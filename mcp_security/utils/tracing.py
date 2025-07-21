"""
OpenTelemetry integration for distributed tracing in MCP+ATCA Security Defense System.

This module provides comprehensive tracing capabilities including:
- Automatic instrumentation for FastAPI, HTTPX, and logging
- Custom span creation and propagation
- Integration with logging system for trace correlation
- Configurable sampling strategies
- Multiple exporter support (OTLP, Jaeger, Console)
"""

import logging
import os
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanProcessor
)
try:
    from opentelemetry.sdk.trace.sampling import (
        TraceIdRatioBased,
        AlwaysOff,
        AlwaysOn,
        ParentBased
    )
except ImportError:
    # Handle different OpenTelemetry versions
    try:
        from opentelemetry.sdk.trace.sampling import ParentBased
        # Try alternative names for other sampling strategies
        try:
            from opentelemetry.sdk.trace.sampling import (
                ALWAYS_OFF as AlwaysOff,
                ALWAYS_ON as AlwaysOn,
                DEFAULT_TRACE_ID_RATIO as TraceIdRatioBased
            )
        except ImportError:
            # Use default sampling strategies
            AlwaysOff = lambda: None
            AlwaysOn = lambda: None
            TraceIdRatioBased = lambda x: None
    except ImportError:
        # Fallback for very old versions
        AlwaysOff = lambda: None
        AlwaysOn = lambda: None
        TraceIdRatioBased = lambda x: None
        ParentBased = lambda x: None
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer
)

from .config import Settings
from .logging import get_logger

# Global tracer instance
_tracer: Optional[Tracer] = None
_tracer_lock = threading.Lock()

logger = get_logger(__name__)


class TracingService:
    """
    Central service for managing OpenTelemetry tracing configuration.
    
    Provides methods for setting up tracing, creating spans, and
    integrating with the logging system.
    """
    
    def __init__(self):
        """Initialize the tracing service."""
        self._initialized = False
        self._tracer_provider: Optional[TracerProvider] = None
        self._processors: List[SpanProcessor] = []
        self._instrumentors = []
    
    def setup(self, settings: Settings) -> None:
        """
        Set up OpenTelemetry tracing with the provided settings.
        
        Args:
            settings: Application settings containing tracing configuration
        """
        if self._initialized:
            return
        
        try:
            # Create resource with service information
            resource = self._create_resource(settings)
            
            # Set up tracer provider
            self._setup_tracer_provider(resource, settings)
            
            # Set up exporters and processors
            self._setup_exporters(settings)
            
            # Set up automatic instrumentation
            self._setup_instrumentation(settings)
            
            # Set as global tracer provider
            trace.set_tracer_provider(self._tracer_provider)
            
            self._initialized = True
            logger.info(
                "OpenTelemetry tracing initialized",
                service_name=settings.otel_service_name,
                exporters=len(self._processors)
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize OpenTelemetry tracing",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def _create_resource(self, settings: Settings) -> Resource:
        """Create OpenTelemetry resource with service information."""
        attributes = {}
        
        # Parse resource attributes from settings
        if settings.otel_resource_attributes:
            for attr in settings.otel_resource_attributes.split(','):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    attributes[key.strip()] = value.strip()
        
        # Ensure required attributes are present
        if 'service.name' not in attributes:
            attributes['service.name'] = settings.otel_service_name
        
        if 'service.version' not in attributes:
            attributes['service.version'] = '0.1.0'
        
        # Add environment information
        attributes['environment'] = 'production' if not settings.debug else 'development'
        attributes['service.instance.id'] = str(uuid4())
        
        return Resource.create(attributes)
    
    def _setup_tracer_provider(self, resource: Resource, settings: Settings) -> None:
        """Set up the tracer provider with sampling configuration."""
        # Configure sampling strategy
        if settings.debug:
            # Sample all traces in development
            sampler = AlwaysOn()
        else:
            # Use ratio-based sampling in production
            sampling_ratio = float(os.getenv('OTEL_TRACES_SAMPLER_ARG', '0.1'))
            sampler = ParentBased(root=TraceIdRatioBased(sampling_ratio))
        
        self._tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler
        )
    
    def _setup_exporters(self, settings: Settings) -> None:
        """Set up span exporters and processors."""
        exporters = []
        
        # OTLP exporter (for production)
        if settings.otel_exporter_otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.otel_exporter_otlp_endpoint,
                headers=self._get_exporter_headers()
            )
            exporters.append(otlp_exporter)
            logger.info(
                "OTLP exporter configured",
                endpoint=settings.otel_exporter_otlp_endpoint
            )
        
        # Console exporter (for development)
        if settings.debug:
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
            logger.info("Console exporter configured for development")
        
        # Create batch processors for each exporter
        for exporter in exporters:
            processor = BatchSpanProcessor(exporter)
            self._processors.append(processor)
            self._tracer_provider.add_span_processor(processor)
    
    def _get_exporter_headers(self) -> Dict[str, str]:
        """Get headers for OTLP exporter from environment."""
        headers = {}
        
        # API key for authentication
        if api_key := os.getenv('OTEL_EXPORTER_OTLP_HEADERS'):
            for header in api_key.split(','):
                if '=' in header:
                    key, value = header.split('=', 1)
                    headers[key.strip()] = value.strip()
        
        return headers
    
    def _setup_instrumentation(self, settings: Settings) -> None:
        """Set up automatic instrumentation for common libraries."""
        try:
            # FastAPI instrumentation
            fastapi_instrumentor = FastAPIInstrumentor()
            fastapi_instrumentor.instrument()
            self._instrumentors.append(fastapi_instrumentor)
            
            # HTTPX instrumentation
            httpx_instrumentor = HTTPXClientInstrumentor()
            httpx_instrumentor.instrument()
            self._instrumentors.append(httpx_instrumentor)
            
            # Logging instrumentation for trace correlation
            logging_instrumentor = LoggingInstrumentor()
            logging_instrumentor.instrument(set_logging_format=True)
            self._instrumentors.append(logging_instrumentor)
            
            logger.info(
                "Automatic instrumentation configured",
                instrumentors=['FastAPI', 'HTTPX', 'Logging']
            )
            
        except Exception as e:
            logger.warning(
                "Some instrumentation failed to initialize",
                error=str(e)
            )
    
    def get_tracer(self, name: str, version: Optional[str] = None) -> Tracer:
        """
        Get a tracer instance.
        
        Args:
            name: Tracer name (typically module name)
            version: Optional tracer version
            
        Returns:
            Tracer: OpenTelemetry tracer instance
        """
        if not self._initialized:
            raise RuntimeError("Tracing service not initialized")
        
        return trace.get_tracer(name, version)
    
    def shutdown(self) -> None:
        """Shutdown tracing and flush any pending spans."""
        try:
            # Shutdown processors
            for processor in self._processors:
                processor.shutdown()
            
            # Uninstrument libraries
            for instrumentor in self._instrumentors:
                instrumentor.uninstrument()
            
            logger.info("OpenTelemetry tracing shutdown complete")
            
        except Exception as e:
            logger.error(
                "Error during tracing shutdown",
                error=str(e)
            )


# Global tracing service instance
_tracing_service = TracingService()


def setup_tracing(settings: Optional[Settings] = None) -> None:
    """
    Set up OpenTelemetry tracing.
    
    Args:
        settings: Optional settings instance. If not provided, will get from config.
    """
    if settings is None:
        from .config import get_settings
        settings = get_settings()
    
    _tracing_service.setup(settings)


def get_tracer(name: str = __name__, version: Optional[str] = None) -> Tracer:
    """
    Get a tracer instance.
    
    Args:
        name: Tracer name
        version: Optional version
        
    Returns:
        Tracer: OpenTelemetry tracer instance
    """
    global _tracer
    
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                _tracer = _tracing_service.get_tracer(name, version)
    
    return _tracer


def create_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
    parent: Optional[Span] = None
) -> Span:
    """
    Create a new span.
    
    Args:
        name: Span name
        kind: Span kind (INTERNAL, SERVER, CLIENT, etc.)
        attributes: Optional span attributes
        parent: Optional parent span
        
    Returns:
        Span: Created span
    """
    tracer = get_tracer()
    
    span = tracer.start_span(
        name=name,
        kind=kind,
        attributes=attributes or {}
    )
    
    return span


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Context manager for tracing an operation.
    
    Args:
        operation_name: Name of the operation
        attributes: Optional attributes to add to the span
        kind: Span kind
        
    Yields:
        Span: The created span
    """
    span = create_span(operation_name, kind=kind, attributes=attributes)
    
    try:
        yield span
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.end()


class AsyncTraceOperation:
    """Async context manager for tracing operations."""
    
    def __init__(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None, kind: SpanKind = SpanKind.INTERNAL):
        self.operation_name = operation_name
        self.attributes = attributes
        self.kind = kind
        self.span = None
    
    async def __aenter__(self):
        self.span = create_span(self.operation_name, kind=self.kind, attributes=self.attributes)
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        self.span.end()


def async_trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Async context manager for tracing an operation.
    
    Args:
        operation_name: Name of the operation
        attributes: Optional attributes to add to the span
        kind: Span kind
        
    Returns:
        AsyncTraceOperation: The async context manager
    """
    return AsyncTraceOperation(operation_name, attributes, kind)


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Decorator for automatically tracing function calls.
    
    Args:
        name: Optional span name. If not provided, uses function name.
        attributes: Optional attributes to add to the span
        kind: Span kind
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_attributes = attributes or {}
            span_attributes.update({
                'function.name': func.__name__,
                'function.module': func.__module__,
            })
            
            with trace_operation(span_name, span_attributes, kind) as span:
                # Add function parameters as attributes (be careful with sensitive data)
                if args:
                    span.set_attribute('function.args.count', len(args))
                if kwargs:
                    span.set_attribute('function.kwargs.count', len(kwargs))
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute('function.result.type', type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute('function.error', str(e))
                    raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            span_attributes = attributes or {}
            span_attributes.update({
                'function.name': func.__name__,
                'function.module': func.__module__,
                'function.async': True
            })
            
            with trace_operation(span_name, span_attributes, kind) as span:
                # Add function parameters as attributes
                if args:
                    span.set_attribute('function.args.count', len(args))
                if kwargs:
                    span.set_attribute('function.kwargs.count', len(kwargs))
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute('function.result.type', type(result).__name__)
                    return result
                except Exception as e:
                    span.set_attribute('function.error', str(e))
                    raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def get_current_span() -> Optional[Span]:
    """
    Get the current active span.
    
    Returns:
        Optional[Span]: Current span or None if no span is active
    """
    return trace.get_current_span()


def get_trace_id() -> Optional[str]:
    """
    Get the current trace ID as a string.
    
    Returns:
        Optional[str]: Trace ID or None if no span is active
    """
    span = get_current_span()
    if span and span.get_span_context().trace_id:
        return format(span.get_span_context().trace_id, '032x')
    return None


def get_span_id() -> Optional[str]:
    """
    Get the current span ID as a string.
    
    Returns:
        Optional[str]: Span ID or None if no span is active
    """
    span = get_current_span()
    if span and span.get_span_context().span_id:
        return format(span.get_span_context().span_id, '016x')
    return None


def inject_trace_context(carrier: Dict[str, str]) -> None:
    """
    Inject trace context into a carrier (e.g., HTTP headers).
    
    Args:
        carrier: Dictionary to inject trace context into
    """
    inject(carrier)


def extract_trace_context(carrier: Dict[str, str]) -> None:
    """
    Extract trace context from a carrier and set as current context.
    
    Args:
        carrier: Dictionary containing trace context
    """
    extract(carrier)


def add_span_attributes(**attributes: Any) -> None:
    """
    Add attributes to the current span.
    
    Args:
        **attributes: Attributes to add to the current span
    """
    span = get_current_span()
    if span:
        for key, value in attributes.items():
            span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current span.
    
    Args:
        name: Event name
        attributes: Optional event attributes
    """
    span = get_current_span()
    if span:
        span.add_event(name, attributes or {})


def set_span_status(status_code: StatusCode, description: Optional[str] = None) -> None:
    """
    Set the status of the current span.
    
    Args:
        status_code: Status code (OK, ERROR, UNSET)
        description: Optional status description
    """
    span = get_current_span()
    if span:
        span.set_status(Status(status_code, description))


def record_exception(exception: Exception, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Record an exception in the current span.
    
    Args:
        exception: Exception to record
        attributes: Optional additional attributes
    """
    span = get_current_span()
    if span:
        span.record_exception(exception, attributes)


def shutdown_tracing() -> None:
    """Shutdown the tracing system."""
    _tracing_service.shutdown()


# Integration with logging system
def get_trace_correlation_data() -> Dict[str, str]:
    """
    Get trace correlation data for logging.
    
    Returns:
        Dict[str, str]: Dictionary containing trace_id and span_id
    """
    correlation_data = {}
    
    if trace_id := get_trace_id():
        correlation_data['trace_id'] = trace_id
    
    if span_id := get_span_id():
        correlation_data['span_id'] = span_id
    
    return correlation_data

# Export all public functions and classes
__all__ = [
    'TracingService',
    'setup_tracing',
    'get_tracer',
    'create_span',
    'trace_operation',
    'async_trace_operation',
    'AsyncTraceOperation',
    'trace_function',
    'get_current_span',
    'get_trace_id',
    'get_span_id',
    'inject_trace_context',
    'extract_trace_context',
    'add_span_attributes',
    'add_span_event',
    'set_span_status',
    'record_exception',
    'shutdown_tracing',
    'get_trace_correlation_data',
]