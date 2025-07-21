"""
Comprehensive logging infrastructure for MCP+ATCA Security Defense System.

This module provides structured logging with JSON formatting, context enrichment,
sensitive data redaction, and integration with OpenTelemetry for distributed tracing.
"""

import json
import logging
import logging.handlers
import os
import re
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4
from functools import wraps
import inspect

import structlog
from structlog.types import FilteringBoundLogger

# Import tracing utilities for trace correlation
try:
    from .tracing import get_trace_correlation_data
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    
    def get_trace_correlation_data() -> Dict[str, str]:
        """Fallback function when tracing is not available."""
        return {}


# Context variables for request-scoped data
_request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


class SecurityEventFilter(logging.Filter):
    """
    Filter to identify security-relevant log events.
    
    This filter helps route security events to dedicated log files
    for easier monitoring and analysis.
    """
    
    def __init__(self):
        """Initialize the security event filter."""
        super().__init__()
        # Keywords that indicate security events
        self.security_keywords = {
            'authentication', 'auth', 'login', 'logout', 'signin', 'signout',
            'authorization', 'permission', 'access', 'denied', 'forbidden',
            'attack', 'intrusion', 'malicious', 'suspicious', 'threat',
            'injection', 'xss', 'csrf', 'sql injection', 'code injection',
            'brute force', 'dictionary attack', 'ddos', 'dos',
            'vulnerability', 'exploit', 'breach', 'compromise',
            'anomaly', 'unusual', 'abnormal', 'outlier',
            'security', 'breach', 'violation', 'alert', 'incident'
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Determine if a log record is security-relevant.
        
        Args:
            record: The log record to filter
            
        Returns:
            bool: True if the record should be processed by this handler
        """
        # Check if the record has security markers
        if hasattr(record, 'security_event') and record.security_event:
            return True
        
        # Check message content for security keywords
        message = record.getMessage().lower()
        for keyword in self.security_keywords:
            if keyword in message:
                return True
        
        # Check logger name for security-related modules
        logger_name = record.name.lower()
        if any(sec_term in logger_name for sec_term in ['security', 'auth', 'psi']):
            return True
        
        return False


class SensitiveDataRedactor:
    """
    Handles redaction of sensitive information from log messages.
    
    Provides configurable patterns and strategies for redacting sensitive data
    like passwords, credit cards, SSNs, and other PII.
    """
    
    def __init__(self):
        """Initialize the redactor with default patterns."""
        # Define regex patterns for common sensitive data
        self.patterns = {
            "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\+\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4})"),
            "password": re.compile(r"(password|pwd|pass)\s*[:=]\s*['\"]?([^'\"\\s\\n\\r]+)", re.IGNORECASE),
            "api_key": re.compile(r"(api[_-]?key|token|secret|sk_[a-z]+_)\s*[:=]?\s*['\"]?([A-Za-z0-9_+/-]{15,})", re.IGNORECASE),
            "jwt": re.compile(r"\beyJ[A-Za-z0-9_/+=.-]+\.eyJ[A-Za-z0-9_/+=.-]+\.[A-Za-z0-9_/+=.-]+\b"),
            "url_credentials": re.compile(r"(https?://)[^:]+:[^@]+@"),
            "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            "mac_address": re.compile(r"\b[0-9A-Fa-f]{2}([-:])[0-9A-Fa-f]{2}(\1[0-9A-Fa-f]{2}){4}\b"),
        }
        
        # Define redaction strategies
        self.strategies = {
            "credit_card": self._mask_credit_card,
            "ssn": self._mask_ssn,
            "email": self._mask_email,
            "phone": self._mask_phone,
            "password": self._mask_password,
            "api_key": self._mask_api_key,
            "jwt": self._mask_jwt,
            "url_credentials": self._mask_url_credentials,
            "ipv4": self._mask_ipv4,
            "mac_address": self._mask_mac_address,
        }
        
        # Fields that should be completely removed or redacted
        self.sensitive_field_names = {
            "password", "pwd", "secret", "token", "api_key", "private_key",
            "credit_card", "ssn", "social_security", "passport_number",
            "auth_token", "access_token", "refresh_token", "session_id",
            "csrf_token", "x_api_key", "authorization", "jwt_token",
            "client_secret", "private_data", "confidential", "sensitive"
        }
        
        # Configuration for redaction behavior
        self.redaction_config = {
            "enabled": True,
            "redact_unknown_patterns": False,
            "preserve_structure": True,
            "redaction_char": "*",
            "min_redaction_length": 3,
        }
    
    def redact_message(self, message: str) -> str:
        """
        Redact sensitive information from a log message.
        
        Args:
            message: The log message to redact
            
        Returns:
            str: The message with sensitive data redacted
        """
        for pattern_name, pattern in self.patterns.items():
            strategy = self.strategies.get(pattern_name, self._default_mask)
            message = pattern.sub(strategy, message)
        
        return message
    
    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive information from a dictionary.
        
        Args:
            data: Dictionary to redact
            
        Returns:
            Dict: Dictionary with sensitive data redacted
        """
        if not isinstance(data, dict):
            return data
        
        redacted = {}
        for key, value in data.items():
            # Check if field name is sensitive
            if key.lower() in self.sensitive_field_names:
                redacted[key] = "[REDACTED]"
            elif isinstance(value, str):
                redacted[key] = self.redact_message(value)
            elif isinstance(value, dict):
                redacted[key] = self.redact_dict(value)
            elif isinstance(value, list):
                redacted[key] = [self.redact_dict(item) if isinstance(item, dict) 
                               else self.redact_message(str(item)) if isinstance(item, str)
                               else item for item in value]
            else:
                redacted[key] = value
        
        return redacted
    
    def _mask_credit_card(self, match: re.Match) -> str:
        """Mask credit card numbers, showing only last 4 digits."""
        card_number = match.group(0).replace("-", "").replace(" ", "")
        return f"****-****-****-{card_number[-4:]}"
    
    def _mask_ssn(self, match: re.Match) -> str:
        """Mask SSN, showing only last 4 digits."""
        return f"***-**-{match.group(0)[-4:]}"
    
    def _mask_email(self, match: re.Match) -> str:
        """Mask email, showing only first char and domain."""
        email = match.group(0)
        parts = email.split("@")
        if len(parts) == 2:
            return f"{parts[0][0]}***@{parts[1]}"
        return "[REDACTED_EMAIL]"
    
    def _mask_phone(self, match: re.Match) -> str:
        """Mask phone number, showing only last 4 digits."""
        phone = match.group(0)
        return f"***-***-{phone[-4:]}"
    
    def _mask_password(self, match: re.Match) -> str:
        """Completely mask password fields."""
        return f"{match.group(1)}=[REDACTED]"
    
    def _mask_api_key(self, match: re.Match) -> str:
        """Mask API keys, showing only first 8 characters."""
        key_part = match.group(2)
        if len(key_part) > 8:
            return f"{match.group(1)}={key_part[:8]}***"
        return f"{match.group(1)}=[REDACTED]"
    
    def _mask_jwt(self, match: re.Match) -> str:
        """Mask JWT tokens."""
        return "[REDACTED_JWT]"
    
    def _mask_url_credentials(self, match: re.Match) -> str:
        """Mask credentials in URLs."""
        return f"{match.group(1)}[REDACTED]:[REDACTED]@"
    
    def _mask_ipv4(self, match: re.Match) -> str:
        """Mask IPv4 addresses, showing only first octet."""
        ip = match.group(0)
        parts = ip.split(".")
        return f"{parts[0]}.***.***.***"
    
    def _mask_mac_address(self, match: re.Match) -> str:
        """Mask MAC addresses, showing only first 6 characters."""
        mac = match.group(0)
        separator = match.group(1)
        return f"**{separator}**{separator}**{separator}**{separator}**{separator}{mac[-2:]}"
    
    def _default_mask(self, match: re.Match) -> str:
        """Default masking strategy."""
        return "[REDACTED]"
    
    def configure_redaction(self, **kwargs) -> None:
        """
        Configure redaction behavior.
        
        Args:
            enabled: Whether redaction is enabled
            redact_unknown_patterns: Whether to redact unknown patterns
            preserve_structure: Whether to preserve data structure
            redaction_char: Character to use for masking
            min_redaction_length: Minimum length for redaction
        """
        self.redaction_config.update(kwargs)
    
    def add_custom_pattern(self, name: str, pattern: str, strategy_func: Optional[callable] = None) -> None:
        """
        Add a custom redaction pattern.
        
        Args:
            name: Name of the pattern
            pattern: Regex pattern string
            strategy_func: Optional custom strategy function
        """
        self.patterns[name] = re.compile(pattern)
        if strategy_func:
            self.strategies[name] = strategy_func
        else:
            self.strategies[name] = self._default_mask
    
    def remove_pattern(self, name: str) -> None:
        """Remove a redaction pattern."""
        self.patterns.pop(name, None)
        self.strategies.pop(name, None)
    
    def get_redaction_stats(self, text: str) -> Dict[str, int]:
        """
        Get statistics about what would be redacted in the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict mapping pattern names to match counts
        """
        stats = {}
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            stats[pattern_name] = len(matches)
        return stats


class SecurityLogger:
    """
    Enhanced logger with security-specific features including context enrichment,
    sensitive data redaction, and structured JSON output.
    """
    
    def __init__(self, name: str, redactor: Optional[SensitiveDataRedactor] = None):
        """
        Initialize the SecurityLogger.
        
        Args:
            name: Logger name (typically module name)
            redactor: Optional custom redactor instance
        """
        self.name = name
        self.redactor = redactor or SensitiveDataRedactor()
        self._logger = structlog.get_logger(name)
        self._local = threading.local()
    
    def _get_context(self) -> Dict[str, Any]:
        """
        Get the current logging context.
        
        Returns:
            Dict: Current context data
        """
        # Get context from ContextVar (request-scoped)
        context = _request_context.get({}).copy()
        
        # Add thread-local context if available
        if hasattr(self._local, "context"):
            context.update(self._local.context)
        
        return context
    
    def _prepare_log_data(self, 
                         level: str,
                         message: str, 
                         **kwargs) -> Dict[str, Any]:
        """
        Prepare log data with context and redaction.
        
        Args:
            level: Log level
            message: Log message
            **kwargs: Additional log data
            
        Returns:
            Dict: Prepared log data
        """
        # Create base log structure
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "logger": self.name,
            "message": message,
            "thread_id": threading.get_ident(),
            "process_id": os.getpid(),
        }
        
        # Add trace correlation data
        if TRACING_AVAILABLE:
            trace_data = get_trace_correlation_data()
            if trace_data:
                log_data.update(trace_data)
        
        # Add context data
        context = self._get_context()
        if context:
            log_data["context"] = context
        
        # Add additional data
        if kwargs:
            # Redact sensitive data from kwargs
            kwargs = self.redactor.redact_dict(kwargs)
            log_data.update(kwargs)
        
        # Redact sensitive data from message
        log_data["message"] = self.redactor.redact_message(message)
        
        return log_data
    
    def set_context(self, **context_data: Any) -> None:
        """
        Set thread-local context data.
        
        Args:
            **context_data: Context data to set
        """
        if not hasattr(self._local, "context"):
            self._local.context = {}
        self._local.context.update(context_data)
    
    def clear_context(self) -> None:
        """Clear thread-local context data."""
        if hasattr(self._local, "context"):
            self._local.context.clear()
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context."""
        log_data = self._prepare_log_data("DEBUG", message, **kwargs)
        self._logger.debug(message, **log_data)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context."""
        log_data = self._prepare_log_data("INFO", message, **kwargs)
        self._logger.info(message, **log_data)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context."""
        log_data = self._prepare_log_data("WARNING", message, **kwargs)
        self._logger.warning(message, **log_data)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with context."""
        log_data = self._prepare_log_data("ERROR", message, **kwargs)
        self._logger.error(message, **log_data)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with context."""
        log_data = self._prepare_log_data("CRITICAL", message, **kwargs)
        self._logger.critical(message, **log_data)
    
    def security_event(self, 
                      event_type: str,
                      message: str, 
                      severity: str = "INFO",
                      user_id: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      resource: Optional[str] = None,
                      action: Optional[str] = None,
                      **kwargs) -> None:
        """
        Log a security-specific event with additional metadata.
        
        Args:
            event_type: Type of security event (authentication, authorization, etc.)
            message: Event description
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            user_id: User identifier
            ip_address: Source IP address
            resource: Resource being accessed
            action: Action being performed
            **kwargs: Additional event data
        """
        security_data = {
            "event_type": event_type,
            "security_category": "security_event",
            "severity": severity.upper(),
        }
        
        if user_id:
            security_data["user_id"] = user_id
        if ip_address:
            security_data["ip_address"] = ip_address
        if resource:
            security_data["resource"] = resource
        if action:
            security_data["action"] = action
        
        # Merge with additional kwargs
        security_data.update(kwargs)
        
        # Log with appropriate level based on severity
        if severity.upper() == "CRITICAL":
            self.critical(message, **security_data)
        elif severity.upper() == "ERROR":
            self.error(message, **security_data)
        elif severity.upper() == "WARNING":
            self.warning(message, **security_data)
        else:
            self.info(message, **security_data)


class LoggingService:
    """
    Central logging service that manages logger configuration and provides
    a unified interface for all logging operations.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Implement singleton pattern for LoggingService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the logging service."""
        if hasattr(self, "_initialized"):
            return
            
        self._initialized = True
        self.redactor = SensitiveDataRedactor()
        self._loggers: Dict[str, SecurityLogger] = {}
        self._setup_complete = False
    
    def setup(self, settings) -> None:
        """
        Set up the logging infrastructure.
        
        Args:
            settings: Application settings instance
        """
        if self._setup_complete:
            return
        
        # Create log directory if it doesn't exist
        log_dir = Path(settings.get_log_dir())
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        self._configure_structlog(settings)
        
        # Set up standard Python logging
        self._configure_python_logging(settings)
        
        self._setup_complete = True
    
    def _configure_structlog(self, settings) -> None:
        """Configure structlog for structured JSON logging."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            self._add_trace_correlation,  # Add trace correlation data
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if settings.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            context_class=dict,
            cache_logger_on_first_use=True,
        )
    
    def _configure_python_logging(self, settings) -> None:
        """Configure Python's standard logging module with advanced rotation and retention."""
        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        if settings.log_format == "json":
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
                '"logger": "%(name)s", "message": "%(message)s"}'
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # Set up multiple file handlers for different log types
        self._setup_file_handlers(settings, formatter, root_logger)
        
        # Set up console handler for development
        if settings.debug:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(console_handler)
        
        # Start retention cleanup background task
        self._setup_retention_cleanup(settings)
    
    def _setup_file_handlers(self, settings, formatter, root_logger) -> None:
        """Set up file handlers with rotation and compression."""
        import gzip
        import shutil
        
        # Main application log with size-based rotation
        size_handler = logging.handlers.RotatingFileHandler(
            filename=settings.log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8"
        )
        # Override doRollover to add compression
        original_doRollover = size_handler.doRollover
        
        def compressed_doRollover():
            """Custom rollover that compresses old log files."""
            original_doRollover()
            # Compress the rotated file
            for i in range(1, settings.log_backup_count + 1):
                log_file = f"{settings.log_file}.{i}"
                if os.path.exists(log_file) and not log_file.endswith('.gz'):
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(f"{log_file}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(log_file)
        
        size_handler.doRollover = compressed_doRollover
        size_handler.setFormatter(formatter)
        size_handler.setLevel(getattr(logging, settings.log_level))
        root_logger.addHandler(size_handler)
        
        # Time-based rotation for daily logs
        log_dir = Path(settings.get_log_dir())
        daily_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / "daily.log",
            when=settings.log_rotation_when,
            interval=1,
            backupCount=settings.log_retention_days,
            encoding="utf-8",
            delay=False,
            utc=True
        )
        daily_handler.setFormatter(formatter)
        daily_handler.setLevel(getattr(logging, settings.log_level))
        root_logger.addHandler(daily_handler)
        
        # Security-specific log file
        security_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / "security.log",
            when=settings.log_rotation_when,
            interval=1,
            backupCount=settings.log_retention_days,
            encoding="utf-8",
            delay=False,
            utc=True
        )
        security_handler.setFormatter(formatter)
        security_handler.setLevel(logging.WARNING)  # Only warnings and above for security
        
        # Add security filter
        security_filter = SecurityEventFilter()
        security_handler.addFilter(security_filter)
        root_logger.addHandler(security_handler)
        
        # Error-only log file
        error_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_dir / "errors.log",
            when=settings.log_rotation_when,
            interval=1,
            backupCount=settings.log_retention_days * 2,  # Keep errors longer
            encoding="utf-8",
            delay=False,
            utc=True
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    def _setup_retention_cleanup(self, settings) -> None:
        """Set up background task for log retention cleanup."""
        import threading
        import time
        import glob
        
        def cleanup_old_logs():
            """Background task to cleanup old log files based on retention policy."""
            while True:
                try:
                    log_dir = Path(settings.get_log_dir())
                    current_time = time.time()
                    retention_seconds = settings.log_retention_days * 24 * 60 * 60
                    
                    # Find all log files
                    log_patterns = [
                        "*.log.*",
                        "*.log.*.gz",
                        "daily.log.*",
                        "security.log.*", 
                        "errors.log.*"
                    ]
                    
                    for pattern in log_patterns:
                        for log_file_path in glob.glob(str(log_dir / pattern)):
                            file_path = Path(log_file_path)
                            if file_path.exists():
                                file_age = current_time - file_path.stat().st_mtime
                                if file_age > retention_seconds:
                                    try:
                                        file_path.unlink()
                                        print(f"Deleted old log file: {file_path}")
                                    except OSError as e:
                                        print(f"Failed to delete log file {file_path}: {e}")
                    
                    # Sleep for 1 hour before next cleanup
                    time.sleep(3600)
                    
                except Exception as e:
                    print(f"Error in log cleanup: {e}")
                    time.sleep(3600)  # Wait an hour before retrying
        
        # Start cleanup thread as daemon
        cleanup_thread = threading.Thread(target=cleanup_old_logs, daemon=True)
        cleanup_thread.start()
    
    def _add_trace_correlation(self, logger, method_name, event_dict):
        """
        Add trace correlation data to log entries.
        
        This processor adds OpenTelemetry trace and span IDs to log entries
        when available, enabling correlation between logs and traces.
        
        Args:
            logger: The logger instance
            method_name: The logging method name
            event_dict: The event dictionary
            
        Returns:
            Updated event dictionary with trace correlation data
        """
        if TRACING_AVAILABLE:
            trace_data = get_trace_correlation_data()
            if trace_data:
                event_dict.update(trace_data)
        
        return event_dict
    
    def get_logger(self, name: str) -> SecurityLogger:
        """
        Get or create a SecurityLogger instance.
        
        Args:
            name: Logger name
            
        Returns:
            SecurityLogger: Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = SecurityLogger(name, self.redactor)
        return self._loggers[name]


# Global logging service instance
_logging_service = LoggingService()


def setup_logging(settings=None) -> None:
    """
    Set up the logging infrastructure.
    
    Args:
        settings: Optional settings instance. If not provided, will get from config.
    """
    if settings is None:
        from .config import get_settings
        settings = get_settings()
    
    _logging_service.setup(settings)


def get_logger(name: str) -> SecurityLogger:
    """
    Get a SecurityLogger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        SecurityLogger: Logger instance
    """
    return _logging_service.get_logger(name)


def set_request_context(**context_data: Any) -> None:
    """
    Set request-scoped context data using ContextVar.
    
    Args:
        **context_data: Context data to set
    """
    current_context = _request_context.get({})
    current_context.update(context_data)
    _request_context.set(current_context)


def clear_request_context() -> None:
    """Clear request-scoped context data."""
    _request_context.set({})


def get_request_context() -> Dict[str, Any]:
    """
    Get current request-scoped context data.
    
    Returns:
        Dict: Current context data
    """
    return _request_context.get({})


# Performance monitoring decorator
def log_performance(logger: Optional[SecurityLogger] = None):
    """
    Decorator to log function execution time and performance metrics.
    
    Args:
        logger: Optional logger instance. If not provided, creates one.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            start_time = time.time()
            function_name = f"{func.__module__}.{func.__name__}"
            
            func_logger.debug(
                f"Starting function execution",
                function=function_name,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                func_logger.info(
                    f"Function execution completed",
                    function=function_name,
                    execution_time_ms=round(execution_time * 1000, 2),
                    success=True
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                func_logger.error(
                    f"Function execution failed",
                    function=function_name,
                    execution_time_ms=round(execution_time * 1000, 2),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    success=False
                )
                
                raise
        
        return wrapper
    return decorator


def redact_sensitive_data(
    *sensitive_params: str,
    redactor: Optional[SensitiveDataRedactor] = None,
    logger: Optional[SecurityLogger] = None
) -> Callable:
    """
    Decorator to automatically redact sensitive data from function arguments and return values.
    
    Args:
        *sensitive_params: Names of parameters that contain sensitive data
        redactor: Custom redactor instance
        logger: Optional logger instance
    
    Usage:
        @redact_sensitive_data('password', 'api_key')
        def login(username: str, password: str, api_key: str):
            # Function implementation
            pass
    """
    if redactor is None:
        redactor = SensitiveDataRedactor()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
            
            # Get function signature for parameter mapping
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Redact sensitive parameters
            safe_params = {}
            for param_name, param_value in bound_args.arguments.items():
                if param_name in sensitive_params:
                    if isinstance(param_value, str):
                        safe_params[param_name] = redactor.redact_message(param_value)
                    elif isinstance(param_value, dict):
                        safe_params[param_name] = redactor.redact_dict(param_value)
                    else:
                        safe_params[param_name] = "[REDACTED]"
                else:
                    safe_params[param_name] = param_value
            
            func_logger.debug(
                f"Function called with redacted parameters",
                function=f"{func.__module__}.{func.__name__}",
                parameters=safe_params
            )
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Log successful execution (don't log return value as it might contain sensitive data)
                func_logger.debug(
                    f"Function execution completed successfully",
                    function=f"{func.__module__}.{func.__name__}"
                )
                
                return result
                
            except Exception as e:
                func_logger.error(
                    f"Function execution failed",
                    function=f"{func.__module__}.{func.__name__}",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


def configure_logging_for_sensitive_operations(
    redaction_enabled: bool = True,
    custom_patterns: Optional[Dict[str, str]] = None,
    log_level: str = "INFO"
) -> SensitiveDataRedactor:
    """
    Configure logging specifically for operations that handle sensitive data.
    
    Args:
        redaction_enabled: Whether to enable data redaction
        custom_patterns: Additional regex patterns for redaction
        log_level: Logging level for sensitive operations
        
    Returns:
        Configured SensitiveDataRedactor instance
    """
    redactor = SensitiveDataRedactor()
    
    # Configure redaction
    redactor.configure_redaction(enabled=redaction_enabled)
    
    # Add custom patterns if provided
    if custom_patterns:
        for name, pattern in custom_patterns.items():
            redactor.add_custom_pattern(name, pattern)
    
    # Set appropriate log level
    logger = get_logger("sensitive_operations")
    
    return redactor


def create_security_audit_log(
    event_type: str,
    message: str,
    **additional_data
) -> None:
    """
    Create a security audit log entry with automatic redaction.
    
    Args:
        event_type: Type of security event
        message: Description of the event
        **additional_data: Additional data to log (will be redacted)
    """
    logger = get_logger("security_audit")
    redactor = SensitiveDataRedactor()
    
    # Redact additional data
    safe_data = redactor.redact_dict(additional_data)
    
    logger.security_event(
        event_type=event_type,
        message=message,
        severity="INFO",
        **safe_data
    )


def validate_redaction_patterns() -> Dict[str, bool]:
    """
    Validate all redaction patterns to ensure they work correctly.
    
    Returns:
        Dict mapping pattern names to validation results
    """
    redactor = SensitiveDataRedactor()
    
    # Test data for each pattern
    test_data = {
        "credit_card": "4532-1234-5678-9012",
        "ssn": "123-45-6789",
        "email": "user@example.com",
        "phone": "(555) 123-4567",
        "password": "password=secret123",
        "api_key": "api_key=sk_test_12345678901234567890",
        "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
        "url_credentials": "https://user:pass@example.com",
        "ipv4": "192.168.1.100",
        "mac_address": "00:1B:63:84:45:E6",
    }
    
    results = {}
    for pattern_name, test_input in test_data.items():
        if pattern_name in redactor.patterns:
            original = test_input
            redacted = redactor.redact_message(test_input)
            # Pattern is valid if the output is different from input
            results[pattern_name] = original != redacted
        else:
            results[pattern_name] = False
    
    return results 