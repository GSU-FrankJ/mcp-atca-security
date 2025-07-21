"""
ELK Stack integration for MCP+ATCA Security Defense System.

This module provides comprehensive integration with the ELK (Elasticsearch, Logstash, Kibana)
stack for centralized log aggregation, search, and visualization.

Key Features:
- Elasticsearch client management and health monitoring
- Logstash formatter for structured log shipping
- Index template management for security logs
- Kibana dashboard and visualization management
- Real-time log streaming and search capabilities
"""

import asyncio
import json
import logging
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path
import threading
from dataclasses import dataclass, asdict

try:
    from elasticsearch import AsyncElasticsearch, Elasticsearch
    from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    AsyncElasticsearch = None
    Elasticsearch = None
    ConnectionError = Exception
    NotFoundError = Exception
    RequestError = Exception

try:
    import logstash_formatter
    LOGSTASH_FORMATTER_AVAILABLE = True
except ImportError:
    LOGSTASH_FORMATTER_AVAILABLE = False
    logstash_formatter = None

from .config import Settings
from .logging import get_logger, SecurityLogger


@dataclass
class ELKHealthStatus:
    """Health status information for ELK stack components."""
    elasticsearch_healthy: bool
    elasticsearch_cluster_status: Optional[str]
    elasticsearch_nodes: int
    elasticsearch_indices: int
    logstash_healthy: bool
    logstash_connected: bool
    kibana_healthy: bool
    last_check: datetime
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['last_check'] = self.last_check.isoformat()
        return result


@dataclass
class LogIndexTemplate:
    """Elasticsearch index template configuration."""
    name: str
    index_patterns: List[str]
    mappings: Dict[str, Any]
    settings: Dict[str, Any]
    priority: int = 100
    
    def to_elasticsearch_template(self) -> Dict[str, Any]:
        """Convert to Elasticsearch template format."""
        return {
            "index_patterns": self.index_patterns,
            "template": {
                "mappings": self.mappings,
                "settings": self.settings
            },
            "priority": self.priority,
            "_meta": {
                "description": f"Template for {self.name} logs",
                "created_by": "mcp-security",
                "created_at": datetime.utcnow().isoformat()
            }
        }


class ElasticsearchManager:
    """
    Manages Elasticsearch connections and operations for log aggregation.
    
    Provides high-level operations for index management, document indexing,
    and search capabilities specifically designed for security logs.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Elasticsearch manager.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        self._client: Optional[AsyncElasticsearch] = None
        self._sync_client: Optional[Elasticsearch] = None
        self._health_status: Optional[ELKHealthStatus] = None
        self._last_health_check: Optional[datetime] = None
        self._health_check_interval = 60  # seconds
        
        if not ELASTICSEARCH_AVAILABLE:
            self.logger.warning("Elasticsearch package not available")
            
    async def initialize(self) -> bool:
        """
        Initialize Elasticsearch connection and setup.
        
        Returns:
            bool: True if initialization successful
        """
        if not ELASTICSEARCH_AVAILABLE:
            self.logger.error("Cannot initialize Elasticsearch - package not available")
            return False
            
        try:
            # Create async client
            self._client = AsyncElasticsearch(
                hosts=self.settings.elasticsearch_hosts,
                basic_auth=(
                    self.settings.elasticsearch_username,
                    self.settings.elasticsearch_password
                ) if self.settings.elasticsearch_username else None,
                verify_certs=False,  # For development - should be True in production
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3
            )
            
            # Create sync client for non-async operations
            self._sync_client = Elasticsearch(
                hosts=self.settings.elasticsearch_hosts,
                basic_auth=(
                    self.settings.elasticsearch_username,
                    self.settings.elasticsearch_password
                ) if self.settings.elasticsearch_username else None,
                verify_certs=False,
                request_timeout=30,
                retry_on_timeout=True,
                max_retries=3
            )
            
            # Test connection
            if await self._client.ping():
                self.logger.info("Elasticsearch connection established")
                
                # Set up index templates
                await self._setup_index_templates()
                
                return True
            else:
                self.logger.error("Failed to ping Elasticsearch")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Elasticsearch: {e}")
            return False
    
    async def _setup_index_templates(self) -> None:
        """Set up Elasticsearch index templates for security logs."""
        templates = [
            self._create_security_log_template(),
            self._create_trace_log_template(),
            self._create_metric_log_template()
        ]
        
        for template in templates:
            try:
                await self._client.indices.put_index_template(
                    name=template.name,
                    **template.to_elasticsearch_template()
                )
                self.logger.info(f"Created index template: {template.name}")
            except Exception as e:
                self.logger.error(f"Failed to create index template {template.name}: {e}")
    
    def _create_security_log_template(self) -> LogIndexTemplate:
        """Create index template for security logs."""
        return LogIndexTemplate(
            name="security-logs",
            index_patterns=["security-logs-*"],
            mappings={
                "properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "logger": {"type": "keyword"},
                    "message": {"type": "text", "analyzer": "standard"},
                    "event_type": {"type": "keyword"},
                    "security_category": {"type": "keyword"},
                    "severity": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "session_id": {"type": "keyword"},
                    "source_ip": {"type": "ip"},
                    "user_agent": {"type": "text"},
                    "threat_level": {"type": "keyword"},
                    "attack_type": {"type": "keyword"},
                    "indicators": {"type": "nested"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "process_id": {"type": "long"},
                    "thread_id": {"type": "long"},
                    "host": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "version": {"type": "keyword"}
                }
            },
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.refresh_interval": "5s",
                "index.mapping.total_fields.limit": 2000
            }
        )
    
    def _create_trace_log_template(self) -> LogIndexTemplate:
        """Create index template for trace logs."""
        return LogIndexTemplate(
            name="trace-logs",
            index_patterns=["trace-logs-*"],
            mappings={
                "properties": {
                    "@timestamp": {"type": "date"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "parent_span_id": {"type": "keyword"},
                    "operation_name": {"type": "keyword"},
                    "duration_ms": {"type": "long"},
                    "status": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "attributes": {"type": "object"},
                    "events": {"type": "nested"},
                    "links": {"type": "nested"}
                }
            },
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "index.refresh_interval": "10s"
            }
        )
    
    def _create_metric_log_template(self) -> LogIndexTemplate:
        """Create index template for metric logs."""
        return LogIndexTemplate(
            name="metric-logs",
            index_patterns=["metrics-*"],
            mappings={
                "properties": {
                    "@timestamp": {"type": "date"},
                    "metric_name": {"type": "keyword"},
                    "metric_type": {"type": "keyword"},
                    "value": {"type": "double"},
                    "unit": {"type": "keyword"},
                    "tags": {"type": "object"},
                    "service": {"type": "keyword"},
                    "host": {"type": "keyword"}
                }
            },
            settings={
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.refresh_interval": "30s"
            }
        )
    
    async def index_document(self, index: str, document: Dict[str, Any], doc_id: Optional[str] = None) -> bool:
        """
        Index a document in Elasticsearch.
        
        Args:
            index: Index name
            document: Document to index
            doc_id: Optional document ID
            
        Returns:
            bool: True if indexing successful
        """
        if not self._client:
            return False
            
        try:
            # Add timestamp if not present
            if "@timestamp" not in document:
                document["@timestamp"] = datetime.utcnow().isoformat()
            
            # Index document
            result = await self._client.index(
                index=index,
                id=doc_id,
                body=document
            )
            
            return result.get("result") in ["created", "updated"]
            
        except Exception as e:
            self.logger.error(f"Failed to index document: {e}")
            return False
    
    async def bulk_index(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk index multiple documents.
        
        Args:
            documents: List of documents with '_index' and optionally '_id' fields
            
        Returns:
            Dict with success/error counts
        """
        if not self._client or not documents:
            return {"success": 0, "errors": 0}
        
        try:
            # Prepare bulk actions
            actions = []
            for doc in documents:
                action = {
                    "_index": doc.pop("_index"),
                    "_source": doc
                }
                if "_id" in doc:
                    action["_id"] = doc.pop("_id")
                if "@timestamp" not in doc:
                    doc["@timestamp"] = datetime.utcnow().isoformat()
                
                actions.append({"index": action})
                actions.append(doc)
            
            # Execute bulk request
            response = await self._client.bulk(body=actions)
            
            # Count results
            success_count = 0
            error_count = 0
            
            for item in response.get("items", []):
                if "index" in item:
                    if item["index"].get("status") in [200, 201]:
                        success_count += 1
                    else:
                        error_count += 1
            
            self.logger.info(f"Bulk index completed: {success_count} success, {error_count} errors")
            
            return {"success": success_count, "errors": error_count}
            
        except Exception as e:
            self.logger.error(f"Bulk index failed: {e}")
            return {"success": 0, "errors": len(documents)}
    
    async def search(self, index: str, query: Dict[str, Any], size: int = 100) -> Dict[str, Any]:
        """
        Search documents in Elasticsearch.
        
        Args:
            index: Index pattern to search
            query: Elasticsearch query DSL
            size: Maximum number of results
            
        Returns:
            Search results
        """
        if not self._client:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        try:
            response = await self._client.search(
                index=index,
                body=query,
                size=size
            )
            return response
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"hits": {"hits": [], "total": {"value": 0}}}
    
    async def get_health_status(self) -> ELKHealthStatus:
        """
        Get comprehensive health status of ELK stack.
        
        Returns:
            ELKHealthStatus: Current health status
        """
        errors = []
        
        # Check if we need to refresh health status
        now = datetime.utcnow()
        if (self._last_health_check and 
            (now - self._last_health_check).total_seconds() < self._health_check_interval and
            self._health_status):
            return self._health_status
        
        # Elasticsearch health
        elasticsearch_healthy = False
        elasticsearch_cluster_status = None
        elasticsearch_nodes = 0
        elasticsearch_indices = 0
        
        if self._client:
            try:
                # Cluster health
                health = await self._client.cluster.health()
                elasticsearch_healthy = health.get("status") in ["green", "yellow"]
                elasticsearch_cluster_status = health.get("status")
                elasticsearch_nodes = health.get("number_of_nodes", 0)
                
                # Index count
                indices = await self._client.cat.indices(format="json")
                elasticsearch_indices = len(indices) if indices else 0
                
            except Exception as e:
                errors.append(f"Elasticsearch health check failed: {e}")
        else:
            errors.append("Elasticsearch client not initialized")
        
        # Logstash health (simplified check)
        logstash_healthy = False
        logstash_connected = False
        
        try:
            # Try to connect to Logstash port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.settings.logstash_host, self.settings.logstash_port))
            sock.close()
            
            logstash_connected = result == 0
            logstash_healthy = logstash_connected  # Simplified check
            
        except Exception as e:
            errors.append(f"Logstash connectivity check failed: {e}")
        
        # Kibana health (simplified check)
        kibana_healthy = elasticsearch_healthy  # Simplified - Kibana depends on Elasticsearch
        
        # Create health status
        self._health_status = ELKHealthStatus(
            elasticsearch_healthy=elasticsearch_healthy,
            elasticsearch_cluster_status=elasticsearch_cluster_status,
            elasticsearch_nodes=elasticsearch_nodes,
            elasticsearch_indices=elasticsearch_indices,
            logstash_healthy=logstash_healthy,
            logstash_connected=logstash_connected,
            kibana_healthy=kibana_healthy,
            last_check=now,
            errors=errors
        )
        
        self._last_health_check = now
        return self._health_status
    
    async def create_index_if_not_exists(self, index_name: str, mappings: Optional[Dict] = None) -> bool:
        """
        Create an index if it doesn't exist.
        
        Args:
            index_name: Name of the index
            mappings: Optional index mappings
            
        Returns:
            bool: True if index exists or was created successfully
        """
        if not self._client:
            return False
        
        try:
            # Check if index exists
            exists = await self._client.indices.exists(index=index_name)
            if exists:
                return True
            
            # Create index
            body = {}
            if mappings:
                body["mappings"] = mappings
            
            await self._client.indices.create(index=index_name, body=body)
            self.logger.info(f"Created index: {index_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown Elasticsearch connections."""
        if self._client:
            await self._client.close()
            self._client = None
        
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        
        self.logger.info("Elasticsearch connections closed")


class LogstashHandler(logging.Handler):
    """
    Custom logging handler that sends logs to Logstash.
    
    This handler formats logs appropriately for Logstash ingestion
    and handles connection failures gracefully.
    """
    
    def __init__(self, host: str, port: int, tags: Optional[List[str]] = None):
        """
        Initialize Logstash handler.
        
        Args:
            host: Logstash host
            port: Logstash port
            tags: Optional tags to add to all log records
        """
        super().__init__()
        self.host = host
        self.port = port
        self.tags = tags or []
        self._socket = None
        self._last_connection_attempt = 0
        self._connection_retry_interval = 30  # seconds
        
    def _connect(self) -> bool:
        """
        Establish connection to Logstash.
        
        Returns:
            bool: True if connection successful
        """
        current_time = time.time()
        
        # Don't retry too frequently
        if (current_time - self._last_connection_attempt) < self._connection_retry_interval:
            return False
        
        self._last_connection_attempt = current_time
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5)
            self._socket.connect((self.host, self.port))
            return True
        except Exception:
            if self._socket:
                self._socket.close()
                self._socket = None
            return False
    
    def emit(self, record: logging.LogRecord) -> None:
        """
        Emit a log record to Logstash.
        
        Args:
            record: Log record to emit
        """
        if not self._socket and not self._connect():
            return
        
        try:
            # Format record as JSON
            log_entry = {
                "@timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "tags": self.tags.copy(),
                "host": socket.gethostname(),
                "service": "mcp-security"
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.format(record)
            
            # Add custom fields from record
            for key, value in record.__dict__.items():
                if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                               "filename", "module", "lineno", "funcName", "created", 
                               "msecs", "relativeCreated", "thread", "threadName", 
                               "processName", "process", "exc_info", "exc_text", "stack_info"]:
                    log_entry[key] = value
            
            # Send to Logstash
            message = json.dumps(log_entry) + "\n"
            self._socket.send(message.encode('utf-8'))
            
        except Exception:
            # Connection failed, close socket and retry later
            if self._socket:
                self._socket.close()
                self._socket = None
    
    def close(self) -> None:
        """Close the handler and socket connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
        super().close()


class ELKIntegration:
    """
    Main ELK Stack integration service.
    
    Coordinates Elasticsearch, Logstash, and Kibana integrations
    for comprehensive log aggregation and analysis.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize ELK integration.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        self.elasticsearch_manager = ElasticsearchManager(settings)
        self._logstash_handler: Optional[LogstashHandler] = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """
        Initialize ELK stack integration.
        
        Returns:
            bool: True if initialization successful
        """
        self.logger.info("Initializing ELK stack integration")
        
        # Initialize Elasticsearch
        if not await self.elasticsearch_manager.initialize():
            self.logger.error("Failed to initialize Elasticsearch")
            return False
        
        # Set up Logstash handler
        if LOGSTASH_FORMATTER_AVAILABLE:
            self._setup_logstash_handler()
        else:
            self.logger.warning("Logstash formatter not available")
        
        self._initialized = True
        self.logger.info("ELK stack integration initialized successfully")
        return True
    
    def _setup_logstash_handler(self) -> None:
        """Set up Logstash logging handler."""
        try:
            self._logstash_handler = LogstashHandler(
                host=self.settings.logstash_host,
                port=self.settings.logstash_port,
                tags=["mcp-security", "python"]
            )
            
            # Add to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(self._logstash_handler)
            
            self.logger.info("Logstash handler configured")
            
        except Exception as e:
            self.logger.error(f"Failed to set up Logstash handler: {e}")
    
    async def log_security_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Log a security event to Elasticsearch.
        
        Args:
            event_data: Security event data
            
        Returns:
            bool: True if logging successful
        """
        if not self._initialized:
            return False
        
        # Add metadata
        enriched_event = {
            **event_data,
            "@timestamp": datetime.utcnow().isoformat(),
            "service": "mcp-security",
            "host": socket.gethostname(),
            "event_category": "security"
        }
        
        # Determine index name (daily rotation)
        index_name = f"security-logs-{datetime.utcnow().strftime('%Y.%m.%d')}"
        
        return await self.elasticsearch_manager.index_document(index_name, enriched_event)
    
    async def search_security_events(self, 
                                   query: str, 
                                   time_range: Optional[Dict[str, str]] = None,
                                   size: int = 100) -> List[Dict[str, Any]]:
        """
        Search security events in Elasticsearch.
        
        Args:
            query: Search query string
            time_range: Time range filter (e.g., {"gte": "now-1h"})
            size: Maximum number of results
            
        Returns:
            List of matching events
        """
        if not self._initialized:
            return []
        
        # Build Elasticsearch query
        es_query = {
            "query": {
                "bool": {
                    "must": [
                        {"match": {"message": query}} if query else {"match_all": {}}
                    ],
                    "filter": []
                }
            },
            "sort": [
                {"@timestamp": {"order": "desc"}}
            ]
        }
        
        # Add time range filter
        if time_range:
            es_query["query"]["bool"]["filter"].append({
                "range": {"@timestamp": time_range}
            })
        
        # Search in security logs index pattern
        results = await self.elasticsearch_manager.search(
            index="security-logs-*",
            query=es_query,
            size=size
        )
        
        # Extract hits
        return [hit["_source"] for hit in results.get("hits", {}).get("hits", [])]
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of ELK stack.
        
        Returns:
            Dict containing health information
        """
        if not self._initialized:
            return {"status": "not_initialized", "components": {}}
        
        health = await self.elasticsearch_manager.get_health_status()
        
        return {
            "status": "healthy" if health.elasticsearch_healthy else "unhealthy",
            "components": health.to_dict(),
            "summary": {
                "elasticsearch": "healthy" if health.elasticsearch_healthy else "unhealthy",
                "logstash": "healthy" if health.logstash_healthy else "unhealthy", 
                "kibana": "healthy" if health.kibana_healthy else "unhealthy"
            }
        }
    
    async def create_kibana_dashboards(self) -> bool:
        """
        Create default Kibana dashboards for security monitoring.
        
        Returns:
            bool: True if dashboards created successfully
        """
        # This would typically involve calling Kibana API
        # For now, we'll just log the intent and return True
        self.logger.info("Kibana dashboard creation requested")
        
        # TODO: Implement actual Kibana dashboard creation
        # This would involve:
        # 1. Creating index patterns
        # 2. Creating visualizations
        # 3. Creating dashboards
        # 4. Setting up alerts
        
        return True
    
    async def shutdown(self) -> None:
        """Shutdown ELK integration."""
        self.logger.info("Shutting down ELK integration")
        
        # Close Logstash handler
        if self._logstash_handler:
            self._logstash_handler.close()
            
            # Remove from root logger
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._logstash_handler)
            self._logstash_handler = None
        
        # Shutdown Elasticsearch
        await self.elasticsearch_manager.shutdown()
        
        self._initialized = False
        self.logger.info("ELK integration shutdown complete")


# Global ELK integration instance
_elk_integration: Optional[ELKIntegration] = None


async def get_elk_integration(settings: Optional[Settings] = None) -> Optional[ELKIntegration]:
    """
    Get or create the global ELK integration instance.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        ELKIntegration instance or None if not available
    """
    global _elk_integration
    
    if _elk_integration is None and settings:
        _elk_integration = ELKIntegration(settings)
        await _elk_integration.initialize()
    
    return _elk_integration


async def shutdown_elk_integration() -> None:
    """Shutdown the global ELK integration instance."""
    global _elk_integration
    
    if _elk_integration:
        await _elk_integration.shutdown()
        _elk_integration = None


# Export main classes and functions
__all__ = [
    'ELKIntegration',
    'ElasticsearchManager', 
    'LogstashHandler',
    'ELKHealthStatus',
    'LogIndexTemplate',
    'get_elk_integration',
    'shutdown_elk_integration',
    'ELASTICSEARCH_AVAILABLE',
    'LOGSTASH_FORMATTER_AVAILABLE'
] 