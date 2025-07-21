"""
Configuration management for MCP+ATCA Security Defense System.

This module provides centralized configuration management using Pydantic models
for type safety and validation.
"""

import os
from functools import lru_cache
from typing import Optional, List

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env files.
    
    Uses Pydantic BaseSettings for automatic validation and type conversion.
    """
    
    # Security Configuration
    security_api_key: str = Field(..., env="SECURITY_API_KEY")
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    worker_processes: int = Field(default=4, env="WORKER_PROCESSES")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # ML Models Configuration
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", 
        env="EMBEDDING_MODEL"
    )
    anomaly_detection_model: str = Field(
        default="isolation-forest", 
        env="ANOMALY_DETECTION_MODEL"
    )
    models_cache_dir: str = Field(default="./models", env="MODEL_CACHE_DIR")
    
    # Performance Configuration
    max_concurrent_requests: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    security_check_timeout: int = Field(default=150, env="SECURITY_CHECK_TIMEOUT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: str = Field(default="./logs/security.log", env="LOG_FILE")
    
    # Monitoring Configuration
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    
    # OpenTelemetry Configuration
    otel_service_name: str = Field(default="mcp-security", env="OTEL_SERVICE_NAME")
    otel_exporter_otlp_endpoint: Optional[str] = Field(None, env="OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_resource_attributes: str = Field(
        default="service.name=mcp-security,service.version=0.1.0", 
        env="OTEL_RESOURCE_ATTRIBUTES"
    )
    
    # ELK Stack Configuration
    elasticsearch_hosts: List[str] = Field(
        default=["http://localhost:9200"], 
        env="ELASTICSEARCH_HOSTS"
    )
    elasticsearch_username: Optional[str] = Field(None, env="ELASTICSEARCH_USERNAME")
    elasticsearch_password: Optional[str] = Field(None, env="ELASTICSEARCH_PASSWORD")
    logstash_host: str = Field(default="localhost", env="LOGSTASH_HOST")
    logstash_port: int = Field(default=5000, env="LOGSTASH_PORT")
    
    # Log Rotation Configuration
    log_max_bytes: int = Field(default=50_000_000, env="LOG_MAX_BYTES")  # 50MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    log_rotation_when: str = Field(default="midnight", env="LOG_ROTATION_WHEN")
    log_retention_days: int = Field(default=30, env="LOG_RETENTION_DAYS")
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard Python logging levels."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @validator("log_format")
    def validate_log_format(cls, v: str) -> str:
        """Validate log format is supported."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"log_format must be one of {valid_formats}")
        return v.lower()
    
    @validator("elasticsearch_hosts", pre=True)
    def parse_elasticsearch_hosts(cls, v):
        """Parse comma-separated elasticsearch hosts if provided as string."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("security_check_timeout")
    def validate_timeout(cls, v: int) -> int:
        """Ensure security check timeout is reasonable (between 50ms and 10s)."""
        if not 50 <= v <= 10000:
            raise ValueError("security_check_timeout must be between 50 and 10000 milliseconds")
        return v
    
    class Config:
        """Pydantic configuration for Settings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_log_dir(self) -> str:
        """Get the directory path for log files."""
        return os.path.dirname(self.log_file)
    
    def get_log_file_base_name(self) -> str:
        """Get the base name of the log file without extension."""
        return os.path.splitext(os.path.basename(self.log_file))[0]


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings instance.
    
    Uses LRU cache to ensure settings are loaded only once and reused.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings() 