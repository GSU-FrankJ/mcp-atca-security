"""
Security Event Logging for MCP+ATCA Security Defense System.

This module provides comprehensive security event logging with:
- Structured security event taxonomy and categorization
- Threat detection and anomaly logging
- SIEM integration capabilities
- Real-time alerting for critical security events
- Compliance and audit trail management
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field, asdict
import threading
from collections import defaultdict, deque
import time
import socket
import hashlib

from .config import Settings
from .logging import get_logger, SecurityLogger

# Import tracing utilities if available
try:
    from .tracing import get_trace_correlation_data
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    
    def get_trace_correlation_data() -> Dict[str, str]:
        return {}

# Import ELK integration if available
try:
    from .elk import get_elk_integration
    ELK_AVAILABLE = True
except ImportError:
    ELK_AVAILABLE = False
    
    async def get_elk_integration(settings=None):
        return None


class ThreatLevel(IntEnum):
    """Threat severity levels for security events."""
    INFORMATIONAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SecurityEventType(Enum):
    """Categories of security events."""
    # Authentication Events
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGIN_BRUTE_FORCE = "auth.login.brute_force"
    AUTH_TOKEN_EXPIRED = "auth.token.expired"
    AUTH_TOKEN_INVALID = "auth.token.invalid"
    AUTH_PRIVILEGE_ESCALATION = "auth.privilege.escalation"
    
    # Access Control Events
    ACCESS_DENIED = "access.denied"
    ACCESS_GRANTED = "access.granted"
    ACCESS_UNAUTHORIZED = "access.unauthorized"
    ACCESS_POLICY_VIOLATION = "access.policy.violation"
    
    # Input Validation Events
    INPUT_INJECTION_ATTEMPT = "input.injection.attempt"
    INPUT_XSS_ATTEMPT = "input.xss.attempt"
    INPUT_MALFORMED = "input.malformed"
    INPUT_SIZE_VIOLATION = "input.size.violation"
    
    # Tool Calling Security Events
    TOOL_CALL_BLOCKED = "tool.call.blocked"
    TOOL_CALL_ANOMALY = "tool.call.anomaly"
    TOOL_PARAMETER_INJECTION = "tool.parameter.injection"
    TOOL_UNAUTHORIZED_ACCESS = "tool.unauthorized.access"
    TOOL_EXECUTION_FAILURE = "tool.execution.failure"
    
    # Prompt Security Events
    PROMPT_INJECTION_DETECTED = "prompt.injection.detected"
    PROMPT_JAILBREAK_ATTEMPT = "prompt.jailbreak.attempt"
    PROMPT_SENSITIVE_DATA = "prompt.sensitive.data"
    PROMPT_ANOMALY = "prompt.anomaly"
    
    # System Security Events
    SYSTEM_INTRUSION_ATTEMPT = "system.intrusion.attempt"
    SYSTEM_MALWARE_DETECTED = "system.malware.detected"
    SYSTEM_CONFIGURATION_CHANGE = "system.config.change"
    SYSTEM_SERVICE_FAILURE = "system.service.failure"
    
    # Data Security Events
    DATA_BREACH_ATTEMPT = "data.breach.attempt"
    DATA_EXFILTRATION = "data.exfiltration"
    DATA_CORRUPTION = "data.corruption"
    DATA_UNAUTHORIZED_ACCESS = "data.unauthorized.access"
    
    # Network Security Events
    NETWORK_SUSPICIOUS_CONNECTION = "network.suspicious.connection"
    NETWORK_DDoS_ATTEMPT = "network.ddos.attempt"
    NETWORK_PORT_SCAN = "network.port.scan"
    NETWORK_PROTOCOL_VIOLATION = "network.protocol.violation"
    
    # Compliance Events
    COMPLIANCE_VIOLATION = "compliance.violation"
    COMPLIANCE_AUDIT_EVENT = "compliance.audit.event"
    COMPLIANCE_DATA_RETENTION = "compliance.data.retention"


class SecurityEventCategory(Enum):
    """High-level categories for security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    TOOL_SECURITY = "tool_security"
    PROMPT_SECURITY = "prompt_security"
    SYSTEM_SECURITY = "system_security"
    DATA_PROTECTION = "data_protection"
    NETWORK_SECURITY = "network_security"
    COMPLIANCE = "compliance"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class SecurityIndicator:
    """Security indicator or IOC (Indicator of Compromise)."""
    type: str  # e.g., "ip", "domain", "hash", "pattern"
    value: str
    confidence: float  # 0.0 to 1.0
    source: str
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AttackVector:
    """Information about attack vector used."""
    technique: str  # MITRE ATT&CK technique ID
    tactic: str     # MITRE ATT&CK tactic
    description: str
    severity: ThreatLevel
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['severity'] = self.severity.value
        return result


@dataclass
class SecurityEvent:
    """Comprehensive security event structure."""
    # Basic Event Information
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: SecurityEventType = SecurityEventType.SYSTEM_SERVICE_FAILURE
    category: SecurityEventCategory = SecurityEventCategory.SYSTEM_SECURITY
    
    # Severity and Impact
    threat_level: ThreatLevel = ThreatLevel.INFORMATIONAL
    confidence: float = 1.0  # Confidence in the detection (0.0 to 1.0)
    
    # Event Details
    title: str = ""
    description: str = ""
    raw_data: Optional[Dict[str, Any]] = None
    
    # Context Information
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Attack Information
    attack_vectors: List[AttackVector] = field(default_factory=list)
    indicators: List[SecurityIndicator] = field(default_factory=list)
    blocked: bool = False
    mitigation_applied: bool = False
    
    # System Context
    service: str = "mcp-security"
    host: str = field(default_factory=socket.gethostname)
    process_id: int = field(default_factory=lambda: os.getpid() if 'os' in globals() else 0)
    
    # Tracing Information
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Additional Metadata
    tags: Set[str] = field(default_factory=set)
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Add trace correlation if available
        if TRACING_AVAILABLE:
            trace_data = get_trace_correlation_data()
            if trace_data:
                self.trace_id = trace_data.get('trace_id')
                self.span_id = trace_data.get('span_id')
        
        # Ensure tags is a set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
    
    def add_indicator(self, indicator_type: str, value: str, confidence: float = 1.0, 
                     source: str = "mcp-security", description: Optional[str] = None) -> None:
        """Add a security indicator to the event."""
        indicator = SecurityIndicator(
            type=indicator_type,
            value=value,
            confidence=confidence,
            source=source,
            description=description
        )
        self.indicators.append(indicator)
    
    def add_attack_vector(self, technique: str, tactic: str, description: str, 
                         severity: ThreatLevel = ThreatLevel.MEDIUM) -> None:
        """Add an attack vector to the event."""
        vector = AttackVector(
            technique=technique,
            tactic=tactic,
            description=description,
            severity=severity
        )
        self.attack_vectors.append(vector)
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the event."""
        self.tags.add(tag)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert security event to dictionary for serialization."""
        result = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "category": self.category.value,
            "threat_level": self.threat_level.value,
            "threat_level_name": self.threat_level.name,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "raw_data": self.raw_data,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "attack_vectors": [vector.to_dict() for vector in self.attack_vectors],
            "indicators": [indicator.to_dict() for indicator in self.indicators],
            "blocked": self.blocked,
            "mitigation_applied": self.mitigation_applied,
            "service": self.service,
            "host": self.host,
            "process_id": self.process_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tags": list(self.tags),
            "custom_fields": self.custom_fields
        }
        
        return result


class ThreatDetectionEngine:
    """
    Real-time threat detection engine for security events.
    
    Analyzes patterns and behaviors to detect potential threats
    and generate appropriate security events.
    """
    
    def __init__(self, settings: Settings):
        """Initialize threat detection engine."""
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        
        # Pattern detection storage
        self._failed_login_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._request_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._blocked_ips: Set[str] = set()
        
        # Detection thresholds
        self.brute_force_threshold = 5  # Failed attempts
        self.brute_force_window = 300   # Time window in seconds
        self.anomaly_threshold = 0.7    # Anomaly score threshold
        
        # Lock for thread safety
        self._lock = threading.Lock()
    
    def analyze_login_attempt(self, ip: str, username: str, success: bool, 
                            user_agent: Optional[str] = None) -> Optional[SecurityEvent]:
        """
        Analyze login attempt for brute force patterns.
        
        Args:
            ip: Source IP address
            username: Username attempted
            success: Whether login was successful
            user_agent: User agent string
            
        Returns:
            SecurityEvent if threat detected, None otherwise
        """
        with self._lock:
            current_time = time.time()
            key = f"{ip}:{username}"
            
            if not success:
                # Track failed attempt
                self._failed_login_attempts[key].append(current_time)
                
                # Count recent failures
                recent_failures = sum(
                    1 for timestamp in self._failed_login_attempts[key]
                    if current_time - timestamp <= self.brute_force_window
                )
                
                # Check for brute force
                if recent_failures >= self.brute_force_threshold:
                    self._blocked_ips.add(ip)
                    
                    event = SecurityEvent(
                        event_type=SecurityEventType.AUTH_LOGIN_BRUTE_FORCE,
                        category=SecurityEventCategory.AUTHENTICATION,
                        threat_level=ThreatLevel.HIGH,
                        title="Brute Force Login Attack Detected",
                        description=f"Multiple failed login attempts detected from {ip} for user {username}",
                        source_ip=ip,
                        user_agent=user_agent,
                        blocked=True,
                        mitigation_applied=True,
                        raw_data={
                            "username": username,
                            "failed_attempts": recent_failures,
                            "time_window": self.brute_force_window
                        }
                    )
                    
                    event.add_indicator("ip", ip, confidence=0.9, description="Brute force source IP")
                    event.add_indicator("username", username, confidence=0.8, description="Targeted username")
                    event.add_attack_vector("T1110.001", "Credential Access", "Password Brute Force", ThreatLevel.HIGH)
                    event.add_tag("brute-force")
                    event.add_tag("authentication")
                    
                    return event
            
            else:
                # Successful login - check if IP was previously blocked
                if ip in self._blocked_ips:
                    event = SecurityEvent(
                        event_type=SecurityEventType.AUTH_LOGIN_SUCCESS,
                        category=SecurityEventCategory.AUTHENTICATION,
                        threat_level=ThreatLevel.MEDIUM,
                        title="Successful Login from Previously Blocked IP",
                        description=f"Successful login from IP {ip} that was previously blocked for brute force",
                        source_ip=ip,
                        user_agent=user_agent,
                        raw_data={"username": username}
                    )
                    
                    event.add_indicator("ip", ip, confidence=0.7, description="Previously blocked IP")
                    event.add_tag("suspicious-login")
                    event.add_tag("authentication")
                    
                    return event
        
        return None
    
    def analyze_prompt_injection(self, prompt: str, user_id: Optional[str] = None, 
                               confidence: float = 0.8) -> Optional[SecurityEvent]:
        """
        Analyze prompt for injection attempts.
        
        Args:
            prompt: The prompt text to analyze
            user_id: Optional user ID
            confidence: Confidence score of the detection
            
        Returns:
            SecurityEvent if injection detected, None otherwise
        """
        # Simple pattern-based detection (would be replaced with ML model)
        injection_patterns = [
            "ignore previous instructions",
            "disregard the above",
            "forget what you were told",
            "act as if you are",
            "pretend to be",
            "roleplay as",
            "jailbreak",
            "developer mode",
            "admin override"
        ]
        
        prompt_lower = prompt.lower()
        detected_patterns = [pattern for pattern in injection_patterns if pattern in prompt_lower]
        
        if detected_patterns:
            event = SecurityEvent(
                event_type=SecurityEventType.PROMPT_INJECTION_DETECTED,
                category=SecurityEventCategory.PROMPT_SECURITY,
                threat_level=ThreatLevel.HIGH if confidence > 0.8 else ThreatLevel.MEDIUM,
                confidence=confidence,
                title="Prompt Injection Attempt Detected",
                description=f"Potential prompt injection detected with patterns: {', '.join(detected_patterns)}",
                user_id=user_id,
                blocked=True,
                mitigation_applied=True,
                raw_data={
                    "prompt_snippet": prompt[:200],  # First 200 chars
                    "detected_patterns": detected_patterns,
                    "prompt_length": len(prompt)
                }
            )
            
            # Add indicators for each detected pattern
            for pattern in detected_patterns:
                event.add_indicator("pattern", pattern, confidence=confidence, 
                                  description="Prompt injection pattern")
            
            event.add_attack_vector("T1059", "Execution", "Command and Scripting Interpreter", ThreatLevel.HIGH)
            event.add_tag("prompt-injection")
            event.add_tag("llm-security")
            
            return event
        
        return None
    
    def analyze_tool_call_anomaly(self, tool_name: str, parameters: Dict[str, Any], 
                                 user_id: Optional[str] = None, 
                                 execution_time: Optional[float] = None) -> Optional[SecurityEvent]:
        """
        Analyze tool call for anomalies.
        
        Args:
            tool_name: Name of the tool being called
            parameters: Tool parameters
            user_id: Optional user ID
            execution_time: Tool execution time in seconds
            
        Returns:
            SecurityEvent if anomaly detected, None otherwise
        """
        # Check for suspicious parameter values
        suspicious_indicators = []
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Check for path traversal
                if "../" in value or "..\\" in value:
                    suspicious_indicators.append(f"Path traversal in {key}")
                
                # Check for command injection
                if any(cmd in value.lower() for cmd in ["rm -rf", "del /", "format c:", "shutdown", "reboot"]):
                    suspicious_indicators.append(f"Command injection in {key}")
                
                # Check for SQL injection
                if any(sql in value.lower() for sql in ["union select", "drop table", "'; --", "1=1"]):
                    suspicious_indicators.append(f"SQL injection in {key}")
        
        # Check for unusual execution time
        if execution_time and execution_time > 30:  # More than 30 seconds
            suspicious_indicators.append(f"Unusually long execution time: {execution_time}s")
        
        if suspicious_indicators:
            event = SecurityEvent(
                event_type=SecurityEventType.TOOL_CALL_ANOMALY,
                category=SecurityEventCategory.TOOL_SECURITY,
                threat_level=ThreatLevel.HIGH,
                title=f"Suspicious Tool Call: {tool_name}",
                description=f"Anomalous tool call detected: {', '.join(suspicious_indicators)}",
                user_id=user_id,
                blocked=True,
                mitigation_applied=True,
                raw_data={
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "execution_time": execution_time,
                    "suspicious_indicators": suspicious_indicators
                }
            )
            
            event.add_indicator("tool", tool_name, confidence=0.8, description="Suspicious tool usage")
            event.add_attack_vector("T1106", "Execution", "Native API", ThreatLevel.HIGH)
            event.add_tag("tool-security")
            event.add_tag("anomaly")
            
            return event
        
        return None


class SecurityEventLogger:
    """
    Main security event logging service.
    
    Coordinates security event detection, logging, and alerting
    across the entire security defense system.
    """
    
    def __init__(self, settings: Settings):
        """Initialize security event logger."""
        self.settings = settings
        self.logger: SecurityLogger = get_logger(__name__)
        self.threat_detection = ThreatDetectionEngine(settings)
        
        # Event storage and processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_processors: List[Callable] = []
        self._alerting_rules: List[Dict[str, Any]] = []
        
        # Statistics
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._threat_level_counts: Dict[ThreatLevel, int] = defaultdict(int)
        
        # Integration components
        self._elk_integration = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize security event logging system."""
        self.logger.info("Initializing security event logging system")
        
        # Initialize ELK integration if available
        if ELK_AVAILABLE:
            self._elk_integration = await get_elk_integration(self.settings)
            if self._elk_integration:
                self.logger.info("ELK integration available for security events")
        
        # Set up default alerting rules
        self._setup_default_alerting_rules()
        
        # Start event processor
        asyncio.create_task(self._process_events())
        
        self._initialized = True
        self.logger.info("Security event logging system initialized")
        return True
    
    def _setup_default_alerting_rules(self) -> None:
        """Set up default alerting rules for critical events."""
        self._alerting_rules = [
            {
                "name": "Critical Threat Alert",
                "condition": {"threat_level": ThreatLevel.CRITICAL},
                "actions": ["log_alert", "send_notification"],
                "cooldown": 300  # 5 minutes
            },
            {
                "name": "High Threat Alert", 
                "condition": {"threat_level": ThreatLevel.HIGH},
                "actions": ["log_alert"],
                "cooldown": 600  # 10 minutes
            },
            {
                "name": "Brute Force Alert",
                "condition": {"event_type": SecurityEventType.AUTH_LOGIN_BRUTE_FORCE},
                "actions": ["log_alert", "block_ip"],
                "cooldown": 900  # 15 minutes
            }
        ]
    
    async def log_event(self, event: SecurityEvent) -> bool:
        """
        Log a security event.
        
        Args:
            event: SecurityEvent to log
            
        Returns:
            bool: True if logging successful
        """
        if not self._initialized:
            return False
        
        try:
            # Add to queue for processing
            await self._event_queue.put(event)
            
            # Update statistics
            self._event_counts[event.event_type.value] += 1
            self._threat_level_counts[event.threat_level] += 1
            
            # Log to standard logger
            self.logger.security_event(
                event_type=event.event_type.value,
                message=event.title or event.description,
                severity=event.threat_level.name,
                event_id=event.event_id,
                threat_level=event.threat_level.value,
                confidence=event.confidence,
                **event.custom_fields
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            return False
    
    async def _process_events(self) -> None:
        """Process security events from the queue."""
        while True:
            try:
                # Get event from queue
                event = await self._event_queue.get()
                
                # Process event
                await self._process_single_event(event)
                
                # Mark task as done
                self._event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error processing security event: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_event(self, event: SecurityEvent) -> None:
        """Process a single security event."""
        # Send to ELK if available
        if self._elk_integration:
            try:
                await self._elk_integration.log_security_event(event.to_dict())
            except Exception as e:
                self.logger.error(f"Failed to send event to ELK: {e}")
        
        # Check alerting rules
        await self._check_alerting_rules(event)
        
        # Run custom processors
        for processor in self._event_processors:
            try:
                await processor(event)
            except Exception as e:
                self.logger.error(f"Event processor failed: {e}")
    
    async def _check_alerting_rules(self, event: SecurityEvent) -> None:
        """Check if event matches any alerting rules."""
        for rule in self._alerting_rules:
            if self._event_matches_condition(event, rule["condition"]):
                await self._execute_alerting_actions(event, rule)
    
    def _event_matches_condition(self, event: SecurityEvent, condition: Dict[str, Any]) -> bool:
        """Check if event matches alerting condition."""
        for key, value in condition.items():
            if key == "threat_level":
                if event.threat_level != value:
                    return False
            elif key == "event_type":
                if event.event_type != value:
                    return False
            elif key == "category":
                if event.category != value:
                    return False
            # Add more condition types as needed
        
        return True
    
    async def _execute_alerting_actions(self, event: SecurityEvent, rule: Dict[str, Any]) -> None:
        """Execute alerting actions for matched rule."""
        for action in rule["actions"]:
            try:
                if action == "log_alert":
                    self.logger.critical(
                        f"SECURITY ALERT: {rule['name']} - {event.title}",
                        event_id=event.event_id,
                        rule_name=rule["name"],
                        alert_triggered=True
                    )
                elif action == "send_notification":
                    # Implement notification sending (email, Slack, etc.)
                    pass
                elif action == "block_ip":
                    if event.source_ip:
                        # Implement IP blocking logic
                        self.logger.warning(f"IP blocking triggered for {event.source_ip}")
                        
            except Exception as e:
                self.logger.error(f"Failed to execute alerting action {action}: {e}")
    
    def add_event_processor(self, processor: Callable[[SecurityEvent], None]) -> None:
        """Add custom event processor."""
        self._event_processors.append(processor)
    
    def add_alerting_rule(self, rule: Dict[str, Any]) -> None:
        """Add custom alerting rule."""
        self._alerting_rules.append(rule)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get security event statistics."""
        return {
            "total_events": sum(self._event_counts.values()),
            "events_by_type": dict(self._event_counts),
            "events_by_threat_level": {
                level.name: count for level, count in self._threat_level_counts.items()
            },
            "high_priority_events": (
                self._threat_level_counts[ThreatLevel.HIGH] + 
                self._threat_level_counts[ThreatLevel.CRITICAL]
            ),
            "alerting_rules_count": len(self._alerting_rules),
            "processors_count": len(self._event_processors)
        }
    
    async def search_events(self, query: str, time_range: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search security events."""
        if self._elk_integration:
            return await self._elk_integration.search_security_events(query, time_range)
        else:
            self.logger.warning("ELK integration not available for event search")
            return []
    
    async def shutdown(self) -> None:
        """Shutdown security event logging system."""
        self.logger.info("Shutting down security event logging system")
        
        # Wait for queue to be processed
        await self._event_queue.join()
        
        self._initialized = False
        self.logger.info("Security event logging system shutdown complete")


# Global security event logger instance
_security_event_logger: Optional[SecurityEventLogger] = None


async def get_security_event_logger(settings: Optional[Settings] = None) -> Optional[SecurityEventLogger]:
    """
    Get or create the global security event logger instance.
    
    Args:
        settings: Optional settings instance
        
    Returns:
        SecurityEventLogger instance or None if not available
    """
    global _security_event_logger
    
    if _security_event_logger is None and settings:
        _security_event_logger = SecurityEventLogger(settings)
        await _security_event_logger.initialize()
    
    return _security_event_logger


async def log_security_event(event_type: SecurityEventType, 
                           title: str,
                           description: str = "",
                           threat_level: ThreatLevel = ThreatLevel.INFORMATIONAL,
                           **kwargs) -> bool:
    """
    Convenience function to log a security event.
    
    Args:
        event_type: Type of security event
        title: Event title
        description: Event description
        threat_level: Threat severity level
        **kwargs: Additional event data
        
    Returns:
        bool: True if logging successful
    """
    logger = await get_security_event_logger()
    if not logger:
        return False
    
    # Determine category from event type
    category_mapping = {
        "auth": SecurityEventCategory.AUTHENTICATION,
        "access": SecurityEventCategory.AUTHORIZATION,
        "input": SecurityEventCategory.INPUT_VALIDATION,
        "tool": SecurityEventCategory.TOOL_SECURITY,
        "prompt": SecurityEventCategory.PROMPT_SECURITY,
        "system": SecurityEventCategory.SYSTEM_SECURITY,
        "data": SecurityEventCategory.DATA_PROTECTION,
        "network": SecurityEventCategory.NETWORK_SECURITY,
        "compliance": SecurityEventCategory.COMPLIANCE
    }
    
    category = SecurityEventCategory.SYSTEM_SECURITY  # Default
    for prefix, cat in category_mapping.items():
        if event_type.value.startswith(prefix):
            category = cat
            break
    
    event = SecurityEvent(
        event_type=event_type,
        category=category,
        threat_level=threat_level,
        title=title,
        description=description,
        **kwargs
    )
    
    return await logger.log_event(event)


async def shutdown_security_event_logger() -> None:
    """Shutdown the global security event logger instance."""
    global _security_event_logger
    
    if _security_event_logger:
        await _security_event_logger.shutdown()
        _security_event_logger = None


# Import os for process ID
import os

# Export main classes and functions
__all__ = [
    'SecurityEvent',
    'SecurityEventType',
    'SecurityEventCategory', 
    'ThreatLevel',
    'SecurityIndicator',
    'AttackVector',
    'ThreatDetectionEngine',
    'SecurityEventLogger',
    'get_security_event_logger',
    'log_security_event',
    'shutdown_security_event_logger',
    'TRACING_AVAILABLE',
    'ELK_AVAILABLE'
] 