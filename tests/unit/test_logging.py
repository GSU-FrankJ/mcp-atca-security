"""
Unit tests for the logging infrastructure.

Tests cover all aspects of the logging system including sensitive data redaction,
context enrichment, structured logging, and performance monitoring.
"""

import json
import logging
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mcp_security.utils.logging import (
    SensitiveDataRedactor,
    SecurityLogger,
    LoggingService,
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    get_request_context,
    log_performance,
)
from mcp_security.utils.config import Settings


class TestSensitiveDataRedactor:
    """Test the sensitive data redaction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.redactor = SensitiveDataRedactor()
    
    def test_credit_card_redaction(self):
        """Test credit card number redaction."""
        test_cases = [
            ("My credit card is 4532-1234-5678-9012", "My credit card is ****-****-****-9012"),
            ("Card: 4532 1234 5678 9012", "Card: ****-****-****-9012"),
            ("Number: 4532123456789012", "Number: ****-****-****-9012"),
        ]
        
        for original, expected in test_cases:
            result = self.redactor.redact_message(original)
            assert result == expected
    
    def test_ssn_redaction(self):
        """Test SSN redaction."""
        message = "SSN: 123-45-6789"
        result = self.redactor.redact_message(message)
        assert result == "SSN: ***-**-6789"
    
    def test_email_redaction(self):
        """Test email redaction."""
        message = "Contact: john.doe@example.com"
        result = self.redactor.redact_message(message)
        assert result == "Contact: j***@example.com"
    
    def test_password_redaction(self):
        """Test password field redaction."""
        test_cases = [
            ("password=secret123", "password=[REDACTED]"),
            ("pwd: mypassword", "pwd=[REDACTED]"),
            ('pass="test123"', "pass=[REDACTED]"),
        ]
        
        for original, expected in test_cases:
            result = self.redactor.redact_message(original)
            assert result == expected
    
    def test_api_key_redaction(self):
        """Test API key redaction."""
        message = "api_key=test_mock_key_abcdefghijklmnop"
        result = self.redactor.redact_message(message)
        assert "test_mock***" in result
    
    def test_jwt_redaction(self):
        """Test JWT token redaction."""
        jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        message = f"Token: {jwt_token}"
        result = self.redactor.redact_message(message)
        assert result == "Token: [REDACTED_JWT]"
    
    def test_dict_redaction(self):
        """Test dictionary redaction."""
        data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "test_mock_key_abcdefghijklmnop",
            "user_data": {
                "email": "test@example.com",
                "ssn": "123-45-6789",
            },
            "safe_data": "this is safe",
        }
        
        result = self.redactor.redact_dict(data)
        
        assert result["username"] == "testuser"
        assert result["password"] == "[REDACTED]"
        assert result["safe_data"] == "this is safe"
        assert result["user_data"]["email"] == "t***@example.com"
        assert result["user_data"]["ssn"] == "***-**-6789"
    
    def test_nested_list_redaction(self):
        """Test redaction in nested lists."""
        data = {
            "items": [
                {"password": "secret1"},
                "normal string",
                {"user": "test@example.com"},
            ]
        }
        
        result = self.redactor.redact_dict(data)
        
        assert result["items"][0]["password"] == "[REDACTED]"
        assert result["items"][1] == "normal string"
        assert result["items"][2]["user"] == "t***@example.com"


class TestSecurityLogger:
    """Test the SecurityLogger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.logger = SecurityLogger("test_logger")
    
    def test_basic_logging_levels(self):
        """Test all logging levels work correctly."""
        with patch.object(self.logger._logger, 'debug') as mock_debug:
            self.logger.debug("Debug message", extra_data="test")
            mock_debug.assert_called_once()
        
        with patch.object(self.logger._logger, 'info') as mock_info:
            self.logger.info("Info message", extra_data="test")
            mock_info.assert_called_once()
        
        with patch.object(self.logger._logger, 'warning') as mock_warning:
            self.logger.warning("Warning message", extra_data="test")
            mock_warning.assert_called_once()
        
        with patch.object(self.logger._logger, 'error') as mock_error:
            self.logger.error("Error message", extra_data="test")
            mock_error.assert_called_once()
        
        with patch.object(self.logger._logger, 'critical') as mock_critical:
            self.logger.critical("Critical message", extra_data="test")
            mock_critical.assert_called_once()
    
    def test_context_enrichment(self):
        """Test context data is properly added to logs."""
        # Set thread-local context
        self.logger.set_context(request_id="req_123", user_id="user_456")
        
        # Set request context
        set_request_context(session_id="sess_789", ip_address="192.168.1.1")
        
        with patch.object(self.logger._logger, 'info') as mock_info:
            self.logger.info("Test message")
            
            # Verify the call was made with context data
            call_args = mock_info.call_args
            log_data = call_args[1]  # kwargs
            
            assert "context" in log_data
            context = log_data["context"]
            assert context["request_id"] == "req_123"
            assert context["user_id"] == "user_456"
            assert context["session_id"] == "sess_789"
            assert context["ip_address"] == "192.168.1.1"
        
        # Clean up
        self.logger.clear_context()
        clear_request_context()
    
    def test_sensitive_data_redaction_in_logs(self):
        """Test that sensitive data is redacted from log messages."""
        with patch.object(self.logger._logger, 'info') as mock_info:
            self.logger.info(
                "User login with password=secret123",
                user_data={"password": "secret123", "email": "test@example.com"}
            )
            
            call_args = mock_info.call_args
            log_data = call_args[1]
            
            # Check message redaction
            assert "password=[REDACTED]" in log_data["message"]
            
            # Check user_data redaction
            assert log_data["user_data"]["password"] == "[REDACTED]"
            assert log_data["user_data"]["email"] == "t***@example.com"
    
    def test_security_event_logging(self):
        """Test security event logging with proper categorization."""
        with patch.object(self.logger._logger, 'warning') as mock_warning:
            self.logger.security_event(
                event_type="authentication_failure",
                message="Failed login attempt",
                severity="WARNING",
                user_id="user_123",
                ip_address="192.168.1.100",
                resource="/api/login",
                action="login",
                attempts=3
            )
            
            call_args = mock_warning.call_args
            log_data = call_args[1]
            
            assert log_data["event_type"] == "authentication_failure"
            assert log_data["security_category"] == "security_event"
            assert log_data["severity"] == "WARNING"
            assert log_data["user_id"] == "user_123"
            assert log_data["ip_address"] == "192.168.1.100"
            assert log_data["resource"] == "/api/login"
            assert log_data["action"] == "login"
            assert log_data["attempts"] == 3
    
    def test_log_data_structure(self):
        """Test that log data has the expected structure."""
        with patch.object(self.logger._logger, 'info') as mock_info:
            self.logger.info("Test message", custom_field="custom_value")
            
            call_args = mock_info.call_args
            log_data = call_args[1]
            
            # Check required fields
            assert "timestamp" in log_data
            assert "level" in log_data
            assert "logger" in log_data
            assert "message" in log_data
            assert "thread_id" in log_data
            assert "process_id" in log_data
            
            # Check custom field
            assert log_data["custom_field"] == "custom_value"
            
            # Check values
            assert log_data["level"] == "INFO"
            assert log_data["logger"] == "test_logger"
            assert log_data["message"] == "Test message"


class TestLoggingService:
    """Test the LoggingService singleton."""
    
    def test_singleton_pattern(self):
        """Test that LoggingService implements singleton pattern."""
        service1 = LoggingService()
        service2 = LoggingService()
        
        assert service1 is service2
    
    def test_logger_creation_and_reuse(self):
        """Test that loggers are created and reused correctly."""
        service = LoggingService()
        
        logger1 = service.get_logger("test_module")
        logger2 = service.get_logger("test_module")
        logger3 = service.get_logger("other_module")
        
        assert logger1 is logger2
        assert logger1 is not logger3
        assert isinstance(logger1, SecurityLogger)
        assert isinstance(logger3, SecurityLogger)
    
    @patch('mcp_security.utils.logging.Path.mkdir')
    def test_setup_creates_log_directory(self, mock_mkdir):
        """Test that setup creates log directory."""
        # Create a mock settings object
        settings = Mock()
        settings.get_log_dir.return_value = "/test/logs"
        settings.log_format = "json"
        settings.log_level = "INFO"
        settings.log_file = "/test/logs/test.log"
        settings.log_max_bytes = 50000000
        settings.log_backup_count = 5
        settings.debug = False
        
        service = LoggingService()
        service._setup_complete = False  # Reset setup state
        service.setup(settings)
        
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestContextFunctions:
    """Test request context management functions."""
    
    def test_request_context_management(self):
        """Test setting, getting, and clearing request context."""
        # Initially empty
        assert get_request_context() == {}
        
        # Set context
        set_request_context(request_id="req_123", user_id="user_456")
        context = get_request_context()
        assert context["request_id"] == "req_123"
        assert context["user_id"] == "user_456"
        
        # Update context
        set_request_context(session_id="sess_789")
        context = get_request_context()
        assert context["request_id"] == "req_123"  # Still there
        assert context["user_id"] == "user_456"    # Still there
        assert context["session_id"] == "sess_789"  # Added
        
        # Clear context
        clear_request_context()
        assert get_request_context() == {}
    
    def test_context_isolation_between_threads(self):
        """Test that context is isolated between threads."""
        contexts = {}
        
        def thread_function(thread_id):
            set_request_context(thread_id=thread_id)
            time.sleep(0.1)  # Allow other threads to run
            contexts[thread_id] = get_request_context()
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=thread_function, args=(f"thread_{i}",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Each thread should have its own context
        assert contexts["thread_0"]["thread_id"] == "thread_0"
        assert contexts["thread_1"]["thread_id"] == "thread_1"
        assert contexts["thread_2"]["thread_id"] == "thread_2"


class TestPerformanceLogging:
    """Test the performance logging decorator."""
    
    def test_successful_function_execution(self):
        """Test performance logging for successful function execution."""
        mock_logger = Mock(spec=SecurityLogger)
        
        @log_performance(mock_logger)
        def test_function(arg1, arg2=None):
            return f"result_{arg1}_{arg2}"
        
        result = test_function("test", arg2="value")
        
        assert result == "result_test_value"
        
        # Check that debug and info were called
        assert mock_logger.debug.called
        assert mock_logger.info.called
        
        # Check debug call (function start)
        debug_call = mock_logger.debug.call_args
        assert "Starting function execution" in debug_call[0][0]
        assert debug_call[1]["function"].endswith("test_function")
        assert debug_call[1]["args_count"] == 1
        assert "arg2" in debug_call[1]["kwargs_keys"]
        
        # Check info call (function completion)
        info_call = mock_logger.info.call_args
        assert "Function execution completed" in info_call[0][0]
        assert info_call[1]["success"] is True
        assert "execution_time_ms" in info_call[1]
    
    def test_failed_function_execution(self):
        """Test performance logging for failed function execution."""
        mock_logger = Mock(spec=SecurityLogger)
        
        @log_performance(mock_logger)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Check that error was called
        assert mock_logger.error.called
        
        # Check error call
        error_call = mock_logger.error.call_args
        assert "Function execution failed" in error_call[0][0]
        assert error_call[1]["success"] is False
        assert error_call[1]["error_type"] == "ValueError"
        assert error_call[1]["error_message"] == "Test error"
        assert "execution_time_ms" in error_call[1]
    
    def test_performance_logging_without_logger(self):
        """Test performance logging creates logger automatically."""
        @log_performance()
        def test_function():
            return "test_result"
        
        # Should not raise an exception
        result = test_function()
        assert result == "test_result"


class TestIntegration:
    """Integration tests for the complete logging system."""
    
    def test_end_to_end_logging_flow(self):
        """Test complete logging flow from setup to log output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            # Create test settings
            settings = Mock()
            settings.get_log_dir.return_value = str(temp_dir)
            settings.log_format = "json"
            settings.log_level = "INFO"
            settings.log_file = str(log_file)
            settings.log_max_bytes = 50000000
            settings.log_backup_count = 5
            settings.debug = True  # Enable console logging
            
            # Setup logging
            setup_logging(settings)
            
            # Get logger and log some messages
            logger = get_logger("integration_test")
            
            # Set context
            set_request_context(request_id="req_integration_test")
            logger.set_context(component="test_component")
            
            # Log messages with sensitive data
            logger.info(
                "User authentication with password=secret123",
                user_data={
                    "username": "testuser",
                    "password": "secret123",
                    "email": "test@example.com"
                }
            )
            
            logger.security_event(
                event_type="data_access",
                message="User accessed sensitive data",
                severity="INFO",
                user_id="user_123",
                resource="/api/sensitive"
            )
            
            # Check that log file was created and contains expected content
            assert log_file.exists()
            
            # Read and verify log content
            log_content = log_file.read_text()
            assert "integration_test" in log_content
            assert "password=[REDACTED]" in log_content
            assert "t***@example.com" in log_content
            assert "security_event" in log_content
            
            # Clean up
            clear_request_context()
            logger.clear_context()
    
    def test_logging_performance_impact(self):
        """Test that logging doesn't significantly impact performance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "perf_test.log"
            
            settings = Mock()
            settings.get_log_dir.return_value = str(temp_dir)
            settings.log_format = "json"
            settings.log_level = "INFO"
            settings.log_file = str(log_file)
            settings.log_max_bytes = 50000000
            settings.log_backup_count = 5
            settings.debug = False
            
            setup_logging(settings)
            logger = get_logger("performance_test")
            
            # Measure logging performance
            start_time = time.time()
            
            for i in range(1000):
                logger.info(f"Performance test message {i}", iteration=i)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should be able to log 1000 messages in reasonable time (< 1 second)
            assert total_time < 1.0
            
            # Verify all messages were logged
            log_content = log_file.read_text()
            assert "Performance test message 0" in log_content
            assert "Performance test message 999" in log_content 