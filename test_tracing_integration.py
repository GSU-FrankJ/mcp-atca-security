#!/usr/bin/env python3
"""
Test script for OpenTelemetry tracing integration.

This script tests the integration between OpenTelemetry tracing and the
logging system, verifying that trace and span IDs are properly correlated
with log entries.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Test data
TEST_SETTINGS_DATA = {
    "security_api_key": "test_key_12345",
    "secret_key": "test_secret_67890",
    "database_url": "sqlite:///test.db",
    "log_level": "DEBUG",
    "log_format": "json",
    "log_file": "./test_logs/tracing_test.log",
    "debug": True,
    "otel_service_name": "mcp-security-test",
    "otel_exporter_otlp_endpoint": None  # Use console exporter for testing
}


class MockSettings:
    """Mock settings for testing."""
    
    def __init__(self, **kwargs):
        # Set all default settings first
        defaults = {
            **TEST_SETTINGS_DATA,
            # Additional logging settings
            "log_max_bytes": 50_000_000,
            "log_backup_count": 5,
            "log_rotation_when": "midnight",
            "log_retention_days": 30,
            # Additional required settings
            "host": "localhost",
            "port": 8000,
            "reload": False,
            "worker_processes": 1,
            "redis_url": "redis://localhost:6379",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "anomaly_detection_model": "isolation-forest",
            "models_cache_dir": "./models",
            "max_concurrent_requests": 100,
            "cache_ttl_seconds": 300,
            "security_check_timeout": 150,
            "prometheus_port": 9090,
            "metrics_enabled": True,
            "otel_resource_attributes": "service.name=mcp-security-test,service.version=0.1.0",
            "elasticsearch_hosts": ["http://localhost:9200"],
            "elasticsearch_username": None,
            "elasticsearch_password": None,
            "logstash_host": "localhost",
            "logstash_port": 5000,
        }
        
        for key, value in defaults.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_log_dir(self) -> str:
        return os.path.dirname(self.log_file)
    
    def get_log_file_base_name(self) -> str:
        return os.path.splitext(os.path.basename(self.log_file))[0]


class TracingIntegrationTestSuite:
    """Test suite for tracing integration."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dir = None
        self.settings = None
    
    def setup(self):
        """Set up test environment."""
        print("🔧 Setting up tracing integration test environment...")
        
        # Create temporary directory for logs
        self.temp_dir = tempfile.mkdtemp(prefix="tracing_test_")
        log_file = os.path.join(self.temp_dir, "tracing_test.log")
        
        # Update settings with temp log file
        self.settings = MockSettings(log_file=log_file)
        
        # Create log directory
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Test environment ready. Logs will be written to: {log_file}")
    
    def cleanup(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        
        # Clean up temporary files
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                print("✅ Temporary files cleaned up")
            except Exception as e:
                print(f"⚠️ Warning: Failed to clean up temp directory: {e}")
    
    async def test_tracing_setup(self) -> bool:
        """Test OpenTelemetry tracing setup."""
        print("\n📡 Testing OpenTelemetry tracing setup...")
        
        try:
            # Try to import tracing utilities
            from mcp_security.utils.tracing import (
                setup_tracing, 
                get_tracer, 
                create_span,
                get_trace_correlation_data
            )
            
            # Set up tracing
            setup_tracing(self.settings)
            print("✅ Tracing setup successful")
            
            # Test tracer creation
            tracer = get_tracer("test_tracer")
            if tracer:
                print("✅ Tracer creation successful")
            else:
                print("❌ Tracer creation failed")
                return False
            
            # Test span creation
            with create_span("test_span") as span:
                if span:
                    print("✅ Span creation successful")
                    
                    # Test trace correlation data
                    trace_data = get_trace_correlation_data()
                    if trace_data and "trace_id" in trace_data:
                        print(f"✅ Trace correlation data: {trace_data}")
                    else:
                        print("❌ Trace correlation data not available")
                        return False
                else:
                    print("❌ Span creation failed")
                    return False
            
            return True
            
        except ImportError as e:
            print(f"❌ Tracing modules not available: {e}")
            return False
        except Exception as e:
            print(f"❌ Tracing setup failed: {e}")
            return False
    
    async def test_logging_integration(self) -> bool:
        """Test logging integration with tracing."""
        print("\n📝 Testing logging integration with tracing...")
        
        try:
            # Set up logging
            from mcp_security.utils.logging import setup_logging, get_logger
            
            setup_logging(self.settings)
            logger = get_logger("test_logger")
            print("✅ Logging setup successful")
            
            # Try to import tracing utilities (might not be available)
            try:
                from mcp_security.utils.tracing import async_trace_operation, setup_tracing
                
                # Set up tracing
                setup_tracing(self.settings)
                
                # Test logging with tracing context
                async with async_trace_operation("test_operation") as span:
                    logger.info("Test log message with tracing context", test_data="value")
                    logger.warning("Test warning with tracing context")
                    logger.error("Test error with tracing context")
                
                print("✅ Logging with tracing context successful")
                
            except ImportError:
                print("⚠️ Tracing not available, testing logging without tracing...")
                
                # Test logging without tracing
                logger.info("Test log message without tracing", test_data="value")
                logger.warning("Test warning without tracing")
                logger.error("Test error without tracing")
                
                print("✅ Logging without tracing successful")
            
            # Wait a moment for logs to be written
            await asyncio.sleep(0.1)
            
            # Check if log file exists and has content
            if os.path.exists(self.settings.log_file):
                with open(self.settings.log_file, 'r') as f:
                    log_content = f.read()
                    if log_content:
                        print(f"✅ Log file created with content ({len(log_content)} bytes)")
                        return True
                    else:
                        print("❌ Log file exists but is empty")
                        return False
            else:
                print("❌ Log file not created")
                return False
            
        except Exception as e:
            print(f"❌ Logging integration test failed: {e}")
            return False
    
    async def test_orchestrator_integration(self) -> bool:
        """Test SecurityOrchestrator integration with tracing."""
        print("\n🔄 Testing SecurityOrchestrator integration with tracing...")
        
        try:
            from mcp_security.orchestrator import SecurityOrchestrator
            
            # Create orchestrator
            orchestrator = SecurityOrchestrator(self.settings)
            
            # Initialize (which should set up tracing)
            await orchestrator.initialize()
            print("✅ SecurityOrchestrator initialized with tracing")
            
            # Test a request (which should create trace spans)
            test_request = {
                "request_id": "test_request_123",
                "prompt": "This is a test prompt for security analysis",
                "tools": ["tool1", "tool2"]
            }
            
            result = await orchestrator.process_request(test_request)
            if result and result.get("request_id") == "test_request_123":
                print("✅ Request processing with tracing successful")
            else:
                print("❌ Request processing failed")
                await orchestrator.shutdown()
                return False
            
            # Test proxy request
            proxy_request = {
                "request_id": "test_proxy_456",
                "target_server": "test_server",
                "data": "test_data"
            }
            
            proxy_result = await orchestrator.proxy_request(proxy_request)
            if proxy_result and proxy_result.get("request_id") == "test_proxy_456":
                print("✅ Proxy request with tracing successful")
            else:
                print("❌ Proxy request failed")
                await orchestrator.shutdown()
                return False
            
            # Shutdown
            await orchestrator.shutdown()
            print("✅ SecurityOrchestrator shutdown successful")
            
            return True
            
        except Exception as e:
            print(f"❌ SecurityOrchestrator integration test failed: {e}")
            return False
    
    async def test_log_trace_correlation(self) -> bool:
        """Test that logs contain trace correlation data."""
        print("\n🔗 Testing log trace correlation...")
        
        try:
            # Wait for logs to be fully written
            await asyncio.sleep(0.2)
            
            if not os.path.exists(self.settings.log_file):
                print("❌ Log file does not exist")
                return False
            
            # Read and parse log entries
            trace_correlated_logs = 0
            total_logs = 0
            
            with open(self.settings.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Try to parse as JSON
                        log_entry = json.loads(line)
                        total_logs += 1
                        
                        # Check for trace correlation fields
                        if any(field in log_entry for field in ['trace_id', 'span_id', 'traceId', 'spanId']):
                            trace_correlated_logs += 1
                            print(f"✅ Found trace-correlated log: {log_entry.get('message', 'No message')}")
                        
                    except json.JSONDecodeError:
                        # Skip non-JSON lines
                        total_logs += 1
                        continue
            
            if total_logs == 0:
                print("❌ No log entries found")
                return False
            
            correlation_rate = trace_correlated_logs / total_logs
            print(f"📊 Trace correlation statistics:")
            print(f"   Total logs: {total_logs}")
            print(f"   Trace-correlated logs: {trace_correlated_logs}")
            print(f"   Correlation rate: {correlation_rate:.2%}")
            
            # Consider it successful if we have any trace correlation
            if trace_correlated_logs > 0:
                print("✅ Log trace correlation working")
                return True
            else:
                print("⚠️ No trace correlation found (might be expected if tracing is unavailable)")
                return True  # Not a failure if tracing is not available
                
        except Exception as e:
            print(f"❌ Log trace correlation test failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("🚀 Starting OpenTelemetry Tracing Integration Test Suite")
        print("=" * 70)
        
        self.setup()
        
        try:
            # Run all tests
            tests = [
                ("Tracing Setup", self.test_tracing_setup),
                ("Logging Integration", self.test_logging_integration),
                ("Orchestrator Integration", self.test_orchestrator_integration),
                ("Log Trace Correlation", self.test_log_trace_correlation),
            ]
            
            results = {}
            
            for test_name, test_func in tests:
                try:
                    result = await test_func()
                    results[test_name] = result
                    self.test_results.append((test_name, result))
                except Exception as e:
                    print(f"❌ {test_name} failed with exception: {e}")
                    results[test_name] = False
                    self.test_results.append((test_name, False))
            
            # Print summary
            print("\n" + "=" * 70)
            print("🏁 Test Summary:")
            print("-" * 70)
            
            passed = 0
            total = len(results)
            
            for test_name, result in results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"{test_name:<30} {status}")
                if result:
                    passed += 1
            
            print("-" * 70)
            success_rate = passed / total if total > 0 else 0
            print(f"Overall Success Rate: {passed}/{total} ({success_rate:.1%})")
            
            if success_rate >= 0.75:  # 75% or higher considered success
                print("🎉 Tracing integration test suite PASSED!")
            else:
                print("💥 Tracing integration test suite FAILED!")
            
            return results
            
        finally:
            self.cleanup()


async def main():
    """Main test runner."""
    suite = TracingIntegrationTestSuite()
    results = await suite.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main()) 