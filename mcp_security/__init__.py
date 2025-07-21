"""
MCP+ATCA Security Defense System

A comprehensive security layer for protecting AI agents and LLM applications
against adversarial tool-calling attacks.
"""

__version__ = "0.1.0"
__author__ = "MCP Security Team"

# Core security modules - lazy import to avoid dependency issues
# from .core import psi, tiads, pes, piff
from .orchestrator import SecurityOrchestrator
from .integrations import MCPAdapter

__all__ = [
    # PSI Engine components (commented out for now)
    # "psi",
    # "tiads", 
    # "pes",
    # "piff",
    "SecurityOrchestrator",
    "MCPAdapter",
] 