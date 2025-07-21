"""
Main FastAPI application for MCP+ATCA Security Defense System.

This module provides the entry point for the security orchestrator server
that sits between AI clients and MCP tool servers.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .utils.config import get_settings
from .utils.logging import setup_logging
from .orchestrator.security_orchestrator import SecurityOrchestrator


# Configure structured logging
logger = structlog.get_logger(__name__)

# Initialize security bearer token
security = HTTPBearer()

# Global orchestrator instance
orchestrator: SecurityOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown operations.
    
    Handles initialization and cleanup of the security orchestrator
    and associated resources.
    """
    global orchestrator
    
    try:
        # Startup: Initialize security orchestrator
        logger.info("Starting MCP Security Defense System")
        settings = get_settings()
        
        orchestrator = SecurityOrchestrator(settings)
        await orchestrator.initialize()
        
        logger.info("Security orchestrator initialized successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to initialize security orchestrator", error=str(e))
        raise
    finally:
        # Shutdown: Clean up resources
        if orchestrator:
            await orchestrator.shutdown()
        logger.info("MCP Security Defense System shutdown complete")


# Create FastAPI application with lifespan management
app = FastAPI(
    title="MCP+ATCA Security Defense System",
    description="Security layer for protecting AI agents against adversarial tool-calling attacks",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify API token for authentication.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        str: The verified token
        
    Raises:
        HTTPException: If token is invalid
    """
    settings = get_settings()
    
    if credentials.credentials != settings.security_api_key:
        logger.warning("Invalid API key attempted", token_prefix=credentials.credentials[:8])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns:
        dict: Health status and system information
    """
    try:
        # Check orchestrator health
        if orchestrator:
            health_status = await orchestrator.health_check()
        else:
            health_status = {"status": "initializing"}
        
        return {
            "status": "healthy",
            "version": "0.1.0",
            "orchestrator": health_status,
            "timestamp": asyncio.get_event_loop().time(),
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Root endpoint providing basic system information.
    
    Returns:
        dict: Basic system information
    """
    return {
        "message": "MCP+ATCA Security Defense System",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.post("/security/analyze")
async def analyze_request(
    request_data: Dict[str, Any],
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Analyze a request for security threats.
    
    This is the main endpoint that receives requests from AI clients,
    processes them through security modules, and returns analysis results.
    
    Args:
        request_data: The request data to analyze
        token: Verified API token
        
    Returns:
        dict: Security analysis results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        if not orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Security orchestrator not initialized"
            )
        
        logger.info("Processing security analysis request", 
                   request_id=request_data.get("request_id"),
                   prompt_length=len(request_data.get("prompt", "")))
        
        # Process request through security orchestrator
        result = await orchestrator.process_request(request_data)
        
        logger.info("Security analysis completed",
                   request_id=request_data.get("request_id"),
                   decision=result.get("decision"),
                   confidence=result.get("confidence"))
        
        return result
        
    except Exception as e:
        logger.error("Security analysis failed", 
                    request_id=request_data.get("request_id"),
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Security analysis failed: {str(e)}"
        )


@app.post("/security/proxy")
async def proxy_request(
    request_data: Dict[str, Any],
    token: str = Depends(verify_token)
) -> Dict[str, Any]:
    """
    Proxy a request through security analysis to MCP tool servers.
    
    This endpoint provides the full proxy functionality: analyze the request,
    and if safe, forward it to the appropriate MCP tool server.
    
    Args:
        request_data: The request data to proxy
        token: Verified API token
        
    Returns:
        dict: Response from tool server or security block response
        
    Raises:
        HTTPException: If proxy operation fails
    """
    try:
        if not orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Security orchestrator not initialized"
            )
        
        logger.info("Processing proxy request",
                   request_id=request_data.get("request_id"),
                   target_server=request_data.get("target_server"))
        
        # Process request through full security pipeline
        result = await orchestrator.proxy_request(request_data)
        
        logger.info("Proxy request completed",
                   request_id=request_data.get("request_id"),
                   success=result.get("success"))
        
        return result
        
    except Exception as e:
        logger.error("Proxy request failed",
                    request_id=request_data.get("request_id"),
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Proxy request failed: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """
    Get security metrics for monitoring and analytics.
    
    Returns:
        dict: Current security metrics
    """
    try:
        if not orchestrator:
            return {"status": "orchestrator_not_initialized"}
        
        metrics = await orchestrator.get_metrics()
        return metrics
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Get configuration
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "mcp_security.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        workers=1 if settings.reload else settings.worker_processes,
    ) 