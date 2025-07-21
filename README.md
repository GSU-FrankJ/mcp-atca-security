# MCP+ATCA Security Defense System

🛡️ **Security Defense System for Modular Context Protocol against Adversarial Tool-Calling Attacks**

A comprehensive security layer that protects AI agents and LLM applications utilizing tool-calling capabilities against sophisticated adversarial attacks including prompt injection, malicious tool misuse, and semantic context exploits.

## 🎯 **Overview**

This system integrates network security-inspired mechanisms into the LLM+Tool ecosystem to enhance resilience against:
- **Prompt Injection Attacks**: Attempts to hijack tool selection or execution
- **Tool Misuse**: Abnormal invocation patterns that may indicate compromise  
- **Semantic Context Exploits**: Crafted prompts designed to manipulate model behavior

## 🏗️ **Architecture**

```
AI Client (Claude/ChatGPT) → Security Orchestrator → MCP Tool Servers
                                     ↓
                           [PSI + TIADS + PES + PIFF]
                                     ↓
                           Security Dashboard & Logging
```

### Core Components

- **🔍 PSI (Prompt Semantics Inspection)**: Token-level embedding analysis for anomaly detection
- **🚨 TIADS (Tool Invocation Anomaly Detection System)**: ML-based behavioral analysis  
- **📦 PES (Prompt Execution Sandbox)**: Safe simulation environment for tool execution
- **🎯 PIFF (Prompt Injection Fuzzing Framework)**: Adversarial testing and vulnerability discovery
- **🎛️ Security Orchestrator**: Central coordination service managing all defense modules

## 🚀 **Quick Start**

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Poetry 1.4+
- Redis (for caching)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd mcp-atca-security

# Install dependencies with Poetry
poetry install

# Set up pre-commit hooks
poetry run pre-commit install

# Start development environment
docker-compose up -d

# Run the security server
poetry run uvicorn mcp_security.main:app --reload --port 8000
```

### Basic Usage

```python
from mcp_security.client import SecurityClient

# Initialize the security client
client = SecurityClient(
    server_url="http://localhost:8000",
    api_key="your-api-key"
)

# Secure request processing
response = await client.secure_request(
    prompt="Analyze this data using the SQL tool",
    tools=["sql_query", "data_analysis"]
)
```

## 📊 **Performance Targets**

- ⚡ **Sub-200ms processing**: All security checks complete within 200ms
- 🎯 **≥90% detection rate**: For prompt injection attacks in test sets
- 📉 **≤5% false positive rate**: In anomaly detection across legitimate use cases  
- 🔄 **99.9% uptime**: With automatic failover capabilities
- 📈 **>95% accuracy**: Tool-call accuracy maintained after defense integration

## 🧪 **Development**

### Project Structure

```
mcp_security/
├── core/                   # Core security modules
│   ├── psi/               # Prompt Semantics Inspection
│   ├── tiads/             # Tool Invocation Anomaly Detection
│   ├── pes/               # Prompt Execution Sandbox
│   └── piff/              # Prompt Injection Fuzzing Framework
├── orchestrator/          # Security Orchestrator
├── integrations/          # MCP protocol adapters
├── dashboard/             # Security monitoring dashboard
├── utils/                 # Shared utilities
└── tests/                 # Test suites
```

### Running Tests

```bash
# Run all tests with coverage
poetry run pytest

# Run specific test categories
poetry run pytest tests/core/test_psi.py
poetry run pytest tests/integration/

# Run security scans
poetry run bandit -r mcp_security/
poetry run safety check
```

### Code Quality

```bash
# Format code
poetry run black mcp_security/
poetry run isort mcp_security/

# Lint code
poetry run flake8 mcp_security/
poetry run mypy mcp_security/

# Run pre-commit checks
poetry run pre-commit run --all-files
```

## 🔧 **Configuration**

### Environment Variables

Create a `.env` file:

```bash
# Security Configuration
SECURITY_API_KEY=your-secret-key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/mcpsecurity

# ML Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ANOMALY_DETECTION_MODEL=isolation-forest

# Performance Tuning
MAX_CONCURRENT_REQUESTS=100
CACHE_TTL_SECONDS=300
SECURITY_CHECK_TIMEOUT=150
```

### Docker Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f security-server

# Rebuild after changes
docker-compose build --no-cache
```

## 📈 **Monitoring & Analytics**

Access the security dashboard at `http://localhost:8000/dashboard` to monitor:

- 📊 Real-time threat detection
- 📈 Security metrics and KPIs  
- 🚨 Alert management
- 📋 Audit logs and forensics
- 🔧 Configuration management

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the full test suite: `poetry run pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Guidelines

- 📝 Write clear, commented code following PEP 8
- ✅ Add tests for new functionality (maintain >90% coverage)
- 📚 Update documentation for API changes
- 🔒 Run security scans before committing
- 🏷️ Use semantic commit messages

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- 📖 **Documentation**: [Full documentation](docs/)
- 🐛 **Bug Reports**: [GitHub Issues](issues/)
- 💬 **Discussions**: [GitHub Discussions](discussions/)
- 📧 **Contact**: security@your-org.com

## 🎯 **Roadmap**

- [x] Core architecture design
- [x] PSI engine implementation  
- [ ] TIADS ML model training
- [ ] PES sandbox environment
- [ ] PIFF fuzzing framework
- [ ] Production deployment
- [ ] Community release

---

**⚠️ Security Notice**: This is research software. Thoroughly test in non-production environments before deployment. 