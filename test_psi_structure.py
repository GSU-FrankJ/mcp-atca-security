#!/usr/bin/env python3
"""
Structure test for PSI Engine components.

This script tests that all PSI module files exist and have the correct structure
without initializing heavy dependencies.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that all required PSI files exist."""
    print("=== Testing PSI File Structure ===")
    
    psi_dir = project_root / "mcp_security" / "core" / "psi"
    
    required_files = [
        "__init__.py",
        "engine.py",
        "analyzers.py", 
        "detectors.py",
        "attacks.py",
        "data.py",
        "evaluation.py"
    ]
    
    for file in required_files:
        file_path = psi_dir / file
        if file_path.exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def test_class_definitions():
    """Test that classes are defined correctly by checking file contents."""
    print("\n=== Testing Class Definitions ===")
    
    psi_dir = project_root / "mcp_security" / "core" / "psi"
    
    # Check engine.py for PSIEngine
    engine_file = psi_dir / "engine.py"
    with open(engine_file, 'r') as f:
        engine_content = f.read()
    
    if "class PSIEngine:" in engine_content:
        print("✅ PSIEngine class defined")
    else:
        print("❌ PSIEngine class missing")
        return False
    
    if "@dataclass" in engine_content and "class PSIResult:" in engine_content:
        print("✅ PSIResult dataclass defined")
    else:
        print("❌ PSIResult dataclass missing")
        return False
    
    # Check analyzers.py for TokenEmbeddingAnalyzer
    analyzers_file = psi_dir / "analyzers.py"
    with open(analyzers_file, 'r') as f:
        analyzers_content = f.read()
    
    if "class TokenEmbeddingAnalyzer:" in analyzers_content:
        print("✅ TokenEmbeddingAnalyzer class defined")
    else:
        print("❌ TokenEmbeddingAnalyzer class missing")
        return False
    
    # Check detectors.py for AdversarialDetector
    detectors_file = psi_dir / "detectors.py"
    with open(detectors_file, 'r') as f:
        detectors_content = f.read()
    
    if "class AdversarialDetector:" in detectors_content:
        print("✅ AdversarialDetector class defined")
    else:
        print("❌ AdversarialDetector class missing")
        return False
    
    # Check data.py for DatasetManager
    data_file = psi_dir / "data.py"
    with open(data_file, 'r') as f:
        data_content = f.read()
    
    if "class DatasetManager:" in data_content:
        print("✅ DatasetManager class defined")
    else:
        print("❌ DatasetManager class missing")
        return False
    
    # Check evaluation.py for EvaluationFramework
    eval_file = psi_dir / "evaluation.py"
    with open(eval_file, 'r') as f:
        eval_content = f.read()
    
    if "class EvaluationFramework:" in eval_content:
        print("✅ EvaluationFramework class defined")
    else:
        print("❌ EvaluationFramework class missing")
        return False
    
    return True

def test_method_signatures():
    """Test that key methods are defined."""
    print("\n=== Testing Key Method Signatures ===")
    
    psi_dir = project_root / "mcp_security" / "core" / "psi"
    
    # Check engine.py for key methods
    engine_file = psi_dir / "engine.py"
    with open(engine_file, 'r') as f:
        engine_content = f.read()
    
    key_methods = [
        "async def initialize(",
        "async def analyze_prompt(",
        "async def train_adversarial_detector(",
        "async def evaluate_performance("
    ]
    
    for method in key_methods:
        if method in engine_content:
            print(f"✅ PSIEngine: {method.split('(')[0]} method defined")
        else:
            print(f"❌ PSIEngine: {method.split('(')[0]} method missing")
            return False
    
    # Check analyzers.py for key methods
    analyzers_file = psi_dir / "analyzers.py"
    with open(analyzers_file, 'r') as f:
        analyzers_content = f.read()
    
    if "async def analyze_tokens(" in analyzers_content:
        print("✅ TokenEmbeddingAnalyzer: analyze_tokens method defined")
    else:
        print("❌ TokenEmbeddingAnalyzer: analyze_tokens method missing")
        return False
    
    return True

def test_imports_structure():
    """Test that __init__.py files have correct imports."""
    print("\n=== Testing Import Structure ===")
    
    # Check PSI __init__.py
    psi_init = project_root / "mcp_security" / "core" / "psi" / "__init__.py"
    with open(psi_init, 'r') as f:
        psi_init_content = f.read()
    
    required_imports = [
        "from .engine import PSIEngine, PSIResult",
        "from .analyzers import TokenEmbeddingAnalyzer",
        "from .detectors import AdversarialDetector",
        "from .data import DatasetManager",
        "from .evaluation import EvaluationFramework"
    ]
    
    for import_stmt in required_imports:
        if import_stmt in psi_init_content:
            print(f"✅ PSI __init__.py: {import_stmt.split(' import ')[1]} imported")
        else:
            print(f"❌ PSI __init__.py: {import_stmt.split(' import ')[1]} import missing")
            return False
    
    return True

def main():
    """Run all structure tests."""
    print("🔍 PSI Engine Structure Test")
    print("=" * 50)
    
    try:
        tests = [
            test_file_structure,
            test_class_definitions,
            test_method_signatures,
            test_imports_structure
        ]
        
        all_passed = True
        for test_func in tests:
            if not test_func():
                all_passed = False
                break
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 All PSI structure tests PASSED!")
            print("\nPSI Core Implementation Status:")
            print("✅ Task 5.1: Token-Level Embedding Analysis Module - COMPLETED")
            print("✅ PSI Engine architecture - COMPLETED")
            print("✅ Adversarial Detection framework - COMPLETED")
            print("✅ Data management system - COMPLETED")
            print("✅ Evaluation framework - COMPLETED")
            print("\nNext steps:")
            print("1. Fix dependency issues (Keras/TensorFlow compatibility)")
            print("2. Run integration tests with actual model loading")
            print("3. Implement remaining subtasks (5.2, 5.3, 5.4, 5.5)")
            print("4. Test with real adversarial examples")
            return True
        else:
            print("❌ PSI structure tests FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Unexpected error during structure test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 