#!/usr/bin/env python3
"""
Basic test script for PSI Engine functionality.

This script tests the core PSI components to ensure they can be imported
and initialized correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Test imports
    print("Testing PSI imports...")
    from mcp_security.core.psi import (
        PSIEngine,
        PSIResult,
        TokenEmbeddingAnalyzer,
        AdversarialDetector,
        DatasetManager,
        EvaluationFramework
    )
    print("✅ PSI imports successful")
    
    # Test basic configuration
    from mcp_security.utils.config import Settings
    print("✅ Settings import successful")
    
    # Create mock settings for testing
    class MockSettings:
        def __init__(self):
            self.psi_embedding_model = 'all-mpnet-base-v2'
            self.psi_window_size = 5
            self.psi_similarity_threshold = 0.7
            self.psi_anomaly_threshold = 0.6
            self.psi_max_tokens = 512
            self.psi_esa_threshold = 0.8
            self.psi_cap_threshold = 0.75
            self.psi_risk_threshold = 0.6
            self.psi_batch_size = 32
            self.psi_performance_target_ms = 200.0
            self.psi_fnr_target = 0.05
            self.psi_data_dir = '.taskmaster/data/psi'
            self.psi_cache_dir = '.taskmaster/cache/psi'
            self.psi_results_dir = '.taskmaster/results/psi'
    
    settings = MockSettings()
    print("✅ Mock settings created")
    
    async def test_basic_functionality():
        """Test basic PSI functionality."""
        print("\n=== Testing PSI Engine Initialization ===")
        
        # Test PSI Engine creation (without full initialization)
        try:
            psi_engine = PSIEngine(settings)
            print("✅ PSI Engine created successfully")
        except Exception as e:
            print(f"❌ PSI Engine creation failed: {e}")
            return False
        
        # Test DatasetManager
        try:
            dataset_manager = DatasetManager(settings)
            print("✅ DatasetManager created successfully")
            
            # Create a sample dataset
            sample_dataset = await dataset_manager.create_sample_dataset()
            print(f"✅ Sample dataset created with {len(sample_dataset)} samples")
            
        except Exception as e:
            print(f"❌ DatasetManager test failed: {e}")
            return False
        
        # Test EvaluationFramework
        try:
            eval_framework = EvaluationFramework(settings)
            print("✅ EvaluationFramework created successfully")
        except Exception as e:
            print(f"❌ EvaluationFramework creation failed: {e}")
            return False
        
        print("\n=== All Basic Tests Passed! ===")
        return True
    
    # Run the test
    print("\nRunning basic functionality tests...")
    success = asyncio.run(test_basic_functionality())
    
    if success:
        print("\n🎉 PSI Engine basic functionality test PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install sentence-transformers faiss-cpu nltk scikit-learn")
        print("2. Run full integration tests")
        print("3. Test with real data")
    else:
        print("\n❌ PSI Engine basic functionality test FAILED!")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPossible issues:")
    print("1. Missing dependencies - run: pip install sentence-transformers faiss-cpu nltk")
    print("2. PYTHONPATH not set correctly")
    print("3. Missing modules in the PSI package")
    sys.exit(1)

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 