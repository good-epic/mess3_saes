
import sys
import os

# Mock simplexity to ensure it's not imported
class MockSimplexity:
    def __getattr__(self, name):
        raise ImportError(f"Attempted to import simplexity.{name}")

sys.modules["simplexity"] = MockSimplexity()

print("Testing imports for analyze_real_saes.py...")
try:
    # We don't import the main module directly because it runs main() if not careful, 
    # but let's just try to import the dependencies we changed.
    
    from aanet_pipeline.extrema import compute_diffusion_extrema
    print("Imported compute_diffusion_extrema successfully.")
    
    from aanet_pipeline.training import train_aanet_model
    print("Imported train_aanet_model successfully.")
    
    from aanet_pipeline.cluster_summary import AAnetDescriptor
    print("Imported AAnetDescriptor successfully.")
    
    from real_data_utils import RealDataSampler
    print("Imported RealDataSampler successfully.")
    
    # Finally try to import the script itself (as a module)
    # We need to make sure it doesn't run main
    import real_data_tests.analyze_real_saes
    print("Imported analyze_real_saes successfully.")
    
    print("VERIFICATION PASSED: No simplexity dependency found.")
    
except ImportError as e:
    print(f"VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"VERIFICATION FAILED with error: {e}")
    import traceback
    traceback.print_exc()
