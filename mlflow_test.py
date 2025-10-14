#!/usr/bin/env python3
"""
Minimal MLFlow connection test script.
Tests authentication, experiment creation, and logging using mlflow_utils helpers.
"""

import mlflow
import os
import sys
from datetime import datetime
from mlflow_utils import start_or_continue_run, log_script_params

# Mock argparse namespace for testing log_script_params
class MockArgs:
    def __init__(self):
        self.test_param_1 = "value1"
        self.test_param_2 = 42
        self.test_param_3 = 3.14

def test_mlflow_connection():
    """Test MLFlow connection and basic operations"""
    
    # Get credentials from environment
    host = os.getenv('DATABRICKS_HOST')
    token = os.getenv('DATABRICKS_TOKEN')
    
    if not host or not token:
        print("❌ Error: Required environment variables not set")
        print("\nSet them like this:")
        print('  export DATABRICKS_HOST="https://dbc-e67ede65-16bc.cloud.databricks.com"')
        print('  export DATABRICKS_TOKEN="dapi..."')
        sys.exit(1)
    
    print("Starting MLFlow test...")
    print(f"🔗 Connecting to: {host}")
    print(f"🔑 Token: {token[:10]}...{token[-4:]}")
    
    try:
        # Step 1: Set up environment for Databricks auth
        os.environ['DATABRICKS_HOST'] = host
        os.environ['DATABRICKS_TOKEN'] = token
        
        # Use "databricks" as tracking URI
        mlflow.set_tracking_uri("databricks")
        
        # Step 2: Test authentication
        print("\n📋 Testing authentication...")
        experiments = mlflow.search_experiments()
        print(f"✅ Authentication successful! Found {len(experiments)} experiments")
        
        # Step 3: Create experiment in /Shared/ directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_experiment_name = f"/Shared/test_experiment_{timestamp}"
        test_run_name = f"test_run_{timestamp}"
        print(f"\n🧪 Creating test experiment: {test_experiment_name}")
        print(f"📝 Run name: {test_run_name}")
        
        # Step 4: Use helper function to start run
        print("\n📝 Using start_or_continue_run() helper...")
        run_id = start_or_continue_run(
            script_name="test_script",
            tracking_uri=host,
            token=token,
            experiment_name=test_experiment_name,
            run_name=test_run_name
        )
        print(f"✅ Helper function worked! Run ID: {run_id}")
        
        # Step 5: Test log_script_params helper
        print("\n📊 Testing log_script_params() helper...")
        mock_args = MockArgs()
        log_script_params("test_script", mock_args)
        print("✅ Parameters logged via helper function")
        
        # Step 6: Log additional params directly
        print("\n📊 Logging additional parameters...")
        mlflow.log_param("direct_param", "direct_value")
        print("✅ Direct parameters logged")
        
        # Step 7: Log metrics
        print("\n📈 Logging test metrics...")
        for i in range(5):
            mlflow.log_metric("test_metric", i * 0.1, step=i)
        mlflow.log_metric("final_accuracy", 0.95)
        print("✅ Metrics logged")
        
        # Step 8: Log artifact
        print("\n📦 Logging test artifact...")
        with open('/tmp/test_artifact.txt', 'w') as f:
            f.write("This is a test artifact\n")
            f.write(f"Created at: {datetime.now()}\n")
            f.write(f"Test params: {mock_args.test_param_1}, {mock_args.test_param_2}\n")
        mlflow.log_artifact('/tmp/test_artifact.txt', 'test_artifacts')
        print("✅ Artifact logged")
        
        # Step 9: End run
        print("\n🏁 Ending run...")
        mlflow.end_run()
        print("✅ Run ended")
        
        # Get experiment info
        experiment = mlflow.get_experiment_by_name(test_experiment_name)
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! All tests passed!")
        print("="*60)
        print(f"\n✅ Helper functions work correctly")
        print(f"✅ Authentication with token works")
        print(f"✅ Can create experiments and runs")
        print(f"✅ Can log params, metrics, and artifacts")
        
        print(f"\n📍 View your test experiment in Databricks:")
        print(f"   {host}/#mlflow/experiments/{experiment.experiment_id}")
        print(f"\n📍 View your test run:")
        print(f"   {host}/#mlflow/experiments/{experiment.experiment_id}/runs/{run_id}")
        
        # Step 10: Attempt to delete
        print("\n🗑️  Attempting to delete test experiment...")
        try:
            client = mlflow.tracking.MlflowClient()
            client.delete_experiment(experiment.experiment_id)
            print("✅ Test experiment deleted successfully")
        except Exception as e:
            print(f"⚠️  Could not delete experiment: {e}")
            print(f"   You can manually delete it from the UI")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed!")
        print(f"   Error: {e}")
        
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        
        return False

if __name__ == "__main__":
    success = test_mlflow_connection()
    sys.exit(0 if success else 1)