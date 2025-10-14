# mlflow_utils.py
import mlflow
import os

def start_or_continue_run(script_name, tracking_uri=None, token=None, experiment_name=None, run_name=None):
    """
    Start new run or continue existing one with proper Databricks authentication.
    
    Args:
        script_name: Name of the script (used for logging)
        tracking_uri: Databricks workspace URL (defaults to DATABRICKS_HOST env var)
        token: Databricks token (defaults to DATABRICKS_TOKEN env var)
        experiment_name: Experiment name - must be absolute path for Databricks
                        e.g., '/Shared/my-experiment'
                        (defaults to EXPERIMENT_NAME env var)
        run_name: Run name (defaults to RUN_NAME env var)
    
    Returns:
        run_id: The MLFlow run ID
    
    Raises:
        ValueError: If required parameters are missing
    """
    # Get tracking_uri (workspace URL)
    tracking_uri = tracking_uri or os.getenv('DATABRICKS_HOST')
    if not tracking_uri:
        raise ValueError(
            "tracking_uri is required. Provide it as an argument or set DATABRICKS_HOST environment variable."
        )
    
    # Get token
    token = token or os.getenv('DATABRICKS_TOKEN')
    if not token:
        raise ValueError(
            "token is required for authentication. Provide it as an argument or set DATABRICKS_TOKEN environment variable."
        )
    
    # Set credentials for Databricks authentication
    os.environ['DATABRICKS_HOST'] = tracking_uri
    os.environ['DATABRICKS_TOKEN'] = token
    
    # For Databricks, use "databricks" as the tracking URI
    mlflow.set_tracking_uri("databricks")
    
    # Test authentication by making an API call
    try:
        mlflow.search_experiments(max_results=1)
    except Exception as e:
        raise ValueError(f"Authentication failed. Check your credentials. Error: {e}")
    
    # Check if continuing existing run
    run_id = os.getenv('MLFLOW_RUN_ID')
    
    if run_id:
        # Continue existing run
        mlflow.start_run(run_id=run_id)
        print(f"[{script_name}] Continuing run: {run_id}")
    else:
        # Start new run - need experiment_name and run_name
        experiment_name = experiment_name or os.getenv('EXPERIMENT_NAME')
        if not experiment_name:
            raise ValueError(
                "experiment_name is required for new runs. "
                "Provide it as an argument or set EXPERIMENT_NAME environment variable."
            )
        
        # Validate Databricks experiment path format
        if not experiment_name.startswith('/'):
            raise ValueError(
                f"Databricks experiment name must be an absolute path starting with '/', "
                f"e.g., '/Shared/my-experiment'. "
                f"Got: '{experiment_name}'"
            )
        
        run_name = run_name or os.getenv('RUN_NAME')
        if not run_name:
            raise ValueError(
                "run_name is required for new runs. "
                "Provide it as an argument or set RUN_NAME environment variable."
            )
        
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        run_id = run.info.run_id
        os.environ['MLFLOW_RUN_ID'] = run_id
        print(f"[{script_name}] Started new run: {run_id}")
        print(f"[{script_name}] Experiment: {experiment_name}")
        print(f"[{script_name}] Run name: {run_name}")
    
    return run_id

def log_script_params(script_name, args):
    """
    Log params with script prefix to avoid collisions.
    
    Args:
        script_name: Name of the script (used as prefix)
        args: argparse.Namespace or object with parameters as attributes
    """
    if not mlflow.active_run():
        raise RuntimeError(
            "No active MLFlow run. Call start_or_continue_run() first."
        )
    
    params = {f"{script_name}_{k}": v for k, v in vars(args).items()}
    mlflow.log_params(params)
