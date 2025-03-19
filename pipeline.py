import subprocess

def run_step(script_name):
    """Run a Python script as a subprocess and handle errors."""
    print(f"\n🚀 Running {script_name}...\n")
    
    try:
        result = subprocess.run(["python3", f"scripts/{script_name}"], capture_output=True, text=True, check=True)
        
        print(result.stdout)  # Print script output
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script_name} (Exit Code: {e.returncode}):\n{e.stderr}")
        exit(1)  # Stop pipeline if a step fails
    except FileNotFoundError:
        print(f"❌ Script not found: scripts/{script_name}")
        exit(1)  # Stop if a script is missing
    except Exception as e:
        print(f"❌ Unexpected error in {script_name}: {e}")
        exit(1)  # Stop execution on unknown errors

if __name__ == "__main__":
    print("\n📌 Starting Customer Churn Prediction Pipeline...\n")

    # ✅ Run the pipeline steps sequentially
    pipeline_steps = [
        "data_ingestion.py",       # Load Data
        "data_preprocessing.py",   # Data Preprocessing
        "train_model.py",          # Model Training & Tracking
        "register_best_model.py",  # Model Registration
        "test_model.py"            # Model Testing & Prediction
    ]

    for script in pipeline_steps:
        run_step(script)

    print("\n✅ Full pipeline executed successfully! 🚀")
