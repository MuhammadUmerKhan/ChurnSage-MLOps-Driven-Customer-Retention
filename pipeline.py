import subprocess

def run_step(script_name):
    """Run a Python script as a subprocess and handle errors."""
    print(f"\n🚀 Running {script_name}...\n")
    result = subprocess.run(["python3", f"scripts/{script_name}"], capture_output=True, text=True)

    print(result.stdout)  # Print script output

    if result.returncode != 0:
        print(f"❌ Error in {script_name}:\n{result.stderr}")
        exit(1)  # Stop pipeline if a step fails

if __name__ == "__main__":
    print("\n📌 Starting Customer Churn Prediction Pipeline...\n")

    # ✅ Run the pipeline steps sequentially
    run_step("data_ingestion.py")    # Load Data
    run_step("data_preprocessing.py") # Data Preprocessing
    run_step("train_model.py")       # Model Training & Tracking
    run_step("register_best_model.py.py")    # Model Registration
    run_step("test_model.py")        # Model Testing & Prediction

    print("\n✅ Full pipeline executed successfully! 🚀")