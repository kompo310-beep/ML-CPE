from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EPOCHS = 50  # Change this value to set the number of training epochs.


def run_script(filename, *args):
    """Run a lab script with the current Python interpreter."""
    script = ROOT / filename
    if not script.is_file():
        print(f"Error: Could not find {filename} in the lab folder.")
        return False

    print(f"\n--- Running {filename} ---")
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=ROOT,
    )
    if result.returncode == 0:
        print(f"--- {filename} finished successfully ---")
        return True

    print(f"--- {filename} failed (exit code {result.returncode}) ---")
    return False


def main():
    if EPOCHS <= 0:
        print("Error: EPOCHS must be a positive integer.")
        return

    # Run each stage in order. Stop if a stage fails.
    steps = [
        ("train.py", ("--epochs", str(EPOCHS))),
        ("evaluate.py", ()),
        ("image_test.py", ()),
        ("realtime_test.py", ()),
    ]

    for filename, args in steps:
        if not run_script(filename, *args):
            print("\nWorkflow stopped because a step failed.")
            return

    print("\nAll workflow steps completed.")


if __name__ == "__main__":
    main()
