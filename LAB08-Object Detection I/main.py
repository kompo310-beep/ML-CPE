import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
EPOCHS = 40  # Change the number of epochs here


def run_script(script_name, *args):
    script_path = PROJECT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"File not found: {script_path}")

    command = [sys.executable, str(script_path), *map(str, args)]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    print("[1/3] Preparing the dataset...")
    run_script("prepare_dataset.py")

    print(f"[2/3] Training YOLO11s for {EPOCHS} epochs on CPU...")
    run_script("train.py", "--epochs", EPOCHS, "--device", "cpu")

    print("[3/3] Detecting objects in the test images...")
    run_script("predict.py")

    print("Done! Check the predictions folder for the results.")


if __name__ == "__main__":
    main()