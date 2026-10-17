from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description="Train YOLO11n on the parking-space dataset.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args=parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    model=YOLO("yolo11n.pt")
    model.train(data=str(ROOT/"dataset"/"data.yaml"),epochs=args.epochs,imgsz=640,batch=4,workers=0,device="",project=str(ROOT/"runs"),name="parking",exist_ok=True,plots=True,seed=42)
    best=Path(model.trainer.save_dir)/"weights"/"best.pt"
    print(f"Best weights: {best}")
if __name__=="__main__": main()
