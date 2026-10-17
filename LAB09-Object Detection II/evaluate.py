from pathlib import Path
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent
WEIGHTS=ROOT/"runs"/"parking"/"weights"/"best.pt"
def main():
    if not WEIGHTS.exists(): raise FileNotFoundError(f"Train first; model not found: {WEIGHTS}")
    m=YOLO(str(WEIGHTS)); r=m.val(data=str(ROOT/"dataset"/"data.yaml"),split="val",imgsz=640,batch=4,workers=0,plots=True,project=str(ROOT/"runs"),name="evaluation",exist_ok=True)
    print(f"Precision: {r.box.mp:.4f}\nRecall: {r.box.mr:.4f}\nmAP50: {r.box.map50:.4f}\nmAP50-95: {r.box.map:.4f}")
if __name__=="__main__": main()

