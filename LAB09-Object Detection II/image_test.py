from pathlib import Path
import random
import cv2
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent; WEIGHTS=ROOT/"runs"/"parking"/"weights"/"best.pt"
def main():
    if not WEIGHTS.exists(): raise FileNotFoundError(f"Train first; model not found: {WEIGHTS}")
    imgs=list((ROOT/"dataset"/"valid"/"images").glob("*.*"))
    if not imgs: raise FileNotFoundError("No test images found")
    model=YOLO(str(WEIGHTS))
    for p in random.Random(42).sample(imgs,min(5,len(imgs))):
        result=model.predict(source=str(p),imgsz=640,conf=.25,device="",verbose=False)[0]
        out=ROOT/"runs"/"image_test"; out.mkdir(parents=True,exist_ok=True)
        cv2.imwrite(str(out/p.name),result.plot())
        print(f"{p.name}: {len(result.boxes)} detections -> runs/image_test/{p.name}")
if __name__=="__main__": main()

