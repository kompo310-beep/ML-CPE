from pathlib import Path
import cv2
from ultralytics import YOLO
ROOT=Path(__file__).resolve().parent; WEIGHTS=ROOT/"runs"/"parking"/"weights"/"best.pt"
def main():
    if not WEIGHTS.exists(): raise FileNotFoundError(f"Train first; model not found: {WEIGHTS}")
    model=YOLO(str(WEIGHTS)); cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Cannot open webcam index 0")
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            result=model.predict(frame,imgsz=640,conf=.25,device="",verbose=False)[0]
            cv2.imshow("Toy Car Detection (Q to quit)",result.plot())
            if cv2.waitKey(1)&0xFF==ord("q"): break
    finally: cap.release(); cv2.destroyAllWindows()
if __name__=="__main__": main()

