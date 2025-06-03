from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List
import io

app = FastAPI(title="Door & Window Detection API")

# Train Model
try:
    model = YOLO('runs/detect/train/weights/best.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/")
def read_root():
    return {"message": "Door & Window Detection API is running!"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")
    
    try:
        results = model(img)[0]
        
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                confidence = float(box.conf[0])
                
                x, y, w, h = box.xywhn[0].tolist()
                
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
                })
        
        return {"detections": detections}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
