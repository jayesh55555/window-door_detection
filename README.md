# Door \& Window Detection in Architectural Blueprints

An AI-powered object detection system that identifies doors and windows in architectural blueprint images using YOLOv8.

## 🎯 Project Overview

This project implements a custom YOLOv8 model trained from scratch to detect doors and windows in construction blueprint-style images. The solution includes manual labeling, model training, and a FastAPI-based inference endpoint.

## 📁 Project Structure

```
├── README.md
├── app.py                      # FastAPI application
├── train.py                    # Model training script
├── split_dataset.py            # Dataset splitting utility
├── data.yaml                   # YOLOv8 configuration file
├── classes.txt                 # Class definitions
├── requirements.txt            # Python dependencies
├── dataset/
│   ├── train/
│   │   ├── images/            # Training images
│   │   └── labels/            # Training labels (YOLO format)
│   └── val/
│       ├── images/            # Validation images
│       └── labels/            # Validation labels (YOLO format)
├── images/                     # Original blueprint images
├── labels/                     # Manual annotations (YOLO format)
├── runs/detect/train/weights/  # Trained model weights
├── screenshots/
└── best_model.pt              # Final trained model
```


## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (recommended for LabelImg compatibility)
- pip package manager


### Installation

1. **Clone the repository:**

```bash
git clone <your-repo-url>
cd door-window-detection
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download the trained model:**
    - The trained model should be available at `runs/detect/train/weights/best.pt`
    - If not available, follow the training steps below

## 📊 Dataset Preparation

### Manual Labeling Process

1. **Install LabelImg:**

```bash
pip install labelImg setuptools
```

2. **Create class definitions:**

```bash
echo -e "door\nwindow" > classes.txt
```

3. **Start labeling:**

```bash
labelImg images/ classes.txt
```

4. **Labeling Instructions:**
    - Set output format to **YOLO**
    - Set save directory to `labels/`
    - Draw bounding boxes around door and window symbols
    - Save annotations for each image

### Dataset Splitting

Run the dataset splitting script:

```bash
python split_dataset.py
```

This creates an 80/20 train/validation split.

## 🔧 Model Training

### Training Configuration

The model is trained using YOLOv8 with the following parameters:

- **Base model:** YOLOv8n (nano)
- **Epochs:** 50
- **Image size:** 640x640
- **Batch size:** 8
- **Classes:** 2 (door, window)


### Start Training

```bash
python train.py
```


### Monitor Training

- Training progress will be displayed in the console
- Loss graphs and metrics are saved in `runs/detect/train/`
- Best model weights saved as `runs/detect/train/weights/best.pt`


## 🌐 API Usage

### Start the API Server

```bash
python app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### `GET /`

Health check endpoint

**Response:**

```json
{
  "message": "Door & Window Detection API is running!"
}
```


#### `POST /detect`

Object detection endpoint

**Parameters:**

- `file`: Image file (PNG, JPG, JPEG)

**Response:**

```json
{
  "detections": [
    {
      "label": "door",
      "confidence": 0.91,
      "bbox": [0.512, 0.432, 0.120, 0.200]
    },
    {
      "label": "window", 
      "confidence": 0.84,
      "bbox": [0.256, 0.678, 0.080, 0.150]
    }
  ]
}
```

**Bbox Format:** `[x_center, y_center, width, height]` (normalized coordinates 0-1)

### Testing with cURL

```bash
# Test with a blueprint image
curl -X POST "http://localhost:8000/detect" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/blueprint.jpg"

# Example with a specific file
curl -X POST "http://localhost:8000/detect" \
-H "Content-Type: multipart/form-data" \
-F "file=@images/blueprint_001.jpg"
```


### Testing with Python

```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("images/test_blueprint.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```


## 📋 Requirements

```
fastapi==0.104.1
uvicorn==0.24.0
ultralytics==8.0.196
opencv-python==4.8.1.78
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
pillow==10.0.1
scikit-learn==1.3.0
```



## 📧 Contact

For questions or issues, contact: gulanijayesh55@gmail.com

## 📄 License

This project is licensed under the MIT License.

---


