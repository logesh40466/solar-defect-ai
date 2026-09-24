import os
import io
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Model
model = YOLO("best.pt")

def is_thermal_or_solar_payload(img_np):
    """
    Industrial heuristic to verify if image is Thermal (Ironbow, Rainbow, FLIR Grayscale)
    or Structured Solar PV Array, and reject casual human portraits / street scenes.
    """
    # 1. Grayscale Thermal (FLIR Black/White Hot) check
    is_gray = False
    if len(img_np.shape) == 2 or (len(img_np.shape) == 3 and np.allclose(img_np[:,:,0], img_np[:,:,1], atol=10)):
        is_gray = True

    # 2. Chromatic Variance (Thermal false-color palettes have high saturation spread)
    if not is_gray and img_np.shape[2] >= 3:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        sat_mean = np.mean(sat)
        sat_std = np.std(sat)
        
        # Thermal palettes have very distinct localized hot clusters
        if sat_mean < 8 and sat_std < 10:
            # Low color variation and not structured thermal
            return False, "Low thermal contrast payload"

    # 3. Structural Grid / Edge Density Check (Solar PV modules have strict parallel busbars/cells)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if not is_gray else (img_np if len(img_np.shape)==2 else img_np[:,:,0])
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    
    # Casual empty walls or blurry random objects have near-zero edge structures
    if edge_density < 0.005:
        return False, "Payload lacks solar module cell structural signatures"

    return True, "Valid Thermal/Solar Telemetry"

def preprocess_thermal_image(img_pil):
    """
    Real-world contrast stretching (CLAHE) to make cold/hot cells 
    instantly visible to the YOLO convolutional layers.
    """
    img_np = np.array(img_pil)
    
    # Apply CLAHE to luminance channel for optimal feature illumination
    if len(img_np.shape) == 3:
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced), img_np
    return img_pil, img_np

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No telemetry payload transmitted"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    
    try:
        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Corrupt or unreadable image file"}), 400

    # Enhance contrast & prepare array
    enhanced_img, img_np = preprocess_thermal_image(raw_img)

    # Validate image authenticity
    is_valid, validation_msg = is_thermal_or_solar_payload(img_np)
    if not is_valid:
        buff = io.BytesIO()
        raw_img.save(buff, format="JPEG", quality=85)
        encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        return jsonify({
            "status": "REJECTED_PAYLOAD",
            "message": validation_msg,
            "total_faults": 0,
            "faults": [],
            "image_data": f"data:image/jpeg;base64,{encoded_img}"
        })

    # Balanced Inference: conf=0.22 captures true hotspots while rejecting noise
    results = model.predict(source=enhanced_img, conf=0.22, imgsz=640, iou=0.45)
    res = results[0]

    # Render bounding boxes on the original thermal image
    res_plot = res.plot()
    annotated_img = Image.fromarray(res_plot)

    # Encode to Base64
    buff = io.BytesIO()
    annotated_img.save(buff, format="JPEG", quality=90)
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Extract & Classify Faults
    faults = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = res.names[cls_id]
            
            # Real-world bounding box geometry validation:
            # Hotspots are typically compact cell areas (not entire human height aspect ratios)
            xyxy = box.xyxy[0].cpu().numpy()
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
            
            # Hotspots are usually square-ish/rectangular cells (aspect ratio < 5.0)
            if aspect_ratio < 5.0:
                faults.append({
                    "type": name,
                    "confidence": round(conf * 100, 1),
                    "box": [round(float(x), 1) for x in xyxy]
                })

    status = "ALERT" if len(faults) > 0 else "OPTIMAL"

    return jsonify({
        "status": status,
        "total_faults": len(faults),
        "faults": faults,
        "image_data": f"data:image/jpeg;base64,{encoded_img}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
