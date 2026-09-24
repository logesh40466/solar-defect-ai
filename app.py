import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")

def is_valid_solar_thermal(img_pil):
    """
    Validates whether the uploaded image matches thermal infrared color distributions
    or high-contrast solar PV structural characteristics.
    """
    arr = np.array(img_pil)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)

    # Calculate thermal palette variance (Ironbow thermal images have distinct chromatic shifts)
    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))
    saturation_variance = rg_diff + rb_diff

    # Everyday street/portrait RGB photos have low thermal contrast ratios
    # Genuine FLIR/Ironbow thermal palettes have high color separation
    if saturation_variance < 18.0:
        return False

    return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # 1. Thermal & Solar Validation Gate
    if not is_valid_solar_thermal(img):
        # Convert original image to base64 without drawing fake boxes
        buff = io.BytesIO()
        img.save(buff, format="JPEG")
        encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        
        return jsonify({
            "status": "INVALID_IMAGE",
            "message": "Non-thermal payload detected. Please upload an infrared solar PV scan.",
            "total_faults": 0,
            "faults": [],
            "image_data": f"data:image/jpeg;base64,{encoded_img}"
        })

    # 2. Run inference with strict confidence threshold (0.35)
    results = model.predict(source=img, conf=0.35, imgsz=480)
    res = results[0]

    # Draw bounding boxes
    res_plot = res.plot()
    annotated_img = Image.fromarray(res_plot)

    # Convert to base64
    buff = io.BytesIO()
    annotated_img.save(buff, format="JPEG")
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Extract detected faults
    faults = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = res.names[cls_id]
            faults.append({"type": name, "confidence": round(conf * 100, 1)})

    status = "ALERT" if len(faults) > 0 else "OPTIMAL"

    return jsonify({
        "status": status,
        "total_faults": len(faults),
        "faults": faults,
        "image_data": f"data:image/jpeg;base64,{encoded_img}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
