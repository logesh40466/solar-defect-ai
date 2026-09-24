import os
import io
import base64
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")

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

    # Run inference
    results = model.predict(source=img, conf=0.25, imgsz=320)
    res = results[0]

    # Draw bounding boxes
    res_plot = res.plot()
    annotated_img = Image.fromarray(res_plot)

    # Convert to base64
    buff = io.BytesIO()
    annotated_img.save(buff, format="JPEG")
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    # Extract fault details
    faults = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = res.names[cls_id]
            faults.append({"type": name, "confidence": round(conf * 100, 1)})

    status = "ALERT" if len(faults) > 0 else "NORMAL"

    return jsonify({
        "status": status,
        "total_faults": len(faults),
        "faults": faults,
        "image_data": f"data:image/jpeg;base64,{encoded_img}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
