import base64
import io
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load unga trained YOLOv8 model weights
model = YOLO("best.pt")


@app.route("/")
def home():
  return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
  if "file" not in request.files:
    return jsonify({"error": "No file uploaded"}), 400

  file = request.files["file"]
  img_bytes = file.read()
  img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

  # Run inference with best.pt
  results = model.predict(source=img, conf=0.25, imgsz=320)
  res = results[0]

  # Draw bounding boxes
  res_plot = res.plot()
  annotated_img = Image.fromarray(res_plot)

  # Convert to base64 for browser display
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
      "image_data": f"data:image/jpeg;base64,{encoded_img}",
  })


if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000, debug=True)
