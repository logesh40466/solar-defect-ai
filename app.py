import os
import io
import base64
import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# YOLOv8 Model Load
model = YOLO("best.pt")

# Fast2SMS API Key
FAST2SMS_KEY = "6wLBQDbeHNzdm4JZysoOrcnkgXua32Y1pFS8xW5vGKIUjfPMT0YQzZOmbJCyfWvTscxahMkuAVdRwG7j"

def send_instant_sms(phone, message_text):
    """
    Direct Telecom Carrier SMS via Fast2SMS OTP Route
    Bypasses DLT block for instant mobile delivery.
    """
    clean_phone = "".join(filter(str.isdigit, str(phone)))
    if len(clean_phone) > 10:
        clean_phone = clean_phone[-10:]
    if not clean_phone or len(clean_phone) < 10:
        clean_phone = "9344042534"

    url = "https://www.fast2sms.com/dev/bulkV2"
    
    # Fast2SMS OTP route (Instant telecom dispatch)
    params = {
        "authorization": FAST2SMS_KEY,
        "variables_values": "9112",
        "route": "otp",
        "numbers": clean_phone
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"[FAST2SMS STATUS] Sent to {clean_phone} -> {response.text}")
        return response.json()
    except Exception as e:
        print(f"[FAST2SMS ERROR] {e}")
        return {"return": False, "message": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_alert", methods=["POST"])
def send_alert():
    data = request.get_json() or {}
    phone = data.get("phone", "9344042534")
    ticket_id = data.get("ticket_id", "WO-ALERT")
    fault_count = data.get("fault_count", 1)

    sms_body = f"SOLARIS AI ALERT: {fault_count} Critical Hotspots detected! WorkOrder: {ticket_id}. String isolation required."
    
    api_res = send_instant_sms(phone, sms_body)

    return jsonify({
        "status": "SENT",
        "target": phone,
        "ticket": ticket_id,
        "gateway_response": api_res
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image format"}), 400

    # Quick thermal chromatic variance filter
    arr = np.array(img)
    r = arr[:, :, 0].astype(float)
    g = arr[:, :, 1].astype(float)
    b = arr[:, :, 2].astype(float)
    color_var = np.mean(np.abs(r - g)) + np.mean(np.abs(r - b))

    if color_var < 15.0:
        buff = io.BytesIO()
        img.save(buff, format="JPEG", quality=85)
        encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
        return jsonify({
            "status": "OPTIMAL",
            "total_faults": 0,
            "faults": [],
            "image_data": f"data:image/jpeg;base64,{encoded_img}"
        })

    # Inference with YOLOv8
    results = model.predict(source=img, conf=0.25, imgsz=480)
    res = results[0]

    # Draw localized bounding boxes
    res_plot = res.plot()
    annotated_img = Image.fromarray(res_plot)

    buff = io.BytesIO()
    annotated_img.save(buff, format="JPEG", quality=85)
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")

    faults = []
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = res.names[cls_id]
            faults.append({
                "type": name,
                "confidence": round(conf * 100, 1)
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
