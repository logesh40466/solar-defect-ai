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

def send_instant_sms(target_phone, message_text):
    """
    Direct Telecom Carrier SMS via Fast2SMS Quick Route (No Domain Verification Required)
    """
    clean_phone = "".join(filter(str.isdigit, str(target_phone)))
    if len(clean_phone) > 10:
        clean_phone = clean_phone[-10:]
    
    if not clean_phone or len(clean_phone) != 10:
        print(f"[FAST2SMS REJECT] Invalid mobile number: {target_phone}")
        return {"return": False, "message": "Invalid 10-digit mobile number"}

    url = "https://www.fast2sms.com/dev/bulkV2"
    
    # Fast2SMS Quick Route JSON payload
    payload = {
        "route": "q",
        "message": message_text,
        "language": "english",
        "flash": 0,
        "numbers": clean_phone
    }
    
    headers = {
        "authorization": FAST2SMS_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        print(f"[FAST2SMS STATUS] Dispatched to logged-in user {clean_phone} -> {response.text}")
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
    
    # Dynamically extract whichever number was entered in the profile
    logged_in_phone = data.get("phone", "")
    ticket_id = data.get("ticket_id", "WO-ALERT")
    fault_count = data.get("fault_count", 1)

    sms_body = f"SOLARIS AI ALERT: {fault_count} Critical Hotspots detected! WorkOrder: {ticket_id}. String isolation required."
    
    # Send directly to active user's phone number
    api_res = send_instant_sms(logged_in_phone, sms_body)

    return jsonify({
        "status": "SENT",
        "target_user_phone": logged_in_phone,
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

    # Lightweight chromatic variance gate
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
