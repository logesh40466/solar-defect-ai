import os
import io
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 Model
model = YOLO("best.pt")

# Email Pipeline (Unga sender Gmail matrum 16-digit Google App Password inga fill pannunga)
SENDER_EMAIL = "your_email@gmail.com"
SENDER_APP_PASSWORD = "xxxx xxxx xxxx xxxx"

def send_instant_email(recipient_email, ticket_id, fault_count):
    if not recipient_email or "@" not in recipient_email:
        print(f"[VOLTIX ALERT REJECT] Invalid Email: {recipient_email}")
        return {"return": False, "message": "Invalid email address"}

    subject = f"🚨 VOLTIX AI CRITICAL ALERT: {fault_count} Hotspots Detected [{ticket_id}]"
    
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #0c121d; color: #ffffff; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #131d2e; border: 1px solid #1e2c42; border-radius: 10px; padding: 25px;">
          <h2 style="color: #00d2ff; margin-top: 0;">VOLTIX AI — Critical O&M Alert</h2>
          <p style="color: #8494ab;">Autonomous Thermographic Fault Localization Engine</p>
          <hr style="border: 0; border-top: 1px solid #1e2c42;">
          
          <p><strong style="color: #ef4444;">Status:</strong> CRITICAL RISK (Action Required)</p>
          <p><strong>Work Order Ticket:</strong> <span style="font-family: monospace; color: #00d2ff;">{ticket_id}</span></p>
          <p><strong>Defects Classified:</strong> <span style="color: #ef4444; font-weight: bold;">{fault_count} Critical Hotspots</span></p>
          <p><strong>Recommended Action:</strong> Immediate PV String isolation & thermographic field bypass inspection.</p>
          
          <hr style="border: 0; border-top: 1px solid #1e2c42;">
          <p style="font-size: 12px; color: #4c5b73;">This is an autonomous telemetry-generated alert dispatched by VOLTIX AI SCADA Core.</p>
        </div>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = f"VOLTIX AI Sentinel <{SENDER_EMAIL}>"
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        print(f"[VOLTIX EMAIL SUCCESS] Dispatched ticket {ticket_id} to {recipient_email}")
        return {"return": True, "message": "Email dispatched successfully"}
    except Exception as e:
        print(f"[VOLTIX EMAIL ERROR] {e}")
        return {"return": False, "message": str(e)}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send_alert", methods=["POST"])
def send_alert():
    data = request.get_json() or {}
    target_email = data.get("email", "")
    ticket_id = data.get("ticket_id", "WO-ALERT")
    fault_count = data.get("fault_count", 1)

    api_res = send_instant_email(target_email, ticket_id, fault_count)

    return jsonify({
        "status": "SENT",
        "technician_email": target_email,
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

    results = model.predict(source=img, conf=0.25, imgsz=480)
    res = results[0]

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
