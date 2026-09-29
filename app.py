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

# Sender Engine (Email anuppa use aagura host credentials)
SENDER_EMAIL = "logeshajay1701@gmail.com"
SENDER_APP_PASSWORD = "ndfu itzq cjus xuoa"

def send_instant_email(recipient_email, ticket_id, fault_count):
    if not recipient_email or "@" not in recipient_email:
        print(f"[VOLTIX ALERT REJECT] Invalid Email: {recipient_email}")
        return False

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
          <p><strong>Recipient Technician:</strong> {recipient_email}</p>
          <p><strong>Defects Classified:</strong> <span style="color: #ef4444; font-weight: bold;">{fault_count} Critical Hotspots</span></p>
          <p><strong>Recommended Action:</strong> Immediate PV String isolation & thermographic field inspection.</p>
          
          <hr style="border: 0; border-top: 1px solid #1e2c42;">
          <p style="font-size: 12px; color: #4c5b73;">Dispatched by VOLTIX AI Core to actively authenticated operator.</p>
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
        print(f"[VOLTIX EMAIL SUCCESS] Alert sent to {recipient_email} for ticket {ticket_id}")
        return True
    except Exception as e:
        print(f"[VOLTIX EMAIL ERROR] {e}")
        return False

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    # User browser-la login panna email inga dynamic-aa capture aagum
    target_email = request.form.get("user_email", "logeshajay1701@gmail.com")

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

    # Defect irundhaa, login panna technician mail-ku direct-aa anuppum
    if len(faults) > 0:
        ticket_id = f"WO-{np.random.randint(100000, 999999)}-PV"
        send_instant_email(target_email, ticket_id, len(faults))

    return jsonify({
        "status": status,
        "total_faults": len(faults),
        "faults": faults,
        "image_data": f"data:image/jpeg;base64,{encoded_img}",
        "dispatched_to": target_email
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
