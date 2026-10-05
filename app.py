import os
import io
import base64
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')

# ================= CROSS-DEVICE CLOUD DATABASE =================
HISTORY_DB_FILE = 'scada_history.json'

def get_history_db():
    if not os.path.exists(HISTORY_DB_FILE):
        return {}
    with open(HISTORY_DB_FILE, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return {}

def save_history_db(data):
    with open(HISTORY_DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

@app.route('/api/history', methods=['GET'])
def get_user_history():
    email = request.args.get('email', '').strip().lower()
    db = get_history_db()
    return jsonify(db.get(email, []))

@app.route('/api/history', methods=['POST'])
def save_user_history():
    payload = request.get_json(force=True)
    email = payload.get('email', '').strip().lower()
    record = payload.get('record')
    if not email or not record:
        return jsonify({"status": "error", "message": "Missing email or record"}), 400
    
    db = get_history_db()
    if email not in db:
        db[email] = []
    db[email].insert(0, record)
    save_history_db(db)
    return jsonify({"status": "success", "count": len(db[email])})
@app.route('/api/history/clear', methods=['POST'])
def clear_user_history():
    payload = request.get_json(force=True)
    email = payload.get('email', '').strip().lower()
    if not email:
        return jsonify({"status": "error"}), 400
    db = get_history_db()
    if email in db:
        db[email] = []
        save_history_db(db)
    return jsonify({"status": "success"})
# ===============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read uploaded image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # --- Thermal Radiometric Profile Validation ---
        np_img = np.array(image)
        hsv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv_img[:, :, 1])
        std_dev = np.std(np_img)

        # Rejects common portrait / skin tones & non-thermal natural scenes
        if avg_saturation < 18 and std_dev < 30:
            return jsonify({
                "status": "error",
                "message": "INVALID TELEMETRY: Non-thermal image detected. Please upload FLIR/IR radiometric frames only."
            }), 400
        # ----------------------------------------------

        # Run YOLOv8 inference
        results = model(image)
        res = results[0]

        faults = []
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0]) * 100
            name = res.names[cls_id]
            faults.append({
                "type": name,
                "confidence": round(conf, 1)
            })

        # Plot bounding boxes
        res_plotted = res.plot()
        res_image = Image.fromarray(res_plotted)

        buffered = io.BytesIO()
        res_image.save(buffered, format="JPEG")
        img_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        user_email = request.form.get('user_email', 'logeshajay1701@gmail.com')

        return jsonify({
            'status': 'Analysis Completed',
            'total_faults': len(faults),
            'faults': faults,
            'image_data': img_base64,
            'dispatched_to': user_email
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
