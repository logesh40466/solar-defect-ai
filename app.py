import os
import io
import base64
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')

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
