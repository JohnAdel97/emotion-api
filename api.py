# ========================
# Corrected Local PC API Code
# ========================
from flask import Flask, request, jsonify 
from flask_cors import CORS
import tensorflow as tf 
import cv2
import numpy as np
import os
import tempfile  # Added for cross-platform temp files
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ======================== Key Fixes ========================
# 1. Match EXACTLY your model's output classes
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise']  # Verify with your model

# 2. Use tensorflow.keras imports consistently
from tensorflow.keras.models import load_model

# 3. Add missing custom objects for EfficientNet
custom_objects = {
    'LeakyReLU': tf.keras.layers.LeakyReLU,
    'BatchNormalization': tf.keras.layers.BatchNormalization,
    'Functional': tf.keras.Model,
    'swish': tf.keras.layers.Activation('swish').call  # Critical for EfficientNet
}

# 4. Initialize model properly
model = load_model('Final.h5', custom_objects=custom_objects, compile=False)

# 5. Enhanced face detection parameters
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def process_image(img):
    """Improved image processing"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # More sensitive
        minNeighbors=5,
        minSize=(50, 50),  # Smaller faces
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
        
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])  # Largest face
    face = img[y:y+h, x:x+w]
    processed = cv2.resize(face, (224, 224))
    return np.expand_dims(processed.astype('float32')/255.0, axis=0)

@app.route('/predict/image', methods=['POST'])
def image_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
            
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        processed = process_image(img)
        
        if processed is None:
            return jsonify({'error': 'No face detected'}), 400
            
        pred = model.predict(processed, verbose=0)
        return jsonify({'emotion': EMOTIONS[np.argmax(pred)]})
        
    except Exception as e:
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

@app.route('/predict/video', methods=['POST'])
def video_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No video uploaded'}), 400
            
        file = request.files['file']
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return jsonify({'error': 'Unsupported video format'}), 400
            
        # Use system temp directory
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return jsonify({'error': 'Unreadable video file'}), 400
            
        results = []
        last_emotion = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        
        # Process every 0.5 seconds using timestamps
        for timestamp in np.arange(0, duration, 0.5):
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            processed = process_image(frame)
            if processed is None:
                continue
                
            pred = model.predict(processed, verbose=0)
            current_emotion = EMOTIONS[np.argmax(pred)]
            
            if current_emotion != last_emotion:
                results.append({
                    'timestamp': round(timestamp, 1),
                    'emotion': current_emotion
                })
                last_emotion = current_emotion
                
        cap.release()
        os.remove(temp_path)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Video processing failed: {str(e)}'}), 500

@app.route('/')
def home():
    return """
    <h1>Emotion Detection API</h1>
    <p>Endpoints:</p>
    <ul>
        <li>POST /predict/image - Analyze image</li>
        <li>POST /predict/video - Analyze video</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)  # Enable threading