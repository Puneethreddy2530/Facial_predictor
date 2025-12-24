from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import random
from PIL import Image

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "AI Facial Insights API (Simple Mode)",
        "version": "1.0.0"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        image_bytes = file.read()
        
        # Verify it's a valid image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400
        
        # Mock predictions with realistic variation
        age = random.randint(18, 65)
        age_band = 'Adult'
        if age < 2:
            age_band = 'Infant'
        elif age < 13:
            age_band = 'Child'
        elif age < 18:
            age_band = 'Teen'
        elif age > 60:
            age_band = 'Senior'
        
        gender = random.choice(['Man', 'Woman'])
        
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_probs = [random.uniform(0, 100) for _ in emotions]
        total = sum(emotion_probs)
        emotion_probs = [p / total * 100 for p in emotion_probs]
        emotion_dict = dict(zip(emotions, emotion_probs))
        dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
        
        races = ['White', 'Black', 'Asian', 'Indian', 'Others']
        race_probs = [random.uniform(0, 100) for _ in races]
        total = sum(race_probs)
        race_probs = [p / total * 100 for p in race_probs]
        race_dict = dict(zip(races, race_probs))
        dominant_race = max(race_dict.items(), key=lambda x: x[1])[0]
        
        return jsonify({
            'age': age,
            'age_band': age_band,
            'gender': gender,
            'dominant_emotion': dominant_emotion,
            'emotion': emotion_dict,
            'dominant_race': dominant_race,
            'race': race_dict,
            'low_quality': False,
            'quality_reason': '',
            'low_confidence': False,
            'facial_area': {'x': 100, 'y': 80, 'w': 200, 'h': 220}
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AI Facial Insights - Simple Backend")
    print("="*60)
    print("  Running in DEMO MODE with mock predictions")
    print("  Server: http://127.0.0.1:8001")
    print("  Health: http://127.0.0.1:8001/")
    print("  Predict: POST http://127.0.0.1:8001/predict")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=8001, debug=False)
