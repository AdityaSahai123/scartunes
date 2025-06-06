from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import os
import tempfile

app = Flask(__name__)

# Load the pre-trained model
model = None

def load_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'model.h5')
        model = keras.models.load_model(model_path)
    return model

# Class names
class_names = ["Contractures", "Hypertrophic", "Keloid", "Normal Fine-Line", "Pitted"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/KnowYourScar')
def prognosis():
    return render_template('KnowYourScar.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        try:
            # Load model if not already loaded
            current_model = load_model()
            
            # Create a temporary file to save the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                file.save(tmp_file.name)
                
                # Read and preprocess the uploaded image
                img = imread(tmp_file.name)
                img = resize(img, (224, 224), mode='reflect', anti_aliasing=True)
                img = np.expand_dims(img, axis=0)
                
                # Make the prediction
                prediction = current_model.predict(img)
                
                # Find the class name with the highest probability
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                
                # Clean up the temporary file
                os.unlink(tmp_file.name)
                
                return jsonify({"prediction": predicted_class_name})
                
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"})

# Vercel requires this
if __name__ == '__main__':
    app.run(debug=True)
else:
    # For Vercel deployment
    app = app
