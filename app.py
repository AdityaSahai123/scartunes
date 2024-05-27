from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import numpy as np
from skimage.io import imread
from skimage.transform import resize

app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('model.h5')

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
        # Read and preprocess the uploaded image
        img = imread(file)
        img = resize(img, (224, 224), mode='reflect', anti_aliasing=True)
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)

        # Find the class name with the highest probability
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]

        return jsonify({"prediction": predicted_class_name})

if __name__ == '__main__':
    app.run(debug=True)
