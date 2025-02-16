from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/sujith/Documents/Projects/CHILI_PLANT_DISEASE_DETECTION/web_interface/static/uploads'

# Load your pre-trained model (ensure the file path is correct)
model = tf.keras.models.load_model('chili_disease_model.h5')
# Define class names matching your model's outputs
class_names = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

def preprocess_image(image_path, target_size=(160, 160)):
    """
    Open an image file, resize it, normalize pixel values, and add a batch dimension.
    """
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET'])
def index():
    # Render the main page for image uploads
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file to the uploads folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image and run the model prediction
    processed_image = preprocess_image(file_path)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    predicted_label = class_names[predicted_class]

    # (Optional) Remove the file after prediction:
    # os.remove(file_path)

    # Return the prediction as JSON (or redirect/render another template)
    return jsonify({'prediction': predicted_label})
from flask import request, render_template

@app.route('/result')
def result():
    prediction = request.args.get('prediction', 'No prediction')
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    # Ensure the uploads folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
