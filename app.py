from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)

# Ensure the "uploads" directory exists
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Load the trained model
MODEL_PATH = r"E35_D5_LeakyRelu_0.0005_A94.h5"  # Update with the correct model path
model = load_model(MODEL_PATH, compile=False)

# Define class names (modify as per your dataset)
Class_names= ['Aloevera', 'Amar poi', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Astma_weed', 'Avacado', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Basale', 'Beans', 'Betel', 'Betel_Nut', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue', 'Coriender', 'Curry_Leaf', 'Doddapatre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganike', 'Gasagase', 'Geranium', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemon_grass', 'Malabar_Nut', 'Mango', 'Marigold', 'Mint', 'Nagadali', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomegranate', 'Pumpkin', 'Raddish', 'Raktachandini', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'Wood_sorel', 'camphor', 'kamakasturi', 'kepala']

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Function to fetch data from CSV
def fetch_data_from_csv(class_name):
    # Update with the correct CSV file path
    df = pd.read_csv("database.csv")

    # Ensure column names match exactly
    if "Common Name" not in df.columns:
        print("Error: 'Common Name' column not found in CSV")
        return None

    # Filter rows where "Common Name" matches
    class_data = df[df["Common Name"] == class_name]

    # Return all matching rows as a list of dictionaries
    return class_data.to_dict(orient='records') if not class_data.empty else None

# Home page route
@app.route("/", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["image"]
        if file and file.filename:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process and predict the image
            preprocessed_image = preprocess_image(file_path)
            predictions = model.predict(preprocessed_image)
            predicted_class_ind = np.argmax(predictions, axis=-1)[0]
            
            pred_class_name = Class_names[predicted_class_ind]

            # Fetch additional data
            data = fetch_data_from_csv(pred_class_name)

            return render_template("result.html", class_name=pred_class_name, data=data if data else None, input_image_path = f"/uploads/{filename}")

    return render_template("index.html")

# About page route
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)