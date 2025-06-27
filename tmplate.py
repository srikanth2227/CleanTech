from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(_name_)
model = load_model("waste_classifier_model.h5")
class_names = ['Biodegradable', 'Non-Recyclable', 'Recyclable']
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("home"))

    img_file = request.files["image"]
    if img_file.filename == "":
        return redirect(url_for("home"))

    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = class_names[np.argmax(prediction)]

    return render_template("predict.html", label=label, img_path=img_path)

if _name_ == "_main_":
    app.run(debug=True)