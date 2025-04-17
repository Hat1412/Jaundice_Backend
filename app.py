from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the model (adjust filename as needed)
model = tf.keras.models.load_model("jaundice_model.h5")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"].read()
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]
    result = "Jaundice" if prediction > 0.5 else "No Jaundice"
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run()
