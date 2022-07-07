from flask import Flask, request, render_template, jsonify
import subprocess

from flask_cors import CORS
import io
import cv2

app = Flask(__name__)
CORS(app)


@app.route("/predict_api", methods=["GET", "POST"])
def predict_classes():
    if request.method == "GET":
        render_template("home.html", value="Image")
    elif request.method == "POST":
        if "file" not in request.files:
            return "Image not uploaded!"
        file = request.files["file"].read()
        try:
            # img = Image.open(io.BytesIO(file))
            preds, pred_proba = subprocess.check_output(
                [f"python3 src/predict.py {file}"], shell=True
            ).decode("utf-8")
            pred_proba = pred_proba.split("-")
            return jsonify(predictions=preds, pred_prob=pred_proba)

        except IOError:
            return jsonify(
                predictions="Not an Image, Upload a proper image file", preds_prob=""
            )


if __name__ == "__main__":
    app.run(debug=True)
