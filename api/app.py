from flask import (
    Flask,
    request,
    render_template,
    make_response,
    jsonify,
    redirect,
    flash,
    url_for,
)

import subprocess

from flask_cors import CORS, cross_origin
import os
import errno

from werkzeug.utils import secure_filename


UPLOAD_FOLDER = "/workspace/ichd/api/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)
CORS(app)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict_api", methods=["GET", "POST"])
@cross_origin()
def predict_classes():
    if request.method == "GET":
        return render_template("home.html", value="Upload")
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("predict_classes"))
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            response = make_response(
                jsonify(predictions="No file selected", pred_probas="", image=""),
                400,
            )
            response.headers["Content-Type"] = "application/json"
            return response
        try:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                out = subprocess.check_output(
                    [
                        "python3",
                        "/workspace/ichd/api/predict.py",
                        f"/workspace/ichd/api/uploads/{filename}",
                    ],
                    shell=False,
                ).decode("utf-8")
                out = out.split("@")
                preds = out[0]
                probas = out[1].strip("\n").split("-")
                grad_cam = out[2].strip("\n")

                response = make_response(
                    jsonify(predictions=preds, pred_probas=probas, image=grad_cam),
                    200,
                )
                response.headers["Content-Type"] = "application/json"
                return response
            else:
                response = make_response(
                    jsonify(
                        predictions="File format expected in PNG or JPG",
                        pred_probas="",
                        image="",
                    ),
                    400,
                )
                response.headers["Content-Type"] = "application/json"
                return response
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass


if __name__ == "__main__":
    app.run(debug=True)
