from flask import Flask, render_template, Response, jsonify, request
from camera import VideoCamera
import os
from flask_ngrok import run_with_ngrok

PEOPLE_FOLDER = os.path.join("static", "people_photo")
app = Flask(__name__)
run_with_ngrok(app)
app.config["UPLOAD_FOLDER"] = PEOPLE_FOLDER
video_camera = None
global_frame = None


@app.route("/", methods=["POST"])
def index():
    return render_template("index.html")


@app.route("/home", methods=["POST", "GET"])
def home():
    return render_template("home.html")


@app.route("/record_status", methods=["POST"])
def record_status():
    global video_camera
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json["status"]

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


def video_stream():
    global video_camera
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()

    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )
        else:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + global_frame + b"\r\n\r\n"
            )


@app.route("/video_viewer")
def video_viewer():
    return Response(
        video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


import os


@app.route("/image_viewer", methods=["POST", "GET"])
def image_viewer():
    full_filename = os.path.join(app.config["UPLOAD_FOLDER"], "my_plot.png")
    return render_template("index.html", user_image=full_filename)


if __name__ == "__main__":
    app.run()
