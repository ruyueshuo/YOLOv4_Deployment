import argparse
import io
import os
import time

import numpy as np
from importlib import import_module
from flask import Flask, render_template, Response, jsonify, request
import cv2
from PIL import Image

import darknet
# darknet = import_module(".").daknet
app = Flask(__name__)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--weights", default="./results/yolov4-tiny-reflective_best.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny-reflective.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./data/reflective.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def load_model():
    """load the pre-trained  model."""
    global network, class_names, class_colors, args
    args = parser()
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )


def preprocess_img(frame):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    frame_rgb = cv2.cvtColor(np.asarray(frame),cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    return darknet_image


@app.route('/predict',methods=['POST'])
def predict():
    """For rendering results on HTML GUI."""

    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        now = time.localtime(time.time() )
        print("POST at :", time.strftime("%Y--%m--%d %H:%M:%S", now))
        if request.files.get("image"):
            print("Inference started ...")

            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            darknet_image = preprocess_img(image)

            # inference
            start_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
            end_time = time.time()

            data["inference_time"] = end_time - start_time
            print("Time cost : {0:.3f}s.".format(data["inference_time"]))

            data["predictions"] = detections

            # indicate that the request was a success
            data["success"] = True

            print("Inference finished ...")

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


@app.route("/predict_api", methods=["POST"])
def predict_api():
    # initialize the data dictionary that will be returned
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        now = time.localtime(time.time() )
        print("POST at :", time.strftime("%Y--%m--%d %H:%M:%S", now))
        if request.files.get("image"):
            print("Inference started ...")

            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image
            darknet_image = preprocess_img(image)

            # inference
            start_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
            end_time = time.time()

            data["inference_time"] = end_time - start_time
            print("Time cost : {0:.3f}s.".format(data["inference_time"]))

            data["predictions"] = detections

            # indicate that the request was a success
            data["success"] = True

            print("Inference finished ...")
    # return the data dictionary as a JSON response
    return jsonify(data)


if __name__ == '__main__':
    args = parser()
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    app.run(host='0.0.0.0', debug=True, threaded=True)