import cv2
import requests

url = 'http://localhost:5000/predict_api'
# r = requests.post(url, json={'experience':2, 'test_score':9, 'interview_score':6})
image_file = "data/test.jpg"

files = {'image': (image_file, open(image_file, 'rb'))}
# file = {'image':"data/test.jpg"}
r = requests.post(url, files=files)

print(r.json())

cap = cv2.VideoCapture("data/test.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite("data/tmp.jpg", frame)
    files = {'image': ("data/tmp.jpg", open("data/tmp.jpg", 'rb'))}
    r = requests.post(url, files=files)

    print(r.json())

cap.release()