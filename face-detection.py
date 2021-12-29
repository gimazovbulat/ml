import cv2, os
import numpy as np
from PIL import Image

image_path = 'face-detection-sample.jpg'
input_image = Image.open(image_path)
gray = input_image.convert('L')
input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = np.array(gray, 'uint8')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(image, minSize=(10, 10), maxSize=(250, 250))
print(faces)
for (x, y, w, h) in faces:
    cv2.imshow("", image[y: y + h, x: x + w])
    cv2.waitKey(0)
    cv2.rectangle(input_image, (x, y), (x + w, y + h), 255, 1)
cv2.imshow('output', input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()