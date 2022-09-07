import time

import requests
import cv2
import numpy as np


while True:
    start = time.time()
    resp = requests.get("http://192.168.4.1/capture.jpg")
    end = time.time()
    print('requests time:', (end-start) * 1000, 'ms')

    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # show the image, provide window name first
    cv2.imshow('ESP32 stream', image)
    cv2.waitKey(20)

# add wait key. window waits until user presses a key

# and finally destroy/close all open windows
cv2.destroyAllWindows()
