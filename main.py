import time

import requests
import cv2 as cv
import numpy as np

def request_image():
    start = time.time()
    resp = requests.get("http://192.168.4.1/capture", timeout=5)
    end = time.time()
    print('\nrequests time:', (end - start) * 100, 'ms', (end - start) ** -1, 'fps')
    print(len(resp.content))
    img_array = np.asarray(bytearray(resp.content), dtype="uint8")
    return cv.imdecode(img_array, cv.IMREAD_COLOR)


if __name__ == "__main__":

    """
    while True:
        try:
            img = request_image()
        except:
            print("no response")
            time.sleep(0.5)
            continue
    """
    img = cv.imread("tests/test1.jpeg")
    #process_image(img)

    #cv.destroyAllWindows()



# add wait key. window waits until user presses a key

# and finally destroy/close all open windows

