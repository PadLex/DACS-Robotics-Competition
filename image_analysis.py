import time

import cv2 as cv
import numpy as np

#HSV (0-179, 0-255, 0-255)
outer_border_color = (177, 138, 211)
outer_border_error = (10, 80, 80)

inner_border_color = (112, 109, 141)
inner_border_error = (10, 40, 40)


def hsv_in_bounds(value, correct, error):
    h = min(abs(value[0] - correct[0]), 180-abs(value[0] - correct[0])) <= error[0] \
        or correct[1] == 0 or correct[2] == 0
    s = abs(value[1] - correct[1]) <= error[1]
    v = abs(value[2] - correct[2]) <= error[2]
    return h and s and v


def analyse_column(hsv_image, column_index):
    inner_border_start = -1
    border_center = -1
    outer_border_end = -1

    for j in range(len(hsv_image)-1, -1, -1):
        pixel = hsv_image[j][column_index]

        """
        print(j, pixel,
              hsv_in_bounds(pixel, inner_border_color, inner_border_error),
              hsv_in_bounds(pixel, outer_border_color, outer_border_error)
              )
        """

        if inner_border_start < 0:
            if hsv_in_bounds(pixel, inner_border_color, inner_border_error):
                inner_border_start = j
                #print("start", pixel)
        elif border_center < 0:
            if hsv_in_bounds(pixel, outer_border_color, outer_border_error):
                border_center = j
                #print("center", pixel)
        elif outer_border_end < 0:
            if not hsv_in_bounds(pixel, outer_border_color, outer_border_error):
                outer_border_end = j
                #print("end", pixel)
                break

    return [inner_border_start, border_center, outer_border_end]


def find_edges(image):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    print(hsv_img[0][0], hsv_img[19][0],
          hsv_in_bounds(hsv_img[0][0], outer_border_color, outer_border_error),
          hsv_in_bounds(hsv_img[19][0], inner_border_color, inner_border_error))

    borders = []

    for i in range(np.shape(hsv_img)[1]):
        borders.append(analyse_column(hsv_img, i))
        #print(i, borders[i])
        image[borders[i][0]][i] = (255, 0, 0)
        image[borders[i][1]][i] = (0, 255, 0)
        image[borders[i][2]][i] = (0, 0, 255)

    print("done")

    cv.imshow('edges', image)
    cv.waitKey(0)

    print(np.shape(image))

"""
cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            cv.imshow('ESP32 stream', image)
            cv.waitKey(0)
            """

if __name__ == "__main__":
    start = time.time()
    print(find_edges(cv.imread("tests/test9.jpeg")))
    end = time.time()
    print('\nanalysis time:', (end - start) * 1000, 'ms', (end - start) ** -1, 'fps')
