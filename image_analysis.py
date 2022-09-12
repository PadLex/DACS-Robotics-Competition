import itertools
import time
from functools import reduce
import math
from multiprocessing import Pool

import cv2 as cv
import numpy as np
from scipy.spatial import KDTree


# HSV (0-179, 0-255, 0-255)
outer_border_color = (177, 138, 211)
outer_border_error = (10, 80, 80)

inner_border_color = (99, 120, 100)
inner_border_error = (20, 80, 80)


def hsv_in_bounds(value, correct, error):
    # s and v and h
    return (
            abs(value[1] - correct[1]) <= error[1] and
            abs(value[2] - correct[2]) <= error[2] and
            (min(abs(value[0] - correct[0]), 180 - abs(value[0] - correct[0])) <= error[0] or
             correct[1] == 0 or correct[2] == 0)
    )


def analyse_column(hsv_image, column_index):
    inner_border_start = -1
    border_center = -1
    outer_border_end = -1

    for j in range(len(hsv_image) - 1, -1, -1):
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
                # print("start", pixel)
        elif border_center < 0:
            if hsv_in_bounds(pixel, outer_border_color, outer_border_error):
                border_center = j
                # print("center", pixel)
        elif outer_border_end < 0:
            if not hsv_in_bounds(pixel, outer_border_color, outer_border_error):
                outer_border_end = j
                # print("end", pixel)
                break

    return [column_index, inner_border_start, border_center, outer_border_end]


def find_edge_points(image, step=1):
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_width = np.shape(hsv_img)[1]
    indexes = range(0, image_width, step)

    # The parallel version is slower for now
    # with Pool() as pool:
    # return pool.starmap(analyse_column, zip(repeat(hsv_img), indexes))

    return list(map(lambda i: analyse_column(hsv_img, i), indexes))


def clean_edge_points(points):
    return list(filter(lambda point: -1 not in point, points))


def edge_error(points, column, index0, index1):
    x0 = points[index0][0]
    y0 = points[index0][column]
    x1 = points[index1][0]
    y1 = points[index1][column]
    line = lambda x: (y1 - y0) / (x1 - x0) * (x - x1) + y1
    distance = lambda point: abs(point[column] - line(point[0]))/len(points)
    return sum(map(distance, itertools.islice(points, index0, index1)))


def find_corners(points, column, half_chunk=2):
    chunk = 2*half_chunk + 1
    xs = [point[0] for point in points]
    ys = [point[column] for point in points]

    smallest_coefficient = 1
    corner1_index = -1

    second_smallest_coefficient = 1
    corner2_index = -1

    for i in range(len(points)-chunk+1):
        coef = abs(np.corrcoef(xs[i:i+chunk], ys[i:i+chunk])[1,0])
        print(xs[i:i+chunk], ys[i:i+chunk])
        print(coef)

        if coef < smallest_coefficient:
            second_smallest_coefficient = smallest_coefficient
            corner2_index = corner1_index
            smallest_coefficient = coef
            corner1_index = i + half_chunk
        elif coef < second_smallest_coefficient:
            second_smallest_coefficient = coef
            corner2_index = i + half_chunk

    print(smallest_coefficient, second_smallest_coefficient)

    return corner1_index, corner2_index



"""
cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            cv.imshow('ESP32 stream', image)
            cv.waitKey(0)
            """


def find_thresholds(path):
    global outer_border_color, outer_border_error, inner_border_color, inner_border_error

    image = cv.imread(path)

    title_window = "Find thresholds"
    cv.namedWindow(title_window)

    def render_borders():
        start = time.time()
        edge_points = clean_edge_points(find_edge_points(image, 4))
        corner_indexes = find_corners(edge_points, 2)
        end_analysis = time.time()

        print('\nanalysis time:', (end_analysis - start) * 1000, 'ms', (end_analysis - start) ** -1, 'fps')

        print("edge error ", edge_error(edge_points, 2, 0, 20), edge_error(edge_points, 2, 20, len(edge_points)-1))

        image_copy = image.copy()


        for x, inner, center, outer in edge_points:
            #image_copy[inner][x] = (0, 255, 0)
            image_copy[center][x] = (255, 0, 0)
            #image_copy[outer][x] = (0, 0, 255)


        print("corners", corner_indexes)
        if corner_indexes[0] != -1:
            corner_point = edge_points[corner_indexes[0]]
            image_copy[corner_point[2]][corner_point[0]] = (0, 255, 0)

        cv.imshow(title_window, image_copy)

    print("outer_border_color outer_border_error inner_border_color inner_border_error")
    print(outer_border_color, outer_border_error, inner_border_color, inner_border_error)

    while True:
        render_borders()
        cv.waitKey(1)
        try:
            tuple_strings = input("new values: ").replace(' ', '').replace('(', '').split(")")
            tuple_strings.pop()
            value_strings = list(map(lambda x: x.split(','), tuple_strings))
            tuples = [[int(y) for y in x] for x in value_strings]
            outer_border_color, outer_border_error, inner_border_color, inner_border_error = tuples
        except:
            print("Bad input")


if __name__ == "__main__":
    find_thresholds("tests/test9.jpeg")
