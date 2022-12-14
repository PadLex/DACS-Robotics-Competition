import functools
import itertools
import time
from functools import reduce
import math
from multiprocessing import Pool
from sklearn.cluster import DBSCAN


import cv2 as cv
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.spatial import KDTree


# HSV (0-179, 0-255, 0-255)
outer_border_color = (177, 138, 211)
outer_border_error = (10, 60, 60)

inner_border_color = (99, 120, 100)
inner_border_error = (20, 70, 70)


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

def remove_obvious_outliers(points, column, image_width, region_size=40, min_population=4, linear_threshold=0.5):
    #horizontal = np.empty(image_width, dtype=object)
    horizontal = [None for _ in range(image_width)]
    for point in points:
        horizontal[point[0]] = point

    to_be_removed = set()

    left_len = int(region_size/2)
    right_len = round(region_size/2)

    for current_point in points:
        center_x = current_point[0]
        neighbourhood = horizontal[max(0, center_x-left_len):min(image_width-1, center_x+right_len)]
        neighbourhood = list(filter(lambda n: n is not None, neighbourhood))

        """
        neighbours_left = horizontal[max(0, center_x-region_size):center_x]
        neighbours_left = list(filter(lambda n: n is not None, neighbours_left))
        neighbours_right = horizontal[center_x, min(image_width - 1, center_x + region_size)]
        neighbours_right.pop(0)
        neighbours_right = list(filter(lambda n: n is not None, neighbours_right))
        """

        if len(neighbourhood) < min_population:
            to_be_removed.add(center_x)

        xs = [point[0] for point in neighbourhood]
        ys = [point[column] for point in neighbourhood]
        if abs(np.corrcoef(xs, ys)[1, 0]) < linear_threshold:
            for neighbour in neighbourhood:
                to_be_removed.add(neighbour[0])

    return list(filter(lambda p: p[0] not in to_be_removed, points))


def get_lines(points, column, derivative_threshold=0.5):
    xs = [point[0] for point in points]
    ys = [point[column] for point in points]

    polynomials = ()
    min_combined_error = math.inf

    for i in range(3, len(points)-2):
        poly_l, data_l = Polynomial.fit(xs[:i], ys[:i], 1, full=True)
        poly_r, data_r = Polynomial.fit(xs[i:], ys[i:], 1, full=True)

        combined_error = data_l[0][0] + data_r[0][0]

        if combined_error < min_combined_error:
            min_combined_error = combined_error
            polynomials = (poly_l, poly_r)

    print("deriv", polynomials[0].deriv()(0) - polynomials[1].deriv()(0))
    if abs(polynomials[0].deriv()(0) - polynomials[1].deriv()(0)) < derivative_threshold:
        return (Polynomial.fit(xs, ys, 1), )

    return polynomials





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
        clean_points = remove_obvious_outliers(edge_points, 2, np.shape(image)[1])
        if len(clean_points) < 4:
            print("Not enough clean points")
            return
        center_lines = get_lines(clean_points, 2)
        outer_lines = get_lines(clean_points, 3)

        end_analysis = time.time()

        print('\nanalysis time:', (end_analysis - start) * 1000, 'ms', (end_analysis - start) ** -1, 'fps')

        image_copy = image.copy()
        #image_copy = np.zeros(np.shape(image), dtype=np.uint8)
        #image_copy.fill(0)

        for x, inner, center, outer in edge_points:
            image_copy[outer][x] = (0, 255, 0)

        for x, inner, center, outer in clean_points:
            image_copy[outer][x] = (255, 255, 255)

        for line in center_lines:
            for x in range(np.shape(image_copy)[1]):
                y = round(line(x))
                if 0 <= y < np.shape(image_copy)[0]:
                    image_copy[y][x] = (255, 0, 0)

        for line in outer_lines:
            for x in range(np.shape(image_copy)[1]):
                y = round(line(x))
                if 0 <= y < np.shape(image_copy)[0]:
                    image_copy[y][x] = (0, 0, 255)

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
    find_thresholds("tests/test4.jpeg")
