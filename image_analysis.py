import functools
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


def split_by_corners(points, column, image_width, half_chunk=2, max_r=0.8):
    chunk_size = 2*half_chunk + 1
    xs = [point[0] for point in points]
    ys = [point[column] for point in points]

    chunks = [(i, abs(np.corrcoef(xs[i:i+chunk_size+1], ys[i:i+chunk_size+1])[1,0])) for i in range(len(points)-chunk_size+1)]
    chunks.sort(key=lambda chunk: chunk[1])
    print(chunks)

    def split_index(chunk_index):
        return chunk_index + half_chunk

    corner1 = chunks.pop(0)
    if corner1[1] > max_r:
        print("No corners")
        return [points]

    split_1 = split_index(corner1[0])
    if points[corner1[0]][0] < image_width/2:
        for chunk in chunks:
            if chunk[1] > max_r:
                break

            x = points[chunk[0]][0]
            if x >= image_width:
                print("Two corners")
                split_2 = split_index(corner1[1])

                return [points[:split_1], points[split_1:split_2], points[split_2:]]

    print("One corner")
    return [points[:split_1], points[split_1:]]


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
        split_points = split_by_corners(edge_points, 2, np.shape(image)[1])
        end_analysis = time.time()

        print('\nanalysis time:', (end_analysis - start) * 1000, 'ms', (end_analysis - start) ** -1, 'fps')

        #image_copy = image.copy()
        image_copy = np.zeros(np.shape(image), dtype=np.uint8)
        image_copy.fill(0)
        #for x, inner, center, outer in edge_points:
            #image_copy[inner][x] = (0, 255, 0)
            #image_copy[center][x] = (255, 0, 0)
            #image_copy[outer][x] = (0, 0, 255)

        for i, points in enumerate(split_points):
            color = [0, 0, 0]
            color[i] = 255
            for x, inner, center, outer in points:
                image_copy[center][x] = color

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
    find_thresholds("tests/cam2.jpeg")
