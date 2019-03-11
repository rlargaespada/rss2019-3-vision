"""
    RSS 2019 | debug_utils.py
    Tools for visualizing keypoints and bounding boxes returned by various
    object detection methods

    Author: Abbie Lee (abbielee@mit.edu)
"""

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import pdb

def get_fname(img_path):
    """
    Input:
        img_path: str; absolute path to the test image
    Return:
        fname: str; file name
    """
    # we know that the file is one directory deep. Find first forward slash.
    path = img_path[2:]
    slash = path.index("/")

    # find beginning of extension
    ext_idx = path.index(".")

    fname = path[slash + 1:ext_idx]

    return fname

def save_img(img, fname):
    """
    Writes test image to a file.
    """
    res_path = "test_results/"
    ext = ".png"
    save_path = res_path + fname + ext

    cv2.imwrite(save_path, img)


def draw_bb(img, template, bounding_box, img_path, show = False, save = False):
    """
    Displays image with detected bounding box and template.

    Input:
        img: np.3darray; the input image with a cone to be detected
        template: np.3darray: template against which we are matching
        bounding_box:
        img_path:
    Return: None
    """
    matches = []
    db_img = cv2.drawMatches(template, None, img, None, None, None, None)

    # shift bb to draw on image
    w = template.shape[1]

    top_left =(int(bounding_box[0][0] + w), bounding_box[0][1])
    bottom_right = (int(bounding_box[1][0]) + w, bounding_box[1][1])

    # Draw bounding box in Red
    cv2.rectangle(db_img, top_left, bottom_right, (0, 0, 255), 2)

    fname = get_fname(img_path) + "_bb"

    if save:
        save_img(db_img, fname)

    if show:
        cv2.imshow("Debug", db_img)
        cv2.waitKey(0)

def draw_kp(img, kp, img_path, show = False, save = False):
    """
    Displays image with detected keypoints
    """

    db_img = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags = 0)
    fname = get_fname(img_path) + "_kp"

    if save:
        save_img(db_img, fname)

    if show:
        cv2.imshow("Debug", db_img)
        cv2.waitKey(0)


def match_kp(img, template, img_kp, temp_kp, bounding_box, matches, mask, img_path, show = False, save = False):
    """
    Displays template and test image with matched keypoints and bounding box
    """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
           singlePointColor = None,
           matchesMask = mask, # draw only inliers
           flags = 2)

    db_img = cv2.drawMatches(template, temp_kp, img, img_kp, matches, None,**draw_params)

    # shift bb to draw on image
    w = template.shape[1]

    top_left =(int(bounding_box[0][0] + w), bounding_box[0][1])
    bottom_right = (int(bounding_box[1][0]) + w, bounding_box[1][1])

    # Draw bounding box in Red
    cv2.rectangle(db_img, top_left, bottom_right, (0, 0, 255), 2)
    fname = get_fname(img_path) + "_match"

    if save:
        save_img(db_img, fname)

    if show:
        cv2.imshow("result", img3)
        cv2.waitKey(0)
