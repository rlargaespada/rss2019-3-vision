"""
RSS 2019 | Bounding box methods using SIFT and Template Matching

Author: Abbie Lee
"""

import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
import pdb

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img):
    """
    Helper function to print out images, for debugging.
    Press any key to continue.
    """
    winname = "Image"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cd_sift_ransac(img, template, img_counter, debug = False):
    """
    Implement the cone detection using SIFT + RANSAC algorithm using tutorial
    at:  https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches

    Input:
    img: np.3darray; the input image with a cone to be detected
    Return:
    bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
    (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
    """
    # Minimum number of matching features
    MIN_MATCH = 10
    # Create SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    # sift = cv2.SIFT()

    # Compute SIFT on template and test image
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    # Find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    # Find and store good matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # If enough good matches, find bounding box
    if len(good) > MIN_MATCH:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        # Create mask
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

        # Form transformation matrix using the best matches
        dst = cv2.perspectiveTransform(pts, M)

        # set bounding box points to top left and bottom right
        x_min = dst[0][0][0]
        y_min = dst[0][0][1]
        x_max = dst[2][0][0]
        y_max = dst[2][0][1]

        bounding_box = ((x_min, y_min), (x_max, y_max))

        # Add offset to box for drawing
        dst += (w, 0)

        if debug:
            debug_bb(img, template, bounding_box, img_counter)
            # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            #        singlePointColor = None,
            #        matchesMask = matchesMask, # draw only inliers
            #        flags = 2)
            #
            # visualize = cv2.drawMatches(template, kp1, img, kp2, good, None,**draw_params)
            #
            # # Draw bounding box in Red
            # visualize = cv2.polylines(visualize, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
            #
            # fname = str(img_counter) + ".png"
            #
            # cv2.imwrite("test_results/" + fname, visualize)
            # # cv2.imshow("result", img3)
            # # cv2.waitKey(0)

        # Return bounding box
        return bounding_box

    else:
        print("[SIFT] not enough matches; matches: ", len(good))

        # Return bounding box of area 0 if no match found
        return ((0,0), (0,0))

def cd_template_matching(img, template, img_counter, debug=False):
    """
    Implement the cone detection using template matching algorithm using tutorial
    at: https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/

    Input:
    img: np.3darray; the input image with a cone to be detected
    Return:
    bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
    (x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
    """
    template_canny = cv2.Canny(template, 50, 200)

    # Perform Canny Edge detection on test image
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(grey_img, 50, 200)

    # Get dimensions of template
    (img_height, img_width) = img_canny.shape[:2]

    # Keep track of best-fit match
    best_match = None

    # Loop over different scales of image template
    for scale in np.linspace(0.5, 1.5, 50):
        # Resize the image
        resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
        (h, w) = resized_template.shape[:2]

        # Check to see if test image is now smaller than template image
        if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
            continue

        ########## YOUR CODE STARTS HERE ##########
        # Use OpenCV template matching functions to find the best match
        # across template scales.
        # Remember to resize the bounding box using the highest scoring scale
        # x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        # image_print(resized_template)
        result = cv2.matchTemplate(img_canny, resized_template, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if best_match is None or maxVal > best_match[0]:
            	# result_final = result
        	best_match = (maxVal, maxLoc, scale, (h, w))

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, scale, (h, w)) = best_match

    # set bounding box params
    x_min = int(maxLoc[0])
    y_min = int(maxLoc[1])
    x_max = int(maxLoc[0] + w)
    y_max = int(maxLoc[1] + h)

    bounding_box = ((x_min, y_min), (x_max, y_max))
    # image_print(result_final)
    # draw a bounding box around the detected region
    if debug:
        debug_bb(img, template, bounding_box, img_counter)

        ########### YOUR CODE ENDS HERE ###########

    return bounding_box

def debug_bb(img, template, bounding_box, img_counter):
    """
    Display image with detected bounding box and template.

    Input:
        img: np.3darray; the input image with a cone to be detected
        template: np.3darray: template against which we are matching
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

    fname = str(img_counter) + ".png"

    cv2.imwrite("test_results/" + fname, db_img)
    # cv2.imshow("Debug", db_img)
    # cv2.waitKey(0)
