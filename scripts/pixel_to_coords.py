#!/usr/bin/env python
import rospy
from lab4.msg import cone_location
from sensor_msgs.msg import Image
import numpy as np
import cv2
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg

class homopgraphy_transform():
    def __init__(self):
        rospy.Subscriber("/zed/rgb/image_rect_color", Image, self.callback) #image from camera
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/relative_cone", cone_location, queue_size=10) #fix publish topic
        self.rate = rospy.Rate(10)
        self.matrix = np.array([[-5.86065426e-05,  7.20049423e-04, -7.91927511e-01], 
                                [-8.32948552e-04,  1.29477073e-04,  4.92479502e-01],
                                [-2.65306532e-05, -3.54739645e-03,  1.00000000e+00]], 
                                dtype=np.float32)

    def callback(self, data):
        #get image into CV readable format
        cv_image = self.bridge.imgmsg_to_cv2(data)
        #color segmentation code
        #code to return a pixel or a group of pixels
            #output: pixels = np.array(u, v, 1])
        coords = self.matrix.dot(pixels)
        for i in range(len(coords)):
            coords[i] = coords[i]/coords[-1]
        cone_loc = cone_location()
        cone_loc.x_pos = coords[0]
        cone_loc.y_pos = coords[1]
        self.pub.publish(cone_loc)

if __name__ == "__main__":
    rospy.init_node("homography_transform")
    homopgraphy_transform()
    rospy.spin()