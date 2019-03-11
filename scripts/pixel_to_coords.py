#!/usr/bin/env python
import rospy
from lab4.msg import cone_location
#from sensor_msgs.msg import Image
import numpy as np
#import cv2
from std_msgs.msg import Int32MultiArray
#from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg

class homopgraphy_transform():
    def __init__(self):
        rospy.Subscriber("/relative_cone", Int32MultiArray, self.callback) #fix, want input pixel coordinates as a list
        self.pub = rospy.Publisher("/relative_cone", cone_location, queue_size=10) #fix publish topic
        self.rate = rospy.Rate(10)
        self.matrix = np.array([[1.23439833e-05,  3.26294215e-04, -5.74007407e-01], #these numbers arent greate, need to fix
                                [-6.19388166e-04,  1.20862702e-04,  3.53063720e-01],
                                [ 2.04221042e-05, -3.24199685e-03,  1.00000000e+00]], 
                                dtype=np.float32)

    def callback(self, data):
        pixels = np.array([data[0], data[1], 1])
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