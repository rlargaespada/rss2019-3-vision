#!/usr/bin/env python
import rospy
from lab4.msg import cone_location, parking_error
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from rospy.numpy_msg import numpy_msg
from visualization_msgs.msg import Marker
#file for making homography matrix

class homography():
    def __init__(self):
        self.src_pts = np.array([[],[],[],[]], dtype=np.float32) #image
        self.dest_pts = np.array([[],[],[],[]], dtype=np.float32) #ground
        self.matrix = self.create_matrix()
        #self.marker_pub = rospy.Publisher("/homography_test_marker", Marker, queue_size=1)
        #self.rate = rospy.Rate(2)


    def create_matrix(self):
        H = cv2.findHomography(self.src_pts, self.dest_pts)
        return H

    def test_matrix(self, test_point):
        pos = self.matrix.dot(test_point)
        for i in range(3):
            pos[i] = pos[i]/pos[2]
        print pos
        # marker = Marker()
        # marker.header.frame_id = "base_link"
        # marker.type = marker.CYLINDER
        # marker.action = marker.ADD
        # marker.scale.x = .2
        # marker.scale.y = .2
        # marker.scale.z = .2
        # marker.color.a = 1.0
        # marker.color.b = 1.0
        # marker.color.g = 1.0
        # marker.pose.orientation.w = 1.0
        # marker.pose.position.x = pos[0]
        # marker.pose.position.y = pos[1]
        # self.marker_pub.publish(marker)

if __name__ == '__main__':
    #rospy.init_node("Homography_test")
    h = homography()
    #rospy.spin()
    #u=
    #v=
    #test_point = np.array([u, v, 1])
    #h.test_matrix(test_point)