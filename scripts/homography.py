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
        self.src_pts = [[693, 495],[1137, 496],[228, 492],[259, 376], [528, 377], [795, 382], [1054, 386], [939, 350],
                                [655, 356], [361, 354]] #image
        self.dest_pts = [[2,0],[2,1.5],[2, -1.5],[5, -3], [5,-1], [5,1], [5,3], [7, 3], [7,0], [7,-3]] #ground
        self.matrix = self.create_matrix()
        #self.marker_pub = rospy.Publisher("/homography_test_marker", Marker, queue_size=1)
        #self.rate = rospy.Rate(2)


    def create_matrix(self):
        src = np.array(self.src_pts, dtype=numpy.float32)
        dest_cnv = self.dest_pts
        for i in range(len(dest_cnv)):
            dest_cnv[i] = dest_cnv[i]/3.281
        dest = np.array(self.dest_pts, dtype=numpy.float32)
            
        H = cv2.findHomography(src, dest)
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