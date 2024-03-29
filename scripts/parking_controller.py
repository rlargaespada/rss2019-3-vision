#!/usr/bin/env python
import rospy
from lab4.msg import cone_location, parking_error
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    def __init__(self):
        rospy.Subscriber("/relative_cone", cone_location, 
            self.relative_cone_callback)    
        self.drive_pub = rospy.Publisher("/vesc/ackermann_cmd_mux/input/navigation", 
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            parking_error, queue_size=10)

        self.parking_distance = 0.7
        self.failure_forward = 0
        self.relative_x = 0
        self.relative_y = 0

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = - msg.y_pos
        print(self.relative_x, self.relative_y)
        drive_cmd = AckermannDriveStamped()
        
        #################################
        # Play with this number too
        # parking_distance = .75 #meters
        
        # Your Code Here.
        # Use relative position and your control law to populate
        # drive_cmd.

        # using Ackerman Steering x pure pursuit to find angle
	
        if self.relative_y == -1:   #Changed to whatever we decide failure to be
            if self.failure_forward < 10:
                self.drive_cmd.drive.steering_angle = 0
                self.failure_forward += 1
            else:
                self.drive_cmd.drive.steering_angle = np.pi/6
            self.drive_cmd.drive.speed = 1
        else:
            self.failure_forward = 0
            dist_to_pt = np.sqrt(self.relative_x**2 + self.relative_y**2)
            if self.relative_x<0:
                dist_to_park = -dist_to_pt+self.parking_distance
                theta = -np.arctan(self.relative_y/self.relative_x)  
            else:
                dist_to_park = dist_to_pt-self.parking_distance
                theta = np.arctan(self.relative_y/self.relative_x) 

            L_1 = dist_to_park # look ahead distance
            L = 0.325 # size of wheel base
            

            delta = np.arctan(2*L*np.sin(theta)/L_1)

            drive_cmd.drive.steering_angle = delta 

            # using proportional (potentially PD or PID) controller to control velocity
            kp = 2
            vel = kp*dist_to_park
            #print(vel)
            vel = max(min(1.0, vel), -1.0) # caps velocity magnitude at 1
            #print(vel)

            drive_cmd.drive.speed = vel
	    rospy.loginfo(dist_to_park)
	    if abs(vel) > .1:
        #################################
       		self.drive_pub.publish(drive_cmd)
        	self.error_publisher()
            else:
		rospy.loginfo("close enough")
        
    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = parking_error()
        
        #################################

        error_msg.x_error = self.relative_x
        error_msg.y_error = self.relative_y
        error_msg.distance_error = np.sqrt(self.relative_x**2 + self.relative_y**2)

        # Your Code Here
        # Populate error_msg with relative_x, relative_y, sqrt(x^2+y^2)

        #################################
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
