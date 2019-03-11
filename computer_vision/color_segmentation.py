import cv2
import imutils
import numpy as np
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
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########

	# hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
	# lower_bounds, upper_bounds = max_min_hsv(hsv_template)

	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	upper_bounds = [30,255,255] # HSV values we are interested in
	lower_bounds = [8,175,175]
	mask = cv2.inRange(img_hsv, np.array(lower_bounds), np.array(upper_bounds))

	kernel = np.ones((5,5), np.uint8)
	eroded = cv2.erode(mask, kernel, iterations=2)
	dilated = cv2.dilate(eroded, kernel, iterations=3)
	im, contours, h = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cv2.imshow('mask', dilated)
	cv2.waitKey(0)

	# while len(contours)==0:
	# 	upper_bounds[0]=upper_bounds[0]+2
	# 	mask = cv2.inRange(img_hsv, np.array(lower_bounds), np.array(upper_bounds))
	# 	kernel = np.ones((5,5), np.uint8)
	# 	eroded = cv2.erode(mask, kernel, iterations=3)
	# 	dilated = cv2.dilate(eroded, kernel, iterations=3)
	# 	im, contours, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# 	cv2.imshow('mask', dilated)
	# 	cv2.waitKey(0)

	if len(contours)>1:
		print("There is more than one orange object.")
		cv2.imshow("image", img)
		cv2.waitKey(0)
		areas = [cv2.contourArea(c) for c in contours]
		i = areas.index(max(areas))
		c = contours[i]

	elif len(contours)<1:
		print("No orange object detected.")
		cv2.imshow("image", img)
		cv2.waitKey(0)
		return ((0,0), (0, 0))
		return None
	else:
		c = contours[0]

	x,y,w,h = cv2.boundingRect(c)
	bounding_box = ((x,y), (x+w, y+h))
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # draws bounding rectangle around cone in OG pic
	cv2.imshow("image", img)
	cv2.waitKey(0)


	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box

def max_min_hsv(temp): # TODOOOOOOOOOOOOOOOO
	print(temp[0][0])
	low = [180,255,255]
	high = [0,0,0]
	for i in temp:
		for p in i:
			if not (p[0] == 0 and p[1] == 0 and p[2]==255):
				low[0] = min(low[0], p[0])
				low[1] = min(low[0], p[0])
				low[2] = min(low[0], p[0])
				high[0] = max(high[0], p[0])
				high[1] = max(high[1], p[1])
				high[2] = max(high[2], p[2])
	return (np.array(low),np.array(high))


if __name__ == "__main__":
	img = cv2.imread("test15.jpg")
	template = cv2.imread("cone_template.png")
	image_print(img)
	cd_color_segmentation(img, template)