
import cv2
import numpy as np

from opwrapper import *


SHOULDER_R = 2
SHOULDER_L = 5
HIP_R = 9 
HIP_L = 12


class OpResult:
    def __init__(self, body, image):
        self.body = body
        self.image = image

# convert keypoints two 2d array
def convert_keypoints(keypoints):
    result = [];
    for e in keypoints:
        newX = e[0]
        newY = e[1]
        result.append([newX, newY])
    return np.array(result)

def get_coach_keypoints(image):

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_range = np.array([20, 25, 25])
	upper_range = np.array([24, 255, 255])
	mask = cv2.inRange(hsv, lower_range, upper_range)

	coach_keypoints = None
	max_yellow_percentage = 0
	op_result = get_keypoints(image)
	for keypoints in op_result.poseKeypoints:

		coords = np.array([keypoints[SHOULDER_R], keypoints[SHOULDER_L], keypoints[HIP_R], keypoints[HIP_L]])
		minX = int(np.min(coords[:,0:1]))
		maxX = int(np.max(coords[:,0:1]))
		minY = int(np.min(coords[:,1:2]))
		maxY = int(np.max(coords[:,1:2]))

		body_area = mask[minY:maxY,minX:maxX]
		yellow_sum = np.sum(body_area)
		yellow_percentage = yellow_sum / (body_area.size * 255)
		print(yellow_percentage)

		if(yellow_percentage > max_yellow_percentage):
			max_yellow_percentage = yellow_percentage
			coach_keypoints = keypoints


	cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('mask', 600, 600)
	cv2.imshow("mask", mask)
	cv2.waitKey(0)

	return OpResult(convert_keypoints(coach_keypoints), op_result.cvOutputData)







