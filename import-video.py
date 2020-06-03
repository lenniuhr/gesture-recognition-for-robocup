import cv2
import sys

import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
 
# Opens the Video file
label = "fold"
num_of_frames = 100
clip = 9
start_index = (clip - 1) * 100 + 1


cap= cv2.VideoCapture("images/video/" + label + " - " + str(clip) + ".mov")
i=0
j=start_index
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		break

	if(i % 3 == 0):
		#frame = np.rot90(frame, 3)
		filename = "images/pose-4/" + label + ' - ' + str(j) + '.jpeg'
		print("writing to file " + filename)
		cv2.imwrite(filename, frame)
		j += 1
	i+=1

	if(j >= start_index + num_of_frames):
		cap.release()
		cv2.destroyAllWindows()
		sys.exit(-1)
 
cap.release()
cv2.destroyAllWindows()