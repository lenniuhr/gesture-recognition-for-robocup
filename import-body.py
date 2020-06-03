
import numpy as np

from importutil import *
from mathutil import *


HEAD = 0
NECK = 1
PELVIS = 8
SHOULDER_R = 2
SHOULDER_L = 5
ELLBOW_R = 3
ELLBOW_L = 6

HAND_R = 4
HAND_L = 7
H_ROOT = 0
H_CENTER = 9 

def get_body_angles(entry):
	body = entry.body
	shoulderR = winkel(body[NECK] - body[SHOULDER_R], body[ELLBOW_R] - body[SHOULDER_R])
	shoulderL = winkel(body[NECK] - body[SHOULDER_L], body[ELLBOW_L] - body[SHOULDER_L])
	ellbowR = winkel(body[SHOULDER_R] - body[ELLBOW_R], body[HAND_R] - body[ELLBOW_R])
	ellbowL = winkel(body[SHOULDER_L] - body[ELLBOW_L], body[HAND_L] - body[ELLBOW_L])
	return np.array([shoulderR, shoulderL, ellbowR, ellbowL]) / 180

def get_bounding_box(entry):

	# get important positions
	body = entry.body
	x_pos = np.array([body[ELLBOW_R][0], body[ELLBOW_L][0], body[HAND_R][0], body[HAND_L][0], body[SHOULDER_R][0], body[SHOULDER_L][0]])
	y_pos = np.array([body[ELLBOW_R][1], body[ELLBOW_L][1], body[HAND_R][1], body[HAND_L][1], body[SHOULDER_R][1], body[SHOULDER_L][1]])

	# get bounding box borders
	minValX = np.min(x_pos[np.nonzero(x_pos)])
	maxValX = np.max(x_pos[np.nonzero(x_pos)])
	minValY = np.min(y_pos[np.nonzero(y_pos)])
	maxValY = np.max(y_pos[np.nonzero(y_pos)])

	boundingBoxWidth = maxValX - minValX
	boundingBoxHeight = maxValY - minValY

	# calculate position relative to bounding box
	ellbowR = [(body[ELLBOW_R][0] - minValX) / boundingBoxWidth, (body[ELLBOW_R][1] - minValY) / boundingBoxHeight]
	ellbowL = [(body[ELLBOW_L][0] - minValX) / boundingBoxWidth, (body[ELLBOW_L][1] - minValY) / boundingBoxHeight]
	handR = [(body[HAND_R][0] - minValX) / boundingBoxWidth, (body[HAND_R][1] - minValY) / boundingBoxHeight]
	handL = [(body[HAND_L][0] - minValX) / boundingBoxWidth, (body[HAND_L][1] - minValY) / boundingBoxHeight]
	shoulderR = [(body[SHOULDER_R][0] - minValX) / boundingBoxWidth, (body[SHOULDER_R][1] - minValY) / boundingBoxHeight]
	shoulderL = [(body[SHOULDER_L][0] - minValX) / boundingBoxWidth, (body[SHOULDER_L][1] - minValY) / boundingBoxHeight]

	return np.array([ellbowR[0], ellbowR[1], ellbowL[0], ellbowL[1], handR[0], handR[1], handL[0], handL[1], shoulderR[0], shoulderR[1], shoulderL[0], shoulderL[1]])

def import_body(dataset, file_prefixes, labels):
	for i in range(0, len(file_prefixes)):
		file_prefix = file_prefixes[i]
		label = labels[i]
		entries = get_raw_entries(dataset, file_prefix)
		#print(entries)
		for entry in entries:
			positions = get_bounding_box(entry)
			angles = get_body_angles(entry)
			inputs = np.concatenate((angles, positions))
			
			#print_body_angles(angles)
			upsert_body_entry(dataset, entry.filename, entry.frame_nr, label, inputs)
		print("Successfully import file prefix '" + file_prefix + "'")

def check_body_entries(dataset, labels):
	for label in labels:
		entries = get_body_entries(dataset, label)
		print("%s: %i" % (label, len(entries)))



#---------- START ----------

print("---------- START IMPORT BODY ----------")

#show_image("images/pose-1/arm-up-left - 13.jpeg")


#file_prefixes = ["arm-up-left", "arm-up-right", "idle", "dab", "clap", "show-left", "show-right", "t-pose"]
#file_prefixes = ["idle"]
#dataset = "pose-1"

#import_keypoints(dataset, file_prefixes)
#import_keypoints(dataset, file_prefixes, start_index = 38, end_index = 74)
#check_raw_entries(dataset, file_prefixes)

#import_body(dataset, file_prefixes, file_prefixes)
#check_body_entries(dataset, file_prefixes)




#dataset = "pose-2"

#file_prefixes = ["idle", "both-arms-up", "arm-up-right", "arm-up-left", "show-right", "show-left", "show-up-right", "show-up-left", "clap", 
#"cheer", "complain", "both-arms-right", "both-arms-left", "t-pose", "fists-together", "arm-bow-right", "arm-bow-left", "cross-arms", "time-out-low", "time-out-high"]
#file_prefixes = ["arm-up-left"]

#import_keypoints(dataset, file_prefixes, start_index = 1, end_index = 37)
#import_keypoints(dataset, file_prefixes, start_index = 38, end_index = 74)
#import_keypoints(dataset, file_prefixes, start_index = 112, end_index = 148)
#check_raw_entries(dataset, file_prefixes)

#import_body(dataset, file_prefixes, file_prefixes)
#check_body_entries(dataset, file_prefixes)


#show_image("images/pose-3/cross-arms - 14.jpeg")
#sys.exit(-1)

#dataset = "pose-3"

#file_prefixes = ["idle", "clap", "time-out", "cross-arms"] 
#file_prefixes = ["idle"]

#import_keypoints(dataset, file_prefixes)
#check_raw_entries(dataset, file_prefixes)

#import_body(dataset, file_prefixes, file_prefixes)
#check_body_entries(dataset, file_prefixes)



dataset = "pose-4"

#file_prefixes = ["dance", "clap", "spin", "time-out", "idle", "fold"] 
file_prefixes = ["idle"]

import_keypoints(dataset, file_prefixes)
check_raw_entries(dataset, file_prefixes)

#import_body(dataset, file_prefixes, file_prefixes)
#check_body_entries(dataset, file_prefixes)





