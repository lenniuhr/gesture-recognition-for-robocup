
import cv2
import enum
import numpy as np

from fileutil import *
from mathutil import *
from opwrapper import *

batch_size = 55

class Image:
	def __init__(self, name, frame_nr, data):
		self.name = name
		self.frame_nr = frame_nr
		self.data = data

def check_raw_entries(dataset, file_prefixes):
	for file_prefix in file_prefixes:
		entries = get_raw_entries(dataset, file_prefix)
		print("%s: %i" % (file_prefix, len(entries)))

# Read all images with given file prefix from the images folder
# Returns a list of Image 
def read_images(folder, file_prefix, start_index = 1, end_index = 10000):
	images = []
	i = start_index
	while(True):
		filename = file_prefix + " - " + str(i) + ".jpeg"
		image = cv2.imread("images/" + folder + "/" + filename)
		if(image is None):
			break
		print("Reading image file '" + filename + "'")
		images.append(Image(filename, i, image))
		i += 1
		if(i > end_index):
			return images;
	return images

# Imports all images with the file prefix to the raw database
def import_keypoints(dataset, file_prefixes):
	for file_prefix in file_prefixes:
		entries = get_raw_entries(dataset, file_prefix)
		start_index = len(entries) + 1
		end_index = start_index + batch_size - 1
		images = read_images(dataset, file_prefix, start_index, end_index)
		for image in images:
			print("Importing image file '" + image.name + "'")
			op_result = get_keypoints(image.data)
			print("Finished importing image file '" + image.name + "'")
			print("End index: " + str(end_index))
			#print("------ BODY ------")
			#print(op_result.body)
			#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
			#cv2.resizeWindow('image', 600, 800)
			#cv2.imshow('image', op_result.image)
			#cv2.waitKey(0)
			upsert_raw_entry(dataset, image.name, image.frame_nr, op_result.body)

def show_image(file):
	image = cv2.imread(file)
	print(image)
	op_result = get_keypoints(image)
	print(op_result)
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600, 600)
	cv2.imshow('image', op_result.image)
	cv2.imwrite("images/op-example-2.jpeg", op_result.image)
	cv2.waitKey(0)

#show_image("images/clap-example - 1.jpeg")

