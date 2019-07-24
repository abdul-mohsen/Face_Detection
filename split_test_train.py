# USAGE
# python split_test_train.py

# This code is meant to split the data and argument the training dataset

# import the necessary packages
from imutils import paths
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,default='dataset',
	help="path to input directory of faces + images")
ap.add_argument("-t", "--test", required=False, default='test_dataset',
	help="path to test dataset")
ap.add_argument("-T", "--train", required=False,default='train_dataset',
	help="path to train dataset")
args = vars(ap.parse_args())

print("[INFO] quantifying faces...")
# Colecting the paths to all images into a list
imagePaths = list(paths.list_images(args["dataset"]))

# taking the label for the name of the folder
labels = [i.split('\\')[1] for i in imagePaths]

# Splite the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
	imagePaths, labels, stratify=labels, test_size=0.2)

# Creating folder in test dataset and train set same as those in the dataset 
for folder in os.listdir(args["dataset"]):
	try:
		# Create target Directory
		os.mkdir("train_dataset/" + folder)
		os.mkdir("test_dataset/" + folder)
	except FileExistsError:
		pass

# Moving test dateset to a folder
for (i, imagePath) in enumerate(X_test):
	os.rename(imagePath,"test_dataset\\" + imagePath.split('\\', 1)[1])

# random example images
images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential(
	[
		# apply the following augmenters to most images
		iaa.Fliplr(0.5), # horizontally flip 50% of all images
		# crop images by -5% to 10% of their height/width
		iaa.CropAndPad(
			percent=(-0.05, .1),
			pad_mode=ia.ALL,
			pad_cval=(0, 255)
		),
		iaa.Affine(
			translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
			rotate=(-30, 30), # rotate by -45 to +45 degrees
			shear=(-20, 20), # shear by -16 to +16 degrees
			order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
			cval=(0, 255), # if mode is constant, use a cval between 0 and 255
		),
		# execute 0 to 5 of the following (less important) augmenters per image
		# don't execute all of them, as that would often be way too strong
		iaa.SomeOf((0, 5),
			[
				iaa.OneOf([
				    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
				    iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
				    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
				]),
				iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
				# search either for all edges or for directed edges,
				# blend the result with the original image using a blobby mask
				iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
				# either change the brightness of the whole image (sometimes
				# per channel) or change the brightness of subareas
				iaa.OneOf([
					iaa.Multiply((0.75, 1.25), per_channel=0.5)
				]),
				iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
			],
			random_order=True
        )
    ],
    random_order=True
)

# Moving images to train dir
for (j, imagePath) in enumerate(X_train):

	# Reading the image
	img = cv2.imread(imagePath)

	# Printing the percent of images that are done
	if j% 10 == 0 :
		print(str(int(j*100/len(X_train))) , '%')

	# Number of Augmentation per image
	numberOfAugmentation = 9	
	for i in range(numberOfAugmentation):

		# Feed the image to the function augment images to generate random augmentation
		image = seq.augment_images([img])[0]

		# Write the agumented image in the train_dataset
		cv2.imwrite("train_dataset\\" + imagePath.split('\\')[1] + "\\" + str(i) + imagePath.split('\\')[2], image)
	
	# Move the ral image to train_dataset
	os.rename(imagePath,"train_dataset\\" + imagePath.split('\\',1)[1])