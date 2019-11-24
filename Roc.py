import numpy as np
import os
from scipy.io import savemat
import cv2

Videos = os.listdir("Test_Data")

GT = np.loadtxt("Annotations.txt",dtype=str,delimiter='  ')
BT = np.loadtxt("Anomaly_Test.txt",dtype=str,delimiter='  ')

print(GT[0,2])


def get_frame_count(video):
	''' Get frame counts and FPS for a video '''
	cap = cv2.VideoCapture(video)
	if not cap.isOpened():
		print("[Error] video={} can not be opened.".format(video))
		sys.exit(-6)

	# get frame counts
	num_frames = int(cap.get(7))
	fps = cap.get(5)
	width = cap.get(3)   # float
	height = cap.get(4)
	# print("width = ",end='')
	print(width)
	# print("height = ",end='')
	print(height)
	# in case, fps was not available, use default of 29.97
	if not fps or fps != fps:
		fps = 29.97

	return num_frames, fps

print(len(Videos))
for i in range(len(BT)):
	print(BT[i])
	if os.path.exists(BT[i].split('.')[0] + ".txt"):
		idx = np.where(GT[:,0] == BT[i].split('/')[1])[0][0]
		print(True,idx)
		frames,fps = get_frame_count("/home/keval/Documents/Video Processing/Data/" + BT[i])
		truth = np.zeros((1,frames))
		if GT[idx,2] != "-1":
			start = int(GT[idx,2])
			end = int(GT[idx,3])
			truth[0,start:end+1] = 1

		if GT[idx,4] != "-1":
			start = int(GT[idx,4])
			end = int(GT[idx,5])
			truth[0,start:end+1] = 1

		savemat("Allmat/" + str(int(idx)) +".mat",{'truth':truth})	








