import cv2
import os

video_path = 'Data2/'
# save_path = 'Train2/'

train_file = 'Anomaly_Test.txt'
f1 = open(train_file, 'r')
lines = f1.readlines()
# img_path = 'Data2/'

fps_count = 0


for line in lines:
    print(line.split('\n')[0])
    cap = cv2.VideoCapture(line.split('\n')[0])
    success, frame = cap.read()
    # print(cap)
    while success:
        # print('1')
        if 'Training' in line:
            # print('1')
            if fps_count%16 ==0:
                cv2.imwrite('Data/Normal_Train/'+str(fps_count)+'.jpg',frame)
            fps_count += 1
            success, frame = cap.read()
        else:
            # if fps_count%4 ==0:
            
            cv2.imwrite('Data/Test/'+str(fps_count)+'.jpg',frame)
            fps_count += 1
            success, frame = cap.read()

        # success, frame = cap.read()



