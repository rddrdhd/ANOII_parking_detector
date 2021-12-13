import cv2
import time
#vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#vid_cap.set(cv2.CAP_PROP_FPS, 30.0)

# v prezentaci je, jak udelat trenovani s haarem.
# do priste nasadit tohohtle HAARa do parkovani¨
# HOG detector v c++ - github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

cap = cv2.VideoCapture(0)  # 0 = kod moji kamery


counter = 0
nFrame = 10
FPS = 0.0
start_time = time.time()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):

    ret, opencv_frame = cap.read()
    # lokace a vahy jednotlivych detekci
    people_rects, people_weight = hog.detectMultiScale(
        opencv_frame, winStride=(4, 4), scale=1.05)  # scale - zvetseni detekcniho okna, winStride - o kolik se to ma posouvat, hitTreshold=1.0 - jen tam kde si je to jistý

    for i, rect in enumerate(people_rects):
        cv2.rectangle(opencv_frame, rect, (0, 0, 255), 2)
        cv2.putText(opencv_frame, "detections: "+str(round(float(people_weight[i]), 3)),
                    (rect[0], rect[1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 3)

    # pocitani FPS
    if(counter == nFrame):
        end_time = time.time()
        allTime = float(end_time - start_time)
        FPS = (float(counter)/allTime)
        counter = 0
        start_time = time.time()
    counter = counter+1

    # FPS text
    cv2.putText(opencv_frame, "FPS: "+str(round(FPS, 2)), (30, 30),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 3)
    # počet detekcí text
    cv2.putText(opencv_frame, "detections: "+str(len(people_rects)), (30, 60),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 3)
    cv2.imshow("opencv_frame", opencv_frame)
    cv2.waitKey(2)
