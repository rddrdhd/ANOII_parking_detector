import cv2
import time
#vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
#vid_cap.set(cv2.CAP_PROP_FPS, 30.0)

# v prezentaci je, jak udelat trenovani s haarem.
# do priste nasadit tohohtle HAARa do parkovani

cap = cv2.VideoCapture(0)  # 0 = kod moji kamery

# XML na oblicej, dalsi here: https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier(
    'face/haarcascade_frontalface_default.xml')

counter = 0
nFrame = 10
FPS = 0.0
start_time = time.time()

while(True):

    ret, opencv_frame = cap.read()

    # prvni parametr 1.1 - zvetsovani po 10 % pro multiscale; posledni parametr - cim mensi tim citlivejsi - 0 jsou vsechny detekce
    faces_rects = face_cascade.detectMultiScale(opencv_frame, 1.1, 3)

    # vykresleni obdelniku do obrazku
    for rect in faces_rects:
        cv2.rectangle(opencv_frame, rect, (0, 0, 255), 2)

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
    cv2.putText(opencv_frame, "detections: "+str(len(faces_rects)), (30, 60),
                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 3)
    cv2.imshow("opencv_frame", opencv_frame)
    cv2.waitKey(2)
