import cv2
import os
import mediapipe as mp
import pyautogui
import math

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('face_recognition_model.yml')

cap = cv2.VideoCapture(0)
hand_detector= mp.solutions.hands.Hands(max_num_hands = 1,min_detection_confidence=0.5, min_tracking_confidence=0.5)
sw, sh = pyautogui.size()

def face_det(frame,_):
    label_names = ["name"]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        label, confidence = face_recognizer.predict(roi_gray)
        if confidence < 100:
            ck_face = False
            if label in label_names:
                ck_face = True
    return ck_face

def mov(): 
    x = int(landmark[8].x*fw)
    y = int(landmark[8].y*fh)
    index_x = sw/fw*x
    index_y = sh/fh*y
    x = int(landmark[8].x*fw)
    y = int(landmark[8].y*fh)
    middle_x = sw/fw*x
    middle_y = sh/fh*y
    pyautogui.moveTo(index_x, index_y)
#    pyautogui.mouseUp(button = "left")
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #check=face_det(frame,_)
    #if(check):
    fh, fw, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks
    if hands:
        for hand in hands:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmark = hand.landmark
            dist = (hand.landmark[8].x - hand.landmark[12].x)**2
            dist += (hand.landmark[8].y - hand.landmark[12].y)**2
            dist1 = math.sqrt(dist)
            dist = (hand.landmark[5].x - hand.landmark[9].x)**2
            dist += (hand.landmark[5].y - hand.landmark[9].y)**2
            dist2 = math.sqrt(dist)
            ratio = dist1/dist2
            dist = (hand.landmark[9].x - hand.landmark[12].x)**2
            dist += (hand.landmark[9].y - hand.landmark[12].y)**2
            dist3 = math.sqrt(dist)

            if ratio > 2 and dist3>0.11:
                mov()
            elif ratio<1.4 and dist3>0.11:

                pyautogui.click(button = "left")
                pyautogui.sleep(1)
            elif dist3<0.01 or hand.landmark[9].y<hand.landmark[12].y:
                pyautogui.click(button = "right")
                pyautogui.sleep(1)
                    
    cv2.imshow('Virtual Mouse', frame)
    cv2.waitKey(1)

