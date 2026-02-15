import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import copy

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands
aiModel = tf.keras.models.load_model("TheOne.keras")
capture = cv2.VideoCapture(0)

use = True

with handsModule.Hands(
    static_image_mode = False,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 2
) as hands:
    
    while (use):
        ret, frame  = capture.read()
        display_frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                mirrored_landmarks = copy.deepcopy(handLandmarks)
                for landmark in mirrored_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x
                drawingModule.draw_landmarks(
                    display_frame,
                    mirrored_landmarks,
                    handsModule.HAND_CONNECTIONS
                )
            landmarks = []
            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            allThecoordinates = [landmarks]
            correctArray = np.array(allThecoordinates).reshape(1, 63)
            Yprediction = aiModel.predict(correctArray)
            labledYPred = np.argmax(Yprediction, axis=1)
            print(f"Predicted Class Index: {labledYPred[0]}")
            cv2.putText(display_frame, f"Predicted Class Index: {labledYPred[0]}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2)
        cv2.imshow("HandTracker", display_frame)
        if cv2.waitKey(1) == 27:
            break