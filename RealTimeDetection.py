import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import copy

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands
aiModel = tf.keras.models.load_model("maybeTheOne.keras")
capture = cv2.VideoCapture(0)
yTest = np.load("y_test.npy")

use = True

with handsModule.Hands(
    static_image_mode = False,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 1
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
            wrist = mirrored_landmarks.landmark[0]

            thumb_to_hand = mirrored_landmarks.landmark[1]
            thumb_base_to_next = mirrored_landmarks.landmark[2]
            thumb_next_to_next = mirrored_landmarks.landmark[3]
            thumb_tip = mirrored_landmarks.landmark[4]

            index_to_hand = mirrored_landmarks.landmark[5]
            index_base_to_next = mirrored_landmarks.landmark[6]
            index_next_to_next = mirrored_landmarks.landmark[7]
            index_tip = mirrored_landmarks.landmark[8]
                
            middle_to_hand = mirrored_landmarks.landmark[9]
            middle_base_to_next = mirrored_landmarks.landmark[10]
            middle_next_to_next = mirrored_landmarks.landmark[11]
            middle_tip = mirrored_landmarks.landmark[12]
                
            ring_to_hand = mirrored_landmarks.landmark[13]
            ring_base_to_next = mirrored_landmarks.landmark[14]
            ring_next_to_next = mirrored_landmarks.landmark[15]
            ring_tip = mirrored_landmarks.landmark[16]

            pinky_to_hand = mirrored_landmarks.landmark[17]
            pinky_base_to_next = mirrored_landmarks.landmark[18]
            pinky_next_to_next = mirrored_landmarks.landmark[19]
            pinky_tip = mirrored_landmarks.landmark[20]
            allThecoordinates = [wrist.x, wrist.y, wrist.z, thumb_to_hand.x, thumb_to_hand.y, thumb_to_hand.z, thumb_base_to_next.x, thumb_base_to_next.y, thumb_base_to_next.z, thumb_next_to_next.x, thumb_next_to_next.y, thumb_next_to_next.z, thumb_tip.x, thumb_tip.y, thumb_tip.z, index_to_hand.x, index_to_hand.y, index_to_hand.z, index_base_to_next.x, index_base_to_next.y, index_base_to_next.z, index_next_to_next.x, index_next_to_next.y, index_next_to_next.z, index_tip.x, index_tip.y, index_tip.z, middle_to_hand.x, middle_to_hand.y, middle_to_hand.z, middle_base_to_next.x, middle_base_to_next.y, middle_base_to_next.z, middle_next_to_next.x, middle_next_to_next.y, middle_next_to_next.z, middle_tip.x, middle_tip.y, middle_tip.z, ring_to_hand.x, ring_to_hand.y, ring_to_hand.z, ring_base_to_next.x, ring_base_to_next.y, ring_base_to_next.z, ring_next_to_next.x, ring_next_to_next.y, ring_next_to_next.z, ring_tip.x, ring_tip.y, ring_tip.z, pinky_to_hand.x, pinky_to_hand.y, pinky_to_hand.z, pinky_base_to_next.x, pinky_base_to_next.y, pinky_base_to_next.z, pinky_next_to_next.x, pinky_next_to_next.y, pinky_next_to_next.z, pinky_tip.x, pinky_tip.y, pinky_tip.z]
            correctArray = np.array(allThecoordinates).reshape(1, 63)
            Yprediction = aiModel.predict(correctArray)
            labledYPred = np.argmax(Yprediction, axis=1)
            print(f"Predicted Class Index: {labledYPred[0]}")
        cv2.imshow("HandTracker", display_frame)
        if cv2.waitKey(1) == 27:
            break