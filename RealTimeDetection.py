import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import copy
import pyttsx3 as pt
import threading

# --- YOUR ORIGINAL SETUP ---
drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands
aiModel = tf.keras.models.load_model("TheBetterOne.keras")
capture = cv2.VideoCapture(0)

# Track if we are currently speaking to avoid crashing the engine
is_speaking = False

def speak_safely(text):
    global is_speaking
    if is_speaking:
        return # Skip if already talking
    
    def run_speech():
        global is_speaking
        is_speaking = True
        try:
            local_engine = pt.init() # Local init is more stable for threads
            local_engine.setProperty('rate', 150)
            local_engine.say(text)
            local_engine.runAndWait()
            local_engine.stop() # Clean up
        finally:
            is_speaking = False
            
    threading.Thread(target=run_speech, daemon=True).start()

thumbSaidTimes = 0
peaceSaidTimes = 0
okaySaidTimes = 0
chairSaidTimes = 0
homeSaidTimes = 0
youSaidTimes = 0
meSaidTimes = 0
haltSaidTimmes = 0

use = True

with handsModule.Hands(
    static_image_mode = False,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7,
    max_num_hands = 2
) as hands:
    
    while (use):
        ret, frame  = capture.read()
        if not ret: continue
        display_frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # New list to store multiple hand results
        labledYPred = [] 

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
                
                # Extract landmarks for THIS hand
                temp_landmarks = []
                for lm in handLandmarks.landmark:
                    temp_landmarks.extend([lm.x, lm.y, lm.z])
                
                # Get prediction for THIS hand
                correctArray = np.array([temp_landmarks]).reshape(1, 63)
                prediction = aiModel.predict(correctArray, verbose=0)
                labledYPred.append(np.argmax(prediction, axis=1)[0])

            # --- CHECKING THE PREDICTIONS ---
            if len(labledYPred) > 0:
                # Two-Hand Logic (Checks index 0 AND 1 safely)
                if len(labledYPred) == 2:
                    if labledYPred[0] == 3 and labledYPred[1] == 3:
                        cv2.putText(display_frame, "Home", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if homeSaidTimes == 0:
                            speak_safely("home")
                            homeSaidTimes = 1
                        thumbSaidTimes=peaceSaidTimes=okaySaidTimes=chairSaidTimes=meSaidTimes=youSaidTimes=haltSaidTimmes=0
                    

                # Single-Hand Logic (Always safe to check index 0 if len > 0)
                if labledYPred[0] == 0:
                    cv2.putText(display_frame, "Good", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if thumbSaidTimes == 0:
                        speak_safely("good")
                        thumbSaidTimes = 1
                    peaceSaidTimes=okaySaidTimes=chairSaidTimes=meSaidTimes=homeSaidTimes=youSaidTimes=haltSaidTimmes=0
                
                elif labledYPred[0] == 1:
                    cv2.putText(display_frame, "Peace", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if peaceSaidTimes == 0:
                        speak_safely("peace")
                        peaceSaidTimes = 1
                    thumbSaidTimes=okaySaidTimes=chairSaidTimes=meSaidTimes=homeSaidTimes=youSaidTimes=haltSaidTimmes=0

                elif labledYPred[0] == 2:
                    cv2.putText(display_frame, "Okay", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if okaySaidTimes == 0:
                        speak_safely("okay")
                        okaySaidTimes = 1
                    thumbSaidTimes=peaceSaidTimes=chairSaidTimes=meSaidTimes=homeSaidTimes=youSaidTimes=haltSaidTimmes=0

                elif labledYPred[0] == 6:
                    cv2.putText(display_frame, "Halt", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if meSaidTimes == 0:
                        speak_safely("halt")
                        meSaidTimes = 1
                    thumbSaidTimes=peaceSaidTimes=okaySaidTimes=chairSaidTimes=homeSaidTimes=youSaidTimes=0
                
                elif labledYPred[0] == 4:
                    cv2.putText(display_frame, "Me", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if meSaidTimes == 0:
                        speak_safely("me")
                        meSaidTimes = 1
                    thumbSaidTimes=peaceSaidTimes=okaySaidTimes=chairSaidTimes=homeSaidTimes=youSaidTimes=haltSaidTimmes=0

                elif labledYPred[0] == 5:
                    cv2.putText(display_frame, "You", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if youSaidTimes == 0:
                        speak_safely("you")
                        youSaidTimes = 1
                    thumbSaidTimes=peaceSaidTimes=okaySaidTimes=chairSaidTimes=meSaidTimes=homeSaidTimes=haltSaidTimmes=0

                elif labledYPred[0] == 7:
                    thumbSaidTimes=peaceSaidTimes=okaySaidTimes=chairSaidTimes=meSaidTimes=homeSaidTimes=youSaidTimes=haltSaidTimmes=0

        cv2.imshow("HandTracker", display_frame)
        if cv2.waitKey(1) == 27:
            break

capture.release()
cv2.destroyAllWindows()