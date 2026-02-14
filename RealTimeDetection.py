import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import copy
from collections import deque # Added for the stability queue

drawingModule = mp.solutions.drawing_utils
handsModule = mp.solutions.hands
aiModel = tf.keras.models.load_model("TestOne.keras")
capture = cv2.VideoCapture(0)

# 1. Label Mapping based on your data distribution
label_map = {
    0: "Thumbs Up",
    1: "Peace",
    2: "OK",
    3: "Hello",
    4: "Thank You"
}

# 2. Buffers
sequence_buffer = []
# This queue stores the last 10 predictions to check for consistency
prediction_history = deque(maxlen=10) 
stable_prediction = "Waiting..."

with handsModule.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
) as hands:
    
    while True:
        ret, frame = capture.read()
        if not ret: break
        
        display_frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for handLandmarks in results.multi_hand_landmarks:
                mirrored_landmarks = copy.deepcopy(handLandmarks)
                for landmark in mirrored_landmarks.landmark:
                    landmark.x = 1.0 - landmark.x
                
                drawingModule.draw_landmarks(
                    display_frame, mirrored_landmarks, handsModule.HAND_CONNECTIONS
                )

                # Extract coordinates
                current_coords = []
                for res in mirrored_landmarks.landmark:
                    current_coords.extend([res.x, res.y, res.z])

                sequence_buffer.append(current_coords)
                sequence_buffer = sequence_buffer[-30:]

                if len(sequence_buffer) == 30:
                    input_data = np.expand_dims(sequence_buffer, axis=0)
                    prediction = aiModel.predict(input_data, verbose=0)
                    
                    predicted_index = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)

                    # Only count the prediction if confidence is decent
                    if confidence > 0.7:
                        prediction_history.append(predicted_index)
                    
                    # 3. STABILITY CHECK:
                    # If the most frequent prediction in our history appears more than 7/10 times
                    if len(prediction_history) == 10:
                        most_common = max(set(prediction_history), key=list(prediction_history).count)
                        if list(prediction_history).count(most_common) >= 7:
                            stable_prediction = f"{label_map[most_common]} ({confidence*100:.0f}%)"
                        else:
                            stable_prediction = "Calculating..."

        else:
            sequence_buffer = []
            prediction_history.clear()
            stable_prediction = "No Hand Detected"

        # UI Overlay
        cv2.rectangle(display_frame, (0,0), (450, 80), (245, 117, 16), -1)
        cv2.putText(display_frame, stable_prediction, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("HandTracker", display_frame)
        if cv2.waitKey(1) == 27:
            break

capture.release()
cv2.destroyAllWindows()