import cv2
import mediapipe
import numpy as np
import os

# Setup MediaPipe modules
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

# Start webcam
capture = cv2.VideoCapture(0)

frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

i = 0

# Create hand tracking object and process frames
with handsModule.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=2
) as hands:
    
    use = True

    while (use):
        # Read frame from webcam
        ret, frame = capture.read()
        frame  = cv2.flip(frame, 1)
        
        # Process frame and detect hands
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Draw landmarks if hands detected
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(
                    frame, 
                    handLandmarks, 
                    handsModule.HAND_CONNECTIONS
                )

                wrist = handLandmarks.landmark[0]

                thumb_to_hand = handLandmarks.landmark[1]
                thumb_base_to_next = handLandmarks.landmark[2]
                thumb_next_to_next = handLandmarks.landmark[3]
                thumb_tip = handLandmarks.landmark[4]

                index_to_hand = handLandmarks.landmark[5]
                index_base_to_next = handLandmarks.landmark[6]
                index_next_to_next = handLandmarks.landmark[7]
                index_tip = handLandmarks.landmark[8]
                
                middle_to_hand = handLandmarks.landmark[9]
                middle_base_to_next = handLandmarks.landmark[10]
                middle_next_to_next = handLandmarks.landmark[11]
                middle_tip = handLandmarks.landmark[12]
                
                ring_to_hand = handLandmarks.landmark[13]
                ring_base_to_next = handLandmarks.landmark[14]
                ring_next_to_next = handLandmarks.landmark[15]
                ring_tip = handLandmarks.landmark[16]

                pinky_to_hand = handLandmarks.landmark[17]
                pinky_base_to_next = handLandmarks.landmark[18]
                pinky_next_to_next = handLandmarks.landmark[19]
                pinky_tip = handLandmarks.landmark[20]

                wrist_x  = int(wrist.x * frameWidth)
                wrist_y = int(wrist.y * frameHeight)
                wrist_z = wrist.z

                thumb_tip_x = int(thumb_tip.x * frameWidth)
                thumb_tip_y = int(thumb_tip.y * frameHeight)
                thumb_tip_z = thumb_tip.z
                thumb_to_hand_x = int(thumb_to_hand.x * frameWidth)
                thumb_to_hand_y = int(thumb_to_hand.y * frameHeight)
                thumb_to_hand_z = thumb_to_hand.z
                thumb_base_to_next_x = int(thumb_base_to_next.x * frameWidth)
                thumb_base_to_next_y = int(thumb_base_to_next.y * frameHeight)
                thumb_base_to_next_z = thumb_base_to_next.z
                thumb_next_to_next_x = int(thumb_next_to_next.x * frameWidth)
                thumb_next_to_next_y = int(thumb_next_to_next.y * frameHeight)
                thumb_next_to_next_z = thumb_next_to_next.z
                
                index_to_hand_x = int(index_to_hand.x * frameWidth)
                index_to_hand_y = int(index_to_hand.y * frameHeight)
                index_to_hand_z = index_to_hand.z
                index_base_to_next_x = int(index_base_to_next.x * frameWidth)
                index_base_to_next_y = int(index_base_to_next.y * frameHeight)
                index_base_to_next_z = index_base_to_next.z
                index_next_to_next_x = int(index_next_to_next.x * frameWidth)
                index_next_to_next_y = int(index_next_to_next.y * frameHeight)
                index_next_to_next_z = index_next_to_next.z
                index_tip_x = int(index_tip.x * frameWidth)
                index_tip_y = int(index_tip.y * frameHeight)
                index_tip_z = index_tip.z
                
                middle_to_hand_x = int(middle_to_hand.x * frameWidth)
                middle_to_hand_y = int(middle_to_hand.y * frameHeight)
                middle_to_hand_z = middle_to_hand.z
                middle_base_to_next_x = int(middle_base_to_next.x * frameWidth)
                middle_base_to_next_y = int(middle_base_to_next.y * frameHeight)
                middle_base_to_next_z = middle_base_to_next.z
                middle_next_to_next_x = int(middle_next_to_next.x * frameWidth)
                middle_next_to_next_y = int(middle_next_to_next.y * frameHeight)
                middle_next_to_next_z = middle_next_to_next.z
                middle_tip_x = int(middle_tip.x * frameWidth)
                middle_tip_y = int(middle_tip.y * frameHeight)
                middle_tip_z = middle_tip.z
                
                ring_to_hand_x = int(ring_to_hand.x * frameWidth)
                ring_to_hand_y = int(ring_to_hand.y * frameHeight)
                ring_to_hand_z = ring_to_hand.z
                ring_base_to_next_x = int(ring_base_to_next.x * frameWidth)
                ring_base_to_next_y = int(ring_base_to_next.y * frameHeight)
                ring_base_to_next_z = ring_base_to_next.z
                ring_next_to_next_x = int(ring_next_to_next.x * frameWidth)
                ring_next_to_next_y = int(ring_next_to_next.y * frameHeight)
                ring_next_to_next_z = ring_next_to_next.z
                ring_tip_x = int(ring_tip.x * frameWidth)
                ring_tip_y = int(ring_tip.y * frameHeight)
                ring_tip_z = ring_tip.z
                
                pinky_to_hand_x = int(pinky_to_hand.x * frameWidth)
                pinky_to_hand_y = int(pinky_to_hand.y * frameHeight)
                pinky_to_hand_z = pinky_to_hand.z
                pinky_base_to_next_x = int(pinky_base_to_next.x * frameWidth)
                pinky_base_to_next_y = int(pinky_base_to_next.y * frameHeight)
                pinky_base_to_next_z = pinky_base_to_next.z
                pinky_next_to_next_x = int(pinky_next_to_next.x * frameWidth)
                pinky_next_to_next_y = int(pinky_next_to_next.y * frameHeight)
                pinky_next_to_next_z = pinky_next_to_next.z
                pinky_tip_x = int(pinky_tip.x * frameWidth)
                pinky_tip_y = int(pinky_tip.y * frameHeight)
                pinky_tip_z = pinky_tip.z
                
                """print(f"Wrist: \n x: {wrist_x} \n y: {wrist_y}")"""

                """print(f"ThumbToHand: \n x: {thumb_to_hand_x} \n y: {thumb_to_hand_y}")
                print(f"ThumbBaseToNext: \n x: {thumb_base_to_next_x} \n y: {thumb_base_to_next_y}")
                print(f"ThumbNextToNext: \n x: {thumb_next_to_next_x} \n y: {thumb_next_to_next_y}")"""
                #print(f"ThumbTip: \n x: {thumb_tip_x} \n y: {thumb_tip_y}")

                """print(f"IndexToHand: \n x: {index_to_hand_x} \n y: {index_to_hand_y}")
                print(f"IndexBaseToNext: \n x: {index_base_to_next_x} \n y: {index_base_to_next_y}")
                print(f"IndexNextToNext: \n x: {index_next_to_next_x} \n y: {index_next_to_next_y}")"""
                #print(f"IndexTip: \n x: {index_tip_x} \n y: {index_tip_y}")

                """print(f"MiddleToHand: \n x: {middle_to_hand_x} \n y: {middle_to_hand_y}")
                print(f"MiddleBaseToNext: \n x: {middle_base_to_next_x} \n y: {middle_base_to_next_y}")
                print(f"MiddleNextToNext: \n x: {middle_next_to_next_x} \n y: {middle_next_to_next_y}")
                print(f"MiddleTip: \n x: {middle_tip_x} \n y: {middle_tip_y}")

                print(f"RingToHand: \n x: {ring_to_hand_x} \n y: {ring_to_hand_y}")
                print(f"RingBaseToNext: \n x: {ring_base_to_next_x} \n y: {ring_base_to_next_y}")
                print(f"RingNextToNext: \n x: {ring_next_to_next_x} \n y: {ring_next_to_next_y}")
                print(f"RingTip: \n x: {ring_tip_x} \n y: {ring_tip_y}")

                print(f"PinkyToHand: \n x: {pinky_to_hand_x} \n y: {pinky_to_hand_y}")
                print(f"PinkyBaseToNext: \n x: {pinky_base_to_next_x} \n y: {pinky_base_to_next_y}")
                print(f"PinkyNextToNext: \n x: {pinky_next_to_next_x} \n y: {pinky_next_to_next_y}")
                print(f"PinkyTip: \n x: {pinky_tip_x} \n y: {pinky_tip_y}")"""

                #use = False
                
                if ((thumb_to_hand_y < wrist_y) and (thumb_base_to_next_y < thumb_to_hand_y) and (thumb_next_to_next_y  < thumb_base_to_next_y) and (thumb_tip_y < thumb_next_to_next_y)) and \
                    ((thumb_tip_x < index_tip_x) and (thumb_tip_x < middle_tip_x) and (thumb_tip_x < ring_tip_x) and (thumb_tip_x < pinky_tip_x)) and \
                    ((thumb_tip_y < index_tip_y) and (thumb_tip_y < middle_tip_y) and (thumb_tip_y < ring_tip_y) and (thumb_tip_y < pinky_tip_y)) and  \
                    ((thumb_tip_x < index_tip_x) and (thumb_tip_x < middle_tip_x) and (thumb_tip_x < ring_tip_x) and (thumb_tip_x < pinky_tip_x)) and \
                    ((index_tip_y > index_to_hand_y) and (middle_tip_y > middle_to_hand_y) and (ring_tip_y > ring_to_hand_y) and (pinky_tip_y > pinky_to_hand_y)):
                    cv2.putText(frame, "THUMBS UP!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Refined logic for Peace Sign
                elif (index_tip_y < index_next_to_next_y < index_base_to_next_y < index_to_hand_y) and \
                    (middle_tip_y < middle_next_to_next_y < middle_base_to_next_y < middle_to_hand_y) and \
                    (ring_tip_y > ring_base_to_next_y) and \
                    (pinky_tip_y > pinky_base_to_next_y) and \
                    (thumb_tip_x > index_tip_x) and \
                    (abs(middle_tip_x - index_tip_x) > 0.05 * frameWidth):
                    cv2.putText(frame, "PEACE!", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif (middle_tip_y < middle_next_to_next_y < middle_base_to_next_y < middle_to_hand_y) and \
                    (ring_tip_y < ring_next_to_next_y < ring_base_to_next_y < ring_to_hand_y) and \
                    (pinky_tip_y < pinky_next_to_next_y < pinky_base_to_next_y < pinky_to_hand_y) and \
                    (thumb_tip_y - index_tip_y <= 30) and \
                    (thumb_tip_x - index_tip_x <= 10):
                    cv2.putText(frame, "OKAY!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 2)
        # Show the result
        cv2.imshow('Hand Tracking - Press ESC to quit', frame)
        
        # Press ESC (key 27) to quit
        if cv2.waitKey(1) == 27:
            path = "C:/Users/mauro/OneDrive/Desktop/science-fair/OkaySignSnapShots"
            if not os.path.exists(path):
                os.makedirs(path)
            snapshot = np.array([
                [thumb_to_hand_x, thumb_to_hand_y, thumb_base_to_next_x, thumb_base_to_next_y, thumb_next_to_next_x, thumb_next_to_next_y, thumb_tip_x, thumb_tip_y],
                [index_to_hand_x, index_to_hand_y, index_base_to_next_x, index_base_to_next_y, index_next_to_next_x, index_next_to_next_y, index_tip_x, index_tip_y],
                [middle_to_hand_x, middle_to_hand_y, middle_base_to_next_x, middle_base_to_next_y, middle_next_to_next_x, middle_next_to_next_y, middle_tip_x, middle_tip_y],
                [ring_to_hand_x, ring_to_hand_y, ring_base_to_next_x, ring_base_to_next_y, ring_next_to_next_x, ring_next_to_next_y, ring_tip_x, ring_tip_y],
                [pinky_to_hand_x, pinky_to_hand_y, pinky_base_to_next_x, pinky_base_to_next_y, pinky_next_to_next_x, pinky_next_to_next_y, pinky_tip_x, pinky_tip_y]
            ])
            np.save(os.path.join(path, f"data{i}"), snapshot)
            print("Saved")
            i += 1
        elif cv2.waitKey(1) == 99:
            break
# Clean up
cv2.destroyAllWindows()
capture.release()