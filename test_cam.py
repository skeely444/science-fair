import cv2
import mediapipe

drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

capture = cv2.VideoCapture(0)
frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

with handsModule.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=1
) as hands:
    
    while (True):
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                
                drawingModule.draw_landmarks(
                    frame, 
                    handLandmarks, 
                    handsModule.HAND_CONNECTIONS
                )
                
                # Get landmarks
                wrist = handLandmarks.landmark[0]
                index_tip = handLandmarks.landmark[8]
                index_mcp = handLandmarks.landmark[5]  # Base of index finger
                
                # Calculate depth relative to wrist
                index_depth = index_tip.z - wrist.z
                
                # If index tip is significantly closer than wrist = pointing at camera!
                if index_depth < -0.05:  # Closer by 5% or more
                    cv2.putText(frame, "POINTING AT CAMERA!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    print(f"Pointing! Depth: {index_depth:.3f}")
                
                # Show depth value on screen
                depth_text = f"Index depth: {index_depth:.3f}"
                cv2.putText(frame, depth_text, (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Depth Detection - Press ESC to quit', frame)
        
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()
capture.release()