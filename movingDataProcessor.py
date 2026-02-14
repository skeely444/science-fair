import cv2
import mediapipe as mp
import numpy as np
import os
import glob

# Setup MediaPipe ONCE
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- NECESSARY CHANGES START ---
SEQUENCE_LENGTH = 30 
FRAME_SKIP = 1 # We want every frame for motion
# --- NECESSARY CHANGES END ---

MAX_DIM = 640  

def process_video(video_path, sign_name, video_number):
    print(f"  Processing video {video_number}: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    # --- NECESSARY CHANGES START ---
    all_video_landmarks = [] # Store all frames of the video here first
    # --- NECESSARY CHANGES END ---
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize logic (Keeping your original)
            h, w = frame.shape[:2]
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # --- NECESSARY CHANGES START ---
                all_video_landmarks.append(landmarks)
                # --- NECESSARY CHANGES END ---

        cap.release()

        # --- NEW LOGIC: SAVE AS SEQUENCES ---
        samples_saved = 0
        save_path = f"new_data/{sign_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Sliding window: Create sequences of 30 frames
        if len(all_video_landmarks) >= SEQUENCE_LENGTH:
            for i in range(len(all_video_landmarks) - SEQUENCE_LENGTH + 1):
                window = all_video_landmarks[i : i + SEQUENCE_LENGTH]
                
                # Save the (30, 63) array
                file_name = f"{sign_name}_vid{video_number}_seq{i}.npy"
                np.save(os.path.join(save_path, file_name), np.array(window))
                samples_saved += 1
        
        return samples_saved
        # --- NEW LOGIC END ---

    except Exception as e:
        print(f"    ⚠️ Error: {e}")
        return 0

# (Keep your existing __main__ and loop logic as it was)


# ============================================
# MAIN PROCESSING - CHANGE THESE PATHS!
# ============================================

# Where are your video folders?
VIDEO_BASE_PATH = "C:/Users/mauro/OneDrive/Desktop/science-fair"  # ← CHANGE THIS!

# What are your sign folders called?
signs = {
    'ThumbsUpVideoFootage': 'good',  # folder name: sign name
    'PeaceVideoFootage': 'peace',
    'OkaySignVideoFootage': 'okay',
    'WaveVideoFootage': 'hello',
    'ThankYouVideoFootage': 'thanks'
}

# Process all videos automatically!
print("🚀 Starting automatic video processing...\n")

total_samples_all = 0

try:
    for folder_name, sign_name in signs.items():
        print(f"\n{'='*60}")
        print(f"📁 Processing {sign_name} videos...")
        print(f"{'='*60}")
        
        # Find all .mp4 files in this folder
        video_folder = os.path.join(VIDEO_BASE_PATH, folder_name)
        video_files = glob.glob(os.path.join(video_folder, '*.mp4'))
        
        # Also check for other formats
        video_files += glob.glob(os.path.join(video_folder, '*.avi'))
        video_files += glob.glob(os.path.join(video_folder, '*.mov'))
        
        print(f"Found {len(video_files)} videos\n")
        
        if len(video_files) == 0:
            print(f"⚠️  WARNING: No videos found in {video_folder}")
            print(f"    Make sure the path is correct!")
            continue
        
        sign_total = 0
        
        for i, video_path in enumerate(video_files, 1):
            samples = process_video(video_path, sign_name, i)
            sign_total += samples
        
        print(f"\n✅ {sign_name} complete: {sign_total} samples from {len(video_files)} videos")
        total_samples_all += sign_total

    print(f"\n{'='*60}")
    print(f"🎉 ALL DONE!")
    print(f"{'='*60}")
    print(f"Total samples collected: {total_samples_all}")
    print(f"\nData saved in 'data/' folder:")
    for sign_name in signs.values():
        folder = f'data/{sign_name}'
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.npy')])
            print(f"  {sign_name}: {count} files")

finally:
    hands.close()
    print("🔧 MediaPipe Hands closed.")