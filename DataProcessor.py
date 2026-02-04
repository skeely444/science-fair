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

FRAME_SKIP = 2 # Change this if you want more/fewer samples
MAX_DIM = 640  # Resize frames so MediaPipe doesn't hang on very large frames

def process_video(video_path, sign_name, video_number):
    """Process one video and save landmarks"""
    
    print(f"  Processing video {video_number}: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"    ❌ Could not open video")
        return 0
    
    frame_count = 0
    samples_saved = 0
    
    # Keep frames to a reasonable size to avoid very long MediaPipe processing
    max_dim = MAX_DIM
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % FRAME_SKIP == 0:
                # Resize if the frame is very large (keeps aspect ratio)
                h, w = frame.shape[:2]
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                except KeyboardInterrupt:
                    # Stop processing the current video on interruption, but don't abort the whole run
                    print("    ⛔ Stopping this video due to interruption (continuing with next video).")
                    break
                except Exception as e:
                    # Catch MediaPipe or other unexpected errors for this frame and stop the video
                    print(f"    ⚠️ Error processing frame: {e}")
                    break
                
                if results and results.multi_hand_landmarks:
                    landmarks = []
                    for lm in results.multi_hand_landmarks[0].landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    os.makedirs(f'data/{sign_name}', exist_ok=True)
                    filename = f'data/{sign_name}/{sign_name}_v{video_number}_f{frame_count}.npy'
                    np.save(filename, landmarks)
                    samples_saved += 1
            
            frame_count += 1
    finally:
        cap.release()

    print(f"    ✅ Saved {samples_saved} samples")
    return samples_saved


# ============================================
# MAIN PROCESSING - CHANGE THESE PATHS!
# ============================================

# Where are your video folders?
VIDEO_BASE_PATH = "C:/Users/mauro/OneDrive/Desktop/science-fair"  # ← CHANGE THIS!

# What are your sign folders called?
signs = {
    'ThumbsUpVideoFootage': 'thumbs_up',  # folder name: sign name
    'PeaceVideoFootage': 'peace',
    'OkaySignVideoFootage': 'ok'
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