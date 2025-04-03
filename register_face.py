# register_face.py
import cv2
import time
import sys
import platform
from face_analyzer import FaceAnalyzer
from database import get_db_connection, initialize_database, add_face

# Configuration
CAMERA_INDEX = 0 # Default camera index (usually 0 or 1)
CAPTURE_DELAY_SECONDS = 2 # Wait time before capturing image
WINDOW_NAME = "Register Face - Press 'c' to capture, 'q' to quit"

# --- Determine Execution Providers ---
# Use CoreML on macOS for potential MPS/ANE acceleration, otherwise default to CPU
if platform.system() == "Darwin":
    print("Detected macOS. Setting providers to ['CoreMLExecutionProvider', 'CPUExecutionProvider'].")
    INSIGHTFACE_PROVIDERS = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
else:
    # Default to CPU for other OS. Add CUDA check here if desired.
    # Example for CUDA: INSIGHTFACE_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("Detected non-macOS. Setting providers to ['CPUExecutionProvider'].")
    INSIGHTFACE_PROVIDERS = ['CPUExecutionProvider']
# --- End Modified section ---

def capture_and_register():
    """Captures video, detects a face, extracts embedding, and registers it."""
    # Initialize Face Analyzer
    analyzer = FaceAnalyzer(providers=INSIGHTFACE_PROVIDERS)
    if not analyzer.app:
        print("Failed to initialize Face Analyzer. Exiting.")
        return

    # Initialize Database Connection
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return
    if not initialize_database(conn):
        print("Failed to initialize the database. Exiting.")
        conn.close()
        return

    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        conn.close()
        return

    print("\n--- Face Registration ---")
    print(f"Look at the camera. The system will try to detect your face.")
    print(f"Press 'c' when ready to capture the image for registration.")
    print(f"Press 'q' to quit.")

    capture_requested = False
    captured_frame = None
    embedding_to_save = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        display_frame = frame.copy() # Work on a copy

        # --- Face Detection for Visual Feedback ---
        # We run detection continuously for feedback, but only get embedding on capture
        face_results = analyzer.analyze_frame(frame) # Analyze the original frame

        if face_results:
            # Draw box around the most confident face found
            best_face = max(face_results, key=lambda x: x['det_score'])
            bbox = best_face['bbox']
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Score: {best_face['det_score']:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            if capture_requested:
                 # Re-analyze the captured frame specifically to get its embedding
                 embedding_to_save = analyzer.get_single_embedding(captured_frame)
                 if embedding_to_save is not None:
                    print("\nFace captured successfully!")
                    break # Exit loop to ask for name
                 else:
                    print("Could not get embedding from the captured frame. Please try again.")
                    capture_requested = False # Reset capture request

        else:
            cv2.putText(display_frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- Display ---
        cv2.imshow(WINDOW_NAME, display_frame)

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit request received.")
            embedding_to_save = None # Ensure we don't proceed if user quits
            break
        elif key == ord('c'):
            if face_results: # Only allow capture if a face is currently detected
                print("Capture key pressed. Analyzing frame...")
                captured_frame = frame.copy() # Save the current frame
                capture_requested = True
            else:
                print("Cannot capture: No face detected in the current frame.")


    # --- Cleanup Camera ---
    cap.release()
    cv2.destroyAllWindows()

    # --- Get Name and Save to DB ---
    if embedding_to_save is not None:
        while True:
            name = input("Enter the name for this person: ").strip()
            if name:
                if add_face(conn, name, embedding_to_save):
                    print(f"Successfully registered '{name}'.")
                else:
                    print(f"Failed to register '{name}'. Please check database logs.")
                break # Exit name input loop
            else:
                print("Name cannot be empty. Please try again.")
    else:
        print("Registration cancelled or failed.")

    # --- Close DB Connection ---
    if conn:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    capture_and_register() 