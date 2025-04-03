# main_auth.py
import cv2
import time
import sys
import platform
from face_analyzer import FaceAnalyzer
from database import get_db_connection, initialize_database, find_similar_face

# Configuration
CAMERA_INDEX = 0 # Default camera index
WINDOW_NAME = "Face Authentication - Press 'q' to quit"

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

FRAME_WIDTH = 640  # Optional: Set a specific width
FRAME_HEIGHT = 480 # Optional: Set a specific height
RECOGNITION_INTERVAL_SECONDS = 0.5 # How often to run full recognition (in seconds)

def run_authentication():
    """Runs the main face authentication loop."""
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
    # No need to initialize table here unless it might not exist,
    # but it's safer to ensure it's checked/created.
    if not initialize_database(conn):
         print("Warning: Failed to initialize the database, but proceeding. Recognition might fail.")
         # Decide if you want to exit here or proceed cautiously
         # conn.close()
         # return

    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        if conn: conn.close()
        return

    # Optional: Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n--- Face Authentication Running ---")
    print(f"Press 'q' in the window to quit.")

    last_recognition_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from camera.")
            break

        current_time = time.time()
        display_frame = frame.copy()
        recognized_names_in_frame = set() # Keep track of names identified in this frame

        # Run full analysis (detection + recognition + DB lookup) periodically
        if current_time - last_recognition_time >= RECOGNITION_INTERVAL_SECONDS:
            last_recognition_time = current_time
            # --- Face Analysis ---
            face_results = analyzer.analyze_frame(frame)

            if not face_results:
                 cv2.putText(display_frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # --- Process Each Detected Face ---
                for face in face_results:
                    bbox = face['bbox']
                    embedding = face['embedding']
                    det_score = face['det_score']

                    # --- Database Lookup ---
                    name, distance = find_similar_face(conn, embedding)

                    # --- Draw Bounding Box and Label ---
                    color = (0, 0, 255) # Red for unknown
                    label = "Unknown"
                    if name:
                        color = (0, 255, 0) # Green for known
                        label = f"{name} ({distance:.2f})"
                        if name not in recognized_names_in_frame:
                             print(f"AUTHENTICATED: {name} (Distance: {distance:.2f})")
                             recognized_names_in_frame.add(name) # Mark as printed for this frame
                    else:
                         # Optional: Label unknown faces with distance if you want
                         # if distance is not None and distance != float('inf'):
                         #     label = f"Unknown ({distance:.2f})"
                         pass # Keep label as "Unknown"

                    # Draw rectangle
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    # Draw label background
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(display_frame, (bbox[0], bbox[1] - text_size[1] - 4), (bbox[0] + text_size[0], bbox[1]), color, -1)
                    # Draw label text
                    cv2.putText(display_frame, label, (bbox[0], bbox[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # White text

        else:
            # --- Optional: Draw boxes from previous detection cycle for smoother visuals ---
            # You could store the last known bounding boxes and labels and redraw them here
            # This avoids flickering if detection/recognition is slower than frame rate.
            # For simplicity, we'll just show the frame without boxes between recognition intervals.
             pass # Or implement logic to redraw previous boxes

        # --- Display ---
        cv2.imshow(WINDOW_NAME, display_frame)

        # --- Key Handling ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit request received.")
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    if conn:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    run_authentication() 