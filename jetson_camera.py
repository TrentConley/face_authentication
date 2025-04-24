import cv2

def gstreamer_pipeline(
    capture_device=0,
    width=1280,
    height=720,
    fps=30,
    flip_method=0
):
    """
    GStreamer pipeline for Jetson Nano CSI camera using NVIDIA Argus camera.
    """
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
    )

def main():
    pipeline = gstreamer_pipeline()
    print(f"Using pipeline:\n{pipeline}\n")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        # fallback: try plain /dev/video0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Also failed to open /dev/video0 via the default backend.")
            return
        else:
            print("Opened /dev/video0 via default backend.")

    win = "USB Camera (press 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or error reading frame.")
            break

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()