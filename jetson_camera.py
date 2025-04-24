import cv2

def gstreamer_pipeline(
    capture_device="/dev/video0",
    width=1280,
    height=720,
    fps=30
):
    """
    GStreamer pipeline for Jetson Nano hardware MJPEG decoding,
    converting from NVMM to system memory (BGRx → BGR), with drop=1.
    """
    return (
        f"v4l2src device={capture_device} io-mode=2 ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1, format=MJPG ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
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