import cv2

def gstreamer_pipeline(
    capture_device="/dev/video0",
    width=1280,
    height=720,
    fps=30
):
    """
    Create a GStreamer pipeline string for Jetson Nano hardware MJPEG decoding.
    """
    return (
        f"v4l2src device={capture_device} io-mode=2 ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1, format=MJPG ! "
        "jpegparse ! nvv4l2decoder mjpeg=1 ! "
        "video/x-raw(memory:NVMM), format=I420 ! "
        "nvvidconv ! video/x-raw, format=BGR ! appsink"
    )

def main():
    # Build and open the GStreamer pipeline
    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    window_name = "USB Camera (press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame.")
            break

        cv2.imshow(window_name, frame)
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()