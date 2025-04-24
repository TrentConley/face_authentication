import cv2
import logging
import glob
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def gstreamer_pipeline(
    capture_device="/dev/video0",
    width=1280,
    height=720,
    fps=30
):
    """
    GStreamer pipeline for USB camera (MJPEG) on Jetson Nano via v4l2src.
    """
    return (
        f"v4l2src device={capture_device} io-mode=2 ! "
        f"image/jpeg, width={width}, height={height}, framerate={fps}/1, format=MJPG ! "
        "jpegparse ! jpegdec ! "
        "nvvidconv ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=1"
    )

def list_video_devices():
    """List /dev/video* devices."""
    devices = glob.glob('/dev/video*')
    devices = sorted(set(devices))
    logging.info(f"Detected video devices: {devices}")
    return devices

def prompt_select_device(devices):
    """Prompt user to select a video device from a list."""
    if not devices:
        logging.warning("No video devices found; defaulting to /dev/video0")
        return '/dev/video0'
    logging.info("Available video devices:")
    for idx, dev in enumerate(devices):
        logging.info(f"  [{idx}]: {dev}")
    choice = input(f"Select device [0-{len(devices)-1}] (default 0): ")
    try:
        idx = int(choice)
        if 0 <= idx < len(devices):
            return devices[idx]
    except Exception:
        pass
    return devices[0]

def main():
    # Detect and select capture device
    devices = list_video_devices()
    selected = prompt_select_device(devices)
    logging.info(f"Selected capture device: {selected}")
    # Build pipeline
    pipeline = gstreamer_pipeline(capture_device=selected)
    logging.info(f"Using GStreamer pipeline: {pipeline}")

    # Open VideoCapture with GStreamer backend
    logging.debug("Opening VideoCapture with GStreamer backend...")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logging.error("Unable to open video source with GStreamer pipeline.")
        # fallback: try plain /dev/video0
        cap = cv2.VideoCapture(selected)
        if not cap.isOpened():
            logging.error("Also failed to open selected device via default backend.")
            return
        else:
            logging.info("Opened selected device via default backend.")

    win = "USB Camera (press 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Stream ended or error reading frame.")
            break

        # Display frame
        cv2.imshow(win, frame)
        logging.debug(f"Displayed frame size: {frame.shape}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()