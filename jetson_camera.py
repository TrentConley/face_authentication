import cv2
import logging
import glob
import os
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logging.info(f"Starting jetson_camera.py - OpenCV version: {cv2.__version__}")
if not hasattr(cv2, 'CAP_GSTREAMER'):
    logging.warning("OpenCV not built with GStreamer support; CAP_GSTREAMER unavailable.")
else:
    logging.debug("GStreamer backend flag present.")

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
    logging.info("Initializing camera capture application.")
    # Desired capture settings
    width, height, fps = 1280, 720, 30

    # Detect and select capture device
    devices = list_video_devices()
    selected = prompt_select_device(devices)
    logging.info(f"Selected capture device: {selected}")

    # Build GStreamer pipeline string (used if default fails)
    pipeline = gstreamer_pipeline(capture_device=selected, width=width, height=height, fps=fps)
    logging.info(f"Prepared GStreamer pipeline: {pipeline}")

    # Attempt default OpenCV (V4L2) backend first with MJPEG settings
    logging.debug(f"Attempting default V4L2 backend for device {selected} with MJPG settings...")
    cap = cv2.VideoCapture(selected)
    if cap.isOpened():
        # Force MJPEG and resolution/FPS
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        logging.info("Opened selected device via default V4L2 backend successfully with MJPG.")
        backend = 'default'
    else:
        # Fallback to GStreamer
        logging.warning("Default V4L2 backend failed; trying GStreamer pipeline...")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logging.error("Unable to open video source with GStreamer pipeline.")
            return
        logging.info("Opened video source with GStreamer backend successfully.")
        backend = 'gstreamer'

    # Log actual capture properties
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Capture properties ({backend}): Width={cap_width}, Height={cap_height}, FPS={cap_fps}")

    win = "USB Camera (press 'q' to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 600)

    # Begin capture loop
    logging.info("Starting capture loop. Press 'q' in the window to exit.")
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Stream ended or error reading frame.")
            break

        # Display frame
        cv2.imshow(win, frame)
        logging.debug(f"Displayed frame size: {frame.shape}")
        frame_count += 1
        # Log FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps_real = frame_count / elapsed if elapsed > 0 else 0
            logging.info(f"Captured {frame_count} frames in {elapsed:.2f}s ({fps_real:.2f} FPS)")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Quit signal received. Exiting capture loop.")
            break
    # End of capture
    total_time = time.time() - start_time
    logging.info(f"Capture loop terminated after {frame_count} frames and {total_time:.2f}s")

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Resources released. Exiting application.")

if __name__ == "__main__":
    main()