import cv2
import numpy as np
import time

def main():
    # Try opening the camera using the default backend
    print("Trying default backend...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device /dev/video0 using default backend.")
        return

    print("Successfully opened video device.")

    # Assume camera is pre-configured by v4l2-ctl
    # Read back properties to see what was actually set
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Actual settings read by OpenCV: {int(width)}x{int(height)} @ {fps}fps, FOURCC: {fourcc_str} ({fourcc_int})")

    # No need to check width/height here as we assume v4l2-ctl set it

    window_name = 'Camera Feed (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # Start with a reasonable window size

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Error: Can't receive frame (stream end or error). Exiting ...")
            # Add a small delay in case it's a temporary issue
            time.sleep(0.1)
            # Try reading again once
            ret, frame = cap.read()
            if not ret or frame is None:
                 break # Exit if still failing

        frame_count += 1

        # --- Bayer Conversion Logic ---
        # Check if the frame is single channel (expected for raw Bayer)
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # Squeeze if it has a trailing dimension of 1
            if len(frame.shape) == 3:
                frame = np.squeeze(frame, axis=2)

            try:
                # Assume BG Bayer pattern based on 'BG10'
                # cvtColor handles 8-bit or 16-bit input for Bayer
                # If the camera outputs 10-bit, OpenCV might read it as 16-bit (upper bits zero)
                # or potentially scale it.
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                display_frame = bgr_frame
            except cv2.error as e:
                print(f"Error during Bayer conversion (COLOR_BAYER_BG2BGR): {e}", end='\r')
                # Display raw frame if conversion fails (will look monochrome/patterned)
                # Convert raw frame to 3 channels for display consistency if needed
                display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Check if frame is already BGR (some backends might auto-convert)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
             print("Frame received is already 3 channels (likely BGR). Displaying directly.", end='\r')
             display_frame = frame
        else:
             print(f"Warning: Received frame with unexpected shape: {frame.shape}. Skipping display.", end='\r')
             continue # Skip this frame

        # Display the resulting frame
        cv2.imshow(window_name, display_frame)

        # Exit on 'q' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n'q' pressed, exiting.")
            break
        elif key != 255: # 255 is returned when no key is pressed
            print(f"\nKey pressed: {key}")


    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    print(f"\nCaptured {frame_count} frames in {elapsed_time:.2f} seconds ({actual_fps:.2f} FPS).")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Released camera and closed windows.")

if __name__ == "__main__":
    main()
