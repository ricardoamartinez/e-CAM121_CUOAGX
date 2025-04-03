import cv2
import numpy as np
import os
import subprocess
import sys
import time

def run_command(command):
    """Helper function to run shell commands."""
    print(f"Executing: {command}")
    try:
        # Add a small delay before commands that might fail if device is busy
        time.sleep(0.1)
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        # print("Command successful.") # Reduce verbosity
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}")
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        print(f"An unexpected error occurred running command: {command}")
        print(f"Error: {e}")
        return False, str(e)

def process_frame(raw_file_path, width, height):
    """Reads raw Bayer frame, processes using BG_12bit, returns 8bit BGR."""
    dtype = np.uint16
    processed_frame = None
    try:
        expected_bytes = width * height * 2
        if not os.path.exists(raw_file_path) or os.path.getsize(raw_file_path) != expected_bytes:
            print(f"Error: Raw file '{raw_file_path}' missing or incorrect size.")
            return None

        raw_bayer_data = np.fromfile(raw_file_path, dtype=dtype, count=width * height)
        if raw_bayer_data.size != width * height:
             print(f"Error: Read {raw_bayer_data.size} pixels, expected {width * height}.")
             return None

        bayer_frame = raw_bayer_data.reshape((height, width))
        # print("  Read and reshaped raw frame.") # Reduce verbosity

        # Perform Bayer to BGR conversion (Reverting to BG pattern from v4l2-ctl)
        bgr_frame_16bit = cv2.cvtColor(bayer_frame, cv2.COLOR_BAYER_BG2BGR)
        # print("  Performed Bayer conversion (BG pattern).") # Reduce verbosity

        # Scale using 12-bit assumption (divide by 16 / right-shift 4)
        processed_frame = (bgr_frame_16bit >> 4).astype(np.uint8)
        # print("  Scaled frame to 8-bit.") # Reduce verbosity

    except Exception as e:
        print(f"  Error processing frame from {raw_file_path}: {e}")
        processed_frame = None # Ensure None is returned on error
    finally:
        # Clean up temporary raw file
        if os.path.exists(raw_file_path):
            try:
                os.remove(raw_file_path)
                # print(f"  Removed temporary file: {raw_file_path}") # Reduce verbosity
            except OSError as e:
                print(f"  Error removing temporary file {raw_file_path}: {e}")

    return processed_frame

def main():
    device = "/dev/video0"
    width = 4056
    height = 3040
    temp_raw_file = "temp_gain_exp.raw"

    gain_values = [0, 100, 500, 1000]
    exposure_values = [449, 5000, 10000, 17666] # min, low-mid, high-mid, max

    processed_frames = {} # Dictionary to store results {label: image}

    # Set environment variables for display
    os.environ['XAUTHORITY'] = '/run/user/1000/gdm/Xauthority'
    os.environ['DISPLAY'] = ':1'

    print("Starting gain/exposure test...")
    for gain in gain_values:
        for exposure in exposure_values:
            label = f"G{gain}_E{exposure // 1000}k" # Abbreviated label
            print(f"\nTesting: {label} (Gain={gain}, Exposure={exposure})")

            # Set format to BG12 first
            success_fmt, _ = run_command(f"v4l2-ctl -d {device} --set-fmt-video=pixelformat='BG12'")
            if not success_fmt:
                 print(f"  Failed to set format to BG12. Skipping combination.")
                 processed_frames[label] = None
                 continue

            # Set gain and exposure
            success_ctrl, _ = run_command(f"v4l2-ctl -d {device} -c gain={gain} -c exposure={exposure}")
            if not success_ctrl:
                print(f"  Failed to set controls. Skipping combination.")
                processed_frames[label] = None
                continue

            # Capture frame
            # Add retry logic for capture in case device is briefly busy
            capture_success = False
            for attempt in range(3):
                 success, output = run_command(f"v4l2-ctl -d {device} --stream-mmap --stream-count=1 --stream-to={temp_raw_file}")
                 if success:
                     capture_success = True
                     break
                 elif "Device or resource busy" in str(output):
                     print("  Device busy, retrying capture...")
                     time.sleep(0.5) # Wait before retry
                 else:
                     break # Other error, don't retry

            if not capture_success:
                print(f"  Failed to capture frame after retries. Skipping combination.")
                processed_frames[label] = None
                if os.path.exists(temp_raw_file): os.remove(temp_raw_file)
                continue

            # Process frame
            frame = process_frame(temp_raw_file, width, height)
            processed_frames[label] = frame # frame will be None if processing failed

    # --- Create Grid Display ---
    tile_width = 480 # Smaller tiles to fit more combinations if needed later
    tile_height = 360
    cols = len(exposure_values)
    rows = len(gain_values)
    grid_img = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    print("\nCreating grid image...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 # Smaller font for labels
    font_color = (0, 255, 0) # Green text
    line_type = 1

    row_idx = 0
    for gain in gain_values:
        col_idx = 0
        for exposure in exposure_values:
            label = f"G{gain}_E{exposure // 1000}k"
            img = processed_frames.get(label) # Use .get() for safety

            if img is None or img.size == 0:
                print(f"  Warning: Image for {label} is missing or empty.")
                img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8) # Black placeholder
                cv2.putText(img, f"{label}", (5, tile_height - 10), font, font_scale, (0,0,255), line_type) # Red text for error/label
            else:
                img = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                cv2.putText(img, label, (5, tile_height - 10), font, font_scale, font_color, line_type) # Label at bottom

            # Place tile in grid
            y_offset = row_idx * tile_height
            x_offset = col_idx * tile_width
            grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img

            col_idx += 1
        row_idx += 1

    # --- Display Grid ---
    window_name = 'Gain/Exposure Test (BG_12bit) (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Try to make window fit common screen sizes
    cv2.resizeWindow(window_name, min(1920, cols * tile_width), min(1080, rows * tile_height))
    cv2.imshow(window_name, grid_img)

    print(f"\nDisplaying grid. Press 'q' in the '{window_name}' window to quit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("Closed display window.")

if __name__ == "__main__":
    main()
