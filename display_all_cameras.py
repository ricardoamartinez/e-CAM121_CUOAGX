import cv2
import numpy as np
import os
import subprocess
import time

def run_command(command):
    """Helper function to run shell commands."""
    print(f"Executing: {command}")
    try:
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("Command successful.")
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
        print("  Read and reshaped raw frame.")

        # Perform Bayer to BGR conversion (Trying GR pattern)
        bgr_frame_16bit = cv2.cvtColor(bayer_frame, cv2.COLOR_BAYER_GR2BGR)
        print("  Performed Bayer conversion (GR pattern).")

        # Scale using 12-bit assumption (divide by 16 / right-shift 4)
        scaled_frame_8bit = (bgr_frame_16bit >> 4).astype(np.uint8)
        processed_frame = scaled_frame_8bit # Assign scaled frame directly
        # Hardware noise reduction control not available via V4L2

    except Exception as e:
        print(f"  Error processing frame from {raw_file_path}: {e}")
        processed_frame = None # Ensure None is returned on error
    finally:
        # Clean up temporary raw file
        if os.path.exists(raw_file_path):
            try:
                os.remove(raw_file_path)
                print(f"  Removed temporary file: {raw_file_path}")
            except OSError as e:
                print(f"  Error removing temporary file {raw_file_path}: {e}")

    return processed_frame

def main():
    camera_devices = [f"/dev/video{i}" for i in range(4)]
    width = 4056
    height = 3040
    gain = 100 # Reduced gain
    exposure = 17666 # Set exposure to maximum
    processed_frames = []

    # Set environment variables for display
    os.environ['XAUTHORITY'] = '/run/user/1000/gdm/Xauthority'
    os.environ['DISPLAY'] = ':1'

    for i, device in enumerate(camera_devices):
        print(f"\nProcessing Camera {i} ({device})...")
        temp_raw_file = f"temp_cam_{i}.raw"

        # Set gain and exposure
        success, _ = run_command(f"v4l2-ctl -d {device} -c gain={gain} -c exposure={exposure}")
        if not success:
            print(f"  Failed to set controls for {device}. Skipping.")
            processed_frames.append(None) # Add placeholder
            continue

        # Capture frame
        success, _ = run_command(f"v4l2-ctl -d {device} --stream-mmap --stream-count=1 --stream-to={temp_raw_file}")
        if not success:
            print(f"  Failed to capture frame from {device}. Skipping.")
            processed_frames.append(None) # Add placeholder
            # Attempt to remove potentially incomplete file
            if os.path.exists(temp_raw_file): os.remove(temp_raw_file)
            continue

        # Process frame
        frame = process_frame(temp_raw_file, width, height)
        processed_frames.append(frame) # frame will be None if processing failed

    # --- Create Grid Display ---
    tile_width = 640 # Target size for each tile in the grid
    tile_height = 480
    grid_rows = 2
    grid_cols = 2
    grid_img = np.zeros((grid_rows * tile_height, grid_cols * tile_width, 3), dtype=np.uint8)

    print("\nCreating grid image...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0) # Green text
    line_type = 2

    for i in range(grid_rows * grid_cols):
        row_idx = i // grid_cols
        col_idx = i % grid_cols
        label = f"Cam {i}"
        img = processed_frames[i] if i < len(processed_frames) else None

        if img is None or img.size == 0:
            print(f"  Warning: Image for Cam {i} is missing or empty.")
            img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8) # Black placeholder
            cv2.putText(img, f"{label} (Error)", (10, tile_height // 2), font, font_scale, (0,0,255), line_type) # Red text for error
        else:
            img = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            cv2.putText(img, label, (10, 30), font, font_scale, font_color, line_type)

        # Place tile in grid
        y_offset = row_idx * tile_height
        x_offset = col_idx * tile_width
        grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img

    # --- Display Grid ---
    window_name = 'All Cameras - Processed Frames (BG_12bit) (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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
