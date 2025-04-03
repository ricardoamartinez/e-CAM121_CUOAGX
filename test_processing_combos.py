import cv2
import numpy as np
import os
import sys
import subprocess
import time

# --- Helper Functions ---

def run_command(command):
    """Helper function to run shell commands."""
    print(f"Executing: {command}")
    try:
        time.sleep(0.1)
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {command}\nError: {e}\nStderr: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        print(f"An unexpected error occurred running command: {command}\nError: {e}")
        return False, str(e)

def scale_frame(bgr_frame_16bit, method):
    """Scales a 16-bit BGR frame to 8-bit using different methods."""
    scaled_frame_8bit = None
    try:
        if method == 'Norm':
            min_val = np.min(bgr_frame_16bit)
            max_val = np.max(bgr_frame_16bit)
            if max_val > min_val:
                normalized = (bgr_frame_16bit.astype(np.float32) - min_val) / (max_val - min_val)
                scaled_frame_8bit = (normalized * 255).astype(np.uint8)
            else:
                scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) + int(min_val * 255 / 65535)
        elif method == '10bit': # Divide by 4
            # Clip to 0-1023 before scaling
            scaled_frame_8bit = (np.clip(bgr_frame_16bit, 0, 1023) / 4.0).astype(np.uint8)
        elif method == '12bit': # Divide by 16 (Right-shift 4)
            scaled_frame_8bit = (bgr_frame_16bit >> 4).astype(np.uint8)
        elif method == '16bit': # Divide by 256 (Right-shift 8)
            scaled_frame_8bit = (bgr_frame_16bit >> 8).astype(np.uint8)
        else:
            print(f"Unknown scaling method: {method}")
            scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8)

    except Exception as e:
        print(f"Error during scaling method '{method}': {e}")
        scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8)

    # Ensure the output is 3 channels BGR
    if scaled_frame_8bit is not None and len(scaled_frame_8bit.shape) == 2:
        scaled_frame_8bit = cv2.cvtColor(scaled_frame_8bit, cv2.COLOR_GRAY2BGR)
    elif scaled_frame_8bit is None:
         scaled_frame_8bit = np.zeros((100,100,3), dtype=np.uint8) # Placeholder

    return scaled_frame_8bit

def process_bayer_frame(bayer_frame_16bit, pattern_code, scaling_method, apply_inversion):
    """Performs Bayer conversion, scaling, and optional inversion."""
    try:
        # Perform Bayer to BGR conversion
        bgr_frame_16bit = cv2.cvtColor(bayer_frame_16bit, pattern_code)

        # Scale to 8-bit
        scaled_frame_8bit = scale_frame(bgr_frame_16bit, scaling_method)

        # Apply inversion if requested
        if apply_inversion:
            processed_frame = 255 - scaled_frame_8bit
        else:
            processed_frame = scaled_frame_8bit

        return processed_frame
    except cv2.error as e:
        # print(f"  Error during Bayer conversion {pattern_code}: {e}")
        return None
    except Exception as e:
        # print(f"  Error during processing: {e}")
        return None

# --- Main Execution ---

def main(raw_file_path):
    width = 4056
    height = 3040
    raw_file = raw_file_path
    dtype = np.uint16 # Read as 16-bit

    if not os.path.exists(raw_file):
        print(f"Error: Raw frame file '{raw_file}' not found.")
        return

    try:
        # --- Read Raw Data (as uint16) ---
        print("Reading raw data as uint16...")
        expected_bytes = width * height * 2
        actual_bytes = os.path.getsize(raw_file)
        if actual_bytes != expected_bytes:
            print(f"Warning: File size {actual_bytes} != expected {expected_bytes} for uint16.")

        raw_data_16 = np.fromfile(raw_file, dtype=dtype, count=width * height)
        if raw_data_16.size != width * height:
            print(f"Error reading uint16 data. Read {raw_data_16.size} elements, expected {width*height}")
            return

        bayer_frame_16bit = raw_data_16.reshape((height, width))
        print(f"Read as uint16. Shape: {bayer_frame_16bit.shape}, Mean: {np.mean(bayer_frame_16bit):.2f}")

        # --- Define Variations ---
        bayer_patterns = {
            "BG": cv2.COLOR_BAYER_BG2BGR,
            "GR": cv2.COLOR_BAYER_GR2BGR,
            "RG": cv2.COLOR_BAYER_RG2BGR,
            "GB": cv2.COLOR_BAYER_GB2BGR
        }
        scaling_methods = ["Norm", "10bit", "12bit", "16bit"]
        inversion_options = [False, True] # NoInv, Inv
        results = {}

        # --- Process Variations ---
        print("\nProcessing all combinations...")
        for pattern_name, pattern_code in bayer_patterns.items():
            for scale_name in scaling_methods:
                for invert in inversion_options:
                    invert_label = "Inv" if invert else "NoInv"
                    label = f"{pattern_name}_{scale_name}_{invert_label}"
                    # print(f"  Testing: {label}") # Reduce verbosity
                    results[label] = process_bayer_frame(bayer_frame_16bit, pattern_code, scale_name, invert)
        print("Processing complete.")

        # --- Create Grid Display ---
        # Arrange as 8 columns (4 scaling x 2 inversion) and 4 rows (Bayer patterns)
        tile_width = 240 # Smaller tiles for 8 columns
        tile_height = int(tile_width * height / width)
        cols = len(scaling_methods) * len(inversion_options) # 8 columns
        rows = len(bayer_patterns) # 4 rows
        grid_img = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

        print("\nCreating grid image...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4 # Smaller font for more labels
        font_color = (0, 255, 0) # Green text
        line_type = 1

        row_idx = 0
        for pattern_name in bayer_patterns.keys():
            col_idx = 0
            for scale_name in scaling_methods:
                for invert in inversion_options:
                    invert_label = "Inv" if invert else "NoInv"
                    label = f"{pattern_name}_{scale_name}_{invert_label}"
                    img = results.get(label)

                    if img is None or img.size == 0:
                        # print(f"  Warning: Image for {label} is missing or empty.")
                        img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8) # Black placeholder
                        cv2.putText(img, f"{label}", (5, tile_height - 5), font, font_scale, (0,0,255), line_type)
                    else:
                        img = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                        cv2.putText(img, label, (5, tile_height - 5), font, font_scale, font_color, line_type) # Label at bottom

                    # Place tile in grid
                    y_offset = row_idx * tile_height
                    x_offset = col_idx * tile_width
                    grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img
                    col_idx += 1
            row_idx += 1

        # --- Display Grid ---
        window_name = 'Processing Combinations Test (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Resize window to fit screen better if needed
        cv2.resizeWindow(window_name, min(1920, cols * tile_width), min(1080, rows * tile_height))
        cv2.imshow(window_name, grid_img)

        print(f"\nDisplaying grid (4 rows=Bayer Pattern, 8 columns=Scaling+Inversion). Press 'q' to quit.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Closed display window.")

if __name__ == "__main__":
    # Set fixed hardware parameters for capture
    device = "/dev/video0"
    gain = 100
    exposure = 10000
    pixel_format = 'BG10'
    raw_file = 'frame_combo_test.raw'

    print("--- Phase 1: Testing Processing Combinations ---")
    print(f"Setting controls: gain={gain}, exposure={exposure}")
    success_ctrl, _ = run_command(f"v4l2-ctl -d {device} -c gain={gain} -c exposure={exposure}")
    if not success_ctrl:
        sys.exit("Failed to set camera controls.")

    print(f"Capturing frame with format {pixel_format}...")
    success_cap, _ = run_command(f"v4l2-ctl -d {device} --set-fmt-video=pixelformat='{pixel_format}' --stream-mmap --stream-count=1 --stream-to={raw_file}")
    if not success_cap:
        sys.exit("Failed to capture frame.")

    print("\nRunning processing test script...")
    main(raw_file)

    # Clean up raw file
    if os.path.exists(raw_file):
        try:
            os.remove(raw_file)
            print(f"Removed temporary file: {raw_file}")
        except OSError as e:
            print(f"Error removing temporary file {raw_file}: {e}")
