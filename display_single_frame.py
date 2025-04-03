import cv2
import numpy as np
import os
import sys
import subprocess
import time

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

def process_bayer_frame(bayer_frame_16bit, pattern_code):
    """Performs Bayer conversion, 12-bit scaling, and inversion."""
    try:
        if bayer_frame_16bit is None: return None
        # Perform Bayer to BGR conversion
        # print(f"Performing Bayer conversion with pattern code: {pattern_code}...")
        bgr_frame_16bit = cv2.cvtColor(bayer_frame_16bit, pattern_code)
        # print("Conversion complete.")

        # Normalize the 16-bit BGR image to 0-255 range for 8-bit display
        min_val = np.min(bgr_frame_16bit)
        max_val = np.max(bgr_frame_16bit)
        # print(f"  BGR frame min/max before normalization: {min_val}/{max_val}") # Reduce verbosity
        if max_val > min_val:
             normalized_frame = (bgr_frame_16bit.astype(np.float32) - min_val) / (max_val - min_val)
             scaled_frame_8bit = (normalized_frame * 255).astype(np.uint8)
             # print("  Normalized BGR frame to uint8.") # Reduce verbosity
        else:
             scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) + int(min_val * 255 / 65535)
             # print("  Image data is flat, scaled uniform color.") # Reduce verbosity

        # Apply brightness inversion
        inverted_frame_8bit = 255 - scaled_frame_8bit
        # print("Applied brightness inversion.") # Reduce verbosity
        return inverted_frame_8bit
    except cv2.error as e:
        # print(f"  Error during Bayer conversion {pattern_code}: {e}")
        return None
    except Exception as e:
        # print(f"  Error during processing: {e}")
        return None

def capture_frame(device, gain, exposure, hdr, pixel_format, temp_file):
    """Sets controls and captures a single frame."""
    print(f"\nCapturing: G={gain}, E={exposure}, H={hdr}")
    # Set controls
    success_hdr, _ = run_command(f"v4l2-ctl -d {device} -c hdr_enable={hdr}")
    success_gain, _ = run_command(f"v4l2-ctl -d {device} -c gain={gain}")
    success_exp, _ = run_command(f"v4l2-ctl -d {device} -c exposure={exposure}")
    if not (success_hdr and success_gain and success_exp):
        print(f"  Failed to set controls for {device}.")
        return None

    # Capture frame with retries
    capture_success = False
    for attempt in range(2):
        success, output = run_command(f"v4l2-ctl -d {device} --set-fmt-video=pixelformat='{pixel_format}' --stream-mmap --stream-count=1 --stream-to={temp_file}")
        if success:
            capture_success = True
            break
        elif "Device or resource busy" in str(output):
            print("  Device busy, retrying capture...")
            time.sleep(0.5)
        else:
            break # Other error

    if not capture_success:
        print(f"  Failed to capture frame.")
        if os.path.exists(temp_file): os.remove(temp_file)
        return None

    # Read Raw Data
    width=4056
    height=3040
    dtype=np.uint16
    bayer_frame_16bit = None
    try:
        expected_bytes = width * height * 2
        if not os.path.exists(temp_file) or os.path.getsize(temp_file) != expected_bytes:
             print(f"  Error: Captured file '{temp_file}' missing or incorrect size.")
        else:
             raw_data_16 = np.fromfile(temp_file, dtype=dtype, count=width * height)
             if raw_data_16.size == width * height:
                 bayer_frame_16bit = raw_data_16.reshape((height, width))
                 print(f"  Read frame. Mean: {np.mean(bayer_frame_16bit):.2f}")
             else:
                 print(f"  Error reading uint16 data. Read {raw_data_16.size} elements.")
    except Exception as read_e:
        print(f"  Error reading raw file: {read_e}")
    finally:
         if os.path.exists(temp_file): os.remove(temp_file)

    return bayer_frame_16bit


def main():
    device = "/dev/video0"
    pixel_format = 'BG10'
    # Use the best hardware settings identified
    gain = 1000
    exposure = 449
    hdr = 0 # Assuming HDR Off looked better, adjust if needed
    # Use BG pattern, Norm scaling, Inversion ON
    bayer_pattern_code = cv2.COLOR_BAYER_BG2BGR # Use BG pattern

    # Set environment variables for display
    os.environ['XAUTHORITY'] = '/run/user/1000/gdm/Xauthority'
    os.environ['DISPLAY'] = ':1'

    # Capture frame with chosen settings
    bayer_frame_16bit = capture_frame(device, gain, exposure, hdr, pixel_format, "temp_single.raw")

    if bayer_frame_16bit is None:
        print("Failed to capture frame for processing.")
        return

    # --- Read and Process Little Endian ---
    print("\nReading raw data as uint16 Little Endian (<u2)...")
    processed_frame_le = None
    try:
        raw_data_le = np.fromfile("temp_single.raw", dtype='<u2', count=width * height)
        if raw_data_le.size == width * height:
            bayer_frame_le = raw_data_le.reshape((height, width))
            print("Processing frame (Little Endian)...")
            processed_frame_le = process_bayer_frame(bayer_frame_le, bayer_pattern_code)
        else:
            print("Error reading Little Endian data.")
    except Exception as e:
        print(f"Error processing Little Endian: {e}")


    # --- Read and Process Big Endian ---
    print("\nReading raw data as uint16 Big Endian (>u2)...")
    processed_frame_be = None
    try:
        raw_data_be = np.fromfile("temp_single.raw", dtype='>u2', count=width * height)
        if raw_data_be.size == width * height:
            bayer_frame_be = raw_data_be.reshape((height, width))
            print("Processing frame (Big Endian)...")
            processed_frame_be = process_bayer_frame(bayer_frame_be, bayer_pattern_code)
        else:
            print("Error reading Big Endian data.")
    except Exception as e:
        print(f"Error processing Big Endian: {e}")


    # --- Create Grid Display (1x2) ---
    tile_width = 640
    tile_height = 480 # Adjust aspect ratio if needed
    cols = 2
    rows = 1
    grid_img = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    print("\nCreating grid image...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 255, 0)
    line_type = 2

    image_map = {
        (0, 0): (processed_frame_le, "Little Endian"),
        (0, 1): (processed_frame_be, "Big Endian"),
    }

    for r in range(rows):
        for c in range(cols):
            img, label = image_map[(r, c)]
            if img is None or img.size == 0:
                img_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                cv2.putText(img_tile, f"{label} (Error)", (10, tile_height // 2), font, font_scale, (0,0,255), line_type)
            else:
                img_tile = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                cv2.putText(img_tile, label, (10, 30), font, font_scale, font_color, line_type)

            y_offset = r * tile_height
            x_offset = c * tile_width
            grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img_tile


    # --- Display Grid ---
    window_name = f'Endian Test (G{gain}_E{exposure}_H{hdr}_BG_Norm_Inv) (Press Q)'
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
