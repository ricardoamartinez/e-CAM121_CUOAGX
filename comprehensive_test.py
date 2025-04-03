import cv2
import numpy as np
import os
import sys
import subprocess
import time
import math

# --- Helper Functions ---

def run_command(command):
    """Helper function to run shell commands."""
    # print(f"Executing: {command}") # Reduce verbosity
    try:
        time.sleep(0.1) # Small delay
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
        if bgr_frame_16bit is None: return None # Handle None input

        if method == 'Norm':
            min_val = np.min(bgr_frame_16bit)
            max_val = np.max(bgr_frame_16bit)
            if max_val > min_val:
                normalized = (bgr_frame_16bit.astype(np.float32) - min_val) / (max_val - min_val)
                scaled_frame_8bit = (normalized * 255).astype(np.uint8)
            else:
                scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) + int(min_val * 255 / 65535)
        elif method == '10bit': # Divide by 4
            scaled_frame_8bit = (np.clip(bgr_frame_16bit, 0, 1023) / 4.0).astype(np.uint8)
        elif method == '12bit': # Divide by 16 (Right-shift 4)
            scaled_frame_8bit = (bgr_frame_16bit >> 4).astype(np.uint8)
        elif method == '16bit': # Divide by 256 (Right-shift 8)
            scaled_frame_8bit = (bgr_frame_16bit >> 8).astype(np.uint8)
        else:
            # print(f"Unknown scaling method: {method}")
            scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8)

    except Exception as e:
        # print(f"Error during scaling method '{method}': {e}")
        scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) if bgr_frame_16bit is not None else None

    # Ensure the output is 3 channels BGR
    if scaled_frame_8bit is not None and len(scaled_frame_8bit.shape) == 2:
        scaled_frame_8bit = cv2.cvtColor(scaled_frame_8bit, cv2.COLOR_GRAY2BGR)
    elif scaled_frame_8bit is None:
         scaled_frame_8bit = np.zeros((100,100,3), dtype=np.uint8) # Placeholder

    return scaled_frame_8bit

def process_bayer_frame(bayer_frame_16bit, pattern_code, scaling_method, apply_inversion):
    """Performs Bayer conversion, scaling, and optional inversion."""
    try:
        if bayer_frame_16bit is None: return None
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

def main():
    device = "/dev/video0"
    width = 4056
    height = 3040
    temp_raw_file = "temp_comprehensive.raw"
    pixel_format = 'BG10' # Request BG10, expect 16bpp file
    dtype = np.uint16 # Read as 16-bit

    # --- Define Parameter Space ---
    # Focus on the hardware settings from the "4th to last row"
    gain_values = [1000]
    exposure_values = [449]
    hdr_values = [0, 1]
    bayer_patterns = { "BG": cv2.COLOR_BAYER_BG2BGR, "GR": cv2.COLOR_BAYER_GR2BGR }
    scaling_methods = ["Norm", "10bit", "12bit", "16bit"]
    inversion_options = [True] # Test WITH inversion enabled

    results = {} # Dictionary to store results {label: image}
    capture_params_list = [] # List to store hardware params for grid layout

    # Set environment variables for display
    os.environ['XAUTHORITY'] = '/run/user/1000/gdm/Xauthority'
    os.environ['DISPLAY'] = ':1'

    print("--- Comprehensive Test ---")
    total_captures = len(gain_values) * len(exposure_values) * len(hdr_values)
    capture_count = 0

    for gain in gain_values:
        for exposure in exposure_values:
            for hdr in hdr_values:
                capture_count += 1
                hw_label = f"G{gain}_E{exposure//1000}k_H{hdr}"
                print(f"\n({capture_count}/{total_captures}) Capturing for: {hw_label}")
                capture_params_list.append({'gain': gain, 'exposure': exposure, 'hdr': hdr})

                # Set controls
                success_hdr, _ = run_command(f"v4l2-ctl -d {device} -c hdr_enable={hdr}")
                success_gain, _ = run_command(f"v4l2-ctl -d {device} -c gain={gain}")
                success_exp, _ = run_command(f"v4l2-ctl -d {device} -c exposure={exposure}")

                if not (success_hdr and success_gain and success_exp):
                    print(f"  Failed to set controls. Skipping capture.")
                    # Store None for all processing variations for this capture
                    for pattern_name in bayer_patterns.keys():
                        for scale_name in scaling_methods:
                            for invert in inversion_options:
                                inv_label = "Inv" if invert else "NoInv"
                                label = f"{hw_label}_{pattern_name}_{scale_name}_{inv_label}"
                                results[label] = None
                    continue # Skip to next hardware combo

                # Capture frame with retries
                capture_success = False
                for attempt in range(2): # Fewer retries for faster test
                    success, output = run_command(f"v4l2-ctl -d {device} --set-fmt-video=pixelformat='{pixel_format}' --stream-mmap --stream-count=1 --stream-to={temp_raw_file}")
                    if success:
                        capture_success = True
                        break
                    elif "Device or resource busy" in str(output):
                        print("  Device busy, retrying capture...")
                        time.sleep(0.5)
                    else:
                        break

                if not capture_success:
                    print(f"  Failed to capture frame. Skipping processing.")
                    bayer_frame_16bit = None
                else:
                    # Read Raw Data
                    try:
                        expected_bytes = width * height * 2
                        if not os.path.exists(temp_raw_file) or os.path.getsize(temp_raw_file) != expected_bytes:
                             print(f"  Error: Captured file '{temp_raw_file}' missing or incorrect size.")
                             bayer_frame_16bit = None
                        else:
                             raw_data_16 = np.fromfile(temp_raw_file, dtype=dtype, count=width * height)
                             if raw_data_16.size == width * height:
                                 bayer_frame_16bit = raw_data_16.reshape((height, width))
                             else:
                                 print(f"  Error reading uint16 data. Read {raw_data_16.size} elements.")
                                 bayer_frame_16bit = None
                    except Exception as read_e:
                        print(f"  Error reading raw file: {read_e}")
                        bayer_frame_16bit = None
                    finally:
                         if os.path.exists(temp_raw_file): os.remove(temp_raw_file)


                # Process frame using different software settings
                for pattern_name, pattern_code in bayer_patterns.items():
                    for scale_name in scaling_methods:
                        for invert in inversion_options:
                            inv_label = "Inv" if invert else "NoInv"
                            label = f"{hw_label}_{pattern_name}_{scale_name}_{inv_label}"
                            if bayer_frame_16bit is not None:
                                results[label] = process_bayer_frame(bayer_frame_16bit, pattern_code, scale_name, invert)
                            else:
                                results[label] = None # Store None if capture failed

    # --- Create Grid Display ---
    # Grid layout: Rows represent Gain/Exposure/HDR combos, Columns represent Bayer/Scaling combos
    num_hw_combos = len(capture_params_list)
    num_proc_combos = len(bayer_patterns) * len(scaling_methods) * len(inversion_options) # Should be 8

    # Determine grid dimensions (2 HW x 8 Proc = 16 images -> 4x4 grid)
    cols = 4
    rows = 4
    print(f"\nTotal images: {len(results)}. Creating {rows}x{cols} grid...")

    tile_width = 480 # Larger tiles for fewer images
    tile_height = int(tile_width * height / width)
    grid_img = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 # Larger font
    font_color = (0, 255, 0)
    line_type = 1

    img_index = 0
    for hw_params in capture_params_list:
        gain = hw_params['gain']
        exposure = hw_params['exposure']
        hdr = hw_params['hdr']
        hw_label = f"G{gain}_E{exposure//1000}k_H{hdr}"

        for pattern_name in bayer_patterns.keys():
            for scale_name in scaling_methods:
                for invert in inversion_options:
                    inv_label = "Inv" if invert else "NoInv"
                    label = f"{hw_label}_{pattern_name}_{scale_name}_{inv_label}"
                    # Use full label for clarity with fewer images
                    # short_label = f"G{gain}E{exposure//1000}H{hdr}_{pattern_name[0]}{scale_name[0]}{inv_label[0]}"

                    img = results.get(label)

                    row_idx = img_index // cols
                    col_idx = img_index % cols

                    if img is None or img.size == 0:
                        img_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                        cv2.putText(img_tile, label, (5, tile_height - 10), font, font_scale, (0,0,255), line_type) # Full label, Red
                    else:
                        img_tile = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                        cv2.putText(img_tile, label, (5, tile_height - 10), font, font_scale, font_color, line_type) # Full label, Green

                    # Place tile in grid
                    y_offset = row_idx * tile_height
                    x_offset = col_idx * tile_width
                    if y_offset + tile_height <= grid_img.shape[0] and x_offset + tile_width <= grid_img.shape[1]:
                         grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img_tile
                    else:
                         print(f"Warning: Tile for {label} out of grid bounds.")

                    img_index += 1

    # --- Display Grid ---
    window_name = 'Focused Test Grid (G1000_E0k) (Press Q to quit)'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid_img)

    print(f"\nDisplaying grid (Rows: HDR0/HDR1, Cols: Bayer/Scaling). Press 'q' to quit.")
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
