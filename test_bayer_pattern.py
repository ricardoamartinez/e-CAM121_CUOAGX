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
    """Performs Bayer conversion and 10-bit scaling."""
    try:
        # Perform Bayer to BGR conversion
        bgr_frame_16bit = cv2.cvtColor(bayer_frame_16bit, pattern_code)

        # Scale using 12-bit assumption (divide by 16 / right-shift 4)
        scaled_frame_8bit = (bgr_frame_16bit >> 4).astype(np.uint8)

        # Apply Gray World automatic white balance
        # Calculate average R, G, B values (handling potential division by zero)
        avg_b = np.mean(bgr_frame_16bit[:, :, 0])
        avg_g = np.mean(bgr_frame_16bit[:, :, 1])
        avg_r = np.mean(bgr_frame_16bit[:, :, 2])
        avg_gray = (avg_b + avg_g + avg_r) / 3.0

        if avg_b == 0 or avg_g == 0 or avg_r == 0:
             print("  Warning: Zero average in a channel, skipping AWB.")
             wb_frame_16bit = bgr_frame_16bit
        else:
             # Calculate scaling factors
             scale_b = avg_gray / avg_b
             scale_g = avg_gray / avg_g
             scale_r = avg_gray / avg_r

             # Apply scaling factors (clip to prevent overflow)
             wb_frame_16bit = np.clip(bgr_frame_16bit * np.array([[[scale_b, scale_g, scale_r]]]), 0, 65535).astype(np.uint16)
             print(f"  Applied Gray World AWB (Scales B:{scale_b:.2f}, G:{scale_g:.2f}, R:{scale_r:.2f})")


        # Scale using 12-bit assumption (divide by 16 / right-shift 4)
        scaled_frame_8bit = (wb_frame_16bit >> 4).astype(np.uint8)

        # Invert the 8-bit image (Removed based on previous feedback)
        # inverted_frame_8bit = 255 - scaled_frame_8bit
        # return inverted_frame_8bit
        return scaled_frame_8bit
    except cv2.error as e:
        print(f"  Error during Bayer conversion {pattern_code}: {e}")
        return None
    except Exception as e:
        print(f"  Error during scaling: {e}")
        return None


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

        # --- Process Frame with AWB ---
        print("Processing frame with BG pattern and Gray World AWB...")
        processed_frame = process_bayer_frame(bayer_frame_16bit, cv2.COLOR_BAYER_BG2BGR)

        if processed_frame is None:
            print("Error processing frame.")
            return

        # --- Display Single Image ---
        window_name = 'Processed Frame (BG, 12bit Scale, GrayWorld AWB) (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) # Adjust size as needed
        cv2.imshow(window_name, processed_frame)

        print(f"\nDisplaying image. Press 'q' in the '{window_name}' window to quit.")
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Closed display window.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python test_bayer_pattern.py <path_to_raw_file>")
        # Fallback to default filename if no argument provided
        main('frame.raw')
