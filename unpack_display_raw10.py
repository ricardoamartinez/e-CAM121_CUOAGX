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

def unpack_mipi_raw10(raw_bytes, width, height):
    """
    Unpacks MIPI RAW10 packed data (4 pixels in 5 bytes) into a uint16 NumPy array.
    Assumes the common packing: P1[9:2], P2[9:2], P3[9:2], P4[9:2], P1[1:0]P2[1:0]P3[1:0]P4[1:0]
    """
    print("Unpacking MIPI RAW10 data using NumPy vectorization...")
    expected_pixels = width * height
    expected_bytes = expected_pixels * 5 // 4 # Each 4 pixels take 5 bytes

    if raw_bytes.size < expected_bytes:
        print(f"Error: Raw data size ({raw_bytes.size}) is smaller than expected ({expected_bytes}).")
        return None
    elif raw_bytes.size > expected_bytes:
        print(f"Warning: Raw data size ({raw_bytes.size}) is larger than expected ({expected_bytes}). Trimming.")
        raw_bytes = raw_bytes[:expected_bytes]
    
    if raw_bytes.size % 5 != 0:
         print(f"Error: Trimmed data size ({raw_bytes.size}) is not a multiple of 5.")
         return None

    # Reshape into groups of 5 bytes
    packed_data = raw_bytes.reshape(-1, 5)

    # Extract MSBs (bytes 0-3) and LSBs (byte 4)
    msb1 = packed_data[:, 0].astype(np.uint16)
    msb2 = packed_data[:, 1].astype(np.uint16)
    msb3 = packed_data[:, 2].astype(np.uint16)
    msb4 = packed_data[:, 3].astype(np.uint16)
    lsbs = packed_data[:, 4].astype(np.uint16)

    # Extract LSB pairs
    lsb1 = (lsbs >> 6) & 0x03
    lsb2 = (lsbs >> 4) & 0x03
    lsb3 = (lsbs >> 2) & 0x03
    lsb4 = (lsbs >> 0) & 0x03

    # Combine MSBs and LSBs
    p1 = (msb1 << 2) | lsb1
    p2 = (msb2 << 2) | lsb2
    p3 = (msb3 << 2) | lsb3
    p4 = (msb4 << 2) | lsb4

    # Interleave the pixels: p1, p2, p3, p4, p1, p2, p3, p4, ...
    # Create an empty array of the final size
    unpacked_pixels = np.empty(expected_pixels, dtype=np.uint16)
    # Assign pixels in strides
    unpacked_pixels[0::4] = p1
    unpacked_pixels[1::4] = p2
    unpacked_pixels[2::4] = p3
    unpacked_pixels[3::4] = p4

    # Check final size before reshape
    if unpacked_pixels.size != expected_pixels:
        print(f"Error: Final unpacked pixel count ({unpacked_pixels.size}) does not match expected ({expected_pixels}).")
        return None

    print("Unpacking successful.")
    return unpacked_pixels.reshape((height, width))


def main(raw_file_path):
    width = 4056
    height = 3040
    raw_file = raw_file_path

    if not os.path.exists(raw_file):
        print(f"Error: Raw frame file '{raw_file}' not found.")
        return

    try:
        # --- Read Raw Bytes ---
        print("Reading raw byte stream...")
        raw_bytes = np.fromfile(raw_file, dtype=np.uint8)
        print(f"Read {raw_bytes.size} bytes.")

        # --- Unpack MIPI RAW10 ---
        bayer_frame_10bit = unpack_mipi_raw10(raw_bytes, width, height)

        if bayer_frame_10bit is None:
            print("Failed to unpack MIPI RAW10 data.")
            return

        print(f"Unpacked frame shape: {bayer_frame_10bit.shape}, Mean: {np.mean(bayer_frame_10bit):.2f}")

        # --- Process Frame ---
        # Perform Bayer to BGR conversion (BG pattern)
        print("Performing Bayer to BGR conversion (BG pattern)...")
        bgr_frame_10bit = cv2.cvtColor(bayer_frame_10bit, cv2.COLOR_BAYER_BG2BGR)
        print("Conversion complete.")

        # Scale the 10-bit BGR image (0-1023 range) to 8-bit (0-255)
        scale_factor = 255.0 / 1023.0
        display_frame_8bit = (bgr_frame_10bit * scale_factor).astype(np.uint8)
        print("Scaled BGR frame to uint8 assuming 10-bit range.")


        # --- Display Image ---
        window_name = 'Unpacked MIPI RAW10 (BG, 10bit Scale) (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) # Adjust size as needed
        cv2.imshow(window_name, display_frame_8bit)

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
        print("Usage: python unpack_display_raw10.py <path_to_raw_file>")
        # Fallback to default filename if no argument provided
        main('frame.raw')
