import cv2
import numpy as np
import os
import sys

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
            scaled_frame_8bit = (np.clip(bgr_frame_16bit, 0, 1023) / 4.0).astype(np.uint8)
        elif method == '12bit': # Divide by 16 (Right-shift 4)
            scaled_frame_8bit = (bgr_frame_16bit >> 4).astype(np.uint8)
        elif method == '16bit': # Divide by 256 (Right-shift 8)
            scaled_frame_8bit = (bgr_frame_16bit >> 8).astype(np.uint8)
        else:
            print(f"Unknown scaling method: {method}")
            scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) # Return black frame on error

    except Exception as e:
        print(f"Error during scaling method '{method}': {e}")
        scaled_frame_8bit = np.zeros_like(bgr_frame_16bit, dtype=np.uint8) # Return black frame on error

    # Ensure the output is 3 channels BGR
    if scaled_frame_8bit is not None and len(scaled_frame_8bit.shape) == 2:
        scaled_frame_8bit = cv2.cvtColor(scaled_frame_8bit, cv2.COLOR_GRAY2BGR)
    elif scaled_frame_8bit is None:
         scaled_frame_8bit = np.zeros((100,100,3), dtype=np.uint8) # Placeholder if scaling failed entirely

    return scaled_frame_8bit


def main(raw_file_path):
    width = 4056 # Use actual camera resolution
    height = 3040 # Use actual camera resolution
    dtype = np.uint16 # Assuming 10-bit Bayer data is read as 16-bit

    if not os.path.exists(raw_file_path):
        print(f"Error: Raw frame file '{raw_file_path}' not found.")
        return

    try:
        # --- Read Raw Data ---
        expected_bytes = width * height * 2
        actual_bytes = os.path.getsize(raw_file_path)
        if actual_bytes != expected_bytes:
             print(f"Warning: File size {actual_bytes} doesn't match expected {expected_bytes} for {width}x{height} 16-bit.")
             # Adjust count if file is smaller, otherwise numpy might error
             read_count = min(actual_bytes // 2, width * height)
             if read_count != width*height:
                 print(f"Adjusting read count to {read_count}")
        else: # File size matches expected_bytes - This else aligns with the outer if
            read_count = width * height # This line is indented under the else

        raw_bayer_data = np.fromfile(raw_file_path, dtype=dtype, count=read_count)
        if raw_bayer_data.size != read_count:
             print(f"Error: Read {raw_bayer_data.size} pixels, expected {read_count}.")
             return
        # Pad if file was smaller than expected
        if raw_bayer_data.size < width * height:
            raw_bayer_data = np.pad(raw_bayer_data, (0, width*height - raw_bayer_data.size))

        bayer_frame = raw_bayer_data.reshape((height, width))
        print("Successfully read and reshaped raw Bayer frame.")
        mean_raw_value = np.mean(bayer_frame)
        print(f"Mean raw pixel value: {mean_raw_value:.4f}")

        # --- Define Variations ---
        bayer_patterns = {
            "BG": cv2.COLOR_BAYER_BG2BGR,
            "GR": cv2.COLOR_BAYER_GR2BGR,
            "RG": cv2.COLOR_BAYER_RG2BGR,
            "GB": cv2.COLOR_BAYER_GB2BGR
        }
        scaling_methods = ["Norm", "10bit", "12bit", "16bit"]

        results = {}

        # --- Process Variations ---
        for pattern_name, pattern_code in bayer_patterns.items():
            print(f"Processing Bayer pattern: {pattern_name}...")
            try:
                bgr_frame_16bit = cv2.cvtColor(bayer_frame, pattern_code)
                for scale_name in scaling_methods:
                    print(f"  Scaling method: {scale_name}...")
                    label = f"{pattern_name}_{scale_name}"
                    results[label] = scale_frame(bgr_frame_16bit, scale_name)
            except cv2.error as e:
                print(f"  Error during Bayer conversion {pattern_name}: {e}")
                # Add black placeholders if conversion fails
                for scale_name in scaling_methods:
                     label = f"{pattern_name}_{scale_name}"
                     results[label] = np.zeros((100,100,3), dtype=np.uint8)


        # --- Create Grid Display ---
        tile_width = 640
        tile_height = 480
        cols = len(scaling_methods)
        rows = len(bayer_patterns)
        grid_img = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

        print("Creating grid image...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        line_type = 2

        row_idx = 0
        for pattern_name in bayer_patterns.keys():
            col_idx = 0
            for scale_name in scaling_methods:
                label = f"{pattern_name}_{scale_name}"
                img = results.get(label)
                if img is None or img.size == 0:
                    print(f"Warning: Image for {label} is missing or empty.")
                    img = np.zeros((tile_height, tile_width, 3), dtype=np.uint8) # Black placeholder
                else:
                    img = cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA)

                # Add label
                cv2.putText(img, label, (10, 30), font, font_scale, font_color, line_type)

                # Place tile in grid
                y_offset = row_idx * tile_height
                x_offset = col_idx * tile_width
                grid_img[y_offset:y_offset + tile_height, x_offset:x_offset + tile_width] = img

                col_idx += 1
            row_idx += 1

        # --- Display Grid ---
        window_name = 'Bayer/Scaling Variations (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Resize window to fit screen better if needed, but show full grid
        # cv2.resizeWindow(window_name, 1920, 1080) # Example resize
        cv2.imshow(window_name, grid_img)

        print(f"Displaying grid. Press 'q' in the '{window_name}' window to quit.")
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
        print("Usage: python display_raw_frame.py <path_to_raw_file>")
        # Fallback to default filename if no argument provided
        main('frame.raw')
