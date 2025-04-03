import cv2
import numpy as np
import os
import sys

def main(raw_file_path):
    width = 4056
    height = 3040
    raw_file = raw_file_path

    if not os.path.exists(raw_file):
        print(f"Error: Raw frame file '{raw_file}' not found.")
        return

    try:
        # --- Read as uint16 ---
        print("Reading raw data as uint16...")
        expected_bytes_16 = width * height * 2
        actual_bytes = os.path.getsize(raw_file)
        if actual_bytes != expected_bytes_16:
            print(f"Warning: File size {actual_bytes} != expected {expected_bytes_16} for uint16.")
        
        # Read assuming uint16
        raw_data_16 = np.fromfile(raw_file, dtype=np.uint16, count=width * height)
        if raw_data_16.size != width * height:
            print(f"Error reading uint16 data. Read {raw_data_16.size} elements, expected {width*height}")
            img_16bit_scaled = np.zeros((100, 100), dtype=np.uint8) # Placeholder
        else:
            # Skip uint16 processing for now
            pass

        # --- Read as uint8 ---
        print("Reading raw data as uint8...")
        expected_bytes_8 = width * height * 2 # Still expect 2 bytes per pixel total
        raw_data_8 = np.fromfile(raw_file, dtype=np.uint8, count=expected_bytes_8)
        
        if raw_data_8.size != expected_bytes_8:
             print(f"Error reading uint8 data. Read {raw_data_8.size} bytes, expected {expected_bytes_8}")
             img_8bit_direct = np.zeros((100, 100), dtype=np.uint8) # Placeholder
        else:
            # Reshape as (height, width * 2) because each pixel takes 2 bytes
            img_8bit_direct = raw_data_8.reshape((height, width * 2))
            print(f"Read as uint8. Shape: {img_8bit_direct.shape}, Mean: {np.mean(img_8bit_direct):.2f}")

            # Apply histogram equalization to enhance contrast
            print("Applying histogram equalization to uint8 data...")
            try:
                # EqualizeHist expects a 2D grayscale image
                # We need to decide how to treat the (height, width*2) shape.
                # Option 1: Treat it as one wide grayscale image
                # Option 2: Try reshaping to (height*2, width) ? Less likely.
                # Option 3: Treat pairs of bytes? Complex.
                # Let's try Option 1 first.
                equalized_img_8bit = cv2.equalizeHist(img_8bit_direct)
                print("Histogram equalization applied.")
            except cv2.error as e:
                 print(f"Error during equalizeHist: {e}")
                 equalized_img_8bit = img_8bit_direct # Fallback to original if error


        # --- Display Single Image ---
        display_img = equalized_img_8bit
        window_name = 'Raw Data (uint8, Equalized) (Press Q to quit)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) # Adjust size as needed

        print("\nDisplaying processed uint8 image...")
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow(window_name, display_img)

        print(f"Displaying image. Press 'q' in the '{window_name}' window to quit.")
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
        print("Usage: python read_raw_test.py <path_to_raw_file>")
        # Fallback to default filename if no argument provided
        main('frame.raw')
