# e-CAM121_CUOAGX 4-Camera + 1 Thermal Viewer for Jetson AGX Orin

This repository contains a script to display the live feed from four e-con Systems e-CAM121_CUOAGX MIPI cameras and one USB thermal camera (e.g., FLIR Boson) connected to an NVIDIA Jetson AGX Orin.

## Script: `run_viewer.sh`

This script launches an optimized GStreamer pipeline that:
*   Uses `nvarguscamerasrc` to capture from all four MIPI cameras (sensor IDs 0-3).
*   Utilizes sensor mode 6 (~2K @ 60fps) for stable performance for the MIPI cameras.
*   Uses `v4l2src` for the USB thermal camera (currently configured for `/dev/video0`).
*   Leverages the Jetson ISP for automatic Bayer processing, AE, and AWB for the MIPI cameras.
*   Composites the five streams into a 3-over-2 grid using `nvcompositor`.
*   Includes `queue` elements for each stream to improve pipeline stability and synchronization.
*   Displays the final output using the low-latency `nveglglessink`.

## Usage

1.  Ensure the Jetson AGX Orin desktop environment is running.
2.  Open a terminal on the Jetson desktop.
3.  Navigate to this directory (`cd /path/to/this_project`).
4.  Make the script executable (if not already): `chmod +x run_viewer.sh`
5.  Run the script: `./run_viewer.sh`

The script will set the necessary display environment variables and launch the GStreamer pipeline. A window showing the 3-over-2 camera grid should appear. Press `Ctrl+C` in the terminal to stop the pipeline.

**Note on Thermal Camera Device:** The device node for the USB thermal camera (e.g., `/dev/video0`, `/dev/video1`, etc.) can sometimes change after a system reboot. If the thermal camera does not appear, you can use the command `sudo v4l2-ctl --list-devices | cat` to list available video devices and identify the correct node for the thermal camera (look for entries like "FLIR" or similar). You may need to update the `device=/dev/videoX` part in the `run_viewer.sh` script accordingly.

## Script: `run_yolo_viewer.py`

This Python script performs real-time YOLOv8 object detection on all five camera streams (4 MIPI RGB, 1 USB Thermal) and displays the results.

### Features:
*   **Input:** Reads from 4 MIPI cameras using `nvarguscamerasrc` and 1 USB thermal camera using `v4l2src`.
*   **Object Detection:** Utilizes a YOLOv8s model accelerated with NVIDIA TensorRT.
    *   The script will automatically attempt to build a TensorRT engine (`yolov8s.engine`) from an ONNX model (`/media/gt/My Passport/onnx_model/yolov8s.onnx`) if the engine file is not found. The engine file is included in this repository.
*   **Processing:**
    *   Frames are captured from GStreamer pipelines using `appsink`.
    *   Frames are preprocessed (resized, normalized) for the YOLOv8 model.
    *   Inference is performed using the TensorRT engine.
    *   Detections are postprocessed (confidence filtering, Non-Maximum Suppression).
    *   Bounding boxes and labels are drawn on the frames using OpenCV.
*   **Display:** Shows the processed video streams with detections in separate OpenCV windows.

### Dependencies:
*   Python 3
*   GStreamer (with Python bindings: `python3-gi`, `python3-gst-1.0`)
*   OpenCV (Python bindings: `python3-opencv`)
*   PyCUDA (`pip install pycuda`)
*   TensorRT (should be pre-installed with JetPack)
*   `trtexec` utility (for engine building, typically comes with TensorRT)

### Usage:
1.  Ensure all dependencies are installed.
2.  If the TensorRT engine (`yolov8s.engine`) is not present or needs rebuilding, ensure the ONNX model is accessible at `/media/gt/My Passport/onnx_model/yolov8s.onnx`. The script will attempt to build it.
3.  Navigate to this directory.
4.  Run the script: `python3 run_yolo_viewer.py`

The script will initialize the cameras, load/build the TensorRT engine, and then open windows for each camera stream, displaying bounding boxes for detected objects. Press `q` in an OpenCV window to close that stream, or `Ctrl+C` in the terminal to stop the entire script.
