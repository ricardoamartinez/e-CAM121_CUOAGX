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
