# e-CAM121_CUOAGX 4-Camera Viewer for Jetson AGX Orin

This repository contains a script to display the live feed from four e-con Systems e-CAM121_CUOAGX cameras connected to an NVIDIA Jetson AGX Orin.

## Script: `run_viewer.sh`

This script launches an optimized GStreamer pipeline that:
*   Uses `nvarguscamerasrc` to capture from all four cameras (sensor IDs 0-3).
*   Utilizes sensor mode 6 (~2K @ 60fps) for stable performance.
*   Leverages the Jetson ISP for automatic Bayer processing, AE, and AWB.
*   Composites the four streams into a 2x2 grid using `nvcompositor`.
*   Displays the final output using the low-latency `nveglglessink`.

## Usage

1.  Ensure the Jetson AGX Orin desktop environment is running.
2.  Open a terminal on the Jetson desktop.
3.  Navigate to this directory (`cd /path/to/ecam121`).
4.  Make the script executable (if not already): `chmod +x run_viewer.sh`
5.  Run the script: `./run_viewer.sh`

The script will set the necessary display environment variables and launch the GStreamer pipeline. A window showing the 2x2 camera grid should appear. Press `Ctrl+C` in the terminal to stop the pipeline.
