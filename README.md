# e-CAM121_CUOAGX Scripts for Jetson AGX Orin

This repository contains scripts developed to test, troubleshoot, and display video feeds from e-con Systems e-CAM121_CUOAGX (Sony IMX412 sensor) cameras connected to an NVIDIA Jetson AGX Orin development kit via a camera link module.

## Overview

The primary goal was to set up a system to view the live feeds from four connected e-CAM121_CUOAGX cameras. Initial attempts using standard V4L2 capture (`v4l2src`) and processing (OpenCV, GStreamer `bayer2rgb`) encountered difficulties due to the camera's raw Bayer format (`BG10` - likely packed MIPI RAW10) and incompatibilities with standard GStreamer elements (`nvbayerdemux` missing, `bayer2rgb` linking errors) on the tested JetPack/L4T version.

Extensive troubleshooting led to the following findings:
*   Direct raw frame capture using `v4l2-ctl` is possible, but requires specific processing for the Bayer format.
*   Displaying graphical output required setting `DISPLAY=:1` and `XAUTHORITY=/run/user/1000/gdm/Xauthority`.
*   The most reliable method for live viewing uses the `nvarguscamerasrc` GStreamer element, which leverages the Jetson ISP for automatic Bayer processing, white balance, and exposure control.

## Scripts

*   **`display_all_cameras.py`**: Python script using OpenCV and `v4l2-ctl` to capture and display single frames from all four cameras in a 2x2 grid. Uses manually set gain/exposure and includes Bayer processing logic (BG pattern, 12-bit scaling). *Note: This was part of the troubleshooting process and is less efficient than the GStreamer pipeline below.*
*   **`display_single_frame.py`**: Python script to capture a single frame from `/dev/video0` using `v4l2-ctl` and test various processing parameters (Bayer pattern, scaling, inversion).
*   **`test_gain_exposure.py`**: Python script to test a grid of different gain and exposure settings for `/dev/video0`, displaying the results for comparison.
*   **`test_bayer_pattern.py`**: Python script to test different Bayer pattern conversions on a captured raw frame.
*   **`test_processing_combos.py`**: Python script to test combinations of Bayer pattern, scaling, and inversion on a single captured frame.
*   **`comprehensive_test.py`**: Python script testing combinations of hardware (gain, exposure, HDR) and software (Bayer, scaling) settings.
*   **`unpack_display_raw10.py`**: Experimental script attempting to unpack MIPI RAW10 format (unsuccessful).
*   **`read_raw_test.py`**: Script to visualize raw captured data interpreted as uint8 vs uint16.
*   **`live_view_loop.sh`**: Basic shell script attempting a pseudo-live view by looping single frame capture/display (inefficient).
*   **`capture_display.py`**: Early OpenCV test script.

## Recommended Usage: GStreamer Pipeline

The most stable and performant method found for viewing all four cameras uses `nvarguscamerasrc`:

```bash
# Ensure display variables are set correctly for your session
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

# Run the pipeline (Sensor Mode 6: ~1080p @ 60fps per camera)
gst-launch-1.0 \
nvcompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 \
                      sink_1::xpos=960 sink_1::ypos=0 sink_1::width=960 sink_1::height=540 \
                      sink_2::xpos=0 sink_2::ypos=540 sink_2::width=960 sink_2::height=540 \
                      sink_3::xpos=960 sink_3::ypos=540 sink_3::width=960 sink_3::height=540 \
! nvvidconv ! 'video/x-raw, format=RGBA' ! queue ! videoconvert ! autovideosink sync=false -e \
nvarguscamerasrc sensor-id=0 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! comp.sink_0 \
nvarguscamerasrc sensor-id=1 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! comp.sink_1 \
nvarguscamerasrc sensor-id=2 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! comp.sink_2 \
nvarguscamerasrc sensor-id=3 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! comp.sink_3
```

To run at full 4K resolution (more demanding):
```bash
# Ensure display variables are set correctly for your session
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

# Run the pipeline (Sensor Mode 0: 4K @ 60fps per camera)
gst-launch-1.0 \
nvcompositor name=comp sink_0::xpos=0 sink_0::ypos=0 sink_0::width=960 sink_0::height=540 \
                      sink_1::xpos=960 sink_1::ypos=0 sink_1::width=960 sink_1::height=540 \
                      sink_2::xpos=0 sink_2::ypos=540 sink_2::width=960 sink_2::height=540 \
                      sink_3::xpos=960 sink_3::ypos=540 sink_3::width=960 sink_3::height=540 \
! nvvidconv ! 'video/x-raw, format=RGBA' ! queue ! videoconvert ! autovideosink sync=false -e \
nvarguscamerasrc sensor-id=0 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! comp.sink_0 \
nvarguscamerasrc sensor-id=1 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! comp.sink_1 \
nvarguscamerasrc sensor-id=2 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! comp.sink_2 \
nvarguscamerasrc sensor-id=3 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! comp.sink_3
```

## Notes
*   The Python scripts relying on `v4l2-ctl` capture and manual Bayer processing were primarily for debugging and may not produce optimal image quality without further tuning (e.g., white balance).
*   Performance, especially with 4K streams, depends heavily on the Jetson AGX Orin's load and power mode.
