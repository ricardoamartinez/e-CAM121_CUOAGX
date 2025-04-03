#!/bin/bash

# Script to launch GStreamer pipeline for 4x e-CAM121_CUOAGX cameras
# Uses nvarguscamerasrc (ISP processing) at ~2K@60fps (Sensor Mode 6) for stability
# Displays a 2x2 grid using autovideosink

# Set required environment variables for display
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

echo "Launching Stable 4x Camera Viewer Pipeline (Sensor Mode 6, autovideosink)..."
echo "Press Ctrl+C in the terminal to stop."

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

echo "Pipeline stopped."
