#!/bin/bash

# Script to launch GStreamer pipeline for 4x e-CAM121_CUOAGX cameras
# Uses nvarguscamerasrc (ISP processing) at 4K@60fps (Sensor Mode 0)
# Displays a 2x2 grid using nveglglessink for low latency.

# Set required environment variables for display
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

echo "Launching Optimized 4x 4K@60fps camera viewer pipeline (nveglglessink)..."
echo "Press Ctrl+C in the terminal to stop."

# Note: Running 4x 4K@60fps is extremely demanding. Performance may vary.
gst-launch-1.0 \
nvcompositor name=comp \
    sink_0::xpos=0   sink_0::ypos=0   sink_0::width=960 sink_0::height=540 \
    sink_1::xpos=960 sink_1::ypos=0   sink_1::width=960 sink_1::height=540 \
    sink_2::xpos=0   sink_2::ypos=540 sink_2::width=960 sink_2::height=540 \
    sink_3::xpos=960 sink_3::ypos=540 sink_3::width=960 sink_3::height=540 \
! 'video/x-raw(memory:NVMM),format=RGBA' \
! nvegltransform ! nveglglessink sync=false -e \
nvarguscamerasrc sensor-id=0 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! comp.sink_0 \
nvarguscamerasrc sensor-id=1 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! comp.sink_1 \
nvarguscamerasrc sensor-id=2 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! comp.sink_2 \
nvarguscamerasrc sensor-id=3 sensor-mode=0 ! 'video/x-raw(memory:NVMM), width=4056, height=3040, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! comp.sink_3

echo "Pipeline stopped."
