#!/bin/bash

# Script to launch GStreamer pipeline for 4x e-CAM121_CUOAGX cameras and 1x USB Thermal Camera
# Uses nvarguscamerasrc (ISP processing) for MIPI cameras (Sensor Mode 6)
# Uses v4l2src for the USB thermal camera
# Displays a 3-over-2 grid using nveglglessink for low latency.

# Set required environment variables for display
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

echo "Launching Low-Latency 4x MIPI + 1x USB Thermal Camera Viewer Pipeline..."
echo "Press Ctrl+C in the terminal to stop."

gst-launch-1.0 \
nvcompositor name=comp \
    sink_0::xpos=0    sink_0::ypos=0    sink_0::width=640 sink_0::height=480 \
    sink_1::xpos=640  sink_1::ypos=0    sink_1::width=640 sink_1::height=480 \
    sink_2::xpos=1280 sink_2::ypos=0    sink_2::width=640 sink_2::height=480 \
    sink_3::xpos=320  sink_3::ypos=480  sink_3::width=640 sink_3::height=480 \
    sink_4::xpos=960  sink_4::ypos=480  sink_4::width=640 sink_4::height=480 \
! 'video/x-raw(memory:NVMM),format=RGBA' \
! nvegltransform ! nveglglessink sync=false -e \
nvarguscamerasrc sensor-id=0 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! queue ! comp.sink_0 \
nvarguscamerasrc sensor-id=1 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! queue ! comp.sink_1 \
nvarguscamerasrc sensor-id=2 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! queue ! comp.sink_2 \
nvarguscamerasrc sensor-id=3 sensor-mode=6 ! 'video/x-raw(memory:NVMM), width=2028, height=1112, format=NV12, framerate=60/1' ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! queue ! comp.sink_3 \
v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! 'video/x-raw(memory:NVMM),format=RGBA' ! queue ! comp.sink_4

echo "Pipeline stopped."
