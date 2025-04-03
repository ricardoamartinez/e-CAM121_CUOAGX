#!/bin/bash

# Set environment variables for display
export XAUTHORITY=/run/user/1000/gdm/Xauthority
export DISPLAY=:1

echo "Starting live view loop (Ctrl+C to stop)..."

while true; do
  # Step 1: Capture a single raw frame (4056x3040 BG10)
  echo "Capturing frame..."
  v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1 --stream-to=frame.raw
  if [ $? -ne 0 ]; then
    echo "Error capturing frame with v4l2-ctl. Exiting."
    break
  fi

  # Step 2: Process and display the frame using Python script
  echo "Processing and displaying frame..."
  python3 display_raw_frame.py
  if [ $? -ne 0 ]; then
    echo "Error running display script. Exiting."
    break
  fi

  # Optional: Add a small delay if needed, though script already waits 1 sec
  # sleep 0.1
done

echo "Live view loop stopped."
