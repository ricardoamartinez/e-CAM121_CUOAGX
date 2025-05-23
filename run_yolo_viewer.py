#!/usr/bin/env python3

import sys
import os
import time
import subprocess
import gi
import warnings # Added
import threading # Added
import queue # Added
import traceback  # add this at the top if not already imported

# Attempt to import and initialize PyCUDA early and make its availability clear
PYCUDA_AVAILABLE = False
try:
    import pycuda.driver as cuda
    print("PyCUDA driver imported.")
    PYCUDA_AVAILABLE = True
except ImportError:
    print("ERROR: PyCUDA not found or failed to initialize CUDA context.")
    print("Please install PyCUDA (e.g., pip install pycuda) and ensure your CUDA environment is correctly set up.")
    print("TensorRT inference with direct GPU buffer management will not be possible without PyCUDA.")
    # Forcing exit if PyCUDA is critical for the application's core functionality
    # sys.exit("Exiting: PyCUDA is required for GPU operations.") 
    # For now, let it proceed to allow testing other parts if engine already exists, 
    # but inference will be non-functional on GPU.
except Exception as e:
    print(f"ERROR: An unexpected error occurred during PyCUDA import/initialization: {e}")
    print("Please check your PyCUDA and CUDA setup.")
    # sys.exit("Exiting: Error during PyCUDA initialization.")

gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstApp
import cv2 # OpenCV for image processing
import tensorrt as trt # For TensorRT runtime
import numpy as np

# Constants
ONNX_MODEL_PATH = "US_BTR80.onnx"
ENGINE_FILE_PATH = "US_BTR80.engine" # .engine or .trt
NUM_MIPI_CAMERAS = 3
THERMAL_CAM_DEVICE = "/dev/video1" # As determined previously
DYNAMIC_THERMAL_CAM_DEVICE = None # Will be set by find_thermal_camera_device

# GStreamer pipeline elements (placeholders, to be detailed)
# Example for one camera with appsink:
# nvarguscamerasrc sensor-id=0 ! ... ! appsink name=sink0
# v4l2src device=/dev/video0 ! ... ! appsink name=sink_thermal

# YOLOv8 specific constants (these may need adjustment based on the exact model)
MIPI_SENSOR_MODE = 6
MIPI_WIDTH = 2028
MIPI_HEIGHT = 1112
MIPI_FRAMERATE = 60
TARGET_WIDTH = 640 # Added
TARGET_HEIGHT = 640 # Added

# Expected YOLO input shape (channels, height, width)
YOLO_MODEL_PRECISION = np.float32
TARGET_YOLO_FPS_PER_STREAM = 2.0 # Target FPS for YOLO processing per stream (Reduced from 10.0)
# Let's assume model expects BGR, CHW, and specific size (e.g., 640x640)
# This will be confirmed/set when loading the engine
YOLO_INPUT_SHAPE = None # (3, 640, 640) - Will be set from engine
YOLO_OUTPUT_NAMES = None # Will be set from engine

# Global list to hold references to active GStreamer pipelines and app sinks
_active_pipelines = []

PYCUDA_INITIALIZED_SUCCESSFULLY = False

# Globals for TensorRT engine and context (if PyCUDA is available)
tensorrt_engine = None
tensorrt_context = None
host_inputs, cuda_inputs, host_outputs, cuda_outputs = [], [], [], []
stream = None
YOLO_MODEL_OUTPUT_SHAPE = None # Will be set from engine

# Globals for combined view
latest_frames = {}
frame_lock = threading.Lock()
TILE_WIDTH = TARGET_WIDTH  # Use target resolution - Will be updated in main()
TILE_HEIGHT = TARGET_HEIGHT # Use target resolution - Will be updated in main()
last_inference_submit_time = {} # For FPS throttling
latest_raw_camera_frames = {} # Stores latest raw BGR frames from cameras
latest_detected_objects = {}  # Stores latest (detections_list, original_frame_shape, timestamp)

# Globals for worker thread # Added section
frame_input_queue = queue.Queue(maxsize=(NUM_MIPI_CAMERAS + 1) * 2) # Allow some buffering
detections_output_queue = queue.Queue() # NEW - for detection results from worker
stop_event = threading.Event()

# Filter the specific deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Use get_tensor_name instead.*")

def get_scaled_dimensions(original_w, original_h, target_w, target_h):
    """Calculates scaled dimensions to fit within target_w, target_h, preserving aspect ratio."""
    if original_w == 0 or original_h == 0: # Prevent division by zero
        return target_w, target_h # Or some other default, like original scaled if possible

    original_aspect = original_w / original_h
    target_aspect = target_w / target_h

    if original_aspect > target_aspect:
        # Original is wider than target: scale by width
        scaled_w = target_w
        scaled_h = int(target_w / original_aspect)
    else:
        # Original is taller than or same aspect as target: scale by height
        scaled_h = target_h
        scaled_w = int(target_h * original_aspect)
    
    # Ensure dimensions are at least 1x1 and even (often preferred by video encoders/elements)
    scaled_w = max(1, scaled_w // 2 * 2) 
    scaled_h = max(1, scaled_h // 2 * 2)
    return scaled_w, scaled_h

def find_thermal_camera_device(keywords=["FLIR", "Boson", "Thermal"]):
    """
    Finds the thermal camera device by parsing v4l2-ctl output.
    Looks for specified keywords in the device names.
    Returns the path to the first video node (e.g., /dev/videoX) found for such a device.
    """
    print("Attempting to find thermal camera device...")
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices', '--verbose'], capture_output=True, text=True, check=True)
        output_lines = result.stdout.splitlines()
        
        current_device_name = None
        for i, line in enumerate(output_lines):
            line_stripped = line.strip()
            if not line_stripped: # Skip empty lines
                continue

            # Check if this line is a device name (usually not indented)
            # and contains any of the keywords
            if not line.startswith(('\\t', ' ')) and any(keyword.lower() in line_stripped.lower() for keyword in keywords):
                current_device_name = line_stripped
                print(f"Found potential thermal camera section: '{current_device_name}'")
                # Look for /dev/videoX in the immediately following lines
                for j in range(i + 1, len(output_lines)):
                    next_line_stripped = output_lines[j].strip()
                    if not next_line_stripped: # Skip empty lines
                        continue
                    if next_line_stripped.startswith("/dev/video"):
                        print(f"Found device node: {next_line_stripped} for {current_device_name}")
                        return next_line_stripped
                    elif not output_lines[j].startswith(('\\t', ' ')):
                        # We've hit the next device name without finding a /dev/video node for the current one
                        current_device_name = None # Reset, as this was not it or format is unexpected
                        break 
                current_device_name = None # Reset if no /dev/video found under it
        
        print("No thermal camera device found matching keywords or /dev/videoX node structure.")
        return None
    except FileNotFoundError:
        print("ERROR: v4l2-ctl command not found. Is v4l-utils installed?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"ERROR: v4l2-ctl command failed: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while finding thermal camera: {e}")
        return None

def check_or_build_engine(onnx_path, engine_path):
    """
    Checks if a TensorRT engine file exists, and if not, builds it from the ONNX model.
    Uses subprocess with a timeout and redirects trtexec output to log files.
    """
    print(f"Checking for TensorRT engine: {engine_path}")
    if not os.path.exists(engine_path):
        print(f"Engine not found. Building from ONNX: {onnx_path}")
        
        trtexec_path = "/usr/src/tensorrt/bin/trtexec"
        if not os.path.exists(trtexec_path):
            print(f"ERROR: trtexec not found at {trtexec_path}.")
            trtexec_path_alt = "/opt/tensorrt/bin/trtexec"
            if os.path.exists(trtexec_path_alt):
                print(f"Found trtexec at alternative path: {trtexec_path_alt}")
                trtexec_path = trtexec_path_alt
            else:
                print(f"Alternative trtexec path {trtexec_path_alt} also not found. Please install TensorRT or adjust path.")
                return False

        # trtexec command. Output will now go to console.
        command_list = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--explicitBatch",
            "--fp16",
            "--workspace=4096",
            "--verbose", # Adding verbose flag for trtexec for more detailed terminal output
            "--best" # Add --best flag for potentially better engine optimization
        ]
        print(f"Executing: {' '.join(command_list)}")
        print(f"Timeout set to 1200 seconds (20 minutes). This may take a while and output will be verbose...")

        try:
            # stdout and stderr are not redirected, so they go to the console.
            process = subprocess.run(
                command_list, 
                check=False # Don't raise CalledProcessError, check returncode manually
            )
            
            if process.returncode != 0:
                print(f"TensorRT engine build failed. trtexec exited with code: {process.returncode}")
                return False
            if not os.path.exists(engine_path):
                print(f"TensorRT engine build command seemed to succeed (exit code 0), but engine file was not created: {engine_path}")
                return False
            print(f"TensorRT engine built successfully: {engine_path}")
        except subprocess.TimeoutExpired: # This block should ideally not be reached now
            print(f"TensorRT engine build timed out. This should not happen with timeout removed.")
            return False
        except Exception as e:
            print(f"An error occurred while trying to build the TensorRT engine with subprocess: {e}")
            return False
    else:
        print(f"Engine found: {engine_path}")
    return True

def initialize_gstreamer_pipelines():
    """
    Initializes GStreamer pipelines for all cameras with appsinks.
    All cameras output BGR (System memory) at YOLO model input dimensions.
    Appsink will emit 'new-sample' signal.
    Uses DYNAMIC_THERMAL_CAM_DEVICE if set and YOLO_INPUT_SHAPE global.
    """
    global _active_pipelines
    global DYNAMIC_THERMAL_CAM_DEVICE # Ensure we're using the global
    global YOLO_INPUT_SHAPE # Access the model's input shape

    appsinks = {}
    pipelines = []

    print("Initializing GStreamer pipelines...")

    if YOLO_INPUT_SHAPE is None:
        print("ERROR: YOLO_INPUT_SHAPE is not set. Cannot determine target GStreamer resolution.")
        # Fallback to original TARGET_WIDTH, TARGET_HEIGHT or error out
        # For now, let's use the constants, but this indicates an issue if inference is expected
        pipeline_target_w = TARGET_WIDTH
        pipeline_target_h = TARGET_HEIGHT
        print(f"Warning: Using default TARGET_WIDTH/HEIGHT for GStreamer: {pipeline_target_w}x{pipeline_target_h}")
    else:
        pipeline_target_h = YOLO_INPUT_SHAPE[2] # Height is dimension 2 (C, H, W)
        pipeline_target_w = YOLO_INPUT_SHAPE[3] # Width is dimension 3
        print(f"GStreamer target resolution set to: {pipeline_target_w}x{pipeline_target_h} (from YOLO_INPUT_SHAPE)")

    # --- MIPI Cameras ---
    for i in range(NUM_MIPI_CAMERAS):
        # Calculate scaled dimensions for MIPI camera to fit within pipeline_target_w/h
        scaled_mipi_w = pipeline_target_w
        scaled_mipi_h = pipeline_target_h

        print(f"MIPI Cam {i}: Original {MIPI_WIDTH}x{MIPI_HEIGHT} -> Scaled by GStreamer to {scaled_mipi_w}x{scaled_mipi_h} (BGR for appsink). OpenCV will pad.")
    
        pipeline_str = (
            f"nvarguscamerasrc sensor-id={i} sensor-mode={MIPI_SENSOR_MODE} ! "
            f"video/x-raw(memory:NVMM),width={MIPI_WIDTH},height={MIPI_HEIGHT},framerate={MIPI_FRAMERATE}/1,format=NV12 ! "
            # 1. Scale with aspect ratio using nvvidconv (output NVMM at scaled_mipi_w x scaled_mipi_h)
            f"nvvidconv ! video/x-raw(memory:NVMM),width={scaled_mipi_w},height={scaled_mipi_h},format=NV12 ! "
            # 2. Convert to BGR system memory at scaled_mipi_w x scaled_mipi_h
            f"nvvidconv ! video/x-raw,format=RGBA,width={scaled_mipi_w},height={scaled_mipi_h} ! "
            f"queue ! "
            f"appsink name=sink{i} emit-signals=true max-buffers=1 drop=true"
        )
        print(f"MIPI Cam {i} pipeline: {pipeline_str}")
        try:
            pipeline = Gst.parse_launch(pipeline_str)
            appsink = pipeline.get_by_name(f"sink{i}")
            if not pipeline or not appsink:
                print(f"ERROR: Failed to create pipeline or get appsink for MIPI Cam {i}")
                # Potentially return or handle error to prevent adding None to lists
                return None, None 
            pipeline.set_state(Gst.State.PLAYING)
            print(f"MIPI Cam {i} pipeline created and set to PLAYING.")
            appsinks[f"mipi_{i}"] = appsink
            pipelines.append(pipeline)
        except Exception as e:
            print(f"ERROR: Failed to parse or create pipeline for MIPI Cam {i}: {e}")
            return None, None
    
    # Thermal Camera
    if DYNAMIC_THERMAL_CAM_DEVICE:
        thermal_cam_id = f"sink_thermal"
        
        # Placeholder for actual thermal camera original dimensions - IMPORTANT: Update these if known!
        # Using 640x512 as a common FLIR Boson resolution for aspect calculation.
        # If your thermal camera is different, these values MUST be updated.
        thermal_original_w = 640 
        thermal_original_h = 512 
        # It would be best to query these using v4l2-ctl if possible, but that's more complex to integrate here.

        scaled_thermal_w, scaled_thermal_h = get_scaled_dimensions(
            thermal_original_w, thermal_original_h, pipeline_target_w, pipeline_target_h
        )
        print(f"Thermal Cam: Assumed Original {thermal_original_w}x{thermal_original_h} -> Scaled by GStreamer to {scaled_thermal_w}x{scaled_thermal_h} (BGR for appsink). OpenCV will pad.")

        pipeline_str_thermal = (
            f"v4l2src device={DYNAMIC_THERMAL_CAM_DEVICE} ! "
            # Assuming v4l2src outputs a raw format that videoscale can handle
            # 1. Scale with aspect ratio using videoscale (system memory)
            f"videoscale ! video/x-raw,width={scaled_thermal_w},height={scaled_thermal_h} ! "
            # 2. Convert to BGR system memory at scaled_thermal_w x scaled_thermal_h
            f"videoconvert ! video/x-raw,format=BGR,width={scaled_thermal_w},height={scaled_thermal_h} ! "
            f"queue ! "
            f"appsink name={thermal_cam_id} emit-signals=true max-buffers=1 drop=true"
        )
        print(f"Thermal Cam pipeline: {pipeline_str_thermal}")
        try:
            pipeline_thermal = Gst.parse_launch(pipeline_str_thermal)
            thermal_appsink = pipeline_thermal.get_by_name("sink_thermal")
            if not pipeline_thermal or not thermal_appsink:
                print(f"ERROR: Failed to create pipeline or get appsink for Thermal Cam")
                # Potentially return or handle error
                return None, None
            pipeline_thermal.set_state(Gst.State.PLAYING)
            print(f"Thermal Cam pipeline created and set to PLAYING.")
            appsinks["thermal"] = thermal_appsink
            pipelines.append(pipeline_thermal)
        except Exception as e:
            print(f"ERROR: Failed to parse or create pipeline for Thermal Cam: {e}")
            return None, None

    _active_pipelines = pipelines # Store pipelines to keep them alive
    return appsinks, pipelines

def load_tensorrt_engine(engine_path):
    """Loads a TensorRT engine from file and creates an execution context."""
    print(f"Loading TensorRT engine from: {engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if not engine:
        print("Failed to deserialize the TensorRT engine.")
        return None, None # Return two Nones

    try:
        context = engine.create_execution_context()
    except Exception as e:
        print(f"[ERROR] Failed to create execution context: {e}")
        return engine, None

    if not context:
        print("Failed to create TensorRT execution context.")
        return engine, None # Return engine and None for context
    
    print("TensorRT engine loaded and context created successfully.")
    return engine, context

def preprocess_frame_yolo(frame, input_shape):
    """
    Prepares a frame for YOLOv8 inference (normalize, CHW).
    Assumes frame is already BGR and at the correct dimensions (H, W) as per input_shape.
    input_shape is expected to be (Channels, Height, Width) e.g. (3, 640, 640)
    """
    # Frame from GStreamer is HWC, BGR, and should be correct size.
    # input_shape is CHW.
    # Assert frame.shape[:2] == (input_shape[1], input_shape[2]), "Frame dimensions mismatch input_shape"
    # No resize needed as GStreamer pipeline should provide correctly sized frames.
    # resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1])) # Removed
    normalized_frame = frame.astype(YOLO_MODEL_PRECISION) / 255.0
    chw_frame = np.transpose(normalized_frame, (2, 0, 1)) # HWC to CHW
    return np.ascontiguousarray(np.expand_dims(chw_frame, axis=0)) # Add batch dimension

def postprocess_yolo_output(outputs, frame_shape, input_shape, confidence_threshold=0.5, nms_threshold=0.45):
    """
    Postprocesses YOLOv8 output to get bounding boxes, scores, and class IDs.
    Assumes model output is [batch_size, num_channels, num_proposals]
    where num_channels = 4 (box_coords) + num_classes.
    Box coords are assumed to be (cx, cy, w, h).
    """
    if not outputs or outputs[0] is None:
        return []
    output_data = outputs[0] 
    if len(output_data.shape) != 3:
        print(f"Unexpected output shape: {output_data.shape}. Expected 3 dimensions (batch, channels, proposals).")
        return []
    output_data = np.transpose(output_data, (0, 2, 1))
    batch_size, num_proposals, num_channels = output_data.shape
    num_classes = num_channels - 4 
    boxes = []
    confidences = []
    class_ids = []
    for i in range(num_proposals):
        proposal = output_data[0, i, :]
        box_coords = proposal[:4]
        class_scores = proposal[4:]
        max_score_class_id = np.argmax(class_scores)
        max_score = class_scores[max_score_class_id]
        if max_score >= confidence_threshold:
            cx, cy, w, h = box_coords
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(max_score))
            class_ids.append(max_score_class_id)
    if not boxes:
        return []
    nms_boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
    indices = cv2.dnn.NMSBoxes(nms_boxes, confidences, confidence_threshold, nms_threshold)
    final_detections = []
    if len(indices) > 0:
        try:
            selected_indices = indices.flatten()
        except AttributeError:
            selected_indices = indices 
        for i in selected_indices:
            box = boxes[i]
            x1, y1, x2, y2 = box
            score = confidences[i]
            class_id = class_ids[i]

            # --- Start Robust Scaling --- 
            if input_shape[1] == 0 or input_shape[0] == 0:
                print(f"ERROR in postprocess: YOLO_INPUT_SHAPE H or W is zero - {input_shape}")
                continue # Skip this detection
            
            scale_x = frame_shape[1] / input_shape[2] # frame_width / yolo_input_width
            scale_y = frame_shape[0] / input_shape[1] # frame_height / yolo_input_height

            # Check for NaN/inf in raw coordinates before scaling
            raw_coords_for_debug = [x1, y1, x2, y2]
            if any(np.isnan(c) or np.isinf(c) for c in raw_coords_for_debug):
                print(f"WARNING in postprocess: NaN/inf in raw box coordinates: {raw_coords_for_debug}. Skipping.")
                continue

            final_x1 = x1 * scale_x
            final_y1 = y1 * scale_y
            final_x2 = x2 * scale_x
            final_y2 = y2 * scale_y

            # Check for NaN/inf in scaled coordinates
            scaled_coords_for_debug = [final_x1, final_y1, final_x2, final_y2]
            if any(np.isnan(c) or np.isinf(c) for c in scaled_coords_for_debug):
                print(f"WARNING in postprocess: NaN/inf in SCALED box coordinates: {scaled_coords_for_debug}. Raw: {raw_coords_for_debug}, Scale:({scale_x},{scale_y}). Skipping.")
                continue
            
            # Clamp to frame boundaries before int conversion
            final_x1 = max(0, min(final_x1, frame_shape[1] - 1))
            final_y1 = max(0, min(final_y1, frame_shape[0] - 1))
            final_x2 = max(0, min(final_x2, frame_shape[1] - 1))
            final_y2 = max(0, min(final_y2, frame_shape[0] - 1))
            
            # Ensure x2 > x1 and y2 > y1 after clamping
            if final_x2 <= final_x1 or final_y2 <= final_y1:
                print(f"WARNING in postprocess: Invalid box after clamping ({final_x1},{final_y1},{final_x2},{final_y2}). Skipping.")
                continue
                
            final_detections.append((int(final_x1), int(final_y1), int(final_x2), int(final_y2), score, class_id))
            # --- End Robust Scaling ---
    return final_detections

def inference_worker(engine, context, bindings, stream, host_input, host_output, dev_input, dev_output, model_input_shape, model_output_shape):
    """Worker thread to perform inference and postprocessing."""
    print("Inference worker thread started.")
    frame_count = 0
    start_time = time.time()
    processing_times = []

    while not stop_event.is_set():
        try:
            cam_name, frame = frame_input_queue.get(timeout=0.1) 
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error getting frame from input queue: {e}")
            continue

        if frame is None: 
            continue

        try:
            iter_start_time = time.time()
            # --- Perform Inference --- 
            original_frame_shape = frame.shape[:2]
            preprocessed_input = preprocess_frame_yolo(frame, model_input_shape)
            np.copyto(host_input, preprocessed_input.ravel())
            
            cuda.memcpy_htod_async(dev_input, host_input, stream)
            print(f"WORKER: Executing TRT for {cam_name}...")
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            print(f"WORKER: DtoH Copy for {cam_name}...")
            cuda.memcpy_dtoh_async(host_output, dev_output, stream)
            print(f"WORKER: Synchronizing stream for {cam_name}...")
            stream.synchronize()
            print(f"WORKER: Inference Sync complete for {cam_name}.")
            
            processed_output_reshaped = host_output.reshape(model_output_shape)
            
            # --- Postprocessing with default threshold ---
            detections = postprocess_yolo_output(
                [processed_output_reshaped], 
                original_frame_shape, 
                model_input_shape,
                confidence_threshold=0.5,
                nms_threshold=0.45
            )
            
            # print(f"Worker found {len(detections)} detections for {cam_name}") # Can be verbose

            # frame_with_boxes = frame # OLD: Worker used to draw
            # for (x1, y1, x2, y2, score, class_id) in detections:
            #     cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     label = f"C{int(class_id)}: {score:.2f}"
            #     cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- Timing (now for all frames processed) --- 
            iter_end_time = time.time()
            processing_time = iter_end_time - iter_start_time
            print(f"WORKER: Processed {cam_name} in {processing_time*1000:.2f} ms")
            processing_times.append(processing_time)
            frame_count += 1
            if frame_count % 100 == 0:
                avg_time = sum(processing_times) / len(processing_times)
                print(f"WORKER TIMING (All Frames): Avg processing time per frame: {avg_time:.4f} seconds ({1.0/avg_time:.2f} FPS estimate for worker)")
                processing_times = []

            # Put the processed frame onto the output queue
            try:
                # frame_output_queue.put_nowait((cam_name, frame_with_boxes)) # OLD
                detections_output_queue.put_nowait((cam_name, original_frame_shape, detections))
            except queue.Full:
                pass

        except Exception as e:
            print(f"Error during inference/processing for {cam_name}: {e}")

    print("Inference worker thread stopping.")

def inference_loop(appsinks, engine, context, stream, host_inputs, cuda_inputs, host_outputs, cuda_outputs, YOLO_MODEL_OUTPUT_SHAPE):
    """
    Main loop to grab frames, perform inference, and display/output results.
    """
    print("Starting inference loop...")

    # ... (Removed PYCUDA_AVAILABLE check as it's handled in main) ...
    # ... (Removed buffer allocation as it's done in main and passed to worker) ...

    # --- Get buffer refs created in main --- 
    # This assumes main() populates these lists correctly
    if not host_inputs:
        print("[ERROR] host_inputs is empty or None")
    if not cuda_inputs:
        print("[ERROR] cuda_inputs is empty or None")
    if not host_outputs:
        print("[ERROR] host_outputs is empty or None")
    if not cuda_outputs:
        print("[ERROR] cuda_outputs is empty or None")
    if not stream:
        print("[ERROR] stream is None")
    if YOLO_MODEL_OUTPUT_SHAPE is None:
        print("[ERROR] YOLO_MODEL_OUTPUT_SHAPE is None")

    if (not host_inputs or not cuda_inputs or not host_outputs or
        not cuda_outputs or not stream or YOLO_MODEL_OUTPUT_SHAPE is None):
        print("CRITICAL ERROR: One or more required inference components are not initialized correctly.")
        return
        
    # Assuming single input/output for simplicity now
    h_input = host_inputs[0]
    d_input = cuda_inputs[0]
    h_output = host_outputs[0]
    d_output = cuda_outputs[0]
    bindings = [int(d_input), int(d_output)]
    # model_output_shape = (1, 84, 8400) # TODO: Get this robustly from engine in main - NOW USING YOLO_MODEL_OUTPUT_SHAPE
    # --- End Get buffer refs --- 

    loop = GLib.MainLoop()
    print("GLib.MainLoop created for main inference.")
    
    # --- Start Inference Worker Thread --- 
    # Ensure YOLO_MODEL_OUTPUT_SHAPE is passed here
    worker_thread = threading.Thread(
        target=inference_worker, 
        args=(engine, context, bindings, stream, h_input, h_output, d_input, d_output, YOLO_INPUT_SHAPE, YOLO_MODEL_OUTPUT_SHAPE),
        daemon=True # Allows main thread to exit even if worker is blocked
    )
    worker_thread.start()
    print("Inference worker thread started.")
    # --- End Start Inference Worker Thread ---

    try:
        for cam_name, appsink_obj in appsinks.items():
            print(f"Connecting new-sample for {cam_name} (main inference)")
            def on_new_sample(sink, current_cam_name_for_log): # Simpler args
                # print(f"on_new_sample called for {current_cam_name_for_log}") # Can be verbose
                sample = sink.pull_sample()
                if sample:
                    buf = sample.get_buffer()
                    caps = sample.get_caps()
                    gst_frame_h = caps.get_structure(0).get_value("height") 
                    gst_frame_w = caps.get_structure(0).get_value("width")
                    
                    img_format_str = caps.get_structure(0).get_value("format")
                    # With simplified pipelines, we expect BGR directly from appsink at scaled dimensions
                    if img_format_str == "BGR":
                        num_channels = 3
                    else:
                        # This case should ideally not happen with the new pipelines
                        print(f"Warning: Unexpected GStreamer format {img_format_str} for {current_cam_name_for_log}. Expected BGR. Attempting to handle as 3 or 4 channel.")
                        num_channels = 4 if "A" in img_format_str or "x" in img_format_str else 3

                    map_flags = Gst.MapFlags.READ
                    success, map_info = buf.map(map_flags)
                    if not success:
                        buf.unmap(map_info) # Ensure unmap even on failure to map
                        return Gst.FlowReturn.OK

                    expected_size = gst_frame_h * gst_frame_w * num_channels
                    if map_info.size < expected_size:
                        print(f"ERROR: Buffer for {current_cam_name_for_log} is too small ({map_info.size} vs {expected_size}). Skipping frame.")
                        buf.unmap(map_info)
                        return Gst.FlowReturn.OK

                    frame_from_gst = np.ndarray((gst_frame_h, gst_frame_w, num_channels), buffer=map_info.data, dtype=np.uint8)
                    frame_writable_copy = frame_from_gst.copy()
                    buf.unmap(map_info)

                    # --- OpenCV Letterboxing --- 
                    # The frame_writable_copy is now aspect-ratio correct but might be smaller 
                    # than pipeline_target_w x pipeline_target_h. We need to pad it.
                    
                    # Get the dimensions of the incoming frame (e.g., 640x360)
                    incoming_h, incoming_w = frame_writable_copy.shape[:2]

                    # Target dimensions for the canvas (YOLO model input size, e.g., 640x640)
                    # These should be available from where pipeline_target_w/h are defined, 
                    # typically from YOLO_INPUT_SHAPE after engine load.
                    # Re-accessing them here for clarity, ensure they are correctly scoped or passed.
                    # Fallback if YOLO_INPUT_SHAPE isn't set, though it should be by now.
                    canvas_h = YOLO_INPUT_SHAPE[1] if YOLO_INPUT_SHAPE and len(YOLO_INPUT_SHAPE) >= 2 else TARGET_HEIGHT
                    canvas_w = YOLO_INPUT_SHAPE[2] if YOLO_INPUT_SHAPE and len(YOLO_INPUT_SHAPE) >= 3 else TARGET_WIDTH

                    if img_format_str != "BGR":
                        # If somehow not BGR, try to convert. This is a fallback.
                        print(f"Frame format for {current_cam_name_for_log} is {img_format_str}, converting to BGR before letterboxing.")
                        if num_channels == 4 and (img_format_str == "BGRx" or img_format_str == "BGRA"):
                            frame_writable_copy = cv2.cvtColor(frame_writable_copy, cv2.COLOR_BGRA2BGR if 'A' in img_format_str else cv2.COLOR_BGRx2BGR)
                        elif num_channels == 4 and (img_format_str == "RGBA"):
                             frame_writable_copy = cv2.cvtColor(frame_writable_copy, cv2.COLOR_RGBA2BGR)
                        # Add other conversions if necessary, or error out
                        elif num_channels == 3: # If it's 3 channels but not BGR (e.g. RGB), this won't fix it without knowing specific format
                            pass # Assume it's BGR-like or hope for the best if it's an unknown 3-channel format

                    # Create a black canvas
                    # Ensure frame_writable_copy is 3 channels if canvas is 3 channels
                    if frame_writable_copy.shape[2] != 3 and num_channels == 4: # e.g. BGRx became 3 channel somehow
                         # This should not happen if format conversion above is correct
                         print(f"Correcting channel mismatch for {current_cam_name_for_log} before letterbox. Is {frame_writable_copy.shape}, num_channels from caps {num_channels}")
                         # frame_writable_copy = cv2.cvtColor(frame_writable_copy, cv2.COLOR_BGRA2BGR) # Example, might need specific conversion

                    # Ensure canvas_h and canvas_w are integers
                    canvas_h = int(canvas_h)
                    canvas_w = int(canvas_w)

                    letterboxed_frame = np.full((canvas_h, canvas_w, 3), 0, dtype=np.uint8)
                    
                    # Calculate offsets to center the frame
                    x_offset = (canvas_w - incoming_w) // 2
                    y_offset = (canvas_h - incoming_h) // 2
                    
                    # Paste the frame onto the canvas
                    # Ensure incoming frame dimensions are not larger than canvas slice
                    paste_h = min(incoming_h, canvas_h - y_offset)
                    paste_w = min(incoming_w, canvas_w - x_offset)

                    if y_offset < 0 : y_offset = 0 # Should not happen if get_scaled_dimensions works
                    if x_offset < 0 : x_offset = 0 # Should not happen
                    
                    # Only try to paste if the frame has valid dimensions
                    if paste_h > 0 and paste_w > 0 and frame_writable_copy.shape[0] > 0 and frame_writable_copy.shape[1] > 0:
                        try:
                            letterboxed_frame[y_offset:y_offset+paste_h, x_offset:x_offset+paste_w] = frame_writable_copy[0:paste_h, 0:paste_w]
                            bgr_frame = letterboxed_frame
                            if bgr_frame.shape[2] == 4:
                                print("[DEBUG] Converting BGRA to BGR")
                                bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGRA2BGR)
                        except ValueError as e:
                            print(f"ERROR during letterboxing paste for {current_cam_name_for_log}: {e}")
                            print(f"Canvas: {letterboxed_frame.shape}, Frame: {frame_writable_copy.shape}, Paste slice: y={y_offset}:{y_offset+paste_h}, x={x_offset}:{x_offset+paste_w}")
                            bgr_frame = frame_writable_copy # Fallback to unpadded frame
                    else:
                        print(f"Warning: Invalid dimensions for letterboxing for {current_cam_name_for_log}. Frame: {incoming_w}x{incoming_h}, Scaled Paste: {paste_w}x{paste_h}. Using unpadded frame.")
                        bgr_frame = frame_writable_copy # Fallback
                    # --- End OpenCV Letterboxing ---

                    # Always update the latest_frames for immediate display
                    with frame_lock:
                        latest_raw_camera_frames[current_cam_name_for_log] = bgr_frame.copy() # Store a copy for display

                    # Throttle frames sent to inference worker
                    current_time = time.time()
                    time_since_last_submit = current_time - last_inference_submit_time.get(current_cam_name_for_log, 0.0)
                    
                    if time_since_last_submit > (1.0 / TARGET_YOLO_FPS_PER_STREAM):
                        try:
                            frame_input_queue.put_nowait((current_cam_name_for_log, bgr_frame)) # Send the original bgr_frame for processing
                            last_inference_submit_time[current_cam_name_for_log] = current_time
                        except queue.Full:
                            # print(f"Input queue full, dropping throttled frame for {current_cam_name_for_log}") # Verbose
                            pass # Drop frame if input queue is full even after throttling
                    # else:
                        # print(f"Skipping inference for {current_cam_name_for_log} due to FPS throttle.") # Can be verbose
                    
                return Gst.FlowReturn.OK

            # Connect signal, passing only necessary args
            appsink_obj.connect("new-sample", on_new_sample, cam_name) 
            
        # check_cv_events now handles combined display based on output queue
        def check_cv_events(): 
            # Process frames from the output queue
            # while not frame_output_queue.empty(): # OLD
            while not detections_output_queue.empty(): # NEW
                try:
                    # cam_name_out, processed_frame = frame_output_queue.get_nowait() # OLD
                    cam_name_out, processed_frame_shape, processed_detections = detections_output_queue.get_nowait()
                    # Update the frame dictionary used for display
                    with frame_lock: 
                        # latest_frames[cam_name_out] = processed_frame # OLD
                        latest_detected_objects[cam_name_out] = (processed_detections, processed_frame_shape, time.time())
                except queue.Empty:
                    break # No more frames for now
                except Exception as e:
                    print(f"Error getting frame from output queue: {e}")
            
            # --- Create Combined View (reads latest_frames) ---
            current_frames_copy = {}
            with frame_lock:
                # current_frames_copy = latest_frames.copy() # OLD: This was for frames with boxes from worker
                # NEW: We need latest_raw_camera_frames for base, and then overlay from latest_detected_objects
                raw_frames_snapshot = latest_raw_camera_frames.copy()
                detections_snapshot = latest_detected_objects.copy()

            cam_order = ["mipi_0", "mipi_1", "mipi_2", "mipi_3", "thermal", "placeholder"]
            display_tiles = []
            placeholder_tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)

            for cam_name in cam_order:
                if cam_name == "placeholder":
                    display_tiles.append(placeholder_tile)
                    continue

                # frame = current_frames_copy.get(cam_name) # OLD
                # NEW display logic:
                display_frame = raw_frames_snapshot.get(cam_name)

                if display_frame is not None:
                    display_frame = display_frame.copy() # Work on a copy for drawing
                    # Check for detections and draw them
                    detection_data = detections_snapshot.get(cam_name)
                    if detection_data:
                        detections_list, _, detection_timestamp = detection_data
                        # Optional: Check if detection_timestamp is fresh enough
                        if time.time() - detection_timestamp < 4.0: # Only draw detections less than 4 sec old (increased from 1.0)
                            for (x1, y1, x2, y2, score, class_id) in detections_list:
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"C{int(class_id)}: {score:.2f}"
                                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    display_tiles.append(display_frame)
                else:
                    display_tiles.append(placeholder_tile)
            
            if len(display_tiles) == 6:
                try:
                    row1 = cv2.hconcat(display_tiles[0:3])
                    row2 = cv2.hconcat(display_tiles[3:6])
                    combined_view = cv2.vconcat([row1, row2])
                    cv2.imshow("Combined View", combined_view)
                except Exception as e:
                    print(f"Error during frame concatenation/display: {e}")
                    # Fallback display might be needed if concat fails due to size mismatch
                    # cv2.imshow("Combined View", cv2.vconcat([cv2.hconcat(display_tiles[0:3]), cv2.hconcat(display_tiles[3:6])]))

            # --- Handle OpenCV Events & Quit ---
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                if loop.is_running():
                    print("Quit signal received in check_cv_events. Quitting GLib.MainLoop.")
                    stop_event.set() # Signal worker thread to stop
                    loop.quit()
                return False
            return True
        
        GLib.timeout_add(30, check_cv_events)

        print("Starting GLib.MainLoop.run() (Combined Display, Worker Thread)...")
        loop.run()
    except KeyboardInterrupt:
        print("Inference loop interrupted by user.")
        stop_event.set() # Signal worker thread to stop
    finally:
        print("Cleaning up (main inference loop)...")
        if 'worker_thread' in locals() and worker_thread.is_alive():
            print("Waiting for worker thread to finish...")
            stop_event.set()
            worker_thread.join(timeout=2.0) # Wait for worker
            if worker_thread.is_alive():
                print("Worker thread did not stop gracefully.")
        for p in _active_pipelines:
            if p: p.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        if loop.is_running():
            loop.quit()
        print("Cleaned up GStreamer pipelines and OpenCV windows.")

def main():
    global PYCUDA_INITIALIZED_SUCCESSFULLY
    global tensorrt_engine, tensorrt_context
    global host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream
    global DYNAMIC_THERMAL_CAM_DEVICE # Make it global here to set it
    global YOLO_INPUT_SHAPE, YOLO_OUTPUT_NAMES, YOLO_MODEL_OUTPUT_SHAPE # Explicitly declare all as global
    global TILE_WIDTH, TILE_HEIGHT # To update them based on YOLO_INPUT_SHAPE

    print("Script starting...")
    Gst.init(None) # Initialize GStreamer

    # Attempt to init CUDA driver and context if PyCUDA is available
    if PYCUDA_AVAILABLE:
        try:
            print("Attempting to init CUDA driver and retain/push primary context...")
            cuda.init()
            # Attempt to retain the primary context for the device
            # This might fail if an X server or another CUDA app has a conflicting context
            # Or if no CUDA device is found/usable
            device = cuda.Device(0) # Assuming device 0
            # Create a context, or get the primary context
            # For primary context:
            # primary_ctx = device.retain_primary_context()
            # primary_ctx.push()
            # For a new context (simpler if primary context handling is tricky):
            ctx = device.make_context() # Create a new context on the device
            ctx.push() # Push the context to the current CPU thread

            PYCUDA_INITIALIZED_SUCCESSFULLY = True
            print("Primary CUDA context pushed successfully.")
        except cuda.Error as e:
            print(f"PyCUDA: CUDA Error during init/context push: {e}")
            print("Continuing without PyCUDA GPU acceleration for TensorRT if possible, or script may fail at inference.")
            PYCUDA_INITIALIZED_SUCCESSFULLY = False
            # Optional: exit if CUDA is absolutely required from the start
            # sys.exit("Exiting: Critical CUDA initialization failed.")
        except Exception as e:
            print(f"PyCUDA: An unexpected error occurred during CUDA initialization: {e}")
            PYCUDA_INITIALIZED_SUCCESSFULLY = False


    # Find the thermal camera device automatically
    DYNAMIC_THERMAL_CAM_DEVICE = find_thermal_camera_device()
    if not DYNAMIC_THERMAL_CAM_DEVICE:
        print("Warning: Could not automatically find a thermal camera. Will try the default or fail.")
        # The initialize_gstreamer_pipelines function will handle the fallback or error.

    print(f"Attempting to use ONNX model: {ONNX_MODEL_PATH}")
    print(f"Target engine file: {ENGINE_FILE_PATH}")

    if not check_or_build_engine(ONNX_MODEL_PATH, ENGINE_FILE_PATH):
        print("Failed to build or verify TensorRT engine. Exiting.")
        sys.exit(1)

    try: 
        if PYCUDA_INITIALIZED_SUCCESSFULLY: 
            # Ensure these are None before attempting to set them from engine
            YOLO_INPUT_SHAPE = None
            YOLO_OUTPUT_NAMES = None
            YOLO_MODEL_OUTPUT_SHAPE = None # Initialize global here as well
            cuda_context = None # Initialize for the finally block
            worker_thread = None # Initialize for the finally block
            model_output_shape_from_engine = None # Temporary holder
            tensorrt_engine = None
            tensorrt_context = None

            try:
                print("Attempting to init CUDA driver and retain/push primary context...")
                cuda.init() # Explicitly initialize the driver API first
                device = cuda.Device(0) # Assuming device 0
                cuda_context = device.retain_primary_context()
                cuda_context.push() 
                print("Primary CUDA context pushed successfully.")
                
                print("starting load tensorrt engine...")
                tensorrt_engine, tensorrt_context = load_tensorrt_engine(ENGINE_FILE_PATH)
                print("exiting tensorrt engine...")
                if not tensorrt_engine or not tensorrt_context:
                     print("Failed to load TensorRT engine or context. Exiting.")
                     sys.exit(1) 

                print(f"[DEBUG] Engine loaded: {tensorrt_engine is not None}")
                print(f"[DEBUG] Context loaded: {tensorrt_context is not None}")

                YOLO_INPUT_SHAPE = tensorrt_engine.get_tensor_shape(tensorrt_engine.get_tensor_name(0))
                YOLO_MODEL_OUTPUT_SHAPE = tensorrt_engine.get_tensor_shape(tensorrt_engine.get_tensor_name(1))

                print(f"[INFO] YOLO_INPUT_SHAPE set to {YOLO_INPUT_SHAPE}")
                print(f"[INFO] YOLO_MODEL_OUTPUT_SHAPE set to {YOLO_MODEL_OUTPUT_SHAPE}")

                # Clear previous buffer lists just in case
                host_inputs.clear()
                cuda_inputs.clear()
                host_outputs.clear()
                cuda_outputs.clear()
                
                # Determine YOLO input and output details and allocate buffers
                output_shapes_map = {} # To store output shapes

                print(f"[DEBUG] TensorRT engine bindings: {tensorrt_engine.num_bindings}")
                for i in range(tensorrt_engine.num_bindings):
                    binding_name = tensorrt_engine.get_tensor_name(i)
                    binding_shape = tensorrt_context.get_tensor_shape(binding_name)
                    binding_dtype = trt.nptype(tensorrt_engine.get_tensor_dtype(binding_name))
                    
                    print(f"Binding {i}: Name='{binding_name}', Shape={binding_shape}, Dtype={binding_dtype}")
                    if tensorrt_engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                        print("Enter if statement")
                        if YOLO_INPUT_SHAPE is None:
                            YOLO_INPUT_SHAPE = tuple(binding_shape)
                            print(f"[DEBUG] Detected YOLO input '{binding_name}' with shape: {YOLO_INPUT_SHAPE} and dtype: {binding_dtype}")
                            # Ensure it's CHW, if NCHW and N=1, take CHW

                        # Allocate buffers regardless of shape already being known
                        host_inputs.append(cuda.pagelocked_empty(trt.volume(binding_shape), dtype=binding_dtype))
                        cuda_inputs.append(cuda.mem_alloc(host_inputs[-1].nbytes))
                        print(f"[DEBUG] Host inputs: {host_inputs}")
                        print(f"[DEBUG] CUDA inputs: {cuda_inputs}")

                    else:
                        print("Enter else statement")
                        # Assuming all other bindings are outputs
                        if YOLO_OUTPUT_NAMES is None: YOLO_OUTPUT_NAMES = []
                        YOLO_OUTPUT_NAMES.append(binding_name)
                        host_outputs.append(cuda.pagelocked_empty(trt.volume(binding_shape), dtype=binding_dtype))
                        cuda_outputs.append(cuda.mem_alloc(host_outputs[-1].nbytes))
                        output_shapes_map[binding_name] = tuple(binding_shape)
                        # Capture the shape of the first output binding encountered
                        if model_output_shape_from_engine is None:
                            model_output_shape_from_engine = tuple(binding_shape)
                            print(f"Detected first YOLO output '{binding_name}' with shape {model_output_shape_from_engine} and dtype {binding_dtype}")
                        # print(f"Detected YOLO output '{binding_name}' with shape {binding_shape} and dtype {binding_dtype}") # Original print

                if YOLO_INPUT_SHAPE is None:
                    print("ERROR: Could not determine YOLO input shape from TensorRT engine.")
                    sys.exit(1)
                if not YOLO_OUTPUT_NAMES:
                    print("ERROR: Could not determine YOLO output names from TensorRT engine.")
                    sys.exit(1)
                if model_output_shape_from_engine is None:
                    print("ERROR: Could not determine model output shape from TensorRT engine bindings.")
                    sys.exit(1)
                
                YOLO_MODEL_OUTPUT_SHAPE = model_output_shape_from_engine

                stream = cuda.Stream()
                PYCUDA_INITIALIZED_SUCCESSFULLY = True # Mark success
                
            except Exception as e:
                print(f"ERROR during explicit CUDA/TensorRT initialization: {e}")
                # Ensure globals are None if setup failed
                tensorrt_engine = None
                tensorrt_context = None
                stream = None
                host_inputs.clear()
                cuda_inputs.clear()
                host_outputs.clear()
                cuda_outputs.clear()
        
        else: # PyCUDA not available path
            print("PyCUDA driver import failed. Inference will be skipped.")
            if YOLO_INPUT_SHAPE is None: 
                YOLO_INPUT_SHAPE = (3, 640, 640) # Default for fallback display
                print(f"Defaulting YOLO_INPUT_SHAPE to {YOLO_INPUT_SHAPE} for display-only mode.")
            # Update TILE_WIDTH and TILE_HEIGHT for display-only mode as well
            if YOLO_INPUT_SHAPE and len(YOLO_INPUT_SHAPE) == 3:
                TILE_HEIGHT = YOLO_INPUT_SHAPE[1]
                TILE_WIDTH = YOLO_INPUT_SHAPE[2]
                print(f"Display TILE_WIDTH and TILE_HEIGHT (display-only) updated to: {TILE_WIDTH}x{TILE_HEIGHT}")

        appsinks, pipelines = initialize_gstreamer_pipelines()
        if not appsinks or not pipelines:
            print("Failed to initialize GStreamer pipelines. Exiting.")
            sys.exit(1)

        # Only start inference loop if PyCUDA/TRT setup was successful
        if PYCUDA_INITIALIZED_SUCCESSFULLY:
            inference_loop(appsinks, tensorrt_engine, tensorrt_context, stream, host_inputs, cuda_inputs, host_outputs, cuda_outputs, YOLO_MODEL_OUTPUT_SHAPE)
        else:
            print("Skipping inference loop as CUDA/TensorRT setup failed or PyCUDA is unavailable.")
            # Optionally start a simple display loop here if desired
            print("Running simple display loop (no inference).")
            loop = GLib.MainLoop()
            try:
                for cam_name, appsink in appsinks.items():
                     # Connect a simplified callback just for display
                     def on_new_sample_display_only(sink, current_cam_name):
                          sample = sink.pull_sample()
                          if sample:
                              buf = sample.get_buffer()
                              caps = sample.get_caps()
                              h = caps.get_structure(0).get_value("height")
                              w = caps.get_structure(0).get_value("width")
                              fmt = caps.get_structure(0).get_value("format")
                              nc = 4 if fmt == "RGBA" else 3
                              success, map_info = buf.map(Gst.MapFlags.READ)
                              if success:
                                  frame = np.ndarray((h, w, nc), buffer=map_info.data, dtype=np.uint8).copy()
                                  bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) if fmt == "RGBA" else frame
                                  with frame_lock:
                                      latest_frames[current_cam_name] = bgr_frame
                                  buf.unmap(map_info)
                          return Gst.FlowReturn.OK
                     appsink.connect("new-sample", on_new_sample_display_only, cam_name)
                
                # Use the same display logic from check_cv_events 
                def check_cv_events_display_only():
                    # Process frames from the output queue (won't be any, but harmless)
                    while not detections_output_queue.empty(): detections_output_queue.get_nowait()
                    
                    current_frames_copy = {}
                    with frame_lock: 
                        raw_frames_snapshot = latest_raw_camera_frames.copy()
                        detections_snapshot = latest_detected_objects.copy() # Though this will be empty here

                    cam_order = ["mipi_0", "mipi_1", "mipi_2", "mipi_3", "thermal", "placeholder"]
                    display_tiles = []
                    placeholder_tile = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
                    for cam_name in cam_order:
                        if cam_name == "placeholder": display_tiles.append(placeholder_tile); continue
                        # frame = current_frames_copy.get(cam_name) # OLD
                        # NEW for display-only mode (no detections to draw)
                        display_frame = raw_frames_snapshot.get(cam_name)

                        if display_frame is not None: 
                            display_tiles.append(display_frame) # Already a BGR frame
                        else: display_tiles.append(placeholder_tile)
                    if len(display_tiles) == 6:
                         try:
                              row1 = cv2.hconcat(display_tiles[0:3])
                              row2 = cv2.hconcat(display_tiles[3:6])
                              combined_view = cv2.vconcat([row1, row2])
                              cv2.imshow("Combined View (No Inference)", combined_view)
                         except Exception as e: print(f"Error display-only concat: {e}")
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27:
                        if loop.is_running(): loop.quit()
                        return False
                    return True
                GLib.timeout_add(30, check_cv_events_display_only)
                loop.run()
            except KeyboardInterrupt:
                 print("Simple display loop interrupted.")
            finally:
                 print("Cleaning up simple display loop...")
                 for p in _active_pipelines: p.set_state(Gst.State.NULL)
                 cv2.destroyAllWindows()
                 if loop.is_running(): loop.quit()

        print("Application finished.")
        return 0

    finally:
        # Ensure worker thread is stopped if it was started
        if 'worker_thread' in locals() and worker_thread is not None and worker_thread.is_alive():
            print("Signalling worker thread to stop (main finally)...")
            stop_event.set()
            worker_thread.join(timeout=2.0)
            if worker_thread.is_alive():
                 print("Worker thread did not stop gracefully (main finally).")
        
        # Ensure primary CUDA context is popped/released
        if 'cuda_context' in locals() and cuda_context is not None:
            try:
                cuda_context.pop()
                print("Primary CUDA context popped.")
            except Exception as e:
                print(f"Error during CUDA context cleanup: {e}")

if __name__ == "__main__":
    sys.exit(main()) 