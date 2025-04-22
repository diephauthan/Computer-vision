import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np
from tracker import EuclideanDistTracker

# Global variable to handle mouse events
capture_image = False

# Mouse event handler
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Check for right mouse button click
        capture_image = True

def detect_objects_by_edges(frame):
    """
    Detect objects using edge detection and contour analysis
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detect edges using Canny algorithm
    edges = cv2.Canny(blurred, 30, 100)
    
    # Apply morphological operations to connect edges
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours and return bounding boxes
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:  # Filter based on area
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:  # Filter based on aspect ratio
                detections.append([x, y, w, h])
    
    return edges, detections  # Return both the edge mask and detections

def detect_objects_combined(frame):
    # Edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Color detection (cho vật màu đỏ)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    # Kết hợp mask màu với mask cạnh
    combined_mask = cv2.bitwise_or(edges, mask_red)
    
    # Áp dụng morphology
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours and return bounding boxes
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 4000:  # Filter based on area
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:  # Filter based on aspect ratio
                detections.append([x, y, w, h])
    
    return edges, detections  # Return both the edge mask and detections

try:
    # Create directory to save images if it doesn't exist
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
        print("Created 'captured_images' directory for saving images")

    # Initialize Basler camera
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        print("No camera found. Please check the connection.")
        sys.exit(1)

    # Connect to the first camera
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    print(f"Connecting to camera: {camera.GetDeviceInfo().GetModelName()}")
    camera.Open()

    # Configure camera
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Initialize Euclidean distance tracker
    tracker = EuclideanDistTracker()
    
    # Create display windows
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)
    
    # Create a smaller edge window
    cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Edges', 400, 300)

    print("Camera is open and displaying images.")
    print("Using edge detection for object tracking.")
    print("Right-click to capture and save an image, press ESC to close.")

    # Display video
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Detect objects using edge detection
            edge_mask, detections = detect_objects_combined(frame)
            
            # Track objects
            boxes_ids = tracker.update(detections)
            
            # Draw tracking results
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(display_frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Display the edge mask in a separate window
            cv2.imshow('Edges', edge_mask)
            
            # Display the frame with tracking information
            cv2.imshow('Basler Camera', display_frame)

            # Check if right mouse button was clicked to capture image
            if capture_image:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Captured and saved image at: {filename}")
                capture_image = False

            # Check for ESC key to exit
            k = cv2.waitKey(1)
            if k == 27:  # ESC
                print("ESC pressed, closing camera...")
                break

        grabResult.Release()

    # Release resources
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Camera closed and program terminated.")

except Exception as e:
    print(f"An error occurred: {e}")
    # Ensure camera is closed in case of error
    try:
        if 'camera' in locals():
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsOpen():
                camera.Close()
            print("Camera connection closed")
        cv2.destroyAllWindows()
    except:
        pass