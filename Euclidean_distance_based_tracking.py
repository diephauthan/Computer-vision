import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np
from tracker import EuclideanDistTracker  # Import from your tracker.py file

def merge_nearby_rectangles(detections, distance_threshold=50):
    """Merge rectangles that are close to each other"""
    if not detections:
        return []
    
    result = []
    used = [False] * len(detections)
    
    for i in range(len(detections)):
        if used[i]:
            continue
            
        current = list(detections[i])
        used[i] = True
        
        # Check other rectangles
        for j in range(i+1, len(detections)):
            if used[j]:
                continue
                
            # Calculate center points
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = detections[j]
            
            cx1, cy1 = x1 + w1//2, y1 + h1//2
            cx2, cy2 = x2 + w2//2, y2 + h2//2
            
            # Calculate distance between centers
            distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            if distance < distance_threshold:
                # Merge rectangles
                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
                current = [x, y, w, h]
                used[j] = True
        
        result.append(current)
    
    return result

# Global variable to handle mouse events
capture_image = False

# Mouse event handler
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Check for right mouse button click
        capture_image = True

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

    # Initialize Euclidean distance tracker from your tracker.py file
    tracker = EuclideanDistTracker()

    # Create object detector (Background Subtractor)
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
    
    
    # Background accumulation frames
    bg_accumulation_frames = 30
    frame_count = 0
    bg_accumulated = False

    # Create display windows
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)
    
    # Create a smaller mask window
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 400, 300)  # Smaller size for the mask window

    print("Camera is open and displaying images.")
    print("Initially accumulating background. Please wait...")
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
            
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Apply background subtraction
            mask = object_detector.apply(frame)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            # Only start tracking after enough frames for background accumulation
            if frame_count < bg_accumulation_frames:
                frame_count += 1
                if frame_count == bg_accumulation_frames:
                    bg_accumulated = True
                    print("Background accumulated. Starting object tracking...")
                
                # Display the frame with a message
                cv2.putText(display_frame, "Accumulating background... Please wait", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Basler Camera', display_frame)
                
                # Also show the mask as it develops
                cv2.imshow('Mask', mask)
                
                # Check for ESC key to exit
                k = cv2.waitKey(1)
                if k == 27:  # ESC
                    print("ESC pressed, closing camera...")
                    break
                    
                grabResult.Release()
                continue
            
            # Process the mask to reduce noise
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to remove noise and fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for cnt in contours:
                # Calculate area and remove small elements
                area = cv2.contourArea(cnt)
                if area > 300000:  # Increased area threshold to filter out small noise
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Further filter based on aspect ratio to avoid very thin or wide rectangles
                    aspect_ratio = float(w) / h
                    if 0.3 < aspect_ratio < 3.0:  # Filter based on reasonable aspect ratio
                        detections.append([x, y, w, h])
            
            # Step 2: Object tracking using your existing tracker class
            detections = merge_nearby_rectangles(detections, distance_threshold=50)
            boxes_ids = tracker.update(detections)
            
            # Draw tracking results
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.putText(display_frame, f"ID: {id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 10, (255, 0, 0), 10)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 10)
            
            # Display the mask in a separate window
            cv2.imshow('Mask', mask)
            
            # Display the frame
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