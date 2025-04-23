import pypylon.pylon as pylon
import cv2
import numpy as np
import sys
import os
import time

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

    # Initialize background subtractor with better parameters
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    
    # Create display windows
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)
    
    # Create windows for different stages of processing
    cv2.namedWindow('Foreground Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Foreground Mask', 400, 300)
    
    cv2.namedWindow('Processed Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Mask', 400, 300)

    print("Camera is open and displaying images.")
    print("Using background subtraction for object detection.")
    print("Right-click to capture and save an image, press ESC to close.")
    print("Press 'r' to reset background model.")
    
    # Learning phase for background subtraction
    print("Learning background... please wait.")
    learning_frames = 30
    frame_count = 0
    
    # Main loop
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Learning phase for background model
            if frame_count < learning_frames:
                # Just apply background subtraction to learn the background
                _ = backSub.apply(frame)
                frame_count += 1
                cv2.putText(display_frame, f"Learning background: {frame_count}/{learning_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow('Basler Camera', display_frame)
                grabResult.Release()
                k = cv2.waitKey(1)
                if k == 27:  # ESC
                    break
                continue
            
            # Apply background subtraction
            fgMask = backSub.apply(frame)
            
            # Show raw foreground mask
            cv2.imshow('Foreground Mask', fgMask)
            
            # Process the foreground mask - improved parameters
            kernel_open = np.ones((5, 5), np.uint8)
            kernel_close = np.ones((15, 15), np.uint8)
            
            # Remove noise with morphological opening
            fgMask_processed = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel_open)
            
            # Fill holes with morphological closing
            fgMask_processed = cv2.morphologyEx(fgMask_processed, cv2.MORPH_CLOSE, kernel_close)
            
            # Apply Gaussian blur for smoother edges
            fgMask_processed = cv2.GaussianBlur(fgMask_processed, (5, 5), 0)
            
            # Apply threshold with adjusted value
            _, fgMask_processed = cv2.threshold(fgMask_processed, 80, 255, cv2.THRESH_BINARY)
            
            # Show processed mask
            cv2.imshow('Processed Mask', fgMask_processed)
            
            # Find contours on the processed mask
            contours, _ = cv2.findContours(fgMask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count significant objects
            significant_objects = 0
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000000:  # Adjusted minimum area threshold
                    significant_objects += 1
                    (x, y, w, h) = cv2.boundingRect(contour)
                    
                    # Draw contour and bounding box
                    cv2.drawContours(frame, [contour], 0, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add text with object area
                    cv2.putText(frame, f"Area: {int(area)}", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display object count
            cv2.putText(frame, f"Objects: {significant_objects}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display the processed frame
            cv2.imshow('Basler Camera', frame)

            # Check if right mouse button was clicked to capture image
            if capture_image:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Captured and saved image at: {filename}")
                capture_image = False

            # Check for key presses
            k = cv2.waitKey(1)
            if k == 27:  # ESC
                print("ESC pressed, closing camera...")
                break
            elif k == ord('r'):  # Reset background model
                backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
                frame_count = 0
                print("Reset background model. Learning new background...")
            elif k == ord('+') or k == ord('='):  # Increase area threshold
                min_area = min_area + 500
                print(f"Minimum contour area increased to: {min_area}")
            elif k == ord('-') or k == ord('_'):  # Decrease area threshold
                min_area = max(500, min_area - 500)
                print(f"Minimum contour area decreased to: {min_area}")

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