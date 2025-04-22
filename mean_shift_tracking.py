import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np

# Global variable for mouse event handling
capture_image = False

# Mouse event handler
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Check for right mouse button click
        capture_image = True

# Mean Shift Tracker class
class MeanShiftTracker:
    def __init__(self):
        # Setup termination criteria: either 10 iterations or move by at least 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.track_window = None
        self.roi_hist = None
        self.tracking_initialized = False

    def init_tracker(self, frame, roi=None):
        # If ROI is provided, use it; otherwise try to detect a face
        if roi is None:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Try to load the face cascade classifier
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Use the first detected face
                    (x, y, w, h) = faces[0]
                    self.track_window = (x, y, w, h)
                else:
                    # If no face detected, use the center region
                    h, w = frame.shape[:2]
                    x = w // 4
                    y = h // 4
                    self.track_window = (x, y, w // 2, h // 2)
                    print("No face detected. Using center region for tracking.")
            except:
                # If error in face detection, use the center region
                h, w = frame.shape[:2]
                x = w // 4
                y = h // 4
                self.track_window = (x, y, w // 2, h // 2)
                print("Face detection unavailable. Using center region for tracking.")
        else:
            self.track_window = roi
            
        # Extract ROI for histogram calculation
        x, y, w, h = self.track_window
        roi = frame[y:y+h, x:x+w]
        
        # Convert ROI to HSV for better tracking
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the ROI if needed
        mask = None
        
        # Calculate histogram for the ROI
        self.roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        
        # Normalize the histogram
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        self.tracking_initialized = True

    def update(self, frame):
        if not self.tracking_initialized:
            return frame
            
        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Calculate back projection
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        
        # Apply meanShift to get the new location
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        
        # Draw the tracking window on the frame
        x, y, w, h = self.track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        return frame

try:
    # Create directory to save images if it doesn't exist
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
        print("Created 'captured_images' directory to save images")

    # Initialize Basler camera
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        print("No camera detected. Please check the connection.")
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

    # Initialize Mean Shift tracker
    ms_tracker = MeanShiftTracker()

    # Create display window
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)

    print("Camera opened, displaying video feed.")
    print("RIGHT-CLICK to capture and save an image, press ESC to close.")

    # Display video
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()

            # Initialize or update tracking
            if not ms_tracker.tracking_initialized:
                ms_tracker.init_tracker(frame)
            else:
                frame = ms_tracker.update(frame)

            # Display the frame
            cv2.imshow('Basler Camera', frame)

            # Check if right mouse button was clicked to capture image
            if capture_image:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image captured and saved at: {filename}")
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
