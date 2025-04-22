import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np

# Global variable for mouse event handling
capture_image = False
roi_select = False
roi_points = []

# Mouse event handler
def mouse_callback(event, x, y, flags, param):
    global capture_image, roi_select, roi_points
    
    if event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button for capturing image
        capture_image = True
    elif event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button for ROI selection
        roi_points = [(x, y)]
        roi_select = True
    elif event == cv2.EVENT_LBUTTONUP and roi_select:
        roi_points.append((x, y))
        roi_select = False

# DCF Tracker class using OpenCV's implementation
class DCFTracker:
    def __init__(self, tracker_type="KCF"):
        """
        Initialize the DCF tracker
        
        tracker_type options:
        - "KCF": Kernelized Correlation Filters
        - "CSRT": Discriminative Correlation Filter with Channel and Spatial Reliability
        """
        self.tracker_type = tracker_type
        self.tracker = None
        self.tracking_initialized = False
        self.bbox = None

    def create_tracker(self):
        """Create the appropriate tracker based on tracker_type"""
        if self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        elif self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        else:
            print(f"Unknown tracker type: {self.tracker_type}. Using KCF instead.")
            return cv2.TrackerKCF_create()

    def init_tracker(self, frame, roi=None):
        """Initialize the tracker with a region of interest"""
        if roi is None:
            # Try to detect a face if no ROI is provided
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            try:
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) > 0:
                    # Use the first detected face
                    (x, y, w, h) = faces[0]
                    self.bbox = (x, y, w, h)
                else:
                    # If no face detected, use the center region
                    h, w = frame.shape[:2]
                    x = w // 4
                    y = h // 4
                    self.bbox = (x, y, w // 2, h // 2)
                    print("No face detected. Using center region for tracking.")
            except:
                # If error in face detection, use the center region
                h, w = frame.shape[:2]
                x = w // 4
                y = h // 4
                self.bbox = (x, y, w // 2, h // 2)
                print("Face detection unavailable. Using center region for tracking.")
        else:
            # Use the provided ROI
            self.bbox = roi
        
        # Create a new tracker instance
        self.tracker = self.create_tracker()
        
        # Initialize the tracker with the frame and bounding box
        success = self.tracker.init(frame, self.bbox)
        
        if success:
            self.tracking_initialized = True
            print(f"Tracker initialized with bounding box: {self.bbox}")
        else:
            print("Failed to initialize tracker")
            
        return success

    def update(self, frame):
        """Update the tracker with a new frame"""
        if not self.tracking_initialized:
            return frame
        
        # Update the tracker
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Tracking success: Draw the bounding box
            self.bbox = bbox
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking ({self.tracker_type})", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Tracking failure: Show error message
            cv2.putText(frame, "Tracking failure detected", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            
        return frame
    
    def reinitialize(self, frame, roi=None):
        """Reinitialize the tracker with a new region of interest"""
        self.tracking_initialized = False
        return self.init_tracker(frame, roi)

def calculate_roi_from_points(points):
    """Calculate ROI (x, y, w, h) from two corner points"""
    if len(points) != 2:
        return None
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    return (x, y, w, h)

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

    # Initialize DCF tracker (can be either KCF or CSRT)
    # CSRT is more accurate but slower, KCF is faster but less accurate
    dcf_tracker = DCFTracker(tracker_type="CSRT")

    # Create display window
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)

    print("Camera opened, displaying video feed.")
    print("LEFT-CLICK and drag to select ROI for tracking")
    print("RIGHT-CLICK to capture and save an image")
    print("Press 'r' to reset tracking")
    print("Press 'c' to switch between KCF and CSRT trackers")
    print("Press ESC to close")

    # Display video
    drawing = False
    current_frame = None
    
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            current_frame = frame.copy()
            
            # Handle ROI selection
            if len(roi_points) == 2:
                roi = calculate_roi_from_points(roi_points)
                if roi and roi[2] > 10 and roi[3] > 10:  # Ensure ROI has reasonable size
                    dcf_tracker.reinitialize(frame, roi)
                roi_points = []
            
            # If tracking is not initialized, try to initialize it
            if not dcf_tracker.tracking_initialized:
                dcf_tracker.init_tracker(frame)
            else:
                # Update tracking
                frame = dcf_tracker.update(frame)
            
            # Show the frame
            cv2.imshow('Basler Camera', frame)
            
            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                print("ESC pressed, closing camera...")
                break
            elif key == ord('r'):  # Reset tracking
                print("Resetting tracking...")
                dcf_tracker.tracking_initialized = False
            elif key == ord('c'):  # Switch tracker type
                if dcf_tracker.tracker_type == "KCF":
                    dcf_tracker.tracker_type = "CSRT"
                else:
                    dcf_tracker.tracker_type = "KCF"
                print(f"Switched to {dcf_tracker.tracker_type} tracker")
                if dcf_tracker.bbox:
                    dcf_tracker.reinitialize(frame, dcf_tracker.bbox)
            
            # Check if right mouse button was clicked to capture image
            if capture_image and current_frame is not None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, current_frame)
                print(f"Image captured and saved at: {filename}")
                capture_image = False

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