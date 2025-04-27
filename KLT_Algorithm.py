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

# KLT tracker initialization
class KLTTracker:
    def __init__(self):
        # Parameters for feature detection - using more lenient quality level
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.05,  # Reduced from 0.3 to detect more features
            minDistance=7,
            blockSize=7
        )
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.tracks = []
        self.prev_gray = None
        self.track_len = 10  # Maximum track length to avoid too long tracks

    def init_tracker(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Detect feature points
        points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        
        if points is not None and len(points) > 0:
            print(f"Detected {len(points)} features to track")
            self.tracks = [[(point[0][0], point[0][1])] for point in points]
        else:
            print("No features detected! Trying with lower quality threshold...")
            # Try again with lower quality threshold
            temp_params = self.feature_params.copy()
            temp_params['qualityLevel'] = 0.01
            points = cv2.goodFeaturesToTrack(gray, mask=None, **temp_params)
            
            if points is not None and len(points) > 0:
                print(f"Detected {len(points)} features with lower threshold")
                self.tracks = [[(point[0][0], point[0][1])] for point in points]
            else:
                print("Still no features found. The image may lack distinct features.")
        
        self.prev_gray = gray

    def update(self, frame):
        if self.prev_gray is None:
            self.init_tracker(frame)
            return frame
            
        frame_with_vis = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to enhance contrast
        gray = cv2.equalizeHist(gray)
        
        if len(self.tracks) > 0:
            # Get the most recent point from each track
            prev_points = np.array([track[-1] for track in self.tracks], dtype=np.float32).reshape(-1, 1, 2)
            
            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None, **self.lk_params)
            
            # Check if new_points is valid
            if new_points is None or new_points.shape[0] == 0:
                print("Error: Invalid new_points!")
                
                # Re-initialize tracker if we lose all points
                self.init_tracker(frame)
                return frame
            
            # Update tracks
            new_tracks = []
            for i, (track, status_flag) in enumerate(zip(self.tracks, status.flatten())):
                if status_flag:
                    x = float(new_points[i][0][0])
                    y = float(new_points[i][0][1])
                    
                    # Add new point to track
                    track.append((x, y))
                    
                    # Limit track length
                    if len(track) > self.track_len:
                        track = track[-self.track_len:]
                    
                    new_tracks.append(track)
                    
                    # Draw current point (larger and filled)
                    cv2.circle(frame_with_vis, (int(x), int(y)), 5, (0, 0, 255), -1)
                    
                    # Draw track line
                    if len(track) > 1:
                        for j in range(len(track) - 1):
                            pt1 = (int(track[j][0]), int(track[j][1]))
                            pt2 = (int(track[j+1][0]), int(track[j+1][1]))
                            cv2.line(frame_with_vis, pt1, pt2, (0, 255, 0), 1)
            
            self.tracks = new_tracks
            
            # Add new points every 10 frames to replace lost ones
            if len(self.tracks) < 20:
                mask = np.zeros_like(gray)
                
                # Fill mask with white where we don't have points
                for track in self.tracks:
                    x, y = track[-1]
                    cv2.circle(mask, (int(x), int(y)), 30, 255, -1)
                    
                # Invert mask to find areas without existing points
                mask = 255 - mask
                
                # Find new points in areas without existing tracks
                new_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if new_points is not None:
                    print(f"Adding {len(new_points)} new points")
                    for point in new_points:
                        self.tracks.append([(point[0][0], point[0][1])])
        else:
            # If no tracks, re-initialize
            self.init_tracker(frame)
            
        # Display count of tracked points
        cv2.putText(frame_with_vis, f"Tracks: {len(self.tracks)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        self.prev_gray = gray
        return frame_with_vis

try:
    # Create directory for saving images if it doesn't exist
    if not os.path.exists('captured_images'):
        os.makedirs('captured_images')
        print("Created 'captured_images' directory for storing images")

    # Initialize Basler camera
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()

    if len(devices) == 0:
        print("No camera found. Please check connection.")
        sys.exit(1)

    # Connect to first camera
    camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))
    print(f"Connecting to camera: {camera.GetDeviceInfo().GetModelName()}")
    camera.Open()

    # Configure camera
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    # Initialize KLT tracker
    klt_tracker = KLTTracker()

    # Create display window
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)

    print("Camera is now open and displaying image.")
    print("RIGHT-CLICK to capture and save image, press ESC to close.")

    # Display video feed
    frame_count = 0
    last_redetect = 0
    
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            frame_count += 1
            
            # Update and display tracking features
            frame_with_points = klt_tracker.update(frame)
            
            # Force feature re-detection periodically
            if frame_count - last_redetect > 100:  # Every 100 frames
                print("Periodic re-detection of features")
                klt_tracker.init_tracker(frame)
                last_redetect = frame_count
            
            # Display the image
            cv2.imshow('Basler Camera', frame_with_points)

            # Check if right mouse button was clicked to capture image
            if capture_image:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_points)
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
    # Ensure camera is closed if an error occurs
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