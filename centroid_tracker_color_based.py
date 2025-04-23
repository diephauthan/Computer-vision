import pypylon.pylon as pylon
import cv2
import sys
import os
import time
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

# Global variable for mouse events
capture_image = False

# Mouse event handler
def mouse_callback(event, x, y, flags, param):
    global capture_image
    if event == cv2.EVENT_RBUTTONDOWN:  # Check for right mouse button click
        capture_image = True

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
 
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
 
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
 
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
 
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        
        # otherwise, we are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
 
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
 
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
 
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
 
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue
 
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
 
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
            
            # compute both the row and column indexes we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
 
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

class ColorDetector:
    def __init__(self):
        # Default color range for red objects in HSV (adjust as needed)
        # For red color that wraps around hue range
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Morphological operation kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    def detect(self, frame):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold for red color (which wraps around in HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Perform morphological operations
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize bounding boxes list
        rects = []
        
        # Process contours
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area > 100000:  # Adjust threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                # Add as (startX, startY, endX, endY) format for the tracker
                rects.append((x, y, x + w, y + h))
        
        return rects, mask

    def update_color_range(self, hsv_point):
        # Sample the HSV value at the clicked point and update color range
        h, s, v = hsv_point
        
        # Set tolerance
        h_tolerance = 15
        s_tolerance = 50
        v_tolerance = 50
        
        # For red color which may wrap around
        if h - h_tolerance < 0:
            self.lower_red1 = np.array([0, max(0, s - s_tolerance), max(0, v - v_tolerance)])
            self.upper_red1 = np.array([h + h_tolerance, min(255, s + s_tolerance), min(255, v + v_tolerance)])
            
            self.lower_red2 = np.array([180 + (h - h_tolerance), max(0, s - s_tolerance), max(0, v - v_tolerance)])
            self.upper_red2 = np.array([180, min(255, s + s_tolerance), min(255, v + v_tolerance)])
        elif h + h_tolerance > 180:
            self.lower_red1 = np.array([h - h_tolerance, max(0, s - s_tolerance), max(0, v - v_tolerance)])
            self.upper_red1 = np.array([180, min(255, s + s_tolerance), min(255, v + v_tolerance)])
            
            self.lower_red2 = np.array([0, max(0, s - s_tolerance), max(0, v - v_tolerance)])
            self.upper_red2 = np.array([(h + h_tolerance) % 180, min(255, s + s_tolerance), min(255, v + v_tolerance)])
        else:
            # Regular case
            self.lower_red1 = np.array([h - h_tolerance, max(0, s - s_tolerance), max(0, v - v_tolerance)])
            self.upper_red1 = np.array([h + h_tolerance, min(255, s + s_tolerance), min(255, v + v_tolerance)])
            
            # No need for second range in this case, so set identical values
            self.lower_red2 = self.lower_red1
            self.upper_red2 = self.upper_red1
        
        print(f"Updated color ranges:")
        print(f"Range 1: {self.lower_red1} to {self.upper_red1}")
        print(f"Range 2: {self.lower_red2} to {self.upper_red2}")

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

    # Initialize trackers
    centroid_tracker = CentroidTracker(maxDisappeared=30)
    color_detector = ColorDetector()

    # Create display windows
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)
    
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 400, 300)

    print("Camera is open and displaying images.")
    print("Right-click to capture and save an image, press ESC to close.")
    print("Press 'c' to enter color calibration mode.")

    # Calibration mode flag
    calibration_mode = False
    
    # Use for calibration
    def calibration_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Sample color at clicked point
            hsv_frame = cv2.cvtColor(param["frame"], cv2.COLOR_BGR2HSV)
            # Use a small region around the clicked point
            x_start = max(0, x - 2)
            y_start = max(0, y - 2)
            x_end = min(hsv_frame.shape[1], x + 3)
            y_end = min(hsv_frame.shape[0], y + 3)
            
            # Get average HSV in small region
            region = hsv_frame[y_start:y_end, x_start:x_end]
            avg_hsv = np.mean(region, axis=(0, 1))
            
            # Update color detector
            color_detector.update_color_range(avg_hsv)
            print(f"Sampled HSV at ({x},{y}): {avg_hsv}")

    # Display video
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            
            # Make a copy for display
            display_frame = frame.copy()
            
            if calibration_mode:
                # Display calibration instructions
                cv2.putText(display_frame, "CALIBRATION MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(display_frame, "Click on colored object to track", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, "Press 'c' again to exit calibration", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Set calibration mouse callback
                param = {"frame": frame}
                cv2.setMouseCallback('Basler Camera', calibration_callback, param)
                
                # Show frame without tracking overlay
                cv2.imshow('Basler Camera', display_frame)
            else:
                # Reset to normal mouse callback
                cv2.setMouseCallback('Basler Camera', mouse_callback)
                
                # Detect colored objects
                rects, mask = color_detector.detect(frame)
                
                # Update object tracking
                objects = centroid_tracker.update(rects)
                
                # Draw tracking information on display frame
                for (objectID, centroid) in objects.items():
                    # Draw ID and centroid
                    text = f"ID {objectID}"
                    cv2.putText(display_frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(display_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                
                # Draw rectangles around detected objects
                for rect in rects:
                    startX, startY, endX, endY = rect
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Display color mask in corner for debugging
                mask_small = cv2.resize(mask, (160, 120))
                # Convert mask to BGR to put on frame
                mask_small_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                # Create ROI and put mask in corner
                roi = display_frame[10:130, 10:170]
                display_frame[10:130, 10:170] = mask_small_bgr
                
                # Show frames
                cv2.imshow('Basler Camera', display_frame)
                cv2.imshow('Mask', mask)

            # Check if right mouse button was clicked to capture image
            if capture_image and not calibration_mode:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/image_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Captured and saved image at: {filename}")
                capture_image = False

            # Check for key presses
            k = cv2.waitKey(1)
            if k == 27:  # ESC
                print("ESC pressed, closing camera...")
                break
            elif k == ord('c'):  # 'c' key to toggle calibration mode
                calibration_mode = not calibration_mode
                if calibration_mode:
                    print("Entering color calibration mode. Click on object to calibrate colors.")
                else:
                    print("Exiting calibration mode, resuming tracking.")
                    # Reset mouse callback
                    cv2.setMouseCallback('Basler Camera', mouse_callback)

        grabResult.Release()

    # Release resources
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Camera closed and program terminated.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
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