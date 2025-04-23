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

    # Initialize tracker with short memory to avoid lingering IDs
    centroid_tracker = CentroidTracker(maxDisappeared=10)

    # Create display window
    cv2.namedWindow('Basler Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Basler Camera', 800, 600)
    cv2.setMouseCallback('Basler Camera', mouse_callback)

    print("Camera is open and displaying images.")
    print("Right-click to capture and save an image, press ESC to close.")
    print("Press 'r' to reset tracking and clear all IDs.")

    # Set up more specific object detection
    # We'll use OpenCV's feature detector instead of background subtraction
    detector = cv2.SimpleBlobDetector_create()
    
    # For manual motion detection
    prev_frame = None
    
    # Maximum allowed objects to track at once (to prevent noise)
    max_objects = 5
    
    # For frame skipping (to reduce processing load and false positives)
    process_every_n_frames = 3
    frame_count = 0

    # Display video
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Convert to OpenCV format
            image = converter.Convert(grabResult)
            frame = image.GetArray()
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Only process every nth frame to reduce load and false positives
            frame_count += 1
            if frame_count % process_every_n_frames == 0:
                # Convert to grayscale for processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                # If we have a previous frame, compare for motion
                if prev_frame is not None:
                    # Compute absolute difference between frames
                    frame_delta = cv2.absdiff(prev_frame, gray)
                    
                    # Apply threshold to highlight areas of motion
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    
                    # Dilate threshold image to fill in holes
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    # Find contours from the threshold image
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Initialize bounding boxes list
                    rects = []
                    
                    # Process only the largest contours (by area)
                    # Sort contours by area in descending order
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Only process the largest few contours (up to max_objects)
                    for contour in contours[:max_objects]:
                        # Filter by area - much higher threshold to eliminate noise
                        area = cv2.contourArea(contour)
                        if area > 1000000:  # Significantly increased minimum area
                            x, y, w, h = cv2.boundingRect(contour)
                            
                            # Additional filtering by aspect ratio and size
                            aspect_ratio = float(w) / h
                            if 0.5 <= aspect_ratio <= 2.0 and w > 30 and h > 30:
                                # Add as (startX, startY, endX, endY) format for the tracker
                                rects.append((x, y, x + w, y + h))
                                
                                # Draw rectangle on display frame
                                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Update object tracking
                    objects = centroid_tracker.update(rects)
                    
                    # Draw tracking information on display frame
                    for (objectID, centroid) in objects.items():
                        # Draw much larger ID and centroid
                        text = f"ID {objectID}"
                        
                        # Increased text size to 2.0 and thickness to 4
                        cv2.putText(display_frame, text, (centroid[0] - 40, centroid[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                        
                        # Increased circle size to 10
                        cv2.circle(display_frame, (centroid[0], centroid[1]), 10, (0, 255, 0), -1)
                
                # Store current frame for next comparison
                prev_frame = gray
            
            # Show frame
            cv2.imshow('Basler Camera', display_frame)

            # Check if right mouse button was clicked to capture image
            if capture_image:
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
            elif k == ord('r'):  # Reset tracking
                print("Resetting object tracking...")
                centroid_tracker = CentroidTracker(maxDisappeared=10)
                prev_frame = None

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