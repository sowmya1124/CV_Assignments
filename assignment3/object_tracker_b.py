import cv2
import math

class ObjectTracker:
    def __init__(self):
        # Dictionary to store the center points of the tracked objects
        self.object_centers = {}
        # Counter to keep track of object IDs
        self.object_id_count = 0

    def update(self, object_rectangles):
        # List to store object bounding boxes and IDs
        tracked_objects = []

        # Update center points of existing objects or assign new IDs
        for rect in object_rectangles:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if the object was detected previously
            object_detected = False
            for obj_id, center in self.object_centers.items():
                dist = math.hypot(cx - center[0], cy - center[1])

                # If distance is within a threshold, update center and ID
                if dist < 25:
                    self.object_centers[obj_id] = (cx, cy)
                    tracked_objects.append([x, y, w, h, obj_id])
                    object_detected = True
                    break

            # If new object detected, assign a new ID
            if not object_detected:
                self.object_centers[self.object_id_count] = (cx, cy)
                tracked_objects.append([x, y, w, h, self.object_id_count])
                self.object_id_count += 1

        # Clean up dictionary by removing unused IDs
        new_object_centers = {}
        for obj_bb_id in tracked_objects:
            _, _, _, _, obj_id = obj_bb_id
            center = self.object_centers[obj_id]
            new_object_centers[obj_id] = center

        # Update dictionary with used IDs
        self.object_centers = new_object_centers.copy()
        return tracked_objects

# Create object tracker instance
tracker = ObjectTracker()

# Capture video from camera
cap = cv2.VideoCapture(0)

# Get video properties
fps = 30  # Assuming a standard FPS for the camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('Object_Tracking_without_Markers.mp4', fourcc, fps, (width, height))

# Background subtractor for object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract Region of Interest (ROI)
    roi = frame[340:720, 500:800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    tracked_boxes_ids = tracker.update(detections)
    for box_id in tracked_boxes_ids:
        x, y, w, h, obj_id = box_id
        #cv2.putText(roi, str(obj_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Write frame with mask to output video
    #out.write(frame)

    # Display frames and masks
    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture, video writer, and close windows
cap.release()
#out.release()
cv2.destroyAllWindows()