import cv2
import supervision as sv
from ultralytics import YOLO

# Initialize YOLOv8 model with tracking
model = YOLO('yolov8n.pt')
model.fuse()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize supervision tools
box_annotator = sv.BoxAnnotator()

# Initialize object counter
object_counter = {}
# Dictionary to track last seen positions
last_positions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Perform detection with tracking
    results = model.track(frame, persist=True)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Count objects with tracking
    for idx, class_id in enumerate(detections.class_id):
        class_name = model.names[class_id]
        track_id = detections.tracker_id[idx] if detections.tracker_id is not None else None
        
        if track_id is not None:
            # Calculate center of current detection
            x1, y1, x2, y2 = detections.xyxy[idx]
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if track_id not in last_positions:
                object_counter[class_name] = object_counter.get(class_name, 0) + 1
                print(f"New {class_name} detected, count: {object_counter[class_name]}")
            else:
                # Calculate distance from last position
                last_x1, last_y1, last_x2, last_y2 = last_positions[track_id]
                last_center = ((last_x1 + last_x2) / 2, (last_y1 + last_y2) / 2)
                distance = ((current_center[0] - last_center[0])**2 + (current_center[1] - last_center[1])**2)**0.5
                
                # Only count if object moved significantly (threshold of 50 pixels)
                if distance > 50:
                    object_counter[class_name] = object_counter.get(class_name, 0) + 1
                    print(f"New {class_name} detected, count: {object_counter[class_name]}")
            
            last_positions[track_id] = detections.xyxy[idx]
    
    # Annotate frame
    labels = [
        f"{model.names.get(class_id, 'Unknown')} {confidence:0.2f}"
        for *_, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(
        scene=frame,
        detections=detections
    )
    
    # Display counts
    count_text = ", ".join([f"{k}: {v}" for k, v in object_counter.items()])
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow("Object Counter", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Additional cleanup to ensure windows are closed
for i in range(1, 5):
    cv2.waitKey(1)
    cv2.destroyAllWindows()