import cv2
import supervision as sv
from ultralytics import YOLO

# Only one class: Gun
class_names = ["Gun"]

# Load the model
model = YOLO("best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Annotators
bounding_box_annotator = sv.RoundBoxAnnotator()
label_annotator = sv.LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with confidence threshold
    results = model(frame, conf=0.5)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Prepare labels with class name and confidence
    labels = [f"{class_names[int(c)]}: {conf:.2f}" 
              for c, conf in zip(detections.class_id, detections.confidence)]

    # Annotate the frame
    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Show the annotated frame
    cv2.imshow("YOLO Object Detection", annotated_image)

    if cv2.waitKey(1) % 256 == 27:  # ESC key
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
