import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

input_video_path = 'people-walking.mp4'

model = YOLO("model/yolov8m.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

people_count = []


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text_color = (255, 255, 255) 
text_position = (10, 30) 


def predict(frame):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = []
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        labels.append(f"#{tracker_id} {results.names[class_id]}")
        if results.names[class_id] == 'person':
            people_count.append(tracker_id)

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)


def count_people():
    video_cap = cv2.VideoCapture(input_video_path)

    while video_cap.isOpened():

        ret, frame = video_cap.read()

        if not ret:
            print("End of the video file...")
            break

        image = predict(frame)
        text = f"People Count: {len(set(people_count))}"
        cv2.putText(image, text, text_position, font, font_scale, text_color, font_thickness)

        cv2.imshow("frame", image)

        if cv2.waitKey(1) == ord('q'):
            break
        
    video_cap.release()
    cv2.destroyAllWindows()

def main():
    count_people()

if __name__ == '__main__':
    main()