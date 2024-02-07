import streamlit as st
st.set_page_config(page_title='People Counting', page_icon='🚶‍♂️', layout="wide")

import pandas as pd
import cv2
import os
from PIL import Image
import supervision as sv
from ultralytics import YOLO

save_path = "video.mp4"
output_video_path = 'output.avi'
final_video_path = 'output.mp4'
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

def convert_avi_to_mp4(avi_file_path, output_name, writer):
    writer.release()
    os.system(f"ffmpeg -i {avi_file_path} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output_name}")
    os.remove(avi_file_path)
    return True

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(3)) 
    frame_height = int(video_cap.get(4)) 
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

def predict(frame):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = []
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        labels.append(f"#{tracker_id} {results.names[class_id]}")
        if results.names[class_id] == 'person':
            people_count.append(tracker_id)

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    
    return label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

def count_people():
    video_cap = cv2.VideoCapture(save_path)
    trackid_dictionary = {}
    empty_container = st.empty()
    writer = create_video_writer(video_cap, output_video_path)

    while video_cap.isOpened():

        ret, frame = video_cap.read()

        if not ret:
            print("End of the video file...")
            break

        image = predict(frame)
        text = f"People Count: {len(set(people_count))}"
        cv2.putText(image, text, text_position, font, font_scale, text_color, font_thickness)

        frame2show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        with empty_container.container():
            c1, c2 = st.columns([0.7, 0.3])
            with c1:
                st.image(frame2show, use_column_width=True)
            with c2:
                # Create or update the DataFrame for people count
                for tracker_id in set(people_count):
                    if tracker_id not in trackid_dictionary:
                        trackid_dictionary[tracker_id] = []

                # Update the count for each person
                for tracker_id in trackid_dictionary:
                    trackid_dictionary[tracker_id].append(len(set(people_count)))

                # Create a DataFrame from the dictionary
                df = pd.DataFrame(list(trackid_dictionary.items()), columns=['People Tracker ID', 'Count'])
                df['Count'] = df['Count'].apply(lambda x: x[-1])  # Get the last count for each person
                st.write("### People Count: ")
                st.dataframe(df['Count'].tail(1), hide_index=True, width=400)
                # st.write(df['Count'].tail(1))
                # st.text("Center Count: {}".format(len(set(people_count))))


        writer.write(image)
        
    video_cap.release()
    cv2.destroyAllWindows()

    convert_avi_to_mp4(output_video_path, final_video_path, writer)
    st.success("Completed.")

    # Display the final video
    st.video(final_video_path)

def main():
    c1, c2 = st.columns([0.15, 0.85], gap='small')
    with c1:
        logo_img = Image.open(r"asset/Ernst-Young-Logo.png")
        st.image(logo_img, use_column_width=True)
    with c2:
        st.title('🚶‍♂️ People Counting')
        st.write('People Detection and Counting through Analytics.')
        st.divider()

    if os.path.exists(save_path):
        os.remove(save_path)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    if os.path.exists(final_video_path):
        os.remove(final_video_path)

    uploaded_file = st.file_uploader("Upload a video", type=['.mp4', '.avi'])
    if st.button('submit', key='btn-1') and uploaded_file:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
                        
        with st.spinner("Processing..."):
            count_people()

        st.divider()

if __name__ == '__main__':
    main()
