import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from PIL import Image



mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

st.title("Face Identification")

st.write("""  
\nDeveloped by:
            \nEbyonds""")

add_selectbox = st.sidebar.selectbox(
    "How would you like to edit your image?",
    ("About Project ",  "Face Detection", "Face Recognition")
)

if add_selectbox == "About Project":
    st.write("1. Background Changer \n2. Face Detection \n3. Face Recognition")
    im=cv2.imread("images/Face.jpg")
    im=cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    st.image(im) 




elif add_selectbox == "Face Detection":
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        fd_image = st.sidebar.file_uploader("Upload a  IMAGE")
        if fd_image is not None:
            fdimage = np.array(Image.open(fd_image))
            st.sidebar.image(fdimage)
            results = face_detection.process(fdimage)
            for landmark in results.detections:
                mp_drawing.draw_detection(fdimage, landmark)
            st.image(fdimage)

elif add_selectbox == "Face Recognition":
    st.write("Upload TWO IMAGES")
    image = st.sidebar.file_uploader("Upload a image to train")
    if image is not None:
        train_image = np.array(Image.open(image))
        st.sidebar.image(train_image)
        image_train = face_recognition.load_image_file(image)
        image_encodings_train = face_recognition.face_encodings(image_train)[0]

        detect_image = st.sidebar.file_uploader("Upload a image to test")
        if detect_image is not None:
            test_image = np.array(Image.open(detect_image))
            st.sidebar.image(test_image)
            image_test = face_recognition.load_image_file(detect_image)
            image_encodings_test = face_recognition.face_encodings(image_test)[0]
            image_location_test = face_recognition.face_locations(image_test)

            results = face_recognition.compare_faces([image_encodings_test], image_encodings_train)[0]
            dst = face_recognition.face_distance([image_encodings_test], image_encodings_train)

            if results:
                for (top, right, bottom, left) in image_location_test:
                    output_image = cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
                st.image(output_image)
                st.write("faces are same")
            else:
                st.write("faces doesn't match")
        else:
            st.write("Upload a pic")
    


else:
    st.write("Choose any of the given options")