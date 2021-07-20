import streamlit as st

import numpy as np



import cv2
# Load the cascade
face_cascade = cv2.CascadeClassifier('face_detector.xml')


uploaded_file = st.file_uploader("Choose a image file", type=["jpg","jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:

    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in faces: 
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    st.image(img, channels="BGR")
