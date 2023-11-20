import streamlit as st
import cv2
import numpy as np

# Function to perform face detection
def detect_faces(image, scaleFactor, minNeighbors, rectangle_color):
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 2)

    return image

# Streamlit app
def main():
    st.title("Face Detection App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # Read the image
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

        # Display the uploaded image
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Instructions
        st.write("### Instructions:")
        st.write("1. Upload an image using the file uploader.")
        st.write("2. Adjust the parameters below for face detection.")
        st.write("3. Click the 'Detect Faces' button.")
        st.write("4. Adjust the color, minNeighbors, and scaleFactor as needed.")

        # Parameters for face detection
        scaleFactor = st.slider("Scale Factor", 1.01, 2.0, 1.1, 0.01)
        minNeighbors = st.slider("Min Neighbors", 1, 10, 3, 1)
        rectangle_color = st.color_picker("Rectangle Color", "#FF0000")

        # Button to trigger face detection
        if st.button("Detect Faces"):
            # Perform face detection
            result_image = detect_faces(image.copy(), scaleFactor, minNeighbors, rectangle_color)

            # Display the result image with detected faces
            st.image(result_image, channels="BGR", caption="Result Image", use_column_width=True)

            # Save the result image
            if st.button("Save Result Image"):
                cv2.imwrite("result_image.jpg", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                st.success("Result image saved successfully.")

if __name__ == "__main__":
    main()