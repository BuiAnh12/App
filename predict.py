import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle


# Load your SVM model with st.cache
@st.cache(allow_output_mutation=True)
def load_model():
    model_file = open("mnist_rbf_defaut.pickle", "rb")
    return pickle.load(model_file)
    # return joblib.load('../svm_model_rbf_C10.joblib')


# Load the SVM model
model = load_model()

st.title("Figure Detection App")

canvas = st_canvas(
    stroke_width=70,
    stroke_color="white",  # Đặt màu nét vẽ thành màu trắng (255, 255, 255 là màu trắng)
    background_color="black",  # Đặt màu nền thành màu đen (0, 0, 0 là màu đen)
    drawing_mode="freedraw",
    key="drawing_canvas",
    height=500,
    width=500,
)
if st.button("Detect Figure"):
    if canvas.image_data is not None:
        try:
            # Convert the canvas drawing to a NumPy array
            img_data = np.array(canvas.image_data, dtype=np.uint8)
            # Resize the image to 28x28 pixels
            img_data = img_data[:, :, :3]
            img_data = cv2.resize(img_data, (28, 28))
            # Convert to grayscale
            img_data_gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
            flattened_image = img_data_gray.flatten()
            # Reshape the flattened image into a 2D array with a single sample
            flattened_image = flattened_image.reshape(1, -1)
            # Use the SVM model to make predictions
            prediction = model.predict(flattened_image)

            # Display the processed image
            st.image(img_data_gray, caption="Processed Image")

            # Display the prediction result
            st.write(f"Predicted Figure: {prediction[0]}")
        except Exception as e:
            st.write(f"Error processing image: {str(e)}")
    else:
        st.write("Vui lòng vẽ trước khi nhấn 'Detect Figure'.")
