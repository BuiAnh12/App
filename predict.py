import streamlit as st
import numpy as np
from PIL import Image, ImageOps  # Import ImageOps
import joblib
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import StandardScaler


@st.cache(allow_output_mutation=True)
def preprocess_image(image):
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))

    # Convert to grayscale
    image = image.convert("L")

    # Invert colors (black background, white drawing)
    image = ImageOps.invert(image)

    # Convert to a NumPy array
    image_array = np.array(image)

    # Normalize the image manually (0 to 1)
    normalized_image = image_array.astype(np.float32) / 255.0

    # Flatten the 2D array to a 1D array
    flattened_image = normalized_image.flatten()

    # Reshape the flattened image into a 2D array with a single sample
    flattened_image = flattened_image.reshape(1, -1)

    # Return the reshaped and scaled image
    return flattened_image


# Load your SVM model with st.cache
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("svm_model_rbf_C10.joblib")


model = load_model()

st.title("Figure Detection App")

# Create a canvas for drawing
canvas = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="white",
    background_color="black",
    drawing_mode="freedraw",
    key="drawing_canvas",
    height=300,
    width=400,
)

if st.button("Detect Figure"):
    if canvas.image_data is not None:
        # Convert the canvas drawing to an image
        img_data = canvas.image_data.astype("uint8")
        im = Image.fromarray(img_data, mode="RGBA")

        # Preprocess the image
        processed_image = preprocess_image(im)

        # Use the SVM model to make predictions
        prediction = model.predict(processed_image)

        # Display the prediction result
        st.write(f"Predicted Figure: {prediction[0]}")
