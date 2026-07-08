
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Page settings
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🩺",
    layout="centered"
)


# Load trained Keras model
@st.cache_resource
def load_covid_model():
    return load_model("model.keras")


model = load_covid_model()


# App title
st.title("🩺 COVID-19 Detection from Chest X-Ray")

st.write(
    "Upload a chest X-ray image and the deep learning model will predict COVID Positive or Normal."
)


# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )


    if st.button("Predict"):

        # Resize according to model input
        img = image.resize((299, 299))


        # Convert image to numpy array
        img_array = np.array(img)


        # Convert grayscale image to RGB
        if len(img_array.shape) == 2:
            img_array = np.stack(
                (img_array,) * 3,
                axis=-1
            )


        # Remove alpha channel if image has RGBA
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]


        # Normalize image
        img_array = img_array / 255.0


        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        # Prediction
        prediction = model.predict(img_array)

        probability = prediction[0][0]


        # Result
        st.subheader("Prediction Result")


        if probability > 0.5:

            st.error("COVID Positive")

            st.write(
                f"Confidence: {probability * 100:.2f}%"
            )

        else:

            st.success("Normal")

            st.write(
                f"Confidence: {(1 - probability) * 100:.2f}%"
            )
