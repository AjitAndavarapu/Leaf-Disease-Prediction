import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
#Tensorflow model prediction

def read_file_as_image(data) -> np.ndarray:
    image = np.array((data))
    return image

MODEL = tf.keras.models.load_model("potatoes.h5")

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(bytes(data),"r"))
    return image


#ui
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home
if(app_mode=="Home"):
    st.header("Plat Disease Recognition System")
    image_path="ajit.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    img = st.file_uploader("Choose an Image:")
    image = read_file_as_image(img.read())
    img_batch = np.expand_dims(image, 0)
    
    if(st.button("Show Image")):
        st.image(img,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        image = read_file_as_image(img)
        img_batch = np.expand_dims(image, 0)
    
        predictions = MODEL.predict(img_batch)

        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print(f"Class : {predicted_class}, Confidence : {float(confidence)}")

