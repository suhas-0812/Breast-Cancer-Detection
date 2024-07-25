import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import glob
import streamlit.components.v1 as components

st.title("Breast Cancer Detection")

# Load the trained model
model = tf.keras.models.load_model('model1.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match the input shape of the model
    image = np.asarray(image)  # Convert to numpy array
    image = image / 255.0  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to match the input shape
    return image

# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    image_paths = glob.glob(directory + '/*.jpg')
    if len(image_paths) == 0:
        st.error(f"No images found in directory: {directory}")
    random.shuffle(image_paths)  # Shuffle the images randomly
    images = []
    labels = []
    for path in image_paths:
        image = Image.open(path).convert('RGB')
        images.append(image)
        if 'normal' in path:
            labels.append(0)  # 0 for Normal
        elif 'sick' in path:
            labels.append(1)  # 1 for Cancer
    return images, labels

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload and test", "Test on Available images", "About the Model"])

if page == "Upload and test":
    st.write("### Upload any frontal grayscale thermogram image to predict breast cancer.")

    uploaded_file = st.file_uploader("Choose a thermogram image...", type="jpg")

    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 3])

        with col1:
            st.write("#### Image")
            image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB mode
            st.image(image, caption='Uploaded Image', width=256)

        with col3:
            st.write("#### Classification Result")
            placeholder_class = st.empty()
            placeholder_result = st.empty()
            placeholder_class.write("Classifying...")

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_label = (prediction > 0.5).astype("int32")

            # Actual and predicted labels
            actual_label_text = 'No Cancer' if predicted_label == 0 else 'Cancer'
            predicted_label_text = 'Normal' if predicted_label == 0 else 'Cancer'

            # Display results
            placeholder_class.write("Classification Completed")
            if predicted_label == 0:
                placeholder_result.success(f"No cancer detected")
            else:
                placeholder_result.error(f"Cancer detected")

elif page == "Test on Available images":
    st.write("### Test on Available Images")

    # Load and display images from the 'Test' directory
    normal_images, normal_labels = load_images_from_directory('Test/normal')
    sick_images, sick_labels = load_images_from_directory('Test/sick')
    
    # Check if images were loaded
    if len(normal_images) == 0 and len(sick_images) == 0:
        st.error("No images found in 'Test/normal' or 'Test/sick' directories.")
    else:
        if 'all_images' not in st.session_state:
            all_images = normal_images + sick_images
            all_labels = normal_labels + sick_labels

            # Shuffle the images and labels together
            combined = list(zip(all_images, all_labels))
            random.shuffle(combined)
            all_images, all_labels = zip(*combined)

            # Store the images and labels in session state
            st.session_state.all_images = all_images
            st.session_state.all_labels = all_labels
        else:
            all_images = st.session_state.all_images
            all_labels = st.session_state.all_labels

        # Display all images in a column
        st.write("#### Available Test Images")

        for i, (image, actual_label) in enumerate(zip(all_images, all_labels)):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.image(image, width=256, caption=f'Test Image {i + 1}')
            
            with col2:
                # Unique key for each button
                button_key = f"test_button_{i}"
                if st.button(f"Test Image {i + 1}", key=button_key):
                    processed_image = preprocess_image(image)
                    prediction = model.predict(processed_image)
                    predicted_label = (prediction > 0.5).astype("int32")
                    actual_label_text = 'Normal' if actual_label == 0 else 'Cancer'
                    predicted_label_text = 'Normal' if predicted_label == 0 else 'Cancer'

                    st.write(f"**Actual:** {actual_label_text}")
                    st.write(f"**Predicted:** {predicted_label_text}")

elif page == "About the Model":
    # Describe the model
    st.write("### Model Architecture")

    st.write("""
    The model used for breast cancer detection is a Convolutional Neural Network (CNN) with the following architecture:

    1. **Conv2D Layer:** 32 filters, 3x3 kernel, ReLU activation, 'same' padding
    2. **MaxPooling2D Layer:** 2x2 pooling size
    3. **Conv2D Layer:** 64 filters, 3x3 kernel, ReLU activation, 'same' padding
    4. **MaxPooling2D Layer:** 2x2 pooling size
    5. **Conv2D Layer:** 128 filters, 3x3 kernel, ReLU activation, 'same' padding
    6. **MaxPooling2D Layer:** 2x2 pooling size
    7. **Flatten Layer:** Flattens the 3D output to 1D
    8. **Dense Layer:** 512 neurons, ReLU activation
    9. **Dropout Layer:** Dropout rate of 0.5
    10. **Dense Layer:** 256 neurons, ReLU activation
    11. **Dropout Layer:** Dropout rate of 0.5
    12. **Dense Layer:** 1 neuron, Sigmoid activation

    The model is compiled with the Adam optimizer and binary cross-entropy loss function, and is evaluated based on accuracy.
    """)

    st.write("### Model Summary")

    # Display the model summary image
    st.image("model_summary.jpg", caption="Model Summary")

    # Display training history images
    st.write("#### Training History")

    col1, col2 = st.columns(2)

    with col1:
        st.image("acc.jpg", caption="Training Accuracy")

    with col2:
        st.image("loss.jpg", caption="Training Loss")

    # Classification metrics
    st.write("#### Classification Metrics")

    st.write("""
    - **Accuracy:** 99.04%
    - **Precision:** 98.79%
    - **Recall:** 99.93%
    """)

    # Confusion matrix
    st.write("### Confusion Matrix")

    st.write("The confusion matrix visualizes the performance of the model on the test set.")

    with open("confusion_matrix.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # Display the Plotly graph
    components.html(html_content, height=500)