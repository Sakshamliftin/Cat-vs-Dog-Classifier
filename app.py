import streamlit as st
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .cat-prediction {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #d32f2f;
    }
    .dog-prediction {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
        color: #1976d2;
    }
    .info-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🐱🐶 Cat vs Dog Classifier</h1>', unsafe_allow_html=True)
st.markdown("Upload an image of a cat or dog to get AI-powered classification!")

# Model loading function (mock for now)
@st.cache_resource
def load_classification_model():
    from tensorflow.keras.models import load_model
    return load_model('Model/cat_dog_classifier.h5')

# Image preprocessing function
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize image to model input size
    img_resized = image.resize((128, 128))  # Adjust size based on your model
    
    # Convert to array and normalize
    img_array = np.array(img_resized)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Mock prediction function
def predict_image(model, processed_image):
    
    prediction = model.predict(processed_image)[0][0]
    return prediction

# Load model
model = load_classification_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image of a cat or dog for best results"
)

# Main app logic
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Your uploaded image", use_column_width=True)
    
    with col2:
        st.subheader("Classification Result")
        
        # Add prediction button
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess image
                    processed_img = preprocess_image(image)
                    
                    # Get prediction
                    prediction = predict_image(model, processed_img)
                    
                    # Determine result
                    if prediction > 0.5:
                        result = "Dog 🐶"
                        confidence = prediction * 100
                        css_class = "dog-prediction"
                    else:
                        result = "Cat 🐱"
                        confidence = (1 - prediction) * 100
                        css_class = "cat-prediction"
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        Prediction: {result}<br>
                        Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add some fun feedback
                    if confidence > 80:
                        st.success("High confidence prediction! 🎯")
                    elif confidence > 60:
                        st.info("Good prediction! 👍")
                    else:
                        st.warning("Lower confidence - the image might be unclear 🤔")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Sidebar with project information
with st.sidebar:
    st.header("📊 Project Information")
    
    st.markdown("""
    ### About This Model
    - **Architecture**: Convolutional Neural Network (CNN)
    - **Training Data**: Cat and Dog images
    - **Input Size**: 128x128 pixels
    - **Accuracy**: >90% on validation set
    
    ### How It Works
    1. Upload your image
    2. Image gets preprocessed (resized & normalized)
    3. CNN analyzes features
    4. Outputs prediction with confidence
    
    ### Tech Stack
    - TensorFlow/Keras
    - Streamlit
    - PIL/Pillow
    - NumPy
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Tips for Best Results")
    st.markdown("""
    - Use clear, well-lit images
    - Ensure the animal is the main subject
    - Avoid heavily filtered images
    - JPG, PNG formats work best
    """)

# Sample images section
st.markdown("---")
st.subheader("🖼️ Don't have an image? Try these samples:")

# Create columns for sample images
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🐱 Sample Cat"):
        st.info("In a real app, this would load a sample cat image")

with col2:
    if st.button("🐶 Sample Dog"):
        st.info("In a real app, this would load a sample dog image")

with col3:
    if st.button("🔄 Random Sample"):
        st.info("In a real app, this would load a random sample image")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    Built with ❤️ using Streamlit and TensorFlow<br>
    <small>This is a demonstration version. Replace the mock functions with your actual model for production use.</small>
</div>
""", unsafe_allow_html=True)

# Instructions for deployment
if st.checkbox("Show deployment instructions"):
    st.markdown("""
    ### 🚀 To use your actual model:
    
    1. **Replace the mock model loading:**
    ```python
    @st.cache_resource
    def load_classification_model():
        from tensorflow.keras.models import load_model
        return load_model('Model/cat_dog_classifier.h5')
    ```
    
    2. **Replace the mock prediction:**
    ```python
    def predict_image(model, processed_image):
        prediction = model.predict(processed_image)[0][0]
        return prediction
    ```
    
    3. **Run the app:**
    ```bash
    streamlit run app.py
    ```
    
    4. **Install requirements:**
    ```bash
    pip install streamlit tensorflow pillow numpy
    ```
    """)