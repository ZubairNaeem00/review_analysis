import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from scipy.sparse import hstack
import nltk
import os
import pandas as pd

# Download stopwords if not already present
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .sentiment-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Amazon Fashion Review Sentiment Analysis")
st.markdown("""
    This application predicts the sentiment of Amazon fashion reviews using a machine learning model 
    trained on TF-IDF word and character features with multiple algorithms.
    """)

# Sidebar for information
st.sidebar.title("ℹ️ Model Information")
st.sidebar.markdown("""
    **Model Details:**
    - **Features:** Combined Word (20K, 1-2 grams) + Character (30K, 3-5 grams) TF-IDF
    - **Algorithms:** 11 trained (Logistic Regression, SVM, Naive Bayes, Random Forest, etc.)
    - **Task:** 2-class sentiment classification (Negative, Positive)
    - **Input:** Product review text
    - **Output:** Sentiment label with confidence
    """)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained sentiment analysis model"""
    model_path = 'best_sentiment_model.pkl'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please ensure the model is saved in the current directory.")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Clean text function
def clean_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_sentiment(review_text, model_package):
    """Predict sentiment for given review text"""
    try:
        # Clean the input text
        cleaned_text = clean_text(review_text)
        
        # Transform using both vectorizers
        text_word = model_package['word_vectorizer'].transform([cleaned_text])
        text_char = model_package['char_vectorizer'].transform([cleaned_text])
        
        # Combine features
        text_combined = hstack([text_word, text_char])
        
        # Convert sparse matrix to dense for model compatibility (especially MLP)
        text_combined_dense = text_combined.toarray()
        
        # Make prediction
        prediction = model_package['model'].predict(text_combined_dense)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model_package['model'].predict_proba(text_combined_dense)[0]
            prob_dict = {label: prob for label, prob in zip(model_package['sentiment_labels'], probabilities)}
        except AttributeError:
            prob_dict = None
        
        return prediction, prob_dict
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None

# Load model at startup
model_package = load_model()

if model_package:
    # Display model performance metrics in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Model Performance")
    st.sidebar.metric("Accuracy", f"{model_package['accuracy']:.4f}")
    st.sidebar.metric("Precision", f"{model_package['precision']:.4f}")
    st.sidebar.metric("Recall", f"{model_package['recall']:.4f}")
    st.sidebar.metric("F1-Score", f"{model_package['f1_score']:.4f}")
    st.sidebar.metric("Algorithm", model_package['algorithm'])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Analyze Sentiment")
        
        # Text input options
        input_method = st.radio("Select input method:", ["Text Input", "Batch Predictions"], horizontal=True)
        
        if input_method == "Text Input":
            # Single review input
            review_input = st.text_area(
                "Enter a product review:",
                height=120,
                placeholder="Example: This product is amazing! Great quality and fast shipping.",
                help="Type or paste a product review to analyze its sentiment"
            )
            
            if st.button("🎯 Analyze Sentiment", use_container_width=True):
                if review_input.strip():
                    with st.spinner("🔄 Analyzing sentiment..."):
                        prediction, probabilities = predict_sentiment(review_input, model_package)
                    
                    if prediction:
                        st.markdown("---")
                        
                        sentiment_colors = {
                            'Positive': 'positive',
                            'Negative': 'negative'
                        }
                        color_class = sentiment_colors.get(prediction, 'negative')
                        
                        st.markdown(f"""
                            <div class="sentiment-box {color_class}">
                                <h3>Predicted Sentiment: <strong>{prediction}</strong></h3>
                                <p><strong>Review:</strong> {review_input}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if probabilities:
                            st.markdown("**Confidence Scores:**")
                            for label, prob in probabilities.items():
                                if label in ['Positive', 'Negative']:
                                    st.progress(prob, text=f"{label}: {prob:.2%}")
                else:
                    st.warning("⚠️ Please enter a review to analyze.")
        
        else:
            # Batch predictions
            st.subheader("📝 Batch Predictions")
            
            batch_input = st.text_area(
                "Enter multiple reviews (one per line):",
                height=150,
                placeholder="Review 1\nReview 2\nReview 3",
                help="Enter multiple reviews separated by line breaks"
            )
            
            if st.button("🎯 Analyze Batch", use_container_width=True):
                if batch_input.strip():
                    reviews = [r.strip() for r in batch_input.split('\n') if r.strip()]
                    
                    if reviews:
                        with st.spinner(f"🔄 Analyzing {len(reviews)} reviews..."):
                            results = []
                            for i, review in enumerate(reviews, 1):
                                prediction, _ = predict_sentiment(review, model_package)
                                results.append({
                                    'Review': review,
                                    'Sentiment': prediction
                                })
                        
                        st.markdown("---")
                        st.markdown("**Batch Results:**")
                        
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        st.markdown("**Summary Statistics:**")
                        col1, col2 = st.columns(2)
                        
                        sentiment_counts = df['Sentiment'].value_counts()
                        for sentiment, col in zip(['Positive', 'Negative'], [col1, col2]):
                            count = sentiment_counts.get(sentiment, 0)
                            percentage = (count / len(df) * 100) if len(df) > 0 else 0
                            with col:
                                st.metric(sentiment, f"{count} ({percentage:.1f}%)")
                else:
                    st.warning("⚠️ Please enter at least one review.")
    
    with col2:
        st.subheader("📚 Sample Reviews")
        st.markdown("""
            **Positive Example:**
            > "Amazing quality! Love this product, highly recommend!"
            
            **Negative Example:**
            > "Poor quality and broke within days. Very disappointed."
        """)
        
        st.markdown("---")
        st.subheader("💡 Tips")
        st.markdown("""
            - Longer reviews work better
            - Write naturally as you would in a real review
            - The model works on English reviews
            - Punctuation is automatically cleaned
        """)

else:
    st.error("""
        ❌ **Model not found!**
        
        Please ensure you have:
        1. Run all cells in `review_analysis.ipynb` to train the model
        2. The `best_sentiment_model.pkl` file is in the same directory as this app
        3. The model file is properly saved
        
        Once the model is ready, refresh this page.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; padding: 2rem 0;">
        <small>Sentiment Analysis Model | Built with Streamlit | Amazon Fashion Reviews Dataset</small>
    </div>
""", unsafe_allow_html=True)
