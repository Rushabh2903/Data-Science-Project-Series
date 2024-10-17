import streamlit as st
import pickle
import numpy as np

# Load the model and vectoriser
model = pickle.load(open('model.pkl', 'rb'))
vectoriser = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit UI
st.title("Sentiment Analysis")

# Input text from the user
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze"):
    if user_input:
        # Preprocess the input text
        input_data = vectoriser.transform([user_input])
        
        # Predict sentiment
        prediction = model.predict(input_data)
        
        # Output the result
        if prediction[0] == 1:
            st.success("The sentiment is Positive! ")
        elif prediction[0]==0:
            st.success("The sentiment is Neutral")
        else:
            st.error("The sentiment is Negative.")
    else:
        st.warning("Please enter some text before analyzing.")
