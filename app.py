import streamlit as st
import pickle
import os


# Load the model and vectorizer
try:
    with open('multinomial_nb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()  # Stop execution if there's an error

# Streamlit app title
st.title("SMS / E-mail Fraud Detection")

# Input text box
input_message = st.text_area("Enter your message:", "")

# Button to trigger prediction
if st.button("Check Message"):
    if input_message:
        # Transform the input message
        input_vect = vectorizer.transform([input_message])
        
        # Make prediction
        prediction = model.predict(input_vect)
        prediction_proba = model.predict_proba(input_vect)[:, 1]

        # Display results
        st.success(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")
        # st.write(f"Probability: {prediction_proba[0]:.2f}")
    else:
        st.warning("Please enter a message.")
