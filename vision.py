# Q&A Chatbot
# from langchain.llms import OpenAI

import os
from dotenv import load_dotenv
import streamlit as st
from PIL import Image
import google.generativeai as genai
from datetime import datetime

load_dotenv()  # take environment variables from .env.

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load OpenAI model and get responses
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    if response.text:  # Check if response is not empty
        return response.text
    else:
        return "No response generated some error occured"

# Function to set up image data
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Initialize Streamlit app
st.set_page_config(page_title="Gemini Image Demo")

# Initialize interaction history
if "history" not in st.session_state:
    st.session_state.history = []


initial_prompt = """
               You are an expert in predicting kidney diseases. You will receive input medical data and test results for patients, and you will have to answer questions based on this data to predict the likelihood of kidney disease.

               If the image is not related to kidney disease, please mention that it's the wrong image.

               If the image is of the pathology report, mention the name of the patient and predict the result.

               I am the patient who is uploading the image.
               Tell the prediction as if you are speaking to the patient and elaborate your answer.

               At the end, add "If you have any questions about your kidney function test results, be sure to talk to your doctor." in bold and red if the person has a disease.
               """
# Header and input fields
st.header("Gemini Application")
input_prompt = st.text_area("Input Prompt:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Button to trigger response generation
submit = st.button("Tell me about the image")

if submit:
    if input_prompt.strip():  # Ensure input is not empty
        # Get the image data
        image_data = input_image_setup(uploaded_file)
        
        # Generate response
        response = get_gemini_response(initial_prompt, image_data, input_prompt)
        
        
        # Append user input, response, and timestamp to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        interaction_entry = {
            "input": input_prompt,
            "response": response,
            "timestamp": timestamp
        }
        st.session_state.history.append(interaction_entry)
        
        # Display response
        st.subheader("The Response is")
        st.write(response)

# Show optimized history
st.sidebar.header("Personalized Interaction History")
num_interactions_to_display = 5  # Adjust the number of interactions to display
for idx, interaction in enumerate(reversed(st.session_state.history[-num_interactions_to_display:])):
    interaction_key = f"interaction_{len(st.session_state.history) - idx}"  # Generate unique key
    st.sidebar.subheader(f"Interaction {len(st.session_state.history) - idx}")
    st.sidebar.text_area(f"Input {interaction_key}:", value=interaction["input"], height=100)
    st.sidebar.text("Response:")
    st.sidebar.write(interaction["response"])
    st.sidebar.text("Timestamp:")
    st.sidebar.write(interaction["timestamp"])
