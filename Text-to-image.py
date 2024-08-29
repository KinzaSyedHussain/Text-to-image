import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import torch
print(torch.__version__)

# Load the pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("AI Image Description Generator")

# Text input for generating a description
text_input = st.text_area("Enter a prompt for image description:", "A beautiful landscape with mountains")

# Generate a description when the user clicks the button
if st.button("Generate Description"):
    st.write("Generating description...")

    # Encode the input text with GPT-2
    input_ids = tokenizer.encode(text_input, return_tensors="pt")

    # Generate the description
    with torch.no_grad():
        # Example with a max_length of 200 tokens
        output = model.generate(input_ids, max_length=200, num_return_sequences=1)


    # Decode and display the generated description
    generated_description = tokenizer.decode(output[0], skip_special_tokens=True)
    st.write("Generated Description:", generated_description)

st.write("Instructions:")
st.write("1. Enter a prompt for the image description you want to generate.")
st.write("2. Click the 'Generate Description' button to create the description.")

# Footer text
st.write("Powered by Hugging Face's GPT-2 model.")


