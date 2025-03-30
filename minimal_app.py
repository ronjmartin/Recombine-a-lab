import streamlit as st

st.title("Minimal Streamlit App")
st.write("This is a minimal Streamlit app to test the environment.")

st.markdown("## Features")
st.markdown("- Simple interface")
st.markdown("- No external dependencies")
st.markdown("- Testing environment setup")

# Add a slider
value = st.slider("Select a value", 0, 100, 50)
st.write(f"Selected value: {value}")

# Add a text input
user_input = st.text_input("Enter some text")
if user_input:
    st.write(f"You entered: {user_input}")

# Add a button
if st.button("Click me"):
    st.success("Button clicked!")

# Display environment information
st.markdown("## Environment Information")
import sys
st.write(f"Python version: {sys.version}")
st.write(f"Python executable: {sys.executable}")