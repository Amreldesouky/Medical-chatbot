import streamlit as st
import os
import google.generativeai as genai
from google.generativeai import types

# Ensure the API key is set in the environment variables

# Ensure the API key is set in the environment variables
# Fetch API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# Initialize the client with the API key
client = genai.Client(api_key=api_key)

# Model ID
model = "gemini-2.5-flash-preview-04-17"

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def get_bot_response(user_input):
    # Access the conversation history from session state
    conversation_history = st.session_state.conversation_history

    # Add user input to conversation history
    conversation_history.append(f"User: {user_input}")

    # Prepare contents to be sent to the model, including the conversation history
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=msg) for msg in conversation_history],
        ),
    ]

    # Configure the content generation settings, including safety settings and system instructions
    generate_content_config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""You are a medical professional, a doctor, with extensive knowledge in general medicine, diagnostics, and patient care. You are here to provide medical information and advice in a clear, accurate, and empathetic manner. When asked about symptoms or conditions, you respond with evidence-based knowledge, but always encourage seeking consultation with a licensed healthcare provider for accurate diagnosis and treatment. Please remember to explain medical terms clearly and offer the best guidance based on the given information"""),
        ],
    )

    # Generate the content stream based on the history
    response_text = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response_text += chunk.text

    # Add the chatbot's response to the conversation history
    conversation_history.append(f"Bot: {response_text.strip()}")

    # Update the session state with the new conversation history
    st.session_state.conversation_history = conversation_history

    return response_text.strip()


# Streamlit Interface
def main():
    st.title("Medical AI Assistant Chatbot")
    st.markdown("**Ask me about symptoms or medical conditions.**")

    # Chat input box
    user_input = st.text_input("Enter your question or symptoms:")

    if user_input:
        bot_response = get_bot_response(user_input)
        # Display the bot's response
        st.write(f"**Bot**: {bot_response}")

    # Display the entire conversation history
    # if st.session_state.conversation_history:
    #     st.markdown("### Conversation History:")
    #     for i in range(0, len(st.session_state.conversation_history), 2):
    #         st.write(f"**User**: {st.session_state.conversation_history[i]}")
    #         st.write(f"**Bot**: {st.session_state.conversation_history[i+1]}")

if __name__ == "__main__":
    main()
