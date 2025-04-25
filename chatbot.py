import streamlit as st
import google.generativeai as genai

# Fetch API key from Streamlit secrets
api_key = st.secrets["gemini_api_key"]

# Configure the API key
genai.configure(api_key=api_key)

# Model ID (use a supported model; gemini-1.5-flash is generally available)
model_name = "gemini-1.5-flash"

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def get_bot_response(user_input):
    # Initialize the model
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a medical professional, a doctor, with extensive knowledge in general medicine, diagnostics, and patient care. You are here to provide medical information and advice in a clear, accurate, and empathetic manner. When asked about symptoms or conditions, you respond with evidence-based knowledge, but always encourage seeking consultation with a licensed healthcare provider for accurate diagnosis and treatment. Please remember to explain medical terms clearly and offer the best guidance based on the given information."
    )

    # Access the conversation history from session state
    conversation_history = st.session_state.conversation_history

    # Add user input to conversation history
    conversation_history.append(f"User: {user_input}")

    # Prepare contents for the model (combine history into a single string)
    contents = "\n".join(conversation_history)

    # Configure safety settings
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
    }

    # Generate the content stream
    response_text = ""
    try:
        response = model.generate_content(
            contents,
            safety_settings=safety_settings,
            stream=True
        )
        for chunk in response:
            response_text += chunk.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

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

    # Optional: Display conversation history (uncomment if needed)
    # if st.session_state.conversation_history:
    #     st.markdown("### Conversation History:")
    #     for i in range(0, len(st.session_state.conversation_history), 2):
    #         st.write(f"**User**: {st.session_state.conversation_history[i][5:]}")
    #         st.write(f"**Bot**: {st.session_state.conversation_history[i+1][4:]}")

if __name__ == "__main__":
    main()
