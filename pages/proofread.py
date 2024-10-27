import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import json

def init_page():
    st.title("英文の校正")
    st.sidebar.title("Options")

def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)
    
def load_api_key():
    load_dotenv()
    return os.environ['OPENAI_API_KEY']

def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-4o", "GPT-4o-mini"))
    model_name = "gpt-4o" if model == "GPT-4o" else "gpt-4o-mini"
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.01
    )
    return ChatOpenAI(temperature=temperature, model_name=model_name)

def analyze_text(llm, text):
    # シンプルなプロンプトを使用
    prompt = ChatPromptTemplate.from_template("""
        Correct the following text to CEFR B2 level and provide the result as a JSON object with the following structure:
        {
            "corrected_text": "Corrected version of the text",
            "corrections": ["List of corrections"],
            "b2_expressions": ["Useful CEFR B2 expressions"]
        }
        Text to correct: {input_text}
    """)

    # テンプレートにユーザーの入力を適用
    formatted_prompt = prompt.format(input_text=text)

    result = llm(formatted_prompt)

    st.write("Raw AI Response:", result)  # レスポンスのデバッグ

    try:
        # AIのレスポンスをJSONとして解析
        response_content = json.loads(result["content"])

        if "corrected_text" not in response_content:
            st.error(f"AI response did not contain 'corrected_text'. Full response: {response_content}")
            return {
                "corrected_text": "Error: 'corrected_text' not found",
                "corrections": ["Error in AI response"],
                "b2_expressions": ["Error in AI response"]
            }

        return response_content

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse the response: {str(e)}")
        return {
            "corrected_text": "Error in processing",
            "corrections": ["Unable to process corrections"],
            "b2_expressions": ["Unable to process expressions"]
        }

def display_results(results):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Corrected Text")
        st.write(results["corrected_text"])
        
        st.subheader("🔍 Corrections Explained")
        for correction in results["corrections"]:
            st.markdown(f"- {correction}")
    
    with col2:
        st.subheader("📚 CEFR B2 Expressions to Learn")
        for expression in results["b2_expressions"]:
            st.markdown(f"- {expression}")

def main():
    api_key = load_api_key()
    client = initialize_openai_client(api_key)
    init_page()    
    llm = select_model()

    input_text = st.text_area(
        "Input Text",
        height=150,
        placeholder="Enter your English text here..."
    )

    if st.button("Analyze Text", disabled=not api_key):
        if not input_text:
            st.error("Please enter some text to analyze.")
            return
            
        try:
            with st.spinner("Analyzing your text..."):
                results = analyze_text(llm, input_text)
                display_results(results)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Full error details for debugging:")
            st.error(e)
            
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to start.")

if __name__ == "__main__":
    main()
