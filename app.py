import base64
import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

# ページを初期化
def init_page():
    st.title("独り言英語コーチングアプリ")
    # サイドバーのタイトルを表示
    st.sidebar.title("Options")

# モデルを選択
def select_model():
    # サイドバーにオプションボタンを設置
    model = st.sidebar.radio("Choose a model:", ("GPT-4o", "GPT-4o-mini"))
    if model == "GPT-4o":
        model_name = "gpt-4o"
    else:
        model_name = "gpt-4o-mini"

    # スライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.01とする
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

# 環境変数からAPIキーを取得
def load_api_key():
    load_dotenv()  # 環境変数を読み込む
    return os.environ['OPENAI_API_KEY']

# OpenAIクライアントを初期化
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# 音声ファイルをアップロード
def upload_audio_file():
    return st.file_uploader(
        "音声ファイルをアップロードしてください", type=["m4a", "mp3", "webm", "mp4", "mpga", "wav"]
    )

# 音声ファイルを表示
def display_audio(audio_file):
    st.audio(audio_file, format="audio/wav")

# 音声ファイルを文字起こし
def transcribe_audio(client, audio_file):
    with st.spinner("音声文字起こしを実行中です..."):
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )
    return transcript

# プロンプトテンプレートを定義
def define_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    """あなたは英語の教師です。入力された英語テキストは生徒が1〜2分程度で英語で独り言です
                    以下の形式に従って、CFER B2レベルの英語で校正した上で、改善ポイントを5つ以内でリストアップし、最後に覚えるべき表現を5つ以内でリストアップしてください
                    日本語で出力してください。それぞれの項目は改行で区切ってください。

                    1. 英語の校正結果      
                    2. 改善ポイント
                    3. 覚えるべきポイント
                    4. CEFR基準での英文の評価
                    """
                )
            ),
            HumanMessagePromptTemplate.from_template("{text}"),
        ]
    )

#   音声文字起こしをLLMで処理
def process_transcript_with_llm(llm, transcript, template):
    result = llm(template.format_messages(text=transcript))
    return result

def main():

    # APIキーを読み込む
    api_key = load_api_key()
    client = initialize_openai_client(api_key)

    # ページを初期化
    init_page()    

    # LLMを選択
    llm = select_model()

    audio_file = upload_audio_file()

    if audio_file is not None:
        display_audio(audio_file)

        if st.button("音声から文字起こしを実行"):
            transcript = transcribe_audio(client, audio_file)
            st.success("音声文字起こしが完了しました！")
            st.write(transcript)

            with st.spinner('AIがあなたの英語についてアドバイスを出力中です...'):
                template = define_prompt_template()
                result = process_transcript_with_llm(llm,transcript, template)

                result_content = result.content
                response_metadata = result.response_metadata

                st.success("AIがあなたの英語についてアドバイスを出力しました！")
                st.write("Content:")
                st.write(result_content)

                st.write("\nResponse Metadata:")
                st.json(response_metadata)

if __name__ == "__main__":
    main()
