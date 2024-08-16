import base64
import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI  # 修正箇所

load_dotenv()  # 環境変数を読み込む

# 環境変数からAPIキーを読み込む
#api_key = os.getenv("OPENAI_API_KEY") 
# TODO: あとで適当な関数に変更
api_key = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=api_key)

st.title("音声文字起こしアプリ")

audio_file = st.file_uploader(
    "音声ファイルをアップロードしてください", type=["m4a", "mp3", "webm", "mp4", "mpga", "wav"]
)

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    if st.button("音声文字起こしを実行する"):
        with st.spinner("音声文字起こしを実行中です..."):
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )
        st.success("音声文字起こしが完了しました！")
        st.write(transcript)

        # プロンプトテンプレートを定義
        template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        """あなたは英語の教師です。入力された英語テキストは生徒が1〜2分程度で英語で独り言です
                        以下の形式に従って、CFER B2レベルの英語で校正した上で、改善ポイントを5つ以内でリストアップし、最後に覚えるべき表現を5つ以内でリストアップしてください
                        日本語で出力してください。

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

        # LLMの設定
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        result = llm(template.format_messages(text=transcript))

        # 出力結果を変数に格納
        result_content = result.content
        response_metadata = result.response_metadata

        # contentを整形して表示
        st.write("Content:")
        st.write(result_content)

        # response_metadataを整形して表示
        st.write("\nResponse Metadata:")
        st.json(response_metadata)
