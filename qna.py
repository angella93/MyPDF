import streamlit as st
import fitz
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="보험 약관 AI 컨설턴트", page_icon="📑", layout="wide")

# 1. 세션 상태 초기화 (대화 기록을 저장하는 리스트)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    temp = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.0, 0.1)
    if st.button("🔄 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 3. 데이터 로딩 (캐싱)
@st.cache_resource
def get_vectorstore():
    pdf_path = "9회주는 암보험Plus_해약환급금 미지급형.pdf"
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return FAISS.from_texts(splitter.split_text(text), OpenAIEmbeddings(model="text-embedding-3-large"))

# 4. 메인 UI 및 채팅 기록 표시
st.title("📑 보험 약관 지능형 질의응답")
vectorstore = get_vectorstore()

# 화면에 이전 대화들을 계속 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 질문 입력 및 답변 생성
if prompt := st.chat_input("질문을 입력하세요:"):
    # 사용자 메시지 화면 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("약관 검색 중..."):
            docs = vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            llm = ChatOpenAI(temperature=temp, model='gpt-4o')
            chain = ChatPromptTemplate.from_template("배경지식: {context}\n\n질문: {question}\n\n답변:") | llm | StrOutputParser()
            response = chain.invoke({"context": context, "question": prompt})

            st.markdown(response)

    # AI 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": response})
