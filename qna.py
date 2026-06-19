import streamlit as st
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="보험 약관 AI 컨설턴트", page_icon="📑", layout="wide")

# 1. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 사이드바 설정
with st.sidebar:
    st.title("⚙️ 설정")
    uploaded_file = st.file_uploader("보험 약관 PDF 업로드", type="pdf")
    temp = st.slider("창의성 (Temperature)", 0.0, 1.0, 0.0, 0.1)

    if st.button("🔄 대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# 3. PDF 처리 및 벡터스토어 생성 함수
@st.cache_resource
def get_vectorstore(uploaded_file_bytes):
    # 업로드된 파일을 임시로 읽어 PDF 텍스트 추출
    doc = fitz.open(stream=uploaded_file_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_text(text)
    return FAISS.from_texts(texts, OpenAIEmbeddings(model="text-embedding-3-large"))

# 4. 메인 UI 및 질문 처리
st.title("📑 보험 약관 지능형 질의응답")

if uploaded_file is not None:
    # 파일 업로드 시 벡터스토어 생성
    with st.spinner("PDF 분석 중..."):
        vectorstore = get_vectorstore(uploaded_file.read())

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 질문 입력
    if prompt := st.chat_input("질문을 입력하세요:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                docs = vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])

                llm = ChatOpenAI(temperature=temp, model='gpt-4o')
                chain = ChatPromptTemplate.from_template(
                    "배경지식: {context}\n\n질문: {question}\n\n답변:"
                ) | llm | StrOutputParser()

                response = chain.invoke({"context": context, "question": prompt})
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("왼쪽 사이드바에서 분석할 보험 약관 PDF 파일을 업로드해 주세요.")
