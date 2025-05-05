import json
import streamlit as st
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

# ========== 1) JSON 파일 읽기 ==========
with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_list = json.load(f)

docs = []
for qa in qa_list:
    q = qa["question"]
    a = qa["answer"]["2_planning_for_AI_usage"]["purpose"]
    e = qa["answer"]["3_prompt_and_feedback_process"]["chatGPT_response"]["analysis_summary"][0]
    full_text = f"Q: {q}\nA: {a}\nExplanation: {e}"
    docs.append(Document(page_content=full_text))

# ========== 2) 벡터스토어 세팅 ==========
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss = FAISS.from_documents(docs, embeddings)
retriever = faiss.as_retriever()

# ========== 3) GPT-Neo 모델 설정 ==========
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", max_length=512, do_sample=True, temperature=0.8)
llm = HuggingFacePipeline(pipeline=generator)

template = """
너는 퀴즈 생성기야. 주제를 바탕으로 {num_questions}개의 {question_type} 문제를 한글로 만들고, 각각 정답과 해설도 함께 제공해줘.
주제: {topic}
"""
prompt = PromptTemplate(input_variables=["topic", "num_questions", "question_type"], template=template)
gen_chain = LLMChain(llm=llm, prompt=prompt)

# ========== 4) Streamlit 앱 UI ==========
st.set_page_config(page_title="한글 퀴즈 생성기", layout="centered")
st.title("📘 한글 퀴즈 생성기 (GPT-Neo 기반)")

mode = st.radio("모드 선택", ["📚 기존 문제 검색", "🧠 새 퀴즈 생성"])

# ========== 5) 기존 문제 검색 ==========
if mode == "📚 기존 문제 검색":
    query = st.text_input("🔍 찾고 싶은 키워드나 주제를 입력하세요")
    if st.button("검색"):
        from langchain.chains import RetrievalQA
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        with st.spinner("검색 중..."):
            result = qa_chain.run(query)
        st.markdown("### ✅ 검색 결과")
        st.write(result)

# ========== 6) 새 퀴즈 생성 ==========
else:
    topic = st.text_input("주제 입력 (예: 인공지능, 마케팅 등)")
    num = st.number_input("문제 개수", min_value=1, max_value=10, value=3)
    qtype = st.selectbox("문제 유형", ["객관식", "OX", "주관식"])

    if st.button("퀴즈 생성"):
        with st.spinner("GPT-Neo가 문제를 생성 중입니다..."):
            result = gen_chain.run({"topic": topic, "num_questions": num, "question_type": qtype})
        st.markdown("### 📝 생성된 퀴즈")
        st.text_area("결과", result, height=400)

        if st.button("➕ 벡터스토어에 저장"):
            new_doc = Document(page_content=result)
            faiss.add_documents([new_doc])
            st.success("✅ 새로운 퀴즈가 벡터스토어에 저장되었습니다.")
