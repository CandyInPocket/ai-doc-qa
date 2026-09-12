import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

st.set_page_config(page_title="AI 文档问答助手", page_icon="📚", layout="wide")
load_dotenv()

def get_config(key, default=None):
    if os.getenv(key):
        return os.getenv(key)
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

API_KEY = get_config("OPENAI_API_KEY")
BASE_URL = get_config("OPENAI_BASE_URL")
CHAT_MODEL = get_config("CHAT_MODEL", "deepseek-chat")

if not API_KEY:
    st.error("缺少 OPENAI_API_KEY。请在 .env 或 Streamlit Secrets 中设置。")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)

@st.cache_resource
def get_chroma_client():
    # 使用本地轻量向量化模型，不需要任何外部 API
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    chroma_client = chromadb.Client()
    return chroma_client, default_ef

def read_uploaded_file(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.read().decode("utf-8", errors="ignore")

def chunk_text(text, chunk_size=800, overlap=150):
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

def build_collection(files):
    chroma_client, ef = get_chroma_client()
    try:
        chroma_client.delete_collection("docs")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="docs", embedding_function=ef)
    
    ids, docs, metas = [], [], []
    for file in files:
        text = read_uploaded_file(file)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            uid = hashlib.md5(f"{file.name}-{i}-{chunk[:50]}".encode()).hexdigest()
            ids.append(uid)
            docs.append(chunk)
            metas.append({"source": file.name, "chunk": i})
            
    if not docs:
        return None, 0
        
    for i in range(0, len(docs), 100):
        collection.add(
            ids=ids[i:i + 100],
            documents=docs[i:i + 100],
            metadatas=metas[i:i + 100],
        )
    return collection, len(docs)

def answer_question(collection, question, top_k=4):
    count = collection.count()
    if count == 0:
        return "知识库为空。", []
    n = min(top_k, count)
    
    # 直接使用文本查询，本地模型会自动处理向量化
    res = collection.query(query_texts=[question], n_results=n)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    
    context = "\n\n".join(
        f"[{i + 1}] 来源：{m['source']}，片段 {m['chunk']}\n{d}"
        for i, (d, m) in enumerate(zip(docs, metas))
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的知识库助手。只能根据用户提供的资料回答；"
                "资料不足时明确说“资料中没有足够信息”。"
                "回答要简洁，并在句末用 [1][2] 标注来源。"
            ),
        },
        {
            "role": "user",
            "content": f"资料：\n{context}\n\n问题：{question}",
        },
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    sources = list(zip(docs, metas))
    return answer, sources

st.title("📚 AI 文档问答助手")
st.caption("上传 PDF/TXT/MD，构建知识库后提问。")

with st.sidebar:
    st.header("设置")
    top_k = st.slider("检索片段数", 1, 10, 4)
    if st.button("清空对话"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "collection" not in st.session_state:
    st.session_state.collection = None

uploaded_files = st.file_uploader(
    "上传文档",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)

with col1:
    if st.button("构建/重建知识库", type="primary", disabled=not uploaded_files):
        with st.spinner("正在解析、切分、嵌入..."):
            collection, n_chunks = build_collection(uploaded_files)
            if collection:
                st.session_state.collection = collection
                st.session_state.messages = []
                st.success(f"知识库构建完成，共 {n_chunks} 个片段。")
            else:
                st.warning("没有提取到有效文本。")

with col2:
    if st.button("清空知识库"):
        st.session_state.collection = None
        st.success("已清空知识库。")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("请输入问题")

if question:
    if st.session_state.collection is None:
        st.warning("请先上传文档并构建知识库。")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                answer, sources = answer_question(
                    st.session_state.collection,
                    question,
                    top_k,
                )
                st.markdown(answer)
                with st.expander("查看引用来源"):
                    for i, (doc, meta) in enumerate(sources, 1):
                        st.markdown(f"**[{i}] {meta['source']} / 片段 {meta['chunk']}**")
                        preview = doc[:1000] + ("..." if len(doc) > 1000 else "")
                        st.text(preview)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
