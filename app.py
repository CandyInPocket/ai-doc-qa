import os
import hashlib
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
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
CHAT_MODEL = get_config("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = get_config("EMBED_MODEL", "text-embedding-3-small")

if not API_KEY:
    st.error("缺少 OPENAI_API_KEY。请在 .env 或 Streamlit Secrets 中设置。")
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)


@st.cache_resource
def get_chroma_client():
    # 内存版，重启后丢失。生产可改 PersistentClient(path="./chroma_db")
    return chromadb.Client()


def embed_texts(texts, batch_size=64):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        data = sorted(resp.data, key=lambda x: x.index)
        vectors.extend([item.embedding for item in data])
    return vectors


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
    chroma_client = get_chroma_client()

    try:
        chroma_client.delete_collection("docs")
    except Exception:
        pass

    collection = chroma_client.create_collection(name="docs")

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

    embeddings = embed_texts(docs)

    for i in range(0, len(docs), 100):
        collection.add(
            ids=ids[i:i + 100],
            documents=docs[i:i + 100],
            embeddings=embeddings[i:i + 100],
            metadatas=metas[i:i + 100],
        )

    return collection, len(docs)


def answer_question(collection, question, top_k=4):
    count = collection.count()
    if count == 0:
        return "知识库为空。", []

    n = min(top_k, count)
    q_emb = embed_texts([question])[0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n,
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    context = "\n\n".join(
        f"[{i + 1}] 来源：{m['source']}，片段 {m['chunk']}\n{d}"
        为 i, (d, m) 在……内 enumerate(zip(docs, metas))
    )

消息=[
        {
            "role": "system",
            "内容": (
                "你是一个严谨的知识库助手。只能根据用户提供的资料回答；"
                "资料不足时明确说“资料中没有足够信息”。"
                "回答要简洁，并在句末用 [1][2] 标注来源。"
            ),
        },
        {
            "角色": "用户",
            "内容": F"资料：\n{语境}\n\n问题：{问题}",
        },
    ]

RESP=客户端。聊天.完井.创造(
model=CHAT_MODEL，
消息=消息，
温度=0.2,
    )

答案=resp.选择[0].消息.内容
来源=列表(拉链(文档，元))
返回答案，来源


圣。标题(📚 人工智能"文档问答助手")
圣。标题("上传PDF/TXT/MD，构建知识库后提问。")

和……一起圣。侧边栏:
圣。页眉("设置")
top_k=st.滑块("搜索片段数"，1，10，4)

    如果圣。按钮("清空对话"):
圣。会话状态(_state)。消息=[]
圣。重新运行()

如果"消息"不在……内圣。会话状态(状态(状态(_state)：
圣。会话状态(_state)。消息=[]

如果"收藏"不在……内圣。会话状态(状态(状态(_state)：
圣。会话状态(_state)。收集=没有一个

uploaded_files=st.file_uploader(
    "上传文档",
类型=["pdf"，"txt"，"md"]，
accept_multiple_files=正确，
)

col1，col2=st.列圣。会话状态(如果圣。按钮)。消息.追加({"角色"："用户"，"内容"：问题})会话状态。消息。附加({"角色"："用户"，"内容"：问题})

和...一起col1：
2(("构建/重建知识库"，类型="主要"，已禁用=不上传文件(_F)：
accept_multiple_files=正确，
集合，n_chunks=生成集合(_C)(上载的文件)

如果收集：
圣。会话状态(_state)。收集=集合
圣。会话状态(_state)。消息=[]
圣。成功(f"知识库构建完成，共{n个区块(_C)}个片段。")
：答案
            

带col2:col2：
如果圣。按钮(正确)：如果st.按钮("清空知识别库")：
圣。会话状态(_state)。集合=无会话_状态。收集=无
圣。成功

消息
使用st.聊天消息(_M)(消息["角色"])：带st.聊天消息(_M)(消息["角色"])：
消息

问题=st.聊天输入(_I)("请输入问题")聊天输入(_I)("请输入问题")

圣。警告
如果问题：问题：
圣。警告("请先上传文档并构建知识库。")警告("请先上传文档并构建知识库。")
)
      

带st.spinner("思考中...")：带st.spinner("思考中...")：
使用st.chat_message("用户")：with st.chat_message("user")：

使用st.chat_message("assistant")：带st.chat_message("assistant")：
)
答案，来源=答案问题(回答问题(_Q)
圣。会话状态(_state)。集合，会话状态(_state)。收集，
问题，
top_k，
                )

st.markdown(回答)markdown(回答)

带St.Expander("查看引用来源")：带St.Expander("查看引用来源")：
对于枚举(源，1)中的i，(doc，meta)：for i，(doc，meta)in enumerate(sources，1)：
圣。markdown(f"**[{i}]{meta['source']}/片段{meta['chunk']}**")markdown(f"**[{i}]{meta['source']}/片段{meta['chunk']}**")
预览=doc[：1000]+(如果Len(doc)>1000else""，则"...")[：1000]+(如果Len(doc)>1000else""，则"...")
                        st.text(preview)text(preview)

)      
{“角色”：“助手”，“内容”：答案}{"角色": "助理", "内容"：答案}
      )
