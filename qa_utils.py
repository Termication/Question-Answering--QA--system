import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def load_documents(folder_path="docs"):
    docs = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_vector_store(texts):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    return FAISS.from_documents(texts, embeddings)

def ask_question(query, vstore):
    llm = ChatGoogleGenerativeAI(
        model="models/text-bison-001",
        google_api_key=google_api_key,
        temperature=0.7
    )
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vstore.similarity_search(query)
    return chain.run(input_documents=docs, question=query)