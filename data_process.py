from utils import *

import os
from dotenv import load_dotenv
load_dotenv()

from glob import glob
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader,PyMuPDFLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def doc2vec():
    # 定义文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 50
    )

    # 读取并分割文件
    dir_path = os.getenv('DATA_PATH')

    documents = []
    for file_path in glob(dir_path + '*.*'): # glob根据指定的模式匹配文件路径
        loader = None
        if '.csv' in file_path:
            loader = CSVLoader(file_path,encoding='utf-8')
        if '.pdf' in file_path:
            loader = PyMuPDFLoader(file_path)
        if '.txt' in file_path:
            loader = TextLoader(file_path,encoding='utf-8')
        if loader:
            documents += loader.load_and_split(text_splitter) # 使用定义的文本分割器加载并分割文本
    
    # 向量化存储
    if documents:
        vdb = Chroma.from_documents(
            documents=documents,
            embedding=get_embeddings_model(),
            persist_directory=os.getenv('DB_PATH')
        )
        vdb.persist()
 