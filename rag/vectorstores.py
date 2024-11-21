import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

class PokemonDocumentStore:
    def __new__(cls, docs: list[Document], collection_name:str, vector_store=Chroma):
        if os.path.exists('data/db'):
            return vector_store(embedding=OpenAIEmbeddings(), persist_directory = 'data/db')
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=128, chunk_overlap=64
        )
        split_docs = text_splitter.split_documents(docs)
        return vector_store.from_documents(split_docs,
            collection_name=collection_name,
            embedding=OpenAIEmbeddings(),
            persist_directory = 'data/db'
        )