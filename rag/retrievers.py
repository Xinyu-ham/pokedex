import json
from models.vision import ImageSimilarity, LabeledImage

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import OpenAI
from langchain_core.retrievers import BaseRetriever

class WikiRetrieverFactory:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        
    def create(self):
        return self.vector_store.as_retriever()
    
class PokemonRetrieverFactory:
    def __init__(self, llm: OpenAI, content_description: str, vector_store: Chroma, verbose: bool=False):
        self.llm = llm
        self.content_description = content_description
        self.vector_store = vector_store
        self.verbose = verbose
        self.attribute_info = [
            AttributeInfo(
                name='source',
                description='Website where the information was scraped from',
                type='string',
            ),
            AttributeInfo(
                name='title',
                description='Title of the document',
                type='string',
            ),
            AttributeInfo(
                name='language',
                description='Language of the document',
                type='string',
            ),
            AttributeInfo(
                name='name',
                description='The name of the pokemon',
                type='string',
            ),
            AttributeInfo(
                name='types',
                description='A list of types of a pokemon, a Pokemon can have 2 types at most',
                type='list[string]',
            ),
            AttributeInfo(
                name='img_url',
                description='The URL of the image of the pokemon in png format',
                type='string',
            ),
        ]

    def create(self):
        return SelfQueryRetriever.from_llm(
            self.llm,
            self.vector_store,
            self.content_description,
            self.attribute_info,
            verbose=self.verbose
        )
    
class PokemonImageRetrieverFactory:
    class ImageRetriever(BaseRetriever):
        docs: list[LabeledImage]
        vit_model_name: str

        def _get_relevant_documents(self, query: str):
            image_similarity_model = ImageSimilarity(self.vit_model_name)
            try:
                query_img = LabeledImage.load_image(query)
            except:
                query_img = LabeledImage.load_image('tmp/query.png')
            similarity_scores = image_similarity_model.compare(self.docs, query_img)
            pokemon, certainty = max(similarity_scores.items(), key=lambda x: x[1])
            return [Document(
                page_content=json.dumps({'pokemon': pokemon, 'certainty': certainty}),
                metadata={'source': 'image_retrival', 'format': 'json'}
            )]
        
    def __init__(self, docs: list[LabeledImage], vit_model_name: str):
        self.docs = docs
        self.vit_model_name = vit_model_name

    def create(self):    
        return self.ImageRetriever(docs=self.docs, vit_model_name=self.vit_model_name)