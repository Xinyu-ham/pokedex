import os
from models.vision import LabeledImage
from langchain_community.document_loaders import JSONLoader

def load_pokemon_record(file: str):
    return JSONLoader(
        file,
        jq_schema='.kwargs',
        content_key='page_content',
        text_content=False,
        metadata_func=lambda x, _: x['metadata']
    ).load()[0]

def load_pokemon_image(file: str):
    name = file.split('/')[-1].split('.')[0]
    return LabeledImage.load_image(file, name)

def get_all_files(directory, extension=''):
    if extension:
        return [f'{directory}/{f}' for f in os.listdir(directory) if f.endswith(extension)]
    return [f'{directory}/{f}' for f in os.listdir(directory)]