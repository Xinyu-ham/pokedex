from rag.retrievers import PokemonImageRetrieverFactory, PokemonRetrieverFactory, WikiRetrieverFactory
from rag.vectorstores import PokemonDocumentStore
from rag.utils import get_all_files, load_pokemon_image, load_pokemon_record

from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'



# Load all the images and documents
img_files = get_all_files('data/img', extension='png')
imgs = [load_pokemon_image(f) for f in img_files]

doc_files = get_all_files('data/doc', extension='json')
docs = [load_pokemon_record(f) for f in doc_files]

vector_store = PokemonDocumentStore(docs, 'pokedex')

# Create the retrievers
img_retriever = PokemonImageRetrieverFactory(imgs, VIT_MODEL_NAME).create()
wiki_retriever = WikiRetrieverFactory(vector_store).create()
pokemon_retriever = PokemonRetrieverFactory(
    ChatOpenAI(model='gpt-4o-mini'),
    'Wiki pages of pokemons',
    vector_store,
    False
).create()

wiki_retriever_tool = create_retriever_tool(
    wiki_retriever,
    'retrieve_pokemon_info',
    'Search and return information about a given Pokemon'
)

pokemon_retriever_tool = create_retriever_tool(
    pokemon_retriever,
    'retrieve_pokemon_by_type',
    'Search and return names of pokemon based on type'
)

image_retriever_tool = create_retriever_tool(
    img_retriever,
    'retrieve_pokemon_by_image',
    'Search for pokemon based on the path of an image file, returns name of pokemon and certainty of prediction'
)

tools = [wiki_retriever_tool, pokemon_retriever_tool, image_retriever_tool]

prompt = hub.pull("hwchase17/react")

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_agent_response(input_dict: dict[str, str]) -> str:
    return agent_executor.invoke(input_dict).get('output')
