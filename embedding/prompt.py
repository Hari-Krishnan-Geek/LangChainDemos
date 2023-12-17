from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains import RetrievalQA 
from langchain.chat_models import ChatOpenAI

# import langchain

# langchain.debug=True

load_dotenv()
chat = ChatOpenAI() 

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory = "emb",
    embedding_function = embeddings
)

retriever = db.as_retriever() # retrive the doc loader function from the database

chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever, # It is used to work irrespective of the db used
    chain_type = "stuff", # stuffs the matching doc(vector) into the prompt
    # verbose=True
)

# other chain_type options are `map_reduce`, `map_rerank`, `refine`

result = chain.run("What is an interesting fact about english languae ?")

print(result)