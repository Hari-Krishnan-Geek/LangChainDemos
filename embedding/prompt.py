from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.chains import RetrievalQA 
from langchain.chat_models import ChatOpenAI
from redundant_fileter_retriever import RedundantFilterRetriever
import langchain


langchain.debug = True

# import langchain

# langchain.debug=True

load_dotenv()
chat = ChatOpenAI() 

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory = "emb",
    embedding_function = embeddings
)

# # A retrieval system is defined as something that can take string queries and return the most 'relevant' Documents from some source.
# retriever = db.as_retriever() # retrive the doc loader function from the database

# chain = RetrievalQA.from_chain_type(
#     llm = chat,
#     retriever = retriever, # It is used to work irrespective of the db used, retrivrs the matching vector, 
#     chain_type = "stuff", # stuffs the matching doc(vector) into the prompt
#     # verbose=True
# )

# # other chain_type options are `map_reduce`, `map_rerank`, `refine`

# result = chain.run("What is an interesting fact about english languae ?")

# print(result)



# using custom retriever

retriever = RedundantFilterRetriever(
    embeddings = embeddings,
    chroma = db
)

chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever, # It is used to work irrespective of the db used, retrivrs the matching vector, 
    chain_type = "stuff", # stuffs the matching doc(vector) into the prompt
    # verbose=True
)

result = chain.run("What is an interesting fact about english languae ?")

print(result)