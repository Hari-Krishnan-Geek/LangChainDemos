from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma

# 
load_dotenv()

embeddings = OpenAIEmbeddings()

# eg:
# emb = embeddings.embed_query("hi there")
# print(emb)

# try to find 200 char and then find nearest separator char
splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200, # at most 200 character
    chunk_overlap = 100
)

loader = TextLoader("facts.txt")
# docs = loader.load()

# print(docs)

#L2 - similarity by distance
# cosine similarity - by angle

docs = loader.load_and_split(text_splitter = splitter)

# every run will store data into db named emb
db = Chroma.from_documents(
    docs,
    embedding = embeddings,
    persist_directory = "emb"
)


results = db.similarity_search_with_score("What is an interesting fact about english languae ?",
                                          k = 2)

# similarity_search() is another function which returns without score



for result in results:
    #result is a tuple
    print(result[1])
    print(result[0].page_content)

