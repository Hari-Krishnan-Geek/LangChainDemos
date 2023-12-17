from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

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

for doc in docs:
    print(doc
            .page_content)
    print("\n")
