from langchain.chat_models import ChatOpenAI 
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, FileChatMessageHistory


load_dotenv()
# API key will be read from environment file
chat = ChatOpenAI(verbose = True)


memory = ConversationBufferMemory (
    chat_memory = FileChatMessageHistory("messages.json"), # can be saved in database any other also
    memory_key = "messages",
    return_messages= True
)


# memory = ConversationSummaryMemory (
#     memory_key = "messages",
#     return_messages= True,
#     llm = chat  # to generate the summary, we can use any other model also here
# )

prompt = ChatPromptTemplate(
    input_variables = ["content", "messages"],
    messages = [
        MessagesPlaceholder(variable_name="messages"), #search for key "messages" to find the earlier messages
        HumanMessagePromptTemplate.from_template ("{content}") # search for `content` in the input
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory = memory, # optional if needs to give the previous data and store the reult
    verbose = True
)

while True:
    content = input(">>")
    result = chain({"content":content})

    print(result)

