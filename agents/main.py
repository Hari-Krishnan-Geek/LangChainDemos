from langchain.chat_models import ChatOpenAI
from langchain.prompts import(
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from langchain.schema import SystemMessage



load_dotenv()

chat = ChatOpenAI()

tables = list_tables()
prompt = ChatPromptTemplate(
    messages = [
        SystemMessage(content=(
                  "You are an AI that has access to  SQLite database\n"
                  f"The database has tables of :{tables}\n"

                  "Do not make any assumptions about what tables exists"
                  "or what columns exist. Instead use the describe_tables function"
                )),
    HumanMessagePromptTemplate.from_template("{input}"),
    # agent scratchpad variable saves the conversation
    MessagesPlaceholder(variable_name = "agent_scratchpad") # look for an ip variable with name `agent_scratchpad```
    ]
)

# tools will not be always used by chatgpt, sometimes it can give proper output without using the tool
tools = [run_query_tool, describe_tables_tool, write_report_tool]

agent = OpenAIFunctionsAgent(
    llm = chat,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(
    agent = agent,
    verbose = True,
    tools = tools
)

# will work without SystemMessage
# agent_executor("How many users are there in the databse?")

# # to call with the SystemMessage
# agent_executor("How many users have provided a shipping address?")



# to call with the write_report_tool
agent_executor("Summarise the top 5 ost popular products. Write the result to a report file")