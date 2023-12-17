from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv


load_dotenv()
# API key will be read from environment file
llm = OpenAI()


code_prompt = PromptTemplate(
    template = "Write a very short {language} function that will {task} ",
    input_variables = ["language", "task"]
)

code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key = "code"
)

# result = code_chain({"language":"python", "task":"add two numbers"})

# print(result)


# Adding two chains together

test_prompt = PromptTemplate(
    template = "Write a test code for {code} in {language}",
    input_variables = ["code", "language"]
)

test_chain = LLMChain(
    llm = llm,
    prompt = test_prompt,
    output_key = "test"
)

seq_chain = SequentialChain(
    chains  = [code_chain, test_chain],
    input_variables = ["language", "task"],
    output_variables = ["test", "code"]
)

# result = seq_chain({"language":"python", "task":"add two numbers"})

result = seq_chain({"language":"python", "task":"add two numbers"})
print(result)

print("GENERATED CODE >>>>")
print(result["code"])

print("GENERATED TEST >>>>")
print(result["test"])