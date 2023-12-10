from langchain.llms import OpenAI

# Login to https://platform.openai.com/docs/overview and create your API KEY
api_key = "<YOUR_API_KEY>"

llm = OpenAI(
		openai_api_key = api_key
	)

result = llm(" write a poem")

print(result)