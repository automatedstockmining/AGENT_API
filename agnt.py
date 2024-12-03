import matplotlib
matplotlib.use('Agg')


from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents import load_agent,initialize_agent,AgentType, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from googlesearch import search
import requests
import trafilatura
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import json
import ast
import gradio as gr
from huggingface_hub import InferenceClient
import tiktoken
import time
import os
from PIL import Image
import os
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from langchain.tools import tool

def truncate_with_qwen(text: str, max_tokens: int = 10000) -> str:
    """
    Truncates the input text based on a token limit.
    
    Parameters:
    - text (str): The input text to be truncated.
    - max_tokens (int): The maximum number of tokens allowed (default: 10,000).

    Returns:
    - str: The truncated text.
    """
    # Ensure input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Use a base encoding like cl100k_base for tokenization
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the text into token IDs
    token_ids = encoding.encode(text)
    
    # Truncate the token list to the maximum allowed tokens
    truncated_token_ids = token_ids[:max_tokens]
    
    # Decode tokens back into a string
    truncated_text = encoding.decode(truncated_token_ids)
    return truncated_text

from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain.chat_models import ChatOpenAI
import langchain
from langchain.agents import load_agent,initialize_agent,AgentType, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from googlesearch import search
import requests
import trafilatura
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import json
import ast
import gradio as gr
from huggingface_hub import InferenceClient
import tiktoken
import time
import os
from PIL import Image
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Retrieve the API key
key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=key)

def fetch_and_process_url(link):
    try:
        # Fetch URL content
        req = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        html_content = req.text  # Use raw HTML directly
        # Extract main content using trafilatura
        return trafilatura.extract(html_content)
    except Exception as e:
        return f"Error fetching or processing {link}: {e}"

def perform_search(query, num_results=5) :
    '''takes in a '''
    try:
        # Perform Google search
        urls = [url for url in search(query, num_results=num_results)]
        print("URLs Found:")
        print(urls)
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return

    # Fetch and process URLs in parallel
    with ThreadPoolExecutor(max_workers=30) as executor:
        results = list(executor.map(fetch_and_process_url, urls))

    # Combine results into a single formatted output
    formatted_text = '\n\n'.join(filter(None, results))  # Skip None or empty results
    return formatted_text


from langchain.tools import tool




@tool
def web_browse(text: str) -> str:
    
    '''
    searches google for what the user is asking for and returns a string of data to respond with. input the user query
    '''
    
    examples = """

    {"user_input": "cisco systems stock price for the last 4 days", "searches": ["cisco stock price last 4 days", "cisco systems stock historical data", "current price of Cisco Systems", "cisco stock price chart"]},
    {"user_input": "Apple stock price yesterday", "searches": ["Apple stock price yesterday", "historical price of Apple stock"]},
    {"user_input": "Tesla quarterly revenue", "searches": ["Tesla latest quarterly revenue", "Tesla revenue report Q3 2024"]},
    {"user_input": "CAPM model for Tesla", "searches": ["Tesla stock beta value", "current risk-free rate", "expected market return for CAPM model"]},
    {"user_input": "Hi", "searches": []},
    {"user_input": "Who are you?", "searches": []},
    {"user_input": "Google earnings per share last quarter", "searches": ["Google EPS last quarter", "Google quarterly earnings report"]},
    {"user_input": "Calculate WACC for Microsoft", "searches": ["Microsoft cost of equity", "Microsoft cost of debt", "Microsoft capital structure", "current risk-free rate", "Microsoft beta"]},
    {"user_input": "Show Amazon stock chart for last 5 years", "searches": ["Amazon stock chart last 5 years", "Amazon historical price data"]},
    {"user_input": "GDP of China in 2023", "searches": ["China GDP 2023", "latest GDP figures for China"]},
    {"user_input": "Portfolio optimization model", "searches": ["efficient frontier portfolio theory", "input data for portfolio optimization model", "expected returns and covariances"]},
    {"user_input": "Find current inflation rate in the US", "searches": ["current US inflation rate", "US CPI data"]},
    {"user_input": "What is NPV and how do you calculate it?", "searches": ["definition of NPV", "how to calculate NPV"]},
    {"user_input": "Dividend yield for Coca-Cola", "searches": ["Coca-Cola dividend yield", "latest Coca-Cola dividend data"]},
    {"user_input": "Sharpe ratio formula example", "searches": ["Sharpe ratio formula", "example calculation of Sharpe ratio"]},
    {"user_input": "What is the current Fed interest rate?", "searches": ["current Federal Reserve interest rate", "latest Fed interest rate decision"]},
    {"user_input": "Generate DCF model for Tesla", "searches": ["Tesla free cash flow data", "Tesla growth rate projections", "current discount rate for Tesla", "steps to build a DCF model"]},
    {"user_input": "Tell me a joke", "searches": []},
    {"user_input": "Explain the concept of opportunity cost", "searches": ["definition of opportunity cost", "examples of opportunity cost in economics"]}

    """

    
    
    
    
    format = '{"user_input": "dynamic input", "searches": ["search1", "search2", "search3"]}'
    response_for_searches = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2000,
        messages=[{'role':'system','content': f'Split the input "{text}" into a dictionary format for searches. Examples: {examples}. Respond in this exact format: {format}. the maximumum number of items you can put in the list of searches in 5. ONLY HAVE '' SURROUNDING THE DICTIONARY NOT THREE '' OR ANYTHING SIMILAR. for basic requests have less searches in the list'}]
    )
    
    searches_resp = response_for_searches.choices[0].message.content
    if searches_resp[:4] == "'''" and searches_resp[-4:] == "'''":
        print('the response had commas around it so the algorithm removed them')
        searches_resp = searches_resp[3:-3]
    else:
        pass
    print(f'the llm response: {searches_resp}')
    print(f'the text is: {text}')
    searches =  ast.literal_eval(searches_resp)



    data = []
    searches_needed = searches['searches']
    for value in searches_needed:
        var = perform_search(value)
        data.append(var)
    pre_summarised_data =  ''.join(data)
    pre_summarised_data = truncate_with_qwen(pre_summarised_data)
    print('tokenized data')
    response_for_summary = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=2000,
        messages=[{'role':'system','content': f"Based on the query '{text}', extract only the relevant data from the following text and organize it into a table format. "
                    f"Use Markdown table syntax with clear column headers. Include all relevant numerical data, dates, or categories as appropriate. "
                    f"Here is the raw data: {pre_summarised_data}. If the request specifies a time range or specific attributes, ensure only that data is included."
                }]
    )
    summary_resp = response_for_summary.choices[0].message.content
    return summary_resp



web_browse_tool = Tool(
    name="WebBrowseTool",
    func=web_browse,
    description='''searches google for what the user is asking for and returns a string of data to respond with'''
    )


import requests
from openai import OpenAI


@tool()
def generate_and_upload_plot(request, output_file="plot.png", openai_api_key=key):
    
    """
    YOU MUST INPUT THE DATA FROM THE SEARCH TOOL INTO THIS TOOL FOR IT TO WORK. 
    THIS TOOL DOES NOT HAVE PRE-LOADED DATA AND REQUIRES IT FROM THE USER. 
    ALWAYS ENSURE THE DATA IS PASSED INTO THE REQUEST AND CLEARLY DESCRIBE THE PLOT NEEDED.
    EXECUTE THIS TOOL ONLY WHEN YOU HAVE ALL REQUIRED INFORMATION.
    """

    # Initialize OpenAI client
    client = OpenAI(api_key=key)

    # Request Matplotlib code from OpenAI
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        max_tokens=2000,
        messages=[{
            'role': 'user',
            'content': (
                f"Based off of {request}, generate matplotlib code to create the chart requested. "
                "only use matplotlib, pandas, numpy etc not other libraries"
                "ALWAYS include plt.savefig('plot.png') at the end of the code to save the figure. "
                "RETURN ONLY THE CODE AND NOTHING ELSE, NO EXPLANATION. DO NOT INCLUDE QUOTES AT THE START OR END OR PYTHON ON ANYTHING SIMILAR, JUST THE RAW PYTHON CODE. ALWAYS CODE THE WHOLE SCRIPT, ASSIGNING THE DATA THE USER MAY PROVIDE AS VARIBALES, NEVER RETURN A SNIPPET, ONLY THE WHOLE SCRIPT"
            )
        }],
        temperature=0
    )

    # Extract the Matplotlib code from the response
    model_response = response.choices[0].message.content.strip()

    try:
        # Debug: Print the generated code (optional)
        print("Executing the following code:\n", model_response)

        # Execute the Matplotlib code
        exec(model_response)

        # Upload the plot to Catbox
        catbox_upload_url = "https://catbox.moe/user/api.php"
        with open(output_file, "rb") as file:
            upload_response = requests.post(
                catbox_upload_url, 
                data={"reqtype": "fileupload"}, 
                files={"fileToUpload": file}
            )

        # Check response and return Catbox URL
        if upload_response.status_code == 200 and upload_response.text.startswith("https://"):
            return upload_response.text.strip()  # Return the Catbox URL
        else:
            return f"Error: Failed to upload to Catbox. Status: {upload_response.status_code}, Response: {upload_response.text}"

    except Exception as e:
        return f"Error: {str(e)}"
plotting_tool = Tool(
    name="plotting_tool",
    func=generate_and_upload_plot,
    description="""
    IN YOUR RESPONSE TAKE THE LINK RETURNED AND PASS IT TO THE USER UNDER 'HERE' REPLACE IT WITH THE LINK
    YOU MUST INPUT THE DATA FROM THE SEARCH TOOL INTO THIS TOOL FOR IT TO WORK. 
    THIS TOOL DOES NOT HAVE PRE-LOADED DATA AND REQUIRES IT FROM THE USER. 
    ALWAYS ENSURE THE DATA IS PASSED INTO THE REQUEST AND CLEARLY DESCRIBE THE PLOT NEEDED.
    EXECUTE THIS TOOL ONLY WHEN YOU HAVE ALL REQUIRED INFORMATION.
    IF YOU DO NOT HAVE ALL O0F THE DATA NEEDED, INPUT WHAT YOU DO HAVE
    """
    )
from flask import Flask, request, jsonify, render_template
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Initialize the LangChain agent
llm = ChatOpenAI(
    api_key=key,
    temperature=0,
    model="gpt-4o-mini"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Example tools (e.g., plotting and web browsing)

all_tools = [plotting_tool, web_browse_tool]

agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    
    handle_parsing_errors=True
)

def interact_with_agent(user_input):
    try:
        response = agent.run(user_input)
        return response
    except Exception as e:
        return str(e)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()
# CORS Configuration
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow cookies or Authorization headers if needed
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Define the input model
class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    """
    Endpoint to interact with the LangChain agent.
    Input:
      - message: User's input query
    Output:
      - response: Agent's response to the query
    """
    try:
        # Run the user query through the LangChain agent
        response = agent.run(query.message)
        return {"response": response}
    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"message": "Welcome to the AGENT_API!"}
