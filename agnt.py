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


from langchain.tools import tool, Tool
import openai
import requests

# Initialize OpenAI client
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
embedding_function = OpenAIEmbeddings(api_key=key)
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
# API Key for Financial Modeling Prep
fmp = os.getenv('FMP_API_KEY')
db = Chroma(persist_directory='./newfmpdocs_embedding_db/',embedding_function=embedding_function)

@tool
def fetch_financial_data(req: str) -> str:
    """
    THIS TOOL SHOULD ALWAYS BE USED FIRST IF USER IS ASKING ABOUT SOMETHING RELATED TO COMPANIES OR FINANCE
    Fetches financial data by determining the appropriate Financial Modeling Prep (FMP) API endpoint
    based on the user's query and contextual information stored in a similarity search database.

    Args:
        req (str): The user request specifying the financial data to be retrieved.

    Returns:
        str: The response from the FMP API, containing the requested financial data in text format.

    Functionality:
        1. Searches a database for the most relevant document using similarity search.
        2. Utilizes OpenAI's GPT model to identify the exact FMP API endpoint required for the query.
        3. Constructs the API request URL and appends the API key.
        4. Sends the request to the FMP API and retrieves the data.
    """
    # Retrieve relevant content from the database
    docs = db.similarity_search(req, k =5)
    search_content = docs[0].page_content
    print(search_content)
    # Generate the API endpoint using GPT
    from datetime import datetime
    today = datetime.today().strftime('%Y-%m-%d')
    message = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0,
        messages=[{
            'role': 'user',
            'content': f'based off of {req} and this data: {search_content} and the fact that the current date is {today} '
                       f'keep in mind that the current date is {today} when you make a request involving the current date'
                       f'return the financial modelling prep endpoint that will give the data needed to answer the question. '
                       f'RETURN SIMPLY THE FULL ENDPOINT AND NOTHING ELSE, WITH NO """ OR COMMAS AROUND IT. '
                       f'whenever the user asks for the current anything make 100% sure that you include the current date in the request: {today} '
        }]
    )
    url = message.choices[0].message.content.strip()
    
    # Append API key to the URL
    if '?' in url:
        url = f'{url}&apikey={fmp}'
    else:
        url = f'{url}?apikey={fmp}'
    print(url)
    # Fetch data from the API
    response = requests.get(url)
    texted_data = response.text

    texted_data = truncate_with_qwen(texted_data)
    print(texted_data)
    summary_response = client.chat.completions.create(
    model='gpt-4o-mini',
    temperature=0,
    messages=[{
        'role': 'user',
        'content': f'gather the numerical data needed to answer {req} from {texted_data}, return the numbers and as little explanation as possible except for a few words to explain what the numebrs are. FOR EXAMPLE, ALWAYS INCLUDE DATES WHERE APPLICABLE NEXT TO NUMBERS AND TEXT SAYING WHAT THE NUMBER IOS BEFORE NUMBERS. IF THE DATA IS NOT PROVIDED RETURN : USE WEB_BROWSE_TOOL AND NOTHING ELSE. ALSO RETURN USE_WEB_BROWSE_TOOL IF THE INPUT WAS ON CRYPTO AND NOT STOCKS' }]
    )
    print(summary_response.choices[0].message.content)
    return summary_response.choices[0].message.content

# Define the tool for LangChain usage
financial_data_tool = Tool(
    name="financial_data_tool",
    func=fetch_financial_data,
    description="""
    THIS TOOL SHOULD ALWAYS BE USED FIRST IF USER IS ASKING ABOUT SOMETHING RELATED TO COMPANIES OR FINANCE
    This tool fetches financial data from the Financial Modeling Prep (FMP) API by determining the correct
    API endpoint using OpenAI's GPT model and contextual similarity search. Provide a clear request
    (e.g., 'current price of cisco stock') to retrieve relevant data.
    """
)

######Chart_img_tool


import requests
import json

# Your CHART-IMG API Key
API_KEY = os.getenv('CHART_IMG_TOKEN')

# API endpoint for Advanced Chart
API_URL = "https://api.chart-img.com/v1/tradingview/advanced-chart"

def upload_to_catbox_binary(image_binary):
    """
    Uploads an image binary to Catbox and returns the URL.

    Args:
        image_binary (bytes): Binary data of the image to upload.

    Returns:
        str: URL of the uploaded image or error message.
    """
    # Catbox API endpoint
    CATBOX_API_URL = "https://catbox.moe/user/api.php"
    
    # File to upload
    files = {
        "fileToUpload": ("chart_img.png", image_binary)  # Provide a filename and binary data
    }
    
    # Data for the POST request
    data = {
        "reqtype": "fileupload"  # Request type
    }

    try:
        # Make the POST request to Catbox
        response = requests.post(CATBOX_API_URL, data=data, files=files)
        
        # Check for successful upload
        if response.status_code == 200:
            # Catbox returns the URL directly in the response body
            return response.text.strip()
        else:
            return f"Error: Failed to upload file. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def fetch_and_upload_chart(symbol, interval, range, theme, studies, chart_style):
    """
    Fetches a chart with technical indicators and uploads it to Catbox.

    Args:
        symbol (str): The TradingView symbol (e.g., "BINANCE:BTCUSDT").
        interval (str): The chart interval (e.g., "1D", "1H").
        range (str): The chart range (e.g., "1M", "6M").
        theme (str): Chart theme, "light" or "dark".
        studies (list): List of technical indicators (e.g., ["MA", "RSI", "BB"]).
        chart_style (str): Style of the chart, e.g., "candle", "line".

    Returns:
        str: URL of the uploaded chart or error message.
    """
    # Define the request parameters
    params = {
        "symbol": symbol,         # Symbol for the chart
        "interval": interval,     # Interval of the chart
        "theme": theme,           # Theme of the chart
        "studies": studies,       # List of technical indicators
        "timezone": "Etc/UTC",    # Chart timezone
        "width": 800,             # Chart width in pixels
        "height": 600,            # Chart height in pixels
        "format": "png",          # Image format
        "style": chart_style,
        "range": range            # Chart range (e.g., "1M", "6M")
    }

    # Authorization header
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    try:
        # Send the request to the CHART-IMG API
        response = requests.get(API_URL, headers=headers, params=params)

        # Check if the response is successful
        if response.status_code == 200:
            # Upload the chart image directly from the binary response content
            catbox_url = upload_to_catbox_binary(response.content)

            return catbox_url
        else:
            return f"Error: Failed to fetch chart. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

# Example usage
def generate_chart_img(request: str) -> str:
    """
    Generate a chart image based on the user's request and upload it to Catbox.

    This function integrates with the CHART-IMG API to create trading charts and
    uses the Catbox API to upload the chart directly, returning the final URL of the chart.
    
    Args:
        request (str): A natural language description of the chart the user wants.
                       Example: "Cisco's chart for the last 1 day as a line with as many indicators as you can think of."

    Returns:
        str: The URL of the uploaded chart image hosted on Catbox.
        
    Example Usage in a LangChain Agent:
        - Input: "Generate a candlestick chart for BTC/USDT over the last 1 month with RSI and MA indicators."
        - Output: A URL to the chart image, e.g., "https://files.catbox.moe/xyz.png"

    Agent Behavior:
        - Accepts a natural language prompt describing the desired chart.
        - Parses the request into API parameters using OpenAI's GPT model.
        - Fetches the chart using the CHART-IMG API.
        - Directly uploads the chart image to Catbox without saving it locally.
        - Returns the final Catbox-hosted URL for user consumption.

    Use Case in LangChain:
        - Add this function to a custom tool for generating and visualizing financial charts.
        - The agent should route user requests related to chart generation to this tool.
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{
            'role': 'user',
            'content': f'''based off of {request} Generate a Python dictionary with parameters for the CHART-IMG API based on the user's request.
            Include the following keys:
            - - enclose each item in the dictionary in " not '
            - do not inlude """python json or anything similar as it will ruin the function. simply return the dictionary with nothing around it no commas etc
            - symbol (str): The TradingView symbol (e.g., "BINANCE:BTCUSDT").
            - interval (str): The chart interval (e.g., "1D", "1H").
            - theme (str): Chart theme ("dark" or "light").
            - studies (list): List of technical indicators (e.g., ["MA", "RSI", "BB"]).
            - style (str): Chart style ("candle", "line", etc.).
            - width (int): Image width in pixels (minimum 320).
            - height (int): Image height in pixels (minimum 220).
            - format (str): Image format ("png" or "jpeg").
            - range (str): Chart range (e.g., "1M", "6M").
            Return only the raw Python dictionary, without explanation or additional text. DO NOT RETURN ''PYTHON OR SIMILAR JUST THE RAW DICTIONARY
            '''
        }]
    )
    resp = response.choices[0].message.content
    print(resp)
    dictionary_resp = json.loads(resp)

    # Fetch and upload the chart image
    catbox_url = fetch_and_upload_chart(
        symbol=dictionary_resp['symbol'],
        interval=dictionary_resp['interval'],
        range=dictionary_resp['range'],
        theme=dictionary_resp['theme'],
        studies=dictionary_resp['studies'],
        chart_style=dictionary_resp['style']
    )
    return f'<Image url="{catbox_url}" />'



chart_img_tool = Tool(
    name="chart_img_tool",
    func=generate_chart_img,
    description="""
    Generate a trading chart image with the CHART-IMG API based on user specifications and uploads it to Catbox.

    Args:
        request (str): A user-friendly description of the chart, including the symbol, interval, theme, style, 
                       and any technical indicators or other preferences.

    Returns:
        str: URL of the generated and uploaded chart image.
    """
)







# Initialize the LangChain agent
llm = ChatOpenAI(
    api_key=key,
    temperature=0,
    model="gpt-4o-mini"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Example tools (e.g., plotting and web browsing)

all_tools = [plotting_tool, web_browse_tool,chart_img_tool]

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
    import re
    def add_exclamation_to_links(text):
        # Regex to find [text](url) patterns and add '!' in front of the square brackets
        updated_text = re.sub(r'(\[.*?\]\(.*?\))', r'!\1', text)
        return updated_text

    try:
        # Run the user query through the LangChain agent
        response = agent.run(query.message)
        
        print(f'before cutting: {response}')
        if response.endswith("```"):
            
            response = re.sub(r'```$', '', response)
            print(f'after cutting: {response}')
            response = add_exclamation_to_links(response)
            return {"response": response}
        else:
            response = add_exclamation_to_links(response)
            return {"response": response}

    except Exception as e:
        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"message": "Welcome to the AGENT_API!"}
