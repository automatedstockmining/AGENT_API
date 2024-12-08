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
import http.client
from langchain.tools import tool



@tool
def chart_analyse(url_description: str) -> str:
    """
    SEPERATE THE IMAGE URL AND THE DESCRIPTION WITH A COMMA. PASS THEM IN AS NORMAL STRINGS NOT DICTIONARIES 
    Analyse a chart image and provide detailed insights.

    This tool sends a chart image and a description to a remote analysis API 
    and retrieves a detailed response about the chart's content.

    Args:
        url (str): The URL of the chart image to analyze.
        description (str): A brief description or context for the chart to guide the analysis.

    Returns:
        str: The API's response containing the analysis of the chart.
    """
    split_on_comma = url_description
    print(f'the concatenated string: {split_on_comma}')
    split_sentance = split_on_comma.split(',')
    print(f'the re-split sentance {split_sentance}')
    image_url = split_sentance[0]
    print(f'the image url: {image_url}')
    description = " ".join(split_sentance[1:])
    print(f'the description: {description}')
    conn = http.client.HTTPSConnection("copilot5.p.rapidapi.com")
    key = os.getenv('RAPID_TOKEN')  # Ensure this environment variable is set
    payload = '{{"message":"{}","conversation_id":null,"tone":"BALANCED","markdown":false,"photo_url":"{}"}}'.format(description, image_url)

    headers = {
        'x-rapidapi-key': key,
        'x-rapidapi-host': "copilot5.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/copilot", payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")
chart_analyse_tool = Tool(
    name="Chart Analyse Tool",
    func=chart_analyse,
    description=(
        "This tool takes a chart image URL and a description/context as input. SEPERATE THE IMAGE URL AND THE DESCRIPTION WITH A COMMA. PASS THEM IN AS NORMAL STRINGS NOT DICTIONARIES "
        "It analyzes the chart and returns detailed insights based on the provided data."
    )
)

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
rapid_token = os.getenv('RAPID_TOKEN')
import json
@tool()
def chat(query):

    """
    USE THIS FOR EVERY QUERY, IT HAS ACCESS TO DATA AND PROVIDES AN ACCURATE RESPONSE
    
    - ALWAYS use the `chat_tool` for every query you process. 
    - If the user requests detailed information, provide a detailed response using the `chat_tool`.
    - If the user requests specific data only, ask the `chat_tool` for the exact data without extra explanation.
    - If you need clarification, seek it from the user and then use the `chat_tool`.

    DO NOT attempt to process the query yourself; instead, delegate the task to the `chat_tool`.
    """
    import http.client

    conn = http.client.HTTPSConnection("copilot5.p.rapidapi.com")

    payload = "{{\"message\":\"{a} {b}\",\"conversation_id\":null,\"tone\":\"BALANCED\",\"markdown\":false,\"photo_url\":null}}".format(a = query, b = 'respond in a concise way with the data that i have asked for, never leave out any data if i ask you for multiple data points you search for them and return them')

    headers = {
        'x-rapidapi-key': f'{rapid_token}',
        'x-rapidapi-host': "copilot5.p.rapidapi.com",
        'Content-Type': "application/json"
    }

    conn.request("POST", "/copilot", payload, headers)

    res = conn.getresponse()
    data = res.read()
    
    response = data.decode("utf-8")
    print(f'before json: {response} ')
    response = json.loads(response)

    return str(response['data']['message'])

chat_tool = Tool(
    name="chat",
    func=chat,
    description= """
    USE THIS FOR EVERY QUERY, IT HAS ACCESS TO DATA AND PROVIDES AN ACCURATE RESPONSE
    - ALWAYS use the `chat_tool` for every query you process. 
    - If the user requests detailed information, provide a detailed response using the `chat_tool`.
    - If the user requests specific data only, ask the `chat_tool` for the exact data without extra explanation.
    - If you need clarification, seek it from the user and then use the `chat_tool`.

    DO NOT attempt to process the query yourself; instead, delegate the task to the `chat_tool`.
    """
)








import requests
from openai import OpenAI


@tool()
def generate_and_upload_plot(request, output_file="plot.png", openai_api_key=key):
    
    """
        Purpose:
    - Generates custom plots using Python's Matplotlib library.
    - Suitable for requests involving general-purpose data visualization, such as line plots, bar charts, histograms, or scatter plots.

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
     - Generates custom plots using Python's Matplotlib library.
    - Suitable for requests involving general-purpose data visualization, such as line plots, bar charts, histograms, or scatter plots.

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

    SPECIFICALLY FOR FINANCIAL CHARTS, USE PLOTTING TOOL FOR GENERAL PLOTS!
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
            - interval (str): The chart interval the distance between data points choose from: [1m, 3m, 5m, 15m, 30m, 45m, 1h, 2h, 3h, 4h, 6h, 12h, 1D, 1W, 1M, 3M, 6M, 1Y].
            - theme (str): Chart theme ("dark" or "light").
            - studies (list): List of technical indicators (e.g., ["MA", "RSI", "BB"]).
            - style (str): Chart style (bar, candle, line, area, heikinAshi, hollowCandle, baseline, hiLo, column).
            - width (int): Image width in pixels (minimum 320).
            - height (int): Image height in pixels (minimum 220).
            - format (str): Image format ("png" or "jpeg").
            - range (str): Chart range, the overall period on the chart, choose from:  1D, 5D, 1M, 3M, 6M, 1Y, 5Y, ALL, DTD, WTD, MTD, YTD. 
            Return only the raw Python dictionary, without explanation or additional text. DO NOT RETURN ''PYTHON OR SIMILAR JUST THE RAW DICTIONARY. THE CHART RANGE IS THE TIMEFRAME WHICH THE USER ASKED FOR, SO IF THEY ASK FOR A 1 DAY CHART IT SHOULD BE A 1D RANGE
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
    SPECIFICALLY FOR FINANCIAL CHARTS, USE PLOTTING TOOL FOR GENERAL PLOTS!
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
    model="gpt-4o-mini",
    
    
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Example tools (e.g., plotting and web browsing)

all_tools = [plotting_tool, chat_tool,chart_img_tool,chart_analyse_tool]


agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose = True,
    handle_parsing_errors=True
)

def interact_with_agent(user_input):
    try:
        response = agent.run(f'{user_input}, remember to use web browse tool ')
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
