import matplotlib
matplotlib.use('Agg')

import re
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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("app.log")  # Log to a file named app.log
    ]
)


@tool
def chart_analyse(url_description: str) -> str:
    """
    takes in a natural language request with a description of how you want the chart analysed, inside this description somewhere, make sure the image url is included. IT RETURNS THE CHART ANALYSIS
    """
    pattern = r'https?://[^\s]+'
    match = re.search(pattern, url_description)
    if not match:
        return "Error: No valid URL found in the input."

    image_url = match.group()
    
    description = url_description.replace(image_url,'')
    print(f'the description {description}')
    
    import requests
    import json

    url = "https://copilot5.p.rapidapi.com/copilot"

    payload = {
        "message": f"{str(description)}",
        "conversation_id": None,
        "tone": "BALANCED",
        "markdown": False,
        "photo_url": f"{str(image_url.strip('.,;!?()[]{}<>'))}"
    }
    headers = {
        "x-rapidapi-key": os.getenv('RAPID_TOKEN'),
        "x-rapidapi-host": "copilot5.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    # Decode the response content as a string
    response_text = response.content.decode('utf-8')

    # Clean and parse the response
    if response_text.startswith("b'") or response_text.startswith('b"'):
        # Remove leading "b'" or 'b"'
        response_text = response_text[2:-1]
        
        # Replace escaped single quotes with double quotes to ensure JSON compatibility
        response_text = response_text.replace("\\'", "'").replace("'", '"')

    # Parse the cleaned response as JSON
    response_data = json.loads(response_text)

    # Extract the desired message
    if 'data' in response_data and 'message' in response_data['data']:
        return response_data['data']['message']
    else:
        return "Key 'data' or 'message' not found in the response."

chart_analyse_tool = Tool(
    name="Chart Analyse Tool",
    func=chart_analyse,
    description=(
        '''takes in a natural language request with a description of how you want the chart analysed, inside this description somewhere, make sure the image url is included. IT RETURNS THE CHART ANALYSIS''')
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
    rapid_token = os.getenv('RAPID_TOKEN')
    payload = "{{\"message\":\"{a} {b}\",\"conversation_id\":null,\"tone\":\"BALANCED\",\"markdown\":false,\"photo_url\":null}}".format(a = query, b = 'respond in a concise way with the data that i have asked for, never leave out any data if i ask you for multiple data points you search for them and return them')

    headers = {
        'x-rapidapi-key': rapid_token,
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
def generate_and_upload_plot(request):
    """
    Purpose:
    - Generates a custom plot based on a natural language request using Python's Matplotlib library.
    - Finds the necessary data, creates the plot, and uploads it to Catbox for public access.

    Arguments:
    - request (str): A natural language description of the desired plot and data.

    Returns:
    - str: URL of the uploaded plot image.
    """
    import os
    import io
    import requests
    import matplotlib.pyplot as plt


   
 
    import requests
    with open("plot_chart.png", "wb") as file:
    # Write an empty byte string to the file
        file.write(b'')
    end_code = '''
plt.savefig('plot_chart.png')
    
    '''
    url = "https://copilot5.p.rapidapi.com/copilot"
    
    payload = {
        "message": "{a}---{b}".format(a = request, b ='Search the web and find the data needed for their request, then return the matplotlib python code to plot this data. RETURN NOTHING ELSE OTHER THAN THE MATPLOTLIB CODE, NO EXPLANATIONS ETC. YOU SEARCH THE WEB FOR THE DATA NEEDED FOR THEIR PLOT AND PLOT THE REAL DATA. DO NOT INCLUDE PLT.SHOW IN THE CODE'),
        "conversation_id": None,
        "tone": "BALANCED",
        "markdown": False,
        "photo_url": None
    }
    headers = {
        "x-rapidapi-key": os.getenv('RAPID_TOKEN'),
        "x-rapidapi-host": "copilot5.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    print(f'the response status {response}')
    response = response.json()
    print('coverted to json')
    response = response['data']['message']
    print(f'got the response: {response}')
    matplotlib_code = f'{response}\n\n {end_code}'
    buffer = io.BytesIO()
    exec(matplotlib_code, {"plt": plt, "io": io, "buffer": buffer})

    print('successfully executed the code')
    print(response)
       # Upload the plot from the memory buffer
    file_path = "plot_chart.png"

    # 0x0.st upload endpoint
    upload_url = "https://0x0.st/"

   
    with open(file_path, "rb") as file:
        response = requests.post(upload_url, files={"file": file})
        upload_url = "https://catbox.moe/user/api.php"
        with open(file_path, "rb") as file:
            payload = {
                "reqtype": "fileupload",
                "userhash": "",  # Optional, leave empty for anonymous upload
            }
            files = {
                "fileToUpload": file
            }
            response = requests.post(upload_url, data=payload, files=files)

        # Check the response
        if response.status_code == 200:
            uploaded_url = response.text.strip()  # The response is the URL
            logging.info(f"File uploaded successfully: {uploaded_url}")
            return uploaded_url
        else:
            logging.error(f"Failed to upload file. Status Code: {response.status_code}")
            logging.error(f"Response Text: {response.text}")
            return None

plotting_tool = Tool(
    name="plotting_tool",
    func=generate_and_upload_plot,
    description=
    """
    Purpose:
    - Generates a custom plot based on a natural language request using Python's Matplotlib library.
    - Finds the necessary data, creates the plot, and uploads it to Catbox for public access.

    Arguments:
    - request (str): A natural language description of the desired plot and data.

    Returns:
    - str: URL of the uploaded plot image.
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

      """
    from black import format_str, FileMode
import autopep8

with open('formatted-indicators-json.txt','r') as file:
    read = file.read()
my_json = ast.literal_eval(read)

save_code = """
# Upload binary content to Catbox
url = "https://catbox.moe/user/api.php"
payload = {"reqtype": "fileupload"}
files = {"fileToUpload": ("chart_image.png", response.content)}

# Send POST request to upload the image
upload_response = requests.post(url, data=payload, files=files)

# Check the response
if upload_response.status_code == 200:
    print(f"{upload_response.text.strip()}")
    
else:
    print(f"Failed to upload to Catbox: {upload_response.status_code} - {upload_response.text}")
"""

def find_indicator(data, name):
    for item in data:
        if item.get("indicator name") == name:
            return item['description']
    return None  # Return None if the indicator is not found


technical_indicators = [
    "Accumulation/Distribution",
    "Accumulative Swing Index",
    "Advance/Decline",
    "Arnaud Legoux Moving Average",
    "Aroon",
    "Average Directional Index",
    "Average True Range",
    "Awesome Oscillator",
    "Balance of Power",
    "Bollinger Bands",
    "Bollinger Bands %B",
    "Bollinger Bands Width",
    "Chaikin Money Flow",
    "Chaikin Oscillator",
    "Chaikin Volatility",
    "Chande Kroll Stop",
    "Chande Momentum Oscillator",
    "Chop Zone",
    "Choppiness Index",
    "Commodity Channel Index",
    "Connors RSI",
    "Coppock Curve",
    "Detrended Price Oscillator",
    "Directional Movement",
    "Donchian Channels",
    "Double EMA",
    "Ease of Movement",
    "Elder's Force Index",
    "Envelopes",
    "Fisher Transform",
    "Historical Volatility",
    "Hull Moving Average",
    "Ichimoku Cloud",
    "Keltner Channels",
    "Klinger Oscillator",
    "Know Sure Thing",
    "Least Squares Moving Average",
    "Linear Regression Curve",
    "Linear Regression Slope",
    "MA Cross",
    "MA with EMA Cross",
    "MACD",
    "Majority Rule",
    "Mass Index",
    "McGinley Dynamic",
    "Momentum",
    "Money Flow Index",
    "Moving Average",
    "Moving Average Adaptive",
    "Moving Average Channel",
    "Moving Average Double",
    "Moving Average Exponential",
    "Moving Average Hamming",
    "Moving Average Multiple",
    "Moving Average Triple",
    "Moving Average Weighted",
    "Net Volume",
    "On Balance Volume",
    "Parabolic SAR",
    "Price Channel",
    "Price Oscillator",
    "Price Volume Trend",
    "Rate Of Change",
    "Relative Strength Index",
    "Relative Vigor Index",
    "SMI Ergodic Indicator/Oscillator",
    "Smoothed Moving Average",
    "Standard Deviation",
    "Standard Error",
    "Standard Error Bands",
    "Stochastic",
    "Stochastic RSI",
    "Super Trend",
    "Trend Strength Index",
    "Triple EMA",
    "TRIX",
    "True Strength Index",
    "Ultimate Oscillator",
    "Volatility Close-to-Close",
    "Volatility Index",
    "Volatility O-H-L-C",
    "Volatility Zero Trend Close-to-Close",
    "Volume",
    "Volume Oscillator",
    "Volume Profile Visible Range",
    "Vortex Indicator",
    "VWAP",
    "VWMA",
    "Williams %R"
]


import logging
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

      """
    from black import format_str, FileMode
import autopep8

with open('formatted-indicators-json.txt','r') as file:
    read = file.read()
my_json = ast.literal_eval(read)

save_code = """
with open('technical_chart.png','wb') as file:

    file.write(response.content)
"""

def find_indicator(data, name):
    for item in data:
        if item.get("indicator name") == name:
            return item['description']
    return None  # Return None if the indicator is not found


technical_indicators = [
    "Accumulation/Distribution",
    "Accumulative Swing Index",
    "Advance/Decline",
    "Arnaud Legoux Moving Average",
    "Aroon",
    "Average Directional Index",
    "Average True Range",
    "Awesome Oscillator",
    "Balance of Power",
    "Bollinger Bands",
    "Bollinger Bands %B",
    "Bollinger Bands Width",
    "Chaikin Money Flow",
    "Chaikin Oscillator",
    "Chaikin Volatility",
    "Chande Kroll Stop",
    "Chande Momentum Oscillator",
    "Chop Zone",
    "Choppiness Index",
    "Commodity Channel Index",
    "Connors RSI",
    "Coppock Curve",
    "Detrended Price Oscillator",
    "Directional Movement",
    "Donchian Channels",
    "Double EMA",
    "Ease of Movement",
    "Elder's Force Index",
    "Envelopes",
    "Fisher Transform",
    "Historical Volatility",
    "Hull Moving Average",
    "Ichimoku Cloud",
    "Keltner Channels",
    "Klinger Oscillator",
    "Know Sure Thing",
    "Least Squares Moving Average",
    "Linear Regression Curve",
    "Linear Regression Slope",
    "MA Cross",
    "MA with EMA Cross",
    "MACD",
    "Majority Rule",
    "Mass Index",
    "McGinley Dynamic",
    "Momentum",
    "Money Flow Index",
    "Moving Average",
    "Moving Average Adaptive",
    "Moving Average Channel",
    "Moving Average Double",
    "Moving Average Exponential",
    "Moving Average Hamming",
    "Moving Average Multiple",
    "Moving Average Triple",
    "Moving Average Weighted",
    "Net Volume",
    "On Balance Volume",
    "Parabolic SAR",
    "Price Channel",
    "Price Oscillator",
    "Price Volume Trend",
    "Rate Of Change",
    "Relative Strength Index",
    "Relative Vigor Index",
    "SMI Ergodic Indicator/Oscillator",
    "Smoothed Moving Average",
    "Standard Deviation",
    "Standard Error",
    "Standard Error Bands",
    "Stochastic",
    "Stochastic RSI",
    "Super Trend",
    "Trend Strength Index",
    "Triple EMA",
    "TRIX",
    "True Strength Index",
    "Ultimate Oscillator",
    "Volatility Close-to-Close",
    "Volatility Index",
    "Volatility O-H-L-C",
    "Volatility Zero Trend Close-to-Close",
    "Volume",
    "Volume Oscillator",
    "Volume Profile Visible Range",
    "Vortex Indicator",
    "VWAP",
    "VWMA",
    "Williams %R"
]


def financial_charting(request):
    print(request)
    go_completion = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[
        {"role": "user", "content": f"Based on the following request: '{request}', determine the technical indicators directly needed for the chart choose from {technical_indicators}. Return only the exact names of the indicators spaced by a , (e.g., 'Volatility Index') with no explanation or additional text. ALWAYS RETURN THE VOLUME ALONG WITH THE INDICATORS THEY ASK FOR. ONLY RETURN THE INDICATORS THEY ASK FOR. IF THEY DONT EXPLICITLY ASK FOR TECHNCIAL INDICATORS RETURN ABSOLUTELY NOTHING, AND ONLY RETRUN TECHNICAL INDICATORS FROM THE LIST, MAKE NONE UP YOURSELF. ONLY UP TO FIVE INDICATORS ARE ALLOWED, NO MORE!"}
    ]
    )
    final_resp = go_completion.choices[0].message.content
    print(final_resp)

    final_resp = final_resp.split(',')
    print(f'the list of indicators: {final_resp}')
    list_of_technicals = []
    [list_of_technicals.append(var) for var in final_resp]
    
    full_data = []
    for fin_var in list_of_technicals:
        new_var = find_indicator(my_json, fin_var)
        if new_var:  # Append only if new_var is not None
            full_data.append(new_var)
    final_data = ''.join(full_data)
    print(f"the data retrieved through RAG {final_data}")


    completion = client.chat.completions.create(
    model="gpt-4o",
    temperature=0,
    messages=[
        {"role": "user", "content": f""""
    Generate Python code for the following request: '{request}'. Use the following data for studies:
    {full_data}. 

    Each indicator's 'name' should appear in the 'studies' section of the payload, with appropriate inputs and overrides. Return up to the response (stop before response.json()) in the code and ONLY return the code, no explanations or additional text. THE RANGE IS THE CHART TIMEFRAME THAT THEY WANT SO IF THEY ASK FOR A 1 DAY CHART THE RANGE SHOULD BE 1D THE RANGE CAN ONLY BE: 1D, 5D, 1M, 3M, 6M, 1Y, 5Y, ALL, DTD, WTD, MTD, YTD, THE INTERVAL I THE PERIOD BETWEEN DATA POINTS ON THE CHART. ALWAYS KEEP "showMainPane": True unless explicity specified. THE NAMES OF THE INDICATORS THAT YOU SHOULD PUT IN THE REQUEST ARE {final_resp} """}
 ]
    )
    response = completion.choices[0].message.content
    print(f'the response \n\n\n {response}')

    if response.startswith("```python"):
        # Remove the starting and ending code block markers
        response = response.replace("```python", "").replace("```", "")
        response = autopep8.fix_code(response)
        
    else:
        pass


    import os
    import io
    import sys
    response = response.replace("false", "False").replace('true','True')
    response = response.replace('{YOUR_API_KEY}',os.getenv('CHART_IMG_TOKEN'))
    response = f'{response}\n\n{save_code}'
    exec(response)
    # File path to upload
    file_path = "technical_chart.png"

    # 0x0.st upload endpoint
    upload_url = "https://0x0.st/"

   
    with open(file_path, "rb") as file:
        response = requests.post(upload_url, files={"file": file})
        upload_url = "https://catbox.moe/user/api.php"
        with open(file_path, "rb") as file:
            payload = {
                "reqtype": "fileupload",
                "userhash": "",  # Optional, leave empty for anonymous upload
            }
            files = {
                "fileToUpload": file
            }
            response = requests.post(upload_url, data=payload, files=files)

        # Check the response
        if response.status_code == 200:
            uploaded_url = response.text.strip()  # The response is the URL
            logging.info(f"File uploaded successfully: {uploaded_url}")
            return uploaded_url
        else:
            logging.error(f"Failed to upload file. Status Code: {response.status_code}")
            logging.error(f"Response Text: {response.text}")
            return None

chart_img_tool = Tool(
    name="chart_img_tool",
    func=financial_charting,
    description="""
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
