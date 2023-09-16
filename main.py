# importing modules

from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from typing_extensions import Concatenate
import streamlit as st
import speech_recognition as sr
import requests
import json
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import time
import wget
import argparse


# import API key
os.environ["OPENAI_API_KEY"] = "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"
API_KEY = os.getenv("OPENAI_API_KEY")

def STT_microphone():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        
        print("Calibrating...")
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("listening now...")
    
        audio = r.listen(source, timeout=5, phrase_time_limit=30,)
        print("Recognizing...")
        
        transcription = r.recognize_whisper(
            audio,
            model="base.en",
            show_dict=True,
        )["text"]

        print("User asked :", transcription)
    return transcription


# read pdf file
def pdf_read(file):

    pdfreader = PdfReader(file)
    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # We need to split the jumbled text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_overlap=200,
        chunk_size=800, # this should be less than max input token taken by llm
        length_function=len
    )
    texts = text_splitter.split_text(raw_text)

    return texts

# get embeddings from OpenAI
def get_embeddings(text):
    
    embeddings = OpenAIEmbeddings()
    # text from pdf is converted into embeddings and stored into vector database
    vector_db = FAISS.from_texts(texts=text, embedding=embeddings)

    return vector_db


def llm_response2query(vector_db,query,use_gpt4=None):
    
    if use_gpt4 == True:
        # initilise LLM from ChatOpenAI in chain,
        llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=API_KEY)
    else:
        # initilise LLM from OpenAI in chain,
        llm = OpenAI(temperature=0.7, model="text-davinci-003", openai_api_key=API_KEY)
   
    # load llm into chain
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    # perform similarity search on query
    docs = vector_db.similarity_search(query=query)
    # run chain answer based on selected doc
    answer = chain.run(input_documents=docs, question=query)
    print("ChatBot replies:", answer)
    return answer


# pass answer from LLM chain to D-id 
def facial_animation(query):
    payload = {
            "script": {
                "type": "text",
                "subtitles": "false",
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural",
                    "voice_config": {
                        "style": "Friendly",
                        "rate": "0.80"
                    }               
                },
                "ssml": "false",
                "input": query
            },
            "config": {
                "fluent": "false",
                "pad_audio": "0.0",
                "stitch": True
            },
            "source_url": "https://create-images-results.d-id.com/google-oauth2%7C107017662203149014763/upl_0ta3JhUnJ1W2qKckIjRMu/image.png"
        }
    
    headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": "Basic Y21sbVlYUXViV0ZvWVcxdGIyUkFiM1YwYkc5dmF5NWpiMjA6OHBGT2sxSjlhcU5nZjR4VmF0MmpP"
        }
    # response_id = resp_id(endpoint, headers, payload)
    # polling = poll(endpoint, headers)
    # result_url = result(endpoint, headers, payload)
    return headers, payload


def resp_id(endpoint, headers, payload):

    # post request with endpoint and json payload - returns the id of the response
    response = requests.post(endpoint, json=payload, headers=headers)
    response_id = response.json()
    return response_id["id"]

def poll(endpoint, headers, payload):

    # poll endpoint and use .get to pull info about the response body
    response_id = resp_id(endpoint, headers, payload)
    polling_endpoint = endpoint + response_id # URL with which you can ask the API if the processing is done 
    poll_response = requests.get(polling_endpoint, headers=headers)
    return poll_response.json()

# loop the response until it completed 
def result(endpoint, headers, payload):
    while True:
        poll_result = poll(endpoint, headers, payload)
        if (poll_result['status'] == 'done'):
            return poll_result["result_url"]
        elif (poll_result['status'] == 'error'):
            return poll_result["error"]
    
def get_response(endpoint, headers, response_id):

    url = endpoint + response_id
    response = requests.get(url, headers=headers)
    r = response.json()

    return r["result_url"]


if __name__ == "__main__":

    # facial animation endpoint
    endpoint = "https://api.d-id.com/talks/"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("PDF_doc", type=str)
    args = parser.parse_args()

    st.title('🤖 VVA Bot Test')
    audio_input_button = st.button("Press the button and speak")
    if audio_input_button:

        transcription = STT_microphone()
        # audio_data = sd.rec(int(5 * 44100), channels=1, blocking=True)
        # st.audio(audio_data, format="wav")
        st.write("User asked:",transcription)

        # read pdf file
        file = args.PDF_doc
        pdf_pages = pdf_read(file)

        # embeddings from pdf pages and store into db
        vector_db = get_embeddings(pdf_pages)

        # pass query to llm and get response based on context from vector db
        query = transcription
        answer = llm_response2query(vector_db, query, use_gpt4=False)
        
        st.write("ChatBot replies:", answer)

        # pass answer to facial animation API
        headers, payload = facial_animation(str(answer))
        # post request
        response_id = resp_id(endpoint, headers, payload)
        # get request 
        time.sleep(30)
        url = get_response(endpoint, headers, response_id)
        print(url)
        # show video on streamlit app
        st.write("*Loading animated bot...*",)
        time.sleep(20)
        st.video(file)



    

   






