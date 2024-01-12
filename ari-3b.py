import torch
from sentence_transformers import SentenceTransformer
import gradio as gr
model =  SentenceTransformer("paraphrase-MiniLM-L6-v2")
from transformers import pipeline
import fitz  # PyMuPDF
generate_text = pipeline(model="databricks/dolly-v2-3b",
                         trust_remote_code=True,return_full_text=True)

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)




def clean_text(text):
    # Remplace tous les sauts de ligne par des espaces
    text = text.replace('\n', ' ')
    # Supprime les espaces supplémentaires
    text = ' '.join(text.split())
    return text

import easyocr
import io
from PIL import Image
import nltk

# Créer le lecteur OCR (peut être réutilisé)
reader = easyocr.Reader(['en'])  # Ici, 'en' est pour l'anglais, ajustez selon vos besoins

def extract_text_from_page(page):
    text = page.get_text()
    if text.strip():
        return clean_text(text)

    text = ''
    for img in page.get_images(full=True):
        base_image = page.get_pixmap()
        pil_image = Image.open(io.BytesIO(base_image.tobytes()))
        numpy_image = np.array(pil_image)

        # EasyOCR renvoie une liste de tuples (bbox, text)
        ocr_results = reader.readtext(numpy_image, paragraph=True)
        for _, detected_text in ocr_results:
            text += detected_text + ' '  # Concatène le texte détecté

    return clean_text(text)


nltk.download('punkt')

def extract_sentences_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    all_sentences = []

    for page in document:
        text = extract_text_from_page(page)
        cleaned_text = clean_text(text)
        sentences = nltk.tokenize.sent_tokenize(cleaned_text)
        all_sentences.extend(sentences)

    document.close()
    return all_sentences






import numpy as np
import faiss
import pandas as pd
def normalization(embedding):
    normal_embedding=np.array(embedding,dtype=np.float32)
    normal_embedding/=np.linalg.norm(normal_embedding,axis=1)[:,np.newaxis]
    return normal_embedding

def creatingIndexes(ensemble):
    d=ensemble.shape[1]
    index=faiss.IndexFlatIP(d)
    index.add(ensemble)
    return index

def searchinIndex(index, normal_embedding):
    D,I= index.search(normal_embedding, 10)
    r=pd.DataFrame({'distance':D[0],'index':I[0]})
    return r


def generation(instruction,pdf):
    extracted_sentences = extract_sentences_from_pdf(pdf)
    embeddings = model.encode(extracted_sentences[:50])
    normal_embeddings=normalization(embeddings)

    index=creatingIndexes(normal_embeddings)
    query_embedding=model.encode([instruction])
    query_normalization=normalization(query_embedding)
    i=searchinIndex(index,query_normalization)

    #print(extracted_sentences[i['index'][0]] + extracted_sentences[i['index'][1]])

    context=extracted_sentences[i['index'][0]] + extracted_sentences[i['index'][1]]

    return llm_context_chain.predict(instruction=instruction, context="").lstrip()

interface = gr.Interface(fn=generation, inputs=["text", "file"], outputs="text")

# Lancez l'interface
interface.launch()

