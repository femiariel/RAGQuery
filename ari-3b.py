import torch
from sentence_transformers import SentenceTransformer
import gradio as gr
from transformers import pipeline
import fitz  # PyMuPDF
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import easyocr
import io
from PIL import Image
import nltk
import numpy as np
import faiss
import pandas as pd



# Method for cleaning texts
def clean_text(text):
    """
    Clean a given text by removing extra spaces and replacing newlines with spaces.
    
    This function takes a text string as input, replaces all newline characters with spaces,
    and removes any extra spaces to produce a clean, single-spaced text string.
    
    Parameters:
    text (str): The input text to be cleaned.
    
    Returns:
    str: The cleaned text.
    """
    text = text.replace('\n', ' ')  # Replace all newlines with spaces
    text = ' '.join(text.split())  # Remove extra spaces
    return text

# Method for extracting text from each page
def extract_text_from_page(page):
    """
    Extract text from a PDF page, using OCR if necessary.
    
    This function first tries to extract text directly from the PDF page.
    If no text is found, it performs OCR on any images within the page to extract text.
    
    Parameters:
    page (fitz.Page): A page object from the PDF document.
    
    Returns:
    str: The extracted and cleaned text from the page.
    """
    text = page.get_text()
    if text.strip():
        return clean_text(text)
    
    text = ''
    for img in page.get_images(full=True):
        base_image = page.get_pixmap()
        pil_image = Image.open(io.BytesIO(base_image.tobytes()))
        numpy_image = np.array(pil_image)

        # EasyOCR returns a list of tuples (bbox, text)
        ocr_results = reader.readtext(numpy_image, paragraph=True)
        for _, detected_text in ocr_results:
            text += detected_text + ' '  # Concatenate detected text

    return clean_text(text)

# Method for extracting text from a PDF
def extract_sentences_from_pdf(pdf_path):
    """
    Extract sentences from a PDF document.
    
    This function opens a PDF document, extracts text from each page,
    cleans the text, and tokenizes it into sentences.
    
    Parameters:
    pdf_path (str): The file path to the PDF document.
    
    Returns:
    list: A list of sentences extracted from the PDF.
    """
    document = fitz.open(pdf_path)
    all_sentences = []

    for page in document:
        text = extract_text_from_page(page)
        cleaned_text = clean_text(text)
        sentences = nltk.tokenize.sent_tokenize(cleaned_text)
        all_sentences.extend(sentences)

    document.close()
    return all_sentences


def normalization(embedding):
    """
    Normalize a set of embedding vectors.
    
    This function takes a matrix of embedding vectors as input and returns a matrix
    where each vector is normalized (so that its norm is equal to 1).
    
    Parameters:
    embedding (array-like): A 2D matrix where each row is an embedding vector.
    
    Returns:
    normal_embedding (np.ndarray): The matrix of normalized embedding vectors.
    """
    normal_embedding = np.array(embedding, dtype=np.float32)  # Convert to NumPy array of type float32
    normal_embedding /= np.linalg.norm(normal_embedding, axis=1)[:, np.newaxis]  # Normalize each vector
    return normal_embedding

def creatingIndexes(ensemble):
    """
    Create an index for vector similarity search using inner product.
    
    This function takes a matrix of vectors as input and creates an index to facilitate
    similarity search (based on inner product) using the FAISS library.
    
    Parameters:
    ensemble (np.ndarray): A 2D matrix where each row is an embedding vector.
    
    Returns:
    index (faiss.IndexFlatIP): A FAISS index for vector similarity search.
    """
    d = ensemble.shape[1]  # Dimension of the vectors
    index = faiss.IndexFlatIP(d)  # Create a FAISS index for inner product
    index.add(ensemble)  # Add vectors to the index
    return index


# Method for searching in the index
def searchinIndex(index, normal_embedding):
    """
    Search for the closest vectors in the index given a normalized embedding.
    
    This function performs a search in the FAISS index using the provided normalized embedding
    and returns the top 10 closest vectors along with their distances.
    
    Parameters:
    index (faiss.Index): The FAISS index to search within.
    normal_embedding (np.ndarray): The normalized embedding to search for.
    
    Returns:
    pd.DataFrame: A DataFrame containing the distances and indices of the top 10 closest vectors.
    """
    D, I = index.search(normal_embedding, 10)  # Perform the search
    r = pd.DataFrame({'distance': D[0], 'index': I[0]})  # Create a DataFrame with distances and indices
    return r

# Method for generating response based on instruction and PDF content
def generation(instruction, pdf):
    """
    Generate a response based on a given instruction and the content of a PDF.
    
    This function extracts sentences from a PDF, encodes them into embeddings, normalizes these embeddings,
    creates an index for similarity search, and then finds the most relevant context to generate a response
    using a language model.
    
    Parameters:
    instruction (str): The instruction or query to base the generation on.
    pdf (str): The file path to the PDF document.
    
    Returns:
    str: The generated response from the language model.
    """
    extracted_sentences = extract_sentences_from_pdf(pdf)  # Extract sentences from the PDF
    embeddings = model.encode(extracted_sentences)  # Encode the sentences into embeddings
    normal_embeddings = normalization(embeddings)  # Normalize the embeddings

    index = creatingIndexes(normal_embeddings)  # Create an index for the embeddings
    query_embedding = model.encode([instruction])  # Encode the instruction into an embedding
    query_normalization = normalization(query_embedding)  # Normalize the instruction embedding
    i = searchinIndex(index, query_normalization)  # Search for the closest embeddings in the index

    # Combine the top two relevant sentences as context
    context = extracted_sentences[i['index'][0]] + extracted_sentences[i['index'][1]]

    # Generate and return the response using the language model
    return llm_context_chain.predict(instruction=instruction, context=context).lstrip()
#embedding model
model =  SentenceTransformer("BAAI/bge-m3")

#using of cpu or gpu according to what is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
generate_text = pipeline(model="databricks/dolly-v2-3b",device=device,
                         trust_remote_code=True,return_full_text=True,max_length=60)

# Download the Punkt sentence tokenizer and the necessary data for French
nltk.download('punkt')
nltk.download('fr_core_news_sm')

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")


hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# Creation of OCR reader
reader = easyocr.Reader(['en'])  # Ici, 'en' est pour l'anglais, ajustez selon vos besoins


interface = gr.Interface(fn=generation, inputs=["text", "file"], outputs="text")

# Interface launching
interface.launch()

