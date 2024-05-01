from dotenv import load_dotenv
import os
load_dotenv()
from typing import List, Dict
from unstructured.cleaners.core import (
    bytes_string_to_string,
    clean,
    clean_bullets,
    clean_dashes,
    clean_extra_whitespace,
    clean_non_ascii_chars,
)
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain_openai import ChatOpenAI
from gliner import GLiNER
from langchain.output_parsers import PydanticOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from collections import defaultdict
from models import Triple, Graph
from  graph_builder import GraphBuilder
from neo4j import GraphDatabase
from helpers import clean_mapping
from models import DocOntology
import logging 
from whyhow import WhyHow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChunkProcessor:
    
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        """
        Initializes the ChunkProcessor class with the file path and chunk parameters.

        Parameters:
        - file_path (str): The path to the PDF file from which text chunks are extracted.
        - chunk_size (int): Size of each text chunk.
        - chunk_overlap (int): Overlap size between consecutive chunks.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_chunks(self, limit: int = None) -> List[str]:
        """
        Process a PDF file and split its content into chunks.

        Args:
            limit (int, optional): The maximum number of chunks to return. Defaults to None, which returns all chunks.

        Returns:
            list: A list of strings, each representing a chunk of the PDF content.
        """

            # create a loader
        loader = PyPDFLoader(self.file_path)
        # load your data
        data = loader.load()

        if not data:
            # TODO: Handle more elegantly.
            raise RuntimeError(
                "No data laoded from the file, or the file is empty."
            )

        # Split your data up into smaller documents with Chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        # Convert Document objects into strings and return the list
        chunks = [
            str(doc.page_content) for doc in text_splitter.split_documents(data)
        ]
        # If limit is specified and less than the length of chunks, return a slice up to limit
        if limit is not None and limit < len(chunks):
            return chunks[:limit]
        # Otherwise, return all chunks
        return chunks
    

    def clean_chunk(self, chunk_txt: str) -> str:
        """
        Cleans a single text chunk by removing unwanted formatting and normalizing the text.

        Parameters:
        - chunk_txt (str): The text chunk to be cleaned.

        Returns:
        - str: The cleaned text chunk.
        """
        text_cleaned_bullets = clean_bullets(chunk_txt)
        text_cleaned_non_ascii = clean_non_ascii_chars(text_cleaned_bullets)
        text_cleaned_bytes = bytes_string_to_string(text_cleaned_non_ascii, encoding="utf-8")

        text_cleaned_dashes = clean_dashes(text_cleaned_bytes)
        text_cleaned_whitespace = clean_extra_whitespace(text_cleaned_dashes)

        return clean(text_cleaned_whitespace, lowercase=True)

    def clean_chunks(self, chunks_lst: list) -> list:
        """
        Cleans a list of text chunks using the `clean_chunk` method.

        Parameters:
        - chunks_lst (list): A list of text chunks to be cleaned.

        Returns:
        - list: A list containing the cleaned text chunks.
        """
        return [self.clean_chunk(chunk) for chunk in chunks_lst]

    def generate_chunks(self):
        """
        Retrieves and cleans text chunks from the file specified during initialization.
        This method combines the functionality of getting and cleaning chunks.

        Returns:
        - list: A list of cleaned text chunks ready for further processing or analysis.
        """
        raw_chunks = self.extract_chunks()
        return self.clean_chunks(raw_chunks)

####### Class for mapping chunks to entities 


class GLiNEREntityExtractor():
    def __init__(self, model_path: str, entities_labels: List[str], entity_keys: Dict[str, str]):
        self.model_path = model_path
        self.entities_labels = entities_labels
        self.entity_keys = entity_keys
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        nltk.download('averaged_perceptron_tagger')
        nltk.download('omw-1.4')
        nltk.download('wordnet')

    def load(self):
        
        self.model = GLiNER.from_pretrained(self.model_path)

    def get_wordnet_pos(self, word: str) -> str:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def normalize_entity(self, entity: str) -> str:
        age_pattern = r'\d+\s+(week|month|year)s?\s+old'
        descriptive_pattern = r"(male|female|young|old|emaciated|large|small|adult|miniature|patients|subjects)"
        cleaned_entity = nltk.re.sub(f"{age_pattern}|{descriptive_pattern}", "", entity, flags=nltk.re.IGNORECASE).strip()
        return self.lemmatizer.lemmatize(cleaned_entity.lower(), self.get_wordnet_pos(cleaned_entity.lower()))

    def extract_entities(self, text: str) -> dict:
        result = self.model.predict_entities(text, self.entities_labels, threshold=0.5)
        extracted_entities = {key: [] for key in self.entity_keys.values()}
        
        for entity in result:
            if entity['label'] in self.entity_keys:
                key = self.entity_keys[entity['label']]
                normalized_entity = self.normalize_entity(entity['text'])
                extracted_entities[key].append(normalized_entity)

        return extracted_entities

    def process_chunks(self, chunks: List[str]) -> List[dict]:
        mapped_entities = []
        for chunk in chunks:
            chunk_text = chunk if isinstance(chunk, str) else chunk['chunk_text']
            entities = self.extract_entities(chunk_text)
            chunk_mapping = {key: list(set(values)) for key, values in entities.items()}
            chunk_mapping["chunk"] = chunk_text
            mapped_entities.append(chunk_mapping)
        
        return mapped_entities
    