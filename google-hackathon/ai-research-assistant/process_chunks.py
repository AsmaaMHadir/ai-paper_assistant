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
    