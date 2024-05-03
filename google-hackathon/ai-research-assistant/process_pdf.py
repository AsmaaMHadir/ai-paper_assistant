from llmsherpa.readers import LayoutPDFReader
import re

# extract pdf dource files and read them as 
def read_file_layout(file_name):
    llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    pdf_reader = LayoutPDFReader(llmsherpa_api_url)
    doc = pdf_reader.read_pdf(file_name)
    return doc



def get_chunk_paths(chunk):
    pattern = re.compile(r'^([^\n]+)$', re.MULTILINE)
    matches = pattern.findall(chunk)
    return matches[0]


def get_chunks_details(chunk,section_title):
    
    chunk_details = []
    for chunk in chunk:
        chunk_text = chunk.to_context_text(include_section_info=True)
        chunk_path =  get_chunk_paths(chunk_text)
        chunk_page_num = chunk.page_idx
        chunk_details.append({"text":chunk.to_text(),
                                "title":chunk_path,
                                "page":chunk_page_num,
                                "source_doc":section_title})
    return chunk_details

def get_doc_sections(doc):
    
    sections_details = []
    sections = doc.sections()
    for section in sections:
        section_text = section.to_text(include_children=True,recurse=True)
        section_title = section.title
        section_chunks = section.chunks()
        section_tables = section.tables()
        section_details = {"text":section_text,
                           "title":section_title,
                           "chunks":get_chunks_details(section_chunks,section_title),
                           "tables":section_tables}
        sections_details.append(section_details)
    return sections_details


def get_sections_chunks(sections):

    chunks = []
    
    for section in sections:
        for chunk in section["chunks"]:
            chunks.extend(chunk["text"])
            
    return chunks 

def get_sections_tables_text(sections):
    context = []
    tables = []

    for sec in sections:
        context.append(sec["text"])
        tables.extend(sec["tables"])
        

    return tables,context


