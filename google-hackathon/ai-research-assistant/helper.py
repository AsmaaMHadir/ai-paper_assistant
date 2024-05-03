from urlextract import URLExtract
import re

def extract_urls_from_section(sections):
    
    extractor = URLExtract()
    urls = []

    for section in sections:
        url = extractor.find_urls(section)
        urls.append(url) 
            
    return urls


def find_github_repo_urls_in_text(text):
    
    # Regular expression to match GitHub repository homepage URLs more flexibly
    github_repo_url_pattern = re.compile(r'https?://github\.com/[\w-]+/[\w-]+/?\b')
    
    # Search the text for matches
    github_repo_urls = github_repo_url_pattern.findall(text)
    
    # Remove duplicates and return the results
    return list(set(github_repo_urls))

import pandas as pd

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten the sublist
        else:
            flat_list.append(item)
    return flat_list

def html_to_markdown_table(html_content):

    # Read HTML tables into a list of DataFrame objects
    dfs = pd.read_html(html_content)
    
    # Convert each DataFrame to Markdown and print
    markdown_tables = []
    for df in dfs:
        markdown_tables.append(df.to_markdown(index=False))
    
    return markdown_tables