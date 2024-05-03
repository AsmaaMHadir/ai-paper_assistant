import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
from config import main_prompt, table_summarization_prompt 

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")


def get_summary(context, user_backgound=""):

    response = model.generate_content(
        [main_prompt]
        + ["Here is the research paper"]
        + context
        + ["Here is the user's background"]
        + [user_backgound],
        request_options={"timeout": 600},
    )

    return response.text


def get_repo_analysis(repo_url, context):

    response = model.generate_content(
        [
            "# Here is a github repository containing code relevant for a research paper the user is reading and equire assistance in understandng"
        ]
        + [repo_url]
        + ["# This is the paper"]
        + [context]
        + [
            "[END]\n\nPlease explain the code in the repository and provide a detailed guide showing the user how this code may help them to get started in implementing the research paper"
        ]
    )

    return response

def get_tables_summaries(table_markdown, context):
    
    response = model.generate_content(
        [table_summarization_prompt] +
        ['# Here is a table extracted from the paper']+
        table_markdown +
        ["# This is the paper:"] +
        context , 
        request_options={"timeout": 600}
    )

    return response.text
    