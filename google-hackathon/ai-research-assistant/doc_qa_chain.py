import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_qa_model(context):

    # Set up the model
    generation_config = {
      "temperature": 1,
      "top_p": 0.95,
      "top_k": 0,
      "max_output_tokens": 8192,
    }

    safety_settings = [
      {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
      },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings)
    llm_context = [
        {
            "role" : "user",
            "parts" : [
                {
                    "text" : "Your name is Tin, and You are an AI Research assistant with expertise in simplifying and explaning the most complex AI/ML research papers. Answer the user's questions based on the context of this research paper he provided you with"
                },
                {
                  "text":" ".join(txt for txt in context)
                }
            ],
        },
        {
            "role" : "model",
            "parts" : [
                {
                    "text" : "Understood"
                }
            ]
        }
    ]
    convo = model.start_chat(history=llm_context)

    return convo




