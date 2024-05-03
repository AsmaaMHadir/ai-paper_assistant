import chainlit as cl
from process_pdf import read_file_layout, get_doc_sections, get_sections_tables_text
from chainlit import AskUserMessage, Message, on_chat_start
from chat import get_summary, get_tables_summaries
from helper import html_to_markdown_table, flatten_list
from doc_qa_chain import get_gemini_qa_model
from google.generativeai.generative_models import ChatSession
import asyncio 

@cl.on_chat_start
async def on_start():
    user_background = await AskUserMessage(content="I would like to tailor my answer to best suit your background. Could you tell me a bit about your professional or academic background?", timeout=10).send()

    
    if user_background:
        
        cl.user_session.set("user_background",user_background["output"]) 
        
        await cl.Message(
            content="Thank you! now please upload the research paper below"
        ).send()
        
        files = None

        # Wait for the user to upload a file
        while files == None:
            files = await cl.AskFileMessage(
                content="PDF Upload", accept=["application/pdf"]
            ).send()

        await cl.Message(
            content=f"Processing the doc.."
        ).send()
        text_file = files[0].path # file path

        paper_doc=await cl.make_async(read_file_layout)(text_file)

        sections = await cl.make_async(get_doc_sections)(paper_doc) # got sections
        await cl.Message(
            content=f"Extracted the document's sections.."
        ).send()


        
        doc_tables,doc_context = await cl.make_async(get_sections_tables_text)(sections)
        convo_model = cl.make_async(get_gemini_qa_model)(doc_context)
        cl.user_session.set("model",convo_model) 
        cl.user_session.set("paper_context",doc_context) 
        
        doc_tables = flatten_list(doc_tables)
        await cl.Message(
            content="Analyzing the paper..."
        ).send()
        
        paper_tables = []
        for table in doc_tables:
            table = table.to_html()
            table = html_to_markdown_table(table)
            paper_tables.extend(table)

        table_summary = await cl.make_async(get_tables_summaries)(paper_tables,doc_context)
        
        user_info = cl.user_session.get("user_background")
        summary = await cl.make_async(get_summary)(doc_context,user_info)
        await cl.Message(
            content=summary + "\n\n" + table_summary
        ).send()
        await cl.Message(
            content= "Ask me more questions about the paper!"
        ).send()
     
        
@cl.on_message
async def on_message(msg: cl.Message):
    query = msg.content
    context = cl.user_session.get("paper_context")
    
    if context:
        convo_model = cl.user_session.get("model")
        if asyncio.iscoroutine(convo_model):
            convo_model = await convo_model

        convo_model.send_message(query)

        ai_answer =  convo_model.last.text
        await cl.Message(
            content= ai_answer
        ).send()
        
        await msg.update()