from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import requests

DB_faiss_path="vectors/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

current_user_data = {}

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieve_user_data(email, password):
    """
    Retrieve user data using python requests
    """
    base_url = "https://wixstocle.pythonanywhere.com/"

    # Login
    login_url = base_url + "api/login/"
    login_data = {
        'email': email,
        'password': password
    }
    response = requests.post(login_url, data=login_data)
    user_token = response.json()['data']['token']

    headers = {
        'Authorization': f'Token {user_token}',
    }

    print(user_token)

    user_data = {}

    # Get user data
    user_data_url = base_url + "api/user/"+email
    response = requests.get(user_data_url, headers=headers)
    user_data["profile"] = response.json()['data']

    #Get user's savings
    savings_url = base_url + "api/saving"
    response = requests.get(savings_url, headers=headers)
    user_data["planned_savings"] = response.json()['data']

    #Get user's expenses
    expenses_url = base_url + "api/expense"
    response = requests.get(expenses_url, headers=headers)
    user_data["transactions"] = response.json()['data']

    return user_data
    

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    
    llm = CTransformers(
        #model = "llama-2-7b-chat.ggmlv3.q4_0.bin",
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_file = "llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        max_new_tokens = 1024,
        temperature = 0.5
    )
    return llm

#QA Model 
def qa_chat():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    user_profile_vector = embeddings.embed_documents(current_user_data)
    # user_vector = np.concatenate([user_profile_vector, savings_vector, expenses_vector], axis=0)
    db = FAISS.load_local(DB_faiss_path, embeddings)
    #db.add_vector(user_profile_vector)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
async def final_result(query):
    qa_result = qa_chat()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    global current_user_data
    chain = qa_chat()
    current_user_data = retrieve_user_data("tony@gmail.com", "user1234")
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = f"Hi {current_user_data['profile']['first_name']}, Welcome to Retire.AI Financial Assistant. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    global current_user_data
    print(current_user_data)
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    prompt_data = {
        "query": message.content,
        "context": current_user_data,
        "question": message.content
    }
    
    res = await chain.acall(prompt_data, callbacks=[cb])
    answer = "Is there anything else, I can help with?" #res["result"]

    #sources = res["source_documents"]

    # response = final_result(message.content, chain)  # Call final_result with the message content
    # print(response)
    # answer = response["result"]
    # sources = response["source_documents"]

    # if sources:
    #     answer += f"\nSources:" + str(sources)
    # else:
    #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()