import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.retrievers import MultiVectorRetriever
from typing import List
import warnings
import pinecone

warnings.filterwarnings("ignore")

load_dotenv()

# Initialize API keys and models
openai_api_key = st.secrets["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))
print("Economics Index Name:", os.getenv("INDEX_INST_IKONOMIKA"))
print("Innovations Index Name:", os.getenv("INDEX_INOVATIONS"))



# Initialize vector stores for both subjects
inovations_index_name = os.getenv("INDEX_INOVATIONS")
economics_index_name = os.getenv("INDEX_INST_IKONOMIKA")

inovations_vectorstore = PineconeVectorStore(index_name=inovations_index_name, embedding=embeddings)
economics_vectorstore = PineconeVectorStore(index_name=economics_index_name, embedding=embeddings)

class MultiSubjectRetriever:
    def __init__(self, vectorstores: List[PineconeVectorStore]):
        self.vectorstores = vectorstores
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for store in self.vectorstores:
            docs = store.as_retriever().get_relevant_documents(query)
            all_docs.extend(docs)
        return all_docs

def get_subject_from_query(query: str) -> tuple[str, list[PineconeVectorStore]]:
    """Determine which subject(s) the query is about and return relevant vectorstores."""
    inovations_docs = inovations_vectorstore.as_retriever().get_relevant_documents(query)
    econ_docs = economics_vectorstore.as_retriever().get_relevant_documents(query)
    
    if inovations_docs and econ_docs:
        return "both", [inovations_vectorstore, economics_vectorstore]
    elif inovations_docs:
        return "inovations", [inovations_vectorstore]
    elif econ_docs:
        return "economics", [economics_vectorstore]
    else:
        return "unknown", []

def get_response(query: str, vectorstores: list[PineconeVectorStore]) -> str:
    """Get response from specified vectorstores."""
    # Create a multi-subject retriever
    retriever = MultiSubjectRetriever(vectorstores)
    
    # Get documents
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "Не намирам релевантна информация за този въпрос в учебниците."

    # Create combined instruction
    combined_instruction = """
    Ти си студент и разполагаш с учебници по иновации и институционална икономика. 
    Отговори на въпроса, базирайки се само на информацията от тези учебници. 
    Отговорът трябва да бъде на български език, точен и изчерпателен.
    
    Въпрос:
    """

    full_query = f"{combined_instruction}\n{query}"
    
    # Create the chain
    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    chain = create_stuff_documents_chain(llm, prompt)
    
    # Generate response
    result = chain.invoke({
        "context": docs,
        "input": full_query
    })
    
    return result

st.set_page_config(page_title="Inovations and InstitutionalEconomics Chatbot", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .chat-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 10px;
        max-height: 500px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
    }
    .message {
        margin: 10px 0;
        padding: 10px;
        border-radius: 10px;
        width: fit-content;
    }
    .user-message {
        background-color: #e0f7fa;
        text-align: left;
    }
    .bot-message {
        background-color: #e1bee7;
        text-align: left;
    }
    .message p {
        margin: 0;
    }
    .user-icon, .bot-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }
    .chat-input {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        background-color: white;
        border-top: 1px solid #ccc;
    }
    .chat-box {
        display: flex;
        align-items: center;
    }
    .input-area {
        display: flex;
        justify-content: center;
        padding-bottom: 60px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📚 inovations & Economics Chatbot")
st.write("Ask questions related to 'иновации' and 'институционална икономика'.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Въведете вашия въпрос:", "")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    with st.spinner("Анализирам въпроса..."):
        subject, relevant_stores = get_subject_from_query(user_input)
        
        if relevant_stores:
            final_answer = get_response(user_input, relevant_stores)
        else:
            final_answer = "Не намирам релевантна информация за този въпрос в учебниците."

        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Bot", final_answer))

# Display chat history
st.write('<div class="chat-container">', unsafe_allow_html=True)

if st.session_state.chat_history:
    for role, message in st.session_state.chat_history:
        if role == "User":
            st.write(f"""
            <div class="chat-box">
                <img src="https://www.iconpacks.net/icons/2/free-user-icon-3296-thumb.png" class="user-icon"/>
                <div class="message user-message">
                    <p><strong>User:</strong> {message}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.write(f"""
            <div class="chat-box">
                <img src="https://img.icons8.com/ios-filled/50/000000/chatbot.png" class="bot-icon"/>
                <div class="message bot-message">
                    <p><strong>Bot:</strong> {message}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.write('</div>', unsafe_allow_html=True)
