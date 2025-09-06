import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# langgraph imports
from langgraph.graph import START, END, StateGraph
from typing import TypedDict


# set up streamlit
st.title("Research Pipeline Agent")

st.subheader("Welcome to the Research Pipeline Agent made with Streamlit, Ollama and LangGraph!")

class AgentState(TypedDict):
    question: str
    answer: str
    


# first
def ask_question(state: AgentState) -> AgentState:
    
    return state

# second
def agent_search(state: AgentState) -> AgentState:
    messages = [
        ("system", "You are a research assistant that helps users find info."),
        ("human", "{question}")
    ]
    chat = ChatPromptTemplate.from_messages(messages)
    llm = ChatOllama(model="llama3.1")
    chain = chat | llm | StrOutputParser()

    res = chain.invoke({"question": state["question"]})

    # Embed & store
    embeddings = OllamaEmbeddings(model="llama3.1")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text = text_splitter.split_text(res)


    Chroma.from_texts(
        texts=text,
        embedding=embeddings,
        persist_directory="db/chroma_store"
    )
    
    print("Data stored in ChromaDB")
    return state
                
    # third - display result
def get_result(state: AgentState) -> AgentState:
    embeding = OllamaEmbeddings(model="llama3.1")
    
    question = state["question"]
    
    vector_store = Chroma(embedding_function=embeding, persist_directory="db/chroma_store")
    
    res = vector_store.similarity_search(question, k=10)
    
    messages = [
        (
            "system", f"""You are ai research summarizer assistant that helps users summarize information.
                        
                        Here is the information: 
                        {[res.page_content for res in res]}
            """
        ),
        
        ("human", "{question}")
    ]
            
    llm = ChatOllama(model="llama3.1")
    
    chat = ChatPromptTemplate.from_messages(messages)
    
    chain = chat | llm | StrOutputParser()
    
    state["answer"] = chain.invoke({"question": question})
    
    return state
    
      
    
# define the state graph
graph = StateGraph(AgentState)

# add the nodes
graph.add_node("ask_question", ask_question)
graph.add_node("agent_search", agent_search)
graph.add_node("get_result", get_result)

# add the edges
graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "agent_search")
graph.add_edge("agent_search", "get_result")
graph.add_edge("get_result", END)

# compile the graph
compiled_graph = graph.compile()

# set the initial state


question = st.text_input("Ask a question:")
ask_btn = st.button("Ask")

if ask_btn and question:
    res = compiled_graph.invoke({"question": question, "answer": ""})
    st.write(res["answer"])

