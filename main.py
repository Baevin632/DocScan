
print("top of file")

from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import uvicorn 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS



from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

import os


from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("START")

app = FastAPI(title="DocScan-Rag")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("BEFORE EMBEDDINGS")

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
print("AFTER EMBEDDINGS")

@app.get("/test-chat")
def test_chat():
    try:
        response = llm.invoke("Say hello")
        return {"answer": response.content}
    except Exception as e:
        return {"error": str(e)}


print("BEFORE GEMINI")
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",  
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
print("AFTER GEMINI")
vector_store=None

class ChatRequest(BaseModel):
    message:str


    
@app.post("/upload")
async def upload_pdf(file: UploadFile = File()):
    global vector_store
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        
        os.remove(file_path)
        return {"message": f"✅ Successfully processed '{file.filename}' ({len(chunks)} chunks)"}
    
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

   
@app.post("/chat")
async def chat(request: ChatRequest):
    global vector_store
    if vector_store is None:
        return {"answer": "Please upload a document first."}
    try:
        
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.message)
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Strong, clean prompt
        prompt = f"""You are a clear, concise, and professional assistant.

        Answer the question using **only** the information from the provided context.

        Rules for answering:
        - Write in simple, natural, easy-to-read English.
        - Use short paragraphs. Do not write long blocks of text.
        - Use bullet points when listing things.
        - Never use **, *, -, #, or any markdown symbols.
        - Always mention the source when possible (e.g., "According to page 2...", "In the Experience section...", etc.).
        - If the answer is not in the document, say: "Sorry, this information is not available in the document."

        Context: {context}

        Question: {request.message}
        Answer:"""

        response = llm.invoke(prompt)
       
        return {"answer": response.text}
    
    
    

    

    except Exception as e:
        return {"answer": f"Error: {str(e)}"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=8000,reload=False)





