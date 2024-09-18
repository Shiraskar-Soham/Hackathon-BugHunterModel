from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Create FastAPI app
app = FastAPI()

# Set model path where the model will be saved and loaded from
model_dir = "/home/soham/Desktop/Hackathon/Python/saved_model"   # Update this path

# Load or train the model
if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, 'config.json')):
    print("Loading saved model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
else:
    print("Training model and saving it...")
    model_name = "Salesforce/codegen-350M-multi"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save the model and tokenizer for future use
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

# Create Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    max_new_tokens=150,
    truncation=True
)

# Load your documents
loader = TextLoader("/home/soham/Desktop/Hackathon/Python/cleaned_java_dataset.txt")
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create a FAISS vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Explicit model name
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Define the template for your prompt
template = (
    "You are an assistant for code generation tasks. "
    "Use the following pieces of retrieved code snippets to generate code based on the given prompt. "
    "If the context provided does not fit the prompt, say that you don't have a relevant example. "
    "Keep the generated code concise and in line with the retrieved examples.\n\n"
    "Context: {context}\n\n"
)

# Define the prompt structure for the LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}"),
    ]
)

# Create the language model chain and retrieval chain
llm = HuggingFacePipeline(pipeline=hf_pipeline)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Define request body schema
class RequestInput(BaseModel):
    input: str

# Helper function to format the code properly
def format_code_block(code: str) -> str:
    # Replace '\n' with actual newlines
    formatted_code = code.replace("\\n", "\n")
    
    # Optional: Remove extra 'Human:' or 'System:' tags if present
    formatted_code = formatted_code.replace("Human:", "").replace("System:", "")
    
    # Strip leading/trailing whitespace
    return formatted_code.strip()

# Define the API endpoint
@app.post("/generate_code")
async def generate_code(request: RequestInput):
    try:
        # Invoke the model with the input provided
        response = rag_chain.invoke({"input": request.input})
        
        # Get the context documents
        context_docs = response.get("context", [])
        print(context_docs)
        
        # Concatenate all page_content
        concatenated_code = "\n\n 1234 ".join(doc.page_content for doc in context_docs)
        
        # Format the concatenated code
        formatted_code = format_code_block(concatenated_code)
        
        # Return the formatted code
        return {formatted_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))