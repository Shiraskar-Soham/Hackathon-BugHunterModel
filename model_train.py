from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Set model path where the model will be saved and loaded from
model_dir = "/home/akarsh/Desktop/saved_model2"  # Update this path

# Check if the model already exists, to load instead of training
if os.path.exists(model_dir):
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

# Create the Hugging Face pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,  # Increase as needed
    max_new_tokens=150  # Adjust as necessary
)

# Load your documents
loader = TextLoader("/home/akarsh/Desktop/Hackathon/cleaned_java_dataset.txt")
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create a FAISS vectorstore
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# Define the template for your prompt
template = (
    "You are an assistant for code generation tasks. "
    "Use the following pieces of retrieved code snippets to generate code based on the given prompt. "
    "If the context provided does not fit the prompt, say that you don't have a relevant example. "
    "Keep the generated code concise and in line with the retrieved examples.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        ("human", "{input}")
    ]
)

# Create the language model chain and retrieval chain
llm = HuggingFacePipeline(pipeline=hf_pipeline)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Provide input for the trained model to generate code
response = rag_chain.invoke({"input": "convert date string to epoch milli second"})
print(response["answer"])
