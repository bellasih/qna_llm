import os
import torch
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, pipeline, AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def define_embedding(model_path:str,
                     model_kwargs:dict,
                     encode_kwargs:dict):
    
    '''
    model_path: define the path to the pre-trained model you want to use
    model_kwargs: dictionary with model configuration options, specifying to use the CPU/GPU for computations
    encode_kwargs: dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    '''

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    return HuggingFaceEmbeddings(
        model_name=model_path,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

def load_dataset(file_path:str, 
                 source_column:str,
                 type_file="csv",
                 chunk_size=300,
                 chunk_overlap=0):
    '''
    file_path: define the path of the dataset that you want to use
    source_column: 
    type_file: 
    chunk_size:
    chunk_overlap:
    '''
    if type_file == "csv":
        loader = CSVLoader(file_path=file_path, source_column=source_column)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(data)
        for doc in docs:
            doc.page_content = doc.page_content.split("\nclean_review_text: ")[-1]
        return docs
    elif type_file == "dataframe":
        loader = DataFrameLoader(pd.read_csv(file_path), page_content_column="clean_review_text")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(data)
        return docs
    elif type_file == "txt":
        loader = TextLoader(file_path, encoding = 'UTF-8')
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(data)
        return docs
    else:
         raise ValueError('The extension of the source should be .csv or dataframe')

def load_faiss_db(db_path,
                  embeddings=None,
                  docs=None, 
                  is_visualize=False):
    '''
    db_path: path that indicate index db
    embeddings: 
    '''
    if os.path.isdir(db_path):
        db = FAISS.load_local(db_path, embeddings=embeddings, distance_strategy=DistanceStrategy.COSINE)
        return db
    else:
        try:
            if not is_visualize:
                db = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
            else:
                db = None
                with tqdm(total=len(docs), desc="Ingesting documents") as pbar:
                    for d in docs:
                        if db:
                            db.add_documents([d])
                        else:
                            db = FAISS.from_documents([d], embeddings, distance_strategy=DistanceStrategy.COSINE)
                        pbar.update(1)  
            db.save_local(db_path)
            return db
        except Exception as error:
            print('Caught this error: ' + repr(error))

def define_llm(model_name:str,
               use_4bit:bool, disable_exllama=None):
    """
    """
    model_config = AutoConfig.from_pretrained(
        model_name,
    )
    
    if disable_exllama != None:
        model_config.quantization_config["disable_exllama"] = False
        model_config.quantization_config["exllama_config"] = {"version":2}

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if use_4bit:
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_nested_quant,
        )

        if compute_dtype == torch.float16 and use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
                
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
        )
    elif (not use_4bit) and device == torch.device("cuda"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
            )
        except:
             model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="cuda:0", 
                config=model_config
            )
    else:
        raise ValueError('Available device is cpu')
    return model, tokenizer

def create_response_chain(model, tokenizer, max_new_tokens, temperature=0.2,
                          repetition_penalty=1.1, return_full_text=True,
                          ):
    """
    """
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=return_full_text,
        max_new_tokens=max_new_tokens,
    )

    prompt_template = """
    ### [INST] 
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Answer the question based on OUR APPLICATION knowledge from the context. OUR APPLICATION is SPOTIFY. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Here is context to help:

    {context}

    ### QUESTION:
    {question} 

    [/INST]
    """

    chosen_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Create llm chain 
    return LLMChain(llm=chosen_llm, prompt=prompt)

def answer_with_rag(question: str,
                    llm_chain: LLMChain,
                    db,
                    k_retrieve:int=4):
    retriever = db.as_retriever(search_type="similarity_score_threshold",
                                search_kwargs={"k": k_retrieve,
                                              "score_threshold":0.2})
    rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
                 | llm_chain)
    result = rag_chain.invoke(question)
    relevant_docs = [doc.page_content for doc in result["context"]]
    return result["text"], relevant_docs