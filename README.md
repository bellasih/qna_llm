# qna_llm

This repository contains code implementations for Retrieval-Augmented Generation (RAG).

## How to Use
### Requirements:
Main libraries and dependencies can be downloaded by executing this command in your terminal:
```
pip install -r requirement.txt
```
### Run The Program
1. To run the program, simply by re-running the available notebook: `Run_LangChain.ipynb` for creating vectorstore and evaluating the performance of LLMs after RAG and `Run_Streamlit_GoogleColab.ipynb` for running the apps in the background
2. If you want to run locally for creating vectorstore and evaluating the performance of LLMs after RAG, you can execute this command (make sure you check the args which are supported):
```
python main.py
```
3. Meanwhile for running the app, you can execute this command:
```
python index.py
```

## Caution
Some results perhaps tend to be bias due to the lack of the GPU computation for handling million of data. I will figure out how to improve the performance soon.