import random
from tqdm import tqdm
import pandas as pd
from langchain_community.chat_models import ChatHuggingFace
from langchain.prompts import ChatPromptTemplate
import datasets
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset
import evaluate

def evaluate_with_ragas(dataset: dict):
    dataset = Dataset.from_dict(dataset)
    result = evaluate(dataset=dataset,
                      metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    df = result.to_pandas()
    return df

def evaluate_with_bleu(answer,reference):
    bleu = evaluate.load('bleu')
    score_bleu = bleu.compute(predictions=[answer], references=reference)
    return score_bleu

def evaluate_with_rouge(answer,reference):
    rouge = evaluate.load('rouge')
    score_rouge = rouge.compute(predictions=[answer], references=reference)
    return score_rouge

def create_qa_pair(model,
                   n_generation:int,
                   docs):
    chat_model = ChatHuggingFace(llm=model)
    QA_generation_prompt = """
    Your task is to write a factoid question and an answer given a context.
    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
    This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

    Provide your answer as follows:

    Output:::
    Factoid question: (your factoid question)
    Answer: (your answer to the factoid question)

    Now here is the context.

    Context: {context}\n
    Output:::"""

    QA_generation_prompt = ChatPromptTemplate.from_template(QA_generation_prompt)
    QA_generation_agent = QA_generation_prompt | chat_model

    print(f"Generating {n_generation} QA couples...")
    outputs = []
    for context in tqdm(random.sample(docs, n_generation)):
        # Generate QA couple
        output_QA_couple = QA_generation_agent.invoke({"context": context.page_content}).content
        try:
            question = output_QA_couple.split("Factoid question: ")[1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[1]
            outputs.append(
                {
                    "context": context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": context.metadata["source"],
                }
            )
        except:
            continue
    return outputs

def eval_dataset_from_critique_agents(model,
                                      outputs:list):
    
    chat_model = ChatHuggingFace(llm=model)
    question_groundedness_critique_prompt = """
    You will be given a context and a question.
    Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
    Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating)
    Total rating: (your rating)

    Now here are the question and context.

    Question: {question}\n
    Context: {context}\n
    Answer::: """

    question_relevance_critique_prompt = """
    You will be given a question.
    Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
    Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating)
    Total rating: (your rating)

    Now here is the question.

    Question: {question}\n
    Answer::: """

    question_standalone_critique_prompt = """
    You will be given a question.
    Your task is to provide a 'total rating' representing how context-independant this question is.
    Give your answer on a scale of 1 to 5, where 1 means that the question only makes sense in a specific context, and 5 means that the question makes sense by itself.
    For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
    The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

    Provide your answer as follows:

    Answer:::
    Evaluation: (your rationale for the rating)
    Total rating: (your rating)

    Now here is the question.

    Question: {question}\n
    Answer::: """

    question_groundedness_critique_prompt = ChatPromptTemplate.from_template(question_groundedness_critique_prompt)
    question_groundedness_critique_agent = question_groundedness_critique_prompt | chat_model

    question_relevance_critique_prompt = ChatPromptTemplate.from_template(question_relevance_critique_prompt)
    question_relevance_critique_agent = question_relevance_critique_prompt | chat_model

    question_standalone_critique_prompt = ChatPromptTemplate.from_template(question_standalone_critique_prompt)
    question_standalone_critique_agent = question_standalone_critique_prompt | chat_model

    print("Generating critique for each QA couple...")
    for output in tqdm(outputs):
        # Critique the generated QA couple
        question_groundedness_evaluation = question_groundedness_critique_agent.invoke(
            {"context": output["context"], "question": output["question"]}
        ).content
        question_relevance_evaluation = question_relevance_critique_agent.invoke({"question": output["question"]}).content
        question_standalone_evaluation = question_standalone_critique_agent.invoke(
            {"question": output["question"]}
        ).content

        try:
            groundedness_score = int(question_groundedness_evaluation.split("Total rating: ")[1][0])
            groundedness_eval = question_groundedness_evaluation.split("Total rating: ")[0].split("Evaluation: ")[1]
            relevance_score = int(question_relevance_evaluation.split("Total rating: ")[1][0])
            relevance_eval = question_relevance_evaluation.split("Total rating: ")[0].split("Evaluation: ")[1]
            standalone_score = int(question_standalone_evaluation.split("Total rating: ")[1][0])
            standalone_eval = question_standalone_evaluation.split("Total rating: ")[0].split("Evaluation: ")[1]
            output.update(
                {
                    "groundedness_score": groundedness_score,
                    "groundedness_eval": groundedness_eval,
                    "relevance_score": relevance_score,
                    "relevance_eval": relevance_eval,
                    "standalone_score": standalone_score,
                    "standalone_eval": standalone_eval,
                }
            )
        except:
            continue
    generated_questions_df = pd.DataFrame.from_dict(outputs)
    generated_questions = generated_questions_df.loc[(generated_questions_df["groundedness_score"] >= 4)
                                                     & (generated_questions_df["relevance_score"] >= 4)
                                                     & (generated_questions["standalone_score"] >= 4)]
    eval_dataset = datasets.Dataset.from_pandas(generated_questions, split="train", preserve_index=False)
    return generated_questions_df, eval_dataset