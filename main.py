from preprocess import PreprocessDataFrame
from rag import *
from evaluation import *
import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def download_file():
    """
    Function for download the dataset from kaggle 
    """
    cmd = ["mkdir ~/.kaggle",
           "mkdir dataset",
           "cp kaggle.json ~/.kaggle/",
           "chmod 600 ~/.kaggle/kaggle.json",
           "kaggle datasets download 'bwandowando/3-4-million-spotify-google-store-reviews'",
           "unzip 3-4-million-spotify-google-store-reviews.zip -d dataset"]
    for c in cmd:
        try:
            os.system(c)
        except:
            continue

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_name', type=str, default='faiss_index', help='vectorstore')
    parser.add_argument('--question', type=str, default='')
    parser.add_argument('--benchmark_data', type=str, default='')
    parser.add_argument('--verctorstore_name', type=str, default='faiss_index_all-mpnet-base-v2_cs500_co50_1000')
    parser.add_argument('--dataset_source', type=str, default='dataset/clean_spotify_dataset.csv', help='filename of data (.csv)')
    parser.add_argument('--download_dataset', type=int, default=0, help='0: not download from kaggle, 1: download from kaggle')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='')
    parser.add_argument('--embedding_name', type=str, default='sentence-transformers/all-mpnet-base-v2', help='available LLMs')
    parser.add_argument('--normalize_embeddings', type=bool, default=True, help='do quantization')
    parser.add_argument('--is_quantize', type=bool, default=True, help='do quantization')
    parser.add_argument('--is_preprocess', type=bool, default=False, help='do preprocess')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--chunk_overlap', type=int, default=50)
    opt = parser.parse_args()
    return vars(opt)

def main(opt):
    if opt["download_dataset"]:
        download_file()

    #do preprocess and save the clean dataset
    if opt["is_preprocess"]:
        df = PreprocessDataFrame("dataset/SPOTIFY_REVIEWS.csv").cleaningDataFrame()
        temp = df[["review_id","clean_review_text", "pseudo_author_id"]].reset_index(drop=True)
        temp.to_csv("dataset/clean_spotify_dataset.csv", index=False)
    
    #load the clean dataframe into docs format
    docs = load_dataset(opt["dataset_source"], "review_id", 
                        type_file="dataframe",
                        chunk_size=opt["chunk_size"], 
                        chunk_overlap=opt["chunk_overlap"])
    
    model_kwargs = {'device': opt["device"]}
    encode_kwargs = {'normalize_embeddings': opt["normalize_embedding"]}
    embeddings = define_embedding(opt["embedding_name"], model_kwargs, encode_kwargs)

    db = load_faiss_db(opt["vectorestore_name"], embeddings=embeddings)
    
    model, tokenizer = define_llm(opt["model_name"], True)
    llm_chain = create_response_chain(model, tokenizer, 256, temperature=0.1)

    try:
        if opt["benchmark_data"] != '':
            benchmark_df = pd.read_csv(opt["benchmark_data"])

            question = benchmark_df["questions"].tolist()
            questions, answers, contexts, ground_truth, scores_bleu = [], [], [], [], []
            for i,q in enumerate(tqdm(question)):
                answer, relevant_docs = answer_with_rag(q, llm_chain, db, k_retrieve=10)
                score = evaluate_with_bleu(answer, benchmark_df["answers"][i])
                questions.append(q)
                answers.append(answer)
                contexts.append(relevant_docs)
                ground_truth.append(benchmark_df["answers"][i])
                scores_bleu.append(score)

            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truth
            }
            try:
                eval_df = evaluate_with_ragas(data)
            except:
                data["score_bleu"] = scores_bleu
                eval_df = pd.DataFrame(data)

            eval_df.to_csv("eval_{}_{}.csv".format(opt["model_name"], opt["benchmark_data"].split(".")[0]),
                            index=False)
        else:
            answer, relevant_docs = answer_with_rag(question, llm_chain, db, k_retrieve=10)
            print(answer)
            print(relevant_docs)
    except Exception as error:
        print('Caught this error: ' + repr(error))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
