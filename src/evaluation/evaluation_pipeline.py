import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.test_case import LLMTestCase
from src.evaluation import get_gemini_model, get_metrics
import time
import os
import json

import numpy as np
from ragas.metrics import SemanticSimilarity
from ragas.metrics.collections import NonLLMStringSimilarity, DistanceMeasure

from ragas.embeddings import embedding_factory
import Levenshtein



def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / max(len(relevant_ids), 1)

def embedding_similarity(output,expected_output):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_pred = model.encode(output, convert_to_tensor=True)
    emb_ref  = model.encode(expected_output, convert_to_tensor=True)
    cosine_score = util.cos_sim(emb_pred, emb_ref).item()
    return cosine_score


def rough_levenshtein_similarity(output,expected_output):
    distance = Levenshtein.distance(output.lower(), expected_output.lower())
    max_len = max(len(expected_output), len(output))
    return 1 - distance / max_len

def rough_f1_similarity(putput,expected_output):
    pred_tokens = set(putput.lower().split())
    ref_tokens = set(expected_output.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



def build_test_cases(rag, df):
    """
    Build a list of test cases for evaluating a Retrieval-Augmented Generation (RAG) model.

    Args:
        rag: An instance of the RAG model, which provides a `query` method to generate answers
             and a `vector_store.retriever().get_relevant_documents` method to retrieve relevant contexts.
        df (pd.DataFrame): A DataFrame containing the test data. Each row should include:
            - "input": The input query for the model.
            - "expected_output" (optional): The expected output for the query.
            - "contexts" (optional): Predefined contexts to be used for the test case.

    Returns:
        list: A list of `LLMTestCase` objects, each representing a test case with
          the following attributes:
            - `input`: The input query.
            - `actual_output`: The model's generated answer.
            - `expected_output`: The expected output for the query.
            - `contexts` (if supported): The predefined contexts for the test case.
            - `retrieval_contexts` (if supported): The contexts retrieved by the model.
            - `additional_metadata` (if `contexts` and `retrieval_contexts`
              are not directly supported):
              A dictionary containing the predefined and retrieved contexts.
    """
    test_cases = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = row.get("answer", "")
        context_groundtruth = row.get("context", "")
        # title = row.get("filename", "")
        
        
        # Get model answer + actual retrieved contexts
        # answer, context = rag.query(question)
        # contexts = json.loads(context)
        # retrieved_contexts = [c.content for c in contexts]
        # retrieval_doc = [c.title for c in contexts]
        answer=""
        retriever = rag.vector_store.retriever()
        results = retriever.invoke(question)
        
        retrieved_contexts = [doc.page_content for doc in results]
        retrieval_doc=[doc.metadata.get('title')for doc in results]
        

        # Build test case
        case_answer = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected,
            retrieval_context=retrieved_contexts,
            metadata={"relevant_context": context_groundtruth} # store ground truth here
        )
        case_contexts=[]
        for retrieved_context in retrieved_contexts: 
            case_context = LLMTestCase(
                input=question,
                actual_output=context_groundtruth,
                expected_output=retrieved_context,
            )
            case_contexts.append(case_context)
        
        test_cases.append([case_answer, case_contexts])
    return test_cases


def run_evaluation(rag, eval_data_path, output_folder, non_LLM_metric_only=True):
    """
    Executes the evaluation process for the given RAG (Retrieval-Augmented Generation)
      model.

    Args:
        rag: The RAG model instance to be evaluated.
        eval_data_path (str): The file path to the evaluation data in excel format.

    Returns:
        dict: A dictionary containing the evaluation results, including detailed metrics.

    This function performs the following steps:
    1. Loads the evaluation data from the specified CSV file.
    2. Builds test cases using the provided RAG model and the loaded data.
    3. Retrieves the Gemini model and associated evaluation metrics.
    4. Runs the evaluation process using the test cases and metrics.
    5. Returns the evaluation results, including detailed information.
    """

    disConfig = DisplayConfig(
        show_indicator=False, print_results=False, verbose_mode=False
    )
    # Load evaluation data
    try:
        df = pd.read_csv(eval_data_path)
    except:
        df = pd.read_excel(eval_data_path)

    # Build test cases
    test_cases = build_test_cases(rag, df)

    # Get Gemini model and metrics
    gemini_model = get_gemini_model()
    metrics = get_metrics(model=gemini_model)

    # Run evaluation
    results = []
    for i, (test_case_answer,test_case_contexts) in enumerate(test_cases):
        row = {
            "case_id": i,
            "input": test_case_answer.input,
            "model_output": test_case_answer.actual_output,
            "ground_truth": test_case_answer.expected_output,
            "context": test_case_contexts[0].actual_output,
            "retrieved_context": test_case_answer.retrieval_context
        }
        embedSim=[embedding_similarity(c,row["context"]) for c in  row["retrieved_context"]]
        rough_levenshtein_similarity_list= [rough_levenshtein_similarity(c,row["context"])  for c in  row["retrieved_context"]]
        rough_f1_similarity_list=[ rough_f1_similarity(c,row["context"])  for c in  row["retrieved_context"]]
        for j in range(1,len(row["retrieved_context"])+1):
            row['EmbedSimilarity_max_'+str(j)]=max(embedSim[:j])
            row['rough_levenshtein_similarity_max_'+str(j)]=max(rough_levenshtein_similarity_list[:j])
            row['rough_f1_similarity_max_'+str(j)]=max(rough_f1_similarity_list[:j])
        print(row)
        print(non_LLM_metric_only)
        print('-------------------------')
        #rough_f1_similarity
        #rough_levenshtein_similarity(output,expected_output)
        if non_LLM_metric_only:
            results.append(row)
            print(f"Json file containing evaluation for case {i}, {output_folder}/case_{i}.json written.")
            with open(f"{output_folder}/case_{i}.json", "a") as f:
                json.dump(row, f, indent=4)
            continue
            
        
            
        for metric in metrics:
            result = evaluate(
                test_cases=[test_case_answer], metrics=[metric], display_config=disConfig
            )
            metric_name = result.test_results[0].metrics_data[0].name
            score = result.test_results[0].metrics_data[0].score
            reason = result.test_results[0].metrics_data[0].reason
            row[f"{metric_name}_score"] = f"{score:.3f}"
            row[f"{metric_name}_reason"] = reason

            if "similarity" in metric_name.lower():
                sim_max=0
                metric_name = result.test_results[0].metrics_data[0].name
                for j,test_case_context in enumerate(test_case_contexts):
                    result = evaluate(
                        test_cases=[test_case_context], metrics=[metric], display_config=disConfig
                    )
                    score = result.test_results[0].metrics_data[0].score
                    reason = result.test_results[0].metrics_data[0].reason
                    row[f"context_{metric_name}_score_{j}"] = f"{score:.3f}"
                    row[f"context_{metric_name}_reason_{j}"] = reason
                    sim_max=max(sim,score )
                row[f"context_max_{metric_name}"]= sim_max
                time.sleep(20) 
                
                


            time.sleep(20)  # To avoid rate limiting
        with open(f"{os.path.dirname(eval_data_path)}/case_{i}.json", "a") as f:
            json.dump(row, f, indent=4)
        print(f"Json file containing evaluation for case {i} written.")
        results.append(row)
    return results
