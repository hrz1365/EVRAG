import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import DisplayConfig
from deepeval.test_case import LLMTestCase
from src.evaluation import get_gemini_model, get_metrics
import time
import os
import json


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
        question = row["input"]
        expected = row.get("expected_output", "")
        ctx_text = row.get("contexts", "")
        contexts = [ctx_text] if isinstance(ctx_text, str) and ctx_text else []

        # Get model answer + actual retrieved contexts
        answer, _ = rag.query(question)
        retriever = rag.vector_store.retriever()
        results = retriever.invoke(question)
        retrieved_contexts = [doc.page_content for doc in results]

        # Build test case
        case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected,
        )

        # Attach contexts for newer DeepEval versions
        if hasattr(case, "context"):
            case.context = contexts
            case.retrieval_context = retrieved_contexts
        else:
            case.additional_metadata = {
                "contexts": contexts,
                "retrieval_contexts": retrieved_contexts,
            }

        test_cases.append(case)
    return test_cases


def run_evaluation(rag, eval_data_path):
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
    df = pd.read_excel(eval_data_path)

    # Build test cases
    test_cases = build_test_cases(rag, df)

    # Get Gemini model and metrics
    gemini_model = get_gemini_model()
    metrics = get_metrics(model=gemini_model)

    # Run evaluation
    results = []
    for i, test_case in enumerate(test_cases):
        row = {
            "case_id": i,
            "input": test_case.input,
            "model_output": test_case.actual_output,
            "ground_truth": test_case.expected_output,
        }
        for metric in metrics:
            result = evaluate(
                test_cases=[test_case], metrics=[metric], display_config=disConfig
            )
            metric_name = result.test_results[0].metrics_data[0].name
            score = result.test_results[0].metrics_data[0].score
            reason = result.test_results[0].metrics_data[0].reason
            row[f"{metric_name}_score"] = f"{score:.3f}"
            row[f"{metric_name}_reason"] = reason
            time.sleep(20)  # To avoid rate limiting
        with open(f"{os.path.dirname(eval_data_path)}/case_{i}.json", "a") as f:
            json.dump(row, f, indent=4)
        print(f"Json file containing evaluation for case {i} written.")
        results.append(row)
    return results
