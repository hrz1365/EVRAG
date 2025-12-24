import argparse
import os
from pathlib import Path
import pandas as pd
from src.pipelines.rag_pipeline import RAGPipeline
from src.evaluation.evaluation_pipeline import run_evaluation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the RAG Pipeline Evaluation or DeepEval evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["query", "validate"],
        default="query",
        help="Mode to run the script in: 'query' for querying the RAG pipeline, "
        "'validate' for running DeepEval evaluation.",
    )
    parser.add_argument(
        "--pdf", type=str, default="", help="Path to the knowledge source PDf."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to ask in query mode (use quotes).",
    )
    parser.add_argument(
        "--validation",
        type=str,
        default="data/validation_set.xlsx",
        help="Path to validation Excel file.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/validation_results.xlsx",
        help="Path to save evaluation results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = Path(args.pdf)
    validation_path = Path(args.validation)
    report_path = Path(args.report)

    print("Initializing RAG Pipeline...")
    rag = RAGPipeline(pdf_path=pdf_path)

    print("Building index from PDF...")
    rag.build_index()
    print("Index built successfully.")

    # Query mode
    if args.mode == "query":
        if args.query is None:
            print("Please provide a query using the --query argument.")
            return

        print(f"Querying RAG Pipeline with question: {args.query}")
        answer, _ = rag.query(args.query)

        print("Answer:")
        print(answer)

    # Validation mode
    elif args.mode == "validate":
        print("Running DeepEval evaluation...")
        results = run_evaluation(rag, validation_path)

        print(f"Saving evaluation report to {report_path}...")
        os.makedirs(report_path.parent, exist_ok=True)
        df_results = pd.DataFrame(results)
        df_results.to_excel(report_path, index=False)

        print("Evaluation report saved successfully.")


if __name__ == "__main__":
    main()
