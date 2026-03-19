from deepeval.models import GeminiModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    # Contextual Recall,
    GEval, 
    # EmbeddingSimilarityMetric,
    # F1ScoreMetric,
    # RoughRecallAtNMetric
)
from deepeval.test_case import LLMTestCaseParams
from dotenv import load_dotenv
import os



load_dotenv()


def get_gemini_model():
    """
    Initializes and returns a GeminiModel instance.

    This function retrieves the API key for the Gemini model from the
    environment variable `GEMINI_API_KEY`. If the environment variable
    is not set, it raises a ValueError. The function then creates and
    returns a GeminiModel instance configured with the specified model
    name, API key, and temperature.

    Returns:
        GeminiModel: An instance of the GeminiModel class configured
        with the specified parameters.

    Raises:
        ValueError: If the `GEMINI_API_KEY` environment variable is not set.
    """
    api_key = os.getenv("GeminiRoya")
    if not api_key:
        raise ValueError("Gemini environment variable not set.")
    model = GeminiModel(model_name="gemini-2.0-flash", api_key=api_key, temperature=0)
    return model


def get_custom_metrics(model=None):
    """
    Generate a list of custom evaluation metrics for assessing the performance of a model.

    This function creates two evaluation metrics:
    1. Factual Correctness: Evaluates whether the actual output is factually correct
       based on the expected output.
    2. Semantic Similarity: Assesses how similar the actual output is to the expected
       output in terms of meaning and coverage.

    If no model is provided, a default model is retrieved using `get_gemini_model()`.

    Args:
        model (Optional): The model to be used for evaluation. If None, a default
                          model is used.

    Returns:
        list: A list containing two GEval metric objects: `correctness_metric` and
              `similarity_metric`.
    """
    if model is None:
        model = get_gemini_model()

    correctness_metric = GEval(
        name="Factual Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_steps=[
            "Check whether any facts in the actual output contradict facts in the expected output.",
            "Heavily penalize omissions of important details.",
            "Vague or stylistic differences are fine; focus on factual accuracy.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
    )

    similarity_metric = GEval(
        name="Semantic Similarity",
        criteria="Determine how similar the actual output is to the expected output in meaning and coverage.",
        evaluation_steps=[
            "Focus on whether both outputs express the same meaning.",
            "Do not penalize if the actual output adds harmless clarifications (e.g., units, nouns, or context words) that do not change the meaning.",
            "Only penalize if the actual output introduces incorrect or contradictory information.",
            "Do not require exact word overlap — prefer semantic equivalence.",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=model,
    )

    return [correctness_metric, similarity_metric]


def get_metrics(model=None):
    """
    Retrieves a list of evaluation metrics for a given model.

    Args:
        model (optional): The model for which the metrics are to be calculated.
            If not provided, a default model is retrieved using `get_gemini_model()`.
        include_custom_metrics (bool, optional): Whether to include custom metrics
            in the returned list. Defaults to True.

    Returns:
        list: A list of metric objects, including base metrics and optionally
        custom metrics if `include_custom_metrics` is True.
    """
    if model is None:
        model = get_gemini_model()

    custom_metrics = get_custom_metrics(model=model)
    LLM_metrics = [
        AnswerRelevancyMetric(model=model, verbose_mode=False),
        FaithfulnessMetric(model=model, verbose_mode=False),
        ContextualPrecisionMetric(model=model, verbose_mode=False),
    ]+custom_metrics

    # non_LLM=[F1ScoreMetric(), RoughRecallAtNMetric(N=k)]

    
        
    return AnswerRelevancyMetric
    
