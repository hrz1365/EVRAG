from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
# import google.genai as genai
import google.generativeai as genai
import  dotenv
from dotenv import load_dotenv, dotenv_values 
import os

class LLMEngine:
    """
    LLMEngine is a class that provides an interface for interacting with a
      large language model (LLM) to generate text based on a given prompt.
      It uses the Hugging Face Transformers library to load
      a pre-trained model and tokenizer, and sets up a text generation pipeline.

    Attributes:
        model_id (str): The identifier of the pre-trained model to use.
          Defaults to "tiiuae/falcon-7b-instruct".
        device (str): The device to run the model on.
          Defaults to "auto", which automatically selects the device.

    Methods:
        __init__(model_id="tiiuae/falcon-7b-instruct", device="auto")
        generate(prompt: str, max_new_tokens: int = 150, temperature: float = 0,
          do_sample: bool = False)
    """

    def __init__(self, model_id="tiiuae/falcon-7b-instruct", device="auto"):
        """
        Initializes the generator with a specified model and device configuration.

        Args:
            model_id (str): The identifier of the pre-trained model to use.
                Defaults to "tiiuae/falcon-7b-instruct".
            device (str): The device to load the model on. Can be "auto", "cpu",
                or a specific GPU identifier. Defaults to "auto".

        Attributes:
            model_id (str): The identifier of the pre-trained model.
            device (str): The device configuration for the model.
            tokenizer (AutoTokenizer): The tokenizer loaded from the pre-trained model.
            model (AutoModelForCausalLM): The causal language model loaded
              from the pre-trained model.
            llm_pipeline (Pipeline): The text generation pipeline initialized
              with the model and tokenizer.
        """
        load_dotenv()
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
        )
        self.llm_pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0,
        do_sample: bool = False,
    ):
        """
        Generates a response based on the given prompt using the language model pipeline.

        Args:
            prompt (str): The input text prompt to generate a response for.
            max_new_tokens (int, optional): The maximum number of new tokens
              to generate. Defaults to 150.
            temperature (float, optional): Sampling temperature for controlling
              randomness. A lower value makes the output more deterministic.
              Defaults to 0.
            do_sample (bool, optional): Whether to use sampling for generation.
                If False, the model uses greedy decoding. Defaults to False.

        Returns:
            str: The generated response text, with the prompt removed
              and leading/trailing whitespace stripped.
        """
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )[0]["generated_text"]
        response = response.replace(prompt, "").strip()
        return response


load_dotenv()


class LLMEngineGemini:
    """
    A class to interact with the Gemini generative AI model.

    Attributes:
      model (genai.GenerativeModel): The generative model instance.

    Methods:
      __init__(model_name: str = "gemini-2.5-flash") -> None:
        Initializes the LLMEngineGemini instance with the specified model name.
        Raises a ValueError if the required API key environment variable is not set.

      generate(prompt: str, max_output_tokens: int = 150, temperature: float = 0) -> str:
        Generates a response from the Gemini model based on the provided prompt.
    """

    def __init__(
        self,
        model_name="gemini-2.5-flash",
        max_output_tokens: int = 150,
        temperature: float = 0.0,
    ) -> None:
        """
        Initializes the generator with a specified model name.

        Args:
          model_name (str, optional): The name of the generative model to use.
            Defaults to "gemini-2.5-pro".

        Raises:
          ValueError: If the "GeminiRoya" environment variable is not set.
        """
        key = os.getenv("GeminiRoya")
        
        if not key:
            raise ValueError("Gemini environment variable not set.")
        
        genai.configure(api_key=key)

        self.model = genai.GenerativeModel(model_name)
        # self.model_name=model_name
        # client = genai.Client(api_key=key)
        # self.model = client.models

    def generate(self, user_prompt: str):
       
        """
        Generates a response based on the provided prompt using the model.

        Args:
          prompt (str): The input prompt to generate a response for.

        Returns:
          str: The generated response text.
        """

        system_instruction="""
        you are a helpful AI assistant operating in a multi-document retrieval-augmented generation (RAG) system.
        Answer the user's question using only the information explicitly stated in the provided documents.
        ***Rules:
        * You may combine information from multiple documents only when they are consistent and directly relevant
        * When answering, cite supporting documents using their identifiers (e.g., title and page number [future of vehicle grid integration , 2]).
        * Do not infer missing details or resolve conflicts on your own
        * If documents contain conflicting information, clearly state that the sources disagree
        * Do not use prior knowledge or external information
        * If the provided documents do not contain sufficient information to answer the question, respond exactly with: "I am sorry, I could not find relevant information."
        Keep answers concise, factual, and grounded in the documents.
        ***Important***
        The source and page is given as part of context. use it for referencing 
        ***Input***
            'content': the content retrieved
            'source': the path to the file
            'page': page number of content
            'title': file name
        """
        messages = [
            {"role": "model", "parts": [system_instruction]},
            {"role": "user", "parts": [user_prompt]},
        ]
        
        # print(messages)
        # return
        # Generate the response
        response = self.model.generate_content(messages)
        # response = self.model.generate_content(contents = prompt , model=self.model_name)
        print(response)

        return response.candidates[0].content.parts[0].text
