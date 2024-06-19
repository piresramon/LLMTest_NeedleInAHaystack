import os
from operator import itemgetter
from typing import Optional

import maritalk
from langchain_community.chat_models import ChatMaritalk
from langchain.prompts import PromptTemplate
import transformers

from .model import ModelProvider


class MariTalk(ModelProvider):
    """
    A wrapper class for interacting with MariTalk's API, providing methods to encode text, generate prompts,
    evaluate models, and create LangChain runnables for language model interactions.

    Attributes:
        model_name (str): The name of the MariTalk model to use for evaluations and interactions.
        model: An instance of the asynchronous MariTalk client for asynchronous API calls.
        tokenizer: A tokenizer instance for encoding and decoding text to and from token representations.
    """
        
    DEFAULT_MODEL_KWARGS: dict = dict(max_tokens  = 300,
                                      temperature = 0)

    def __init__(self,
                 model_name: str = "sabia-2-medium",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS):
        """
        Initializes the MariTalk model provider with a specific model.

        Args:
            model_name (str): The name of the MariTalk model to use. Defaults to 'sabia-2-medium'.
            model_kwargs (dict): Model configuration. Defaults to {max_tokens: 300, temperature: 0}.
        
        Raises:
            ValueError: If MARITALK_API_SECRET_KEY is not found in the environment.
        """
        api_key = os.getenv('MARITALK_API_SECRET_KEY')
        if (not api_key):
            raise ValueError("MARITALK_API_SECRET_KEY must be in env.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.api_key = api_key
        self.model = maritalk.MariTalk(
            api_url="https://chat-dev.maritaca.ai/api",
            key=self.api_key,
            model=self.model_name
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("maritaca-ai/sabia-2-tokenizer-medium")
    
    async def evaluate_model(self, prompt: str) -> str:
        """
        Evaluates a given prompt using the MariTalk model and retrieves the model's response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The content of the model's response to the prompt.
        """
        response_chunks = []

        async_generator = self.model.generate(
                messages=prompt,
                stream=True,
                return_async_generator=True,
                **self.model_kwargs
            )
        
        async for chunk in async_generator:
            response_chunks.append(chunk.get('text', ''))

        return ''.join(response_chunks)
    
    def generate_prompt(self, context: str, retrieval_question: str) -> str | list[dict[str, str]]:
        """
        Generates a structured prompt for querying the model, based on a given context and retrieval question.

        Args:
            context (str): The context or background information relevant to the question.
            retrieval_question (str): The specific question to be answered by the model.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the structured prompt, including roles and content for system and user messages.
        """
        return [{
                "role": "user",
                "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
            },
            {
                "role": "assistant",
                "content": "Ok."
            },
            {
                "role": "user",
                "content": f"[Context]\n{context}\n[Context]\n\n{retrieval_question} Don't give information outside the document or repeat your findings"
            }]
    
    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens[:context_length])
    
    def get_langchain_runnable(self, context: str) -> str:
        """
        Creates a LangChain runnable that constructs a prompt based on a given context and a question, 
        queries the MariTalk model, and returns the model's response. This method leverages the LangChain 
        library to build a sequence of operations: extracting input variables, generating a prompt, 
        querying the model, and processing the response.

        Args:
            context (str): The context or background information relevant to the user's question. 
            This context is provided to the model to aid in generating relevant and accurate responses.

        Returns:
            str: A LangChain runnable object that can be executed to obtain the model's response to a 
            dynamically provided question. The runnable encapsulates the entire process from prompt 
            generation to response retrieval.

        Example:
            To use the runnable:
                - Define the context and question.
                - Execute the runnable with these parameters to get the model's response.
        """

        template = """You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
        \n ------- \n 
        {context} 
        \n ------- \n
        Here is the user question: \n --- --- --- \n {question} \n Don't give information outside the document or repeat your findings."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        )
        # Create a LangChain runnable
        model = ChatMaritalk(temperature=0, model=self.model_name, api_key=self.api_key)
        chain = ( {"context": lambda x: context,
                  "question": itemgetter("question")} 
                | prompt 
                | model 
                )
        return chain
    

