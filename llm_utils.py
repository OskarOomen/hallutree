"""
LLM Utilities using LangChain with GPT-4o and LangSmith tracing.
"""

import os
import logging
from typing import List, Optional
from functools import wraps
import time

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from OpenAI API calls
logging.getLogger("httpx").setLevel(logging.WARNING)

class LLMConfig(BaseModel):
    """Configuration for LLM settings."""
    model_name: str = "gpt-4o"
    temperature: float = 0
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    langsmith_project: Optional[str] = None
    langsmith_api_key: Optional[str] = None


class LLMUtils:
    """Main LLM utility class with LangChain integration."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM utilities with configuration."""
        self.config = config or LLMConfig()
        self._setup_environment()
        self._setup_llm()
    
    def _setup_environment(self):
        """Setup environment variables."""
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        if not self.config.langsmith_api_key:
            self.config.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if not self.config.langsmith_project:
            self.config.langsmith_project = os.getenv("LANGSMITH_PROJECT", "ragatomics")
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        if not self.config.langsmith_api_key:
            raise ValueError("LangSmith API key is required")
    
    def _setup_llm(self):
        """Setup the LLM instance."""
        callback_manager = None
        
        self.llm_base = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            callback_manager=callback_manager
        )
        self.llm = self.llm_base | StrOutputParser()

    def simple_completion(self, sys_prompt: str, human_prompt: str) -> str:
        """Simple completion with system message."""
        return self.llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=human_prompt)])
    
    def simple_completion_no_system(self, human_prompt: str) -> str:
        """Simple completion without a system message."""
        return self.llm.invoke([HumanMessage(content=human_prompt)])
    
    def chat_completion(self, messages: List[BaseMessage]) -> str:
        return self.llm.invoke(messages)


def trace_llm_call(func_name: Optional[str] = None):
    """Decorator to trace LLM function calls with LangSmith."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            function_name = func_name or func.__name__
            for i in range(3): # repeat after rate limit error because it often just needs to wait a fraction of a second
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    # Only log errors, not successful completions
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"LLM call '{function_name}' failed after {execution_time:.2f}s: {e}")
                    time.sleep(0.2)
            time.sleep(1)
            raise Exception("LLM call failed after 3 attempts")
        return wrapper
    return decorator


# Convenience functions
def create_llm_utils(
    model_name: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    langsmith_project: Optional[str] = None
) -> LLMUtils:
    """Create an LLMUtils instance with default settings."""
    config = LLMConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        langsmith_project=langsmith_project
    )
    return LLMUtils(config)


@trace_llm_call("text_generation")
def generate(llm_utils: LLMUtils, sys_prompt: str, human_prompt: str) -> str:
    """Generate text using the LLM."""
    return llm_utils.simple_completion(sys_prompt, human_prompt)

@trace_llm_call("text_generation_no_system")
def generate_no_system(llm_utils: LLMUtils, human_prompt: str) -> str:
    """Generate text using the LLM without a system message."""
    return llm_utils.simple_completion_no_system(human_prompt)

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    try:
        llm = create_llm_utils()
        
        # Simple completion
        response = generate_no_system(llm, "What is the capital of France?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set OPENAI_API_KEY and LANGSMITH_API_KEY environment variables")