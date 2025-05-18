"""
Local LLM integration module for SmartFileManager.

This module provides the interface for integrating with local LLM models,
allowing for content-based file analysis and metadata extraction.
"""

import os
import logging
import json
import tempfile
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

# Setup logging
logger = logging.getLogger(__name__)

class LLMHandler(ABC):
    """Base abstract class for LLM integration."""
    
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM for a given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM is available for use.
        
        Returns:
            True if LLM is available, False otherwise
        """
        pass


class TransformersLLMHandler(LLMHandler):
    """Handler for using local Transformers LLM models."""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize the Transformers LLM handler.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
        # Try to load the model and tokenizer
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Error loading Transformers model: {e}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if not self.model_name:
                # Default to a smaller model suitable for local use
                self.model_name = "gpt2"
            
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Successfully loaded model: {self.model_name}")
            
        except ImportError:
            logger.error("transformers library not installed. Please install with: pip install transformers")
        except Exception as e:
            logger.error(f"Error initializing Transformers model: {e}")
            raise
    
    def is_available(self) -> bool:
        """
        Check if the Transformers model is loaded and ready to use.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.tokenizer is not None
    
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the loaded model.
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response
        """
        if not self.is_available():
            logger.error("Model not available for inference")
            return ""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response (adjust parameters as needed)
            output = self.model.generate(
                inputs["input_ids"],
                max_length=100 + len(inputs["input_ids"][0]),
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
            
            # Decode and clean up the response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt from the response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""


class OllamaLLMHandler(LLMHandler):
    """Handler for using local Ollama LLM server."""
    
    def __init__(self, model_name: str = "llama3", api_url: str = "http://localhost:11434/api"):
        """
        Initialize the Ollama LLM handler.
        
        Args:
            model_name: Name of the Ollama model to use
            api_url: URL of the Ollama API
        """
        self.model_name = model_name
        self.api_url = api_url
        self.is_initialized = False
        
        # Try to initialize the handler
        try:
            self._initialize()
        except Exception as e:
            logger.error(f"Error initializing Ollama handler: {e}")
    
    def _initialize(self):
        """Initialize the Ollama handler."""
        try:
            import requests
            
            # Test the API connection
            try:
                response = requests.get(f"{self.api_url}/tags")
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    available_models = [m['name'] for m in models]
                    
                    if self.model_name not in available_models:
                        logger.warning(f"Model {self.model_name} not found in Ollama. Available models: {available_models}")
                    else:
                        logger.info(f"Successfully connected to Ollama. Using model: {self.model_name}")
                        self.is_initialized = True
                else:
                    logger.error(f"Error connecting to Ollama API: {response.status_code} {response.text}")
            except Exception as e:
                logger.error(f"Error connecting to Ollama API: {e}")
                
        except ImportError:
            logger.error("requests library not installed. Please install with: pip install requests")
    
    def is_available(self) -> bool:
        """
        Check if the Ollama API is available.
        
        Returns:
            True if API is available, False otherwise
        """
        return self.is_initialized
    
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the Ollama API.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            The API's response
        """
        if not self.is_available():
            logger.error("Ollama API not available")
            return ""
        
        try:
            import requests
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.api_url}/generate", json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Error from Ollama API: {response.status_code} {response.text}")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {e}")
            return ""


class LLMFactory:
    """Factory for creating LLM handlers based on available backends."""
    
    @staticmethod
    def create_llm_handler(config: Dict[str, Any] = None) -> Optional[LLMHandler]:
        """
        Create an LLM handler based on available backends and configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            An initialized LLM handler or None if no handler could be created
        """
        if config is None:
            config = {}
        
        # Try Ollama first (usually better for local deployment)
        handler = OllamaLLMHandler(
            model_name=config.get('ollama_model', 'llama3'),
            api_url=config.get('ollama_api_url', 'http://localhost:11434/api')
        )
        
        if handler.is_available():
            logger.info("Using Ollama LLM handler")
            return handler
        
        # Fall back to Transformers
        handler = TransformersLLMHandler(
            model_name=config.get('transformers_model', 'gpt2'),
            device=config.get('device', 'cpu')
        )
        
        if handler.is_available():
            logger.info("Using Transformers LLM handler")
            return handler
        
        logger.warning("No LLM handler could be initialized")
        return None
