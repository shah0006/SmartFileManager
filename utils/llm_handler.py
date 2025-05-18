"""
LLM Handler Utility Module.

This module provides an interface for interacting with local LLMs (Transformers and Ollama)
for metadata extraction and content analysis.
"""

import os
import logging
import json
import time
import requests
from typing import Dict, Any, Optional, Union

# Setup logging
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for local LLM interactions."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM handler.
        
        Args:
            config: Configuration dictionary with LLM settings
        """
        # Default configuration
        self.config = {
            "provider": "auto",  # Options: "auto", "transformers", "ollama", "none"
            "model": "auto",     # Model to use (auto selects a default)
            "max_tokens": 1000,    # Max tokens in response (increased for better metadata extraction)
            "temperature": 0.1,    # Lower value for more deterministic answers (better for metadata)
            "timeout": 60,         # Timeout in seconds (increased for larger models)
            "api_url": "http://localhost:11434/api/generate"  # Ollama API URL
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Initialize provider
        self.provider = self.config["provider"]
        self.model = self.config["model"]
        
        # Auto-detect provider if set to auto
        if self.provider == "auto":
            self.provider = self._auto_detect_provider()
        
        # Auto-select model if set to auto
        if self.model == "auto":
            self.model = self._auto_select_model()
        
        # Initialize the selected provider
        self._initialize_provider()
    
    def _auto_detect_provider(self) -> str:
        """
        Auto-detect the available LLM provider.
        
        Returns:
            Provider name (transformers, ollama, or none)
        """
        # Check for Ollama first (prioritize it)
        try:
            # Check if Ollama is running by attempting a ping
            response = requests.get(
                self.config["api_url"].replace("/generate", "/ping"),
                timeout=2
            )
            if response.status_code == 200:
                logger.info("Detected Ollama API")
                return "ollama"
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        
        # Then try Transformers as fallback
        try:
            # Use importlib to check if transformers is available
            import importlib.util
            if importlib.util.find_spec("transformers") is not None:
                logger.info("Detected Transformers library")
                return "transformers"
        except ImportError:
            logger.debug("Transformers not available")
        
        # If all fails
        logger.warning("No LLM provider detected, using none")
        return "none"
    
    def _auto_select_model(self) -> str:
        """
        Auto-select an appropriate model based on provider.
        
        Returns:
            Model name
        """
        if self.provider == "ollama":
            # Use the latest Llama model for best results
            return "llama3"
        elif self.provider == "transformers":
            # Choose a model suitable for text generation
            return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            return "none"
    
    def _initialize_provider(self):
        """Initialize the selected LLM provider."""
        if self.provider == "transformers":
            self._initialize_transformers()
        elif self.provider == "ollama":
            # No initialization needed for Ollama API
            logger.info(f"Using Ollama with model: {self.model}")
        else:
            logger.warning("No LLM provider initialized")
    
    def _initialize_transformers(self):
        """Initialize Transformers model."""
        try:
            # Import required libraries
            import importlib
            
            # Dynamically import transformers
            transformers_spec = importlib.util.find_spec("transformers")
            if transformers_spec is not None:
                transformers = importlib.import_module("transformers")
                AutoTokenizer = getattr(transformers, "AutoTokenizer")
                AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
                
                # Import torch
                torch_spec = importlib.util.find_spec("torch")
                if torch_spec is not None:
                    torch = importlib.import_module("torch")
                    
                    logger.info(f"Initializing Transformers with model: {self.model}")
                    
                    # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            
            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model with appropriate precision
            if device == "cuda":
                # Use half precision for GPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                # Full precision for CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model
                ).to(device)
            
            logger.info("Transformers model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Transformers: {e}")
            self.provider = "none"
    
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM.
        
        Args:
            prompt: Prompt text to send to the LLM
            
        Returns:
            LLM response text
        """
        if self.provider == "transformers":
            return self._get_transformers_response(prompt)
        elif self.provider == "ollama":
            return self._get_ollama_response(prompt)
        else:
            logger.warning("No LLM provider available to generate response")
            return ""
    
    def _get_transformers_response(self, prompt: str) -> str:
        """
        Get a response using Transformers.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        try:
            import torch
            
            # Prepare inputs
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=self.config["max_tokens"],
                    temperature=self.config["temperature"],
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode the response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating Transformers response: {e}")
            return ""
    
    def _get_ollama_response(self, prompt: str) -> str:
        """
        Get a response using Ollama API.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Response text
        """
        try:
            # Add more context for better results with metadata extraction
            enhanced_prompt = f"""You are a helpful AI assistant specialized in metadata extraction.
Your task is to extract structured information from the text provided.
Be precise, accurate, and focus on objective factual information.
If a piece of information is not present, indicate that it's not available.
Do not make up information or hallucinate details.

Here is the text to analyze:

{prompt}"""

            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config["temperature"],
                    "num_predict": self.config["max_tokens"],
                    "top_p": 0.9,  # Add top_p sampling for better deterministic outputs
                    "stop": ["<end>", "###"]  # Add stop tokens to prevent rambling
                }
            }
            
            # Make the API request
            logger.info(f"Sending request to Ollama API with model: {self.model}")
            start_time = time.time()
            
            response = requests.post(
                self.config["api_url"],
                json=payload,
                timeout=self.config["timeout"]
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Ollama API response received in {elapsed_time:.2f}s")
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Clean up the response text
                response_text = response_text.strip()
                
                # Log some stats about the response
                tokens = data.get("eval_count", 0)
                logger.debug(f"Generated {tokens} tokens in {elapsed_time:.2f}s")
                
                return response_text
            else:
                logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                return ""
                
        except requests.Timeout:
            logger.error(f"Timeout occurred when calling Ollama API (>{self.config['timeout']}s)")
            return ""
        except Exception as e:
            logger.error(f"Error generating Ollama response: {e}")
            return ""
            
    def is_available(self) -> bool:
        """
        Check if the LLM is available.
        
        Returns:
            True if available, False otherwise
        """
        return self.provider != "none"


class CachingLLMHandler(LLMHandler):
    """LLM handler with response caching for efficiency."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_file: Optional[str] = None):
        """
        Initialize the caching LLM handler.
        
        Args:
            config: Configuration dictionary with LLM settings
            cache_file: Path to the cache file (default: use in-memory cache only)
        """
        super().__init__(config)
        
        # Initialize cache
        self.cache = {}
        self.cache_file = cache_file
        
        # Load cache if file exists
        if cache_file and os.path.exists(cache_file):
            self._load_cache()
    
    def get_response(self, prompt: str) -> str:
        """
        Get a response from the LLM with caching.
        
        Args:
            prompt: Prompt text to send to the LLM
            
        Returns:
            LLM response text
        """
        # Check if in cache
        cache_key = self._get_cache_key(prompt)
        if cache_key in self.cache:
            logger.debug(f"Using cached response for prompt: {prompt[:50]}...")
            return self.cache[cache_key]
        
        # Get response from LLM
        response = super().get_response(prompt)
        
        # Cache the response
        if response:
            self.cache[cache_key] = response
            self._save_cache()
        
        return response
    
    def _get_cache_key(self, prompt: str) -> str:
        """
        Generate a cache key for a prompt.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Cache key
        """
        # Simple hash of provider, model, and prompt
        key = f"{self.provider}_{self.model}_{prompt}"
        
        # Use a deterministic hash for the key
        import hashlib
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_cache(self):
        """Load the cache from file."""
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached responses from {self.cache_file}")
        except Exception as e:
            logger.warning(f"Error loading LLM cache: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """Save the cache to file."""
        if not self.cache_file:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} cached responses to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Error saving LLM cache: {e}")


def get_llm_handler(config: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Union[LLMHandler, CachingLLMHandler]:
    """
    Factory function to create an appropriate LLM handler.
    
    Args:
        config: Configuration dictionary with LLM settings
        use_cache: Whether to use caching
        
    Returns:
        LLM handler instance
    """
    try:
        # Prioritize Ollama if no specific config provided
        if config is None:
            config = {
                "provider": "auto",  # Will prioritize Ollama in auto-detect
                "model": "llama3",   # Default to latest Llama model
                "temperature": 0.1,    # Low temperature for deterministic responses
                "max_tokens": 1000     # Increased token limit for comprehensive responses
            }
        
        if use_cache:
            # Determine cache file location
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "llm_cache")
            cache_file = os.path.join(cache_dir, "response_cache.json")
            
            return CachingLLMHandler(config, cache_file)
        else:
            return LLMHandler(config)
    except Exception as e:
        logger.error(f"Error creating LLM handler: {e}")
        return LLMHandler({"provider": "none"})
