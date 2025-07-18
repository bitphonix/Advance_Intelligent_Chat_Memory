import tiktoken
from typing import List, Dict, Any
import os

try:
    import google.generativeai as genai
    HAS_GOOGLE_TOKENIZATION = True
except ImportError:
    HAS_GOOGLE_TOKENIZATION = False
    print("Warning: google-generativeai not available. Using fallback for Gemini token counting.")

class TokenCounter:
    """Handles token counting for different LLM providers"""
    
    def __init__(self):
        self.openai_encodings = {
            'gpt-3.5-turbo': 'cl100k_base',
            'gpt-4': 'cl100k_base',
            'gpt-4-turbo': 'cl100k_base',
            'gpt-4o': 'cl100k_base',
            'text-davinci-003': 'p50k_base'
        }
        
        # Configure Gemini if API key is available
        if HAS_GOOGLE_TOKENIZATION:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
        
    def count_tokens_openai(self, text: str, model: str = 'gpt-3.5-turbo') -> int:
        """Count tokens for OpenAI models using tiktoken"""
        try:
            encoding_name = self.openai_encodings.get(model, 'cl100k_base')
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting OpenAI tokens: {e}")
            return self._fallback_token_count(text)
    
    def count_tokens_gemini(self, text: str, model_name: str = 'gemini-1.5-flash') -> int:
        """Count tokens for Gemini models using Google's tokenization"""
        try:
            if HAS_GOOGLE_TOKENIZATION:
                model = genai.GenerativeModel(model_name)
                result = model.count_tokens(text)
                return result.total_tokens
            else:
                # Fallback to OpenAI encoding
                encoding = tiktoken.get_encoding('cl100k_base')
                return len(encoding.encode(text))
        except Exception as e:
            print(f"Error counting Gemini tokens: {e}")
            return self._fallback_token_count(text)
    
    def count_tokens(self, text: str, model_type: str = 'gemini', model_name: str = None) -> int:
        """Universal token counter"""
        if model_type.lower() == 'openai':
            return self.count_tokens_openai(text, model_name or 'gpt-3.5-turbo')
        elif model_type.lower() == 'gemini':
            return self.count_tokens_gemini(text, model_name or 'gemini-1.5-flash')
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'openai' or 'gemini'.")
    
    def count_message_tokens(self, messages: List[Dict[str, Any]], model_type: str = 'gemini', model_name: str = None) -> int:
        """Count tokens for a list of messages"""
        total_tokens = 0
        
        for message in messages:
            content = message.get('content', '')
            role = message.get('role', '')
            
            # Count content tokens
            content_tokens = self.count_tokens(content, model_type, model_name)
            
            # Add overhead for role and formatting
            if model_type.lower() == 'openai':
                # OpenAI has specific overhead per message
                overhead = 4  # tokens for role and formatting
            else:
                # Gemini overhead estimation
                role_tokens = self.count_tokens(f"role: {role}", model_type, model_name)
                overhead = role_tokens + 2  # formatting overhead
            
            total_tokens += content_tokens + overhead
        
        return total_tokens
    
    def _fallback_token_count(self, text: str) -> int:
        """Fallback token estimation"""
        return len(text) // 4

# Test the token counter
if __name__ == "__main__":
    counter = TokenCounter()
    
    test_messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you help me with a programming problem?"}
    ]
    
    print("=== Token Counter Test ===")
    print(f"Total tokens (Gemini): {counter.count_message_tokens(test_messages, 'gemini')}")
    print(f"Total tokens (OpenAI): {counter.count_message_tokens(test_messages, 'openai')}")