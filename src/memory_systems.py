from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.token_counter import TokenCounter
import os
from datetime import datetime

class BaseMemory(ABC):
    """Abstract base class for all memory systems"""
    
    def __init__(self, token_limit: int = 4000, model_type: str = 'gemini'):
        self.token_limit = token_limit
        self.model_type = model_type
        self.token_counter = TokenCounter()
        self.messages = []
        self.total_messages_added = 0
        
    @abstractmethod
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a new message to memory"""
        pass
        
    @abstractmethod
    def get_context(self) -> List[Dict[str, Any]]:
        """Get the current context for the LLM"""
        pass
        
    def get_current_tokens(self) -> int:
        """Get current token count"""
        return self.token_counter.count_message_tokens(self.messages, self.model_type)
        
    def is_over_limit(self) -> bool:
        """Check if we're over the token limit"""
        return self.get_current_tokens() > self.token_limit
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_messages': len(self.messages),
            'total_added': self.total_messages_added,
            'current_tokens': self.get_current_tokens(),
            'token_limit': self.token_limit,
            'utilization': f"{(self.get_current_tokens() / self.token_limit) * 100:.1f}%",
            'memory_type': self.__class__.__name__
        }
    
    def clear_memory(self) -> None:
        """Clear all messages from memory"""
        self.messages = []
        self.total_messages_added = 0

class SlidingWindowMemory(BaseMemory):
    """Sliding window memory - removes oldest messages when over token limit"""
    
    def __init__(self, token_limit: int = 4000, model_type: str = 'gemini'):
        super().__init__(token_limit, model_type)
        self.discarded_count = 0
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message and trim if necessary"""
        self.messages.append(message)
        self.total_messages_added += 1
        self._trim_messages()
        
    def _trim_messages(self) -> None:
        """Remove oldest messages if over token limit"""
        while self.is_over_limit() and len(self.messages) > 1:
            discarded = self.messages.pop(0)
            self.discarded_count += 1
            print(f"🗑️ [Sliding] Discarded: {discarded['content'][:50]}...")
            
    def get_context(self) -> List[Dict[str, Any]]:
        """Return all current messages"""
        return self.messages.copy()
        
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['discarded_messages'] = self.discarded_count
        return stats

class FixedWindowMemory(BaseMemory):
    """Fixed window memory - keeps exactly N recent messages"""
    
    def __init__(self, max_messages: int = 20, token_limit: int = 4000, model_type: str = 'gemini'):
        super().__init__(token_limit, model_type)
        self.max_messages = max_messages
        self.discarded_count = 0
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message and maintain fixed window size"""
        self.messages.append(message)
        self.total_messages_added += 1
        
        # First trim by message count
        while len(self.messages) > self.max_messages:
            discarded = self.messages.pop(0)
            self.discarded_count += 1
            print(f"🗑️ [Fixed] Discarded (count): {discarded['content'][:50]}...")
            
        # Then trim by token count
        while self.is_over_limit() and len(self.messages) > 1:
            discarded = self.messages.pop(0)
            self.discarded_count += 1
            print(f"🗑️ [Fixed] Discarded (token): {discarded['content'][:50]}...")
            
    def get_context(self) -> List[Dict[str, Any]]:
        return self.messages.copy()
        
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats['max_messages'] = self.max_messages
        stats['discarded_messages'] = self.discarded_count
        return stats

class SummaryBufferMemory(BaseMemory):
    """Summary buffer memory - summarizes old messages using LLM"""
    
    def __init__(self, token_limit: int = 4000, summary_threshold: int = 3000, 
                 model_type: str = 'gemini', api_key: str = None):
        super().__init__(token_limit, model_type)
        self.summary_threshold = summary_threshold
        self.api_key = api_key or os.getenv('GEMINI_API_KEY' if model_type == 'gemini' else 'OPENAI_API_KEY')
        self.summary = None
        self.summarized_count = 0
        self.llm_client = self._initialize_llm_client()
        
    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client"""
        if self.model_type.lower() == 'gemini':
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                    return genai.GenerativeModel('gemini-1.5-flash')
                return None
            except ImportError:
                return None
        elif self.model_type.lower() == 'openai':
            try:
                import openai
                if self.api_key:
                    return openai.OpenAI(api_key=self.api_key)
                return None
            except ImportError:
                return None
        return None
        
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message and summarize if needed"""
        self.messages.append(message)
        self.total_messages_added += 1
        
        if self.get_current_tokens() > self.summary_threshold:
            self._create_summary()
            
    def _create_summary(self) -> None:
        """Create summary of old messages using LLM"""
        if len(self.messages) <= 3:
            return
            
        # Keep last 2 messages as recent context
        messages_to_summarize = self.messages[:-2]
        recent_messages = self.messages[-2:]
        
        # Generate summary
        summary_text = self._generate_summary(messages_to_summarize)
        
        if self.summary is None:
            self.summary = {
                'role': 'system',
                'content': f"Previous conversation summary: {summary_text}"
            }
        else:
            self.summary['content'] += f"\n\nAdditional context: {summary_text}"
            
        # Replace old messages with summary + recent messages
        self.messages = [self.summary] + recent_messages
        self.summarized_count += len(messages_to_summarize)
        
        print(f"📋 [Summary] Summarized {len(messages_to_summarize)} messages")
        
    def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Generate summary using LLM or fallback"""
        if self.llm_client is None:
            return self._fallback_summary(messages)
            
        try:
            conversation_text = self._format_messages_for_summary(messages)
            
            prompt = f"""Please summarize the following conversation in 2-3 sentences, preserving key topics and important details:

{conversation_text}

Summary:"""

            if self.model_type.lower() == 'gemini':
                response = self.llm_client.generate_content(prompt)
                return response.text.strip()
            elif self.model_type.lower() == 'openai':
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            print(f"⚠️ Error generating LLM summary: {e}")
            return self._fallback_summary(messages)
            
    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Fallback summary generation"""
        user_messages = [msg['content'] for msg in messages if msg.get('role') == 'user']
        assistant_messages = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
        
        topics = []
        for msg in user_messages[:3]:
            words = msg.split()
            topics.extend([word for word in words if len(word) > 5])
        
        topics_str = ', '.join(list(set(topics))[:5])
        
        return f"Conversation covered: {topics_str}. {len(user_messages)} questions, {len(assistant_messages)} responses."
        
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages for summarization"""
        formatted_lines = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_lines.append(f"{role.capitalize()}: {content}")
        return '\n'.join(formatted_lines)
        
    def get_context(self) -> List[Dict[str, Any]]:
        return self.messages.copy()
        
    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats.update({
            'summary_threshold': self.summary_threshold,
            'summarized_count': self.summarized_count,
            'has_summary': self.summary is not None,
            'llm_available': self.llm_client is not None
        })
        return stats
        
    def clear_memory(self) -> None:
        super().clear_memory()
        self.summary = None
        self.summarized_count = 0

# Test memory systems
if __name__ == "__main__":
    print("🧠 Testing Memory Systems...\n")
    
    test_messages = [
        {"role": "user", "content": "Hello! How are you today?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you explain machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "What are the main types?"},
        {"role": "assistant", "content": "There are three main types: supervised learning, unsupervised learning, and reinforcement learning."},
    ]
    
    memory_systems = [
        ("Sliding Window", SlidingWindowMemory(token_limit=200)),
        ("Fixed Window", FixedWindowMemory(max_messages=4, token_limit=200)),
        ("Summary Buffer", SummaryBufferMemory(token_limit=300, summary_threshold=200)),
    ]
    
    for name, memory in memory_systems:
        print(f"=== Testing {name} Memory ===")
        for msg in test_messages:
            memory.add_message(msg)
        
        print(f"Final Stats: {memory.get_stats()}")
        print(f"Context Length: {len(memory.get_context())}")
        print()