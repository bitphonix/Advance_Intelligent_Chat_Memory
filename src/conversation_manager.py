from typing import List, Dict, Any, Optional
from src.memory_systems import SummaryBufferMemory, SlidingWindowMemory, FixedWindowMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

class ConversationManager:
    """Advanced conversation manager with LangChain integration"""
    
    def __init__(self, model_type: str = 'gemini', memory_type: str = 'summary_buffer', 
                 token_limit: int = 4000, summary_threshold: int = 3000):
        """
        Initialize conversation manager
        
        Args:
            model_type: 'gemini' or 'openai'
            memory_type: 'summary_buffer', 'sliding_window', or 'fixed_window'
            token_limit: Maximum tokens to maintain
            summary_threshold: When to start summarizing (for summary_buffer)
        """
        self.model_type = model_type.lower()
        self.memory_type = memory_type.lower()
        
        # Initialize API keys
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.openai_key = os.getenv('OPENAI_API_KEY')
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()
        
        # Initialize memory system
        self.memory = self._initialize_memory_system(token_limit, summary_threshold)
        
        # Conversation state
        self.conversation_active = False
        
    def _initialize_llm_client(self):
        """Initialize LangChain LLM client"""
        if self.model_type == 'gemini':
            if self.gemini_key:
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=self.gemini_key,
                    temperature=0.7
                )
            else:
                print("❌ No Gemini API key found in environment")
                return None
                
        elif self.model_type == 'openai':
            if HAS_OPENAI and self.openai_key:
                return ChatOpenAI(
                    model="gpt-3.5-turbo",
                    openai_api_key=self.openai_key,
                    temperature=0.7
                )
            else:
                print("❌ OpenAI not available or no API key found")
                return None
        
        return None
        
    def _initialize_memory_system(self, token_limit: int, summary_threshold: int):
        """Initialize the specified memory system"""
        api_key = self.gemini_key if self.model_type == 'gemini' else self.openai_key
        
        if self.memory_type == 'summary_buffer':
            return SummaryBufferMemory(
                token_limit=token_limit,
                summary_threshold=summary_threshold,
                model_type=self.model_type,
                api_key=api_key
            )
        elif self.memory_type == 'sliding_window':
            return SlidingWindowMemory(
                token_limit=token_limit,
                model_type=self.model_type
            )
        elif self.memory_type == 'fixed_window':
            return FixedWindowMemory(
                max_messages=20,
                token_limit=token_limit,
                model_type=self.model_type
            )
        else:
            raise ValueError(f"Unknown memory type: {self.memory_type}")
    
    def add_system_message(self, message: str) -> None:
        """Add a system message to set context"""
        system_msg = {"role": "system", "content": message}
        self.memory.add_message(system_msg)
    
    def get_response(self, user_input: str) -> str:
        """Get AI response maintaining conversation context"""
        # Add user message to memory
        user_msg = {"role": "user", "content": user_input}
        self.memory.add_message(user_msg)
        
        # Get current context
        context = self.memory.get_context()
        
        # Generate response using LangChain
        if self.llm_client:
            try:
                # Convert to LangChain messages
                langchain_messages = self._convert_to_langchain_messages(context)
                
                # Generate response
                response = self.llm_client.invoke(langchain_messages)
                assistant_response = response.content
                
                # Add assistant response to memory
                assistant_msg = {"role": "assistant", "content": assistant_response}
                self.memory.add_message(assistant_msg)
                
                return assistant_response
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                print(f"❌ {error_msg}")
                
                # Add error to memory
                assistant_msg = {"role": "assistant", "content": "I apologize, but I encountered an error. Please try again."}
                self.memory.add_message(assistant_msg)
                
                return assistant_msg["content"]
        else:
            fallback_msg = "I'm here to help! (Please check your API key configuration)"
            assistant_msg = {"role": "assistant", "content": fallback_msg}
            self.memory.add_message(assistant_msg)
            return fallback_msg
    
    def _convert_to_langchain_messages(self, context: List[Dict[str, Any]]) -> List:
        """Convert context to LangChain message format"""
        langchain_messages = []
        
        for msg in context:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        return langchain_messages
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get detailed conversation statistics"""
        stats = self.memory.get_stats()
        stats.update({
            'model_type': self.model_type,
            'memory_type': self.memory_type,
            'llm_available': self.llm_client is not None,
            'conversation_active': self.conversation_active
        })
        return stats
    
    def clear_conversation(self) -> None:
        """Clear the conversation history"""
        self.memory.clear_memory()
        self.conversation_active = False
        print("🧹 Conversation cleared!")
    
    def get_memory_messages(self) -> List[Dict[str, Any]]:
        """Get raw memory messages (similar to memory.chat_memory.messages)"""
        return self.memory.messages.copy()
    
    def get_moving_summary_buffer(self) -> Optional[str]:
        """Get current summary buffer (if using summary memory)"""
        if hasattr(self.memory, 'summary') and self.memory.summary:
            return self.memory.summary.get('content', '')
        return None
    
    def switch_memory_type(self, new_memory_type: str) -> None:
        """Switch to a different memory type"""
        if new_memory_type.lower() not in ['summary_buffer', 'sliding_window', 'fixed_window']:
            raise ValueError(f"Unknown memory type: {new_memory_type}")
        
        # Save current stats
        old_stats = self.get_conversation_stats()
        
        # Initialize new memory system
        self.memory_type = new_memory_type.lower()
        self.memory = self._initialize_memory_system(
            token_limit=old_stats['token_limit'],
            summary_threshold=old_stats.get('summary_threshold', 3000)
        )
        
        print(f"🔄 Switched to {new_memory_type} memory system")
    
    def switch_model(self, new_model_type: str) -> None:
        """Switch to a different model type"""
        if new_model_type.lower() not in ['gemini', 'openai']:
            raise ValueError(f"Unknown model type: {new_model_type}")
        
        self.model_type = new_model_type.lower()
        self.llm_client = self._initialize_llm_client()
        
        print(f"🔄 Switched to {new_model_type} model")

# Test the conversation manager
if __name__ == "__main__":
    print("🤖 Testing Conversation Manager with LangChain Integration...\n")
    
    # Test different memory types
    memory_types = ['summary_buffer', 'sliding_window', 'fixed_window']
    
    for memory_type in memory_types:
        print(f"=== Testing {memory_type.title()} Memory ===")
        
        manager = ConversationManager(
            model_type='gemini',
            memory_type=memory_type,
            token_limit=1000,
            summary_threshold=800
        )
        
        # Test conversation
        test_inputs = [
            "Hello! Can you explain what machine learning is?",
            "How does it differ from traditional programming?",
            "What are the main types of machine learning?",
            "Can you give me an example of supervised learning?"
        ]
        
        for i, user_input in enumerate(test_inputs):
            print(f"\n👤 User: {user_input}")
            response = manager.get_response(user_input)
            print(f"🤖 Assistant: {response[:100]}...")
            
            # Show stats
            stats = manager.get_conversation_stats()
            print(f"📊 Stats: {stats['current_tokens']} tokens, {stats['total_messages']} messages")
        
        # Show memory access methods
        print(f"\n📋 Memory Messages: {len(manager.get_memory_messages())} messages")
        
        if memory_type == 'summary_buffer':
            summary = manager.get_moving_summary_buffer()
            if summary:
                print(f"📝 Summary Buffer: {summary[:100]}...")
        
        print(f"Final Stats: {manager.get_conversation_stats()}")
        print("-" * 50)