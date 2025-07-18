#!/usr/bin/env python3
"""
Interactive Chat Interface for Intelligent Memory System
"""

import os
import sys
from typing import Dict, Any
from src.conversation_manager import ConversationManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InteractiveChatInterface:
    """Interactive chat interface with memory management"""
    
    def __init__(self):
        self.manager = None
        self.running = False
        
    def display_welcome(self):
        """Display welcome message and options"""
        print("🧠 " + "="*60)
        print("    INTELLIGENT CHAT MEMORY SYSTEM")
        print("="*64)
        print("\n🎯 Features:")
        print("   • Multiple memory strategies (Summary Buffer, Sliding Window, Fixed Window)")
        print("   • Support for Gemini and OpenAI models")
        print("   • Smart context management")
        print("   • Real-time memory statistics")
        print("\n💡 Commands:")
        print("   /help     - Show all commands")
        print("   /stats    - Show memory statistics")
        print("   /clear    - Clear conversation")
        print("   /switch   - Switch memory type or model")
        print("   /memory   - Show memory contents")
        print("   /quit     - Exit chat")
        print("\n" + "="*64)
        
    def setup_initial_configuration(self):
        """Setup initial configuration"""
        print("\n🔧 INITIAL SETUP")
        print("-" * 20)
        
        # Check API keys
        gemini_key = os.getenv('GEMINI_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        print(f"📊 API Keys Status:")
        print(f"   Gemini: {'✅ Available' if gemini_key else '❌ Not found'}")
        print(f"   OpenAI: {'✅ Available' if openai_key else '❌ Not found'}")
        
        if not gemini_key and not openai_key:
            print("\n❌ No API keys found! Please set up your .env file.")
            print("   Copy .env.example to .env and add your API keys.")
            return False
        
        # Model selection
        print(f"\n🤖 Model Selection:")
        available_models = []
        if gemini_key:
            available_models.append('gemini')
        if openai_key:
            available_models.append('openai')
        
        if len(available_models) == 1:
            model_type = available_models[0]
            print(f"   Auto-selected: {model_type}")
        else:
            print("   Available models:")
            for i, model in enumerate(available_models, 1):
                print(f"   {i}. {model}")
            
            while True:
                try:
                    choice = input("\nSelect model (1-2, or press Enter for Gemini): ").strip()
                    if not choice:
                        model_type = 'gemini'
                        break
                    choice = int(choice)
                    if 1 <= choice <= len(available_models):
                        model_type = available_models[choice - 1]
                        break
                    print("❌ Invalid choice. Please try again.")
                except ValueError:
                    print("❌ Please enter a number.")
        
        # Memory type selection
        print(f"\n🧠 Memory Type Selection:")
        memory_types = ['summary_buffer', 'sliding_window', 'fixed_window']
        
        print("   Available memory types:")
        for i, mem_type in enumerate(memory_types, 1):
            descriptions = {
                'summary_buffer': 'Summarizes old messages using AI',
                'sliding_window': 'Removes oldest messages when limit reached',
                'fixed_window': 'Keeps fixed number of recent messages'
            }
            print(f"   {i}. {mem_type.replace('_', ' ').title()} - {descriptions[mem_type]}")
        
        while True:
            try:
                choice = input("\nSelect memory type (1-3, or press Enter for Summary Buffer): ").strip()
                if not choice:
                    memory_type = 'summary_buffer'
                    break
                choice = int(choice)
                if 1 <= choice <= len(memory_types):
                    memory_type = memory_types[choice - 1]
                    break
                print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a number.")
        
        # Advanced configuration
        print(f"\n⚙️ Advanced Configuration:")
        
        try:
            token_limit = input("Token limit (press Enter for 4000): ").strip()
            token_limit = int(token_limit) if token_limit else 4000
        except ValueError:
            token_limit = 4000
        
        try:
            summary_threshold = input("Summary threshold (press Enter for 3000): ").strip()
            summary_threshold = int(summary_threshold) if summary_threshold else 3000
        except ValueError:
            summary_threshold = 3000
        
        # Initialize conversation manager
        try:
            self.manager = ConversationManager(
                model_type=model_type,
                memory_type=memory_type,
                token_limit=token_limit,
                summary_threshold=summary_threshold
            )
            
            print(f"\n✅ Configuration Complete!")
            print(f"   Model: {model_type}")
            print(f"   Memory: {memory_type}")
            print(f"   Token Limit: {token_limit}")
            print(f"   Summary Threshold: {summary_threshold}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error initializing manager: {e}")
            return False
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands"""
        command = command.lower().strip()
        
        if command == '/help':
            self.show_help()
            
        elif command == '/stats':
            self.show_stats()
            
        elif command == '/clear':
            self.manager.clear_conversation()
            
        elif command == '/switch':
            self.handle_switch_command()
            
        elif command == '/memory':
            self.show_memory_contents()
            
        elif command == '/quit':
            print("\n👋 Thanks for using Intelligent Chat Memory System!")
            return False
            
        else:
            print(f"❌ Unknown command: {command}")
            print("   Type /help for available commands.")
        
        return True
    
    def show_help(self):
        """Show help information"""
        print("\n📖 HELP - Available Commands:")
        print("-" * 30)
        print("   /help     - Show this help message")
        print("   /stats    - Show memory and conversation statistics")
        print("   /clear    - Clear conversation history")
        print("   /switch   - Switch memory type or model")
        print("   /memory   - Show current memory contents")
        print("   /quit     - Exit the chat")
        print("\n💬 Chat Tips:")
        print("   • Just type your message and press Enter")
        print("   • Memory is automatically managed")
        print("   • Use /stats to monitor memory usage")
        print("   • Long conversations are supported!")
    
    def show_stats(self):
        """Show detailed statistics"""
        stats = self.manager.get_conversation_stats()
        
        print("\n📊 CONVERSATION STATISTICS:")
        print("-" * 30)
        print(f"   Model: {stats['model_type'].title()}")
        print(f"   Memory Type: {stats['memory_type'].replace('_', ' ').title()}")
        print(f"   Messages: {stats['total_messages']}")
        print(f"   Total Added: {stats['total_added']}")
        print(f"   Current Tokens: {stats['current_tokens']}")
        print(f"   Token Limit: {stats['token_limit']}")
        print(f"   Utilization: {stats['utilization']}")
        print(f"   LLM Available: {'✅' if stats['llm_available'] else '❌'}")
        
        # Additional stats for specific memory types
        if 'summarized_count' in stats:
            print(f"   Summarized Messages: {stats['summarized_count']}")
            print(f"   Has Summary: {'✅' if stats['has_summary'] else '❌'}")
        
        if 'discarded_messages' in stats:
            print(f"   Discarded Messages: {stats['discarded_messages']}")
    
    def show_memory_contents(self):
        """Show memory contents"""
        messages = self.manager.get_memory_messages()
        
        print(f"\n🧠 MEMORY CONTENTS ({len(messages)} messages):")
        print("-" * 40)
        
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            preview = content[:100] + "..." if len(content) > 100 else content
            
            print(f"   {i}. [{role.upper()}] {preview}")
        
        # Show summary buffer if available
        summary = self.manager.get_moving_summary_buffer()
        if summary:
            print(f"\n📝 SUMMARY BUFFER:")
            print(f"   {summary[:200]}...")
    
    def handle_switch_command(self):
        """Handle switch command"""
        print("\n🔄 SWITCH OPTIONS:")
        print("   1. Switch memory type")
        print("   2. Switch model")
        print("   3. Cancel")
        
        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()
                if choice == '1':
                    self.switch_memory_type()
                    break
                elif choice == '2':
                    self.switch_model()
                    break
                elif choice == '3':
                    print("   Cancelled.")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a number.")
    
    def switch_memory_type(self):
        """Switch memory type"""
        memory_types = ['summary_buffer', 'sliding_window', 'fixed_window']
        
        print("\n🧠 Available Memory Types:")
        for i, mem_type in enumerate(memory_types, 1):
            current = "← CURRENT" if mem_type == self.manager.memory_type else ""
            print(f"   {i}. {mem_type.replace('_', ' ').title()} {current}")
        
        while True:
            try:
                choice = input("\nSelect memory type (1-3): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(memory_types):
                    new_type = memory_types[choice - 1]
                    if new_type != self.manager.memory_type:
                        self.manager.switch_memory_type(new_type)
                    else:
                        print("   Already using this memory type.")
                    break
                print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a number.")
    
    def switch_model(self):
        """Switch model"""
        available_models = []
        if os.getenv('GEMINI_API_KEY'):
            available_models.append('gemini')
        if os.getenv('OPENAI_API_KEY'):
            available_models.append('openai')
        
        if len(available_models) == 1:
            print("   Only one model available.")
            return
        
        print("\n🤖 Available Models:")
        for i, model in enumerate(available_models, 1):
            current = "← CURRENT" if model == self.manager.model_type else ""
            print(f"   {i}. {model.title()} {current}")
        
        while True:
            try:
                choice = input("\nSelect model (1-2): ").strip()
                choice = int(choice)
                if 1 <= choice <= len(available_models):
                    new_model = available_models[choice - 1]
                    if new_model != self.manager.model_type:
                        self.manager.switch_model(new_model)
                    else:
                        print("   Already using this model.")
                    break
                print("❌ Invalid choice. Please try again.")
            except ValueError:
                print("❌ Please enter a number.")
    
    def run_chat_loop(self):
        """Main chat loop"""
        print("\n💬 CHAT STARTED - Type your message or /help for commands")
        print("="*60)
        
        self.running = True
        message_count = 0
        
        while self.running:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self.running = self.handle_command(user_input)
                    continue
                
                # Process regular message
                print("\n🤖 Assistant: ", end="", flush=True)
                
                try:
                    response = self.manager.get_response(user_input)
                    print(response)
                    
                    message_count += 1
                    
                    # Show stats every 10 messages
                    if message_count % 10 == 0:
                        stats = self.manager.get_conversation_stats()
                        print(f"\n📊 Quick Stats: {stats['total_messages']} messages, {stats['current_tokens']} tokens ({stats['utilization']})")
                        
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print("Please try again or type /help for commands.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Chat ended. Goodbye!")
                break
    
    def run(self):
        """Main run method"""
        try:
            self.display_welcome()
            
            if not self.setup_initial_configuration():
                return
            
            self.run_chat_loop()
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please check your configuration and try again.")

def main():
    """Main function"""
    chat = InteractiveChatInterface()
    chat.run()

if __name__ == "__main__":
    main()