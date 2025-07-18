# Intelligent Chat Memory System

A sophisticated chat memory management system that maintains context across unlimited conversational length using smart memory strategies.

## Features

- **Multiple Memory Strategies**:
  - Summary Buffer Memory: Summarizes old messages using AI
  - Sliding Window Memory: Removes oldest messages when limit reached
  - Fixed Window Memory: Keeps fixed number of recent messages

- **Multi-Model Support**:
  - Google Gemini (default)
  - OpenAI GPT models

- **LangChain Integration**:
  - Professional message handling
  - Seamless model switching
  - Advanced conversation management

- **Interactive Chat Interface**:
  - Real-time memory statistics
  - Dynamic configuration switching
  - Comprehensive command system

## Installation

1. Clone or download the project
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

## Usage

### Interactive Chat
```bash
python src/interactive_chat.py
```

### Testing Components
```bash
# Test memory systems
python src/memory_systems.py

# Test conversation manager
python src/conversation_manager.py

# Test token counter
python src/token_counter.py
```

### Available Commands

- `/help` - Show all commands
- `/stats` - Show memory statistics
- `/clear` - Clear conversation
- `/switch` - Switch memory type or model
- `/memory` - Show memory contents
- `/quit` - Exit chat

## Memory Strategies

### Summary Buffer Memory
- Automatically summarizes old messages when token limit is approached
- Preserves recent context while condensing historical information
- Uses AI to generate intelligent summaries

### Sliding Window Memory
- Maintains a sliding window of recent messages
- Automatically removes oldest messages when over token limit
- Simple but effective for many use cases

### Fixed Window Memory
- Keeps exactly N most recent messages
- Provides predictable memory usage
- Good for applications with strict memory constraints

## Configuration

### Environment Variables
```bash
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Memory Configuration
- `token_limit`: Maximum tokens to maintain (default: 4000)
- `summary_threshold`: When to start summarizing (default: 3000)
- `max_messages`: Maximum messages for fixed window (default: 20)

## Advanced Features

### Memory Access
```python
# Get raw memory messages
messages = manager.get_memory_messages()

# Get summary buffer (if using summary memory)
summary = manager.get_moving_summary_buffer()

# Get detailed statistics
stats = manager.get_conversation_stats()
```

### Dynamic Switching
```python
# Switch memory type during conversation
manager.switch_memory_type('sliding_window')

# Switch model type
manager.switch_model('openai')
```

## Testing

The system includes comprehensive testing for:
- Long conversations (500+ messages)
- Memory efficiency
- Token optimization
- Error handling
- Model switching

Run tests with:
```bash
python -m pytest tests/  # If you add test files
```

## Skills Demonstrated

- Advanced context window management
- Memory system architecture design
- Automatic summarization implementation
- State management in conversational AI
- Performance optimization for long conversations
- Cost and performance management for LLMs
- LangChain integration patterns

## Contributing

This is a learning project. Feel free to:
- Add new memory strategies
- Implement additional models
- Enhance the chat interface
- Add more sophisticated summarization
- Implement semantic search capabilities

## License

MIT License - See LICENSE file for details