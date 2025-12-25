# Custom Chatbot with RAG

**Intelligent chatbot system combining large language models with retrieval augmented generation for domain-specific conversations about fictional characters.**

This project demonstrates advanced natural language processing techniques using embeddings, semantic search, and custom prompt engineering to create a specialized chatbot that can answer questions about fictional characters from a curated dataset.

## Features

- **Retrieval Augmented Generation (RAG)** - Combines LLM capabilities with custom knowledge base
- **Semantic Search** - Uses OpenAI embeddings for intelligent document retrieval
- **Character Analysis** - Advanced extraction of personality traits, relationships, and occupations
- **Interactive Interface** - Multi-mode chatbot with question answering, recommendations, and comparisons
- **Performance Optimization** - Robust error handling and batch processing for embeddings
- **Custom Prompt Engineering** - Specialized prompts for better character-specific responses

## Dataset

The chatbot is built around a **fictional character descriptions dataset** containing:
- 55 unique characters from various media (plays, movies, shows, etc.)
- Detailed character descriptions with personality traits and backgrounds
- Settings ranging from ancient Greece to modern USA
- Relationship networks between characters
- Character occupations and age demographics

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection for API calls

### Setup

```bash
git clone https://github.com/yourusername/custom-chatbot.git
cd custom-chatbot
pip install -r requirements.txt
```

### Configuration

1. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Quick Start

### Basic Usage

```python
from src.chatbot_engine import ChatbotEngine
from src.data_processor import CharacterDataProcessor

# Initialize the system
processor = CharacterDataProcessor()
df = processor.load_and_process_data("data/character_descriptions.csv")

chatbot = ChatbotEngine(df)

# Ask a question
answer = chatbot.answer_question("Who is Emily and what is her background?")
print(answer)
```

### Interactive Mode

```bash
python -m src.interactive_chatbot
```

### Web Interface

```bash
python -m src.gradio_interface
```

## Core Components

### 1. Data Processing
- **Character Analysis** - Extracts personality traits, relationships, occupations
- **Text Enhancement** - Creates enriched character profiles for better retrieval
- **Embedding Generation** - Creates semantic embeddings using OpenAI's text-embedding-ada-002

### 2. Retrieval System
- **Semantic Search** - Finds most relevant characters based on query embeddings
- **Question Enhancement** - Improves queries for better semantic matching
- **Context Selection** - Intelligently selects relevant context within token limits

### 3. Generation System
- **Custom Prompting** - Specialized prompts for character-specific responses
- **Multi-turn Conversations** - Maintains context across interactions
- **Response Formatting** - Structured and informative answers

### 4. Advanced Features
- **Character Recommendations** - Suggests characters based on desired traits
- **Character Comparisons** - Detailed analysis comparing multiple characters
- **Relationship Mapping** - Identifies and explains character connections

## Usage Examples

### Question Answering

```python
# Basic character information
chatbot.answer_question("Tell me about Emily's personality")

# Setting-based queries
chatbot.answer_question("Which characters are set in England?")

# Relationship queries
chatbot.answer_question("What relationships exist between characters?")

# Trait-based queries
chatbot.answer_question("Which characters are introverted?")
```

### Character Recommendations

```python
# Get character recommendations
recommendations = chatbot.recommend_characters(
    "strong female characters with leadership qualities", 
    n=3
)
print(recommendations)
```

### Character Comparisons

```python
# Compare two characters
comparison = chatbot.compare_characters("Emily", "Jack")
print(comparison)
```

## Architecture

```
custom-chatbot/
├── src/
│   ├── data_processor.py      # Data loading and enhancement
│   ├── embedding_manager.py   # Embedding generation and management
│   ├── retrieval_engine.py    # Semantic search and retrieval
│   ├── chatbot_engine.py      # Main chatbot logic
│   ├── interactive_chatbot.py # Command-line interface
│   └── gradio_interface.py    # Web interface
├── data/
│   └── character_descriptions.csv
├── examples/
│   └── usage_examples.py
└── tests/
    └── test_chatbot.py
```

## Performance Features

### Robust Error Handling
- Graceful fallbacks for API failures
- Alternative embedding strategies
- Error recovery mechanisms

### Optimization Techniques
- Batch processing for embeddings
- Token counting and management
- Efficient context selection
- Caching for repeated queries

### Scalability
- Modular architecture for easy extension
- Support for different embedding models
- Configurable parameters
- Easy dataset replacement

## Configuration

The system uses a configuration file `config/chatbot_config.yaml`:

```yaml
models:
  embedding_model: "text-embedding-ada-002"
  completion_model: "gpt-3.5-turbo-instruct"

generation:
  max_prompt_tokens: 1800
  max_answer_tokens: 300
  temperature: 0.7

retrieval:
  max_context_entries: 5
  similarity_threshold: 0.8

features:
  enable_recommendations: true
  enable_comparisons: true
  enable_interactive_mode: true
```

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Characters | 55 |
| Average Description Length | 261 characters |
| Unique Settings | 6 (USA, England, Ancient Greece, etc.) |
| Media Types | 7 (Play, Movie, Sitcom, etc.) |
| Extracted Personality Traits | 6 categories |
| Character Relationships | Network of connections |

### Character Distribution

**By Setting:**
- USA: 21 characters
- Ancient Greece: 10 characters  
- England: 8 characters
- Texas: 6 characters
- Australia: 5 characters
- Italy: 5 characters

**By Medium:**
- Play: 18 characters
- Reality Show: 8 characters
- Musical: 7 characters
- Movie: 6 characters

## Advanced Features

### Personality Analysis
Automatic extraction and categorization of character traits:
- **Positive traits**: kind, warm, friendly, caring
- **Extroverted traits**: outgoing, sociable, bubbly
- **Confident traits**: self-assured, bold, courageous
- **And more categories...**

### Relationship Networks
Identifies and maps relationships between characters:
- Family connections (parent-child, siblings)
- Romantic relationships (married, dating, engaged)
- Professional connections
- Friendships and social ties

### Smart Question Enhancement
Improves user queries for better semantic matching:
- Expands trait categories with examples
- Adds relationship context for relationship queries
- Maintains original intent while improving retrieval

## API Reference

### ChatbotEngine

```python
class ChatbotEngine:
    def answer_question(self, question: str) -> str
    def recommend_characters(self, query: str, n: int = 3) -> str
    def compare_characters(self, char1: str, char2: str) -> str
    def get_character_info(self, character_name: str) -> dict
```

### CharacterDataProcessor

```python
class CharacterDataProcessor:
    def load_and_process_data(self, file_path: str) -> pd.DataFrame
    def extract_personality_traits(self, description: str) -> list
    def extract_relationships(self, description: str) -> list
    def create_enhanced_text(self, character_data: dict) -> str
```

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

Run specific tests:

```bash
python -m pytest tests/test_chatbot.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -m "Add feature"`
6. Push to branch: `git push origin feature-name`
7. Submit a pull request

## Troubleshooting

### Common Issues

**OpenAI API errors**
- Check API key configuration
- Verify API quota and billing
- Ensure internet connectivity

**Empty or poor responses**
- Verify dataset is loaded correctly
- Check embedding generation
- Review question enhancement logic

**Performance issues**
- Reduce batch size for embeddings
- Adjust token limits
- Use caching for repeated queries

## Future Enhancements

- **Multi-language support** for international character datasets
- **Voice interface** with speech-to-text and text-to-speech
- **Visual character cards** with generated images
- **Expanded datasets** from books, anime, games
- **Real-time learning** from user interactions
- **Integration with external APIs** for richer character data

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenAI for providing powerful language models and embeddings
- The open-source community for NLP tools and techniques
- Contributors to the character descriptions dataset


---

*This project demonstrates advanced applications of retrieval augmented generation (RAG) for domain-specific conversational AI, showcasing the power of combining large language models with curated knowledge bases.*
