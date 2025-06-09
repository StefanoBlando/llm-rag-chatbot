#!/usr/bin/env python3
"""Usage examples for the Custom Chatbot with RAG."""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import CharacterDataProcessor
from src.chatbot_engine import ChatbotEngine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_1_basic_setup_and_questions():
    """Example 1: Basic setup and question answering."""
    print("="*50)
    print("EXAMPLE 1: Basic Setup and Question Answering")
    print("="*50)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your environment variables")
        return
    
    # Initialize components
    print("📚 Loading and processing character data...")
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    
    print(f"✅ Loaded {len(df)} characters")
    
    # Initialize chatbot
    print("🤖 Initializing chatbot engine...")
    chatbot = ChatbotEngine(df, api_key)
    
    # Ask some questions
    questions = [
        "Who is Emily and what is her background?",
        "Which characters are set in England?",
        "Tell me about characters with strong personalities."
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        answer = chatbot.answer_question(question)
        print(f"🤖 Answer: {answer}")
        print("-" * 50)


def example_2_character_recommendations():
    """Example 2: Character recommendation system."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Character Recommendation System")
    print("="*50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your environment variables")
        return
    
    # Setup
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    chatbot = ChatbotEngine(df, api_key)
    
    # Test recommendations
    recommendation_queries = [
        "strong female leaders with determination",
        "introverted characters who prefer solitude",
        "characters dealing with family relationships"
    ]
    
    for i, query in enumerate(recommendation_queries, 1):
        print(f"\n🎯 Recommendation {i}: '{query}'")
        recommendations = chatbot.recommend_characters(query, n=3)
        print(recommendations)
        print("-" * 50)


def example_3_character_comparisons():
    """Example 3: Character comparison functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Character Comparison")
    print("="*50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your environment variables")
        return
    
    # Setup
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    chatbot = ChatbotEngine(df, api_key)
    
    # Character pairs to compare
    comparisons = [
        ("Emily", "Jack"),
        ("Sarah", "Alice")
    ]
    
    for i, (char1, char2) in enumerate(comparisons, 1):
        print(f"\n⚖️ Comparison {i}: {char1} vs {char2}")
        comparison = chatbot.compare_characters(char1, char2)
        print(comparison)
        print("-" * 50)


def example_4_data_analysis():
    """Example 4: Dataset analysis and statistics."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Dataset Analysis")
    print("="*50)
    
    # Data processing and analysis
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    
    # Get statistics
    stats = processor.get_dataset_statistics(df)
    
    print("📊 Dataset Statistics:")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Average description length: {stats['avg_description_length']:.0f} characters")
    print(f"Min/Max description length: {stats['min_description_length']}/{stats['max_description_length']}")
    
    print("\n🌍 Settings distribution:")
    for setting, count in stats['settings_distribution'].items():
        print(f"  • {setting}: {count} characters")
    
    print("\n🎭 Medium distribution:")
    for medium, count in stats['medium_distribution'].items():
        print(f"  • {medium}: {count} characters")
    
    print("\n📚 Most detailed characters:")
    for char in stats['most_detailed_characters']:
        print(f"  • {char['Name']}: {char['Description_Length']} characters")
    
    # Network analysis
    network_stats = processor.analyze_character_networks(df)
    print(f"\n🔗 Character connections:")
    print(f"Total connections found: {network_stats['total_connections']}")
    print("Most connected characters:")
    for char, connections in network_stats['most_connected_characters']:
        print(f"  • {char}: {connections} mentions")


def example_5_advanced_search():
    """Example 5: Advanced search functionality."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Advanced Search Features")
    print("="*50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your environment variables")
        return
    
    # Setup
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    chatbot = ChatbotEngine(df, api_key)
    
    # Search by setting
    print("🌍 Characters in England:")
    england_chars = chatbot.get_characters_by_setting("England")
    for char in england_chars[:5]:  # Show first 5
        print(f"  • {char['name']} ({char['medium']})")
    
    # Search by medium
    print("\n🎭 Characters in Plays:")
    play_chars = chatbot.get_characters_by_medium("Play")
    for char in play_chars[:5]:  # Show first 5
        print(f"  • {char['name']} ({char['setting']})")
    
    # Search by traits
    print("\n🎯 Characters with specific traits:")
    trait_searches = [
        ["confident", "brave"],
        ["shy", "quiet"],
        ["warm", "caring"]
    ]
    
    for traits in trait_searches:
        print(f"\nTraits: {', '.join(traits)}")
        trait_chars = chatbot.search_by_traits(traits)
        for char in trait_chars[:3]:  # Show first 3
            matching = ', '.join(char.get('matching_traits', []))
            print(f"  • {char['name']} - Matching: {matching}")


def example_6_basic_vs_custom_comparison():
    """Example 6: Compare basic LLM responses vs custom RAG responses."""
    print("\n" + "="*50)
    print("EXAMPLE 6: Basic vs Custom Response Comparison")
    print("="*50)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your environment variables")
        return
    
    # Setup
    processor = CharacterDataProcessor()
    df = processor.load_and_process_data("data/character_descriptions.csv")
    chatbot = ChatbotEngine(df, api_key)
    
    # Test questions that benefit from the custom knowledge base
    test_questions = [
        "What is Emily's relationship with Alice?",
        "Which characters are aspiring actors?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        
        print("\n🔹 Basic LLM Response:")
        basic_answer = chatbot.get_basic_answer(question)
        print(basic_answer)
        
        print("\n🔹 Custom RAG Response:")
        custom_answer = chatbot.answer_question(question)
        print(custom_answer)
        
        print("-" * 50)


def main():
    """Run all examples."""
    print("🤖 Custom Chatbot with RAG - Usage Examples")
    print("=" * 60)
    
    try:
        example_1_basic_setup_and_questions()
        example_2_character_recommendations()
        example_3_character_comparisons()
        example_4_data_analysis()
        example_5_advanced_search()
        example_6_basic_vs_custom_comparison()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("For interactive usage, try:")
        print("  • python -m src.interactive_chatbot")
        print("  • python -m src.gradio_interface")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Install requirements: pip install -r requirements.txt")
        print("3. Ensure data/character_descriptions.csv exists")


if __name__ == "__main__":
    main()
