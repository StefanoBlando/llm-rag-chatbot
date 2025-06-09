"""Custom Chatbot with RAG - Main package."""

__version__ = "0.1.0"
__author__ = "Stefano Blando"
__description__ = "Intelligent chatbot system with RAG for fictional character conversations"

# Import main components for easy access
from .data_processor import CharacterDataProcessor
from .embedding_manager import EmbeddingManager
from .chatbot_engine import ChatbotEngine
from .interactive_chatbot import InteractiveChatbot

__all__ = [
    "CharacterDataProcessor",
    "EmbeddingManager", 
    "ChatbotEngine",
    "InteractiveChatbot"
]
