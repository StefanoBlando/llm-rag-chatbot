"""Main chatbot engine combining all components."""

import openai
import pandas as pd
import tiktoken
from typing import List, Dict, Any, Optional
from .embedding_manager import EmbeddingManager
from .data_processor import CharacterDataProcessor


class ChatbotEngine:
    """Main chatbot engine for character-based conversations."""
    
    def __init__(self, df: pd.DataFrame, api_key: str, api_base: str = None):
        """Initialize the chatbot engine.
        
        Args:
            df: Processed DataFrame with character data
            api_key: OpenAI API key
            api_base: Optional custom API base URL
        """
        self.df = df
        self.completion_model = "gpt-3.5-turbo-instruct"
        
        # Configure OpenAI
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(api_key, api_base)
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Generate embeddings if not already present
        if 'embeddings' not in self.df.columns:
            self._generate_embeddings()
    
    def _generate_embeddings(self):
        """Generate embeddings for all character texts."""
        texts = self.df['text'].tolist()
        embeddings = self.embedding_manager.generate_embeddings_batch(texts)
        self.df['embeddings'] = embeddings
    
    def answer_question(self, question: str, max_prompt_tokens: int = 1800, max_answer_tokens: int = 300) -> str:
        """Answer a question about characters using RAG.
        
        Args:
            question: User's question
            max_prompt_tokens: Maximum tokens for the prompt
            max_answer_tokens: Maximum tokens for the answer
            
        Returns:
            Generated answer string
        """
        prompt = self._create_prompt(question, max_prompt_tokens)
        
        try:
            response = openai.Completion.create(
                model=self.completion_model,
                prompt=prompt,
                max_tokens=max_answer_tokens,
                temperature=0.7
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I apologize, but I'm having trouble generating an answer right now. Please try again."
    
    def _create_prompt(self, question: str, max_token_count: int) -> str:
        """Create a prompt with relevant context from the character dataset.
        
        Args:
            question: User's question
            max_token_count: Maximum number of tokens for the prompt
            
        Returns:
            Formatted prompt string
        """
        prompt_template = """
You are a Character Information Assistant with deep knowledge of fictional characters.
Answer the question based on the context below. If the question can't be answered based on the 
context, say "I don't know" and suggest a better question the user might ask about fictional characters.

The context below contains detailed information about fictional characters including:
- Their names, descriptions, and backgrounds
- The medium they appear in (like plays, films, etc.)
- Their setting and time period
- Their personality traits and characteristics
- Their relationships with other characters
- Their age and occupation

Context: 

{}

---

Question: {}
Answer:

Please provide a detailed, thoughtful response. If there are multiple relevant characters, compare and contrast them.
If the question is about relationships, explain the nature of these relationships clearly.
"""
        
        # Count tokens in the template and question
        current_token_count = len(self.tokenizer.encode(prompt_template)) + len(self.tokenizer.encode(question))
        
        # Get relevant context
        context = self._get_relevant_context(question, max_token_count - current_token_count)
        
        return prompt_template.format(context, question)
    
    def _get_relevant_context(self, question: str, available_tokens: int) -> str:
        """Get relevant context for the question within token limits.
        
        Args:
            question: User's question
            available_tokens: Available tokens for context
            
        Returns:
            Formatted context string
        """
        # Find most similar documents
        relevant_df = self.embedding_manager.find_most_similar(question, self.df, top_k=10)
        
        context_parts = []
        current_tokens = 0
        
        for _, row in relevant_df.iterrows():
            text = row['text']
            text_tokens = len(self.tokenizer.encode(text))
            
            if current_tokens + text_tokens <= available_tokens:
                context_parts.append(text)
                current_tokens += text_tokens
            else:
                break
        
        return "\n\n###\n\n".join(context_parts)
    
    def recommend_characters(self, query: str, n: int = 3) -> str:
        """Recommend characters based on a user query.
        
        Args:
            query: Description of desired character traits
            n: Number of recommendations to return
            
        Returns:
            Formatted recommendation string
        """
        # Add prefix to help the embedding model understand this is a recommendation query
        recommendation_query = f"Find characters who are {query}"
        
        # Get characters sorted by relevance
        relevant_df = self.embedding_manager.find_most_similar(recommendation_query, self.df, top_k=n)
        
        # Format the recommendations
        result = f"Based on '{query}', here are some character recommendations:\n\n"
        
        for i, (_, row) in enumerate(relevant_df.iterrows(), 1):
            result += f"{i}. **{row['Name']}**\n"
            result += f"   Medium: {row['Medium']}\n"
            result += f"   Setting: {row['Setting']}\n"
            
            # Add a brief highlight from the description
            brief = row['Description'][:150] + "..." if len(row['Description']) > 150 else row['Description']
            result += f"   Description: {brief}\n\n"
        
        return result
    
    def compare_characters(self, char1: str, char2: str) -> str:
        """Compare two characters from the dataset.
        
        Args:
            char1: Name of first character
            char2: Name of second character
            
        Returns:
            Detailed comparison string
        """
        # Find the characters in the dataset
        char1_data = self.df[self.df['Name'].str.lower() == char1.lower()]
        char2_data = self.df[self.df['Name'].str.lower() == char2.lower()]
        
        if len(char1_data) == 0:
            return f"Character '{char1}' not found in the dataset."
        if len(char2_data) == 0:
            return f"Character '{char2}' not found in the dataset."
        
        # Create a comparison prompt
        comparison_prompt = f"""
Compare and contrast these two characters:

Character 1:
{char1_data['text'].values[0]}

Character 2:
{char2_data['text'].values[0]}

Compare them in terms of:
1. Personality traits and temperament
2. Backgrounds and origins
3. Motivations and goals
4. Relationships with others
5. Key similarities and differences

Comparison:
"""
        
        # Get the comparison
        try:
            response = openai.Completion.create(
                model=self.completion_model,
                prompt=comparison_prompt,
                max_tokens=400,
                temperature=0.7
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating comparison: {e}")
            return "I'm having trouble generating the comparison right now. Please try again."
    
    def get_character_info(self, character_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Dictionary with character information
        """
        char_data = self.df[self.df['Name'].str.lower() == character_name.lower()]
        
        if len(char_data) == 0:
            return {"error": f"Character '{character_name}' not found in dataset"}
        
        row = char_data.iloc[0]
        
        return {
            "name": row['Name'],
            "description": row['Description'],
            "medium": row['Medium'],
            "setting": row['Setting'],
            "enhanced_info": row['text'],
            "description_length": row.get('Description_Length', len(row['Description']))
        }
    
    def get_characters_by_setting(self, setting: str) -> List[Dict[str, str]]:
        """Get all characters from a specific setting.
        
        Args:
            setting: Setting name
            
        Returns:
            List of character dictionaries
        """
        setting_chars = self.df[self.df['Setting'].str.lower() == setting.lower()]
        
        characters = []
        for _, row in setting_chars.iterrows():
            characters.append({
                "name": row['Name'],
                "description": row['Description'],
                "medium": row['Medium']
            })
        
        return characters
    
    def get_characters_by_medium(self, medium: str) -> List[Dict[str, str]]:
        """Get all characters from a specific medium.
        
        Args:
            medium: Medium name (e.g., "Play", "Movie")
            
        Returns:
            List of character dictionaries
        """
        medium_chars = self.df[self.df['Medium'].str.lower() == medium.lower()]
        
        characters = []
        for _, row in medium_chars.iterrows():
            characters.append({
                "name": row['Name'],
                "description": row['Description'],
                "setting": row['Setting']
            })
        
        return characters
    
    def search_by_traits(self, traits: List[str]) -> List[Dict[str, str]]:
        """Search for characters with specific personality traits.
        
        Args:
            traits: List of trait keywords to search for
            
        Returns:
            List of matching character dictionaries
        """
        matching_chars = []
        
        for _, row in self.df.iterrows():
            description_lower = row['Description'].lower()
            text_lower = row['text'].lower()
            
            # Check if any of the traits appear in the character's information
            if any(trait.lower() in description_lower or trait.lower() in text_lower for trait in traits):
                matching_chars.append({
                    "name": row['Name'],
                    "description": row['Description'],
                    "medium": row['Medium'],
                    "setting": row['Setting'],
                    "matching_traits": [trait for trait in traits if trait.lower() in text_lower]
                })
        
        return matching_chars
    
    def get_basic_answer(self, question: str, max_answer_tokens: int = 300) -> str:
        """Get a basic answer without using the custom knowledge base.
        
        Args:
            question: User's question
            max_answer_tokens: Maximum tokens for the answer
            
        Returns:
            Basic answer string
        """
        prompt = f"Question: {question}\nAnswer:"
        
        try:
            response = openai.Completion.create(
                model=self.completion_model,
                prompt=prompt,
                max_tokens=max_answer_tokens,
                temperature=0.7
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"Error generating basic answer: {e}")
            return "I'm having trouble generating an answer right now."
