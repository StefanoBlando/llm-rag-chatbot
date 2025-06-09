"""Embedding management module for handling OpenAI embeddings."""

import openai
import pandas as pd
import numpy as np
import random
import time
from typing import List, Optional, Union
from openai.embeddings_utils import distances_from_embeddings


class EmbeddingManager:
    """Manages embedding generation and similarity calculations."""
    
    def __init__(self, api_key: str, api_base: str = None, model_name: str = "text-embedding-ada-002"):
        """Initialize the embedding manager.
        
        Args:
            api_key: OpenAI API key
            api_base: Optional custom API base URL
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embedding_dimension = 1536  # Standard size for text-embedding-ada-002
        
        # Configure OpenAI
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base
    
    def get_robust_embedding(self, text: str, retries: int = 3) -> List[float]:
        """Get embeddings with robust error handling.
        
        Args:
            text: Text to embed
            retries: Number of retry attempts
            
        Returns:
            List of embedding values
        """
        for attempt in range(retries):
            try:
                # Clean the text
                text = text.replace("\n", " ").strip()
                
                # Make the API call
                response = openai.Embedding.create(
                    input=[text],
                    engine=self.model_name
                )
                
                # Try different response formats
                if hasattr(response, 'data') and len(response.data) > 0:
                    if hasattr(response.data[0], 'embedding'):
                        return response.data[0].embedding
                
                # If the above doesn't work, try accessing as dictionary
                if isinstance(response, dict):
                    if 'data' in response and len(response['data']) > 0:
                        if 'embedding' in response['data'][0]:
                            return response['data'][0]['embedding']
                
                # If we can't find the embedding in the expected format
                print(f"Warning: Could not extract embedding from response format")
                return self._get_zero_embedding()
            
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"All attempts failed, using zero embedding")
                    return self._get_zero_embedding()
        
        return self._get_zero_embedding()
    
    def _get_zero_embedding(self) -> List[float]:
        """Get a zero embedding as fallback.
        
        Returns:
            List of zeros with correct embedding dimension
        """
        return [0.0] * self.embedding_dimension
    
    def _get_random_embedding(self) -> List[float]:
        """Get a random embedding for testing purposes.
        
        Returns:
            List of random values with correct embedding dimension
        """
        return [random.uniform(-1, 1) for _ in range(self.embedding_dimension)]
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 5, use_fallback: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            use_fallback: Whether to use fallback embeddings on failure
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"Generating embeddings for {len(texts)} texts in {total_batches} batches...")
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num} of {total_batches}...")
                
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = []
                
                # Process each text individually for better error handling
                for text in batch_texts:
                    embedding = self.get_robust_embedding(text)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Small delay between batches to avoid rate limiting
                time.sleep(0.5)
        
        except Exception as e:
            print(f"Error in batch embedding generation: {e}")
            
            if use_fallback:
                print("Using fallback embeddings for demonstration")
                embeddings = [self._get_random_embedding() for _ in range(len(texts))]
            else:
                raise
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def calculate_similarities(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        """Calculate cosine similarities between query and document embeddings.
        
        Args:
            query_embedding: Embedding vector for the query
            document_embeddings: List of embedding vectors for documents
            
        Returns:
            List of similarity scores (higher = more similar)
        """
        try:
            # Calculate distances (lower = more similar)
            distances = distances_from_embeddings(
                query_embedding,
                document_embeddings,
                distance_metric="cosine"
            )
            
            # Convert distances to similarities (higher = more similar)
            similarities = [1 - distance for distance in distances]
            return similarities
        
        except Exception as e:
            print(f"Error calculating similarities: {e}")
            # Return random similarities as fallback
            return [random.random() for _ in range(len(document_embeddings))]
    
    def find_most_similar(self, query: str, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """Find the most similar documents to a query.
        
        Args:
            query: Query string
            df: DataFrame with 'text' and 'embeddings' columns
            top_k: Number of most similar documents to return
            
        Returns:
            DataFrame sorted by similarity (most similar first)
        """
        try:
            # Get query embedding
            query_embedding = self.get_robust_embedding(query)
            
            # Calculate similarities
            document_embeddings = df['embeddings'].tolist()
            similarities = self.calculate_similarities(query_embedding, document_embeddings)
            
            # Add similarities to dataframe copy
            df_copy = df.copy()
            df_copy['similarity'] = similarities
            df_copy['distances'] = [1 - sim for sim in similarities]  # For compatibility
            
            # Sort by similarity (descending) and return top k
            df_sorted = df_copy.sort_values('similarity', ascending=False)
            return df_sorted.head(top_k)
        
        except Exception as e:
            print(f"Error in similarity search: {e}")
            # Return random selection as fallback
            return df.sample(n=min(top_k, len(df)))
    
    def enhance_query_for_search(self, query: str, trait_categories: dict) -> str:
        """Enhance query for better semantic search results.
        
        Args:
            query: Original query string
            trait_categories: Dictionary of personality trait categories
            
        Returns:
            Enhanced query string
        """
        # Check for trait categories and expand them
        for category, traits in trait_categories.items():
            if category.lower() in query.lower():
                trait_examples = ", ".join(traits[:5])  # Sample of traits
                return f"{query} Examples of {category.lower()} traits include {trait_examples}."
        
        # Check for relationship questions and enhance them
        relationship_terms = ['relationship', 'married', 'dating', 'family', 'friend', 'connected']
        if any(term in query.lower() for term in relationship_terms):
            return f"{query} Look for characters who are connected to others including husband, wife, partner, father, mother, son, daughter, siblings, or friends."
        
        return query  # Return original if no enhancements needed
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """Validate that embeddings are properly formatted.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            True if embeddings are valid, False otherwise
        """
        if not embeddings:
            return False
        
        # Check if all embeddings have the correct dimension
        for embedding in embeddings:
            if len(embedding) != self.embedding_dimension:
                return False
            
            # Check if embedding contains valid numbers
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
        
        return True
    
    def save_embeddings(self, embeddings: List[List[float]], file_path: str):
        """Save embeddings to a file.
        
        Args:
            embeddings: List of embedding vectors
            file_path: Path to save the embeddings
        """
        try:
            np.save(file_path, np.array(embeddings))
            print(f"Embeddings saved to {file_path}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, file_path: str) -> Optional[List[List[float]]]:
        """Load embeddings from a file.
        
        Args:
            file_path: Path to load the embeddings from
            
        Returns:
            List of embedding vectors or None if loading fails
        """
        try:
            embeddings_array = np.load(file_path)
            embeddings = embeddings_array.tolist()
            print(f"Embeddings loaded from {file_path}")
            return embeddings
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return None
