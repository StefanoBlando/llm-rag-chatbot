"""Data processing module for character descriptions dataset."""

import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Tuple, Any


class CharacterDataProcessor:
    """Processes and enhances character descriptions dataset."""
    
    def __init__(self):
        """Initialize the data processor with trait categories and patterns."""
        # Define age descriptors and patterns
        self.age_descriptors = {
            'young': '18-30',
            'early 20s': '20-24',
            'mid 20s': '24-27',
            'late 20s': '27-30',
            'early 30s': '30-34',
            'mid 30s': '34-37',
            'late 30s': '37-40',
            'middle-aged': '40-60',
            '40s': '40-50',
            '50s': '50-60',
            '60s': '60-70',
            'old': '65+',
            'elderly': '70+',
            'teen': '13-19',
            'teenage': '13-19',
            'adolescent': '13-19'
        }
        
        # List of common occupations to search for
        self.common_occupations = [
            'actor', 'actress', 'artist', 'author', 'banker', 'businessperson', 'businessman', 'businesswoman',
            'chef', 'dancer', 'doctor', 'engineer', 'farmer', 'journalist', 'lawyer', 'manager', 
            'musician', 'nurse', 'performer', 'professor', 'scientist', 'soldier', 'teacher', 'writer',
            'student', 'librarian', 'police', 'officer', 'guard', 'director', 'producer', 'assistant'
        ]
        
        # Define expanded trait categories and associated words
        self.trait_categories = {
            'Positive': ['kind', 'warm', 'friendly', 'caring', 'loving', 'generous', 'compassionate', 
                        'gentle', 'good-natured', 'optimistic', 'cheerful', 'happy', 'joyful', 'content'],
            'Negative': ['cruel', 'cold', 'unfriendly', 'mean', 'hateful', 'selfish', 'greedy', 
                        'harsh', 'bad-tempered', 'pessimistic', 'gloomy', 'sad', 'miserable', 'malicious'],
            'Extroverted': ['outgoing', 'sociable', 'talkative', 'lively', 'bubbly', 'chatty', 
                           'gregarious', 'expressive', 'energetic', 'vibrant', 'charismatic'],
            'Introverted': ['shy', 'quiet', 'reserved', 'introspective', 'thoughtful', 'private', 
                           'solitary', 'contemplative', 'reflective', 'withdrawn', 'secluded'],
            'Confident': ['confident', 'self-assured', 'secure', 'self-reliant', 'bold', 'fearless', 
                         'daring', 'courageous', 'brave', 'valiant', 'heroic', 'audacious'],
            'Insecure': ['insecure', 'self-doubting', 'uncertain', 'anxious', 'nervous', 'worried', 
                        'fearful', 'apprehensive', 'timid', 'hesitant', 'doubtful']
        }
        
        # Flatten trait list for easier checking
        self.all_traits = []
        for category, traits in self.trait_categories.items():
            for trait in traits:
                self.all_traits.append((trait, category))
    
    def load_and_process_data(self, file_path: str) -> pd.DataFrame:
        """Load and process the character descriptions dataset.
        
        Args:
            file_path: Path to the CSV file containing character descriptions
            
        Returns:
            Processed DataFrame with enhanced character information
        """
        # Load the character descriptions dataset
        df = pd.read_csv(file_path)
        
        # Calculate description lengths
        df["Description_Length"] = df["Description"].apply(len)
        
        # Process each character and extract enhanced information
        character_traits = {}
        character_relationships = {}
        age_groups = {}
        occupations = {}
        
        for i, row in df.iterrows():
            name = row['Name']
            description = row['Description'].lower()
            
            # Extract age group
            age_groups[name] = self._extract_age_group(description)
            
            # Extract occupation
            occupations[name] = self._extract_occupation(description)
            
            # Extract traits
            character_traits[name] = self._extract_personality_traits(description)
            
            # Extract relationships
            character_relationships[name] = self._extract_relationships(description, df['Name'].tolist(), name)
        
        # Store extracted information for use in text enhancement
        self.character_traits = character_traits
        self.character_relationships = character_relationships
        self.age_groups = age_groups
        self.occupations = occupations
        
        # Apply enhanced text formatting
        df["text"] = df.apply(self._create_enhanced_text, axis=1)
        
        return df
    
    def _extract_age_group(self, description: str) -> str:
        """Extract age group from character description.
        
        Args:
            description: Character description text
            
        Returns:
            Age group string or "Unknown"
        """
        for age_term, age_range in self.age_descriptors.items():
            if age_term in description:
                return age_range
        return "Unknown"
    
    def _extract_occupation(self, description: str) -> str:
        """Extract occupation from character description.
        
        Args:
            description: Character description text
            
        Returns:
            Occupation string or "Unknown"
        """
        for job in self.common_occupations:
            if job in description:
                # Find the complete phrase around the occupation
                match = re.search(r'(?:[a-z]+\s+){0,3}' + job + r'(?:\s+[a-z]+){0,3}', description)
                if match:
                    return match.group(0)
        return "Unknown"
    
    def _extract_personality_traits(self, description: str) -> List[str]:
        """Extract personality traits from character description.
        
        Args:
            description: Character description text
            
        Returns:
            List of personality traits with categories
        """
        traits = []
        for trait, category in self.all_traits:
            if trait in description:
                traits.append(f"{trait} ({category})")
        return traits
    
    def _extract_relationships(self, description: str, all_names: List[str], current_name: str) -> List[str]:
        """Extract relationships from character description.
        
        Args:
            description: Character description text
            all_names: List of all character names in dataset
            current_name: Name of current character
            
        Returns:
            List of relationship descriptions
        """
        relationships = []
        
        # Look for relationship keywords
        relationship_keywords = ['married', 'husband', 'wife', 'partner', 'boyfriend', 'girlfriend',
                               'relationship', 'dating', 'engaged', 'lover', 'spouse',
                               'father', 'mother', 'parent', 'son', 'daughter', 'child',
                               'brother', 'sister', 'sibling', 'family', 'friend']
        
        for keyword in relationship_keywords:
            if keyword in description:
                # Find the sentence containing this keyword
                sentences = description.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        # Clean up the sentence
                        clean_sentence = sentence.strip()
                        if clean_sentence and clean_sentence not in relationships:
                            relationships.append(clean_sentence)
        
        # Look for direct character name mentions
        for other_name in all_names:
            if other_name.lower() in description and other_name.lower() != current_name.lower():
                sentences = description.split('.')
                for sentence in sentences:
                    if other_name.lower() in sentence.lower():
                        clean_sentence = sentence.strip()
                        if clean_sentence and clean_sentence not in relationships:
                            relationships.append(f"Connected to {other_name}: {clean_sentence}")
        
        return relationships
    
    def _create_enhanced_text(self, row: pd.Series) -> str:
        """Create enhanced text representation of a character.
        
        Args:
            row: DataFrame row containing character information
            
        Returns:
            Enhanced text description with all extracted information
        """
        name = row['Name']
        
        # Base information
        enhanced_text = f"Name: {name}\n"
        enhanced_text += f"Description: {row['Description']}\n"
        enhanced_text += f"Medium: {row['Medium']}\n"
        enhanced_text += f"Setting: {row['Setting']}"
        
        # Add age information
        if name in self.age_groups and self.age_groups[name] != "Unknown":
            enhanced_text += f"\nAge: {self.age_groups[name]}"
        
        # Add occupation information
        if name in self.occupations and self.occupations[name] != "Unknown":
            enhanced_text += f"\nOccupation: {self.occupations[name]}"
        
        # Add personality traits
        if name in self.character_traits and self.character_traits[name]:
            enhanced_text += f"\nPersonality Traits: {', '.join(self.character_traits[name])}"
        
        # Add relationships
        if name in self.character_relationships and self.character_relationships[name]:
            enhanced_text += f"\nRelationship Context: {'. '.join(self.character_relationships[name])}"
        
        return enhanced_text
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_characters': len(df),
            'avg_description_length': df['Description_Length'].mean(),
            'min_description_length': df['Description_Length'].min(),
            'max_description_length': df['Description_Length'].max(),
            'settings_distribution': df['Setting'].value_counts().to_dict(),
            'medium_distribution': df['Medium'].value_counts().to_dict(),
            'most_detailed_characters': df.nlargest(5, 'Description_Length')[['Name', 'Description_Length']].to_dict('records')
        }
        
        return stats
    
    def analyze_character_networks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze character relationship networks.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary containing network analysis results
        """
        # Count mentions of each character in other character descriptions
        character_mentions = Counter()
        
        for name in df['Name']:
            for other_name in df['Name']:
                if name != other_name:
                    # Check if this character is mentioned in others' descriptions
                    other_descriptions = df[df['Name'] == other_name]['Description'].values
                    for desc in other_descriptions:
                        if name.lower() in desc.lower():
                            character_mentions[name] += 1
        
        # Find most connected characters
        most_connected = character_mentions.most_common(5)
        
        return {
            'character_mentions': dict(character_mentions),
            'most_connected_characters': most_connected,
            'total_connections': sum(character_mentions.values())
        }
