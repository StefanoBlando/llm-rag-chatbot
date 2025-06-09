"""Interactive command-line chatbot interface."""

import os
from dotenv import load_dotenv
from .data_processor import CharacterDataProcessor
from .chatbot_engine import ChatbotEngine

# Load environment variables
load_dotenv()


class InteractiveChatbot:
    """Interactive command-line interface for the character chatbot."""
    
    def __init__(self, data_file: str = "data/character_descriptions.csv"):
        """Initialize the interactive chatbot.
        
        Args:
            data_file: Path to the character descriptions CSV file
        """
        self.data_file = data_file
        self.chatbot = None
        self._setup_chatbot()
    
    def _setup_chatbot(self):
        """Set up the chatbot engine with data processing."""
        print("Initializing chatbot...")
        
        # Get API credentials
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        
        if not api_key:
            print("Error: OPENAI_API_KEY not found in environment variables.")
            print("Please set your OpenAI API key in a .env file or environment variable.")
            return
        
        try:
            # Process data
            processor = CharacterDataProcessor()
            df = processor.load_and_process_data(self.data_file)
            
            # Initialize chatbot
            self.chatbot = ChatbotEngine(df, api_key, api_base)
            
            print(f"Chatbot initialized with {len(df)} characters!")
            
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            print("Please check your data file and API configuration.")
    
    def run(self):
        """Run the interactive chatbot session."""
        if not self.chatbot:
            print("Chatbot not properly initialized. Exiting.")
            return
        
        print("\n" + "="*60)
        print("🤖 CHARACTER CHATBOT - Interactive Mode")
        print("="*60)
        
        self._show_help()
        
        while True:
            try:
                user_input = input("\n💬 Your query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thanks for chatting! Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                
                elif user_input.lower().startswith('recommend:'):
                    self._handle_recommendation(user_input)
                
                elif user_input.lower().startswith('compare:'):
                    self._handle_comparison(user_input)
                
                elif user_input.lower().startswith('info:'):
                    self._handle_character_info(user_input)
                
                elif user_input.lower().startswith('setting:'):
                    self._handle_setting_query(user_input)
                
                elif user_input.lower().startswith('medium:'):
                    self._handle_medium_query(user_input)
                
                elif user_input.lower().startswith('traits:'):
                    self._handle_traits_query(user_input)
                
                else:
                    self._handle_general_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'help' for assistance.")
    
    def _show_help(self):
        """Display help information."""
        print("\n📚 AVAILABLE COMMANDS:")
        print("-" * 40)
        print("🔍 QUESTIONS:")
        print("  Just type your question naturally!")
        print("  Examples:")
        print("  • Who is Emily and what is her background?")
        print("  • Which characters are in romantic relationships?")
        print("  • Tell me about characters in England.")
        print("  • What personality traits does Sarah have?")
        
        print("\n🎯 RECOMMENDATIONS:")
        print("  recommend: [description of desired traits]")
        print("  Examples:")
        print("  • recommend: strong leadership qualities")
        print("  • recommend: introverted and thoughtful characters")
        print("  • recommend: characters with family conflicts")
        
        print("\n⚖️ COMPARISONS:")
        print("  compare: [character1] and [character2]")
        print("  Examples:")
        print("  • compare: Emily and Jack")
        print("  • compare: Sarah and Alice")
        
        print("\n📋 SPECIFIC SEARCHES:")
        print("  info: [character name]     - Get detailed character info")
        print("  setting: [setting name]    - Characters from specific setting")
        print("  medium: [medium name]      - Characters from specific medium")
        print("  traits: [trait1, trait2]   - Characters with specific traits")
        
        print("\n🛠️ UTILITY:")
        print("  help     - Show this help message")
        print("  stats    - Show dataset statistics")
        print("  exit/quit/q - End the session")
    
    def _show_stats(self):
        """Display dataset statistics."""
        try:
            df = self.chatbot.df
            
            print("\n📊 DATASET STATISTICS:")
            print("-" * 40)
            print(f"📚 Total Characters: {len(df)}")
            print(f"📏 Avg Description Length: {df['Description_Length'].mean():.0f} chars")
            
            print(f"\n🌍 Settings ({len(df['Setting'].unique())}):")
            for setting, count in df['Setting'].value_counts().head(6).items():
                print(f"  • {setting}: {count} characters")
            
            print(f"\n🎭 Media Types ({len(df['Medium'].unique())}):")
            for medium, count in df['Medium'].value_counts().head(6).items():
                print(f"  • {medium}: {count} characters")
                
        except Exception as e:
            print(f"❌ Error showing stats: {e}")
    
    def _handle_recommendation(self, user_input: str):
        """Handle character recommendation requests."""
        query = user_input[len('recommend:'):].strip()
        if not query:
            print("❌ Please specify what kind of characters you're looking for.")
            print("Example: recommend: brave and loyal characters")
            return
        
        print(f"\n🔍 Finding characters matching: '{query}'...")
        try:
            result = self.chatbot.recommend_characters(query, n=3)
            print(f"\n✨ {result}")
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
    
    def _handle_comparison(self, user_input: str):
        """Handle character comparison requests."""
        comparison_query = user_input[len('compare:'):].strip()
        if ' and ' not in comparison_query:
            print("❌ Please use format: compare: Character1 and Character2")
            return
        
        try:
            chars = comparison_query.split(' and ')
            char1 = chars[0].strip()
            char2 = chars[1].strip()
            
            print(f"\n🔍 Comparing {char1} and {char2}...")
            result = self.chatbot.compare_characters(char1, char2)
            print(f"\n⚖️ {result}")
        except Exception as e:
            print(f"❌ Error generating comparison: {e}")
    
    def _handle_character_info(self, user_input: str):
        """Handle specific character information requests."""
        character_name = user_input[len('info:'):].strip()
        if not character_name:
            print("❌ Please specify a character name.")
            print("Example: info: Emily")
            return
        
        try:
            info = self.chatbot.get_character_info(character_name)
            if 'error' in info:
                print(f"❌ {info['error']}")
                # Show available characters
                available = self.chatbot.df['Name'].tolist()[:10]
                print(f"Available characters include: {', '.join(available)}...")
            else:
                print(f"\n📋 CHARACTER INFORMATION:")
                print(f"Name: {info['name']}")
                print(f"Medium: {info['medium']}")
                print(f"Setting: {info['setting']}")
                print(f"Description: {info['description']}")
        except Exception as e:
            print(f"❌ Error getting character info: {e}")
    
    def _handle_setting_query(self, user_input: str):
        """Handle setting-based character queries."""
        setting = user_input[len('setting:'):].strip()
        if not setting:
            print("❌ Please specify a setting.")
            print("Example: setting: England")
            return
        
        try:
            characters = self.chatbot.get_characters_by_setting(setting)
            if not characters:
                print(f"❌ No characters found in setting: {setting}")
                available_settings = self.chatbot.df['Setting'].unique().tolist()
                print(f"Available settings: {', '.join(available_settings)}")
            else:
                print(f"\n🌍 CHARACTERS IN {setting.upper()}:")
                for char in characters:
                    print(f"• {char['name']} ({char['medium']})")
        except Exception as e:
            print(f"❌ Error searching by setting: {e}")
    
    def _handle_medium_query(self, user_input: str):
        """Handle medium-based character queries."""
        medium = user_input[len('medium:'):].strip()
        if not medium:
            print("❌ Please specify a medium.")
            print("Example: medium: Play")
            return
        
        try:
            characters = self.chatbot.get_characters_by_medium(medium)
            if not characters:
                print(f"❌ No characters found in medium: {medium}")
                available_mediums = self.chatbot.df['Medium'].unique().tolist()
                print(f"Available mediums: {', '.join(available_mediums)}")
            else:
                print(f"\n🎭 CHARACTERS IN {medium.upper()}:")
                for char in characters:
                    print(f"• {char['name']} ({char['setting']})")
        except Exception as e:
            print(f"❌ Error searching by medium: {e}")
    
    def _handle_traits_query(self, user_input: str):
        """Handle trait-based character queries."""
        traits_str = user_input[len('traits:'):].strip()
        if not traits_str:
            print("❌ Please specify traits to search for.")
            print("Example: traits: brave, loyal, kind")
            return
        
        try:
            traits = [trait.strip() for trait in traits_str.split(',')]
            characters = self.chatbot.search_by_traits(traits)
            
            if not characters:
                print(f"❌ No characters found with traits: {', '.join(traits)}")
            else:
                print(f"\n🎯 CHARACTERS WITH TRAITS {', '.join(traits)}:")
                for char in characters[:5]:  # Limit to top 5
                    matching = ', '.join(char.get('matching_traits', []))
                    print(f"• {char['name']} - Matching: {matching}")
        except Exception as e:
            print(f"❌ Error searching by traits: {e}")
    
    def _handle_general_question(self, user_input: str):
        """Handle general questions using RAG."""
        print("\n🔍 Searching for answer...")
        try:
            answer = self.chatbot.answer_question(user_input)
            print(f"\n🤖 {answer}")
        except Exception as e:
            print(f"❌ Error generating answer: {e}")


def main():
    """Main function to run the interactive chatbot."""
    chatbot = InteractiveChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
