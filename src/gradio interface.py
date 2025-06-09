"""Gradio web interface for the character chatbot."""

import gradio as gr
import os
from dotenv import load_dotenv
from .data_processor import CharacterDataProcessor
from .chatbot_engine import ChatbotEngine

# Load environment variables
load_dotenv()


class ChatbotWebInterface:
    """Web interface for the character chatbot using Gradio."""
    
    def __init__(self, data_file: str = "data/character_descriptions.csv"):
        """Initialize the web interface.
        
        Args:
            data_file: Path to the character descriptions CSV file
        """
        self.chatbot = None
        self.df = None
        self._setup_chatbot(data_file)
    
    def _setup_chatbot(self, data_file: str):
        """Set up the chatbot engine."""
        try:
            # Get API credentials
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = os.getenv("OPENAI_API_BASE")
            
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            # Process data
            processor = CharacterDataProcessor()
            self.df = processor.load_and_process_data(data_file)
            
            # Initialize chatbot
            self.chatbot = ChatbotEngine(self.df, api_key, api_base)
            
        except Exception as e:
            print(f"Error setting up chatbot: {e}")
            self.chatbot = None
    
    def chat_response(self, message: str, history: list) -> tuple:
        """Generate chatbot response for the chat interface.
        
        Args:
            message: User's message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not self.chatbot:
            response = "❌ Chatbot not available. Please check configuration."
            history.append([message, response])
            return "", history
        
        try:
            # Handle different types of queries
            if message.lower().startswith('recommend:'):
                query = message[len('recommend:'):].strip()
                response = self.chatbot.recommend_characters(query, n=3)
            elif message.lower().startswith('compare:') and ' and ' in message.lower():
                comparison_query = message[len('compare:'):].strip()
                chars = comparison_query.split(' and ')
                char1 = chars[0].strip()
                char2 = chars[1].strip()
                response = self.chatbot.compare_characters(char1, char2)
            else:
                response = self.chatbot.answer_question(message)
            
            history.append([message, response])
            return "", history
            
        except Exception as e:
            error_response = f"❌ Error: {str(e)}"
            history.append([message, error_response])
            return "", history
    
    def get_character_recommendations(self, traits: str, num_recommendations: int) -> str:
        """Get character recommendations based on traits.
        
        Args:
            traits: Description of desired character traits
            num_recommendations: Number of recommendations to return
            
        Returns:
            Formatted recommendations string
        """
        if not self.chatbot:
            return "❌ Chatbot not available."
        
        if not traits.strip():
            return "Please enter traits or characteristics you're looking for."
        
        try:
            return self.chatbot.recommend_characters(traits, n=num_recommendations)
        except Exception as e:
            return f"❌ Error generating recommendations: {str(e)}"
    
    def compare_two_characters(self, char1: str, char2: str) -> str:
        """Compare two characters.
        
        Args:
            char1: Name of first character
            char2: Name of second character
            
        Returns:
            Comparison result string
        """
        if not self.chatbot:
            return "❌ Chatbot not available."
        
        if not char1.strip() or not char2.strip():
            return "Please enter both character names to compare."
        
        try:
            return self.chatbot.compare_characters(char1, char2)
        except Exception as e:
            return f"❌ Error comparing characters: {str(e)}"
    
    def get_character_info(self, character_name: str) -> str:
        """Get detailed information about a character.
        
        Args:
            character_name: Name of the character
            
        Returns:
            Character information string
        """
        if not self.chatbot:
            return "❌ Chatbot not available."
        
        if not character_name.strip():
            return "Please enter a character name."
        
        try:
            info = self.chatbot.get_character_info(character_name)
            if 'error' in info:
                available = self.df['Name'].tolist()[:10]
                return f"{info['error']}\n\nAvailable characters include: {', '.join(available)}..."
            else:
                return f"**{info['name']}**\n\n**Medium:** {info['medium']}\n**Setting:** {info['setting']}\n\n**Description:** {info['description']}"
        except Exception as e:
            return f"❌ Error getting character info: {str(e)}"
    
    def get_dataset_stats(self) -> str:
        """Get dataset statistics.
        
        Returns:
            Formatted statistics string
        """
        if self.df is None:
            return "❌ Dataset not available."
        
        try:
            stats = f"**Dataset Statistics:**\n\n"
            stats += f"📚 Total Characters: {len(self.df)}\n"
            stats += f"📏 Average Description Length: {self.df['Description_Length'].mean():.0f} characters\n\n"
            
            stats += f"**Settings:**\n"
            for setting, count in self.df['Setting'].value_counts().head(6).items():
                stats += f"• {setting}: {count} characters\n"
            
            stats += f"\n**Media Types:**\n"
            for medium, count in self.df['Medium'].value_counts().head(6).items():
                stats += f"• {medium}: {count} characters\n"
            
            return stats
        except Exception as e:
            return f"❌ Error getting statistics: {str(e)}"
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Character Chatbot",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as demo:
            
            gr.Markdown("""
            # 🤖 Character Chatbot with RAG
            
            **Intelligent chatbot for exploring fictional characters using Retrieval Augmented Generation**
            
            This chatbot can answer questions about 55 fictional characters from various media, 
            provide recommendations, and compare characters based on their traits and backgrounds.
            """)
            
            with gr.Tabs():
                # Main Chat Tab
                with gr.TabItem("💬 Chat"):
                    gr.Markdown("### Ask me anything about the characters!")
                    
                    chatbot = gr.Chatbot(
                        height=400,
                        show_label=False,
                        avatar_images=("🧑‍💻", "🤖")
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask about characters, their relationships, traits, settings...",
                            container=False,
                            scale=7
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
                    
                    gr.Examples(
                        examples=[
                            "Who is Emily and what is her background?",
                            "Which characters are in romantic relationships?",
                            "Tell me about characters set in England",
                            "Which characters are introverted?",
                            "recommend: strong female characters with leadership qualities",
                            "compare: Emily and Jack"
                        ],
                        inputs=msg
                    )
                
                # Recommendations Tab
                with gr.TabItem("🎯 Recommendations"):
                    gr.Markdown("### Get character recommendations based on specific traits")
                    
                    with gr.Row():
                        with gr.Column():
                            traits_input = gr.Textbox(
                                label="Describe the type of character you're looking for",
                                placeholder="e.g., brave leaders, introverted artists, characters with family conflicts",
                                lines=2
                            )
                            num_recs = gr.Slider(
                                minimum=1, maximum=5, value=3, step=1,
                                label="Number of recommendations"
                            )
                            rec_btn = gr.Button("Get Recommendations", variant="primary")
                        
                        with gr.Column():
                            rec_output = gr.Markdown(label="Recommendations")
                    
                    gr.Examples(
                        examples=[
                            ["strong leadership qualities", 3],
                            ["introverted and thoughtful characters", 2],
                            ["characters with internal conflicts", 3],
                            ["friendly and outgoing personalities", 2],
                            ["characters in romantic relationships", 3]
                        ],
                        inputs=[traits_input, num_recs]
                    )
                
                # Comparison Tab
                with gr.TabItem("⚖️ Compare"):
                    gr.Markdown("### Compare two characters in detail")
                    
                    with gr.Row():
                        with gr.Column():
                            char1_input = gr.Textbox(
                                label="First Character",
                                placeholder="e.g., Emily"
                            )
                            char2_input = gr.Textbox(
                                label="Second Character", 
                                placeholder="e.g., Jack"
                            )
                            compare_btn = gr.Button("Compare Characters", variant="primary")
                        
                        with gr.Column():
                            compare_output = gr.Markdown(label="Comparison")
                    
                    gr.Examples(
                        examples=[
                            ["Emily", "Jack"],
                            ["Sarah", "Alice"],
                            ["Tom", "George"]
                        ],
                        inputs=[char1_input, char2_input]
                    )
                
                # Character Info Tab
                with gr.TabItem("📋 Character Info"):
                    gr.Markdown("### Get detailed information about a specific character")
                    
                    with gr.Row():
                        with gr.Column():
                            char_name_input = gr.Textbox(
                                label="Character Name",
                                placeholder="e.g., Emily"
                            )
                            info_btn = gr.Button("Get Character Info", variant="primary")
                        
                        with gr.Column():
                            info_output = gr.Markdown(label="Character Information")
                
                # Dataset Stats Tab
                with gr.TabItem("📊 Dataset"):
                    gr.Markdown("### Dataset Overview and Statistics")
                    
                    stats_btn = gr.Button("Show Dataset Statistics", variant="primary")
                    stats_output = gr.Markdown()
                    
                    # Show stats immediately
                    demo.load(self.get_dataset_stats, outputs=stats_output)
            
            # Event handlers
            msg.submit(self.chat_response, [msg, chatbot], [msg, chatbot])
            submit_btn.click(self.chat_response, [msg, chatbot], [msg, chatbot])
            
            rec_btn.click(
                self.get_character_recommendations,
                [traits_input, num_recs],
                rec_output
            )
            
            compare_btn.click(
                self.compare_two_characters,
                [char1_input, char2_input],
                compare_output
            )
            
            info_btn.click(
                self.get_character_info,
                char_name_input,
                info_output
            )
            
            stats_btn.click(
                self.get_dataset_stats,
                outputs=stats_output
            )
        
        return demo
    
    def launch(self, share: bool = False, debug: bool = False):
        """Launch the Gradio interface.
        
        Args:
            share: Whether to create a public link
            debug: Whether to enable debug mode
        """
        if not self.chatbot:
            print("❌ Cannot launch: Chatbot not properly initialized")
            print("Please check your OpenAI API key and data file.")
            return
        
        demo = self.create_interface()
        demo.launch(share=share, debug=debug)


def main():
    """Main function to launch the web interface."""
    interface = ChatbotWebInterface()
    interface.launch(share=True, debug=False)


if __name__ == "__main__":
    main()
