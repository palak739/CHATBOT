import nltk
import re
import logging
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag, word_tokenize

# Ensure required nltk data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Setup logging to file
logging.basicConfig(filename='chatbot_conversation.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

pairs = [
    [r"hi|hello|hey", ["Hello! How can I help you today?", "Hi there! How may I assist you?"]],
    [r"my name is (.*)", ["Hello %1! How can I assist you today?"]],
    [r"(.*) your name?", ["I am your friendly chatbot!"]],
    [r"how are you?", ["I'm just a bot, but I'm doing well. How about you?"]],
    [r"tell me a joke", ["Why don't skeletons fight each other? They don't have the guts!"]],
    [r"(.*) (help|assist) (.*)", ["Sure! How can I assist you with %3?"]],
    [r"bye|exit", ["Goodbye! Have a great day!", "See you later!"]],
    [r"(.*)", ["I'm sorry, I didn't understand that. Could you rephrase?", "Could you please elaborate?"]]
]

class RuleBasedChatbot:
    def __init__(self, pairs):
        self.chat = Chat(pairs, reflections)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.context = {}

    def analyze_sentiment(self, text):
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            if scores['compound'] >= 0.05:
                return 'positive'
            elif scores['compound'] <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return 'neutral'

    def pos_tagging(self, text):
        try:
            tokens = word_tokenize(text)
            return pos_tag(tokens)
        except Exception as e:
            logging.error(f"POS tagging error: {e}")
            return []

    def respond(self, user_input):
        # Log user input
        logging.info(f"User: {user_input}")

        # Analyze sentiment
        sentiment = self.analyze_sentiment(user_input)
        pos_tags = self.pos_tagging(user_input)

        response = self.chat.respond(user_input)

        # Log chatbot response
        logging.info(f"Chatbot: {response}")

        return response

    def chat_with_bot(self):
        print("Hello, I am your advanced chatbot! Type 'exit' to end the conversation.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    print("Chatbot: Goodbye! Have a nice day!")
                    logging.info("Chatbot session ended by user.")
                    break
                response = self.respond(user_input)
                print(f"Chatbot: {response}")
            except Exception as e:
                print("Chatbot: Sorry, something went wrong. Please try again.")
                logging.error(f"Error during chat: {e}")

if __name__ == "__main__":
    chatbot = RuleBasedChatbot(pairs)
    chatbot.chat_with_bot()
