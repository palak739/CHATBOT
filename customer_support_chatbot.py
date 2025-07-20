import nltk
import logging
from nltk.chat.util import Chat, reflections
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag

# Ensure required nltk data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Setup logging to file
logging.basicConfig(filename='customer_support_chatbot.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Define customer support pairs
pairs = [
    [r"hi|hello|hey", ["Hello! Welcome to Customer Support. How can I assist you today?"]],
    [r"my order (.*) (status|track|tracking)", ["Please provide your order number, and I will check the status for you."]],
    [r"order number is (.*)", ["Thank you. Let me check the status of order %1. Please wait a moment."]],
    [r"(.*) refund (.*)", ["I'm sorry to hear that. Could you please provide your order number for the refund process?"]],
    [r"(.*) cancel (.*) order", ["To cancel your order, please provide your order number."]],
    [r"(.*) technical issue (.*)", ["I understand you're facing technical issues. Could you please describe the problem in detail?"]],
    [r"(.*) speak to (a )?human", ["I will connect you to a customer service representative shortly."]],
    [r"thank you|thanks", ["You're welcome! If you have any other questions, feel free to ask."]],
    [r"bye|exit", ["Thank you for contacting Customer Support. Have a great day!"]],
    [r"(.*)", ["I'm sorry, I didn't understand that. Could you please rephrase or provide more details?"]]
]

class CustomerSupportChatbot:
    def __init__(self, pairs):
        self.chat = Chat(pairs, reflections)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            if scores['compound'] <= -0.5:
                return 'negative'
            elif scores['compound'] >= 0.5:
                return 'positive'
            else:
                return 'neutral'
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")
            return 'neutral'

    def respond(self, user_input):
        logging.info(f"User: {user_input}")
        sentiment = self.analyze_sentiment(user_input)
        response = self.chat.respond(user_input)
        logging.info(f"Chatbot: {response}")
        return response

    def chat_with_bot(self):
        print("Welcome to the Customer Support Chatbot! Type 'exit' to end the conversation.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'exit':
                    print("Chatbot: Thank you for contacting Customer Support. Have a great day!")
                    logging.info("Chatbot session ended by user.")
                    break
                response = self.respond(user_input)
                print(f"Chatbot: {response}")
            except Exception as e:
                print("Chatbot: Sorry, something went wrong. Please try again.")
                logging.error(f"Error during chat: {e}")

if __name__ == "__main__":
    chatbot = CustomerSupportChatbot(pairs)
    chatbot.chat_with_bot()
