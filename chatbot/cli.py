from chatbot import ChatBot
if __name__ == "__main__":
    bot = ChatBot()
    print("RAG Chatbot (type 'exit' to quit)")
    while True:
        try:
            user_input = input().strip()
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input:
                continue
            response = bot.ask_question_without_context(user_input)
            print(response)
        except KeyboardInterrupt:
            print("\nBye!")
            break