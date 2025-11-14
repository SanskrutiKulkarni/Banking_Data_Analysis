import google.generativeai as genai
import os
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from llm.chat_history import ChatHistoryManager

load_dotenv()

class BankingChatAssistant:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-pro-vision")
        self.history = ChatHistoryManager()
    
    def analyze(self, text_query: str, context: dict, image_data=None):
        # Build prompt
        prompt = f"""Analyze this banking customer:
        - Age: {context.get('age', 'N/A')}
        - Balance: ${context.get('balance', 0):,}
        - Subscription Probability: {context.get('prediction', 0):.2%}
        
        User Question: {text_query}"""
        
        # Prepare content
        content = [prompt]
        if image_data:
            if isinstance(image_data, bytes):
                img = Image.open(BytesIO(image_data))
            else:
                img = image_data
            content.append(img)
        
        # Generate response
        response = self.model.generate_content(content)
        
        # Store in history
        self.history.add_message(
            role="user",
            content=text_query,
            metadata={"context": context, "has_image": bool(image_data)}
        )
        self.history.add_message(
            role="assistant",
            content=response.text,
            metadata={"context": context}
        )
        
        return response.text