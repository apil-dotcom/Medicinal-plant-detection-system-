import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI/OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Load plant data (optional, for reference)
with open("model/plants.json", "r", encoding="utf-8") as f:
    plant_data = json.load(f)

def chat_with_ai(question, plant_name=None):
    """
    question: user's question string
    plant_name: the currently detected plant (optional)
    """
    # Build a system prompt that restricts responses
    system_prompt = """
You are a helpful medicinal plant expert chatbot.

Rules:
1. Only answer questions related to medicinal plants, their benefits, or scientific info.
2. If the question is unrelated to medicinal plants, reply: "⚠️ Sorry, I only answer medicinal plant questions."
3. If a plant_name is provided, assume the user is asking about that plant.

Keep answers simple, clear, and concise.
"""

    # Include plant info if available
    plant_info_text = ""
    if plant_name and plant_name in plant_data:
        info = plant_data[plant_name]
        plant_info_text = f"""
Currently Detected Plant: {plant_name}
Scientific Name: {info.get('scientific', 'Unknown')}
Benefits: {', '.join(info.get('benefits', []))}
"""

    # Prepare messages for OpenAI chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": plant_info_text + "\nUser Question: " + question}
    ]

    try:
        response = client.chat.completions.create(
            model="tngtech/deepseek-r1t2-chimera:free",
            messages=messages
        )

        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        print("Error:", e)
        return "❌ AI service error. Please try again later."