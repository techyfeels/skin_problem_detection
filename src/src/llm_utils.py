import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Hardcoded HAUM Product Data
HAUM_PRODUCTS = [
    {
        "name": "LCID Salicylic Acid 2%",
        "description": "A serum with 2% Salicylic Acid. Best for exfoliation, acne treatment, and deep pore cleansing.",
        "benefits": ["treats acne", "exfoliates", "unclogs pores"]
    },
    {
        "name": "C+ALPHA",
        "description": "Contains Vitamin C 7% and Alpha Arbutin 2%. Brightens skin, fades dark spots, and provides antioxidant protection.",
        "benefits": ["brightening", "fades dark spots", "antioxidant"]
    },
    {
        "name": "A Pure Retinol 0.8%",
        "description": "An anti-aging cream serum with 0.8% Retinol. Helps reduce fine lines, wrinkles, and improves skin texture.",
        "benefits": ["anti-aging", "reduces wrinkles", "improves texture"]
    },
    {
        "name": "FACE ON Calm & Repair",
        "description": "Lightweight moisturizer with Ectoin 1%, Milk Protein, and Creatine. Soothes irritated and sensitive skin.",
        "benefits": ["moisturizing", "soothing", "repairing", "sensitive skin"]
    },
    {
        "name": "AHA Toner",
        "description": "Toner with Glycolic Acid 7%. Exfoliates dead skin cells and brightens the complexion.",
        "benefits": ["exfoliation", "brightening"]
    },
    {
        "name": "WASH ON Gentle & Calm",
        "description": "Facial wash with Pentavitin 1%. Cleanses while maintaining skin hydration and softness.",
        "benefits": ["cleansing", "hydrating", "gentle"]
    },
    {
        "name": "Aloecid Niacinamide 10%",
        "description": "Serum with Niacinamide 10% and Aloe Vera. Controls oil, minimizes pores, and soothes skin.",
        "benefits": ["oil control", "pore minimizing", "soothing"]
    },
    {
        "name": "Hydracalm Toner",
        "description": "Hydrating toner with Milk Protein. Deeply hydrates and calms the skin.",
        "benefits": ["hydrating", "calming"]
    },
    {
        "name": "Sunstick SPF 39 PA ++++",
        "description": "Practical sun protection stick. Protects against UV rays.",
        "benefits": ["sun protection", "uv protection"]
    }
]

def get_product_context():
    """Formats the product list into a string for the prompt."""
    context = "Here is the list of available HAUM skincare products:\n"
    for p in HAUM_PRODUCTS:
        context += f"- {p['name']}: {p['description']} (Key benefits: {', '.join(p['benefits'])})\n"
    return context

def get_skin_advice(detection_results, user_query, chat_history):
    """
    Generates a response using OpenAI GPT-4o-mini based on detection results and user query.
    
    Args:
        detection_results (dict): Dictionary of detected classes and their counts (e.g., {'acne': 2}).
        user_query (str): The user's question.
        chat_history (list): List of previous chat messages.
        
    Returns:
        str: The LLM's response.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY not found in environment variables."

    client = OpenAI(api_key=api_key)

    product_context = get_product_context()
    
    # Format detection results for the prompt
    detection_summary = ", ".join([f"{k}: {v}" for k, v in detection_results.items()])
    if not detection_summary:
        detection_summary = "No specific skin conditions detected."

    system_prompt = f"""You are a helpful Skin Analyzer Assistant.
    The user has uploaded an image and our object detection model found the following: {detection_summary}.
    
    Your goal is to answer the user's questions and recommend suitable products from the HAUM skincare line ONLY if applicable to their condition or query.
    
    {product_context}
    
    Rules:
    1. Be polite and helpful.
    2. Focus on the detected skin problems (if any) and the user's question.
    3. Use the provided HAUM product list to make recommendations. Do not recommend products outside this list.
    4. If the user asks about something unrelated to skin or these products, politely guide them back.
    5. Keep answers concise but informative.
    """

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (simple version, last few messages)
    for msg in chat_history[-5:]: # Keep context short
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {str(e)}"
