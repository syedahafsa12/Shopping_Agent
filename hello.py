import logging
import os
import random
from dotenv import load_dotenv

# Suppress debug messages
logging.getLogger().setLevel(logging.ERROR)

load_dotenv()

# Retrieve your API key from the environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Missing GEMINI_API_KEY in environment variables.")
    exit(1)

#########################################
# LangChain / LangGraph Setup
#########################################
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import mock data (ensure mock_data.py is in the same folder)
from mock_data import mock_data

# Global variables for cart and user history
cart = []
user_history = []

#########################################
# TOOL DEFINITIONS - CORE FEATURES
#########################################
@tool
def show_all_products():
    """Return all available products."""
    return mock_data

@tool
def recommend_products(query: str):
    """
    Suggest up to 3 matching products based on the user query.
    """
    results = []
    for category, products in mock_data.items():
        for product in products:
            if query.lower() in product["name"].lower() or query.lower() in category:
                results.append(product)
    if not results:
        return f"No products found for: {query}"
    response = ""
    for prod in results[:3]:
        response += f"{prod['name']} ({prod['price']})\nDescription: {prod['description']}\n\n"
    return response.strip()

@tool
def get_shopping_advice(query: str):
    """Provide a concise shopping tip."""
    prompt = f"User wants {query}. Give one short, clear shopping tip."
    response = model.invoke([{"role": "system", "content": prompt}])
    if response and isinstance(response, list) and "content" in response[0]:
        return response[0]["content"].strip()
    return "No advice available."

@tool
def get_shipping_info(query: str):
    """Return a brief shipping option."""
    prompt = f"Suggest one brief shipping option for {query} products."
    response = model.invoke([{"role": "system", "content": prompt}])
    if response and isinstance(response, list) and "content" in response[0]:
        return response[0]["content"].strip()
    return "No shipping info available."

@tool
def checkout(address: str, phone_no: int, card_no: int):
    """
    Simulate a dummy payment process and finalize checkout.
    Clears the cart and returns a confirmation message.
    """
    if not cart:
        return "Your cart is empty."
    total_price = sum(float(prod["price"].replace("$", "")) for prod in cart)
    cart.clear()
    return f"Checkout complete! Total: ${total_price:.2f}. Delivery to {address}."

#########################################
# TOOL DEFINITIONS - ADDITIONAL FEATURES
#########################################
@tool
def product_search(query: str):
    """
    Search for products by name or category.
    Also stores the query in user history.
    """
    user_history.append(query)
    results = []
    for category, products in mock_data.items():
        for product in products:
            if query.lower() in product["name"].lower() or query.lower() in category:
                results.append(f"{product['name']} ({product['price']})")
    if not results:
        return "No matching products found."
    return "\n".join(results)

@tool
def filter_by_price(max_price: float):
    """
    Return products under the specified max price.
    """
    results = []
    for category, products in mock_data.items():
        for product in products:
            price = float(product["price"].replace("$", ""))
            if price <= max_price:
                results.append(f"{product['name']} (${price:.2f})")
    if not results:
        return f"No products found under ${max_price:.2f}."
    return "Products under ${:.2f}:\n".format(max_price) + "\n".join(results)

@tool
def price_comparison(product_name: str):
    """
    Simulate price comparisons for the given product.
    Returns dummy prices from different sellers.
    """
    sellers = ["Seller A", "Seller B", "Seller C"]
    base_price = None
    for category, products in mock_data.items():
        for product in products:
            if product["name"].lower() == product_name.lower():
                base_price = float(product["price"].replace("$", ""))
                break
    if base_price is None:
        return f"{product_name} not found."
    comparisons = [f"{seller}: ${base_price + random.uniform(-50, 50):.2f}" for seller in sellers]
    return "Price Comparison for " + product_name + ":\n" + "\n".join(comparisons)

@tool
def discounts_and_deals():
    """Return a random discount or deal."""
    deals = [
        "10% off on all laptops!",
        "Buy one get one free on select accessories.",
        "20% discount on smartphones until Friday."
    ]
    return random.choice(deals)

@tool
def user_preferences_history():
    """Return a summary of the user's past searches/preferences."""
    if not user_history:
        return "No search history available."
    recent = user_history[-3:]
    return "Recent searches: " + ", ".join(recent)

@tool
def customer_reviews(product_name: str):
    """Provide dummy customer reviews for a product."""
    reviews = {
        "Laptop A": "4.5/5 - Great performance and battery life.",
        "Phone A": "4.0/5 - Good value for money.",
        "Tablet A": "4.2/5 - Versatile and easy to use.",
    }
    return reviews.get(product_name, "No reviews available for this product.")

@tool
def compare_products(product_list: str):
    """
    Compare products given a comma-separated list.
    Returns a short comparison summary.
    """
    names = [name.strip().lower() for name in product_list.split(",")]
    found = []
    for category, products in mock_data.items():
        for product in products:
            if product["name"].lower() in names:
                found.append(product)
    if len(found) < 2:
        return "Need at least two products to compare."
    summary = "Comparison:\n"
    for prod in found:
        summary += f"{prod['name']}: {prod['price']}, {prod['description']}\n"
    return summary.strip()

@tool
def trending_products(category: str):
    """
    Simulate trending products by randomly selecting one from a category.
    """
    cat = category.lower()
    if cat not in mock_data or not mock_data[cat]:
        return "No trending products available."
    product = random.choice(mock_data[cat])
    return f"Trending in {category.title()}: {product['name']} ({product['price']})."

# Map all tools by name
tools = [
    show_all_products, recommend_products, get_shopping_advice, get_shipping_info,
    checkout, product_search, filter_by_price, price_comparison, discounts_and_deals,
    user_preferences_history, customer_reviews, compare_products, trending_products
]
tools_by_name = {tool.name: tool for tool in tools}

#########################################
# SYSTEM PROMPT
#########################################
system_prompt = (
    "You are a concise, friendly shopping assistant with full conversation memory. "
    "When a user asks to buy a product, use your tools (such as 'recommend_products') to suggest items. "
    "If the user says something like 'confirm order for Headphones B', ask for details and then simulate a checkout "
    "using dummy data (for example, address='123 Main St', phone_no=123456, card_no=12345678) and provide a shipping estimate. "
    "For other queries, use context to respond helpfully."
)

#########################################
# AGENT DEFINITION
#########################################
@task
def call_model(messages):
    """Call the generative model with the system prompt and conversation."""
    response = model.bind_tools(tools).invoke(
        [{"role": "system", "content": system_prompt}] + messages
    )
    return response

@task
def call_tool(tool_call):
    """Execute a tool call based on its name and arguments."""
    tool_fn = tools_by_name[tool_call["name"]]
    observation = tool_fn.invoke(tool_call["args"])
    return ToolMessage(content=observation, tool_call_id=tool_call["id"])

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def agent(messages, previous):
    """Main agent logic with conversation memory."""
    if previous is not None:
        messages = add_messages(previous, messages)
    llm_response = call_model(messages).result()
    while llm_response.tool_calls:
        tool_result_futures = [call_tool(tc) for tc in llm_response.tool_calls]
        tool_results = [future.result() for future in tool_result_futures]
        messages = add_messages(messages, [llm_response, *tool_results])
        llm_response = call_model(messages).result()
    messages = add_messages(messages, llm_response)
    return entrypoint.final(value=llm_response, save=messages)

# Instantiate the model after tool definitions
model = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.0-flash")

#########################################
# CONVERSATION LOOP (Terminal)
#########################################
def conversation_loop():
    conversation = []
    print("🛍 AI Shopping Assistant (Terminal Mode)")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        conversation.append({"role": "user", "content": user_input})
        response = agent.invoke(conversation, config={"configurable": {"thread_id": "terminal"}})
        assistant_response = response.content.strip()
        conversation.append({"role": "assistant", "content": assistant_response})
        print("\n--- Last 5 Messages ---")
        for msg in conversation[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            print(f"{role}: {msg['content']}")
        print("------------------------\n")

if __name__ == "__main__":
    conversation_loop()
