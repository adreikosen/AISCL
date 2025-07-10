from dotenv import load_dotenv
import os
import json
from httpx._transports import base
from openai import OpenAI
from model import NetworkOptimization
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = NetworkOptimization("Network_Optimization_Model")

model.add_plant("Ahmedabad", 4800)
model.add_plant("Patna", 3500)
model.add_plant("Hyderabad", 2200)

# Add distribution centers with demands
model.add_distribution_center("Bhopal", 4800)
model.add_distribution_center("Indore", 5700)

# Set shipping costs
model.set_shipping_cost("Ahmedabad", "Bhopal", 16500)
model.set_shipping_cost("Ahmedabad", "Indore", 10600)
model.set_shipping_cost("Patna", "Bhopal", 12200)
model.set_shipping_cost("Patna", "Indore", 12600)
model.set_shipping_cost("Hyderabad", "Bhopal", 10300)
model.set_shipping_cost("Hyderabad", "Indore", 9240)

def get_available_functions() -> List[Dict]:
    """Return the available functions for the AI to call."""
    return [
        {
            "type": "function",
            "function": {
                "name": "add_plant",
                "description": "Add a new plant with its capacity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the plant"},
                        "capacity": {"type": "number", "description": "Production capacity of the plant"}
                    },
                    "required": ["name", "capacity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_distribution_center",
                "description": "Add a new distribution center with its demand",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the distribution center"},
                        "demand": {"type": "number", "description": "Demand of the distribution center"}
                    },
                    "required": ["name", "demand"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_shipping_cost",
                "description": "Set the shipping cost between a plant and a distribution center",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plant": {"type": "string", "description": "Name of the plant"},
                        "dc": {"type": "string", "description": "Name of the distribution center"},
                        "cost": {"type": "number", "description": "Cost per unit to ship from plant to DC"}
                    },
                    "required": ["plant", "dc", "cost"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "solve_optimization",
                "description": "Solve the optimization problem and get the solution",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_model_info",
                "description": "Get information about the current model including the current plants, current distribution centers, and shipping costs",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_capacity",
                "description": "Update capacity of existing plant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the plant"},
                        "capacity": {"type": "number", "description": "Capacity of the plant"}
                    },
                    "required": ["name", "capacity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "update_demand",
                "description": "Update demand of existing distribution center",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the distribution center"},
                        "demand": {"type": "number", "description": "Demand of the distribution center"}
                    },
                    "required": ["name", "demand"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remove_plant",
                "description": "Remove existing plant",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the plant"},
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remove_distribution_center",
                "description": "Remove existing distribution center",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the distribution center"},
                    },
                    "required": ["name"]
                }
            }
        },
        
    ]

def execute_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """Execute the function call and return the result as a string."""
    try:
        if function_name == "add_plant":
            model.add_plant(function_args["name"], function_args["capacity"])
            return f"Added plant {function_args['name']} with capacity {function_args['capacity']}"
            
        elif function_name == "add_distribution_center":
            model.add_distribution_center(function_args["name"], function_args["demand"])
            return f"Added distribution center {function_args['name']} with demand {function_args['demand']}"
            
        elif function_name == "set_shipping_cost":
            model.set_shipping_cost(
                function_args["plant"],
                function_args["dc"],
                function_args["cost"]
            )
            return f"Set shipping cost from {function_args['plant']} to {function_args['dc']} to {function_args['cost']}"
            
        elif function_name == "solve_optimization":
            solution = model.solve()
            return f"Optimization complete. {solution}"
        elif function_name == "get_model_info":
            plants = model.plants
            distribution_centers = model.distribution_centers
            shipping_costs = model.costs
            capacity = model.capacity
            demand = model.demand
            return f"Plants: {plants}\nDistribution Centers: {distribution_centers}\nShipping Costs: {shipping_costs}\nCapacity: {capacity}\nDemand: {demand}"
        elif function_name == "update_capacity":
            model.update_capacity(function_args["name"], function_args["capacity"])
            return f"Updated capacity of {function_args['name']} to {function_args['capacity']}"
        elif function_name == "update_demand":
            model.update_demand(function_args["name"], function_args["demand"])
            return f"Updated demand of {function_args['name']} to {function_args['demand']}"
        elif function_name == "remove_plant":
            model.remove_plant(function_args["name"])
            return f"Removed plant {function_args['name']}"
        elif function_name == "remove_distribution_center":
            model.remove_distribution_center(function_args["name"])
            return f"Removed distribution center {function_args['name']}"
        return f"Unknown function: {function_name}"
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"

def chat_with_ai(user_input: str, messages: List[Dict] = None) -> str:
    """Send a message to the AI and get a response, handling function calls."""
    if messages is None:
        messages = []
    
    # Add user message to the conversation
    messages.append({"role": "user", "content": user_input})
    
    # Get AI response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=get_available_functions(),
        tool_choice="auto"
        )
    response_message = response.choices[0].message
    messages.append(response_message)
    
    # Check if the AI wants to call a function
    if response_message.tool_calls:
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            function_response = execute_function_call(function_name, function_args)
            
            # Add the function response to the messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": function_response
            })
        
        # Get a new response from the AI with the function's response
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        return second_response.choices[0].message.content
    
    return response_message.content

def main():
    """Main function to run the chatbot."""
    print("Hello I am Kit your friendly network optimization chatbot!")
    print("You can add plants, distribution centers, set shipping costs, and solve the optimization problem.")
    print("Type 'exit' to quit.\n")
    
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant for network optimization. You can help users model and solve transportation problems."
    }]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye :( - Kit signing off!")
            break
            
        response = chat_with_ai(user_input, messages)
        print(f"\nKit: {response}")

if __name__ == "__main__":
    # Initialize the optimization model
    main()
