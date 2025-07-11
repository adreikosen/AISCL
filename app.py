from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from model import NetworkOptimization

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
model = None
snapshots = {}

def initialize_model():
    """Initialize the model with default values."""
    global model
    model = NetworkOptimization("Base_Model")
    
    # Default model setup
    model.add_plant("Ahmedabad", 4800)
    model.add_plant("Patna", 3500)
    model.add_plant("Hyderabad", 2200)
    model.add_distribution_center("Bhopal", 4800)
    model.add_distribution_center("Indore", 5700)
    model.set_shipping_cost("Ahmedabad", "Bhopal", 16500)
    model.set_shipping_cost("Ahmedabad", "Indore", 10600)
    model.set_shipping_cost("Patna", "Bhopal", 12200)
    model.set_shipping_cost("Patna", "Indore", 12600)
    model.set_shipping_cost("Hyderabad", "Bhopal", 10300)
    model.set_shipping_cost("Hyderabad", "Indore", 9240)

def create_snapshot(snapshot_name: str = None) -> str:
    """
    Create a snapshot of the current model state.
    
    Args:
        snapshot_name (str, optional): Name for the snapshot. If None, a default name will be generated.
        
    Returns:
        str: Confirmation message with the snapshot name
    """
    global model, snapshots
    if snapshot_name is None:
        snapshot_name = f"snapshot_{len(snapshots) + 1}"
    snapshots[snapshot_name] = model.copy(snapshot_name)
    return f"Created snapshot '{snapshot_name}' of the current model."

def compare_with_snapshot(snapshot_name: str) -> str:
    """
    Compare the current model with a previously saved snapshot.
    
    Args:
        snapshot_name (str): Name of the snapshot to compare with
        
    Returns:
        str: Formatted comparison results
    """
    global model, snapshots
    if snapshot_name not in snapshots:
        return f"Error: No snapshot found with name '{snapshot_name}'"
    
    snapshot = snapshots[snapshot_name]
    differences = NetworkOptimization.compare_models(snapshot, model)
    
    # Format the differences into a readable string
    output = ["\n=== Model Comparison ==="]
    output.append(f"Comparing: {differences['name'][0]} (snapshot) vs {differences['name'][1]} (current)")
    
    # Plants section
    plants = differences['plants']
    if plants['added'] or plants['removed'] or plants['capacity_changes']:
        output.append("\n=== Plants ===")
        if plants['added']:
            output.append(f"Added plants: {', '.join(plants['added'])}")
        if plants['removed']:
            output.append(f"Removed plants: {', '.join(plants['removed'])}")
        for plant, (old_cap, new_cap) in plants['capacity_changes'].items():
            output.append(f"Changed capacity for {plant}: {old_cap} → {new_cap}")
    
    # Distribution Centers section
    dcs = differences['distribution_centers']
    if dcs['added'] or dcs['removed'] or dcs['demand_changes']:
        output.append("\n=== Distribution Centers ===")
        if dcs['added']:
            output.append(f"Added DCs: {', '.join(dcs['added'])}")
        if dcs['removed']:
            output.append(f"Removed DCs: {', '.join(dcs['removed'])}")
        for dc, (old_demand, new_demand) in dcs['demand_changes'].items():
            output.append(f"Changed demand for {dc}: {old_demand} → {new_demand}")
    
    # Cost changes section
    if differences['cost_changes']:
        output.append("\n=== Cost Changes ===")
        for plant, dc_changes in differences['cost_changes'].items():
            for dc, (old_cost, new_cost) in dc_changes.items():
                output.append(f"Cost {plant} → {dc}: ${old_cost} → ${new_cost}")
    
    # Add solution comparison
    try:
        old_solution = snapshot.solve()
        new_solution = model.solve()
        
        output.append("\n=== Solution Comparison ===")
        output.append(f"Snapshot total cost: ${old_solution.get('total_cost', 'N/A'):,.2f}")
        output.append(f"Current total cost: ${new_solution.get('total_cost', 'N/A'):,.2f}")
        
        if old_solution.get('total_cost') and new_solution.get('total_cost'):
            cost_diff = new_solution['total_cost'] - old_solution['total_cost']
            change = "increase" if cost_diff > 0 else "decrease"
            output.append(f"Change in total cost: ${abs(cost_diff):,.2f} ({change})")
    
    except Exception as e:
        output.append(f"\nCould not compare solutions due to error: {str(e)}")
    
    return "\n".join(output)

def list_snapshots() -> str:
    """List all saved snapshots."""
    global snapshots
    if not snapshots:
        return "No snapshots available."
    return "Available snapshots:\n" + "\n".join(f"- {name}" for name in snapshots.keys())

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
                "description": "Get information about the current model including plants, distribution centers, and shipping costs",
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
                        "capacity": {"type": "number", "description": "New capacity of the plant"}
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
                        "demand": {"type": "number", "description": "New demand of the distribution center"}
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
                        "name": {"type": "string", "description": "Name of the plant"}
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
                        "name": {"type": "string", "description": "Name of the distribution center"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "clear_model",
                "description": "Clear the model",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_snapshot",
                "description": "Create a snapshot of the current model state for later comparison",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "snapshot_name": {
                            "type": "string",
                            "description": "Name for the snapshot. If not provided, a default name will be generated"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_with_snapshot",
                "description": "Compare the current model with a previously saved snapshot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "snapshot_name": {
                            "type": "string",
                            "description": "Name of the snapshot to compare with"
                        }
                    },
                    "required": ["snapshot_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_snapshots",
                "description": "List all saved snapshots",
                "parameters": {"type": "object", "properties": {}}
            }
        }
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
            snapshot_name = f"pre_solve_{len(snapshots) + 1}"
            create_snapshot(snapshot_name)
            solution = model.solve()
            return f"Optimization complete. {solution}"
            
        elif function_name == "get_model_info":
            plants = model.plants
            distribution_centers = model.distribution_centers
            shipping_costs = model.costs
            capacity = model.capacity
            demand = model.demand
            return f"""Current Model State:
Plants: {plants}
Distribution Centers: {distribution_centers}
Shipping Costs: {shipping_costs}
Capacity: {capacity}
Demand: {demand}"""
            
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
            
        elif function_name == "clear_model":
            model.clear_model()
            return "Model cleared successfully"
            
        elif function_name == "create_snapshot":
            return create_snapshot(function_args.get("snapshot_name"))
            
        elif function_name == "compare_with_snapshot":
            return compare_with_snapshot(function_args["snapshot_name"])
            
        elif function_name == "list_snapshots":
            return list_snapshots()
            
        else:        
            return f"Unknown function: {function_name}"
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"

def chat_with_ai(user_input: str, messages: List[Dict] = None) -> str:
    """Send a message to the AI and get a response, handling function calls and API errors."""
    if messages is None:
        messages = [] 
    try:
        # Add user message to the conversation
        messages.append({"role": "user", "content": user_input})
        try:
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
                
                try:
                    # Get a new response from the AI with the function's response
                    second_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    return second_response.choices[0].message.content
                except Exception as e:
                    return f"I'm sorry, but I encountered an error while processing the function response: {str(e)}. Please try again."
            
            return response_message.content
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return "I'm currently receiving too many requests and am overwhelmed. Please wait a moment and try again."
            elif "authentication" in error_msg.lower():
                return "There was an authentication error with the AI service. Please check your API key and try again."
            elif "timeout" in error_msg.lower():
                return "The request to the API service timed out. Please check your internet connection and try again."
            else:
                return f"I'm sorry, but I encountered an error: {error_msg}. Please try again or contact support if the issue persists."
                
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}. Please try again later."

def main():
    """Main function to run the chatbot."""
    global model
    
    # Initialize the model
    initialize_model()
    
    print("Hello I am Kit your friendly network optimization chatbot!")
    print("You can add plants, distribution centers, set shipping costs, and solve the optimization problem.")
    print("Type 'exit' to quit.\n")
    
    messages = [{
        "role": "system",
        "content": """You are a helpful assistant for network optimization. You can help users model and solve network optimization problems in supply chain. 
        You can also help users create and compare different scenarios by taking snapshots of the model at different points in time.
        When comparing models, make sure to highlight the key differences and their business implications. A base model has been preloaded into the system."""
    }]
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye :( - Kit signing off!")
                break
                
            response = chat_with_ai(user_input, messages)
            print(f"\nKit: {response}")
            print(snapshots)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using Kit. Have a great day!")
            break
        except Exception as e:
            print(f"\nKit: Oops! Something went wrong: {str(e)}. Let's try that again.")

if __name__ == "__main__":
    main()
