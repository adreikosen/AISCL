from dotenv import load_dotenv
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import requests
from model import NetworkOptimization

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
model = None
snapshots = {}

def geocode_location(location_name: str) -> Optional[Tuple[float, float]]:
    """
    Look up latitude and longitude for a location name using Nominatim geocoding service.
    
    Args:
        location_name (str): Name of the location to geocode (e.g., "Chicago, IL")
        
    Returns:
        Optional[Tuple[float, float]]: (latitude, longitude) if found, None otherwise
    """
    try:
        base_url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 1
        }
        
        headers = {
            'User-Agent': 'NetworkOptimizationApp/1.0'  # Required by Nominatim usage policy
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            return float(data[0]['lat']), float(data[0]['lon'])
            
    except Exception as e:
        print(f"Error geocoding location {location_name}: {str(e)}")
    
    return None

def add_plant_with_geocoding(name: str, capacity: int, city: str = None, state: str = None) -> str:
    """
    Add a plant with automatic geocoding of its location.
    
    Args:
        name (str): Name of the plant
        capacity (int): Production capacity
        city (str, optional): City name for geocoding
        state (str, optional): State/region name for more accurate geocoding
        
    Returns:
        str: Status message
    """
    global model
    if model is None:
        return "Error: Model not initialized"
    
    try:
        # Try to geocode the location if city is provided
        lat, lon = None, None
        if city:
            location_query = f"{city}"
            if state:
                location_query += f", {state}"
            coords = geocode_location(location_query)
            if coords:
                lat, lon = coords
        
        # Add the plant with or without location data
        model.add_plant(name, capacity, lat, lon)
        
        if lat is not None and lon is not None:
            return f"Added plant '{name}' with capacity {capacity} at location ({lat:.4f}, {lon:.4f})"
        else:
            return f"Added plant '{name}' with capacity {capacity} (location not set)"
            
    except Exception as e:
        return f"Error adding plant: {str(e)}"

def add_store_with_geocoding(name: str, demand: int, city: str = None, state: str = None) -> str:
    """
    Add a store with automatic geocoding of its location.
    
    Args:
        name (str): Name of the store
        demand (int): Demand amount
        city (str, optional): City name for geocoding
        state (str, optional): State/region name for more accurate geocoding
        
    Returns:
        str: Status message
    """
    global model
    if model is None:
        return "Error: Model not initialized"
    
    try:
        # Try to geocode the location if city is provided
        lat, lon = None, None
        if city:
            location_query = f"{city}"
            if state:
                location_query += f", {state}"
            coords = geocode_location(location_query)
            if coords:
                lat, lon = coords
        
        # Add the store with or without location data
        model.add_store(name, demand, lat, lon)
        
        if lat is not None and lon is not None:
            return f"Added store '{name}' with demand {demand} at location ({lat:.4f}, {lon:.4f})"
        else:
            return f"Added store '{name}' with demand {demand} (location not set)"
            
    except Exception as e:
        return f"Error adding store: {str(e)}"

def add_rdc_with_geocoding(name: str, city: str = None, state: str = None) -> str:
    """
    Add a regional distribution center with automatic geocoding of its location.
    
    Args:
        name (str): Name of the RDC
        city (str, optional): City name for geocoding
        state (str, optional): State/region name for more accurate geocoding
        
    Returns:
        str: Status message
    """
    global model
    if model is None:
        return "Error: Model not initialized"
    
    try:
        # Try to geocode the location if city is provided
        lat, lon = None, None
        if city:
            location_query = f"{city}"
            if state:
                location_query += f", {state}"
            coords = geocode_location(location_query)
            if coords:
                lat, lon = coords
        
        # Add the RDC with or without location data
        model.add_regional_distribution_center(name, lat, lon)
        
        if lat is not None and lon is not None:
            return f"Added RDC '{name}' at location ({lat:.4f}, {lon:.4f})"
        else:
            return f"Added RDC '{name}' (location not set)"
            
    except Exception as e:
        return f"Error adding RDC: {str(e)}"

def initialize_model():
    """Initialize the model with default values using US cities."""
    global model
    model = NetworkOptimization("US_Base_Model")
    
    # Add plants (manufacturing facilities) with production capacities (in units) and locations
    model.add_plant("Chicago_IL", 10000, lat=41.8781, lon=-87.6298)    # Major manufacturing hub in the Midwest
    model.add_plant("Houston_TX", 8500, lat=29.7604, lon=-95.3698)     # Major industrial center in the South
    model.add_plant("LosAngeles_CA", 9500, lat=34.0522, lon=-118.2437)  # Major West Coast manufacturing center
    
    # Add Regional Distribution Centers with locations
    model.add_regional_distribution_center("RDC1_Chicago", lat=41.8781, lon=-87.6298)  # Same as Chicago plant
    model.add_regional_distribution_center("RDC2_Atlanta", lat=33.7490, lon=-84.3880)  # Major logistics hub
    model.add_regional_distribution_center("RDC3_Denver", lat=39.7392, lon=-104.9903)  # Central distribution point
    
    # Add stores with demands (in units) and locations
    model.add_store("NewYork_NY", 8000, lat=40.7128, lon=-74.0060)    # High population density
    model.add_store("Miami_FL", 6000, lat=25.7617, lon=-80.1918)      # Major Southeastern hub
    model.add_store("Seattle_WA", 5000, lat=47.6062, lon=-122.3321)   # Pacific Northwest
    model.add_store("Dallas_TX", 7000, lat=32.7767, lon=-96.7970)     # Major Texas hub
    
    # Set shipping costs from plants to RDCs (in dollars per unit)
    model.set_shipping_cost("Chicago_IL", "RDC1_Chicago", 500)    # Local transfer
    model.set_shipping_cost("Chicago_IL", "RDC2_Atlanta", 1200)   # Chicago to Atlanta
    model.set_shipping_cost("Houston_TX", "RDC2_Atlanta", 800)    # Houston to Atlanta
    model.set_shipping_cost("Houston_TX", "RDC3_Denver", 1000)    # Houston to Denver
    model.set_shipping_cost("LosAngeles_CA", "RDC3_Denver", 700)  # LA to Denver
    model.set_shipping_cost("LosAngeles_CA", "RDC1_Chicago", 1500) # Cross-country
    
    # Set shipping costs from RDCs to stores (in dollars per unit)
    model.set_shipping_cost("RDC1_Chicago", "NewYork_NY", 800)    # Chicago to NYC
    model.set_shipping_cost("RDC2_Atlanta", "Miami_FL", 600)      # Atlanta to Miami
    model.set_shipping_cost("RDC2_Atlanta", "Dallas_TX", 500)     # Atlanta to Dallas
    model.set_shipping_cost("RDC3_Denver", "Seattle_WA", 700)     # Denver to Seattle
    model.set_shipping_cost("RDC3_Denver", "Dallas_TX", 600)      # Denver to Dallas

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
    
    # Stores section
    stores = differences['stores']
    if stores['added'] or stores['removed'] or stores['demand_changes']:
        output.append("\n=== Stores ===")
        if stores['added']:
            output.append(f"Added stores: {', '.join(stores['added'])}")
        if stores['removed']:
            output.append(f"Removed stores: {', '.join(stores['removed'])}")
        for store, (old_demand, new_demand) in stores['demand_changes'].items():
            output.append(f"Changed demand for {store}: {old_demand} → {new_demand}")
    
    # Cost changes section
    if differences['cost_changes']:
        output.append("\n=== Cost Changes ===")
        for plant, store_changes in differences['cost_changes'].items():
            for store, (old_cost, new_cost) in store_changes.items():
                output.append(f"Cost {plant} → {store}: ${old_cost} → ${new_cost}")
    
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

def restore_snapshot(snapshot_name: str) -> str:
    """
    Restore the model state from a previously saved snapshot.
    
    Args:
        snapshot_name (str): Name of the snapshot to restore
        
    Returns:
        str: Confirmation message with the snapshot name
    """
    global model, snapshots
    if snapshot_name not in snapshots:
        return f"Error: No snapshot found with name '{snapshot_name}'"
    
    # Create a copy of the current state before restoring
    backup_name = f"pre_restore_{len(snapshots) + 1}"
    snapshots[backup_name] = model.copy(backup_name)
    
    # Restore the snapshot
    model = snapshots[snapshot_name]
    return f"Successfully restored model state from snapshot '{snapshot_name}'. Previous state saved as '{backup_name}'."

def get_available_functions() -> List[Dict]:
    """Return the available functions for the AI to call."""
    return [
        {
            "type": "function",
            "function": {
                "name": "add_plant_with_geocoding",
                "description": "Add a new manufacturing plant with automatic geocoding of its location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the plant"},
                        "capacity": {"type": "integer", "description": "Production capacity of the plant"},
                        "city": {"type": "string", "description": "City where the plant is located"},
                        "state": {"type": "string", "description": "State/region where the plant is located"}
                    },
                    "required": ["name", "capacity"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "remove_regional_distribution_center",
                "description": "Remove a regional distribution center",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the RDC to remove"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_rdc_with_geocoding",
                "description": "Add a new regional distribution center with automatic geocoding of its location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the RDC"},
                        "city": {"type": "string", "description": "City where the RDC is located"},
                        "state": {"type": "string", "description": "State/region where the RDC is located"}
                    },
                    "required": ["name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "add_store_with_geocoding",
                "description": "Add a new store with automatic geocoding of its location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the store"},
                        "demand": {"type": "integer", "description": "Demand of the store"},
                        "city": {"type": "string", "description": "City where the store is located"},
                        "state": {"type": "string", "description": "State/region where the store is located"}
                    },
                    "required": ["name", "demand"]
                }
            }
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "add_plant",
        #         "description": "Add a new plant with its capacity",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "name": {"type": "string", "description": "Name of the plant"},
        #                 "capacity": {"type": "number", "description": "Production capacity of the plant"}
        #             },
        #             "required": ["name", "capacity"]
        #         }
        #     }
        # },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "add_store",
        #         "description": "Add a new store with its demand",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "name": {"type": "string", "description": "Name of the store"},
        #                 "demand": {"type": "number", "description": "Demand of the store"}
        #             },
        #             "required": ["name", "demand"]
        #         }
        #     }
        # },
        {
            "type": "function",
            "function": {
                "name": "set_shipping_cost",
                "description": "Set the shipping cost between a plant and a store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plant": {"type": "string", "description": "Name of the plant"},
                        "store": {"type": "string", "description": "Name of the store"},
                        "cost": {"type": "number", "description": "Cost per unit to ship from plant to store"}
                    },
                    "required": ["plant", "store", "cost"]
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
                "description": "Get information about the current model including plants, stores, and shipping costs",
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
                "description": "Update demand of existing store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the store"},
                        "demand": {"type": "number", "description": "New demand of the store"}
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
                "name": "remove_store",
                "description": "Remove existing store",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the store"}
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
        },
        {
            "type": "function",
            "function": {
                "name": "restore_snapshot",
                "description": "Restore the model state from a previously saved snapshot",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "snapshot_name": {
                            "type": "string",
                            "description": "Name of the snapshot to restore"
                        }
                    },
                    "required": ["snapshot_name"]
                }
            }
        }
    ]

def execute_function_call(function_name: str, function_args: Dict[str, Any]) -> str:
    """Execute a function call and return the result as a string."""
    try:
        if function_name == "add_plant_with_geocoding":
            return add_plant_with_geocoding(**function_args)
        elif function_name == "add_store_with_geocoding":
            return add_store_with_geocoding(**function_args)
        elif function_name == "add_rdc_with_geocoding":
            return add_rdc_with_geocoding(**function_args)
        # elif function_name == "add_plant":
        #     model.add_plant(function_args["name"], function_args["capacity"])
        #     return f"Added plant {function_args['name']} with capacity {function_args['capacity']}"
            
        # elif function_name == "add_store":
        #     model.add_store(function_args["name"], function_args["demand"])
        #     return f"Added store {function_args['name']} with demand {function_args['demand']}"
            
        elif function_name == "set_shipping_cost":
            model.set_shipping_cost(
                function_args["plant"],
                function_args["store"],
                function_args["cost"]
            )
            return f"Set shipping cost from {function_args['plant']} to {function_args['store']} to {function_args['cost']}"
            
        elif function_name == "solve_optimization":
            solution = model.solve()
            snapshot_name = f"pre_solve_{len(snapshots) + 1}"
            create_snapshot(snapshot_name)
            return f"Optimization complete. {solution}"
            
        elif function_name == "get_model_info":
            plants = model.plants
            stores = model.stores
            shipping_costs = model.costs
            capacity = model.capacity
            demand = model.demand
            regional_distribution_centers = model.regional_distribution_centers
            return f"""Current Model State:
                Plants: {plants}
                Stores: {stores}
                Shipping Costs: {shipping_costs}
                Capacity: {capacity}
                Demand: {demand}
                Regional Distribution Centers: {regional_distribution_centers}"""
            
        elif function_name == "update_capacity":
            model.update_capacity(function_args["name"], function_args["capacity"])
            return f"Updated capacity of {function_args['name']} to {function_args['capacity']}"
            
        elif function_name == "update_demand":
            model.update_demand(function_args["name"], function_args["demand"])
            return f"Updated demand of {function_args['name']} to {function_args['demand']}"
            
        elif function_name == "remove_plant":
            model.remove_plant(function_args["name"])
            return f"Removed plant {function_args['name']}"
            
        elif function_name == "remove_store":
            model.remove_store(function_args["name"])
            return f"Removed store {function_args['name']}"
            
        elif function_name == "remove_regional_distribution_center":
            model.remove_regional_distribution_center(function_args["name"])
            return f"Removed regional distribution center {function_args['name']}"
            
        elif function_name == "clear_model":
            model.clear_model()
            return "Model cleared successfully"
            
        elif function_name == "compare_with_snapshot":
            return compare_with_snapshot(function_args["snapshot_name"])
            
        elif function_name == "list_snapshots":
            return list_snapshots()
            
        elif function_name == "restore_snapshot":
            return restore_snapshot(function_args["snapshot_name"])
            
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
            else:
                messages.append(response_message)
                             
            
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

initialize_model()
def main():
    """Main function to run the chatbot."""
    print("Hello I am Kit your friendly network optimization chatbot!")
    print("You can add plants, stores, set shipping costs, and solve the optimization problem.")
    print("Type 'exit' to quit.\n")
    
    messages = [{
        "role": "system",
        "content": """You are a helpful assistant for network optimization. You can help users model and solve network optimization problems in supply chain. In this model there are plants that have capacity which ship to regional distribution centers which must adhere to a conservation of flow requirement and then the regional distribution centers ship to stores. The goal is to minimize the total shipping cost while satisfying the demand at each store. """
    }]
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye :( - Kit signing off!")
                break
                
            response = chat_with_ai(user_input, messages)
            print(f"\nKit: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Thanks for using Kit. Have a great day!")
            break
        except Exception as e:
            print(f"\nKit: Oops! Something went wrong: {str(e)}. Let's try that again.")

if __name__ == "__main__":
    main()
