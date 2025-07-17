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
    model = NetworkOptimization("US_Transportation_Network")
   # Create model instance
    model = NetworkOptimization("US_Transportation_Network")
    
    # Add plants with capacities (in units) and locations
    model.add_plant("Chicago_IL", 13000, 41.8781, -87.6298)    # Major manufacturing hub
    model.add_plant("Houston_TX", 14500, 29.7604, -95.3698)     # Major industrial center
    model.add_plant("LosAngeles_CA", 12500, 34.0522, -118.2437)  # West Coast manufacturing
    
    # Add regional distribution centers
    model.add_regional_distribution_center("RDC1_Chicago", 41.8781, -87.6298)  # Midwest hub
    model.add_regional_distribution_center("RDC2_Atlanta", 33.7490, -84.3880)  # Southeast hub
    #model.add_regional_distribution_center("RDC3_Denver", 39.7392, -104.9903)  # Central hub
    
    # Add stores with demands (in units) and locations
    model.add_store("NewYork_NY", 7000, 40.7128, -74.0060)     # High population density
    model.add_store("Miami_FL", 5500, 25.7617, -80.1918)       # Southeast retail hub
    model.add_store("Seattle_WA", 5000, 47.6062, -122.3321)    # Pacific Northwest
    model.add_store("Dallas_TX", 6500, 32.7767, -96.7970)      # Major Texas market
    model.add_store("Boston_MA", 5000, 42.3601, -71.0589)      # Major Northeast market
    model.add_store("LasVegas_NV", 8000, 36.1699, -115.1398)      # Major West Coast market
    
    # Set shipping costs (in dollars per unit)
    # Plants to RDCs
    model.set_shipping_cost("Chicago_IL", "RDC1_Chicago", 100)  # Local transfer within Chicago
    model.set_shipping_cost("Chicago_IL", "RDC2_Atlanta", 1404.20)  # Chicago → ATL: $1,404.20
    model.set_shipping_cost("Houston_TX", "RDC2_Atlanta", 1666.00)  # Houston → ATL: $1,666.00
    model.set_shipping_cost("Houston_TX", "RDC1_Chicago", 2237.20)  # Houston → Chicago: $2,237.20
    model.set_shipping_cost("LosAngeles_CA", "RDC2_Atlanta", 4641.00)  # LA → ATL: $4,641.00
    model.set_shipping_cost("LosAngeles_CA", "RDC1_Chicago", 4165.00)  # LA → Chicago: $4,165.00
    
    # RDCs to Stores
    model.set_shipping_cost("RDC1_Chicago", "NewYork_NY", 1880.20)  # Chicago → NYC: $1,880.20
    model.set_shipping_cost("RDC2_Atlanta", "Miami_FL", 1428.00)  # Atlanta → Miami: $1,428.00
    model.set_shipping_cost("RDC2_Atlanta", "Dallas_TX", 1713.60)  # Atlanta → Dallas: $1,713.60
    model.set_shipping_cost("RDC1_Chicago", "Seattle_WA", 4117.40)  # Chicago → Seattle: $4,117.40
    model.set_shipping_cost("RDC1_Chicago", "LasVegas_NV", 3617.60)  # Chicago → Las Vegas: $3,617.60
    model.set_shipping_cost("RDC2_Atlanta", "LasVegas_NV", 4165.00)  # Atlanta → Las Vegas: $4,165.00
    model.set_shipping_cost("RDC1_Chicago", "Boston_MA", 2023.00)  # Chicago → Boston: $2,023.00
    model.set_shipping_cost("RDC2_Atlanta", "Boston_MA", 2261.00)  # Atlanta → Boston: $2,261.00
    model.set_shipping_cost("RDC2_Atlanta", "Seattle_WA", 5188.40)  # Atlanta → Seattle: $5,188.40

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
                "name": "set_shipping_cost",
                "description": "Set the shipping cost between a source and destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Name of the source"},
                        "destination": {"type": "string", "description": "Name of the destination"},
                        "cost": {"type": "number", "description": "Cost per unit to ship from plant to store"}
                    },
                    "required": ["source", "destination", "cost"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name":"get_shipping_cost",
                "description": "Get the shipping cost from source to destination",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string", "description": "Source node (plant or RDC)"},
                        "dst": {"type": "string", "description": "Destination node (store or RDC)"}
                    },
                    "required": ["src", "dst"]
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
                "name": "clear_model",
                "description": "Clear the model",
                "parameters": {"type": "object", "properties": {}}
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
        elif function_name == "get_shipping_cost":
            return model.get_shipping_cost(function_args["src"], function_args["dst"])
        elif function_name == "set_shipping_cost":
            model.set_shipping_cost(
                function_args.get("source") or function_args.get("src"),
                function_args.get("destination") or function_args.get("dst"),
                function_args["cost"]
            )
            return f"Set shipping cost from {function_args.get('source') or function_args.get('src')} to {function_args.get('destination') or function_args.get('dst')} to {function_args['cost']}"
            
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