import streamlit as st
import time
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
from model import NetworkOptimization
from app import (
    model as app_model,
    chat_with_ai,
    add_store_with_geocoding,
    add_plant_with_geocoding,
    add_rdc_with_geocoding,
    initialize_model
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page config
st.set_page_config(layout="wide", page_title="Network Optimization Dashboard")

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        color: #4f8bf9;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .assistant-message {
        background-color: #e6f3ff;
    }
</style>
""", unsafe_allow_html=True)

def initialize_dashboard_model():
    """Initialize the model with default values for the dashboard."""
    try:
        # Initialize model update timestamp
        if 'last_model_update' not in st.session_state:
            st.session_state.last_model_update = time.time()
        # Only initialize if not already done
        if 'model' not in st.session_state or st.session_state.model is None:
            st.session_state.model = app_model
            # Default model setup with error handling
            initialize_model()
        return True
    except Exception as e:
        st.error(f"Failed to create model instance: {str(e)}")
        return False

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [{
        "role": "system",
        "content": """You are a helpful assistant for network optimization. You can help users model and solve network optimization problems in supply chain."
        You can also help users create and compare different scenarios by taking snapshots of the model at different points in time.
        When comparing models, make sure to highlight the key differences and their business implications. A base model has been preloaded into the system."""
    }]
    st.session_state.snapshots = {}
    st.session_state.locations = {}
    st.session_state.last_clicked = None

def create_info_tables(model):
    """Create information tables for plants, RDCs, stores, and shipping costs."""
    try:
        # Initialize containers for each table
        plants_data = []
        rdcs_data = []
        stores_data = []
        shipping_costs = []
        
        # Get solution for optimal shipping routes
        solution = model.solve()
        solution_routes = set()
        if solution and 'solution' in solution and solution['solution']:
            solution_routes = set(solution['solution'].keys())
        
        # Prepare plants data
        for plant in getattr(model, 'plants', []):
            capacity = getattr(model, 'capacity', {}).get(plant, 'N/A')
            location = getattr(model, 'locations', {}).get(plant, {})
            lat = location.get('lat', 'N/A')
            lon = location.get('lon', 'N/A')
            coord_str = f"({lat}, {lon})" if lat != 'N/A' and lon != 'N/A' else 'N/A'
            plants_data.append({
                'Name': plant,
                'Capacity': capacity,
                'Coordinates': coord_str
            })
        
        # Prepare RDCs data
        for rdc in getattr(model, 'regional_distribution_centers', []):
            location = getattr(model, 'locations', {}).get(rdc, {})
            lat = location.get('lat', 'N/A')
            lon = location.get('lon', 'N/A')
            coord_str = f"({lat}, {lon})" if lat != 'N/A' and lon != 'N/A' else 'N/A'
            rdcs_data.append({
                'Name': rdc,
                'Coordinates': coord_str
            })
        
        # Prepare stores data
        for store in getattr(model, 'stores', []):
            demand = getattr(model, 'demand', {}).get(store, 'N/A')
            location = getattr(model, 'locations', {}).get(store, {})
            lat = location.get('lat', 'N/A')
            lon = location.get('lon', 'N/A')
            coord_str = f"({lat}, {lon})" if lat != 'N/A' and lon != 'N/A' else 'N/A'
            stores_data.append({
                'Name': store,
                'Demand': demand,
                'Coordinates': coord_str
            })
        
        # Prepare shipping costs data
        for from_node in getattr(model, 'costs', {}):
            for to_node in getattr(model, 'costs', {}).get(from_node, {}):
                try:
                    cost = model.get_shipping_cost(from_node, to_node)
                    is_optimal = (from_node, to_node) in solution_routes
                    shipping_costs.append({
                        'Source': from_node,
                        'Destination': to_node,
                        'Cost per Unit': f"${cost:.2f}",
                        'In Optimal Solution': 'Yes' if is_optimal else 'No'
                    })
                except Exception as e:
                    continue
        
        return {
            'plants': pd.DataFrame(plants_data),
            'rdcs': pd.DataFrame(rdcs_data),
            'stores': pd.DataFrame(stores_data),
            'shipping_costs': pd.DataFrame(shipping_costs)
        }
    except Exception as e:
        st.error(f"Error creating info tables: {str(e)}")
        return None

def get_model_data():
    """Extract data from the model for visualization."""
    try:
        st.session_state.model = app_model
        # Check if model is initialized
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            st.warning("Model not initialized. Creating a new instance...")
            
        # Get solution from model
        solution = st.session_state.model.solve()
        
        # Check if solution is valid
        if solution is None or not isinstance(solution, dict):
            st.warning("No valid solution found. The model might be empty or invalid.")
            return None
            
        # Initialize data structures
        nodes = []
        edges = []
        non_solution_edges = []
        
        # Add plants with error handling
        for plant in getattr(st.session_state.model, 'plants', []):
            try:
                capacity = getattr(st.session_state.model, 'capacity', {}).get(plant, 'N/A')
                nodes.append({
                    'id': plant,
                    'type': 'plant',
                    'label': f"{plant}\n(Capacity: {capacity})",
                    'size': 15,
                    'color': 'red'
                })
            except Exception as e:
                st.warning(f"Error processing plant {plant}: {str(e)}")
                continue
        
        # Add distribution centers with error handling
        for dc in getattr(st.session_state.model, 'stores', []):
            try:
                demand = getattr(st.session_state.model, 'demand', {}).get(dc, 'N/A')
                nodes.append({
                    'id': dc,
                    'type': 'dc',
                    'label': f"{dc}\n(Demand: {demand})",
                    'size': 15,
                    'color': 'gold'
                })
            except Exception as e:
                st.warning(f"Error processing stores {dc}: {str(e)}")
                continue

        # Add regional distribution centers with error handling
        for rdc in getattr(st.session_state.model, 'regional_distribution_centers', []):
            try:
                demand = getattr(st.session_state.model, 'demand', {}).get(rdc, 'N/A')
                nodes.append({
                    'id': rdc,
                    'type': 'rdc',
                    'label': f"{rdc}",
                    'size': 15,
                    'color': 'blue'
                })
            except Exception as e:
                st.warning(f"Error processing stores {dc}: {str(e)}")
                continue

        print("ALL COSTS:", getattr(st.session_state.model, 'costs'))
        for from_node in getattr(st.session_state.model, 'costs'):
            print("FROM NODE: ", from_node)
            for to_node in getattr(st.session_state.model, 'costs')[from_node]:
                print("TO NODE: ", to_node)
                try:
                    cost = st.session_state.model.get_shipping_cost(from_node, to_node)
                    print("costVar: ", st.session_state.model.get_shipping_cost(from_node, to_node))
                    non_solution_edges.append({
                        'from': from_node,
                        'to': to_node,
                        'label': f" units\n(${cost}/unit)",
                        'color': 'gray',
                        'cost': cost
                        })
                except Exception as e:
                    st.warning(f"Error processing edge from {from_node} to {to_node}: {str(e)}")
                    continue
        
        # Add edges with shipping quantities if solution exists
        if 'solution' in solution and solution['solution']:
            for (p, dc), qty in solution['solution'].items():
                try:
                    cost = getattr(st.session_state.model, 'costs', {}).get(p, {}).get(dc, 'N/A')
                    edges.append({
                        'from': p,
                        'to': dc,
                        'label': f"{int(qty)} units\n(${cost}/unit)",
                        'value': qty,
                        'color': '#90ee91'
                    })
                except Exception as e:
                    st.warning(f"Error processing edge from {p} to {dc}: {str(e)}")
                    continue
        
        return {
            'nodes': nodes,
            'edges': edges,
            'solution': solution,
            'non_solution_edges': non_solution_edges
        }
        
    except Exception as e:
        import traceback
        st.error(f"Error in get_model_data: {str(e)}\n\nTraceback:\n{traceback.format_exc()}")
        return None

def create_map():
    # Create a base map centered on the United States
    m = folium.Map(height=500, zoom_control=True, location=[39.8283, -98.5795], zoom_start=4)  # Centered on the geographic center of the US
    
    # Get model data with error handling
    
    model_data = get_model_data()
    print("Nodes:", model_data['nodes'])
    if model_data is None or 'nodes' not in model_data:
        st.warning("No model data available. Please initialize the model first.")
        return m
    
    x_loc = []
    y_loc = []
    # Add nodes to the map using model-provided locations
    for node in model_data['nodes']:
        node_id = node['id']
        print(st.session_state.model.locations[node_id]['lat'])
        print(st.session_state.model.locations[node_id]['lon'])

        
        # Get location from model locations
        if node_id in st.session_state.model.locations:
            if st.session_state.model.locations[node_id]['lat'] in y_loc and st.session_state.model.locations[node_id]['lon'] in x_loc:
                loc = (st.session_state.model.locations[node_id]['lat'] + 0.15, 
                       st.session_state.model.locations[node_id]['lon'])
            else:
                loc = (st.session_state.model.locations[node_id]['lat'], 
                       st.session_state.model.locations[node_id]['lon'])
        else:
            # If no location is set, use default coordinates
            loc = (39.8283, -98.5795)
            
        # Add marker with popup
        popup = f"<b>{node['id']}</b><br>"
        if node['type'] == 'plant':
            popup += f"Type: Plant<br>Capacity: {st.session_state.model.capacity[node['id']]}"
            icon = folium.Icon(icon='glyphicon-wrench', color='red')
        elif node['type'] == 'rdc':
            popup += f"Type: RDC"
            icon = folium.Icon(icon='glyphicon-transfer', color='blue')
        else:
            popup += f"Type: Store<br>Demand: {st.session_state.model.demand[node['id']]}"
            icon = folium.Icon(icon='glyphicon-tag', color='orange')
        
        folium.Marker(
            location=loc,
            popup=popup,
            icon=icon,
            tooltip=node['id']
        ).add_to(m)

        y_loc.append(loc[0])
        x_loc.append(loc[1])
        print("y list: ", y_loc)
        print("x list: ", x_loc)
    
    # Add all edges to map
    for edge in model_data['non_solution_edges']:
        from_loc = st.session_state.model.locations.get(edge['from'])
        to_loc = st.session_state.model.locations.get(edge['to'])
        if from_loc and to_loc:
            folium.PolyLine(
                locations=[(from_loc['lat'], from_loc['lon']), (to_loc['lat'], to_loc['lon'])],
                color='gray',
                weight=2,
                opacity=0.7,
                tooltip=f"{edge['from']} to {edge['to']}: {edge['label']}"
            ).add_to(m)

    # Add solution edges to the map
    for edge in model_data['edges']:
        from_loc = st.session_state.model.locations.get(edge['from'])
        to_loc = st.session_state.model.locations.get(edge['to'])
        if from_loc and to_loc:
            folium.PolyLine(
                locations=[(from_loc['lat'], from_loc['lon']), (to_loc['lat'], to_loc['lon'])],
                color='#90ee91',
                weight=3,
                opacity=1,
                tooltip=f"{edge['from']} to {edge['to']}: {edge['label']}"
            ).add_to(m)

    return m

def add_new_node():
    with st.expander("Add New Node"):
        node_type = st.selectbox("Node Type", ["Plant", "Store", "Regional Distribution Center"])
        name = st.text_input("Node Name")
        city = st.text_input("City")
        state = st.text_input("State/Region")
        
        if node_type == "Plant":
            capacity = st.number_input("Capacity", min_value=0)
            if st.button("Add Plant"):
                result = add_plant_with_geocoding(name, capacity, city, state)
                st.success(result)
                st.rerun()
        elif node_type == "Store":
            demand = st.number_input("Demand", min_value=0)
            if st.button("Add Store Center"):
                result = add_store_with_geocoding(name, demand, city, state)
                st.success(result)
                st.rerun()
        else:
            if st.button("Add Regional Distribution Center"):
                result = add_rdc_with_geocoding(name, city, state)
                st.success(result)
                st.rerun()

def create_plotly_graph():
    model_data = get_model_data()
    print("Nodes:", model_data['nodes'], "Edges:", model_data['edges'])
    if not model_data:
        return go.Figure()
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    node_types = ['Plant', 'RDC', 'Store']

    for node in model_data['nodes']:
        if node['id'] in st.session_state.model.locations:
            if st.session_state.model.locations[node['id']]['lat'] in node_y and st.session_state.model.locations[node['id']]['lon'] in node_x:
                loc = (st.session_state.model.locations[node['id']]['lat'] + 0.1, 
                    st.session_state.model.locations[node['id']]['lon'] + 0.1)
            else:
                loc = (st.session_state.model.locations[node['id']]['lat'], 
                    st.session_state.model.locations[node['id']]['lon'])
            node_x.append(loc[1])  # lng
            node_y.append(loc[0])   # lat
        else:
            node_x.append(0)
            node_y.append(0)
        
        color = None
        if node['type'] == 'plant': 
            color = 'red'
            node_types.append('Plant')
        elif node['type'] == 'rdc': 
            color = 'blue'
            node_types.append('RDC')
        else: 
            color = 'orange'
            node_types.append('Store')
        node_text.append(node_types[-1] + ': ' + node['label'])
        node_colors.append(color)
        node_sizes.append(20)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    non_solution_edge_x = []
    non_solution_edge_y = []
    non_solution_edge_text = []
    
    for edge in model_data['edges']:
        if edge['from'] in st.session_state.model.locations and edge['to'] in st.session_state.model.locations:
            from_loc = st.session_state.model.locations[edge['from']]
            to_loc = st.session_state.model.locations[edge['to']]
            
            edge_x.extend([from_loc['lon'], to_loc['lon'], None])
            edge_y.extend([from_loc['lat'], to_loc['lat'], None])
            edge_text.append(edge['label'])

    for non_solution_edge in model_data['non_solution_edges']:
        if non_solution_edge['from'] in st.session_state.model.locations and non_solution_edge['to'] in st.session_state.model.locations:
            from_loc = st.session_state.model.locations[non_solution_edge['from']]
            to_loc = st.session_state.model.locations[non_solution_edge['to']]
            
            non_solution_edge_x.extend([from_loc['lon'], to_loc['lon'], None])
            non_solution_edge_y.extend([from_loc['lat'], to_loc['lat'], None])
            non_solution_edge_text.append(non_solution_edge['label'])

    # Create edge traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#5ce600'),
        hoverinfo='text',
        mode='lines',
        text=edge_text,
        hoverlabel=dict(bgcolor='white')
    )
    
    #Create non-essential edge traces
    non_solution_edge_trace = go.Scatter(
        x=non_solution_edge_x, y=non_solution_edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=non_solution_edge_text,
        hoverlabel=dict(bgcolor='white')
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=2,
            opacity=1,
        )
    )
    
    # Create figure
    fig = go.Figure(data=[non_solution_edge_trace, edge_trace, node_trace],
                   layout=go.Layout(
                       title='Network Optimization Model',
                       showlegend=True,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=30),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig



def main():
    st.title("Network Optimization Dashboard")
    
    # Initialize the model if not already done
    if not initialize_dashboard_model():
        st.error("Failed to initialize the dashboard model. Please check the logs for more information.")
        return
    
    # Initialize locations in session state if not exists
    if 'locations' not in st.session_state:
        st.session_state.locations = {}
    
    
    # Initialize chat history if not exists
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "Hello! I'm your supply chain optimization assistant. How can I help you today?"
        }]
        st.rerun()
    
    # Ensure chat messages are plain dictionaries (avoid Pydantic deprecation)
    def _to_dict(m):
        if isinstance(m, dict):
            return m
        # Prefer `model_dump` (Pydantic ≥2)
        if hasattr(m, "model_dump"):
            return m.model_dump()
        # Fallback to legacy `dict()`
        if hasattr(m, "dict"):
            return m.dict()
        # Last resort
        return vars(m)

    st.session_state.chat_messages = [_to_dict(msg) for msg in st.session_state.chat_messages]
    
    # Get model data with error handling
    model_data = get_model_data()
    if model_data is None:
        st.warning("No model data available. The model might be empty or invalid.")
    
    # Add timestamp to force refresh when model changes
    if 'last_model_update' not in st.session_state:
        st.session_state.last_model_update = time.time()
    
    # Add a hidden element to force refresh when model changes
    st.markdown(f'<div style="display:none">{st.session_state.last_model_update}</div>', unsafe_allow_html=True)
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Map View", "Network Graph"])
    
    with tab1:
        # Add new node UI
        add_new_node()
        
        # Create and display map if we have model data
        if model_data and 'nodes' in model_data and model_data['nodes']:
            m = create_map()
            st_folium(m, width='100%', height=500)
        else:
            st.warning("No nodes available to display on the map.")
    
    with tab2:
        # Create and display network graph if we have model data
        if model_data and 'nodes' in model_data and model_data['nodes']:
            fig = create_plotly_graph()
            if fig:
                st.plotly_chart(fig, use_container_width=False)
            else:
                st.warning("Could not generate network graph.")
        else:
            st.warning("No nodes available to display in the network graph.")

    tables = create_info_tables(st.session_state.model)
        
    if tables:
        st.header("Plants")
        st.dataframe(tables['plants'], use_container_width=True, hide_index=True)
        
        st.header("Regional Distribution Centers")
        st.dataframe(tables['rdcs'], use_container_width=True, hide_index=True)
        
        st.header("Stores")
        st.dataframe(tables['stores'], use_container_width=True, hide_index=True)
        
        st.header("Shipping Costs")
        # Apply styling to highlight optimal routes
        def highlight_optimal(row):
            return ['background-color: #90EE90' if row['In Optimal Solution'] == 'Yes' else '' for _ in row]
            
        styled_df = tables['shipping_costs'].style.apply(highlight_optimal, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("Rows highlighted in green indicate routes that are part of the optimal solution.")
    
    # Sidebar for model information and controls
    with st.sidebar:
        processingRequest = False
        st.header("Model Information")
        
        # Display model metrics if available
        if 'model' in st.session_state and st.session_state.model is not None:
            try:
                solution = st.session_state.model.solve()
                if solution and 'total_cost' in solution and solution['total_cost'] is not None:
                    st.metric("Total Cost", f"${float(solution['total_cost']):,.2f}")
                else:
                    st.warning("No solution available. The model might be empty or invalid.")
            except Exception as e:
                st.error(f"Error calculating model metrics: {str(e)}")
        
        # Chat interface container
        # Create containers with dynamic heights
        chat_history_container = st.container(height=475)
        chat_box_container = st.container()
        
        print(st.session_state.chat_messages)
        # Display chat history
        with chat_history_container:
            st.subheader("Chat with the AI")
            for message in st.session_state.chat_messages:
                if message["content"] is not None:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

        # Display chat box
        with chat_box_container:

            if prompt := st.chat_input("How can I help?"):
                with st.spinner("Processing..."):
                    # Get AI response
                    chat_with_ai(prompt, st.session_state.chat_messages)
                
                # Rerun to update the chat display
                print(st.session_state.chat_messages)
                st.rerun()


def update_city_coordinates(city_name, lat, lon):
    """Update the coordinates for a city in the CITY_COORDINATES dictionary.
    
    Args:
        city_name (str): Name of the city
        lat (float): Latitude of the city
        lon (float): Longitude of the city
    """
    global CITY_COORDINATES
    CITY_COORDINATES[city_name] = (lat, lon)
    st.session_state.locations[city_name] = (lat, lon)
    st.success(f"Updated coordinates for {city_name}: ({lat}, {lon})")

# Run the app
if __name__ == "__main__":
    main()
