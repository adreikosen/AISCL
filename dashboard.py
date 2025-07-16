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
    add_distribution_center_with_geocoding,
    add_plant_with_geocoding
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Geographic coordinates for the cities (latitude, longitude)
CITY_COORDINATES = {
    "Chicago_IL": (41.8781, -87.6298),
    "Houston_TX": (29.7604, -95.3698),
    "LosAngeles_CA": (34.0522, -118.2437),
    "NewYork_NY": (40.7128, -74.0060),
    "Atlanta_GA": (33.7490, -84.3880),
    "Denver_CO": (39.7392, -104.9903)
}

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

def get_model_data():
    """Extract data from the model for visualization."""
    try:
        # Check if model is initialized
        if not hasattr(st.session_state, 'model') or st.session_state.model is None:
            st.warning("Model not initialized. Creating a new instance...")
            st.session_state.model = app_model  
            
        # Get solution from model
        solution = st.session_state.model.solve()
        
        # Check if solution is valid
        if solution is None or not isinstance(solution, dict):
            st.warning("No valid solution found. The model might be empty or invalid.")
            return None
            
        # Initialize data structures
        nodes = []
        edges = []
        
        # Add plants with error handling
        for plant in getattr(st.session_state.model, 'plants', []):
            try:
                capacity = getattr(st.session_state.model, 'capacity', {}).get(plant, 'N/A')
                nodes.append({
                    'id': plant,
                    'type': 'plant',
                    'label': f"{plant}\n(Capacity: {capacity})",
                    'size': 15,
                    'color': 'blue'
                })
            except Exception as e:
                st.warning(f"Error processing plant {plant}: {str(e)}")
                continue
        
        # Add distribution centers with error handling
        for dc in getattr(st.session_state.model, 'distribution_centers', []):
            try:
                demand = getattr(st.session_state.model, 'demand', {}).get(dc, 'N/A')
                nodes.append({
                    'id': dc,
                    'type': 'dc',
                    'label': f"{dc}\n(Demand: {demand})",
                    'size': 15,
                    'color': 'green'
                })
            except Exception as e:
                st.warning(f"Error processing distribution center {dc}: {str(e)}")
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
                        'color': 'gray'
                    })
                except Exception as e:
                    st.warning(f"Error processing edge from {p} to {dc}: {str(e)}")
                    continue
        
        return {
            'nodes': nodes,
            'edges': edges,
            'solution': solution
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
    if model_data is None or 'nodes' not in model_data:
        st.warning("No model data available. Please initialize the model first.")
        return m
    
    # Add nodes to the map using model-provided locations
    for node in model_data['nodes']:
        node_id = node['id']
        # Get location from model locations
        if node_id in st.session_state.model.locations:
            loc = (st.session_state.model.locations[node_id]['lat'], 
                   st.session_state.model.locations[node_id]['lon'])
        else:
            # If no location is set, use default coordinates
            loc = (39.8283, -98.5795)
            
        # Add marker with popup
        popup = f"<b>{node['id']}</b><br>"
        if node['type'] == 'plant':
            popup += f"Type: Plant<br>Capacity: {st.session_state.model.capacity[node['id']]}"
            icon = folium.Icon(icon='industry', color='blue')
        else:
            popup += f"Type: Distribution Center<br>Demand: {st.session_state.model.demand[node['id']]}"
            icon = folium.Icon(icon='warehouse', color='green')
        
        folium.Marker(
            location=loc,
            popup=popup,
            icon=icon,
            tooltip=node['id']
        ).add_to(m)
    
    # Add edges to the map
    for edge in model_data['edges']:
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
    
    return m

def add_new_node():
    with st.expander("Add New Node"):
        node_type = st.selectbox("Node Type", ["Plant", "Distribution Center"])
        name = st.text_input("Node Name")
        city = st.text_input("City")
        state = st.text_input("State/Region")
        
        if node_type == "Plant":
            capacity = st.number_input("Capacity", min_value=0)
            if st.button("Add Plant"):
                result = add_plant_with_geocoding(name, capacity, city, state)
                st.success(result)
                st.rerun()
        else:
            demand = st.number_input("Demand", min_value=0)
            if st.button("Add Distribution Center"):
                result = add_distribution_center_with_geocoding(name, demand, city, state)
                st.success(result)
                st.rerun()

def create_plotly_graph():
    model_data = get_model_data()
    if not model_data:
        return go.Figure()
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in model_data['nodes']:
        if node['id'] in st.session_state.model.locations:
            loc = (st.session_state.model.locations[node['id']]['lat'], 
                   st.session_state.model.locations[node['id']]['lon'])
            node_x.append(loc[1])  # lng
            node_y.append(loc[0])   # lat
        else:
            node_x.append(0)
            node_y.append(0)
        
        node_text.append(node['label'])
        node_colors.append('blue' if node['type'] == 'plant' else 'green')
        node_sizes.append(20)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in model_data['edges']:
        if edge['from'] in st.session_state.model.locations and edge['to'] in st.session_state.model.locations:
            from_loc = st.session_state.model.locations[edge['from']]
            to_loc = st.session_state.model.locations[edge['to']]
            
            edge_x.extend([from_loc['lon'], to_loc['lon'], None])
            edge_y.extend([from_loc['lat'], to_loc['lat'], None])
            edge_text.append(edge['label'])
    
    # Create edge traces
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text,
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
            line_width=2
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Network Optimization Model',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=30),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def main():
    st.title("Network Optimization Dashboard")
    
    # Initialize the dashboard model
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
            st_folium(m, width=1200, height=600)
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
        
        # Display chat history
        with chat_history_container:
            st.subheader("Chat with the AI")
            for message in st.session_state.chat_messages:
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
    if 'model' not in st.session_state:
        initialize_dashboard_model()
    main()
