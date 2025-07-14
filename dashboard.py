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
    initialize_model,
    model as app_model,
    chat_with_ai,
    get_available_functions,
    execute_function_call
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Geographic coordinates for the cities (latitude, longitude)
CITY_COORDINATES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Patna": (25.5941, 85.1376),
    "Hyderabad": (17.3850, 78.4867),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577)
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

# Function to run model commands
def run_model_command(command):
    """Execute a model command using the AI assistant and return the response."""
    try:
        # Update model timestamp to trigger refresh
        st.session_state.last_model_update = time.time()
        # Initialize messages if not exists
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = [{
                "role": "system",
                "content": """You are a helpful assistant for network optimization. You can help users model and solve network optimization problems in supply chain.
                You can also help users create and compare different scenarios by taking snapshots of the model at different points in time.
                When comparing models, make sure to highlight the key differences and their business implications. A base model has been preloaded into the system."""
            }]
        
        # Get response from the AI - chat_with_ai will handle adding the user message
        response = chat_with_ai(command, st.session_state.chat_messages)
        
        # The response should already be a string from chat_with_ai
        response_content = str(response)
        
        # The chat_with_ai function now handles adding messages to the history
        # So we just need to ensure the last message is from the assistant
        if st.session_state.chat_messages and st.session_state.chat_messages[-1].get('role') != 'assistant':
            # If for some reason the last message isn't from the assistant, add it
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": response_content
            })
        
        return response_content
        
    except Exception as e:
        error_msg = f"Error in run_model_command: {str(e)}"
        if 'chat_messages' in st.session_state:
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
        return error_msg

# Function to get model data
def get_model_data():
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

# Function to create folium map
def create_map():
    # Create a base map centered on India (where the default plants/DCs are located)
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    # Add drawing tools
    Draw(export=True).add_to(m)
    
    # Get model data with error handling
    model_data = get_model_data()
    if model_data is None or 'nodes' not in model_data:
        st.warning("No model data available. Please initialize the model first.")
        return m
    
    # Define city coordinates
    CITY_COORDINATES = {
        "Ahmedabad": (23.0225, 72.5714),
        "Patna": (25.5941, 85.1376),
        "Hyderabad": (17.3850, 78.4867),
        "Bhopal": (23.2599, 77.4126),
        "Indore": (22.7196, 75.8577)
    }
    
    # Add nodes to the map
    for node in model_data['nodes']:
        node_id = node['id']
        # Use saved location, then city coordinates, or default to center of India
        if node_id in st.session_state.locations:
            loc = st.session_state.locations[node_id]
        elif node_id in CITY_COORDINATES:
            loc = CITY_COORDINATES[node_id]
            st.session_state.locations[node_id] = loc  # Save to session state
        else:
            # Default to center of India if city not in our coordinates
            loc = (20.5937, 78.9629)
            st.session_state.locations[node_id] = loc
        
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
    model_data = get_model_data()
    for edge in model_data['edges']:
        if edge['from'] in st.session_state.locations and edge['to'] in st.session_state.locations:
            folium.PolyLine(
                locations=[st.session_state.locations[edge['from']], st.session_state.locations[edge['to']]],
                color='gray',
                weight=2,
                opacity=0.7,
                tooltip=f"{edge['from']} to {edge['to']}: {edge['label']}"
            ).add_to(m)
    
    return m

# Function to create plotly network graph
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
        if node['id'] in st.session_state.locations:
            loc = st.session_state.locations[node['id']]
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
        if edge['from'] in st.session_state.locations and edge['to'] in st.session_state.locations:
            from_loc = st.session_state.locations[edge['from']]
            to_loc = st.session_state.locations[edge['to']]
            
            edge_x.extend([from_loc[1], to_loc[1], None])
            edge_y.extend([from_loc[0], to_loc[0], None])
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

# Main app layout
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
    
    # Ensure all messages are in dict format
    st.session_state.chat_messages = [
        msg if isinstance(msg, dict) else msg.dict() 
        for msg in st.session_state.chat_messages
    ]
    
    # Add a manual refresh button
    if st.sidebar.button("🔄 Refresh Model Data"):
        st.session_state.last_refresh = time.time()
        st.rerun()
    
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
        # Create and display map if we have model data
        if model_data and 'nodes' in model_data and model_data['nodes']:
            m = create_map()
            
            # Display the map
            st_data = st_folium(m, width=1200, height=600)
            
            # Handle map interactions
            if st_data and st_data.get("last_active_drawing"):
                drawing = st_data["last_active_drawing"]
                if drawing and "properties" in drawing and "feature" in st.session_state:
                    node_id = st.session_state.feature
                    try:
                        lat = drawing["geometry"]["coordinates"][1]
                        lon = drawing["geometry"]["coordinates"][0]
                        # Update both session state and CITY_COORDINATES
                        st.session_state.locations[node_id] = (lat, lon)
                        if node_id not in CITY_COORDINATES:
                            # Add a warning that this is a new city that should be added to CITY_COORDINATES
                            st.warning(f"City '{node_id}' is not in the coordinates database. Consider adding it to CITY_COORDINATES.")
                        st.rerun()
                    except (KeyError, IndexError) as e:
                        st.error(f"Error updating node location: {str(e)}")
        else:
            st.warning("No nodes available to display on the map.")
    
    with tab2:
        # Create and display network graph if we have model data
        if model_data and 'nodes' in model_data and model_data['nodes']:
            fig = create_plotly_graph()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not generate network graph.")
        else:
            st.warning("No nodes available to display in the network graph.")
    
    # Sidebar for model information and controls
    with st.sidebar:
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
        
        # Node selection for moving on map
        st.subheader("Move Node on Map")
        if model_data and 'nodes' in model_data and model_data['nodes']:
            node_options = [node['id'] for node in model_data['nodes']]
            selected_node = st.selectbox("Select a node to move:", [""] + node_options, key="node_selector_sidebar")
            if selected_node:
                st.session_state.feature = selected_node
                st.info(f"Click on the map to move {selected_node} to a new location")
        else:
            st.warning("No nodes available for movement.")
        
    # Chat interface in the sidebar
    with st.sidebar:
        st.subheader("Chat with the AI")
        
        # Add CSS to make the chat area scrollable
        st.markdown("""
        <style>
            .chat-container {
                display: flex;
                flex-direction: column;
                height: calc(100vh - 200px);
            }
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding-right: 10px;
                margin-bottom: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                background: #f9f9f9;
            }
            .stChatInput {
                position: sticky;
                bottom: 0;
                background: white;
                padding: 10px 0;
                z-index: 100;
            }
        </style>
        """, unsafe_allow_html=True)
        
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            # Convert to dict if needed (for OpenAI message objects)
            if hasattr(message, 'dict'):
                msg_dict = message.dict()
            else:
                msg_dict = message
            
            # Skip system messages or messages without role/content
            if not msg_dict or not isinstance(msg_dict, dict):
                continue
                
            role = msg_dict.get("role")
            content = msg_dict.get("content")
            
            # Only display non-system messages with content
            if role != "system" and content:
                with st.chat_message(role):
                    st.markdown(content)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-messages
        
        # Chat input at the bottom of the sidebar
        if prompt := st.chat_input("What would you like to do?"):
            if not prompt.strip():
                st.warning("Please enter a valid command.")
            else:
                try:
                    # Just call run_model_command - it will handle adding messages to history
                    response = run_model_command(prompt)
                    
                except Exception as e:
                    error_msg = f"Error processing command: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
                
                # Rerun to update the UI
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close chat-container
        

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
