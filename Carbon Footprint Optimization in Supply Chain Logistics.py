import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt

st.title("Carbon Footprint Optimization in Supply Chain Logistics")
st.write("""
Traditional logistics systems prioritize cost and time, often neglecting environmental impact.
This app uses a deep learning model to optimize delivery routes for minimal carbon emissions,
helping companies make green logistics decisions.
""")

# Dynamic input using Streamlit only
n_routes = st.number_input("Enter number of possible routes:", min_value=1, step=1, value=3)
route_data = []
for i in range(n_routes):
    st.subheader(f"Route {i+1}")
    distance = st.number_input(f"  Distance (km) for Route {i+1}:", min_value=0.0, step=1.0)
    fuel_usage = st.number_input(f"  Fuel usage (liters) for Route {i+1}:", min_value=0.0, step=0.1)
    weather = st.slider(f"  Weather severity (0-1) for Route {i+1}:", 0.0, 1.0, 0.0, 0.01)
    traffic = st.slider(f"  Traffic level (0-1) for Route {i+1}:", 0.0, 1.0, 0.0, 0.01)
    cargo_weight = st.number_input(f"  Cargo weight (tons) for Route {i+1}:", min_value=0.0, step=0.1)
    route_data.append([distance, fuel_usage, weather, traffic, cargo_weight])

if st.button("Optimize and Visualize"):
    route_data = np.array(route_data)

    # Dummy target: carbon emissions (for demonstration, real model needs real data)
    carbon_emissions = route_data[:, 0] * route_data[:, 1] * (1 + route_data[:, 2] + route_data[:, 3]) * (1 + 0.1 * route_data[:, 4])
    carbon_emissions = carbon_emissions.reshape(-1, 1)

    # Data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(route_data)

    # Build a simple deep learning model
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train the model (for demonstration, using the dummy target)
    model.fit(X_scaled, carbon_emissions, epochs=100, verbose=0)

    # Predict carbon emissions for each route
    predicted_emissions = model.predict(X_scaled).flatten()

    # Find the best route (minimum predicted emission)
    best_route_idx = np.argmin(predicted_emissions)
    st.success(f"Best route is Route {best_route_idx+1} with predicted carbon emission: {predicted_emissions[best_route_idx]:.2f}")

    # Visualization - Bar Chart
    st.subheader("Predicted Carbon Emissions for Each Route")
    st.bar_chart(pd.DataFrame({'Route': [f"Route {i+1}" for i in range(n_routes)], 'Predicted Emissions': predicted_emissions}).set_index('Route'))

    # Visualization - Route Graph
    st.subheader("Route Graph with Predicted Emissions")
    G = nx.DiGraph()
    for i in range(len(route_data)):
        G.add_node(f"Route {i+1}", emissions=predicted_emissions[i])
    for i in range(len(route_data)-1):
        G.add_edge(f"Route {i+1}", f"Route {i+2}")

    pos = nx.spring_layout(G)
    labels = {node: f"{node}\n{G.nodes[node]['emissions']:.2f}" for node in G.nodes}
    fig, ax = plt.subplots(figsize=(8, 5))
    nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=2000, ax=ax)
    plt.title("Route Graph with Predicted Emissions")
    st.pyplot(fig)
