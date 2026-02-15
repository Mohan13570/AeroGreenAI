import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib
import cv2
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(layout="wide", page_title="AeroGreen AI")
st.title("🌱 AeroGreen AI - Smart Farming")

# =====================================
# CONSTANTS
# =====================================
FARM_SIZE = 100
GRID_DIV = 5
PLANT_SPACING = 15
ZONES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]

# =====================================
# SESSION STATE INIT
# =====================================
def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

# Farm & Environment States
init_state("day", 0)
init_state("temperature", 28)
init_state("crop_health", 85.0)
init_state("rain_probability", 0)
init_state("rain_today", False)
init_state("yield_value", 0.0)
init_state("event_log", ["Day 0: System initialized. Awaiting simulation start."])
init_state("rodent_zone", None)

# ML Results
init_state("ml_results", {"disease": None, "soil": None, "pest": None, "rodent": None})

# Zone-Specific Data Maps
init_state("soil_moisture", {zone: random.randint(60, 80) for zone in ZONES})
init_state("zone_temperature", {zone: 28.0 + random.uniform(-2, 2) for zone in ZONES})
init_state("zone_nutrients", {zone: random.randint(70, 100) for zone in ZONES})
init_state("zone_microorganisms", {zone: random.randint(60, 90) for zone in ZONES}) 
init_state("fruit_size", {zone: 0.5 for zone in ZONES})

# Sustainability States
init_state("total_water_used", 0.0)
init_state("total_water_harvested", 0.0)
init_state("pesticide_liters", 0.0)
init_state("carbon_offset", 0.0)
init_state("current_health", 100)

# Drone Telemetry States
init_state("drone_position", (0, 0))
init_state("drone_battery", 100.0)
init_state("patrol_time_total", 0.0) 
init_state("detection_accuracy", 94.2)
init_state("drone_status", "Docked")

# Historical Tracking
init_state("temperature_history", [])
init_state("yield_history", [])
init_state("health_history", [])
init_state("moisture_history", [])

# AI Irrigation Intelligence
init_state("ai_irrigation_log", [])
init_state("water_saved_by_ai", 0.0)

# Yield Risk Engine
init_state("yield_risk_index", 0.0)

# Smart Alerts
init_state("ai_alerts", [])

# Drone Path Tracking
init_state("drone_path", [(0, 0)])

# 🚁 Drone state initialization
if "drone_position" not in st.session_state:
    st.session_state.drone_position = (0, 0)

if "drone_path" not in st.session_state:
    st.session_state.drone_path = [(0, 0)]

# =====================================
# PHASE 1 – INTELLIGENCE ENGINE
# =====================================

def calculate_yield_risk():
    """
    Calculates yield risk based on temperature, soil moisture, and crop health.
    0 = No risk
    100 = Severe risk
    """
    temp = st.session_state.temperature
    moisture = sum(st.session_state.soil_moisture.values()) / len(st.session_state.soil_moisture)
    health = st.session_state.crop_health

    risk = 0

    # Temperature risk
    if temp > 35:
        risk += (temp - 35) * 2
    elif temp < 10:
        risk += (10 - temp) * 2

    # Moisture risk
    if moisture < 30:
        risk += (30 - moisture) * 1.5
    elif moisture > 80:
        risk += (moisture - 80) * 1.2

    # Health risk influence
    risk += (100 - health) * 0.5

    # Clamp value
    risk = max(0, min(100, risk))

    return round(risk, 2)

# =====================================
# SIMULATION LOGIC FUNCTION
# =====================================
target_day = 45

def advance_simulation(days_to_advance):
    for _ in range(days_to_advance):
        st.session_state.day += 1

        # 1. WEATHER & ENVIRONMENT
        st.session_state.rain_probability = random.randint(0, 100)
        st.session_state.rain_today = st.session_state.rain_probability > 70
        st.session_state.temperature = random.randint(22, 36)
        plant_height_sim = min(100.0, 1.0 + (st.session_state.day * 2.2))

        # 2. DRONE TELEMETRY
        if st.session_state.rain_today:
            st.session_state.drone_status = "Grounded (Rain)"
            st.session_state.drone_battery = max(0.0, st.session_state.drone_battery - 0.5)
        else:
            st.session_state.drone_status = "Patrolling"
            daily_flight_time = random.uniform(15, 45) 
            st.session_state.patrol_time_total += daily_flight_time
            st.session_state.drone_battery = max(0.0, st.session_state.drone_battery - (daily_flight_time * 0.15))
            
            # Autonomous 2D drone movement (random direction)
            x, y = st.session_state.drone_position
            possible_moves = []
            
            # Check boundaries and add valid moves
            if x > 0:
                possible_moves.append((x - 1, y))   # Up
            if x < 1:
                possible_moves.append((x + 1, y))   # Down
            if y > 0:
                possible_moves.append((x, y - 1))   # Left
            if y < 4:
                possible_moves.append((x, y + 1))   # Right
            
            # Pick random valid move
            if possible_moves:
                new_position = random.choice(possible_moves)
                st.session_state.drone_position = new_position
                st.session_state.drone_path.append(new_position)

        if st.session_state.drone_battery < 20:
             st.session_state.drone_battery = 100.0
             st.session_state.event_log.insert(0, f"Day {st.session_state.day}: 🔋 Drone auto-recharged to 100%.")

        # 3. RODENT EVENT & PESTICIDE
        if st.session_state.day > 15 and random.random() < 0.3:
            st.session_state.rodent_zone = random.choice(ZONES)
            st.session_state.pesticide_liters += 0.15
            st.session_state.detection_accuracy = max(85.0, st.session_state.detection_accuracy - 0.2)
        else:
            st.session_state.rodent_zone = None
            st.session_state.detection_accuracy = min(99.9, st.session_state.detection_accuracy + 0.05)

        # 4. WATER, SOIL, NUTRIENTS, TEMP & MICROBES
        if st.session_state.rain_today:
            harvested = random.uniform(8.0, 15.0) 
            st.session_state.total_water_harvested += harvested
            for zone in ZONES:
                st.session_state.soil_moisture[zone] += random.randint(10, 20)
                st.session_state.soil_moisture[zone] = min(100, st.session_state.soil_moisture[zone])

        for zone in ZONES:
            # Moisture
            st.session_state.soil_moisture[zone] -= random.randint(4, 7)
            if st.session_state.soil_moisture[zone] < 30 and not st.session_state.rain_today:
                st.session_state.soil_moisture[zone] += 20
                st.session_state.total_water_used += 4.0
            st.session_state.soil_moisture[zone] = max(10, st.session_state.soil_moisture[zone])
            
            # Growth
            st.session_state.fruit_size[zone] = min(5.0, st.session_state.fruit_size[zone] + 0.2)
            
            # Zone temperature fluctuates around the daily global average
            st.session_state.zone_temperature[zone] = st.session_state.temperature + random.uniform(-3.0, 3.0)
            
            # Nutrients deplete over time based on growth
            st.session_state.zone_nutrients[zone] = max(0.0, st.session_state.zone_nutrients[zone] - random.uniform(0.5, 1.5))
            
            # Microorganisms logic: they thrive in good moisture (40-80%) and temps (20-30C)
            moist = st.session_state.soil_moisture[zone]
            temp = st.session_state.zone_temperature[zone]
            if 40 <= moist <= 80 and 20 <= temp <= 30:
                st.session_state.zone_microorganisms[zone] = min(100.0, st.session_state.zone_microorganisms[zone] + random.uniform(0.5, 2.0))
            else:
                # Extreme conditions kill off microbes
                st.session_state.zone_microorganisms[zone] = max(0.0, st.session_state.zone_microorganisms[zone] - random.uniform(0.5, 2.5))

        # 5. YIELD & CARBON
        avg_moisture = sum(st.session_state.soil_moisture.values()) / len(ZONES)
        avg_nutrients = sum(st.session_state.zone_nutrients.values()) / len(ZONES)
        avg_microbes = sum(st.session_state.zone_microorganisms.values()) / len(ZONES)
        
        # Yield factors in moisture, nutrients, and biological health
        st.session_state.yield_value = round((st.session_state.day * avg_moisture * (avg_nutrients/100) * (avg_microbes/100)) / 10, 2)
        st.session_state.carbon_offset = (plant_height_sim / 100) * len(ZONES) * 0.5

        # 6. EVENT LOGGING
        weather_str = "🌧️ Rained" if st.session_state.rain_today else "☀️ Sunny"
        log_msg = f"Day {st.session_state.day}: {weather_str} ({st.session_state.temperature}°C)."
        if st.session_state.rodent_zone:
            log_msg += f" 🚨 Rodent detected in {st.session_state.rodent_zone}."
        st.session_state.event_log.insert(0, log_msg)

    # =====================================
    # PHASE 1 – Dynamic Risk & History Update
    # =====================================

    # Update Yield Risk Index
    st.session_state.yield_risk_index = calculate_yield_risk()

    # Store Historical Data
    st.session_state.temperature_history.append(st.session_state.temperature)
    avg_m = sum(st.session_state.soil_moisture.values()) / len(ZONES)
    st.session_state.moisture_history.append(avg_m)
    st.session_state.yield_history.append(st.session_state.yield_value)
    st.session_state.health_history.append(st.session_state.crop_health)

    # Limit history size to avoid memory overflow
    if len(st.session_state.temperature_history) > 200:
        st.session_state.temperature_history.pop(0)
        st.session_state.moisture_history.pop(0)
        st.session_state.health_history.pop(0)

# =====================================
# INTERACTIVE SIDEBAR
# =====================================
with st.sidebar:
    st.header("🕹️ Farm Controls")
    
    if st.button("➡️ Next Day", use_container_width=True, type="primary"):
        advance_simulation(1)
        
    st.markdown("---")
    st.subheader("⏩ Fast Forward")
    skip_days = st.slider("Select days to skip:", min_value=1, max_value=45, value=5)
    if st.button(f"Advance {skip_days} Days", use_container_width=True):
        advance_simulation(skip_days)

    st.markdown("---")
    st.subheader("🛠️ Manual Override")
    
    if st.button("🌧️ Force Rain", use_container_width=True):
        st.session_state.rain_today = True
        st.session_state.rain_probability = 100
        for zone in ZONES:
            st.session_state.soil_moisture[zone] = min(100, st.session_state.soil_moisture[zone] + random.randint(10, 20))
        st.session_state.event_log.insert(0, f"Day {st.session_state.day}: 🌧️ Manual Rain Triggered!")

    if st.button("🚨 Spawn Rodent", use_container_width=True):
        target_zone = random.choice(ZONES)
        st.session_state.rodent_zone = target_zone
        st.session_state.event_log.insert(0, f"Day {st.session_state.day}: 🚨 Manual Rodent Spawned in {target_zone}!")

    if st.button("💧 Irrigate All", use_container_width=True):
        for zone in ZONES:
            st.session_state.soil_moisture[zone] = min(100, st.session_state.soil_moisture[zone] + 30)
        st.session_state.event_log.insert(0, f"Day {st.session_state.day}: 💧 Manual Irrigation Activated!")
        
    if st.button("🌱 Fertilize All", use_container_width=True):
        for zone in ZONES:
            st.session_state.zone_nutrients[zone] = min(100.0, st.session_state.zone_nutrients[zone] + 40.0)
            st.session_state.zone_microorganisms[zone] = min(100.0, st.session_state.zone_microorganisms[zone] + 15.0)
        st.session_state.event_log.insert(0, f"Day {st.session_state.day}: 🌱 Organic Fertilizer Applied!")
        
    st.markdown("---")
    if st.button("🔄 Reset Farm", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# =====================================
# GLOBAL RENDER VARIABLES
# =====================================
def get_stage(day):
    if day < 8: return "seedling"
    elif day < 15: return "flower"
    elif day < 30: return "green"
    elif day < 45: return "ripening"
    else: return "ripe"

stage = get_stage(st.session_state.day)
plant_height = min(100.0, 1.0 + (st.session_state.day * 2.2)) 
dynamic_drone_height = max(100.0, plant_height + 50.0)
dynamic_pole_height = max(50.0, plant_height + 20.0)

# =====================================
# REUSABLE 3D RENDERING FUNCTION
# =====================================
def create_farm_figure(overlay_type="None"):
    fig = go.Figure()

    # SOIL GRID SETUP
    x = np.linspace(-FARM_SIZE, FARM_SIZE, 60)
    y = np.linspace(-FARM_SIZE, FARM_SIZE, 60)
    xg, yg = np.meshgrid(x, y)
    zg = np.zeros_like(xg)

    # RENDER SOIL OR CHOSEN MAP OVERLAY
    if overlay_type != "None":
        surface_colors = np.zeros_like(xg)
        
        # Populate grid based on zone values
        for i in range(xg.shape[0]):
            for j in range(xg.shape[1]):
                y_val = yg[i, j]
                x_val = xg[i, j]
                row = "A" if y_val >= 0 else "B"
                col = min(5, max(1, int((x_val + FARM_SIZE) // (FARM_SIZE * 2 / 5)) + 1))
                zone = f"{row}{col}"
                
                if overlay_type == "Soil Moisture":
                    surface_colors[i, j] = st.session_state.soil_moisture.get(zone, 50)
                elif overlay_type == "Temperature Heat Map":
                    surface_colors[i, j] = st.session_state.zone_temperature.get(zone, 25)
                elif overlay_type == "Nutrient Distribution":
                    surface_colors[i, j] = st.session_state.zone_nutrients.get(zone, 50)
                elif overlay_type == "Microorganism Activity":
                    surface_colors[i, j] = st.session_state.zone_microorganisms.get(zone, 50)

        # Set color scales and limits based on overlay type
        if overlay_type == "Soil Moisture":
            cscale, cmin, cmax, title = "RdYlBu", 0, 100, "Moisture %"
        elif overlay_type == "Temperature Heat Map":
            cscale, cmin, cmax, title = "Plasma", 15, 45, "Temp °C"
        elif overlay_type == "Nutrient Distribution":
            cscale, cmin, cmax, title = "YlGn", 0, 100, "Nutrients %"
        elif overlay_type == "Microorganism Activity":
            cscale, cmin, cmax, title = "Viridis", 0, 100, "Bio-Index"

        fig.add_trace(go.Surface(
            x=xg, y=yg, z=zg,
            surfacecolor=surface_colors,
            colorscale=cscale, 
            cmin=cmin, cmax=cmax,
            showscale=True,
            colorbar=dict(title=title, x=0.9, y=0.5, len=0.6),
            hoverinfo='skip'
        ))
    else:
        # Default Dirt
        soil_color = "#3e2615" if st.session_state.rain_today else "#5c3a21"
        fig.add_trace(go.Surface(
            x=xg, y=yg, z=zg, 
            colorscale=[[0, soil_color], [1, soil_color]], 
            showscale=False, 
            hoverinfo='skip'
        ))

    # GRID LINES
    section = FARM_SIZE * 2 / GRID_DIV
    for i in range(GRID_DIV + 1):
        val = -FARM_SIZE + i * section
        fig.add_trace(go.Scatter3d(x=[val, val], y=[-FARM_SIZE, FARM_SIZE], z=[0.1, 0.1], mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=2), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[-FARM_SIZE, FARM_SIZE], y=[val, val], z=[0.1, 0.1], mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=2), showlegend=False, hoverinfo='skip'))

    # IRRIGATION PIPES
    for zone_index in range(10):
        angle = zone_index * (2 * math.pi / 10)
        px, py = 60 * math.cos(angle), 60 * math.sin(angle)
        fig.add_trace(go.Scatter3d(x=[px, px+10], y=[py, py+10], z=[0.5, 0.5], mode="lines", line=dict(color="blue", width=6), showlegend=False, hoverinfo='skip'))

    # BATCH RENDER PLANTS
    stem_x, stem_y, stem_z = [], [], []
    leaf_x, leaf_y, leaf_z = [], [], []
    fruit_x, fruit_y, fruit_z = [], [], []

    for x_pos in np.arange(-80, 80, PLANT_SPACING):
        for y_pos in np.arange(-80, 80, PLANT_SPACING):
            stem_x.extend([float(x_pos), float(x_pos), None]) 
            stem_y.extend([float(y_pos), float(y_pos), None])
            stem_z.extend([0.0, float(plant_height), None])
            
            leaf_x.append(float(x_pos))
            leaf_y.append(float(y_pos))
            leaf_z.append(float(plant_height))
            
            for h in np.arange(8.0, plant_height, 8.0):
                leaf_angle = h * 1.5 
                leaf_offset_x = 3.5 * math.cos(leaf_angle) 
                leaf_offset_y = 3.5 * math.sin(leaf_angle)
                leaf_x.append(float(x_pos) + leaf_offset_x)
                leaf_y.append(float(y_pos) + leaf_offset_y)
                leaf_z.append(h)
            
            if stage in ["flower", "green", "ripening", "ripe"]:
                fruit_height = float(plant_height) + 1.0 
                fruit_x.append(float(x_pos)) 
                fruit_y.append(float(y_pos))
                fruit_z.append(fruit_height)

    fig.add_trace(go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", line=dict(color="#4B3621", width=6), name="Stems", hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(x=leaf_x, y=leaf_y, z=leaf_z, mode="markers", marker=dict(size=5, color="darkgreen"), name="Leaves"))

    if fruit_x:
        color_map = {"flower": "yellow", "green": "lime", "ripening": "orange", "ripe": "red"}
        fig.add_trace(go.Scatter3d(x=fruit_x, y=fruit_y, z=fruit_z, mode="markers", marker=dict(size=8, color=color_map[stage]), name="Fruits"))

    # DRONE MOVEMENT
    if st.session_state.rodent_zone:
        zone_idx = ZONES.index(st.session_state.rodent_zone)
        zone_angle = zone_idx * (2 * math.pi / 10)
        drone_color = "red"
    else:
        row, col = st.session_state.drone_position
        zone_index = row * 5 + col
        zone_angle = zone_index * (2 * math.pi / 10)
        drone_color = "yellow"
        
    drone_x = 60 * math.cos(zone_angle)
    drone_y = 60 * math.sin(zone_angle)
    fig.add_trace(go.Scatter3d(x=[drone_x], y=[drone_y], z=[dynamic_drone_height], mode="markers", marker=dict(size=12, color=drone_color), name="Drone"))

    # CLOUDS
    cloud_rng = random.Random(st.session_state.day)
    cloud_x, cloud_y, cloud_z, cloud_sizes = [], [], [], []
    x_grid = np.linspace(-FARM_SIZE + 20, FARM_SIZE - 20, 5)
    y_grid = np.linspace(-FARM_SIZE + 20, FARM_SIZE - 20, 3)
    
    for gx in x_grid:
        for gy in y_grid:
            cx = gx + cloud_rng.uniform(-10, 10)
            cy = gy + cloud_rng.uniform(-10, 10)
            cz = cloud_rng.uniform(170, 190)
            for _ in range(12): 
                cloud_x.append(cx + cloud_rng.uniform(-15, 15))
                cloud_y.append(cy + cloud_rng.uniform(-15, 15))
                cloud_z.append(cz + cloud_rng.uniform(-5, 5))
                cloud_sizes.append(cloud_rng.uniform(30, 55))

    cloud_color = "#424b54" if st.session_state.rain_today else "#ffffff"
    fig.add_trace(go.Scatter3d(x=cloud_x, y=cloud_y, z=cloud_z, mode="markers", marker=dict(size=cloud_sizes, color=cloud_color, opacity=0.9 if st.session_state.rain_today else 0.7, line=dict(width=0)), name="Clouds", hoverinfo='skip'))

    # RAIN
    if st.session_state.rain_today:
        rain_x, rain_y, rain_z = [], [], []
        for _ in range(180):
            rx = cloud_rng.uniform(-FARM_SIZE, FARM_SIZE)
            ry = cloud_rng.uniform(-FARM_SIZE, FARM_SIZE)
            rz_top = cloud_rng.uniform(10, 160) 
            rain_x.extend([rx, rx, None])
            rain_y.extend([ry, ry, None])
            rain_z.extend([rz_top, rz_top - 6, None])
        fig.add_trace(go.Scatter3d(x=rain_x, y=rain_y, z=rain_z, mode="lines", line=dict(color="rgba(173, 216, 230, 0.6)", width=6), name="Rain", hoverinfo='skip'))
    
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False, range=[0, 200]), bgcolor="#546e7a" if st.session_state.rain_today else "skyblue", camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)), aspectmode='manual', aspectratio=dict(x=1, y=1, z=1.5)), margin=dict(l=0, r=0, b=0, t=0), height=650, showlegend=False)
    
    return fig

# =====================================
# TAB LAYOUT
# =====================================
tab1, tab2, tab3, tab4, tab5, tab6, tab_overview = st.tabs([
    "🛰️ 3D Farm Visual", 
    "🗺️ Spatial Mapping", 
    "📊 Analytics Dashboard", 
    "🌍 Sustainability", 
    "🚁 AeroGreen Drone Command",
    "🤖 ML Trained Models",
    "🏡 Farm Overview"
])

# -------------------------------------
# TAB 7: FARM OVERVIEW
# -------------------------------------
with tab_overview:
    st.markdown("### 🗺️ Farm Overview & Camera Coverage")
    st.caption("Health status heatmap (derived from live session values) and camera coverage report")

    # Build zone grid coords (2 rows x 5 cols)
    zones_grid = [[f"A{c+1}" for c in range(5)], [f"B{c+1}" for c in range(5)]]
    zone_coords = {}
    for i in range(2):
        for j in range(5):
            zone_coords[zones_grid[i][j]] = (j + 0.5, 1.5 - i)

    # Determine health status from session state values
    def compute_zone_health(z):
        moist = st.session_state.soil_moisture.get(z, 50)
        nut = st.session_state.zone_nutrients.get(z, 75)
        micro = st.session_state.zone_microorganisms.get(z, 70)
        # Danger if any critical low
        if moist < 30 or nut < 40 or micro < 40:
            return "Danger"
        # Warning if any suboptimal
        if moist < 45 or nut < 60 or micro < 60:
            return "Warning"
        return "Healthy"

    health_status = {z: compute_zone_health(z) for row in zones_grid for z in row}
    color_map = {"Healthy": "#b9fbc0", "Warning": "#ffe066", "Danger": "#ff6b6b"}

    # Camera positions and coverage radius (engineering placeholders)
    camera_radius = 1.8
    cameras = [
        (2.5, 2.2, "North Cam"),
        (2.5, -0.2, "South Cam"),
        (-0.2, 1.0, "West Cam"),
        (5.2, 1.0, "East Cam")
    ]

    coverage_report = {cam[2]: [] for cam in cameras}
    for cam_x, cam_y, cam_name in cameras:
        for zone, (zx, zy) in zone_coords.items():
            distance = math.sqrt((cam_x - zx) ** 2 + (cam_y - zy) ** 2)
            if distance <= camera_radius:
                coverage_report[cam_name].append(zone)

    # Draw the figure using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    for zone, (x, y) in zone_coords.items():
        color = color_map.get(health_status.get(zone, "Healthy"), "#b9fbc0")
        rect = patches.FancyBboxPatch((x - 0.5, y - 0.5), 0.96, 0.96, boxstyle="round,pad=0.02", facecolor=color, edgecolor="white", zorder=1)
        ax.add_patch(rect)
        ax.text(x, y, zone, ha='center', va='center', fontsize=10, fontweight='bold')

    for cam_x, cam_y, cam_name in cameras:
        circle = plt.Circle((cam_x, cam_y), camera_radius, color="#4dabf7", fill=True, alpha=0.1, linestyle='--', zorder=0)
        outline = plt.Circle((cam_x, cam_y), camera_radius, color="#4dabf7", fill=False, linewidth=1.5, linestyle='--', zorder=2)
        ax.add_patch(circle)
        ax.add_patch(outline)
        ax.text(cam_x, cam_y, "📷", fontsize=15, ha='center', va='center', zorder=3)

    legend_elements = [patches.Patch(facecolor=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 3)
    ax.axis("off")
    ax.set_aspect("equal")
    st.pyplot(fig)

    # Coverage analytics
    st.subheader("📡 Coverage Report")
    cols = st.columns(4)
    for i, (cam_name, covered_zones) in enumerate(coverage_report.items()):
        with cols[i]:
            st.metric(label=cam_name, value=f"{len(covered_zones)} Zones")
            st.caption(", ".join(covered_zones) if covered_zones else "No coverage")

    all_covered = set([z for sub in coverage_report.values() for z in sub])
    blind_spots = set(zone_coords.keys()) - all_covered
    if blind_spots:
        st.error(f"⚠️ Blind Spots Detected: {', '.join(sorted(blind_spots))}")
    else:
        st.success("✅ Full Coverage Confirmed")

# -------------------------------------
# TAB 1: 3D FARM VISUAL
# -------------------------------------
with tab1:
    st.markdown(f"### 🌞 Farm Day: {st.session_state.day} | 🌱 Stage: {stage.capitalize()} | 📏 Avg Height: {round(plant_height, 2)} cm")
    fig = go.Figure()

    # SOIL
    x = np.linspace(-FARM_SIZE, FARM_SIZE, 60)
    y = np.linspace(-FARM_SIZE, FARM_SIZE, 60)
    xg, yg = np.meshgrid(x, y)
    zg = np.zeros_like(xg)
    soil_color = "#3e2615" if st.session_state.rain_today else "#5c3a21"
    fig.add_trace(go.Surface(x=xg, y=yg, z=zg, colorscale=[[0, soil_color], [1, soil_color]], showscale=False, hoverinfo='skip'))

    # GRID
    section = FARM_SIZE * 2 / GRID_DIV
    for i in range(GRID_DIV + 1):
        val = -FARM_SIZE + i * section
        fig.add_trace(go.Scatter3d(x=[val, val], y=[-FARM_SIZE, FARM_SIZE], z=[0.1, 0.1], mode="lines", line=dict(color="white", width=4), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[-FARM_SIZE, FARM_SIZE], y=[val, val], z=[0.1, 0.1], mode="lines", line=dict(color="white", width=4), showlegend=False, hoverinfo='skip'))

    # IRRIGATION PIPES
    for zone_index in range(10):
        angle = zone_index * (2 * math.pi / 10)
        px, py = 60 * math.cos(angle), 60 * math.sin(angle)
        fig.add_trace(go.Scatter3d(x=[px, px+10], y=[py, py+10], z=[0.5, 0.5], mode="lines", line=dict(color="blue", width=6), showlegend=False, hoverinfo='skip'))

    # BATCH RENDER PLANTS
    stem_x, stem_y, stem_z = [], [], []
    leaf_x, leaf_y, leaf_z = [], [], []
    fruit_x, fruit_y, fruit_z = [], [], []

    for x_pos in np.arange(-80, 80, PLANT_SPACING):
        for y_pos in np.arange(-80, 80, PLANT_SPACING):
            
            # 1. STEMS 
            stem_x.extend([float(x_pos), float(x_pos), None]) 
            stem_y.extend([float(y_pos), float(y_pos), None])
            stem_z.extend([0.0, float(plant_height), None])
            
            # 2. TOP LEAF
            leaf_x.append(float(x_pos))
            leaf_y.append(float(y_pos))
            leaf_z.append(float(plant_height))
            
            # 3. SIDE LEAVES 
            for h in np.arange(8.0, plant_height, 8.0):
                leaf_angle = h * 1.5 
                leaf_offset_x = 3.5 * math.cos(leaf_angle) 
                leaf_offset_y = 3.5 * math.sin(leaf_angle)
                
                leaf_x.append(float(x_pos) + leaf_offset_x)
                leaf_y.append(float(y_pos) + leaf_offset_y)
                leaf_z.append(h)
            
            # 4. FRUITS
            if stage in ["flower", "green", "ripening", "ripe"]:
                fruit_height = float(plant_height) + 1.0 
                fruit_x.append(float(x_pos)) 
                fruit_y.append(float(y_pos))
                fruit_z.append(fruit_height)

    # Add Stems Trace
    fig.add_trace(go.Scatter3d(x=stem_x, y=stem_y, z=stem_z, mode="lines", line=dict(color="#4B3621", width=6), name="Stems", hoverinfo='skip'))
    
    # Add Leaves Trace
    fig.add_trace(go.Scatter3d(x=leaf_x, y=leaf_y, z=leaf_z, mode="markers", marker=dict(size=5, color="darkgreen"), name="Leaves"))

    # Add Fruits Trace
    if fruit_x:
        color_map = {"flower": "yellow", "green": "lime", "ripening": "orange", "ripe": "red"}
        fig.add_trace(go.Scatter3d(x=fruit_x, y=fruit_y, z=fruit_z, mode="markers", marker=dict(size=8, color=color_map[stage]), name="Fruits"))

    # CAMERAS
    for pos in [(-50, -50), (50, -50), (-50, 50), (50, 50), (0, 0)]:
        fig.add_trace(go.Scatter3d(x=[pos[0], pos[0]], y=[pos[1], pos[1]], z=[0, dynamic_pole_height], mode="lines", line=dict(color="black", width=8), showlegend=False, hoverinfo='skip'))

    # DRONE MOVEMENT
    if st.session_state.rodent_zone:
        zone_idx = ZONES.index(st.session_state.rodent_zone)
        zone_angle = zone_idx * (2 * math.pi / 10)
        drone_color = "red"
    else:
        row, col = st.session_state.drone_position
        zone_index = row * 5 + col
        zone_angle = zone_index * (2 * math.pi / 10)
        drone_color = "yellow"
        
    drone_x = 60 * math.cos(zone_angle)
    drone_y = 60 * math.sin(zone_angle)

    fig.add_trace(go.Scatter3d(x=[drone_x], y=[drone_y], z=[dynamic_drone_height], mode="markers", marker=dict(size=12, color=drone_color), name="Drone"))

   # =====================================
    # ORGANIZED CLOUDS (Grid Layout, No Outline)
    # =====================================
    cloud_rng = random.Random(st.session_state.day)
    cloud_x, cloud_y, cloud_z, cloud_sizes = [], [], [], []

    # Organize 15 clusters into a clean 3x5 grid
    num_cols, num_rows = 5, 3
    x_grid = np.linspace(-FARM_SIZE + 20, FARM_SIZE - 20, num_cols)
    y_grid = np.linspace(-FARM_SIZE + 20, FARM_SIZE - 20, num_rows)
    
    for gx in x_grid:
        for gy in y_grid:
            # Subtle jitter for a natural yet organized feel
            cx = gx + cloud_rng.uniform(-10, 10)
            cy = gy + cloud_rng.uniform(-10, 10)
            cz = cloud_rng.uniform(170, 190)
            
            for _ in range(12): 
                cloud_x.append(cx + cloud_rng.uniform(-15, 15))
                cloud_y.append(cy + cloud_rng.uniform(-15, 15))
                cloud_z.append(cz + cloud_rng.uniform(-5, 5))
                cloud_sizes.append(cloud_rng.uniform(30, 55))

    # Clean styling without outlines
    cloud_color = "#424b54" if st.session_state.rain_today else "#ffffff"
    cloud_opacity = 0.9 if st.session_state.rain_today else 0.7

    fig.add_trace(go.Scatter3d(
        x=cloud_x, y=cloud_y, z=cloud_z,
        mode="markers",
        marker=dict(
            size=cloud_sizes, 
            color=cloud_color, 
            opacity=cloud_opacity, 
            line=dict(width=0) # Outline strictly removed
        ),
        name="Clouds",
        hoverinfo='skip'
    ))

    # Add Rain Droplets if it's raining
    if st.session_state.rain_today:
        rain_x, rain_y, rain_z = [], [], []
        # Generate 400 rain streaks
        for _ in range(180):
            rx = cloud_rng.uniform(-FARM_SIZE, FARM_SIZE)
            ry = cloud_rng.uniform(-FARM_SIZE, FARM_SIZE)
            rz_top = cloud_rng.uniform(10, 160) # Random height between ground and clouds
            rz_bottom = rz_top - 6 # Make the rain droplet 8 units long
            
            # Use None to break the lines, so they render as separate droplets
            rain_x.extend([rx, rx, None])
            rain_y.extend([ry, ry, None])
            rain_z.extend([rz_top, rz_bottom, None])

        fig.add_trace(go.Scatter3d(
            x=rain_x, y=rain_y, z=rain_z,
            mode="lines",
            line=dict(color="rgba(173, 216, 230, 0.6)", width=6), # Light, semi-transparent blue
            name="Rain",
            hoverinfo='skip'
        ))
    
    # Update layout to STRETCH the Z-axis
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False, range=[0, 200]), 
            bgcolor="#546e7a" if st.session_state.rain_today else "skyblue",
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)), 
            aspectmode='manual', 
            aspectratio=dict(x=1, y=1, z=1.5) 
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=650, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    

# -------------------------------------
# TAB 2: SPATIAL MAPPING
# -------------------------------------
with tab2:
    st.markdown(f"### 🗺️ Live Spatial GIS Map (Day {st.session_state.day})")
    
    # Selection and Stats
    overlay_choice = st.radio("Select Map Layer:", ["Soil Moisture", "Temperature Heat Map", "Nutrient Distribution", "Microorganism Activity"], horizontal=True)
    
    col_map, col_table = st.columns([2, 1])
    
    with col_map:
        st.plotly_chart(create_farm_figure(overlay_type=overlay_choice), use_container_width=True)
    
    with col_table:
        st.markdown(f"#### 📋 {overlay_choice} Data")
        
        # Mapping selected criteria to actual state keys
        criteria_map = {
            "Soil Moisture": ("soil_moisture", "%"),
            "Temperature Heat Map": ("zone_temperature", "°C"),
            "Nutrient Distribution": ("zone_nutrients", "%"),
            "Microorganism Activity": ("zone_microorganisms", " Index")
        }
        state_key, unit = criteria_map[overlay_choice]
        
        # Construct DataFrame for the specific selection
        zone_data = []
        for z in ZONES:
            val = st.session_state[state_key][z]
            zone_data.append({"Zone": z, f"Value ({unit})": round(val, 1)})
        
        analysis_df = pd.DataFrame(zone_data)
        
        # Add visual highlighting for the table
        st.dataframe(
            analysis_df.style.background_gradient(cmap="Greens" if "Nutrient" in overlay_choice else "Blues" if "Moisture" in overlay_choice else "YlOrRd"),
            use_container_width=True,
            hide_index=True,
            height=450
        )
        
        avg_val = sum(st.session_state[state_key].values()) / 10
        st.metric(f"Current Global Avg ({unit})", round(avg_val, 2))

# -------------------------------------
# TAB 3: ANALYTICS DASHBOARD
# -------------------------------------
with tab3:
    st.markdown("### 📈 Live Farm Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("🌡️ Temperature", f"{st.session_state.temperature}°C")
    m2.metric("🌧️ Rain Probability", f"{st.session_state.rain_probability}%")
    m3.metric("🌱 Growth Stage", stage.capitalize())
    m4.metric("🌾 Plant Height", f"{round(plant_height, 2)} cm")
    m5.metric("📦 Est. Yield", f"{st.session_state.yield_value} kg")
    m6.metric("Yield Risk Index (%)", f"{st.session_state.yield_risk_index}")

    st.markdown("---")
    st.markdown("### 📊 Historical Trends")

    if len(st.session_state.temperature_history) > 0:

        history_df = pd.DataFrame({
            "Day": list(range(len(st.session_state.temperature_history))),
            "Temperature": st.session_state.temperature_history,
            "Moisture": st.session_state.moisture_history,
            "Yield": st.session_state.yield_history,
            "Health": st.session_state.health_history
        })

        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(
            x=history_df["Day"],
            y=history_df["Temperature"],
            mode="lines",
            name="Temperature"
        ))
        fig_history.add_trace(go.Scatter(
            x=history_df["Day"],
            y=history_df["Moisture"],
            mode="lines",
            name="Moisture"
        ))
        fig_history.add_trace(go.Scatter(
            x=history_df["Day"],
            y=history_df["Yield"],
            mode="lines",
            name="Yield"
        ))
        fig_history.add_trace(go.Scatter(
            x=history_df["Day"],
            y=history_df["Health"],
            mode="lines",
            name="Health"
        ))

        fig_history.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_history, use_container_width=True)
    
    st.markdown("---")
    col_table, col_log = st.columns([1.5, 1])
    with col_table:
        st.markdown("#### 📍 Zone Analysis Data")
        df = pd.DataFrame({
            "Zone": ZONES,
            "Moisture (%)": [st.session_state.soil_moisture[z] for z in ZONES],
            "Nutrients (%)": [round(st.session_state.zone_nutrients[z], 1) for z in ZONES],
            "Bio-Index": [round(st.session_state.zone_microorganisms[z], 1) for z in ZONES],
            "Temp (°C)": [round(st.session_state.zone_temperature[z], 1) for z in ZONES]
        })
        df["Status"] = df["Moisture (%)"].apply(lambda m: "🔴 Auto-Irrigating" if m < 30 else "🔵 High Moisture" if m > 85 else "🟢 Optimal")
        df["Threats"] = ["🚨 Rodent" if z == st.session_state.rodent_zone else "✅ Clear" for z in ZONES]
        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_log:
        st.markdown("#### 📜 Farm Event Log")
        with st.container(height=280):
            for entry in st.session_state.event_log[:20]:
                st.text(entry)

    st.markdown("---")
    st.markdown("### 📋 Season Summary")
    
    progress = min(100, int((st.session_state.day / target_day) * 100))
    s1, s2, s3 = st.columns(3)
    
    with s1:
        st.write("**📅 Season Timeline**")
        st.progress(progress / 100)
        st.caption(f"{progress}% Complete ({st.session_state.day} / {target_day} Days)")
        
    with s2:
        st.write("**💪 Crop Health Outlook**")
        avg_m = sum(st.session_state.soil_moisture.values()) / len(ZONES)
        health_status = "⚠️ Stress Detected" if avg_m < 35 else "💧 Saturated" if avg_m > 80 else "✅ Vigorous"
        st.info(f"Current Condition: {health_status}")
        
    with s3:
        st.write("**📊 Yield Prediction**")
        final_est = round(st.session_state.yield_value * (target_day / max(1, st.session_state.day)), 2)
        st.success(f"Harvest Target: ~{final_est} kg")

# -------------------------------------
# TAB 4: SUSTAINABILITY
# -------------------------------------
with tab4:
    st.markdown("### 🌍 Environmental Impact & Health Dashboard")
    s1, s2, s3 = st.columns(3)
    
    net_water = st.session_state.total_water_used - st.session_state.total_water_harvested
    s1.metric("💧 Net Water Usage", f"{round(net_water, 1)} L", delta=f"-{round(st.session_state.total_water_harvested, 1)} Harvested")
    
    pest_status = "Optimal" if st.session_state.pesticide_liters < (st.session_state.day * 0.1) else "High"
    s2.metric("🧪 Pesticide Applied", f"{round(st.session_state.pesticide_liters, 2)} L", pest_status, delta_color="inverse")
    s3.metric("🌳 CO2 Sequestered", f"{round(st.session_state.carbon_offset, 2)} kg", "Carbon Sink")

    st.markdown("---")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("#### 🌱 Live Crop Health")
        health_fig = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.current_health, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"}, 'steps': [{'range': [0, 40], 'color': "#ff4b4b"}, {'range': [40, 75], 'color': "#ffa500"}, {'range': [75, 100], 'color': "#2ca02c"}], 'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': st.session_state.current_health}}))
        health_fig.update_layout(height=300, margin=dict(t=50, b=0, l=10, r=10))
        st.plotly_chart(health_fig, use_container_width=True)

    with g2:
        st.markdown("#### 🌧️ Rainwater Harvested")
        harvest_fig = go.Figure(go.Indicator(mode="gauge+number", value=st.session_state.total_water_harvested, gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "#00ccff"}}))
        harvest_fig.update_layout(height=300, margin=dict(t=50, b=0, l=10, r=10))
        st.plotly_chart(harvest_fig, use_container_width=True)
    
    final_est = round(st.session_state.yield_value * (target_day/ max(1, st.session_state.day)), 2)
    st.success(f"📊 Projected Harvest: ~{final_est} kg")
    
    if st.session_state.current_health < 60:
        st.warning("💡 **AI Health Alert:** High temperature or uneven moisture is stressing the crops. Check Zone Analytics.")
    else:
        st.info("💡 **AI Sustainability Tip:** Crop health is stable. Rain harvesting is currently meeting a significant portion of your water budget.")

# -------------------------------------
# TAB 5: AeroGreen Drone Command
# -------------------------------------
with tab5:
    st.markdown("### 🛰️ Drone Command & Telemetry")
    d1, d2, d3, d4 = st.columns(4)

    b_color = "normal" if st.session_state.drone_battery > 30 else "inverse"
    d1.metric("🔋 Battery Life", f"{round(st.session_state.drone_battery, 1)}%", delta_color=b_color)
    d2.metric("⏱️ Total Flight Time", f"{round(st.session_state.patrol_time_total / 60, 1)} hrs")
    d3.metric("🎯 Detection Accuracy", f"{round(st.session_state.detection_accuracy, 1)}%")
    status_icon = "🟢" if st.session_state.drone_status == "Patrolling" else "🔴"
    d4.metric("📡 System Status", f"{status_icon} {st.session_state.drone_status}")

    st.markdown("---")
    col_info, col_gauge = st.columns([1, 1])

    with col_info:
        st.markdown("#### ☁️ Flight Weather Safety")
        wind_speed = round(random.uniform(5, 25) if st.session_state.rain_today else random.uniform(2, 12), 1)
        st.write(f"**Current Wind Speed:** {wind_speed} km/h")
        if wind_speed > 20:
            st.error("⚠️ High Wind Warning: Automated patrol restricted.")
        else:
            st.success("✅ Wind speeds within safe flight envelope.")
            
        st.write(f"**Visibility:** {'Low' if st.session_state.rain_today else 'High'}")
        st.write(f"**Last Rodent Ping:** {st.session_state.rodent_zone if st.session_state.rodent_zone else 'None'}")

    with col_gauge:
        batt_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=st.session_state.drone_battery,
            title={'text': "Battery Charge"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "lime" if st.session_state.drone_battery > 20 else "red"},
                'steps': [
                    {'range': [0, 20], 'color': "maroon"},
                    {'range': [20, 100], 'color': "#232323"}
                ]
            }
        ))
        batt_fig.update_layout(height=250, margin=dict(t=0, b=0))
        st.plotly_chart(batt_fig, use_container_width=True)

    # ---------------------------
    # NAVIGATION GRID
    # ---------------------------

    st.markdown("### 🛰️ Drone Navigation Grid")

    rows = 2
    cols = 5

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("⬅ Left", key="tab5_left"):
        x, y = st.session_state.drone_position
        if y > 0:
            st.session_state.drone_position = (x, y - 1)

    if col2.button("➡ Right", key="tab5_right"):
        x, y = st.session_state.drone_position
        if y < cols - 1:
            st.session_state.drone_position = (x, y + 1)

    if col3.button("⬆ Up", key="tab5_up"):
        x, y = st.session_state.drone_position
        if x > 0:
            st.session_state.drone_position = (x - 1, y)

    if col4.button("⬇ Down", key="tab5_down"):
        x, y = st.session_state.drone_position
        if x < rows - 1:
            st.session_state.drone_position = (x + 1, y)

    # Save Path
    if st.session_state.drone_position not in st.session_state.drone_path:
        st.session_state.drone_path.append(st.session_state.drone_position)

    # Draw Grid
    fig_nav = go.Figure()

    for i in range(rows):
        for j in range(cols):
            fig_nav.add_shape(
                type="rect",
                x0=j, y0=i,
                x1=j+1, y1=i+1,
                line=dict(color="gray")
            )

    if len(st.session_state.drone_path) > 1:
        x_path = [pos[1] + 0.5 for pos in st.session_state.drone_path]
        y_path = [pos[0] + 0.5 for pos in st.session_state.drone_path]

        fig_nav.add_trace(go.Scatter(
            x=x_path,
            y=y_path,
            mode="lines+markers",
            name="Drone Path"
        ))

    x, y = st.session_state.drone_position

    fig_nav.add_trace(go.Scatter(
        x=[y + 0.5],
        y=[x + 0.5],
        mode="markers",
        marker=dict(size=20),
        name="Current Position"
    ))

    fig_nav.update_layout(
        height=450,
        xaxis=dict(range=[0, cols], showgrid=False, zeroline=False),
        yaxis=dict(range=[0, rows], showgrid=False, zeroline=False),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig_nav.update_yaxes(scaleanchor="x", scaleratio=1)

    st.plotly_chart(fig_nav, use_container_width=True)

    zone_labels = [
        ["A1", "A2", "A3", "A4", "A5"],
        ["B1", "B2", "B3", "B4", "B5"]
    ]

    current_zone = zone_labels[x][y]
    st.success(f"📍 Drone currently in Zone: {current_zone}")

    if st.button("🔄 Reset Drone Path", key="tab5_reset"):
        st.session_state.drone_position = (0, 0)
        st.session_state.drone_path = [(0, 0)]

# -------------------------------------
# TAB 6: ML TRAINED MODELS
# -------------------------------------
with tab6:
    st.markdown("### 🤖 ML Trained Models - Detection Systems")
    st.markdown("Upload images and run the trained detection models for Disease, Soil, Pest, and Rodent.")

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(["🌿 Disease", "🌍 Soil", "🪲 Pest", "🐀 Rodent"])

    def load_ml_model(name):
        try:
            return joblib.load(f"{name}_model.pkl")
        except Exception:
            return None

    with ml_tab1:
        st.markdown("#### 🌿 Disease Detection")
        disease_file = st.file_uploader("Upload tomato leaf image", type=["jpg","jpeg","png"], key="disease_upload")
        if disease_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(disease_file, width=300)
            with col2:
                if st.button("Run Disease Detection", key="disease_btn"):
                    model = load_ml_model("disease")
                    if model is None:
                        st.error("disease_model.pkl not found. Train model first.")
                    else:
                        img = Image.open(disease_file).resize((128,128))
                        arr = np.array(img)
                        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                        feat = hsv.flatten().reshape(1,-1)
                        pred = model.predict(feat)[0]
                        conf = model.predict_proba(feat).max()
                        classes = ["Healthy", "Leaf Spot", "Blight", "Mosaic Virus"]
                        st.success(f"Detected: {classes[pred]}")
                        st.info(f"Confidence: {round(conf*100,2)}%")

    with ml_tab2:
        st.markdown("#### 🌍 Soil Moisture Detection")
        soil_file = st.file_uploader("Upload soil image", type=["jpg","jpeg","png"], key="soil_upload")
        if soil_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(soil_file, width=300)
            with col2:
                if st.button("Run Soil Detection", key="soil_btn"):
                    model = load_ml_model("soil")
                    if model is None:
                        st.error("soil_model.pkl not found. Train model first.")
                    else:
                        img = Image.open(soil_file).resize((128,128))
                        arr = np.array(img)
                        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                        feat = hsv.flatten().reshape(1,-1)
                        pred = model.predict(feat)[0]
                        conf = model.predict_proba(feat).max()
                        classes = ["Dry Soil", "Moist Soil", "Waterlogged Soil"]
                        st.success(f"Detected: {classes[pred]}")
                        st.info(f"Confidence: {round(conf*100,2)}%")

    with ml_tab3:
        st.markdown("#### 🪲 Pest Detection")
        pest_file = st.file_uploader("Upload plant/leaf image", type=["jpg","jpeg","png"], key="pest_upload")
        if pest_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(pest_file, width=300)
            with col2:
                if st.button("Run Pest Detection", key="pest_btn"):
                    model = load_ml_model("pest")
                    if model is None:
                        st.error("pest_model.pkl not found. Train model first.")
                    else:
                        img = Image.open(pest_file).resize((128,128))
                        arr = np.array(img)
                        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                        feat = hsv.flatten().reshape(1,-1)
                        pred = model.predict(feat)[0]
                        conf = model.predict_proba(feat).max()
                        classes = ["No Pest Detected", "Pest Detected"]
                        st.success(f"Detected: {classes[pred]}")
                        st.info(f"Confidence: {round(conf*100,2)}%")

    with ml_tab4:
        st.markdown("#### 🐀 Rodent Detection")
        rodent_file = st.file_uploader("Upload field image", type=["jpg","jpeg","png"], key="rodent_upload")
        if rodent_file:
            col1, col2 = st.columns(2)
            with col1:
                st.image(rodent_file, width=300)
            with col2:
                if st.button("Run Rodent Detection", key="rodent_btn"):
                    model = load_ml_model("rodent")
                    if model is None:
                        st.error("rodent_model.pkl not found. Train model first.")
                    else:
                        img = Image.open(rodent_file).resize((128,128))
                        arr = np.array(img)
                        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                        feat = hsv.flatten().reshape(1,-1)
                        pred = model.predict(feat)[0]
                        conf = model.predict_proba(feat).max()
                        classes = ["No Rodent Detected", "Rodent Detected"]
                        st.success(f"Detected: {classes[pred]}")
                        st.info(f"Confidence: {round(conf*100,2)}%")
