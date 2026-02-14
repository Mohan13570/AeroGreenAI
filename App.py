import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import math

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
ZONES = ["A1", "A2", "A3", "A4", "A5",
         "B1", "B2", "B3", "B4", "B5"]

# =====================================
# SESSION STATE INIT
# =====================================

def init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

init_state("day", 0)
init_state("soil_moisture", {zone: random.randint(60, 80) for zone in ZONES})
init_state("rain_probability", 0)
init_state("rain_today", False)
init_state("rodent_zone", None)
init_state("drone_position", 0)

# Dashboard variables
init_state("temperature", 28)
init_state("yield_value", 0.0)
init_state("fruit_size", {zone: 0.5 for zone in ZONES})
init_state("event_log", ["Day 0: System initialized. Awaiting simulation start."])

# =====================================
# SIMULATION LOGIC FUNCTION
# =====================================

def advance_simulation(days_to_advance):
    for _ in range(days_to_advance):
        st.session_state.day += 1

        # WEATHER
        st.session_state.rain_probability = random.randint(0, 100)
        st.session_state.rain_today = st.session_state.rain_probability > 70
        st.session_state.temperature = random.randint(22, 36)

        # SOIL DRYING & IRRIGATION
        for zone in ZONES:
            st.session_state.soil_moisture[zone] -= random.randint(4, 7)
            
            # Auto-irrigate if it gets too dry
            if st.session_state.soil_moisture[zone] < 30 and not st.session_state.rain_today:
                st.session_state.soil_moisture[zone] += 20
                
            st.session_state.soil_moisture[zone] = max(10, st.session_state.soil_moisture[zone])

        # RAIN EFFECT
        if st.session_state.rain_today:
            for zone in ZONES:
                st.session_state.soil_moisture[zone] += random.randint(10, 20)
                st.session_state.soil_moisture[zone] = min(100, st.session_state.soil_moisture[zone])

        # GROWTH / FRUIT SIZE
        for zone in ZONES:
            st.session_state.fruit_size[zone] = min(5.0, st.session_state.fruit_size[zone] + 0.2)

        # RODENT EVENT AFTER FRUIT STAGE
        if st.session_state.day > 15 and random.random() < 0.3:
            st.session_state.rodent_zone = random.choice(ZONES)
        else:
            st.session_state.rodent_zone = None

        # DRONE PATROL MOVEMENT
        st.session_state.drone_position = (st.session_state.drone_position + 1) % 10

        # YIELD CALCULATION
        avg_moisture = sum(st.session_state.soil_moisture.values()) / len(ZONES)
        st.session_state.yield_value = round((st.session_state.day * avg_moisture) / 12, 2)

        # EVENT LOGGING
        weather_str = "🌧️ Rained" if st.session_state.rain_today else "☀️ Sunny"
        log_msg = f"Day {st.session_state.day}: {weather_str} ({st.session_state.temperature}°C)."
        if st.session_state.rodent_zone:
            log_msg += f" 🚨 Rodent detected in {st.session_state.rodent_zone}."
        st.session_state.event_log.insert(0, log_msg)


# =====================================
# INTERACTIVE SIDEBAR
# =====================================

with st.sidebar:
    st.header("🕹️ Farm Controls")
    
    # Single day advance
    if st.button("➡️ Next Day", use_container_width=True, type="primary"):
        advance_simulation(1)
        
    st.markdown("---")
    
    # Multi-day fast forward - SLIDER UPDATED TO 45 DAYS
    st.subheader("⏩ Fast Forward")
    skip_days = st.slider("Select days to skip:", min_value=1, max_value=45, value=5)
    if st.button(f"Advance {skip_days} Days", use_container_width=True):
        advance_simulation(skip_days)

    st.markdown("---")
    
    # =====================================
    # MANUAL TRIGGERS
    # =====================================
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
        
    st.markdown("---")
    
    # Reset Simulation
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

# Dynamic Plant Height Calculation: Reaches 100cm at day 45
plant_height = min(100.0, 1.0 + (st.session_state.day * 2.2)) 

# Dynamic equipment heights to avoid clipping through tall plants
dynamic_drone_height = max(100.0, plant_height + 50.0)
dynamic_pole_height = max(50.0, plant_height + 20.0)

# =====================================
# TAB LAYOUT
# =====================================

tab1,tab2,tab3,tab4= st.tabs(["🛰️ 3D Farm Visual", "📊 Analytics Dashboard","🌍 Sustainability", "🚁AeroGreen Drone Command"])

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
        zone_angle = st.session_state.drone_position * (2 * math.pi / 10)
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
# TAB 2: ANALYTICS DASHBOARD
# -------------------------------------
with tab2:
    st.markdown("### 📈 Live Farm Metrics")
    
    # Top Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🌡️ Temperature", f"{st.session_state.temperature}°C")
    m2.metric("🌧️ Rain Probability", f"{st.session_state.rain_probability}%")
    m3.metric("🌱 Growth Stage", stage.capitalize())
    m4.metric("🌾 Plant Height", f"{round(plant_height, 2)} cm")
    m5.metric("📦 Est. Yield", f"{st.session_state.yield_value} kg")

    st.markdown("---")
    
    # Soil Moisture Bar Chart
    st.markdown("#### 💧 Soil Moisture by Zone")
    moisture_values = [st.session_state.soil_moisture[z] for z in ZONES]
    
    # Color code bars based on moisture health
    bar_colors = ['#ff4b4b' if m < 30 else '#1f77b4' if m > 85 else '#2ca02c' for m in moisture_values]

    bar_fig = go.Figure(data=[
        go.Bar(
            x=ZONES, 
            y=moisture_values,
            marker_color=bar_colors,
            text=[f"{m}%" for m in moisture_values],
            textposition='auto'
        )
    ])
    
    bar_fig.update_layout(
        yaxis=dict(range=[0, 100], title="Moisture Level (%)"),
        xaxis=dict(title="Farm Zones"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=300
    )
    
    # Add target lines for optimal moisture range
    bar_fig.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Minimum Threshold")
    bar_fig.add_hline(y=85, line_dash="dash", line_color="blue", annotation_text="Maximum Threshold")

    st.plotly_chart(bar_fig, use_container_width=True)
    
    st.markdown("---")

    # Data Table & Event Log Row
    col_table, col_log = st.columns([1.5, 1])
    
    with col_table:
        st.markdown("#### 📍 Zone Analysis Data")
        
        df = pd.DataFrame({
            "Zone": ZONES,
            "Soil Moisture (%)": [st.session_state.soil_moisture[z] for z in ZONES],
            "Fruit Size Index": [round(st.session_state.fruit_size[z], 2) for z in ZONES]
        })
        
        def get_status(moisture):
            if moisture < 30: return "🔴 Auto-Irrigating"
            if moisture > 85: return "🔵 High Moisture"
            return "🟢 Optimal"
            
        df["Status"] = df["Soil Moisture (%)"].apply(get_status)
        df["Threats"] = ["🚨 Rodent" if z == st.session_state.rodent_zone else "✅ Clear" for z in ZONES]

        st.dataframe(df, use_container_width=True, hide_index=True)

    with col_log:
        st.markdown("#### 📜 Farm Event Log")
        log_container = st.container(height=280)
        with log_container:
            for entry in st.session_state.event_log[:20]: # Show latest 20 events
                st.text(entry)

    # =====================================
    # SEASON SUMMARY SECTION
    # =====================================
    st.markdown("---")
    st.markdown("### 📋 Season Summary")
    
    # Calculate Season Progress
    target_day = 45
    progress = min(100, int((st.session_state.day / target_day) * 100))
    
    s1, s2, s3 = st.columns(3)
    
    with s1:
        st.write("**📅 Season Timeline**")
        st.progress(progress / 100)
        st.caption(f"{progress}% Complete ({st.session_state.day} / {target_day} Days)")
        
    with s2:
        st.write("**💪 Crop Health Outlook**")
        avg_m = sum(st.session_state.soil_moisture.values()) / len(ZONES)
        if avg_m < 35: health_status = "⚠️ Stress Detected"
        elif avg_m > 80: health_status = "💧 Saturated"
        else: health_status = "✅ Vigorous"
        st.info(f"Current Condition: {health_status}")
        
    with s3:
        st.write("**📊 Yield Prediction**")
        # Projects current yield average out to day 45
        final_est = round(st.session_state.yield_value * (target_day / max(1, st.session_state.day)), 2)
        st.success(f"Harvest Target: ~{final_est} kg")

        # Initialize Sustainability Metrics
init_state("total_water_used", 0.0)
init_state("total_water_harvested", 0.0)
init_state("pesticide_liters", 0.0)
init_state("carbon_offset", 0.0)


# --- Inside the for-loop in advance_simulation ---

# 1. Rain Harvesting: Capture water during rain events
if st.session_state.rain_today:
    harvested = random.uniform(8.0, 15.0) 
    st.session_state.total_water_harvested += harvested

# 2. Water Budgeting: Track irrigation volume
for zone in ZONES:
    if st.session_state.soil_moisture[zone] < 30 and not st.session_state.rain_today:
        # Each auto-irrigation event is estimated at 4 Liters
        st.session_state.total_water_used += 4.0

# 3. Pesticide Tracking: Every rodent event requires organic treatment
if st.session_state.rodent_zone:
    st.session_state.pesticide_liters += 0.15 

# 4. Carbon Sequestration: Biomass estimation based on plant height
# Calculation: (Height/100) * Number of Zones * Growth Factor
st.session_state.carbon_offset = (plant_height / 100) * len(ZONES) * 0.5


with tab3:
    st.markdown("### 🌍 Environmental Impact & Health Dashboard")
    
    # Corrected yield calculation line
    final_est = round(st.session_state.yield_value * (target_day/ max(1, st.session_state.day)), 2)
    
    # Top Row Metrics
    s1, s2, s3 = st.columns(3)
    
    # 1. Net Water Calculation
    net_water = st.session_state.total_water_used - st.session_state.total_water_harvested
    s1.metric("💧 Net Water Usage", f"{round(net_water, 1)} L", 
              delta=f"-{round(st.session_state.total_water_harvested, 1)} Harvested")
    
    # 2. Pesticide Load (Organic)
    pest_status = "Optimal" if st.session_state.pesticide_liters < (st.session_state.day * 0.1) else "High"
    s2.metric("🧪 Pesticide Applied", f"{round(st.session_state.pesticide_liters, 2)} L", 
              pest_status, delta_color="inverse")
    
    # 3. Carbon Sequestration
    s3.metric("🌳 CO2 Sequestered", f"{round(st.session_state.carbon_offset, 2)} kg", "Carbon Sink")

    st.markdown("---")

    # Double Gauge Row: Crop Health vs. Rain Harvesting
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("#### 🌱 Live Crop Health")
        # Initialize state for health if not present
        if "current_health" not in st.session_state:
            st.session_state.current_heal
