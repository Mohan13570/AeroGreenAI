# 🌱 AeroGreen AI  
## Intelligent Autonomous Smart Farming Simulation Platform

---

## 🚀 Overview

**AeroGreen AI** is an AI-powered smart agriculture simulation platform designed to optimize irrigation, monitor crop growth, detect pest threats, and enable autonomous drone response for sustainable farming.

This project demonstrates a scalable 10-acre intelligent tomato farm ecosystem that integrates:

- 🌤 Weather-aware irrigation  
- 💧 Dynamic soil moisture modeling  
- 🍅 Real-time tomato growth lifecycle simulation  
- 🐀 Rodent detection & pesticide response  
- 🚁 Autonomous drone surveillance  
- 📷 Distributed multi-camera monitoring  
- 📊 Web-based farmer dashboard  

The system is built entirely in Python and runs in a browser, making it deployable, mobile-compatible, and hardware-agnostic.

---

## 🌍 Problem Statement

Modern agriculture faces critical challenges:

- Over-irrigation leading to water wastage  
- Unpredictable climate conditions  
- Delayed pest detection  
- Crop losses due to rodents  
- Manual field monitoring inefficiencies  
- Excessive pesticide usage  

Small and medium farmers often lack affordable intelligent automation systems.

**AeroGreen AI aims to simulate a scalable, AI-driven, sustainable farm management ecosystem.**

---

## 🧠 Core System Capabilities

### 1️⃣ Climate-Aware Irrigation Engine

- Soil moisture decreases naturally over time.
- Daily rain probability is simulated.
- If rain is expected → irrigation is postponed.
- If moisture drops below threshold → irrigation is triggered.
- Prevents unnecessary water consumption.

#### Decision Logic

```python
if soil_moisture < 30 and not rain_today:
    irrigation = "Start Irrigation"

elif rain_probability > 70:
    irrigation = "Delay Irrigation"
```

---

### 2️⃣ Realistic Tomato Growth Lifecycle Simulation

The system models biological crop growth over time.

#### Growth Stages

| Stage | Days | Visual Change |
|-------|------|--------------|
| Seedling | 0–4 | Small green stem |
| Flowering | 5–9 | Yellow flowers |
| Green Fruit | 10–19 | Small green tomatoes |
| Ripening | 20–29 | Orange tomatoes |
| Ripe | 30+ | Fully red tomatoes |

🕒 1 farm day = 10 real-world seconds.

Plant height increases dynamically as days progress.

---

### 3️⃣ Rodent Detection & Autonomous Response

Rodent events occur only during fruit-bearing stages.

When detected:

- 🚨 Zone flagged  
- 🚁 Drone dispatched automatically  
- 🧪 Pesticide protocol simulated  
- 📊 Farmer alerted in dashboard  

---

### 4️⃣ Autonomous Drone Surveillance

The drone:

- Patrols farm continuously  
- Moves in circular route during normal conditions  
- Redirects to affected zone when rodent detected  
- Simulates targeted intervention  

---

### 5️⃣ Distributed Camera Monitoring

5 surveillance poles are positioned:

- 4 quadrant cameras (each covers ~2.5 acres)  
- 1 central monitoring camera  

The layout ensures complete 10-acre visibility.

---

## 🏗 System Architecture

```
10 Acre Farm (10 Zones)
        ↓
Soil Moisture Engine
        ↓
Weather Simulation Engine
        ↓
AI Decision Layer
        ↓
Irrigation / Rodent Logic
        ↓
Drone Response Engine
        ↓
Farmer Dashboard (Web-Based)
```

---

## 🧮 Farm Layout

- Total Area: 10 Acres  
- Divided into 10 Zones (A1–B5)  
- 5 Camera Poles  
- Central Monitoring  
- Autonomous Drone Layer  

---

## ⚙️ Technical Stack

| Component | Technology |
|------------|------------|
| Frontend | Streamlit |
| 3D Visualization | Plotly 3D |
| Simulation Engine | Python |
| State Management | Streamlit Session State |
| Mathematical Modeling | NumPy |
| Deployment | Streamlit Cloud / Render |

---

## 🔄 Simulation Engine Design

Each farm day (10 seconds):

1️⃣ Soil moisture reduces  
2️⃣ Weather probability generated  
3️⃣ Rain impact applied  
4️⃣ Irrigation decision executed  
5️⃣ Rodent event probability checked  
6️⃣ Drone target assigned  
7️⃣ Plant growth updated  

This mimics real agricultural cycles.

---

## 🌱 Sustainability Impact

AeroGreen AI contributes to:

- 💧 Water conservation  
- ⚡ Energy efficiency  
- 🧪 Reduced pesticide misuse  
- 📉 Lower crop losses  
- 📈 Yield optimization  
- 🌍 Climate-resilient agriculture  

---

## 📊 Farmer Dashboard

Displays:

- Current farm day  
- Rain probability  
- Rain event status  
- Soil moisture per zone  
- Irrigation decision per zone  
- Rodent alerts  
- Drone dispatch alerts  

Fully browser-based and mobile compatible.

---

## 📈 Scalability

Although demonstrated on 10 acres, the architecture is modular.

To scale:

- Add more zones  
- Add additional camera nodes  
- Deploy multiple drones  
- Integrate real IoT sensors  
- Connect to live weather APIs  
- Deploy cloud backend infrastructure  

System supports horizontal expansion.

---

## 🔮 Future Enhancements

- Real IoT sensor integration  
- Real weather API connection  
- Computer vision disease detection  
- Reinforcement learning irrigation optimization  
- Predictive yield analytics  
- Edge AI deployment  
- Multi-crop support  

---

## 🖥 How to Run

### 1️⃣ Install dependencies

```bash
pip install streamlit plotly numpy
```

### 2️⃣ Run the app

```bash
streamlit run app.py
```

### 3️⃣ Open in browser

Works on:
- Desktop  
- Mobile browser  
- Cloud deployment  

---

## 🎯 Project Vision

AeroGreen AI envisions a future where:

- Farms operate autonomously  
- Irrigation adapts instantly to climate  
- Pest threats are mitigated automatically  
- Farmers monitor entire fields from their phone  
- Sustainability and profitability go hand-in-hand  

---

## 🏁 Conclusion

AeroGreen AI is a proof-of-concept intelligent smart agriculture ecosystem integrating climate awareness, automation, AI decision systems, and autonomous response mechanisms into a unified sustainable farming solution.

It demonstrates how digital intelligence can transform traditional agriculture into a precision-driven, environmentally responsible system.

---

## 📄 License

This project is developed for educational and hackathon purposes.
