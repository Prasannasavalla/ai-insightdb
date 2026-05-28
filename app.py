import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.linear_model import LinearRegression

# 1. Setup a premium widescreen analytics page layout
st.set_page_config(page_title="AI-InsightDB Corporate Suite", page_icon="📈", layout="wide")

# 2. Inject Enterprise-Grade Light Theme CSS
st.markdown("""
    <style>
    /* Main background background canvas */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
    }
    /* Main Page Header Typography */
    .title-text {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.1rem;
    }
    /* Subtitle text styling */
    .sub-caption {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    /* Polished white metric cards with subtle drop shadows */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px 0 rgba(0, 0, 0, 0.03);
    }
    /* Value numbering typography inside metric cards */
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
    /* Label subtext inside metric cards */
    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-top: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- MAIN SCREEN HEADER ---
st.markdown("<div class='title-text'>AI-InsightDB // Business Intelligence Engine 📈</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-caption'>Automated descriptive analytics and machine learning trend forecasting wrapped in a local AI executive briefing layer.</div>", unsafe_allow_html=True)
st.divider()

OLLAMA_URL = "http://localhost:11434/api/generate"

# --- SIDEBAR UTILITY GATEWAY ---
with st.sidebar:
    st.markdown("<h3 style='color:#0f172a; margin-top:0;'>📊 Data Control Pipeline</h3>", unsafe_allow_html=True)
    st.write("Upload operations, financial transactions, or sensor tables to evaluate.")
    
    use_sample = st.checkbox("Load Synthetic Operational Data Sample", value=True)
    st.divider()
    st.markdown("**Infrastructure Verification:**")
    st.success("⚡ Model: Scikit-Learn Engine")
    st.success("🧠 Insights: Ollama Local Gateway")
    st.caption("100% Secure Air-Gapped Compliance Mode")

# --- STEP 1: DATA INGESTION MATRIX ---
if use_sample:
    np.random.seed(42)
    days = np.array(range(1, 31))
    operational_yield = 150 + (days * 2.5) + np.random.normal(0, 5, 30)
    defect_rate = np.clip(12 - (days * 0.3) + np.random.normal(0, 1, 30), 1, 15)
    
    df = pd.DataFrame({
        "Day_Sequence": days,
        "Production_Yield_Units": np.round(operational_yield, 1),
        "Defect_Rate_Percentage": np.round(defect_rate, 2)
    })
else:
    uploaded_file = st.sidebar.file_uploader("Upload Target CSV Matrix", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("System awaiting input: Upload a corporate CSV file or activate the synthetic sample generator in the control sidebar.")
        st.stop()

# --- DISPLAY RAW DATA TABLE SUMMARY ---
with st.container(border=True):
    st.markdown("<h4 style='color:#1e293b; margin-top:0;'>📋 Source Ledger Verification</h4>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=180)

st.write("")

# --- DYNAMIC METRIC CARDS ROW ---
# Using our custom high-contrast white cards instead of raw text strings
m_col1, m_col2, m_col3 = st.columns(3)

# Calculate stats and Linear Regression ahead of cards
X = df[[df.columns[0]]].values
y = df[df.columns[1]].values
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]

next_step = int(df.columns[0] + 1 if isinstance(df.columns[0], (int, float)) else len(df) + 1)
prediction = model.predict([[next_step]])[0]

with m_col1:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{df[df.columns[1]].mean():.1f}</div><div class='metric-label'>SAMPLE RUN MEAN</div></div>", unsafe_allow_html=True)
with m_col2:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>+{slope:.2f}</div><div class='metric-label'>CALCULATED STEP TRAJECTORY</div></div>", unsafe_allow_html=True)
with m_col3:
    st.markdown(f"<div class='metric-card'><div class='metric-val'>{prediction:.1f}</div><div class='metric-label'>NEXT FORECASTED BOUNDARY TARGET</div></div>", unsafe_allow_html=True)

st.write("")

# --- SPLIT LAYOUT FOR CHARTING AND TABLES ---
col_viz, col_stats = st.columns([2, 1], gap="medium")

with col_viz:
    with st.container(border=True):
        fig = px.line(df, x=df.columns[0], y=df.columns[1], title=f"Primary Operational Trend Over Time ({df.columns[1]})", markers=True)
        # Using the clean default light layout style template for Plotly graphs
        fig.update_layout(template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

with col_stats:
    with st.container(border=True):
        st.markdown("<p style='font-weight:600; margin-top:0; color:#1e293b;'>📊 Descriptive Telemetry Summary</p>", unsafe_allow_html=True)
        stats_df = df.describe().T[['mean', 'min', 'max']]
        st.dataframe(stats_df, use_container_width=True)

# --- STEP 3: CONVERT NUMBERS TO AI GENERATED BUSINESS INSIGHTS ---
st.write("")
st.markdown("### 🔬 Executive Analytics Intelligence Summary")

data_summary_text = f"""
Dataset Overview:
Columns analyzed: {list(df.columns)}
Mean values: {dict(df.mean())}
Calculated growth trajectory slope: {slope:.4f} units per cycle.
Next predicted future data milestone target value: {prediction:.2f}
"""

trigger_ai = st.button("Compile Executive Intelligence Briefing 🚀", type="primary", use_container_width=True)

if trigger_ai:
    with st.spinner("Local Qwen LLM analyzing numerical data parameters... Compiling briefing matrix."):
        
        system_prompt = f"""
        You are an expert Senior Business Data Analyst. Review this data summary and provide an executive intelligence brief.
        Keep it sharp, professional, and clear. Avoid fluff.
        
        Data to evaluate:
        {data_summary_text}
        
        Structure your analysis strictly using this layout:
        ### 🔍 Structural Trend Analysis:
        (Analyze the growth slope and current mean numbers)
        
        ### 🎯 Forecast & Business Impact:
        (Explain what the next predicted milestone target value means for resource planning)
        
        ### 💡 Strategic Operational Advice:
        (Provide actionable business advice based on these metrics)
        """
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "qwen2.5:0.5b",
                    "prompt": system_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                ai_insights = response.json().get("response", "No brief compiled.")
                st.balloons()
                with st.container(border=True):
                    st.markdown(ai_insights)
            else:
                st.error("Local inference model connection error: Verify background service initialization.")
                
        except requests.exceptions.ConnectionError:
            st.error("📡 Connection Matrix Severed: Ensure Ollama is actively running in the background (`ollama run qwen2.5:0.5b`).")
else:
    with st.container(border=True):
        st.info("System primed. Click 'Compile Executive Intelligence Briefing' above to route mathematical matrices into the localized NLP engine layer.")