

https://github.com/user-attachments/assets/03c1e0f3-34e0-486b-81a2-fc8b50d2a0d4

# AI-InsightDB // Autonomous Business Intelligence Engine 🤖

Raw numerical data spreadsheets tell a vital story, but business leaders often lack the time to parse rows of statistics or write complex forecasting models from scratch. 

I built **AI-InsightDB** to act as an automated, completely air-gapped data analyst. This full-stack web dashboard takes in multi-column operational datasets, cleans the tracking vectors, executes underlying machine learning regression algorithms to calculate future trends, and feeds the numerical outputs straight to a local, offline LLM to compile a natural-language executive intelligence brief.

---

##  How the Pipeline Executes

The data flow runs entirely on local machine hardware without utilizing external APIs or leaking sensitive company data tables to cloud trackers:

1. **Ingestion & Profiling Layer:** Loads raw dataframes via Pandas and runs quick descriptive statistical parameters (mean, boundary limits, missing elements).
2. **Predictive Modeling Layer:** Feeds data columns to a Scikit-Learn `LinearRegression` model to extract overall growth trajectory slopes and forecast future milestone values.
3. **Local AI Interpretation Layer:** Compresses the underlying statistical matrices into a concise text payload, passing it natively to an open-source `qwen2.5:0.5b` model running on an offline Ollama background service. The model translates mathematical trends into actionable business recommendations.

---

## The Data Stack

* **Data Wrangling & ML:** Pandas, NumPy, Scikit-Learn
* **Dynamic Visualization:** Plotly Express (Interactive charting)
* **Application Layout Interface:** Streamlit (Custom Dark CSS styling modules)
* **Local Processing Engine:** Ollama Gateway

---

##  Step-by-Step Local Deployment

Follow these terminal setup instructions to run the analytical engine on your device:

```bash
# 1. Clone the repository workspace and step into it
git clone [https://github.com/prasannasavalla/ai-insightdb.git](https://github.com/prasannasavalla/ai-insightdb.git)
cd ai-insightdb

# 2. Re-initialize your isolated project environment sandbox
python -m venv analyst-env

# 3. Activate the environment
# On Windows Command Prompt:
analyst-env\Scripts\activate.bat
# On macOS / Linux: source analyst-env/bin/activate

# 4. Pull down the analytics package requirements
pip install -r requirements.txt

# 5. Ensure Ollama is active on your computer, and fetch the tiny model
ollama run qwen2.5:0.5b

# 6. Launch your interactive dashboard interface!
streamlit run app.py
