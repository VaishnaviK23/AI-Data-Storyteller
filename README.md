# 📊 AI Data Storyteller

**AI Data Storyteller** is an interactive Streamlit application that uses GPT-4 and Plotly to generate smart visualizations and data insights from user-uploaded datasets. It leverages the LangGraph framework to manage a structured, modular data analysis pipeline.

---

## 🚀 Features

- Upload CSV or Excel datasets
- Automatic data type analysis
- Insightful suggestions using GPT-4
- Plotly-powered visualizations (Line, Bar, Scatter, Box)
- LLM agent retrieves stats for selected columns
- Natural language summary of dataset
- Downloadable HTML report
- Built with LangGraph for extensibility and clarity

---

## 🧰 Tech Stack

- **Frontend**: Streamlit  
- **LLM**: OpenAI GPT-4 via LangChain  
- **Workflow Engine**: LangGraph  
- **Visualization**: Plotly Express  
- **Templating**: Jinja2  
- **Environment**: python-dotenv  

---

## 📦 Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/VaishnaviK23/AI-Data-Storyteller.git
    cd AI-Data-Storyteller
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set your OpenAI API key:**

    Create a `.env` file in the root directory:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📂 How to Use

1. Upload a `.csv` or `.xlsx` file.
2. Select the X and Y columns for visualization.
3. Choose a chart type (`Line`, `Bar`, `Scatter`, `Box`).
4. Click **Generate Visualization**.
5. View:
   - 📈 Interactive Plotly chart  
   - 🧠 GPT-generated data summary  
   - 🔍 Agent insights on selected column  
6. Download the final report as an HTML file.

---

## 🔄 LangGraph Workflow

The app’s logic is orchestrated using **LangGraph** with the following node-based flow:

1. `analyze_types` → Detect data types  
2. `clarify` → Ask GPT-4 what analysis might be useful  
3. Conditionally route to:  
   - `generate_chart` → If user selected chart type  
   - `summarize` → If no chart type selected  
4. `agent_node` → Use LangChain Agent to analyze column  
5. `summarize` → Final dataset overview  
6. `generate_report` → Render an HTML report for download  

This structure allows **flexibility**, **scalability**, and **transparency** for future extension.

---

## 📝 Example Output

- 📈 Interactive chart rendered using Plotly  
- 🧠 GPT-4 generated natural language summary  
- 🔍 Agent-based insight on column statistics  
- 📄 HTML report with summary, insights, and preview
