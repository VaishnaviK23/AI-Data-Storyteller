import streamlit as st
import pandas as pd
import plotly.express as px
import openai
import os
import io
import chardet
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from jinja2 import Template
from dotenv import load_dotenv
from typing import TypedDict, List

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

st.set_page_config(page_title="AI Data Storyteller", layout="wide")
st.title("📊 AI Data Storyteller")

# --- LangGraph States and Nodes ---
class GraphState(TypedDict):
    df: List[dict]
    x_col: str
    y_col: str
    chart_type: str
    data_types: dict
    clarification: str
    summary: str
    fig: object
    agent_result: str
    report_html: str

def data_type_analysis_node(state: GraphState):
    df = pd.DataFrame(state["df"])
    types = {
        "Numerical Columns": df.select_dtypes(include="number").columns.tolist(),
        "Categorical Columns": df.select_dtypes(include="object").columns.tolist(),
        "Datetime Columns": df.select_dtypes(include="datetime").columns.tolist()
    }
    return {"data_types": types, "df": state["df"]}

def ask_clarifying_question_node(state: GraphState):
    df = pd.DataFrame(state["df"])
    columns = df.columns.tolist()
    prompt = f"User uploaded a dataset with columns: {columns}. What type of analysis would be insightful?"
    response = llm.invoke(prompt)
    return {"clarification": response.content, "df": state["df"]}

def generate_chart_node(state: GraphState):
    df = pd.DataFrame(state["df"])
    x_col = state.get("x_col")
    y_col = state.get("y_col")
    chart_type = state.get("chart_type")
    fig = None

    if x_col in df.columns and y_col in df.columns:
        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col)
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_col, y=y_col)
        elif chart_type == "Box":
            fig = px.box(df, x=x_col, y=y_col)

    return {"fig": fig, "df": state["df"]}

def agent_node(state: GraphState):
    def get_column_stats(column: str) -> str:
        df = pd.DataFrame(state["df"])
        return df[column].describe().to_string()

    tools = [
        Tool(
            name="get_column_stats",
            func=get_column_stats,
            description="Get statistics for a given column from the dataframe. Input should be the column name."
        )
    ]

    executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )

    column = state.get("x_col")
    if not column:
        df = pd.DataFrame(state["df"])
        column = df.select_dtypes(include="number").columns[0]

    # Provide the input as a plain string prompt
    prompt = f"Please get statistics for the column '{column}' using the tool get_column_stats."
    result = executor.run(prompt)

    return {"agent_result": result, "df": state["df"]}

def summary_node(state: GraphState):
    df = pd.DataFrame(state["df"])
    
    # Identify column types
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    summary_parts = []

    summary_parts.append(f"The dataset contains {len(df)} rows and {len(df.columns)} columns.")

    # Summarize categorical columns
    for col in categorical_cols:
        top_val = df[col].value_counts().idxmax()
        unique_vals = df[col].nunique()
        summary_parts.append(f"Column '{col}' has {unique_vals} unique values. Most frequent: '{top_val}'.")

    # Summarize numerical columns
    for col in numerical_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        col_mean = df[col].mean()
        summary_parts.append(f"Column '{col}' - min: {col_min:.2f}, max: {col_max:.2f}, average: {col_mean:.2f}")

    # Summarize datetime columns
    for col in datetime_cols:
        min_date = df[col].min()
        max_date = df[col].max()
        summary_parts.append(f"Column '{col}' ranges from {min_date} to {max_date}.")

    # Construct final prompt
    analysis_summary = "\n".join(summary_parts)
    prompt = f"Generate a concise and readable paragraph summarizing the dataset based on this profile:\n\n{analysis_summary}"
    summary = llm.invoke(prompt)

    return {"summary": summary.content, "df": state["df"]}

def report_node(state: GraphState):
    df = pd.DataFrame(state["df"])
    summary = state.get("summary") or "No summary available."
    agent_result = state.get("agent_result") or ""
    html_template = Template("""
    <html>
    <head><title>AI Data Report</title></head>
    <body>
        <h1>Dataset Summary</h1>
        <pre>{{ summary }}</pre>
        <h2>Agent Insight</h2>
        <pre>{{ agent_result }}</pre>
        <h2>Data Preview</h2>
        {{ table | safe }}
    </body>
    </html>
    """)
    report_html = html_template.render(summary=summary, agent_result=agent_result, table=df.head().to_html())
    return {"report_html": report_html, "df": state["df"]}

# --- LangGraph Construction ---
builder = StateGraph(GraphState)
builder.add_node("analyze_types", RunnableLambda(data_type_analysis_node))
builder.add_node("clarify", RunnableLambda(ask_clarifying_question_node))
builder.add_node("generate_chart", RunnableLambda(generate_chart_node))
builder.add_node("agent_node", RunnableLambda(agent_node))
builder.add_node("summarize", RunnableLambda(summary_node))
builder.add_node("generate_report", RunnableLambda(report_node))

builder.set_entry_point("analyze_types")
builder.add_edge("analyze_types", "clarify")

# Conditional edge from clarify based on chart_type
builder.add_conditional_edges(
    "clarify",
    lambda state: "generate_chart" if state.get("chart_type") else "summarize",
    {
        "generate_chart": "generate_chart",
        "summarize": "summarize"
    }
)

builder.add_edge("generate_chart", "agent_node")
builder.add_edge("agent_node", "summarize")
builder.add_edge("summarize", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()

# --- Streamlit Interface ---
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    x_col, y_col, chart_type = None, None, None

    st.write("Select visualization preferences (used in later node):")
    with st.form(key='form1'):
        if uploaded_file.name.endswith(".csv"):
            try:
                raw_data = uploaded_file.getvalue()
                detected_encoding = chardet.detect(raw_data)['encoding']
                temp_df = pd.read_csv(io.StringIO(raw_data.decode(detected_encoding)))
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
        elif uploaded_file.name.endswith(".xlsx"):
            try:
                temp_df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
        else:
            st.error("Unsupported file type. Please upload a .csv or .xlsx file.")

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", temp_df.columns)
        with col2:
            y_col = st.selectbox("Y-axis", temp_df.select_dtypes(include="number").columns)

        chart_type = st.selectbox("Chart Type", ["", "Line", "Bar", "Scatter", "Box"])
        submit_button = st.form_submit_button(label="Generate Visualization")

    if submit_button:
        inputs = {
            "df": temp_df,
            "x_col": x_col,
            "y_col": y_col,
            "chart_type": chart_type
        }

        final_state = graph.invoke(inputs)

        if "fig" in final_state:
            st.subheader("📈 Visualization")
            st.plotly_chart(final_state["fig"])

        if final_state.get("clarification"):
            st.subheader("💡 GPT-4 Suggestions")
            st.markdown(final_state["clarification"])

        st.subheader("📝 Summary")
        st.markdown(final_state["summary"])

        if final_state.get("agent_result"):
            st.subheader("🧠 Agent Insights")
            st.markdown(final_state["agent_result"])
        

        st.subheader("📅 Download Report")
        st.download_button("Download HTML Report", final_state["report_html"], file_name="data_report.html")
