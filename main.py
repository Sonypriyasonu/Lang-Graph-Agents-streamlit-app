import os
import streamlit as st
from langgraph.graph import StateGraph
import graphviz
from utils import (
    create_basic_agent_graph,
    create_multi_agent_graph,
    create_looping_graph,
    extract_text_from_pdf
)


# Mapping use cases to their graph builders
graph_builders = {
    "Single Agent": create_basic_agent_graph,
    "Multi-Agent": create_multi_agent_graph,
    "Cyclic Graph": create_looping_graph,
}

JOB_DESCRIPTION = """
We are looking for a Software Engineer with strong Python skills, experience in building scalable web applications, and familiarity with cloud services (AWS/GCP).  
The candidate should have excellent problem-solving skills and the ability to work in a fast-paced agile environment.  
Preferred qualifications include knowledge of containerization (Docker, Kubernetes) and CI/CD pipelines.
"""

st.set_page_config(page_title="LangGraph Agent Explorer", layout="wide")

use_case = st.sidebar.selectbox("Choose a LangGraph use case", list(graph_builders.keys()))

if use_case == "Cyclic Graph":
    st.title("📄 Resume Qualification Checker")
else:
    st.title("🧠 LangGraph Agent Explorer")
    
if use_case == "Cyclic Graph":
    
    st.subheader("Job Description")
    st.markdown(JOB_DESCRIPTION, unsafe_allow_html=True)
    
    uploaded_pdf = st.file_uploader("Upload Resume PDF", type=["pdf"])
    if uploaded_pdf:
        resume_text = extract_text_from_pdf(uploaded_pdf)
        st.subheader("Extracted Resume Text (preview)")
        st.write(resume_text[:1000])

        if st.button("Check Resume Against Job Requirements"):
            st.info("Building graph and invoking agents...")
            graph = graph_builders[use_case](resume_text)
            result = graph.invoke({"input": resume_text})

            if result.get("approved", False):
                st.success("🎉 Resume PASSES the job requirements.")
            else:
                st.error("⚠️ Resume DOES NOT meet the job requirements.")

            st.subheader("Reason")
            st.write(result.get("reason", "No reason available"))
                
            with st.sidebar:
                    st.subheader("📊 Graph Flow")
                    dot = graphviz.Digraph()

                    # Define nodes
                    dot.node("start", "Start")
                    dot.node("review", "🔄 Review Resume")
                    dot.node("end", "End")

                    # Define edges
                    dot.edge("start", "review")
                    dot.edge("review", "review", label="Needs re-review (approved = None)")  # cycle edge
                    dot.edge("review", "end", label="Approved or Rejected (approved != None)")

                    st.graphviz_chart(dot)
    else:
        st.warning("Please upload a resume PDF.")
else:
    user_input = st.text_area("Enter your input (prompt, query, topic):", height=100)

    if st.button("Run LangGraph Agent"):
        if not user_input:
            st.warning("Please enter a prompt or input text.")
        else:
            st.info("Building graph and invoking agents...")
            graph = graph_builders[use_case](user_input)
            result = graph.invoke({"input": user_input})

            st.success("Agent Execution Completed ✅")
            
            if use_case == "Single Agent":
                st.subheader("🔍 Final Output")
                st.write(result.get("output", "No output available"))
                
                with st.sidebar:
                    st.subheader("📊 Graph Flow")

                    dot = graphviz.Digraph()

                    # Define nodes
                    dot.node("start", "Start")
                    dot.node("agent", "🤖 Agent")
                    dot.node("end", "End")

                    # Define edges
                    dot.edge("start", "agent")
                    dot.edge("agent", "end")

                    st.graphviz_chart(dot)
            elif use_case == "Multi-Agent":
                tab1, tab2, tab3 = st.tabs(["🔍 Researcher", "✍️ Writer", "📝 Editor"])

                with tab1:
                    st.subheader("Researcher Agent Output")
                    st.write(result.get("research_output", "No output available"))

                with tab2:
                    st.subheader("Writer Agent Output")
                    st.write(result.get("writer_output", "No output available"))

                with tab3:
                    st.subheader("Editor Agent Output")
                    st.write(result.get("editor_output", "No output available"))
                
                with st.sidebar:
                    st.subheader("📊 Graph Flow")

                    dot = graphviz.Digraph()

                    # Define nodes
                    dot.node("start", "Start")
                    dot.node("researcher", "🔍 Researcher Agent")
                    dot.node("writer", "✍️ Writer Agent")
                    dot.node("editor", "📝 Editor Agent")
                    dot.node("end", "End")

                    # Define edges
                    dot.edge("start", "researcher")
                    dot.edge("researcher", "writer")
                    dot.edge("writer", "editor")
                    dot.edge("editor", "end")

                    # Display the graph
                    st.graphviz_chart(dot)

            else:
                st.subheader("🔍 Final Output")
                st.write(result.get("output", "No output available"))
