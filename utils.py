from langgraph.graph import StateGraph
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import TypedDict
from graphviz import Digraph
import PyPDF2
from langchain.schema import HumanMessage
import re,json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("Google API key is not set")
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.7, google_api_key=api_key)


# Define a state schema
# class AgentState(TypedDict):
#     input: str
#     output: str
    
class AgentState(TypedDict):
    input: str
    output: str
    research_output: str
    writer_output: str
    editor_output: str
    job_requirements: str
    approved: bool
    resume_text: str
    evaluation: str
    reason: str

# --------- Utility function to extract text from PDF ---------
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# --------- 1. Basic Single Agent ---------
def create_basic_agent_graph(user_input):
    prompt = PromptTemplate.from_template("Answer the following question: {question}")
    chain = LLMChain(llm=llm, prompt=prompt)

    def process_state(state: AgentState):
        return {"output": chain.run({"question": state["input"]})}

    graph = StateGraph(AgentState)
    graph.add_node("main", process_state)
    graph.set_entry_point("main")
    graph.set_finish_point("main")
    return graph.compile()

# --------- 2. Multi-Agent Graph ---------
def create_multi_agent_graph(user_input):
    # Create individual agent graphs
    research_graph = create_research_agent_graph(user_input)
    writer_graph = create_writer_agent_graph(user_input)
    editor_graph = create_editor_agent_graph(user_input)

    def start(state: AgentState):
        return {"input": state["input"]}

    def research_node(state: AgentState):
        research_output = research_graph.invoke({"input": state["input"]})
        return {"research_output": research_output["output"]}

    def writer_node(state: AgentState):
        writer_output = writer_graph.invoke({"input": state["research_output"]})
        return {"writer_output": writer_output["output"]}

    def editor_node(state: AgentState):
        editor_output = editor_graph.invoke({"input": state["writer_output"]})
        return {"editor_output": editor_output["output"]}

    def end(state: AgentState):
        return {
            "research_output": state["research_output"],
            "writer_output": state["writer_output"],
            "editor_output": state["editor_output"]
        }

    graph = StateGraph(AgentState)
    graph.add_node("start", start)
    graph.add_node("research", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)
    graph.add_node("end", end)

    graph.set_entry_point("start")
    graph.add_edge("start", "research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", "end")
    graph.set_finish_point("end")

    return graph.compile()


# --------- 3. Cyclic Graph (loop until condition met) ---------
def create_looping_graph(user_input):
    job_description = """
    We are looking for a Software Engineer with strong Python skills, experience in building scalable web applications, and familiarity with cloud services (AWS/GCP). 
    The candidate should have excellent problem-solving skills and the ability to work in a fast-paced agile environment.
    Preferred qualifications include knowledge of containerization (Docker, Kubernetes) and CI/CD pipelines.
    """

    def start_node(state: AgentState):
        return {
            **state,
            "job_requirements": job_description,
            "approved": None,
            "resume_text": state["input"],
            "evaluation": "",
            "reason": ""
        }

    def review_resume(state: AgentState):
        prompt_text = f"""
Job Requirements:
{state['job_requirements']}

Resume:
{state['resume_text']}

Does the resume meet the job requirements?

Respond ONLY in this exact JSON format, enclosed within triple single quotes ''' like this:

```json
{{
  "approved": true,
  "reason": "Provide a clear explanation for the decision."
}}
```

No other text outside the triple quotes.
"""

        evaluation = llm([HumanMessage(content=prompt_text)])
        with open("response.txt", "w", encoding="utf-8") as f:
            f.write(evaluation.content if hasattr(evaluation, "content") else str(evaluation))
        # Extract the JSON between triple single quotes
        match = re.search(r"```json(.*?)```", evaluation.content if hasattr(evaluation, "content") else evaluation, re.DOTALL)
        if not match:
            raise ValueError("Could not find JSON response enclosed in ```json ```")

        json_str = match.group(1).strip()

        # Parse JSON
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nJSON content was:\n{json_str}")

        approved = result.get("approved", False)
        reason = result.get("reason", "")

        return {
            **state,
            "job_requirements": state["job_requirements"],
            "approved": approved,
            "resume_text": state["resume_text"],
            "evaluation": evaluation.content if hasattr(evaluation, "content") else evaluation,
            "reason": reason,
        }

    def condition(state: AgentState):
        if state.get("approved") is not None:
            return "end"
        return "review"

    def end_node(state: AgentState):
        return state  # You could also format final output here

    graph = StateGraph(AgentState)
    graph.add_node("start", start_node)
    graph.add_node("review", review_resume)
    graph.add_node("end", end_node)

    graph.add_edge("start", "review")
    graph.add_conditional_edges("review", condition, {"review": "review", "end": "end"})

    graph.set_entry_point("start")
    graph.set_finish_point("end")

    return graph.compile()


# --------- 4. Researcher Agent ---------
def create_research_agent_graph(user_input):
    prompt = PromptTemplate.from_template("Research the latest trends in {topic}. Focus on identifying pros and cons and the overall narrative.")
    chain = LLMChain(llm=llm, prompt=prompt)

    def process_state(state: AgentState):
        return {"output": chain.run({"topic": state["input"]})}

    graph = StateGraph(AgentState)
    graph.add_node("research", process_state)
    graph.set_entry_point("research")
    graph.set_finish_point("research")
    return graph.compile()

# --------- 5. Writer Agent ---------
def create_writer_agent_graph(user_input):
    prompt = PromptTemplate.from_template("Write an engaging article on {topic}. Focus on the latest trends and how it's impacting the industry.")
    chain = LLMChain(llm=llm, prompt=prompt)

    def process_state(state: AgentState):
        return {"output": chain.run({"topic": state["input"]})}

    graph = StateGraph(AgentState)
    graph.add_node("write", process_state)
    graph.set_entry_point("write")
    graph.set_finish_point("write")
    return graph.compile()

# --------- 6. Editor Agent ---------
def create_editor_agent_graph(user_input):
    prompt = PromptTemplate.from_template("Edit the article on {topic} for grammatical errors and ensure it's ready for publication.")
    chain = LLMChain(llm=llm, prompt=prompt)

    def process_state(state: AgentState):
        return {"output": chain.run({"topic": state["input"]})}

    graph = StateGraph(AgentState)
    graph.add_node("edit", process_state)
    graph.set_entry_point("edit")
    graph.set_finish_point("edit")
    return graph.compile()
