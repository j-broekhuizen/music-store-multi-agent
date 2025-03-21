from langchain_openai import ChatOpenAI
import operator
from langgraph.checkpoint.memory import MemorySaver
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, List, Tuple, Union
from pprint import pprint

load_dotenv()

checkpointer = MemorySaver()

code_research_deployment_url = "https://reusable-researcher-a9f30a6efd655a6b9aa752ff1463c9f1.us.langgraph.app"
code_write_deployment_url = "https://react-write-code-da2ca8fb2e095891ba51ba0f66542b4a.us.langgraph.app"
code_test_deployment_url = "https://react-test-code-674354e49ca850e98faf729296112548.us.langgraph.app"

code_write_remote_graph = RemoteGraph("agent", url=code_write_deployment_url)
code_test_remote_graph = RemoteGraph("agent", url=code_test_deployment_url)
research_assistant_remote_graph = RemoteGraph("agent", url=code_research_deployment_url)

model = ChatOpenAI(model="o3-mini")

class Config(BaseModel):
    context_for_researcher: list[str]

class State(TypedDict):
    original_objective: str
    action_plan: List[str]
    messages: Annotated[list[AnyMessage], add_messages]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Response(BaseModel):
    """Response to user."""
    response: str

class Step(BaseModel):
    description: str = Field(description="Description of the step to be performed")
    subagent: Literal["code_research_subagent", "code_writing_subagent", "code_testing_subagent"] = Field(
        description="Name of the subagent that should execute this step"
    )
    
class Plan(BaseModel):
    steps: List[Step] = Field(
        description="Different steps that the subagents should follow to answer the customer's request, in chronological order"
    )

class ReplannerResponse(BaseModel):
    action: Union[Response, Plan] = Field(
        description="The action to perform. If you no longer need to use the subagents to solve the customer's request, use Response. "
        "If you still need to use the subagents to solve the problem, construct and return a list of steps to be performed by the subagents, use Plan."
    )

supervisor_prompt = """
You are a supervisor/planner for a multi-agent team that handles queries from customers. The multi-agent team you are supervising is responsible for handling questions related to developer productivity. Your team is composed of three subagents that you can use to help answer the customer's request:

1. code_research_subagent: this subagent searches the internet for information related to programming languages, libraries, and frameworks. this subagent should be used when the customer wants to learn more about the functionality of a specific programming language, library, or framework, or is just generally
curious about some specific topic.
2. code_writing_subagent: this subagent writes code based on the user's request.
3. code_testing_subagent: this subagent tests the user's provided code, writes tests, and suggests improvements.

Your role is to create an action plan that the subagents can follow and execute to thoroughly answer the customer's request. Your action plan should specify the exact steps that should be followed by the subagent(s) in order to successfully answer the customer's request, and which subagent should perform each step. Return the action plan as a list of objects, where each object contains the following fields:
- step: a detailed step that should be followed by the subagent(s) in order to successfully answer a portion of the customer's request.
- subagent_to_perform_step: the subagent that should perform the step.

Return the action plan in chronological order, starting with the first step to be performed, and ending with the last step to be performed. You should try your best not to have multiple steps performed by the same subagent. This is inefficient. If you need a subagent to perform a task, try and group it all into one step.

If you do not need a subagent to answer the customer's request, do not include it in the action plan/list of steps. Your goal is to be as efficient as possible, while ensuring that the customer's request is answered thoroughly and to the best of your ability.

Be sure the action plan is thorough and that it covers all aspects of the customer's request. Take a deep breath and think carefully before responding. Go forth and make the customer delighted!
"""

# helpers 
def format_steps(steps_list):
    """
    Convert a list of Step objects into a neatly formatted string.
    
    Args:
        steps_list: List of Step objects
        
    Returns:
        A formatted string containing all key information
    """
    formatted_output = "# STEPS ALREADY PERFORMED BY THE SUBAGENTS\n\n"
    
    for i, step in enumerate(steps_list, 1):
        print(step, "step in format_steps")
        context_on_step = step[0]
        result_of_step = step[1]['content']
        formatted_output += f"## Step {i}\n\n"
        formatted_output += f"### Task Description\n{context_on_step}\n\n"
        formatted_output += f"### Result\nThe following is the response returned after the subagent performed the task above:\n\n"
        # Add the result content
        formatted_output += f"{result_of_step}\n\n"    
        # Add separator between steps
        if i < len(steps_list):
            formatted_output += "---\n\n"
    return formatted_output

def format_action_plan(steps_list):
    """
    Process a list of Step objects and format them into a comprehensive summary string.
    
    Args:
        steps_list: List of Step objects with description and subagent attributes
        
    Returns:
        Formatted string with key information about agent actions
    """
    summary = "# ACTION PLAN FROM PREVIOUS SUPERVISOR/PLANNER\n\n"
    
    for i, step in enumerate(steps_list, 1):
        # Extract information from Step object
        task_description = step.description
        subagent = step.subagent
        
        # Format the action summary
        summary += f"## Action {i}\n\n"
        summary += f"### Agent\n{subagent}\n\n"
        summary += f"### Task\n{task_description}\n\n"
        
        # Add separator between actions except for the last one
        if i < len(steps_list):
            summary += "---\n\n"
    return summary

def supervisor(state: State, config: Config) -> dict:
    print("\n" + "="*50)
    print("🎯 SUPERVISOR FUNCTION CALLED")
    print("="*50)
    
    first_message = state["messages"][-1]
    structured_model = model.with_structured_output(Plan)
    # Filter out tool messages from conversation history
    filtered_messages = [msg for msg in state["messages"][-3:] 
                         if not (hasattr(msg, 'type') and msg.type in ["tool", "tool_call"])]
    result = structured_model.invoke([
        SystemMessage(content=supervisor_prompt)
    ] + filtered_messages)
    return {
        "messages": state["messages"], 
        "action_plan": result.steps,
        "original_objective": first_message.content,
    }

def agent_executor(state: State, config: Config) -> dict:
    print("\n" + "="*50)
    print("🤖 AGENT EXECUTOR FUNCTION CALLED")
    print("="*50)
    
    plan = state["action_plan"]
    total_plan = "\n".join([
        f"{i+1}. {step.subagent} will: {step.description}" 
        for i, step in enumerate(plan)
    ])
    first_task_subagent = plan[0].subagent
    first_task_description = plan[0].description
    first_task_subagent_response = first_task_subagent + " executed logic to answer the following request from the planner/supervisor: " + first_task_description
    task_formatted_prompt = f"""For the following plan: 
    {total_plan}
    You are tasked with executing the first step #1. {first_task_subagent} will: {first_task_description}
    """
    if first_task_subagent == "code_research_subagent":
        config = {
            "configurable": {
                "source_urls": ["https://react.dev/reference/react/hooks"]
            }
        }
        response = research_assistant_remote_graph.invoke({"query": first_task_description}, config)
        final_response = response["research_for_subagent"]
    elif first_task_subagent == "code_writing_subagent":
        response = code_write_remote_graph.invoke({"messages": [HumanMessage(content=task_formatted_prompt)]})["messages"]
        final_response = response["messages"][-1].content
    elif first_task_subagent == "code_testing_subagent":
        response = code_test_remote_graph.invoke({"messages": [HumanMessage(content=task_formatted_prompt)]})["messages"]
        final_response = response["messages"][-1].content
    return {
        "messages": state["messages"],
        "action_plan": plan,
        "past_steps": [
            (first_task_subagent_response, final_response)
        ]
    }

def replanner(state: State, config: Config) -> dict:
    print("\n" + "="*50)
    print("🔄 REPLANNER FUNCTION CALLED")
    print("="*50)
    # works
    original_objective = state["original_objective"]
    # needs to be cleaned up
    previous_action_plan = state["action_plan"]
    # needs to be cleaned up
    previous_steps = state["past_steps"]
    formatted_action_plan = format_action_plan(previous_action_plan)
    formatted_steps = format_steps(previous_steps)
    print(formatted_steps, "formatted steps in replanner")
    print(formatted_action_plan, "formatted action plan in replanner")
    replanner_prompt = f"""
    You are a supervisor/planner for a multi-agent team that handles queries from customers. The multi-agent team you are supervising is responsible for handling questions related to developer productivity. Your team is composed of three subagents that you can use to help answer the customer's request:

1. code_research_subagent: this subagent searches the internet for information related to programming languages, libraries, and frameworks. this subagent should be used when the customer wants to learn more about the functionality of a specific programming language, library, or framework, or is just generally
curious about some specific topic.
2. code_writing_subagent: this subagent writes code based on the user's request.
3. code_testing_subagent: this subagent tests the user's provided code, writes tests, and suggests improvements.

Your role is to create an action plan that the subagents can follow and execute to thoroughly answer the customer's request. Your action plan should specify the exact steps that should be followed by the subagent(s) in order to successfully answer the customer's request, and which subagent should perform each step. Return the action plan as a list of objects, where each object contains the following fields:
- step: a detailed step that should be followed by the subagent(s) in order to successfully answer a portion of the customer's request.
- subagent_to_perform_step: the subagent that should perform the step.

Return the action plan in chronological order, starting with the first step to be performed, and ending with the last step to be performed. You should try your best not to have multiple steps performed by the same subagent. This is inefficient. If you need a subagent to perform a task, try and group it all into one step.

If you do not need a subagent to answer the customer's request, do not include it in the action plan/list of steps. Your goal is to be as efficient as possible, while ensuring that the customer's request is answered thoroughly and to the best of your ability.

Before you, another supervisor/planner has already created an action plan/list of steps for the subagents to follow. You should use this action plan as a starting point. Given the following information, update the existing action plan/list of steps to thoroughly answer the customer's request.

The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your original objective/request from the customer was this:
{original_objective}

The original action plan constructed by the previous supervisor/planner was this:
{formatted_action_plan}

The subagents have already executed the following steps:
{formatted_steps}

Update the plan accordingly. If no more steps are needed and you are ready to respond to the customer, use Response. If you still need to use the subagents to solve the problem, construct and return a list of steps to be performed by the subagents using Plan.

Take a deep breath and think carefully before responding. Go forth and make the customer delighted!
"""
    structured_model = model.with_structured_output(ReplannerResponse)
    result = structured_model.invoke([SystemMessage(content=replanner_prompt)])
    if isinstance(result.action, Response):
        return {"response": result.action.response}
    else:
        return {"plan": result.action.steps}

def should_end(state: State):
    print("\n" + "="*50)
    print("🎬 SHOULD_END FUNCTION CALLED")
    print("="*50)
    if "response" in state and state["response"]:
        return END
    else:
        return "agent_executor"

builder = StateGraph(State, config_schema=Config)
builder.add_node("supervisor", supervisor)
builder.add_node("agent_executor", agent_executor)
builder.add_node("replanner", replanner)
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "agent_executor")
builder.add_edge("agent_executor", "replanner")
builder.add_conditional_edges(
    "replanner",
    should_end,
    ["agent_executor", END],
)
graph = builder.compile()

config = {
    "configurable": {
        "context_for_researcher": [
            "https://react.dev/reference/react/hooks",
            "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
            "https://python.langchain.com/docs/introduction/",
        ]
    }
}

result = graph.invoke({ "messages": [HumanMessage(content="educate me on the use of useMemo. then write me code for a python script that adds two numbers together. then tell me what useRouter is, and what the benefit of NextJS is.")] }, config)
print(result, "result from graph call")

# planner -> agent_executor -> replanner -> agent_executor -> replanner -> END