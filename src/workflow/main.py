import os
from typing import TypedDict, List
import sys
from dotenv import load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_groq import ChatGroq
import sys
from langgraph.graph import StateGraph, START, END
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.config import grok_api_key
from langchain.globals import set_verbose,set_debug

set_verbose(True)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
# agent_executor.invoke(
#     {"input": "Who directed the 2023 film Oppenheimer and what is their age in days?"}
# )

set_debug(True)

load_dotenv()
API_KEY=grok_api_key

class ReleaseState(TypedDict):
    release_info: str
    prs_and_commits: List[str]
    release_notes: str


llm = ChatGroq(api_key=grok_api_key, model="openai/gpt-oss-120b")
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github, include_release_tools=True)  # Prebuilt Github tools
tools = toolkit.get_tools()
tools_map = {t.name: t for t in tools}


def get_latest_release(state: ReleaseState) -> ReleaseState:
    tool = tools_map.get("Get latest release")
    if not tool:
        raise RuntimeError("GitHub tool 'Get latest release' not found.")
    result = tool.run({"repo": os.getenv("GITHUB_REPOSITORY")})
    state["release_info"] = str(result)
    return state


def collect_prs_commits(state: ReleaseState) -> ReleaseState:
    tool = tools_map.get("List open pull requests (PRs)")
    if not tool:
        raise RuntimeError("GitHub tool 'List pull requests' not found.")
    result = tool.run({"repo": os.getenv("GITHUB_REPOSITORY"), "state": "closed"})
    state["prs_and_commits"] = [str(result)]
    return state


def generate_release_notes(state: ReleaseState) -> ReleaseState:
    prompt = f"""
    You are a release notes generator.
    Latest release info: {state['release_info']}
    PRs & commits since last release: {state['prs_and_commits']}
    Generate release notes for version {state['release_info'].split('v')[-1]}.
    Write professional release notes in Markdown format.
    """
    response = llm.invoke(prompt)
    state["release_notes"] = response.content
    return state


# Graph
graph = StateGraph(ReleaseState)

graph.add_node("get_latest_release", get_latest_release)
graph.add_node("collect_prs_commits", collect_prs_commits)
graph.add_node("generate_release_notes", generate_release_notes)

graph.add_edge(START, "get_latest_release")
graph.add_edge("get_latest_release", "collect_prs_commits")
graph.add_edge("collect_prs_commits", "generate_release_notes")
graph.add_edge("generate_release_notes", END)

app = graph.compile()

if __name__ == "__main__":
    final_state = app.invoke({})
    print("\n Generated Release Notes:\n")
    print(final_state["release_notes"])
