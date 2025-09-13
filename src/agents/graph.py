from dotenv import load_dotenv
from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import sys
from src.agents.utils import rename_tool, get_issue_tools, get_release_tools, get_code_review_tools, get_documentation_tools

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")
github = GitHubAPIWrapper()
toolkit = GitHubToolkit.from_github_api_wrapper(github, include_release_tools=True)
tools = [rename_tool(tool) for tool in toolkit.get_tools()]

documentation_agent = create_react_agent(
    llm,
    tools=get_documentation_tools(tools),
    prompt=(
        "You are a documentation agent. Your task is to help write and maintain project documentation, "
        "including README files, API docs, and other relevant materials. Use the provided tools to read existing files, "
        "search code, and create or update documentation as needed. Always ensure that the documentation is clear, concise, "
        "and up-to-date with the latest project changes. Use the tools available to you effectively."
    ),
    name="documentation_agent",
)

release_notes_agent = create_react_agent(
    llm,
    tools=get_release_tools(tools),
    prompt=(
        "You are a release notes agent. Your task is to generate comprehensive and clear release notes for new software releases. "
        "Use the provided tools to gather information about the latest releases, including pull requests, commits, and issues. "
        "Ensure that the release notes highlight key features, bug fixes, and any other important changes in a user-friendly manner."
    ),
    name="workflow",
)

issue_agent = create_react_agent(
    llm,
    tools=get_issue_tools(tools),
    prompt=(
        "You are an issue management agent. Your task is to help manage and resolve issues within the project. "
        "Use the provided tools to search for existing issues, retrieve detailed information about specific issues, "
        "and add comments or updates as necessary. Ensure that issues are addressed promptly and that all relevant information is documented clearly."
    ),
    name="issue_agent",
)

code_review_agent = create_react_agent(
    llm,
    tools=get_code_review_tools(tools),
    prompt=(
        "You are a code review agent. Your task is to assist in reviewing code changes and pull requests within the project. "
        "Use the provided tools to list open pull requests, examine the files changed in each pull request, and read specific files as needed. "
        "Provide constructive feedback on code quality, adherence to coding standards, and potential improvements to ensure high-quality contributions."
    ),
    name="code_review_agent",
)

github_agent = create_supervisor(
    agents=[documentation_agent, release_notes_agent, issue_agent, code_review_agent],
    model=llm,
    prompt=(
        "You are a supervisor agent overseeing multiple specialized agents: "
        "- documentation_agent: handles project documentation tasks "
        "- workflow: generates release notes for new software versions "
        "- issue_agent: manages and resolves project issues "
        "- code_review_agent: assists in reviewing code changes and pull requests "
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()



if __name__ == "__main__":
    res = github_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Create a documentation and update it in the readme."
            }
        ]
    })
    print(res)
