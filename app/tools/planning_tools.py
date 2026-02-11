from typing import List
from langchain.tools import tool


@tool
def todo_write(tasks: List[str]) -> str:
    """
    Creates a list of todo list before calling other tools or subagents

    :returns: a string cointaing the list of tasks to be done by the deep agent
    """
    formatted_tasks = "\n".join([f"- {task}" for task in tasks])
    return f"Todo list created:\n{formatted_tasks}"