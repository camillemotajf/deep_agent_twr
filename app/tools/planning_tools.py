from typing import List
from langchain.tools import tool


@tool
def todo_write(tasks: List[str]) -> str:
    formatted_tasks = "\n".join([f"- {task}" for task in tasks])
    return f"Todo list created:\n{formatted_tasks}"