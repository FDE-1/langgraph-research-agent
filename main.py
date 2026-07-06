import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

from langchain_core.tools import BaseTool, StructuredTool
from tavily import TavilyClient

from langgraph_research_agent.agent import Agent
from langgraph_research_agent.tools.web_search import web_search_func
from langgraph_research_agent.tools.calculator import calculator
from langgraph_research_agent.tools.save_file import save_file
from langgraph_research_agent.tools.wikipedia import wikipedia
from langgraph_research_agent.tools.search_memory import search_memory

console = Console()

def init_agent() -> Agent:
    """Init the different agent"""
    tavily_client = TavilyClient()

    def web_search_bound(query: str, max_result: int = 5) -> dict[str, object]:
        """Search the web via Tavily and return a dict.
        Use when you have to search on a general subject.
        Do not use for specific search and calculator."""
        return web_search_func(tavily_client, query, max_result)

    web_search = StructuredTool.from_function(
        func=web_search_bound,
        name="web_search",
        handle_tool_error=True,
    )

    tools: list[BaseTool] = [
        web_search,
        calculator,
        save_file,
        wikipedia,
        search_memory
    ]
    
    return Agent(funcs=tools)

def main() -> None:
    welcome_message = Text("Welcome to the agent of langgraph research 🧠\n", justify="center", style="bold blue")
    welcome_message.append("Write your question here. Write 'quit' or 'exit' to exit.", style="italic cyan")
    console.print(Panel(welcome_message, border_style="blue", title="LangGraph Agent"))

    try:
        with console.status("[bold yellow]Loading of the different tools...[/bold yellow]"):
            agent = init_agent()
            session_id = str(uuid.uuid4())
    except Exception as e:
        console.print(f"[bold red]Fail to initialize : {e}[/bold red]")
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            
            if user_input.lower() in ["quit", "exit", "/quit", "/exit", "q"]:
                console.print("\n[bold blue]Goodbye ![/bold blue] 👋")
                break
                
            if not user_input.strip():
                continue

            with console.status("[bold magenta]Agent is thinking...[/bold magenta]", spinner="dots"):
                response = agent.run(user_input, thread_id=session_id)
            
            console.print(Panel(
                Markdown(response), 
                title="[bold magenta]Assistant[/bold magenta]", 
                border_style="magenta",
                padding=(1, 2)
            ))

        except KeyboardInterrupt:
            console.print("\n[bold red]Interruption by the user. Goodbye ![/bold red] 👋")
            break
        except Exception as e:
            console.print(f"\n[bold red]An error occured in the execution :[/bold red] {e}")

if __name__ == "__main__":
    main()