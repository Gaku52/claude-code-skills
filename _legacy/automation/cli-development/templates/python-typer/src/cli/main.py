"""Main CLI application."""
import typer
from rich.console import Console
from typing import Optional

from .commands import create, list_cmd, delete

app = typer.Typer(
    name="mycli",
    help="A powerful CLI tool for project management",
    add_completion=True,
)
console = Console()

# グローバルステート
class State:
    def __init__(self) -> None:
        self.verbose: bool = False
        self.config_path: Optional[str] = None

state = State()

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path"),
) -> None:
    """
    My CLI Tool - A powerful project generator.

    Use --verbose for detailed output.
    """
    state.verbose = verbose
    state.config_path = config

    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")

# コマンド登録
app.add_typer(create.app, name="create", help="Create a new project")
app.command(name="list")(list_cmd.list_projects)
app.command(name="delete")(delete.delete_project)

if __name__ == "__main__":
    app()
