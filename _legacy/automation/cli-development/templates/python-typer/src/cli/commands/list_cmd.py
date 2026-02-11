"""List command."""
import typer
from rich.console import Console
from rich.table import Table

console = Console()

def list_projects(
    all: bool = typer.Option(False, "--all", "-a", help="Show all projects")
) -> None:
    """List all projects."""
    table = Table(title="Projects")

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Template", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Size", justify="right", style="blue")

    # サンプルデータ
    table.add_row("myapp", "React", "2026-01-03", "10 MB")
    table.add_row("api", "Node.js", "2026-01-02", "5 MB")
    table.add_row("dashboard", "Vue", "2026-01-01", "8 MB")

    console.print(table)
