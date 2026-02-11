"""Delete command."""
import typer
from rich.console import Console
import time

console = Console()

def delete_project(
    name: str = typer.Argument(..., help="Project name"),
    force: bool = typer.Option(False, "--force", "-f", help="Force delete without confirmation"),
) -> None:
    """Delete a project."""
    if not force:
        confirmed = typer.confirm(f'Delete project "{name}"?')
        if not confirmed:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Abort()

    console.print(f"[red]Deleting project: {name}[/red]")
    time.sleep(1)
    console.print("[green]✓ Project deleted[/green]")
