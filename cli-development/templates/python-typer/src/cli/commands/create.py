"""Create command."""
import typer
from rich.console import Console
from rich.progress import track
from rich.panel import Panel
import time

app = typer.Typer()
console = Console()

@app.command()
def project(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option("default", "--template", "-t", help="Template to use"),
    typescript: bool = typer.Option(True, "--typescript/--no-typescript"),
    install: bool = typer.Option(True, "--install/--no-install", help="Install dependencies"),
) -> None:
    """
    Create a new project from a template.

    Example:
        mycli create myapp
        mycli create myapp --template react --no-typescript
    """
    console.print(Panel.fit(
        f"[cyan]Creating project: {name}[/cyan]",
        title="Project Generator",
        border_style="cyan"
    ))

    console.print(f"Template: [green]{template}[/green]")
    console.print(f"TypeScript: [green]{typescript}[/green]")

    # プロジェクト作成処理
    tasks = [
        "Creating directory structure",
        "Copying template files",
        "Configuring project",
    ]

    if install:
        tasks.append("Installing dependencies")

    for task in track(tasks, description="Setting up project"):
        time.sleep(0.5)

    console.print("\n[bold green]✓ Project created successfully![/bold green]\n")
    console.print("[cyan]Next steps:[/cyan]")
    console.print(f"  cd {name}")
    if not install:
        console.print("  npm install")
    console.print("  npm run dev")
