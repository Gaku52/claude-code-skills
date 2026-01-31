"""Test commands."""
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_create_command():
    """Test create command."""
    result = runner.invoke(app, ["create", "project", "myapp"])
    assert result.exit_code == 0
    assert "Creating project: myapp" in result.stdout

def test_create_with_template():
    """Test create with template option."""
    result = runner.invoke(app, ["create", "project", "myapp", "--template", "react"])
    assert result.exit_code == 0
    assert "react" in result.stdout.lower()

def test_list_command():
    """Test list command."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "Projects" in result.stdout

def test_delete_command_with_force():
    """Test delete command with force flag."""
    result = runner.invoke(app, ["delete", "myapp", "--force"])
    assert result.exit_code == 0
    assert "Deleting project: myapp" in result.stdout

def test_help():
    """Test help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout

def test_version():
    """Test version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
