from invoke import task

@task
def test(c):
    """Run tests with pytest."""
    c.run("poetry run pytest", pty=True)

@task
def lint(c):
    """Run linting with Ruff."""
    c.run("poetry run ruff .", pty=True)

@task
def format(c):
    """Format the code using ruff."""
    c.run("poetry run ruff format src/ tasks.py tests/")

@task
def type(c):
    """Run type checks with MyPy."""
    c.run("poetry run mypy src", pty=True)

@task
def docs(c):
    """Generate HTML documentation with pdoc."""
    c.run("poetry run pdoc src --output-dir docs --html")

@task
def run(c, config="config/config.yaml"):
    """Run the data pipeline with the specified configuration file."""
    c.run(f"poetry run softeng-data-pipeline --config {config}")
