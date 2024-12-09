from invoke import task

@task
def test(c):
    """Run all unit and integration tests except GUI tests."""
    c.run("pytest --ignore=tests/integration/test_gui.py -v")

@task
def lint(c):
    c.run("ruff .")

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
