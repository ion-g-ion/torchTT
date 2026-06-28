# Contributing to torchTT

Thank you for your interest in contributing to `torchTT`! We welcome contributions from the community to help improve the library.

## How to Contribute

1.  **Report Bugs**: If you find a bug, please open an issue describing the problem, including a minimal reproducible example.
2.  **Suggest Features**: If you have an idea for a new feature, feel free to open an issue to discuss it.
3.  **Submit Pull Requests**: We accept pull requests for bug fixes, improvements, and new features.

## Development Setup

To set up a development environment:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork**:
    ```bash
    git clone https://github.com/your-username/torchTT.git
    cd torchTT
    ```
3.  **Install the package in editable mode** with development dependencies:
    ```bash
    pip install -e ".[dev]"
    ```
    Or using `uv`:
    ```bash
    uv sync --extra dev
    ```

## Running Tests

We use `pytest` for testing. Ensure all tests pass before submitting a pull request:

```bash
pytest tests/
```

## Code Style

Please ensure your code follows standard Python coding conventions (PEP 8). 

## Submission Guidelines

1.  Create a new branch for your changes: `git checkout -b my-feature-branch`.
2.  Commit your changes with clear, descriptive commit messages.
3.  Push your branch to your fork: `git push origin my-feature-branch`.
4.  Open a Pull Request against the `main` branch of the original repository.
5.  Describe your changes in detail in the PR description.

We will review your PR and provide feedback as soon as possible. Thank you for contributing!

