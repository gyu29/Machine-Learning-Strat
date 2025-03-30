# Contributing to S&P 500 Price Prediction and Trading Strategy

Thank you for your interest in contributing to this project! This document provides guidelines and best practices for contributing to the S&P 500 Price Prediction and Trading Strategy project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### 1. Fork and Clone the Repository

1. Fork the repository by clicking the "Fork" button on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/Machine-Learning-Strat.git
   cd Machine-Learning-Strat
   ```

### 2. Set Up Development Environment

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Making Changes

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes following these guidelines:
   - Follow PEP 8 style guide for Python code
   - Add docstrings to all new functions and classes
   - Include type hints for function parameters and return values
   - Write clear, descriptive commit messages
   - Keep commits focused and atomic

3. Test your changes:
   ```bash
   ./run_tests.sh
   ```

### 4. Project Structure Guidelines

- Place new source code in the appropriate directory under `src/`
- Add new tests in the `tests/` directory
- Include example notebooks in the `notebooks/` directory
- Update documentation as needed

### 5. Documentation

- Update the README.md if you add new features or make significant changes
- Add docstrings to all new functions and classes
- Include comments for complex algorithms or business logic
- Update the example notebooks if you modify existing functionality

### 6. Pull Request Process

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request on GitHub:
   - Use a clear, descriptive title
   - Provide a detailed description of your changes
   - Reference any related issues
   - Include screenshots or visualizations if applicable

3. Ensure your PR:
   - Passes all tests
   - Follows the project's coding standards
   - Includes appropriate documentation
   - Has no merge conflicts

### 7. Review Process

- All PRs require at least one review from a maintainer
- Address review comments promptly
- Make requested changes and push updates to your PR
- Keep the PR up to date with the main branch

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Maximum line length: 100 characters
- Use type hints for function parameters and return values

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Include integration tests for complex features
- Maintain or improve test coverage

### Performance

- Optimize code for efficiency
- Profile code when making performance-critical changes
- Document performance considerations
- Include benchmarks for significant changes

### Security

- Never commit sensitive data or API keys
- Use environment variables for configuration
- Follow security best practices
- Report security vulnerabilities privately

## Getting Help

- Open an issue for bug reports or feature requests
- Join discussions in existing issues
- Contact maintainers for urgent matters

## Recognition

Contributors will be recognized in the project's README.md and release notes. Significant contributions may result in maintainer status.

Thank you for contributing to this project! 