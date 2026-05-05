# Contributing to MoralVerse.AI

Thank you for your interest in contributing to MoralVerse.AI! This document provides guidelines and instructions for contributing to the project.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Commit Convention](#commit-convention)
4. [Code Style](#code-style)
5. [Testing](#testing)

---

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- Virtual environment (`venv`)
- API keys for LLM/Image/TTS services (see `.env.example`)

### Initial Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mhamzii594-wq/MoralVerse-FYP.git
   cd MoralVerse_FYP
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # Windows
   source .venv/bin/activate     # Unix/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Start development server:**
   ```bash
   python manage.py runserver
   ```

---

## Development Workflow

### Branch Strategy (Git Flow)

```
main (stable, production-ready)
  ↑
  └── develop (integration branch)
        ↑
        ├── feature/your-feature-name
        ├── bugfix/issue-name
        └── docs/documentation-update
```

### Creating a Feature Branch

```bash
# Ensure you're on develop and up-to-date
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/descriptive-name

# Example: feature/async-story-generation
# Example: bugfix/image-consistency-issue
```

### Making Changes

1. **Write your code**
   - Follow project conventions (see [Code Style](#code-style))
   - Keep changes focused and atomic

2. **Test your changes**
   ```bash
   python manage.py test
   ```

3. **Commit with conventional messages** (see [Commit Convention](#commit-convention))
   ```bash
   git add .
   git commit -m "feat(scope): description"
   ```

### Pushing and Creating Pull Request

```bash
# Push your branch
git push -u origin feature/your-feature-name

# Go to GitHub and create a Pull Request
# - Base: develop
# - Compare: feature/your-feature-name
# - Fill out PR template with description
```

---

## Commit Convention

We follow **Conventional Commits** for clear and structured commit history.

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat** - New feature
- **fix** - Bug fix
- **docs** - Documentation changes
- **style** - Code style (formatting, missing semicolons, etc.)
- **refactor** - Code refactoring without feature changes
- **test** - Adding or updating tests
- **chore** - Dependency updates, tooling changes
- **perf** - Performance improvements

### Scope (Optional but recommended)
Common scopes:
- `llm-engine` - Story generation
- `image-engine` - Image generation
- `tts-engine` - Text-to-speech
- `avatar-processor` - Avatar handling
- `video-engine` - Video assembly
- `decision-engine` - Branching logic
- `pipeline` - Main orchestration
- `models` - Database models
- `views` - Django views/REST endpoints

### Examples

```bash
# Feature
git commit -m "feat(llm-engine): add Groq provider support"

# Bug fix
git commit -m "fix(image-engine): resolve character consistency in Scene 2"

# Documentation
git commit -m "docs(readme): update Windows setup instructions"

# Refactoring
git commit -m "refactor(pipeline): optimize image generation step"

# Performance
git commit -m "perf(tts-engine): add response caching for audio"

# With body
git commit -m "feat(decision-engine): implement full LLM regeneration

- Regenerates scenes 2-4 based on user decision
- Maintains character consistency via anchors
- Caches previous scenes for performance"
```

---

## Code Style

### Python (PEP 8)
We follow [PEP 8](https://pep8.org/) style guide with the following preferences:

- **Line length:** 120 characters (soft limit)
- **Indentation:** 4 spaces
- **Naming:** 
  - Variables/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Django Specific
- Use Django ORM for database queries (no raw SQL unless necessary)
- Properly document all models with docstrings
- Use type hints in function signatures
- Create migrations for any model changes

### Code Quality
Optional but recommended:
- Use `black` for code formatting
- Use `flake8` for linting
- Add type hints with `mypy`

```bash
# Install (optional)
pip install black flake8 mypy

# Format code
black ai_modules/ core/

# Check style
flake8 ai_modules/ core/
```

---

## Testing

### Running Tests
```bash
# Run all tests
python manage.py test

# Run specific test file
python manage.py test core.tests.test_models

# Run with verbosity
python manage.py test --verbosity=2
```

### Writing Tests
- Place tests in `tests/` directory
- Follow naming convention: `test_<feature>.py`
- Use Django TestCase for database tests
- Mock external API calls

Example:
```python
from django.test import TestCase
from core.models import StoryRequest

class StoryRequestTestCase(TestCase):
    def setUp(self):
        self.story = StoryRequest.objects.create(
            child_name="Test Child",
            child_age=8,
            moral_theme="Honesty"
        )
    
    def test_story_creation(self):
        self.assertEqual(self.story.child_name, "Test Child")
        self.assertEqual(self.story.status, "pending")
```

---

## Key Project Files

### Configuration
- `moralverse/settings.py` - Django settings
- `.env.example` - Environment variables template
- `requirements.txt` - Python dependencies

### AI Modules
- `ai_modules/llm_engine.py` - Story generation
- `ai_modules/image_engine.py` - Scene illustrations
- `ai_modules/tts_engine.py` - Voice narration
- `ai_modules/avatar_processor.py` - Avatar conversion
- `ai_modules/video_engine.py` - Video assembly

### Core App
- `core/models.py` - Database models
- `core/views.py` - Request handlers
- `core/services/pipeline.py` - Main orchestration

### Documentation
- `README.md` - User guide
- `PROJECT_ANALYSIS.md` - Technical architecture
- `VERSION_CONTROL_PLAN.md` - Git strategy

---

## Questions?

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Start a discussion in the repository

---

**Thank you for contributing to MoralVerse.AI!** 🚀
