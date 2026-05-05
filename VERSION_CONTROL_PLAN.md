# Version Control Plan for MoralVerse.AI

## 📋 Current Status
✅ `.gitignore` already exists (well-configured)
✅ `.env.example` exists for team reference
⚠️ Git repository likely NOT initialized yet

---

## 🎯 Implementation Plan (4 Phases)

### **Phase 1: Local Git Setup** ✓ Quick & Local
**Duration:** 5-10 minutes

**Steps:**
1. Initialize Git repository
   ```bash
   cd c:\Users\user\Desktop\fyp.031.full_pipeline_working\MoralVerse_FYP
   git init
   ```

2. Configure Git identity
   ```bash
   git config user.name "Your Name"
   git config user.email "your.email@example.com"
   
   # (Global config - optional, skip if already done)
   # git config --global user.name "Your Name"
   # git config --global user.email "your.email@example.com"
   ```

3. Create initial commit
   ```bash
   git add .
   git commit -m "Initial commit: Full MoralVerse.AI pipeline working"
   ```

**Output:**
- Git repository initialized locally
- All files tracked (except those in .gitignore)
- Ready for local version history

---

### **Phase 2: Branch Strategy Setup** 🌿
**Duration:** 10-15 minutes

**Recommended Branch Structure (Git Flow):**

```
main (production-ready releases)
  ↑
  └── release/v1.0 (release candidate)
        ↑
        └── develop (integration branch)
              ↑
              ├── feature/character-consistency
              ├── feature/async-tasks
              ├── feature/advanced-decisions
              ├── bugfix/avatar-processing
              └── docs/update-readme
```

**Branch Naming Conventions:**
- **main** - Stable, production-ready code only
- **develop** - Integration branch for features
- **feature/** - New features (e.g., `feature/full-decision-regeneration`)
- **bugfix/** - Bug fixes (e.g., `bugfix/image-generation-timeout`)
- **docs/** - Documentation updates
- **hotfix/** - Critical production fixes

**Initial Setup:**
```bash
# Create develop branch from main
git checkout -b develop
git push -u origin develop

# Set main as default branch on GitHub (after remote setup)
```

---

### **Phase 3: Remote Repository Setup** 🌐
**Duration:** 10-20 minutes

**Option A: GitHub (Recommended for FYP)**
1. Go to https://github.com/new
2. Create repository: `MoralVerse-FYP` (or similar)
3. Initialize with:
   - ❌ NO README (you have one)
   - ❌ NO .gitignore (you have one)
   - ❌ NO LICENSE (add later)
4. Connect local repo to remote:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/MoralVerse-FYP.git
   git branch -M main
   git push -u origin main
   ```

**Option B: GitLab (Alternative)**
- Similar process on https://gitlab.com/projects/new

**Verification:**
```bash
git remote -v  # Should show: origin https://...
git branch -a  # Should show: main, develop (if pushed)
```

---

### **Phase 4: Collaboration & Maintenance Setup** 👥
**Duration:** 15-30 minutes

#### A. GitHub Configuration
1. **Branch Protection Rules**
   - Go to Settings → Branches
   - Protect `main` branch:
     - ✅ Require pull request reviews before merging
     - ✅ Require status checks to pass
     - ✅ Require branches to be up to date

2. **Pull Request Template** (Create `.github/pull_request_template.md`)
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Feature
   - [ ] Bug fix
   - [ ] Documentation
   
   ## Testing
   How to test these changes?
   
   ## Checklist
   - [ ] Code follows project conventions
   - [ ] Comments added for complex logic
   - [ ] Tests added/updated
   - [ ] No new warnings generated
   ```

3. **CONTRIBUTING.md** (for team collaboration)
   ```markdown
   # Contributing to MoralVerse.AI
   
   ## Setup
   1. Fork repo
   2. Clone: `git clone https://github.com/YOUR_USERNAME/MoralVerse-FYP.git`
   3. Create feature branch: `git checkout -b feature/your-feature`
   
   ## Commit Convention
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation
   - style: Code style (formatting, missing semicolons, etc.)
   - refactor: Code refactoring
   - test: Adding tests
   
   Example: `git commit -m "feat: add async task queue for story generation"`
   
   ## Pull Request Process
   1. Ensure all tests pass
   2. Create PR to `develop` branch
   3. Wait for review & approval
   4. Squash & merge
   ```

#### B. Commit Convention (Conventional Commits)
**Format:** `<type>(<scope>): <subject>`

**Examples:**
```bash
git commit -m "feat(llm-engine): add Groq provider support"
git commit -m "fix(image-engine): resolve character consistency in Scene 2"
git commit -m "docs(readme): update setup instructions for Windows"
git commit -m "refactor(pipeline): optimize image generation step"
git commit -m "test(decision-engine): add branching logic tests"
```

#### C. .gitattributes (Optional - For Consistency)
Create `.gitattributes`:
```
* text=auto
*.py text eol=lf
*.js text eol=lf
*.html text eol=lf
*.md text eol=lf
*.json text eol=lf
*.mp4 binary
*.png binary
*.jpg binary
```

---

## 📊 Gitignore Status Review

**Current `.gitignore` Covers:**
✅ Python cache (`__pycache__/`, `*.pyc`)
✅ Virtual environment (`.venv/`, `venv/`)
✅ Django static files (`staticfiles/`)
✅ Generated media (videos, audio, images, subtitles)
✅ Secrets (`.env`)
✅ System files (`.DS_Store`, `Thumbs.db`)

**Recommendations to Add:**
```
# Database snapshots
db.sqlite3
*.db

# Logs
logs/*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
*.tmp

# Testing
.pytest_cache/
htmlcov/
.coverage

# Build artifacts
dist/
build/
*.egg-info/

# Temporary files
*.tmp
*.bak
```

---

## 🔐 Security Best Practices

### Already Implemented ✅
- `.env` in `.gitignore` (secrets protected)
- `.env.example` tracked (reference for team)

### Additional Steps:
1. **Add pre-commit hook** (optional but recommended)
   ```bash
   pip install pre-commit
   ```
   Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       rev: v4.4.0
       hooks:
         - id: check-ast
         - id: check-json
         - id: check-yaml
         - id: detect-private-key
         - id: end-of-file-fixer
         - id: trailing-whitespace
     
     - repo: https://github.com/psf/black
       rev: 23.3.0
       hooks:
         - id: black
   ```

2. **Secret scanning** (GitHub feature)
   - Settings → Security & analysis → Enable secret scanning
   - GitHub will warn if API keys are accidentally committed

3. **GitHub Security Alert Settings**
   - Enable "Dependabot alerts" for vulnerable dependencies
   - Enable "Dependabot security updates"

---

## 📈 Workflow Example (Feature Development)

```bash
# Step 1: Start new feature
git checkout develop
git pull origin develop
git checkout -b feature/full-decision-regeneration

# Step 2: Make changes, commit with convention
git add ai_modules/decision_engine.py
git commit -m "feat(decision-engine): implement full LLM regeneration on user choice"

# Step 3: Push to GitHub
git push -u origin feature/full-decision-regeneration

# Step 4: Create Pull Request on GitHub UI
# - Add description
# - Link to issues if applicable
# - Request review

# Step 5: After approval & CI passes
git checkout develop
git pull origin develop
git merge --squash feature/full-decision-regeneration
git commit -m "feat(decision-engine): implement full LLM regeneration on user choice"
git push origin develop

# Step 6: Delete feature branch
git branch -d feature/full-decision-regeneration
git push origin --delete feature/full-decision-regeneration
```

---

## 🔄 Continuous Integration (CI/CD) - Optional but Recommended

### GitHub Actions Setup
Create `.github/workflows/tests.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python manage.py test
    
    - name: Check code style
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## 📅 Recommended Timeline for FYP Submission

```
Week 1: Phase 1 (Local setup)           → Ready to commit
Week 2: Phase 2 (Branches)              → Organization
Week 3: Phase 3 (Remote setup)          → Team collaboration ready
Week 4: Phase 4 (Maintenance setup)     → Professional grade

Before Final Submission:
- Ensure all work is in main branch
- Tag release: git tag -a v1.0 -m "Final FYP submission"
- Push tags: git push origin v1.0
```

---

## 🎯 Quick Decision Tree

**Choose Based On:**

| Scenario | Recommendation |
|----------|---|
| Working alone | Phase 1 + Phase 3 (local + GitHub) |
| Team project | All 4 phases + CI/CD |
| FYP submission | Phase 1 + 3 + tag as v1.0 |
| Production ready | All phases + security features |

---

## 📝 Commands Reference

```bash
# Initialize
git init
git add .
git commit -m "Initial commit: description"

# Branching
git checkout -b feature/name        # Create & switch
git branch -a                        # List all branches
git merge feature/name               # Merge branch
git branch -d feature/name           # Delete local branch

# Remote
git remote add origin <url>         # Add remote
git push -u origin main             # Push & track
git pull origin develop             # Fetch & merge
git fetch                           # Fetch without merge

# History
git log --oneline                   # Commit history
git show <commit-hash>              # Show commit details
git diff main develop               # Compare branches
git status                          # Current status

# Undo (if needed)
git reset --soft HEAD~1             # Undo last commit, keep changes
git revert <commit-hash>            # Create reverse commit
git clean -fd                       # Remove untracked files
```

---

## ✅ Implementation Checklist

- [ ] Phase 1: Initialize Git locally
- [ ] Phase 1: First commit
- [ ] Phase 2: Create develop branch
- [ ] Phase 3: Create GitHub repository
- [ ] Phase 3: Push to GitHub
- [ ] Phase 4: Configure branch protection (main)
- [ ] Phase 4: Add PR template
- [ ] Phase 4: Add CONTRIBUTING.md
- [ ] Phase 4: Enable secret scanning
- [ ] Optional: Setup GitHub Actions (CI/CD)
- [ ] Optional: Add pre-commit hooks
- [ ] Final: Tag release v1.0 for FYP submission

---

## 📞 Next Steps

**Ready to implement?** I can help with:
1. **Execute all Phase 1 setup** (git init, first commit)
2. **Create necessary files** (.github/pull_request_template.md, CONTRIBUTING.md, .gitattributes)
3. **Configure GitHub** (after you create repository)
4. **Setup GitHub Actions** (CI/CD pipeline)

**Just let me know which phase(s) you want to start with!** 🚀
