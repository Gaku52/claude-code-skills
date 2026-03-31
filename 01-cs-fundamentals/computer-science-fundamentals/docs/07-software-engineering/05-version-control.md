# Version Control

> Mastering Git is a fundamental skill for modern software engineers, on par with reading and writing.

## What You Will Learn in This Chapter

- [ ] Understand Git's internal model (DAG)
- [ ] Be able to explain branching strategies
- [ ] Learn practical Git workflows used in production
- [ ] Master how to write commit messages
- [ ] Understand the trade-offs between merge and rebase
- [ ] Acquire conflict resolution procedures
- [ ] Learn automation through Git Hooks
- [ ] Understand CI/CD integration patterns
- [ ] Learn best practices for managing large-scale repositories
- [ ] Master practical troubleshooting techniques


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Introduction to System Design](./04-system-design-basics.md)

---

## 1. Git's Internal Model

### 1.1 DAG and Object Model

```
Git = Directed Acyclic Graph (DAG) + Content-Addressable Storage

  Commit: A snapshot (tree of the entire file set)
  Branch: A pointer to a commit (just a reference)
  HEAD: A pointer to the current branch/commit

  DAG Structure:
  C1 <- C2 <- C3 <- C4      main
                 \
                  C5 <- C6  feature

  Each commit holds the hash of its parent commit
  -> Tamper-proof history chain (same principle as blockchain)

  Object Model:
  - blob: File contents (SHA-1 hash)
  - tree: Directory (references to blobs and trees)
  - commit: Tree + parent commit + metadata
  - tag: Named reference to a commit
```

### 1.2 Object Model in Detail

```
Git's Four Object Types:

  1. blob (Binary Large Object)
     +----------------------------------+
     | blob 15\0Hello, World!\n         |
     | SHA-1: af5626b4a114abcb82d63db   |
     +----------------------------------+
     - The raw contents of a file
     - Does not include the file name (names are managed by trees)
     - Files with identical contents share a single blob

  2. tree (corresponds to a directory)
     +----------------------------------------------+
     | tree                                          |
     | 100644 blob a1b2c3... README.md               |
     | 100644 blob d4e5f6... main.py                 |
     | 040000 tree 7a8b9c... src/                    |
     +----------------------------------------------+
     - Holds file names and modes (permissions)
     - References to child blobs and trees

  3. commit
     +----------------------------------------------+
     | commit                                        |
     | tree    d8329fc...                             |
     | parent  a0b1c2d...                             |
     | author  Alice <alice@example.com> 1234567890   |
     | committer Alice <alice@example.com> 1234567890 |
     |                                                |
     | Add user authentication feature                |
     +----------------------------------------------+
     - Reference to the root tree
     - Reference to parent commit(s) (merge commits have two or more)
     - Author and committer (may be different people)
     - Commit message

  4. tag (annotated tag)
     +----------------------------------------------+
     | tag                                           |
     | object  e3f4a5b...                            |
     | type    commit                                |
     | tag     v1.0.0                                |
     | tagger  Alice <alice@example.com> 1234567890  |
     |                                               |
     | Release version 1.0.0                         |
     +----------------------------------------------+
```

```bash
# Commands to inspect the object model
git cat-file -t HEAD        # Display the object type
# commit

git cat-file -p HEAD        # Display the object contents
# tree d8329fc...
# parent a0b1c2d...
# author Alice <alice@example.com> 1704067200 +0900
# committer Alice <alice@example.com> 1704067200 +0900
#
# Add user authentication feature

git cat-file -p HEAD^{tree}  # Contents of the root tree
# 100644 blob af5626b... README.md
# 040000 tree 7a8b9c0... src

git cat-file -p af5626b      # Contents of a blob
# (file contents are displayed)

# Storage location of objects
ls .git/objects/
# af/  7a/  d8/  ...
# First 2 characters are the directory name, the rest is the file name
```

### 1.3 How References Work

```
Types of References:

  1. Branch References
     .git/refs/heads/main          -> commit hash
     .git/refs/heads/feature/login -> commit hash

  2. Remote Tracking References
     .git/refs/remotes/origin/main -> commit hash

  3. Tag References
     .git/refs/tags/v1.0.0         -> tag object hash

  4. HEAD Reference
     .git/HEAD -> ref: refs/heads/main (normal)
     .git/HEAD -> a0b1c2d... (detached HEAD)

  5. Special References
     ORIG_HEAD  -> HEAD before merge/rebase/reset
     MERGE_HEAD -> The other commit during a merge
     FETCH_HEAD -> Result of the last fetch

  Relative Reference Notation:
  HEAD~1     -> 1st parent of HEAD (= HEAD~)
  HEAD~2     -> 2nd ancestor of HEAD
  HEAD^1     -> First parent of HEAD (same as HEAD~ for regular commits)
  HEAD^2     -> Second parent of HEAD (only meaningful for merge commits)

  Visual Explanation:
  C1 <- C2 <- C3 <- M (merge commit, parents are C3 and C5)
                 \
                  C4 <- C5

  M~1 = C3     (follow first parent)
  M~2 = C2     (follow first parent again)
  M^1 = C3     (first parent)
  M^2 = C5     (second parent = source branch of merge)
```

### 1.4 Staging Area (Index)

```
Git's Three States:

  Working Directory    Staging Area (Index)    Repository
  +----------------+  +--------------------+  +----------------+
  | Edited files   |  | Changes to commit  |  | Committed      |
  |                |  |                    |  | history        |
  | file.py        |->| file.py            |->| commit abc     |
  | (modified)     |add| (staged)          |commit|             |
  +----------------+  +--------------------+  +----------------+
        ^                                          |
        +------------------------------------------+
                        checkout / restore

  Why does the staging area exist:
  1. Granular control over commits
     - You can commit only parts of a file (git add -p)
     - Split multiple changes into logical commits

  2. Review before committing
     - Verify planned changes with git diff --staged
     - Ensure no unintended changes are included

  3. Workspace for merge conflict resolution
     - Resolve conflicts in staging, then commit
```

```bash
# Practical staging usage

# Stage an entire file
git add file.py

# Interactively stage by hunk (chunk of changes)
git add -p file.py
# y: stage this hunk
# n: skip this hunk
# s: split the hunk further
# e: manually edit the hunk

# Verify staging contents
git diff --staged     # Staged changes
git diff              # Unstaged changes
git diff HEAD         # All changes (staged + unstaged)

# Undo staging
git restore --staged file.py  # Unstage a file (changes are preserved)
git restore file.py           # Discard file changes (caution!)
```

---

## 2. Branching Strategies

### 2.1 Major Branching Strategies

```
Major Branching Strategies:

  1. GitHub Flow (Simple):
     main --*--*--*--*--*--*--
                \     /
     feature    *--*--*

     -> main is always deployable
     -> Create PRs from feature branches
     -> Deploy after merging

  2. Git Flow (Strict):
     main    --*------------*--
     develop --*--*--*--*--*--
                \  /     \  /
     feature    *--*      *--*
     release              *--*--*

     -> Separate branches for development (develop), stable (main), and release
     -> Suited for large-scale projects

  3. Trunk-Based Development (Modern):
     main --*--*--*--*--*--*--*--
               \/  \/
     short     *    *    <- Short-lived branches (within 1 day)

     -> Merge to main frequently (multiple times per day)
     -> Feature flags hide incomplete features
     -> Adopted by Google, Facebook
```

### 2.2 Detailed Comparison of Each Strategy

```
Branching Strategy Comparison Table:

  +----------------------------------------------------------+
  | Attribute       | GitHub Flow | Git Flow   | Trunk-Based  |
  +----------------------------------------------------------+
  | Branch count    | Few         | Many       | Minimal      |
  | Complexity      | Low         | High       | Low          |
  | Release freq.   | High        | Low        | Highest      |
  | CI/CD fit       | High        | Moderate   | Highest      |
  | Team size       | Small-Med   | Large      | Small-Large  |
  | Learning cost   | Low         | High       | Low          |
  | Conflicts       | Moderate    | Many       | Few          |
  | Rollback        | Easy        | Easy       | Flag toggle  |
  | Target env.     | Web/SaaS   | Packages   | Web/SaaS     |
  | Notable adopter | GitHub     | Enterprise | Google       |
  +----------------------------------------------------------+

  Selection Guidelines:
  - Web apps / SaaS -> GitHub Flow or Trunk-Based
  - Mobile apps (periodic releases) -> Git Flow
  - Fully established CI/CD -> Trunk-Based
  - Team unfamiliar with Git -> GitHub Flow (simplest)
  - Supporting multiple versions simultaneously -> Git Flow
```

### 2.3 GitHub Flow in Practice

```bash
# --- GitHub Flow Workflow ---

# 1. Create a new branch from main
git switch main
git pull origin main
git switch -c feature/add-user-auth

# 2. Develop and commit
# ... edit code ...
git add src/auth.py tests/test_auth.py
git commit -m "feat: implement login authentication"

git add src/middleware.py
git commit -m "feat: add authentication middleware"

git add tests/test_middleware.py
git commit -m "test: add tests for authentication middleware"

# 3. Push to remote
git push -u origin feature/add-user-auth

# 4. Create a Pull Request
gh pr create --title "feat: add user authentication" --body "
## Overview
Implemented login authentication and middleware.

## Changes
- JWT-based authentication
- Authentication middleware
- Tests

## How to Test
\`\`\`bash
pytest tests/test_auth.py tests/test_middleware.py
\`\`\`
"

# 5. Merge after review
# (via GitHub UI or CLI)
gh pr merge --squash

# 6. Clean up local branches
git switch main
git pull origin main
git branch -d feature/add-user-auth
```

### 2.4 Trunk-Based Development in Practice

```bash
# --- Trunk-Based Development Workflow ---

# 1. Create a short-lived branch (to be merged within 1 day)
git switch main
git pull origin main
git switch -c short-lived/add-button

# 2. Make small commits
git add src/components/Button.tsx
git commit -m "feat: add new button component (behind flag)"

# 3. Hide incomplete features with feature flags
# config/features.py
# FEATURE_FLAGS = {
#     "new_button": False,  # Off in production
# }

# 4. Merge promptly
git push -u origin short-lived/add-button
gh pr create --title "feat: add button component" --body "Flag: new_button"
gh pr merge --squash

# 5. Update main and move to next task
git switch main
git pull origin main

# Gradual feature flag rollout:
# 1. Dev environment: flag ON -> test
# 2. Staging: flag ON -> QA
# 3. Production: canary release (ON for 1% of users)
# 4. Gradually increase ON percentage (10% -> 50% -> 100%)
# 5. ON for all users -> remove flag code
```

---

## 3. Commit Messages

### 3.1 Conventional Commits

```
Conventional Commits Specification:

  Format:
  <type>(<scope>): <description>

  <body>

  <footer>

  Type Values:
  +--------------------------------------------------+
  | type     | Description             | Example      |
  +--------------------------------------------------+
  | feat     | New feature             | Add login    |
  | fix      | Bug fix                 | Fix NPE      |
  | docs     | Documentation           | Update README|
  | style    | Code style              | Formatting   |
  | refactor | Refactoring             | Split function|
  | perf     | Performance improvement | Optimize query|
  | test     | Tests                   | Add tests    |
  | build    | Build system            | webpack config|
  | ci       | CI configuration        | GitHub Actions|
  | chore    | Miscellaneous tasks     | Update deps  |
  | revert   | Revert a commit         | Revert change|
  +--------------------------------------------------+

  scope (optional): Area affected by the change
  -> auth, api, ui, db, config, etc.

  Indicating BREAKING CHANGE:
  -> Append ! after type: feat!: ...
  -> Add BREAKING CHANGE: ... in the footer
```

### 3.2 Writing Good Commit Messages

```bash
# --- Examples of Good Commit Messages ---

# Good: Short one-line message (for simple changes)
git commit -m "fix: correct email validation error during user registration"

# Good: Message with body (for complex changes)
git commit -m "$(cat <<'EOF'
feat(auth): implement JWT-based authentication

Added JWT token authentication in addition to password authentication.
Token expiration is 24 hours, with automatic renewal via refresh tokens.

- Access token issuance and verification
- Refresh token rotation
- Token blacklist for invalidation

Closes #123
EOF
)"

# --- Examples of Bad Commit Messages ---
# Bad: Unclear what was done
git commit -m "fix"
git commit -m "update"
git commit -m "WIP"
git commit -m "WIP"
git commit -m "misc changes"

# Bad: Commit too large
git commit -m "add auth, fix bugs, refactor UI, add tests"
# -> Should be split into multiple commits

# Bad: Only implementation details, no intent
git commit -m "add if statement"
git commit -m "rename variable x to userId"
```

### 3.3 Commit Granularity

```
Appropriate Commit Granularity:

  Principle: 1 commit = 1 logical change

  Good Granularity Examples:
  1. "feat: implement user registration API"
  2. "test: add tests for user registration API"
  3. "fix: correct email address validation"
  4. "refactor: convert UserService to repository pattern"

  Bad Granularity Examples:
  Too Fine-Grained:
  1. "define function signature"
  2. "implement function body"
  3. "add one test"
  4. "add another test"

  Too Coarse-Grained:
  1. "implement entire user management (auth, CRUD, API, tests, docs)"

  Techniques for Splitting Commits:

  1. Stage changes selectively
     git add -p            # Select by hunk
     git add src/auth.py   # Select by file

  2. Split after working
     git reset HEAD~1      # Undo last commit (changes preserved)
     git add -p            # Re-stage by hunk and commit

  3. Clean up with fixup/squash
     git commit --fixup HEAD~2   # Mark as fix for 2 commits ago
     git rebase -i HEAD~5        # Interactive rebase to reorganize
     # Place fixup after the corresponding pick to consolidate
```

---

## 4. Merge and Rebase

### 4.1 Types of Merges

```
Three Ways to Merge:

  1. Fast-Forward Merge
     Before:
     main    A <- B <- C
     feature           \ D <- E

     After (git merge feature):
     main    A <- B <- C <- D <- E

     -> Applied automatically when main is behind
     -> No merge commit is created
     -> Linear history

  2. 3-Way Merge (Standard Merge)
     Before:
     main    A <- B <- C <- F
     feature           \ D <- E

     After (git merge feature):
     main    A <- B <- C <- F <- M
                       \ D <- E /
     -> M is a merge commit (has two parents)
     -> Branch history is preserved

  3. Squash Merge
     Before:
     main    A <- B <- C
     feature           \ D <- E <- F

     After (git merge --squash feature && git commit):
     main    A <- B <- C <- S
     -> S combines D+E+F into a single commit
     -> Does not preserve the fine-grained history of the feature branch
     -> Commonly used for PR merges
```

### 4.2 How Rebase Works

```
Rebase:

  Before:
  main    A <- B <- C <- D
  feature       \ E <- F <- G

  git switch feature
  git rebase main

  After:
  main    A <- B <- C <- D
  feature                \ E' <- F' <- G'

  -> "Replays" E, F, G after D
  -> E', F', G' are new commits (hashes change)
  -> History becomes linear

  Advantages of Rebase:
  - Clean history (linear, no branches)
  - Easier to read git log
  - Easier to use bisect

  Rebase Caveats:
  - Never rebase published commits!
    (Other people's references to those commits will break)
  - Force push is required after rebasing a pushed branch
    git push --force-with-lease origin feature
    (--force-with-lease is a safer force push)
```

### 4.3 When to Use Merge vs Rebase

```
Choosing Between Merge and Rebase:

  Use Merge When:
  - Integrating into the main branch
  - Combining shared branches
  - You want to preserve history accurately
  - You want to record merges (when and what was integrated)

  Use Rebase When:
  - Keeping a personal feature branch up to date with main
  - Cleaning up commit history
  - Tidying up a branch before PR merge

  Recommended Workflow:
  1. Develop on a feature branch
  2. Incorporate main changes: git rebase main
  3. Review the PR
  4. Merge into main: squash merge (GitHub UI)

  Golden Rule:
  "Never rebase published history"
  -> Your own branch: rebase OK
  -> Shared branch: use merge
```

### 4.4 Interactive Rebase

```bash
# --- Interactive Rebase (cleaning up commit history) ---

# Reorganize the latest 5 commits
git rebase -i HEAD~5

# Editor opens:
# pick a1b2c3d feat: add user model
# pick d4e5f6g feat: implement user API
# pick 7a8b9c0 fix: typo fix
# pick 1d2e3f4 feat: add validation
# pick 5a6b7c8 fix: fix validation bug

# Edit as follows:
# pick a1b2c3d feat: add user model
# pick d4e5f6g feat: implement user API
# fixup 7a8b9c0 fix: typo fix                 <- merge into above commit
# pick 1d2e3f4 feat: add validation
# fixup 5a6b7c8 fix: fix validation bug       <- merge into above commit

# Rebase commands:
# pick   = use as is
# reword = change commit message
# edit   = edit commit contents
# squash = merge into above commit (combine messages)
# fixup  = merge into above commit (discard message)
# drop   = delete commit

# You can also reorder commits (just move lines around)
```

---

## 5. Conflict Resolution

### 5.1 Causes of Conflicts

```
Cases Where Conflicts Occur:

  1. Changes to the same line
     main:    line = "Hello"  -> line = "Hello, World!"
     feature: line = "Hello"  -> line = "Hi there!"

  2. Changes to adjacent lines (in some cases)
     One side deletes the line, the other edits it

  3. File deletion and modification
     main:    file.py deleted
     feature: file.py edited

  4. File rename
     main:    old.py -> new.py
     feature: old.py edited

  Cases Where Conflicts Do NOT Occur:
  - Changes to different files
  - Changes to different parts of the same file
  - One side adds a file (no conflict with existing files)
```

### 5.2 Conflict Resolution Procedure

```bash
# --- Conflict Resolution Procedure ---

# 1. Execute merge (or rebase)
git merge feature
# Auto-merging src/user.py
# CONFLICT (content): Merge conflict in src/user.py
# Automatic merge failed; fix conflicts and then commit the result.

# 2. Check the conflict status
git status
# Unmerged paths:
#   both modified:   src/user.py

# 3. Inspect conflict markers
# Conflict markers in the file:
# <<<<<<< HEAD
# name = "Hello, World!"
# =======
# name = "Hi there!"
# >>>>>>> feature

# 4. Resolve manually
# Remove markers and edit to the correct content:
# name = "Hello, World!"

# 5. Stage the resolution
git add src/user.py

# 6. Create the merge commit
git commit  # Message is auto-generated for merges

# --- Conflict Resolution During Rebase ---
git rebase main
# CONFLICT in src/user.py

# After resolving:
git add src/user.py
git rebase --continue  # Proceed to apply the next commit

# Abort the rebase and revert:
git rebase --abort
```

### 5.3 Tips to Minimize Conflicts

```
How to Minimize Conflicts:

  1. Keep PRs small
     - Fewer files changed
     - Fewer lines changed (target: under 200-400 lines)
     - Merge PRs promptly

  2. Incorporate main frequently
     git switch feature
     git rebase main  # Run daily

  3. Design file structure appropriately
     - Avoid cramming too much code into a single file
     - Be mindful of separation of concerns

  4. Team communication
     - Share plans when editing the same file
     - Leverage pair programming

  5. Standardize formatters
     - Share settings via .editorconfig
     - Enforce Prettier, Black, etc. in CI
     - Eliminate meaningless diffs from auto-formatting

  6. Lock file conflict mitigation
     # Specify merge strategy in .gitattributes
     package-lock.json merge=ours
     yarn.lock merge=ours
```

---

## 6. Practical Git Commands

### 6.1 Daily Commands

```bash
# --- Basic Operations ---
git status                  # Check status
git add -p                  # Interactive staging
git commit -m "msg"         # Commit
git push origin branch      # Push
git pull --rebase           # Pull (rebase merge)

# --- Branch Operations ---
git switch -c feature       # Create branch + switch
git switch main             # Switch to main
git branch -d feature       # Delete merged branch
git branch -D feature       # Force-delete unmerged branch

# --- Viewing History ---
git log --oneline --graph   # Graph display
git log --oneline -20       # Last 20 entries
git log --since="2024-01-01" --until="2024-01-31"  # Date range
git log --author="Alice"    # Filter by author
git log -p -- path/to/file  # File change history with diffs
git blame file.py           # Last author of each line

# --- Viewing Diffs ---
git diff                    # Working tree changes
git diff --staged           # Staged changes
git diff main..feature      # Diff between branches
git diff HEAD~3..HEAD       # Diff of last 3 commits
git diff --stat             # Change statistics
```

### 6.2 Useful Commands

```bash
# --- stash (temporary shelving) ---
git stash                   # Shelve changes
git stash push -m "WIP: login feature"  # Shelve with message
git stash list              # List shelved items
git stash pop               # Restore latest and remove
git stash apply stash@{1}   # Restore specific item (keep it)
git stash drop stash@{0}    # Delete specific item

# --- cherry-pick (apply a specific commit) ---
git cherry-pick abc1234     # Apply a specific commit to current branch
git cherry-pick abc1234..def5678  # Range specification
git cherry-pick --no-commit abc1234  # Apply changes without committing

# --- Retrieve a specific file from another branch ---
git checkout main -- path/to/file.py  # Get file from main
git restore --source main -- path/to/file.py  # Same (newer syntax)

# --- Amend commits ---
git commit --amend          # Amend last commit (change message)
git commit --amend --no-edit  # Add files without changing the message

# --- Tags ---
git tag v1.0.0              # Lightweight tag
git tag -a v1.0.0 -m "Release 1.0.0"  # Annotated tag
git push origin v1.0.0      # Push a tag
git push origin --tags       # Push all tags
```

### 6.3 Troubleshooting

```bash
# --- reflog (HEAD movement history) ---
git reflog
# abc1234 HEAD@{0}: commit: feat: add new feature
# def5678 HEAD@{1}: rebase finished: ...
# 7a8b9c0 HEAD@{2}: rebase: starting

# Recovery using reflog
git reset --hard HEAD@{2}   # Revert to the state 2 operations ago

# --- bisect (find the commit that introduced a bug) ---
git bisect start
git bisect bad              # Current commit has the bug
git bisect good v1.0.0      # v1.0.0 does not have the bug
# -> Git checks out a midpoint commit
# Test and judge good/bad
git bisect good  # or  git bisect bad
# -> Binary search narrows it down
git bisect reset            # Done, return to original branch

# Automation (with a test script)
git bisect start HEAD v1.0.0
git bisect run python -m pytest tests/test_bug.py

# --- Undo erroneous commits ---
# Method 1: revert (undo with a new commit) = safe
git revert abc1234
git revert HEAD~3..HEAD     # Revert last 3 commits

# Method 2: reset (rewind history) = use with caution
git reset --soft HEAD~1     # Undo commit (changes remain staged)
git reset --mixed HEAD~1    # Undo commit (changes remain in working tree)
git reset --hard HEAD~1     # Undo commit (all changes are deleted!)

# --- Recover a deleted branch ---
git reflog | grep "feature/important"
# abc1234 HEAD@{5}: checkout: moving from feature/important to main
git branch feature/important abc1234  # Restore the branch

# --- Fix accidentally committed large files ---
# Completely remove from history (caution: rewrites history)
git filter-branch --force --tree-filter \
  'rm -f path/to/large-file.zip' HEAD

# Newer tool: git-filter-repo
pip install git-filter-repo
git filter-repo --path path/to/large-file.zip --invert-paths
```

### 6.4 Alias Configuration

```bash
# --- Useful Git Aliases ---
git config --global alias.st "status"
git config --global alias.co "checkout"
git config --global alias.sw "switch"
git config --global alias.br "branch"
git config --global alias.ci "commit"
git config --global alias.lg "log --oneline --graph --decorate --all"
git config --global alias.last "log -1 HEAD --format='%H %s'"
git config --global alias.unstage "restore --staged"
git config --global alias.undo "reset --soft HEAD~1"
git config --global alias.amend "commit --amend --no-edit"
git config --global alias.wip "commit -am 'WIP'"
git config --global alias.branches "branch -a -v"
git config --global alias.tags "tag -l -n1"
git config --global alias.stashes "stash list"
git config --global alias.cleanup "!git branch --merged | grep -v '\\*\\|main\\|master\\|develop' | xargs -n 1 git branch -d"

# Usage examples
git st           # git status
git lg           # Pretty graph display
git undo         # Undo last commit
git cleanup      # Batch-delete merged branches
```

---

## 7. Git Hooks

### 7.1 Types of Git Hooks

```
Types of Git Hooks:

  Client-Side Hooks (Local):
  +--------------------------------------------------+
  | Hook Name            | Timing                     |
  +--------------------------------------------------+
  | pre-commit           | Before commit              |
  | prepare-commit-msg   | Before commit message edit |
  | commit-msg           | After commit message final |
  | post-commit          | After commit               |
  | pre-push             | Before push                |
  | pre-rebase           | Before rebase              |
  | post-checkout        | After checkout             |
  | post-merge           | After merge                |
  +--------------------------------------------------+

  Server-Side Hooks:
  +--------------------------------------------------+
  | Hook Name            | Timing                     |
  +--------------------------------------------------+
  | pre-receive          | Before push is received    |
  | update               | Before each branch update  |
  | post-receive         | After push is received     |
  +--------------------------------------------------+
```

### 7.2 Practical Git Hook Examples

```bash
#!/bin/bash
# .git/hooks/pre-commit (or managed via husky)

# --- Lint Check ---
echo "Running lint..."
npm run lint
if [ $? -ne 0 ]; then
    echo "Lint errors found. Please fix before committing."
    exit 1
fi

# --- Run Tests ---
echo "Running tests..."
npm run test -- --bail
if [ $? -ne 0 ]; then
    echo "Tests failed. Please fix before committing."
    exit 1
fi

# --- Detect Secrets ---
echo "Checking for secrets..."
PATTERNS="(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]"
if git diff --cached --name-only | xargs grep -lE "$PATTERNS" 2>/dev/null; then
    echo "Potential secrets detected. Please review."
    exit 1
fi

# --- Detect Large Files ---
MAX_SIZE=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ "$size" -gt "$MAX_SIZE" ]; then
            echo "File $file exceeds 5MB. Use Git LFS instead."
            exit 1
        fi
    fi
done

echo "All checks passed."
exit 0
```

```bash
#!/bin/bash
# .git/hooks/commit-msg
# Enforce Conventional Commits format

commit_msg=$(cat "$1")
pattern="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?: .{1,72}"

if ! echo "$commit_msg" | head -1 | grep -qE "$pattern"; then
    echo "Invalid commit message format."
    echo ""
    echo "Expected format: <type>(<scope>): <description>"
    echo "  type: feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
    echo "  scope: optional, e.g., (auth), (api)"
    echo "  description: max 72 characters"
    echo ""
    echo "Example: feat(auth): implement login authentication"
    exit 1
fi

exit 0
```

### 7.3 Automation with husky + lint-staged

```json
// package.json
{
  "scripts": {
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss}": [
      "stylelint --fix",
      "prettier --write"
    ],
    "*.{json,md}": [
      "prettier --write"
    ]
  }
}
```

```bash
# husky setup
npm install --save-dev husky lint-staged
npx husky install

# pre-commit hook
npx husky add .husky/pre-commit "npx lint-staged"

# commit-msg hook (Conventional Commits check)
npm install --save-dev @commitlint/cli @commitlint/config-conventional
npx husky add .husky/commit-msg 'npx commitlint --edit "$1"'
```

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2, 'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf',
       'test', 'build', 'ci', 'chore', 'revert']
    ],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
  },
};
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions Basics

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    needs: lint
    strategy:
      matrix:
        node-version: [18, 20, 22]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run test -- --coverage
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.node-version }}
          path: coverage/

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
      - run: echo "Deploying to production..."
      # Actual deployment commands
```

### 8.2 Pull Request Automation

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  # PR size check
  pr-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check PR size
        run: |
          LINES_CHANGED=$(git diff --stat origin/${{ github.base_ref }}...HEAD | tail -1 | awk '{print $4+$6}')
          echo "Lines changed: $LINES_CHANGED"
          if [ "$LINES_CHANGED" -gt 1000 ]; then
            echo "::warning::PR is too large ($LINES_CHANGED lines). Consider splitting."
          fi

  # Auto-assign reviewers
  auto-assign:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v2
        with:
          configuration-path: '.github/auto-assign.yml'

  # Auto-labeling
  labeler:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

### 8.3 Semantic Versioning and Releases

```
Semantic Versioning (SemVer):
  MAJOR.MINOR.PATCH

  MAJOR: Incompatible changes
  MINOR: Backward-compatible new features
  PATCH: Backward-compatible bug fixes

  Examples:
  1.0.0 -> 1.0.1  Patch (bug fix)
  1.0.0 -> 1.1.0  Minor (new feature)
  1.0.0 -> 2.0.0  Major (breaking change)

  Pre-release:
  2.0.0-alpha.1
  2.0.0-beta.1
  2.0.0-rc.1 (Release Candidate)

  Mapping to Conventional Commits:
  feat: -> MINOR version bump
  fix:  -> PATCH version bump
  feat!: or BREAKING CHANGE: -> MAJOR version bump
```

```bash
# Automated releases with semantic-release
# Automatically determines version from Conventional Commits

# Install
npm install --save-dev semantic-release @semantic-release/git @semantic-release/changelog

# .releaserc.json
# {
#   "branches": ["main"],
#   "plugins": [
#     "@semantic-release/commit-analyzer",
#     "@semantic-release/release-notes-generator",
#     "@semantic-release/changelog",
#     "@semantic-release/npm",
#     "@semantic-release/github",
#     ["@semantic-release/git", {
#       "assets": ["CHANGELOG.md", "package.json"],
#       "message": "chore(release): ${nextRelease.version}"
#     }]
#   ]
# }
```

---

## 9. Managing Large-Scale Repositories

### 9.1 Monorepo vs Multirepo

```
Monorepo:
  All projects in a single repository

  +-------------------------------------+
  | monorepo/                           |
  | +-- packages/                       |
  | |   +-- web/        (frontend)      |
  | |   +-- api/        (backend)       |
  | |   +-- shared/     (shared libs)   |
  | |   +-- mobile/     (mobile app)    |
  | +-- tools/                          |
  | +-- package.json                    |
  | +-- turbo.json (or nx.json)         |
  +-------------------------------------+

  Advantages:
  - Easy code sharing
  - Atomic changes (modify API and frontend simultaneously)
  - Unified CI/CD
  - Centralized dependency management

  Disadvantages:
  - Repository grows large
  - Build time increases
  - Access control is difficult
  - Git operations may slow down

  Representative Tools: Turborepo, Nx, Lerna, Bazel

Multirepo:
  Separate repositories per project

  Advantages:
  - Lightweight repositories
  - Independent CI/CD
  - Fine-grained access control
  - Team autonomy

  Disadvantages:
  - Code sharing is difficult
  - Cross-cutting changes are cumbersome
  - Dependency management is complex
  - Version inconsistencies
```

### 9.2 Managing Large Files

```bash
# --- Git LFS (Large File Storage) ---

# Install
git lfs install

# Specify file patterns to track
git lfs track "*.psd"
git lfs track "*.zip"
git lfs track "*.mp4"
git lfs track "assets/images/*.png"

# .gitattributes is auto-generated
# *.psd filter=lfs diff=lfs merge=lfs -text
# *.zip filter=lfs diff=lfs merge=lfs -text

# Use add/commit/push as usual
git add .gitattributes
git add large-file.psd
git commit -m "feat: add design file"
git push origin main

# Check LFS status
git lfs ls-files     # List LFS-managed files
git lfs status       # LFS status
```

### 9.3 .gitignore Best Practices

```bash
# --- Recommended .gitignore Settings ---

# --- OS ---
.DS_Store
Thumbs.db
*.swp
*~

# --- Editor/IDE ---
.vscode/settings.json
.idea/
*.sublime-workspace
*.code-workspace

# --- Language/Framework ---
# Node.js
node_modules/
npm-debug.log*
yarn-error.log*
.pnpm-debug.log*

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# Java
*.class
target/
*.jar
*.war

# Go
/vendor/

# --- Build Artifacts ---
dist/
build/
out/
.next/
.nuxt/

# --- Environment Variables / Secrets ---
.env
.env.local
.env.production
*.pem
*.key
credentials.json

# --- Tests/Coverage ---
coverage/
.nyc_output/
*.lcov
htmlcov/

# --- Logs ---
*.log
logs/

# --- Databases ---
*.sqlite3
*.db

# Global .gitignore (shared across all repositories)
git config --global core.excludesfile ~/.gitignore_global
```

---

## 10. Security

### 10.1 Managing Sensitive Information

```
Managing Sensitive Information in Git Repositories:

  Items That Must NEVER Be Committed:
  - Passwords, API keys
  - Private keys (.pem, .key)
  - .env files (production values)
  - credentials.json
  - Database connection strings

  Countermeasures:
  1. Add to .gitignore
  2. Commit .env.example (with empty or dummy values)
  3. Pre-detect with git-secrets
  4. If accidentally committed, immediately rotate the keys

  Setting Up git-secrets:
```

```bash
# Installing and configuring git-secrets
brew install git-secrets  # macOS

# Configure in a repository
cd my-project
git secrets --install
git secrets --register-aws  # Register AWS key patterns

# Add custom patterns
git secrets --add 'password\s*=\s*["\047][^"\047]+["\047]'
git secrets --add 'PRIVATE[_-]?KEY'

# Automatic checks before commit (added to pre-commit hook)
git secrets --scan         # Scan entire repository
git secrets --scan-history # Scan entire history
```

### 10.2 GPG Signing

```bash
# --- Commit Signing (GPG) ---

# Generate GPG key
gpg --full-generate-key

# Check key ID
gpg --list-secret-keys --keyid-format=long
# sec   rsa4096/ABC1234567890DEF 2024-01-01 [SC]

# Configure in Git
git config --global user.signingkey ABC1234567890DEF
git config --global commit.gpgsign true  # Sign all commits

# Signed commit
git commit -S -m "feat: signed commit"

# Verify signatures
git log --show-signature
# gpg: Good signature from "Alice <alice@example.com>"

# Register public key on GitHub
gpg --armor --export ABC1234567890DEF
# -> Add to GitHub Settings -> SSH and GPG keys
```

---

## 11. Advanced Git Techniques

### 11.1 worktree (Multiple Working Directories)

```bash
# --- git worktree ---
# Check out multiple branches simultaneously from one repository

# Create a new worktree
git worktree add ../project-hotfix hotfix/critical-bug
git worktree add ../project-review feature/new-ui

# List worktrees
git worktree list
# /path/to/project          abc1234 [main]
# /path/to/project-hotfix   def5678 [hotfix/critical-bug]
# /path/to/project-review   7a8b9c0 [feature/new-ui]

# Remove a worktree
git worktree remove ../project-hotfix

# Use cases:
# - Continue development while running a long build in another directory
# - Check out another branch for code review
# - Handle an urgent hotfix without interrupting current work
```

### 11.2 Submodules

```bash
# --- git submodule ---
# Include an external repository as a subdirectory

# Add a submodule
git submodule add https://github.com/example/library.git libs/library
git commit -m "feat: add library as submodule"

# Clone a repository with submodules
git clone --recurse-submodules https://github.com/example/project.git
# Or
git clone https://github.com/example/project.git
cd project
git submodule init
git submodule update

# Update submodules
git submodule update --remote  # Fetch latest from remote
cd libs/library
git checkout v2.0.0            # Pin to a specific version
cd ../..
git add libs/library
git commit -m "chore: update library to v2.0.0"

# Submodule caveats:
# - Adds complexity (--recurse-submodules required for cloning)
# - CI configuration becomes more complex
# - Alternative: consider package managers like npm/pip
```

### 11.3 Performance Optimization

```bash
# --- Performance Improvements for Large Repositories ---

# Shallow clone (limit history)
git clone --depth 1 https://github.com/large/repo.git
git clone --depth 100 https://github.com/large/repo.git
git clone --shallow-since="2024-01-01" https://github.com/large/repo.git

# Partial clone (lazy fetch blobs)
git clone --filter=blob:none https://github.com/large/repo.git

# Sparse checkout (specific directories only)
git clone --no-checkout https://github.com/large/repo.git
cd repo
git sparse-checkout init --cone
git sparse-checkout set packages/web packages/shared
git checkout main
# -> Only packages/web and packages/shared are checked out

# Accelerate with fsmonitor (file monitoring daemon)
git config core.fsmonitor true
git config core.untrackedcache true

# gc (garbage collection)
git gc --aggressive  # Optimize the repository
git prune            # Remove unreachable objects

# Performance inspection
git count-objects -vH  # Object count and size
```

---

## 12. Team Development Best Practices

### 12.1 CODEOWNERS

```bash
# --- .github/CODEOWNERS ---
# Auto-assign reviewers per file/directory

# Default owner
* @team-lead

# Frontend
/src/components/ @frontend-team
/src/pages/ @frontend-team
*.tsx @frontend-team
*.css @frontend-team

# Backend
/src/api/ @backend-team
/src/models/ @backend-team
/src/services/ @backend-team

# Infrastructure
/terraform/ @infra-team
/docker/ @infra-team
/.github/workflows/ @infra-team
Dockerfile @infra-team

# Documentation
/docs/ @tech-writers
*.md @tech-writers

# Security-related
/src/auth/ @security-team
/src/middleware/auth* @security-team
```

### 12.2 Pull Request Template

```markdown
<!-- .github/pull_request_template.md -->

## Overview
<!-- What did you change in this PR? -->

## Type of Change
- [ ] New feature (feat)
- [ ] Bug fix (fix)
- [ ] Refactoring (refactor)
- [ ] Documentation (docs)
- [ ] Tests (test)
- [ ] Other

## Changes
<!-- Describe the changes in detail -->

## How to Test
<!-- Describe the testing procedure -->

## Screenshots
<!-- Attach screenshots if there are UI changes -->

## Checklist
- [ ] Added/updated tests
- [ ] Updated documentation
- [ ] No breaking changes
- [ ] Confirmed performance impact
- [ ] Confirmed security impact

## Related Issue
Closes #
```

### 12.3 Branch Protection Rules

```
Recommended Branch Protection Settings (main branch):

  +--------------------------------------------------+
  | Setting                      | Recommended        |
  +--------------------------------------------------+
  | Require pull request         | ON                 |
  | Required reviewers           | 1-2 people         |
  | Dismiss stale reviews        | ON                 |
  | Require status checks        | ON                 |
  | Require branches up to date  | ON                 |
  | Require conversation         | ON                 |
  | resolution                   |                    |
  | Require signed commits       | As needed          |
  | Include administrators       | ON                 |
  | Allow force pushes           | OFF                |
  | Allow deletions              | OFF                |
  +--------------------------------------------------+

  Required Status Checks:
  - CI (lint, test, build)
  - Security scan
  - Code coverage threshold
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Also create test code

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate the input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Slow version: {slow_time:.4f} sec")
    print(f"Fast version: {fast_time:.6f} sec")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be aware of algorithmic time complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|---------|
| Initialization error | Incorrect configuration file | Verify configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increasing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check the error message**: Read the stack trace and identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Incremental verification**: Verify hypotheses using log output or a debugger
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input/output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O wait**: Check disk and network I/O conditions
4. **Check concurrent connections**: Check connection pool status

| Problem Type | Diagnostic Tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is paramount. Understanding deepens not just through theory, but by actually writing code and verifying how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping ahead to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in professional practice?

Knowledge of this topic is frequently applied in daily development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Git Internals | DAG + content-addressable. Tamper-proof |
| Branches | Just pointers. Lightweight to create/delete |
| Strategy | GitHub Flow (simple), Trunk-Based (modern) |
| Commits | Conventional Commits, 1 commit = 1 logical change |
| Merge vs Rebase | Shared branches: merge, personal branches: rebase |
| Conflicts | Minimize with small PRs + frequent rebases |
| Hooks | pre-commit for lint, test, and secret detection |
| CI/CD | Automate with GitHub Actions |
| Security | Secrets go in .gitignore + git-secrets |
| Essential Skills | rebase, stash, bisect, reflog |

---

## Recommended Next Guides

---

## References
1. Chacon, S. & Straub, B. "Pro Git." 2nd Edition, Apress, 2014.
2. Driessen, V. "A Successful Git Branching Model." 2010.
3. Conventional Commits Specification. https://www.conventionalcommits.org/
4. GitHub Flow Guide. https://docs.github.com/en/get-started/quickstart/github-flow
5. Atlassian Git Tutorials. https://www.atlassian.com/git/tutorials
6. Trunk Based Development. https://trunkbaseddevelopment.com/
7. Semantic Versioning. https://semver.org/
