[日本語版](../../ja/05-infrastructure/version-control-and-jujutsu/SKILL.md)

# Version Control and Jujutsu

> Git is the foundation of modern development, yet few truly understand its internals. This guide dives deep into Git's internal object model, advanced operations, and the next-generation VCS Jujutsu (jj) -- exploring the depths of version control.

## Target Audience

- Engineers who want to understand Git's internal architecture
- Developers looking to master advanced Git operations (rebase, bisect, reflog, etc.)
- Those interested in the next-generation VCS, Jujutsu

## Prerequisites

- Basic Git operations (add, commit, push, pull, branch)
- Basic terminal operations

## Study Guide

### 00-git-internals — Git Internals

| # | File | Description |
|---|------|-------------|

### 01-advanced-git — Advanced Git Operations

| # | File | Description |
|---|------|-------------|

### 02-jujutsu — Jujutsu (jj)

| # | File | Description |
|---|------|-------------|

## Quick Reference

```
Git Internal Objects:
  blob   — File contents
  tree   — Directory structure
  commit — Snapshot + metadata
  tag    — Named reference

Advanced Git Commands:
  git rebase -i HEAD~5        — Interactively edit the last 5 commits
  git bisect start/bad/good   — Identify the commit that introduced a bug
  git reflog                  — View HEAD movement history
  git worktree add ../feature — Work on a branch in a separate directory
  git log -S "keyword"        — Search for code changes

Jujutsu (jj) Basics:
  jj init                     — Initialize a repository
  jj status                   — Check status
  jj describe -m "message"    — Set commit message
  jj new                      — Create a new change set
  jj squash                   — Squash into parent
  jj git push                 — Push to Git remote
```

## References

1. Chacon, S. & Straub, B. "Pro Git." git-scm.com/book, 2024.
2. Git. "Git Internals." git-scm.com/book/en/v2/Git-Internals, 2024.
3. Jujutsu. "Documentation." martinvonz.github.io/jj, 2024.
