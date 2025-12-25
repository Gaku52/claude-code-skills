# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 2: Implement Sub Agents (code-reviewer, test-runner, etc.)
- Add Agent orchestration guide
- Create validation scripts

## [1.0.1] - 2025-12-25

### Security
- Updated `.gitignore` for Phase 2 preparation
  - Added Node.js / TypeScript exclusions (node_modules, dist, build, etc.)
  - Added environment variable exclusions (.env, .env.local, etc.)
  - Added test coverage exclusions

### Added
- `.env.example` template file for Phase 2 environment setup
  - GitHub Personal Access Token configuration
  - Debug and logging options
  - Custom skills directory path option

### Changed
- Updated README.md version to 1.0.1
- Updated STATUS.md with security enhancement details
- Updated all documentation dates to 2025-12-25

### Notes
- Preparing for Phase 2: Sub Agents implementation
- All 26 Skills remain complete and stable
- Enhanced security posture for upcoming TypeScript development

## [1.0.0] - 2024-12-24

### Added
- **All 26 Skills completed** 🎉
  - Web開発系（6個）: documentation, web-development, react-development, nextjs-development, frontend-performance, web-accessibility
  - バックエンド開発系（4個）: backend-development, nodejs-development, python-development, database-design
  - スクリプト・自動化系（3個）: script-development, cli-development, mcp-development
  - iOS開発系（5個）: ios-development, ios-project-setup, swiftui-patterns, networking-data, ios-security
  - 品質・テスト系（3個）: testing-strategy, code-review, quality-assurance
  - DevOps・CI/CD系（3個）: git-workflow, ci-cd-automation, dependency-management
  - ナレッジ管理系（2個）: incident-logger, lessons-learned
- Phase 2 design documentation
  - PHASE2_DESIGN.md - Detailed technical design
  - QUICKSTART.md - Step-by-step implementation guide
  - ROADMAP.md - Phase 1-5 overall roadmap
  - MONOREPO_STRUCTURE.md - Repository structure design

### Notes
- Phase 1 complete: Full-Stack Development Skills Framework
- Phase 2 ready: Sub Agents implementation can begin
- Production-ready release with comprehensive skill coverage

## [0.1.0] - 2024-12-24

### Added
- Initial repository structure
- 25 Skills skeleton (directory structure)
- Complete `git-workflow` Skill as template
  - SKILL.md with comprehensive guide
  - README.md with usage instructions
  - Detailed commit message guide (guides/05-commit-messages.md)
  - Pre-commit checklist
  - Pull request template
- Project documentation
  - README.md with progress tracker
  - CONTRIBUTING.md with contribution guidelines
  - This CHANGELOG.md

### Structure Created
- Skills categories:
  - Product Planning & Design (5 skills)
  - iOS Development (6 skills)
  - Quality & Testing (4 skills)
  - DevOps & CI/CD (4 skills)
  - Release & Operations (3 skills)
  - Knowledge Management (3 skills)

### Notes
- This is the foundational release
- Only `git-workflow` is fully documented
- Other 24 skills have directory structure only
- Ready for iterative development

---

## Version History

- **0.1.0** (2024-12-24) - Initial release with git-workflow template
- **0.2.0** (TBD) - Core skills completion
- **0.5.0** (TBD) - Beta release with all basic skills
- **1.0.0** (TBD) - Production-ready release
