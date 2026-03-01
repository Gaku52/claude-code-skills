# Claude Code Skills

<!-- PROGRESS_BADGES_START -->
![Skills](https://img.shields.io/badge/Skills-36-blue)
![Guides](https://img.shields.io/badge/Guides-901-success)
![Characters](https://img.shields.io/badge/Characters-29053K-informational)
<!-- PROGRESS_BADGES_END -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ソフトウェア開発の全領域をカバーする体系的な知識ベース。
CS基礎からAI活用まで、36 Skills / 901ガイドファイル / 約1,200万字。

> **Status:** Phase 1 完了（コンテンツ生成済み）→ レビュー待ち

## 概要

| 指標 | 値 |
|------|-----|
| Skills数 | 36 |
| ガイドファイル数 | 901 |
| 総文字数 | 約1,200万字 |
| カテゴリ数 | 8 |

## カテゴリ一覧

### 01-cs-fundamentals — CS基礎 (4 Skills / 131 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| computer-science-fundamentals | 55 | ハードウェア、データ表現、計算理論、プログラミングパラダイム |
| algorithm-and-data-structures | 24 | アルゴリズム設計、データ構造、計算量解析 |
| operating-system-guide | 20 | プロセス管理、メモリ、ファイルシステム、カーネル |
| programming-language-fundamentals | 32 | 型システム、コンパイラ、言語設計、メモリモデル |

### 02-programming — プログラミング言語・技法 (6 Skills / 118 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| object-oriented-programming | 20 | SOLID原則、デザインパターン、継承vs合成 |
| async-and-error-handling | 18 | 非同期処理、Promise、エラーハンドリング戦略 |
| typescript-complete-guide | 25 | 型システム、ジェネリクス、ユーティリティ型 |
| go-practical-guide | 18 | Goroutine、チャネル、Go実践パターン |
| rust-systems-programming | 25 | 所有権、ライフタイム、unsafe、並行処理 |
| regex-and-text-processing | 12 | 正規表現、テキスト処理、パーサー |

### 03-software-design — ソフトウェア設計・品質 (3 Skills / 58 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| clean-code-principles | 20 | 命名規則、関数設計、リファクタリング |
| design-patterns-guide | 20 | GoFパターン、アーキテクチャパターン |
| system-design-guide | 18 | スケーラビリティ、分散システム、設計面接 |

### 04-web-and-network — Web・ネットワーク (4 Skills / 75 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| network-fundamentals | 20 | TCP/IP、DNS、HTTP、TLS |
| browser-and-web-platform | 18 | レンダリング、DOM、Web API |
| web-application-development | 20 | フルスタック開発、SPA/SSR、状態管理 |
| api-and-library-guide | 17 | REST/GraphQL設計、ライブラリ選定 |

### 05-infrastructure — インフラ・DevOps (7 Skills / 130 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| linux-cli-mastery | 22 | シェル、コマンド、システム管理 |
| docker-container-guide | 22 | コンテナ、Docker Compose、マルチステージビルド |
| aws-cloud-guide | 29 | EC2、Lambda、S3、IAM、VPC |
| devops-and-github-actions | 17 | CI/CD、GitHub Actions、自動化 |
| development-environment-setup | 14 | エディタ、ツールチェーン、dotfiles |
| windows-application-development | 14 | WPF、WinUI、Win32 API |
| version-control-and-jujutsu | 12 | Git高度操作、Jujutsu |

### 06-data-and-security — データ・セキュリティ (3 Skills / 63 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| sql-and-query-mastery | 19 | SQL最適化、インデックス、トランザクション |
| security-fundamentals | 25 | 暗号化、OWASP、脆弱性対策 |
| authentication-and-authorization | 19 | OAuth2、JWT、RBAC |

### 07-ai — AI・LLM (8 Skills / 125 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| llm-and-ai-comparison | 20 | LLMモデル比較、ベンチマーク |
| ai-analysis-guide | 16 | AIによるデータ分析、プロンプト設計 |
| ai-audio-generation | 14 | AI音声生成、音楽生成 |
| ai-visual-generation | 14 | AI画像・動画生成 |
| ai-automation-and-monetization | 15 | AI自動化、収益化戦略 |
| ai-era-development-workflow | 15 | AI時代の開発ワークフロー |
| ai-era-gadgets | 12 | AIガジェット、ハードウェア |
| custom-ai-agents | 19 | AIエージェント設計・実装 |

### 08-hobby — 趣味 (1 Skill / 201 files)

| Skill | ファイル数 | 内容 |
|-------|-----------|------|
| dj-skills-guide | 201 | DJテクニック、Rekordbox、Ableton Live、楽曲制作 |

## ディレクトリ構成

```
skills/
├── 01-cs-fundamentals/          # CS基礎 (4)
├── 02-programming/              # プログラミング (6)
├── 03-software-design/          # 設計・品質 (3)
├── 04-web-and-network/          # Web・ネットワーク (4)
├── 05-infrastructure/           # インフラ・DevOps (7)
├── 06-data-and-security/        # データ・セキュリティ (3)
├── 07-ai/                       # AI・LLM (8)
├── 08-hobby/                    # 趣味 (1)
├── _original-skills/            # Phase 1以前の26 Skills（アーカイブ）
├── _legacy/                     # レガシーディレクトリ
├── _meta/                       # プロジェクト管理（SESSION_ARCHIVE等）
└── README.md
```

### Skill内部構造

```
skill-name/
├── SKILL.md       # 概要・目次
├── docs/          # ガイドファイル群（メインコンテンツ）
└── README.md      # 使い方
```

## 使い方

### Claude Codeでの使用

```bash
git clone https://github.com/Gaku52/claude-code-skills.git ~/.claude/skills
```

Claude Codeが `~/.claude/skills/` を自動参照。開発時にSkillの知識が適用される。

### 手動参照

```bash
# Skill概要を見る
cat ~/.claude/skills/02-programming/typescript-complete-guide/SKILL.md

# 特定ガイドを読む
cat ~/.claude/skills/05-infrastructure/docker-container-guide/docs/multi-stage-build.md
```

## 今後の予定

1. **レビュー**: 901ファイルの品質チェック（正確性、網羅性、日本語品質）
2. **Phase 2**: レビュー結果に基づく改善・追加

## License

MIT License - See [LICENSE](LICENSE)

---

**最終更新**: 2026-02-12
**バージョン**: 2.0.0 (Phase 1 Complete — 36 Skills / 901 files)
