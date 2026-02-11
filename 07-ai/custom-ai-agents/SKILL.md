# カスタム AI エージェント

> AI エージェントは LLM に自律的な行動能力を与える。エージェントアーキテクチャ、ツール統合、マルチエージェント、MCP プロトコル、プロダクションデプロイまで、AI エージェント開発の全てを解説する。

## このSkillの対象者

- AI エージェントを設計・実装したいエンジニア
- LLM アプリケーションを高度化したい方
- マルチエージェントシステムに興味がある方

## 前提知識

- LLM API の使用経験
- TypeScript or Python の実務経験
- Web 開発の基礎

## 学習ガイド

### 00-fundamentals — エージェントの基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-agent-overview.md]] | AI エージェントの定義、歴史、現在のアーキテクチャ |
| 01 | [[docs/00-fundamentals/01-reasoning-patterns.md]] | ReAct、CoT、Tree of Thought、Plan-and-Execute |
| 02 | [[docs/00-fundamentals/02-memory-systems.md]] | 短期/長期メモリ、ベクトル DB、会話履歴管理 |
| 03 | [[docs/00-fundamentals/03-tool-use.md]] | Tool Use、Function Calling、ツール設計原則 |

### 01-architecture — アーキテクチャ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-architecture/00-single-agent.md]] | シングルエージェント設計、ループ構造、エラー処理 |
| 01 | [[docs/01-architecture/01-multi-agent.md]] | マルチエージェント、オーケストレーション、協調パターン |
| 02 | [[docs/01-architecture/02-workflow-agents.md]] | ワークフローエージェント、条件分岐、並列実行 |
| 03 | [[docs/01-architecture/03-human-in-the-loop.md]] | Human-in-the-Loop、承認フロー、フィードバックループ |

### 02-tools — ツールとフレームワーク

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-tools/00-langchain.md]] | LangChain（Agent/Chain/Tool/Memory）、LangGraph |
| 01 | [[docs/02-tools/01-anthropic-sdk.md]] | Anthropic SDK、Claude Agent SDK、Tool Use 実装 |
| 02 | [[docs/02-tools/02-mcp-protocol.md]] | MCP（Model Context Protocol）、サーバー/クライアント |
| 03 | [[docs/02-tools/03-other-frameworks.md]] | CrewAI、AutoGen、Semantic Kernel、Vercel AI SDK |

### 03-advanced — 高度なトピック

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-advanced/00-rag-agents.md]] | RAG エージェント、動的検索、自己反省型 RAG |
| 01 | [[docs/03-advanced/01-code-agents.md]] | コード生成エージェント、サンドボックス、テスト実行 |
| 02 | [[docs/03-advanced/02-browser-agents.md]] | ブラウザ操作エージェント、Playwright、スクレイピング |
| 03 | [[docs/03-advanced/03-safety-and-guardrails.md]] | エージェント安全性、権限制御、コスト制限、監査 |

### 04-deployment — デプロイと運用

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-deployment/00-production-architecture.md]] | プロダクションアーキテクチャ、スケーリング、キュー |
| 01 | [[docs/04-deployment/01-evaluation.md]] | エージェント評価、テスト、ベンチマーク、A/B テスト |
| 02 | [[docs/04-deployment/02-monitoring.md]] | エージェント監視、コスト追跡、LangSmith、Helicone |

## クイックリファレンス

```
エージェントパターン選定:
  単純タスク → シングルエージェント + Tool Use
  複雑タスク → Plan-and-Execute
  専門分野混合 → マルチエージェント
  人間の承認必要 → Human-in-the-Loop
  外部ツール多数 → MCP 統合

フレームワーク選定:
  軽量・柔軟 → Anthropic SDK 直接
  フルスタック → LangChain / LangGraph
  マルチエージェント → CrewAI / AutoGen
  TypeScript → Vercel AI SDK
```

## 参考文献

1. Anthropic. "Tool Use Documentation." docs.anthropic.com, 2024.
2. LangChain. "Documentation." langchain.com/docs, 2024.
3. Anthropic. "MCP Specification." modelcontextprotocol.io, 2024.
