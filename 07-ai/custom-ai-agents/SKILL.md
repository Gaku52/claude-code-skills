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

### 01-architecture — アーキテクチャ

| # | ファイル | 内容 |
|---|---------|------|

### 02-tools — ツールとフレームワーク

| # | ファイル | 内容 |
|---|---------|------|

### 03-advanced — 高度なトピック

| # | ファイル | 内容 |
|---|---------|------|

### 04-deployment — デプロイと運用

| # | ファイル | 内容 |
|---|---------|------|

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
