# LLM と AI モデル比較

> LLM の世界は急速に進化する。Claude、GPT、Gemini、Llama の特徴比較、プロンプトエンジニアリング、RAG、ファインチューニング、評価手法まで、LLM 活用の全てを体系的に解説する。

## このSkillの対象者

- LLM を活用したアプリケーションを開発するエンジニア
- AI モデルの選定・比較を行う方
- プロンプトエンジニアリングを極めたい方

## 前提知識

- AI/ML の基礎概念
- Web API の基礎知識
- Python or TypeScript の基礎

## 学習ガイド

### 00-fundamentals — LLM の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-llm-overview.md]] | LLM の仕組み、Transformer、スケーリング則 |
| 01 | [[docs/00-fundamentals/01-tokenization.md]] | トークナイザー、BPE、SentencePiece、コンテキスト長 |
| 02 | [[docs/00-fundamentals/02-training-and-alignment.md]] | 事前学習、SFT、RLHF、DPO、Constitutional AI |
| 03 | [[docs/00-fundamentals/03-inference-optimization.md]] | 量子化、KV キャッシュ、推論最適化、vLLM |

### 01-models — モデル比較

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-models/00-claude-family.md]] | Claude Opus/Sonnet/Haiku、特徴、API、ベストプラクティス |
| 01 | [[docs/01-models/01-gpt-family.md]] | GPT-4o/GPT-4o-mini、ChatGPT、Function Calling |
| 02 | [[docs/01-models/02-gemini-and-others.md]] | Gemini、Llama、Mistral、Qwen、Command R |
| 03 | [[docs/01-models/03-model-selection.md]] | モデル選定基準、ベンチマーク、コスト比較、用途別推奨 |

### 02-techniques — 活用技術

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-techniques/00-prompt-engineering.md]] | プロンプト設計、Few-shot、CoT、System Prompt |
| 01 | [[docs/02-techniques/01-rag.md]] | RAG アーキテクチャ、ベクトル DB、チャンキング、リランキング |
| 02 | [[docs/02-techniques/02-function-calling.md]] | Tool Use/Function Calling、構造化出力、JSON モード |
| 03 | [[docs/02-techniques/03-fine-tuning.md]] | ファインチューニング、LoRA、QLoRA、データ準備 |
| 04 | [[docs/02-techniques/04-multimodal.md]] | マルチモーダル（画像/音声/動画入力）、Vision API |

### 03-applications — アプリケーション

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-applications/00-chatbot-development.md]] | チャットボット開発、会話管理、メモリ、ストリーミング |
| 01 | [[docs/03-applications/01-ai-search.md]] | AI 検索、セマンティック検索、ハイブリッド検索 |
| 02 | [[docs/03-applications/02-code-generation.md]] | AI コード生成、Copilot、Claude Code、Cursor |
| 03 | [[docs/03-applications/03-content-generation.md]] | コンテンツ生成、要約、翻訳、構造化データ抽出 |

### 04-evaluation — 評価と安全性

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-evaluation/00-evaluation-methods.md]] | LLM 評価手法、自動評価、人間評価、ベンチマーク |
| 01 | [[docs/04-evaluation/01-safety-and-guardrails.md]] | ガードレール、コンテンツフィルタリング、ジェイルブレイク対策 |
| 02 | [[docs/04-evaluation/02-cost-optimization.md]] | コスト最適化、キャッシュ、バッチ処理、モデル選択戦略 |

## クイックリファレンス

```
LLM モデル比較（2024年末時点）:

  モデル          │ 強み              │ コスト  │ 用途
  ──────────────┼──────────────────┼───────┼────────
  Claude Opus   │ 推論・分析・コード  │ 高     │ 複雑なタスク
  Claude Sonnet │ バランス           │ 中     │ 汎用
  GPT-4o        │ マルチモーダル      │ 高     │ 汎用
  Gemini Pro    │ 長文脈・検索統合    │ 中     │ Google 連携
  Llama 3       │ オープン・カスタム  │ 自前   │ セルフホスト
  Mistral       │ コスパ・EU 拠点    │ 低-中  │ 欧州案件
```

## 参考文献

1. Anthropic. "Claude Documentation." docs.anthropic.com, 2024.
2. OpenAI. "API Documentation." platform.openai.com, 2024.
3. Vaswani, A. et al. "Attention Is All You Need." NeurIPS, 2017.
