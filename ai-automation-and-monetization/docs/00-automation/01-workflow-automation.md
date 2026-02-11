# ワークフロー自動化 — Zapier、n8n、Make 実践ガイド

> 3大ワークフロー自動化プラットフォーム（Zapier、n8n、Make）を比較し、AIと連携した実用的な自動化フローの設計・構築・運用を実践的に解説する。

---

## この章で学ぶこと

1. **3大プラットフォームの特徴と選定基準** — Zapier、n8n、Makeの強み・弱みを理解し、ユースケースに応じた選択ができる
2. **AI連携ワークフローの設計パターン** — トリガー→AI処理→アクションの基本パターンから複雑な分岐処理まで
3. **本番運用のベストプラクティス** — エラーハンドリング、モニタリング、コスト最適化の実践手法

---

## 1. プラットフォーム比較

### 1.1 3大プラットフォーム概要

```
┌─────────────────────────────────────────────────────────────┐
│            ワークフロー自動化プラットフォーム比較              │
├──────────┬──────────────┬──────────────┬───────────────────┤
│          │   Zapier     │    Make      │      n8n          │
├──────────┼──────────────┼──────────────┼───────────────────┤
│ 種類     │ SaaS         │ SaaS         │ OSS / セルフホスト│
│ 価格     │ $20-$800/月  │ $9-$300/月   │ 無料(自前) /$20+  │
│ 連携数   │ 7,000+       │ 1,800+       │ 400+              │
│ AI連携   │ ネイティブ    │ HTTP経由     │ ネイティブ+HTTP   │
│ 難易度   │ 初級         │ 中級         │ 中級〜上級        │
│ 実行上限 │ プラン依存    │ Ops数依存    │ 無制限(自前)      │
└──────────┴──────────────┴──────────────┴───────────────────┘
```

### 1.2 詳細比較表

| 比較項目 | Zapier | Make | n8n |
|---------|--------|------|-----|
| 初期設定の容易さ | 非常に簡単 | 簡単 | やや複雑 |
| ビジュアルエディタ | リスト型 | フロー図型 | フロー図型 |
| 条件分岐 | Paths機能 | Router | IF/Switch |
| ループ処理 | 制限あり | Iterator | Loop/SplitInBatches |
| Webhook | 有料プラン | 全プラン | 全プラン |
| データ変換 | Formatter | 組込み関数 | Function/Code |
| セルフホスト | 不可 | 不可 | Docker/npm |
| API制限 | 月間タスク数 | Operations数 | なし(自前) |
| チーム機能 | Business+ | Team+ | 全プラン |
| デバッグ | 実行ログ | 実行履歴 | 実行ログ+デバッガ |

---

## 2. Zapier + AI ワークフロー

### 2.1 基本構造

```
Zapier AI ワークフロー基本構造:

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Trigger  │───▶│ AI Step  │───▶│ Filter/  │───▶│ Action   │
  │ (Gmail)  │    │ (GPT-4)  │    │ Branch   │    │ (Slack)  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘
       │                                                │
       │           ┌──────────┐                         │
       └──────────▶│ Formatter│─────────────────────────┘
                   │ (整形)   │
                   └──────────┘
```

### 2.2 実装例: 顧客問い合わせ自動分類

```python
# Zapier フロー定義（概念的にPythonで表現）
zapier_flow = {
    "name": "顧客問い合わせ自動分類・対応",
    "trigger": {
        "app": "Gmail",
        "event": "新規メール受信",
        "filter": "from:*@customer.com"
    },
    "steps": [
        {
            "app": "ChatGPT (OpenAI)",
            "action": "Conversation",
            "config": {
                "model": "gpt-4",
                "prompt": """
以下のメールを分析し、JSON形式で返答してください:
- category: billing/technical/sales/other
- urgency: high/medium/low
- summary: 50文字以内の要約
- suggested_response: 推奨返信文

メール本文:
{{trigger.body}}
""",
                "memory_key": "customer_{{trigger.from}}"
            }
        },
        {
            "app": "Formatter",
            "action": "Text - Extract JSON",
            "config": {"input": "{{step1.response}}"}
        },
        {
            "app": "Paths",
            "conditions": [
                {
                    "name": "緊急対応",
                    "condition": "{{step2.urgency}} == high",
                    "actions": [
                        {"app": "Slack", "channel": "#urgent-support"},
                        {"app": "PagerDuty", "severity": "high"}
                    ]
                },
                {
                    "name": "通常対応",
                    "condition": "{{step2.urgency}} != high",
                    "actions": [
                        {"app": "Notion", "database": "Support Tickets"},
                        {"app": "Gmail", "draft": "{{step2.suggested_response}}"}
                    ]
                }
            ]
        }
    ]
}
```

---

## 3. n8n + AI ワークフロー

### 3.1 n8nのセルフホスト構成

```yaml
# docker-compose.yml - n8n セルフホスト
version: '3.8'
services:
  n8n:
    image: n8nio/n8n:latest
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
      - N8N_ENCRYPTION_KEY=${ENCRYPTION_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - n8n_data:/home/node/.n8n
    restart: unless-stopped

  postgres:
    image: postgres:16
    environment:
      - POSTGRES_DB=n8n
      - POSTGRES_USER=n8n
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  n8n_data:
  postgres_data:
```

### 3.2 n8nワークフロー: ドキュメント要約パイプライン

```json
{
  "name": "AI Document Summarizer",
  "nodes": [
    {
      "type": "n8n-nodes-base.webhook",
      "name": "Document Upload",
      "parameters": {
        "httpMethod": "POST",
        "path": "summarize",
        "responseMode": "lastNode"
      }
    },
    {
      "type": "n8n-nodes-base.code",
      "name": "Preprocess",
      "parameters": {
        "jsCode": "// ドキュメントをチャンクに分割\nconst text = $input.first().json.document;\nconst chunkSize = 3000;\nconst chunks = [];\nfor (let i = 0; i < text.length; i += chunkSize) {\n  chunks.push({ chunk: text.slice(i, i + chunkSize), index: i / chunkSize });\n}\nreturn chunks.map(c => ({ json: c }));"
      }
    },
    {
      "type": "@n8n/n8n-nodes-langchain.openAi",
      "name": "Summarize Chunks",
      "parameters": {
        "model": "gpt-4",
        "prompt": "以下のテキストを日本語で3行に要約:\n\n{{$json.chunk}}"
      }
    },
    {
      "type": "n8n-nodes-base.code",
      "name": "Merge Summaries",
      "parameters": {
        "jsCode": "const summaries = $input.all().map(i => i.json.text).join('\\n');\nreturn [{ json: { combined: summaries } }];"
      }
    },
    {
      "type": "@n8n/n8n-nodes-langchain.openAi",
      "name": "Final Summary",
      "parameters": {
        "model": "gpt-4",
        "prompt": "以下の部分要約を統合し、構造化された最終要約を作成:\n\n{{$json.combined}}"
      }
    }
  ]
}
```

---

## 4. Make (Integromat) + AI ワークフロー

### 4.1 Make のシナリオ設計

```
Make シナリオ: SNS投稿自動生成

  ┌─────────┐   ┌─────────┐   ┌─────────┐
  │ RSS     │──▶│ OpenAI  │──▶│ Router  │
  │ Watch   │   │ 要約+   │   │         │
  │ (ブログ)│   │ SNS文生成│   │         │
  └─────────┘   └─────────┘   └────┬────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌──────────┐   ┌──────────┐   ┌──────────┐
              │ Twitter  │   │ LinkedIn │   │ Facebook │
              │ Post     │   │ Post     │   │ Post     │
              └──────────┘   └──────────┘   └──────────┘
```

### 4.2 Make HTTP + OpenAI

```python
# Make HTTP モジュール設定（概念的表現）
make_scenario = {
    "modules": [
        {
            "type": "RSS - Watch Items",
            "config": {
                "url": "https://myblog.com/feed",
                "limit": 5
            }
        },
        {
            "type": "HTTP - Make a Request",
            "config": {
                "url": "https://api.openai.com/v1/chat/completions",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer {{OPENAI_API_KEY}}",
                    "Content-Type": "application/json"
                },
                "body": {
                    "model": "gpt-4",
                    "messages": [{
                        "role": "user",
                        "content": "以下のブログ記事からSNS投稿を3種類生成:\n"
                                   "- Twitter用(140文字)\n"
                                   "- LinkedIn用(300文字、ビジネス調)\n"
                                   "- Facebook用(200文字、カジュアル)\n\n"
                                   "記事: {{1.title}} - {{1.description}}"
                    }]
                }
            }
        },
        {
            "type": "JSON - Parse",
            "config": {
                "input": "{{2.data.choices[0].message.content}}"
            }
        }
    ]
}
```

---

## 5. 共通設計パターン

### 5.1 エラーハンドリングパターン

```python
# ワークフローエラーハンドリングの3層構造
class WorkflowErrorHandler:
    """ワークフロー共通エラーハンドリング"""

    def __init__(self):
        self.retry_count = 0
        self.max_retries = 3
        self.dead_letter_queue = []

    def handle_error(self, error, step_name: str, input_data: dict):
        """3層エラーハンドリング"""
        # Layer 1: リトライ
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            wait_time = 2 ** self.retry_count
            print(f"[{step_name}] リトライ {self.retry_count}/{self.max_retries} "
                  f"({wait_time}秒後)")
            time.sleep(wait_time)
            return "retry"

        # Layer 2: フォールバック
        if hasattr(self, f"fallback_{step_name}"):
            print(f"[{step_name}] フォールバック実行")
            return getattr(self, f"fallback_{step_name}")(input_data)

        # Layer 3: Dead Letter Queue
        self.dead_letter_queue.append({
            "step": step_name,
            "error": str(error),
            "input": input_data,
            "timestamp": time.time()
        })
        self.notify_admin(step_name, error)
        return "failed"

    def fallback_ai_summarize(self, input_data):
        """AI要約のフォールバック: 安いモデルで再試行"""
        return call_ai(input_data["text"], model="gpt-3.5-turbo")

    def notify_admin(self, step_name, error):
        """管理者通知"""
        send_slack(f"ワークフロー障害: {step_name} - {error}")
```

---

## 6. アンチパターン

### アンチパターン1: ステップ過多のリニアフロー

```
BAD: 20ステップが直列に並ぶ
  Step1 → Step2 → Step3 → ... → Step20
  - 1箇所の失敗で全体が停止
  - デバッグが困難
  - 実行時間が長い

GOOD: モジュール化 + 並列実行
  ┌─ Module A (Step1→2→3) ─┐
  │                         ├──▶ Merge → Final
  └─ Module B (Step4→5→6) ─┘
  - 独立テスト可能
  - 並列実行で高速化
  - 障害の影響範囲が限定的
```

### アンチパターン2: AIへの過度な依存

```python
# BAD: 全判断をAIに委ねる
def process_order(order):
    decision = call_ai(f"この注文を処理すべきか?: {order}")
    if "はい" in decision:
        charge_customer(order)  # AIの回答で課金処理

# GOOD: AIは補助、ルールベースと併用
def process_order(order):
    # ルールベースのバリデーション（確実）
    if order.amount > 100000:
        return flag_for_review(order)
    if order.customer.is_blocked:
        return reject(order)

    # AIは追加の異常検知にのみ使用
    fraud_score = call_ai(f"不正スコアを0-100で判定: {order}")
    if int(fraud_score) > 80:
        return flag_for_review(order)

    return approve(order)
```

---

## 7. FAQ

### Q1: Zapier、Make、n8n のどれを選ぶべき？

**A:** 判断基準は3つ。(1) 予算 — 無料~低コストならn8n（セルフホスト）、中予算ならMake、予算十分ならZapier。(2) 技術力 — 非エンジニアはZapier、エンジニアはn8n。(3) 規模 — 小規模はZapier、大量処理はn8nかMake。月1,000タスク以下ならZapier無料プラン、1万タスク以上ならn8nセルフホストが費用対効果最良。

### Q2: AI APIのコストが心配。抑える方法は？

**A:** 3つの戦略がある。(1) キャッシュ — 同一入力の結果をRedis/DBに保存し再利用、(2) モデル選択 — 簡単なタスクはgpt-3.5-turbo（GPT-4の1/20コスト）、(3) プロンプト最適化 — 不要なコンテキストを削り入力トークン数を減らす。実測で月$500→$80に削減した事例もある。

### Q3: ワークフローのテスト方法は？

**A:** 3段階でテストする。(1) ユニットテスト — 各ステップを個別にモックデータで実行、(2) 統合テスト — テスト用Webhookで全フロー実行、(3) 本番モニタリング — 成功率・実行時間・コストをダッシュボードで監視。n8nはデバッガ内蔵、Zapierは実行ログで確認可能。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Zapier | 最多連携、初心者向け、コスト高め |
| Make | コスパ良好、ビジュアル設計、中級者向け |
| n8n | OSS、無制限実行、技術者向け |
| AI連携の鍵 | キャッシュ + モデル選択 + プロンプト最適化 |
| エラー処理 | リトライ → フォールバック → Dead Letter Queue |
| 設計原則 | モジュール化、並列化、Human-in-the-Loop |

---

## 次に読むべきガイド

- [02-document-processing.md](./02-document-processing.md) — ドキュメント処理の自動化
- [03-email-communication.md](./03-email-communication.md) — メール/コミュニケーション自動化
- [../02-monetization/01-cost-management.md](../02-monetization/01-cost-management.md) — API費用最適化

---

## 参考文献

1. **Zapier公式ドキュメント** — https://zapier.com/help — トリガー・アクション一覧とベストプラクティス
2. **n8n Documentation** — https://docs.n8n.io — ノード設定、セルフホスト、AI統合ガイド
3. **Make (Integromat) Help Center** — https://www.make.com/en/help — シナリオ設計とHTTPモジュール活用
4. **"Workflow Automation with AI" — Packt (2024)** — AI連携ワークフローの設計パターン集
