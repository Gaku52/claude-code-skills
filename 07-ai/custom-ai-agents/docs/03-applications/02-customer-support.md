# カスタマーサポートエージェント

> チャットボット・FAQ・エスカレーション――顧客の問い合わせを自動分類し、適切な回答を生成し、必要に応じて人間のオペレーターに引き継ぐサポートエージェントの設計。

## この章で学ぶこと

1. サポートエージェントのワークフロー設計（分類・回答・エスカレーション）
2. RAGベースのナレッジ検索と回答生成の実装パターン
3. 人間のオペレーターとの連携（ハンドオフ）設計

---

## 1. サポートエージェントの全体像

```
カスタマーサポートエージェントのフロー

[顧客の問い合わせ]
       |
       v
[意図分類] ──→ スパム/不正 ──→ [ブロック]
       |
       ├── FAQ対応可能 ──→ [ナレッジ検索] ──→ [回答生成] ──→ [顧客]
       |
       ├── 技術的問題 ──→ [トラブルシュート] ──→ [解決?]
       |                                          ├── YES → [顧客]
       |                                          └── NO  → [エスカレーション]
       |
       ├── アカウント操作 ──→ [本人確認] ──→ [操作実行] ──→ [顧客]
       |
       └── 複雑/感情的 ──→ [人間オペレーター] (即エスカレーション)
```

---

## 2. 基本的なサポートエージェント

### 2.1 意図分類

```python
# 問い合わせの意図分類
import anthropic

class IntentClassifier:
    INTENTS = {
        "billing": "請求・支払い関連",
        "technical": "技術的な問題・バグ報告",
        "account": "アカウント管理（変更、解約等）",
        "product": "製品に関する質問",
        "complaint": "クレーム・苦情",
        "general": "その他の一般的な問い合わせ"
    }

    def __init__(self):
        self.client = anthropic.Anthropic()

    def classify(self, message: str) -> dict:
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # 高速・低コスト
            max_tokens=256,
            messages=[{"role": "user", "content": f"""
以下の顧客メッセージを分類してください。

メッセージ: {message}

カテゴリ: {list(self.INTENTS.keys())}

JSON形式で出力:
{{"intent": "カテゴリ名", "confidence": 0.0-1.0, "sentiment": "positive/neutral/negative", "urgency": "low/medium/high"}}
"""}]
        )
        return json.loads(response.content[0].text)

# 使用例
classifier = IntentClassifier()
result = classifier.classify("先月の請求が二重になっています！至急確認してください")
# {"intent": "billing", "confidence": 0.95, "sentiment": "negative", "urgency": "high"}
```

### 2.2 RAGベースの回答生成

```python
# ナレッジベースからの回答生成
class SupportKnowledgeBase:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = anthropic.Anthropic()

    def answer(self, question: str, customer_context: dict = None) -> dict:
        # 1. 関連ナレッジを検索
        relevant_docs = self.vector_store.search(question, top_k=5)

        # 2. コンテキスト構築
        context = "\n\n".join([
            f"--- ドキュメント: {doc['title']} ---\n{doc['content']}"
            for doc in relevant_docs
        ])

        customer_info = ""
        if customer_context:
            customer_info = f"""
顧客情報:
- プラン: {customer_context.get('plan', '不明')}
- 利用期間: {customer_context.get('tenure', '不明')}
- 過去の問い合わせ: {customer_context.get('history', 'なし')}
"""

        # 3. 回答生成
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下の情報を基に、顧客の質問に回答してください。

{customer_info}

ナレッジベース:
{context}

顧客の質問: {question}

ルール:
- ナレッジベースに情報がない場合は「確認いたします」と回答
- 推測や憶測は絶対にしない
- 丁寧で簡潔な言葉遣いを使う
- 具体的な手順がある場合は番号付きリストで示す
"""}]
        )

        answer_text = response.content[0].text

        # 4. 信頼度評価
        has_knowledge = any(
            self._is_relevant(doc, question)
            for doc in relevant_docs
        )

        return {
            "answer": answer_text,
            "confidence": 0.9 if has_knowledge else 0.3,
            "sources": [doc["title"] for doc in relevant_docs[:3]],
            "should_escalate": not has_knowledge
        }
```

### 2.3 完全なサポートエージェント

```python
# 完全なカスタマーサポートエージェント
class CustomerSupportAgent:
    def __init__(self, knowledge_base, crm_system):
        self.classifier = IntentClassifier()
        self.kb = knowledge_base
        self.crm = crm_system
        self.client = anthropic.Anthropic()
        self.escalation_threshold = 0.5

    def handle_inquiry(self, customer_id: str, message: str) -> dict:
        # 1. 顧客情報取得
        customer = self.crm.get_customer(customer_id)

        # 2. 意図分類
        intent = self.classifier.classify(message)

        # 3. 即時エスカレーション判定
        if self._needs_immediate_escalation(intent, customer):
            return self._escalate(customer_id, message, intent, "自動エスカレーション")

        # 4. 意図に応じた処理
        if intent["intent"] == "billing":
            response = self._handle_billing(customer, message)
        elif intent["intent"] == "technical":
            response = self._handle_technical(customer, message)
        elif intent["intent"] == "account":
            response = self._handle_account(customer, message)
        else:
            response = self.kb.answer(message, customer)

        # 5. 信頼度チェック
        if response["confidence"] < self.escalation_threshold:
            return self._escalate(customer_id, message, intent, "低信頼度")

        # 6. 履歴保存
        self.crm.log_interaction(customer_id, message, response["answer"])

        return {
            "response": response["answer"],
            "intent": intent,
            "escalated": False
        }

    def _needs_immediate_escalation(self, intent: dict, customer: dict) -> bool:
        """即時エスカレーションが必要かを判定"""
        # 強いネガティブ感情
        if intent["sentiment"] == "negative" and intent["urgency"] == "high":
            return True
        # VIP顧客
        if customer.get("tier") == "enterprise":
            return True
        # 苦情
        if intent["intent"] == "complaint":
            return True
        return False

    def _escalate(self, customer_id, message, intent, reason) -> dict:
        """人間のオペレーターにエスカレーション"""
        ticket = self.crm.create_escalation_ticket(
            customer_id=customer_id,
            message=message,
            intent=intent,
            reason=reason,
            priority="high" if intent.get("urgency") == "high" else "normal"
        )
        return {
            "response": "担当者におつなぎいたします。少々お待ちください。",
            "intent": intent,
            "escalated": True,
            "ticket_id": ticket["id"]
        }
```

---

## 3. エスカレーション設計

```
エスカレーションの判断マトリクス

                    感情: ポジティブ/中立    感情: ネガティブ
                    +-------------------+-------------------+
  信頼度: 高 (>0.8) | 自動回答          | 自動回答 +        |
                    |                   | トーン注意        |
                    +-------------------+-------------------+
  信頼度: 中 (0.5-) | 自動回答 +        | エスカレーション    |
                    | 確認文付き        |                   |
                    +-------------------+-------------------+
  信頼度: 低 (<0.5) | エスカレーション    | 即時エスカレーション |
                    |                   | (優先度高)         |
                    +-------------------+-------------------+
```

---

## 4. 会話管理

```python
# マルチターン会話の管理
class ConversationManager:
    def __init__(self):
        self.sessions = {}

    def get_or_create_session(self, customer_id: str) -> dict:
        if customer_id not in self.sessions:
            self.sessions[customer_id] = {
                "messages": [],
                "intent_history": [],
                "created_at": time.time(),
                "resolved": False
            }
        return self.sessions[customer_id]

    def add_message(self, customer_id: str, role: str, content: str):
        session = self.get_or_create_session(customer_id)
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })

    def get_context_summary(self, customer_id: str) -> str:
        """会話の要約を生成（長い会話の圧縮用）"""
        session = self.sessions.get(customer_id)
        if not session or len(session["messages"]) < 10:
            return ""

        # 古い部分を要約
        old_messages = session["messages"][:-5]
        summary = self.llm.summarize(old_messages)
        return f"これまでの会話の要約: {summary}"
```

---

## 5. 比較表

### 5.1 サポートチャネル比較

| チャネル | 対応速度 | コスト | 顧客満足度 | 対応可能時間 |
|---------|---------|--------|-----------|------------|
| AIチャット | 即時 | 最低 | 中-高 | 24/7 |
| 人間チャット | 1-5分 | 中 | 高 | 営業時間 |
| メール | 数時間-1日 | 低 | 中 | 24/7受付 |
| 電話 | 待ち時間あり | 最高 | 最高 | 営業時間 |
| FAQ/セルフ | 即時 | 最低 | 低-中 | 24/7 |

### 5.2 自動化レベル比較

| レベル | 説明 | 解決率 | 適用場面 |
|--------|------|--------|---------|
| L0 ルール | if-else定型回答 | 20-30% | よくある質問 |
| L1 検索 | FAQ検索+テンプレート | 40-50% | ナレッジベースあり |
| L2 RAG | 文書検索+LLM生成 | 50-65% | 豊富なドキュメント |
| L3 エージェント | 自律的問題解決 | 60-75% | ツール統合あり |
| L4 完全自律 | アカウント操作含む | 70-85% | CRM/DB統合あり |

---

## 6. トーン・言語設計

```python
# 回答のトーン調整
class ToneAdjuster:
    TONE_GUIDELINES = {
        "positive": "明るく前向きな言葉遣い。顧客の良い体験を喜ぶ。",
        "neutral": "丁寧でプロフェッショナル。事実ベースの対応。",
        "negative": "共感を示す。「ご不便をおかけし申し訳ございません」から始める。お詫びの後に解決策を提示。",
        "angry": "最大限の共感。感情を否定しない。具体的な解決ステップを即座に提示。エスカレーション選択肢も。"
    }

    def adjust(self, answer: str, sentiment: str) -> str:
        guidelines = self.TONE_GUIDELINES.get(sentiment, self.TONE_GUIDELINES["neutral"])
        response = self.llm.generate(f"""
以下の回答を、顧客の感情に配慮してリライトしてください。

トーンガイドライン: {guidelines}

元の回答: {answer}

リライト後:
""")
        return response
```

---

## 7. アンチパターン

### アンチパターン1: 一律テンプレート回答

```
# NG: 全ての問い合わせに同じテンプレート
"お問い合わせありがとうございます。担当部署に確認の上、
 3営業日以内にご回答いたします。"

# OK: 問い合わせ内容に応じたパーソナライズ回答
"二重請求のご連絡ありがとうございます。
 確認したところ、2月15日の請求 ¥3,980 が重複しておりました。
 本日中に返金処理を行います。返金は5営業日以内にカードに反映されます。"
```

### アンチパターン2: エスカレーションの遅延

```python
# NG: 自力解決に固執して顧客を待たせる
for attempt in range(10):  # 10回試行...
    answer = generate_answer(question)
    if answer.confidence > 0.3:  # 低い閾値
        return answer

# OK: 早期エスカレーション
answer = generate_answer(question)
if answer.confidence < 0.7 or customer.is_frustrated():
    return escalate_to_human(question)  # 素早く人間に引き継ぎ
```

---

## 8. FAQ

### Q1: サポートエージェントの効果をどう測定する？

主要KPI:
- **自動解決率**: 人間の介入なしに解決した割合（目標: 60-80%）
- **初回回答解決率**: 最初の回答で問題が解決した割合
- **CSAT**: 顧客満足度スコア（1-5）
- **平均対応時間**: 問い合わせから解決までの時間
- **エスカレーション率**: 人間に引き継いだ割合

### Q2: 多言語対応の方法は？

2つのアプローチ:
1. **検出→翻訳→処理→翻訳**: 入力言語を検出し、内部処理は単一言語で行い、回答を元言語に翻訳
2. **ネイティブ多言語**: LLMの多言語能力を活用し、入力言語のまま処理・回答

Claudeの場合は後者が推奨。日本語入力にそのまま日本語で回答可能。

### Q3: 個人情報の扱いは？

- **マスキング**: クレジットカード番号等はマスクしてからLLMに渡す
- **ログ管理**: 会話ログから個人情報を除外して保存
- **データ保持期間**: GDPR/個人情報保護法に準拠した保持期間設定
- **LLMプロバイダのポリシー確認**: データがモデル学習に使われないことを確認

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアフロー | 意図分類 → ナレッジ検索 → 回答生成 → エスカレーション |
| 意図分類 | 高速モデル（Haiku）で分類、感情・緊急度も判定 |
| 回答生成 | RAGベース、ナレッジベースの情報のみ使用 |
| エスカレーション | 信頼度×感情のマトリクスで判断 |
| トーン | 顧客の感情に応じた言葉遣いの調整 |
| KPI | 自動解決率、CSAT、平均対応時間 |

## 次に読むべきガイド

- [03-data-agents.md](./03-data-agents.md) — データ分析エージェント
- [../01-patterns/02-workflow-agents.md](../01-patterns/02-workflow-agents.md) — ワークフロー設計の詳細
- [../04-production/00-deployment.md](../04-production/00-deployment.md) — サポートエージェントのデプロイ

## 参考文献

1. Anthropic, "Customer service agent cookbook" — https://docs.anthropic.com/en/docs/about-claude/use-case-guides/customer-service
2. Zendesk, "AI in Customer Service" — https://www.zendesk.com/blog/ai-customer-service/
3. Intercom, "AI-First Customer Service" — https://www.intercom.com/ai-bot
