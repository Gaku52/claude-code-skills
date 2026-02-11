# メール/コミュニケーション自動化 — 自動返信、要約、分類

> メールとビジネスコミュニケーションにAIを統合し、自動返信、要約生成、優先度分類を実現する実践的な設計と実装を解説する。

---

## この章で学ぶこと

1. **メールAI分類システム** — 受信メールの自動カテゴリ分類と優先度判定の設計・実装
2. **AI自動返信エンジン** — コンテキスト理解に基づく返信ドラフト生成とトーン制御
3. **コミュニケーション要約** — 長いメールスレッド、会議録、チャットの自動要約

---

## 1. メールAI処理アーキテクチャ

### 1.1 全体構成

```
┌──────────────────────────────────────────────────────────┐
│           メールAI処理パイプライン                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  受信        分析            判断           アクション     │
│  ┌─────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │IMAP │─▶│分類      │─▶│ルーティング│─▶│自動返信  │    │
│  │/API │  │感情分析  │  │優先度判定 │  │転送      │    │
│  │     │  │意図抽出  │  │担当割当  │  │タスク作成│    │
│  └─────┘  └──────────┘  └──────────┘  └──────────┘    │
│      │         │              │              │          │
│      ▼         ▼              ▼              ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ダッシュボード & ログ                 │    │
│  │  メトリクス | 精度 | コスト | 対応時間            │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 1.2 メール処理のライフサイクル

```
メール受信 → 前処理 → AI分析 → 判定 → アクション → 学習

  ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
  │ 1 │──▶│ 2 │──▶│ 3 │──▶│ 4 │──▶│ 5 │──▶│ 6 │
  └───┘   └───┘   └───┘   └───┘   └───┘   └───┘
  受信     HTML    分類     緊急?   返信    フィード
  解析     →Text   感情     スパム?  転送    バック
           ヘッダ   意図     VIP?   タスク   精度改善
```

---

## 2. メール分類エンジン

### 2.1 分類システム実装

```python
import anthropic
from dataclasses import dataclass
from enum import Enum

class EmailCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    PARTNERSHIP = "partnership"
    SPAM = "spam"
    PERSONAL = "personal"
    NEWSLETTER = "newsletter"

class Priority(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class EmailAnalysis:
    category: EmailCategory
    priority: Priority
    sentiment: str  # positive/neutral/negative
    intent: str
    summary: str
    suggested_action: str
    confidence: float

class EmailClassifier:
    """AIメール分類エンジン"""

    CLASSIFICATION_PROMPT = """
以下のメールを分析し、JSON形式で結果を返してください。

分析項目:
- category: billing / technical / sales / partnership / spam / personal / newsletter
- priority: urgent / high / medium / low
- sentiment: positive / neutral / negative
- intent: 1文で送信者の意図
- summary: 50文字以内の要約
- suggested_action: 推奨アクション
- confidence: 0.0-1.0の信頼度

判断基準:
- urgent: 即座の対応が必要（障害報告、請求トラブル等）
- high: 今日中に対応すべき（重要顧客、締切案件）
- medium: 2-3日以内に対応（一般問い合わせ）
- low: 対応不要 or いつでも可（ニュースレター等）

メール:
From: {sender}
Subject: {subject}
Date: {date}
Body:
{body}
"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.vip_list = set()
        self.rules = []

    def add_vip(self, email: str):
        """VIPリスト追加"""
        self.vip_list.add(email.lower())

    def classify(self, email: dict) -> EmailAnalysis:
        """メール分類"""
        # ルールベースの前処理
        if self._is_obvious_spam(email):
            return EmailAnalysis(
                category=EmailCategory.SPAM,
                priority=Priority.LOW,
                sentiment="neutral",
                intent="スパム",
                summary="スパムメール",
                suggested_action="自動削除",
                confidence=0.99
            )

        # AI分析
        prompt = self.CLASSIFICATION_PROMPT.format(**email)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        result = self._parse_response(response.content[0].text)

        # VIP補正
        if email["sender"].lower() in self.vip_list:
            if result.priority != Priority.URGENT:
                result.priority = Priority.HIGH

        return result

    def _is_obvious_spam(self, email: dict) -> bool:
        """ルールベーススパム判定"""
        spam_keywords = ["当選", "無料", "今すぐクリック", "unsubscribe"]
        subject_body = f"{email['subject']} {email['body']}".lower()
        return sum(1 for kw in spam_keywords if kw in subject_body) >= 2

    def _parse_response(self, text: str) -> EmailAnalysis:
        """レスポンスパース"""
        import json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return EmailAnalysis(
                category=EmailCategory(data["category"]),
                priority=Priority(data["priority"]),
                sentiment=data["sentiment"],
                intent=data["intent"],
                summary=data["summary"],
                suggested_action=data["suggested_action"],
                confidence=data.get("confidence", 0.8)
            )
        except Exception:
            return EmailAnalysis(
                category=EmailCategory.PERSONAL,
                priority=Priority.MEDIUM,
                sentiment="neutral",
                intent="分類不能",
                summary="AI分類に失敗",
                suggested_action="手動確認",
                confidence=0.0
            )
```

### 2.2 分類精度の比較

| 分類方式 | 精度 | 速度 | コスト/1000通 | 適用場面 |
|---------|------|------|-------------|---------|
| キーワードマッチ | 60-70% | 即時 | 0円 | 簡易フィルタ |
| ルールベース | 75-85% | 即時 | 0円 | 定型パターン |
| 従来ML (SVM/NB) | 85-90% | 高速 | ~$0.01 | 大量処理 |
| GPT-3.5-turbo | 90-93% | 中 | ~$0.50 | コスト重視 |
| GPT-4 / Claude | 95-98% | 低速 | ~$5.00 | 精度重視 |
| ハイブリッド | 96-99% | 中 | ~$1.00 | 最適バランス |

---

## 3. AI自動返信エンジン

### 3.1 返信生成システム

```python
class AutoReplyEngine:
    """AI自動返信エンジン"""

    REPLY_PROMPT = """
あなたは{company_name}のカスタマーサポート担当です。
以下のメールに対する返信ドラフトを作成してください。

トーン: {tone}
言語: 日本語
署名: {signature}

ルール:
- 丁寧だが簡潔に
- 具体的な解決策を提示
- 不明点は正直に伝え、確認する旨を記載
- 個人情報は含めない

元メール:
From: {sender_name} <{sender_email}>
Subject: {subject}
Body:
{body}

過去のやり取り（あれば）:
{thread_history}
"""

    def __init__(self, api_key: str, company_name: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.company_name = company_name
        self.templates = {}

    def generate_reply(self, email: dict,
                       tone: str = "professional",
                       thread_history: str = "") -> dict:
        """返信ドラフト生成"""
        signature = self._get_signature(email.get("assigned_to"))

        prompt = self.REPLY_PROMPT.format(
            company_name=self.company_name,
            tone=tone,
            signature=signature,
            sender_name=email.get("sender_name", ""),
            sender_email=email["sender"],
            subject=email["subject"],
            body=email["body"],
            thread_history=thread_history or "なし"
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        reply_text = response.content[0].text
        return {
            "subject": f"Re: {email['subject']}",
            "body": reply_text,
            "status": "draft",  # 自動送信ではなくドラフト
            "confidence": self._assess_confidence(reply_text, email)
        }

    def _assess_confidence(self, reply: str, email: dict) -> float:
        """返信の信頼度判定"""
        score = 0.8
        if len(reply) < 50:
            score -= 0.2  # 短すぎる
        if "確認" in reply or "お調べ" in reply:
            score -= 0.1  # 不確実性あり
        if email.get("priority") == "urgent":
            score -= 0.1  # 緊急案件は人間確認推奨
        return max(0.0, min(1.0, score))

    def _get_signature(self, assigned_to: str = None) -> str:
        return f"""
--
{self.company_name} カスタマーサポート
{assigned_to or "担当者"}
"""
```

### 3.2 トーン制御マトリクス

```
トーン制御パラメータ:

  フォーマル度
  高 ┤ ● 公式謝罪    ● 契約関連
     │
  中 ┤ ● 一般サポート ● ビジネス提案
     │
  低 ┤ ● 社内連絡     ● カジュアル問い合わせ
     └──┬────────────┬────────────┬──
       冷静         中立         親しみ
                感情トーン
```

---

## 4. メールスレッド要約

### 4.1 スレッド要約エンジン

```python
class ThreadSummarizer:
    """メールスレッド要約エンジン"""

    SUMMARY_PROMPT = """
以下のメールスレッドを分析し、構造化された要約を作成してください。

出力形式:
1. 概要（2-3文）
2. 参加者と役割
3. 経緯（時系列）
4. 現在のステータス
5. 未解決事項
6. 必要なアクション

スレッド:
{thread}
"""

    def summarize_thread(self, messages: list[dict]) -> dict:
        """スレッド全体を要約"""
        thread_text = self._format_thread(messages)

        # 長いスレッドはチャンク処理
        if len(thread_text) > 10000:
            return self._summarize_long_thread(messages)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": self.SUMMARY_PROMPT.format(thread=thread_text)
            }]
        )
        return {"summary": response.content[0].text}

    def _summarize_long_thread(self, messages: list[dict]) -> dict:
        """長いスレッドの段階的要約"""
        # Phase 1: 各メールを個別要約
        individual_summaries = []
        for msg in messages:
            summary = self._summarize_single(msg)
            individual_summaries.append(summary)

        # Phase 2: 個別要約を統合
        combined = "\n".join(
            f"[{s['date']}] {s['from']}: {s['summary']}"
            for s in individual_summaries
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"以下のメール要約を統合:\n{combined}"
            }]
        )
        return {"summary": response.content[0].text}

    def _format_thread(self, messages: list[dict]) -> str:
        """スレッドをテキスト形式に整形"""
        parts = []
        for msg in messages:
            parts.append(
                f"---\nFrom: {msg['sender']}\n"
                f"Date: {msg['date']}\n"
                f"Subject: {msg['subject']}\n\n"
                f"{msg['body']}\n"
            )
        return "\n".join(parts)
```

### 4.2 会議録要約

```python
class MeetingSummarizer:
    """会議録AI要約"""

    def summarize_meeting(self, transcript: str) -> dict:
        """会議録を構造化要約"""
        prompt = f"""
以下の会議録を分析し、JSON形式で要約:

{{
  "title": "会議タイトル",
  "date": "日付",
  "participants": ["参加者リスト"],
  "duration": "所要時間",
  "summary": "3行要約",
  "decisions": ["決定事項"],
  "action_items": [
    {{"task": "タスク内容", "assignee": "担当者", "deadline": "期限"}}
  ],
  "open_issues": ["未解決事項"],
  "next_meeting": "次回予定"
}}

会議録:
{transcript}
"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return self._parse_json(response.content[0].text)
```

---

## 5. 統合ワークフロー

### 5.1 完全自動化フロー

```
完全自動化メールワークフロー:

  受信 ──▶ 分類 ──▶ ルーティング
              │
    ┌─────────┼─────────┬──────────────┐
    ▼         ▼         ▼              ▼
 [スパム]  [サポート]  [営業]       [その他]
    │         │         │              │
 自動削除  ┌──┴──┐   CRM登録      Inbox保留
           │     │
        [自動]  [手動]
           │     │
        ドラフト  担当者
        生成     通知
           │
        信頼度
        チェック
         │    │
       ≥0.8  <0.8
         │    │
       自動   人間
       送信   レビュー
```

---

## 6. アンチパターン

### アンチパターン1: 無条件自動送信

```python
# BAD: AI生成文をそのまま自動送信
def handle_email(email):
    reply = ai.generate_reply(email)
    send_email(reply)  # 顧客にいきなり送信 — 危険！

# GOOD: 信頼度に基づく段階的自動化
def handle_email(email):
    analysis = classifier.classify(email)
    reply = ai.generate_reply(email)

    if analysis.category == EmailCategory.SPAM:
        archive(email)  # スパムは自動アーカイブ
    elif reply["confidence"] >= 0.9 and analysis.priority == Priority.LOW:
        send_email(reply)  # 低優先度+高信頼のみ自動送信
    elif reply["confidence"] >= 0.7:
        save_as_draft(reply)  # ドラフト保存、1クリック送信
        notify_agent(email)
    else:
        escalate_to_human(email)  # 人間にエスカレーション
```

### アンチパターン2: コンテキスト無視の返信

```python
# BAD: 個別メールだけ見て返信
def generate_reply(email):
    return ai.reply(email["body"])  # 過去のやり取りを無視

# GOOD: スレッド全体のコンテキストを考慮
def generate_reply(email):
    thread = fetch_thread(email["thread_id"])
    customer_history = fetch_customer_history(email["sender"])

    context = {
        "thread": thread,
        "customer_tier": customer_history.tier,
        "past_issues": customer_history.recent_issues,
        "sentiment_trend": analyze_sentiment_trend(thread)
    }

    return ai.reply_with_context(email, context)
```

---

## 7. FAQ

### Q1: 自動返信で顧客満足度は下がらない？

**A:** 適切に実装すれば むしろ向上する。調査によると、(1) 応答速度が10分→30秒に短縮され顧客満足度15%向上、(2) 24時間対応が可能に、(3) 回答の一貫性が保たれる。ただし「AIが対応している」旨を隠さないこと、複雑な案件は速やかに人間にエスカレーションすることが条件。

### Q2: プライバシーとセキュリティの対策は？

**A:** 3層の対策を推奨。(1) データマスキング — メール送信前にPII（個人情報）を自動マスク、(2) API選択 — データ保持しないAPI（Anthropic API等）を使用、(3) オンプレミス — 機密度の高い業界はセルフホストLLM（Llama等）を検討。GDPRやAPPI（日本の個人情報保護法）の要件も確認すること。

### Q3: 既存メールシステムとの統合方法は？

**A:** 3つのアプローチがある。(1) IMAP/SMTP直接 — 最も柔軟だが実装コスト高、(2) Gmail API / Microsoft Graph API — Google/Microsoft環境なら最適、(3) Zapier/n8n経由 — 最も簡単で即日稼働可能。おすすめは(3)で始めて、規模拡大後に(2)へ移行する段階的アプローチ。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 分類 | ルールベース + AI のハイブリッドが最良 |
| 自動返信 | 信頼度スコアで自動/ドラフト/人間を切り替え |
| 要約 | 長いスレッドは段階的要約（個別→統合） |
| セキュリティ | PIIマスキング + データ非保持API + 暗号化 |
| 導入順序 | 分類→要約→ドラフト生成→段階的自動送信 |
| KPI | 応答時間、解決率、顧客満足度、コスト/通 |

---

## 次に読むべきガイド

- [../01-business/02-content-creation.md](../01-business/02-content-creation.md) — コンテンツ制作の自動化
- [../02-monetization/01-cost-management.md](../02-monetization/01-cost-management.md) — API費用最適化
- [00-automation-overview.md](./00-automation-overview.md) — AI自動化の全体像

---

## 参考文献

1. **Gmail API Documentation** — https://developers.google.com/gmail/api — Gmail統合の公式ガイド
2. **Microsoft Graph API** — https://learn.microsoft.com/graph — Outlook/Teams統合
3. **"AI-Powered Customer Service" — Harvard Business Review (2024)** — AI顧客対応の効果測定研究
4. **Anthropic API Documentation** — https://docs.anthropic.com — Claude APIのベストプラクティス
