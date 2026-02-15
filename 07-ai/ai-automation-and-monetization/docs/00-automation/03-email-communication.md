# メール/コミュニケーション自動化 — 自動返信、要約、分類

> メールとビジネスコミュニケーションにAIを統合し、自動返信、要約生成、優先度分類を実現する実践的な設計と実装を解説する。

---

## この章で学ぶこと

1. **メールAI分類システム** — 受信メールの自動カテゴリ分類と優先度判定の設計・実装
2. **AI自動返信エンジン** — コンテキスト理解に基づく返信ドラフト生成とトーン制御
3. **コミュニケーション要約** — 長いメールスレッド、会議録、チャットの自動要約
4. **マルチチャネル統合** — Slack、Teams、チャット等の統合コミュニケーション管理
5. **運用設計とKPI** — 精度モニタリング、コスト管理、段階的自動化の実践
6. **セキュリティとコンプライアンス** — PII保護、GDPR/APPI対応、監査ログ

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

### 1.3 技術スタック選定ガイド

```
┌──────────────────────────────────────────────────────────────┐
│          メール自動化 技術スタック比較                          │
├──────────────┬────────────────────────────────────────────────┤
│ コンポーネント │ 選択肢と推奨                                   │
├──────────────┼────────────────────────────────────────────────┤
│ メール受信    │ Gmail API (推奨) / Microsoft Graph / IMAP      │
│ AI処理       │ Claude API (推奨) / GPT-4 / Gemini             │
│ キューイング  │ Redis + BullMQ (推奨) / SQS / RabbitMQ        │
│ データ保存    │ PostgreSQL (推奨) / Supabase / DynamoDB        │
│ ワークフロー  │ n8n (推奨) / Zapier / Temporal                │
│ フロントエンド │ Next.js (推奨) / React / Vue                  │
│ モニタリング  │ Grafana + Prometheus / Datadog / New Relic     │
│ 通知         │ Slack Webhook / Discord / LINE Notify          │
└──────────────┴────────────────────────────────────────────────┘

選定基準:
  ● 個人/小規模: Gmail API + Claude API + Supabase + n8n
  ● 中規模チーム: Microsoft Graph + Claude API + PostgreSQL + Temporal
  ● エンタープライズ: オンプレMTA + セルフホストLLM + Kubernetes
```

### 1.4 データフロー詳細設計

```python
# メールパイプラインの詳細データフロー定義
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class ProcessingStage(Enum):
    """処理ステージの定義"""
    RECEIVED = "received"
    PREPROCESSED = "preprocessed"
    CLASSIFIED = "classified"
    ROUTED = "routed"
    ACTION_TAKEN = "action_taken"
    FEEDBACK_COLLECTED = "feedback_collected"


@dataclass
class EmailMessage:
    """メールメッセージの統一データモデル"""
    message_id: str
    thread_id: str
    sender: str
    sender_name: str
    recipients: list[str]
    cc: list[str] = field(default_factory=list)
    subject: str = ""
    body_text: str = ""
    body_html: str = ""
    attachments: list[dict] = field(default_factory=list)
    headers: dict = field(default_factory=dict)
    received_at: datetime = field(default_factory=datetime.now)
    labels: list[str] = field(default_factory=list)

    # AI処理結果
    stage: ProcessingStage = ProcessingStage.RECEIVED
    classification: Optional[dict] = None
    sentiment: Optional[str] = None
    priority: Optional[str] = None
    suggested_reply: Optional[str] = None
    action_taken: Optional[str] = None
    processing_time_ms: int = 0
    ai_cost_usd: float = 0.0

    def to_ai_context(self) -> str:
        """AI処理用のコンテキスト文字列に変換"""
        return (
            f"From: {self.sender_name} <{self.sender}>\n"
            f"To: {', '.join(self.recipients)}\n"
            f"CC: {', '.join(self.cc)}\n"
            f"Subject: {self.subject}\n"
            f"Date: {self.received_at.isoformat()}\n"
            f"---\n"
            f"{self.body_text}"
        )

    def estimated_tokens(self) -> int:
        """トークン数の概算（日本語は1文字≒1.5トークン）"""
        text_length = len(self.body_text)
        return int(text_length * 1.5)


@dataclass
class ProcessingResult:
    """パイプライン処理結果"""
    message_id: str
    success: bool
    stage: ProcessingStage
    classification: Optional[dict] = None
    reply_draft: Optional[str] = None
    action: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    confidence: float = 0.0
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
    SUPPORT = "support"
    COMPLAINT = "complaint"
    FEEDBACK = "feedback"
    INTERNAL = "internal"

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
    language: str = "ja"
    topics: list[str] = None
    entities: list[dict] = None

class EmailClassifier:
    """AIメール分類エンジン"""

    CLASSIFICATION_PROMPT = """
以下のメールを分析し、JSON形式で結果を返してください。

分析項目:
- category: billing / technical / sales / partnership / spam / personal / newsletter / support / complaint / feedback / internal
- priority: urgent / high / medium / low
- sentiment: positive / neutral / negative
- intent: 1文で送信者の意図
- summary: 50文字以内の要約
- suggested_action: 推奨アクション
- confidence: 0.0-1.0の信頼度
- language: メールの言語コード（ja / en / zh 等）
- topics: 関連トピックのリスト（最大3つ）
- entities: 抽出したエンティティ（人名、会社名、金額、日付等）

判断基準:
- urgent: 即座の対応が必要（障害報告、請求トラブル、セキュリティインシデント等）
- high: 今日中に対応すべき（重要顧客、締切案件、クレーム初回対応）
- medium: 2-3日以内に対応（一般問い合わせ、機能要望）
- low: 対応不要 or いつでも可（ニュースレター、広告、FYI共有）

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
        self.classification_cache = {}
        self.stats = {
            "total_classified": 0,
            "cache_hits": 0,
            "rule_based": 0,
            "ai_classified": 0,
        }

    def add_vip(self, email: str):
        """VIPリスト追加"""
        self.vip_list.add(email.lower())

    def add_rule(self, condition: callable, result: EmailAnalysis):
        """カスタムルール追加"""
        self.rules.append({"condition": condition, "result": result})

    def classify(self, email: dict) -> EmailAnalysis:
        """メール分類"""
        self.stats["total_classified"] += 1

        # ルールベースの前処理
        if self._is_obvious_spam(email):
            self.stats["rule_based"] += 1
            return EmailAnalysis(
                category=EmailCategory.SPAM,
                priority=Priority.LOW,
                sentiment="neutral",
                intent="スパム",
                summary="スパムメール",
                suggested_action="自動削除",
                confidence=0.99
            )

        # カスタムルールチェック
        for rule in self.rules:
            if rule["condition"](email):
                self.stats["rule_based"] += 1
                return rule["result"]

        # ニュースレター自動検出
        if self._is_newsletter(email):
            self.stats["rule_based"] += 1
            return EmailAnalysis(
                category=EmailCategory.NEWSLETTER,
                priority=Priority.LOW,
                sentiment="neutral",
                intent="情報配信",
                summary=f"ニュースレター: {email['subject'][:30]}",
                suggested_action="既読にしてアーカイブ",
                confidence=0.95
            )

        # AI分析
        self.stats["ai_classified"] += 1
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

    def classify_batch(self, emails: list[dict]) -> list[EmailAnalysis]:
        """バッチ分類（コスト効率化）"""
        results = []
        ai_batch = []

        # ルールベースで処理可能なものを先に処理
        for email in emails:
            if self._is_obvious_spam(email):
                results.append((email["message_id"], EmailAnalysis(
                    category=EmailCategory.SPAM,
                    priority=Priority.LOW,
                    sentiment="neutral",
                    intent="スパム",
                    summary="スパムメール",
                    suggested_action="自動削除",
                    confidence=0.99
                )))
            elif self._is_newsletter(email):
                results.append((email["message_id"], EmailAnalysis(
                    category=EmailCategory.NEWSLETTER,
                    priority=Priority.LOW,
                    sentiment="neutral",
                    intent="情報配信",
                    summary=f"ニュースレター: {email['subject'][:30]}",
                    suggested_action="アーカイブ",
                    confidence=0.95
                )))
            else:
                ai_batch.append(email)

        # AI分類が必要なものをバッチ処理
        if ai_batch:
            batch_prompt = self._build_batch_prompt(ai_batch)
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": batch_prompt}]
            )
            batch_results = self._parse_batch_response(
                response.content[0].text, ai_batch
            )
            results.extend(batch_results)

        return results

    def _build_batch_prompt(self, emails: list[dict]) -> str:
        """バッチ分類用プロンプト生成"""
        email_texts = []
        for i, email in enumerate(emails):
            email_texts.append(
                f"--- メール {i+1} (ID: {email['message_id']}) ---\n"
                f"From: {email['sender']}\n"
                f"Subject: {email['subject']}\n"
                f"Body: {email['body'][:500]}\n"
            )

        return (
            "以下の複数メールをそれぞれ分類してください。\n"
            "各メールについて category, priority, sentiment, summary を"
            "JSON配列で返してください。\n\n"
            + "\n".join(email_texts)
        )

    def _is_obvious_spam(self, email: dict) -> bool:
        """ルールベーススパム判定"""
        spam_keywords = [
            "当選", "無料プレゼント", "今すぐクリック",
            "unsubscribe", "配信停止", "出会い系",
            "儲かる", "副業で月収", "限定オファー"
        ]
        subject_body = f"{email['subject']} {email['body']}".lower()
        match_count = sum(1 for kw in spam_keywords if kw in subject_body)
        return match_count >= 2

    def _is_newsletter(self, email: dict) -> bool:
        """ニュースレター判定"""
        newsletter_indicators = [
            "list-unsubscribe" in str(email.get("headers", {})).lower(),
            "noreply" in email["sender"].lower(),
            "newsletter" in email["sender"].lower(),
            "mail.substack.com" in email["sender"].lower(),
            "配信" in email.get("subject", ""),
        ]
        return sum(newsletter_indicators) >= 2

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
                confidence=data.get("confidence", 0.8),
                language=data.get("language", "ja"),
                topics=data.get("topics", []),
                entities=data.get("entities", [])
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

    def _parse_batch_response(self, text, emails):
        """バッチレスポンスのパース"""
        import json
        results = []
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            data_list = json.loads(text[start:end])
            for i, data in enumerate(data_list):
                msg_id = emails[i]["message_id"]
                analysis = EmailAnalysis(
                    category=EmailCategory(data.get("category", "personal")),
                    priority=Priority(data.get("priority", "medium")),
                    sentiment=data.get("sentiment", "neutral"),
                    intent=data.get("intent", ""),
                    summary=data.get("summary", ""),
                    suggested_action=data.get("suggested_action", "手動確認"),
                    confidence=data.get("confidence", 0.7)
                )
                results.append((msg_id, analysis))
        except Exception:
            for email in emails:
                results.append((email["message_id"], EmailAnalysis(
                    category=EmailCategory.PERSONAL,
                    priority=Priority.MEDIUM,
                    sentiment="neutral",
                    intent="バッチ分類失敗",
                    summary="手動確認が必要",
                    suggested_action="手動確認",
                    confidence=0.0
                )))
        return results

    def get_stats(self) -> dict:
        """分類統計情報の取得"""
        total = self.stats["total_classified"]
        if total == 0:
            return self.stats
        return {
            **self.stats,
            "rule_based_pct": round(
                self.stats["rule_based"] / total * 100, 1
            ),
            "ai_classified_pct": round(
                self.stats["ai_classified"] / total * 100, 1
            ),
        }
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

### 2.3 ハイブリッド分類戦略の詳細

```python
class HybridClassificationStrategy:
    """ルールベース + ML + LLM のハイブリッド分類戦略"""

    def __init__(self, classifier: EmailClassifier):
        self.classifier = classifier
        self.ml_model = None  # 学習済みモデル（scikit-learn等）
        self.confidence_threshold = 0.85

    def classify(self, email: dict) -> EmailAnalysis:
        """3段階のフォールバック分類"""

        # Stage 1: ルールベース（コスト0、即時）
        rule_result = self._rule_based_classify(email)
        if rule_result and rule_result.confidence >= 0.95:
            rule_result.suggested_action += " [ルールベース判定]"
            return rule_result

        # Stage 2: MLモデル（コスト極小、高速）
        if self.ml_model:
            ml_result = self._ml_classify(email)
            if ml_result and ml_result.confidence >= self.confidence_threshold:
                ml_result.suggested_action += " [ML判定]"
                return ml_result

        # Stage 3: LLM（コスト高、高精度）
        llm_result = self.classifier.classify(email)
        llm_result.suggested_action += " [LLM判定]"
        return llm_result

    def _rule_based_classify(self, email: dict) -> EmailAnalysis | None:
        """ルールベース分類"""
        subject = email.get("subject", "").lower()
        sender = email.get("sender", "").lower()
        body = email.get("body", "").lower()

        # 請求関連
        billing_keywords = ["請求", "invoice", "支払い", "料金", "プラン変更"]
        if any(kw in subject or kw in body[:200] for kw in billing_keywords):
            return EmailAnalysis(
                category=EmailCategory.BILLING,
                priority=Priority.HIGH,
                sentiment="neutral",
                intent="請求関連の問い合わせ",
                summary=f"請求関連: {email['subject'][:30]}",
                suggested_action="請求チームに転送",
                confidence=0.92
            )

        # 障害報告
        incident_keywords = [
            "障害", "ダウン", "エラー", "使えない",
            "動かない", "バグ", "500エラー", "タイムアウト"
        ]
        if any(kw in subject or kw in body[:300] for kw in incident_keywords):
            return EmailAnalysis(
                category=EmailCategory.TECHNICAL,
                priority=Priority.URGENT,
                sentiment="negative",
                intent="障害またはバグの報告",
                summary=f"障害報告: {email['subject'][:30]}",
                suggested_action="エンジニアリングチームに即時エスカレーション",
                confidence=0.90
            )

        # 解約・退会
        churn_keywords = ["解約", "退会", "キャンセル", "やめたい", "解除"]
        if any(kw in subject or kw in body[:300] for kw in churn_keywords):
            return EmailAnalysis(
                category=EmailCategory.SUPPORT,
                priority=Priority.HIGH,
                sentiment="negative",
                intent="解約・退会の意思表示",
                summary=f"解約要望: {email['subject'][:30]}",
                suggested_action="カスタマーサクセスに転送（リテンション対応）",
                confidence=0.93
            )

        return None

    def _ml_classify(self, email: dict) -> EmailAnalysis | None:
        """MLモデルによる分類（事前学習済み）"""
        if not self.ml_model:
            return None

        features = self._extract_features(email)
        prediction = self.ml_model.predict([features])
        confidence = max(self.ml_model.predict_proba([features])[0])

        if confidence < self.confidence_threshold:
            return None

        category_map = {
            0: EmailCategory.BILLING,
            1: EmailCategory.TECHNICAL,
            2: EmailCategory.SALES,
            3: EmailCategory.SUPPORT,
            4: EmailCategory.SPAM,
            5: EmailCategory.NEWSLETTER,
            6: EmailCategory.PERSONAL,
        }

        return EmailAnalysis(
            category=category_map.get(prediction[0], EmailCategory.PERSONAL),
            priority=Priority.MEDIUM,
            sentiment="neutral",
            intent="ML分類による判定",
            summary=email["subject"][:50],
            suggested_action="自動分類済み",
            confidence=float(confidence)
        )

    def _extract_features(self, email: dict) -> list:
        """メールから特徴量を抽出"""
        subject = email.get("subject", "")
        body = email.get("body", "")
        return [
            len(subject),
            len(body),
            body.count("?"),
            body.count("!"),
            1 if "noreply" in email.get("sender", "") else 0,
            len(email.get("cc", [])),
            len(email.get("attachments", [])),
        ]
```

### 2.4 分類カテゴリの詳細定義

```python
# 各カテゴリの詳細定義と自動アクションマッピング
CATEGORY_CONFIG = {
    "billing": {
        "display_name": "請求・支払い",
        "description": "料金、請求書、プラン変更、返金に関するメール",
        "auto_actions": [
            "請求チームのSlackチャネルに通知",
            "CRMに問い合わせチケット作成",
            "顧客の契約情報を自動取得して添付",
        ],
        "sla_hours": 4,
        "escalation_after_hours": 8,
        "template_replies": {
            "price_inquiry": "料金に関するお問い合わせありがとうございます...",
            "refund_request": "ご返金のご要望を承りました...",
            "plan_change": "プラン変更のお手続きについてご案内します...",
        },
        "keywords_ja": ["請求", "料金", "支払い", "返金", "プラン", "契約"],
        "keywords_en": ["billing", "invoice", "payment", "refund", "plan"],
    },
    "technical": {
        "display_name": "技術サポート",
        "description": "バグ報告、機能の使い方、API関連の技術的な質問",
        "auto_actions": [
            "技術サポートキューに追加",
            "関連するFAQ/ドキュメントを自動検索",
            "エラーログとの照合（直近24時間）",
        ],
        "sla_hours": 8,
        "escalation_after_hours": 24,
        "template_replies": {
            "bug_report": "バグのご報告ありがとうございます...",
            "how_to": "ご質問の機能についてご説明します...",
            "api_question": "API仕様についてお答えします...",
        },
        "keywords_ja": ["エラー", "バグ", "動かない", "使い方", "API"],
        "keywords_en": ["error", "bug", "not working", "how to", "API"],
    },
    "sales": {
        "display_name": "営業・セールス",
        "description": "新規問い合わせ、デモ依頼、見積もり要求",
        "auto_actions": [
            "CRMにリード登録",
            "営業チームにSlack通知",
            "会社情報を自動リサーチ",
        ],
        "sla_hours": 2,
        "escalation_after_hours": 4,
        "template_replies": {
            "demo_request": "デモのご要望ありがとうございます...",
            "pricing_inquiry": "お見積りについてご回答します...",
            "trial_request": "トライアルのご案内をいたします...",
        },
        "keywords_ja": ["導入", "検討", "デモ", "見積", "価格"],
        "keywords_en": ["demo", "pricing", "trial", "enterprise", "quote"],
    },
    "complaint": {
        "display_name": "クレーム・苦情",
        "description": "不満、苦情、改善要求の強い表現を含むメール",
        "auto_actions": [
            "マネージャーに即時通知",
            "過去のやり取り履歴を自動取得",
            "クレーム管理システムにエスカレーション登録",
        ],
        "sla_hours": 1,
        "escalation_after_hours": 2,
        "template_replies": {
            "service_complaint": "ご不便をおかけし誠に申し訳ございません...",
            "quality_issue": "品質に関するご指摘、真摯に受け止めます...",
        },
        "keywords_ja": ["不満", "ひどい", "最悪", "怒り", "訴える", "消費者庁"],
        "keywords_en": ["terrible", "worst", "unacceptable", "lawsuit"],
    },
}
```

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
- 1つのメールにつき1つの明確なアクションを提案
- 顧客の名前で呼びかける（分かる場合）

元メール:
From: {sender_name} <{sender_email}>
Subject: {subject}
Body:
{body}

過去のやり取り（あれば）:
{thread_history}

ナレッジベース参照（あれば）:
{knowledge_context}
"""

    def __init__(self, api_key: str, company_name: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.company_name = company_name
        self.templates = {}
        self.knowledge_base = None
        self.tone_configs = {
            "professional": {
                "description": "ビジネス標準の丁寧なトーン",
                "examples": ["お世話になっております", "ご確認いただけますと幸いです"],
                "avoid": ["了解です", "OK", "〜っす"],
            },
            "friendly": {
                "description": "親しみやすく柔らかいトーン",
                "examples": ["いつもありがとうございます！", "お気軽にご連絡ください"],
                "avoid": ["拝啓", "敬具", "小職"],
            },
            "formal": {
                "description": "非常にフォーマルなトーン（公式謝罪等）",
                "examples": ["誠に申し訳ございません", "深くお詫び申し上げます"],
                "avoid": ["すみません", "ごめんなさい"],
            },
            "empathetic": {
                "description": "共感を示すトーン（クレーム対応等）",
                "examples": [
                    "ご不便をおかけして申し訳ございません",
                    "お気持ちは十分に理解しております"
                ],
                "avoid": ["しかしながら", "ですが"],
            },
        }

    def generate_reply(self, email: dict,
                       tone: str = "professional",
                       thread_history: str = "",
                       knowledge_context: str = "") -> dict:
        """返信ドラフト生成"""
        signature = self._get_signature(email.get("assigned_to"))

        prompt = self.REPLY_PROMPT.format(
            company_name=self.company_name,
            tone=self._get_tone_instruction(tone),
            signature=signature,
            sender_name=email.get("sender_name", ""),
            sender_email=email["sender"],
            subject=email["subject"],
            body=email["body"],
            thread_history=thread_history or "なし",
            knowledge_context=knowledge_context or "なし"
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        reply_text = response.content[0].text
        confidence = self._assess_confidence(reply_text, email)

        return {
            "subject": f"Re: {email['subject']}",
            "body": reply_text,
            "status": self._determine_status(confidence, email),
            "confidence": confidence,
            "tone": tone,
            "tokens_used": (
                response.usage.input_tokens + response.usage.output_tokens
            ),
            "model": "claude-sonnet-4-20250514",
        }

    def generate_multiple_drafts(
        self, email: dict, tones: list[str] = None
    ) -> list[dict]:
        """複数トーンのドラフトを同時生成"""
        if tones is None:
            tones = ["professional", "friendly"]

        drafts = []
        for tone in tones:
            draft = self.generate_reply(email, tone=tone)
            draft["tone_label"] = self.tone_configs[tone]["description"]
            drafts.append(draft)

        return drafts

    def _get_tone_instruction(self, tone: str) -> str:
        """トーン設定の詳細指示を生成"""
        config = self.tone_configs.get(tone, self.tone_configs["professional"])
        return (
            f"{config['description']}\n"
            f"使用例: {', '.join(config['examples'])}\n"
            f"避ける表現: {', '.join(config['avoid'])}"
        )

    def _assess_confidence(self, reply: str, email: dict) -> float:
        """返信の信頼度判定"""
        score = 0.8

        # 長さチェック
        if len(reply) < 50:
            score -= 0.2  # 短すぎる
        elif len(reply) > 2000:
            score -= 0.1  # 長すぎる

        # 不確実性の表現
        uncertain_phrases = ["確認", "お調べ", "分かりかねます", "不明"]
        uncertainty_count = sum(
            1 for phrase in uncertain_phrases if phrase in reply
        )
        score -= uncertainty_count * 0.05

        # 緊急案件は人間確認推奨
        if email.get("priority") == "urgent":
            score -= 0.15

        # クレーム対応は慎重に
        if email.get("category") == "complaint":
            score -= 0.2

        # 金額に言及している場合は慎重に
        import re
        if re.search(r'[¥$€]\d+|円|ドル', reply):
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _determine_status(self, confidence: float, email: dict) -> str:
        """信頼度に基づくステータス決定"""
        category = email.get("category", "")
        priority = email.get("priority", "medium")

        # 高リスクカテゴリは常にドラフト
        if category in ["complaint", "billing"]:
            return "draft_review_required"

        # 緊急は人間レビュー必須
        if priority == "urgent":
            return "draft_review_required"

        # 信頼度に基づく判定
        if confidence >= 0.9:
            return "ready_to_send"
        elif confidence >= 0.7:
            return "draft"
        else:
            return "needs_human_review"

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
  高 ┤ ● 公式謝罪    ● 契約関連    ● 法務案件
     │
  中 ┤ ● 一般サポート ● ビジネス提案 ● 新規顧客対応
     │
  低 ┤ ● 社内連絡     ● カジュアル問い合わせ ● 既存顧客FU
     └──┬────────────┬────────────┬──────────┬──
       冷静         中立         親しみ      熱意
                感情トーン

  カテゴリ別推奨トーン:
  ┌───────────────┬──────────────┬────────────────┐
  │ カテゴリ       │ 推奨トーン    │ 注意点          │
  ├───────────────┼──────────────┼────────────────┤
  │ クレーム       │ formal +     │ まず謝罪から    │
  │               │ empathetic   │ 言い訳しない    │
  ├───────────────┼──────────────┼────────────────┤
  │ 技術サポート   │ professional │ 専門用語の説明  │
  │               │              │ 手順を明確に    │
  ├───────────────┼──────────────┼────────────────┤
  │ 営業問い合わせ │ friendly +   │ 押し売りしない  │
  │               │ professional │ 次のステップ提示│
  ├───────────────┼──────────────┼────────────────┤
  │ 解約防止      │ empathetic + │ 引き止めすぎない│
  │               │ professional │ 代替案を提示    │
  └───────────────┴──────────────┴────────────────┘
```

### 3.3 ナレッジベース連携による返信品質向上

```python
class KnowledgeAugmentedReplyEngine:
    """ナレッジベース連携の返信エンジン（RAG方式）"""

    def __init__(self, reply_engine: AutoReplyEngine, vector_store):
        self.reply_engine = reply_engine
        self.vector_store = vector_store  # Pinecone, Qdrant, pgvector等

    def generate_reply_with_knowledge(
        self, email: dict, top_k: int = 5
    ) -> dict:
        """ナレッジベースを参照した返信生成"""

        # 1. 関連ナレッジの検索
        query = f"{email['subject']} {email['body'][:300]}"
        relevant_docs = self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            filter={"type": {"$in": ["faq", "doc", "past_reply"]}}
        )

        # 2. コンテキスト構築
        knowledge_context = self._build_context(relevant_docs)

        # 3. 過去の類似対応の取得
        past_replies = self._find_similar_past_replies(email)

        # 4. ナレッジ付き返信生成
        reply = self.reply_engine.generate_reply(
            email=email,
            knowledge_context=knowledge_context,
            thread_history=past_replies
        )

        # 5. 参照元情報の付与
        reply["references"] = [
            {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "relevance_score": doc.metadata.get("score", 0),
            }
            for doc in relevant_docs
        ]

        return reply

    def _build_context(self, docs: list) -> str:
        """検索結果からコンテキスト文字列を構築"""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[参考{i}] {doc.metadata.get('title', '無題')}\n"
                f"{doc.page_content[:500]}\n"
            )
        return "\n---\n".join(context_parts)

    def _find_similar_past_replies(self, email: dict) -> str:
        """過去の類似メール対応を検索"""
        results = self.vector_store.similarity_search(
            query=email["body"][:200],
            top_k=3,
            filter={"type": "past_reply", "rating": {"$gte": 4}}
        )
        if not results:
            return "過去の類似対応なし"

        past = []
        for r in results:
            past.append(
                f"[過去の対応例]\n"
                f"元メール: {r.metadata.get('original_subject', '')}\n"
                f"返信: {r.page_content[:300]}..."
            )
        return "\n\n".join(past)

    def feedback_loop(self, reply_id: str, rating: int, comment: str = ""):
        """返信品質のフィードバックを記録し学習に活用"""
        self.vector_store.update_metadata(
            id=reply_id,
            metadata={
                "rating": rating,
                "feedback_comment": comment,
                "feedback_at": datetime.now().isoformat(),
            }
        )

        # 高評価の返信をナレッジベースに追加
        if rating >= 4:
            reply_data = self.vector_store.get(reply_id)
            self.vector_store.upsert(
                id=f"past_reply_{reply_id}",
                content=reply_data["content"],
                metadata={
                    "type": "past_reply",
                    "rating": rating,
                    "category": reply_data.get("category", ""),
                    "original_subject": reply_data.get("subject", ""),
                }
            )
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

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

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
        return {
            "summary": response.content[0].text,
            "message_count": len(messages),
            "participants": list({m["sender"] for m in messages}),
            "date_range": {
                "start": messages[0]["date"],
                "end": messages[-1]["date"],
            },
        }

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
                "content": f"以下のメール要約を統合して、最終的な要約を作成してください:\n{combined}"
            }]
        )
        return {
            "summary": response.content[0].text,
            "method": "hierarchical",
            "message_count": len(messages),
        }

    def _summarize_single(self, msg: dict) -> dict:
        """個別メールの要約"""
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # 個別要約はHaikuで十分
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": (
                    f"以下のメールを1-2文で要約:\n"
                    f"From: {msg['sender']}\n"
                    f"Date: {msg['date']}\n"
                    f"Body: {msg['body'][:1000]}"
                )
            }]
        )
        return {
            "from": msg["sender"],
            "date": msg["date"],
            "summary": response.content[0].text,
        }

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

    def generate_action_items(self, thread_summary: str) -> list[dict]:
        """要約からアクションアイテムを抽出"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "以下のメールスレッド要約からアクションアイテムを抽出し、"
                    "JSON配列で返してください。\n"
                    "各アイテム: {\"task\": \"タスク\", \"assignee\": \"担当者\", "
                    "\"deadline\": \"期限\", \"priority\": \"high/medium/low\"}\n\n"
                    f"要約:\n{thread_summary}"
                )
            }]
        )
        import json
        try:
            text = response.content[0].text
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except Exception:
            return []
```

### 4.2 会議録要約

```python
class MeetingSummarizer:
    """会議録AI要約"""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def summarize_meeting(self, transcript: str,
                          meeting_type: str = "general") -> dict:
        """会議録を構造化要約"""
        type_instructions = {
            "general": "一般的な会議の要約を作成",
            "standup": "デイリースタンドアップの要点を抽出",
            "sprint_review": "スプリントレビューの成果と課題を整理",
            "retrospective": "振り返りのKeep/Problem/Tryを整理",
            "sales": "商談の進捗と次のアクションを明確化",
            "one_on_one": "1on1の議論内容とフォローアップ事項を整理",
        }

        instruction = type_instructions.get(meeting_type, type_instructions["general"])

        prompt = f"""
以下の会議録を分析し、JSON形式で要約してください。
会議タイプ: {meeting_type}
指示: {instruction}

出力形式:
{{
  "title": "会議タイトル",
  "type": "{meeting_type}",
  "date": "日付",
  "participants": ["参加者リスト"],
  "duration": "所要時間",
  "summary": "3行要約",
  "key_points": ["主要論点"],
  "decisions": ["決定事項"],
  "action_items": [
    {{"task": "タスク内容", "assignee": "担当者", "deadline": "期限"}}
  ],
  "open_issues": ["未解決事項"],
  "risks": ["リスクや懸念事項"],
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

    def summarize_from_audio(self, audio_path: str) -> dict:
        """音声ファイルから会議要約を生成するパイプライン"""
        # Step 1: 音声→テキスト変換（Whisper等）
        transcript = self._transcribe_audio(audio_path)

        # Step 2: 話者分離（Diarization）
        diarized = self._diarize_transcript(transcript)

        # Step 3: AI要約
        return self.summarize_meeting(diarized)

    def _transcribe_audio(self, audio_path: str) -> str:
        """音声ファイルの文字起こし"""
        import openai
        client = openai.OpenAI()
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ja",
                response_format="verbose_json",
            )
        return transcript.text

    def _diarize_transcript(self, transcript: str) -> str:
        """話者分離（簡易実装）"""
        # 実際にはpyannote-audio等を使用
        return transcript

    def _parse_json(self, text: str) -> dict:
        """JSONパース（エラー耐性あり）"""
        import json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:
            return {"raw_summary": text, "parse_error": True}
```

### 4.3 Slackメッセージ要約

```python
class SlackSummarizer:
    """Slackチャネル/スレッドの要約エンジン"""

    def __init__(self, api_key: str, slack_token: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.slack_token = slack_token

    def summarize_channel(
        self, channel_id: str, hours: int = 24
    ) -> dict:
        """チャネルの直近N時間の要約"""
        import requests
        from datetime import datetime, timedelta

        oldest = (datetime.now() - timedelta(hours=hours)).timestamp()

        response = requests.get(
            "https://slack.com/api/conversations.history",
            headers={"Authorization": f"Bearer {self.slack_token}"},
            params={
                "channel": channel_id,
                "oldest": str(oldest),
                "limit": 200,
            }
        )
        messages = response.json().get("messages", [])

        if not messages:
            return {"summary": "この期間のメッセージはありません"}

        # メッセージの整形
        formatted = []
        for msg in reversed(messages):  # 時系列順に
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            ts = datetime.fromtimestamp(
                float(msg["ts"])
            ).strftime("%H:%M")
            formatted.append(f"[{ts}] {user}: {text}")

        thread_text = "\n".join(formatted)

        # AI要約
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"以下のSlackチャネルのメッセージを要約してください。\n"
                    f"期間: 直近{hours}時間\n"
                    f"出力: 概要、主要な議論、決定事項、アクションアイテム\n\n"
                    f"メッセージ:\n{thread_text}"
                )
            }]
        )

        return {
            "summary": response.content[0].text,
            "channel_id": channel_id,
            "period_hours": hours,
            "message_count": len(messages),
            "unique_participants": len({m.get("user") for m in messages}),
        }

    def daily_digest(self, channel_ids: list[str]) -> str:
        """複数チャネルのデイリーダイジェスト生成"""
        digests = []
        for ch_id in channel_ids:
            summary = self.summarize_channel(ch_id, hours=24)
            digests.append(summary)

        # 全チャネルの統合ダイジェスト
        combined = "\n\n".join(
            f"## #{d.get('channel_name', d['channel_id'])}\n"
            f"メッセージ数: {d['message_count']}\n"
            f"{d['summary']}"
            for d in digests
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    "以下の各チャネルの要約を統合して、"
                    "チーム全体のデイリーダイジェストを作成してください。\n"
                    "重要度順に整理し、注意が必要な事項を冒頭に配置。\n\n"
                    f"{combined}"
                )
            }]
        )

        return response.content[0].text
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

### 5.2 Gmail API統合の実装

```python
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


class GmailIntegration:
    """Gmail APIとの統合クラス"""

    def __init__(self, credentials_path: str):
        self.creds = Credentials.from_authorized_user_file(
            credentials_path,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.modify",
            ]
        )
        self.service = build("gmail", "v1", credentials=self.creds)

    def fetch_unread(self, max_results: int = 50) -> list[dict]:
        """未読メールの取得"""
        results = self.service.users().messages().list(
            userId="me",
            q="is:unread",
            maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        emails = []

        for msg_ref in messages:
            msg = self.service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="full"
            ).execute()

            email_data = self._parse_message(msg)
            emails.append(email_data)

        return emails

    def _parse_message(self, msg: dict) -> dict:
        """Gmailメッセージのパース"""
        headers = {
            h["name"].lower(): h["value"]
            for h in msg["payload"]["headers"]
        }

        body = ""
        if "parts" in msg["payload"]:
            for part in msg["payload"]["parts"]:
                if part["mimeType"] == "text/plain":
                    data = part["body"].get("data", "")
                    body = base64.urlsafe_b64decode(data).decode("utf-8")
                    break
        elif "body" in msg["payload"]:
            data = msg["payload"]["body"].get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8")

        return {
            "message_id": msg["id"],
            "thread_id": msg["threadId"],
            "sender": headers.get("from", ""),
            "sender_name": self._extract_name(headers.get("from", "")),
            "recipients": headers.get("to", "").split(","),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "body": body,
            "labels": msg.get("labelIds", []),
            "headers": headers,
        }

    def _extract_name(self, from_header: str) -> str:
        """From ヘッダーから名前を抽出"""
        if "<" in from_header:
            return from_header.split("<")[0].strip().strip('"')
        return from_header

    def send_reply(self, original_msg_id: str,
                   thread_id: str,
                   to: str,
                   subject: str,
                   body: str) -> dict:
        """返信メールの送信"""
        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to
        message["subject"] = subject
        message["In-Reply-To"] = original_msg_id
        message["References"] = original_msg_id

        raw = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        result = self.service.users().messages().send(
            userId="me",
            body={
                "raw": raw,
                "threadId": thread_id,
            }
        ).execute()

        return result

    def add_label(self, message_id: str, label_name: str):
        """メールにラベルを追加"""
        # ラベルID取得（既存 or 新規作成）
        label_id = self._get_or_create_label(label_name)

        self.service.users().messages().modify(
            userId="me",
            id=message_id,
            body={
                "addLabelIds": [label_id],
            }
        ).execute()

    def _get_or_create_label(self, label_name: str) -> str:
        """ラベルの取得または作成"""
        labels = self.service.users().labels().list(
            userId="me"
        ).execute()

        for label in labels.get("labels", []):
            if label["name"] == label_name:
                return label["id"]

        # ラベル新規作成
        new_label = self.service.users().labels().create(
            userId="me",
            body={
                "name": label_name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            }
        ).execute()

        return new_label["id"]

    def mark_as_read(self, message_id: str):
        """既読にする"""
        self.service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]}
        ).execute()
```

### 5.3 n8nワークフロー定義

```json
{
  "name": "AI Email Automation Pipeline",
  "nodes": [
    {
      "name": "Gmail Trigger",
      "type": "n8n-nodes-base.gmailTrigger",
      "parameters": {
        "pollTimes": { "item": [{ "mode": "everyMinute", "minute": 5 }] },
        "filters": { "readStatus": "unread" }
      },
      "position": [100, 300]
    },
    {
      "name": "Extract Email Data",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": {
          "string": [
            { "name": "sender", "value": "={{ $json.from }}" },
            { "name": "subject", "value": "={{ $json.subject }}" },
            { "name": "body", "value": "={{ $json.textPlain }}" }
          ]
        }
      },
      "position": [300, 300]
    },
    {
      "name": "AI Classification",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "https://api.anthropic.com/v1/messages",
        "headers": {
          "x-api-key": "={{ $credentials.anthropicApiKey }}",
          "content-type": "application/json",
          "anthropic-version": "2023-06-01"
        },
        "body": "={{ JSON.stringify({ model: 'claude-sonnet-4-20250514', max_tokens: 512, messages: [{ role: 'user', content: 'メールを分類: ' + $json.body }] }) }}"
      },
      "position": [500, 300]
    },
    {
      "name": "Route by Category",
      "type": "n8n-nodes-base.switch",
      "parameters": {
        "rules": [
          { "value": "spam", "output": 0 },
          { "value": "support", "output": 1 },
          { "value": "sales", "output": 2 },
          { "value": "billing", "output": 3 }
        ]
      },
      "position": [700, 300]
    },
    {
      "name": "Slack Notification",
      "type": "n8n-nodes-base.slack",
      "parameters": {
        "channel": "#support-alerts",
        "text": "新規メール: {{ $json.subject }} from {{ $json.sender }}"
      },
      "position": [900, 200]
    },
    {
      "name": "CRM Update",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "https://api.hubspot.com/crm/v3/objects/contacts",
        "body": "={{ JSON.stringify({ properties: { email: $json.sender } }) }}"
      },
      "position": [900, 400]
    }
  ]
}
```

### 5.4 Microsoft Graph API統合

```python
import msal
import requests


class OutlookIntegration:
    """Microsoft Graph APIとのOutlook統合"""

    GRAPH_API_URL = "https://graph.microsoft.com/v1.0"

    def __init__(self, client_id: str, client_secret: str, tenant_id: str):
        self.app = msal.ConfidentialClientApplication(
            client_id,
            authority=f"https://login.microsoftonline.com/{tenant_id}",
            client_credential=client_secret,
        )
        self._token = None

    def _get_token(self) -> str:
        """アクセストークン取得"""
        if self._token:
            return self._token

        result = self.app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        self._token = result["access_token"]
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    def fetch_messages(self, user_id: str,
                       folder: str = "inbox",
                       unread_only: bool = True,
                       top: int = 50) -> list[dict]:
        """メッセージ取得"""
        params = {
            "$top": top,
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,toRecipients,body,receivedDateTime,isRead",
        }
        if unread_only:
            params["$filter"] = "isRead eq false"

        response = requests.get(
            f"{self.GRAPH_API_URL}/users/{user_id}/mailFolders/{folder}/messages",
            headers=self._headers(),
            params=params,
        )
        response.raise_for_status()

        messages = response.json().get("value", [])
        return [self._normalize_message(msg) for msg in messages]

    def _normalize_message(self, msg: dict) -> dict:
        """Graph APIレスポンスを共通形式に変換"""
        sender = msg.get("from", {}).get("emailAddress", {})
        return {
            "message_id": msg["id"],
            "sender": sender.get("address", ""),
            "sender_name": sender.get("name", ""),
            "subject": msg.get("subject", ""),
            "body": msg.get("body", {}).get("content", ""),
            "date": msg.get("receivedDateTime", ""),
            "is_read": msg.get("isRead", False),
            "recipients": [
                r["emailAddress"]["address"]
                for r in msg.get("toRecipients", [])
            ],
        }

    def send_reply(self, user_id: str, message_id: str,
                   reply_body: str) -> dict:
        """返信送信"""
        response = requests.post(
            f"{self.GRAPH_API_URL}/users/{user_id}/messages/{message_id}/reply",
            headers=self._headers(),
            json={
                "message": {
                    "body": {
                        "contentType": "Text",
                        "content": reply_body,
                    }
                }
            }
        )
        response.raise_for_status()
        return {"status": "sent", "message_id": message_id}

    def create_category(self, user_id: str, display_name: str,
                        color: str = "preset0") -> dict:
        """カテゴリ作成"""
        response = requests.post(
            f"{self.GRAPH_API_URL}/users/{user_id}/outlook/masterCategories",
            headers=self._headers(),
            json={
                "displayName": display_name,
                "color": color,
            }
        )
        return response.json()
```

---

## 6. セキュリティとコンプライアンス

### 6.1 PII（個人情報）マスキング

```python
import re
from typing import Tuple


class PIIMasker:
    """個人情報の自動マスキング"""

    PATTERNS = {
        "email": {
            "pattern": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "replacement": "[EMAIL_MASKED]",
            "description": "メールアドレス",
        },
        "phone_jp": {
            "pattern": r'0\d{1,4}-?\d{1,4}-?\d{4}',
            "replacement": "[PHONE_MASKED]",
            "description": "日本の電話番号",
        },
        "credit_card": {
            "pattern": r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
            "replacement": "[CARD_MASKED]",
            "description": "クレジットカード番号",
        },
        "my_number": {
            "pattern": r'\d{4}\s?\d{4}\s?\d{4}',
            "replacement": "[MYNUMBER_MASKED]",
            "description": "マイナンバー",
        },
        "postal_code": {
            "pattern": r'〒?\d{3}-?\d{4}',
            "replacement": "[POSTAL_MASKED]",
            "description": "郵便番号",
        },
        "ip_address": {
            "pattern": r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
            "replacement": "[IP_MASKED]",
            "description": "IPアドレス",
        },
        "bank_account": {
            "pattern": r'[普通|当座]\s?\d{7,8}',
            "replacement": "[ACCOUNT_MASKED]",
            "description": "銀行口座番号",
        },
    }

    def mask(self, text: str,
             categories: list[str] = None) -> Tuple[str, list[dict]]:
        """テキスト中のPIIをマスキング"""
        masked_text = text
        detections = []

        patterns_to_check = (
            {k: v for k, v in self.PATTERNS.items() if k in categories}
            if categories
            else self.PATTERNS
        )

        for pii_type, config in patterns_to_check.items():
            matches = list(re.finditer(config["pattern"], masked_text))
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "description": config["description"],
                    "position": match.span(),
                    "original_length": len(match.group()),
                })
            masked_text = re.sub(
                config["pattern"],
                config["replacement"],
                masked_text
            )

        return masked_text, detections

    def unmask(self, masked_text: str,
               original_text: str,
               detections: list[dict]) -> str:
        """マスキング解除（権限チェック後に使用）"""
        result = masked_text
        for det in reversed(detections):
            start, end = det["position"]
            original_value = original_text[start:end]
            result = result.replace(
                self.PATTERNS[det["type"]]["replacement"],
                original_value,
                1
            )
        return result


# 使用例
masker = PIIMasker()
text = "山田太郎様 (taro@example.com) へ請求書を送付。電話: 090-1234-5678"
masked, detections = masker.mask(text)
# → "山田太郎様 ([EMAIL_MASKED]) へ請求書を送付。電話: [PHONE_MASKED]"
```

### 6.2 監査ログ設計

```python
from datetime import datetime
from enum import Enum
import json
import hashlib


class AuditAction(Enum):
    EMAIL_RECEIVED = "email_received"
    EMAIL_CLASSIFIED = "email_classified"
    REPLY_GENERATED = "reply_generated"
    REPLY_SENT = "reply_sent"
    REPLY_EDITED = "reply_edited"
    ESCALATED = "escalated"
    PII_DETECTED = "pii_detected"
    PII_MASKED = "pii_masked"
    DATA_EXPORTED = "data_exported"
    SETTINGS_CHANGED = "settings_changed"


class AuditLogger:
    """メール自動化の監査ログ"""

    def __init__(self, db_client):
        self.db = db_client

    def log(self, action: AuditAction, details: dict,
            user_id: str = "system",
            email_id: str = None) -> str:
        """監査ログの記録"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action.value,
            "user_id": user_id,
            "email_id": email_id,
            "details": details,
            "checksum": self._compute_checksum(details),
        }

        # DB保存
        result = self.db.table("audit_logs").insert(log_entry).execute()
        return result.data[0]["id"]

    def _compute_checksum(self, details: dict) -> str:
        """改ざん検知用チェックサム"""
        content = json.dumps(details, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def query_logs(self, filters: dict = None,
                   start_date: str = None,
                   end_date: str = None,
                   limit: int = 100) -> list[dict]:
        """監査ログの検索"""
        query = self.db.table("audit_logs").select("*")

        if filters:
            for key, value in filters.items():
                query = query.eq(key, value)
        if start_date:
            query = query.gte("timestamp", start_date)
        if end_date:
            query = query.lte("timestamp", end_date)

        query = query.order("timestamp", desc=True).limit(limit)
        return query.execute().data

    def generate_compliance_report(
        self, start_date: str, end_date: str
    ) -> dict:
        """コンプライアンスレポート生成"""
        logs = self.query_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        report = {
            "period": {"start": start_date, "end": end_date},
            "total_emails_processed": sum(
                1 for l in logs if l["action"] == "email_received"
            ),
            "auto_replies_sent": sum(
                1 for l in logs if l["action"] == "reply_sent"
            ),
            "human_escalations": sum(
                1 for l in logs if l["action"] == "escalated"
            ),
            "pii_detections": sum(
                1 for l in logs if l["action"] == "pii_detected"
            ),
            "data_exports": sum(
                1 for l in logs if l["action"] == "data_exported"
            ),
            "settings_changes": [
                l for l in logs if l["action"] == "settings_changed"
            ],
        }

        total = report["total_emails_processed"]
        if total > 0:
            report["auto_reply_rate"] = round(
                report["auto_replies_sent"] / total * 100, 1
            )
            report["escalation_rate"] = round(
                report["human_escalations"] / total * 100, 1
            )

        return report
```

### 6.3 GDPR/APPI対応チェックリスト

```
メール自動化におけるデータ保護対応:

  ■ データ収集と利用
  ┌──────────────────────────────────────────────────┐
  │ □ メール内容をAI APIに送信する旨をプライバシー    │
  │   ポリシーに明記                                  │
  │ □ データ処理の法的根拠を明確化                     │
  │   (同意 / 正当な利益 / 契約の履行)                 │
  │ □ 利用目的を具体的に列挙                          │
  │   (分類、返信生成、要約、品質改善)                 │
  │ □ 第三者提供先（AI APIプロバイダ）の明示           │
  └──────────────────────────────────────────────────┘

  ■ データ保持と削除
  ┌──────────────────────────────────────────────────┐
  │ □ メール本文の保持期間を定義（推奨: 90日以内）      │
  │ □ AI APIプロバイダのデータ保持ポリシーを確認        │
  │   (Anthropic: 入力データを学習に使用しない)         │
  │ □ 自動削除のスケジュール設定                       │
  │ □ ユーザーからの削除要求への対応手順               │
  │ □ バックアップデータの削除手順                     │
  └──────────────────────────────────────────────────┘

  ■ セキュリティ対策
  ┌──────────────────────────────────────────────────┐
  │ □ 通信の暗号化（TLS 1.3）                        │
  │ □ 保存データの暗号化（AES-256）                   │
  │ □ PIIマスキングの適用                             │
  │ □ アクセス制御（RBAC）の実装                      │
  │ □ 監査ログの記録と保持                            │
  │ □ インシデント対応手順の策定                       │
  │ □ 定期的なセキュリティ監査の実施                   │
  └──────────────────────────────────────────────────┘

  ■ ユーザーの権利保護
  ┌──────────────────────────────────────────────────┐
  │ □ アクセス権: 保存データの開示請求への対応         │
  │ □ 訂正権: 誤ったデータの修正手順                  │
  │ □ 削除権: データ削除要求への対応（忘れられる権利）  │
  │ □ 異議申立権: AI自動処理への異議申立手順           │
  │ □ データポータビリティ: データエクスポート機能      │
  │ □ オプトアウト: AI処理を拒否するオプション         │
  └──────────────────────────────────────────────────┘
```

---

## 7. 運用モニタリングとKPI

### 7.1 KPIダッシュボード設計

```python
class EmailAutomationDashboard:
    """メール自動化のKPIダッシュボード"""

    def __init__(self, db_client):
        self.db = db_client

    def get_daily_metrics(self, date: str) -> dict:
        """日次メトリクスの取得"""
        return {
            "date": date,
            "volume": {
                "total_received": self._count_emails(date, "received"),
                "auto_classified": self._count_emails(date, "classified"),
                "auto_replied": self._count_emails(date, "auto_replied"),
                "human_handled": self._count_emails(date, "human_handled"),
                "escalated": self._count_emails(date, "escalated"),
            },
            "quality": {
                "classification_accuracy": self._calc_accuracy(date),
                "reply_approval_rate": self._calc_approval_rate(date),
                "false_positive_rate": self._calc_false_positive(date),
                "customer_satisfaction": self._calc_csat(date),
            },
            "performance": {
                "avg_classification_time_ms": self._avg_time(date, "classify"),
                "avg_reply_generation_time_ms": self._avg_time(date, "reply"),
                "avg_first_response_time_min": self._avg_response_time(date),
                "p95_response_time_min": self._p95_response_time(date),
            },
            "cost": {
                "total_api_cost_usd": self._total_api_cost(date),
                "cost_per_email_usd": self._cost_per_email(date),
                "tokens_used": self._total_tokens(date),
                "estimated_monthly_cost": self._total_api_cost(date) * 30,
            },
            "category_breakdown": self._category_breakdown(date),
        }

    def _count_emails(self, date: str, status: str) -> int:
        """ステータス別メール数カウント"""
        result = self.db.table("email_logs") \
            .select("id", count="exact") \
            .eq("date", date) \
            .eq("status", status) \
            .execute()
        return result.count or 0

    def _calc_accuracy(self, date: str) -> float:
        """分類精度の計算（人間フィードバックベース）"""
        result = self.db.table("classification_feedback") \
            .select("correct") \
            .eq("date", date) \
            .execute()

        if not result.data:
            return 0.0

        correct = sum(1 for r in result.data if r["correct"])
        return round(correct / len(result.data) * 100, 1)

    def _calc_approval_rate(self, date: str) -> float:
        """AI返信の承認率（ドラフトが変更なしで送信された割合）"""
        result = self.db.table("reply_logs") \
            .select("status, edited") \
            .eq("date", date) \
            .execute()

        if not result.data:
            return 0.0

        approved = sum(
            1 for r in result.data
            if r["status"] == "sent" and not r["edited"]
        )
        return round(approved / len(result.data) * 100, 1)

    def _calc_false_positive(self, date: str) -> float:
        """偽陽性率（誤って自動返信されたメールの割合）"""
        result = self.db.table("reply_logs") \
            .select("feedback_score") \
            .eq("date", date) \
            .eq("auto_sent", True) \
            .execute()

        if not result.data:
            return 0.0

        false_positives = sum(
            1 for r in result.data
            if r.get("feedback_score", 5) <= 2
        )
        return round(false_positives / len(result.data) * 100, 1)

    def _category_breakdown(self, date: str) -> dict:
        """カテゴリ別の内訳"""
        result = self.db.table("email_logs") \
            .select("category, priority") \
            .eq("date", date) \
            .execute()

        breakdown = {}
        for r in (result.data or []):
            cat = r["category"]
            if cat not in breakdown:
                breakdown[cat] = {"count": 0, "priority_dist": {}}
            breakdown[cat]["count"] += 1
            pri = r.get("priority", "medium")
            breakdown[cat]["priority_dist"][pri] = \
                breakdown[cat]["priority_dist"].get(pri, 0) + 1

        return breakdown

    # 他のメソッドは省略（各メトリクスのDB集計）
    def _avg_time(self, date, op): return 0
    def _avg_response_time(self, date): return 0
    def _p95_response_time(self, date): return 0
    def _total_api_cost(self, date): return 0.0
    def _cost_per_email(self, date): return 0.0
    def _total_tokens(self, date): return 0
    def _calc_csat(self, date): return 0.0


    def generate_weekly_report(self, start_date: str,
                               end_date: str) -> str:
        """週次レポートの生成"""
        # 各日のメトリクスを集計
        daily_metrics = []
        from datetime import datetime, timedelta
        current = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        while current <= end:
            metrics = self.get_daily_metrics(current.date().isoformat())
            daily_metrics.append(metrics)
            current += timedelta(days=1)

        # サマリー計算
        total_received = sum(
            m["volume"]["total_received"] for m in daily_metrics
        )
        total_auto_replied = sum(
            m["volume"]["auto_replied"] for m in daily_metrics
        )
        avg_accuracy = sum(
            m["quality"]["classification_accuracy"] for m in daily_metrics
        ) / len(daily_metrics) if daily_metrics else 0

        total_cost = sum(
            m["cost"]["total_api_cost_usd"] for m in daily_metrics
        )

        report = f"""
## メール自動化 週次レポート
期間: {start_date} 〜 {end_date}

### ボリューム
- 総受信数: {total_received}通
- 自動返信: {total_auto_replied}通 ({total_auto_replied/max(total_received,1)*100:.1f}%)

### 品質
- 分類精度: {avg_accuracy:.1f}%

### コスト
- 総API費用: ${total_cost:.2f}
- 1通あたり: ${total_cost/max(total_received,1):.4f}
"""
        return report
```

### 7.2 アラート設定

```python
class AlertManager:
    """メール自動化のアラート管理"""

    def __init__(self, slack_webhook_url: str):
        self.webhook_url = slack_webhook_url
        self.alert_rules = [
            {
                "name": "分類精度低下",
                "condition": lambda m: m["quality"]["classification_accuracy"] < 90,
                "severity": "warning",
                "message": "分類精度が90%を下回っています: {value}%",
            },
            {
                "name": "応答時間超過",
                "condition": lambda m: m["performance"]["avg_first_response_time_min"] > 30,
                "severity": "warning",
                "message": "平均応答時間が30分を超えています: {value}分",
            },
            {
                "name": "APIコスト急増",
                "condition": lambda m: m["cost"]["total_api_cost_usd"] > 50,
                "severity": "critical",
                "message": "日次APIコストが$50を超えています: ${value}",
            },
            {
                "name": "エスカレーション急増",
                "condition": lambda m: (
                    m["volume"]["escalated"] / max(m["volume"]["total_received"], 1) > 0.3
                ),
                "severity": "warning",
                "message": "エスカレーション率が30%を超えています: {value}%",
            },
            {
                "name": "偽陽性率上昇",
                "condition": lambda m: m["quality"]["false_positive_rate"] > 5,
                "severity": "critical",
                "message": "偽陽性率が5%を超えています: {value}%。自動送信を一時停止してください。",
            },
        ]

    def check_and_alert(self, metrics: dict):
        """メトリクスをチェックしてアラート送信"""
        import requests

        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    self._send_alert(rule, metrics)
            except Exception as e:
                print(f"Alert check failed for {rule['name']}: {e}")

    def _send_alert(self, rule: dict, metrics: dict):
        """Slackにアラート送信"""
        import requests

        severity_emoji = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }

        emoji = severity_emoji.get(rule["severity"], ":bell:")

        payload = {
            "text": (
                f"{emoji} *メール自動化アラート: {rule['name']}*\n"
                f"重要度: {rule['severity']}\n"
                f"{rule['message']}\n"
                f"日時: {metrics.get('date', 'N/A')}"
            )
        }

        requests.post(self.webhook_url, json=payload)
```

---

## 8. アンチパターン

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

### アンチパターン3: エラーハンドリングの欠如

```python
# BAD: API障害時にメールが処理されない
def process_email(email):
    result = ai.classify(email)  # APIがダウンしたら全停止
    reply = ai.generate_reply(email)
    send_email(reply)

# GOOD: フォールバックと再試行の実装
def process_email(email):
    try:
        result = ai.classify(email)
    except APIError as e:
        # フォールバック: ルールベース分類
        result = rule_based_classify(email)
        log_error("AI classification failed, using rule-based", e)

    try:
        reply = ai.generate_reply(email)
    except APIError as e:
        # フォールバック: テンプレート返信
        reply = template_reply(result.category, email)
        log_error("AI reply generation failed, using template", e)

    # 再試行キューに入れる
    if result.confidence < 0.5:
        retry_queue.add(email, retry_after_minutes=30)
        return

    send_with_retry(reply, max_retries=3, backoff_seconds=5)
```

### アンチパターン4: 一律のAIモデル使用

```python
# BAD: 全てのメールに高コストモデルを使用
def classify_all(emails):
    for email in emails:
        # 1通あたり$0.01のコスト × 10,000通/日 = $100/日
        result = claude_opus.classify(email)

# GOOD: タスクの複雑度に応じたモデル選択
def classify_all(emails):
    for email in emails:
        # Stage 1: ルールベース（コスト$0）
        if is_obvious_category(email):
            yield rule_based_result(email)
            continue

        # Stage 2: 軽量モデル（コスト$0.001）
        haiku_result = claude_haiku.classify(email)
        if haiku_result.confidence >= 0.90:
            yield haiku_result
            continue

        # Stage 3: 高精度モデル（コスト$0.01）— 必要な場合のみ
        yield claude_sonnet.classify(email)
```

### アンチパターン5: ログと監査の不備

```python
# BAD: AI処理の記録を残さない
def auto_reply(email):
    reply = ai.generate(email)
    send(reply)  # 何が送られたか後から確認できない

# GOOD: 全ての処理を監査可能な形で記録
def auto_reply(email):
    # 入力の記録
    audit.log(AuditAction.EMAIL_RECEIVED, {
        "message_id": email["id"],
        "sender": email["sender"],
        "subject": email["subject"],
    })

    # PII検出とマスキング
    masked_body, pii_detections = masker.mask(email["body"])
    if pii_detections:
        audit.log(AuditAction.PII_DETECTED, {
            "types": [d["type"] for d in pii_detections],
            "count": len(pii_detections),
        })

    # AI処理
    reply = ai.generate(masked_body)
    audit.log(AuditAction.REPLY_GENERATED, {
        "confidence": reply["confidence"],
        "model": reply["model"],
        "tokens": reply["tokens_used"],
    })

    # 送信
    if reply["confidence"] >= 0.9:
        send(reply)
        audit.log(AuditAction.REPLY_SENT, {
            "auto": True,
            "reply_length": len(reply["body"]),
        })
    else:
        save_draft(reply)
        audit.log(AuditAction.REPLY_GENERATED, {
            "status": "draft",
            "reason": "low_confidence",
        })
```

---

## 9. 段階的導入ロードマップ

### 9.1 4フェーズ導入計画

```
Phase 1 (Week 1-2): 観察モード
┌──────────────────────────────────────────┐
│ ● メール受信のフック設定                   │
│ ● AI分類を実行するが結果はログのみ         │
│ ● 実際の処理は変更なし（シャドウモード）    │
│ ● 分類精度のベンチマーク取得               │
│ 成功基準: 分類精度 85%以上                 │
└──────────────────────────────────────────┘
          │
          ▼
Phase 2 (Week 3-4): アシストモード
┌──────────────────────────────────────────┐
│ ● 分類結果をラベル/タグとして付与          │
│ ● 返信ドラフトを生成してサジェスト         │
│ ● スレッド要約を自動生成                   │
│ ● 人間が全ての送信を承認                   │
│ 成功基準: ドラフト承認率 70%以上           │
└──────────────────────────────────────────┘
          │
          ▼
Phase 3 (Week 5-8): 半自動モード
┌──────────────────────────────────────────┐
│ ● 低リスクメール（ニュースレター、FAQ）の   │
│   自動処理                                │
│ ● 高信頼度ドラフトのワンクリック送信       │
│ ● 自動エスカレーションルールの適用         │
│ 成功基準: 自動処理率 40%以上              │
└──────────────────────────────────────────┘
          │
          ▼
Phase 4 (Week 9+): 完全自動モード
┌──────────────────────────────────────────┐
│ ● 信頼度90%以上の返信を自動送信           │
│ ● 異常検知による自動エスカレーション       │
│ ● フィードバックループによる継続的改善     │
│ ● 月次レビューで精度と方針を調整           │
│ 成功基準: 自動処理率 60%以上、CSAT維持    │
└──────────────────────────────────────────┘
```

### 9.2 各フェーズの詳細チェックリスト

```python
deployment_checklist = {
    "phase_1_observe": {
        "duration": "2週間",
        "tasks": [
            "メールプロバイダとのAPI接続設定",
            "AI分類パイプラインの構築（ログのみモード）",
            "分類結果と人間の判断を比較するダッシュボード構築",
            "ベースライン精度の測定（最低200通で評価）",
            "PIIマスキングの動作確認",
            "エラーハンドリングとリトライ機構のテスト",
        ],
        "exit_criteria": [
            "分類精度 85%以上（人間判断と比較）",
            "システムの安定稼働確認（エラー率 < 1%）",
            "PII検出漏れなし（テストデータで検証）",
        ],
    },
    "phase_2_assist": {
        "duration": "2週間",
        "tasks": [
            "分類結果のラベル自動付与",
            "返信ドラフト生成機能の有効化",
            "ドラフト承認UIの構築",
            "フィードバック収集機能の実装",
            "チーム向けトレーニング実施",
            "ナレッジベースの初期構築",
        ],
        "exit_criteria": [
            "ドラフト承認率 70%以上",
            "担当者の平均対応時間 30%以上短縮",
            "チームメンバーの満足度調査で好意的回答 80%以上",
        ],
    },
    "phase_3_semi_auto": {
        "duration": "4週間",
        "tasks": [
            "低リスクカテゴリの自動処理ルール設定",
            "自動送信の信頼度閾値チューニング",
            "エスカレーションフローの最適化",
            "コスト最適化（モデル選択、キャッシュ）",
            "カスタマー向け通知設定（AI対応の透明性）",
            "週次レビューミーティングの開始",
        ],
        "exit_criteria": [
            "自動処理率 40%以上",
            "偽陽性率（誤送信） 2%以下",
            "顧客満足度の維持または向上",
            "月次APIコストが予算内",
        ],
    },
    "phase_4_full_auto": {
        "duration": "継続",
        "tasks": [
            "高信頼度返信の自動送信有効化",
            "フィードバックループの自動化",
            "月次精度レビューと閾値調整",
            "新パターンの検出と対応ルール追加",
            "コンプライアンス監査の定期実施",
            "障害対応手順の文書化とドリル",
        ],
        "exit_criteria": [
            "自動処理率 60%以上",
            "分類精度 95%以上維持",
            "CSAT（顧客満足度）の維持",
            "運用工数 50%以上削減",
        ],
    },
}
```

---

## 10. FAQ

### Q1: 自動返信で顧客満足度は下がらない？

**A:** 適切に実装すれば むしろ向上する。調査によると、(1) 応答速度が10分→30秒に短縮され顧客満足度15%向上、(2) 24時間対応が可能に、(3) 回答の一貫性が保たれる。ただし「AIが対応している」旨を隠さないこと、複雑な案件は速やかに人間にエスカレーションすることが条件。段階的な導入（まず分類→要約→ドラフト→自動返信）で品質を確認しながら進めることが重要。

### Q2: プライバシーとセキュリティの対策は？

**A:** 3層の対策を推奨。(1) データマスキング — メール送信前にPII（個人情報）を自動マスク、(2) API選択 — データ保持しないAPI（Anthropic API等）を使用、(3) オンプレミス — 機密度の高い業界はセルフホストLLM（Llama等）を検討。GDPRやAPPI（日本の個人情報保護法）の要件も確認すること。特に医療、金融、法務分野では規制当局への事前確認が必要になる場合がある。

### Q3: 既存メールシステムとの統合方法は？

**A:** 3つのアプローチがある。(1) IMAP/SMTP直接 — 最も柔軟だが実装コスト高、(2) Gmail API / Microsoft Graph API — Google/Microsoft環境なら最適、(3) Zapier/n8n経由 — 最も簡単で即日稼働可能。おすすめは(3)で始めて、規模拡大後に(2)へ移行する段階的アプローチ。

### Q4: AIモデルのコストを抑えるには？

**A:** 5つの主要な最適化戦略がある。(1) ハイブリッド分類 — ルールベースで処理可能なメールをAI呼び出し前にフィルタリングする（スパム、ニュースレター等で30-50%削減可能）。(2) モデル選択 — 分類にはHaiku（低コスト）、返信生成にはSonnet（高品質）と使い分ける。(3) キャッシング — 類似の問い合わせに対する回答をキャッシュし、同一回答の再生成を防ぐ。(4) バッチ処理 — 複数メールをまとめて1回のAPI呼び出しで分類する。(5) プロンプト最適化 — 入力トークン数を最小化する（メール本文の先頭500文字のみ使用等）。

### Q5: 多言語メールへの対応は？

**A:** Claude/GPT-4は100以上の言語をサポートしているため、多言語メールの分類と返信生成は標準機能で対応可能。実装のポイント: (1) まずメールの言語を検出し、分類プロンプトとは別に処理する。(2) 返信は元メールと同じ言語で生成するよう指示する。(3) 社内テンプレートを主要言語（日本語、英語、中国語等）で準備しておく。(4) 翻訳が必要な場合は、DeepL APIとの併用も検討する（特にフォーマルな文書の場合）。

### Q6: 自動化の効果測定はどうする？

**A:** 導入前後のメトリクスを比較する。主要KPI: (1) 平均初回応答時間（MTTR）— 導入前10分→導入後30秒が目安、(2) 1通あたりの対応コスト — 人件費+APIコストで計算、(3) 顧客満足度（CSAT）— 導入前後で維持または向上、(4) 担当者の対応件数 — 1人あたり2-3倍の処理能力向上が期待できる、(5) エスカレーション率 — 低いほど自動化が成功している証拠。ROI計算は「人件費削減額 - APIコスト - 開発コスト」で行う。

### Q7: チームの抵抗をどう克服する？

**A:** 段階的な導入とメリットの可視化が鍵。(1) 最初はAIをアシスタント（ドラフト生成）として位置づけ、人間の判断を尊重する。(2) 「仕事を奪う」ではなく「定型業務を自動化して、より価値の高い業務に集中できる」と説明する。(3) パイロットチームで成功事例を作り、他チームへ展開する。(4) フィードバックを積極的に収集し、改善に反映する。(5) 対応品質が向上した具体的な数値（応答速度、CSAT等）を共有する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 分類 | ルールベース + AI のハイブリッドが最良 |
| 自動返信 | 信頼度スコアで自動/ドラフト/人間を切り替え |
| 要約 | 長いスレッドは段階的要約（個別→統合） |
| マルチチャネル | Slack/Teams/メールを統合した一元管理 |
| セキュリティ | PIIマスキング + データ非保持API + 暗号化 + 監査ログ |
| 導入順序 | 観察→アシスト→半自動→完全自動の4フェーズ |
| コスト最適化 | ハイブリッド分類 + モデル使い分け + キャッシュ |
| KPI | 応答時間、解決率、顧客満足度、コスト/通、偽陽性率 |

---

## 次に読むべきガイド

- [../01-business/02-content-creation.md](../01-business/02-content-creation.md) — コンテンツ制作の自動化
- [../02-monetization/01-cost-management.md](../02-monetization/01-cost-management.md) — API費用最適化
- [00-automation-overview.md](./00-automation-overview.md) — AI自動化の全体像
- [02-document-processing.md](./02-document-processing.md) — 文書処理の自動化

---

## 参考文献

1. **Gmail API Documentation** — https://developers.google.com/gmail/api — Gmail統合の公式ガイド
2. **Microsoft Graph API** — https://learn.microsoft.com/graph — Outlook/Teams統合
3. **"AI-Powered Customer Service" — Harvard Business Review (2024)** — AI顧客対応の効果測定研究
4. **Anthropic API Documentation** — https://docs.anthropic.com — Claude APIのベストプラクティス
5. **n8n Documentation** — https://docs.n8n.io — ワークフロー自動化プラットフォーム
6. **"Customer Service AI Best Practices" — Zendesk (2025)** — AIカスタマーサービスの導入ガイドライン
7. **GDPR and AI Processing** — https://gdpr.eu/artificial-intelligence — EU AI規制とデータ保護の関係
8. **個人情報保護委員会** — https://www.ppc.go.jp — 日本のAPPI（個人情報保護法）ガイドライン
