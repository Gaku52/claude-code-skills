# カスタマーサポートエージェント

> チャットボット・FAQ・エスカレーション――顧客の問い合わせを自動分類し、適切な回答を生成し、必要に応じて人間のオペレーターに引き継ぐサポートエージェントの設計。

## この章で学ぶこと

1. サポートエージェントのワークフロー設計（分類・回答・エスカレーション）
2. RAGベースのナレッジ検索と回答生成の実装パターン
3. 人間のオペレーターとの連携（ハンドオフ）設計
4. マルチチャネル対応と統一的な顧客体験の構築
5. 感情分析・トーン調整による顧客満足度向上
6. サポートエージェントの評価指標と継続改善サイクル

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

### 1.1 サポートエージェント成熟度モデル

カスタマーサポートエージェントは段階的に成熟度を高めていくことが推奨される。一度に全機能を実装するのではなく、段階的なアプローチで確実に価値を提供する。

```
成熟度レベル

Level 1: FAQ応答 ─────────────────────────────────────────
  - キーワードマッチング
  - 固定テンプレート回答
  - 解決率: 15-25%

Level 2: RAG応答 ─────────────────────────────────────────
  - ベクトル検索によるナレッジ検索
  - LLMによる自然な回答生成
  - 信頼度ベースのエスカレーション
  - 解決率: 40-55%

Level 3: コンテキスト統合 ────────────────────────────────
  - CRM/注文システム統合
  - 顧客履歴を考慮した回答
  - マルチターン会話管理
  - 解決率: 55-70%

Level 4: アクション実行 ─────────────────────────────────
  - 返金処理・アカウント変更
  - 本人確認フロー
  - トランザクション操作
  - 解決率: 65-80%

Level 5: プロアクティブ ─────────────────────────────────
  - 問題の予兆検知
  - 先回り対応
  - パーソナライズド提案
  - 解決率: 75-90%
```

### 1.2 アーキテクチャ概要

```python
"""
カスタマーサポートエージェントのアーキテクチャ概要
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class SupportTier(Enum):
    """サポート階層"""
    SELF_SERVICE = "self_service"   # セルフサービス（FAQ、ヘルプセンター）
    AI_AGENT = "ai_agent"           # AIエージェント対応
    HUMAN_L1 = "human_l1"           # 人間オペレーター（一般）
    HUMAN_L2 = "human_l2"           # 人間オペレーター（専門）
    HUMAN_L3 = "human_l3"           # エンジニア/マネージャー


class ChannelType(Enum):
    """対応チャネル"""
    WEB_CHAT = "web_chat"
    EMAIL = "email"
    LINE = "line"
    SLACK = "slack"
    PHONE = "phone"
    IN_APP = "in_app"


@dataclass
class SupportTicket:
    """サポートチケット"""
    ticket_id: str
    customer_id: str
    channel: ChannelType
    subject: str
    messages: list = field(default_factory=list)
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    urgency: str = "medium"
    current_tier: SupportTier = SupportTier.AI_AGENT
    assigned_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolution_summary: Optional[str] = None
    csat_score: Optional[int] = None
    tags: list = field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def response_time_seconds(self) -> Optional[float]:
        if len(self.messages) >= 2:
            return self.messages[1]["timestamp"] - self.messages[0]["timestamp"]
        return None

    @property
    def handle_time_seconds(self) -> Optional[float]:
        if self.resolved_at:
            return self.resolved_at - self.created_at
        return None


@dataclass
class CustomerProfile:
    """顧客プロファイル"""
    customer_id: str
    name: str
    email: str
    plan: str = "free"
    tier: str = "standard"          # standard / premium / enterprise
    language: str = "ja"
    timezone: str = "Asia/Tokyo"
    tenure_months: int = 0
    lifetime_value: float = 0.0
    recent_tickets: list = field(default_factory=list)
    satisfaction_history: list = field(default_factory=list)
    preferred_channel: ChannelType = ChannelType.WEB_CHAT
    notes: str = ""

    @property
    def average_csat(self) -> Optional[float]:
        if not self.satisfaction_history:
            return None
        return sum(self.satisfaction_history) / len(self.satisfaction_history)

    @property
    def is_at_risk(self) -> bool:
        """チャーンリスクの簡易判定"""
        if self.average_csat and self.average_csat < 3.0:
            return True
        recent_negative = sum(
            1 for s in self.satisfaction_history[-5:]
            if s <= 2
        )
        return recent_negative >= 2
```

---

## 2. 基本的なサポートエージェント

### 2.1 意図分類

```python
# 問い合わせの意図分類
import anthropic
import json
from typing import Optional


class IntentClassifier:
    """問い合わせの意図を分類するクラス"""

    INTENTS = {
        "billing": "請求・支払い関連",
        "technical": "技術的な問題・バグ報告",
        "account": "アカウント管理（変更、解約等）",
        "product": "製品に関する質問",
        "complaint": "クレーム・苦情",
        "shipping": "配送・物流関連",
        "refund": "返品・返金",
        "feature_request": "機能要望",
        "general": "その他の一般的な問い合わせ"
    }

    # サブインテントの定義
    SUB_INTENTS = {
        "billing": [
            "invoice_question",      # 請求書の質問
            "payment_failure",       # 支払い失敗
            "double_charge",         # 二重請求
            "plan_change",           # プラン変更
            "discount_inquiry",      # 割引・クーポン
        ],
        "technical": [
            "bug_report",            # バグ報告
            "performance_issue",     # パフォーマンス問題
            "integration_error",     # 連携エラー
            "how_to",               # 使い方の質問
            "data_loss",            # データ消失
        ],
        "account": [
            "password_reset",        # パスワードリセット
            "account_locked",        # アカウントロック
            "profile_update",        # プロフィール更新
            "cancellation",          # 解約
            "data_export",           # データエクスポート
        ],
    }

    def __init__(self):
        self.client = anthropic.Anthropic()

    def classify(self, message: str, conversation_history: Optional[list] = None) -> dict:
        """問い合わせを分類する"""

        # 会話履歴がある場合はコンテキストとして含める
        history_context = ""
        if conversation_history:
            recent = conversation_history[-3:]  # 直近3メッセージ
            history_context = "\n".join([
                f"{'顧客' if m['role'] == 'user' else 'サポート'}: {m['content']}"
                for m in recent
            ])
            history_context = f"\n過去の会話:\n{history_context}\n"

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # 高速・低コスト
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
以下の顧客メッセージを分類してください。
{history_context}
メッセージ: {message}

メインカテゴリ: {list(self.INTENTS.keys())}
サブカテゴリ（該当する場合）: {json.dumps(self.SUB_INTENTS, ensure_ascii=False)}

JSON形式で出力:
{{
  "intent": "メインカテゴリ名",
  "sub_intent": "サブカテゴリ名またはnull",
  "confidence": 0.0-1.0,
  "sentiment": "positive/neutral/negative/angry",
  "urgency": "low/medium/high/critical",
  "language": "検出された言語コード",
  "key_entities": ["関連するエンティティのリスト"],
  "requires_auth": true/false
}}
"""}]
        )
        result = json.loads(response.content[0].text)

        # 信頼度が低い場合の2段階分類
        if result.get("confidence", 0) < 0.6:
            result = self._reclassify_with_context(message, result)

        return result

    def _reclassify_with_context(self, message: str, initial_result: dict) -> dict:
        """信頼度が低い場合にSonnetモデルで再分類"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": f"""
以下の顧客メッセージの分類結果の信頼度が低いため、再分析してください。

メッセージ: {message}
初回分類結果: {json.dumps(initial_result, ensure_ascii=False)}

より正確なJSON形式で出力:
{{
  "intent": "メインカテゴリ名",
  "sub_intent": "サブカテゴリ名またはnull",
  "confidence": 0.0-1.0,
  "sentiment": "positive/neutral/negative/angry",
  "urgency": "low/medium/high/critical",
  "language": "検出された言語コード",
  "key_entities": ["関連するエンティティのリスト"],
  "requires_auth": true/false,
  "reclassified": true,
  "reclassification_reason": "再分類の理由"
}}
"""}]
        )
        return json.loads(response.content[0].text)


# 使用例
classifier = IntentClassifier()
result = classifier.classify("先月の請求が二重になっています！至急確認してください")
# {
#   "intent": "billing",
#   "sub_intent": "double_charge",
#   "confidence": 0.95,
#   "sentiment": "negative",
#   "urgency": "high",
#   "language": "ja",
#   "key_entities": ["先月の請求", "二重請求"],
#   "requires_auth": true
# }
```

### 2.2 RAGベースの回答生成

```python
# ナレッジベースからの回答生成
import hashlib
from datetime import datetime


class SupportKnowledgeBase:
    """サポート用ナレッジベース検索・回答生成"""

    def __init__(self, vector_store, cache=None):
        self.vector_store = vector_store
        self.client = anthropic.Anthropic()
        self.cache = cache or {}

    def answer(self, question: str, customer_context: dict = None) -> dict:
        """質問に対する回答を生成する"""

        # 0. キャッシュチェック（同一質問の高速回答）
        cache_key = self._cache_key(question)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["cached_at"] < 3600:  # 1時間有効
                return {**cached["result"], "from_cache": True}

        # 1. 関連ナレッジを検索
        relevant_docs = self.vector_store.search(question, top_k=5)

        # 2. ドキュメントの関連性フィルタリング
        filtered_docs = self._filter_relevant_docs(relevant_docs, question)

        # 3. コンテキスト構築
        context = "\n\n".join([
            f"--- ドキュメント: {doc['title']} (更新日: {doc.get('updated_at', '不明')}) ---\n{doc['content']}"
            for doc in filtered_docs
        ])

        customer_info = ""
        if customer_context:
            customer_info = f"""
顧客情報:
- 顧客ID: {customer_context.get('customer_id', '不明')}
- プラン: {customer_context.get('plan', '不明')}
- 利用期間: {customer_context.get('tenure', '不明')}
- 過去の問い合わせ: {customer_context.get('history_summary', 'なし')}
- 利用中の機能: {customer_context.get('features', '不明')}
"""

        # 4. 回答生成
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
- 回答に含める情報元のドキュメント名を記録する
- 顧客のプランや利用状況に合わせた回答をする

JSON形式で出力:
{{
  "answer": "回答テキスト",
  "used_sources": ["使用したドキュメント名"],
  "answer_type": "direct_answer / guidance / partial / no_info",
  "follow_up_questions": ["顧客が次に聞きそうな質問"]
}}
"""}]
        )

        parsed = json.loads(response.content[0].text)
        answer_text = parsed["answer"]
        answer_type = parsed.get("answer_type", "direct_answer")

        # 5. 信頼度評価
        confidence = self._calculate_confidence(
            filtered_docs, question, answer_type
        )

        result = {
            "answer": answer_text,
            "confidence": confidence,
            "sources": parsed.get("used_sources", []),
            "answer_type": answer_type,
            "follow_up_questions": parsed.get("follow_up_questions", []),
            "should_escalate": confidence < 0.5 or answer_type == "no_info",
            "from_cache": False,
        }

        # 6. キャッシュ保存
        if confidence > 0.8:
            self.cache[cache_key] = {
                "result": result,
                "cached_at": time.time(),
            }

        return result

    def _filter_relevant_docs(self, docs: list, question: str) -> list:
        """関連性の低いドキュメントを除外"""
        filtered = []
        for doc in docs:
            similarity = doc.get("similarity", 0)
            if similarity > 0.7:  # 類似度閾値
                filtered.append(doc)
        return filtered if filtered else docs[:3]

    def _calculate_confidence(
        self, docs: list, question: str, answer_type: str
    ) -> float:
        """回答の信頼度を計算"""
        if answer_type == "no_info":
            return 0.1
        if answer_type == "partial":
            return 0.4

        # ドキュメントの類似度スコアの平均
        if docs:
            avg_similarity = sum(
                d.get("similarity", 0.5) for d in docs
            ) / len(docs)
        else:
            avg_similarity = 0.2

        # ドキュメントの鮮度による補正
        freshness_bonus = 0
        for doc in docs:
            updated = doc.get("updated_at")
            if updated:
                days_old = (datetime.now() - datetime.fromisoformat(updated)).days
                if days_old < 30:
                    freshness_bonus += 0.05
                elif days_old > 365:
                    freshness_bonus -= 0.05

        return min(max(avg_similarity + freshness_bonus, 0.0), 1.0)

    def _cache_key(self, question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
```

### 2.3 完全なサポートエージェント

```python
# 完全なカスタマーサポートエージェント
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CustomerSupportAgent:
    """完全なカスタマーサポートエージェント"""

    def __init__(self, knowledge_base, crm_system, action_executor=None):
        self.classifier = IntentClassifier()
        self.kb = knowledge_base
        self.crm = crm_system
        self.action_executor = action_executor
        self.client = anthropic.Anthropic()
        self.escalation_threshold = 0.5
        self.conversation_manager = ConversationManager()
        self.tone_adjuster = ToneAdjuster()
        self.metrics = SupportMetrics()

    def handle_inquiry(self, customer_id: str, message: str,
                       channel: str = "web_chat") -> dict:
        """問い合わせを処理する"""
        start_time = time.time()

        # 1. 顧客情報取得
        customer = self.crm.get_customer(customer_id)
        logger.info(f"Handling inquiry from {customer_id} via {channel}")

        # 2. 会話セッション管理
        session = self.conversation_manager.get_or_create_session(customer_id)
        self.conversation_manager.add_message(customer_id, "user", message)

        # 3. 意図分類（会話履歴を含む）
        intent = self.classifier.classify(
            message,
            conversation_history=session.get("messages", [])
        )
        session["intent_history"].append(intent)

        # 4. 即時エスカレーション判定
        if self._needs_immediate_escalation(intent, customer):
            result = self._escalate(
                customer_id, message, intent, "自動エスカレーション"
            )
            self.metrics.record(customer_id, intent, result, time.time() - start_time)
            return result

        # 5. 本人確認が必要な場合
        if intent.get("requires_auth") and not session.get("authenticated"):
            return self._request_authentication(customer_id, intent)

        # 6. 意図に応じた処理
        handler = self._get_handler(intent["intent"])
        response = handler(customer, message, intent)

        # 7. トーン調整
        adjusted_answer = self.tone_adjuster.adjust(
            response["answer"],
            intent.get("sentiment", "neutral"),
            customer
        )
        response["answer"] = adjusted_answer

        # 8. 信頼度チェック
        if response["confidence"] < self.escalation_threshold:
            result = self._escalate(
                customer_id, message, intent, "低信頼度"
            )
            self.metrics.record(customer_id, intent, result, time.time() - start_time)
            return result

        # 9. フォローアップ質問の提案
        follow_ups = response.get("follow_up_questions", [])

        # 10. 会話履歴保存
        self.conversation_manager.add_message(
            customer_id, "assistant", response["answer"]
        )
        self.crm.log_interaction(customer_id, message, response["answer"])

        result = {
            "response": response["answer"],
            "intent": intent,
            "escalated": False,
            "confidence": response["confidence"],
            "sources": response.get("sources", []),
            "follow_up_questions": follow_ups,
            "processing_time": time.time() - start_time,
        }

        self.metrics.record(customer_id, intent, result, time.time() - start_time)
        return result

    def _get_handler(self, intent_type: str):
        """意図に対応するハンドラーを返す"""
        handlers = {
            "billing": self._handle_billing,
            "technical": self._handle_technical,
            "account": self._handle_account,
            "shipping": self._handle_shipping,
            "refund": self._handle_refund,
            "feature_request": self._handle_feature_request,
        }
        return handlers.get(intent_type, self._handle_general)

    def _handle_billing(self, customer: dict, message: str,
                        intent: dict) -> dict:
        """請求関連の問い合わせを処理"""
        # 請求履歴を取得
        billing_history = self.crm.get_billing_history(
            customer["customer_id"], limit=6
        )

        # ナレッジベース + 請求データで回答
        enhanced_context = {
            **customer,
            "billing_history": billing_history,
            "history_summary": f"直近6件の請求: {json.dumps(billing_history, ensure_ascii=False)}"
        }

        response = self.kb.answer(message, enhanced_context)

        # 二重請求の場合は自動返金フラグ
        if intent.get("sub_intent") == "double_charge":
            response["action_suggested"] = "refund_review"
            response["answer"] += "\n\n請求内容を確認しております。二重請求が確認された場合、自動的に返金処理を行います。"

        return response

    def _handle_technical(self, customer: dict, message: str,
                          intent: dict) -> dict:
        """技術的な問い合わせを処理"""
        # ステータスページの確認
        system_status = self._check_system_status()

        if system_status.get("has_incident"):
            incident = system_status["current_incident"]
            return {
                "answer": f"現在、{incident['service']}にて障害が発生しております。"
                          f"\n\nステータス: {incident['status']}"
                          f"\n影響範囲: {incident['impact']}"
                          f"\n復旧見込み: {incident['eta']}"
                          f"\n\n最新情報はステータスページ(https://status.example.com)をご確認ください。",
                "confidence": 0.95,
                "sources": ["system_status_page"],
            }

        # トラブルシューティングガイドの検索
        response = self.kb.answer(message, customer)

        # データ消失の場合は即エスカレーション
        if intent.get("sub_intent") == "data_loss":
            response["confidence"] = 0.0  # 強制エスカレーション
            response["urgency_override"] = "critical"

        return response

    def _handle_account(self, customer: dict, message: str,
                        intent: dict) -> dict:
        """アカウント関連の問い合わせを処理"""
        sub_intent = intent.get("sub_intent")

        if sub_intent == "password_reset":
            # パスワードリセットメールの自動送信
            if self.action_executor:
                self.action_executor.send_password_reset(customer["email"])
            return {
                "answer": f"パスワードリセット用のメールを {self._mask_email(customer['email'])} に送信しました。"
                          "\n\nメールが届かない場合は、迷惑メールフォルダをご確認ください。"
                          "\n10分経っても届かない場合は、再度お知らせください。",
                "confidence": 0.95,
                "sources": ["account_management"],
            }

        if sub_intent == "cancellation":
            # 解約は人間対応（リテンション機会）
            return {
                "answer": "解約についてのお問い合わせですね。"
                          "\n担当者が最適なご提案をさせていただきますので、少々お待ちください。",
                "confidence": 0.3,  # 意図的に低くしてエスカレーション
                "sources": [],
            }

        return self.kb.answer(message, customer)

    def _handle_shipping(self, customer: dict, message: str,
                         intent: dict) -> dict:
        """配送関連の問い合わせを処理"""
        # 注文・配送情報の取得
        orders = self.crm.get_recent_orders(customer["customer_id"])
        if orders:
            latest_order = orders[0]
            tracking_info = self.crm.get_tracking_info(latest_order["order_id"])
            enhanced_context = {
                **customer,
                "order_info": latest_order,
                "tracking_info": tracking_info,
            }
            return self.kb.answer(message, enhanced_context)
        return self.kb.answer(message, customer)

    def _handle_refund(self, customer: dict, message: str,
                       intent: dict) -> dict:
        """返品・返金の問い合わせを処理"""
        response = self.kb.answer(message, customer)
        # 返金ポリシーの確認
        response["answer"] += "\n\n【返金ポリシー】\n- 購入後30日以内: 全額返金\n- 30-60日: 50%返金\n- 60日以降: 返金不可"
        return response

    def _handle_feature_request(self, customer: dict, message: str,
                                intent: dict) -> dict:
        """機能要望を処理"""
        # 機能要望をプロダクトチームに記録
        self.crm.log_feature_request(
            customer_id=customer["customer_id"],
            request=message,
            plan=customer.get("plan", "free"),
        )
        return {
            "answer": "貴重なご意見をいただきありがとうございます。"
                      "\nプロダクトチームに共有させていただきます。"
                      "\n\n今後のアップデート情報は、リリースノートページでご確認いただけます。",
            "confidence": 0.9,
            "sources": [],
        }

    def _handle_general(self, customer: dict, message: str,
                        intent: dict) -> dict:
        """一般的な問い合わせを処理"""
        return self.kb.answer(message, customer)

    def _needs_immediate_escalation(self, intent: dict,
                                    customer: dict) -> bool:
        """即時エスカレーションが必要かを判定"""
        # 強いネガティブ感情
        if intent.get("sentiment") == "angry":
            return True
        if intent["sentiment"] == "negative" and intent["urgency"] in ("high", "critical"):
            return True
        # VIP顧客
        if customer.get("tier") == "enterprise":
            return True
        # 苦情
        if intent["intent"] == "complaint":
            return True
        # データ消失
        if intent.get("sub_intent") == "data_loss":
            return True
        # クリティカル緊急度
        if intent.get("urgency") == "critical":
            return True
        return False

    def _escalate(self, customer_id: str, message: str,
                  intent: dict, reason: str) -> dict:
        """人間のオペレーターにエスカレーション"""
        # 会話サマリーを生成
        session = self.conversation_manager.get_or_create_session(customer_id)
        summary = self.conversation_manager.get_context_summary(customer_id)

        # 適切なチームの選択
        team = self._select_escalation_team(intent)

        ticket = self.crm.create_escalation_ticket(
            customer_id=customer_id,
            message=message,
            intent=intent,
            reason=reason,
            priority="high" if intent.get("urgency") in ("high", "critical") else "normal",
            team=team,
            conversation_summary=summary,
        )

        # エスカレーション通知
        logger.warning(
            f"Escalation: customer={customer_id}, reason={reason}, "
            f"team={team}, ticket={ticket['id']}"
        )

        return {
            "response": "担当者におつなぎいたします。少々お待ちください。"
                        f"\n\nチケット番号: {ticket['id']}"
                        "\nお急ぎの場合は、このチケット番号をお伝えください。",
            "intent": intent,
            "escalated": True,
            "ticket_id": ticket["id"],
            "escalation_team": team,
            "escalation_reason": reason,
        }

    def _select_escalation_team(self, intent: dict) -> str:
        """意図に基づいてエスカレーション先チームを選択"""
        team_map = {
            "billing": "billing_team",
            "technical": "tech_support",
            "account": "account_team",
            "complaint": "customer_success",
            "refund": "billing_team",
            "shipping": "logistics_team",
        }
        return team_map.get(intent["intent"], "general_support")

    def _request_authentication(self, customer_id: str,
                                intent: dict) -> dict:
        """本人確認を要求"""
        return {
            "response": "セキュリティのため、本人確認をさせていただきます。"
                        "\n\nご登録のメールアドレスに確認コードを送信しました。"
                        "\n6桁の確認コードを入力してください。",
            "intent": intent,
            "escalated": False,
            "requires_auth": True,
            "auth_method": "email_otp",
        }

    def _check_system_status(self) -> dict:
        """システムステータスを確認"""
        # 実際にはステータスページAPIを呼び出す
        return {"has_incident": False}

    def _mask_email(self, email: str) -> str:
        """メールアドレスをマスク"""
        parts = email.split("@")
        if len(parts) == 2:
            name = parts[0]
            masked = name[0] + "***" + name[-1] if len(name) > 2 else "***"
            return f"{masked}@{parts[1]}"
        return "***"
```

---

## 3. エスカレーション設計

### 3.1 エスカレーション判断マトリクス

```
エスカレーションの判断マトリクス

                    感情: ポジティブ/中立    感情: ネガティブ      感情: 怒り
                    +-------------------+-------------------+-------------------+
  信頼度: 高 (>0.8) | 自動回答          | 自動回答 +        | 自動回答 +        |
                    |                   | トーン注意        | エスカレーション    |
                    |                   |                   | 選択肢提示        |
                    +-------------------+-------------------+-------------------+
  信頼度: 中 (0.5-) | 自動回答 +        | エスカレーション    | 即時エスカレーション |
                    | 確認文付き        |                   | (優先度高)         |
                    +-------------------+-------------------+-------------------+
  信頼度: 低 (<0.5) | エスカレーション    | 即時エスカレーション | 即時エスカレーション |
                    |                   | (優先度高)         | (最優先)           |
                    +-------------------+-------------------+-------------------+
```

### 3.2 段階的エスカレーションエンジン

```python
class EscalationEngine:
    """段階的エスカレーション管理"""

    def __init__(self, notification_service, queue_manager):
        self.notification = notification_service
        self.queue = queue_manager
        self.rules = self._load_escalation_rules()

    def evaluate(self, ticket: SupportTicket, customer: CustomerProfile,
                 agent_response: dict) -> dict:
        """エスカレーションの必要性と方法を評価"""
        score = self._calculate_escalation_score(
            ticket, customer, agent_response
        )

        if score < 30:
            return {"action": "auto_respond", "tier": SupportTier.AI_AGENT}
        elif score < 50:
            return {
                "action": "auto_respond_with_review",
                "tier": SupportTier.AI_AGENT,
                "review_required": True,
            }
        elif score < 70:
            return {
                "action": "escalate_l1",
                "tier": SupportTier.HUMAN_L1,
                "priority": "normal",
            }
        elif score < 90:
            return {
                "action": "escalate_l2",
                "tier": SupportTier.HUMAN_L2,
                "priority": "high",
            }
        else:
            return {
                "action": "escalate_l3",
                "tier": SupportTier.HUMAN_L3,
                "priority": "critical",
            }

    def _calculate_escalation_score(
        self, ticket: SupportTicket, customer: CustomerProfile,
        agent_response: dict
    ) -> int:
        """エスカレーションスコアを計算（0-100）"""
        score = 0

        # 信頼度に基づくスコア
        confidence = agent_response.get("confidence", 0.5)
        score += int((1 - confidence) * 30)

        # 感情スコア
        sentiment_scores = {
            "positive": 0, "neutral": 5,
            "negative": 20, "angry": 40,
        }
        score += sentiment_scores.get(ticket.sentiment, 10)

        # 緊急度スコア
        urgency_scores = {
            "low": 0, "medium": 5,
            "high": 15, "critical": 30,
        }
        score += urgency_scores.get(ticket.urgency, 5)

        # 顧客ティアスコア
        tier_scores = {
            "standard": 0, "premium": 10, "enterprise": 25,
        }
        score += tier_scores.get(customer.tier, 0)

        # チャーンリスク
        if customer.is_at_risk:
            score += 15

        # 連続問い合わせ（同じ問題で3回以上）
        repeat_count = self._count_repeat_inquiries(
            customer.customer_id, ticket.intent
        )
        if repeat_count >= 3:
            score += 20

        return min(score, 100)

    def _count_repeat_inquiries(self, customer_id: str,
                                intent: str) -> int:
        """同一意図の直近問い合わせ回数をカウント"""
        recent = self.queue.get_recent_tickets(
            customer_id, days=7
        )
        return sum(1 for t in recent if t.intent == intent)

    def execute_escalation(self, ticket: SupportTicket,
                           decision: dict) -> dict:
        """エスカレーションを実行"""
        tier = decision["tier"]
        priority = decision.get("priority", "normal")

        # チケットを更新
        ticket.current_tier = tier

        # キューに追加
        queue_position = self.queue.enqueue(ticket, priority)

        # 通知送信
        if priority in ("high", "critical"):
            self.notification.send_urgent_alert(
                team=self._get_team_for_tier(tier),
                ticket=ticket,
                priority=priority,
            )

        # 顧客への待ち時間通知
        estimated_wait = self.queue.estimate_wait_time(tier, priority)

        return {
            "ticket_id": ticket.ticket_id,
            "tier": tier.value,
            "priority": priority,
            "queue_position": queue_position,
            "estimated_wait_minutes": estimated_wait,
        }

    def _get_team_for_tier(self, tier: SupportTier) -> str:
        team_map = {
            SupportTier.HUMAN_L1: "general_support",
            SupportTier.HUMAN_L2: "specialist_support",
            SupportTier.HUMAN_L3: "engineering_escalation",
        }
        return team_map.get(tier, "general_support")
```

### 3.3 スムーズなハンドオフ

```python
class HandoffManager:
    """AIエージェントから人間オペレーターへのスムーズなハンドオフ"""

    def __init__(self, llm_client):
        self.client = llm_client

    def prepare_handoff_package(self, ticket: SupportTicket,
                                session: dict,
                                customer: CustomerProfile) -> dict:
        """人間オペレーター向けのハンドオフパッケージを準備"""

        # 会話サマリーの生成
        conversation_summary = self._summarize_conversation(
            session["messages"]
        )

        # 試行済みの解決策
        attempted_solutions = self._extract_attempted_solutions(
            session["messages"]
        )

        # 推奨アクション
        recommended_actions = self._suggest_actions(
            ticket, customer, conversation_summary
        )

        return {
            "ticket_id": ticket.ticket_id,
            "customer": {
                "id": customer.customer_id,
                "name": customer.name,
                "plan": customer.plan,
                "tier": customer.tier,
                "tenure_months": customer.tenure_months,
                "lifetime_value": customer.lifetime_value,
                "avg_csat": customer.average_csat,
                "at_risk": customer.is_at_risk,
            },
            "issue": {
                "intent": ticket.intent,
                "sentiment": ticket.sentiment,
                "urgency": ticket.urgency,
                "summary": conversation_summary,
            },
            "context": {
                "attempted_solutions": attempted_solutions,
                "recommended_actions": recommended_actions,
                "related_incidents": self._find_related_incidents(ticket),
            },
            "conversation_history": session["messages"],
            "handoff_time": datetime.now().isoformat(),
        }

    def _summarize_conversation(self, messages: list) -> str:
        """会話を要約"""
        conversation_text = "\n".join([
            f"{'顧客' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in messages
        ])

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": f"""
以下のサポート会話を3文以内で要約してください。
問題の内容、顧客の感情、試みた解決策を含めてください。

会話:
{conversation_text}

要約:
"""}]
        )
        return response.content[0].text

    def _extract_attempted_solutions(self, messages: list) -> list:
        """試行済みの解決策を抽出"""
        solutions = []
        for msg in messages:
            if msg["role"] == "assistant":
                # 番号付きリストや「お試しください」を含む回答を解決策として抽出
                if any(kw in msg["content"] for kw in
                       ["お試しください", "手順", "方法", "以下を"]):
                    solutions.append({
                        "suggestion": msg["content"][:200],
                        "timestamp": msg.get("timestamp"),
                    })
        return solutions

    def _suggest_actions(self, ticket: SupportTicket,
                         customer: CustomerProfile,
                         summary: str) -> list:
        """オペレーター向けの推奨アクションを生成"""
        actions = []

        if customer.is_at_risk:
            actions.append({
                "action": "retention_offer",
                "description": "チャーンリスクあり。リテンションオファーの検討を推奨。",
                "priority": "high",
            })

        if ticket.urgency in ("high", "critical"):
            actions.append({
                "action": "priority_handling",
                "description": "緊急度が高いため、優先的な対応を推奨。",
                "priority": "high",
            })

        if customer.tier == "enterprise":
            actions.append({
                "action": "account_manager_notify",
                "description": "エンタープライズ顧客。アカウントマネージャーへの通知を推奨。",
                "priority": "medium",
            })

        return actions

    def _find_related_incidents(self, ticket: SupportTicket) -> list:
        """関連するインシデントを検索"""
        # 実際にはインシデント管理システムを検索
        return []
```

---

## 4. 会話管理

### 4.1 マルチターン会話マネージャー

```python
# マルチターン会話の管理
class ConversationManager:
    """マルチターン会話のセッション管理"""

    def __init__(self, max_session_age: int = 3600,
                 max_messages: int = 50):
        self.sessions = {}
        self.max_session_age = max_session_age
        self.max_messages = max_messages

    def get_or_create_session(self, customer_id: str) -> dict:
        if customer_id in self.sessions:
            session = self.sessions[customer_id]
            # セッションの有効期限チェック
            if time.time() - session["last_activity"] > self.max_session_age:
                # 期限切れセッションをアーカイブして新規作成
                self._archive_session(customer_id, session)
                return self._create_session(customer_id)
            session["last_activity"] = time.time()
            return session
        return self._create_session(customer_id)

    def _create_session(self, customer_id: str) -> dict:
        session = {
            "session_id": f"sess_{customer_id}_{int(time.time())}",
            "messages": [],
            "intent_history": [],
            "created_at": time.time(),
            "last_activity": time.time(),
            "resolved": False,
            "authenticated": False,
            "context_variables": {},   # カスタム変数
            "interaction_count": 0,
        }
        self.sessions[customer_id] = session
        return session

    def add_message(self, customer_id: str, role: str, content: str):
        session = self.get_or_create_session(customer_id)
        session["messages"].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        session["interaction_count"] += 1

        # メッセージ数が上限を超えた場合は圧縮
        if len(session["messages"]) > self.max_messages:
            self._compress_session(customer_id)

    def get_context_summary(self, customer_id: str) -> str:
        """会話の要約を生成（長い会話の圧縮用）"""
        session = self.sessions.get(customer_id)
        if not session or len(session["messages"]) < 6:
            return ""

        # 古い部分を要約
        old_messages = session["messages"][:-5]
        conversation_text = "\n".join([
            f"{'顧客' if m['role'] == 'user' else 'サポート'}: {m['content']}"
            for m in old_messages
        ])

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": f"""
以下の会話を簡潔に要約してください:
{conversation_text}
"""}]
        )
        return f"これまでの会話の要約: {response.content[0].text}"

    def _compress_session(self, customer_id: str):
        """古いメッセージを要約で置換"""
        session = self.sessions[customer_id]
        summary = self.get_context_summary(customer_id)

        # 古いメッセージを要約に置換し、直近5件を保持
        session["messages"] = [
            {"role": "system", "content": summary, "timestamp": time.time()}
        ] + session["messages"][-5:]

    def _archive_session(self, customer_id: str, session: dict):
        """セッションをアーカイブ"""
        logger.info(
            f"Archiving session {session['session_id']} "
            f"({session['interaction_count']} interactions)"
        )

    def get_session_metrics(self, customer_id: str) -> dict:
        """セッションの統計情報を返す"""
        session = self.sessions.get(customer_id)
        if not session:
            return {}

        messages = session["messages"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]

        return {
            "session_id": session["session_id"],
            "total_messages": len(messages),
            "user_messages": len(user_msgs),
            "assistant_messages": len(assistant_msgs),
            "duration_seconds": time.time() - session["created_at"],
            "resolved": session["resolved"],
        }
```

### 4.2 コンテキスト変数管理

```python
class ContextVariableManager:
    """会話中に収集した情報を管理"""

    REQUIRED_VARIABLES = {
        "billing": ["order_id", "amount", "date"],
        "technical": ["error_message", "browser", "os"],
        "account": ["email", "auth_verified"],
        "shipping": ["order_id", "tracking_number"],
    }

    def __init__(self, llm_client):
        self.client = llm_client

    def extract_variables(self, message: str, intent: str,
                          existing_vars: dict) -> dict:
        """メッセージからコンテキスト変数を抽出"""
        required = self.REQUIRED_VARIABLES.get(intent, [])
        missing = [v for v in required if v not in existing_vars]

        if not missing:
            return existing_vars

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": f"""
以下のメッセージから情報を抽出してください。

メッセージ: {message}
抽出したい情報: {missing}
既存の情報: {json.dumps(existing_vars, ensure_ascii=False)}

JSON形式で抽出結果を出力（見つからない場合はnull）:
"""}]
        )
        extracted = json.loads(response.content[0].text)
        return {**existing_vars, **{k: v for k, v in extracted.items() if v}}

    def get_missing_info_prompt(self, intent: str,
                                existing_vars: dict) -> Optional[str]:
        """不足情報を聞くプロンプトを生成"""
        required = self.REQUIRED_VARIABLES.get(intent, [])
        missing = [v for v in required if v not in existing_vars]

        if not missing:
            return None

        prompts = {
            "order_id": "ご注文番号をお知らせいただけますでしょうか？",
            "amount": "該当の金額をお教えください。",
            "date": "いつ頃の件でしょうか？",
            "error_message": "表示されているエラーメッセージをお知らせください。",
            "browser": "ご利用のブラウザを教えてください（Chrome、Safari等）。",
            "os": "ご利用のOS（Windows、Mac等）を教えてください。",
            "email": "ご登録のメールアドレスを教えてください。",
            "tracking_number": "追跡番号をお持ちでしたらお知らせください。",
        }

        questions = [prompts.get(m, f"{m}を教えてください。") for m in missing[:2]]
        return "\n".join(questions)
```

---

## 5. 比較表

### 5.1 サポートチャネル比較

| チャネル | 対応速度 | コスト/件 | 顧客満足度 | 対応可能時間 | 複雑な問題 | 導入難易度 |
|---------|---------|----------|-----------|------------|-----------|-----------|
| AIチャット | 即時 | $0.01-0.10 | 中-高 | 24/7 | 低-中 | 中 |
| 人間チャット | 1-5分 | $5-15 | 高 | 営業時間 | 高 | 低 |
| メール | 数時間-1日 | $3-8 | 中 | 24/7受付 | 中-高 | 低 |
| 電話 | 待ち時間あり | $10-25 | 最高 | 営業時間 | 最高 | 低 |
| FAQ/セルフ | 即時 | $0.001 | 低-中 | 24/7 | 低 | 中 |
| LINE/SNS | 数分-数時間 | $2-10 | 中-高 | 営業時間+ | 中 | 中 |
| アプリ内 | 即時-数分 | $0.05-0.50 | 高 | 24/7 | 中 | 高 |

### 5.2 自動化レベル比較

| レベル | 説明 | 解決率 | 適用場面 | 実装コスト | ROI回収期間 |
|--------|------|--------|---------|-----------|------------|
| L0 ルール | if-else定型回答 | 20-30% | よくある質問 | 低 | 1-2ヶ月 |
| L1 検索 | FAQ検索+テンプレート | 40-50% | ナレッジベースあり | 中 | 2-4ヶ月 |
| L2 RAG | 文書検索+LLM生成 | 50-65% | 豊富なドキュメント | 中-高 | 3-6ヶ月 |
| L3 エージェント | 自律的問題解決 | 60-75% | ツール統合あり | 高 | 4-8ヶ月 |
| L4 完全自律 | アカウント操作含む | 70-85% | CRM/DB統合あり | 最高 | 6-12ヶ月 |

### 5.3 LLMモデル選定ガイド

| 用途 | 推奨モデル | レイテンシ | コスト | 理由 |
|------|-----------|----------|--------|------|
| 意図分類 | Haiku | ~200ms | 最低 | 高速分類、低コスト |
| 回答生成（一般） | Sonnet | ~1s | 中 | バランスの良い品質 |
| 複雑な問題解決 | Opus | ~3s | 高 | 高精度な推論 |
| 感情分析 | Haiku | ~200ms | 最低 | 高速・十分な精度 |
| 会話要約 | Haiku | ~300ms | 最低 | コスト効率重視 |
| トーン調整 | Sonnet | ~800ms | 中 | ニュアンスの再現 |
| ナレッジ作成 | Opus | ~5s | 高 | 高品質な文書生成 |

### 5.4 サポートツール・プラットフォーム比較

| ツール | AI対応 | マルチチャネル | カスタマイズ性 | 価格帯 | 特徴 |
|--------|--------|-------------|-------------|--------|------|
| Zendesk | Answer Bot | 全チャネル | 中 | $49-215/agent | 業界標準、豊富な連携 |
| Intercom | Fin AI | チャット中心 | 高 | $74-??/agent | 会話型UX、プロダクトツアー |
| Freshdesk | Freddy AI | 全チャネル | 中 | $0-95/agent | コスパ良好、無料プランあり |
| カスタム構築 | 完全制御 | 自由 | 最高 | 開発コスト | 完全な柔軟性 |
| Helpscout | AI Drafts | メール中心 | 低 | $20-65/user | シンプル、メール特化 |

---

## 6. トーン・言語設計

### 6.1 トーン調整エンジン

```python
# 回答のトーン調整
class ToneAdjuster:
    """顧客の感情に応じたトーン調整"""

    TONE_GUIDELINES = {
        "positive": {
            "description": "明るく前向きな言葉遣い。顧客の良い体験を喜ぶ。",
            "opening": "ありがとうございます！",
            "closing": "他にもお手伝いできることがございましたら、お気軽にお知らせください。",
            "emoji_ok": True,
        },
        "neutral": {
            "description": "丁寧でプロフェッショナル。事実ベースの対応。",
            "opening": "お問い合わせいただきありがとうございます。",
            "closing": "ご不明な点がございましたら、お気軽にお問い合わせください。",
            "emoji_ok": False,
        },
        "negative": {
            "description": "共感を示す。「ご不便をおかけし申し訳ございません」から始める。お詫びの後に解決策を提示。",
            "opening": "ご不便をおかけし、大変申し訳ございません。",
            "closing": "今後このようなことがないよう改善に努めてまいります。",
            "emoji_ok": False,
        },
        "angry": {
            "description": "最大限の共感。感情を否定しない。具体的な解決ステップを即座に提示。エスカレーション選択肢も。",
            "opening": "ご迷惑をおかけし、心よりお詫び申し上げます。お気持ちは十分に理解しております。",
            "closing": "早急に対応させていただきます。ご納得いただけない場合は、責任者におつなぎすることも可能です。",
            "emoji_ok": False,
        }
    }

    # NG表現リスト
    BANNED_PHRASES = [
        "それはできません",
        "無理です",
        "そんなはずはありません",
        "お客様の勘違いでは",
        "前にも説明しましたが",
        "マニュアルに書いてあります",
        "弊社の責任ではありません",
    ]

    # 置換マップ
    PHRASE_REPLACEMENTS = {
        "できません": "現時点では対応が難しい状況です。代替案として、",
        "わかりません": "確認いたします。少々お時間をいただけますでしょうか。",
        "それは仕様です": "現在の仕様ではそのような動作となっております。ご要望として開発チームに共有させていただきます。",
    }

    def __init__(self):
        self.client = anthropic.Anthropic()

    def adjust(self, answer: str, sentiment: str,
               customer: dict = None) -> str:
        """回答のトーンを調整する"""

        # 1. NG表現チェック
        answer = self._replace_banned_phrases(answer)

        # 2. 顧客ティアに応じた調整
        formality_level = "formal"
        if customer and customer.get("tier") == "enterprise":
            formality_level = "very_formal"

        guidelines = self.TONE_GUIDELINES.get(
            sentiment, self.TONE_GUIDELINES["neutral"]
        )

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下の回答を、顧客の感情に配慮してリライトしてください。

トーンガイドライン: {guidelines['description']}
冒頭の挨拶: {guidelines['opening']}
締めの言葉: {guidelines['closing']}
丁寧さレベル: {formality_level}

元の回答: {answer}

ルール:
- 情報の正確性は変えない
- 冒頭に適切な挨拶を入れる
- 末尾に締めの言葉を入れる
- 以下の表現は絶対に使わない: {self.BANNED_PHRASES}

リライト後:
"""}]
        )
        return response.content[0].text

    def _replace_banned_phrases(self, text: str) -> str:
        """NG表現を置換"""
        for banned in self.BANNED_PHRASES:
            if banned in text:
                replacement = self.PHRASE_REPLACEMENTS.get(banned, "")
                if replacement:
                    text = text.replace(banned, replacement)
        return text
```

### 6.2 多言語対応

```python
class MultiLanguageSupport:
    """多言語サポート対応"""

    SUPPORTED_LANGUAGES = {
        "ja": {"name": "日本語", "formality": "keigo"},
        "en": {"name": "English", "formality": "professional"},
        "zh": {"name": "中文", "formality": "formal"},
        "ko": {"name": "한국어", "formality": "jondaenmal"},
        "es": {"name": "Español", "formality": "usted"},
    }

    def __init__(self):
        self.client = anthropic.Anthropic()

    def detect_and_respond(self, message: str, answer: str,
                           preferred_language: str = None) -> dict:
        """メッセージの言語を検出し、適切な言語で回答"""

        # 言語検出
        detected_lang = self._detect_language(message)

        # 優先言語の設定がある場合はそちらを使用
        target_lang = preferred_language or detected_lang

        if target_lang not in self.SUPPORTED_LANGUAGES:
            target_lang = "en"  # フォールバック

        # 回答が対象言語でない場合は翻訳
        if not self._is_language(answer, target_lang):
            lang_config = self.SUPPORTED_LANGUAGES[target_lang]
            answer = self._translate_with_tone(
                answer, target_lang, lang_config["formality"]
            )

        return {
            "answer": answer,
            "detected_language": detected_lang,
            "response_language": target_lang,
        }

    def _detect_language(self, text: str) -> str:
        """テキストの言語を検出"""
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": f"Detect the language of this text and reply with only the ISO 639-1 code: {text[:200]}"}]
        )
        return response.content[0].text.strip().lower()

    def _is_language(self, text: str, lang: str) -> bool:
        """テキストが指定言語かどうか判定"""
        detected = self._detect_language(text)
        return detected == lang

    def _translate_with_tone(self, text: str, target_lang: str,
                             formality: str) -> str:
        """トーンを維持しながら翻訳"""
        lang_name = self.SUPPORTED_LANGUAGES[target_lang]["name"]
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"""
以下のカスタマーサポート回答を{lang_name}に翻訳してください。
丁寧さレベル: {formality}
カスタマーサポートとしての適切なトーンを維持してください。

原文: {text}

翻訳:
"""}]
        )
        return response.content[0].text
```

---

## 7. メトリクスと評価

### 7.1 サポートメトリクス

```python
class SupportMetrics:
    """サポートエージェントのメトリクス収集・分析"""

    def __init__(self, storage=None):
        self.storage = storage or {}
        self.metrics_buffer = []

    def record(self, customer_id: str, intent: dict,
               result: dict, processing_time: float):
        """メトリクスを記録"""
        metric = {
            "timestamp": time.time(),
            "customer_id": customer_id,
            "intent": intent.get("intent"),
            "sub_intent": intent.get("sub_intent"),
            "sentiment": intent.get("sentiment"),
            "urgency": intent.get("urgency"),
            "escalated": result.get("escalated", False),
            "confidence": result.get("confidence", 0),
            "processing_time_ms": processing_time * 1000,
            "channel": result.get("channel", "web_chat"),
        }
        self.metrics_buffer.append(metric)

        # バッファが100件を超えたらフラッシュ
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()

    def calculate_kpis(self, period_days: int = 30) -> dict:
        """KPIを計算"""
        cutoff = time.time() - (period_days * 86400)
        recent = [m for m in self.metrics_buffer if m["timestamp"] > cutoff]

        if not recent:
            return {"error": "No data available"}

        total = len(recent)
        escalated = sum(1 for m in recent if m["escalated"])
        auto_resolved = total - escalated

        # 意図別の分布
        intent_dist = {}
        for m in recent:
            intent = m["intent"]
            if intent not in intent_dist:
                intent_dist[intent] = 0
            intent_dist[intent] += 1

        # 平均処理時間
        avg_processing_time = sum(
            m["processing_time_ms"] for m in recent
        ) / total

        # 信頼度の分布
        high_confidence = sum(1 for m in recent if m["confidence"] > 0.8)
        medium_confidence = sum(1 for m in recent if 0.5 <= m["confidence"] <= 0.8)
        low_confidence = sum(1 for m in recent if m["confidence"] < 0.5)

        return {
            "period_days": period_days,
            "total_inquiries": total,
            "auto_resolution_rate": auto_resolved / total * 100,
            "escalation_rate": escalated / total * 100,
            "avg_processing_time_ms": avg_processing_time,
            "intent_distribution": intent_dist,
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },
            "sentiment_breakdown": self._count_by_field(recent, "sentiment"),
            "urgency_breakdown": self._count_by_field(recent, "urgency"),
        }

    def generate_improvement_report(self) -> dict:
        """改善レポートを生成"""
        kpis = self.calculate_kpis()
        if "error" in kpis:
            return kpis

        recommendations = []

        # 自動解決率が低い場合
        if kpis["auto_resolution_rate"] < 60:
            recommendations.append({
                "area": "auto_resolution",
                "current": f"{kpis['auto_resolution_rate']:.1f}%",
                "target": "60-80%",
                "suggestion": "ナレッジベースの拡充と意図分類の精度向上を推奨。"
                             "エスカレーション理由の分析を行い、頻出パターンのナレッジを追加。",
            })

        # 低信頼度が多い場合
        total = kpis["total_inquiries"]
        low_conf_rate = kpis["confidence_distribution"]["low"] / total * 100
        if low_conf_rate > 30:
            recommendations.append({
                "area": "knowledge_coverage",
                "current": f"低信頼度 {low_conf_rate:.1f}%",
                "target": "低信頼度 < 20%",
                "suggestion": "ナレッジベースのカバレッジ不足。"
                             "低信頼度の問い合わせを分析し、不足ドキュメントを特定・追加。",
            })

        # 処理時間が長い場合
        if kpis["avg_processing_time_ms"] > 5000:
            recommendations.append({
                "area": "response_time",
                "current": f"{kpis['avg_processing_time_ms']:.0f}ms",
                "target": "< 3000ms",
                "suggestion": "応答時間の改善が必要。"
                             "キャッシュの活用、モデルの最適化、ベクトル検索のチューニングを検討。",
            })

        return {
            "kpis": kpis,
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }

    def _count_by_field(self, records: list, field: str) -> dict:
        counts = {}
        for r in records:
            val = r.get(field, "unknown")
            counts[val] = counts.get(val, 0) + 1
        return counts

    def _flush_metrics(self):
        """メトリクスをストレージに永続化"""
        if self.storage is not None:
            # 実際にはデータベースやメトリクスサービスに送信
            pass
```

### 7.2 CSAT（顧客満足度）収集

```python
class CSATCollector:
    """顧客満足度の収集と分析"""

    def __init__(self, crm):
        self.crm = crm
        self.client = anthropic.Anthropic()

    def generate_survey_prompt(self, ticket: SupportTicket) -> str:
        """チケット解決後のCSATアンケートを生成"""
        return (
            "お問い合わせいただきありがとうございました。\n\n"
            "今回のサポート対応についてお聞かせください。\n\n"
            "1. 非常に不満\n"
            "2. 不満\n"
            "3. 普通\n"
            "4. 満足\n"
            "5. 非常に満足\n\n"
            "番号でお答えください。\n"
            "また、改善点がございましたらコメントもお願いします。"
        )

    def process_csat_response(self, ticket_id: str,
                              response_text: str) -> dict:
        """CSATレスポンスを処理"""
        # スコア抽出
        score = self._extract_score(response_text)
        comment = self._extract_comment(response_text)

        # 感情分析（コメントがある場合）
        sentiment = None
        if comment:
            sentiment = self._analyze_sentiment(comment)

        result = {
            "ticket_id": ticket_id,
            "score": score,
            "comment": comment,
            "sentiment": sentiment,
            "collected_at": datetime.now().isoformat(),
        }

        # 低スコアの場合はアラート
        if score and score <= 2:
            self._trigger_low_csat_alert(ticket_id, result)

        return result

    def _extract_score(self, text: str) -> Optional[int]:
        """テキストからスコアを抽出"""
        for char in text:
            if char.isdigit() and 1 <= int(char) <= 5:
                return int(char)
        return None

    def _extract_comment(self, text: str) -> str:
        """テキストからコメント部分を抽出"""
        # 数字以外の部分をコメントとして扱う
        comment = "".join(c for c in text if not c.isdigit()).strip()
        return comment if len(comment) > 5 else ""

    def _analyze_sentiment(self, comment: str) -> str:
        """コメントの感情を分析"""
        response = self.client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": f"Classify sentiment as positive/neutral/negative: {comment}"}]
        )
        return response.content[0].text.strip().lower()

    def _trigger_low_csat_alert(self, ticket_id: str, result: dict):
        """低CSATスコアのアラートを発火"""
        logger.warning(
            f"Low CSAT alert: ticket={ticket_id}, "
            f"score={result['score']}, comment={result.get('comment', 'N/A')}"
        )
```

---

## 8. マルチチャネル統合

### 8.1 チャネルアダプター

```python
from abc import ABC, abstractmethod


class ChannelAdapter(ABC):
    """チャネルアダプターの基底クラス"""

    @abstractmethod
    def receive_message(self, raw_payload: dict) -> dict:
        """チャネル固有のペイロードを統一形式に変換"""
        pass

    @abstractmethod
    def send_response(self, customer_id: str, message: str,
                      metadata: dict = None) -> bool:
        """統一形式のレスポンスをチャネル固有の形式で送信"""
        pass

    @abstractmethod
    def format_rich_content(self, content: dict) -> dict:
        """リッチコンテンツ（ボタン、カルーセル等）をチャネル形式に変換"""
        pass


class WebChatAdapter(ChannelAdapter):
    """Webチャットアダプター"""

    def receive_message(self, raw_payload: dict) -> dict:
        return {
            "customer_id": raw_payload["user_id"],
            "message": raw_payload["text"],
            "channel": "web_chat",
            "metadata": {
                "page_url": raw_payload.get("current_page"),
                "session_id": raw_payload.get("session_id"),
                "user_agent": raw_payload.get("user_agent"),
            },
        }

    def send_response(self, customer_id: str, message: str,
                      metadata: dict = None) -> bool:
        # WebSocket経由でリアルタイム送信
        return self._send_via_websocket(customer_id, {
            "type": "message",
            "text": message,
            "metadata": metadata,
        })

    def format_rich_content(self, content: dict) -> dict:
        """Webチャット用のリッチコンテンツ"""
        if content["type"] == "buttons":
            return {
                "type": "button_group",
                "buttons": [
                    {"label": b["label"], "value": b["value"],
                     "style": b.get("style", "default")}
                    for b in content["buttons"]
                ],
            }
        elif content["type"] == "carousel":
            return {
                "type": "carousel",
                "items": content["items"],
            }
        return {"type": "text", "text": str(content)}

    def _send_via_websocket(self, customer_id: str,
                            payload: dict) -> bool:
        # WebSocket送信の実装
        return True


class LINEAdapter(ChannelAdapter):
    """LINEアダプター"""

    def __init__(self, channel_access_token: str):
        self.token = channel_access_token
        self.api_base = "https://api.line.me/v2/bot"

    def receive_message(self, raw_payload: dict) -> dict:
        event = raw_payload["events"][0]
        return {
            "customer_id": event["source"]["userId"],
            "message": event["message"]["text"],
            "channel": "line",
            "metadata": {
                "reply_token": event["replyToken"],
                "message_type": event["message"]["type"],
            },
        }

    def send_response(self, customer_id: str, message: str,
                      metadata: dict = None) -> bool:
        import requests
        reply_token = metadata.get("reply_token") if metadata else None

        if reply_token:
            # リプライAPI使用（無料）
            url = f"{self.api_base}/message/reply"
            payload = {
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": message}],
            }
        else:
            # プッシュAPI使用（有料）
            url = f"{self.api_base}/message/push"
            payload = {
                "to": customer_id,
                "messages": [{"type": "text", "text": message}],
            }

        resp = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })
        return resp.status_code == 200

    def format_rich_content(self, content: dict) -> dict:
        """LINE用のFlex Messageに変換"""
        if content["type"] == "buttons":
            return {
                "type": "flex",
                "altText": "選択肢",
                "contents": {
                    "type": "bubble",
                    "body": {
                        "type": "box",
                        "layout": "vertical",
                        "contents": [
                            {
                                "type": "button",
                                "action": {
                                    "type": "message",
                                    "label": b["label"],
                                    "text": b["value"],
                                },
                                "style": "primary" if b.get("style") == "primary" else "secondary",
                            }
                            for b in content["buttons"]
                        ],
                    },
                },
            }
        return {"type": "text", "text": str(content)}


class EmailAdapter(ChannelAdapter):
    """メールアダプター"""

    def __init__(self, smtp_config: dict):
        self.smtp_config = smtp_config

    def receive_message(self, raw_payload: dict) -> dict:
        return {
            "customer_id": raw_payload["from_email"],
            "message": raw_payload["body_text"],
            "channel": "email",
            "metadata": {
                "subject": raw_payload["subject"],
                "message_id": raw_payload["message_id"],
                "in_reply_to": raw_payload.get("in_reply_to"),
                "cc": raw_payload.get("cc", []),
                "attachments": raw_payload.get("attachments", []),
            },
        }

    def send_response(self, customer_id: str, message: str,
                      metadata: dict = None) -> bool:
        import smtplib
        from email.mime.text import MIMEText

        msg = MIMEText(message, "plain", "utf-8")
        msg["Subject"] = f"Re: {metadata.get('subject', 'お問い合わせ')}"
        msg["From"] = self.smtp_config["from_email"]
        msg["To"] = customer_id
        if metadata and metadata.get("message_id"):
            msg["In-Reply-To"] = metadata["message_id"]

        try:
            with smtplib.SMTP(self.smtp_config["host"],
                              self.smtp_config["port"]) as server:
                server.starttls()
                server.login(
                    self.smtp_config["username"],
                    self.smtp_config["password"],
                )
                server.send_message(msg)
            return True
        except Exception as e:
            logger.error(f"Email send failed: {e}")
            return False

    def format_rich_content(self, content: dict) -> dict:
        """メール用のHTML形式に変換"""
        if content["type"] == "buttons":
            html_buttons = "".join([
                f'<a href="{b["value"]}" style="display:inline-block;padding:10px 20px;'
                f'background:#007bff;color:white;text-decoration:none;border-radius:4px;'
                f'margin:5px;">{b["label"]}</a>'
                for b in content["buttons"]
            ])
            return {"type": "html", "content": html_buttons}
        return {"type": "text", "text": str(content)}


class ChannelRouter:
    """チャネルルーター: 統一的なメッセージルーティング"""

    def __init__(self, support_agent: CustomerSupportAgent):
        self.agent = support_agent
        self.adapters: dict[str, ChannelAdapter] = {}

    def register_adapter(self, channel: str, adapter: ChannelAdapter):
        """チャネルアダプターを登録"""
        self.adapters[channel] = adapter

    def route_message(self, channel: str, raw_payload: dict) -> dict:
        """メッセージをルーティングして処理"""
        adapter = self.adapters.get(channel)
        if not adapter:
            raise ValueError(f"Unknown channel: {channel}")

        # 1. チャネル固有形式を統一形式に変換
        unified = adapter.receive_message(raw_payload)

        # 2. サポートエージェントで処理
        result = self.agent.handle_inquiry(
            customer_id=unified["customer_id"],
            message=unified["message"],
            channel=channel,
        )

        # 3. レスポンスをチャネル固有形式で送信
        adapter.send_response(
            customer_id=unified["customer_id"],
            message=result["response"],
            metadata=unified.get("metadata"),
        )

        return result
```

---

## 9. プロアクティブサポート

### 9.1 問題予兆検知

```python
class ProactiveSupportEngine:
    """プロアクティブサポート: 問題が発生する前に先回り対応"""

    def __init__(self, analytics_db, notification_service, llm_client):
        self.analytics = analytics_db
        self.notification = notification_service
        self.client = llm_client

    def detect_at_risk_customers(self) -> list:
        """チャーンリスクのある顧客を検出"""
        indicators = [
            self._check_usage_decline(),
            self._check_repeated_errors(),
            self._check_support_frequency(),
            self._check_payment_issues(),
        ]

        at_risk = set()
        for indicator_results in indicators:
            for customer_id, risk_score in indicator_results:
                if risk_score > 0.7:
                    at_risk.add(customer_id)

        return list(at_risk)

    def _check_usage_decline(self) -> list:
        """利用量の減少を検出"""
        # 直近30日と前30日の比較
        results = self.analytics.query("""
            SELECT customer_id,
                   current_usage / NULLIF(previous_usage, 0) as usage_ratio
            FROM (
                SELECT customer_id,
                       SUM(CASE WHEN date >= CURRENT_DATE - 30
                           THEN usage_count ELSE 0 END) as current_usage,
                       SUM(CASE WHEN date >= CURRENT_DATE - 60
                           AND date < CURRENT_DATE - 30
                           THEN usage_count ELSE 0 END) as previous_usage
                FROM usage_logs
                GROUP BY customer_id
            ) sub
            WHERE current_usage / NULLIF(previous_usage, 0) < 0.5
        """)
        return [(r["customer_id"], 1 - r["usage_ratio"]) for r in results]

    def _check_repeated_errors(self) -> list:
        """繰り返しエラーを検出"""
        results = self.analytics.query("""
            SELECT customer_id, COUNT(*) as error_count
            FROM error_logs
            WHERE timestamp >= CURRENT_DATE - 7
            GROUP BY customer_id
            HAVING COUNT(*) >= 5
        """)
        return [
            (r["customer_id"], min(r["error_count"] / 20, 1.0))
            for r in results
        ]

    def _check_support_frequency(self) -> list:
        """サポート問い合わせ頻度の急増を検出"""
        results = self.analytics.query("""
            SELECT customer_id, COUNT(*) as ticket_count
            FROM support_tickets
            WHERE created_at >= CURRENT_DATE - 14
            GROUP BY customer_id
            HAVING COUNT(*) >= 3
        """)
        return [
            (r["customer_id"], min(r["ticket_count"] / 10, 1.0))
            for r in results
        ]

    def _check_payment_issues(self) -> list:
        """支払い問題を検出"""
        results = self.analytics.query("""
            SELECT customer_id, COUNT(*) as failure_count
            FROM payment_events
            WHERE status = 'failed'
            AND timestamp >= CURRENT_DATE - 30
            GROUP BY customer_id
            HAVING COUNT(*) >= 1
        """)
        return [
            (r["customer_id"], min(r["failure_count"] / 3, 1.0))
            for r in results
        ]

    def generate_proactive_message(self, customer_id: str,
                                   risk_indicators: list) -> str:
        """プロアクティブメッセージを生成"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": f"""
以下のリスク指標に基づいて、顧客への先回りサポートメッセージを生成してください。

顧客ID: {customer_id}
リスク指標: {json.dumps(risk_indicators, ensure_ascii=False)}

ルール:
- 押し付けがましくない
- 具体的な支援を提案する
- 顧客の状況への理解を示す
- 短く簡潔に（200文字以内）

メッセージ:
"""}]
        )
        return response.content[0].text
```

### 9.2 自動フォローアップ

```python
class AutoFollowUpManager:
    """自動フォローアップ管理"""

    FOLLOW_UP_RULES = {
        "after_resolution": {
            "delay_hours": 24,
            "message_template": "先日のお問い合わせ（{subject}）について、その後問題は解決しましたでしょうか？",
        },
        "after_escalation": {
            "delay_hours": 4,
            "message_template": "エスカレーションチケット（{ticket_id}）の進捗をお知らせします。現在、{status}です。",
        },
        "after_feature_request": {
            "delay_days": 30,
            "message_template": "先日いただいた機能要望（{feature}）について、開発チームからアップデートがあります。",
        },
        "payment_retry": {
            "delay_hours": 2,
            "message_template": "お支払いの処理に問題がございました。お手数ですが、お支払い方法をご確認ください。",
        },
    }

    def __init__(self, scheduler, channel_router: ChannelRouter):
        self.scheduler = scheduler
        self.router = channel_router

    def schedule_follow_up(self, ticket: SupportTicket,
                           follow_up_type: str,
                           custom_params: dict = None):
        """フォローアップをスケジュール"""
        rule = self.FOLLOW_UP_RULES.get(follow_up_type)
        if not rule:
            return

        params = custom_params or {}
        message = rule["message_template"].format(
            subject=ticket.subject,
            ticket_id=ticket.ticket_id,
            **params,
        )

        delay_seconds = rule.get("delay_hours", 0) * 3600
        delay_seconds += rule.get("delay_days", 0) * 86400

        self.scheduler.schedule(
            delay_seconds=delay_seconds,
            callback=self._send_follow_up,
            args=(ticket.customer_id, message, ticket.ticket_id),
        )

    def _send_follow_up(self, customer_id: str, message: str,
                        ticket_id: str):
        """フォローアップを送信"""
        # チケットがまだ未解決の場合のみ送信
        # 実際にはチケット状態を確認
        logger.info(
            f"Sending follow-up to {customer_id} for ticket {ticket_id}"
        )
```

---

## 10. ナレッジベース管理

### 10.1 ナレッジ自動生成

```python
class KnowledgeBaseManager:
    """ナレッジベースの自動生成・更新・品質管理"""

    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.client = llm_client

    def generate_from_resolved_tickets(self, tickets: list) -> list:
        """解決済みチケットからナレッジ記事を自動生成"""
        new_articles = []

        # 類似チケットをクラスタリング
        clusters = self._cluster_similar_tickets(tickets)

        for cluster in clusters:
            if len(cluster) < 3:  # 3件未満は汎用性が低い
                continue

            # 代表的な質問と回答を生成
            article = self._generate_article(cluster)
            if article:
                new_articles.append(article)

        return new_articles

    def _cluster_similar_tickets(self, tickets: list) -> list:
        """類似チケットをクラスタリング"""
        # 簡易実装: 意図+サブインテントでグルーピング
        clusters = {}
        for ticket in tickets:
            key = f"{ticket.get('intent', 'general')}_{ticket.get('sub_intent', 'none')}"
            if key not in clusters:
                clusters[key] = []
            clusters[key].append(ticket)
        return list(clusters.values())

    def _generate_article(self, cluster: list) -> Optional[dict]:
        """チケットクラスタからナレッジ記事を生成"""
        # 代表的な質問と回答を収集
        examples = []
        for ticket in cluster[:5]:
            examples.append({
                "question": ticket.get("initial_message", ""),
                "resolution": ticket.get("resolution_summary", ""),
            })

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": f"""
以下の解決済みサポートチケットから、ナレッジベース記事を生成してください。

チケット例:
{json.dumps(examples, ensure_ascii=False, indent=2)}

JSON形式で出力:
{{
  "title": "記事タイトル",
  "category": "カテゴリ",
  "question": "よくある質問の形式",
  "answer": "回答（手順がある場合は番号付きリスト）",
  "keywords": ["検索用キーワード"],
  "related_articles": ["関連記事のキーワード"]
}}
"""}]
        )

        try:
            article = json.loads(response.content[0].text)
            article["source_ticket_count"] = len(cluster)
            article["generated_at"] = datetime.now().isoformat()
            return article
        except json.JSONDecodeError:
            return None

    def audit_knowledge_base(self) -> dict:
        """ナレッジベースの品質監査"""
        all_articles = self.vector_store.get_all_articles()

        stale_articles = []
        duplicate_candidates = []
        low_usage = []

        for article in all_articles:
            # 古い記事の検出
            updated = article.get("updated_at")
            if updated:
                age_days = (datetime.now() - datetime.fromisoformat(updated)).days
                if age_days > 180:
                    stale_articles.append({
                        "id": article["id"],
                        "title": article["title"],
                        "age_days": age_days,
                    })

            # 利用頻度の低い記事
            usage = article.get("usage_count", 0)
            if usage < 5:
                low_usage.append({
                    "id": article["id"],
                    "title": article["title"],
                    "usage_count": usage,
                })

        return {
            "total_articles": len(all_articles),
            "stale_articles": stale_articles,
            "low_usage_articles": low_usage,
            "audit_date": datetime.now().isoformat(),
        }
```

---

## 11. アンチパターン

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

### アンチパターン3: コンテキスト無視

```python
# NG: 毎回同じ質問を最初からやり直し
def handle_message(customer_id, message):
    # 過去の会話を見ない
    return generate_answer(message)

# OK: 会話コンテキストを維持
def handle_message(customer_id, message):
    session = conversation_manager.get_session(customer_id)
    context = session.get("messages", [])
    variables = session.get("context_variables", {})
    return generate_answer(message, context=context, variables=variables)
```

### アンチパターン4: 感情の無視

```python
# NG: 感情に関係なく事務的に対応
def respond(message, intent):
    answer = knowledge_base.search(message)
    return answer  # 感情に関係なく同じトーン

# OK: 感情に応じたトーン調整
def respond(message, intent):
    answer = knowledge_base.search(message)
    sentiment = intent.get("sentiment", "neutral")
    if sentiment in ("negative", "angry"):
        answer = tone_adjuster.add_empathy(answer, sentiment)
    return answer
```

### アンチパターン5: ハルシネーション（幻覚）の放置

```python
# NG: LLMの回答をそのまま返す
def answer_question(question):
    response = llm.generate(question)
    return response  # ナレッジに無い情報も自信満々に回答

# OK: ナレッジベースの情報のみ使用し、不明な場合はエスカレーション
def answer_question(question):
    docs = knowledge_base.search(question, top_k=5)
    if not docs or max(d["similarity"] for d in docs) < 0.7:
        return {"answer": "確認いたします", "should_escalate": True}
    response = llm.generate(question, context=docs)
    return {"answer": response, "grounded": True}
```

### アンチパターン6: 個人情報の無防備な扱い

```python
# NG: 個人情報をそのままLLMに渡す
def handle_billing(customer):
    prompt = f"顧客 {customer['name']}、カード番号 {customer['card_number']}..."
    return llm.generate(prompt)

# OK: 個人情報をマスクしてからLLMに渡す
def handle_billing(customer):
    masked = {
        "name": mask_name(customer["name"]),
        "card": f"****{customer['card_number'][-4:]}",
        "plan": customer["plan"],  # 非機密情報はそのまま
    }
    prompt = f"顧客プラン: {masked['plan']}、カード末尾: {masked['card']}..."
    return llm.generate(prompt)
```

---

## 12. 実装チェックリスト

### Must（必須）

- [ ] 意図分類の実装（信頼度付き）
- [ ] RAGベースのナレッジ検索・回答生成
- [ ] 信頼度ベースのエスカレーション判定
- [ ] 人間オペレーターへのハンドオフ機構
- [ ] 個人情報のマスキング処理
- [ ] 回答へのソース（根拠）の付与
- [ ] 基本的なKPI計測（自動解決率、処理時間）
- [ ] NG表現フィルタリング

### Should（推奨）

- [ ] マルチターン会話管理
- [ ] 顧客の感情に応じたトーン調整
- [ ] CSAT（顧客満足度）収集の仕組み
- [ ] マルチチャネル対応（Web + LINE等）
- [ ] 会話コンテキストの圧縮（長い会話対応）
- [ ] ナレッジベースの品質監査
- [ ] 2段階意図分類（低信頼度時の再分類）
- [ ] 本人確認フロー

### Nice to Have（あると良い）

- [ ] プロアクティブサポート（チャーン予兆検知）
- [ ] 解決済みチケットからのナレッジ自動生成
- [ ] 自動フォローアップ
- [ ] 多言語対応
- [ ] A/Bテストによる回答品質の継続改善
- [ ] ダッシュボードでのリアルタイムモニタリング
- [ ] VoC（Voice of Customer）分析

---

## 13. FAQ

### Q1: サポートエージェントの効果をどう測定する？

主要KPI:
- **自動解決率**: 人間の介入なしに解決した割合（目標: 60-80%）
- **初回回答解決率（FCR）**: 最初の回答で問題が解決した割合
- **CSAT**: 顧客満足度スコア（1-5）
- **平均対応時間（AHT）**: 問い合わせから解決までの時間
- **エスカレーション率**: 人間に引き継いだ割合
- **コスト/チケット**: 1チケットあたりの対応コスト

副次指標:
- **リピート問い合わせ率**: 同じ問題での再問い合わせ割合
- **ナレッジヒット率**: ナレッジベースから回答が見つかった割合
- **顧客離反率（チャーン率）**: サポート後の解約率
- **NPS（Net Promoter Score）**: 推奨度スコア

### Q2: 多言語対応の方法は？

2つのアプローチ:
1. **検出→翻訳→処理→翻訳**: 入力言語を検出し、内部処理は単一言語で行い、回答を元言語に翻訳
2. **ネイティブ多言語**: LLMの多言語能力を活用し、入力言語のまま処理・回答

Claudeの場合は後者が推奨。日本語入力にそのまま日本語で回答可能。ただし、ナレッジベースが日本語のみの場合、他言語の質問に対する検索精度が落ちる可能性がある。その場合は質問を日本語に翻訳してから検索し、回答を元言語に戻すハイブリッドアプローチが有効。

### Q3: 個人情報の扱いは？

- **マスキング**: クレジットカード番号、パスワード等はマスクしてからLLMに渡す
- **ログ管理**: 会話ログから個人情報を除外して保存（PII検出ツール活用）
- **データ保持期間**: GDPR/個人情報保護法に準拠した保持期間設定
- **LLMプロバイダのポリシー確認**: データがモデル学習に使われないことを確認
- **暗号化**: 保存時・転送時の暗号化を徹底
- **アクセス制御**: ログへのアクセスを最小権限原則で管理

### Q4: ナレッジベースのメンテナンス頻度は？

推奨サイクル:
- **日次**: 解決済みチケットからの候補記事生成（自動）
- **週次**: 低信頼度回答のレビューとナレッジ追加
- **月次**: 利用頻度の低い記事のレビュー・更新・アーカイブ
- **四半期**: 全体的な品質監査と構造の見直し

ナレッジベースの鮮度が回答品質に直結するため、定期的な更新プロセスをチーム内に組み込むことが重要。

### Q5: エスカレーション先の人間オペレーターが不在の場合は？

対応策:
1. **非同期対応**: メールやチケットシステムで非同期にエスカレーションし、営業時間内に対応
2. **待ち時間通知**: 推定待ち時間を顧客に伝え、コールバック予約の提案
3. **部分回答**: AIが可能な範囲で部分的に回答し、残りを人間が対応
4. **オンコール体制**: 重要度の高い問い合わせ用にオンコールオペレーターを配置

```python
def handle_after_hours_escalation(ticket, customer):
    """営業時間外のエスカレーション処理"""
    if ticket.urgency == "critical" and customer.tier == "enterprise":
        # エンタープライズ顧客のクリティカルはオンコール
        return notify_on_call_team(ticket)

    # 通常はチケット作成+翌営業日対応
    ticket_id = create_async_ticket(ticket)
    return {
        "response": f"申し訳ございませんが、現在営業時間外です。"
                    f"\n\nチケット番号: {ticket_id}"
                    f"\n翌営業日（{next_business_day()}）に担当者よりご連絡いたします。"
                    f"\n\nお急ぎの場合は、ヘルプセンター(https://help.example.com)もご利用ください。",
        "escalated": True,
        "async": True,
    }
```

### Q6: ボットだと気づかれないようにすべきか？

AIであることは**明示すべき**。理由:
1. **透明性**: 顧客はAIと話していることを知る権利がある
2. **期待値管理**: 人間ではないことを理解していると、能力の限界に対する不満が減る
3. **法規制**: 一部の地域ではAI開示が法的要件

推奨アプローチ:
```
初回メッセージ例:
"こんにちは！AIアシスタントの[名前]です。
お問い合わせ内容を確認し、お手伝いいたします。
必要に応じて、人間のオペレーターにおつなぎすることも可能です。"
```

### Q7: サポートエージェントの立ち上げに必要なデータ量は？

最低限必要なデータ:
- **FAQ/ナレッジ記事**: 50-100件以上
- **過去の対応履歴**: 500-1000件（意図分類のチューニング用）
- **製品ドキュメント**: 主要機能のドキュメント全般

段階的なアプローチ:
1. まずFAQの上位20問から始める
2. 2-4週間のパイロット運用でデータを蓄積
3. エスカレーション理由を分析してナレッジを拡充
4. 月次でカバレッジを10-20%ずつ拡大

---

## まとめ

| 項目 | 内容 |
|------|------|
| コアフロー | 意図分類 → ナレッジ検索 → 回答生成 → エスカレーション |
| 意図分類 | 高速モデル（Haiku）で分類、感情・緊急度も判定 |
| 回答生成 | RAGベース、ナレッジベースの情報のみ使用 |
| エスカレーション | 信頼度 x 感情 x 顧客ティアのマトリクスで判断 |
| トーン | 顧客の感情に応じた言葉遣いの調整 |
| マルチチャネル | アダプターパターンで統一的なルーティング |
| プロアクティブ | 問題予兆検知、チャーンリスク分析、自動フォローアップ |
| ナレッジ管理 | 解決済みチケットからの自動生成、品質監査 |
| KPI | 自動解決率、CSAT、平均対応時間、コスト/チケット |

## 次に読むべきガイド

- [03-data-agents.md](./03-data-agents.md) -- データ分析エージェント
- [../01-patterns/02-workflow-agents.md](../01-patterns/02-workflow-agents.md) -- ワークフロー設計の詳細
- [../04-production/00-deployment.md](../04-production/00-deployment.md) -- サポートエージェントのデプロイ

## 参考文献

1. Anthropic, "Customer service agent cookbook" -- https://docs.anthropic.com/en/docs/about-claude/use-case-guides/customer-service
2. Zendesk, "AI in Customer Service" -- https://www.zendesk.com/blog/ai-customer-service/
3. Intercom, "AI-First Customer Service" -- https://www.intercom.com/ai-bot
4. Gartner, "Predicts: Customer Service and Support" -- https://www.gartner.com/en/customer-service-support
5. McKinsey, "The next frontier of customer engagement: AI-enabled customer service" -- https://www.mckinsey.com/capabilities/operations/our-insights
