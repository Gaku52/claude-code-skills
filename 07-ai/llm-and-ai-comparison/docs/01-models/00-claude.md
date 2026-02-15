# Claude — Anthropic の AI アシスタント

> Constitutional AI を基盤とする Claude ファミリーの特徴、API 活用法、他モデルとの差別化ポイントを解説する。

## この章で学ぶこと

1. **Claude ファミリー**の各モデル（Haiku / Sonnet / Opus）の特性と使い分け
2. **Constitutional AI** の原理と Claude の安全性設計
3. **Claude API** の実践的な使い方とベストプラクティス
4. **Extended Thinking** による高精度推論
5. **Claude Code** とエージェント的活用
6. **MCP (Model Context Protocol)** によるツール統合

---

## 1. Claude ファミリーの概要

### ASCII 図解 1: Claude モデルファミリー

```
Claude モデルファミリー (2024-2025)
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Claude 3.5 / 4 ファミリー                                  │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Haiku     │  │   Sonnet    │  │    Opus     │     │
│  │              │  │              │  │              │     │
│  │ 超高速       │  │ バランス     │  │ 最高性能     │     │
│  │ 最低コスト    │  │ コスパ最強   │  │ 複雑推論     │     │
│  │ 分類/抽出    │  │ 汎用        │  │ 研究/分析    │     │
│  │ リアルタイム  │  │ コーディング │  │ 長文創作     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                            │
│  性能:   Haiku ←───── Sonnet ────── Opus ──→ 高            │
│  速度:   Haiku ──→ 高  Sonnet ──── Opus ←─── 低            │
│  コスト: Haiku ←── 低  Sonnet ──── Opus ──→ 高             │
│                                                            │
│  共通: 200K コンテキスト、マルチモーダル対応、ツール使用      │
│                                                            │
│  Claude 4 Opus の追加機能:                                  │
│  ├── Extended Thinking（拡張思考モード）                     │
│  ├── 超長文の正確な理解と生成                                │
│  ├── 高度なコード理解・デバッグ                              │
│  └── マルチステップの複雑な推論                              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.1 モデル選択の判断基準

```
┌────────────────────────────────────────────────────────────┐
│              Claude モデル選択ガイド                         │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Haiku を選ぶべき場面:                                      │
│  ├── 大量のテキスト分類（感情分析、カテゴリ分類）            │
│  ├── シンプルなデータ抽出（名前、日付、金額の取得）          │
│  ├── 定型的な応答生成（FAQ、テンプレート記入）              │
│  ├── リアルタイム応答が必要（チャットボット）               │
│  ├── コスト制約が厳しい大量処理                             │
│  └── 前処理/フィルタリング（ルーティングの前段）            │
│                                                            │
│  Sonnet を選ぶべき場面:                                     │
│  ├── コード生成・レビュー                                   │
│  ├── 文書作成・翻訳                                        │
│  ├── 中程度の分析・要約                                     │
│  ├── ツール使用を伴うタスク                                 │
│  ├── RAG パイプラインの回答生成                             │
│  └── コストと品質のバランスが重要な場面                      │
│                                                            │
│  Opus を選ぶべき場面:                                       │
│  ├── 複雑な多段階推論（数学、論理パズル）                   │
│  ├── 大規模コードベースの理解・リファクタリング              │
│  ├── 研究論文の深い分析・批評                               │
│  ├── 長文の創作・編集                                       │
│  ├── 微妙なニュアンスの理解が必要な場面                      │
│  └── Extended Thinking が有効な複雑問題                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### コード例 1: Claude API の基本使用法

```python
import anthropic

client = anthropic.Anthropic()  # ANTHROPIC_API_KEY 環境変数を使用

# 基本的なメッセージ送信
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="あなたは日本語で回答する技術アシスタントです。",
    messages=[
        {"role": "user", "content": "Pythonのデコレータを説明してください"}
    ]
)

print(response.content[0].text)
print(f"入力トークン: {response.usage.input_tokens}")
print(f"出力トークン: {response.usage.output_tokens}")
print(f"モデル: {response.model}")
print(f"停止理由: {response.stop_reason}")
```

### コード例 2: ストリーミング応答

```python
import anthropic

client = anthropic.Anthropic()

# ストリーミングで応答を受け取る（体感レイテンシの大幅改善）
with client.messages.stream(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system="あなたは技術文書の専門家です。",
    messages=[
        {"role": "user", "content": "マイクロサービスアーキテクチャの利点と課題を説明してください"}
    ]
) as stream:
    full_response = ""
    for text in stream.text_stream:
        print(text, end="", flush=True)
        full_response += text

    print()

    # ストリーム完了後のメタデータ
    final_message = stream.get_final_message()
    print(f"\n入力トークン: {final_message.usage.input_tokens}")
    print(f"出力トークン: {final_message.usage.output_tokens}")
```

### コード例 3: マルチターン会話

```python
import anthropic
from typing import List, Dict

class ClaudeConversation:
    """Claude とのマルチターン会話管理"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        system: str = "",
        max_tokens: int = 2048,
    ):
        self.client = anthropic.Anthropic()
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.messages: List[Dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, user_message: str) -> str:
        """メッセージを送信して応答を取得"""
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
        )

        assistant_message = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_message})

        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return assistant_message

    def reset(self):
        """会話をリセット"""
        self.messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_stats(self) -> Dict:
        """会話の統計情報"""
        return {
            "turns": len(self.messages) // 2,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
        }

# 使用例
conv = ClaudeConversation(
    system="あなたは親切なプログラミング講師です。段階的に教えてください。"
)

print(conv.chat("Pythonのリスト内包表記を教えてください"))
print(conv.chat("条件付きのリスト内包表記はどう書きますか？"))
print(conv.chat("ネストされた場合はどうなりますか？"))
print(f"\n統計: {conv.get_stats()}")
```

### コード例 4: Vision（画像理解）

```python
import anthropic
import base64
import httpx

client = anthropic.Anthropic()

# パターン1: ローカルファイルから
def analyze_local_image(image_path: str, prompt: str) -> str:
    """ローカル画像を分析"""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # 拡張子からメディアタイプを判定
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    ext = "." + image_path.rsplit(".", 1)[-1].lower()
    media_type = media_types.get(ext, "image/png")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data,
                    },
                },
                {"type": "text", "text": prompt}
            ],
        }]
    )
    return response.content[0].text

# パターン2: URLから
def analyze_url_image(image_url: str, prompt: str) -> str:
    """URL画像を分析"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image_url,
                    },
                },
                {"type": "text", "text": prompt}
            ],
        }]
    )
    return response.content[0].text

# パターン3: 複数画像の比較
def compare_images(image_paths: list, prompt: str) -> str:
    """複数画像を比較分析"""
    content = []
    for path in image_paths:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        })
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text

# 使用例
result = analyze_local_image(
    "architecture_diagram.png",
    "このアーキテクチャ図を分析し、以下を教えてください:\n"
    "1. 全体構成の概要\n"
    "2. 潜在的なボトルネック\n"
    "3. 改善提案"
)
print(result)
```

---

## 2. Constitutional AI

### ASCII 図解 2: Constitutional AI の仕組み

```
Constitutional AI (CAI) のプロセス:

Phase 1: 自己批評 (Supervised Learning)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 有害な        │ →  │ モデルが応答  │ →  │ 憲法原則に   │
│ プロンプトを  │    │ を生成       │    │ 基づき自己   │
│ 収集          │    │ (Red Team)   │    │ 批評を実行   │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ 改善版の応答  │
                                        │ を自己生成    │
                                        │ (Revision)   │
                                        └──────────────┘

Phase 2: RLAIF (RL from AI Feedback)
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 応答ペア      │ →  │ AI が憲法原則│ →  │ 報酬モデルを │
│ (元 vs 改善)  │    │ に基づき好み │    │ 学習し       │
└──────────────┘    │ を判定       │    │ PPO で最適化 │
                    │ (人間の代わり)│    └──────────────┘
                    └──────────────┘

「憲法」（Constitution）の例:
┌──────────────────────────────────────────────────────┐
│ 1. 有用性: ユーザーの意図を正確に理解し助けること      │
│ 2. 正直さ: 不確実な場合は正直にそう述べること          │
│ 3. 無害性: 危害を助長する情報を提供しないこと          │
│ 4. 公正性: 偏見や差別的な応答を避けること              │
│ 5. プライバシー: 個人情報を保護すること                │
│ 6. 透明性: AIであることを隠さないこと                  │
└──────────────────────────────────────────────────────┘

従来の RLHF との違い:
  RLHF:  人間のアノテーターが好みを判定 → 高コスト、スケールしにくい
  RLAIF: AI が原則に基づいて判定 → 低コスト、一貫性が高い、スケール可能
```

### 2.1 Claude の安全性の実務的影響

```
┌────────────────────────────────────────────────────────────┐
│          Claude の安全性特性がプロダクトに与える影響          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  利点:                                                     │
│  ├── ガードレールが組み込み済み → 追加の安全対策が軽量     │
│  ├── 不適切な出力が少ない → ブランドリスクの低減           │
│  ├── 不確実性の表明 → ハルシネーションの自己申告           │
│  ├── 拒否の品質が高い → ユーザー体験を損なわない拒否      │
│  └── 多言語での安全性 → 日本語でも安全性が維持される       │
│                                                            │
│  注意点:                                                   │
│  ├── 過度な安全性 → 正当なリクエストの過剰拒否（稀に）    │
│  ├── 医療/法律の免責 → 常に「専門家に相談」と付記        │
│  ├── 創作の制限 → 暴力的/性的な創作コンテンツに制限      │
│  └── 最新情報 → 学習データの期限がある                    │
│                                                            │
│  対策:                                                     │
│  ├── システムプロンプトで許容範囲を明示                    │
│  ├── コンテキストを十分に提供して誤解を防ぐ                │
│  └── 必要に応じて Anthropic のカスタム対応を相談           │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### コード例 5: システムプロンプトでの安全性活用

```python
import anthropic

client = anthropic.Anthropic()

# Claude の安全性特性を活用したシステム設計
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    system="""あなたは企業の法務アシスタントです。

以下の原則に従ってください:
1. 法的助言ではなく一般的な情報提供であることを明記する
2. 不確実な場合は「確認が必要」と述べる
3. 個人情報の取り扱いには特に注意する
4. 専門家への相談を適切に推奨する
5. 関連する法律や条文を参照する際は正確を期す
6. 回答の最後に免責事項を記載する

Claude の Constitutional AI による安全性に加え、
上記のアプリケーション固有のガードレールを設定しています。""",
    messages=[{
        "role": "user",
        "content": "退職時の有給休暇の扱いについて教えてください"
    }]
)

print(response.content[0].text)
```

---

## 3. Claude API の高度な機能

### 3.1 Extended Thinking（拡張思考）

```python
import anthropic

client = anthropic.Anthropic()

# Extended Thinking を有効にした推論
response = client.messages.create(
    model="claude-opus-4-20250514",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000,  # 思考に使えるトークン数
    },
    messages=[{
        "role": "user",
        "content": """以下の複雑なシステム設計問題を分析してください。

大規模ECサイトで以下の要件を満たすアーキテクチャを設計:
1. 秒間10,000リクエストの処理
2. 99.99%の可用性
3. 在庫の整合性保証（二重販売の防止）
4. グローバル展開（日本、米国、欧州）
5. リアルタイムの価格更新
6. 不正検知の組み込み

トレードオフを含めて詳細に分析してください。"""
    }]
)

# 思考過程と最終回答を分離して表示
for block in response.content:
    if block.type == "thinking":
        print("=== 思考過程 ===")
        print(block.thinking[:500] + "...")  # 長いので一部のみ
    elif block.type == "text":
        print("\n=== 最終回答 ===")
        print(block.text)

print(f"\n思考トークン: {response.usage.cache_creation_input_tokens}")
print(f"出力トークン: {response.usage.output_tokens}")
```

### ASCII 図解 3: Extended Thinking の仕組み

```
┌────────────────────────────────────────────────────────────┐
│              Extended Thinking の動作                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  通常モード:                                               │
│  入力 ──→ [Claude の推論] ──→ 出力                         │
│            (内部で完結)                                     │
│                                                            │
│  Extended Thinking モード:                                 │
│  入力 ──→ [<thinking>...</thinking>] ──→ 出力              │
│            思考過程が明示される                              │
│            ├── 問題の分解                                   │
│            ├── 複数のアプローチの検討                        │
│            ├── トレードオフの分析                            │
│            ├── 自己検証と修正                               │
│            └── 結論の導出                                   │
│                                                            │
│  効果:                                                     │
│  ├── 数学・論理問題: 正解率 +20-40%                        │
│  ├── コード生成: バグ率 -30%                               │
│  ├── 複雑な分析: 根拠の透明化                              │
│  └── トークンコスト: 思考分が追加（budget で制御可能）       │
│                                                            │
│  いつ使うべきか:                                           │
│  ✓ 複雑な多段階推論                                       │
│  ✓ 高精度が求められる分析                                  │
│  ✓ 数学的・論理的問題                                     │
│  ✗ 単純な応答生成                                          │
│  ✗ リアルタイム性が重要                                    │
│  ✗ コスト最適化が優先                                      │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 3.2 ツール使用（Function Calling）

```python
import anthropic
import json

client = anthropic.Anthropic()

# 複数ツールの定義
tools = [
    {
        "name": "get_weather",
        "description": "指定された都市の現在の天気を取得する",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "都市名（例: 東京、大阪、ニューヨーク）"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度の単位",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_database",
        "description": "社内データベースから情報を検索する",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "検索クエリ"
                },
                "table": {
                    "type": "string",
                    "enum": ["customers", "orders", "products"],
                    "description": "検索対象テーブル"
                },
                "limit": {
                    "type": "integer",
                    "description": "最大件数",
                    "default": 10
                }
            },
            "required": ["query", "table"]
        }
    },
    {
        "name": "send_email",
        "description": "メールを送信する",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "宛先メールアドレス"},
                "subject": {"type": "string", "description": "件名"},
                "body": {"type": "string", "description": "本文"},
            },
            "required": ["to", "subject", "body"]
        }
    }
]

def execute_tool(name: str, input_data: dict) -> str:
    """ツールを実行する（実際のアプリケーションでは外部APIを呼び出す）"""
    if name == "get_weather":
        return json.dumps({
            "city": input_data["city"],
            "temperature": 22,
            "condition": "晴れ",
            "humidity": 45,
        }, ensure_ascii=False)
    elif name == "search_database":
        return json.dumps({
            "results": [{"id": 1, "name": "テストデータ"}],
            "total": 1,
        }, ensure_ascii=False)
    elif name == "send_email":
        return json.dumps({"status": "sent", "message_id": "msg_123"})
    return json.dumps({"error": "Unknown tool"})

def chat_with_tools(user_message: str) -> str:
    """ツール使用を含むチャット"""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )

        # ツール呼び出しがない場合は最終回答
        if response.stop_reason == "end_turn":
            return response.content[0].text

        # ツール呼び出しの処理
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

# 使用例
print(chat_with_tools("東京の天気を調べて、それに基づいた服装アドバイスをください"))
```

### 3.3 プロンプトキャッシュ

```python
import anthropic

client = anthropic.Anthropic()

# 大量のコンテキスト（ドキュメント、コードベース等）をキャッシュ
large_context = """
（ここに大量のドキュメント、API仕様書、コードベースなど）
... 数千〜数万トークンのコンテンツ ...
"""

# 1回目: キャッシュを作成
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "あなたは以下のドキュメントに基づいて質問に答えるアシスタントです。"
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}  # キャッシュ対象
        }
    ],
    messages=[{"role": "user", "content": "このドキュメントの概要を教えてください"}],
)

print(f"1回目 - キャッシュ作成トークン: {response1.usage.cache_creation_input_tokens}")
print(f"1回目 - キャッシュ読み取り: {response1.usage.cache_read_input_tokens}")

# 2回目以降: キャッシュから読み取り（90%コスト削減）
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "あなたは以下のドキュメントに基づいて質問に答えるアシスタントです。"
        },
        {
            "type": "text",
            "text": large_context,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[{"role": "user", "content": "セキュリティに関する部分を説明してください"}],
)

print(f"2回目 - キャッシュ作成トークン: {response2.usage.cache_creation_input_tokens}")
print(f"2回目 - キャッシュ読み取り: {response2.usage.cache_read_input_tokens}")
# → キャッシュヒットにより大幅なコスト削減
```

### 3.4 バッチ API

```python
import anthropic
import json

client = anthropic.Anthropic()

# 大量のリクエストをバッチで処理（50%コスト削減、24時間以内に完了）
batch_requests = []
for i in range(100):
    batch_requests.append({
        "custom_id": f"request_{i}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": f"製品レビュー #{i} を分析してください: ..."}
            ]
        }
    })

# バッチの作成
batch = client.batches.create(requests=batch_requests)
print(f"バッチID: {batch.id}")
print(f"ステータス: {batch.processing_status}")

# バッチのステータス確認
batch_status = client.batches.retrieve(batch.id)
print(f"処理済み: {batch_status.request_counts.succeeded}")
print(f"失敗: {batch_status.request_counts.errored}")

# 結果の取得（完了後）
if batch_status.processing_status == "ended":
    for result in client.batches.results(batch.id):
        print(f"ID: {result.custom_id}")
        if result.result.type == "succeeded":
            print(f"回答: {result.result.message.content[0].text[:100]}...")
```

---

## 4. Claude Code とエージェント活用

### ASCII 図解 4: Claude Code のアーキテクチャ

```
┌────────────────────────────────────────────────────────────┐
│                    Claude Code                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ ユーザー  │ →  │ Claude Code  │ →  │ Claude API   │     │
│  │ (ターミナル)│    │ CLI          │    │ (Opus/Sonnet)│     │
│  └──────────┘    └──────┬───────┘    └──────────────┘     │
│                         │                                  │
│                    利用可能なツール                          │
│                    ┌─────────────────┐                     │
│                    │ Read    - ファイル読み取り              │
│                    │ Write   - ファイル書き込み              │
│                    │ Edit    - ファイル編集                  │
│                    │ Bash    - コマンド実行                  │
│                    │ Glob    - ファイル検索                  │
│                    │ Grep    - テキスト検索                  │
│                    │ Task    - サブタスク委任                │
│                    │ MCP     - 外部ツール統合                │
│                    └─────────────────┘                     │
│                                                            │
│  ワークフロー:                                              │
│  1. ユーザーが自然言語で指示                                │
│  2. Claude がタスクを分解                                   │
│  3. 必要なツールを選択・実行                                │
│  4. 結果を確認・修正を繰り返し                              │
│  5. 最終結果を報告                                          │
│                                                            │
│  特徴:                                                     │
│  ├── エージェント的な自律実行                               │
│  ├── ファイルシステムの直接操作                              │
│  ├── Git 操作（commit, diff, etc.）                        │
│  ├── テスト実行と修正のループ                               │
│  ├── MCP によるツール拡張                                   │
│  └── /compact でメモリ管理                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### コード例 6: MCP サーバーの実装

```python
"""
MCP (Model Context Protocol) サーバーの実装例
Claude Code や他の MCP クライアントから呼び出し可能なツールを提供
"""
from mcp.server import Server
from mcp.types import Tool, TextContent
import json
import sqlite3

# MCP サーバーの初期化
server = Server("my-tools")

@server.tool()
async def query_database(query: str, database: str = "main.db") -> str:
    """SQLiteデータベースに対してSELECTクエリを実行する

    Args:
        query: 実行するSELECTクエリ
        database: データベースファイルパス

    Returns:
        クエリ結果のJSON文字列
    """
    # セキュリティチェック: SELECT のみ許可
    if not query.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "SELECT クエリのみ許可されています"})

    try:
        conn = sqlite3.connect(database)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return json.dumps({"results": rows, "count": len(rows)}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

@server.tool()
async def analyze_logs(
    log_path: str,
    level: str = "ERROR",
    last_n: int = 100,
) -> str:
    """ログファイルを分析して指定レベル以上のログを抽出する

    Args:
        log_path: ログファイルのパス
        level: フィルタするログレベル (DEBUG, INFO, WARN, ERROR)
        last_n: 最後のN行を対象にする

    Returns:
        フィルタされたログエントリのJSON
    """
    level_order = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
    min_level = level_order.get(level, 0)

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-last_n:]

        filtered = []
        for line in lines:
            for lvl, order in level_order.items():
                if lvl in line and order >= min_level:
                    filtered.append(line.strip())
                    break

        return json.dumps({
            "total_lines": len(lines),
            "filtered_count": len(filtered),
            "entries": filtered[:50],  # 最大50件
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})

# Claude Code の設定ファイル (.claude/mcp.json)
mcp_config = {
    "mcpServers": {
        "my-tools": {
            "command": "python",
            "args": ["mcp_server.py"],
            "env": {
                "DATABASE_PATH": "/path/to/db"
            }
        }
    }
}
```

---

## 5. 実践的なユースケース

### 5.1 コードレビューアシスタント

```python
import anthropic
from pathlib import Path

class CodeReviewAssistant:
    """Claude を使ったコードレビューアシスタント"""

    def __init__(self):
        self.client = anthropic.Anthropic()

    def review_file(self, file_path: str) -> str:
        """ファイルのコードレビュー"""
        code = Path(file_path).read_text()
        ext = Path(file_path).suffix

        language_map = {
            ".py": "Python",
            ".ts": "TypeScript",
            ".js": "JavaScript",
            ".go": "Go",
            ".rs": "Rust",
            ".java": "Java",
        }
        language = language_map.get(ext, "Unknown")

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=f"""あなたはシニアソフトウェアエンジニアです。
{language}のコードレビューを行います。

レビュー観点:
1. バグ・ロジックエラー
2. セキュリティ脆弱性
3. パフォーマンス問題
4. 可読性・保守性
5. テスタビリティ
6. エラーハンドリング

各問題には以下の情報を含めてください:
- 行番号（概算）
- 深刻度: Critical / High / Medium / Low
- 問題の説明
- 修正案（コード付き）""",
            messages=[{
                "role": "user",
                "content": f"以下の {language} コードをレビューしてください:\n\n```{ext[1:]}\n{code}\n```"
            }]
        )
        return response.content[0].text

    def review_diff(self, diff: str) -> str:
        """Git diff のレビュー"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="""あなたはコードレビュアーです。
Git diff を分析し、以下を確認してください:
1. 変更の目的は明確か
2. 意図しない変更がないか
3. テストの追加・更新が必要か
4. パフォーマンスへの影響
5. 後方互換性の問題""",
            messages=[{
                "role": "user",
                "content": f"以下の diff をレビューしてください:\n\n```diff\n{diff}\n```"
            }]
        )
        return response.content[0].text

# 使用例
reviewer = CodeReviewAssistant()
review = reviewer.review_file("src/api/handlers.py")
print(review)
```

### 5.2 ドキュメント分析パイプライン

```python
import anthropic
from typing import List, Dict
import json

class DocumentAnalyzer:
    """長文ドキュメントの分析パイプライン"""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def analyze(self, document: str, analysis_type: str = "comprehensive") -> Dict:
        """ドキュメントの包括的分析"""
        analyses = {
            "comprehensive": self._comprehensive_analysis,
            "summary": self._summary,
            "key_points": self._extract_key_points,
            "risks": self._risk_analysis,
            "action_items": self._extract_action_items,
        }

        analyzer = analyses.get(analysis_type, self._comprehensive_analysis)
        return analyzer(document)

    def _comprehensive_analysis(self, document: str) -> Dict:
        """包括的な分析"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system="ドキュメントアナリストとして分析してください。",
            messages=[{
                "role": "user",
                "content": f"""以下のドキュメントを包括的に分析し、JSON形式で出力してください。

<document>
{document}
</document>

出力形式:
{{
    "title": "ドキュメントのタイトル/主題",
    "summary": "200字以内の要約",
    "key_points": ["要点1", "要点2", ...],
    "risks": ["リスク1", "リスク2", ...],
    "action_items": ["アクション1", "アクション2", ...],
    "sentiment": "positive/neutral/negative",
    "confidence": 0.0-1.0,
    "topics": ["トピック1", "トピック2", ...],
    "questions": ["追加調査が必要な点1", ...]
}}"""
            }]
        )

        try:
            return json.loads(response.content[0].text)
        except json.JSONDecodeError:
            return {"raw_analysis": response.content[0].text}

    def compare_documents(self, doc1: str, doc2: str) -> Dict:
        """2つのドキュメントの比較分析"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": f"""以下の2つのドキュメントを比較分析してください。

<document_1>
{doc1}
</document_1>

<document_2>
{doc2}
</document_2>

比較観点:
1. 共通点
2. 相違点
3. 矛盾する内容
4. 一方にのみ含まれる情報
5. 推奨事項"""
            }]
        )
        return {"comparison": response.content[0].text}
```

---

## 6. コスト最適化

### 比較表 1: Claude モデルの詳細比較

| 項目 | Haiku | Sonnet | Opus |
|------|-------|--------|------|
| 最適用途 | 分類、抽出、軽量タスク | 汎用、コーディング、分析 | 複雑推論、研究、創作 |
| 入力料金 (/1M tokens) | $0.80 | $3.00 | $15.00 |
| 出力料金 (/1M tokens) | $4.00 | $15.00 | $75.00 |
| キャッシュ読み取り | $0.08 | $0.30 | $1.50 |
| バッチ入力 | $0.40 | $1.50 | $7.50 |
| バッチ出力 | $2.00 | $7.50 | $37.50 |
| コンテキスト長 | 200K | 200K | 200K |
| 速度 (tokens/sec) | 非常に速い | 速い | 中程度 |
| Extended Thinking | なし | なし | あり |
| コーディング能力 | 良好 | 優秀 | 最高 |
| 推論能力 | 良好 | 優秀 | 最高 |
| ビジョン対応 | あり | あり | あり |

### コスト最適化戦略

```python
from dataclasses import dataclass
from typing import List

@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

class ClaudeCostOptimizer:
    """Claude の利用コストを最適化する"""

    PRICING = {
        "haiku": {"input": 0.80, "output": 4.00, "cache_read": 0.08},
        "sonnet": {"input": 3.00, "output": 15.00, "cache_read": 0.30},
        "opus": {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    }

    @classmethod
    def estimate_cost(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        batch: bool = False,
    ) -> CostEstimate:
        """コスト概算"""
        pricing = cls.PRICING[model]
        batch_discount = 0.5 if batch else 1.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"] * batch_discount
        output_cost = (output_tokens / 1_000_000) * pricing["output"] * batch_discount
        cache_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read"]

        total = input_cost + output_cost + cache_cost

        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total,
        )

    @classmethod
    def compare_models(
        cls,
        input_tokens: int,
        output_tokens: int,
        requests_per_day: int = 1000,
    ) -> List[CostEstimate]:
        """モデル間のコスト比較"""
        results = []
        for model in ["haiku", "sonnet", "opus"]:
            estimate = cls.estimate_cost(model, input_tokens, output_tokens)
            daily_cost = estimate.cost_usd * requests_per_day
            monthly_cost = daily_cost * 30
            results.append({
                "model": model,
                "per_request": f"${estimate.cost_usd:.4f}",
                "daily": f"${daily_cost:.2f}",
                "monthly": f"${monthly_cost:.2f}",
            })
            print(f"{model:>8}: ${estimate.cost_usd:.4f}/req, "
                  f"${daily_cost:.2f}/day, ${monthly_cost:.2f}/month")
        return results

# コスト比較
print("=== 1000入力/500出力トークンのコスト比較 ===")
print(f"(1日1000リクエスト想定)")
ClaudeCostOptimizer.compare_models(1000, 500, 1000)

print("\n=== プロンプトキャッシュの効果 ===")
# キャッシュなし
no_cache = ClaudeCostOptimizer.estimate_cost("sonnet", 50000, 500)
# キャッシュあり（90%がキャッシュヒット）
with_cache = ClaudeCostOptimizer.estimate_cost(
    "sonnet", 5000, 500, cache_read_tokens=45000
)
print(f"キャッシュなし: ${no_cache.cost_usd:.4f}")
print(f"キャッシュあり: ${with_cache.cost_usd:.4f}")
print(f"節約率: {(1 - with_cache.cost_usd/no_cache.cost_usd)*100:.1f}%")
```

### 比較表 2: Claude vs 他モデルの特徴比較

| 特徴 | Claude 4 | GPT-4o | Gemini 2.0 Pro | DeepSeek V3 |
|------|----------|--------|----------------|-------------|
| 安全性アプローチ | Constitutional AI | RLHF | 非公開 | RLHF |
| コンテキスト長 | 200K | 128K | 1M+ | 128K |
| Extended Thinking | あり (Opus) | o1-pro 別モデル | Flash Thinking | R1 別モデル |
| 日本語能力 | 優秀 | 優秀 | 良好 | 良好 |
| コード生成 | 最高水準 | 優秀 | 良好 | 優秀 |
| 長文理解 | 非常に優秀 | 良好 | 最高 | 良好 |
| ツール使用 | 優秀 | 優秀 | 良好 | 限定的 |
| バッチ API | あり (50%引) | あり (50%引) | あり | なし |
| プロンプトキャッシュ | あり (90%引) | あり | コンテキストキャッシュ | なし |
| 価格帯 | 中程度 | 中程度 | 中程度 | 低価格 |
| オープンソース | なし | なし | なし | あり |

---

## アンチパターン

### アンチパターン 1: モデル選択の固定化

```
誤: 全タスクで Opus を使用
  → コストが不必要に高い、レイテンシも増大

正: タスクに応じてモデルを使い分ける
  Haiku: 分類、感情分析、簡単な質問応答、前処理
  Sonnet: コード生成、文書作成、一般的な分析、RAG回答
  Opus: 複雑な推論、研究レベルの分析、長文創作、Extended Thinking

  コスト例（1日1万リクエスト、平均1K入力/500出力トークン）:
  - 全て Opus: $285/日 = $8,550/月
  - 全て Sonnet: $57/日 = $1,710/月
  - 混合（Haiku 70% + Sonnet 25% + Opus 5%）: $22/日 = $660/月
```

### アンチパターン 2: プロンプトキャッシュの未活用

```
誤: 同じシステムプロンプトや共通コンテキストを毎回フル送信
  → 大量のトークン消費、レイテンシ増加

正: Claude のプロンプトキャッシュを活用
  - 1024トークン以上の共通部分を cache_control で指定
  - 2回目以降は 90% のコスト削減
  - TTL は 5 分（最後のアクセスから）

  効果の例:
  50Kトークンのドキュメントを10回質問する場合
  - キャッシュなし: $1.50 (50K × 10 × $3/1M)
  - キャッシュあり: $0.285 (50K × $3/1M + 50K × 9 × $0.30/1M)
  → 81% のコスト削減
```

### アンチパターン 3: ストリーミング未使用

```
誤: 常にブロッキングで全体の応答を待つ
  → ユーザーは長時間白い画面を見る

正: ストリーミングを活用して体感レイテンシを改善
  - Time to First Token (TTFT) が重要
  - Sonnet の TTFT は通常 200-500ms
  - フル応答の 500-3000ms を待つ必要がない
```

### アンチパターン 4: エラーハンドリングの欠如

```python
# NG: エラーハンドリングなし
response = client.messages.create(...)

# OK: 適切なリトライとフォールバック
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=lambda e: isinstance(e, (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        anthropic.APIConnectionError,
    ))
)
def safe_call(messages, model="claude-sonnet-4-20250514"):
    try:
        return client.messages.create(
            model=model,
            max_tokens=2048,
            messages=messages,
        )
    except anthropic.BadRequestError as e:
        # トークン制限超過等 → フォールバック
        print(f"BadRequest: {e}")
        return None
```

---

## FAQ

### Q1: Claude の最大の強みは何ですか？

**A:** 複数の強みがありますが、特に以下が際立っています:
1. **長文コンテキストの理解力**: 200Kトークンのコンテキストを実用的に活用でき、大規模コードベースや長文ドキュメントの分析に適している
2. **安全性**: Constitutional AI による組み込みの安全性が、プロダクション利用時のリスクを低減する
3. **コーディング能力**: 特に Claude Code との組み合わせで、エージェント的なコーディング支援が可能
4. **日本語能力**: 自然な日本語での応答品質が高い

### Q2: Claude API のレートリミットはどうなっていますか？

**A:** ティア制で管理されています:
- **Tier 1**: ~50 RPM, ~40K TPM/日
- **Tier 2**: ~1000 RPM, ~80K TPM/日
- **Tier 3**: ~2000 RPM, ~160K TPM/日
- **Tier 4**: ~4000 RPM, ~400K TPM/日

バッチ API を使えばリミットの影響を大幅に軽減できます。利用額に応じてティアが自動昇格します。

### Q3: Claude Code とは何ですか？

**A:** Anthropic 公式の CLI ツールで、ターミナルから Claude と対話しながらコーディングができます。ファイルの読み書き、Git 操作、テスト実行などをエージェント的に行え、MCP（Model Context Protocol）で外部ツールとも統合できます。VS Code の拡張機能としても利用可能です。

### Q4: Extended Thinking はいつ使うべきですか？

**A:** 以下の場面で特に効果的です:
- 数学的・論理的な問題の解決
- 複雑なコードのデバッグ
- 多角的な分析が必要な場面
- 高精度が求められるが、レイテンシは許容できる場合

budget_tokens で思考量を制御でき、コストとのバランスを取れます。

### Q5: プロンプトキャッシュの最適な使い方は？

**A:** 以下のパターンが効果的です:
- **RAG パイプライン**: 取得したドキュメントをキャッシュし、同じドキュメントに対する複数の質問を処理
- **コードレビュー**: コードベースをキャッシュし、異なる観点でレビュー
- **長いシステムプロンプト**: 詳細な指示をキャッシュ
- 最低1024トークン以上のブロックが対象、TTLは5分

### Q6: Claude でハルシネーションを防ぐにはどうすればよいですか？

**A:** 複数のアプローチを組み合わせます:
1. **RAG の活用**: 事実に基づく回答が必要な場合、外部知識を注入する
2. **引用の要求**: 「根拠を明示してください」「情報源を示してください」
3. **不確実性の表明を促す**: 「確信がない場合はそう述べてください」
4. **温度を0に設定**: 事実確認では決定的な出力を使用
5. **検証ステップの追加**: 回答後に自己検証させる

---

## まとめ

| 項目 | 要点 |
|------|------|
| Claude ファミリー | Haiku（速度）/ Sonnet（バランス）/ Opus（性能）の3段階 |
| Constitutional AI | 憲法原則に基づく自己批評でアラインメント |
| API 機能 | メッセージ、ストリーミング、ツール使用、ビジョン対応 |
| Extended Thinking | Opus で利用可能な拡張思考モード |
| コンテキスト | 200K トークンの長文コンテキスト |
| プロンプトキャッシュ | 繰り返しコンテキストのコストを 90% 削減 |
| バッチ API | 大量リクエストを 50% コスト削減で処理 |
| Claude Code | CLI ベースのAIコーディングアシスタント |
| MCP | Model Context Protocol によるツール統合 |

---

## 次に読むべきガイド

- [01-gpt.md](./01-gpt.md) -- GPT ファミリーとの比較
- [04-model-comparison.md](./04-model-comparison.md) -- 全モデルの横断比較
- [../02-applications/02-function-calling.md](../02-applications/02-function-calling.md) -- Function Calling の詳細

---

## 参考文献

1. Anthropic. (2023). "Claude's Constitution." https://www.anthropic.com/index/claudes-constitution
2. Bai, Y. et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv:2212.08073*. https://arxiv.org/abs/2212.08073
3. Anthropic. "Claude API Documentation." https://docs.anthropic.com/
4. Anthropic. "Model Context Protocol (MCP)." https://modelcontextprotocol.io/
5. Anthropic. "Claude Code Documentation." https://docs.anthropic.com/claude-code/
6. Anthropic. "Prompt Caching." https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
