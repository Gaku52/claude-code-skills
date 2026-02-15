# プロンプト駆動開発 ── 仕様からプロンプトへ、プロンプトからコードへ

> ソフトウェア開発の起点を「コードを書くこと」から「プロンプトを設計すること」へ移行させ、仕様→プロンプト→コード→検証の新しい開発サイクルを体系的に習得する。

---

## この章で学ぶこと

1. **プロンプト駆動開発（PDD）のプロセス** ── 仕様定義からコード生成までの一貫したワークフローを理解する
2. **効果的なプロンプト設計パターン** ── 再現性と品質を高めるプロンプトテンプレートを習得する
3. **プロンプトの反復改善手法** ── AIの出力品質を段階的に向上させるテクニックを身につける
4. **実践的なPDDワークフロー** ── 実プロジェクトでPDDを導入・運用する方法を習得する
5. **チームでのPDD標準化** ── プロンプトの品質管理とナレッジ共有の体制を構築する

---

## 1. プロンプト駆動開発（PDD）とは

### 1.1 開発パラダイムの変遷

```
手続型開発          オブジェクト指向      テスト駆動開発(TDD)   プロンプト駆動開発(PDD)
(1960s-)           (1990s-)            (2000s-)            (2024s-)

コード → 動作      設計 → コード        テスト → コード      プロンプト → コード

┌──────┐          ┌──────┐           ┌──────┐            ┌──────────┐
│手続き │          │クラス │           │Red   │            │ 仕様定義  │
│を書く │          │設計  │           │Green │            │    ↓     │
│  ↓   │          │  ↓   │           │Refac │            │プロンプト │
│デバッグ│          │実装  │           │ tor  │            │    ↓     │
└──────┘          └──────┘           └──────┘            │AI生成    │
                                                         │    ↓     │
                                                         │検証・改善 │
                                                         └──────────┘
```

### 1.2 PDDのワークフロー

```
┌──────────────────────────────────────────────────────┐
│              プロンプト駆動開発サイクル                  │
│                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │ 1.仕様  │───►│2.プロン │───►│ 3.生成  │         │
│   │  定義   │    │ プト設計│    │ (AI)    │         │
│   └─────────┘    └─────────┘    └────┬────┘         │
│        ▲                              │              │
│        │                              ▼              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐         │
│   │ 6.統合  │◄───│5.改善   │◄───│ 4.検証  │         │
│   │         │    │(反復)   │    │ (人間)  │         │
│   └─────────┘    └─────────┘    └─────────┘         │
│                                                      │
│   各ステップの所要時間:                                │
│   仕様(10min) → プロンプト(5min) → 生成(1min)        │
│   → 検証(5min) → 改善(3min) → 統合(5min)            │
│   合計: 約30分 (従来: 2-4時間)                        │
└──────────────────────────────────────────────────────┘
```

### 1.3 PDDの基本原則

プロンプト駆動開発を成功させるための5つの基本原則を理解する。

```
原則1: Specification First（仕様優先）
  - コードを書く前に、必ず仕様を明文化する
  - 仕様がプロンプトの品質を決定し、プロンプトがコード品質を決定する
  - 曖昧な仕様 → 曖昧なプロンプト → 曖昧なコード（ゴミイン・ゴミアウト）

原則2: Incremental Refinement（段階的改善）
  - 一度で完璧を目指さず、反復的に改善する
  - 各ラウンドで1つの品質軸（機能正確性、エラー処理、パフォーマンスなど）に集中
  - 3ラウンド以内で実用品質に到達することを目標とする

原則3: Context is King（コンテキストが王）
  - AIは提供された情報の範囲内でしか最適解を出せない
  - 既存コード、規約、アーキテクチャ決定を必ずコンテキストとして与える
  - コンテキストの質がプロンプトの効果を指数関数的に向上させる

原則4: Human in the Loop（人間の関与）
  - AIの出力を無条件に受け入れない
  - ドメイン知識、セキュリティ、パフォーマンスの観点で必ず人間がレビュー
  - 最終的な品質責任は人間にある

原則5: Prompt as Asset（プロンプトは資産）
  - 優れたプロンプトはコードと同等の資産価値を持つ
  - バージョン管理し、レビューし、再利用可能にする
  - チームのプロンプトライブラリは組織の競争力になる
```

### 1.4 PDDと従来手法の比較

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│    観点       │   従来開発    │     TDD     │     PDD      │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ 開発の起点    │ コード       │ テスト       │ プロンプト    │
│ 設計の表現    │ UMLなど      │ テストケース │ 構造化文書    │
│ 反復単位      │ 実装→デバッグ │ Red→Green   │ 生成→検証    │
│ 品質の保証    │ コードレビュー│ テスト通過   │ プロンプト品質│
│ スケール      │ 線形的       │ 線形的       │ 指数的       │
│ 学習曲線      │ 高い         │ 中程度       │ 新しいスキル  │
│ 再利用性      │ ライブラリ    │ テスト       │ テンプレート  │
│ ドキュメント  │ 別途作成     │ テスト=文書  │ プロンプト=文書│
└──────────────┴──────────────┴──────────────┴──────────────┘
```

### 1.5 PDDが適するケースと適さないケース

```python
# PDDが特に効果を発揮するケース
pdd_suitable_cases = {
    "CRUD操作": "定型的なAPI・画面の大量生成",
    "ボイラープレート": "設定ファイル、初期コード、スキャフォールド",
    "テストコード": "網羅的なテストケースの生成",
    "データ変換": "ETL、マイグレーション、フォーマット変換",
    "ドキュメント": "APIドキュメント、JSDoc/docstring",
    "プロトタイピング": "概念実証、MVP開発",
    "標準パターン": "認証、RBAC、監査ログなど",
}

# PDDの効果が限定的なケース
pdd_limited_cases = {
    "高度なアルゴリズム": "競プロ的な最適化、数学的証明が必要なケース",
    "ドメイン固有知識": "業界特有の複雑なビジネスルール",
    "パフォーマンス最適化": "μsec単位のチューニング",
    "セキュリティクリティカル": "暗号実装、認証基盤の核心部分",
    "レガシー統合": "文書化されていない古いシステムとの統合",
    "ハードウェア連携": "ドライバ、組み込み系の低レイヤー",
}

# 判定フレームワーク
def should_use_pdd(task: dict) -> str:
    """タスクがPDDに適しているかを判定する"""
    score = 0
    if task.get("is_well_defined"):        score += 2  # 仕様が明確
    if task.get("has_standard_pattern"):    score += 2  # 標準パターンが存在
    if task.get("is_repetitive"):          score += 1  # 繰り返し作業
    if task.get("needs_domain_expertise"): score -= 2  # 専門知識が必要
    if task.get("is_security_critical"):   score -= 2  # セキュリティ重要
    if task.get("has_existing_examples"):  score += 1  # 参考例がある

    if score >= 3:
        return "PDD推奨: プロンプトで効率的に生成可能"
    elif score >= 1:
        return "PDD部分適用: 骨格をPDDで生成し、詳細を手動で調整"
    else:
        return "従来開発推奨: 手動実装の方が安全かつ効率的"
```

---

## 2. プロンプト設計パターン

### コード例1: 基本テンプレート（CRISP形式）

```markdown
# CRISP プロンプトテンプレート

## Context（文脈）
- プロジェクト: ECサイトの注文管理システム
- 技術スタック: Python 3.12, FastAPI, SQLAlchemy, PostgreSQL
- 既存コードの規約: PEP 8準拠、型ヒント必須、docstring必須

## Role（AIの役割）
あなたはシニアバックエンドエンジニアです。
クリーンアーキテクチャとDDDに精通しています。

## Intent（意図・目的）
注文のキャンセル機能を実装したい。
キャンセル可能な条件（発送前のみ）を厳密にチェックし、
在庫の復元とユーザーへの通知も行う必要がある。

## Specifics（具体的要件）
- エンドポイント: POST /api/v1/orders/{order_id}/cancel
- キャンセル条件: ステータスが "pending" または "confirmed" のみ
- 副作用: 在庫数の復元、キャンセルメールの送信
- エラー: 既にキャンセル済み(409)、発送済み(422)

## Pattern（出力形式）
- ドメイン層、ユースケース層、プレゼンテーション層に分離
- 各層のファイルを別々に出力
- テストコードも含める
```

### コード例2: 段階的詳細化パターン

```python
# === ステップ1: 大枠の設計を依頼 ===
prompt_step1 = """
注文キャンセル機能の設計を以下の観点で提案してください:
1. ドメインモデルの変更点
2. ユースケースのフロー
3. 必要なインターフェース
コードは不要。箇条書きで構造だけ示してください。
"""

# === ステップ2: 設計をレビューしてコード生成 ===
prompt_step2 = """
上記の設計に同意します。以下の修正を加えてコードを生成してください:
- OrderCancelledイベントを追加
- 冪等性を保証する（同じリクエストを2回送っても安全）
- CancelReasonをenumで定義
"""

# === ステップ3: テストとエッジケース ===
prompt_step3 = """
生成されたコードに対して以下のテストを作成してください:
1. 正常系: pending状態の注文をキャンセル
2. 正常系: confirmed状態の注文をキャンセル
3. 異常系: shipped状態の注文をキャンセル試行
4. 異常系: 既にキャンセル済みの注文を再キャンセル
5. 境界: 在庫が0の商品を含む注文のキャンセル
6. 並行: 同時に2つのキャンセルリクエスト
"""
```

### コード例3: コンテキスト注入パターン

```markdown
# 既存コードをコンテキストとして提供し、一貫性を保つ

## 既存のドメインモデル（参考）
```python
# domain/order.py の既存コード
class Order:
    def __init__(self, order_id: OrderId, items: list[OrderItem]):
        self._id = order_id
        self._items = items
        self._status = OrderStatus.PENDING
        self._events: list[DomainEvent] = []

    def confirm(self) -> None:
        if self._status != OrderStatus.PENDING:
            raise OrderAlreadyConfirmedException(self._id)
        self._status = OrderStatus.CONFIRMED
        self._events.append(OrderConfirmed(self._id))
```

## 依頼
上記の既存パターン（イベント発行、例外クラス、命名規則）に
完全に一致する形で `cancel` メソッドを追加してください。
既存コードとの差分のみを出力してください。
```

### コード例4: 制約指定パターン

```python
# AIの出力を制約で制御する

PROMPT_WITH_CONSTRAINTS = """
以下の制約に従ってReactコンポーネントを作成してください。

## 機能要件
ユーザー一覧テーブル（検索・ソート・ページネーション付き）

## 制約条件（必ず守ること）
- DO: TypeScript strict mode で型安全にする
- DO: TanStack Table v8 を使う
- DO: サーバーサイドページネーション対応
- DO: ローディング・エラー・空状態の3つのUIステートを実装
- DO: アクセシビリティ（aria属性）を含める

- DON'T: any型を使わない
- DON'T: useEffectの中でデータフェッチしない（TanStack Queryを使う）
- DON'T: CSSをインラインで書かない（Tailwind CSSを使う）
- DON'T: 1コンポーネント200行を超えない
"""
```

### コード例5: プロンプトのバージョン管理

```yaml
# .prompts/order-cancel.yaml
# プロンプトをコードと同様にバージョン管理する

metadata:
  id: order-cancel-v3
  author: "team-backend"
  created: "2025-03-15"
  model: "claude-sonnet-4-20250514"
  quality_score: 0.92  # 過去の出力品質スコア

context:
  project: "ec-platform"
  module: "order-management"
  conventions: |
    - Clean Architecture (domain / usecase / infra / presentation)
    - Domain events for side effects
    - Result type for error handling (no exceptions in domain layer)

prompt: |
  注文キャンセルのユースケースを実装してください。

  入力: order_id (UUID), reason (CancelReason enum), cancelled_by (UserId)
  出力: Result[CancelledOrder, CancelError]

  ビジネスルール:
  1. キャンセル可能なステータス: PENDING, CONFIRMED
  2. 発送後はキャンセル不可 → ReturnRequestへ誘導
  3. キャンセル時に在庫を復元
  4. OrderCancelledイベントを発行

  制約:
  - 冪等性を保証すること
  - 楽観ロックでの並行制御

validation:
  - "CancelError型が定義されていること"
  - "Result型で返していること"
  - "DomainEventが発行されていること"
  - "テストが5件以上含まれていること"
```

### コード例6: マルチモーダルプロンプトパターン

```markdown
# 画像やダイアグラムを含むプロンプト

## UIモックアップからのコード生成

### プロンプト構造
1. スクリーンショットまたはFigmaエクスポートを添付
2. 以下のテキストプロンプトを付与

## テキストプロンプト

添付のUI設計をReactコンポーネントとして実装してください。

### 技術仕様
- フレームワーク: Next.js 14 App Router
- スタイリング: Tailwind CSS + shadcn/ui
- 状態管理: React Server Components + useActionState

### レイアウト解析指示
1. 添付画像の各セクションを独立したコンポーネントに分割
2. レスポンシブデザイン（モバイル→デスクトップ）
3. カラーコードは画像から正確に抽出
4. フォントサイズは相対的な比率を維持

### コンポーネント構造の期待形式
```
src/
  components/
    layout/
      Header.tsx
      Sidebar.tsx
      MainContent.tsx
    features/
      UserProfile/
        UserProfileCard.tsx
        UserProfileStats.tsx
        index.tsx
```

### 出力形式
- 各ファイルを個別に出力
- コンポーネントごとにStorybookのストーリーも生成
- レスポンシブのブレークポイント: sm(640px), md(768px), lg(1024px)
```

### コード例7: ドメインエキスパートプロンプトパターン

```python
# 特定ドメインの知識をAIに注入して精度を高める

DOMAIN_EXPERT_PROMPT = """
## ドメイン知識（金融取引システム）

### 用語定義
- 約定（やくじょう）: 売買注文が成立すること
- 受渡日（うけわたしび）: 約定日の2営業日後（T+2）
- 洗替（あらいがえ）: 含み損益を日次で再計算すること
- ネッティング: 同一通貨の債権債務を相殺すること

### ビジネスルール（厳守）
1. 取引金額は1億円未満の場合、自動承認
2. 1億円以上10億円未満は部長承認が必要
3. 10億円以上は役員承認が必要
4. 同一顧客への1日の合計取引額が50億円を超える場合、アラート

### 規制要件
- 金商法に基づく取引記録の7年間保存
- 反社会的勢力チェック（毎取引時）
- マネーロンダリング検知（パターンマッチング）

## 依頼
上記のドメイン知識に基づき、取引承認ワークフローのドメインモデルを
実装してください。TypeScript + Prisma で記述し、承認ステート管理に
State パターンを使用してください。
"""
```

---

## 3. プロンプト品質の評価基準

### 3.1 CLEAR基準

| 基準 | 説明 | チェック項目 |
|------|------|-------------|
| **C**oncrete（具体的） | 曖昧さがない | 入出力の型、エラーケースが明記されているか |
| **L**ayered（段階的） | 複雑さを分解 | 1プロンプトの責務が適切に限定されているか |
| **E**xample-rich（例が豊富） | 期待する形式を示す | 入出力例やコードスニペットが含まれているか |
| **A**ctionable（実行可能） | 即座にコードに変換可能 | AIが追加質問なしに実装できるか |
| **R**eproducible（再現可能） | 誰が実行しても同じ結果 | モデル、バージョン、コンテキストが固定されているか |

### 3.2 プロンプト品質 vs コード品質の相関

| プロンプト品質 | コード品質の傾向 | 修正回数 | 総所要時間 |
|--------------|----------------|---------|-----------|
| 曖昧（1行） | 動くが設計が悪い | 5-10回 | 従来と同等 |
| 基本的（要件列挙） | 機能は正しい | 2-3回 | 従来の50% |
| 構造化（CRISP） | 設計も品質も高い | 0-1回 | 従来の25% |
| 完全（例+制約付き） | プロダクション品質 | 0回 | 従来の15% |

### 3.3 プロンプト品質評価スコアカード

```python
from dataclasses import dataclass
from enum import Enum

class QualityLevel(Enum):
    POOR = 1
    BASIC = 2
    GOOD = 3
    EXCELLENT = 4

@dataclass
class PromptScoreCard:
    """プロンプトの品質を定量的に評価するスコアカード"""

    # CLEAR基準の各スコア（1-4）
    concrete: QualityLevel      # 具体性
    layered: QualityLevel       # 段階性
    example_rich: QualityLevel  # 例の豊富さ
    actionable: QualityLevel    # 実行可能性
    reproducible: QualityLevel  # 再現可能性

    @property
    def total_score(self) -> int:
        """総合スコア（5-20）"""
        return sum([
            self.concrete.value,
            self.layered.value,
            self.example_rich.value,
            self.actionable.value,
            self.reproducible.value,
        ])

    @property
    def quality_grade(self) -> str:
        """品質グレード"""
        score = self.total_score
        if score >= 18:
            return "A: プロダクション品質のコードが期待できる"
        elif score >= 14:
            return "B: 軽微な修正で使用可能なコードが期待できる"
        elif score >= 10:
            return "C: 骨格は正しいが大幅な修正が必要"
        else:
            return "D: プロンプトの再設計が必要"

    def improvement_suggestions(self) -> list[str]:
        """改善提案を生成"""
        suggestions = []
        if self.concrete.value <= 2:
            suggestions.append(
                "具体性向上: 入出力の型、エラーケース、境界条件を明記する"
            )
        if self.layered.value <= 2:
            suggestions.append(
                "段階性向上: 1つのプロンプトで扱う範囲を絞り、複数ステップに分割する"
            )
        if self.example_rich.value <= 2:
            suggestions.append(
                "例の追加: 入出力の具体例、期待するコードスタイルのサンプルを含める"
            )
        if self.actionable.value <= 2:
            suggestions.append(
                "実行可能性向上: 技術スタック、ライブラリバージョン、環境情報を追加する"
            )
        if self.reproducible.value <= 2:
            suggestions.append(
                "再現性向上: モデル名、Temperature設定、使用ツールを固定する"
            )
        return suggestions


# 使用例
score = PromptScoreCard(
    concrete=QualityLevel.EXCELLENT,
    layered=QualityLevel.GOOD,
    example_rich=QualityLevel.GOOD,
    actionable=QualityLevel.EXCELLENT,
    reproducible=QualityLevel.BASIC,
)
print(f"総合スコア: {score.total_score}/20")
print(f"品質グレード: {score.quality_grade}")
for suggestion in score.improvement_suggestions():
    print(f"  改善: {suggestion}")
```

### 3.4 品質メトリクスの自動計測

```python
import re
from typing import NamedTuple

class PromptMetrics(NamedTuple):
    """プロンプトの定量的メトリクス"""
    word_count: int          # 単語数
    has_context: bool        # コンテキスト情報の有無
    has_constraints: bool    # 制約条件の有無
    has_examples: bool       # 例の有無
    has_error_cases: bool    # エラーケースの記述
    specificity_score: float # 具体性スコア（0-1）
    estimated_quality: str   # 推定品質レベル

def analyze_prompt(prompt: str) -> PromptMetrics:
    """プロンプトを分析してメトリクスを算出する"""
    words = prompt.split()
    word_count = len(words)

    # コンテキスト情報の検出
    context_patterns = [
        r"プロジェクト|技術スタック|既存|アーキテクチャ|規約",
        r"context|project|stack|architecture|convention",
    ]
    has_context = any(
        re.search(p, prompt, re.IGNORECASE) for p in context_patterns
    )

    # 制約条件の検出
    constraint_patterns = [
        r"制約|DON'?T|禁止|必ず|MUST|SHOULD NOT",
        r"constraint|restriction|requirement",
    ]
    has_constraints = any(
        re.search(p, prompt, re.IGNORECASE) for p in constraint_patterns
    )

    # 例の検出
    has_examples = bool(re.search(r"例[：:]|例えば|```|example|e\.g\.", prompt))

    # エラーケースの検出
    has_error_cases = bool(
        re.search(r"エラー|異常|例外|失敗|error|exception|failure", prompt, re.IGNORECASE)
    )

    # 具体性スコアの計算
    specificity_indicators = [
        has_context, has_constraints, has_examples, has_error_cases,
        word_count > 50, word_count > 100, word_count > 200,
        bool(re.search(r"\d+", prompt)),  # 数値を含む
        bool(re.search(r"(int|str|bool|float|list|dict|string|number)", prompt)),
    ]
    specificity_score = sum(specificity_indicators) / len(specificity_indicators)

    # 推定品質レベル
    if specificity_score >= 0.8:
        estimated_quality = "EXCELLENT"
    elif specificity_score >= 0.6:
        estimated_quality = "GOOD"
    elif specificity_score >= 0.4:
        estimated_quality = "BASIC"
    else:
        estimated_quality = "POOR"

    return PromptMetrics(
        word_count=word_count,
        has_context=has_context,
        has_constraints=has_constraints,
        has_examples=has_examples,
        has_error_cases=has_error_cases,
        specificity_score=specificity_score,
        estimated_quality=estimated_quality,
    )
```

---

## 4. 反復改善のテクニック

### 4.1 フィードバックループ

```
┌────────────────────────────────────────────────┐
│          プロンプト反復改善プロセス               │
│                                                │
│  Round 1: 初回生成                              │
│  ┌──────────┐    ┌──────────┐                  │
│  │プロンプト │───►│  出力    │──► 評価: 60点    │
│  │ (v1)     │    │ (Draft1) │                  │
│  └──────────┘    └──────────┘                  │
│       │                                        │
│       │ 修正: "エラーハンドリングが不足"          │
│       ▼                                        │
│  Round 2: 改善                                  │
│  ┌──────────┐    ┌──────────┐                  │
│  │プロンプト │───►│  出力    │──► 評価: 80点    │
│  │ (v2)     │    │ (Draft2) │                  │
│  └──────────┘    └──────────┘                  │
│       │                                        │
│       │ 修正: "テストのエッジケース追加"          │
│       ▼                                        │
│  Round 3: 完成                                  │
│  ┌──────────┐    ┌──────────┐                  │
│  │プロンプト │───►│  出力    │──► 評価: 95点    │
│  │ (v3)     │    │ (Final)  │                  │
│  └──────────┘    └──────────┘                  │
└────────────────────────────────────────────────┘
```

### 4.2 反復改善の具体的テクニック

```python
# テクニック1: 差分指示法
# 前の出力を参照し、変更点のみを指示する

DIFF_INSTRUCTION_PROMPT = """
前回生成したコードに以下の修正を加えてください。
変更部分のみ出力してください（変更のないファイルは省略）。

## 修正指示
1. OrderService.cancel() にリトライロジックを追加
   - 最大3回、指数バックオフ（1s, 2s, 4s）
   - OptimisticLockException の場合のみリトライ

2. CancelledOrder のレスポンスに以下を追加
   - refund_amount: 返金額（税込）
   - refund_estimated_date: 返金予定日（3営業日後）

3. テストに以下のケースを追加
   - リトライ成功のケース
   - リトライ上限超過のケース
"""

# テクニック2: 観点切り替え法
# 異なる観点からレビューと改善を依頼する

PERSPECTIVE_SWITCH_PROMPTS = {
    "security": """
    生成されたコードをセキュリティの観点でレビューしてください:
    - SQLインジェクション、XSSの可能性
    - 認可チェックの漏れ
    - 情報漏洩リスク（ログ出力、エラーメッセージ）
    - レートリミットの必要性
    修正が必要な箇所を具体的に指摘してください。
    """,

    "performance": """
    生成されたコードをパフォーマンスの観点でレビューしてください:
    - N+1クエリの有無
    - 不要なメモリアロケーション
    - キャッシュ戦略の妥当性
    - インデックスの必要性
    改善案を具体的なコード修正として示してください。
    """,

    "maintainability": """
    生成されたコードを保守性の観点でレビューしてください:
    - SOLID原則への準拠
    - 関数の責務の明確さ
    - テスタビリティ
    - 命名の適切さ
    リファクタリング案を具体的に示してください。
    """,
}

# テクニック3: 比較生成法
# 複数のアプローチを生成させて比較する

COMPARISON_PROMPT = """
以下の機能を3つの異なるアプローチで実装してください。

## 機能: 注文キャンセル処理

### アプローチA: ドメインイベント方式
- 利点・欠点を明記

### アプローチB: サーガパターン方式
- 利点・欠点を明記

### アプローチC: ステートマシン方式
- 利点・欠点を明記

最後に、推奨アプローチとその理由を述べてください。
"""
```

### 4.3 プロンプトチェーニング

```python
# 複数のプロンプトを連鎖させて複雑な成果物を構築する

class PromptChain:
    """プロンプトを連鎖的に実行して段階的に成果物を構築する"""

    def __init__(self, ai_client):
        self.client = ai_client
        self.context = {}  # 各ステップの出力を蓄積

    def execute_chain(self, feature_spec: dict) -> dict:
        """プロンプトチェーンを実行"""

        # Step 1: アーキテクチャ設計
        arch_prompt = f"""
        以下の機能のアーキテクチャを設計してください。

        機能: {feature_spec['name']}
        要件: {feature_spec['requirements']}

        以下の形式で出力:
        1. コンポーネント図（テキスト形式）
        2. データフロー
        3. API設計（エンドポイント一覧）
        4. データモデル（ER図テキスト形式）
        """
        self.context['architecture'] = self.client.generate(arch_prompt)

        # Step 2: ドメインモデル実装
        domain_prompt = f"""
        以下のアーキテクチャ設計に基づき、ドメインモデルを実装してください。

        ## アーキテクチャ設計
        {self.context['architecture']}

        ## 制約
        - Python 3.12 + dataclasses
        - 値オブジェクトは frozen=True
        - ドメインイベントを含める
        - ファクトリメソッドを使用
        """
        self.context['domain'] = self.client.generate(domain_prompt)

        # Step 3: ユースケース実装
        usecase_prompt = f"""
        以下のドメインモデルを使用してユースケースを実装してください。

        ## ドメインモデル
        {self.context['domain']}

        ## 制約
        - Repository インターフェースを定義（実装は後のステップ）
        - トランザクション境界を明確にする
        - Result型でエラーを表現
        """
        self.context['usecase'] = self.client.generate(usecase_prompt)

        # Step 4: インフラ層実装
        infra_prompt = f"""
        以下のRepository インターフェースの実装を作成してください。

        ## ユースケース（Repository インターフェース定義を含む）
        {self.context['usecase']}

        ## 制約
        - SQLAlchemy 2.0 + asyncio
        - PostgreSQL用
        - マイグレーション（Alembic）も含める
        """
        self.context['infra'] = self.client.generate(infra_prompt)

        # Step 5: テスト生成
        test_prompt = f"""
        以下の全レイヤーのテストを作成してください。

        ## ドメインモデル
        {self.context['domain']}

        ## ユースケース
        {self.context['usecase']}

        ## 制約
        - pytest + pytest-asyncio
        - ドメイン層: 単体テスト（モック不要）
        - ユースケース層: Repository をモック
        - インフラ層: testcontainers でPostgreSQLを起動
        - カバレッジ90%以上を目標
        """
        self.context['tests'] = self.client.generate(test_prompt)

        return self.context
```

---

## 5. 実践的なPDDワークフロー

### 5.1 フルスタック機能開発のPDDフロー

```python
# 実際のプロジェクトでPDDを適用する完全なワークフロー例

class PDDWorkflow:
    """
    プロンプト駆動開発の実践ワークフロー

    典型的な機能開発（ユーザー検索機能）での適用例
    """

    # Phase 1: 仕様定義（人間が行う）
    SPEC = """
    ## ユーザー検索機能の仕様

    ### 目的
    管理画面からユーザーを高速に検索し、詳細情報を表示する

    ### 検索条件
    - フリーテキスト（名前、メール、電話番号に部分一致）
    - ステータスフィルター（active, suspended, deleted）
    - 登録期間（from - to）
    - ソート（名前、登録日、最終ログイン日）

    ### 非機能要件
    - レスポンスタイム: 200ms以下（100万件のデータ）
    - ページネーション: オフセット方式、1ページ20件
    - アクセス制御: ADMIN, SUPPORT ロールのみアクセス可能
    """

    # Phase 2: プロンプト設計（人間がテンプレートを活用）
    PROMPTS = {
        "api_design": """
        ## Context
        - FastAPI + SQLAlchemy + PostgreSQL
        - 既存のユーザーテーブル: users (id, name, email, phone,
          status, created_at, last_login_at)
        - 認証: JWT + RBAC (roles: ADMIN, SUPPORT, USER)

        ## 依頼
        ユーザー検索APIのエンドポイントを設計してください。

        ### 検索仕様
        - GET /api/v1/admin/users/search
        - クエリパラメータ: q(フリーテキスト), status,
          from_date, to_date, sort_by, sort_order, page, per_page
        - レスポンス: ページネーション付きのユーザー一覧

        ### 出力形式
        - Pydanticのリクエスト/レスポンスモデル
        - FastAPIのルーター
        - 検索サービス
        - SQLAlchemyのクエリビルダー
        - テストコード（pytest）
        """,

        "frontend": """
        ## Context
        - Next.js 14 App Router + TypeScript
        - UI: shadcn/ui + Tailwind CSS
        - 状態管理: TanStack Query v5
        - 先ほど設計したAPI: GET /api/v1/admin/users/search

        ## 依頼
        管理画面のユーザー検索ページを実装してください。

        ### UI仕様
        - 検索フォーム（デバウンス付きテキスト入力）
        - フィルターパネル（ステータス、期間）
        - 結果テーブル（ソート対応、ページネーション）
        - ローディング、エラー、空結果の各状態

        ### 出力形式
        - page.tsx（Server Component）
        - SearchForm.tsx（Client Component）
        - UserTable.tsx（Client Component）
        - useUserSearch.ts（カスタムフック）
        - 型定義ファイル
        """,
    }

    # Phase 3: 検証チェックリスト（人間が確認）
    VERIFICATION = """
    ## 検証チェックリスト

    ### 機能検証
    - [ ] フリーテキスト検索が名前・メール・電話番号で動作する
    - [ ] ステータスフィルターが正しく適用される
    - [ ] 日付範囲フィルターが正しく動作する
    - [ ] ソートが昇順・降順で動作する
    - [ ] ページネーションが正しく動作する

    ### セキュリティ検証
    - [ ] ADMIN, SUPPORT以外のロールはアクセスできない
    - [ ] SQLインジェクション対策がされている
    - [ ] フリーテキストにXSSペイロードを入れても安全

    ### パフォーマンス検証
    - [ ] 100万件のテストデータで200ms以下
    - [ ] 適切なインデックスが定義されている
    - [ ] N+1クエリが発生していない

    ### UX検証
    - [ ] 検索デバウンスが300ms程度
    - [ ] ローディング中にスケルトンが表示される
    - [ ] エラー時にリトライボタンが表示される
    """
```

### 5.2 レガシーコードリファクタリングのPDD

```python
# レガシーコードを段階的にリファクタリングするPDDアプローチ

LEGACY_REFACTORING_PROMPTS = {
    "step1_analysis": """
    ## Context
    以下のレガシーコードを分析してください。

    ```python
    # legacy_order_processor.py（800行の神クラス）
    class OrderProcessor:
        def __init__(self, db):
            self.db = db

        def process(self, order_data):
            # バリデーション、在庫チェック、決済、通知、ログ...
            # 800行のメソッド
            ...
    ```

    ## 依頼
    1. 責務を分析して一覧化してください
    2. 依存関係を図示してください
    3. リファクタリングの優先順位を提案してください
    4. 段階的な移行計画（5ステップ以内）を作成してください
    """,

    "step2_interface": """
    ## Context
    前回の分析結果に基づき、新しいインターフェースを設計してください。

    ## 制約
    - 既存のOrderProcessorを一度に全て書き換えない
    - Strangler Fig パターンで段階的に移行
    - 各ステップでテストが通る状態を維持
    - 新しいコードはClean Architecture準拠
    """,

    "step3_migration": """
    ## Context
    設計済みのインターフェースに基づき、最初の移行ステップを実装してください。

    ## 移行対象
    バリデーションロジックをOrderValidatorクラスに抽出

    ## 制約
    - OrderProcessorはOrderValidatorに委譲する形に変更
    - 外部インターフェースは変更しない
    - 移行前後で全テストがパスすること
    - 移行のロールバック手順も含める
    """,
}
```

### 5.3 API設計のPDD

```python
# OpenAPI仕様をプロンプトで生成するワークフロー

API_DESIGN_PDD = """
## Context
- マイクロサービスアーキテクチャ
- API Gateway: Kong
- 認証: OAuth 2.0 + JWT
- バージョニング: URL path方式 (/api/v1/)
- ドキュメント: OpenAPI 3.1

## 依頼
以下のリソースに対するREST APIを設計してください。

### リソース: プロジェクト管理
- Project: プロジェクトの作成・更新・削除・一覧・詳細
- Task: プロジェクト内のタスク管理
- Member: プロジェクトメンバーの管理

### 設計要件
1. HATEOAS準拠のレスポンス形式
2. Cursor-based pagination（大量データ対応）
3. Partial response（fieldsパラメータ）
4. 一括操作API（batch endpoint）
5. Webhook通知の登録API

### 出力形式
1. OpenAPI 3.1 YAML形式
2. 各エンドポイントにリクエスト/レスポンスの例を含める
3. エラーレスポンスのスキーマ（RFC 7807準拠）
4. 認可ルール（どのロールがどのエンドポイントにアクセスできるか）
"""
```

---

## 6. プロンプトテンプレートライブラリ

### 6.1 汎用テンプレート集

```yaml
# .prompts/templates/crud-api.yaml
name: "CRUD API生成テンプレート"
description: "標準的なCRUD APIを生成するための汎用テンプレート"
version: "2.0"

parameters:
  - name: entity_name
    description: "エンティティ名（例: User, Product）"
    required: true
  - name: fields
    description: "フィールド定義（名前: 型の配列）"
    required: true
  - name: tech_stack
    description: "技術スタック"
    default: "Python + FastAPI + SQLAlchemy"
  - name: auth_required
    description: "認証要否"
    default: true
  - name: soft_delete
    description: "論理削除の使用"
    default: true

template: |
  ## Context
  技術スタック: {{tech_stack}}
  エンティティ: {{entity_name}}

  ## フィールド定義
  {{#each fields}}
  - {{this.name}}: {{this.type}} {{#if this.required}}(必須){{/if}}
  {{/each}}

  ## 依頼
  {{entity_name}}のCRUD APIを以下の仕様で実装してください。

  ### エンドポイント
  - POST   /api/v1/{{entity_name | lower}}s     - 作成
  - GET    /api/v1/{{entity_name | lower}}s     - 一覧（ページネーション付き）
  - GET    /api/v1/{{entity_name | lower}}s/:id - 詳細
  - PUT    /api/v1/{{entity_name | lower}}s/:id - 更新
  - DELETE /api/v1/{{entity_name | lower}}s/:id - 削除{{#if soft_delete}}（論理削除）{{/if}}

  ### 共通仕様
  {{#if auth_required}}- JWT認証必須{{/if}}
  - バリデーション（Pydantic）
  - エラーハンドリング（RFC 7807形式）
  - ログ出力（構造化ログ）
  - テストコード（pytest、カバレッジ90%以上）
```

### 6.2 コードレビュープロンプトテンプレート

```yaml
# .prompts/templates/code-review.yaml
name: "AIコードレビューテンプレート"
version: "1.5"

template: |
  以下のコードを多角的にレビューしてください。

  ## レビュー対象コード
  ```
  {{code}}
  ```

  ## レビュー観点（各観点で1-5のスコアをつけてください）

  ### 1. 正確性（Correctness）
  - ビジネスロジックに誤りがないか
  - エッジケースの処理漏れがないか
  - 型の不整合がないか

  ### 2. セキュリティ（Security）
  - インジェクション系の脆弱性
  - 認証・認可の漏れ
  - 機密情報の露出

  ### 3. パフォーマンス（Performance）
  - 不要な計算やIO
  - メモリリーク
  - 最適化の余地

  ### 4. 可読性（Readability）
  - 命名の適切さ
  - コメントの過不足
  - 関数の長さと複雑度

  ### 5. テスタビリティ（Testability）
  - 依存性の注入
  - モックの容易さ
  - 副作用の分離

  ## 出力形式
  各観点のスコアと具体的な改善提案をコード付きで示してください。
  致命的な問題は🔴、改善推奨は🟡、軽微は🟢でマークしてください。
```

### 6.3 テスト生成テンプレート

```yaml
# .prompts/templates/test-generation.yaml
name: "包括的テスト生成テンプレート"
version: "2.0"

template: |
  以下のコードに対する包括的なテストスイートを作成してください。

  ## 対象コード
  ```
  {{code}}
  ```

  ## テスト要件

  ### カテゴリ別テストケース
  1. **正常系（Happy Path）**
     - 典型的な入力での正常動作
     - 各分岐パスの通過確認

  2. **境界値（Boundary）**
     - 最小値、最大値
     - 空リスト、空文字列
     - ゼロ、負数

  3. **異常系（Error Cases）**
     - 不正な入力
     - null/undefined
     - 型不一致
     - リソース不足（メモリ、ディスク）

  4. **並行性（Concurrency）**
     - 同時実行
     - レースコンディション
     - デッドロック

  5. **統合（Integration）**
     - 外部サービスとの連携
     - DB操作
     - ファイルI/O

  ## 制約
  - テストフレームワーク: {{test_framework}}
  - カバレッジ目標: {{coverage_target}}%
  - モックライブラリ: {{mock_library}}
  - 各テストは独立して実行可能であること
  - Given-When-Thenの形式でテスト名を記述
```

---

## 7. 高度なプロンプトテクニック

### 7.1 メタプロンプティング

```python
# プロンプトを生成するプロンプト（メタプロンプト）

META_PROMPT = """
あなたはプロンプトエンジニアです。
以下の要件に基づいて、最適なプロンプトを生成してください。

## 要件
- 目的: {purpose}
- 技術: {tech_stack}
- 複雑度: {complexity}  # low / medium / high
- 品質要件: {quality_requirements}

## プロンプト設計の指針
1. CRISP形式を使用
2. 具体的な入出力例を含める
3. 制約条件を明示する
4. 品質チェックリストを含める

## 出力
生成したプロンプトを以下の形式で出力してください:
1. プロンプト本文
2. 使用方法の説明
3. 想定される出力の品質レベル
4. さらに改善するためのヒント
"""

# 使用例: 認証機能のプロンプトを自動生成
auth_meta = META_PROMPT.format(
    purpose="OAuth 2.0 + PKCE認証フローの実装",
    tech_stack="Next.js 14 + NextAuth.js v5",
    complexity="high",
    quality_requirements="セキュリティ監査をパスできるレベル",
)
```

### 7.2 自己改善プロンプト

```python
# AIに自分のプロンプトを評価・改善させる

SELF_IMPROVEMENT_PROMPT = """
## ステップ1
以下のプロンプトでコードを生成してください。

{original_prompt}

## ステップ2
生成したコードの品質を自己評価してください（1-10点）。

## ステップ3
品質が8点未満の場合、プロンプトのどの部分が不足していたか分析し、
改善版のプロンプトを提案してください。

## ステップ4
改善版プロンプトでコードを再生成し、品質の変化を報告してください。
"""

# 自動改善ループの実装
class SelfImprovingPrompt:
    """プロンプトを自動的に改善するループ"""

    def __init__(self, ai_client, initial_prompt: str, target_score: int = 8):
        self.client = ai_client
        self.prompt = initial_prompt
        self.target_score = target_score
        self.history: list[dict] = []

    def run(self, max_iterations: int = 5) -> dict:
        for i in range(max_iterations):
            # 生成
            output = self.client.generate(self.prompt)

            # 自己評価
            evaluation = self.client.generate(f"""
            以下のコードの品質を1-10点で評価してください。
            スコアと改善点を JSON形式で出力。

            {output}
            """)

            score = evaluation['score']
            self.history.append({
                'iteration': i + 1,
                'prompt': self.prompt,
                'score': score,
                'feedback': evaluation['improvements'],
            })

            if score >= self.target_score:
                return {
                    'final_prompt': self.prompt,
                    'final_output': output,
                    'iterations': i + 1,
                    'history': self.history,
                }

            # プロンプト改善
            self.prompt = self.client.generate(f"""
            元のプロンプト:
            {self.prompt}

            評価フィードバック:
            {evaluation['improvements']}

            上記のフィードバックを反映した改善版プロンプトを出力してください。
            """)

        return {
            'final_prompt': self.prompt,
            'iterations': max_iterations,
            'history': self.history,
            'note': 'ターゲットスコアに達しませんでした',
        }
```

### 7.3 条件分岐プロンプト

```python
# 出力内容を条件に応じて動的に変化させるプロンプト

CONDITIONAL_PROMPT_TEMPLATE = """
## 基本要件
{base_requirement}

## 条件分岐

### if ターゲット環境 == "production"
- エラーハンドリングを完全に実装
- 構造化ログ（JSON形式）を出力
- Prometheus メトリクスを追加
- ヘルスチェックエンドポイントを含める
- Graceful shutdown を実装

### elif ターゲット環境 == "staging"
- エラーハンドリングを実装
- デバッグログを有効化
- テストデータ生成ユーティリティを含める

### elif ターゲット環境 == "development"
- 基本的なエラーハンドリングのみ
- ホットリロード対応
- コンソールログ
- swagger UIを有効化

## 現在のターゲット環境: {target_env}
上記の条件に従い、適切なコードを生成してください。
"""

# 使用例
prompt = CONDITIONAL_PROMPT_TEMPLATE.format(
    base_requirement="ユーザー管理API（CRUD）",
    target_env="production",
)
```

---

## アンチパターン

### アンチパターン 1: ワンショット万能プロンプト

```markdown
# BAD: 1つのプロンプトで全てを解決しようとする
"ECサイトの全機能を実装して。ユーザー管理、商品管理、注文管理、
決済連携、在庫管理、レコメンド、通知機能を含めて。
フロントはReact、バックはFastAPI、DBはPostgreSQLで。"

# → 出力が膨大かつ品質が低い。コンテキスト制限にも引っかかる。

# GOOD: 機能単位で分割し、依存関係順に生成する
"Step 1: ドメインモデル（User, Product, Order）の定義"
"Step 2: Userの CRUD API実装"
"Step 3: Productの CRUD API実装（Step 2の規約に従う）"
# ...段階的に構築
```

### アンチパターン 2: コンテキスト不足プロンプト

```markdown
# BAD: プロジェクト固有の情報を提供しない
"ログイン機能を作って"

# → 汎用的すぎるコードが生成され、既存コードと整合しない

# GOOD: 既存コードと規約をコンテキストとして提供
"以下の既存認証モジュール（auth/service.py）のパターンに従い、
OAuth2.0によるGoogleログイン機能を追加してください。
既存のSessionManagerを再利用し、UserRepositoryに
google_idフィールドを追加する想定です。"
```

### アンチパターン 3: プロンプトの放置

```markdown
# BAD: プロンプトを一度書いたら放置する
# 3ヶ月前に書いたプロンプトをそのまま使い続ける
# → モデルが更新され、技術スタックも変わり、品質が劣化

# GOOD: プロンプトも定期的にメンテナンスする
# .prompts/CHANGELOG.md
## 2025-04-01
- claude-sonnet-4-20250514 対応: Temperature指定を削除（不要になったため）
- React 19対応: use()フックの使用を許可する制約を追加
- テスト: Vitest v2対応のプロンプトに更新
```

### アンチパターン 4: 出力の無条件受け入れ

```python
# BAD: AIの出力をそのまま本番コードに投入
generated_code = ai.generate(prompt)
deploy(generated_code)  # 危険！

# GOOD: 必ず検証プロセスを経る
generated_code = ai.generate(prompt)

# Step 1: 静的解析
lint_result = run_linter(generated_code)
assert lint_result.errors == 0

# Step 2: 型チェック
type_result = run_type_checker(generated_code)
assert type_result.errors == 0

# Step 3: テスト実行
test_result = run_tests(generated_code)
assert test_result.passed

# Step 4: セキュリティスキャン
security_result = run_security_scan(generated_code)
assert security_result.critical == 0

# Step 5: 人間によるレビュー
review = request_human_review(generated_code)
assert review.approved

# Step 6: デプロイ
deploy(generated_code)
```

### アンチパターン 5: コンテキストウィンドウの浪費

```markdown
# BAD: 不要な情報をプロンプトに詰め込む
"以下のプロジェクト全体のファイル一覧です（500ファイル）。
この中のorder.pyを修正してください..."

# → コンテキストウィンドウを浪費し、肝心の指示が薄まる

# GOOD: 必要最小限のコンテキストを厳選する
"以下のorder.pyと、関連するorder_repository.py、
order_event.pyを提供します。
order.pyのcancelメソッドを修正してください。"
```

---

## 8. チームでのPDD運用

### 8.1 プロンプトレビュー体制

```yaml
# .github/PULL_REQUEST_TEMPLATE/prompt_review.md

## プロンプト変更レビューチェックリスト

### 必須確認項目
- [ ] CLEAR基準を満たしているか
- [ ] 既存のプロンプトとの一貫性があるか
- [ ] テンプレートパラメータが文書化されているか
- [ ] バリデーション条件が定義されているか
- [ ] 想定される出力例が含まれているか

### 品質スコア
- 具体性: /4
- 段階性: /4
- 例の豊富さ: /4
- 実行可能性: /4
- 再現可能性: /4
- **合計: /20**

### セキュリティ確認
- [ ] プロンプトインジェクションに対する防御があるか
- [ ] 機密情報がハードコードされていないか
- [ ] 出力にセンシティブなデータが含まれないか

### レビュアーコメント
（自由記述）
```

### 8.2 プロンプト品質ダッシュボード

```python
# チーム全体のプロンプト品質を可視化するダッシュボード

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class PromptUsageRecord:
    """プロンプト使用記録"""
    prompt_id: str
    user: str
    timestamp: datetime
    model: str
    quality_score: float         # 0-1
    iteration_count: int          # 何回改善したか
    output_accepted: bool         # 出力が受け入れられたか
    time_saved_minutes: Optional[float] = None  # 推定節約時間

@dataclass
class TeamPromptMetrics:
    """チーム全体のプロンプトメトリクス"""
    records: list[PromptUsageRecord] = field(default_factory=list)

    @property
    def average_quality(self) -> float:
        """平均品質スコア"""
        if not self.records:
            return 0
        return sum(r.quality_score for r in self.records) / len(self.records)

    @property
    def acceptance_rate(self) -> float:
        """出力受け入れ率"""
        if not self.records:
            return 0
        accepted = sum(1 for r in self.records if r.output_accepted)
        return accepted / len(self.records)

    @property
    def average_iterations(self) -> float:
        """平均反復回数"""
        if not self.records:
            return 0
        return sum(r.iteration_count for r in self.records) / len(self.records)

    @property
    def total_time_saved(self) -> float:
        """総節約時間（時間）"""
        return sum(
            r.time_saved_minutes for r in self.records
            if r.time_saved_minutes is not None
        ) / 60

    def top_prompts(self, n: int = 10) -> list[dict]:
        """品質スコアの高いプロンプトTop N"""
        from collections import defaultdict
        prompt_scores = defaultdict(list)
        for r in self.records:
            prompt_scores[r.prompt_id].append(r.quality_score)

        averaged = [
            {"prompt_id": pid, "avg_score": sum(scores) / len(scores), "usage_count": len(scores)}
            for pid, scores in prompt_scores.items()
        ]
        return sorted(averaged, key=lambda x: x["avg_score"], reverse=True)[:n]

    def generate_report(self) -> str:
        """週次レポートを生成"""
        return f"""
        ## プロンプト駆動開発 週次レポート

        ### サマリー
        - 総使用回数: {len(self.records)}
        - 平均品質スコア: {self.average_quality:.2f}
        - 出力受け入れ率: {self.acceptance_rate:.1%}
        - 平均反復回数: {self.average_iterations:.1f}
        - 総節約時間: {self.total_time_saved:.1f}時間

        ### トッププロンプト
        {json.dumps(self.top_prompts(5), indent=2, ensure_ascii=False)}
        """
```

### 8.3 PDD導入ロードマップ

```
Phase 1: 試験導入（1-2週間）
├── チーム内に2-3名のPDDチャンピオンを選出
├── 既存のボイラープレート作業をPDDに置き換え
├── 基本テンプレート（CRISP）の研修
└── 成果と課題をドキュメント化

Phase 2: 拡大（3-4週間）
├── チーム全員にPDD研修を実施
├── プロンプトテンプレートライブラリの構築開始
├── プロンプトレビューをコードレビューに統合
├── 品質メトリクスの計測開始
└── 週次レトロスペクティブにPDDの振り返りを追加

Phase 3: 標準化（5-8週間）
├── プロンプト品質基準の正式策定
├── CI/CDパイプラインへのプロンプト検証の組み込み
├── ナレッジベースの運用開始
├── 他チームへの横展開
└── ROI分析と経営報告

Phase 4: 最適化（継続的）
├── プロンプトの自動改善パイプライン
├── チーム横断のベストプラクティス共有
├── 新しいAIモデルへの適応
├── メトリクスに基づく継続的改善
└── 業界カンファレンスでの知見共有
```

---

## FAQ

### Q1: PDDはTDD（テスト駆動開発）と併用できるか？

完全に併用可能であり、むしろ相性が良い。手順は「(1) テストの仕様をプロンプトで記述 → (2) AIがテストコードを生成 → (3) テストの正しさを人間がレビュー → (4) 実装コードをプロンプトで生成 → (5) テストが通ることを確認」となる。TDDの「Red→Green→Refactor」サイクルの各段階でAIを活用できる。

### Q2: プロンプトの再利用性を高めるにはどうすればよいか？

3つの方法がある。(1) テンプレート化: CRISPなどの形式でチーム共有テンプレートを作成、(2) パラメータ化: 変数部分を `{entity_name}` のようにプレースホルダーにする、(3) バージョン管理: `.prompts/` ディレクトリでGit管理し、品質スコアをメタデータとして記録する。

### Q3: プロンプトの品質をチーム内でどう標準化すればよいか？

「プロンプトレビュー」をコードレビューと同様のプロセスとして導入する。CLEAR基準によるチェックリストを作成し、PRにプロンプトも含める。優れたプロンプトはチームWikiに登録し、パターンライブラリとして蓄積する。月次で「プロンプト品質向上会」を実施し、ベストプラクティスを更新する。

### Q4: PDDでのAIモデル選択はどうすればよいか？

タスクの複雑度に応じてモデルを使い分ける。(1) 定型的なCRUD生成やボイラープレート: 軽量・高速なモデル（Claude Haiku等）で十分、(2) 設計判断を伴う中程度の複雑さ: バランス型モデル（Claude Sonnet等）が最適、(3) アーキテクチャ設計や複雑なリファクタリング: 最高性能モデル（Claude Opus等）を使用。コスト最適化のために、段階的に上位モデルにエスカレーションする戦略が有効。

### Q5: プロンプトインジェクションへの対策はどうすればよいか？

ユーザー入力をプロンプトに組み込む場合は特に注意が必要。対策として (1) ユーザー入力とシステムプロンプトを明確に分離する、(2) 入力のサニタイズ（特殊文字のエスケープ）を行う、(3) AIの出力を信頼せず、必ずバリデーションを行う、(4) 権限の最小化（AIにシステムコマンドの実行権限を与えない）を徹底する。

### Q6: PDDの効果をどう測定すればよいか？

以下のメトリクスを追跡する。(1) 開発速度: 同種のタスクの完了時間の変化、(2) 品質: バグ発生率、コードレビューでの指摘件数の変化、(3) 再利用性: プロンプトテンプレートの使用頻度と種類数、(4) 満足度: 開発者のサーベイ（PDDに対する満足度と生産性の自己評価）。導入前のベースラインを必ず計測しておくこと。

---

## まとめ

| 項目 | 要点 |
|------|------|
| PDDの定義 | 仕様→プロンプト→生成→検証のサイクルで開発する手法 |
| 設計パターン | CRISP形式、段階的詳細化、コンテキスト注入、制約指定 |
| 品質基準 | CLEAR（具体的・段階的・例付き・実行可能・再現可能） |
| 反復改善 | 平均2-3回の改善で95点品質に到達 |
| バージョン管理 | プロンプトもコードと同様にGit管理する |
| チーム運用 | プロンプトレビュー、品質ダッシュボード、テンプレートライブラリ |
| 注意点 | ワンショット禁止、コンテキスト必須、出力の無条件受け入れ禁止 |
| 高度テクニック | メタプロンプティング、自己改善、プロンプトチェーニング |

---

## 次に読むべきガイド

- [../01-ai-coding/00-github-copilot.md](../01-ai-coding/00-github-copilot.md) ── GitHub Copilotでのプロンプト実践
- [../01-ai-coding/01-claude-code.md](../01-ai-coding/01-claude-code.md) ── Claude Codeでの高度なPDD
- [../02-workflow/00-ai-testing.md](../02-workflow/00-ai-testing.md) ── PDD+TDDの統合アプローチ

---

## 参考文献

1. Elvis Saravia, "Prompt Engineering Guide," 2024. https://www.promptingguide.ai/
2. Anthropic, "Prompt Engineering Documentation," 2025. https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering
3. Lilian Weng, "Prompt Engineering," lilianweng.github.io, 2023. https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
4. Harrison Chase, "LangChain: Building applications with LLMs," 2024. https://python.langchain.com/docs/
