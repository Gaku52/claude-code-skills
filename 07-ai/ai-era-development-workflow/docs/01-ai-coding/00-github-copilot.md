# GitHub Copilot ── 設定、効果的な使い方、制限

> GitHub Copilotの仕組みから実践的な活用法、知っておくべき制限事項までを網羅し、日常のコーディングで最大限の生産性向上を実現する方法を学ぶ。

---

## この章で学ぶこと

1. **Copilotのアーキテクチャと設定** ── 補完エンジンの仕組みを理解し、最適な環境設定を行う
2. **効果的な利用パターン** ── 補完精度を最大化するテクニックとワークフローを習得する
3. **制限事項と代替戦略** ── Copilotが苦手とする領域を知り、適切に対処する方法を学ぶ
4. **チーム導入と運用** ── 組織でCopilotを効果的に展開するための戦略とガバナンスを学ぶ
5. **パフォーマンス計測と最適化** ── Copilotの効果を定量的に評価し、継続的に改善する方法を身につける

---

## 1. GitHub Copilotのアーキテクチャ

### 1.1 動作の仕組み

```
┌─────────────────────────────────────────────────────┐
│                  GitHub Copilot 動作フロー            │
│                                                     │
│  エディタ (VSCode / JetBrains / Neovim)             │
│  ┌────────────────────────────────────────┐         │
│  │  カーソル位置の前後のコード              │         │
│  │  開いているファイルのコンテキスト         │         │
│  │  ファイルパス・言語情報                  │         │
│  └─────────────┬──────────────────────────┘         │
│                │ 送信                               │
│                ▼                                    │
│  ┌────────────────────────────────────────┐         │
│  │  GitHub Copilot サーバー                │         │
│  │  ┌──────────────┐  ┌───────────────┐  │         │
│  │  │ コンテキスト  │  │ LLMモデル     │  │         │
│  │  │ 構築エンジン  │─►│ (GPT-4o等)    │  │         │
│  │  └──────────────┘  └───────┬───────┘  │         │
│  └────────────────────────────┼──────────┘         │
│                │              │                     │
│                │ 候補返却     │                     │
│                ▼              ▼                     │
│  ┌────────────────────────────────────────┐         │
│  │  補完候補（グレーテキスト / Ghost Text）  │         │
│  │  Tab で受け入れ / Esc で拒否            │         │
│  └────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────┘
```

### 1.2 Copilotの製品ラインナップ

```
┌─────────────────────────────────────────────────┐
│           GitHub Copilot 製品体系                 │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐             │
│  │ Individual  │  │  Business    │             │
│  │ $10/月      │  │  $19/月/人   │             │
│  │             │  │              │             │
│  │ ・コード補完 │  │ ・Individual │             │
│  │ ・チャット   │  │   の全機能   │             │
│  │ ・CLI       │  │ ・組織管理   │             │
│  │             │  │ ・ポリシー   │             │
│  └─────────────┘  │ ・監査ログ   │             │
│                   └──────────────┘             │
│  ┌──────────────────────────────┐               │
│  │  Enterprise   $39/月/人      │               │
│  │  ・Business の全機能          │               │
│  │  ・Fine-tuning               │               │
│  │  ・Knowledge Base 連携        │               │
│  │  ・IP補償                     │               │
│  └──────────────────────────────┘               │
└─────────────────────────────────────────────────┘
```

### 1.3 コンテキスト構築エンジンの内部動作

Copilotがどのようにコンテキストを構築してLLMに送信するかを理解することは、補完精度を高めるために不可欠である。

```
┌──────────────────────────────────────────────────────┐
│          コンテキスト構築の詳細フロー                    │
│                                                      │
│  Step 1: ローカル情報の収集                            │
│  ┌────────────────────────────────────────┐          │
│  │ ・現在のファイルの全内容                  │          │
│  │ ・カーソル位置（行番号、カラム）           │          │
│  │ ・ファイルパス（言語推定に使用）           │          │
│  │ ・開いているタブのファイル一覧             │          │
│  └──────────────┬─────────────────────────┘          │
│                 ▼                                    │
│  Step 2: コンテキストの優先順位付け                    │
│  ┌────────────────────────────────────────┐          │
│  │ 優先度1: カーソル前後のコード（最大2000行）│          │
│  │ 優先度2: 同一ファイル内のimport文          │          │
│  │ 優先度3: 関連ファイル（同名 .test / .d.ts）│          │
│  │ 優先度4: 開いているタブの内容              │          │
│  │ 優先度5: 隣接ディレクトリのファイル         │          │
│  └──────────────┬─────────────────────────┘          │
│                 ▼                                    │
│  Step 3: トークン制限内でのプロンプト構築              │
│  ┌────────────────────────────────────────┐          │
│  │ 総トークン予算: 約8,000トークン          │          │
│  │ ・Prefix（カーソル前）: 約4,000トークン   │          │
│  │ ・Suffix（カーソル後）: 約2,000トークン   │          │
│  │ ・関連ファイル: 約2,000トークン           │          │
│  └──────────────┬─────────────────────────┘          │
│                 ▼                                    │
│  Step 4: LLM呼出と候補生成                            │
│  ┌────────────────────────────────────────┐          │
│  │ ・Fill-in-the-Middle (FIM) 形式で送信    │          │
│  │ ・複数候補を並列生成（通常3候補）         │          │
│  │ ・post-processingでフィルタリング         │          │
│  └────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
```

### 1.4 Copilot Agentモードのアーキテクチャ

2025年後半からCopilotに追加されたAgentモードは、従来のインライン補完とは根本的に異なるアーキテクチャを持つ。

```
┌──────────────────────────────────────────────────────┐
│            Copilot Agent Mode アーキテクチャ            │
│                                                      │
│  ┌────────────────────────────────────────────┐      │
│  │  ユーザー指示                                │      │
│  │  "認証機能を実装して、テストも書いて"         │      │
│  └──────────────┬─────────────────────────────┘      │
│                 ▼                                    │
│  ┌────────────────────────────────────────────┐      │
│  │  Planning Agent                             │      │
│  │  ┌────────────────────────────────────┐    │      │
│  │  │ 1. タスクを分解                      │    │      │
│  │  │ 2. ファイル構成を計画                 │    │      │
│  │  │ 3. 実行順序を決定                    │    │      │
│  │  └────────────────────────────────────┘    │      │
│  └──────────────┬─────────────────────────────┘      │
│                 ▼                                    │
│  ┌────────────────────────────────────────────┐      │
│  │  Execution Agent                            │      │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │      │
│  │  │ ファイル  │ │ コード   │ │ ターミナル│  │      │
│  │  │ 検索     │ │ 編集     │ │ コマンド  │  │      │
│  │  └──────────┘ └──────────┘ └──────────┘  │      │
│  │  ┌──────────┐ ┌──────────┐               │      │
│  │  │ テスト   │ │ エラー   │               │      │
│  │  │ 実行     │ │ 修正     │               │      │
│  │  └──────────┘ └──────────┘               │      │
│  └──────────────┬─────────────────────────────┘      │
│                 ▼                                    │
│  ┌────────────────────────────────────────────┐      │
│  │  結果の提示                                  │      │
│  │  ・変更ファイルの差分プレビュー               │      │
│  │  ・テスト結果のサマリー                       │      │
│  │  ・Accept/Reject の選択肢                    │      │
│  └────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

---

## 2. 最適な設定

### コード例1: VSCode設定

```jsonc
// .vscode/settings.json
{
  // Copilot基本設定
  "github.copilot.enable": {
    "*": true,
    "plaintext": false,     // プレーンテキストでは無効
    "markdown": true,        // Markdownでは有効
    "yaml": true,
    "json": true
  },

  // インライン補完の表示設定
  "editor.inlineSuggest.enabled": true,
  "editor.inlineSuggest.showToolbar": "onHover",

  // Copilot Chat設定
  "github.copilot.chat.localeOverride": "ja",  // 日本語で回答

  // 除外パターン（機密ファイルをCopilotに送信しない）
  "github.copilot.advanced": {
    "debug.overrideEngine": "",
    "inlineSuggest.count": 3  // 候補数
  }
}
```

### コード例2: .copilotignore でファイルを除外

```gitignore
# .copilotignore - Copilotに送信しないファイル

# 機密情報
.env
.env.local
*.pem
*.key
credentials/

# 生成ファイル（ノイズになる）
dist/
node_modules/
*.min.js

# ライセンス上の問題があるコード
vendor/proprietary/
```

### コード例3: 効果的なコメント駆動補完

```python
# Copilotの補完精度を高めるコメントの書き方

# BAD: 曖昧なコメント
# データを処理する
def process():
    pass  # → 何を処理するか不明で、低品質な補完

# GOOD: 具体的な仕様をコメントで記述
# 売上CSVファイルを読み込み、月別・カテゴリ別に集計する
# 入力: CSVファイルパス（ヘッダー: date, category, amount）
# 出力: dict[str, dict[str, float]] = {月: {カテゴリ: 合計}}
# エラー: FileNotFoundError, csv.Error
def aggregate_sales(filepath: str) -> dict[str, dict[str, float]]:
    # → Copilotが正確な実装を補完
    import csv
    from collections import defaultdict
    from datetime import datetime

    result: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            month = datetime.strptime(row['date'], '%Y-%m-%d').strftime('%Y-%m')
            category = row['category']
            amount = float(row['amount'])
            result[month][category] += amount

    return dict(result)
```

### コード例4: Copilot Chatの活用

```python
# Copilot Chat (Cmd+I) の効果的な使い方

# 1. コード説明を求める
# 選択範囲 → /explain → 日本語で説明が返る

# 2. テスト生成
# 関数を選択 → /tests → pytestのテストが生成される

# 3. リファクタリング
# コードブロックを選択 → "この関数をリファクタリングして。
# 単一責任の原則に従い、3つの関数に分割して"

# 4. バグ修正
# エラーメッセージをペースト → /fix → 修正コードが提案される

# 5. ドキュメント生成
# 関数を選択 → /doc → docstringが生成される
```

### コード例5: Copilot CLIの活用

```bash
# GitHub Copilot CLI（ターミナル補完）

# コマンドの説明を求める
gh copilot explain "find . -name '*.py' -exec grep -l 'import os' {} +"

# 自然言語からコマンドを生成
gh copilot suggest "過去7日間に変更されたPythonファイルを検索"
# → find . -name "*.py" -mtime -7

# Gitの複雑な操作
gh copilot suggest "mainブランチとの差分があるファイル一覧を表示"
# → git diff --name-only main...HEAD

# システム管理
gh copilot suggest "ポート3000を使っているプロセスを見つけて終了"
# → lsof -ti:3000 | xargs kill -9
```

### コード例6: JetBrains IDEでのCopilot設定

```xml
<!-- JetBrains IDE (IntelliJ IDEA / PyCharm / WebStorm) の設定 -->
<!-- Settings → Plugins → "GitHub Copilot" をインストール -->

<!-- Copilotの動作カスタマイズ -->
<!-- Settings → Languages & Frameworks → GitHub Copilot -->
```

```kotlin
// JetBrains IDEでのCopilot活用例

// 1. コード補完はVSCodeとほぼ同じ操作感
// Tab: 受け入れ、Esc: 拒否

// 2. JetBrains固有の強み
// - リファクタリング機能との併用
//   Copilotで生成 → IntelliJのリファクタリングで整理

// 3. Copilot Chatの利用
// Tool Window → GitHub Copilot Chat

// 実践例: Kotlinでのデータクラス生成
// コメントで仕様を記述すると、Copilotが補完する

// ユーザー情報を管理するデータクラス
// - id: UUID（自動生成）
// - name: 氏名（1-100文字）
// - email: メールアドレス（RFC 5322準拠）
// - role: 権限（ADMIN, EDITOR, VIEWER）
// - createdAt: 作成日時
// - updatedAt: 更新日時

data class User(
    val id: UUID = UUID.randomUUID(),
    val name: String,
    val email: String,
    val role: UserRole,
    val createdAt: Instant = Instant.now(),
    val updatedAt: Instant = Instant.now()
) {
    init {
        require(name.length in 1..100) { "Name must be 1-100 characters" }
        require(email.matches(Regex("^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+$"))) {
            "Invalid email format"
        }
    }
}

enum class UserRole {
    ADMIN, EDITOR, VIEWER
}
```

### コード例7: Neovimでのcopilot.vim設定

```lua
-- Neovim の init.lua でCopilot設定

-- copilot.vimプラグインのインストール（lazy.nvim使用）
return {
  {
    "github/copilot.vim",
    event = "InsertEnter",
    config = function()
      -- 補完の有効/無効を言語別に設定
      vim.g.copilot_filetypes = {
        ["*"] = true,
        ["markdown"] = true,
        ["yaml"] = true,
        ["json"] = true,
        ["plaintext"] = false,  -- プレーンテキストでは無効
      }

      -- キーマッピングのカスタマイズ
      vim.g.copilot_no_tab_map = true
      vim.keymap.set("i", "<C-J>", 'copilot#Accept("\\<CR>")', {
        expr = true,
        replace_keycodes = false,
      })
      vim.keymap.set("i", "<C-]>", "<Plug>(copilot-next)")     -- 次の候補
      vim.keymap.set("i", "<C-[>", "<Plug>(copilot-previous)") -- 前の候補
      vim.keymap.set("i", "<C-\\>", "<Plug>(copilot-dismiss)") -- 拒否

      -- 除外ディレクトリの設定
      vim.g.copilot_workspace_folders = {
        vim.fn.expand("~/projects/current-project"),
      }
    end,
  },

  -- copilot-cmp（nvim-cmpとの統合）
  {
    "zbirenbaum/copilot-cmp",
    dependencies = { "zbirenbaum/copilot.lua" },
    config = function()
      require("copilot_cmp").setup({
        suggestion = { enabled = false },
        panel = { enabled = false },
      })
    end,
  },
}
```

---

## 3. 制限事項と対処法

### 3.1 Copilotの得意・不得意

| 得意な領域 | 不得意な領域 |
|-----------|-------------|
| 定型的なCRUD操作 | 複雑なビジネスロジック |
| 標準ライブラリの使用 | ドメイン固有の処理 |
| テストコードの生成 | セキュリティクリティカルな実装 |
| ドキュメントコメント | マルチファイルの大規模リファクタリング |
| 正規表現の作成 | プロジェクト全体のアーキテクチャ設計 |
| データ変換ロジック | 社内独自フレームワークの利用 |

### 3.2 補完品質の比較（言語別）

| 言語 | 補完精度 | 理由 |
|------|---------|------|
| Python | 非常に高い | 学習データが豊富、コミュニティが大きい |
| TypeScript | 非常に高い | 型情報がコンテキストとして有効 |
| Java | 高い | 定型パターンが多く予測しやすい |
| Rust | 中程度 | 所有権システムの理解が完全ではない |
| Haskell | 中程度 | 関数型パターンの学習データが少ない |
| COBOL | 低い | 学習データが限定的 |

### 3.3 フレームワーク別の補完品質

| フレームワーク | 補完精度 | 得意なパターン | 注意点 |
|--------------|---------|-------------|--------|
| React | 非常に高い | コンポーネント定義、hooks | 最新のServer Componentsは精度低下 |
| Next.js | 高い | ルーティング、API Routes | App Router vs Pages Routerの混同 |
| Django | 非常に高い | モデル定義、ビュー、フォーム | カスタムミドルウェアは精度低下 |
| FastAPI | 高い | エンドポイント定義、Pydanticモデル | 複雑なDependency Injectionは弱い |
| Spring Boot | 高い | Controller、Service、Repository | AOP設定の補完精度は中程度 |
| Ruby on Rails | 高い | MVC全般、マイグレーション | metaprogrammingパターンは弱い |
| Flutter | 中程度 | Widget定義、State管理 | カスタムRenderObjectは弱い |
| SwiftUI | 中程度 | ビュー定義、修飾子 | 複雑なアニメーションは弱い |

### 3.4 制限への具体的な対処法

```python
# 制限1: 古いAPIバージョンのコードを生成する問題
# 対処: コメントでバージョンを明示する

# GOOD: バージョンを明示
# Python 3.12, FastAPI 0.109, Pydantic v2 を使用
from pydantic import BaseModel, field_validator  # v2のAPIを指定

class UserCreate(BaseModel):
    name: str
    email: str

    # Pydantic v2の書き方を明示
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email")
        return v

# 制限2: プロジェクト固有の命名規則を知らない問題
# 対処: 同一ファイル内に例を示す

# プロジェクトの命名規則:
# - サービス名: XxxService（例: OrderService, PaymentService）
# - リポジトリ名: XxxRepository（例: OrderRepository）
# - DTOクラス名: XxxDto（例: OrderDto, OrderCreateDto）

class OrderService:
    """注文関連のビジネスロジック"""
    def __init__(self, order_repo: OrderRepository):
        self.order_repo = order_repo

    # → 以降のメソッド補完では命名規則が反映される

# 制限3: ビジネスロジックの文脈を理解できない問題
# 対処: ドメイン用語をコメントで定義する

# ドメイン用語:
# - SKU: Stock Keeping Unit（在庫管理単位）
# - MOQ: Minimum Order Quantity（最小発注数量）
# - Lead Time: 発注から納品までの日数
# - Safety Stock: 安全在庫（需要変動のバッファ）

def calculate_reorder_point(
    average_daily_demand: float,
    lead_time_days: int,
    safety_stock: int
) -> int:
    """再発注点の計算（ROP = 平均日次需要 × リードタイム + 安全在庫）"""
    return int(average_daily_demand * lead_time_days) + safety_stock
```

### 3.5 セキュリティリスクと対策

```
┌──────────────────────────────────────────────────────────┐
│         Copilot セキュリティリスクマトリクス                │
│                                                          │
│  リスク              影響度    発生頻度    対策            │
│  ─────────────────────────────────────────────────       │
│  機密情報の漏洩        高       中        .copilotignore │
│  脆弱なコードの生成    中       高        レビュー必須    │
│  ライセンス違反        中       低        public code    │
│                                         filter有効化    │
│  古いAPIの使用         低       高        バージョン明示  │
│  意図しないデータ送信  中       中        ネットワーク     │
│                                         ポリシー設定    │
│                                                          │
│  セキュリティチェックリスト:                               │
│  □ .copilotignore が設定されている                        │
│  □ public code filter が有効                             │
│  □ 組織のセキュリティポリシーに準拠                        │
│  □ 生成コードのセキュリティレビュープロセスがある           │
│  □ 機密リポジトリでのCopilot使用ポリシーが定義済み         │
│  □ SOC2/ISO27001要件との整合性を確認済み                  │
└──────────────────────────────────────────────────────────┘
```

```python
# セキュリティ観点でCopilotの生成コードを検証するチェッカー

import ast
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    severity: Severity
    category: str
    message: str
    line_number: int
    suggestion: str


class CopilotSecurityChecker:
    """Copilotが生成したコードのセキュリティ問題を検出する"""

    # 危険なパターン定義
    DANGEROUS_PATTERNS = [
        {
            "pattern": r"eval\(",
            "severity": Severity.CRITICAL,
            "category": "injection",
            "message": "eval()の使用は任意コード実行の脆弱性を生む",
            "suggestion": "ast.literal_eval()またはjson.loads()を使用する",
        },
        {
            "pattern": r"exec\(",
            "severity": Severity.CRITICAL,
            "category": "injection",
            "message": "exec()の使用は任意コード実行の脆弱性を生む",
            "suggestion": "安全な代替手段を検討する",
        },
        {
            "pattern": r"subprocess\.call\(.*, shell=True",
            "severity": Severity.HIGH,
            "category": "injection",
            "message": "shell=Trueはコマンドインジェクションのリスク",
            "suggestion": "shell=Falseでリスト形式の引数を使用する",
        },
        {
            "pattern": r"pickle\.loads?\(",
            "severity": Severity.HIGH,
            "category": "deserialization",
            "message": "pickleの読み込みは任意コード実行の脆弱性を生む",
            "suggestion": "json形式などの安全なシリアライゼーションを使用する",
        },
        {
            "pattern": r"password\s*=\s*['\"][^'\"]+['\"]",
            "severity": Severity.CRITICAL,
            "category": "hardcoded_secret",
            "message": "パスワードがハードコードされている",
            "suggestion": "環境変数またはシークレット管理サービスを使用する",
        },
        {
            "pattern": r"(api_key|secret_key|token)\s*=\s*['\"][^'\"]+['\"]",
            "severity": Severity.CRITICAL,
            "category": "hardcoded_secret",
            "message": "APIキーまたはシークレットがハードコードされている",
            "suggestion": "環境変数またはシークレット管理サービスを使用する",
        },
        {
            "pattern": r"verify\s*=\s*False",
            "severity": Severity.MEDIUM,
            "category": "ssl",
            "message": "SSL証明書の検証が無効化されている",
            "suggestion": "本番環境では必ずverify=Trueにする",
        },
        {
            "pattern": r"md5\(|sha1\(",
            "severity": Severity.MEDIUM,
            "category": "crypto",
            "message": "弱いハッシュアルゴリズムの使用",
            "suggestion": "SHA-256以上のアルゴリズムを使用する",
        },
    ]

    def check_code(self, code: str) -> list[SecurityFinding]:
        """コードをスキャンしてセキュリティ問題を検出する"""
        findings: list[SecurityFinding] = []

        for line_num, line in enumerate(code.split("\n"), 1):
            for pattern_def in self.DANGEROUS_PATTERNS:
                if re.search(pattern_def["pattern"], line):
                    findings.append(SecurityFinding(
                        severity=pattern_def["severity"],
                        category=pattern_def["category"],
                        message=pattern_def["message"],
                        line_number=line_num,
                        suggestion=pattern_def["suggestion"],
                    ))

        return findings

    def generate_report(self, findings: list[SecurityFinding]) -> str:
        """検出結果のレポートを生成する"""
        if not findings:
            return "セキュリティ問題は検出されませんでした。"

        report_lines = ["## セキュリティスキャン結果\n"]
        report_lines.append(f"検出数: {len(findings)}\n")

        # 重大度別にグループ化
        by_severity = {}
        for f in findings:
            by_severity.setdefault(f.severity.value, []).append(f)

        for severity in ["critical", "high", "medium", "low"]:
            if severity in by_severity:
                report_lines.append(f"\n### {severity.upper()}")
                for finding in by_severity[severity]:
                    report_lines.append(
                        f"- L{finding.line_number}: {finding.message}"
                    )
                    report_lines.append(f"  修正案: {finding.suggestion}")

        return "\n".join(report_lines)


# 使用例
checker = CopilotSecurityChecker()
code_to_check = '''
import subprocess
password = "admin123"
subprocess.call(f"echo {user_input}", shell=True)
'''
findings = checker.check_code(code_to_check)
print(checker.generate_report(findings))
```

---

## 4. 補完精度を高めるテクニック

### テクニック図解

```
┌─────────────────────────────────────────────────┐
│         Copilot 補完精度向上テクニック             │
│                                                 │
│  1. ファイル名を明確に                            │
│     ✗ utils.py                                  │
│     ✓ order_cancellation_service.py             │
│                                                 │
│  2. 関連ファイルを開いておく                      │
│     タブで開いているファイル = コンテキスト         │
│     → モデル定義ファイルを開くと補完精度UP         │
│                                                 │
│  3. 型ヒント / JSDocを先に書く                    │
│     型情報 → 補完の制約 → 精度向上               │
│                                                 │
│  4. テストファイルで実装の意図を示す              │
│     テストを先に書く → 実装ファイルの補完が向上    │
│                                                 │
│  5. 段階的に補完を受け入れる                      │
│     Ctrl+→ で単語単位の部分受け入れ              │
└─────────────────────────────────────────────────┘
```

### 4.1 テスト駆動でCopilotの精度を上げる

```python
# テストを先に書くことでCopilotの補完精度を飛躍的に向上させる

# Step 1: テストファイルを先に作成（test_order_service.py）
import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from app.services.order_service import OrderService
from app.models.order import Order, OrderStatus, OrderItem


class TestOrderService:
    """OrderServiceのテスト"""

    def test_create_order_with_valid_items(self):
        """有効なアイテムで注文を作成できる"""
        service = OrderService()
        items = [
            OrderItem(product_id="PROD-001", quantity=2, unit_price=Decimal("1500")),
            OrderItem(product_id="PROD-002", quantity=1, unit_price=Decimal("3000")),
        ]
        order = service.create_order(customer_id="CUST-001", items=items)
        assert order.status == OrderStatus.PENDING
        assert order.total == Decimal("6000")

    def test_create_order_with_empty_items_raises_error(self):
        """空のアイテムリストでは注文できない"""
        service = OrderService()
        with pytest.raises(ValueError, match="At least one item is required"):
            service.create_order(customer_id="CUST-001", items=[])

    def test_cancel_order_within_grace_period(self):
        """猶予期間内のキャンセルは返金される"""
        service = OrderService()
        order = service.create_order(
            customer_id="CUST-001",
            items=[OrderItem(product_id="PROD-001", quantity=1, unit_price=Decimal("1000"))],
        )
        result = service.cancel_order(order.id, reason="customer_request")
        assert result.status == OrderStatus.CANCELLED
        assert result.refund_amount == Decimal("1000")

    def test_cancel_shipped_order_raises_error(self):
        """出荷済み注文はキャンセルできない"""
        service = OrderService()
        order = service.create_order(
            customer_id="CUST-001",
            items=[OrderItem(product_id="PROD-001", quantity=1, unit_price=Decimal("1000"))],
        )
        service.ship_order(order.id)
        with pytest.raises(ValueError, match="Cannot cancel shipped order"):
            service.cancel_order(order.id, reason="customer_request")


# Step 2: 実装ファイルに移動（order_service.py）
# → テストファイルがタブで開いているため、
#   Copilotはテストの仕様を参考にして正確に実装を補完する
```

### 4.2 型ヒントによる精度向上

```typescript
// TypeScriptの型定義を先に書くとCopilotの精度が劇的に向上する

// Step 1: 型定義ファイルを作成（types/order.ts）
export interface Order {
  id: string;
  customerId: string;
  items: OrderItem[];
  status: OrderStatus;
  total: number;
  createdAt: Date;
  updatedAt: Date;
  shippedAt?: Date;
  cancelledAt?: Date;
  cancellationReason?: string;
}

export interface OrderItem {
  productId: string;
  productName: string;
  quantity: number;
  unitPrice: number;
  subtotal: number;
}

export enum OrderStatus {
  PENDING = "PENDING",
  CONFIRMED = "CONFIRMED",
  SHIPPED = "SHIPPED",
  DELIVERED = "DELIVERED",
  CANCELLED = "CANCELLED",
}

export interface CreateOrderRequest {
  customerId: string;
  items: Array<{
    productId: string;
    quantity: number;
  }>;
  shippingAddress: Address;
  paymentMethod: PaymentMethod;
}

export interface OrderSummary {
  totalOrders: number;
  totalRevenue: number;
  averageOrderValue: number;
  statusBreakdown: Record<OrderStatus, number>;
}

// Step 2: サービスファイルに移動
// → 型定義ファイルが開いているため、Copilotは
//   CreateOrderRequestの構造に沿った実装を正確に補完する
```

### 4.3 コンテキストウィンドウの効果的な活用

```
┌──────────────────────────────────────────────────────────┐
│       コンテキスト最適化のための開くべきファイル戦略        │
│                                                          │
│  タスク              開いておくべきファイル                │
│  ─────────────────────────────────────────────────       │
│  新しいAPI実装       ┌─────────────────────────┐         │
│                     │ 1. 既存の類似APIファイル   │         │
│                     │ 2. リクエスト/レスポンス型 │         │
│                     │ 3. ルーティング設定       │         │
│                     │ 4. テストファイル         │         │
│                     └─────────────────────────┘         │
│                                                          │
│  DB操作追加          ┌─────────────────────────┐         │
│                     │ 1. スキーマ定義           │         │
│                     │ 2. 既存リポジトリ         │         │
│                     │ 3. マイグレーションファイル │         │
│                     └─────────────────────────┘         │
│                                                          │
│  テスト作成          ┌─────────────────────────┐         │
│                     │ 1. テスト対象ファイル      │         │
│                     │ 2. 既存テスト（同ディレクトリ）│     │
│                     │ 3. テストヘルパー/フィクスチャ│     │
│                     │ 4. 型定義ファイル          │         │
│                     └─────────────────────────┘         │
│                                                          │
│  フロントエンド       ┌─────────────────────────┐         │
│  コンポーネント作成   │ 1. 類似コンポーネント     │         │
│                     │ 2. 共通UIコンポーネント    │         │
│                     │ 3. APIクライアント         │         │
│                     │ 4. スタイル変数/テーマ     │         │
│                     └─────────────────────────┘         │
└──────────────────────────────────────────────────────────┘
```

### 4.4 プロンプトエンジニアリングテクニック

```python
# Copilotの補完を最大限活用するプロンプトテクニック集

# テクニック1: 段階的詳細化（Progressive Refinement）
# まず大枠のコメントを書き、徐々に詳細化する

# === バッチ処理エンジン ===
# 目的: 大量のデータを効率的に処理する
# 要件:
#   - 入力: ジェネレータ（メモリ効率を考慮）
#   - バッチサイズ: 設定可能（デフォルト100件）
#   - エラーハンドリング: 個別エラーはスキップしてログ記録
#   - リトライ: 指数バックオフで最大3回
#   - 進捗報告: コールバック関数で通知
#   - 並行処理: asyncioベースで最大同時実行数を制限

import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, TypeVar, Generic

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """バッチ処理の設定"""
    batch_size: int = 100
    max_retries: int = 3
    max_concurrency: int = 10
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0


@dataclass
class BatchResult(Generic[R]):
    """バッチ処理の結果"""
    successful: list[R] = field(default_factory=list)
    failed: list[tuple[Exception, T]] = field(default_factory=list)
    total_processed: int = 0
    total_errors: int = 0


class BatchProcessor(Generic[T, R]):
    """
    大量データの非同期バッチ処理エンジン。

    使用例:
        processor = BatchProcessor(
            process_fn=send_email,
            config=BatchConfig(batch_size=50, max_concurrency=5)
        )
        result = await processor.run(email_generator())
    """

    def __init__(
        self,
        process_fn: Callable[[T], R],
        config: BatchConfig | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        self.process_fn = process_fn
        self.config = config or BatchConfig()
        self.on_progress = on_progress
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    async def _process_with_retry(self, item: T) -> R:
        """リトライ付きで単一アイテムを処理する"""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:
                    if asyncio.iscoroutinefunction(self.process_fn):
                        return await self.process_fn(item)
                    return self.process_fn(item)
            except Exception as e:
                last_error = e
                delay = min(
                    self.config.retry_base_delay * (2 ** attempt),
                    self.config.retry_max_delay
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries} failed: {e}. "
                    f"Retrying in {delay}s"
                )
                await asyncio.sleep(delay)
        raise last_error

    async def run(self, items: AsyncGenerator[T, None] | list[T]) -> BatchResult[R]:
        """バッチ処理を実行する"""
        result = BatchResult()
        batch: list[T] = []

        async def process_batch(batch_items: list[T]):
            tasks = [self._process_with_retry(item) for item in batch_items]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for item, outcome in zip(batch_items, outcomes):
                if isinstance(outcome, Exception):
                    result.failed.append((outcome, item))
                    result.total_errors += 1
                else:
                    result.successful.append(outcome)
                result.total_processed += 1
                if self.on_progress:
                    self.on_progress(result.total_processed, result.total_errors)

        if isinstance(items, list):
            for i in range(0, len(items), self.config.batch_size):
                await process_batch(items[i:i + self.config.batch_size])
        else:
            async for item in items:
                batch.append(item)
                if len(batch) >= self.config.batch_size:
                    await process_batch(batch)
                    batch = []
            if batch:
                await process_batch(batch)

        return result
```

---

## 5. チーム導入ガイド

### 5.1 導入フェーズと計画

```
┌──────────────────────────────────────────────────────────┐
│            Copilot チーム導入ロードマップ                   │
│                                                          │
│  Phase 1: パイロット（2週間）                              │
│  ┌────────────────────────────────────────────┐          │
│  │ ・3-5名のアーリーアダプターを選定            │          │
│  │ ・Individual プランで試用開始               │          │
│  │ ・週次で使用感をフィードバック               │          │
│  │ ・セキュリティチェックリストの作成            │          │
│  └────────────────────────────────────────────┘          │
│                    ▼                                     │
│  Phase 2: 評価（2週間）                                   │
│  ┌────────────────────────────────────────────┐          │
│  │ ・定量指標の計測（補完受入率、開発速度）      │          │
│  │ ・定性フィードバック収集                     │          │
│  │ ・.copilotignore の標準化                   │          │
│  │ ・ガイドラインの策定                        │          │
│  └────────────────────────────────────────────┘          │
│                    ▼                                     │
│  Phase 3: 拡大展開（1ヶ月）                               │
│  ┌────────────────────────────────────────────┐          │
│  │ ・Business プランに移行                     │          │
│  │ ・全開発チームへ展開                        │          │
│  │ ・トレーニングセッション実施                 │          │
│  │ ・コーディング規約の更新（Copilot考慮）      │          │
│  └────────────────────────────────────────────┘          │
│                    ▼                                     │
│  Phase 4: 最適化（継続）                                  │
│  ┌────────────────────────────────────────────┐          │
│  │ ・月次で効果測定レポートを作成               │          │
│  │ ・ベストプラクティスのナレッジベース構築      │          │
│  │ ・Enterprise検討（Knowledge Base活用）      │          │
│  │ ・CI/CDパイプラインとの統合                  │          │
│  └────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### 5.2 効果測定フレームワーク

```python
# Copilot導入効果の測定スクリプト

import json
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional


@dataclass
class CopilotMetrics:
    """Copilot利用メトリクス"""
    date: date
    developer_id: str

    # 補完関連
    suggestions_shown: int = 0      # 表示された補完候補数
    suggestions_accepted: int = 0    # 受け入れた補完候補数
    characters_accepted: int = 0     # 受け入れた文字数

    # 生産性関連
    lines_of_code_written: int = 0   # 書いたコード行数
    pull_requests_created: int = 0   # 作成したPR数
    time_to_first_commit: float = 0  # 最初のコミットまでの時間（分）

    # 品質関連
    bugs_introduced: int = 0         # 導入したバグ数
    code_review_iterations: int = 0  # コードレビューの往復回数
    test_coverage_change: float = 0  # テストカバレッジの変化

    @property
    def acceptance_rate(self) -> float:
        """補完受入率"""
        if self.suggestions_shown == 0:
            return 0.0
        return self.suggestions_accepted / self.suggestions_shown * 100

    @property
    def productivity_score(self) -> float:
        """総合生産性スコア（0-100）"""
        # 加重平均で算出
        scores = {
            "acceptance_rate": min(self.acceptance_rate / 40 * 30, 30),  # 最大30点
            "code_volume": min(self.lines_of_code_written / 200 * 20, 20),  # 最大20点
            "pr_velocity": min(self.pull_requests_created / 3 * 20, 20),  # 最大20点
            "quality": max(0, 30 - self.bugs_introduced * 10 -
                          self.code_review_iterations * 5),  # 最大30点
        }
        return sum(scores.values())


class CopilotDashboard:
    """チーム全体のCopilot利用状況ダッシュボード"""

    def __init__(self):
        self.metrics_history: list[CopilotMetrics] = []

    def add_metrics(self, metrics: CopilotMetrics):
        self.metrics_history.append(metrics)

    def team_summary(self) -> dict:
        """チーム全体のサマリーを返す"""
        if not self.metrics_history:
            return {"error": "No data available"}

        total_suggestions = sum(m.suggestions_shown for m in self.metrics_history)
        total_accepted = sum(m.suggestions_accepted for m in self.metrics_history)
        avg_acceptance = total_accepted / total_suggestions * 100 if total_suggestions > 0 else 0

        developers = set(m.developer_id for m in self.metrics_history)
        avg_productivity = sum(
            m.productivity_score for m in self.metrics_history
        ) / len(self.metrics_history)

        return {
            "period": {
                "start": min(m.date for m in self.metrics_history).isoformat(),
                "end": max(m.date for m in self.metrics_history).isoformat(),
            },
            "team_size": len(developers),
            "total_suggestions_shown": total_suggestions,
            "total_suggestions_accepted": total_accepted,
            "average_acceptance_rate": round(avg_acceptance, 1),
            "average_productivity_score": round(avg_productivity, 1),
            "total_characters_accepted": sum(
                m.characters_accepted for m in self.metrics_history
            ),
            "estimated_time_saved_hours": round(
                sum(m.characters_accepted for m in self.metrics_history) / 500 * 0.5, 1
            ),
        }

    def generate_report(self) -> str:
        """レポートを生成する"""
        summary = self.team_summary()
        report = f"""
## Copilot 利用レポート

### 期間
{summary.get('period', {}).get('start', 'N/A')} 〜 {summary.get('period', {}).get('end', 'N/A')}

### チーム概要
- チーム人数: {summary.get('team_size', 0)}名
- 補完候補表示数: {summary.get('total_suggestions_shown', 0):,}
- 補完受入数: {summary.get('total_suggestions_accepted', 0):,}
- 平均受入率: {summary.get('average_acceptance_rate', 0)}%
- 平均生産性スコア: {summary.get('average_productivity_score', 0)}/100
- 受入文字数: {summary.get('total_characters_accepted', 0):,}
- 推定節約時間: {summary.get('estimated_time_saved_hours', 0)}時間

### 評価
受入率の目標は25-35%が健全な範囲。
これ以上高い場合はコードレビューの厳密化を検討。
これ以下の場合はコンテキスト改善やトレーニングが必要。
"""
        return report
```

### 5.3 組織ポリシーテンプレート

```markdown
# GitHub Copilot 利用ポリシー

## 1. 利用対象者
- 正社員の開発者全員（契約社員は要個別承認）
- QAエンジニア、SREも利用可能

## 2. 利用禁止事項
- [ ] 機密情報（顧客データ、認証情報）を含むファイルでの使用
- [ ] セキュリティクリティカルな暗号化・認証コードの無検証での使用
- [ ] 外部委託先のリポジトリでの使用（契約確認が必要）
- [ ] オフショアチームへのライセンス提供（法務確認が必要）

## 3. 必須設定
- .copilotignore の設定（テンプレートを使用）
- public code filter の有効化
- チャットログの保存期間設定

## 4. コードレビュー要件
- Copilot生成コードも通常のコードレビュープロセスを適用
- セキュリティ関連のコードは追加のセキュリティレビューを実施
- 生成コードには // Generated with Copilot のコメントを推奨

## 5. 品質基準
- 生成コードのテストカバレッジ80%以上
- Lintエラーゼロ
- ドキュメント付きの公開API
```

---

## 6. トラブルシューティング

### 6.1 よくある問題と解決策

```
┌──────────────────────────────────────────────────────────────┐
│           Copilot トラブルシューティングガイド                  │
│                                                              │
│  問題1: 補完候補が表示されない                                 │
│  ─────────────────────────────────────────────               │
│  確認1: VSCodeステータスバーのCopilotアイコン                  │
│          → 有効(緑) / 無効(灰) / エラー(赤)                   │
│  確認2: ネットワーク接続                                      │
│          → プロキシ設定、ファイアウォールを確認                 │
│  確認3: ファイルタイプの除外設定                               │
│          → settings.json の copilot.enable を確認             │
│  確認4: 拡張機能の競合                                        │
│          → 他のAI補完拡張機能を無効化                          │
│  確認5: 認証状態                                              │
│          → GitHub Copilot: Sign Out → 再ログイン              │
│                                                              │
│  問題2: 補完品質が低い                                        │
│  ─────────────────────────────────────────────               │
│  対策1: コメントで文脈を補強する                               │
│  対策2: 型ヒント/JSDocを追加する                               │
│  対策3: 関連ファイルをタブで開く                               │
│  対策4: ファイル名を明確にする                                 │
│  対策5: ワークスペース設定を確認する                           │
│                                                              │
│  問題3: 古いコード/APIが生成される                             │
│  ─────────────────────────────────────────────               │
│  対策1: コメントでバージョンを明示する                         │
│  対策2: 最新のサンプルコードを同ファイル内に配置               │
│  対策3: @docs や Web検索で最新情報を参照                      │
│  対策4: .cursorrules / CLAUDE.md でバージョン固定              │
│                                                              │
│  問題4: 他のエディタ拡張と競合する                             │
│  ─────────────────────────────────────────────               │
│  対策1: TabNine等の他AI補完を無効化                            │
│  対策2: IntelliCode補完との優先順位を設定                      │
│  対策3: キーバインドの競合を解決                               │
│         (Cmd+K, Ctrl+Space等)                                │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 パフォーマンス最適化

```jsonc
// VSCode設定によるCopilotパフォーマンス最適化

{
  // 大規模リポジトリでの最適化
  "files.watcherExclude": {
    "**/node_modules/**": true,
    "**/dist/**": true,
    "**/build/**": true,
    "**/.git/objects/**": true,
    "**/vendor/**": true
  },

  // Copilot応答速度の改善
  "editor.quickSuggestions": {
    "strings": false  // 文字列内での補完を無効化（ノイズ減少）
  },

  // メモリ使用量の最適化
  "editor.maxTokenizationLineLength": 5000,

  // ネットワーク最適化（プロキシ環境）
  "http.proxy": "http://proxy.company.com:8080",
  "http.proxyStrictSSL": true,
  "github.copilot.advanced": {
    "debug.useProxy": true
  }
}
```

### 6.3 企業ネットワークでの設定

```bash
# 企業プロキシ環境でのCopilot設定

# 1. プロキシ設定の確認
echo $HTTP_PROXY
echo $HTTPS_PROXY

# 2. VS Codeのプロキシ設定
# Settings → proxy で検索
# "http.proxy": "http://proxy.example.com:8080"

# 3. npm（copilot-cli）のプロキシ設定
npm config set proxy http://proxy.example.com:8080
npm config set https-proxy http://proxy.example.com:8080

# 4. Git のプロキシ設定
git config --global http.proxy http://proxy.example.com:8080

# 5. 認証テスト
gh auth status  # GitHubへの接続を確認
gh copilot --version  # Copilot CLIの動作確認

# 6. ファイアウォールで許可が必要なドメイン
# - github.com
# - api.github.com
# - copilot-proxy.githubusercontent.com
# - *.githubcopilot.com
# - default.exp-tas.com
```

---

## 7. Copilot Extensions と今後の展望

### 7.1 Copilot Extensions

```
┌──────────────────────────────────────────────────────────┐
│           Copilot Extensions エコシステム                  │
│                                                          │
│  ┌──────────────────────────────────┐                    │
│  │  Copilot Extensions              │                    │
│  │  ・サードパーティがCopilotを拡張   │                    │
│  │  ・Chatで@拡張名 で呼び出し       │                    │
│  │  ・GitHub Marketplace で配布     │                    │
│  └──────────────┬───────────────────┘                    │
│                 │                                        │
│  ┌──────────────▼───────────────────┐                    │
│  │  主要なExtensions                │                    │
│  │                                  │                    │
│  │  @docker     Docker関連の支援     │                    │
│  │  @sentry     エラー追跡統合       │                    │
│  │  @datadog    モニタリング統合     │                    │
│  │  @mongodb    MongoDB操作支援     │                    │
│  │  @azure      Azure開発支援       │                    │
│  │  @hashicorp  Terraform支援       │                    │
│  └──────────────────────────────────┘                    │
│                                                          │
│  利用例:                                                  │
│  > @docker このアプリのDockerfileを最適化して              │
│  > @sentry 最近の本番エラーをまとめて                      │
│  > @azure このAPIをAzure Functionsにデプロイして          │
└──────────────────────────────────────────────────────────┘
```

### 7.2 Copilot Workspace

```
┌──────────────────────────────────────────────────────────┐
│            Copilot Workspace 概要                         │
│                                                          │
│  Copilot Workspaceは、IssueからPRまでの全工程を           │
│  AIが支援する次世代開発環境。                               │
│                                                          │
│  ┌─────────┐                                            │
│  │ Issue   │ ← 自然言語で問題を記述                      │
│  └────┬────┘                                            │
│       ▼                                                  │
│  ┌─────────┐                                            │
│  │ 分析    │ ← AIがIssueを分析し、影響範囲を特定          │
│  └────┬────┘                                            │
│       ▼                                                  │
│  ┌─────────┐                                            │
│  │ 計画    │ ← 変更計画を提案（ファイル・変更内容）        │
│  └────┬────┘                                            │
│       ▼                                                  │
│  ┌─────────┐                                            │
│  │ 実装    │ ← 計画に基づいてコードを生成                 │
│  └────┬────┘                                            │
│       ▼                                                  │
│  ┌─────────┐                                            │
│  │ 検証    │ ← テスト実行、リンク確認                     │
│  └────┬────┘                                            │
│       ▼                                                  │
│  ┌─────────┐                                            │
│  │ PR作成  │ ← レビュー可能なPRを自動作成                 │
│  └─────────┘                                            │
│                                                          │
│  従来のフロー:                                            │
│  Issue → 開発者が分析 → 設計 → 実装 → テスト → PR         │
│  所要時間: 数時間〜数日                                    │
│                                                          │
│  Workspace:                                               │
│  Issue → AI分析 → AI提案 → 人間が確認・修正 → PR          │
│  所要時間: 数分〜数時間                                    │
└──────────────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン 1: Tab連打開発

```python
# BAD: Copilotの提案を連続でTabで受け入れ続ける
# → 意図しないロジックが混入するリスク

# 例: Copilotが提案した認証コード
def verify_token(token: str) -> bool:
    # Tabで受け入れたが、実は期限切れチェックが抜けている
    decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return decoded is not None  # ← 期限切れでもTrueを返す！

# GOOD: 各提案を読んで理解してから受け入れる
def verify_token(token: str) -> bool:
    try:
        decoded = jwt.decode(
            token, SECRET_KEY,
            algorithms=["HS256"],
            options={"verify_exp": True}  # 期限切れを検証
        )
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False
```

### アンチパターン 2: Copilot依存のコード理解放棄

```
❌ 悪い習慣:
   - 補完されたコードを読まずにそのまま使う
   - "動いているから正しい"と判断する
   - Copilotなしではコードが書けなくなる

✅ 良い習慣:
   - 受け入れる前に必ずコードを読む
   - 補完内容を声に出して説明できるか確認
   - 週に1回はCopilotオフでコーディング練習
   - 補完されたコードにテストを書く
```

### アンチパターン 3: コンテキスト汚染

```python
# BAD: 無関係なコードがファイル内に散在し、
# Copilotのコンテキストを汚染する

# =============================================
# ここから一時的なデバッグコード（後で消す）
# =============================================
# import pdb; pdb.set_trace()
# print("DEBUG: user_data =", user_data)
# # TODO: このif文は意味不明だが消すと壊れる
# if True:
#     pass
# =============================================

class OrderService:
    # → Copilotがこのデバッグコードの影響を受けて
    #   低品質な補完を生成する

# GOOD: ファイルをクリーンに保つ
# - デバッグコードは用が済んだらすぐ削除
# - TODO/HACKコメントは定期的にクリーンアップ
# - 未使用のimportは自動削除（isort / autoflake）
```

### アンチパターン 4: テストなしでのCopilot生成コードの投入

```python
# BAD: Copilotが生成したコードをテストなしで本番投入
def calculate_discount(price: float, coupon: str) -> float:
    # Copilotの補完をそのまま使用
    if coupon == "SUMMER20":
        return price * 0.8
    elif coupon == "VIP50":
        return price * 0.5
    return price
    # → 負の価格、0円、float精度問題を考慮していない

# GOOD: 必ずテストを書いてから本番投入
def calculate_discount(price: Decimal, coupon: str) -> Decimal:
    """割引価格を計算する"""
    if price < 0:
        raise ValueError("Price must be non-negative")

    discount_map = {
        "SUMMER20": Decimal("0.80"),
        "VIP50": Decimal("0.50"),
    }

    multiplier = discount_map.get(coupon, Decimal("1.00"))
    discounted = (price * multiplier).quantize(Decimal("0.01"))
    return max(discounted, Decimal("0"))

# テスト
class TestCalculateDiscount:
    def test_valid_coupon(self):
        assert calculate_discount(Decimal("1000"), "SUMMER20") == Decimal("800.00")

    def test_no_coupon(self):
        assert calculate_discount(Decimal("1000"), "INVALID") == Decimal("1000.00")

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            calculate_discount(Decimal("-100"), "SUMMER20")

    def test_zero_price(self):
        assert calculate_discount(Decimal("0"), "VIP50") == Decimal("0.00")
```

---

## エッジケース分析

### エッジケース 1: 大規模モノレポでのCopilot

```
問題: 数千ファイルのモノレポでCopilotの精度が低下する

原因:
- コンテキストウィンドウの制限（約8Kトークン）
- 類似名ファイルが多く、誤ったコンテキストが選択される
- 異なるチームの異なる規約が混在

対策:
1. ワークスペースフォルダで対象ディレクトリを限定
   {
     "folders": [
       {"path": "packages/my-service"}  // ← 関連パッケージのみ
     ]
   }

2. .copilotignore で無関係なパッケージを除外
   packages/other-service/
   packages/legacy-code/

3. ファイル命名でスコープを明確化
   packages/order-service/src/services/OrderCalculationService.ts
   (パッケージ名がファイルパスに含まれる)
```

### エッジケース 2: レガシーコードベースでのCopilot

```
問題: 古いパターン（jQuery、ES5、callback地獄）を学習して
     それに合わせたコードを生成してしまう

対策:
1. 新しいコードの「お手本ファイル」を作成
   → Copilotはそのパターンを参照して補完する

2. コメントで明示的にモダンパターンを指定
   // React 18 + TypeScript + hooks パターンで実装
   // jQuery, class componentは使用しない

3. 新旧コードを別ディレクトリに分離
   src/
   ├── legacy/  (.copilotignoreに追加)
   └── modern/  (Copilotがこちらのみ参照)
```

---

## FAQ

### Q1: CopilotがSuggestionを出さない場合の対処法は？

原因は主に3つ。(1) ネットワーク接続の問題 → ステータスバーのCopilotアイコンを確認、(2) ファイルタイプが除外されている → settings.jsonの `github.copilot.enable` を確認、(3) コンテキスト不足 → コメントや型ヒントを追加する。それでも解決しない場合は `Copilot: Toggle` でON/OFFを試す。

### Q2: Copilotが生成したコードの著作権はどうなるか？

GitHubのTOSによると、Copilotの出力に対してユーザーが著作権を持つ。ただし、学習データと酷似したコード（verbatim copy）が出力されるリスクがある。Enterprise版にはIP補償が含まれる。OSSライセンスとの互換性を確保するため、`public code filter` を有効にすることを推奨する。

### Q3: CopilotとCursorのどちらを選ぶべきか？

用途で判断する。Copilotは「既存エディタに追加する補完ツール」として優秀で、VSCodeやJetBrainsを離れたくない場合に最適。CursorはAIを前提に設計されたIDEで、マルチファイル編集やコードベース全体の理解が必要な場合に優位。両方を試して判断するのが理想的だが、コストを抑えたいなら まずCopilot Individualから始めるとよい。

### Q4: Copilotの補完を受け入れた後に元に戻すには？

通常のUndo（Cmd+Z / Ctrl+Z）で元に戻せる。Copilotの補完受入は通常のテキスト編集として扱われるため、エディタの標準的なUndo機能が使える。複数行の補完を受け入れた場合も、1回のUndoで全体が取り消される。

### Q5: Copilotの利用統計を確認するには？

GitHub.com → Settings → Copilot → Usage で確認可能。Business/Enterpriseプランでは管理者ダッシュボードで組織全体の統計が見られる。VSCode内ではステータスバーのCopilotアイコンをクリックすると、セッション中の受入率が表示される。

### Q6: 社内のプライベートライブラリをCopilotに学習させることはできるか？

Enterprise版のKnowledge Base機能を使えば、社内リポジトリのコードをCopilotのコンテキストとして活用できる。ただし、これはFine-tuningではなくRAG（検索拡張生成）ベースのアプローチであり、補完時に関連コードを検索して参照する仕組みである。Individual/Businessプランではこの機能は利用できない。

### Q7: CopilotとClaude Codeは併用できるか？

問題なく併用可能。エディタ内ではCopilotがインライン補完を提供し、ターミナルではClaude Codeがエージェントタスクを実行するという役割分担が効果的。ただし、Cursor IDE + Copilotの組み合わせはAI補完機能が競合する場合がある。VSCode + Copilot + ターミナルのClaude Codeという組み合わせが最も衝突が少ない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 仕組み | エディタからコンテキストをサーバーに送信、LLMが補完候補を返却 |
| 設定 | .copilotignoreで機密除外、言語別有効/無効化 |
| 精度向上 | 型ヒント、明確なファイル名、関連ファイルを開く |
| Chatの活用 | /explain, /tests, /fix, /doc で4大ユースケース |
| CLI | `gh copilot suggest` でターミナル操作も補完 |
| 制限 | 複雑なビジネスロジック、セキュリティ実装は人間が判断 |
| チーム導入 | パイロット → 評価 → 拡大展開 → 最適化の4段階 |
| 効果測定 | 受入率25-35%が健全、定期レポートで改善 |
| セキュリティ | .copilotignore、public code filter、レビュープロセス |

---

## 次に読むべきガイド

- [01-claude-code.md](./01-claude-code.md) ── Claude Codeでのエージェント型開発
- [02-cursor-and-windsurf.md](./02-cursor-and-windsurf.md) ── AI IDEとの比較
- [03-ai-coding-best-practices.md](./03-ai-coding-best-practices.md) ── AIコーディングのベストプラクティス

---

## 参考文献

1. GitHub, "GitHub Copilot Documentation," 2025. https://docs.github.com/en/copilot
2. Albert Ziegler et al., "Productivity Assessment of Neural Code Completion," ACM, 2022. https://doi.org/10.1145/3520312.3534864
3. GitHub, "GitHub Copilot Trust Center," 2025. https://resources.github.com/copilot-trust-center/
4. GitHub, "Copilot Extensions Documentation," 2025. https://docs.github.com/en/copilot/github-copilot-extensions
5. GitHub, "Copilot Workspace Technical Preview," 2025. https://githubnext.com/projects/copilot-workspace
6. Thomas Dohmke, "GitHub Copilot X: The AI-Powered Developer Experience," GitHub Blog, 2023. https://github.blog/2023-03-22-github-copilot-x-the-ai-powered-developer-experience/
