# DRY / KISS / YAGNI ── 重複排除・単純化・不要機能の排除

> ソフトウェア開発における3つの基本原則。DRYは知識の重複を排除し、KISSは複雑さを避け、YAGNIは不要な先行実装を防ぐ。この3原則の適切なバランスが、保守しやすいコードを生む。

---

## この章で学ぶこと

1. **DRY原則の正しい理解** ── 単なるコード重複排除ではなく「知識の一元化」を理解する
2. **KISSの実践方法** ── シンプルさを保ちながら要件を満たす設計を身につける
3. **YAGNIの判断基準** ── 先行投資すべきケースと不要なケースを見極める
4. **3原則の相互関係と矛盾** ── 原則が衝突する場面での優先順位判断を習得する
5. **Rule of Three の実践** ── 重複排除のタイミングを判断する経験則を学ぶ

---

## 前提知識

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| クリーンコード概要 | コード品質の基本概念 | [00-clean-code-overview.md](./00-clean-code-overview.md) |
| リファクタリング基礎 | コードの構造変更手法 | [リファクタリング技法](../02-refactoring/01-refactoring-techniques.md) |
| 関数の基本 | 関数定義、引数、戻り値 | [関数設計](../01-practices/01-functions.md) |

---

## 1. DRY ── Don't Repeat Yourself

### 1.1 定義と本質

```
+-----------------------------------------------------------+
|  DRY (Don't Repeat Yourself)                              |
|  ─────────────────────────────────────────────────         |
|  「すべての知識はシステム内で唯一の、                      |
|    曖昧でない、権威のある表現を持たなければならない」      |
|                    ── Andrew Hunt & David Thomas           |
|                       『The Pragmatic Programmer』         |
+-----------------------------------------------------------+
```

**重要な注意点:** DRYは「コードの文字列的な重複排除」ではなく、**「知識の一元化」**である。この区別を誤ると、かえってコードの品質が悪化する。

### 1.2 DRYの対象 ── 何が「重複」なのか

```
     DRYが適用される「知識」の種類
     ┌──────────────────────────────┐
     │  ビジネスロジック             │  例: 税計算ルール
     ├──────────────────────────────┤
     │  データスキーマ               │  例: ユーザー定義
     ├──────────────────────────────┤
     │  設定値                       │  例: API エンドポイント
     ├──────────────────────────────┤
     │  アルゴリズム                 │  例: ソート手順
     ├──────────────────────────────┤
     │  バリデーションルール         │  例: メール形式チェック
     ├──────────────────────────────┤
     │  データベーススキーマ         │  例: テーブル定義
     ├──────────────────────────────┤
     │  APIコントラクト             │  例: リクエスト/レスポンス形式
     └──────────────────────────────┘
     ※ コードの見た目が同じでも、
       表現している「知識」が異なれば重複ではない
```

### 1.3 WHY ── なぜDRYが重要なのか

DRY違反の根本的な問題は、**変更漏れ**（Shotgun Surgery）である。

```
  DRY違反時の変更コスト

  「消費税率を10%から12%に変更する」

  DRY準拠:                      DRY違反:
  ┌──────────────┐              ┌──────────────┐
  │ TaxCalculator │              │ InvoiceService│ ← 10% → 12%
  │ TAX_RATE=0.10│              │ tax = x*0.10  │
  │ → 0.12に変更 │              ├──────────────┤
  │ (1箇所のみ)  │              │ CartService   │ ← 10% → 12%
  └──────────────┘              │ tax = x*0.10  │
                                ├──────────────┤
  変更箇所: 1                    │ ReportService │ ← 10% → 12%
  変更漏れリスク: 0%             │ tax = x*0.10  │
                                ├──────────────┤
                                │ API Response  │ ← 10% → 12%
                                │ tax_rate: 0.10│
                                └──────────────┘
                                変更箇所: 4
                                変更漏れリスク: 高い
```

### 1.4 コード例

**コード例1: 真のDRY違反 ── 同じ知識が複数箇所にある**

```python
# DRY違反: 税計算ロジックが2箇所に存在
class InvoiceService:
    def calculate_total(self, subtotal: float) -> float:
        tax = subtotal * 0.10  # 消費税10%
        return subtotal + tax

class CartService:
    def calculate_total(self, subtotal: float) -> float:
        tax = subtotal * 0.10  # 消費税10% ← 同じ知識の重複！
        return subtotal + tax


# DRY適用: 税計算の知識を一元化
class TaxCalculator:
    """税計算の唯一の権威ある情報源"""
    TAX_RATE = 0.10

    @classmethod
    def calculate_tax(cls, amount: float) -> float:
        """税額を計算する"""
        return amount * cls.TAX_RATE

    @classmethod
    def calculate_total_with_tax(cls, subtotal: float) -> float:
        """税込み合計を計算する"""
        return subtotal + cls.calculate_tax(subtotal)


class InvoiceService:
    def calculate_total(self, subtotal: float) -> float:
        return TaxCalculator.calculate_total_with_tax(subtotal)

class CartService:
    def calculate_total(self, subtotal: float) -> float:
        return TaxCalculator.calculate_total_with_tax(subtotal)
```

**コード例2: DRY違反ではないケース ── 偶然の類似**

```python
# これはDRY違反ではない！
# 見た目は似ているが、異なる「知識」を表現している

def validate_username(name: str) -> bool:
    """ユーザー名は3文字以上20文字以下（ユーザー名のビジネスルール）"""
    return 3 <= len(name) <= 20

def validate_product_name(name: str) -> bool:
    """商品名は3文字以上20文字以下（商品名のビジネスルール）"""
    return 3 <= len(name) <= 20

# 無理に共通化するとこうなる（悪い例）:
def validate_name_length(name: str, min_len: int = 3, max_len: int = 20) -> bool:
    return min_len <= len(name) <= max_len

# 問題: ユーザー名ルールが変わっても商品名ルールは変わらない
# 例: 「ユーザー名は5文字以上に変更」→ 商品名まで影響する
# これは「偶然の一致」であり、別の「知識」
```

**コード例3: DRYの適用パターン ── 定数の一元化**

```typescript
// DRY違反: 同じ値がコードの各所に散在
class UserValidator {
  validate(email: string): boolean {
    return /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/.test(email);
  }
}

class RegistrationForm {
  isValidEmail(email: string): boolean {
    return /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/.test(email);
  }
}

// DRY適用: 正規表現を1箇所で定義
const EMAIL_PATTERN = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;

function isValidEmail(email: string): boolean {
  return EMAIL_PATTERN.test(email);
}

// すべての箇所がこの関数を使う
class UserValidator {
  validate(email: string): boolean {
    return isValidEmail(email);
  }
}
```

**コード例4: DRYの適用 ── テンプレートメソッドパターン**

```python
from abc import ABC, abstractmethod
from datetime import datetime

# DRY違反: レポート生成の共通フローが各クラスに重複
class SalesReport:
    def generate(self, data):
        header = f"=== 売上レポート ===\n日付: {datetime.now()}\n"
        body = self._format_sales(data)
        footer = f"\n--- 以上 ---\n"
        return header + body + footer

class InventoryReport:
    def generate(self, data):
        header = f"=== 在庫レポート ===\n日付: {datetime.now()}\n"
        body = self._format_inventory(data)
        footer = f"\n--- 以上 ---\n"
        return header + body + footer


# DRY適用: テンプレートメソッドパターン
class BaseReport(ABC):
    """レポート生成の共通フローを定義"""

    def generate(self, data) -> str:
        header = self._build_header()
        body = self._build_body(data)
        footer = self._build_footer()
        return header + body + footer

    def _build_header(self) -> str:
        return f"=== {self.title} ===\n日付: {datetime.now()}\n"

    def _build_footer(self) -> str:
        return f"\n--- 以上 ---\n"

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def _build_body(self, data) -> str:
        pass

class SalesReport(BaseReport):
    @property
    def title(self) -> str:
        return "売上レポート"

    def _build_body(self, data) -> str:
        return "\n".join(f"  {item['name']}: {item['amount']}円" for item in data)

class InventoryReport(BaseReport):
    @property
    def title(self) -> str:
        return "在庫レポート"

    def _build_body(self, data) -> str:
        return "\n".join(f"  {item['name']}: {item['stock']}個" for item in data)
```

### 1.5 Rule of Three（三度目の法則）

DRY化のタイミングを判断する経験則として、「Rule of Three」がある。

```
  Rule of Three

  1回目の重複 → そのまま（偶然の一致かもしれない）
  2回目の重複 → メモしておく（まだ様子見）
  3回目の重複 → 共通化する（パターンが確立された）

  理由:
  ・1-2回では共通化の正しい抽象が見えない
  ・3回になれば共通パターンが明確になる
  ・早すぎる共通化は間違った抽象を生む
```

### 1.6 DRY違反の種類と対処法

```
  コード重複の分類

  ┌─────────────────────────────────────────────────┐
  │  Type 1: 完全なクローン                          │
  │  → コピペされた同一コード                        │
  │  → 対処: 関数/メソッド抽出                       │
  ├─────────────────────────────────────────────────┤
  │  Type 2: パラメータ化されたクローン              │
  │  → 定数やリテラルだけが異なるコード              │
  │  → 対処: パラメータ化して共通関数に              │
  ├─────────────────────────────────────────────────┤
  │  Type 3: 構造的なクローン                        │
  │  → 一部の文が追加/削除/変更されたコード          │
  │  → 対処: テンプレートメソッド or ストラテジー    │
  ├─────────────────────────────────────────────────┤
  │  Type 4: 意味的なクローン                        │
  │  → 異なるコードだが同じ結果を生む                │
  │  → 対処: 最もシンプルな実装に統一                │
  └─────────────────────────────────────────────────┘
```

---

## 2. KISS ── Keep It Simple, Stupid

### 2.1 定義

```
+-----------------------------------------------------------+
|  KISS (Keep It Simple, Stupid)                            |
|  ─────────────────────────────────────────────────         |
|  「シンプルさは最高の洗練である」── Leonardo da Vinci     |
|  「必要十分な最もシンプルな解法を選べ」                    |
|                                                           |
|  類似原則:                                                |
|  ・Occam's Razor: 「必要以上に仮定を増やすな」           |
|  ・UNIX哲学: 「一つのことをうまくやる」                   |
|  ・Einstein: 「できるだけ単純に。でも単純すぎないように」  |
+-----------------------------------------------------------+
```

### 2.2 WHY ── なぜシンプルさが重要なのか

```
  複雑さのコストモデル

  コスト
    ^
    |                        ####
    |                   #####
    |              #####         ← 複雑なコードの保守コスト
    |         #####
    |    #####
    |####
    |
    |  ****************************  ← シンプルなコードの保守コスト
    |****
    +------------------------------------> 時間
    初期  1月  3月  6月  1年  2年
```

### 2.3 シンプルさの測定基準

| 指標 | 測定方法 | 目安 |
|------|---------|------|
| サイクロマティック複雑度 | 分岐数を計測 | 関数あたり10以下 |
| 認知的複雑度 | ネスト深度を加味 | 関数あたり15以下 |
| 関数の行数 | 物理行を計測 | 20行以下 |
| 引数の数 | パラメータ数を計測 | 3個以下 |
| 抽象化の段数 | 呼び出し階層の深さ | 3段以下 |
| import文の数 | 依存モジュール数 | 10個以下 |

### 2.4 コード例

**コード例5: 過度に複雑な実装 vs シンプルな実装**

```javascript
// KISS違反: 過度に複雑なバリデーション
class UserValidator {
  constructor() {
    this.validationChain = new ValidationChainBuilder()
      .addValidator(new NotNullValidator())
      .addValidator(new StringLengthValidator(1, 100))
      .addValidator(new RegexValidator(/^[a-zA-Z0-9_]+$/))
      .addValidator(new BlacklistValidator(BANNED_WORDS))
      .setErrorHandler(new ValidationErrorAggregator())
      .setLocalizationProvider(new I18nValidationMessages('ja'))
      .build();
  }

  validate(username) {
    return this.validationChain.execute(
      new ValidationContext(username, 'username')
    );
  }
}

// KISS適用: シンプルで十分
function validateUsername(username) {
  if (!username || username.length === 0) {
    return { valid: false, error: 'ユーザー名は必須です' };
  }
  if (username.length > 100) {
    return { valid: false, error: 'ユーザー名は100文字以内です' };
  }
  if (!/^[a-zA-Z0-9_]+$/.test(username)) {
    return { valid: false, error: '英数字とアンダースコアのみ使用できます' };
  }
  return { valid: true, error: null };
}
```

**コード例6: シンプルなデータ変換**

```python
# KISS違反: 過度にジェネリックな変換パイプライン
class DataTransformPipeline:
    def __init__(self):
        self.transformers = []

    def add_transformer(self, transformer):
        self.transformers.append(transformer)
        return self

    def execute(self, data):
        result = data
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result

pipeline = DataTransformPipeline()
pipeline.add_transformer(StripWhitespaceTransformer())
pipeline.add_transformer(LowercaseTransformer())
pipeline.add_transformer(RemoveSpecialCharsTransformer())
result = pipeline.execute(user_input)


# KISS適用: 直接的で明快
def normalize_input(text: str) -> str:
    """入力テキストを正規化する"""
    return text.strip().lower().replace('-', '').replace('.', '')
```

**コード例7: KISS適用 ── 設定管理**

```python
# KISS違反: 多層の抽象化を持つ設定管理
class ConfigurationManager:
    def __init__(self):
        self._providers = []
        self._cache = {}
        self._observers = []
        self._encryption_service = EncryptionService()

    def register_provider(self, provider):
        self._providers.append(provider)

    def get(self, key: str, default=None):
        if key in self._cache:
            return self._cache[key]
        for provider in reversed(self._providers):
            value = provider.get(key)
            if value is not None:
                if self._is_encrypted(key):
                    value = self._encryption_service.decrypt(value)
                self._cache[key] = value
                self._notify_observers(key, value)
                return value
        return default


# KISS適用: 必要十分なシンプルさ
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    """アプリケーション設定（イミュータブル）"""
    database_url: str
    api_key: str
    debug: bool = False
    max_connections: int = 10

    @classmethod
    def from_env(cls) -> "AppConfig":
        """環境変数から設定を読み込む"""
        return cls(
            database_url=os.environ["DATABASE_URL"],
            api_key=os.environ["API_KEY"],
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            max_connections=int(os.environ.get("MAX_CONNECTIONS", "10")),
        )
```

### 2.5 Simplicity vs Simplistic

| Simplicity（良い単純さ） | Simplistic（悪い単純さ） |
|------------------------|------------------------|
| 複雑な問題を明快に解決 | 問題を無視して単純化 |
| 適切な抽象化で整理 | 必要な抽象化を省略 |
| エッジケースを考慮 | エッジケースを無視 |
| エラーハンドリング適切 | エラーハンドリング不足 |
| テスト可能な構造 | テスト不可能な構造 |

---

## 3. YAGNI ── You Aren't Gonna Need It

### 3.1 定義

```
+-----------------------------------------------------------+
|  YAGNI (You Aren't Gonna Need It)                         |
|  ─────────────────────────────────────────────────         |
|  「実際に必要になるまで、その機能を実装するな」            |
|                    ── Ron Jeffries (XP共同創始者)          |
|                                                           |
|  「今必要でない機能に費やすコスト:                         |
|    実装コスト + テストコスト + 保守コスト                  |
|    + 読解コスト + 結局使わないリスク                       |
|    + 実際のニーズとのズレリスク」                         |
+-----------------------------------------------------------+
```

### 3.2 WHY ── 不要な先行実装のコスト

```
  YAGNI違反の実際のコスト

  機能A: 今必要 → 実装 → 使われる → コスト回収

  機能B: 将来必要かも → 実装 → 結局使われない
    ├── 実装コスト: 3日
    ├── テスト作成: 1日
    ├── レビュー: 0.5日
    ├── ドキュメント: 0.5日
    ├── 保守コスト: 0.5日/月 x 12ヶ月 = 6日
    └── 合計: 11日分の工数が無駄に

  機能C: 将来必要かも → 実装 → 実際のニーズと異なる
    ├── 上記の11日 + リファクタリング: 5日
    └── 合計: 16日分の工数が無駄に
```

### 3.3 コード例

**コード例8: 過度な先行実装 vs 必要十分な実装**

```typescript
// YAGNI違反: 現時点で不要な拡張ポイントを大量に用意
interface LogTransport {
  send(entry: LogEntry): Promise<void>;
}
interface LogFormatter {
  format(entry: LogEntry): string;
}
interface LogFilter {
  shouldLog(entry: LogEntry): boolean;
}

class Logger {
  private transports: LogTransport[] = [];
  private formatters: Map<string, LogFormatter> = new Map();
  private filters: LogFilter[] = [];
  private bufferSize: number;
  private flushInterval: number;
  private retryPolicy: RetryPolicy;
  private encryptionProvider?: EncryptionProvider;
  // 実際に使うのはコンソール出力だけなのに...
}


// YAGNI適用: 今必要なものだけ実装
enum LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 }

class Logger {
  constructor(private level: LogLevel = LogLevel.INFO) {}

  info(message: string): void {
    if (this.level <= LogLevel.INFO) {
      console.log(`[INFO] ${new Date().toISOString()} ${message}`);
    }
  }

  error(message: string, error?: Error): void {
    if (this.level <= LogLevel.ERROR) {
      console.error(`[ERROR] ${new Date().toISOString()} ${message}`, error);
    }
  }

  warn(message: string): void {
    if (this.level <= LogLevel.WARN) {
      console.warn(`[WARN] ${new Date().toISOString()} ${message}`);
    }
  }
  // ファイル出力が必要になったら、その時に拡張する
}
```

**コード例9: YAGNI適用 ── APIレスポンス**

```python
import json
from dataclasses import dataclass

# YAGNI違反: あらゆるフォーマットに先行対応
class ApiResponse:
    def __init__(self, data, status=200):
        self.data = data
        self.status = status
        self._formatters = {
            'json': JsonFormatter(),
            'xml': XmlFormatter(),
            'csv': CsvFormatter(),
            'yaml': YamlFormatter(),
            'msgpack': MsgpackFormatter(),
        }

    def to_format(self, format_type: str) -> bytes:
        return self._formatters[format_type].format(self.data)


# YAGNI適用: 今はJSONのみ
@dataclass
class ApiResponse:
    data: dict
    status: int = 200

    def to_json(self) -> str:
        return json.dumps({
            'status': self.status,
            'data': self.data
        }, ensure_ascii=False)
# XMLやCSVが必要になったらその時に追加する
```

### 3.4 YAGNIの例外 ── 先行投資すべきケース

| 先行投資が正当なケース | 理由 |
|---------------------|------|
| セキュリティ対策 | 後から追加は困難。最初から組み込む |
| データベーススキーマ設計 | マイグレーションコストが高い |
| APIの公開インターフェース | 後方互換性の維持が必要 |
| ロギング基盤 | 後から追加すると全コードに影響 |
| テスト基盤 | テスト可能な設計は最初から |
| i18n（国際化）基盤 | 文字列のハードコードは後から修正困難 |

---

## 4. 3原則の相互関係と矛盾

### 4.1 相互関係図

```
          DRY                KISS               YAGNI
     「重複するな」     「シンプルに」      「今不要なら作るな」
          |                  |                   |
          +--------+---------+--------+----------+
                   |                  |
           矛盾が発生する場面       調和する場面
                   |                  |
      DRY追求 → 過度な抽象化    3原則すべてが
      → KISS違反になりうる      同じ方向を向く場面:
                                「シンプルに、重複なく、
      KISS追求 → 重複を許容      必要なものだけ」
      → DRY違反になりうる

      DRY追求 → 将来の再利用を
      見越した抽象化
      → YAGNI違反になりうる
```

### 4.2 矛盾の解決ガイド

| 状況 | 優先すべき原則 | 理由 |
|------|---------------|------|
| 2箇所の重複、変更頻度低い | KISS > DRY | 抽象化コストに見合わない |
| 3箇所以上の重複 | DRY > KISS | 変更漏れのリスクが高い |
| 将来の拡張のための抽象化 | YAGNI > OCP | 不確実な未来に投資しない |
| ビジネスルールの重複 | DRY > YAGNI | ルール変更時の一貫性が重要 |
| 小規模スクリプトの重複 | KISS > DRY | 抽象化よりもコピペが明快 |
| テストコードの重複 | KISS > DRY | テストの可読性を優先 |
| セキュリティ関連の機能 | 安全側に倒す | YAGNIは安全性に適用しない |

### 4.3 判断フローチャート

```
  3原則の判断フロー

  重複を発見
  │
  ├── 同じ「知識」を表現しているか？
  │   ├── No → 放置（偶然の一致）
  │   └── Yes → 何箇所の重複か？
  │       ├── 2箇所 → 変更頻度は？
  │       │   ├── 高い → DRY化する
  │       │   └── 低い → Rule of Three（3箇所目を待つ）
  │       └── 3箇所以上 → DRY化する
  │
  ├── DRY化の方法を検討
  │   ├── シンプルに関数/定数に抽出可能？
  │   │   └── Yes → 抽出する（KISS準拠）
  │   └── 複雑な抽象化が必要？
  │       ├── 今の要件で必要？ → 実装する
  │       └── 将来の要件のため？ → YAGNI（今は見送り）
```

---

## 5. 実践的な判断フロー

| 判断ポイント | 質問 | Yes → | No → |
|-------------|------|-------|------|
| 重複発見 | 同じ「知識」か？ | DRY化を検討 | 放置（偶然の一致） |
| DRY化検討 | 3箇所以上？ | 共通化する | 2箇所ならRule of Three |
| 共通化方法 | シンプルに抽出可能？ | 関数/定数に抽出 | 設計パターン適用を検討 |
| 新機能要求 | 今スプリントで必要？ | 実装する | YAGNI（後回し） |
| 実装方法 | 最もシンプルな方法で動く？ | その方法で実装 | さらにシンプルにできないか検討 |
| 抽象化検討 | 具体的なユースケースが3つ以上？ | 抽象化する | 具体的な実装のまま |

---

## 6. レイヤー間のDRY

### 6.1 フロントエンド/バックエンド間の重複

```python
# レイヤー間DRY違反: バリデーションルールが分散

# 改善: 単一の定義から各層のルールを生成
AGE_MIN = 0
AGE_MAX = 150

def validate_age(age: int) -> bool:
    return AGE_MIN <= age <= AGE_MAX

def get_age_schema() -> dict:
    """JSON Schemaとしてフロントエンドと共有"""
    return {
        "type": "integer",
        "minimum": AGE_MIN,
        "maximum": AGE_MAX
    }

def get_age_constraint_sql() -> str:
    """DB制約として共有"""
    return f"CHECK (age >= {AGE_MIN} AND age <= {AGE_MAX})"
```

### 6.2 マイクロサービス間のDRY

```
  マイクロサービスでのDRY判断

  サービス間の共有:
  ┌─────────────────────────────────────────────────┐
  │ 共有すべきもの           │ 共有すべきでないもの   │
  ├─────────────────────────┼─────────────────────┤
  │ APIコントラクト(OpenAPI) │ ビジネスロジック      │
  │ イベントスキーマ         │ データベーススキーマ   │
  │ 認証トークン形式         │ 内部実装詳細          │
  │ 共通ドメイン型          │ ユーティリティ関数    │
  └─────────────────────────┴─────────────────────┘

  原則: サービスの独立性 > DRY
  → 多少の重複は許容し、独立デプロイ可能性を維持
```

---

## 7. アンチパターン

### アンチパターン1: WET（Write Everything Twice）コード

```python
# NG: 同じバリデーションロジックが微妙に異なるパターンで存在
def validate_email_frontend(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_email_backend(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))  # 微妙に違う！

# OK: 単一の定義
EMAIL_PATTERN = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def validate_email(email: str) -> bool:
    """メールアドレスの形式を検証する（唯一の正規表現定義）"""
    return bool(re.match(EMAIL_PATTERN, email))
```

### アンチパターン2: Speculative Generality（投機的汎用性）

```java
// NG: 使われないフレームワークを先に作る
public interface DataExporter<T, F extends ExportFormat, C extends ExportConfig> {
    ExportResult<T> export(Collection<T> data, F format, C config);
    void registerPlugin(ExportPlugin<T> plugin);
    void setMiddleware(ExportMiddleware<T>... middlewares);
}
// 実際に必要なのは「CSVでユーザー一覧を出力する」だけ

// OK: 必要最小限
public class UserCsvExporter {
    public String export(List<User> users) {
        StringBuilder csv = new StringBuilder("name,email\n");
        for (User u : users) {
            csv.append(u.getName()).append(",").append(u.getEmail()).append("\n");
        }
        return csv.toString();
    }
}
```

### アンチパターン3: DRY原理主義（Wrong Abstraction）

```python
# NG: 異なるコンテキストの偶然の類似を無理に共通化
class GenericProcessor:
    def process(self, type: str, data: dict) -> dict:
        if type == 'user_registration':
            validated = self._validate(data, USER_RULES)
            result = self._save(data, 'users')
            self._notify(data['email'], 'welcome')
        elif type == 'email_campaign':
            validated = self._validate(data, CAMPAIGN_RULES)
            result = self._save(data, 'campaigns')
            self._notify(data['recipients'], 'campaign')
        return result

# OK: 各コンテキストは独立
class UserRegistrationService:
    def register(self, user_data: dict) -> User:
        self.validator.validate(user_data)
        user = self.repository.save(user_data)
        self.mailer.send_welcome(user.email)
        return user

class EmailCampaignService:
    def launch(self, campaign_data: dict) -> Campaign:
        self.validator.validate(campaign_data)
        campaign = self.repository.save(campaign_data)
        self.mailer.send_campaign(campaign.recipients)
        return campaign
```

Sandi Metz の名言:

> 「間違った抽象化よりも重複のほうがマシである」
> ("Duplication is far cheaper than the wrong abstraction")

---

## 8. 実践演習

### 演習1（基礎）: DRY違反の検出と修正

以下のコードからDRY違反を特定し、適切に共通化せよ。

```python
class OrderService:
    def calculate_domestic_shipping(self, weight: float) -> float:
        if weight <= 1.0:
            base = 500
        elif weight <= 5.0:
            base = 800
        elif weight <= 10.0:
            base = 1200
        else:
            base = 1200 + (weight - 10) * 100
        tax = base * 0.10
        return base + tax

    def calculate_express_shipping(self, weight: float) -> float:
        if weight <= 1.0:
            base = 500
        elif weight <= 5.0:
            base = 800
        elif weight <= 10.0:
            base = 1200
        else:
            base = 1200 + (weight - 10) * 100
        express_surcharge = base * 0.50
        base = base + express_surcharge
        tax = base * 0.10
        return base + tax
```

**期待される出力例:**

```python
TAX_RATE = 0.10
EXPRESS_SURCHARGE_RATE = 0.50

def _calculate_base_shipping(weight: float) -> float:
    """重量に基づく基本送料を計算する"""
    if weight <= 1.0:
        return 500.0
    elif weight <= 5.0:
        return 800.0
    elif weight <= 10.0:
        return 1200.0
    else:
        return 1200.0 + (weight - 10.0) * 100.0

def _apply_tax(amount: float) -> float:
    return amount * (1 + TAX_RATE)

class OrderService:
    def calculate_domestic_shipping(self, weight: float) -> float:
        base = _calculate_base_shipping(weight)
        return _apply_tax(base)

    def calculate_express_shipping(self, weight: float) -> float:
        base = _calculate_base_shipping(weight)
        base_with_surcharge = base * (1 + EXPRESS_SURCHARGE_RATE)
        return _apply_tax(base_with_surcharge)
```

### 演習2（応用）: DRY vs KISS の判断

以下の2パターンのうちどちらが適切か判断し理由を述べよ。

**パターンA（DRY重視）:**
```python
def format_entity(entity: dict, entity_type: str) -> str:
    template = TEMPLATES[entity_type]
    fields = FIELD_MAPPINGS[entity_type]
    result = template['header']
    for field in fields:
        result += f"  {field['label']}: {entity.get(field['key'], 'N/A')}\n"
    result += template['footer']
    return result
```

**パターンB（KISS重視）:**
```python
def format_user(user: dict) -> str:
    return f"名前: {user.get('name', 'N/A')}\nメール: {user.get('email', 'N/A')}"

def format_product(product: dict) -> str:
    return f"商品名: {product.get('name', 'N/A')}\n価格: {product.get('price', 'N/A')}円"
```

**期待される分析:** パターンBが適切。各フォーマットは異なる「知識」を表現しており、偶然の構造的類似。パターンAはテンプレート設定の管理が複雑化し、KISS違反。

### 演習3（発展）: 3原則のバランス設計

ECサイトの商品検索機能を設計せよ。

現在の要件: 商品名の部分一致検索のみ
将来の可能性: カテゴリ絞り込み、価格範囲、ソート、ページネーション

**期待される出力例:**

```python
from dataclasses import dataclass

@dataclass
class SearchQuery:
    """検索パラメータ（現在は名前のみ、将来フィールド追加可能）"""
    name: str

class ProductRepository:
    def search(self, query: SearchQuery) -> list:
        return self.db.query(
            "SELECT * FROM products WHERE name LIKE %s",
            (f"%{query.name}%",)
        )

class ProductSearchService:
    def __init__(self, repository: ProductRepository):
        self.repository = repository

    def search(self, name: str) -> list:
        query = SearchQuery(name=name)
        return self.repository.search(query)
```

設計判断の根拠:
- **YAGNI**: 今はname検索のみ実装。ページネーション等は後回し
- **DIP**: リポジトリ層を分離（テスト容易性確保）
- **KISS**: SearchQueryは将来フィールド追加可能だが今はシンプル


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---

## 9. FAQ

### Q1: DRYを徹底すると、かえってコードが複雑にならないか？

その通り。DRYは「知識の重複排除」であり「コードの文字列的な重複排除」ではない。異なるコンテキストの偶然の類似を無理に共通化すると、不自然な結合が生まれKISS違反になる。**Rule of Three**（3回目の重複で共通化）が実践的なガイドライン。

### Q2: YAGNIに従うと、後から大きな設計変更が必要にならないか？

YAGNI は「設計を考えるな」ではなく「実装を先延ばしにせよ」。クリーンな設計（低結合・高凝集）を保っていれば、後からの拡張は容易。不要な先行実装は、実際のニーズとずれた設計を固定化するリスクがある。

### Q3: KISSの「シンプル」は主観的ではないか？

ある程度は主観的だが、客観的指標がある: サイクロマティック複雑度、認知的複雑度、依存関係の数、抽象化の段数、「関数名だけで動作が予想できるか」テスト。

### Q4: テストコードにもDRYを適用すべきか？

テストコードでは**DRYよりもKISS（可読性）を優先**。テストは仕様書として読まれるため自己完結的であるべき。ただしテストデータ生成やモック設定は共通化してよい。

### Q5: マイクロサービスでのDRYはどう考えるべきか？

サービスの独立性 > DRY。サービス間の共有ライブラリはカップリングを生む。多少の重複を許容し、独立デプロイ可能性を維持する。

### Q6: 3原則が互いに矛盾する場合、どのように優先順位を決めるべきか？

3原則が矛盾するケースは実務では頻繁に発生する。一般的な優先順位は以下の通り。

1. **KISS > DRY**: 共通化によってコードが複雑になるなら、多少の重複を許容する。Sandi Metz の「間違った抽象化よりも重複のほうがマシ」が判断基準。
2. **YAGNI > DRY**: 将来の重複を予測して先に抽象化を作るのは避ける。実際に3回目の重複が発生してから共通化する。
3. **KISS > YAGNI**: シンプルさの維持と将来の拡張性が矛盾する場合は稀だが、「拡張ポイントを設けること自体がシンプルさを損なう」場合は拡張ポイントを作らない。

ただし、これらは機械的に適用するルールではない。最終的な判断基準は「**6ヶ月後にこのコードを読む開発者が、最も短時間で理解・変更できるのはどの選択か**」である。

```
  判断フローチャート:

  重複を発見
    │
    ├─ 3回以上出現しているか？ ─No─→ そのまま放置（YAGNI）
    │
   Yes
    │
    ├─ 同じ「知識」を表現しているか？ ─No─→ 偶然の一致、別々に保つ（KISS）
    │
   Yes
    │
    ├─ シンプルに共通化できるか？ ─No─→ 共通化を見送る（KISS > DRY）
    │
   Yes
    │
    └─→ 共通化する（DRY適用）
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 原則 | 一言 | 適用のコツ | 行き過ぎの兆候 |
|------|------|-----------|---------------|
| DRY | 知識を一元化 | Rule of Three | 不自然な抽象化 |
| KISS | シンプルに保つ | 最も直接的な方法を選ぶ | 機能不足 |
| YAGNI | 今必要なものだけ | 要件駆動で実装 | 拡張困難な設計 |

### 3原則の適用チェックリスト

| チェック項目 | 原則 |
|------------|------|
| この重複は同じ「知識」を表現しているか？ | DRY |
| 共通化は最もシンプルな方法で実現できるか？ | KISS |
| この抽象化は今のユースケースで必要か？ | YAGNI |
| 3箇所以上で使われる共通パターンか？ | Rule of Three |
| この設計変更で可読性は向上するか？ | KISS |
| 将来の要件ではなく今の要件に基づいているか？ | YAGNI |

---

## 次に読むべきガイド

- [結合度と凝集度](./03-coupling-cohesion.md) ── DRYとKISSを支えるモジュール設計原則
- [SOLID原則](./01-solid.md) ── 特にOCPとDRYの関係
- [関数設計](../01-practices/01-functions.md) ── KISS原則を関数レベルで実践する
- [リファクタリング技法](../02-refactoring/01-refactoring-techniques.md) ── DRY化のための具体的技法
- [コードスメル](../02-refactoring/00-code-smells.md) ── 重複やコードの複雑さの検出
- デザインパターン: Behavioral ── StrategyやTemplateMethodによるDRY化

---

## 参考文献

1. **Andrew Hunt, David Thomas** 『The Pragmatic Programmer: Your Journey to Mastery』 Addison-Wesley, 2019 (20th Anniversary Edition)
2. **Kent Beck** 『Extreme Programming Explained: Embrace Change』 Addison-Wesley, 2004 (2nd Edition)
3. **Sandi Metz** 『Practical Object-Oriented Design: An Agile Primer Using Ruby』 Addison-Wesley, 2018 (2nd Edition)
4. **John Ousterhout** 『A Philosophy of Software Design』 Yaknyam Press, 2018
5. **Sandi Metz** "The Wrong Abstraction" (blog post, 2016) ── DRYの過剰適用に関する重要な議論
6. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018
7. **Ron Jeffries** "You're NOT Gonna Need It!" (XP Magazine, 1998) ── YAGNIの原典
8. **Donald Knuth** "Structured Programming with go to Statements" Computing Surveys, 1974 ── 「早すぎる最適化」の原典
