# DRY / KISS / YAGNI ── 重複排除・単純化・不要機能の排除

> ソフトウェア開発における3つの基本原則。DRYは知識の重複を排除し、KISSは複雑さを避け、YAGNIは不要な先行実装を防ぐ。この3原則の適切なバランスが、保守しやすいコードを生む。

---

## この章で学ぶこと

1. **DRY原則の正しい理解** ── 単なるコード重複排除ではなく「知識の一元化」を理解する
2. **KISSの実践方法** ── シンプルさを保ちながら要件を満たす設計を身につける
3. **YAGNIの判断基準** ── 先行投資すべきケースと不要なケースを見極める

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

### 1.2 DRYの対象

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
     └──────────────────────────────┘
     ※ コードの見た目が同じでも、
       表現している「知識」が異なれば重複ではない
```

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
    TAX_RATE = 0.10

    @classmethod
    def calculate_tax(cls, amount: float) -> float:
        return amount * cls.TAX_RATE

    @classmethod
    def calculate_total_with_tax(cls, subtotal: float) -> float:
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
    """ユーザー名は3文字以上20文字以下"""
    return 3 <= len(name) <= 20

def validate_product_name(name: str) -> bool:
    """商品名は3文字以上20文字以下"""
    return 3 <= len(name) <= 20

# 無理に共通化すると、一方の変更が他方に影響する
# ユーザー名ルールが変わっても商品名ルールは変わらない
# → これは偶然の一致であり、別の「知識」
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
+-----------------------------------------------------------+
```

**コード例3: 過度に複雑な実装 vs シンプルな実装**

```javascript
// KISS違反: 過度に複雑
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

**コード例4: シンプルなデータ変換**

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
|    + 読解コスト + 結局使わないリスク」                     |
+-----------------------------------------------------------+
```

**コード例5: 過度な先行実装 vs 必要十分な実装**

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
  // ... 実際に使うのはコンソール出力だけなのに

  constructor(config: LoggerConfig) {
    // 100行の初期化コード...
  }
}

// YAGNI適用: 今必要なものだけ実装
class Logger {
  private level: LogLevel;

  constructor(level: LogLevel = LogLevel.INFO) {
    this.level = level;
  }

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
  // ファイル出力が必要になったら、その時に拡張する
}
```

---

## 4. 3原則の相互関係と矛盾

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
                                  必要なものだけ」
```

### 矛盾の解決ガイド

| 状況 | 優先すべき原則 | 理由 |
|------|---------------|------|
| 2箇所の重複、変更頻度低い | KISS > DRY | 抽象化コストに見合わない |
| 3箇所以上の重複 | DRY > KISS | 変更漏れのリスクが高い |
| 将来の拡張のための抽象化 | YAGNI > OCP | 不確実な未来に投資しない |
| ビジネスルールの重複 | DRY > YAGNI | ルール変更時の一貫性が重要 |

---

## 5. 実践的な判断フロー

| 判断ポイント | 質問 | Yes → | No → |
|-------------|------|-------|------|
| 重複発見 | 同じ「知識」か？ | DRY化を検討 | 放置（偶然の一致） |
| DRY化検討 | 3箇所以上？ | 共通化する | 2箇所ならRule of Three |
| 共通化方法 | シンプルに抽出可能？ | 関数/定数に抽出 | 設計パターン適用を検討 |
| 新機能要求 | 今スプリントで必要？ | 実装する | YAGNI（後回し） |
| 実装方法 | 最もシンプルな方法で動く？ | その方法で実装 | さらにシンプルにできないか検討 |

---

## 6. アンチパターン

### アンチパターン1: WET（Write Everything Twice）コード

```python
# アンチパターン: 同じバリデーションロジックがフロント/バック/DBに分散
# フロントエンド
def validate_email_frontend(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

# バックエンド（微妙に異なるパターン！）
def validate_email_backend(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return bool(re.match(pattern, email))

# DB制約（さらに異なるルール！）
# CHECK (email ~ '^[^@]+@[^@]+\.[^@]+$')

# 改善: 単一の定義から各層のバリデーションを生成
EMAIL_PATTERN = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'

def validate_email(email: str) -> bool:
    return bool(re.match(EMAIL_PATTERN, email))
```

### アンチパターン2: Speculative Generality（投機的汎用性）

```java
// アンチパターン: 使われないフレームワークを先に作る
public interface DataExporter<T, F extends ExportFormat, C extends ExportConfig> {
    ExportResult<T> export(Collection<T> data, F format, C config);
    void registerPlugin(ExportPlugin<T> plugin);
    void setMiddleware(ExportMiddleware<T>... middlewares);
}
// 実際に必要なのは「CSVでユーザー一覧を出力する」だけ

// 改善: 必要最小限
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

---

## 7. FAQ

### Q1: DRYを徹底すると、かえってコードが複雑にならないか？

その通り。DRYは「知識の重複排除」であり「コードの文字列的な重複排除」ではない。異なるコンテキストの偶然の類似を無理に共通化すると、不自然な結合が生まれKISS違反になる。**Rule of Three**（3回目の重複で共通化）が実践的なガイドライン。

### Q2: YAGNIに従うと、後から大きな設計変更が必要にならないか？

YAGNI は「設計を考えるな」ではなく「実装を先延ばしにせよ」。クリーンな設計（低結合・高凝集）を保っていれば、後からの拡張は容易になる。逆に不要な先行実装は、実際のニーズとずれた設計を固定化するリスクがある。

### Q3: KISSの「シンプル」は主観的ではないか？

ある程度は主観的だが、以下の客観的指標がある:
- **サイクロマティック複雑度**: 分岐数を計測
- **依存関係の数**: importの数
- **抽象化の段数**: 処理を追うために何段のジャンプが必要か
- **名前から推測できるか**: 関数名だけで動作が予想できるか

---

## まとめ

| 原則 | 一言 | 適用のコツ | 行き過ぎの兆候 |
|------|------|-----------|---------------|
| DRY | 知識を一元化 | Rule of Three | 不自然な抽象化 |
| KISS | シンプルに保つ | 最も直接的な方法を選ぶ | 機能不足 |
| YAGNI | 今必要なものだけ | 要件駆動で実装 | 拡張困難な設計 |

---

## 次に読むべきガイド

- [結合度と凝集度](./03-coupling-cohesion.md) ── DRYとKISSを支えるモジュール設計原則
- [関数設計](../01-practices/01-functions.md) ── KISS原則を関数レベルで実践する
- [リファクタリング技法](../02-refactoring/01-refactoring-techniques.md) ── DRY化のための具体的技法

---

## 参考文献

1. **Andrew Hunt, David Thomas** 『The Pragmatic Programmer: Your Journey to Mastery』 Addison-Wesley, 2019 (20th Anniversary Edition)
2. **Kent Beck** 『Extreme Programming Explained: Embrace Change』 Addison-Wesley, 2004 (2nd Edition)
3. **Sandi Metz** 『Practical Object-Oriented Design: An Agile Primer Using Ruby』 Addison-Wesley, 2018 (2nd Edition)
4. **John Ousterhout** 『A Philosophy of Software Design』 Yaknyam Press, 2018
