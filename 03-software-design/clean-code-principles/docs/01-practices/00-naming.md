# 命名規則 ── 変数・関数・クラスの命名術

> コードは書く時間の10倍読まれる。良い名前は最高のドキュメントであり、悪い名前は最悪の技術的負債である。命名はプログラミングで最も重要かつ最も困難なスキルの一つ。

---

## この章で学ぶこと

1. **命名の基本原則** ── 意図を明確に伝える名前の付け方を理解する
2. **要素別の命名規則** ── 変数・関数・クラス・定数それぞれの命名パターンを身につける
3. **命名のアンチパターン** ── 避けるべき命名習慣と改善方法を把握する
4. **認知科学から見た命名** ── 人間の記憶と認知の観点から良い名前の条件を理解する
5. **チーム開発における命名戦略** ── 命名規約の策定と自動強制の方法を習得する

---

## 前提知識

このガイドを最大限に活用するには、以下の知識が必要です。

| 前提知識 | 必要レベル | 参照先 |
|---------|----------|--------|
| クリーンコードの概要 | 読了推奨 | [クリーンコード概論](../00-principles/00-clean-code-overview.md) |
| 1つ以上のプログラミング言語 | 基本的なコーディング経験 | -- |
| IDEの基本操作 | リネーム機能を使える | -- |

---

## 1. 命名の基本原則

### 1.1 なぜ命名が重要か

Robert C. Martin は「プログラマーの仕事時間の70%はコードを読むことに費やされる」と述べている。つまり、良い名前は**チーム全体の生産性に直結する投資**である。

```
+-----------------------------------------------------------+
|  良い名前の3条件                                          |
|  ─────────────────────────────────────                    |
|  1. 意図が明確 (Intention-Revealing)                      |
|     → 何のために存在するか分かる                          |
|  2. 発音可能 (Pronounceable)                              |
|     → チームで口頭議論できる                              |
|  3. 検索可能 (Searchable)                                 |
|     → IDE/grepで見つけられる                              |
+-----------------------------------------------------------+
```

### 1.2 認知科学から見た命名

人間の短期記憶（ワーキングメモリ）は7±2チャンク（Miller's Law）しか保持できない。良い名前は「1つのチャンク」で意味を伝え、読み手の認知負荷を最小化する。

```
  名前の良さと文脈理解にかかる時間

  理解時間
    ^
    |  ***
    |     ***
    |        ***
    |           ****
    |               *****
    |                    ********
    +------------------------------> 名前の質
     d  data  val  userData  activeUserList
     ↑                              ↑
   即座に理解不可能            即座に理解可能
```

| 命名の質 | 認知コスト | 例 | 読み手の反応 |
|---------|----------|-----|------------|
| 暗号的 | 極めて高い | `d`, `x2`, `tmp` | 「何これ？コード全体を読まないと分からない」 |
| 曖昧 | 高い | `data`, `info`, `result` | 「何のdata？」 |
| 具体的 | 低い | `userAge`, `orderTotal` | 「ユーザーの年齢ね」 |
| 自己説明的 | 最小 | `activeUserCount`, `isEmailVerified` | 即座に理解 |

**命名が悪いコードの読解プロセス:**

```
  悪い命名:
  p = g(d, f)

  読み手の思考プロセス:
  1. pって何？ → コンテキストを探す（20秒）
  2. gって何の関数？ → 関数定義を見に行く（30秒）
  3. dとfは何のデータ？ → 呼び出し元を確認（20秒）
  → 合計: 70秒+ で1行の理解

  良い命名:
  discountedPrice = calculateDiscount(originalPrice, discountRate)

  読み手の思考プロセス:
  1. 一目で理解 → 「元の価格に割引率を適用して割引価格を計算する」
  → 合計: 3秒 で1行の理解
```

### 1.3 命名の5つの基本ルール

| ルール | 説明 | 悪い例 | 良い例 |
|-------|------|--------|--------|
| 1. 意図を表す | 何のために存在するか | `d` | `elapsedDays` |
| 2. 誤解を招かない | 読み手が間違った推測をしない | `accountList`（Setなのに） | `accounts` or `accountSet` |
| 3. 区別がつく | 似た名前で混乱しない | `product` vs `productData` | `product` vs `productDetail` |
| 4. 発音可能 | チームで議論できる | `genymdhms` | `generationTimestamp` |
| 5. 検索可能 | IDEで一意に検索できる | `e`, `t`, `MAX` | `maxRetryCount` |

**コード例1: 意図を明確にする命名**

```python
# === 悪い命名: 何のデータか分からない ===
d = 86400
l = get_list()
for i in l:
    if i.s == 1:
        process(i)

# 読み手の疑問:
# - d は何の数値？ → 1日の秒数？定数？
# - l は何のリスト？ → ユーザー？注文？
# - i.s は何？ → status? score? size?
# - 1 は何を意味する？ → アクティブ？完了？


# === 良い命名: コードが自己説明的 ===
SECONDS_PER_DAY = 86400
active_users = get_active_users()
for user in active_users:
    if user.status == UserStatus.ACTIVE:
        send_notification(user)

# 読み手は即座に理解:
# - 1日の秒数を定数として定義
# - アクティブユーザーのリストを取得
# - 各ユーザーのステータスがアクティブなら通知を送信
```

**コード例2: スコープに応じた名前の長さ**

```python
# ============================================================
# ルール: スコープが広いほど名前は長く、狭いほど短くてよい
# ============================================================

# ループ変数（短いスコープ）: 短い名前でOK
for i in range(10):
    matrix[i][i] = 1

# ただし意味がある場合は明示する
for row_index in range(height):
    for col_index in range(width):
        grid[row_index][col_index] = calculate_cell(row_index, col_index)

# モジュールレベル定数（長いスコープ）: 長くて具体的に
MAX_LOGIN_ATTEMPTS_BEFORE_LOCKOUT = 5
DEFAULT_SESSION_TIMEOUT_MINUTES = 30
MINIMUM_PASSWORD_LENGTH = 8

# クラスメンバー（中程度のスコープ）: 適度な長さ
class UserService:
    def __init__(self):
        self.retry_count = 0            # クラス内でのコンテキストがある
        self.last_login_timestamp = None  # クラス名が文脈を提供

# グローバルに近い変数: 最も具体的に
user_session_timeout_seconds = 1800
database_connection_pool_max_size = 20
```

```
  スコープと名前の長さの関係

  名前の長さ
    ^
    |                          ★ グローバル定数
    |                    ★ クラスフィールド
    |              ★ メソッド引数
    |        ★ ローカル変数
    |  ★ ループ変数
    +──────────────────────────→ スコープの広さ
    狭い                       広い

  例:
  i (ループ)
  total (ローカル)
  order_count (メソッド引数)
  active_user_count (クラスフィールド)
  MAX_LOGIN_ATTEMPTS_BEFORE_LOCKOUT (グローバル定数)
```

---

## 2. 要素別の命名規則

### 2.1 変数名

```
  ┌────────────────────────────────────────────────┐
  │ 変数命名のガイドライン                          │
  ├──────────┬─────────────────────────────────────┤
  │ bool     │ is/has/can/should + 形容詞/過去分詞 │
  │          │ isActive, hasPermission, canEdit     │
  ├──────────┼─────────────────────────────────────┤
  │ 数値     │ 単位を含める                         │
  │          │ timeoutMs, fileSizeBytes, ageYears   │
  ├──────────┼─────────────────────────────────────┤
  │ コレクション│ 複数形 or xxxList/xxxMap           │
  │          │ users, orderItems, nameToEmail       │
  ├──────────┼─────────────────────────────────────┤
  │ 一時変数 │ 用途を示す                           │
  │          │ tempFile, swapValue, accumulator     │
  ├──────────┼─────────────────────────────────────┤
  │ Optional │ maybe/optional + 名詞               │
  │          │ maybeUser, optionalAddress           │
  ├──────────┼─────────────────────────────────────┤
  │ 日時     │ 種類 + At/On/Since                  │
  │          │ createdAt, publishedOn, activeSince  │
  └──────────┴─────────────────────────────────────┘
```

**コード例3: ブール変数の命名**

```typescript
// === 悪い: trueの意味が不明 ===
let flag = true;
let check = false;
let status = true;
let enable = false;
let login = true;

// === 良い: true/falseの意味が明確 ===
let isVisible = true;
let hasAdminPermission = false;
let shouldAutoSave = true;
let canDeletePost = user.role === 'admin';
let wasProcessed = order.processedAt !== null;
let isEmailVerified = !!user.emailVerifiedAt;
let hasExceededLimit = currentCount > MAX_ALLOWED;

// ブール変数命名のベストプラクティス
// | プレフィックス | 意味           | 例                     |
// |-------------|---------------|------------------------|
// | is          | 状態である     | isActive, isLoading    |
// | has         | 持っている     | hasError, hasChildren  |
// | can         | 可能である     | canEdit, canDelete     |
// | should      | すべきである   | shouldUpdate, shouldRetry |
// | was         | 過去の状態     | wasDeleted, wasNotified |
// | will        | 未来の状態     | willExpire, willRetry  |
```

**コード例4: 数値変数の単位を含める**

```python
# === 悪い: 単位が不明 ===
timeout = 5000     # ミリ秒？秒？分？
file_size = 1024   # バイト？KB？MB？
distance = 100     # メートル？キロメートル？マイル？
age = 25           # 年？月？日？

# === 良い: 単位を名前に含める ===
timeout_ms = 5000
timeout_seconds = 5
file_size_bytes = 1048576
file_size_mb = 1.0
distance_km = 100.0
age_years = 25

# === さらに良い: 型で単位を表現（型安全） ===
from dataclasses import dataclass

@dataclass
class Duration:
    milliseconds: int

    @classmethod
    def from_seconds(cls, seconds: int) -> 'Duration':
        return cls(milliseconds=seconds * 1000)

    @classmethod
    def from_minutes(cls, minutes: int) -> 'Duration':
        return cls(milliseconds=minutes * 60 * 1000)

# 使用例
connection_timeout = Duration.from_seconds(30)
session_timeout = Duration.from_minutes(15)
```

### 2.2 関数名

| パターン | 用途 | 例 |
|---------|------|-----|
| `get/fetch/find` | データ取得 | `getUserById`, `fetchOrders` |
| `create/build/make` | 生成 | `createUser`, `buildQuery` |
| `update/modify/set` | 更新 | `updateProfile`, `setName` |
| `delete/remove/clear` | 削除 | `deleteUser`, `removeItem` |
| `is/has/can/should` | 判定 | `isValid`, `hasAccess` |
| `validate/check/verify` | 検証 | `validateEmail`, `checkAuth` |
| `convert/transform/to` | 変換 | `toJSON`, `convertToCSV` |
| `calculate/compute` | 計算 | `calculateTotal`, `computeHash` |
| `parse/extract` | 解析・抽出 | `parseDate`, `extractToken` |
| `ensure/require` | 前提条件の保証 | `ensureAuthenticated`, `requireAdmin` |
| `try/attempt` | 失敗する可能性 | `tryConnect`, `attemptLogin` |
| `register/subscribe` | イベント登録 | `registerHandler`, `subscribeToTopic` |

**get/fetch/find の使い分け:**

```typescript
// get: 既にメモリにある、または即座に取得可能なもの
class User {
  getName(): string { return this.name; }        // フィールドアクセス
  getAge(): number { return calculateAge(this.birthDate); } // 計算
}

// fetch: 外部リソース（API、DB）から非同期取得
async function fetchUserFromAPI(id: string): Promise<User> {
  return await api.get(`/users/${id}`);
}

// find: 検索（見つからない可能性がある → Optional/null を返す）
function findUserByEmail(email: string): User | undefined {
  return users.find(u => u.email === email);
}

// 混同すると危険な例
class UserService {
  // 悪い: getUser がDBアクセスするなら fetch の方が適切
  async getUser(id: string): Promise<User> {
    return await this.db.query('SELECT * FROM users WHERE id = ?', [id]);
  }

  // 良い: 非同期外部取得であることが名前から分かる
  async fetchUser(id: string): Promise<User> {
    return await this.db.query('SELECT * FROM users WHERE id = ?', [id]);
  }
}
```

**コード例5: 関数名の改善**

```python
# === 悪い: 何をする関数か分からない ===
def handle(data):
    pass

def do_it(x, y):
    pass

def process(items):
    pass

def run(config):
    pass

def execute(params):
    pass

# === 良い: 動詞+名詞で動作と対象を明示 ===
def validate_email_format(email: str) -> bool:
    """メールアドレスの形式を検証する"""
    pass

def calculate_monthly_revenue(transactions: list[Transaction]) -> Decimal:
    """月次売上を計算する"""
    pass

def send_password_reset_email(user: User) -> None:
    """パスワードリセットメールを送信する"""
    pass

def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """摂氏を華氏に変換する"""
    return celsius * 9 / 5 + 32

def find_expired_subscriptions(cutoff_date: date) -> list[Subscription]:
    """期限切れのサブスクリプションを検索する"""
    pass
```

### 2.3 クラス名

```
  ┌────────────────────────────────────────────┐
  │ クラス命名のガイドライン                    │
  ├────────────┬───────────────────────────────┤
  │ エンティティ │ 名詞: User, Order, Product   │
  ├────────────┼───────────────────────────────┤
  │ サービス    │ 名詞+Service: PaymentService  │
  │            │ 動詞er: OrderProcessor         │
  ├────────────┼───────────────────────────────┤
  │ リポジトリ  │ 名詞+Repository               │
  │            │ UserRepository                 │
  ├────────────┼───────────────────────────────┤
  │ ファクトリ  │ 名詞+Factory                  │
  │            │ ConnectionFactory              │
  ├────────────┼───────────────────────────────┤
  │ バリデータ  │ 名詞+Validator                │
  │            │ EmailValidator                 │
  ├────────────┼───────────────────────────────┤
  │ ビルダー   │ 名詞+Builder                  │
  │            │ QueryBuilder                   │
  ├────────────┼───────────────────────────────┤
  │ アダプター  │ 名詞+Adapter                  │
  │            │ StripePaymentAdapter           │
  ├────────────┼───────────────────────────────┤
  │ 例外       │ 名詞+Error/Exception          │
  │            │ InvalidInputError              │
  ├────────────┼───────────────────────────────┤
  │ インターフェース │ I+名詞 or 形容詞able     │
  │            │ IPaymentGateway, Serializable  │
  └────────────┴───────────────────────────────┘
```

**コード例6: 名前空間を活用した命名**

```java
// === 悪い: プレフィックスで名前空間を代用 ===
class AppUserAccountValidationService { }
class AppUserAccountRepository { }
class AppUserAccountDTO { }
// → 名前が長すぎて可読性が低い

// === 良い: パッケージ/モジュールで名前空間を構成 ===
package com.example.user.account;

class ValidationService { }
class Repository { }
class AccountDTO { }

// 使用時: 文脈から意味が明確
import com.example.user.account.ValidationService;
// → パッケージが文脈を提供するので、クラス名を短くできる
```

### 2.4 定数と列挙型の命名

**コード例7: 定数と列挙型**

```typescript
// === 定数: マジックナンバーに意味のある名前をつける ===

// 悪い: マジックナンバー
if (password.length < 8) { ... }
if (retryCount > 3) { ... }
if (status === 2) { ... }

// 良い: 意図が明確な定数
const MINIMUM_PASSWORD_LENGTH = 8;
const MAX_RETRY_ATTEMPTS = 3;

if (password.length < MINIMUM_PASSWORD_LENGTH) { ... }
if (retryCount > MAX_RETRY_ATTEMPTS) { ... }

// === 列挙型: 選択肢の集合に名前をつける ===

// 悪い: 文字列リテラルで状態を管理
let status = 'active';
if (status === 'active' || status === 'pending') { ... }
// → タイポに気づけない、補完が効かない

// 良い: 列挙型で型安全に
enum OrderStatus {
  PENDING = 'pending',
  CONFIRMED = 'confirmed',
  SHIPPED = 'shipped',
  DELIVERED = 'delivered',
  CANCELLED = 'cancelled',
}

enum UserRole {
  ADMIN = 'admin',
  EDITOR = 'editor',
  VIEWER = 'viewer',
}

// 使用
if (order.status === OrderStatus.PENDING || order.status === OrderStatus.CONFIRMED) {
  // タイポがコンパイルエラーで検出される
}
```

---

## 3. 命名の高度なテクニック

### 3.1 対称的な命名

対になる概念には対称的な名前をつける。

| 概念 | 良い対称的な命名 | 悪い非対称な命名 |
|------|----------------|----------------|
| 開始/終了 | `start` / `stop` | `start` / `end` |
| 追加/削除 | `add` / `remove` | `add` / `delete` |
| 開く/閉じる | `open` / `close` | `open` / `shutdown` |
| 取得/設定 | `get` / `set` | `get` / `put` |
| 送信/受信 | `send` / `receive` | `send` / `get` |
| 表示/非表示 | `show` / `hide` | `show` / `invisible` |
| 有効/無効 | `enable` / `disable` | `enable` / `off` |
| 登録/解除 | `register` / `unregister` | `register` / `remove` |
| 圧縮/展開 | `compress` / `decompress` | `compress` / `expand` |
| シリアライズ/デシリアライズ | `serialize` / `deserialize` | `serialize` / `parse` |

### 3.2 ドメイン用語の統一（ユビキタス言語）

```python
# === 悪い: 同じ概念に異なる名前を使う ===

# ファイル1: user_controller.py
def get_client(client_id):  # "client" と呼んでいる
    pass

# ファイル2: order_service.py
def create_order(customer_id):  # "customer" と呼んでいる
    pass

# ファイル3: notification.py
def notify_user(user_id):  # "user" と呼んでいる
    pass

# → "client", "customer", "user" は同じ概念？異なる概念？混乱を招く


# === 良い: 用語集（Glossary）を定義して統一 ===

# 用語集:
# - User: システムにログインする人
# - Customer: 商品を購入する人（User の一種）
# - Guest: ログインしていない訪問者

# ファイル1: user_controller.py
def get_user(user_id):  # 統一された用語
    pass

# ファイル2: order_service.py
def create_order(customer_id):  # Customer は User の特殊な役割
    pass

# ファイル3: notification.py
def notify_user(user_id):  # 統一された用語
    pass
```

### 3.3 コンテキストを活用した命名

**コード例8: コンテキストによる冗長性の排除**

```typescript
// === 悪い: コンテキストの冗長な繰り返し ===
class User {
  userName: string;        // "User" が冗長
  userEmail: string;       // "User" が冗長
  userAge: number;         // "User" が冗長
  userAddress: Address;    // "User" が冗長

  getUserName(): string { return this.userName; }
  setUserEmail(email: string): void { this.userEmail = email; }
}

// === 良い: クラスがコンテキストを提供 ===
class User {
  name: string;            // User.name で十分に明確
  email: string;           // User.email で十分に明確
  age: number;             // User.age で十分に明確
  address: Address;        // User.address で十分に明確

  getName(): string { return this.name; }
  setEmail(email: string): void { this.email = email; }
}

// === ただし、コンテキストの外では具体的に ===
// 関数の引数としてクラスの外に出る場合は具体的にする
function sendEmail(recipientEmail: string, senderEmail: string): void {
  // "email" だけでは sender か recipient か不明
}
```

---

## 4. 言語別の命名慣習

### 4.1 命名規約マトリクス

| 要素 | Python | JavaScript/TS | Java | Go | Rust |
|------|--------|--------------|------|-----|------|
| 変数 | snake_case | camelCase | camelCase | camelCase | snake_case |
| 関数 | snake_case | camelCase | camelCase | CamelCase(公開)/camelCase(非公開) | snake_case |
| クラス | PascalCase | PascalCase | PascalCase | PascalCase | PascalCase |
| 定数 | UPPER_SNAKE | UPPER_SNAKE | UPPER_SNAKE | CamelCase | UPPER_SNAKE |
| ファイル | snake_case | camelCase/kebab | PascalCase | snake_case | snake_case |
| パッケージ | snake_case | kebab-case | lowercase | lowercase | snake_case |
| インターフェース | なし(Protocol) | I+PascalCase/PascalCase | I+PascalCase | PascalCase+er | PascalCase |
| 列挙型 | PascalCase | PascalCase | PascalCase | PascalCase | PascalCase |

### 4.2 言語固有の慣習

**コード例9: 言語別の慣習例**

```python
# === Python の命名慣習 ===
# PEP 8 に従う

# モジュールレベル定数
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT_SECONDS = 30

# 関数と変数: snake_case
def calculate_total_price(items: list[OrderItem]) -> Decimal:
    subtotal = sum(item.price * item.quantity for item in items)
    return subtotal

# クラス: PascalCase
class OrderProcessor:
    # プライベート: _single_leading_underscore
    def _validate_order(self, order: Order) -> bool:
        pass

    # 名前マングリング: __double_leading_underscore（極めて稀に使用）
    def __internal_state(self):
        pass

    # ダンダーメソッド: __name__（Pythonの特殊メソッド）
    def __str__(self) -> str:
        pass
```

```go
// === Go の命名慣習 ===
// Go は大文字/小文字で公開/非公開を区別する

// 公開: CamelCase（大文字始まり）
func CalculateTax(amount float64) float64 {
    return amount * taxRate
}

// 非公開: camelCase（小文字始まり）
func calculateDiscount(amount float64) float64 {
    return amount * discountRate
}

// インターフェース: 動詞+er
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// 1メソッドのインターフェースは Doer 形式
type Stringer interface {
    String() string
}
```

```typescript
// === TypeScript の命名慣習 ===

// インターフェース: PascalCase（I プレフィックスは賛否あり）
interface PaymentGateway {
  charge(amount: number): Promise<PaymentResult>;
}

// 型エイリアス: PascalCase
type UserId = string;
type OrderStatus = 'pending' | 'confirmed' | 'shipped';

// 列挙型: PascalCase（メンバーもPascalCase）
enum HttpStatus {
  Ok = 200,
  NotFound = 404,
  InternalServerError = 500,
}

// ジェネリクス: 単一文字 or 意味のある名前
function identity<T>(value: T): T { return value; }
function mapArray<TInput, TOutput>(
  items: TInput[],
  transform: (item: TInput) => TOutput
): TOutput[] {
  return items.map(transform);
}
```

| 原則 | 説明 |
|------|------|
| プロジェクト内で統一 | 言語慣習よりもプロジェクト内の一貫性が重要 |
| Linter で自動強制 | ESLint, pylint, checkstyle で命名規則を強制 |
| レビューで確認 | 自動化できない意味の明確さはレビューで補完 |

---

## 5. 命名のリファクタリング

### 5.1 段階的な命名改善プロセス

```
  命名リファクタリングのフロー

  Step 1: 仮名で実装
  ┌────────────────────────────────┐
  │ temp_result = do_stuff(data)   │
  └────────────────────────────────┘
                │
                ▼
  Step 2: 全体の文脈が見えたら改善
  ┌────────────────────────────────────────┐
  │ validated_order = validate_order(raw_order) │
  └────────────────────────────────────────┘
                │
                ▼
  Step 3: レビューで更に磨く
  ┌──────────────────────────────────────────────┐
  │ validated_order = ensure_order_valid(raw_input) │
  └──────────────────────────────────────────────┘
```

**コード例10: IDEのリネーム機能を活用**

```python
# Step 1: 動くコードを書く（仮名でもOK）
def proc(d):
    r = []
    for x in d:
        if x > 0:
            r.append(x * 2)
    return r

# Step 2: 意図が分かったらリネーム（IDE: Shift+F6 / F2）
def double_positive_numbers(numbers: list[float]) -> list[float]:
    doubled = []
    for number in numbers:
        if number > 0:
            doubled.append(number * 2)
    return doubled

# Step 3: Pythonic に洗練
def double_positive_numbers(numbers: list[float]) -> list[float]:
    return [n * 2 for n in numbers if n > 0]
```

### 5.2 コードレビューでの命名チェックリスト

| チェック項目 | 質問 | 例 |
|------------|------|-----|
| 意図が明確か | 「この名前を見て即座に用途が分かるか？」 | `d` → `elapsedDays` |
| 誤解の可能性 | 「別の意味に解釈されないか？」 | `filter` → `excludeInactive` |
| 一貫性 | 「同じ概念に同じ単語を使っているか？」 | `get`/`fetch`/`retrieve` 混在 → 統一 |
| 略語の妥当性 | 「この略語はチーム全員が理解できるか？」 | `usr` → `user` |
| 否定形の排除 | 「二重否定になっていないか？」 | `isNotInactive` → `isActive` |
| コンテキスト | 「クラス名と合わせて冗長ではないか？」 | `User.userName` → `User.name` |

---

## 6. アンチパターン

### アンチパターン1: ハンガリアン記法の誤用

```typescript
// NG: 型をプレフィックスに入れる（現代のIDEでは不要）
let strName: string = "太郎";
let intAge: number = 25;
let arrUsers: User[] = [];
let bIsActive: boolean = true;
let objConfig: Config = {};

// OK: 型情報は型システムに任せる
let name: string = "太郎";
let age: number = 25;
let users: User[] = [];
let isActive: boolean = true;
let config: Config = {};

// 例外: UI系でのプレフィックスは許容される場合がある
// btnSubmit, txtEmail, lblError はUI要素の種類を示す
```

### アンチパターン2: 略語・暗号的な命名

```python
# NG: 解読が必要な名前
def calc_ttl_w_dsc(itms, dsc_pct):
    ttl = 0
    for itm in itms:
        ttl += itm.prc * itm.qty
    return ttl * (1 - dsc_pct / 100)

# OK: 省略せずフルスペルで
def calculate_total_with_discount(
    items: list[OrderItem],
    discount_percent: float
) -> float:
    subtotal = sum(item.price * item.quantity for item in items)
    return subtotal * (1 - discount_percent / 100)
```

### アンチパターン3: 汎用的すぎる名前

```python
# NG: 何でも意味が通ってしまう曖昧な名前
data = get_data()           # 何のデータ？
result = process(data)      # 何の結果？
info = fetch_info()         # 何の情報？
manager = get_manager()     # 何を管理？
handler = create_handler()  # 何をハンドル？
temp = calculate()          # 何の一時値？

# OK: 具体的な名前
user_profiles = fetch_active_user_profiles()
monthly_revenue = calculate_monthly_revenue(transactions)
server_health_info = check_server_health()
connection_manager = create_database_connection_pool()
request_handler = create_api_request_handler()
interpolated_value = interpolate_between(start, end, ratio)
```

---

## 7. 演習問題

### 演習1（基礎）: 命名の改善

以下のコードの変数名・関数名を改善してください。

```python
def f(l, n):
    r = []
    for i in l:
        if i.a > n:
            r.append(i)
    return r

d = f(get_all(), 18)
for i in d:
    s(i.e, "Welcome!")
```

**期待される回答:**

```python
def find_users_older_than(users: list[User], minimum_age: int) -> list[User]:
    eligible_users = []
    for user in users:
        if user.age > minimum_age:
            eligible_users.append(user)
    return eligible_users

# さらにPythonic に
def find_users_older_than(users: list[User], minimum_age: int) -> list[User]:
    return [user for user in users if user.age > minimum_age]

adult_users = find_users_older_than(get_all_users(), minimum_age=18)
for user in adult_users:
    send_email(user.email, "Welcome!")
```

### 演習2（応用）: ドメイン用語の統一

以下のコードベースで、同じ概念に異なる名前が使われている箇所を特定し、用語を統一してください。

```typescript
// user_controller.ts
function getClient(clientId: string): Client { ... }
function updateCustomerProfile(customerId: string, data: any): void { ... }

// notification_service.ts
function notifyMember(memberId: string, message: string): void { ... }

// billing_service.ts
function chargeAccount(accountHolderId: string, amount: number): void { ... }

// analytics_service.ts
function trackUserAction(userId: string, action: string): void { ... }
```

**期待される回答:**

```typescript
// 用語集を定義:
// - User: システムの利用者（統一された用語）
// - UserProfile: ユーザーのプロフィール情報
// - Account: 課金対象のアカウント

// user_controller.ts
function getUser(userId: string): User { ... }
function updateUserProfile(userId: string, profile: UserProfileUpdate): void { ... }

// notification_service.ts
function notifyUser(userId: string, message: string): void { ... }

// billing_service.ts
function chargeUserAccount(userId: string, amount: number): void { ... }

// analytics_service.ts
function trackUserAction(userId: string, action: string): void { ... }
```

### 演習3（発展）: 命名規約ドキュメントの作成

チームのために以下の要素を含む命名規約を作成してください。
- ブール変数のプレフィックスルール
- 非同期関数の命名パターン
- エラー型の命名規則
- APIエンドポイントの命名規則

**期待される回答:**

```markdown
# 命名規約

## 1. ブール変数
- プレフィックス: is, has, can, should, was, will
- 例: isActive, hasPermission, canEdit, shouldRetry

## 2. 非同期関数
- 外部API呼び出し: fetch + 名詞 (fetchUsers, fetchOrderById)
- DB操作: find/save/delete + 名詞 (findUserByEmail, saveOrder)
- 処理: process/handle + 名詞 + Async (processPaymentAsync)

## 3. エラー型
- 基底: AppError
- パターン: [名詞] + Error (ValidationError, NotFoundError)
- HTTP関連: Http + [ステータス] + Error (HttpNotFoundError)

## 4. APIエンドポイント
- RESTful: /api/v1/{resource}/{id}
- コレクション: 複数形 (/users, /orders)
- アクション: POST /api/v1/orders/{id}/cancel
```

---

## 8. FAQ

### Q1: 長い名前は悪いのか？

長い名前自体は悪くない。**意味が曖昧な短い名前のほうが遥かに有害**。ただし「関数名が長すぎて1行に収まらない」場合は、その関数が複数の責任を持っている兆候かもしれない。名前の長さではなく、責任の分離を見直す。

目安:
- 変数名: 5-25文字
- 関数名: 10-40文字
- クラス名: 5-30文字
- 名前が40文字を超える場合は設計を見直す

### Q2: 命名に迷って時間がかかりすぎる場合はどうすべきか？

仮の名前（`temp_xxx`）を付けて先に実装し、全体の文脈が見えた段階でリネームする。IDE のリファクタリング機能を使えばリネームは安全に行える。**命名は反復的なプロセス**として捉える。

実践的なアプローチ:
1. 最初は動詞+名詞で仮名をつける（5秒ルール）
2. テストが通ったらより良い名前を考える
3. コードレビューで第三者の目で確認する
4. 3ヶ月後の自分が理解できるかを基準にする

### Q3: 日本語の変数名は使ってよいか？

技術的には多くの言語で使用可能だが、以下の理由から英語が推奨される。
- 国際的なチームでの可読性
- ライブラリ/フレームワークとの一貫性
- 技術用語は英語のほうが正確
- StackOverflow等の情報が英語圏に集中

ただし、ドメイン固有の日本語概念（「確定申告」「源泉徴収」等）はコメントで補足するか、ローマ字で表現する（`kakutei_shinkoku`）。

### Q4: 命名規約はどの程度厳密にすべきか？

Linterで自動強制できるルール（ケーシング、長さ）は厳密に適用する。意味的な命名の品質はコードレビューで人間が確認する。重要なのは**一貫性**であり、チーム全体で同じルールに従うことが生産性を上げる。

### Q5: レガシーコードの命名を一括で変更すべきか？

一括変更はリスクが高い。以下の段階的アプローチを推奨する。

1. **新規コード**: 即座に新しい命名規約を適用
2. **変更するコード**: 変更のついでにリネーム
3. **頻繁に読むコード**: 優先的にリネーム
4. **安定した古いコード**: 触らない（リスクに対してリターンが低い）

### Q6: 同じ概念に対してgetとfetchどちらを使うべきか？

チーム内で一貫した基準を設けることが最も重要だが、一般的には以下の使い分けが広く受け入れられている。

| 動詞 | 意味 | 典型的な用途 |
|------|------|------------|
| `get` | 同期的に即座に値を返す。計算コストが低い | メモリ上のプロパティ取得、キャッシュからの読み出し |
| `fetch` | 非同期的に外部リソースから取得する | HTTP API呼び出し、外部サービスへの問い合わせ |
| `find` | 検索して見つからない可能性がある（null/undefinedを返す） | DBクエリ、コレクション内の検索 |
| `load` | ファイルやリソースを読み込んで初期化する | 設定ファイルの読み込み、モジュールの遅延ロード |
| `retrieve` | アーカイブや長期保存から復元する | バックアップからの復旧、キャッシュ再構築 |

```typescript
// get: 同期・軽量
function getUserName(user: User): string { return user.name; }

// fetch: 非同期・外部通信
async function fetchUserProfile(userId: string): Promise<UserProfile> {
  return await api.get(`/users/${userId}/profile`);
}

// find: 見つからない可能性あり
function findUserByEmail(email: string): User | null {
  return users.find(u => u.email === email) ?? null;
}
```

### Q7: テストメソッドの命名はどうすべきか？

テストメソッド名は通常のメソッドよりも長くなっても構わない。テスト名は「何を」「どの条件で」「どうなるべきか」を明確に表現すべきである。代表的なパターンは以下の通り。

```typescript
// パターン1: should + 期待動作 + when + 条件
it('should return empty array when no users match the criteria', () => { ... });

// パターン2: given-when-then をアンダースコアで区切る
test('given_expired_token_when_authenticate_then_throw_error', () => { ... });

// パターン3: メソッド名_条件_期待結果（xUnit スタイル）
test('calculateTotal_withDiscountCode_appliesDiscount', () => { ... });
```

いずれのパターンでも重要なのは、テストが失敗したときに名前だけで原因が推測できることである。`test1`、`testCalculate`のような曖昧な名前は避ける。

---

## まとめ

| 要素 | 命名の鍵 | 例 | パターン |
|------|---------|-----|---------|
| 変数 | 何を格納しているか | `activeUserCount` | 名詞/形容詞+名詞 |
| ブール | true/falseの意味 | `isAuthenticated` | is/has/can + 形容詞 |
| 関数 | 何をするか | `calculateShippingCost` | 動詞 + 名詞 |
| クラス | 何を表現するか | `PaymentProcessor` | 名詞 / 名詞+役割 |
| 定数 | 何の値か | `MAX_RETRY_COUNT` | UPPER_SNAKE_CASE |
| 列挙 | 選択肢の集合 | `OrderStatus.SHIPPED` | PascalCase.UPPER |
| インターフェース | 何ができるか | `Serializable` | 形容詞able / 名詞 |

| 原則 | 説明 | 確認方法 |
|------|------|---------|
| 意図の明確さ | 名前だけで目的が分かる | 「この名前を見て3秒で理解できるか？」 |
| 一貫性 | 同じ概念に同じ単語 | 用語集（Glossary）を維持 |
| 適切な長さ | スコープに比例 | ループ変数は短く、グローバルは長く |
| 検索可能性 | grepで一意に見つかる | IDE検索で確認 |
| 発音可能性 | 口頭で議論できる | チームミーティングで使ってみる |

---

## 次に読むべきガイド

- [関数設計](./01-functions.md) ── 命名と密接に関わる関数の設計原則
- [コメント](./03-comments.md) ── 名前で表現しきれない情報の補足方法
- [コードレビューチェックリスト](../03-practices-advanced/04-code-review-checklist.md) ── 命名のレビュー観点
- [DRY/KISS/YAGNI](../00-principles/02-dry-kiss-yagni.md) ── 名前のシンプルさとKISS原則

---

## 参考文献

1. **Robert C. Martin** 『Clean Code: A Handbook of Agile Software Craftsmanship』 Prentice Hall, 2008 (Chapter 2: Meaningful Names) ── 命名の基本原則
2. **Dustin Boswell, Trevor Foucher** 『The Art of Readable Code』 O'Reilly Media, 2011 ── 読みやすいコードの命名技法
3. **Steve McConnell** 『Code Complete: A Practical Handbook of Software Construction』 Microsoft Press, 2004 (Chapter 11: The Power of Variable Names) ── 変数命名の詳細なガイドライン
4. **George A. Miller** "The Magical Number Seven, Plus or Minus Two" Psychological Review, 1956 ── ワーキングメモリの容量に関する古典的論文
5. **Eric Evans** 『Domain-Driven Design: Tackling Complexity in the Heart of Software』 Addison-Wesley, 2003 ── ユビキタス言語の概念
6. **Python Software Foundation** "PEP 8 -- Style Guide for Python Code" ── Pythonの命名規約
7. **Effective Go** (golang.org) ── Goの命名慣習
8. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 ── Rename Variable、Rename Method の手順
