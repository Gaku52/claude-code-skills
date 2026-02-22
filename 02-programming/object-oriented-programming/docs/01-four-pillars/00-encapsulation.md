# カプセル化

> カプセル化は「データとそれを操作するメソッドを1つの単位にまとめ、内部の実装詳細を隠蔽する」原則。OOPの4つの柱の中で最も基本的かつ重要。

## この章で学ぶこと

- [ ] カプセル化の2つの側面（バンドリングと情報隠蔽）を理解する
- [ ] アクセス修飾子の使い分けを把握する
- [ ] 不変オブジェクトの設計を学ぶ
- [ ] Tell, Don't Ask 原則を実践できるようになる
- [ ] 防衛的コピーとデータ漏洩防止を理解する
- [ ] 各言語のカプセル化メカニズムの違いを把握する

---

## 1. カプセル化の2つの側面

```
カプセル化 = バンドリング + 情報隠蔽

  バンドリング（Bundling）:
    → 関連するデータとメソッドを1つのクラスにまとめる
    → 「このデータはこのメソッドで操作する」という意図を明確化

  情報隠蔽（Information Hiding）:
    → 内部の実装詳細を外部から見えなくする
    → 外部には必要最小限のインターフェースのみ公開
    → 内部実装を変更しても外部に影響しない

  ┌──────────────────────────────────┐
  │         BankAccount              │
  │  ┌──────────────────────────┐   │
  │  │ private:                 │   │
  │  │   balance: number        │   │  内部（隠蔽）
  │  │   transactions: Log[]    │   │
  │  │   validate(amount)       │   │
  │  └──────────────────────────┘   │
  │  ┌──────────────────────────┐   │
  │  │ public:                  │   │
  │  │   deposit(amount)        │   │  外部（公開API）
  │  │   withdraw(amount)       │   │
  │  │   getBalance()           │   │
  │  └──────────────────────────┘   │
  └──────────────────────────────────┘
```

### 1.1 バンドリングの深掘り

バンドリングは単に「データとメソッドをまとめる」だけでなく、「意味のある単位」を形成することが重要。

```typescript
// TypeScript: バンドリングの良い例と悪い例

// ❌ 悪い例: 関連性の薄いデータとメソッドの混在
class Miscellaneous {
  userName: string = "";
  productPrice: number = 0;
  logLevel: string = "info";

  formatUserName(): string { return this.userName.trim(); }
  calculateTax(): number { return this.productPrice * 0.1; }
  writeLog(message: string): void { console.log(`[${this.logLevel}] ${message}`); }
}

// ✅ 良い例: 意味のある単位にバンドリング
class UserName {
  constructor(private readonly value: string) {
    if (value.trim().length === 0) {
      throw new Error("ユーザー名は空にできません");
    }
  }

  format(): string {
    return this.value.trim();
  }

  equals(other: UserName): boolean {
    return this.format() === other.format();
  }

  toString(): string {
    return this.format();
  }
}

class Price {
  constructor(
    private readonly amount: number,
    private readonly currency: string = "JPY",
  ) {
    if (amount < 0) throw new Error("価格は0以上である必要があります");
  }

  calculateTax(rate: number = 0.1): Price {
    return new Price(Math.floor(this.amount * rate), this.currency);
  }

  withTax(rate: number = 0.1): Price {
    return new Price(Math.floor(this.amount * (1 + rate)), this.currency);
  }

  format(): string {
    return `${this.amount.toLocaleString()} ${this.currency}`;
  }

  add(other: Price): Price {
    if (this.currency !== other.currency) {
      throw new Error("通貨が異なります");
    }
    return new Price(this.amount + other.amount, this.currency);
  }

  compareTo(other: Price): number {
    if (this.currency !== other.currency) {
      throw new Error("通貨が異なります");
    }
    return this.amount - other.amount;
  }
}

class Logger {
  constructor(
    private readonly context: string,
    private level: "debug" | "info" | "warn" | "error" = "info",
  ) {}

  setLevel(level: "debug" | "info" | "warn" | "error"): void {
    this.level = level;
  }

  debug(message: string): void {
    if (this.shouldLog("debug")) {
      console.log(`[DEBUG][${this.context}] ${message}`);
    }
  }

  info(message: string): void {
    if (this.shouldLog("info")) {
      console.log(`[INFO][${this.context}] ${message}`);
    }
  }

  warn(message: string): void {
    if (this.shouldLog("warn")) {
      console.warn(`[WARN][${this.context}] ${message}`);
    }
  }

  error(message: string, error?: Error): void {
    if (this.shouldLog("error")) {
      console.error(`[ERROR][${this.context}] ${message}`, error?.stack ?? "");
    }
  }

  private shouldLog(level: string): boolean {
    const levels = ["debug", "info", "warn", "error"];
    return levels.indexOf(level) >= levels.indexOf(this.level);
  }
}
```

### 1.2 情報隠蔽の本質

情報隠蔽の目的は「変更の影響を局所化する」こと。内部実装を隠すことで、外部コードに影響を与えずに内部を自由に変更できる。

```python
# Python: 情報隠蔽による変更の局所化

# バージョン1: リストで在庫管理
class Inventory:
    """在庫管理システム（バージョン1: リスト実装）"""

    def __init__(self):
        self._items: list[dict] = []  # 内部実装はリスト

    def add_item(self, name: str, quantity: int, price: int) -> None:
        """商品を追加"""
        for item in self._items:
            if item["name"] == name:
                item["quantity"] += quantity
                return
        self._items.append({
            "name": name,
            "quantity": quantity,
            "price": price,
        })

    def remove_item(self, name: str, quantity: int) -> bool:
        """商品を取り出す"""
        for item in self._items:
            if item["name"] == name:
                if item["quantity"] >= quantity:
                    item["quantity"] -= quantity
                    if item["quantity"] == 0:
                        self._items.remove(item)
                    return True
                return False
        return False

    def get_stock(self, name: str) -> int:
        """在庫数を取得"""
        for item in self._items:
            if item["name"] == name:
                return item["quantity"]
        return 0

    def get_total_value(self) -> int:
        """在庫の総額"""
        return sum(item["quantity"] * item["price"] for item in self._items)

    def get_all_items(self) -> list[tuple[str, int, int]]:
        """全商品の情報を返す（内部構造を漏らさない）"""
        return [(item["name"], item["quantity"], item["price"]) for item in self._items]


# バージョン2: 辞書で在庫管理に変更
# → 外部のコードは一切変更不要！（公開APIが同じため）
class InventoryV2:
    """在庫管理システム（バージョン2: 辞書実装で高速化）"""

    def __init__(self):
        self._items: dict[str, dict] = {}  # 内部実装を辞書に変更

    def add_item(self, name: str, quantity: int, price: int) -> None:
        if name in self._items:
            self._items[name]["quantity"] += quantity
        else:
            self._items[name] = {"quantity": quantity, "price": price}

    def remove_item(self, name: str, quantity: int) -> bool:
        if name not in self._items:
            return False
        item = self._items[name]
        if item["quantity"] >= quantity:
            item["quantity"] -= quantity
            if item["quantity"] == 0:
                del self._items[name]
            return True
        return False

    def get_stock(self, name: str) -> int:
        item = self._items.get(name)
        return item["quantity"] if item else 0

    def get_total_value(self) -> int:
        return sum(
            item["quantity"] * item["price"]
            for item in self._items.values()
        )

    def get_all_items(self) -> list[tuple[str, int, int]]:
        return [
            (name, item["quantity"], item["price"])
            for name, item in self._items.items()
        ]


# 利用側のコード: バージョン1でも2でも同じコードが動く
def process_order(inventory, item_name: str, quantity: int) -> bool:
    """注文処理: 内部実装を知らなくても使える"""
    stock = inventory.get_stock(item_name)
    if stock < quantity:
        print(f"在庫不足: {item_name}（在庫: {stock}, 注文: {quantity}）")
        return False

    inventory.remove_item(item_name, quantity)
    print(f"出荷完了: {item_name} x {quantity}")
    return True

# 使用例（どちらのバージョンでも同じように動く）
inv = Inventory()  # または InventoryV2()
inv.add_item("ノートPC", 10, 150000)
inv.add_item("マウス", 50, 3000)

process_order(inv, "ノートPC", 3)
print(f"在庫総額: {inv.get_total_value():,}円")
```

### 1.3 カプセル化と契約による設計

```
契約による設計（Design by Contract）:
  → カプセル化されたオブジェクトは「契約」を持つ

  事前条件（Precondition）:
    → メソッドを呼ぶ前に満たすべき条件
    → 例: deposit(amount) → amount > 0

  事後条件（Postcondition）:
    → メソッド実行後に保証される条件
    → 例: deposit(amount) → 残高が amount だけ増加

  不変条件（Invariant）:
    → オブジェクトの生存期間中、常に成り立つ条件
    → 例: 残高 >= 0
```

```typescript
// TypeScript: 契約による設計とカプセル化

class DateRange {
  // 不変条件: start <= end
  private readonly _start: Date;
  private readonly _end: Date;

  constructor(start: Date, end: Date) {
    // 事前条件の検証
    if (start > end) {
      throw new Error(
        `開始日(${start.toISOString()})は終了日(${end.toISOString()})以前である必要があります`
      );
    }
    this._start = new Date(start.getTime()); // 防衛的コピー
    this._end = new Date(end.getTime());     // 防衛的コピー
  }

  // 事後条件: 返される日付は start 以後 end 以前
  get start(): Date {
    return new Date(this._start.getTime()); // 防衛的コピーを返す
  }

  get end(): Date {
    return new Date(this._end.getTime());
  }

  // 事前条件: date は null/undefined でない
  // 事後条件: start <= date <= end のとき true
  contains(date: Date): boolean {
    return date >= this._start && date <= this._end;
  }

  // 事後条件: 返される DateRange は不変条件を満たす
  // （intersection が存在しない場合は null）
  intersect(other: DateRange): DateRange | null {
    const newStart = new Date(Math.max(this._start.getTime(), other._start.getTime()));
    const newEnd = new Date(Math.min(this._end.getTime(), other._end.getTime()));

    if (newStart > newEnd) return null;
    return new DateRange(newStart, newEnd);
  }

  getDurationMs(): number {
    return this._end.getTime() - this._start.getTime();
  }

  getDurationDays(): number {
    return Math.ceil(this.getDurationMs() / (1000 * 60 * 60 * 24));
  }

  toString(): string {
    const fmt = (d: Date) => d.toISOString().split("T")[0];
    return `${fmt(this._start)} ~ ${fmt(this._end)} (${this.getDurationDays()}日間)`;
  }
}

// 使用例
const q1 = new DateRange(new Date("2025-01-01"), new Date("2025-03-31"));
const feb = new DateRange(new Date("2025-02-01"), new Date("2025-02-28"));

console.log(q1.contains(new Date("2025-02-15"))); // true
console.log(q1.intersect(feb)?.toString());        // "2025-02-01 ~ 2025-02-28 (27日間)"

// 不変条件の違反は不可能
// const invalid = new DateRange(new Date("2025-12-31"), new Date("2025-01-01")); // Error!
```

---

## 2. アクセス修飾子

```
┌──────────────┬───────────┬──────────┬───────────┬──────────┐
│ 修飾子       │ クラス内  │ サブクラス│ パッケージ│ 外部     │
├──────────────┼───────────┼──────────┼───────────┼──────────┤
│ private      │ ○        │ ×       │ ×        │ ×       │
│ protected    │ ○        │ ○       │ △(Java)  │ ×       │
│ package      │ ○        │ ×       │ ○        │ ×       │
│ public       │ ○        │ ○       │ ○        │ ○       │
└──────────────┴───────────┴──────────┴───────────┴──────────┘

原則: 最も制限的なアクセスレベルを選ぶ
  → まず private にして、必要に応じて公開範囲を広げる
```

### 2.1 各言語のアクセス制御

```typescript
// TypeScript
class User {
  public name: string;        // どこからでもアクセス可
  protected email: string;    // サブクラスからアクセス可
  private password: string;   // クラス内のみ
  readonly id: string;        // 読み取り専用

  constructor(name: string, email: string, password: string) {
    this.id = crypto.randomUUID();
    this.name = name;
    this.email = email;
    this.password = password;
  }
}
```

```python
# Python: 規約ベース（強制ではない）
class User:
    def __init__(self, name: str, email: str, password: str):
        self.name = name          # public（規約）
        self._email = email       # protected（規約: アンダースコア1つ）
        self.__password = password # private（名前マングリング）

    @property
    def email(self) -> str:       # プロパティでアクセス制御
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if "@" not in value:
            raise ValueError("Invalid email")
        self._email = value
```

```java
// Java: 厳格なアクセス制御
public class User {
    private final String id;           // private + final = 不変
    private String name;
    private String email;

    public User(String name, String email) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.email = email;
    }

    // getter: 読み取りのみ公開
    public String getName() { return name; }
    public String getEmail() { return email; }

    // setter: バリデーション付き
    public void setEmail(String email) {
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Invalid email");
        }
        this.email = email;
    }

    // id の setter は提供しない → 外部から変更不可
}
```

### 2.2 アクセス修飾子の詳細比較

```
各言語のアクセス制御メカニズムの比較:

┌──────────┬──────────────────────────────────────────────────────┐
│ Java     │ private, package-private(デフォルト), protected,     │
│          │ public の4段階。コンパイル時に厳格に検証              │
├──────────┼──────────────────────────────────────────────────────┤
│ C#       │ private, protected, internal, protected internal,   │
│          │ private protected, public の6段階                    │
├──────────┼──────────────────────────────────────────────────────┤
│ C++      │ private, protected, public の3段階 + friend         │
│          │ コンパイル時検証。friend でカプセル化を限定的に突破   │
├──────────┼──────────────────────────────────────────────────────┤
│ TypeScript│ private, protected, public の3段階                  │
│          │ コンパイル時のみ検証（JSランタイムでは制限なし）      │
│          │ ECMAScript #private も使用可能（ランタイム制限あり）  │
├──────────┼──────────────────────────────────────────────────────┤
│ Python   │ 規約ベース。_protected, __private（名前マングリング）│
│          │ 強制力なし。「大人の合意」に依存                      │
├──────────┼──────────────────────────────────────────────────────┤
│ Kotlin   │ private, protected, internal, public の4段階         │
│          │ internal はモジュールスコープ                         │
├──────────┼──────────────────────────────────────────────────────┤
│ Rust     │ pub なし=クレート内プライベート、pub=公開             │
│          │ pub(crate), pub(super), pub(in path) で細かく制御    │
├──────────┼──────────────────────────────────────────────────────┤
│ Go       │ 大文字始まり=公開、小文字始まり=パッケージ内プライベート│
│          │ 2段階のみ。シンプルだが柔軟性は低い                   │
├──────────┼──────────────────────────────────────────────────────┤
│ Swift    │ open, public, internal, fileprivate, private の5段階 │
│          │ open は継承/オーバーライド可能な公開                   │
└──────────┴──────────────────────────────────────────────────────┘
```

```kotlin
// Kotlin: アクセス修飾子の活用

// internal: モジュール内でのみアクセス可能（マルチモジュール開発に有効）
internal class DatabasePool {
    private val connections = mutableListOf<Connection>()
    private val maxSize = 10

    internal fun getConnection(): Connection {
        return if (connections.isNotEmpty()) {
            connections.removeFirst()
        } else {
            createNewConnection()
        }
    }

    internal fun returnConnection(conn: Connection) {
        if (connections.size < maxSize) {
            connections.add(conn)
        } else {
            conn.close()
        }
    }

    private fun createNewConnection(): Connection {
        // プライベート: 接続の作成方法は完全に隠蔽
        return Connection("jdbc:postgresql://localhost/mydb")
    }
}

// 公開API: ユーザーが使うインターフェース
class UserRepository(private val pool: DatabasePool) {
    fun findById(id: Long): User? {
        val conn = pool.getConnection()
        try {
            // データベース操作
            return conn.query("SELECT * FROM users WHERE id = ?", id)
        } finally {
            pool.returnConnection(conn)
        }
    }

    fun save(user: User): Unit {
        val conn = pool.getConnection()
        try {
            conn.execute("INSERT INTO users (name, email) VALUES (?, ?)",
                user.name, user.email)
        } finally {
            pool.returnConnection(conn)
        }
    }
}
```

```rust
// Rust: モジュールシステムによるアクセス制御

mod database {
    // pub なし = このモジュール内でのみアクセス可能
    struct ConnectionConfig {
        host: String,
        port: u16,
        database: String,
    }

    // pub(crate) = クレート内でのみアクセス可能
    pub(crate) struct Connection {
        config: ConnectionConfig,
        is_active: bool,
    }

    impl Connection {
        // pub(crate) = クレート内で使える
        pub(crate) fn new(host: &str, port: u16, database: &str) -> Self {
            Connection {
                config: ConnectionConfig {
                    host: host.to_string(),
                    port,
                    database: database.to_string(),
                },
                is_active: true,
            }
        }

        // pub = 外部から使える
        pub fn execute(&self, query: &str) -> Result<Vec<String>, String> {
            if !self.is_active {
                return Err("接続が無効です".to_string());
            }
            // クエリ実行ロジック
            Ok(vec![format!("実行: {}", query)])
        }

        // プライベート: モジュール内のみ
        fn reset(&mut self) {
            self.is_active = true;
        }

        pub fn close(&mut self) {
            self.is_active = false;
        }
    }

    // pub = 外部に公開するファサード
    pub struct Database {
        connections: Vec<Connection>,
    }

    impl Database {
        pub fn new(host: &str, port: u16, database: &str, pool_size: usize) -> Self {
            let connections = (0..pool_size)
                .map(|_| Connection::new(host, port, database))
                .collect();
            Database { connections }
        }

        pub fn get_connection(&mut self) -> Option<&Connection> {
            self.connections.iter().find(|c| c.is_active)
        }
    }
}

// 外部から使えるのは pub なメンバのみ
fn main() {
    let mut db = database::Database::new("localhost", 5432, "myapp", 5);
    if let Some(conn) = db.get_connection() {
        let result = conn.execute("SELECT 1");
        println!("{:?}", result);
    }
    // database::ConnectionConfig にはアクセス不可
    // conn.config にもアクセス不可
}
```

### 2.3 TypeScript の ECMAScript Private Fields

```typescript
// TypeScript: # による真のプライベートフィールド

class SecureWallet {
  // ECMAScript private fields: ランタイムレベルで本当にプライベート
  #balance: number;
  #transactions: Array<{ type: string; amount: number; date: Date }>;
  #owner: string;

  constructor(owner: string, initialBalance: number = 0) {
    if (initialBalance < 0) {
      throw new Error("初期残高は0以上である必要があります");
    }
    this.#owner = owner;
    this.#balance = initialBalance;
    this.#transactions = [];

    if (initialBalance > 0) {
      this.#recordTransaction("initial", initialBalance);
    }
  }

  // 公開API
  get owner(): string {
    return this.#owner;
  }

  get balance(): number {
    return this.#balance;
  }

  deposit(amount: number): void {
    this.#validatePositiveAmount(amount);
    this.#balance += amount;
    this.#recordTransaction("deposit", amount);
  }

  withdraw(amount: number): void {
    this.#validatePositiveAmount(amount);
    if (amount > this.#balance) {
      throw new Error(`残高不足: 残高=${this.#balance}, 出金額=${amount}`);
    }
    this.#balance -= amount;
    this.#recordTransaction("withdrawal", amount);
  }

  getStatement(): string {
    const lines = [
      `=== ${this.#owner} のウォレット ===`,
      `残高: ${this.#balance.toLocaleString()}円`,
      `--- 取引履歴 ---`,
    ];

    for (const tx of this.#transactions) {
      const sign = tx.type === "withdrawal" ? "-" : "+";
      const date = tx.date.toLocaleDateString("ja-JP");
      lines.push(`  ${date} ${tx.type}: ${sign}${tx.amount.toLocaleString()}円`);
    }

    return lines.join("\n");
  }

  // プライベートメソッド: 外部からアクセス不可
  #validatePositiveAmount(amount: number): void {
    if (amount <= 0) {
      throw new Error("金額は正の数である必要があります");
    }
    if (!Number.isFinite(amount)) {
      throw new Error("金額は有限の数である必要があります");
    }
  }

  #recordTransaction(type: string, amount: number): void {
    this.#transactions.push({ type, amount, date: new Date() });
  }
}

// 使用例
const wallet = new SecureWallet("田中太郎", 100000);
wallet.deposit(50000);
wallet.withdraw(30000);
console.log(wallet.getStatement());

// 以下は全てエラーになる（TypeScriptでもJavaScriptでも）
// wallet.#balance = 999999;  // SyntaxError
// wallet.#transactions;       // SyntaxError
// (wallet as any).#balance;  // SyntaxError
// Object.keys(wallet);       // #フィールドは列挙されない
```

---

## 3. ゲッター/セッター論争

```
「全フィールドに getter/setter を付ける」は悪い慣習:

  悪い例（Anemic Domain Model）:
    class User {
      getName() / setName()
      getAge() / setAge()
      getEmail() / setEmail()
      getBalance() / setBalance()
    }
    → 単なるデータの器。振る舞いが外部に漏れ出す
    → カプセル化の意味がない

  良い例（Rich Domain Model）:
    class BankAccount {
      deposit(amount)     ← ビジネスロジックを内包
      withdraw(amount)    ← バリデーション含む
      getBalance()        ← 読み取りのみ
      // setBalance() は存在しない
    }
    → オブジェクトが自分の責任で状態を管理

指針:
  getter: 必要なものだけ公開
  setter: 原則として作らない。代わりにビジネスメソッドを提供
  → 「Tell, Don't Ask」原則
```

```typescript
// Tell, Don't Ask の例

// ❌ Ask（状態を聞いてから外部で判断）
if (account.getBalance() >= amount) {
  account.setBalance(account.getBalance() - amount);
}

// ✅ Tell（オブジェクトに指示する）
account.withdraw(amount); // 内部でバリデーション + 更新
```

### 3.1 Anemic Domain Model の問題点

```typescript
// TypeScript: Anemic vs Rich Domain Model の比較

// ❌ Anemic Domain Model: データの器でしかない
class AnemicOrder {
  id: string = "";
  customerId: string = "";
  items: Array<{ productId: string; quantity: number; price: number }> = [];
  status: string = "pending";
  totalAmount: number = 0;
  discountRate: number = 0;
  shippingCost: number = 0;
  createdAt: Date = new Date();
}

// ビジネスロジックが外部に散乱
class OrderService {
  calculateTotal(order: AnemicOrder): void {
    let subtotal = 0;
    for (const item of order.items) {
      subtotal += item.price * item.quantity;
    }
    order.totalAmount = subtotal * (1 - order.discountRate) + order.shippingCost;
  }

  canCancel(order: AnemicOrder): boolean {
    return order.status === "pending" || order.status === "confirmed";
  }

  cancel(order: AnemicOrder): void {
    if (!this.canCancel(order)) {
      throw new Error("この注文はキャンセルできません");
    }
    order.status = "cancelled";
  }

  // 問題: 誰でも order.status = "shipped" と直接変更できてしまう
  // → バリデーションを回避される可能性
}

// ✅ Rich Domain Model: オブジェクトが自分の責任で状態を管理
class OrderItem {
  constructor(
    public readonly productId: string,
    public readonly productName: string,
    private _quantity: number,
    public readonly unitPrice: number,
  ) {
    if (_quantity <= 0) throw new Error("数量は1以上である必要があります");
    if (unitPrice < 0) throw new Error("単価は0以上である必要があります");
  }

  get quantity(): number {
    return this._quantity;
  }

  get subtotal(): number {
    return this._quantity * this.unitPrice;
  }

  updateQuantity(newQuantity: number): OrderItem {
    return new OrderItem(this.productId, this.productName, newQuantity, this.unitPrice);
  }
}

type OrderStatus = "pending" | "confirmed" | "shipping" | "delivered" | "cancelled";

class Order {
  private _status: OrderStatus = "pending";
  private _items: OrderItem[] = [];
  private _discountRate: number = 0;
  private _shippingCost: number = 0;
  public readonly id: string;
  public readonly customerId: string;
  public readonly createdAt: Date;

  constructor(id: string, customerId: string) {
    this.id = id;
    this.customerId = customerId;
    this.createdAt = new Date();
  }

  // 状態は読み取りのみ公開
  get status(): OrderStatus {
    return this._status;
  }

  get items(): ReadonlyArray<OrderItem> {
    return [...this._items]; // 防衛的コピー
  }

  // ビジネスメソッド: 状態遷移のルールを内包
  addItem(item: OrderItem): void {
    if (this._status !== "pending") {
      throw new Error("確定済みの注文に商品を追加できません");
    }
    this._items.push(item);
  }

  removeItem(productId: string): void {
    if (this._status !== "pending") {
      throw new Error("確定済みの注文から商品を削除できません");
    }
    this._items = this._items.filter(i => i.productId !== productId);
  }

  applyDiscount(rate: number): void {
    if (rate < 0 || rate > 0.5) {
      throw new Error("割引率は0〜50%の範囲である必要があります");
    }
    this._discountRate = rate;
  }

  setShippingCost(cost: number): void {
    if (cost < 0) throw new Error("送料は0以上である必要があります");
    this._shippingCost = cost;
  }

  // 合計額の計算は Order が責任を持つ
  get subtotal(): number {
    return this._items.reduce((sum, item) => sum + item.subtotal, 0);
  }

  get discountAmount(): number {
    return Math.floor(this.subtotal * this._discountRate);
  }

  get totalAmount(): number {
    return this.subtotal - this.discountAmount + this._shippingCost;
  }

  // 状態遷移: ルールを内包
  confirm(): void {
    if (this._status !== "pending") {
      throw new Error(`注文の確認ができません（現在のステータス: ${this._status}）`);
    }
    if (this._items.length === 0) {
      throw new Error("空の注文は確認できません");
    }
    this._status = "confirmed";
  }

  ship(): void {
    if (this._status !== "confirmed") {
      throw new Error(`出荷できません（現在のステータス: ${this._status}）`);
    }
    this._status = "shipping";
  }

  deliver(): void {
    if (this._status !== "shipping") {
      throw new Error(`配達完了にできません（現在のステータス: ${this._status}）`);
    }
    this._status = "delivered";
  }

  cancel(): void {
    if (this._status === "delivered" || this._status === "cancelled") {
      throw new Error(`この注文はキャンセルできません（ステータス: ${this._status}）`);
    }
    this._status = "cancelled";
  }

  getSummary(): string {
    const lines = [
      `注文 #${this.id} [${this._status}]`,
      `顧客: ${this.customerId}`,
      `--- 商品 ---`,
    ];
    for (const item of this._items) {
      lines.push(`  ${item.productName} x${item.quantity} = ${item.subtotal.toLocaleString()}円`);
    }
    lines.push(`小計: ${this.subtotal.toLocaleString()}円`);
    if (this._discountRate > 0) {
      lines.push(`割引: -${this.discountAmount.toLocaleString()}円 (${this._discountRate * 100}%)`);
    }
    if (this._shippingCost > 0) {
      lines.push(`送料: ${this._shippingCost.toLocaleString()}円`);
    }
    lines.push(`合計: ${this.totalAmount.toLocaleString()}円`);
    return lines.join("\n");
  }
}

// 使用例
const order = new Order("ORD-001", "CUST-123");
order.addItem(new OrderItem("P001", "MacBook Pro", 1, 298000));
order.addItem(new OrderItem("P002", "Magic Mouse", 2, 13800));
order.applyDiscount(0.1);
order.setShippingCost(0); // 送料無料

console.log(order.getSummary());
order.confirm();
order.ship();
// order.addItem(...); // Error: 確定済みの注文に商品を追加できません
```

### 3.2 プロパティパターン（スマートなgetter/setter）

```python
# Python: property を使ったスマートなアクセス制御

class Temperature:
    """温度クラス: 内部的にはケルビンで保持"""

    def __init__(self, kelvin: float):
        self.kelvin = kelvin  # property 経由でバリデーション

    @property
    def kelvin(self) -> float:
        return self._kelvin

    @kelvin.setter
    def kelvin(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"絶対零度以下は不可能です: {value}K")
        self._kelvin = value

    @property
    def celsius(self) -> float:
        """摂氏温度（計算プロパティ）"""
        return self._kelvin - 273.15

    @celsius.setter
    def celsius(self, value: float) -> None:
        self.kelvin = value + 273.15  # ケルビンのバリデーションを再利用

    @property
    def fahrenheit(self) -> float:
        """華氏温度（計算プロパティ）"""
        return self.celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5 / 9

    @property
    def is_freezing(self) -> bool:
        """水の凝固点以下かどうか"""
        return self.celsius <= 0

    @property
    def is_boiling(self) -> bool:
        """水の沸点以上かどうか"""
        return self.celsius >= 100

    def __repr__(self) -> str:
        return f"Temperature({self.celsius:.1f}°C / {self.fahrenheit:.1f}°F / {self.kelvin:.1f}K)"


# 使用例
t = Temperature(373.15)
print(t)              # Temperature(100.0°C / 212.0°F / 373.1K)
print(t.is_boiling)   # True

t.celsius = 25        # 摂氏で設定
print(t)              # Temperature(25.0°C / 77.0°F / 298.1K)

t.fahrenheit = 0      # 華氏で設定
print(t)              # Temperature(-17.8°C / 0.0°F / 255.4K)
print(t.is_freezing)  # True

try:
    t.kelvin = -10    # ValueError: 絶対零度以下は不可能です
except ValueError as e:
    print(e)
```

```java
// Java: Record による軽量なデータキャリア（Java 16+）

// Record: 不変のデータキャリア（getter 自動生成、setterなし）
public record Point(double x, double y) {
    // バリデーション: コンパクトコンストラクタ
    public Point {
        if (Double.isNaN(x) || Double.isNaN(y)) {
            throw new IllegalArgumentException("座標にNaNは使用できません");
        }
    }

    // 追加メソッド
    public double distanceTo(Point other) {
        double dx = this.x - other.x;
        double dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    public Point translate(double dx, double dy) {
        return new Point(x + dx, y + dy);
    }

    public Point scale(double factor) {
        return new Point(x * factor, y * factor);
    }
}

// Record: 複合的なデータ
public record Address(
    String postalCode,
    String prefecture,
    String city,
    String street,
    String building   // null 許容
) {
    public Address {
        if (postalCode == null || !postalCode.matches("\\d{3}-\\d{4}")) {
            throw new IllegalArgumentException("郵便番号の形式が不正です: " + postalCode);
        }
        if (prefecture == null || prefecture.isBlank()) {
            throw new IllegalArgumentException("都道府県は必須です");
        }
        if (city == null || city.isBlank()) {
            throw new IllegalArgumentException("市区町村は必須です");
        }
        if (street == null || street.isBlank()) {
            throw new IllegalArgumentException("番地は必須です");
        }
    }

    public String toSingleLine() {
        var sb = new StringBuilder();
        sb.append("〒").append(postalCode).append(" ");
        sb.append(prefecture).append(city).append(street);
        if (building != null && !building.isBlank()) {
            sb.append(" ").append(building);
        }
        return sb.toString();
    }
}

// 使用例
var p1 = new Point(3, 4);
var p2 = new Point(0, 0);
System.out.println(p1.distanceTo(p2)); // 5.0
System.out.println(p1.x());            // 3.0（自動生成のgetter）
// p1.x = 10;  // コンパイルエラー: Record は不変

var addr = new Address("100-0001", "東京都", "千代田区", "千代田1-1", null);
System.out.println(addr.toSingleLine()); // 〒100-0001 東京都千代田区千代田1-1
```

---

## 4. 不変オブジェクト

```
不変オブジェクト（Immutable Object）:
  → 生成後に状態が変化しないオブジェクト
  → スレッドセーフ（ロック不要）
  → 予測可能（副作用なし）
  → ハッシュキーとして安全

作り方:
  1. 全フィールドを final/readonly にする
  2. setter を提供しない
  3. コンストラクタで全ての値を設定
  4. ミュータブルな参照を外部に漏らさない
```

### 4.1 TypeScript の不変オブジェクト

```typescript
// TypeScript: 不変オブジェクト
class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: string,
  ) {
    if (amount < 0) throw new Error("金額は0以上");
  }

  // 変更メソッドは新しいオブジェクトを返す
  add(other: Money): Money {
    if (this.currency !== other.currency) {
      throw new Error("通貨が異なります");
    }
    return new Money(this.amount + other.amount, this.currency);
  }

  multiply(factor: number): Money {
    return new Money(this.amount * factor, this.currency);
  }

  toString(): string {
    return `${this.amount} ${this.currency}`;
  }
}

const price = new Money(1000, "JPY");
const tax = price.multiply(0.1);      // 新しいオブジェクト
const total = price.add(tax);         // 新しいオブジェクト
// price は変わらない（不変）
```

### 4.2 Kotlin の data class

```kotlin
// Kotlin: data class（不変オブジェクトの簡潔な記法）
data class Point(val x: Double, val y: Double) {
    fun distanceTo(other: Point): Double =
        sqrt((x - other.x).pow(2) + (y - other.y).pow(2))

    // copy() で一部だけ変えた新しいオブジェクトを生成
    fun translate(dx: Double, dy: Double): Point =
        copy(x = x + dx, y = y + dy)
}

val p1 = Point(1.0, 2.0)
val p2 = p1.translate(3.0, 4.0)  // Point(4.0, 6.0)
// p1 は Point(1.0, 2.0) のまま
```

### 4.3 不変オブジェクトの実践パターン

```typescript
// TypeScript: 実践的な不変オブジェクト設計

// 不変のコレクションを持つオブジェクト
class Playlist {
  private constructor(
    public readonly name: string,
    public readonly owner: string,
    private readonly _songs: ReadonlyArray<Song>,
    public readonly createdAt: Date,
  ) {}

  static create(name: string, owner: string): Playlist {
    return new Playlist(name, owner, [], new Date());
  }

  get songs(): ReadonlyArray<Song> {
    return this._songs;
  }

  get songCount(): number {
    return this._songs.length;
  }

  get totalDuration(): number {
    return this._songs.reduce((total, song) => total + song.durationSec, 0);
  }

  // 変更操作は新しいオブジェクトを返す
  addSong(song: Song): Playlist {
    if (this._songs.some(s => s.id === song.id)) {
      throw new Error(`「${song.title}」は既に追加されています`);
    }
    return new Playlist(
      this.name,
      this.owner,
      [...this._songs, song],
      this.createdAt,
    );
  }

  removeSong(songId: string): Playlist {
    const filtered = this._songs.filter(s => s.id !== songId);
    if (filtered.length === this._songs.length) {
      throw new Error(`ID: ${songId} の曲が見つかりません`);
    }
    return new Playlist(this.name, this.owner, filtered, this.createdAt);
  }

  reorder(fromIndex: number, toIndex: number): Playlist {
    if (fromIndex < 0 || fromIndex >= this._songs.length ||
        toIndex < 0 || toIndex >= this._songs.length) {
      throw new Error("インデックスが範囲外です");
    }
    const songs = [...this._songs];
    const [moved] = songs.splice(fromIndex, 1);
    songs.splice(toIndex, 0, moved);
    return new Playlist(this.name, this.owner, songs, this.createdAt);
  }

  rename(newName: string): Playlist {
    if (newName.trim().length === 0) {
      throw new Error("プレイリスト名は空にできません");
    }
    return new Playlist(newName, this.owner, this._songs, this.createdAt);
  }

  getSummary(): string {
    const minutes = Math.floor(this.totalDuration / 60);
    const lines = [
      `${this.name} (by ${this.owner})`,
      `${this.songCount}曲 / ${minutes}分`,
    ];
    for (let i = 0; i < this._songs.length; i++) {
      const song = this._songs[i];
      lines.push(`  ${i + 1}. ${song.title} - ${song.artist} (${song.formatDuration()})`);
    }
    return lines.join("\n");
  }
}

class Song {
  constructor(
    public readonly id: string,
    public readonly title: string,
    public readonly artist: string,
    public readonly durationSec: number,
  ) {}

  formatDuration(): string {
    const min = Math.floor(this.durationSec / 60);
    const sec = this.durationSec % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
  }
}

// 使用例: 全ての操作が新しいオブジェクトを返す
let playlist = Playlist.create("ドライブ", "田中");

playlist = playlist
  .addSong(new Song("s1", "Highway Star", "Deep Purple", 378))
  .addSong(new Song("s2", "Born to Run", "Springsteen", 270))
  .addSong(new Song("s3", "Drive", "The Cars", 235));

console.log(playlist.getSummary());
// ドライブ (by 田中)
// 3曲 / 14分
//   1. Highway Star - Deep Purple (6:18)
//   2. Born to Run - Springsteen (4:30)
//   3. Drive - The Cars (3:55)

// 元のplaylistは変更されない（不変）
const reordered = playlist.reorder(2, 0);
console.log(reordered.songs[0].title); // "Drive"
console.log(playlist.songs[0].title);  // "Highway Star"（変更なし）
```

### 4.4 ミュータブルな内部状態の漏洩防止

```java
// Java: 防衛的コピーによる内部状態の保護

import java.util.*;

public class Schedule {
    private final String name;
    private final List<Event> events;  // ミュータブルなリスト

    public Schedule(String name, List<Event> events) {
        this.name = name;
        // 防衛的コピー: 外部のリストへの参照を持たない
        this.events = new ArrayList<>(events);
        // さらにリスト内のオブジェクトもコピーすべき（深いコピー）
    }

    public String getName() { return name; }

    // ❌ 悪い例: 内部リストの参照を直接返す
    // public List<Event> getEvents() { return events; }
    // → 外部で events.add() されると内部状態が壊れる

    // ✅ 良い例1: 不変ビューを返す
    public List<Event> getEvents() {
        return Collections.unmodifiableList(events);
    }

    // ✅ 良い例2: 防衛的コピーを返す
    public List<Event> getEventsCopy() {
        return new ArrayList<>(events);
    }

    // ✅ 良い例3: ストリームを返す（Java 8+）
    public java.util.stream.Stream<Event> eventStream() {
        return events.stream();
    }

    public void addEvent(Event event) {
        events.add(event);
        events.sort(Comparator.comparing(Event::startTime));
    }

    public boolean hasConflict(Event newEvent) {
        return events.stream().anyMatch(e -> e.overlapsWith(newEvent));
    }
}

public record Event(
    String title,
    java.time.LocalDateTime startTime,
    java.time.LocalDateTime endTime
) {
    public Event {
        if (startTime.isAfter(endTime)) {
            throw new IllegalArgumentException("開始時刻は終了時刻より前である必要があります");
        }
    }

    public boolean overlapsWith(Event other) {
        return this.startTime.isBefore(other.endTime)
            && other.startTime.isBefore(this.endTime);
    }

    public java.time.Duration duration() {
        return java.time.Duration.between(startTime, endTime);
    }
}
```

```python
# Python: 防衛的コピーと __slots__

from copy import deepcopy
from datetime import datetime
from typing import Iterator

class Config:
    """設定クラス: 内部辞書の漏洩を防ぐ"""

    __slots__ = ("_data", "_frozen")  # __dict__ を無効化してメモリ効率化

    def __init__(self, initial: dict | None = None):
        object.__setattr__(self, "_data", dict(initial or {}))
        object.__setattr__(self, "_frozen", False)

    def set(self, key: str, value) -> None:
        if self._frozen:
            raise RuntimeError("この設定は凍結されています")
        self._data[key] = deepcopy(value)  # 値も防衛的コピー

    def get(self, key: str, default=None):
        value = self._data.get(key, default)
        return deepcopy(value)  # 返す値も防衛的コピー

    def freeze(self) -> None:
        """設定を凍結（以後変更不可）"""
        object.__setattr__(self, "_frozen", True)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        status = "frozen" if self._frozen else "mutable"
        return f"Config({status}, {len(self._data)} items)"

    # __setattr__ と __delattr__ をブロック
    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Config のフィールドに直接アクセスできません")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Config のフィールドを削除できません")


# 使用例
config = Config({"db_host": "localhost", "db_port": 5432})
config.set("features", ["auth", "logging"])

# 取得した値を変更しても内部に影響しない
features = config.get("features")
features.append("hacked!")
print(config.get("features"))  # ["auth", "logging"]（変更されていない）

# 凍結
config.freeze()
try:
    config.set("db_host", "evil.com")  # RuntimeError
except RuntimeError as e:
    print(e)  # "この設定は凍結されています"
```

---

## 5. カプセル化のアンチパターン

```
1. 全公開（public フィールド）:
   → 内部実装への依存が発生
   → 変更すると利用者コードが全て壊れる

2. 過剰な getter/setter:
   → Anemic Domain Model
   → カプセル化の意味がない

3. 内部コレクションの漏洩:
   class Team {
     getMembers(): Member[] { return this.members; }
   }
   → 外部で members.push() されると内部状態が壊れる
   → 防衛的コピーまたは ReadonlyArray を返す

4. フレンドクラスの乱用（C++）:
   → カプセル化の境界を曖昧にする

5. リフレクションによるアクセス:
   → private を無視してアクセス可能
   → テスト以外では使わない
```

### 5.1 アンチパターンの詳細と対策

```typescript
// TypeScript: アンチパターンと対策

// ❌ アンチパターン1: 内部コレクションの漏洩
class TeamBad {
  private members: string[] = [];

  addMember(name: string): void {
    this.members.push(name);
  }

  getMembers(): string[] {
    return this.members; // 内部配列の参照を直接返す！
  }
}

const teamBad = new TeamBad();
teamBad.addMember("田中");
const membersBad = teamBad.getMembers();
membersBad.push("不正なメンバー"); // 外部から内部状態を破壊！
console.log(teamBad.getMembers()); // ["田中", "不正なメンバー"]

// ✅ 対策: ReadonlyArray + 防衛的コピー
class TeamGood {
  private members: string[] = [];

  addMember(name: string): void {
    if (name.trim().length === 0) {
      throw new Error("メンバー名は空にできません");
    }
    if (this.members.includes(name)) {
      throw new Error(`${name} は既にメンバーです`);
    }
    this.members.push(name);
  }

  removeMember(name: string): void {
    const index = this.members.indexOf(name);
    if (index === -1) {
      throw new Error(`${name} はメンバーではありません`);
    }
    this.members.splice(index, 1);
  }

  getMembers(): ReadonlyArray<string> {
    return [...this.members]; // 防衛的コピーを返す
  }

  hasMember(name: string): boolean {
    return this.members.includes(name);
  }

  get size(): number {
    return this.members.length;
  }
}

// ❌ アンチパターン2: God Object（何でも知っている巨大クラス）
class ApplicationBad {
  // ユーザー管理
  private users: Map<string, any> = new Map();
  createUser(name: string): void { /* ... */ }
  deleteUser(id: string): void { /* ... */ }

  // 商品管理
  private products: Map<string, any> = new Map();
  addProduct(name: string, price: number): void { /* ... */ }
  removeProduct(id: string): void { /* ... */ }

  // 注文管理
  private orders: Map<string, any> = new Map();
  createOrder(userId: string, productId: string): void { /* ... */ }

  // メール送信
  sendEmail(to: string, subject: string, body: string): void { /* ... */ }

  // ログ出力
  log(message: string): void { /* ... */ }

  // → 1つのクラスが全責任を持ちすぎ
  // → 変更理由が多すぎて保守不能
}

// ✅ 対策: 責任の分離 + ファサードパターン
class UserRepository {
  private users = new Map<string, { name: string; email: string }>();

  create(name: string, email: string): string {
    const id = crypto.randomUUID();
    this.users.set(id, { name, email });
    return id;
  }

  findById(id: string): { name: string; email: string } | undefined {
    return this.users.get(id);
  }

  delete(id: string): boolean {
    return this.users.delete(id);
  }
}

class ProductCatalog {
  private products = new Map<string, { name: string; price: number }>();

  add(name: string, price: number): string {
    const id = crypto.randomUUID();
    this.products.set(id, { name, price });
    return id;
  }

  findById(id: string): { name: string; price: number } | undefined {
    return this.products.get(id);
  }
}

class OrderService {
  constructor(
    private readonly users: UserRepository,
    private readonly products: ProductCatalog,
  ) {}

  createOrder(userId: string, productId: string): string {
    const user = this.users.findById(userId);
    if (!user) throw new Error("ユーザーが見つかりません");

    const product = this.products.findById(productId);
    if (!product) throw new Error("商品が見つかりません");

    const orderId = crypto.randomUUID();
    // 注文処理...
    return orderId;
  }
}
```

### 5.2 Feature Envy（機能の横恋慕）

```python
# Python: Feature Envy の検出と修正

# ❌ Feature Envy: 他のオブジェクトのデータに依存しすぎ
class ReportGeneratorBad:
    def generate_salary_report(self, employee) -> str:
        """従業員の給与レポートを生成（悪い例）"""
        base = employee.base_salary
        bonus = employee.bonus_rate * base
        tax = (base + bonus) * employee.tax_rate
        net = base + bonus - tax

        # employee のデータを全部取り出して外部で計算
        # → employee 自身が計算すべき
        return (
            f"名前: {employee.name}\n"
            f"基本給: {base:,}円\n"
            f"賞与: {bonus:,.0f}円\n"
            f"税金: {tax:,.0f}円\n"
            f"手取り: {net:,.0f}円"
        )


# ✅ 修正: 計算を Employee に移動
class Employee:
    def __init__(self, name: str, base_salary: int,
                 bonus_rate: float, tax_rate: float):
        self._name = name
        self._base_salary = base_salary
        self._bonus_rate = bonus_rate
        self._tax_rate = tax_rate

    @property
    def name(self) -> str:
        return self._name

    def calculate_bonus(self) -> int:
        return int(self._base_salary * self._bonus_rate)

    def calculate_tax(self) -> int:
        gross = self._base_salary + self.calculate_bonus()
        return int(gross * self._tax_rate)

    def calculate_net_salary(self) -> int:
        gross = self._base_salary + self.calculate_bonus()
        return gross - self.calculate_tax()

    def get_salary_breakdown(self) -> dict[str, int]:
        """給与の内訳を返す（データの公開は最小限）"""
        return {
            "base_salary": self._base_salary,
            "bonus": self.calculate_bonus(),
            "tax": self.calculate_tax(),
            "net_salary": self.calculate_net_salary(),
        }

class ReportGeneratorGood:
    def generate_salary_report(self, employee: Employee) -> str:
        """従業員の給与レポートを生成（良い例）"""
        breakdown = employee.get_salary_breakdown()
        return (
            f"名前: {employee.name}\n"
            f"基本給: {breakdown['base_salary']:,}円\n"
            f"賞与: {breakdown['bonus']:,}円\n"
            f"税金: {breakdown['tax']:,}円\n"
            f"手取り: {breakdown['net_salary']:,}円"
        )


# 使用例
emp = Employee("田中太郎", 400000, 0.2, 0.3)
report = ReportGeneratorGood()
print(report.generate_salary_report(emp))
```

---

## 6. モジュールレベルのカプセル化

### 6.1 パッケージ/モジュールによる境界設定

```
カプセル化はクラスだけでなく、モジュール/パッケージレベルでも重要:

  ┌─── public API ──────────────────────┐
  │                                      │
  │  UserService  ←─ 外部が使うクラス    │
  │  UserDTO      ←─ 外部に返すデータ    │
  │                                      │
  │  ┌─── internal ──────────────────┐  │
  │  │                                │  │
  │  │  UserRepository  ←─ 内部実装  │  │
  │  │  UserValidator   ←─ 内部実装  │  │
  │  │  UserMapper      ←─ 内部実装  │  │
  │  │  user_queries.sql ←─ 内部     │  │
  │  │                                │  │
  │  └────────────────────────────────┘  │
  │                                      │
  └──────────────────────────────────────┘
```

```python
# Python: __all__ とモジュールレベルのカプセル化

# user_module/__init__.py
from .service import UserService
from .dto import UserDTO, CreateUserRequest

# 公開するクラスのみ __all__ に記載
__all__ = ["UserService", "UserDTO", "CreateUserRequest"]

# 内部実装は公開しない
# UserRepository, UserValidator, UserMapper は __all__ に含めない
```

```typescript
// TypeScript: barrel export によるモジュールカプセル化

// user/index.ts（公開API）
export { UserService } from "./service";
export { UserDTO, CreateUserRequest } from "./dto";

// 以下は export しない（内部実装）
// UserRepository, UserValidator, UserMapper
```

### 6.2 依存関係の方向とカプセル化

```
依存関係逆転の原則（DIP）とカプセル化:

  ❌ 悪い例: 上位モジュールが下位モジュールの実装に依存
    OrderService → MySQLOrderRepository
    → データベースの変更が OrderService に波及

  ✅ 良い例: 抽象に依存
    OrderService → OrderRepository（interface）
                   ↑
    MySQLOrderRepository implements OrderRepository
    → データベースの変更が OrderService に影響しない
```

```typescript
// TypeScript: 依存関係逆転によるカプセル化の強化

// 抽象（インターフェース）
interface OrderRepository {
  save(order: Order): Promise<void>;
  findById(id: string): Promise<Order | null>;
  findByCustomer(customerId: string): Promise<Order[]>;
}

interface NotificationService {
  notify(userId: string, message: string): Promise<void>;
}

interface PaymentGateway {
  charge(amount: number, currency: string, paymentMethodId: string): Promise<PaymentResult>;
  refund(transactionId: string): Promise<void>;
}

interface PaymentResult {
  transactionId: string;
  status: "success" | "failed";
}

// 上位モジュール: 抽象にのみ依存
class OrderUseCase {
  constructor(
    private readonly orderRepo: OrderRepository,
    private readonly payment: PaymentGateway,
    private readonly notification: NotificationService,
  ) {}

  async placeOrder(
    customerId: string,
    items: Array<{ productId: string; quantity: number; price: number }>,
    paymentMethodId: string,
  ): Promise<string> {
    // 注文の作成
    const orderId = crypto.randomUUID();
    const totalAmount = items.reduce((sum, i) => sum + i.price * i.quantity, 0);

    // 支払い処理（PaymentGateway の実装詳細は知らない）
    const result = await this.payment.charge(totalAmount, "JPY", paymentMethodId);
    if (result.status === "failed") {
      throw new Error("支払いに失敗しました");
    }

    // 注文の永続化（OrderRepository の実装詳細は知らない）
    const order = new Order(orderId, customerId, items, result.transactionId);
    await this.orderRepo.save(order);

    // 通知（NotificationService の実装詳細は知らない）
    await this.notification.notify(
      customerId,
      `注文 #${orderId} が確定しました。合計: ${totalAmount.toLocaleString()}円`,
    );

    return orderId;
  }
}

// 下位モジュール: インターフェースを実装（実装詳細はカプセル化）
class PostgresOrderRepository implements OrderRepository {
  // PostgreSQL 固有の実装は完全に隠蔽
  private pool: any; // pg.Pool

  constructor(connectionString: string) {
    // DB接続の詳細は内部に閉じる
  }

  async save(order: Order): Promise<void> {
    // SQL INSERT の詳細は外部から見えない
  }

  async findById(id: string): Promise<Order | null> {
    // SQL SELECT の詳細は外部から見えない
    return null;
  }

  async findByCustomer(customerId: string): Promise<Order[]> {
    return [];
  }
}

class StripePaymentGateway implements PaymentGateway {
  // Stripe SDK の使い方は完全に隠蔽
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async charge(amount: number, currency: string, paymentMethodId: string): Promise<PaymentResult> {
    // Stripe API 呼び出しの詳細は外部から見えない
    return { transactionId: "txn_xxx", status: "success" };
  }

  async refund(transactionId: string): Promise<void> {
    // Stripe の返金処理は外部から見えない
  }
}

class SlackNotificationService implements NotificationService {
  constructor(private webhookUrl: string) {}

  async notify(userId: string, message: string): Promise<void> {
    // Slack API の使い方は外部から見えない
  }
}

// テスト用のモック実装も簡単に作れる
class InMemoryOrderRepository implements OrderRepository {
  private orders = new Map<string, Order>();

  async save(order: Order): Promise<void> {
    this.orders.set(order.id, order);
  }

  async findById(id: string): Promise<Order | null> {
    return this.orders.get(id) ?? null;
  }

  async findByCustomer(customerId: string): Promise<Order[]> {
    return [...this.orders.values()].filter(o => o.customerId === customerId);
  }
}
```

---

## 7. カプセル化とテスト

### 7.1 テスタビリティとカプセル化のバランス

```
問題: private メソッドをテストしたい

  → private メソッドのテストは「コードの臭い」かもしれない
  → private メソッドが複雑すぎるなら、別クラスに抽出すべきサイン

  対策:
    1. public メソッド経由で間接的にテストする
    2. 複雑な private ロジックは別クラスに抽出する
    3. package-private（Java）を使ってテストクラスからアクセス可能にする
```

```python
# Python: テスタビリティを考慮したカプセル化

# ❌ テストしにくい設計
class OrderProcessorBad:
    def process(self, order_data: dict) -> str:
        # 1. バリデーション（テストしたい）
        if not order_data.get("customer_id"):
            raise ValueError("顧客IDが必要です")
        if not order_data.get("items"):
            raise ValueError("商品が必要です")

        # 2. 合計計算（テストしたい）
        total = 0
        for item in order_data["items"]:
            total += item["price"] * item["quantity"]
            if item["quantity"] > 100:
                total *= 0.9  # 大量割引

        # 3. 決済処理（外部API呼び出し）
        payment_result = self._call_payment_api(total)

        # 4. 通知（外部API呼び出し）
        self._send_notification(order_data["customer_id"], total)

        return payment_result["transaction_id"]

    def _call_payment_api(self, amount: float) -> dict:
        # 外部API呼び出し...
        return {"transaction_id": "xxx"}

    def _send_notification(self, customer_id: str, amount: float) -> None:
        # 外部API呼び出し...
        pass


# ✅ テストしやすい設計: 責任の分離
from typing import Protocol

class PaymentGateway(Protocol):
    def charge(self, amount: int) -> str: ...

class NotificationSender(Protocol):
    def send(self, recipient: str, message: str) -> None: ...

class OrderValidator:
    """バリデーションロジックを独立したクラスに抽出"""

    def validate(self, order_data: dict) -> list[str]:
        errors = []
        if not order_data.get("customer_id"):
            errors.append("顧客IDが必要です")
        if not order_data.get("items"):
            errors.append("商品が必要です")
        else:
            for i, item in enumerate(order_data["items"]):
                if item.get("price", 0) <= 0:
                    errors.append(f"商品{i}: 価格は正の数である必要があります")
                if item.get("quantity", 0) <= 0:
                    errors.append(f"商品{i}: 数量は正の数である必要があります")
        return errors

class PriceCalculator:
    """価格計算ロジックを独立したクラスに抽出"""

    BULK_THRESHOLD = 100
    BULK_DISCOUNT = 0.1

    def calculate_total(self, items: list[dict]) -> int:
        total = 0
        for item in items:
            subtotal = item["price"] * item["quantity"]
            if item["quantity"] > self.BULK_THRESHOLD:
                subtotal = int(subtotal * (1 - self.BULK_DISCOUNT))
            total += subtotal
        return total

class OrderProcessor:
    """注文処理: 各コンポーネントを合成"""

    def __init__(
        self,
        validator: OrderValidator,
        calculator: PriceCalculator,
        payment: PaymentGateway,
        notification: NotificationSender,
    ):
        self._validator = validator
        self._calculator = calculator
        self._payment = payment
        self._notification = notification

    def process(self, order_data: dict) -> str:
        # 1. バリデーション
        errors = self._validator.validate(order_data)
        if errors:
            raise ValueError(f"バリデーションエラー: {', '.join(errors)}")

        # 2. 合計計算
        total = self._calculator.calculate_total(order_data["items"])

        # 3. 決済
        transaction_id = self._payment.charge(total)

        # 4. 通知
        self._notification.send(
            order_data["customer_id"],
            f"注文完了: {total:,}円"
        )

        return transaction_id


# テスト
import unittest
from unittest.mock import Mock

class TestOrderValidator(unittest.TestCase):
    def setUp(self):
        self.validator = OrderValidator()

    def test_empty_customer_id(self):
        errors = self.validator.validate({"items": [{"price": 100, "quantity": 1}]})
        assert "顧客IDが必要です" in errors

    def test_valid_order(self):
        errors = self.validator.validate({
            "customer_id": "C001",
            "items": [{"price": 100, "quantity": 1}],
        })
        assert len(errors) == 0

class TestPriceCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = PriceCalculator()

    def test_simple_calculation(self):
        items = [{"price": 100, "quantity": 3}]
        assert self.calculator.calculate_total(items) == 300

    def test_bulk_discount(self):
        items = [{"price": 100, "quantity": 200}]
        # 200 > 100 なので10%割引
        assert self.calculator.calculate_total(items) == 18000

class TestOrderProcessor(unittest.TestCase):
    def test_successful_order(self):
        mock_payment = Mock()
        mock_payment.charge.return_value = "TXN-001"
        mock_notification = Mock()

        processor = OrderProcessor(
            validator=OrderValidator(),
            calculator=PriceCalculator(),
            payment=mock_payment,
            notification=mock_notification,
        )

        result = processor.process({
            "customer_id": "C001",
            "items": [{"price": 1000, "quantity": 2}],
        })

        assert result == "TXN-001"
        mock_payment.charge.assert_called_once_with(2000)
        mock_notification.send.assert_called_once()
```

---

## 8. カプセル化の設計指針チェックリスト

```
クラス設計時のカプセル化チェックリスト:

□ フィールドは全て private（または最も制限的なアクセスレベル）か？
□ getter は本当に必要なものだけ公開しているか？
□ setter の代わりにビジネスメソッドを提供しているか？
□ 内部コレクションの参照を直接外部に渡していないか？
□ コンストラクタで不変条件を確立しているか？
□ ミュータブルな引数を防衛的にコピーしているか？
□ 不変にできるフィールドは final/readonly にしているか？
□ Tell, Don't Ask 原則に従っているか？
□ 1つのクラスが持つ責任は1つだけか？
□ 内部実装の変更が外部に影響しない設計か？
□ テスタビリティを損なわない範囲でカプセル化しているか？
□ モジュール/パッケージレベルでもアクセス制御しているか？
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| バンドリング | データとメソッドを意味のある単位にまとめる |
| 情報隠蔽 | 内部実装を隠し、公開APIのみ提供。変更の影響を局所化 |
| アクセス修飾子 | 最も制限的なレベルを選ぶ。言語ごとに仕組みが異なる |
| getter/setter | setterは原則不要。ビジネスメソッドを提供 |
| Tell, Don't Ask | 状態を聞かずにオブジェクトに指示する |
| 不変オブジェクト | 変更時は新しいオブジェクトを返す。スレッドセーフ |
| 防衛的コピー | 内部コレクションの参照を外部に漏らさない |
| Rich Domain Model | オブジェクトが自分の責任で状態とロジックを管理 |
| 契約による設計 | 事前条件・事後条件・不変条件を定義・検証 |
| モジュールカプセル化 | パッケージ/モジュールレベルでもアクセス制御 |
| テスタビリティ | 複雑なprivateロジックは別クラスに抽出 |

---

## 次に読むべきガイド
→ [[01-inheritance.md]] — 継承

---

## 参考文献
1. Bloch, J. "Effective Java." Item 15-17: Minimize accessibility, Use immutability. 2018.
2. Fowler, M. "Anemic Domain Model." martinfowler.com, 2003.
3. Parnas, D. "On the Criteria To Be Used in Decomposing Systems into Modules." CACM, 1972.
4. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
5. Evans, E. "Domain-Driven Design." Addison-Wesley, 2003.
6. Martin, R. "Clean Code." Prentice Hall, 2008.
7. Hunt, A. & Thomas, D. "The Pragmatic Programmer." 2nd Ed, Addison-Wesley, 2019.
