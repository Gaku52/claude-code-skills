# Encapsulation

> Encapsulation is the principle of "bundling data together with the methods that operate on it, and hiding the internal implementation details." It is the most fundamental and important of OOP's four pillars.

## What You Will Learn in This Chapter

- [ ] Understand the two facets of encapsulation (bundling and information hiding)
- [ ] Grasp how to use access modifiers appropriately
- [ ] Learn how to design immutable objects
- [ ] Be able to practice the Tell, Don't Ask principle
- [ ] Understand defensive copying and preventing data leakage
- [ ] Grasp the differences in encapsulation mechanisms across languages


## Prerequisites

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. The Two Facets of Encapsulation

```
Encapsulation = Bundling + Information Hiding

  Bundling:
    -> Group related data and methods into a single class
    -> Clarify the intent: "this data is manipulated by these methods"

  Information Hiding:
    -> Make internal implementation details invisible from outside
    -> Expose only the minimum necessary interface
    -> Internal implementation changes do not affect external code

  +----------------------------------+
  |         BankAccount              |
  |  +--------------------------+    |
  |  | private:                 |    |
  |  |   balance: number        |    |  internal (hidden)
  |  |   transactions: Log[]    |    |
  |  |   validate(amount)       |    |
  |  +--------------------------+    |
  |  +--------------------------+    |
  |  | public:                  |    |
  |  |   deposit(amount)        |    |  external (public API)
  |  |   withdraw(amount)       |    |
  |  |   getBalance()           |    |
  |  +--------------------------+    |
  +----------------------------------+
```

### 1.1 A Deeper Look at Bundling

Bundling is not simply about "grouping data and methods together"; the important thing is to form a "meaningful unit."

```typescript
// TypeScript: Good and bad examples of bundling

// Bad: mixing data and methods with little relation to each other
class Miscellaneous {
  userName: string = "";
  productPrice: number = 0;
  logLevel: string = "info";

  formatUserName(): string { return this.userName.trim(); }
  calculateTax(): number { return this.productPrice * 0.1; }
  writeLog(message: string): void { console.log(`[${this.logLevel}] ${message}`); }
}

// Good: bundle into meaningful units
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

### 1.2 The Essence of Information Hiding

The purpose of information hiding is to "localize the impact of changes." By hiding internal implementation, you can freely change internals without affecting external code.

```python
# Python: Localizing changes through information hiding

# Version 1: inventory management with a list
class Inventory:
    """Inventory management system (version 1: list implementation)"""

    def __init__(self):
        self._items: list[dict] = []  # internal implementation is a list

    def add_item(self, name: str, quantity: int, price: int) -> None:
        """Add a product"""
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
        """Withdraw a product"""
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
        """Get the stock quantity"""
        for item in self._items:
            if item["name"] == name:
                return item["quantity"]
        return 0

    def get_total_value(self) -> int:
        """Total value of the inventory"""
        return sum(item["quantity"] * item["price"] for item in self._items)

    def get_all_items(self) -> list[tuple[str, int, int]]:
        """Return info on all products (without leaking internal structure)"""
        return [(item["name"], item["quantity"], item["price"]) for item in self._items]


# Version 2: changed to dict-based inventory management
# -> External code requires no changes at all! (because the public API is the same)
class InventoryV2:
    """Inventory management system (version 2: dict implementation for speed)"""

    def __init__(self):
        self._items: dict[str, dict] = {}  # internal implementation changed to a dict

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


# Caller-side code: the same code works with either version 1 or 2
def process_order(inventory, item_name: str, quantity: int) -> bool:
    """Order processing: works without knowing the internal implementation"""
    stock = inventory.get_stock(item_name)
    if stock < quantity:
        print(f"在庫不足: {item_name}(在庫: {stock}, 注文: {quantity})")
        return False

    inventory.remove_item(item_name, quantity)
    print(f"出荷完了: {item_name} x {quantity}")
    return True

# Usage (works the same with either version)
inv = Inventory()  # or InventoryV2()
inv.add_item("ノートPC", 10, 150000)
inv.add_item("マウス", 50, 3000)

process_order(inv, "ノートPC", 3)
print(f"在庫総額: {inv.get_total_value():,}円")
```

### 1.3 Encapsulation and Design by Contract

```
Design by Contract:
  -> An encapsulated object has a "contract"

  Precondition:
    -> Condition that must hold before calling a method
    -> Example: deposit(amount) -> amount > 0

  Postcondition:
    -> Condition guaranteed to hold after the method executes
    -> Example: deposit(amount) -> balance increases by amount

  Invariant:
    -> Condition that always holds during the object's lifetime
    -> Example: balance >= 0
```

```typescript
// TypeScript: Design by contract and encapsulation

class DateRange {
  // Invariant: start <= end
  private readonly _start: Date;
  private readonly _end: Date;

  constructor(start: Date, end: Date) {
    // Check the precondition
    if (start > end) {
      throw new Error(
        `開始日(${start.toISOString()})は終了日(${end.toISOString()})以前である必要があります`
      );
    }
    this._start = new Date(start.getTime()); // defensive copy
    this._end = new Date(end.getTime());     // defensive copy
  }

  // Postcondition: the returned date is on or after start and on or before end
  get start(): Date {
    return new Date(this._start.getTime()); // return a defensive copy
  }

  get end(): Date {
    return new Date(this._end.getTime());
  }

  // Precondition: date is not null/undefined
  // Postcondition: true when start <= date <= end
  contains(date: Date): boolean {
    return date >= this._start && date <= this._end;
  }

  // Postcondition: the returned DateRange satisfies the invariant
  // (null if no intersection exists)
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

// Usage
const q1 = new DateRange(new Date("2025-01-01"), new Date("2025-03-31"));
const feb = new DateRange(new Date("2025-02-01"), new Date("2025-02-28"));

console.log(q1.contains(new Date("2025-02-15"))); // true
console.log(q1.intersect(feb)?.toString());        // "2025-02-01 ~ 2025-02-28 (27日間)"

// Violating the invariant is impossible
// const invalid = new DateRange(new Date("2025-12-31"), new Date("2025-01-01")); // Error!
```

---

## 2. Access Modifiers

```
+--------------+-----------+----------+-----------+----------+
| Modifier     | In class  | Subclass | Package   | Outside  |
+--------------+-----------+----------+-----------+----------+
| private      | yes       | no       | no        | no       |
| protected    | yes       | yes      | partial(Java) | no   |
| package      | yes       | no       | yes       | no       |
| public       | yes       | yes      | yes       | yes      |
+--------------+-----------+----------+-----------+----------+

Principle: choose the most restrictive access level
  -> Start with private and widen the scope only when needed
```

### 2.1 Access Control in Each Language

```typescript
// TypeScript
class User {
  public name: string;        // accessible from anywhere
  protected email: string;    // accessible from subclasses
  private password: string;   // within the class only
  readonly id: string;        // read-only

  constructor(name: string, email: string, password: string) {
    this.id = crypto.randomUUID();
    this.name = name;
    this.email = email;
    this.password = password;
  }
}
```

```python
# Python: convention-based (not enforced)
class User:
    def __init__(self, name: str, email: str, password: str):
        self.name = name          # public (by convention)
        self._email = email       # protected (convention: single underscore)
        self.__password = password # private (name mangling)

    @property
    def email(self) -> str:       # access control via property
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if "@" not in value:
            raise ValueError("Invalid email")
        self._email = value
```

```java
// Java: strict access control
public class User {
    private final String id;           // private + final = immutable
    private String name;
    private String email;

    public User(String name, String email) {
        this.id = UUID.randomUUID().toString();
        this.name = name;
        this.email = email;
    }

    // getters: only expose reads
    public String getName() { return name; }
    public String getEmail() { return email; }

    // setter: with validation
    public void setEmail(String email) {
        if (!email.contains("@")) {
            throw new IllegalArgumentException("Invalid email");
        }
        this.email = email;
    }

    // No setter is provided for id -> cannot be modified from outside
}
```

### 2.2 Detailed Comparison of Access Modifiers

```
Comparison of access control mechanisms across languages:

+----------+------------------------------------------------------+
| Java     | private, package-private (default), protected,       |
|          | public (4 levels). Strictly checked at compile time  |
+----------+------------------------------------------------------+
| C#       | private, protected, internal, protected internal,    |
|          | private protected, public (6 levels)                 |
+----------+------------------------------------------------------+
| C++      | private, protected, public (3 levels) + friend       |
|          | Compile-time checks. friend allows selective breaks  |
+----------+------------------------------------------------------+
| TypeScript| private, protected, public (3 levels)               |
|          | Checked only at compile time (no restriction at JS   |
|          | runtime). ECMAScript #private also available (with   |
|          | runtime restriction).                                |
+----------+------------------------------------------------------+
| Python   | Convention-based. _protected, __private (name        |
|          | mangling). Not enforced. Relies on "consenting       |
|          | adults" agreement.                                   |
+----------+------------------------------------------------------+
| Kotlin   | private, protected, internal, public (4 levels)      |
|          | internal has module scope                            |
+----------+------------------------------------------------------+
| Rust     | No pub = private to the crate, pub = public          |
|          | Fine-grained control via pub(crate), pub(super),     |
|          | pub(in path)                                         |
+----------+------------------------------------------------------+
| Go       | Uppercase start = public, lowercase start = private  |
|          | to the package. Only two levels. Simple but less     |
|          | flexible.                                            |
+----------+------------------------------------------------------+
| Swift    | open, public, internal, fileprivate, private         |
|          | (5 levels). open = public that allows                |
|          | inheritance/override.                                |
+----------+------------------------------------------------------+
```

```kotlin
// Kotlin: putting access modifiers to work

// internal: accessible only within the module (useful for multi-module development)
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
        // private: the way connections are created is fully hidden
        return Connection("jdbc:postgresql://localhost/mydb")
    }
}

// Public API: the interface users will use
class UserRepository(private val pool: DatabasePool) {
    fun findById(id: Long): User? {
        val conn = pool.getConnection()
        try {
            // database operation
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
// Rust: access control via the module system

mod database {
    // No pub = accessible only within this module
    struct ConnectionConfig {
        host: String,
        port: u16,
        database: String,
    }

    // pub(crate) = accessible only within the crate
    pub(crate) struct Connection {
        config: ConnectionConfig,
        is_active: bool,
    }

    impl Connection {
        // pub(crate) = usable within the crate
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

        // pub = usable from outside
        pub fn execute(&self, query: &str) -> Result<Vec<String>, String> {
            if !self.is_active {
                return Err("接続が無効です".to_string());
            }
            // query execution logic
            Ok(vec![format!("実行: {}", query)])
        }

        // private: within the module only
        fn reset(&mut self) {
            self.is_active = true;
        }

        pub fn close(&mut self) {
            self.is_active = false;
        }
    }

    // pub = the facade exposed to the outside
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

// From outside, only pub members are accessible
fn main() {
    let mut db = database::Database::new("localhost", 5432, "myapp", 5);
    if let Some(conn) = db.get_connection() {
        let result = conn.execute("SELECT 1");
        println!("{:?}", result);
    }
    // database::ConnectionConfig is inaccessible
    // conn.config is also inaccessible
}
```

### 2.3 TypeScript's ECMAScript Private Fields

```typescript
// TypeScript: truly private fields via #

class SecureWallet {
  // ECMAScript private fields: genuinely private at the runtime level
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

  // Public API
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

  // Private method: inaccessible from outside
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

// Usage
const wallet = new SecureWallet("田中太郎", 100000);
wallet.deposit(50000);
wallet.withdraw(30000);
console.log(wallet.getStatement());

// All of the following produce errors (in both TypeScript and JavaScript)
// wallet.#balance = 999999;  // SyntaxError
// wallet.#transactions;       // SyntaxError
// (wallet as any).#balance;  // SyntaxError
// Object.keys(wallet);       // # fields are not enumerated
```

---

## 3. The Getter/Setter Debate

```
"Putting a getter/setter on every field" is a bad habit:

  Bad example (Anemic Domain Model):
    class User {
      getName() / setName()
      getAge() / setAge()
      getEmail() / setEmail()
      getBalance() / setBalance()
    }
    -> Just a container for data. Behavior leaks to the outside.
    -> Encapsulation is pointless.

  Good example (Rich Domain Model):
    class BankAccount {
      deposit(amount)     <- contains business logic
      withdraw(amount)    <- includes validation
      getBalance()        <- read only
      // setBalance() does not exist
    }
    -> The object manages its own state responsibly.

Guidelines:
  getter: expose only what is needed
  setter: don't create them as a rule; provide business methods instead
  -> The "Tell, Don't Ask" principle
```

```typescript
// Example of Tell, Don't Ask

// Ask (query state and decide externally)
if (account.getBalance() >= amount) {
  account.setBalance(account.getBalance() - amount);
}

// Tell (give instructions to the object)
account.withdraw(amount); // validation + update happens internally
```

### 3.1 Problems with the Anemic Domain Model

```typescript
// TypeScript: comparing Anemic vs. Rich Domain Model

// Anemic Domain Model: nothing more than a data container
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

// Business logic scattered outside the object
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

  // Problem: anyone can directly set order.status = "shipped"
  // -> Validation can be bypassed
}

// Rich Domain Model: the object manages its own state responsibly
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

  // State is exposed as read-only
  get status(): OrderStatus {
    return this._status;
  }

  get items(): ReadonlyArray<OrderItem> {
    return [...this._items]; // defensive copy
  }

  // Business methods: contain the state transition rules
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

  // The Order is responsible for calculating the total
  get subtotal(): number {
    return this._items.reduce((sum, item) => sum + item.subtotal, 0);
  }

  get discountAmount(): number {
    return Math.floor(this.subtotal * this._discountRate);
  }

  get totalAmount(): number {
    return this.subtotal - this.discountAmount + this._shippingCost;
  }

  // State transitions: rules are contained here
  confirm(): void {
    if (this._status !== "pending") {
      throw new Error(`注文の確認ができません(現在のステータス: ${this._status})`);
    }
    if (this._items.length === 0) {
      throw new Error("空の注文は確認できません");
    }
    this._status = "confirmed";
  }

  ship(): void {
    if (this._status !== "confirmed") {
      throw new Error(`出荷できません(現在のステータス: ${this._status})`);
    }
    this._status = "shipping";
  }

  deliver(): void {
    if (this._status !== "shipping") {
      throw new Error(`配達完了にできません(現在のステータス: ${this._status})`);
    }
    this._status = "delivered";
  }

  cancel(): void {
    if (this._status === "delivered" || this._status === "cancelled") {
      throw new Error(`この注文はキャンセルできません(ステータス: ${this._status})`);
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

// Usage
const order = new Order("ORD-001", "CUST-123");
order.addItem(new OrderItem("P001", "MacBook Pro", 1, 298000));
order.addItem(new OrderItem("P002", "Magic Mouse", 2, 13800));
order.applyDiscount(0.1);
order.setShippingCost(0); // free shipping

console.log(order.getSummary());
order.confirm();
order.ship();
// order.addItem(...); // Error: 確定済みの注文に商品を追加できません
```

### 3.2 The Property Pattern (Smart Getters/Setters)

```python
# Python: smart access control using property

class Temperature:
    """Temperature class: internally stored in Kelvin"""

    def __init__(self, kelvin: float):
        self.kelvin = kelvin  # validation via the property

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
        """Temperature in Celsius (computed property)"""
        return self._kelvin - 273.15

    @celsius.setter
    def celsius(self, value: float) -> None:
        self.kelvin = value + 273.15  # reuse the Kelvin validation

    @property
    def fahrenheit(self) -> float:
        """Temperature in Fahrenheit (computed property)"""
        return self.celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5 / 9

    @property
    def is_freezing(self) -> bool:
        """Whether at or below the freezing point of water"""
        return self.celsius <= 0

    @property
    def is_boiling(self) -> bool:
        """Whether at or above the boiling point of water"""
        return self.celsius >= 100

    def __repr__(self) -> str:
        return f"Temperature({self.celsius:.1f}°C / {self.fahrenheit:.1f}°F / {self.kelvin:.1f}K)"


# Usage
t = Temperature(373.15)
print(t)              # Temperature(100.0°C / 212.0°F / 373.1K)
print(t.is_boiling)   # True

t.celsius = 25        # set via Celsius
print(t)              # Temperature(25.0°C / 77.0°F / 298.1K)

t.fahrenheit = 0      # set via Fahrenheit
print(t)              # Temperature(-17.8°C / 0.0°F / 255.4K)
print(t.is_freezing)  # True

try:
    t.kelvin = -10    # ValueError: 絶対零度以下は不可能です
except ValueError as e:
    print(e)
```

```java
// Java: lightweight data carriers via Record (Java 16+)

// Record: immutable data carrier (getters auto-generated, no setter)
public record Point(double x, double y) {
    // Validation: compact constructor
    public Point {
        if (Double.isNaN(x) || Double.isNaN(y)) {
            throw new IllegalArgumentException("座標にNaNは使用できません");
        }
    }

    // Additional methods
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

// Record: composite data
public record Address(
    String postalCode,
    String prefecture,
    String city,
    String street,
    String building   // allows null
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

// Usage
var p1 = new Point(3, 4);
var p2 = new Point(0, 0);
System.out.println(p1.distanceTo(p2)); // 5.0
System.out.println(p1.x());            // 3.0 (auto-generated getter)
// p1.x = 10;  // compile error: Record is immutable

var addr = new Address("100-0001", "東京都", "千代田区", "千代田1-1", null);
System.out.println(addr.toSingleLine()); // 〒100-0001 東京都千代田区千代田1-1
```

---

## 4. Immutable Objects

```
Immutable Object:
  -> An object whose state does not change after construction
  -> Thread-safe (no locks required)
  -> Predictable (no side effects)
  -> Safe to use as a hash key

How to create one:
  1. Make all fields final/readonly
  2. Do not provide setters
  3. Set every value in the constructor
  4. Do not leak mutable references to the outside
```

### 4.1 Immutable Objects in TypeScript

```typescript
// TypeScript: immutable object
class Money {
  constructor(
    public readonly amount: number,
    public readonly currency: string,
  ) {
    if (amount < 0) throw new Error("金額は0以上");
  }

  // Mutating methods return a new object
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
const tax = price.multiply(0.1);      // new object
const total = price.add(tax);         // new object
// price is unchanged (immutable)
```

### 4.2 Kotlin's data class

```kotlin
// Kotlin: data class (concise notation for immutable objects)
data class Point(val x: Double, val y: Double) {
    fun distanceTo(other: Point): Double =
        sqrt((x - other.x).pow(2) + (y - other.y).pow(2))

    // copy() creates a new object with only some fields changed
    fun translate(dx: Double, dy: Double): Point =
        copy(x = x + dx, y = y + dy)
}

val p1 = Point(1.0, 2.0)
val p2 = p1.translate(3.0, 4.0)  // Point(4.0, 6.0)
// p1 remains Point(1.0, 2.0)
```

### 4.3 Practical Patterns for Immutable Objects

```typescript
// TypeScript: practical immutable object design

// Object holding an immutable collection
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

  // Mutating operations return a new object
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

// Usage: every operation returns a new object
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

// The original playlist is not modified (immutable)
const reordered = playlist.reorder(2, 0);
console.log(reordered.songs[0].title); // "Drive"
console.log(playlist.songs[0].title);  // "Highway Star" (unchanged)
```

### 4.4 Preventing Leakage of Mutable Internal State

```java
// Java: protecting internal state with defensive copies

import java.util.*;

public class Schedule {
    private final String name;
    private final List<Event> events;  // mutable list

    public Schedule(String name, List<Event> events) {
        this.name = name;
        // Defensive copy: do not hold a reference to the external list
        this.events = new ArrayList<>(events);
        // Ideally also copy the objects inside the list (deep copy)
    }

    public String getName() { return name; }

    // Bad: returning the reference to the internal list directly
    // public List<Event> getEvents() { return events; }
    // -> External events.add() calls would corrupt internal state

    // Good 1: return an unmodifiable view
    public List<Event> getEvents() {
        return Collections.unmodifiableList(events);
    }

    // Good 2: return a defensive copy
    public List<Event> getEventsCopy() {
        return new ArrayList<>(events);
    }

    // Good 3: return a stream (Java 8+)
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
# Python: defensive copies and __slots__

from copy import deepcopy
from datetime import datetime
from typing import Iterator

class Config:
    """Configuration class: prevent leakage of the internal dict"""

    __slots__ = ("_data", "_frozen")  # disable __dict__ for memory efficiency

    def __init__(self, initial: dict | None = None):
        object.__setattr__(self, "_data", dict(initial or {}))
        object.__setattr__(self, "_frozen", False)

    def set(self, key: str, value) -> None:
        if self._frozen:
            raise RuntimeError("この設定は凍結されています")
        self._data[key] = deepcopy(value)  # defensive copy of the value too

    def get(self, key: str, default=None):
        value = self._data.get(key, default)
        return deepcopy(value)  # defensive copy of the returned value too

    def freeze(self) -> None:
        """Freeze the configuration (no further modifications allowed)"""
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

    # Block __setattr__ and __delattr__
    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Config のフィールドに直接アクセスできません")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Config のフィールドを削除できません")


# Usage
config = Config({"db_host": "localhost", "db_port": 5432})
config.set("features", ["auth", "logging"])

# Mutating the returned value does not affect the internal state
features = config.get("features")
features.append("hacked!")
print(config.get("features"))  # ["auth", "logging"] (not changed)

# Freeze
config.freeze()
try:
    config.set("db_host", "evil.com")  # RuntimeError
except RuntimeError as e:
    print(e)  # "この設定は凍結されています"
```

---

## 5. Encapsulation Anti-patterns

```
1. Exposing everything (public fields):
   -> Creates dependencies on internal implementation
   -> Changing it breaks all callers

2. Excessive getters/setters:
   -> Anemic Domain Model
   -> Defeats the purpose of encapsulation

3. Leaking internal collections:
   class Team {
     getMembers(): Member[] { return this.members; }
   }
   -> External members.push() calls corrupt internal state
   -> Return a defensive copy or a ReadonlyArray

4. Abuse of friend classes (C++):
   -> Blurs the boundary of encapsulation

5. Access via reflection:
   -> Can bypass private and access fields
   -> Use only in testing

```

### 5.1 Detailed Anti-patterns and Remedies

```typescript
// TypeScript: anti-patterns and remedies

// Anti-pattern 1: leaking internal collections
class TeamBad {
  private members: string[] = [];

  addMember(name: string): void {
    this.members.push(name);
  }

  getMembers(): string[] {
    return this.members; // returns the reference to the internal array directly!
  }
}

const teamBad = new TeamBad();
teamBad.addMember("田中");
const membersBad = teamBad.getMembers();
membersBad.push("不正なメンバー"); // internal state corrupted from outside!
console.log(teamBad.getMembers()); // ["田中", "不正なメンバー"]

// Remedy: ReadonlyArray + defensive copy
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
    return [...this.members]; // return a defensive copy
  }

  hasMember(name: string): boolean {
    return this.members.includes(name);
  }

  get size(): number {
    return this.members.length;
  }
}

// Anti-pattern 2: God Object (a huge class that knows everything)
class ApplicationBad {
  // User management
  private users: Map<string, any> = new Map();
  createUser(name: string): void { /* ... */ }
  deleteUser(id: string): void { /* ... */ }

  // Product management
  private products: Map<string, any> = new Map();
  addProduct(name: string, price: number): void { /* ... */ }
  removeProduct(id: string): void { /* ... */ }

  // Order management
  private orders: Map<string, any> = new Map();
  createOrder(userId: string, productId: string): void { /* ... */ }

  // Email sending
  sendEmail(to: string, subject: string, body: string): void { /* ... */ }

  // Logging
  log(message: string): void { /* ... */ }

  // -> A single class carries too many responsibilities
  // -> Too many reasons to change, making it unmaintainable
}

// Remedy: separation of responsibilities + facade pattern
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
    // order processing...
    return orderId;
  }
}
```

### 5.2 Feature Envy

```python
# Python: detecting and fixing Feature Envy

# Feature Envy: too dependent on another object's data
class ReportGeneratorBad:
    def generate_salary_report(self, employee) -> str:
        """Generate a salary report for the employee (bad example)"""
        base = employee.base_salary
        bonus = employee.bonus_rate * base
        tax = (base + bonus) * employee.tax_rate
        net = base + bonus - tax

        # Pulls out all of the employee's data and computes externally
        # -> employee itself should do the calculation
        return (
            f"名前: {employee.name}\n"
            f"基本給: {base:,}円\n"
            f"賞与: {bonus:,.0f}円\n"
            f"税金: {tax:,.0f}円\n"
            f"手取り: {net:,.0f}円"
        )


# Fix: move calculations into Employee
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
        """Return the salary breakdown (minimum data exposure)"""
        return {
            "base_salary": self._base_salary,
            "bonus": self.calculate_bonus(),
            "tax": self.calculate_tax(),
            "net_salary": self.calculate_net_salary(),
        }

class ReportGeneratorGood:
    def generate_salary_report(self, employee: Employee) -> str:
        """Generate a salary report for the employee (good example)"""
        breakdown = employee.get_salary_breakdown()
        return (
            f"名前: {employee.name}\n"
            f"基本給: {breakdown['base_salary']:,}円\n"
            f"賞与: {breakdown['bonus']:,}円\n"
            f"税金: {breakdown['tax']:,}円\n"
            f"手取り: {breakdown['net_salary']:,}円"
        )


# Usage
emp = Employee("田中太郎", 400000, 0.2, 0.3)
report = ReportGeneratorGood()
print(report.generate_salary_report(emp))
```

---

## 6. Module-level Encapsulation

### 6.1 Setting Boundaries via Packages/Modules

```
Encapsulation matters not only for classes but also at the module/package level:

  +--- public API -----------------------+
  |                                      |
  |  UserService  <- class used outside  |
  |  UserDTO      <- data returned out   |
  |                                      |
  |  +--- internal ------------------+   |
  |  |                                |  |
  |  |  UserRepository  <- internal  |  |
  |  |  UserValidator   <- internal  |  |
  |  |  UserMapper      <- internal  |  |
  |  |  user_queries.sql <- internal |  |
  |  |                                |  |
  |  +--------------------------------+  |
  |                                      |
  +--------------------------------------+
```

```python
# Python: __all__ and module-level encapsulation

# user_module/__init__.py
from .service import UserService
from .dto import UserDTO, CreateUserRequest

# List only the classes to expose in __all__
__all__ = ["UserService", "UserDTO", "CreateUserRequest"]

# Do not expose internal implementations
# UserRepository, UserValidator, UserMapper are not included in __all__
```

```typescript
// TypeScript: module encapsulation via barrel export

// user/index.ts (public API)
export { UserService } from "./service";
export { UserDTO, CreateUserRequest } from "./dto";

// The following are not exported (internal implementations)
// UserRepository, UserValidator, UserMapper
```

### 6.2 Direction of Dependencies and Encapsulation

```
Dependency Inversion Principle (DIP) and encapsulation:

  Bad: a higher-level module depends on a lower-level module's implementation
    OrderService -> MySQLOrderRepository
    -> Database changes propagate into OrderService

  Good: depend on an abstraction
    OrderService -> OrderRepository (interface)
                   ^
    MySQLOrderRepository implements OrderRepository
    -> Database changes don't affect OrderService
```

```typescript
// TypeScript: strengthening encapsulation through dependency inversion

// Abstractions (interfaces)
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

// Higher-level module: depends only on abstractions
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
    // Create the order
    const orderId = crypto.randomUUID();
    const totalAmount = items.reduce((sum, i) => sum + i.price * i.quantity, 0);

    // Payment processing (knows nothing of PaymentGateway's implementation details)
    const result = await this.payment.charge(totalAmount, "JPY", paymentMethodId);
    if (result.status === "failed") {
      throw new Error("支払いに失敗しました");
    }

    // Persist the order (knows nothing of OrderRepository's implementation details)
    const order = new Order(orderId, customerId, items, result.transactionId);
    await this.orderRepo.save(order);

    // Notification (knows nothing of NotificationService's implementation details)
    await this.notification.notify(
      customerId,
      `注文 #${orderId} が確定しました。合計: ${totalAmount.toLocaleString()}円`,
    );

    return orderId;
  }
}

// Lower-level module: implements the interface (implementation details are encapsulated)
class PostgresOrderRepository implements OrderRepository {
  // PostgreSQL-specific implementation is fully hidden
  private pool: any; // pg.Pool

  constructor(connectionString: string) {
    // DB connection details stay internal
  }

  async save(order: Order): Promise<void> {
    // SQL INSERT details are invisible from outside
  }

  async findById(id: string): Promise<Order | null> {
    // SQL SELECT details are invisible from outside
    return null;
  }

  async findByCustomer(customerId: string): Promise<Order[]> {
    return [];
  }
}

class StripePaymentGateway implements PaymentGateway {
  // How the Stripe SDK is used is fully hidden
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async charge(amount: number, currency: string, paymentMethodId: string): Promise<PaymentResult> {
    // Stripe API call details are invisible from outside
    return { transactionId: "txn_xxx", status: "success" };
  }

  async refund(transactionId: string): Promise<void> {
    // Stripe refund handling is invisible from outside
  }
}

class SlackNotificationService implements NotificationService {
  constructor(private webhookUrl: string) {}

  async notify(userId: string, message: string): Promise<void> {
    // How the Slack API is used is invisible from outside
  }
}

// Mock implementations for tests are also easy to create
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

## 7. Encapsulation and Testing

### 7.1 Balancing Testability and Encapsulation

```
Problem: you want to test a private method

  -> Testing private methods may itself be a "code smell"
  -> If the private method is too complex, it may be a sign that it
     should be extracted into a separate class

  Remedies:
    1. Test it indirectly via a public method
    2. Extract the complex private logic into another class
    3. Use package-private (Java) to make it accessible from the test class
```

```python
# Python: encapsulation with testability in mind

# Hard-to-test design
class OrderProcessorBad:
    def process(self, order_data: dict) -> str:
        # 1. Validation (want to test)
        if not order_data.get("customer_id"):
            raise ValueError("顧客IDが必要です")
        if not order_data.get("items"):
            raise ValueError("商品が必要です")

        # 2. Total calculation (want to test)
        total = 0
        for item in order_data["items"]:
            total += item["price"] * item["quantity"]
            if item["quantity"] > 100:
                total *= 0.9  # bulk discount

        # 3. Payment processing (external API call)
        payment_result = self._call_payment_api(total)

        # 4. Notification (external API call)
        self._send_notification(order_data["customer_id"], total)

        return payment_result["transaction_id"]

    def _call_payment_api(self, amount: float) -> dict:
        # external API call...
        return {"transaction_id": "xxx"}

    def _send_notification(self, customer_id: str, amount: float) -> None:
        # external API call...
        pass


# Test-friendly design: separation of responsibilities
from typing import Protocol

class PaymentGateway(Protocol):
    def charge(self, amount: int) -> str: ...

class NotificationSender(Protocol):
    def send(self, recipient: str, message: str) -> None: ...

class OrderValidator:
    """Extract the validation logic into its own class"""

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
    """Extract the price calculation logic into its own class"""

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
    """Order processing: compose the components"""

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
        # 1. Validation
        errors = self._validator.validate(order_data)
        if errors:
            raise ValueError(f"バリデーションエラー: {', '.join(errors)}")

        # 2. Total calculation
        total = self._calculator.calculate_total(order_data["items"])

        # 3. Payment
        transaction_id = self._payment.charge(total)

        # 4. Notification
        self._notification.send(
            order_data["customer_id"],
            f"注文完了: {total:,}円"
        )

        return transaction_id


# Tests
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
        # 200 > 100, so a 10% discount applies
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

## 8. Encapsulation Design Checklist

```
Checklist for encapsulation when designing a class:

[ ] Are all fields private (or at the most restrictive access level)?
[ ] Are only the truly necessary getters exposed?
[ ] Do you provide business methods instead of setters?
[ ] Do you avoid handing out references to internal collections directly?
[ ] Do you establish invariants in the constructor?
[ ] Do you defensively copy mutable arguments?
[ ] Are fields that can be immutable marked final/readonly?
[ ] Do you follow the Tell, Don't Ask principle?
[ ] Does each class have only a single responsibility?
[ ] Is the design such that changes to the internal implementation
    don't impact the outside?
[ ] Is encapsulation kept at a level that does not harm testability?
[ ] Do you also apply access control at the module/package level?
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Deeper understanding comes from not only the theory but also actually writing code and observing how it behaves.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping straight to advanced topics. We recommend firmly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

The knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and when designing architecture.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Bundling | Group data and methods into meaningful units |
| Information hiding | Hide internal implementation and expose only the public API; localize the impact of changes |
| Access modifiers | Choose the most restrictive level; mechanisms differ between languages |
| Getter/setter | Setters are generally unnecessary; provide business methods instead |
| Tell, Don't Ask | Instruct the object rather than querying its state |
| Immutable objects | Return new objects on modification; thread-safe |
| Defensive copying | Do not leak references to internal collections |
| Rich Domain Model | Objects manage their own state and logic |
| Design by contract | Define and check preconditions, postconditions, and invariants |
| Module encapsulation | Apply access control at the package/module level as well |
| Testability | Extract complex private logic into separate classes |

---

## Recommended Next Guides

---

## References
1. Bloch, J. "Effective Java." Item 15-17: Minimize accessibility, Use immutability. 2018.
2. Fowler, M. "Anemic Domain Model." martinfowler.com, 2003.
3. Parnas, D. "On the Criteria To Be Used in Decomposing Systems into Modules." CACM, 1972.
4. Meyer, B. "Object-Oriented Software Construction." 2nd Ed, Prentice Hall, 1997.
5. Evans, E. "Domain-Driven Design." Addison-Wesley, 2003.
6. Martin, R. "Clean Code." Prentice Hall, 2008.
7. Hunt, A. & Thomas, D. "The Pragmatic Programmer." 2nd Ed, Addison-Wesley, 2019.
