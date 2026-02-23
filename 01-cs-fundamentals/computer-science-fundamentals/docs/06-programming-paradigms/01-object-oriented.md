# オブジェクト指向プログラミング

> OOPの本質は「カプセル化」「継承」「ポリモーフィズム」「抽象化」の4本柱であり、大規模ソフトウェアの複雑さを管理する手法である。適切に使えばコードの再利用性・保守性・拡張性が飛躍的に向上するが、乱用すると逆に複雑さを増す諸刃の剣でもある。

## この章で学ぶこと

- [ ] OOPの4大原則を説明し、それぞれの実装パターンを示せる
- [ ] SOLID原則を理解し、違反パターンと改善方法を説明できる
- [ ] 継承よりコンポジションを選ぶべき理由を具体例で示せる
- [ ] デザインパターンの代表例を実装できる
- [ ] OOPと他のパラダイムの融合手法を理解できる
- [ ] 実務でのクラス設計のベストプラクティスを適用できる

---

## 1. OOPの歴史と背景

### 1.1 OOPの誕生と発展

オブジェクト指向プログラミングは1960年代のSimulaに端を発し、1970年代のSmalltalkで本格的に体系化された。その後、C++、Java、C#、Python、Ruby等の主要言語に採用され、現在のソフトウェア開発の主流パラダイムとなっている。

```
OOPの歴史年表:

1967年  Simula 67
        - Ole-Johan Dahl と Kristen Nygaard が開発
        - クラスとオブジェクトの概念を初めて導入
        - シミュレーション用言語として設計

1972年  Smalltalk
        - Alan Kay が Xerox PARC で開発
        - 「すべてがオブジェクト」という純粋OOP
        - メッセージパッシングモデル
        - GUIの概念もここから

1979年  C++
        - Bjarne Stroustrup が開発
        - C言語にOOP機能を追加
        - 多重継承、テンプレート
        - 産業界で広く普及

1995年  Java
        - Sun Microsystems (James Gosling)
        - 「Write Once, Run Anywhere」
        - 単一継承 + インターフェース
        - ガベージコレクション

1995年  JavaScript
        - Brendan Eich が開発
        - プロトタイプベースOOP
        - クラスベースとは異なるアプローチ

2000年  C#
        - Microsoft (.NET Framework)
        - Java に影響を受けた設計
        - プロパティ、イベント、デリゲート

2004年  Scala
        - OOPと関数型の融合
        - トレイト（ミックスイン）

2014年  Swift
        - Apple が開発
        - プロトコル指向プログラミング
        - 値型（struct）を重視
```

### 1.2 OOPの基本的な考え方

OOPの核心は「現実世界のモデリング」にある。現実の物事をオブジェクトとして表現し、オブジェクト同士の相互作用としてプログラムを構築する。

```python
# OOPの根本的な考え方: 現実世界をモデル化する

# 例: ECサイトのドメインモデル
# 現実世界の概念をクラスとして表現する

class Customer:
    """顧客 - 商品を購入する人"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        self._orders: list['Order'] = []

    def place_order(self, cart: 'ShoppingCart') -> 'Order':
        """カートの中身から注文を作成する"""
        order = Order(customer=self, items=cart.items.copy())
        self._orders.append(order)
        cart.clear()
        return order

    @property
    def order_history(self) -> list['Order']:
        return self._orders.copy()

class Product:
    """商品 - 販売される物"""
    def __init__(self, name: str, price: int, stock: int):
        self.name = name
        self.price = price
        self._stock = stock

    def is_available(self) -> bool:
        return self._stock > 0

    def reduce_stock(self, quantity: int) -> None:
        if quantity > self._stock:
            raise ValueError(f"在庫不足: 残り{self._stock}個")
        self._stock -= quantity

class ShoppingCart:
    """買い物カゴ - 購入前の一時的な商品の集まり"""
    def __init__(self):
        self.items: list[tuple[Product, int]] = []

    def add(self, product: Product, quantity: int = 1) -> None:
        if not product.is_available():
            raise ValueError(f"{product.name}は在庫切れです")
        self.items.append((product, quantity))

    @property
    def total(self) -> int:
        return sum(p.price * q for p, q in self.items)

    def clear(self) -> None:
        self.items.clear()

class Order:
    """注文 - 確定した購入"""
    _next_id = 1

    def __init__(self, customer: Customer, items: list):
        self.order_id = Order._next_id
        Order._next_id += 1
        self.customer = customer
        self.items = items
        self.status = "pending"

    @property
    def total(self) -> int:
        return sum(p.price * q for p, q in self.items)

# 使用例
customer = Customer("田中太郎", "tanaka@example.com")
laptop = Product("ノートPC", 150000, 10)
mouse = Product("マウス", 3000, 50)

cart = ShoppingCart()
cart.add(laptop, 1)
cart.add(mouse, 2)
print(f"カート合計: {cart.total}円")  # 156000円

order = customer.place_order(cart)
print(f"注文ID: {order.order_id}, 合計: {order.total}円")
```

---

## 2. OOPの4大原則

### 2.1 カプセル化（Encapsulation）

カプセル化とは、データ（状態）とそれを操作するメソッド（振る舞い）をひとまとめにし、外部からの不正なアクセスを制限する仕組みである。情報隠蔽（Information Hiding）とも呼ばれる。

```python
# === カプセル化の基本 ===

# ❌ カプセル化なし: データが剥き出し
class BadBankAccount:
    def __init__(self):
        self.balance = 0  # 誰でもアクセスし放題
        self.transactions = []

# 外部から直接変更できてしまう
account = BadBankAccount()
account.balance = -1000000  # 不正な状態！
account.transactions = []    # 履歴を消去！


# ✅ カプセル化あり: データを保護
class BankAccount:
    """銀行口座 - 内部状態を適切に保護"""

    def __init__(self, account_number: str, initial_balance: int = 0):
        self._account_number = account_number  # protected（慣習）
        self.__balance = initial_balance        # private（name mangling）
        self.__transactions: list[dict] = []
        self.__is_frozen = False

    @property
    def balance(self) -> int:
        """残高の読み取り専用プロパティ"""
        return self.__balance

    @property
    def account_number(self) -> str:
        """口座番号の読み取り専用プロパティ"""
        return self._account_number

    def deposit(self, amount: int, description: str = "") -> bool:
        """入金処理"""
        if self.__is_frozen:
            raise RuntimeError("口座が凍結されています")
        if amount <= 0:
            raise ValueError("入金額は正の値である必要があります")

        self.__balance += amount
        self.__record_transaction("deposit", amount, description)
        return True

    def withdraw(self, amount: int, description: str = "") -> bool:
        """出金処理"""
        if self.__is_frozen:
            raise RuntimeError("口座が凍結されています")
        if amount <= 0:
            raise ValueError("出金額は正の値である必要があります")
        if amount > self.__balance:
            raise ValueError(f"残高不足（残高: {self.__balance}円）")

        self.__balance -= amount
        self.__record_transaction("withdrawal", amount, description)
        return True

    def get_statement(self, last_n: int = 10) -> list[dict]:
        """取引明細を取得（直近N件）"""
        return self.__transactions[-last_n:].copy()  # コピーを返す

    def freeze(self) -> None:
        """口座を凍結"""
        self.__is_frozen = True

    def __record_transaction(self, tx_type: str, amount: int, desc: str):
        """内部メソッド: 取引を記録"""
        from datetime import datetime
        self.__transactions.append({
            "type": tx_type,
            "amount": amount,
            "description": desc,
            "timestamp": datetime.now().isoformat(),
            "balance_after": self.__balance
        })

# 使用例
account = BankAccount("1234-5678", 100000)
account.deposit(50000, "給与振込")
account.withdraw(30000, "家賃")
print(f"残高: {account.balance}円")  # 120000円

# 不正操作は不可能
# account.__balance = -1000000  # AttributeError
# account.balance = 0           # プロパティのため設定不可
```

```java
// Java でのカプセル化（アクセス修飾子）
public class Employee {
    // アクセス修飾子の4段階
    // private:   クラス内部のみ
    // (default): 同一パッケージ内
    // protected: 同一パッケージ + サブクラス
    // public:    どこからでもアクセス可能

    private String id;
    private String name;
    private int salary;
    private Department department;

    public Employee(String id, String name, int salary) {
        this.id = id;
        setName(name);
        setSalary(salary);
    }

    // ゲッター: 読み取りアクセスを提供
    public String getId() { return id; }
    public String getName() { return name; }
    public int getSalary() { return salary; }

    // セッター: バリデーション付きで書き込みアクセスを制御
    public void setName(String name) {
        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("名前は必須です");
        }
        this.name = name.trim();
    }

    public void setSalary(int salary) {
        if (salary < 0) {
            throw new IllegalArgumentException("給与は0以上である必要があります");
        }
        this.salary = salary;
    }

    // ビジネスロジック
    public void raiseSalary(double percentage) {
        if (percentage < 0 || percentage > 50) {
            throw new IllegalArgumentException("昇給率は0-50%の範囲です");
        }
        this.salary = (int)(this.salary * (1 + percentage / 100));
    }
}
```

```typescript
// TypeScript でのカプセル化
class UserAccount {
    private _email: string;
    private _passwordHash: string;
    private _loginAttempts: number = 0;
    private _isLocked: boolean = false;
    readonly createdAt: Date;

    constructor(email: string, password: string) {
        this.validateEmail(email);
        this._email = email;
        this._passwordHash = this.hashPassword(password);
        this.createdAt = new Date();
    }

    // getアクセサ（読み取り専用）
    get email(): string { return this._email; }
    get isLocked(): boolean { return this._isLocked; }

    // パスワードのハッシュは外部に公開しない
    // get passwordHash は定義しない

    changeEmail(newEmail: string, currentPassword: string): void {
        if (!this.verifyPassword(currentPassword)) {
            throw new Error("現在のパスワードが正しくありません");
        }
        this.validateEmail(newEmail);
        this._email = newEmail;
    }

    login(password: string): boolean {
        if (this._isLocked) {
            throw new Error("アカウントがロックされています");
        }

        if (this.verifyPassword(password)) {
            this._loginAttempts = 0;
            return true;
        }

        this._loginAttempts++;
        if (this._loginAttempts >= 5) {
            this._isLocked = true;
        }
        return false;
    }

    private validateEmail(email: string): void {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            throw new Error("無効なメールアドレス形式です");
        }
    }

    private hashPassword(password: string): string {
        // 実際にはbcrypt等を使用
        return `hashed_${password}`;
    }

    private verifyPassword(password: string): boolean {
        return this._passwordHash === this.hashPassword(password);
    }
}
```

### 2.2 継承（Inheritance）

継承は既存クラス（親クラス/スーパークラス）の機能を引き継いで新しいクラス（子クラス/サブクラス）を作成する仕組みである。コードの再利用と「is-a」関係のモデリングに使用する。

```python
# === 継承の基本パターン ===

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Optional


# 基底クラス（親クラス）
class Employee(ABC):
    """従業員の基底クラス"""

    _next_id = 1

    def __init__(self, name: str, hire_date: date, base_salary: int):
        self.employee_id = Employee._next_id
        Employee._next_id += 1
        self.name = name
        self.hire_date = hire_date
        self._base_salary = base_salary

    @property
    def years_of_service(self) -> int:
        """勤続年数"""
        return (date.today() - self.hire_date).days // 365

    @abstractmethod
    def calculate_pay(self) -> int:
        """月給を計算する（サブクラスで実装必須）"""
        pass

    @abstractmethod
    def get_role(self) -> str:
        """役職名を返す"""
        pass

    def __str__(self) -> str:
        return f"{self.get_role()}: {self.name} (ID: {self.employee_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, id={self.employee_id})"


# 正社員
class FullTimeEmployee(Employee):
    """正社員"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 bonus_rate: float = 0.1):
        super().__init__(name, hire_date, base_salary)
        self.bonus_rate = bonus_rate

    def calculate_pay(self) -> int:
        # 勤続年数に応じた昇給込み
        seniority_bonus = self.years_of_service * 5000
        return self._base_salary + seniority_bonus

    def calculate_annual_bonus(self) -> int:
        """年間ボーナスを計算"""
        return int(self._base_salary * 12 * self.bonus_rate)

    def get_role(self) -> str:
        return "正社員"


# 契約社員
class ContractEmployee(Employee):
    """契約社員"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 contract_end: date):
        super().__init__(name, hire_date, base_salary)
        self.contract_end = contract_end

    def calculate_pay(self) -> int:
        return self._base_salary

    def is_contract_active(self) -> bool:
        return date.today() <= self.contract_end

    def get_role(self) -> str:
        return "契約社員"


# パートタイム
class PartTimeEmployee(Employee):
    """パートタイム従業員"""

    def __init__(self, name: str, hire_date: date, hourly_rate: int,
                 hours_per_month: int):
        super().__init__(name, hire_date, 0)
        self.hourly_rate = hourly_rate
        self.hours_per_month = hours_per_month

    def calculate_pay(self) -> int:
        return self.hourly_rate * self.hours_per_month

    def get_role(self) -> str:
        return "パートタイム"


# マネージャー（正社員の拡張）
class Manager(FullTimeEmployee):
    """マネージャー - 正社員に管理機能を追加"""

    def __init__(self, name: str, hire_date: date, base_salary: int,
                 management_allowance: int = 50000):
        super().__init__(name, hire_date, base_salary, bonus_rate=0.15)
        self.management_allowance = management_allowance
        self._subordinates: list[Employee] = []

    def calculate_pay(self) -> int:
        return super().calculate_pay() + self.management_allowance

    def add_subordinate(self, employee: Employee) -> None:
        if employee not in self._subordinates:
            self._subordinates.append(employee)

    def get_subordinates(self) -> list[Employee]:
        return self._subordinates.copy()

    def get_role(self) -> str:
        return "マネージャー"


# ポリモーフィズムの活用
def print_payroll(employees: list[Employee]) -> None:
    """全従業員の給与明細を出力 - 型に関係なく統一的に処理"""
    total = 0
    for emp in employees:
        pay = emp.calculate_pay()
        total += pay
        print(f"  {emp} → 月給: {pay:,}円")
    print(f"  {'─' * 40}")
    print(f"  合計: {total:,}円")


# 使用例
employees = [
    Manager("佐藤部長", date(2015, 4, 1), 500000),
    FullTimeEmployee("田中一郎", date(2020, 4, 1), 300000),
    ContractEmployee("鈴木二郎", date(2024, 4, 1), 350000, date(2026, 3, 31)),
    PartTimeEmployee("山田花子", date(2023, 10, 1), 1200, 80),
]
print_payroll(employees)
```

```java
// Java での継承とインターフェース

// インターフェース: 契約（何ができるか）を定義
interface Payable {
    int calculatePay();
}

interface Reportable {
    String generateReport();
}

// 抽象クラス: 共通実装を提供
abstract class Employee implements Payable, Reportable {
    protected final String name;
    protected final String employeeId;
    protected int baseSalary;

    protected Employee(String name, String employeeId, int baseSalary) {
        this.name = name;
        this.employeeId = employeeId;
        this.baseSalary = baseSalary;
    }

    // テンプレートメソッドパターン
    @Override
    public final String generateReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 従業員レポート ===\n");
        sb.append("名前: ").append(name).append("\n");
        sb.append("ID: ").append(employeeId).append("\n");
        sb.append("月給: ").append(calculatePay()).append("円\n");
        appendExtraInfo(sb);  // サブクラスでカスタマイズ可能
        return sb.toString();
    }

    // サブクラスがオーバーライドしてレポートに情報を追加
    protected void appendExtraInfo(StringBuilder sb) {
        // デフォルトでは何もしない
    }

    public abstract String getRole();
}

// 具象クラス
class FullTimeEmployee extends Employee {
    private double bonusRate;

    public FullTimeEmployee(String name, String id, int salary, double bonusRate) {
        super(name, id, salary);
        this.bonusRate = bonusRate;
    }

    @Override
    public int calculatePay() {
        return baseSalary;
    }

    public int calculateAnnualBonus() {
        return (int)(baseSalary * 12 * bonusRate);
    }

    @Override
    public String getRole() { return "正社員"; }

    @Override
    protected void appendExtraInfo(StringBuilder sb) {
        sb.append("年間ボーナス: ").append(calculateAnnualBonus()).append("円\n");
    }
}
```

### 2.3 ポリモーフィズム（Polymorphism）

ポリモーフィズムとは「多態性」のことで、同じインターフェースに対して異なる実装を提供する仕組みである。コンパイル時ポリモーフィズム（メソッドオーバーロード）と実行時ポリモーフィズム（メソッドオーバーライド）の2種類がある。

```python
# === ポリモーフィズムの実践例 ===

from abc import ABC, abstractmethod
from typing import Protocol


# 1. 抽象基底クラスによるポリモーフィズム
class NotificationSender(ABC):
    """通知送信の抽象インターフェース"""

    @abstractmethod
    def send(self, recipient: str, message: str) -> bool:
        """通知を送信する"""
        pass

    @abstractmethod
    def get_channel_name(self) -> str:
        """チャネル名を返す"""
        pass


class EmailSender(NotificationSender):
    def __init__(self, smtp_server: str):
        self.smtp_server = smtp_server

    def send(self, recipient: str, message: str) -> bool:
        print(f"[Email] {recipient} に送信: {message}")
        # SMTP送信処理...
        return True

    def get_channel_name(self) -> str:
        return "メール"


class SlackSender(NotificationSender):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, recipient: str, message: str) -> bool:
        print(f"[Slack] #{recipient} に送信: {message}")
        # Webhook POST処理...
        return True

    def get_channel_name(self) -> str:
        return "Slack"


class SMSSender(NotificationSender):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def send(self, recipient: str, message: str) -> bool:
        print(f"[SMS] {recipient} に送信: {message}")
        # SMS API呼び出し...
        return True

    def get_channel_name(self) -> str:
        return "SMS"


class LineSender(NotificationSender):
    def __init__(self, channel_token: str):
        self.channel_token = channel_token

    def send(self, recipient: str, message: str) -> bool:
        print(f"[LINE] {recipient} に送信: {message}")
        return True

    def get_channel_name(self) -> str:
        return "LINE"


# ポリモーフィズムを活用する通知サービス
class NotificationService:
    """通知の送信を統一的に管理"""

    def __init__(self):
        self._senders: list[NotificationSender] = []

    def register_sender(self, sender: NotificationSender) -> None:
        self._senders.append(sender)

    def broadcast(self, message: str, recipients: dict[str, str]) -> dict:
        """全チャネルで通知を送信"""
        results = {}
        for sender in self._senders:
            channel = sender.get_channel_name()
            if channel in recipients:
                success = sender.send(recipients[channel], message)
                results[channel] = success
        return results


# 使用例: 送信者の具体的な型を知らなくても動作する
service = NotificationService()
service.register_sender(EmailSender("smtp.example.com"))
service.register_sender(SlackSender("https://hooks.slack.com/xxx"))
service.register_sender(SMSSender("api-key-123"))
service.register_sender(LineSender("channel-token-abc"))

results = service.broadcast(
    "サーバー障害が発生しました",
    {
        "メール": "admin@example.com",
        "Slack": "alerts",
        "SMS": "090-1234-5678",
        "LINE": "U1234567890"
    }
)
```

```python
# 2. Protocol（構造的部分型）によるポリモーフィズム（Python 3.8+）

from typing import Protocol, runtime_checkable


@runtime_checkable
class Serializable(Protocol):
    """シリアライズ可能なオブジェクトのプロトコル"""
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...


@runtime_checkable
class Validatable(Protocol):
    """バリデーション可能なオブジェクトのプロトコル"""
    def validate(self) -> list[str]: ...
    def is_valid(self) -> bool: ...


# Protocol を満たすクラス（明示的な継承は不要）
class UserProfile:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def to_dict(self) -> dict:
        return {"name": self.name, "age": self.age, "email": self.email}

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def validate(self) -> list[str]:
        errors = []
        if not self.name:
            errors.append("名前は必須です")
        if self.age < 0 or self.age > 150:
            errors.append("年齢が不正です")
        if "@" not in self.email:
            errors.append("メールアドレスが不正です")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# Protocol を使った汎用関数
def save_to_file(obj: Serializable, filepath: str) -> None:
    """Serializableなオブジェクトをファイルに保存"""
    with open(filepath, 'w') as f:
        f.write(obj.to_json())

def validate_and_report(obj: Validatable) -> None:
    """Validatableなオブジェクトを検証してレポート"""
    errors = obj.validate()
    if errors:
        print(f"バリデーションエラー: {', '.join(errors)}")
    else:
        print("バリデーション成功")

# 使用例
user = UserProfile("田中太郎", 30, "tanaka@example.com")
print(isinstance(user, Serializable))   # True（構造的部分型）
print(isinstance(user, Validatable))    # True
save_to_file(user, "/tmp/user.json")
validate_and_report(user)
```

```typescript
// TypeScript でのポリモーフィズム（構造的部分型が標準）

// インターフェース定義
interface Shape {
    area(): number;
    perimeter(): number;
    describe(): string;
}

interface Drawable {
    draw(ctx: CanvasRenderingContext2D): void;
}

// 複数のインターフェースを実装
class Circle implements Shape, Drawable {
    constructor(
        public readonly centerX: number,
        public readonly centerY: number,
        public readonly radius: number
    ) {}

    area(): number {
        return Math.PI * this.radius ** 2;
    }

    perimeter(): number {
        return 2 * Math.PI * this.radius;
    }

    describe(): string {
        return `円（半径: ${this.radius}）`;
    }

    draw(ctx: CanvasRenderingContext2D): void {
        ctx.beginPath();
        ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
        ctx.stroke();
    }
}

class Rectangle implements Shape, Drawable {
    constructor(
        public readonly x: number,
        public readonly y: number,
        public readonly width: number,
        public readonly height: number
    ) {}

    area(): number {
        return this.width * this.height;
    }

    perimeter(): number {
        return 2 * (this.width + this.height);
    }

    describe(): string {
        return `長方形（${this.width} x ${this.height}）`;
    }

    draw(ctx: CanvasRenderingContext2D): void {
        ctx.strokeRect(this.x, this.y, this.width, this.height);
    }
}

class Triangle implements Shape {
    constructor(
        public readonly a: number,
        public readonly b: number,
        public readonly c: number
    ) {}

    area(): number {
        // ヘロンの公式
        const s = (this.a + this.b + this.c) / 2;
        return Math.sqrt(s * (s - this.a) * (s - this.b) * (s - this.c));
    }

    perimeter(): number {
        return this.a + this.b + this.c;
    }

    describe(): string {
        return `三角形（辺: ${this.a}, ${this.b}, ${this.c}）`;
    }
}

// ポリモーフィズムを活用した関数
function printShapeInfo(shapes: Shape[]): void {
    for (const shape of shapes) {
        console.log(`${shape.describe()}: 面積=${shape.area().toFixed(2)}, 周長=${shape.perimeter().toFixed(2)}`);
    }
}

function totalArea(shapes: Shape[]): number {
    return shapes.reduce((sum, shape) => sum + shape.area(), 0);
}
```

### 2.4 抽象化（Abstraction）

抽象化は、複雑な実装の詳細を隠蔽し、本質的なインターフェースだけを公開する概念である。利用者は「何ができるか」だけを知っていればよく、「どのように実現するか」は知る必要がない。

```python
# === 抽象化の実践例: データベースアクセス層 ===

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class QueryResult:
    """クエリ結果の抽象表現"""
    rows: list[dict[str, Any]]
    row_count: int
    affected_rows: int = 0


class DatabaseConnection(ABC):
    """データベース接続の抽象インターフェース

    利用者はこのインターフェースだけを知っていれば、
    具体的なDB（PostgreSQL、MySQL、SQLite等）を意識する必要がない。
    """

    @abstractmethod
    def connect(self) -> None:
        """データベースに接続する"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """接続を切断する"""
        pass

    @abstractmethod
    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        """SQLクエリを実行する"""
        pass

    @abstractmethod
    def begin_transaction(self) -> None:
        """トランザクションを開始する"""
        pass

    @abstractmethod
    def commit(self) -> None:
        """トランザクションをコミットする"""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """トランザクションをロールバックする"""
        pass

    # コンテキストマネージャとしても使える（テンプレートメソッド）
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        self.disconnect()
        return False


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQLの具体的な実装"""

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self._conn = None

    def connect(self) -> None:
        import psycopg2
        self._conn = psycopg2.connect(
            host=self.host, port=self.port,
            dbname=self.dbname, user=self.user, password=self.password
        )

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        cursor = self._conn.cursor()
        cursor.execute(query, params)

        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return QueryResult(rows=rows, row_count=len(rows))
        else:
            return QueryResult(rows=[], row_count=0, affected_rows=cursor.rowcount)

    def begin_transaction(self) -> None:
        self._conn.autocommit = False

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class SQLiteConnection(DatabaseConnection):
    """SQLiteの具体的な実装"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

    def connect(self) -> None:
        import sqlite3
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row

    def disconnect(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: Optional[tuple] = None) -> QueryResult:
        cursor = self._conn.cursor()
        cursor.execute(query, params or ())

        if cursor.description:
            rows = [dict(row) for row in cursor.fetchall()]
            return QueryResult(rows=rows, row_count=len(rows))
        else:
            return QueryResult(rows=[], row_count=0, affected_rows=cursor.rowcount)

    def begin_transaction(self) -> None:
        self._conn.execute("BEGIN")

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


# 抽象化されたインターフェースを使うリポジトリ
class UserRepository:
    """ユーザーリポジトリ - 具体的なDBに依存しない"""

    def __init__(self, db: DatabaseConnection):
        self._db = db  # 抽象に依存（DI）

    def find_by_id(self, user_id: int) -> Optional[dict]:
        result = self._db.execute(
            "SELECT * FROM users WHERE id = %s", (user_id,)
        )
        return result.rows[0] if result.rows else None

    def find_all(self) -> list[dict]:
        result = self._db.execute("SELECT * FROM users ORDER BY id")
        return result.rows

    def create(self, name: str, email: str) -> int:
        result = self._db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id",
            (name, email)
        )
        return result.rows[0]["id"]


# 使用例: DB実装を切り替え可能
# 開発環境
# db = SQLiteConnection(":memory:")
# 本番環境
# db = PostgreSQLConnection("db.example.com", 5432, "myapp", "user", "pass")
# repo = UserRepository(db)
```

---

## 3. SOLID原則

SOLID原則はRobert C. Martin（Uncle Bob）が提唱した5つのオブジェクト指向設計原則である。保守性・拡張性・テスト容易性の高いソフトウェアを作るための指針となる。

### 3.1 S — Single Responsibility Principle（単一責任原則）

```
単一責任原則（SRP）:
  「クラスが変更される理由は、ただ1つであるべき」

  より正確には:
  「クラスは1つのアクターに対してのみ責任を持つべき」
  （アクター = そのクラスの変更を要求する人や組織）
```

```python
# ❌ SRP違反: 1つのクラスが複数の責任を持っている
class User:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def save_to_database(self):
        """データベースに保存 → 永続化の責任"""
        pass

    def send_welcome_email(self):
        """ウェルカムメールを送信 → 通知の責任"""
        pass

    def generate_report(self):
        """レポートを生成 → レポーティングの責任"""
        pass

    def validate(self):
        """バリデーション → 検証の責任"""
        pass


# ✅ SRP準拠: 各クラスが1つの責任のみ持つ
class User:
    """ユーザーのドメインモデル（ビジネスルールのみ）"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email


class UserValidator:
    """ユーザーデータのバリデーション"""
    @staticmethod
    def validate(user: User) -> list[str]:
        errors = []
        if not user.name or len(user.name) < 2:
            errors.append("名前は2文字以上必要です")
        if "@" not in user.email:
            errors.append("有効なメールアドレスを入力してください")
        return errors


class UserRepository:
    """ユーザーの永続化"""
    def __init__(self, db_connection):
        self._db = db_connection

    def save(self, user: User) -> int:
        # DB保存処理
        pass

    def find_by_email(self, email: str) -> Optional[User]:
        # DB検索処理
        pass


class UserNotifier:
    """ユーザーへの通知"""
    def __init__(self, email_service):
        self._email_service = email_service

    def send_welcome(self, user: User) -> None:
        self._email_service.send(
            to=user.email,
            subject="ようこそ！",
            body=f"{user.name}様、会員登録ありがとうございます。"
        )


class UserReportGenerator:
    """ユーザー関連レポートの生成"""
    def generate_activity_report(self, user: User) -> str:
        # レポート生成処理
        pass
```

### 3.2 O — Open/Closed Principle（開放/閉鎖原則）

```
開放/閉鎖原則（OCP）:
  「ソフトウェアエンティティは拡張に対して開いており、
   修正に対して閉じているべき」

  → 新しい機能を追加する際に、既存のコードを変更する必要がないように設計する
```

```python
# ❌ OCP違反: 新しい割引タイプを追加するたびに既存コードを修正
class DiscountCalculator:
    def calculate(self, order_total: int, discount_type: str) -> int:
        if discount_type == "percentage":
            return int(order_total * 0.9)
        elif discount_type == "fixed":
            return order_total - 1000
        elif discount_type == "member":
            return int(order_total * 0.85)
        # 新しい割引を追加するたびにここに elif を追加...
        else:
            return order_total


# ✅ OCP準拠: 新しい割引を追加しても既存コードは変更不要
class DiscountStrategy(ABC):
    """割引戦略の抽象基底クラス"""

    @abstractmethod
    def apply(self, order_total: int) -> int:
        pass

    @abstractmethod
    def description(self) -> str:
        pass


class PercentageDiscount(DiscountStrategy):
    def __init__(self, rate: float):
        self.rate = rate

    def apply(self, order_total: int) -> int:
        return int(order_total * (1 - self.rate))

    def description(self) -> str:
        return f"{int(self.rate * 100)}%割引"


class FixedAmountDiscount(DiscountStrategy):
    def __init__(self, amount: int):
        self.amount = amount

    def apply(self, order_total: int) -> int:
        return max(0, order_total - self.amount)

    def description(self) -> str:
        return f"{self.amount}円引き"


class MemberDiscount(DiscountStrategy):
    def __init__(self, member_rank: str):
        self.member_rank = member_rank
        self._rates = {"gold": 0.15, "silver": 0.10, "bronze": 0.05}

    def apply(self, order_total: int) -> int:
        rate = self._rates.get(self.member_rank, 0)
        return int(order_total * (1 - rate))

    def description(self) -> str:
        return f"会員割引（{self.member_rank}）"


# 新しい割引を追加しても既存コードは一切変更不要
class CouponDiscount(DiscountStrategy):
    """クーポン割引 - 後から追加"""
    def __init__(self, coupon_code: str, discount_rate: float):
        self.coupon_code = coupon_code
        self.discount_rate = discount_rate

    def apply(self, order_total: int) -> int:
        return int(order_total * (1 - self.discount_rate))

    def description(self) -> str:
        return f"クーポン {self.coupon_code}"


class TimeLimitedDiscount(DiscountStrategy):
    """期間限定割引 - さらに後から追加"""
    def __init__(self, name: str, rate: float, end_date: date):
        self._name = name
        self._rate = rate
        self._end_date = end_date

    def apply(self, order_total: int) -> int:
        if date.today() <= self._end_date:
            return int(order_total * (1 - self._rate))
        return order_total

    def description(self) -> str:
        return f"期間限定: {self._name}"


# 割引を適用するコード（変更不要）
class OrderProcessor:
    def apply_discount(self, total: int, strategy: DiscountStrategy) -> int:
        discounted = strategy.apply(total)
        print(f"  {strategy.description()}: {total:,}円 → {discounted:,}円")
        return discounted

    def apply_best_discount(self, total: int, strategies: list[DiscountStrategy]) -> int:
        """最もお得な割引を適用"""
        best = min(strategies, key=lambda s: s.apply(total))
        return self.apply_discount(total, best)
```

### 3.3 L — Liskov Substitution Principle（リスコフの置換原則）

```
リスコフの置換原則（LSP）:
  「サブタイプはそのベースタイプと置換可能でなければならない」

  → 親クラスを使っている箇所で、子クラスに置き換えても
    プログラムの正しさが保たれること
```

```python
# ❌ LSP違反: 典型的な「正方形・長方形」問題
class Rectangle:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value

    def area(self) -> int:
        return self._width * self._height


class Square(Rectangle):
    """正方形 - 幅と高さが常に等しい"""
    def __init__(self, side: int):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value: int):
        self._width = value
        self._height = value  # 高さも変更！

    @Rectangle.height.setter
    def height(self, value: int):
        self._width = value  # 幅も変更！
        self._height = value


# この関数は Rectangle の契約を前提としている
def test_rectangle(rect: Rectangle):
    rect.width = 5
    rect.height = 10
    assert rect.area() == 50  # Square を渡すと失敗！（area = 100）


# ✅ LSP準拠: 共通のインターフェースで設計
class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

class Square(Shape):
    def __init__(self, side: float):
        self.side = side

    def area(self) -> float:
        return self.side ** 2
```

```python
# ❌ LSP違反: もう1つの典型例
class Bird:
    def fly(self):
        print("飛びます")

class Penguin(Bird):
    def fly(self):
        raise NotImplementedError("ペンギンは飛べません！")


# ✅ LSP準拠: インターフェースを適切に分離
class Bird(ABC):
    @abstractmethod
    def move(self) -> str:
        pass

class FlyingBird(Bird):
    def move(self) -> str:
        return "空を飛びます"

    def fly(self) -> str:
        return "羽ばたいて飛行"

class SwimmingBird(Bird):
    def move(self) -> str:
        return "水中を泳ぎます"

    def swim(self) -> str:
        return "水中を泳ぐ"

class Sparrow(FlyingBird):
    pass

class Penguin(SwimmingBird):
    pass

# Bird のリストとして統一的に扱える
birds: list[Bird] = [Sparrow(), Penguin()]
for bird in birds:
    print(bird.move())  # すべて正常に動作
```

### 3.4 I — Interface Segregation Principle（インターフェース分離原則）

```
インターフェース分離原則（ISP）:
  「クライアントに不要なメソッドへの依存を強制してはならない」

  → 大きなインターフェースを、用途ごとに小さなインターフェースに分割する
```

```python
# ❌ ISP違反: 巨大なインターフェース
class Machine(ABC):
    @abstractmethod
    def print_document(self, doc): pass

    @abstractmethod
    def scan_document(self, doc): pass

    @abstractmethod
    def fax_document(self, doc): pass

    @abstractmethod
    def staple_pages(self, pages): pass


# シンプルなプリンタでも全メソッドの実装を強制される
class SimplePrinter(Machine):
    def print_document(self, doc):
        print("印刷中...")

    def scan_document(self, doc):
        raise NotImplementedError("スキャン非対応")  # 不要な実装

    def fax_document(self, doc):
        raise NotImplementedError("FAX非対応")  # 不要な実装

    def staple_pages(self, pages):
        raise NotImplementedError("ステープラー非対応")  # 不要な実装


# ✅ ISP準拠: 小さなインターフェースに分割
class Printer(ABC):
    @abstractmethod
    def print_document(self, doc) -> None: pass

class Scanner(ABC):
    @abstractmethod
    def scan_document(self, doc) -> bytes: pass

class Fax(ABC):
    @abstractmethod
    def fax_document(self, doc, number: str) -> bool: pass

class Stapler(ABC):
    @abstractmethod
    def staple_pages(self, pages) -> None: pass


# 必要なインターフェースだけを実装
class SimplePrinter(Printer):
    def print_document(self, doc) -> None:
        print("印刷中...")

class MultiFunctionPrinter(Printer, Scanner, Fax):
    def print_document(self, doc) -> None:
        print("印刷中...")

    def scan_document(self, doc) -> bytes:
        return b"scanned_data"

    def fax_document(self, doc, number: str) -> bool:
        print(f"FAX送信中... {number}")
        return True

# 関数もピンポイントな型で受け取れる
def do_printing(printer: Printer, doc) -> None:
    printer.print_document(doc)  # SimplePrinter でも MultiFunctionPrinter でもOK
```

### 3.5 D — Dependency Inversion Principle（依存性逆転原則）

```
依存性逆転原則（DIP）:
  「高レベルモジュールは低レベルモジュールに依存してはならない。
   両方とも抽象に依存すべきである」

  → 具体的な実装ではなく、抽象（インターフェース）に依存する
```

```python
# ❌ DIP違反: 高レベルモジュールが低レベルモジュールに直接依存
class MySQLDatabase:
    def query(self, sql: str) -> list:
        # MySQL固有の処理
        pass

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # 具体クラスに直接依存！

    def get_user(self, user_id: int):
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")


# ✅ DIP準拠: 抽象に依存し、依存性注入（DI）を使う
class Database(ABC):
    """データベースの抽象インターフェース"""
    @abstractmethod
    def query(self, sql: str, params: tuple = ()) -> list: pass

    @abstractmethod
    def execute(self, sql: str, params: tuple = ()) -> int: pass


class MySQLDatabase(Database):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def query(self, sql: str, params: tuple = ()) -> list:
        # MySQL固有の処理
        pass

    def execute(self, sql: str, params: tuple = ()) -> int:
        pass


class PostgreSQLDatabase(Database):
    def __init__(self, connection_string: str):
        self._conn_str = connection_string

    def query(self, sql: str, params: tuple = ()) -> list:
        # PostgreSQL固有の処理
        pass

    def execute(self, sql: str, params: tuple = ()) -> int:
        pass


class InMemoryDatabase(Database):
    """テスト用のインメモリDB"""
    def __init__(self):
        self._data: dict[str, list] = {}

    def query(self, sql: str, params: tuple = ()) -> list:
        return []

    def execute(self, sql: str, params: tuple = ()) -> int:
        return 0


class UserService:
    """ユーザーサービス - 抽象に依存"""

    def __init__(self, db: Database):  # 抽象型で受け取る（DI）
        self._db = db

    def get_user(self, user_id: int) -> dict:
        results = self._db.query(
            "SELECT * FROM users WHERE id = %s", (user_id,)
        )
        return results[0] if results else None

    def create_user(self, name: str, email: str) -> int:
        return self._db.execute(
            "INSERT INTO users (name, email) VALUES (%s, %s)",
            (name, email)
        )


# 環境に応じてDIで切り替え
import os
def create_user_service() -> UserService:
    env = os.getenv("APP_ENV", "development")

    if env == "production":
        db = PostgreSQLDatabase("postgresql://prod-server/myapp")
    elif env == "development":
        db = MySQLDatabase("mysql://localhost/myapp_dev")
    else:  # test
        db = InMemoryDatabase()

    return UserService(db)
```

---

## 4. 継承 vs コンポジション

### 4.1 継承の問題点

```python
# === 継承の乱用による問題 ===

# ❌ 深い継承ツリー（Fragile Base Class Problem）
class Animal: pass
class Mammal(Animal): pass
class DomesticMammal(Mammal): pass
class Dog(DomesticMammal): pass
class GuideDog(Dog): pass  # 5段階の継承！

# 問題点:
# 1. 親クラスの変更が全子クラスに波及（脆い基底クラス問題）
# 2. 理解するために全階層を読む必要がある
# 3. 新しい分類が困難（泳ぐ犬? → SwimmingDog をどこに置く?）


# ❌ 多重継承のダイヤモンド問題（Python）
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):  # B と C の両方から継承
    pass

d = D()
d.method()  # "B" が呼ばれる（MROに基づく）
# → 予測困難な振る舞い
print(D.__mro__)  # メソッド解決順序を確認できる
```

### 4.2 コンポジションの利点

```python
# ✅ コンポジション: 機能をパーツとして組み合わせる

# 能力をクラスとして定義
class WalkAbility:
    def walk(self, distance: int) -> str:
        return f"{distance}m 歩きました"

class SwimAbility:
    def swim(self, distance: int) -> str:
        return f"{distance}m 泳ぎました"

class FlyAbility:
    def fly(self, distance: int) -> str:
        return f"{distance}m 飛びました"

class BarkAbility:
    def bark(self) -> str:
        return "ワンワン！"

class GuideAbility:
    def __init__(self):
        self._is_trained = False

    def train(self) -> None:
        self._is_trained = True

    def guide(self, destination: str) -> str:
        if not self._is_trained:
            raise RuntimeError("訓練が完了していません")
        return f"{destination}まで案内します"


# 動物クラスは能力をコンポジションで持つ
class Dog:
    """犬 - 歩行と吠えの能力を持つ"""
    def __init__(self, name: str):
        self.name = name
        self.walker = WalkAbility()
        self.barker = BarkAbility()
        self.swimmer: SwimAbility | None = None  # オプション
        self.guide: GuideAbility | None = None     # オプション

    def make_guide_dog(self) -> None:
        """盲導犬にする"""
        self.guide = GuideAbility()
        self.guide.train()

    def make_swimmer(self) -> None:
        """泳げるようにする"""
        self.swimmer = SwimAbility()


class Duck:
    """鴨 - 歩行、飛行、泳ぎの能力を持つ"""
    def __init__(self, name: str):
        self.name = name
        self.walker = WalkAbility()
        self.flyer = FlyAbility()
        self.swimmer = SwimAbility()


# 使用例
dog = Dog("ポチ")
print(dog.walker.walk(100))    # 100m 歩きました
print(dog.barker.bark())       # ワンワン！

dog.make_guide_dog()
print(dog.guide.guide("駅"))   # 駅まで案内します

duck = Duck("ガーコ")
print(duck.walker.walk(50))    # 50m 歩きました
print(duck.flyer.fly(200))     # 200m 飛びました
print(duck.swimmer.swim(30))   # 30m 泳ぎました
```

### 4.3 継承とコンポジションの使い分け

```
継承 vs コンポジション: 使い分けの指針

  継承を使うべき場面:
  ┌────────────────────────────────────────────────────┐
  │ ✅ 明確な「is-a」関係がある                        │
  │    → Dog is an Animal（犬は動物である）            │
  │ ✅ リスコフの置換原則を満たせる                    │
  │    → サブクラスは親の契約を完全に守る              │
  │ ✅ 継承階層が浅い（2-3段まで）                     │
  │ ✅ フレームワークが要求する場合                    │
  │    → Django の Model, View 等                      │
  └────────────────────────────────────────────────────┘

  コンポジションを使うべき場面:
  ┌────────────────────────────────────────────────────┐
  │ ✅ 「has-a」関係がある                              │
  │    → Car has an Engine（車はエンジンを持つ）       │
  │ ✅ 複数の機能を組み合わせたい                      │
  │ ✅ 実行時に動的に振る舞いを変えたい                │
  │ ✅ テスト時にモックに差し替えたい                  │
  │ ✅ 継承階層が深くなりそうな場合                    │
  └────────────────────────────────────────────────────┘

  迷ったらコンポジションを選べ（GoFの格言）:
  「継承よりコンポジションを好め」
  （Favor composition over inheritance）
```

### 4.4 ミックスイン/トレイトパターン

```python
# ミックスイン: 継承とコンポジションの中間的なアプローチ

import json
import logging
from datetime import datetime


class TimestampMixin:
    """タイムスタンプ機能を付与するミックスイン"""
    created_at: datetime
    updated_at: datetime

    def init_timestamps(self):
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def touch(self):
        """更新日時を現在時刻に更新"""
        self.updated_at = datetime.now()


class SerializableMixin:
    """JSON シリアライズ機能を付与するミックスイン"""

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls.__new__(cls)
        for key, value in data.items():
            setattr(instance, key, value)
        return instance


class LoggableMixin:
    """ロギング機能を付与するミックスイン"""

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    def log_info(self, message: str):
        self._logger.info(f"[{self.__class__.__name__}] {message}")

    def log_error(self, message: str):
        self._logger.error(f"[{self.__class__.__name__}] {message}")


class ValidatableMixin:
    """バリデーション機能を付与するミックスイン"""

    def validate(self) -> list[str]:
        errors = []
        for attr_name in dir(self):
            if attr_name.startswith('validate_'):
                method = getattr(self, attr_name)
                error = method()
                if error:
                    errors.append(error)
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# ミックスインを組み合わせて使う
class Article(TimestampMixin, SerializableMixin, LoggableMixin, ValidatableMixin):
    def __init__(self, title: str, content: str, author: str):
        self.title = title
        self.content = content
        self.author = author
        self.init_timestamps()
        self.log_info(f"記事「{title}」を作成")

    def validate_title(self) -> str | None:
        if not self.title or len(self.title) < 5:
            return "タイトルは5文字以上必要です"
        return None

    def validate_content(self) -> str | None:
        if not self.content or len(self.content) < 100:
            return "本文は100文字以上必要です"
        return None


# 使用例
article = Article("Python入門", "Python is..." * 50, "田中太郎")
print(article.to_json())           # SerializableMixin
print(article.is_valid())          # ValidatableMixin
article.touch()                     # TimestampMixin
article.log_info("記事を更新")     # LoggableMixin
```

---

## 5. 実務でのクラス設計パターン

### 5.1 Value Object（値オブジェクト）

```python
from dataclasses import dataclass


@dataclass(frozen=True)  # 不変オブジェクト
class Money:
    """金額を表す値オブジェクト"""
    amount: int
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("金額は0以上である必要があります")
        if self.currency not in ("JPY", "USD", "EUR"):
            raise ValueError(f"未対応の通貨: {self.currency}")

    def __add__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("異なる通貨は加算できません")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: 'Money') -> 'Money':
        if self.currency != other.currency:
            raise ValueError("異なる通貨は減算できません")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | float) -> 'Money':
        return Money(int(self.amount * factor), self.currency)

    def format(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        elif self.currency == "USD":
            return f"${self.amount / 100:.2f}"
        elif self.currency == "EUR":
            return f"€{self.amount / 100:.2f}"
        return f"{self.amount} {self.currency}"


@dataclass(frozen=True)
class EmailAddress:
    """メールアドレスの値オブジェクト"""
    value: str

    def __post_init__(self):
        if not self._is_valid_email(self.value):
            raise ValueError(f"無効なメールアドレス: {self.value}")

    @staticmethod
    def _is_valid_email(email: str) -> bool:
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @property
    def domain(self) -> str:
        return self.value.split('@')[1]

    @property
    def local_part(self) -> str:
        return self.value.split('@')[0]


@dataclass(frozen=True)
class DateRange:
    """日付範囲の値オブジェクト"""
    start: date
    end: date

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("開始日は終了日より前である必要があります")

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end

    def overlaps(self, other: 'DateRange') -> bool:
        return self.start <= other.end and other.start <= self.end


# 使用例
price = Money(1500)
tax = price * 0.1
total = price + tax
print(total.format())  # ¥1,650

email = EmailAddress("user@example.com")
print(email.domain)  # example.com

period = DateRange(date(2025, 1, 1), date(2025, 12, 31))
print(period.days)  # 364
print(period.contains(date(2025, 6, 15)))  # True
```

### 5.2 Entity（エンティティ）

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4


@dataclass
class OrderItem:
    """注文明細"""
    product_name: str
    unit_price: Money
    quantity: int

    @property
    def subtotal(self) -> Money:
        return self.unit_price * self.quantity


@dataclass
class Order:
    """注文エンティティ - IDで同一性を判断"""
    id: UUID = field(default_factory=uuid4)
    customer_name: str = ""
    items: list[OrderItem] = field(default_factory=list)
    status: str = "draft"
    created_at: datetime = field(default_factory=datetime.now)

    def add_item(self, product_name: str, unit_price: int, quantity: int) -> None:
        if self.status != "draft":
            raise RuntimeError("確定済みの注文には商品を追加できません")
        item = OrderItem(product_name, Money(unit_price), quantity)
        self.items.append(item)

    def remove_item(self, index: int) -> None:
        if self.status != "draft":
            raise RuntimeError("確定済みの注文からは商品を削除できません")
        self.items.pop(index)

    @property
    def total(self) -> Money:
        return Money(sum(item.subtotal.amount for item in self.items))

    def confirm(self) -> None:
        if not self.items:
            raise ValueError("商品がない注文は確定できません")
        if not self.customer_name:
            raise ValueError("顧客名が設定されていません")
        self.status = "confirmed"

    def cancel(self) -> None:
        if self.status == "shipped":
            raise RuntimeError("発送済みの注文はキャンセルできません")
        self.status = "cancelled"

    def ship(self) -> None:
        if self.status != "confirmed":
            raise RuntimeError("確定済みの注文のみ発送できます")
        self.status = "shipped"

    # エンティティの同一性は ID で判断
    def __eq__(self, other) -> bool:
        if not isinstance(other, Order):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


# 使用例
order = Order(customer_name="田中太郎")
order.add_item("ノートPC", 150000, 1)
order.add_item("マウス", 3000, 2)
print(f"合計: {order.total.format()}")  # ¥156,000
order.confirm()
order.ship()
```

### 5.3 Service（サービスクラス）

```python
class OrderService:
    """注文に関するビジネスロジック"""

    def __init__(
        self,
        order_repo: 'OrderRepository',
        product_repo: 'ProductRepository',
        payment_service: 'PaymentService',
        notification_service: 'NotificationService'
    ):
        self._order_repo = order_repo
        self._product_repo = product_repo
        self._payment = payment_service
        self._notification = notification_service

    def place_order(self, customer_id: str, cart_items: list[dict]) -> Order:
        """注文を確定する"""
        # 1. 在庫チェック
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            if not product or product.stock < item['quantity']:
                raise ValueError(f"在庫不足: {item['product_id']}")

        # 2. 注文作成
        order = Order(customer_name=customer_id)
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            order.add_item(product.name, product.price, item['quantity'])

        order.confirm()

        # 3. 決済処理
        payment_result = self._payment.charge(
            customer_id=customer_id,
            amount=order.total.amount
        )
        if not payment_result.success:
            raise RuntimeError(f"決済失敗: {payment_result.error}")

        # 4. 在庫を減らす
        for item in cart_items:
            product = self._product_repo.find_by_id(item['product_id'])
            product.reduce_stock(item['quantity'])
            self._product_repo.save(product)

        # 5. 注文を保存
        self._order_repo.save(order)

        # 6. 通知
        self._notification.send(
            customer_id,
            f"ご注文ありがとうございます。注文ID: {order.id}"
        )

        return order
```

---

## 6. 現代のOOP

### 6.1 データクラスとイミュータビリティ

```python
# Python dataclass
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)  # frozen=True で不変に
class Point:
    x: float
    y: float

    def distance_to(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def translate(self, dx: float, dy: float) -> 'Point':
        """移動した新しいPointを返す（元は変更しない）"""
        return replace(self, x=self.x + dx, y=self.y + dy)


@dataclass(frozen=True)
class Config:
    """アプリケーション設定（不変）"""
    database_url: str
    debug: bool = False
    max_connections: int = 10
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)

    def with_debug(self, debug: bool) -> 'Config':
        """デバッグ設定を変更した新しいConfigを返す"""
        return replace(self, debug=debug)

    def with_max_connections(self, n: int) -> 'Config':
        return replace(self, max_connections=n)
```

```kotlin
// Kotlin data class
data class User(
    val id: Long,
    val name: String,
    val email: String,
    val role: Role = Role.USER
) {
    // data class は equals, hashCode, toString, copy を自動生成

    fun withRole(newRole: Role): User = copy(role = newRole)
}

enum class Role { USER, ADMIN, MODERATOR }

// 使用例
val user = User(1, "田中太郎", "tanaka@example.com")
val admin = user.copy(role = Role.ADMIN)  // 不変的な更新
println(admin)  // User(id=1, name=田中太郎, email=tanaka@example.com, role=ADMIN)
```

```typescript
// TypeScript での不変オブジェクト
interface UserData {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly createdAt: Date;
}

class ImmutableUser implements UserData {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly createdAt: Date;

    constructor(data: UserData) {
        this.id = data.id;
        this.name = data.name;
        this.email = data.email;
        this.createdAt = data.createdAt;
    }

    // 変更時は新しいインスタンスを返す
    withName(name: string): ImmutableUser {
        return new ImmutableUser({ ...this, name });
    }

    withEmail(email: string): ImmutableUser {
        return new ImmutableUser({ ...this, email });
    }
}
```

### 6.2 OOPと関数型の融合

```python
# 現代のOOP: 関数型プログラミングとの融合

from dataclasses import dataclass
from typing import Callable
from functools import reduce


# 1. データクラス + 純粋関数
@dataclass(frozen=True)
class Transaction:
    amount: int
    category: str
    description: str
    is_income: bool


# ビジネスロジックを純粋関数として定義
def total_by_category(
    transactions: list[Transaction],
    category: str
) -> int:
    """指定カテゴリの合計金額を計算する純粋関数"""
    return sum(
        t.amount if t.is_income else -t.amount
        for t in transactions
        if t.category == category
    )


def filter_transactions(
    transactions: list[Transaction],
    predicate: Callable[[Transaction], bool]
) -> list[Transaction]:
    """条件に合う取引を抽出する高階関数"""
    return [t for t in transactions if predicate(t)]


def summarize(transactions: list[Transaction]) -> dict[str, int]:
    """カテゴリ別集計"""
    categories = set(t.category for t in transactions)
    return {
        cat: total_by_category(transactions, cat)
        for cat in categories
    }


# 2. メソッドチェーン（Fluent Interface）
class TransactionQuery:
    """取引データに対するクエリビルダー"""

    def __init__(self, transactions: list[Transaction]):
        self._transactions = transactions

    def income_only(self) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.is_income])

    def expense_only(self) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if not t.is_income])

    def by_category(self, category: str) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.category == category])

    def above(self, amount: int) -> 'TransactionQuery':
        return TransactionQuery([t for t in self._transactions if t.amount > amount])

    def total(self) -> int:
        return sum(t.amount for t in self._transactions)

    def count(self) -> int:
        return len(self._transactions)

    def to_list(self) -> list[Transaction]:
        return self._transactions.copy()


# 使用例: メソッドチェーンで直感的なクエリ
transactions = [
    Transaction(300000, "給与", "月給", True),
    Transaction(80000, "家賃", "マンション", False),
    Transaction(5000, "食費", "スーパー", False),
    Transaction(3000, "食費", "コンビニ", False),
    Transaction(50000, "副業", "フリーランス", True),
]

query = TransactionQuery(transactions)
food_total = query.expense_only().by_category("食費").total()
print(f"食費合計: {food_total}円")  # 8000円

large_expenses = query.expense_only().above(10000).count()
print(f"1万円以上の支出: {large_expenses}件")  # 1件
```

### 6.3 プロトコル指向プログラミング（Swift流）

```python
# Protocol Oriented Programming（POP）のPython実装

from typing import Protocol, runtime_checkable


@runtime_checkable
class Equatable(Protocol):
    def __eq__(self, other: object) -> bool: ...

@runtime_checkable
class Hashable(Equatable, Protocol):
    def __hash__(self) -> int: ...

@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...

@runtime_checkable
class Displayable(Protocol):
    def display(self) -> str: ...

@runtime_checkable
class Persistable(Protocol):
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> 'Persistable': ...


# プロトコルに準拠するクラス（明示的な継承なし）
@dataclass(frozen=True)
class Temperature:
    celsius: float

    @property
    def fahrenheit(self) -> float:
        return self.celsius * 9 / 5 + 32

    def display(self) -> str:
        return f"{self.celsius}°C ({self.fahrenheit:.1f}°F)"

    def __lt__(self, other: 'Temperature') -> bool:
        return self.celsius < other.celsius

    def __le__(self, other: 'Temperature') -> bool:
        return self.celsius <= other.celsius

    def to_dict(self) -> dict:
        return {"celsius": self.celsius}

    @classmethod
    def from_dict(cls, data: dict) -> 'Temperature':
        return cls(celsius=data["celsius"])


# プロトコルベースの汎用関数
def find_max(items: list[Comparable]) -> Comparable:
    return max(items)

def display_all(items: list[Displayable]) -> None:
    for item in items:
        print(item.display())

temps = [Temperature(20), Temperature(35), Temperature(-5)]
display_all(temps)
hottest = find_max(temps)
print(f"最高気温: {hottest.display()}")
```

---

## 7. OOPのアンチパターンと回避策

### 7.1 避けるべきパターン

```
OOPのアンチパターン一覧:

1. God Object（神クラス）
   ❌ 1つのクラスがすべてを管理する
   ✅ 責任ごとにクラスを分割

2. Anemic Domain Model（貧血ドメインモデル）
   ❌ ゲッター/セッターだけのデータ入れ物
   ✅ ビジネスロジックをエンティティ内に持たせる

3. Feature Envy（特性の横恋慕）
   ❌ あるクラスが別クラスのデータに過度にアクセス
   ✅ データを持つクラスにメソッドを移動

4. Shotgun Surgery（散弾銃手術）
   ❌ 1つの変更が多数のクラスに波及
   ✅ 関連する責任を1つのクラスに集約

5. Inappropriate Intimacy（不適切な親密さ）
   ❌ クラス間が密結合で内部詳細に依存
   ✅ インターフェースを通じた疎結合

6. Dead Code（死んだコード）
   ❌ 使われていないクラスやメソッドの放置
   ✅ 定期的に未使用コードを削除

7. Premature Abstraction（早すぎる抽象化）
   ❌ 将来の拡張を見越した過度な抽象化
   ✅ YAGNI原則: 現在必要なものだけを実装

8. Parallel Inheritance Hierarchies（並行継承階層）
   ❌ あるクラスを追加すると別の階層にもクラスが必要
   ✅ コンポジションで統合
```

```python
# ❌ Anemic Domain Model（貧血ドメインモデル）
class UserAnemic:
    """ただのデータ入れ物 - ビジネスロジックがない"""
    def __init__(self):
        self.name = ""
        self.email = ""
        self.status = ""
        self.login_count = 0
        self.last_login = None

# ロジックがサービス層に散在
class UserServiceAnemic:
    def activate_user(self, user: UserAnemic):
        if user.status == "pending":
            user.status = "active"

    def deactivate_user(self, user: UserAnemic):
        if user.status == "active":
            user.status = "inactive"

    def can_login(self, user: UserAnemic) -> bool:
        return user.status == "active"


# ✅ Rich Domain Model（リッチドメインモデル）
class UserRich:
    """ビジネスロジックを内包するリッチなドメインモデル"""

    def __init__(self, name: str, email: str):
        self._name = name
        self._email = email
        self._status = "pending"
        self._login_count = 0
        self._last_login: datetime | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_active(self) -> bool:
        return self._status == "active"

    def activate(self) -> None:
        """アカウントを有効化する"""
        if self._status != "pending":
            raise RuntimeError(f"ステータス '{self._status}' からは有効化できません")
        self._status = "active"

    def deactivate(self, reason: str) -> None:
        """アカウントを無効化する"""
        if self._status != "active":
            raise RuntimeError("アクティブなアカウントのみ無効化できます")
        self._status = "inactive"

    def login(self) -> None:
        """ログイン処理"""
        if not self.is_active:
            raise PermissionError("非アクティブなアカウントではログインできません")
        self._login_count += 1
        self._last_login = datetime.now()

    def can_login(self) -> bool:
        return self.is_active
```

---

## 8. 言語別OOPの特徴比較

```
言語別OOPの特徴:

┌──────────────┬───────────┬──────────┬──────────────┬────────────┐
│              │ Python    │ Java     │ TypeScript   │ Go         │
├──────────────┼───────────┼──────────┼──────────────┼────────────┤
│ クラス       │ あり      │ あり     │ あり         │ struct     │
│ 継承         │ 多重継承  │ 単一継承 │ 単一継承     │ なし       │
│ インター     │ ABC/      │ interface│ interface    │ interface  │
│ フェース     │ Protocol  │          │              │ (暗黙的)   │
│ アクセス制御 │ 慣習(_)   │ 4段階    │ 3段階        │ 大文字/    │
│              │           │          │ (public等)   │ 小文字     │
│ ジェネリクス │ typing    │ あり     │ あり         │ あり(1.18) │
│ データクラス │ dataclass │ record   │ N/A          │ struct     │
│ Null安全性   │ Optional  │ Optional │ strictNull   │ nil + err  │
│ 不変性       │ frozen    │ final    │ readonly     │ N/A        │
│ パターン     │ match     │ sealed   │ discriminated│ switch     │
│ マッチ       │ (3.10+)   │ (21+)    │ union        │ type       │
└──────────────┴───────────┴──────────┴──────────────┴────────────┘

特筆事項:
- Go はクラスがなく、struct + interface + composition で OOP を実現
- Rust は trait ベースで、クラス継承はない
- Swift は Protocol-Oriented Programming を推奨
- Kotlin は data class + sealed class + delegation で簡潔な OOP
```

---

## 9. テスト容易性を考慮したOOP設計

```python
# テスタブルなクラス設計

from abc import ABC, abstractmethod
from typing import Protocol


# === 依存性注入でテスト容易に ===

class Clock(Protocol):
    """時計のプロトコル"""
    def now(self) -> datetime: ...


class RealClock:
    """本番用の時計"""
    def now(self) -> datetime:
        return datetime.now()


class FakeClock:
    """テスト用のフェイク時計"""
    def __init__(self, fixed_time: datetime):
        self._time = fixed_time

    def now(self) -> datetime:
        return self._time

    def advance(self, seconds: int) -> None:
        from datetime import timedelta
        self._time += timedelta(seconds=seconds)


class SessionManager:
    """セッション管理 - テスト容易な設計"""

    def __init__(self, clock: Clock, timeout_minutes: int = 30):
        self._clock = clock  # 時計を注入
        self._timeout_minutes = timeout_minutes
        self._sessions: dict[str, datetime] = {}

    def create_session(self, user_id: str) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = self._clock.now()
        return session_id

    def is_valid(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False

        from datetime import timedelta
        created = self._sessions[session_id]
        elapsed = self._clock.now() - created
        return elapsed < timedelta(minutes=self._timeout_minutes)

    def refresh(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id] = self._clock.now()


# テスト例
def test_session_expiry():
    """セッションの有効期限をテスト"""
    fake_clock = FakeClock(datetime(2025, 1, 1, 12, 0, 0))
    manager = SessionManager(clock=fake_clock, timeout_minutes=30)

    session_id = manager.create_session("user-1")

    # セッション作成直後は有効
    assert manager.is_valid(session_id) == True

    # 29分後はまだ有効
    fake_clock.advance(29 * 60)
    assert manager.is_valid(session_id) == True

    # 31分後は期限切れ
    fake_clock.advance(2 * 60)
    assert manager.is_valid(session_id) == False
    print("テスト通過: セッション有効期限")


def test_session_refresh():
    """セッションのリフレッシュをテスト"""
    fake_clock = FakeClock(datetime(2025, 1, 1, 12, 0, 0))
    manager = SessionManager(clock=fake_clock, timeout_minutes=30)

    session_id = manager.create_session("user-1")

    # 25分後にリフレッシュ
    fake_clock.advance(25 * 60)
    manager.refresh(session_id)

    # リフレッシュから10分後はまだ有効
    fake_clock.advance(10 * 60)
    assert manager.is_valid(session_id) == True
    print("テスト通過: セッションリフレッシュ")


test_session_expiry()
test_session_refresh()
```

---

## 10. OOPを使わない方が良い場面

```
OOPが適さないケース:

1. 小規模スクリプト:
   → 手続き型で関数をいくつか定義するだけで十分
   → クラス設計はオーバーヘッド

2. データ変換パイプライン:
   → 関数型（map/filter/reduce）が適切
   → 状態を持たない変換処理の連鎖

3. 数値計算・科学計算:
   → NumPy等の配列操作が主体
   → OOPの抽象化は性能上のオーバーヘッド

4. シンプルなCLIツール:
   → 引数をパースして処理を実行するだけ
   → クラスは不要な複雑さ

5. 設定ファイルの処理:
   → 辞書（dict）やdataclassで十分
   → メソッドが不要ならクラスにしない

6. ワンショットのデータ処理:
   → スクリプトとして書いた方が明確
   → 再利用性が不要なら抽象化しない

判断基準:
  - 状態管理が必要 → OOP検討
  - 複数の型に共通の振る舞い → OOP（ポリモーフィズム）
  - データ変換が主体 → 関数型
  - 小規模・使い捨て → 手続き型
  - 並行処理が主体 → アクターモデルや CSP
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 4大原則 | カプセル化、継承、ポリモーフィズム、抽象化 |
| SOLID | 5つの設計原則。保守性と拡張性を向上 |
| SRP | 1クラス1責任。変更の理由は1つだけ |
| OCP | 拡張に開き、修正に閉じる。Strategy パターン等で実現 |
| LSP | サブタイプは親と置換可能でなければならない |
| ISP | クライアントに不要なメソッドへの依存を強制しない |
| DIP | 抽象に依存し、具体に依存しない。DIで実現 |
| 継承 vs コンポジション | 「継承よりコンポジション」。柔軟性と疎結合を重視 |
| 値オブジェクト | 不変、等価性で比較、ドメインの概念を表現 |
| エンティティ | ID で識別、状態を持つ、ライフサイクルがある |
| 現代OOP | 関数型との融合、不変性重視、軽量データクラス |
| テスト容易性 | DI、Protocol、テストダブルで検証しやすい設計 |
| アンチパターン | God Object、Anemic Model、Feature Envy を避ける |

---

## 次に読むべきガイド
→ [[02-functional.md]] — 関数型プログラミング

---

## 参考文献
1. Martin, R. C. "Clean Architecture." Prentice Hall, 2017.
2. Gamma, E. et al. "Design Patterns: Elements of Reusable Object-Oriented Software (GoF)." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
4. Martin, R. C. "Agile Software Development, Principles, Patterns, and Practices." Prentice Hall, 2002.
5. Evans, E. "Domain-Driven Design: Tackling Complexity in the Heart of Software." Addison-Wesley, 2003.
6. Freeman, S. and Pryce, N. "Growing Object-Oriented Software, Guided by Tests." Addison-Wesley, 2009.
7. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN Notices, 1993.
8. Liskov, B. "Data Abstraction and Hierarchy." ACM SIGPLAN Notices, 1988.
9. Meyer, B. "Object-Oriented Software Construction." 2nd Edition, Prentice Hall, 1997.
10. Fowler, M. "Refactoring: Improving the Design of Existing Code." 2nd Edition, Addison-Wesley, 2018.
