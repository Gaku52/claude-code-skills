# SOLID原則 ── オブジェクト指向設計の5大原則

> SOLID原則は、変更に強く拡張しやすいソフトウェアを設計するための5つの基本原則である。Robert C. Martinが提唱し、Michael Feathersが頭文字をとって命名した。

---

## この章で学ぶこと

1. **SOLID各原則の意味と目的** ── SRP、OCP、LSP、ISP、DIPの本質を理解する
2. **原則違反の兆候と影響** ── 違反が引き起こす設計上の問題を把握する
3. **各原則の実践的な適用方法** ── 具体的なコード例で正しい設計を身につける

---

## 1. SOLID原則の全体像

```
+------------------------------------------------------------------+
|                    SOLID 原則                                     |
+------------------------------------------------------------------+
| S - Single Responsibility Principle (単一責任の原則)              |
|     → クラスを変更する理由は1つだけにせよ                         |
+------------------------------------------------------------------+
| O - Open/Closed Principle (開放/閉鎖の原則)                       |
|     → 拡張に開き、修正に閉じよ                                   |
+------------------------------------------------------------------+
| L - Liskov Substitution Principle (リスコフの置換原則)            |
|     → 子クラスは親クラスと置換可能であれ                          |
+------------------------------------------------------------------+
| I - Interface Segregation Principle (インターフェース分離の原則)  |
|     → クライアントが使わないメソッドに依存させるな                |
+------------------------------------------------------------------+
| D - Dependency Inversion Principle (依存性逆転の原則)             |
|     → 抽象に依存し、具象に依存するな                              |
+------------------------------------------------------------------+
```

---

## 2. S ── 単一責任の原則 (SRP)

### 定義

> 「クラスを変更する理由は、たった1つだけであるべきだ」── Robert C. Martin

```
   変更理由が複数あるクラス        SRP適用後
   ┌─────────────────┐      ┌──────────────┐
   │   Employee       │      │  Employee     │
   │  ─────────────   │      │  ──────────── │
   │  calculatePay()  │──→   │  getName()    │
   │  generateReport()│      │  getDept()    │
   │  saveToDatabase() │     └──────────────┘
   └─────────────────┘      ┌──────────────┐
    変更理由: 3つ             │ PayCalculator │
    ・給与計算ルール          │  ──────────── │
    ・レポート形式            │  calculate()  │
    ・DB保存方法              └──────────────┘
                              ┌──────────────┐
                              │ ReportGenerator│
                              │  ──────────── │
                              │  generate()   │
                              └──────────────┘
                              ┌──────────────┐
                              │ EmployeeRepo  │
                              │  ──────────── │
                              │  save()       │
                              └──────────────┘
                              変更理由: 各1つ
```

**コード例1: SRP違反と改善**

```python
# SRP違反: Userクラスが認証・永続化・通知すべてを担当
class User:
    def __init__(self, name, email, password):
        self.name = name
        self.email = email
        self.password = password

    def authenticate(self, password):
        return bcrypt.check(self.password, password)

    def save(self):
        db.execute("INSERT INTO users ...", self.name, self.email)

    def send_welcome_email(self):
        smtp.send(self.email, "Welcome!", f"こんにちは {self.name}")


# SRP適用: 各責任を専用クラスに分離
class User:
    """ユーザーのドメインモデル（データ表現のみ）"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email


class AuthenticationService:
    """認証ロジックを担当"""
    def authenticate(self, user: User, password: str) -> bool:
        stored_hash = self.credential_store.get_hash(user.email)
        return bcrypt.check(stored_hash, password)


class UserRepository:
    """ユーザーの永続化を担当"""
    def save(self, user: User) -> None:
        self.db.execute("INSERT INTO users ...", user.name, user.email)


class NotificationService:
    """通知送信を担当"""
    def send_welcome(self, user: User) -> None:
        self.mailer.send(user.email, "Welcome!", f"こんにちは {user.name}")
```

---

## 3. O ── 開放/閉鎖の原則 (OCP)

### 定義

> 「ソフトウェアの構成要素は拡張に対して開かれ、修正に対して閉じていなければならない」── Bertrand Meyer

**コード例2: OCP違反と改善**

```typescript
// OCP違反: 新しい図形を追加するたびにこのクラスを修正する必要がある
class AreaCalculator {
  calculate(shape: any): number {
    if (shape.type === 'circle') {
      return Math.PI * shape.radius ** 2;
    } else if (shape.type === 'rectangle') {
      return shape.width * shape.height;
    } else if (shape.type === 'triangle') {
      return (shape.base * shape.height) / 2;
    }
    // 新しい図形を追加するたびに if 分岐が増える...
    throw new Error(`Unknown shape: ${shape.type}`);
  }
}

// OCP適用: 新しい図形はクラス追加のみで対応（既存コード変更不要）
interface Shape {
  area(): number;
}

class Circle implements Shape {
  constructor(private radius: number) {}
  area(): number {
    return Math.PI * this.radius ** 2;
  }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number {
    return this.width * this.height;
  }
}

// 新しい図形の追加: 既存コードを一切変更しない
class Pentagon implements Shape {
  constructor(private side: number) {}
  area(): number {
    return (Math.sqrt(5 * (5 + 2 * Math.sqrt(5))) / 4) * this.side ** 2;
  }
}

class AreaCalculator {
  calculate(shape: Shape): number {
    return shape.area();  // 多態性で処理を委譲
  }
}
```

---

## 4. L ── リスコフの置換原則 (LSP)

### 定義

> 「S が T の派生型であれば、プログラム中で T 型のオブジェクトを S 型のオブジェクトに置換しても、プログラムの性質は変わらない」── Barbara Liskov

**コード例3: LSP違反の典型例（Rectangle/Square問題）**

```python
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


# LSP違反: Square は Rectangle の契約を破る
class Square(Rectangle):
    def __init__(self, side: int):
        super().__init__(side, side)

    @Rectangle.width.setter
    def width(self, value: int):
        self._width = value
        self._height = value  # 幅を変えると高さも変わる！

    @Rectangle.height.setter
    def height(self, value: int):
        self._width = value
        self._height = value

# この関数は Rectangle の契約を前提としている
def test_area(rect: Rectangle):
    rect.width = 5
    rect.height = 4
    assert rect.area() == 20  # Square だと失敗！(5*5=25)


# LSP準拠: 共通インターフェースで設計
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> int:
        pass

class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    def area(self) -> int:
        return self._width * self._height

class Square(Shape):
    def __init__(self, side: int):
        self._side = side

    def area(self) -> int:
        return self._side ** 2
```

---

## 5. I ── インターフェース分離の原則 (ISP)

### 定義

> 「クライアントは自分が利用しないメソッドに依存することを強制されるべきではない」

**コード例4: ISP違反と改善**

```java
// ISP違反: 巨大なインターフェース
interface Worker {
    void work();
    void eat();
    void sleep();
    void attendMeeting();
    void writeReport();
}

// ロボットはeat/sleepできないが、実装を強制される
class Robot implements Worker {
    public void work() { /* 作業する */ }
    public void eat() { throw new UnsupportedOperationException(); }   // 不要
    public void sleep() { throw new UnsupportedOperationException(); } // 不要
    public void attendMeeting() { throw new UnsupportedOperationException(); }
    public void writeReport() { throw new UnsupportedOperationException(); }
}


// ISP適用: 役割ごとにインターフェースを分離
interface Workable {
    void work();
}

interface Feedable {
    void eat();
}

interface Restable {
    void sleep();
}

interface Communicable {
    void attendMeeting();
    void writeReport();
}

// 人間: すべてを実装
class HumanWorker implements Workable, Feedable, Restable, Communicable {
    public void work() { /* 作業する */ }
    public void eat() { /* 食事する */ }
    public void sleep() { /* 睡眠する */ }
    public void attendMeeting() { /* 会議に出る */ }
    public void writeReport() { /* レポートを書く */ }
}

// ロボット: 必要なものだけ実装
class RobotWorker implements Workable {
    public void work() { /* 作業する */ }
}
```

---

## 6. D ── 依存性逆転の原則 (DIP)

### 定義

> 「上位モジュールは下位モジュールに依存してはならない。両者とも抽象に依存すべきである」

```
  DIP違反                        DIP適用
  ┌──────────┐                  ┌──────────┐
  │ OrderSvc  │                 │ OrderSvc  │
  └─────┬─────┘                 └─────┬─────┘
        │ 直接依存                     │ 抽象に依存
        v                             v
  ┌──────────┐              ┌────────────────┐
  │ MySQLRepo │              │ <<interface>>   │
  └──────────┘              │ OrderRepository │
                             └───────┬────────┘
                                     │ 実装
                          ┌──────────┼──────────┐
                          v          v          v
                     MySQLRepo  PostgresRepo  InMemoryRepo
```

**コード例5: DIP違反と改善**

```python
# DIP違反: 上位モジュールが具象クラスに直接依存
class OrderService:
    def __init__(self):
        self.repository = MySQLOrderRepository()  # 具象への直接依存
        self.notifier = EmailNotifier()            # 具象への直接依存

    def place_order(self, order):
        self.repository.save(order)
        self.notifier.notify(order.customer, "注文を受け付けました")


# DIP適用: 抽象（インターフェース）に依存
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: Order) -> None:
        pass

class Notifier(ABC):
    @abstractmethod
    def notify(self, recipient: str, message: str) -> None:
        pass

class OrderService:
    def __init__(self, repository: OrderRepository, notifier: Notifier):
        self.repository = repository  # 抽象に依存
        self.notifier = notifier      # 抽象に依存

    def place_order(self, order: Order) -> None:
        self.repository.save(order)
        self.notifier.notify(order.customer, "注文を受け付けました")

# 利用時: 具象を注入（依存性注入）
service = OrderService(
    repository=PostgreSQLOrderRepository(connection_string),
    notifier=SlackNotifier(webhook_url)
)
```

---

## 7. SOLID原則の相互関係

| 原則 | 主な焦点 | 他の原則との関係 |
|------|----------|------------------|
| SRP | クラスの責任範囲 | ISPのクラス版。凝集度を高める |
| OCP | 拡張の柔軟性 | DIPと組み合わせて多態性で実現 |
| LSP | 継承の正しさ | OCPの前提条件。型安全性を保証 |
| ISP | インターフェースの粒度 | SRPのインターフェース版 |
| DIP | 依存の方向 | OCPを実現するための手段 |

---

## 8. 適用の判断基準

| 状況 | SOLID適用 | 過度な適用を避ける |
|------|-----------|-------------------|
| 頻繁に変更される箇所 | 積極的に適用 | ── |
| 安定したユーティリティ | 最低限で十分 | 過度な抽象化はYAGNI違反 |
| プロトタイプ/PoC | 後回しでよい | 設計に時間をかけすぎない |
| チーム開発のコアロジック | 必須 | ── |
| 1回限りのスクリプト | 不要 | オーバーエンジニアリング |

---

## 9. アンチパターン

### アンチパターン1: God Class（SRP違反の極致）

```python
# 1つのクラスに全責任を詰め込む
class Application:
    def authenticate_user(self): ...
    def process_payment(self): ...
    def generate_report(self): ...
    def send_notification(self): ...
    def validate_input(self): ...
    def manage_cache(self): ...
    def handle_logging(self): ...
    # 1000行以上のメソッドが続く...
```

### アンチパターン2: 過度な抽象化（SOLID原理主義）

```java
// 1メソッドのためにインターフェース + 実装 + ファクトリ + DI設定
interface StringFormatter { String format(String s); }
class UpperCaseFormatter implements StringFormatter { ... }
class StringFormatterFactory { ... }
class StringFormatterConfig { ... }
// 実際にはただの s.toUpperCase() で十分
```

---

## 10. FAQ

### Q1: SOLID原則はすべて同時に適用すべきか？

すべてを一度に適用する必要はない。まずSRPから始め、変更が多い箇所にOCPとDIPを適用するのが実践的。プロジェクトの規模と変更頻度に応じて段階的に導入する。

### Q2: SOLIDは関数型プログラミングにも適用できるか？

概念的には適用可能。SRPは「関数は1つのことをする」、OCPは「高階関数で拡張する」、DIPは「関数の注入」に対応する。ただし用語はOOP文脈で定義されたものなので、関数型では別の原則名（純粋性、合成可能性など）で語られることが多い。

### Q3: LSP違反をどうやって検出するか？

- 派生クラスで例外をスローしている
- `instanceof` / `typeof` チェックが増えている
- 基底クラスの事前条件を強化、事後条件を弱化している
- 「is-a」関係が成り立たない継承がある

---

## まとめ

| 原則 | 一言で | 違反の兆候 |
|------|--------|-----------|
| SRP | 変更理由は1つ | 巨大クラス、頻繁な修正 |
| OCP | 拡張で対応 | if/switch の増殖 |
| LSP | 置換可能 | instanceof チェック |
| ISP | 小さなIF | 空のメソッド実装 |
| DIP | 抽象に依存 | new の直接呼び出し |

---

## 次に読むべきガイド

- [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) ── 重複排除と単純化の原則
- [結合度と凝集度](./03-coupling-cohesion.md) ── モジュール設計の基盤
- [合成 vs 継承](../03-practices-advanced/01-composition-over-inheritance.md) ── LSPの先にある設計判断

---

## 参考文献

1. **Robert C. Martin** 『Agile Software Development: Principles, Patterns, and Practices』 Prentice Hall, 2002
2. **Robert C. Martin** 『Clean Architecture: A Craftsman's Guide to Software Structure and Design』 Prentice Hall, 2017
3. **Barbara Liskov, Jeannette Wing** "A Behavioral Notion of Subtyping" ACM Transactions on Programming Languages and Systems, 1994
4. **Bertrand Meyer** 『Object-Oriented Software Construction』 Prentice Hall, 1997 (2nd Edition)
