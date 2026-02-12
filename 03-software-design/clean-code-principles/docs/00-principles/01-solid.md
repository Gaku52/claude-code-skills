# SOLID原則 ── オブジェクト指向設計の5大原則

> SOLID原則は、変更に強く拡張しやすいソフトウェアを設計するための5つの基本原則である。Robert C. Martinが提唱し、Michael Feathersが頭文字をとって命名した。

---

## この章で学ぶこと

1. **SOLID各原則の意味と目的** ── SRP、OCP、LSP、ISP、DIPの本質を理解する
2. **原則違反の兆候と影響** ── 違反が引き起こす設計上の問題を把握する
3. **各原則の実践的な適用方法** ── 具体的なコード例で正しい設計を身につける
4. **原則間の相互関係** ── 5つの原則がどのように連携し補完し合うかを理解する
5. **適用の判断基準** ── 過度な適用を避け、実践的なバランス感覚を身につける

---

## 前提知識

| 前提知識 | 説明 | 参照リンク |
|---------|------|-----------|
| オブジェクト指向プログラミング | クラス、継承、ポリモーフィズム、インターフェース | `../../02-programming/` |
| クリーンコード概要 | コード品質の基本概念と測定指標 | [00-clean-code-overview.md](./00-clean-code-overview.md) |
| 抽象化の概念 | 抽象クラス、インターフェース、依存関係 | `../../02-programming/` |

---

## 1. SOLID原則の全体像

### 1.1 各原則の概要

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

### 1.2 なぜSOLID原則が必要なのか ── WHYの深掘り

ソフトウェアは「最初に動くものを作る」だけなら比較的簡単だが、「変更し続けられるものを作る」のが困難である。SOLID原則が解決する根本的な問題は、**変更の波及**である。

```
  変更の波及モデル

  SOLID原則なし                    SOLID原則あり
  ┌──────────────────┐            ┌──────────────────┐
  │    変更要求       │            │    変更要求       │
  │      │           │            │      │           │
  │      v           │            │      v           │
  │  ┌───────┐       │            │  ┌───────┐       │
  │  │ ClassA│       │            │  │ ClassA│       │
  │  └───┬───┘       │            │  └───────┘       │
  │      │ 波及      │            │  (変更はここで完結)│
  │  ┌───┼───┐       │            │                  │
  │  v   v   v       │            │  ClassB, ClassC  │
  │  B   C   D       │            │  → 影響なし      │
  │  │   │   │       │            │                  │
  │  v   v   v       │            │                  │
  │  E   F   G       │            │                  │
  │  (6クラスに波及)  │            │  (1クラスのみ変更)│
  └──────────────────┘            └──────────────────┘
```

SOLID原則の各原則が解決する具体的な問題:

| 原則 | 解決する問題 | 違反した場合の症状 |
|------|------------|------------------|
| SRP | 1つの変更が無関係な機能に影響する | 頻繁な予期しないバグ |
| OCP | 新機能追加のたびに既存コードを修正する | if/switch分岐の増殖 |
| LSP | 派生クラスが基底クラスの前提を破る | instanceof チェックの増加 |
| ISP | 不要なメソッドへの依存を強制される | 空のメソッド実装 |
| DIP | 具象クラスへの直接依存でテスト困難 | モック作成が不可能 |

### 1.3 SOLID原則の歴史的背景

SOLID原則の各原則は、それぞれ異なる時代に異なる研究者により提唱された。

```
  タイムライン

  1988  Barbara Liskov  → LSP の原型論文
  1994  Liskov & Wing   → LSP の正式定義
  1996  Robert C. Martin → OCP, DIP を論文発表
  1997  Bertrand Meyer  → OCP の先駆的記述（Object-Oriented Software Construction）
  2000  Robert C. Martin → SRP, ISP を体系化
  2004  Michael Feathers → 5原則を "SOLID" と命名
  2017  Robert C. Martin → Clean Architecture で SOLID を再定義
```

---

## 2. S ── 単一責任の原則 (SRP)

### 2.1 定義

> 「クラスを変更する理由は、たった1つだけであるべきだ」── Robert C. Martin

Robert C. Martin は後に定義を洗練させ、「変更理由」を「アクター」として再定義した:

> 「モジュールはたった1つのアクター（利害関係者）に対して責任を負うべきだ」── Clean Architecture (2017)

この再定義により、「変更理由」の曖昧さが解消された。アクターとは、そのコードの変更を要求しうる人やグループのことである。

```
   変更理由が複数あるクラス        SRP適用後
   ┌─────────────────┐      ┌──────────────┐
   │   Employee       │      │  Employee     │
   │  ─────────────   │      │  ──────────── │
   │  calculatePay()  │──→   │  getName()    │
   │  generateReport()│      │  getDept()    │
   │  saveToDatabase() │     └──────────────┘
   └─────────────────┘      ┌──────────────┐
    アクター: 3つ             │ PayCalculator │
    ・CFO(給与計算ルール)     │  ──────────── │
    ・COO(レポート形式)       │  calculate()  │
    ・CTO(DB保存方法)         └──────────────┘
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
                              アクター: 各1つ
```

### 2.2 SRP違反の検出方法

SRP違反を検出するための実践的なチェックリスト:

```
  SRP違反チェックリスト

  □ クラス名に「And」「Or」「Manager」「Handler」が含まれる
  □ クラスの説明に「〜して、〜して、〜する」が必要
  □ import文が10個以上ある
  □ クラスが200行を超えている
  □ テスト時に無関係なモック/スタブが必要
  □ 異なるチーム/部門から変更要求が来る
  □ 変更のたびに無関係なテストが壊れる
```

### 2.3 コード例

**コード例1: SRP違反と改善 ── ユーザー管理**

```python
# SRP違反: Userクラスが認証・永続化・通知すべてを担当
class User:
    def __init__(self, name: str, email: str, password: str):
        self.name = name
        self.email = email
        self.password = password

    def authenticate(self, password: str) -> bool:
        """認証ロジック（セキュリティチームが管理）"""
        return bcrypt.check(self.password, password)

    def save(self) -> None:
        """永続化ロジック（インフラチームが管理）"""
        db.execute("INSERT INTO users ...", self.name, self.email)

    def send_welcome_email(self) -> None:
        """通知ロジック（マーケティングチームが管理）"""
        smtp.send(self.email, "Welcome!", f"こんにちは {self.name}")


# SRP適用: 各責任を専用クラスに分離
class User:
    """ユーザーのドメインモデル（データ表現のみ）"""
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email


class AuthenticationService:
    """認証ロジックを担当（アクター: セキュリティチーム）"""
    def __init__(self, credential_store: "CredentialStore"):
        self.credential_store = credential_store

    def authenticate(self, user: User, password: str) -> bool:
        stored_hash = self.credential_store.get_hash(user.email)
        return bcrypt.check(stored_hash, password)


class UserRepository:
    """ユーザーの永続化を担当（アクター: インフラチーム）"""
    def __init__(self, db: "Database"):
        self.db = db

    def save(self, user: User) -> None:
        self.db.execute("INSERT INTO users ...", user.name, user.email)

    def find_by_email(self, email: str) -> User | None:
        row = self.db.query("SELECT * FROM users WHERE email = %s", email)
        return User(row['name'], row['email']) if row else None


class NotificationService:
    """通知送信を担当（アクター: マーケティングチーム）"""
    def __init__(self, mailer: "Mailer"):
        self.mailer = mailer

    def send_welcome(self, user: User) -> None:
        self.mailer.send(user.email, "Welcome!", f"こんにちは {user.name}")
```

**コード例2: SRP適用の実践 ── ログ解析**

```python
# SRP違反: 1つのクラスがパース・フィルタ・集計・出力を担当
class LogAnalyzer:
    def analyze(self, log_file: str) -> None:
        # パース
        entries = []
        with open(log_file) as f:
            for line in f:
                parts = line.strip().split(' ')
                entries.append({
                    'timestamp': parts[0],
                    'level': parts[1],
                    'message': ' '.join(parts[2:])
                })

        # フィルタ
        errors = [e for e in entries if e['level'] == 'ERROR']

        # 集計
        counts = {}
        for error in errors:
            msg = error['message'][:50]
            counts[msg] = counts.get(msg, 0) + 1

        # 出力
        for msg, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{count:5d} | {msg}")


# SRP適用: 各責任を分離
from dataclasses import dataclass
from typing import Iterator

@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str

class LogParser:
    """ログファイルのパースを担当"""
    def parse(self, log_file: str) -> list[LogEntry]:
        entries = []
        with open(log_file) as f:
            for line in f:
                entries.append(self._parse_line(line))
        return entries

    def _parse_line(self, line: str) -> LogEntry:
        parts = line.strip().split(' ', 2)
        return LogEntry(
            timestamp=parts[0],
            level=parts[1],
            message=parts[2] if len(parts) > 2 else ''
        )

class LogFilter:
    """ログエントリのフィルタリングを担当"""
    def filter_by_level(
        self, entries: list[LogEntry], level: str
    ) -> list[LogEntry]:
        return [e for e in entries if e.level == level]

class LogAggregator:
    """ログの集計を担当"""
    def count_by_message(
        self, entries: list[LogEntry], prefix_length: int = 50
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            key = entry.message[:prefix_length]
            counts[key] = counts.get(key, 0) + 1
        return counts

class LogReporter:
    """集計結果の出力を担当"""
    def print_summary(self, counts: dict[str, int]) -> None:
        for msg, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{count:5d} | {msg}")
```

---

## 3. O ── 開放/閉鎖の原則 (OCP)

### 3.1 定義

> 「ソフトウェアの構成要素は拡張に対して開かれ、修正に対して閉じていなければならない」── Bertrand Meyer

この原則の本質は、**新しい振る舞いを追加する際に、既存のコードを変更しなくて済む設計**を作ることにある。

### 3.2 OCPの実現手段

OCPを実現するための主要なパターンは3つある:

```
  OCPの実現手段

  ┌─────────────────────────────────────────────────────┐
  │ 1. ポリモーフィズム（最も一般的）                      │
  │    → インターフェースを定義し、実装を差し替え可能にする │
  │                                                     │
  │ 2. ストラテジーパターン                                │
  │    → アルゴリズムをオブジェクトとして注入する          │
  │                                                     │
  │ 3. テンプレートメソッドパターン                        │
  │    → 骨格をベースクラスに定義し、詳細を派生で実装      │
  └─────────────────────────────────────────────────────┘
```

### 3.3 コード例

**コード例3: OCP違反と改善 ── 図形の面積計算**

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
  perimeter(): number;
}

class Circle implements Shape {
  constructor(private radius: number) {}
  area(): number {
    return Math.PI * this.radius ** 2;
  }
  perimeter(): number {
    return 2 * Math.PI * this.radius;
  }
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number {
    return this.width * this.height;
  }
  perimeter(): number {
    return 2 * (this.width + this.height);
  }
}

// 新しい図形の追加: 既存コードを一切変更しない
class Pentagon implements Shape {
  constructor(private side: number) {}
  area(): number {
    return (Math.sqrt(5 * (5 + 2 * Math.sqrt(5))) / 4) * this.side ** 2;
  }
  perimeter(): number {
    return 5 * this.side;
  }
}

class AreaCalculator {
  calculate(shape: Shape): number {
    return shape.area();  // 多態性で処理を委譲
  }

  calculateTotal(shapes: Shape[]): number {
    return shapes.reduce((total, shape) => total + shape.area(), 0);
  }
}
```

**コード例4: OCP適用 ── 割引計算のストラテジーパターン**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class Order:
    subtotal: Decimal
    customer_type: str
    item_count: int

# OCP違反: 新しい割引ルールの追加には既存コードの修正が必要
class DiscountCalculatorBad:
    def calculate(self, order: Order) -> Decimal:
        if order.customer_type == 'vip':
            return order.subtotal * Decimal('0.20')
        elif order.customer_type == 'regular' and order.item_count >= 10:
            return order.subtotal * Decimal('0.10')
        elif order.customer_type == 'employee':
            return order.subtotal * Decimal('0.30')
        # 新しい割引ルール追加のたびに elif が増える
        return Decimal('0')


# OCP適用: ストラテジーパターンで拡張可能に
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, order: Order) -> Decimal:
        """割引額を計算する"""
        pass

    @abstractmethod
    def is_applicable(self, order: Order) -> bool:
        """この割引が適用可能か判定する"""
        pass

class VipDiscount(DiscountStrategy):
    def calculate(self, order: Order) -> Decimal:
        return order.subtotal * Decimal('0.20')

    def is_applicable(self, order: Order) -> bool:
        return order.customer_type == 'vip'

class BulkDiscount(DiscountStrategy):
    MIN_ITEMS = 10
    def calculate(self, order: Order) -> Decimal:
        return order.subtotal * Decimal('0.10')

    def is_applicable(self, order: Order) -> bool:
        return order.item_count >= self.MIN_ITEMS

class EmployeeDiscount(DiscountStrategy):
    def calculate(self, order: Order) -> Decimal:
        return order.subtotal * Decimal('0.30')

    def is_applicable(self, order: Order) -> bool:
        return order.customer_type == 'employee'

# 新しい割引を追加: SeasonalDiscount クラスを作るだけ
class SeasonalDiscount(DiscountStrategy):
    """季節限定割引（新規追加でも既存コード変更なし）"""
    def calculate(self, order: Order) -> Decimal:
        return order.subtotal * Decimal('0.15')

    def is_applicable(self, order: Order) -> bool:
        from datetime import date
        month = date.today().month
        return month in (7, 8, 12)  # 夏と年末

class DiscountCalculator:
    """割引計算のオーケストレーター（修正に閉じている）"""
    def __init__(self, strategies: list[DiscountStrategy]):
        self.strategies = strategies

    def calculate_best_discount(self, order: Order) -> Decimal:
        applicable = [
            s.calculate(order)
            for s in self.strategies
            if s.is_applicable(order)
        ]
        return max(applicable, default=Decimal('0'))

# 使用例: 戦略を注入
calculator = DiscountCalculator([
    VipDiscount(),
    BulkDiscount(),
    EmployeeDiscount(),
    SeasonalDiscount(),  # 新しい戦略を追加するだけ
])
```

---

## 4. L ── リスコフの置換原則 (LSP)

### 4.1 定義

> 「S が T の派生型であれば、プログラム中で T 型のオブジェクトを S 型のオブジェクトに置換しても、プログラムの性質は変わらない」── Barbara Liskov

### 4.2 LSPの契約モデル

LSP を正しく理解するには、「契約による設計（Design by Contract）」の概念が重要である。

```
  契約モデル

  基底クラスが定義する契約:
  ┌───────────────────────────────────────┐
  │  事前条件 (Precondition)              │
  │  → メソッド呼び出し前に満たすべき条件  │
  │  → 派生クラスは事前条件を強化できない  │
  │                                       │
  │  事後条件 (Postcondition)              │
  │  → メソッド呼び出し後に保証される条件  │
  │  → 派生クラスは事後条件を弱化できない  │
  │                                       │
  │  不変条件 (Invariant)                  │
  │  → オブジェクトが常に満たす条件        │
  │  → 派生クラスも維持しなければならない  │
  └───────────────────────────────────────┘

  違反の例:
  ・事前条件の強化: 基底は正の数を受け付けるが、派生は偶数のみ
  ・事後条件の弱化: 基底は非nullを返すが、派生はnullを返す場合がある
  ・不変条件の破壊: 基底はソート済みを保証するが、派生はしない
```

### 4.3 コード例

**コード例5: LSP違反の典型例（Rectangle/Square問題）**

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

    @abstractmethod
    def perimeter(self) -> int:
        pass

class Rectangle(Shape):
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

    def area(self) -> int:
        return self._width * self._height

    def perimeter(self) -> int:
        return 2 * (self._width + self._height)

class Square(Shape):
    def __init__(self, side: int):
        self._side = side

    def area(self) -> int:
        return self._side ** 2

    def perimeter(self) -> int:
        return 4 * self._side
```

**コード例6: LSP違反の実践的な例 ── コレクション**

```java
// LSP違反: ReadOnlyList が List の「追加可能」という契約を破る
class ReadOnlyList<T> extends ArrayList<T> {
    @Override
    public boolean add(T element) {
        throw new UnsupportedOperationException("読み取り専用です");
    }

    @Override
    public T remove(int index) {
        throw new UnsupportedOperationException("読み取り専用です");
    }
}

// List を受け取る関数は add() が使えることを前提としている
void addDefaultItems(List<String> list) {
    list.add("default1");  // ReadOnlyList だと実行時エラー！
    list.add("default2");
}


// LSP準拠: 適切なインターフェースを使い分ける
// Java の標準ライブラリは既にこの区別を提供している
void readItems(Iterable<String> items) {
    // 読み取りのみ → Iterable で十分
    for (String item : items) {
        System.out.println(item);
    }
}

void modifyItems(List<String> items) {
    // 変更が必要 → List を要求（ReadOnlyListは渡されない）
    items.add("new item");
}
```

### 4.4 LSP違反の検出パターン

| 検出パターン | 例 | 対処法 |
|------------|-----|--------|
| `instanceof` チェック | `if (shape instanceof Circle)` | ポリモーフィズムに置換 |
| `UnsupportedOperationException` | `throw new UnsupportedOperationException()` | インターフェース分離（ISP） |
| ダウンキャスト | `(Circle) shape` | 設計の見直し |
| 条件分岐で型判定 | `if (type == "square")` | ストラテジーパターン |
| 派生クラスで事前条件を強化 | 基底は正数、派生は正の偶数のみ | 契約の再設計 |

---

## 5. I ── インターフェース分離の原則 (ISP)

### 5.1 定義

> 「クライアントは自分が利用しないメソッドに依存することを強制されるべきではない」── Robert C. Martin

### 5.2 ISPの内部メカニズム

ISP が解決する問題は「不必要な再コンパイル」と「不必要な再デプロイ」である。クライアントが使わないメソッドを含むインターフェースに依存すると、そのメソッドの変更時にクライアントも影響を受ける。

```
  ISP違反: 太ったインターフェースの問題

  ┌───────────────┐
  │  FatInterface  │
  │  ─────────────│
  │  methodA()     │ ← ClientA が使用
  │  methodB()     │ ← ClientB が使用
  │  methodC()     │ ← ClientC が使用
  └───────┬───────┘
          │
     ┌────┼────┐
     v    v    v
  ClientA ClientB ClientC

  methodB() の変更 → ClientA, ClientC も再コンパイル必要
  （使っていないのに！）


  ISP適用: インターフェースを分離

  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ InterfaceA│  │InterfaceB│  │InterfaceC│
  │ methodA() │  │methodB() │  │methodC() │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       v              v              v
    ClientA       ClientB        ClientC

  methodB() の変更 → ClientB のみ再コンパイル
```

### 5.3 コード例

**コード例7: ISP違反と改善 ── Worker インターフェース**

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
    public void eat() { throw new UnsupportedOperationException(); }   // LSP違反も!
    public void sleep() { throw new UnsupportedOperationException(); }
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

// AIアシスタント: 作業とコミュニケーション
class AiAssistant implements Workable, Communicable {
    public void work() { /* 作業する */ }
    public void attendMeeting() { /* 議事録を取る */ }
    public void writeReport() { /* レポートを生成する */ }
}
```

**コード例8: ISP適用 ── リポジトリインターフェース**

```typescript
// ISP違反: 全CRUD操作を1つのインターフェースに
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<void>;
  update(entity: T): Promise<void>;
  delete(id: string): Promise<void>;
  bulkInsert(entities: T[]): Promise<void>;
  executeRawQuery(sql: string): Promise<any>;
}

// 読み取り専用のレポートサービスでも全メソッドが見える
class ReportService {
  constructor(private repo: Repository<Order>) {}
  // save, delete, executeRawQuery は使わないのに依存している
}


// ISP適用: 用途別にインターフェースを分離
interface Readable<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
}

interface Writable<T> {
  save(entity: T): Promise<void>;
  update(entity: T): Promise<void>;
}

interface Deletable {
  delete(id: string): Promise<void>;
}

interface BulkOperable<T> {
  bulkInsert(entities: T[]): Promise<void>;
}

// 完全なCRUDリポジトリ
interface CrudRepository<T>
  extends Readable<T>, Writable<T>, Deletable {}

// レポートサービスは読み取り専用インターフェースのみに依存
class ReportService {
  constructor(private repo: Readable<Order>) {}

  async generateMonthlyReport(): Promise<Report> {
    const orders = await this.repo.findAll();
    // ... レポート生成ロジック
  }
}

// 管理画面は全機能を利用
class AdminService {
  constructor(private repo: CrudRepository<Order>) {}

  async deleteOrder(id: string): Promise<void> {
    await this.repo.delete(id);
  }
}
```

---

## 6. D ── 依存性逆転の原則 (DIP)

### 6.1 定義

> 「上位モジュールは下位モジュールに依存してはならない。両者とも抽象に依存すべきである」── Robert C. Martin

> 「抽象は詳細に依存してはならない。詳細が抽象に依存すべきである」

### 6.2 DIPの内部メカニズム

DIP は「依存関係の方向を逆転させる」ことで、上位のビジネスロジックを下位のインフラ詳細から独立させる。

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
   具象に直接依存              └───────┬────────┘
   → MySQLを変更すると              │ 実装
     OrderSvcも影響           ┌─────┼─────┐
                              v     v     v
                         MySQL  Postgres InMemory
                          Repo   Repo    Repo
   → どの実装に変えても OrderSvc は影響を受けない
```

### 6.3 コード例

**コード例9: DIP違反と改善 ── 通知システム**

```python
# DIP違反: 上位モジュールが具象クラスに直接依存
class OrderService:
    def __init__(self):
        self.repository = MySQLOrderRepository()  # 具象への直接依存
        self.notifier = EmailNotifier()            # 具象への直接依存
        self.logger = FileLogger()                 # 具象への直接依存

    def place_order(self, order: "Order") -> None:
        self.repository.save(order)
        self.notifier.notify(order.customer, "注文を受け付けました")
        self.logger.log(f"注文 {order.id} を処理しました")


# DIP適用: 抽象（インターフェース）に依存
from abc import ABC, abstractmethod

class OrderRepository(ABC):
    @abstractmethod
    def save(self, order: "Order") -> None:
        pass

    @abstractmethod
    def find_by_id(self, order_id: str) -> "Order | None":
        pass

class Notifier(ABC):
    @abstractmethod
    def notify(self, recipient: str, message: str) -> None:
        pass

class Logger(ABC):
    @abstractmethod
    def log(self, message: str) -> None:
        pass


class OrderService:
    """上位モジュール: 抽象にのみ依存"""
    def __init__(
        self,
        repository: OrderRepository,
        notifier: Notifier,
        logger: Logger
    ):
        self.repository = repository
        self.notifier = notifier
        self.logger = logger

    def place_order(self, order: "Order") -> None:
        self.repository.save(order)
        self.notifier.notify(order.customer, "注文を受け付けました")
        self.logger.log(f"注文 {order.id} を処理しました")


# 下位モジュール: 抽象を実装
class PostgreSQLOrderRepository(OrderRepository):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    def save(self, order: "Order") -> None:
        # PostgreSQL固有の実装
        pass

    def find_by_id(self, order_id: str) -> "Order | None":
        pass

class SlackNotifier(Notifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def notify(self, recipient: str, message: str) -> None:
        # Slack API を使った通知
        pass

class CloudWatchLogger(Logger):
    def log(self, message: str) -> None:
        # AWS CloudWatch への送信
        pass


# 組み立て（Composition Root）
service = OrderService(
    repository=PostgreSQLOrderRepository("postgresql://..."),
    notifier=SlackNotifier("https://hooks.slack.com/..."),
    logger=CloudWatchLogger()
)

# テスト時: モックを注入
class MockRepository(OrderRepository):
    def __init__(self):
        self.saved_orders = []

    def save(self, order):
        self.saved_orders.append(order)

    def find_by_id(self, order_id):
        return next((o for o in self.saved_orders if o.id == order_id), None)

test_service = OrderService(
    repository=MockRepository(),
    notifier=MockNotifier(),
    logger=MockLogger()
)
```

**コード例10: DIP と依存性注入（DI）の関係**

```typescript
// DIP はアーキテクチャ原則、DI はそれを実現する実装手法

// 1. コンストラクタインジェクション（最も推奨）
class UserService {
  constructor(
    private readonly repository: UserRepository,
    private readonly hasher: PasswordHasher,
    private readonly mailer: Mailer
  ) {}

  async register(email: string, password: string): Promise<User> {
    const hashedPassword = await this.hasher.hash(password);
    const user = new User(email, hashedPassword);
    await this.repository.save(user);
    await this.mailer.sendWelcome(email);
    return user;
  }
}

// 2. セッターインジェクション（オプショナルな依存に）
class ReportGenerator {
  private formatter: ReportFormatter = new DefaultFormatter();

  setFormatter(formatter: ReportFormatter): void {
    this.formatter = formatter;
  }
}

// 3. メソッドインジェクション（呼び出し毎に異なる依存）
class DataProcessor {
  process(data: RawData, transformer: DataTransformer): ProcessedData {
    return transformer.transform(data);
  }
}
```

---

## 7. SOLID原則の相互関係

### 7.1 関係図

```
  ┌──────────────────────────────────────────────────────┐
  │              SOLID原則の相互関係                       │
  │                                                      │
  │  ┌─────┐      前提条件      ┌─────┐                  │
  │  │ LSP │ ─────────────────→ │ OCP │                  │
  │  └──┬──┘                    └──┬──┘                  │
  │     │                          │                      │
  │     │ 型安全性                  │ 実現手段             │
  │     │                          │                      │
  │     v                          v                      │
  │  ┌─────┐      IF版          ┌─────┐                  │
  │  │ ISP │ ←──────────────── │ SRP │                  │
  │  └──┬──┘                    └─────┘                  │
  │     │                                                │
  │     │ 依存の最小化                                    │
  │     v                                                │
  │  ┌─────┐                                             │
  │  │ DIP │ ← OCP を実現するための手段                   │
  │  └─────┘                                             │
  └──────────────────────────────────────────────────────┘
```

### 7.2 関係の詳細

| 原則 | 主な焦点 | 他の原則との関係 |
|------|----------|------------------|
| SRP | クラスの責任範囲 | ISPのクラス版。凝集度を高める |
| OCP | 拡張の柔軟性 | DIPと組み合わせて多態性で実現。LSPが前提条件 |
| LSP | 継承の正しさ | OCPの前提条件。型安全性を保証 |
| ISP | インターフェースの粒度 | SRPのインターフェース版。DIPの依存を最小化 |
| DIP | 依存の方向 | OCPを実現するための手段。ISPで依存を最小化 |

### 7.3 実践での組み合わせ

```python
# SOLID原則の組み合わせ例: 通知サービス

# SRP: 通知送信の責任のみ
# OCP: 新しい通知チャネルはクラス追加で対応
# LSP: すべてのNotifierはsendメソッドの契約を守る
# ISP: 同期/非同期を分離
# DIP: 抽象に依存

from abc import ABC, abstractmethod

# ISP: 同期通知と非同期通知を分離
class SyncNotifier(ABC):
    @abstractmethod
    def send(self, recipient: str, message: str) -> bool:
        """同期的にメッセージを送信し、成否を返す"""
        pass

class AsyncNotifier(ABC):
    @abstractmethod
    async def send(self, recipient: str, message: str) -> str:
        """非同期でメッセージを送信し、ジョブIDを返す"""
        pass

# LSP: 各実装はインターフェースの契約を完全に守る
class EmailNotifier(SyncNotifier):
    """SRP: メール送信のみを担当"""
    def __init__(self, smtp_config: dict):
        self.smtp = SmtpClient(smtp_config)

    def send(self, recipient: str, message: str) -> bool:
        try:
            self.smtp.send_mail(recipient, message)
            return True
        except SmtpError:
            return False

class SlackNotifier(AsyncNotifier):
    """SRP: Slack通知のみを担当"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, recipient: str, message: str) -> str:
        response = await http_post(self.webhook_url, {"text": message})
        return response["job_id"]

# OCP: 新しい通知チャネル追加時、既存コードを変更しない
class SmsNotifier(SyncNotifier):
    """新規追加: SMS通知"""
    def __init__(self, api_key: str):
        self.sms_client = SmsClient(api_key)

    def send(self, recipient: str, message: str) -> bool:
        return self.sms_client.send_sms(recipient, message)

# DIP: NotificationService は抽象にのみ依存
class NotificationService:
    def __init__(self, notifiers: list[SyncNotifier]):
        self.notifiers = notifiers

    def notify_all(self, recipient: str, message: str) -> dict[str, bool]:
        results = {}
        for notifier in self.notifiers:
            name = type(notifier).__name__
            results[name] = notifier.send(recipient, message)
        return results
```

---

## 8. 適用の判断基準

### 8.1 適用すべき場面と避けるべき場面

| 状況 | SOLID適用 | 過度な適用を避ける |
|------|-----------|-------------------|
| 頻繁に変更される箇所 | 積極的に適用 | -- |
| 安定したユーティリティ | 最低限で十分 | 過度な抽象化はYAGNI違反 |
| プロトタイプ/PoC | 後回しでよい | 設計に時間をかけすぎない |
| チーム開発のコアロジック | 必須 | -- |
| 1回限りのスクリプト | 不要 | オーバーエンジニアリング |
| ライブラリ/フレームワーク | 必須 | 利用者の使いやすさも考慮 |
| マイクロサービスの境界 | 必須（特にDIP） | サービス内部は適宜 |

### 8.2 段階的適用のガイドライン

```
  SOLID原則の段階的適用フロー

  Step 1: SRP から始める（最も直感的）
  ├── 巨大クラスを見つけたら分割
  └── 「この関数は何をするか1文で説明できるか？」

  Step 2: DIP を適用（テスト容易性の向上）
  ├── 外部サービス依存をインターフェースで抽象化
  └── コンストラクタインジェクションを導入

  Step 3: OCP を意識（変更が多い箇所）
  ├── if/switch の増殖を発見したらポリモーフィズム化
  └── ストラテジーパターンの適用

  Step 4: ISP で微調整
  ├── 太ったインターフェースを分離
  └── クライアントごとに必要最小限のインターフェース

  Step 5: LSP で品質保証
  ├── 継承関係の正当性を検証
  └── 契約テストの追加
```

### 8.3 過度な適用のコスト

```
  SOLID適用のコスト-ベネフィット曲線

  ベネフィット
    ^
    |        *****
    |    ****     ***
    |  **             **
    | *                 *         ← 適度な適用
    |*                   *
    |                     **      ← 過度な適用
    +-------------------------> SOLID適用度
    0%   25%   50%   75%  100%

    0-50%: 適用するほどベネフィット増大
    50-75%: ベネフィットは緩やかに増大
    75-100%: 抽象化のオーバーヘッドがベネフィットを上回る
```

---

## 9. SOLID原則と他のパラダイム

### 9.1 関数型プログラミングとSOLID

| SOLID原則 | 関数型での対応概念 | 説明 |
|-----------|-----------------|------|
| SRP | 純粋関数 | 各関数は1つの変換のみ |
| OCP | 高階関数 | 関数を引数で受け取り動作を拡張 |
| LSP | 参照透過性 | 同じ入力には常に同じ出力 |
| ISP | 型クラス（Haskell） | 必要な振る舞いのみを要求 |
| DIP | 関数の注入 | 具体的な関数ではなく関数型を受け取る |

```python
# 関数型でのOCP: 高階関数による拡張
from typing import Callable

# ソート戦略を関数として注入（OCP + DIP）
def sort_users(
    users: list[dict],
    key_fn: Callable[[dict], any] = lambda u: u['name']
) -> list[dict]:
    return sorted(users, key=key_fn)

# 新しいソート基準の追加: 既存コード変更なし
by_age = lambda u: u['age']
by_score_desc = lambda u: -u['score']

sort_users(users, key_fn=by_age)
sort_users(users, key_fn=by_score_desc)
```

---

## 10. アンチパターン

### アンチパターン1: God Class（SRP違反の極致）

```python
# NG: 1つのクラスに全責任を詰め込む
class Application:
    def authenticate_user(self): ...
    def process_payment(self): ...
    def generate_report(self): ...
    def send_notification(self): ...
    def validate_input(self): ...
    def manage_cache(self): ...
    def handle_logging(self): ...
    # 1000行以上のメソッドが続く...

# OK: 責任ごとにクラスを分割
class AuthService: ...
class PaymentService: ...
class ReportService: ...
class NotificationService: ...
class InputValidator: ...
class CacheManager: ...
class Logger: ...
```

### アンチパターン2: 過度な抽象化（SOLID原理主義）

```java
// NG: 1メソッドのためにインターフェース + 実装 + ファクトリ + DI設定
interface StringFormatter { String format(String s); }
class UpperCaseFormatter implements StringFormatter {
    public String format(String s) { return s.toUpperCase(); }
}
class StringFormatterFactory {
    public StringFormatter create(String type) { ... }
}
class StringFormatterConfig {
    @Bean
    public StringFormatter formatter() { return new UpperCaseFormatter(); }
}
// 実際にはただの s.toUpperCase() で十分

// OK: 必要性が生じたら抽象化する
String formatted = input.toUpperCase();
```

### アンチパターン3: Leaky Abstraction（抽象の漏洩）

```python
# NG: インターフェースがDBの詳細を漏洩
class UserRepository(ABC):
    @abstractmethod
    def find_by_sql(self, sql: str) -> list[User]:
        """SQLクエリでユーザーを検索する"""
        pass  # SQL前提 → RDB以外の実装で困る

    @abstractmethod
    def set_connection_pool_size(self, size: int) -> None:
        """接続プールサイズを設定する"""
        pass  # 接続プール前提 → インメモリ実装で無意味

# OK: ドメインの言葉でインターフェースを定義
class UserRepository(ABC):
    @abstractmethod
    def find_by_email(self, email: str) -> User | None:
        """メールアドレスでユーザーを検索する"""
        pass

    @abstractmethod
    def find_active_users(self, since: datetime) -> list[User]:
        """指定日以降にアクティブなユーザーを検索する"""
        pass
```

---

## 11. 実践演習

### 演習1（基礎）: SRP違反の検出と修正

以下のクラスからSRP違反を特定し、責任を分離せよ。

```python
class ReportManager:
    def __init__(self, db_connection):
        self.db = db_connection

    def fetch_sales_data(self, start_date, end_date):
        query = f"SELECT * FROM sales WHERE date BETWEEN '{start_date}' AND '{end_date}'"
        return self.db.execute(query)

    def calculate_totals(self, sales_data):
        total = sum(item['amount'] for item in sales_data)
        tax = total * 0.10
        return {'subtotal': total, 'tax': tax, 'total': total + tax}

    def format_as_html(self, report_data):
        html = "<html><body>"
        html += f"<h1>売上レポート</h1>"
        html += f"<p>合計: {report_data['total']}円</p>"
        html += "</body></html>"
        return html

    def send_email(self, recipient, html_content):
        import smtplib
        server = smtplib.SMTP('localhost')
        server.sendmail('reports@company.com', recipient, html_content)
        server.quit()

    def generate_and_send(self, start_date, end_date, recipient):
        data = self.fetch_sales_data(start_date, end_date)
        totals = self.calculate_totals(data)
        html = self.format_as_html(totals)
        self.send_email(recipient, html)
```

**期待される分析:**

責任の分離:
- **SalesDataRepository**: データ取得（アクター: DBA/インフラチーム）
- **SalesCalculator**: 集計計算（アクター: 経理部門）
- **HtmlReportFormatter**: HTML整形（アクター: デザインチーム）
- **EmailSender**: メール送信（アクター: インフラチーム）
- **ReportService**: オーケストレーション（責任を持たない調整役）

**期待される出力例:**

```python
class SalesDataRepository:
    def __init__(self, db):
        self.db = db

    def fetch(self, start_date: str, end_date: str) -> list[dict]:
        return self.db.execute(
            "SELECT * FROM sales WHERE date BETWEEN %s AND %s",
            (start_date, end_date)
        )

class SalesCalculator:
    TAX_RATE = Decimal('0.10')

    def calculate_totals(self, sales_data: list[dict]) -> dict:
        subtotal = sum(Decimal(str(item['amount'])) for item in sales_data)
        tax = subtotal * self.TAX_RATE
        return {'subtotal': subtotal, 'tax': tax, 'total': subtotal + tax}

class ReportFormatter(ABC):
    @abstractmethod
    def format(self, report_data: dict) -> str: ...

class HtmlReportFormatter(ReportFormatter):
    def format(self, report_data: dict) -> str:
        return f"""<html><body>
        <h1>売上レポート</h1>
        <p>合計: {report_data['total']}円</p>
        </body></html>"""

class EmailSender:
    def __init__(self, smtp_host: str, from_address: str):
        self.smtp_host = smtp_host
        self.from_address = from_address

    def send(self, recipient: str, content: str) -> None:
        # SMTP送信ロジック
        pass

class ReportService:
    def __init__(self, repo, calculator, formatter, sender):
        self.repo = repo
        self.calculator = calculator
        self.formatter = formatter
        self.sender = sender

    def generate_and_send(self, start_date, end_date, recipient):
        data = self.repo.fetch(start_date, end_date)
        totals = self.calculator.calculate_totals(data)
        content = self.formatter.format(totals)
        self.sender.send(recipient, content)
```

### 演習2（応用）: OCP を適用した拡張設計

以下の決済処理クラスを、新しい決済方法の追加時に既存コードを変更しなくて済むように設計し直せ。

```python
class PaymentProcessor:
    def process(self, payment_method: str, amount: float) -> bool:
        if payment_method == "credit_card":
            # クレジットカード決済ロジック
            return self._process_credit_card(amount)
        elif payment_method == "bank_transfer":
            # 銀行振込ロジック
            return self._process_bank_transfer(amount)
        elif payment_method == "paypal":
            # PayPalロジック
            return self._process_paypal(amount)
        else:
            raise ValueError(f"未対応の決済方法: {payment_method}")
```

**期待される出力例:**

```python
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float) -> bool: ...

    @abstractmethod
    def name(self) -> str: ...

class CreditCardPayment(PaymentMethod):
    def process(self, amount: float) -> bool:
        # クレジットカード決済ロジック
        return True
    def name(self) -> str:
        return "credit_card"

class BankTransferPayment(PaymentMethod):
    def process(self, amount: float) -> bool:
        # 銀行振込ロジック
        return True
    def name(self) -> str:
        return "bank_transfer"

# 新しい決済方法の追加: 既存コード変更なし
class CryptoPayment(PaymentMethod):
    def process(self, amount: float) -> bool:
        # 暗号通貨決済ロジック
        return True
    def name(self) -> str:
        return "crypto"

class PaymentProcessor:
    def __init__(self):
        self._methods: dict[str, PaymentMethod] = {}

    def register(self, method: PaymentMethod) -> None:
        self._methods[method.name()] = method

    def process(self, method_name: str, amount: float) -> bool:
        if method_name not in self._methods:
            raise ValueError(f"未対応の決済方法: {method_name}")
        return self._methods[method_name].process(amount)
```

### 演習3（発展）: SOLID原則を全面適用した設計

以下の要件を、SOLID原則に準拠して設計・実装せよ。

**要件:** 図書館の書籍管理システム
- 書籍の登録・検索・貸出・返却
- 貸出通知（メール/SMS）
- 延滞チェックと罰金計算
- 複数のデータストア対応（DB/ファイル/インメモリ）

**期待される設計の概要:**

```
  SRP: 各クラスが単一の責任
  ├── Book (ドメインモデル)
  ├── BookRepository (永続化)
  ├── LoanService (貸出ビジネスロジック)
  ├── FineCalculator (罰金計算)
  ├── NotificationService (通知送信)
  └── LibraryFacade (オーケストレーション)

  OCP: 通知チャネルの追加にクラス追加のみ
  DIP: LoanService → BookRepository(抽象) に依存
  ISP: 検索用/管理用で別インターフェース
  LSP: すべてのRepository実装が契約を守る
```

---

## 12. FAQ

### Q1: SOLID原則はすべて同時に適用すべきか？

すべてを一度に適用する必要はない。まずSRPから始め、変更が多い箇所にOCPとDIPを適用するのが実践的。プロジェクトの規模と変更頻度に応じて段階的に導入する。小規模なスクリプトやプロトタイプにSOLIDを完全適用するのはオーバーエンジニアリングである。

### Q2: SOLIDは関数型プログラミングにも適用できるか？

概念的には適用可能。SRPは「関数は1つのことをする」、OCPは「高階関数で拡張する」、DIPは「関数の注入」に対応する。ただし用語はOOP文脈で定義されたものなので、関数型では別の原則名（純粋性、合成可能性、参照透過性など）で語られることが多い。

### Q3: LSP違反をどうやって検出するか？

以下のコードスメルが検出のヒント:
- 派生クラスで `UnsupportedOperationException` をスローしている
- `instanceof` / `typeof` チェックが増えている
- 基底クラスの事前条件を強化、事後条件を弱化している
- 「is-a」関係が成り立たない継承がある（正方形 is-a 長方形の問題）

自動検出の方法として、基底クラスのテストスイートを派生クラスに対しても実行する「契約テスト」がある。

### Q4: DIPとDIコンテナは必須か？

DIPは原則であり、DIコンテナはその実現手段の一つに過ぎない。コンストラクタインジェクションだけでもDIPは実現できる。DIコンテナが必要になるのは、依存グラフが複雑になった大規模アプリケーションの場合。小規模なプロジェクトではマニュアルDI（手動で依存を組み立てる）で十分。

### Q5: SOLIDとマイクロサービスの関係は？

マイクロサービスアーキテクチャはSOLID原則の「サービスレベル」での適用と見ることができる:
- **SRP**: 各サービスは1つのビジネスドメインに責任を持つ
- **OCP**: 新機能は新サービスの追加で対応
- **LSP**: サービスのAPIコントラクトを守る
- **ISP**: 必要なAPIのみを公開する
- **DIP**: サービス間はメッセージキュー等の抽象を介して通信

---

## まとめ

| 原則 | 一言で | 違反の兆候 | 改善手法 |
|------|--------|-----------|---------|
| SRP | 変更理由は1つ | 巨大クラス、頻繁な修正 | Extract Class |
| OCP | 拡張で対応 | if/switch の増殖 | Strategy, Template Method |
| LSP | 置換可能 | instanceof チェック | インターフェース再設計 |
| ISP | 小さなIF | 空のメソッド実装 | Interface分割 |
| DIP | 抽象に依存 | new の直接呼び出し | コンストラクタインジェクション |

### 各原則の適用優先度ガイド

| 優先度 | 原則 | 理由 |
|--------|------|------|
| 1（最初に） | SRP | 最も直感的で効果が大きい |
| 2 | DIP | テスト容易性が劇的に向上 |
| 3 | OCP | 変更の多い箇所で効果を発揮 |
| 4 | ISP | DIPの効果を強化 |
| 5 | LSP | 継承を使う場合に重要 |

---

## 次に読むべきガイド

- [DRY/KISS/YAGNI](./02-dry-kiss-yagni.md) ── 重複排除と単純化の原則
- [結合度と凝集度](./03-coupling-cohesion.md) ── モジュール設計の基盤
- [デメテルの法則](./04-law-of-demeter.md) ── 結合度を下げる具体的規則
- [合成 vs 継承](../03-practices-advanced/01-composition-over-inheritance.md) ── LSPの先にある設計判断
- [デザインパターン: Creational](../../design-patterns-guide/00-creational/) ── OCPとDIPを実現するパターン
- [デザインパターン: Behavioral](../../design-patterns-guide/02-behavioral/) ── StrategyパターンなどOCP実現手段
- [システム設計: アーキテクチャ](../../system-design-guide/02-architecture/) ── SOLIDのアーキテクチャレベル適用

---

## 参考文献

1. **Robert C. Martin** 『Agile Software Development: Principles, Patterns, and Practices』 Prentice Hall, 2002
2. **Robert C. Martin** 『Clean Architecture: A Craftsman's Guide to Software Structure and Design』 Prentice Hall, 2017
3. **Barbara Liskov, Jeannette Wing** "A Behavioral Notion of Subtyping" ACM Transactions on Programming Languages and Systems, 1994
4. **Bertrand Meyer** 『Object-Oriented Software Construction』 Prentice Hall, 1997 (2nd Edition)
5. **Martin Fowler** 『Refactoring: Improving the Design of Existing Code』 Addison-Wesley, 2018 (2nd Edition)
6. **Sandi Metz** 『Practical Object-Oriented Design: An Agile Primer Using Ruby』 Addison-Wesley, 2018
7. **Michael Feathers** 『Working Effectively with Legacy Code』 Prentice Hall, 2004
8. **Mark Seemann** 『Dependency Injection: Principles, Practices, and Patterns』 Manning Publications, 2019
