# LSP（リスコフの置換原則）+ ISP（インターフェース分離の原則）

> LSPは「サブタイプは親タイプの代替として使えるべき」、ISPは「クライアントに不要なメソッドへの依存を強制しない」。型の正しさとインターフェースの適切な粒度を保証する原則。

## この章で学ぶこと

- [ ] LSP 違反のパターンとその回避方法を理解する
- [ ] ISP による適切なインターフェース設計を把握する
- [ ] 実践的な設計判断の基準を学ぶ
- [ ] LSP と ISP を組み合わせた堅牢な型階層の設計方法を習得する
- [ ] 実務での違反検出とリファクタリング手法を身につける

---

## 1. LSP: リスコフの置換原則

```
定義（Barbara Liskov, 1987）:
  「S が T のサブタイプならば、T型のオブジェクトを
   S型のオブジェクトで置き換えてもプログラムの正しさは変わらない」

平易に:
  → 親クラスが使えるところにサブクラスを入れても壊れない
  → サブクラスは親クラスの「契約」を守る

契約:
  1. 事前条件を強化しない（受け入れ範囲を狭めない）
  2. 事後条件を弱化しない（保証を減らさない）
  3. 不変条件を維持する

形式的な定義（Design by Contract との関係）:
  サブタイプ S が T の正当な代替であるために:
  - S の事前条件 ≤ T の事前条件（より緩い or 同じ）
  - S の事後条件 ≥ T の事後条件（より厳しい or 同じ）
  - S は T の不変条件をすべて維持する
  - S は T が送出する例外のサブタイプのみを送出する
```

### 1.1 LSPの歴史的背景

```
1987年:
  Barbara Liskov が OOPSLA の基調講演 "Data Abstraction and Hierarchy"
  で最初にこの概念を提示

1994年:
  Liskov と Wing が "A Behavioral Notion of Subtyping" を発表
  行動的サブタイピングの形式的定義を確立

2002年:
  Robert C. Martin が SOLID 原則の一部として整理
  実務者向けに分かりやすくまとめ直した

核心的な洞察:
  → 継承は「コードの再利用」ではなく「振る舞いの互換性」のために使うべき
  → 型の階層は「実装の階層」ではなく「契約の階層」であるべき
  → サブタイプは単にメソッドを持っているだけでなく、
    意味的に正しく振る舞わなければならない
```

### 1.2 LSP 違反の典型例: 正方形と長方形

```typescript
// ❌ LSP違反の典型例
class Rectangle {
  constructor(protected width: number, protected height: number) {}

  setWidth(w: number): void { this.width = w; }
  setHeight(h: number): void { this.height = h; }
  area(): number { return this.width * this.height; }
}

class Square extends Rectangle {
  setWidth(w: number): void {
    this.width = w;
    this.height = w; // 正方形なので幅と高さを同じにする
  }
  setHeight(h: number): void {
    this.width = h;
    this.height = h;
  }
}

// このテストが壊れる = LSP違反
function testRectangle(rect: Rectangle): void {
  rect.setWidth(5);
  rect.setHeight(4);
  console.assert(rect.area() === 20); // Square だと 16 になる！
}

testRectangle(new Rectangle(0, 0)); // ✅ 20
testRectangle(new Square(0, 0));    // ❌ 16（LSP違反）
```

```typescript
// ✅ LSP準拠: 共通のインターフェースで抽象化
interface Shape {
  area(): number;
}

class Rectangle implements Shape {
  constructor(private width: number, private height: number) {}
  area(): number { return this.width * this.height; }
}

class Square implements Shape {
  constructor(private side: number) {}
  area(): number { return this.side * this.side; }
}
// Square は Rectangle を継承しない → LSP問題が発生しない
```

### 1.3 LSP 違反のパターン一覧

```
パターン1: メソッドの例外追加
  親: withdraw(amount) — 常に成功
  子: withdraw(amount) — 残高不足で例外 ← 事前条件の強化

パターン2: 空実装
  親: save() — データを保存
  子: save() — 何もしない ← 事後条件の弱化

パターン3: 型チェック
  if (animal instanceof Dog) {
    animal.fetch();
  }
  → ポリモーフィズムが壊れている = LSP違反の兆候

パターン4: 戻り値の型の変更
  親: findAll() → 常に配列を返す
  子: findAll() → 条件によって null を返す ← 事後条件の弱化

パターン5: 副作用の追加
  親: calculate(x) → 計算結果を返す（副作用なし）
  子: calculate(x) → 計算結果を返す + ログを書き込む + DBに保存
  → 予期しない副作用 ← 不変条件の違反

パターン6: 状態変更の範囲
  親: setName(name) → name フィールドのみ変更
  子: setName(name) → name 変更 + updatedAt 変更 + 通知送信
  → 呼び出し側が予期しない状態変更
```

```python
# ❌ LSP違反: 空実装
class Bird:
    def fly(self) -> str:
        return "飛んでいます"

class Penguin(Bird):
    def fly(self) -> str:
        raise NotImplementedError("ペンギンは飛べません")  # LSP違反

# ✅ LSP準拠: インターフェースを分離
from abc import ABC, abstractmethod

class Bird(ABC):
    @abstractmethod
    def move(self) -> str: ...

class FlyingBird(Bird):
    def move(self) -> str:
        return "飛んでいます"

class Penguin(Bird):
    def move(self) -> str:
        return "泳いでいます"  # 正当な実装
```

### 1.4 Design by Contract と LSP

```python
# Design by Contract (DbC) を使った LSP の形式化
from abc import ABC, abstractmethod
from typing import List


class SortedCollection(ABC):
    """ソート済みコレクションの契約"""

    @abstractmethod
    def add(self, item: int) -> None:
        """
        事前条件: なし（任意の整数を受け付ける）
        事後条件: アイテムが追加され、コレクションはソート済み状態を維持する
        不変条件: コレクションは常にソート済み
        """
        ...

    @abstractmethod
    def get_all(self) -> List[int]:
        """
        事前条件: なし
        事後条件: ソート済みのリストを返す
        """
        ...

    def _check_invariant(self) -> bool:
        """不変条件: 常にソート済み"""
        items = self.get_all()
        return all(items[i] <= items[i + 1] for i in range(len(items) - 1))


class AscendingSortedCollection(SortedCollection):
    """✅ LSP準拠: 昇順でソート済み"""

    def __init__(self):
        self._items: List[int] = []

    def add(self, item: int) -> None:
        import bisect
        bisect.insort(self._items, item)
        assert self._check_invariant(), "不変条件違反"

    def get_all(self) -> List[int]:
        return list(self._items)


class UniqueAscendingSortedCollection(SortedCollection):
    """✅ LSP準拠: 重複なし昇順ソート
    事後条件を強化（重複排除も保証）→ OK
    """

    def __init__(self):
        self._items: List[int] = []

    def add(self, item: int) -> None:
        if item not in self._items:
            import bisect
            bisect.insort(self._items, item)
        assert self._check_invariant(), "不変条件違反"

    def get_all(self) -> List[int]:
        return list(self._items)


class BoundedSortedCollection(SortedCollection):
    """❌ LSP違反: 値の範囲を制限（事前条件の強化）"""

    def __init__(self, min_val: int = 0, max_val: int = 100):
        self._items: List[int] = []
        self._min = min_val
        self._max = max_val

    def add(self, item: int) -> None:
        if item < self._min or item > self._max:
            raise ValueError(f"値は{self._min}〜{self._max}の範囲内である必要があります")
        import bisect
        bisect.insort(self._items, item)

    def get_all(self) -> List[int]:
        return list(self._items)


# テスト: LSP準拠を検証
def test_sorted_collection(collection: SortedCollection):
    """親クラスの契約に基づくテスト"""
    collection.add(5)
    collection.add(1)
    collection.add(3)
    items = collection.get_all()
    assert items == sorted(items), "ソート済みでない！"
    print(f"✅ {type(collection).__name__}: {items}")


test_sorted_collection(AscendingSortedCollection())           # ✅ [1, 3, 5]
test_sorted_collection(UniqueAscendingSortedCollection())     # ✅ [1, 3, 5]
# test_sorted_collection(BoundedSortedCollection(min_val=3))  # ❌ ValueError
```

### 1.5 共変性・反変性と LSP

```typescript
// 共変性（Covariance）と反変性（Contravariance）はLSPと密接に関連する

// === 戻り値の共変性（LSP準拠）===
// 親の戻り値型のサブタイプを返すのはOK
class Animal {
  name: string;
  constructor(name: string) { this.name = name; }
}

class Dog extends Animal {
  breed: string;
  constructor(name: string, breed: string) {
    super(name);
    this.breed = breed;
  }
}

class AnimalFactory {
  create(): Animal {
    return new Animal("some animal");
  }
}

class DogFactory extends AnimalFactory {
  // ✅ 戻り値の共変性: Dog は Animal のサブタイプ
  create(): Dog {
    return new Dog("Buddy", "Labrador");
  }
}

// === 引数の反変性（LSP準拠）===
// 親の引数型のスーパータイプを受け取るのはOK
interface AnimalHandler {
  handle(animal: Dog): void;  // Dog のみ受け付ける
}

class GeneralAnimalHandler implements AnimalHandler {
  // より広い型（Animal）を受け付けるのは安全
  handle(animal: Animal): void {
    console.log(`Handling ${animal.name}`);
  }
}

// === LSP違反: 引数を狭める ===
// class StrictDogHandler extends GeneralHandler {
//   handle(animal: PurebredDog): void { ... }
//   // ❌ 引数を狭めている（事前条件の強化）
// }
```

```java
// Java での共変戻り値型
public class AnimalShelter {
    public Animal adopt() {
        return new Animal("Unknown");
    }
}

public class DogShelter extends AnimalShelter {
    // ✅ 共変戻り値: 戻り値型をサブタイプに変更（Java 5+）
    @Override
    public Dog adopt() {
        return new Dog("Buddy");
    }
}

// 呼び出し側
AnimalShelter shelter = new DogShelter();
Animal animal = shelter.adopt();  // Dog が返るが、Animal として使える → LSP準拠
```

### 1.6 LSP違反の実務での検出方法

```typescript
// LSP違反を検出する5つのサイン

// サイン1: instanceof チェックの存在
// ❌ ポリモーフィズムの崩壊
function processShape(shape: Shape): number {
  if (shape instanceof Circle) {
    return Math.PI * (shape as Circle).radius ** 2;
  } else if (shape instanceof Rectangle) {
    return (shape as Rectangle).width * (shape as Rectangle).height;
  }
  throw new Error("Unknown shape");
}

// ✅ ポリモーフィズムで解決
function processShape(shape: Shape): number {
  return shape.area();  // 各クラスが自分のarea()を実装
}

// サイン2: NotImplementedError / UnsupportedOperationException
// ❌ サブクラスが親のメソッドを拒否
class ReadOnlyList<T> extends ArrayList<T> {
  add(item: T): void {
    throw new Error("UnsupportedOperation: read-only list");
  }
}

// ✅ インターフェースを分離して解決
interface ReadableList<T> {
  get(index: number): T;
  size(): number;
}

interface WritableList<T> extends ReadableList<T> {
  add(item: T): void;
  remove(index: number): T;
}

// サイン3: 空のメソッド実装
// ❌ 何もしない save()
class CacheOnlyRepository implements Repository {
  save(entity: Entity): void {
    // 何もしない（キャッシュのみなので永続化不要）
  }
}

// サイン4: 条件分岐での型判定
// ❌ 型に基づく分岐
function calculatePay(employee: Employee): number {
  switch (employee.type) {
    case "fulltime": return employee.salary;
    case "parttime": return employee.hourlyRate * employee.hours;
    case "contractor": return employee.dailyRate * employee.days;
  }
}

// ✅ ポリモーフィズムで解決
interface Payable {
  calculatePay(): number;
}

class FullTimeEmployee implements Payable {
  constructor(private salary: number) {}
  calculatePay(): number { return this.salary; }
}

class PartTimeEmployee implements Payable {
  constructor(private hourlyRate: number, private hours: number) {}
  calculatePay(): number { return this.hourlyRate * this.hours; }
}

class Contractor implements Payable {
  constructor(private dailyRate: number, private days: number) {}
  calculatePay(): number { return this.dailyRate * this.days; }
}

// サイン5: ドキュメントでの「このメソッドは呼ばないでください」
// → インターフェースの設計に問題がある明確なサイン
```

### 1.7 LSP準拠のリファクタリングパターン

```python
# パターン1: 「抽出して委譲」
# 継承関係をコンポジションに変換する

# ❌ LSP違反: Stack が List のサブタイプ
class MyList:
    def __init__(self):
        self._items = []

    def add(self, item):
        self._items.append(item)

    def get(self, index):
        return self._items[index]

    def remove(self, index):
        return self._items.pop(index)

    def size(self):
        return len(self._items)


class Stack(MyList):
    """スタックはリストの一種？ → No! LSP違反"""
    def push(self, item):
        self.add(item)

    def pop(self):
        return self.remove(self.size() - 1)

    # get(index) が使える → スタックの契約（LIFO）が壊れる！


# ✅ LSP準拠: コンポジションで実装
class Stack:
    """スタック: LIFO構造"""

    def __init__(self):
        self._items = []  # コンポジション: リストを内部で使用

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("スタックが空です")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("スタックが空です")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)

    # get(index) は公開しない → LIFO契約を維持
```

```typescript
// パターン2: 「インターフェース抽出」
// 共通のインターフェースを抽出して、個別に実装する

// ❌ LSP違反: 永続オブジェクトと一時オブジェクトの混在
class PersistentEntity {
  id: string;
  save(): void { /* DBに保存 */ }
  delete(): void { /* DBから削除 */ }
  validate(): boolean { return true; }
}

class TemporaryEntity extends PersistentEntity {
  save(): void { /* 何もしない */ }    // ❌ 空実装
  delete(): void { /* 何もしない */ }  // ❌ 空実装
}

// ✅ インターフェース抽出でLSP準拠
interface Validatable {
  validate(): boolean;
}

interface Persistable extends Validatable {
  save(): void;
  delete(): void;
}

class PersistentEntity implements Persistable {
  id: string;
  save(): void { /* DBに保存 */ }
  delete(): void { /* DBから削除 */ }
  validate(): boolean { return true; }
}

class TemporaryEntity implements Validatable {
  validate(): boolean { return true; }
  // save() と delete() はそもそも持たない → LSP問題なし
}
```

```java
// パターン3: 「テンプレートメソッド + フック」
// 親クラスのアルゴリズムの一部をサブクラスでカスタマイズ

public abstract class DataExporter {
    // テンプレートメソッド: アルゴリズムの骨格
    public final void export(List<Record> records) {
        validate(records);
        List<String> formatted = format(records);
        String output = join(formatted);
        write(output);
        afterExport(records);  // フック
    }

    // 共通の実装
    protected void validate(List<Record> records) {
        if (records == null || records.isEmpty()) {
            throw new IllegalArgumentException("レコードが空です");
        }
    }

    // サブクラスが実装する抽象メソッド
    protected abstract List<String> format(List<Record> records);
    protected abstract String join(List<String> formatted);
    protected abstract void write(String output);

    // オプショナルなフック（デフォルトでは何もしない）
    protected void afterExport(List<Record> records) {
        // サブクラスで必要に応じてオーバーライド
    }
}

// ✅ LSP準拠: テンプレートメソッドの契約を守りつつカスタマイズ
public class CsvExporter extends DataExporter {
    @Override
    protected List<String> format(List<Record> records) {
        return records.stream()
            .map(r -> String.join(",", r.getValues()))
            .collect(Collectors.toList());
    }

    @Override
    protected String join(List<String> formatted) {
        return String.join("\n", formatted);
    }

    @Override
    protected void write(String output) {
        Files.writeString(Path.of("export.csv"), output);
    }
}

public class JsonExporter extends DataExporter {
    @Override
    protected List<String> format(List<Record> records) {
        ObjectMapper mapper = new ObjectMapper();
        return records.stream()
            .map(r -> mapper.writeValueAsString(r))
            .collect(Collectors.toList());
    }

    @Override
    protected String join(List<String> formatted) {
        return "[" + String.join(",", formatted) + "]";
    }

    @Override
    protected void write(String output) {
        Files.writeString(Path.of("export.json"), output);
    }

    @Override
    protected void afterExport(List<Record> records) {
        logger.info("JSON export completed: {} records", records.size());
    }
}
```

---

## 2. ISP: インターフェース分離の原則

```
定義:
  「クライアントに、使わないメソッドへの依存を強制してはならない」

平易に:
  → インターフェースは小さく、焦点を絞る
  → 「太った」インターフェースを「細い」インターフェースに分割

なぜ重要か:
  → 不要なメソッドを実装する負担を減らす
  → 変更の影響を最小限にする
  → テスト時のモック作成が容易
  → インターフェースの凝集度を高める

ISPの形式的な基準:
  1. インターフェースの各メソッドは、そのインターフェースの
     すべての実装者が意味的に実装できるべき
  2. インターフェースの各メソッドは、そのインターフェースを
     使用するすべてのクライアントが必要とするべき
  3. どちらかが満たされない場合、インターフェースを分割する
```

### 2.1 ISPの歴史的背景

```
1996年:
  Robert C. Martin が "The Interface Segregation Principle" を発表
  Xerox社のプリンタソフトウェアの実際の設計問題から原則を導出

問題の背景:
  Xerox社では、多機能プリンタの全機能を1つの巨大なインターフェースで定義
  → 新しいタイプのプリンタを追加するたびに、不要なメソッドの実装が必要
  → コンパイル時間の増大（C++）
  → テストの肥大化

解決:
  機能ごとにインターフェースを分離
  → 各プリンタは必要なインターフェースのみ実装
  → コンパイル時間の短縮
  → テストの簡素化

教訓:
  → 「太った」インターフェースは、クライアントとサーバー双方に負担をかける
  → 小さなインターフェースの方が、再利用性・テスト容易性が高い
```

### 2.2 ISP リファクタリング: デバイスの例

```typescript
// ❌ ISP違反: 巨大インターフェース
interface SmartDevice {
  print(doc: Document): void;
  scan(): Image;
  fax(doc: Document, number: string): void;
  copy(doc: Document): Document;
  staple(doc: Document): void;
}

// シンプルなプリンターは fax, scan, staple が不要!
class SimplePrinter implements SmartDevice {
  print(doc: Document): void { /* 実装 */ }
  scan(): Image { throw new Error("Not supported"); } // 空実装...
  fax(): void { throw new Error("Not supported"); }   // 空実装...
  copy(): Document { throw new Error("Not supported"); }
  staple(): void { throw new Error("Not supported"); }
}

// ✅ ISP適用: 細かいインターフェースに分離
interface Printer {
  print(doc: Document): void;
}

interface Scanner {
  scan(): Image;
}

interface Faxer {
  fax(doc: Document, number: string): void;
}

// 必要なインターフェースだけ実装
class SimplePrinter implements Printer {
  print(doc: Document): void { /* 実装 */ }
}

class MultiFunctionDevice implements Printer, Scanner, Faxer {
  print(doc: Document): void { /* 実装 */ }
  scan(): Image { /* 実装 */ return new Image(); }
  fax(doc: Document, number: string): void { /* 実装 */ }
}

// 利用側も必要なインターフェースだけに依存
function printReport(printer: Printer): void {
  // Printer だけに依存。Scanner, Faxer は知らない
  printer.print(report);
}
```

### 2.3 実践例: リポジトリのISP

```typescript
// ❌ ISP違反: CRUDが全て必要
interface Repository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

// 読み取り専用サービスにも書き込みメソッドが見える

// ✅ ISP適用: 読み取りと書き込みを分離
interface ReadRepository<T> {
  findAll(): Promise<T[]>;
  findById(id: string): Promise<T | null>;
}

interface WriteRepository<T> {
  create(data: Partial<T>): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<void>;
}

interface Repository<T> extends ReadRepository<T>, WriteRepository<T> {}

// 読み取り専用サービス
class ReportService {
  constructor(private repo: ReadRepository<Order>) {}
  // 書き込みメソッドにアクセスできない = 安全
}
```

### 2.4 ISP実践例: ユーザー管理サービス

```typescript
// ❌ ISP違反: 巨大なユーザーサービスインターフェース
interface UserService {
  // 認証
  login(email: string, password: string): Promise<AuthToken>;
  logout(token: string): Promise<void>;
  refreshToken(token: string): Promise<AuthToken>;

  // プロフィール管理
  getProfile(userId: string): Promise<UserProfile>;
  updateProfile(userId: string, data: Partial<UserProfile>): Promise<void>;
  uploadAvatar(userId: string, image: Buffer): Promise<string>;

  // 管理者機能
  listAllUsers(): Promise<User[]>;
  banUser(userId: string): Promise<void>;
  unbanUser(userId: string): Promise<void>;
  assignRole(userId: string, role: string): Promise<void>;

  // 通知設定
  getNotificationSettings(userId: string): Promise<NotificationSettings>;
  updateNotificationSettings(userId: string, settings: Partial<NotificationSettings>): Promise<void>;

  // 課金
  getSubscription(userId: string): Promise<Subscription>;
  updateSubscription(userId: string, plan: string): Promise<void>;
  cancelSubscription(userId: string): Promise<void>;
}

// ✅ ISP適用: 責務ごとにインターフェースを分離
interface AuthenticationService {
  login(email: string, password: string): Promise<AuthToken>;
  logout(token: string): Promise<void>;
  refreshToken(token: string): Promise<AuthToken>;
}

interface ProfileService {
  getProfile(userId: string): Promise<UserProfile>;
  updateProfile(userId: string, data: Partial<UserProfile>): Promise<void>;
  uploadAvatar(userId: string, image: Buffer): Promise<string>;
}

interface AdminService {
  listAllUsers(): Promise<User[]>;
  banUser(userId: string): Promise<void>;
  unbanUser(userId: string): Promise<void>;
  assignRole(userId: string, role: string): Promise<void>;
}

interface NotificationSettingsService {
  getNotificationSettings(userId: string): Promise<NotificationSettings>;
  updateNotificationSettings(
    userId: string,
    settings: Partial<NotificationSettings>
  ): Promise<void>;
}

interface SubscriptionService {
  getSubscription(userId: string): Promise<Subscription>;
  updateSubscription(userId: string, plan: string): Promise<void>;
  cancelSubscription(userId: string): Promise<void>;
}

// ログイン画面: 認証機能のみ必要
class LoginController {
  constructor(private auth: AuthenticationService) {}

  async handleLogin(email: string, password: string): Promise<AuthToken> {
    return this.auth.login(email, password);
  }
}

// ユーザーダッシュボード: プロフィールと通知設定のみ
class DashboardController {
  constructor(
    private profile: ProfileService,
    private notifications: NotificationSettingsService,
  ) {}

  async getDashboardData(userId: string) {
    const [userProfile, notifSettings] = await Promise.all([
      this.profile.getProfile(userId),
      this.notifications.getNotificationSettings(userId),
    ]);
    return { userProfile, notifSettings };
  }
}

// 管理画面: 管理者機能のみ
class AdminController {
  constructor(private admin: AdminService) {}

  async banMaliciousUser(userId: string): Promise<void> {
    await this.admin.banUser(userId);
  }
}
```

### 2.5 ISP実践例: イベントハンドラの分離

```python
from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import dataclass
from datetime import datetime


# ❌ ISP違反: すべてのイベントを処理する巨大リスナー
class EventListener(ABC):
    @abstractmethod
    def on_user_created(self, user_id: str) -> None: ...

    @abstractmethod
    def on_user_updated(self, user_id: str) -> None: ...

    @abstractmethod
    def on_user_deleted(self, user_id: str) -> None: ...

    @abstractmethod
    def on_order_created(self, order_id: str) -> None: ...

    @abstractmethod
    def on_order_shipped(self, order_id: str) -> None: ...

    @abstractmethod
    def on_payment_received(self, payment_id: str) -> None: ...

    @abstractmethod
    def on_payment_refunded(self, payment_id: str) -> None: ...


# メール通知サービスは注文イベントだけ必要なのに
# 全メソッドの実装を強制される
class EmailNotificationService(EventListener):
    def on_user_created(self, user_id: str) -> None:
        pass  # 不要だが実装必須

    def on_user_updated(self, user_id: str) -> None:
        pass  # 不要だが実装必須

    def on_user_deleted(self, user_id: str) -> None:
        pass  # 不要だが実装必須

    def on_order_created(self, order_id: str) -> None:
        self._send_order_confirmation(order_id)

    def on_order_shipped(self, order_id: str) -> None:
        self._send_shipping_notification(order_id)

    def on_payment_received(self, payment_id: str) -> None:
        pass  # 不要

    def on_payment_refunded(self, payment_id: str) -> None:
        self._send_refund_notification(payment_id)


# ✅ ISP適用: イベントごとにリスナーを分離
class UserEventListener(Protocol):
    def on_user_created(self, user_id: str) -> None: ...
    def on_user_updated(self, user_id: str) -> None: ...
    def on_user_deleted(self, user_id: str) -> None: ...


class OrderEventListener(Protocol):
    def on_order_created(self, order_id: str) -> None: ...
    def on_order_shipped(self, order_id: str) -> None: ...


class PaymentEventListener(Protocol):
    def on_payment_received(self, payment_id: str) -> None: ...
    def on_payment_refunded(self, payment_id: str) -> None: ...


# メール通知サービス: 必要なイベントのみ処理
class EmailNotificationService:
    """注文と返金の通知のみ担当"""

    def on_order_created(self, order_id: str) -> None:
        print(f"注文確認メール送信: {order_id}")

    def on_order_shipped(self, order_id: str) -> None:
        print(f"出荷通知メール送信: {order_id}")

    def on_payment_refunded(self, payment_id: str) -> None:
        print(f"返金通知メール送信: {payment_id}")


# 監査ログサービス: ユーザーイベントのみ処理
class AuditLogService:
    """ユーザー操作の監査ログを記録"""

    def on_user_created(self, user_id: str) -> None:
        print(f"監査ログ: ユーザー作成 {user_id}")

    def on_user_updated(self, user_id: str) -> None:
        print(f"監査ログ: ユーザー更新 {user_id}")

    def on_user_deleted(self, user_id: str) -> None:
        print(f"監査ログ: ユーザー削除 {user_id}")


# イベントバス: リスナーを適切なインターフェースで登録
class EventBus:
    def __init__(self):
        self._user_listeners: list[UserEventListener] = []
        self._order_listeners: list[OrderEventListener] = []
        self._payment_listeners: list[PaymentEventListener] = []

    def register_user_listener(self, listener: UserEventListener) -> None:
        self._user_listeners.append(listener)

    def register_order_listener(self, listener: OrderEventListener) -> None:
        self._order_listeners.append(listener)

    def register_payment_listener(self, listener: PaymentEventListener) -> None:
        self._payment_listeners.append(listener)

    def emit_order_created(self, order_id: str) -> None:
        for listener in self._order_listeners:
            listener.on_order_created(order_id)

    def emit_user_created(self, user_id: str) -> None:
        for listener in self._user_listeners:
            listener.on_user_created(user_id)


# 使用例
bus = EventBus()
bus.register_order_listener(EmailNotificationService())
bus.register_user_listener(AuditLogService())
```

### 2.6 ISP実践例: ファイルシステム操作

```go
package filesystem

import "io"

// ❌ ISP違反: 巨大なファイルシステムインターフェース
type FileSystem interface {
    Read(path string) ([]byte, error)
    Write(path string, data []byte) error
    Delete(path string) error
    Rename(old, new string) error
    List(dir string) ([]string, error)
    Mkdir(path string) error
    MkdirAll(path string) error
    Chmod(path string, mode int) error
    Chown(path string, uid, gid int) error
    Stat(path string) (FileInfo, error)
    Symlink(oldname, newname string) error
    ReadLink(path string) (string, error)
    Watch(path string, callback func(Event)) error
}

// ✅ ISP適用: 責務ごとに分離
type FileReader interface {
    Read(path string) ([]byte, error)
    Stat(path string) (FileInfo, error)
}

type FileWriter interface {
    Write(path string, data []byte) error
    Mkdir(path string) error
    MkdirAll(path string) error
}

type FileDeleter interface {
    Delete(path string) error
}

type DirectoryLister interface {
    List(dir string) ([]string, error)
}

type FilePermissions interface {
    Chmod(path string, mode int) error
    Chown(path string, uid, gid int) error
}

type FileWatcher interface {
    Watch(path string, callback func(Event)) error
}

// 読み取り専用のバックアップサービス
type BackupService struct {
    reader FileReader
    lister DirectoryLister
}

func NewBackupService(reader FileReader, lister DirectoryLister) *BackupService {
    return &BackupService{reader: reader, lister: lister}
}

func (s *BackupService) BackupDirectory(dir string) ([]BackupEntry, error) {
    files, err := s.lister.List(dir)
    if err != nil {
        return nil, err
    }

    var entries []BackupEntry
    for _, file := range files {
        data, err := s.reader.Read(file)
        if err != nil {
            return nil, err
        }
        info, _ := s.reader.Stat(file)
        entries = append(entries, BackupEntry{
            Path:    file,
            Data:    data,
            Size:    info.Size(),
            ModTime: info.ModTime(),
        })
    }
    return entries, nil
}

// 書き込み権限が必要なデプロイサービス
type DeployService struct {
    reader  FileReader
    writer  FileWriter
    deleter FileDeleter
}

func NewDeployService(
    reader FileReader,
    writer FileWriter,
    deleter FileDeleter,
) *DeployService {
    return &DeployService{reader, writer, deleter}
}
```

---

## 3. LSP と ISP の関係

```
LSP: サブタイプの正しさを保証
  → 「このクラスは本当に親の代替として使えるか？」
  → 使えない → インターフェースの設計が間違っている

ISP: インターフェースの粒度を最適化
  → 「このインターフェースは細かすぎ？太すぎ？」
  → 不要なメソッドがある → 分割する

LSP違反 → ISPで解決できることが多い:
  Penguin が Bird.fly() を実装できない
  → Bird インターフェースが太すぎる
  → Movable, Flyable に分割（ISP）
  → Penguin は Movable のみ実装（LSP準拠）

相互関係の図:

  ISP違反                    LSP違反
    │                          │
    ▼                          ▼
  太いインターフェース → 空実装・例外スローが必要
    │                          │
    ▼                          ▼
  ISPで分割         →   自然にLSP準拠に

つまり:
  ISP は LSP 違反を「予防する」役割を果たす
  適切に分離されたインターフェースは、
  LSP違反が起きにくい設計を自然に導く
```

### 3.1 LSP + ISP の統合的な設計例

```typescript
// 実践的な例: 決済システム

// === Step 1: ISPで適切な粒度のインターフェースを設計 ===

interface ChargeablePayment {
  charge(amount: number): Promise<PaymentResult>;
  getChargeLimit(): number;
}

interface RefundablePayment {
  refund(transactionId: string, amount: number): Promise<RefundResult>;
  getRefundPolicy(): RefundPolicy;
}

interface RecurringPayment {
  setupRecurring(interval: string, amount: number): Promise<SubscriptionId>;
  cancelRecurring(subscriptionId: string): Promise<void>;
}

interface PaymentInfoProvider {
  getLastFourDigits(): string;
  getExpirationDate(): string;
  getPaymentType(): string;
}

// === Step 2: 各決済手段がLSP準拠で実装 ===

class CreditCardPayment implements
  ChargeablePayment,
  RefundablePayment,
  RecurringPayment,
  PaymentInfoProvider
{
  constructor(
    private cardNumber: string,
    private expiry: string,
    private cvv: string,
  ) {}

  async charge(amount: number): Promise<PaymentResult> {
    // クレジットカード決済の実装
    return { success: true, transactionId: "cc_" + Date.now() };
  }

  getChargeLimit(): number {
    return 1000000; // 100万円
  }

  async refund(transactionId: string, amount: number): Promise<RefundResult> {
    return { success: true, refundId: "ref_" + Date.now() };
  }

  getRefundPolicy(): RefundPolicy {
    return { maxDays: 30, partialAllowed: true };
  }

  async setupRecurring(interval: string, amount: number): Promise<string> {
    return "sub_" + Date.now();
  }

  async cancelRecurring(subscriptionId: string): Promise<void> {
    // サブスクリプションのキャンセル処理
  }

  getLastFourDigits(): string {
    return this.cardNumber.slice(-4);
  }

  getExpirationDate(): string {
    return this.expiry;
  }

  getPaymentType(): string {
    return "credit_card";
  }
}

class BankTransferPayment implements ChargeablePayment, PaymentInfoProvider {
  // 銀行振込: 課金と情報提供のみ
  // RefundablePayment, RecurringPayment は実装しない → ISP準拠
  // → 「返金できない」メソッドの空実装を強制されない → LSP準拠

  constructor(
    private bankCode: string,
    private accountNumber: string,
  ) {}

  async charge(amount: number): Promise<PaymentResult> {
    return { success: true, transactionId: "bt_" + Date.now() };
  }

  getChargeLimit(): number {
    return 5000000; // 500万円（銀行振込は限度額が高い）
  }

  getLastFourDigits(): string {
    return this.accountNumber.slice(-4);
  }

  getExpirationDate(): string {
    return "N/A"; // 銀行口座に有効期限はない
  }

  getPaymentType(): string {
    return "bank_transfer";
  }
}

class ConvenienceStorePayment implements ChargeablePayment {
  // コンビニ決済: 課金のみ
  // 返金不可、定期支払い不可、カード情報なし

  async charge(amount: number): Promise<PaymentResult> {
    if (amount > 300000) {
      return { success: false, error: "コンビニ決済の上限は30万円です" };
    }
    return { success: true, transactionId: "cvs_" + Date.now() };
  }

  getChargeLimit(): number {
    return 300000; // 30万円
  }
}

// === Step 3: 利用側は必要なインターフェースだけに依存 ===

class CheckoutService {
  // 課金のみ必要
  async processPayment(
    payment: ChargeablePayment,
    amount: number,
  ): Promise<PaymentResult> {
    const limit = payment.getChargeLimit();
    if (amount > limit) {
      return { success: false, error: `決済限度額(${limit})を超えています` };
    }
    return payment.charge(amount);
  }
}

class RefundService {
  // 返金可能な決済手段のみ
  async processRefund(
    payment: RefundablePayment,
    transactionId: string,
    amount: number,
  ): Promise<RefundResult> {
    const policy = payment.getRefundPolicy();
    // ポリシーに基づいた返金処理
    return payment.refund(transactionId, amount);
  }
}

class SubscriptionService {
  // 定期支払い対応の決済手段のみ
  async createSubscription(
    payment: RecurringPayment,
    plan: { interval: string; amount: number },
  ): Promise<string> {
    return payment.setupRecurring(plan.interval, plan.amount);
  }
}
```

---

## 4. 判断基準

```
LSPチェックリスト:
  □ サブクラスは親の全メソッドを正しく実装しているか？
  □ 空実装や例外スロー（UnsupportedOperation）がないか？
  □ instanceof による型チェックが不要か？
  □ 親クラスのテストがサブクラスでも通るか？
  □ 事前条件を強化していないか？
  □ 事後条件を弱化していないか？
  □ 不変条件を維持しているか？
  □ 戻り値の型は共変か？（サブタイプを返すのはOK）
  □ 引数の型は反変か？（スーパータイプを受け取るのはOK）

ISPチェックリスト:
  □ インターフェースの実装者が全メソッドを使っているか？
  □ インターフェースの利用者が全メソッドを必要としているか？
  □ インターフェースのメソッド数は5個以下か？
  □ インターフェースの凝集度は高いか？
  □ 1つの変更理由だけを持つか？（SRP的観点）
  □ インターフェースの名前が具体的か？（「Service」は広すぎる）
  □ モックの作成が容易か？

実務での分割ガイドライン:
  1. 「このインターフェースを実装するとき、
      全メソッドに意味のある実装を書けるか？」
     → 書けない → 分割が必要

  2. 「このインターフェースを利用するとき、
      全メソッドが必要か？」
     → 不要なものがある → 分割が必要

  3. 「このインターフェースの変更頻度は
      全メソッドで同じか？」
     → 違う → 変更頻度ごとに分割
```

### 4.1 過度な分割の回避

```
ISPの落とし穴: 過度な分割（Over-Segregation）

❌ 行き過ぎた分割:
  interface Readable { read(): string; }
  interface Writable { write(data: string): void; }
  interface Closable { close(): void; }
  interface Flushable { flush(): void; }
  interface Seekable { seek(position: number): void; }
  interface Positionable { getPosition(): number; }
  interface Sizeable { getSize(): number; }
  // ... 7個のインターフェース、使い勝手が悪い

✅ 適切な粒度:
  interface ReadableStream {
    read(): string;
    getPosition(): number;
    getSize(): number;
  }

  interface WritableStream {
    write(data: string): void;
    flush(): void;
    getPosition(): number;
  }

  interface Closable {
    close(): void;
  }

  // 3個のインターフェースで十分

判断のコツ:
  → 「一緒に使われるメソッド」は同じインターフェースに
  → 「別々のクライアントが使うメソッド」は別のインターフェースに
  → 凝集度（Cohesion）を意識する
  → 実際のクライアントのユースケースから逆算する
```

---

## 5. テストにおける LSP と ISP

```typescript
// LSP準拠のテスト: 親クラスのテストがサブクラスでも通る

// テストの共通化パターン
abstract class CollectionTestBase<T extends Collection<number>> {
  abstract createCollection(): T;

  testAddAndContains(): void {
    const collection = this.createCollection();
    collection.add(42);
    assert(collection.contains(42), "追加した要素が含まれるべき");
  }

  testRemove(): void {
    const collection = this.createCollection();
    collection.add(42);
    collection.remove(42);
    assert(!collection.contains(42), "削除した要素は含まれないべき");
  }

  testSize(): void {
    const collection = this.createCollection();
    assert(collection.size() === 0, "初期サイズは0");
    collection.add(1);
    collection.add(2);
    assert(collection.size() === 2, "2つ追加後のサイズは2");
  }
}

// ✅ LSP準拠: 全サブクラスが同じテストに通る
class ArrayListTest extends CollectionTestBase<ArrayList<number>> {
  createCollection() { return new ArrayList<number>(); }
}

class LinkedListTest extends CollectionTestBase<LinkedList<number>> {
  createCollection() { return new LinkedList<number>(); }
}

class HashSetTest extends CollectionTestBase<HashSet<number>> {
  createCollection() { return new HashSet<number>(); }
}

// ISP とモック: 小さなインターフェースはモックが容易
// ❌ 太いインターフェース: モック作成が大変
const mockFullRepository: jest.Mocked<Repository<User>> = {
  findAll: jest.fn(),
  findById: jest.fn(),
  create: jest.fn(),
  update: jest.fn(),
  delete: jest.fn(),
  count: jest.fn(),
  findByEmail: jest.fn(),
  search: jest.fn(),
  // ... 多数のメソッドをモック
};

// ✅ ISP適用: 必要なメソッドだけモック
const mockReader: jest.Mocked<ReadRepository<User>> = {
  findAll: jest.fn().mockResolvedValue([]),
  findById: jest.fn().mockResolvedValue(null),
};

// テストが簡潔で意図が明確
describe("ReportService", () => {
  it("should generate report from all users", async () => {
    mockReader.findAll.mockResolvedValue([
      { id: "1", name: "田中" },
      { id: "2", name: "佐藤" },
    ]);

    const service = new ReportService(mockReader);
    const report = await service.generate();

    expect(mockReader.findAll).toHaveBeenCalled();
    expect(report.userCount).toBe(2);
  });
});
```

```python
# Python: pytest でのLSP準拠テスト
import pytest
from abc import ABC, abstractmethod


class ShapeTestBase(ABC):
    """Shape の LSP テスト基底クラス"""

    @abstractmethod
    def create_shape(self) -> "Shape":
        """テスト対象のShapeインスタンスを返す"""
        ...

    def test_area_is_non_negative(self):
        """面積は常に非負"""
        shape = self.create_shape()
        assert shape.area() >= 0

    def test_area_is_numeric(self):
        """面積は数値を返す"""
        shape = self.create_shape()
        assert isinstance(shape.area(), (int, float))

    def test_perimeter_is_non_negative(self):
        """周囲長は常に非負"""
        shape = self.create_shape()
        assert shape.perimeter() >= 0

    def test_string_representation(self):
        """文字列表現が空でない"""
        shape = self.create_shape()
        assert len(str(shape)) > 0


class TestCircle(ShapeTestBase):
    def create_shape(self):
        return Circle(radius=5)

    def test_circle_specific_area(self):
        circle = Circle(radius=10)
        assert abs(circle.area() - 314.159) < 0.01


class TestRectangle(ShapeTestBase):
    def create_shape(self):
        return Rectangle(width=4, height=5)

    def test_rectangle_specific_area(self):
        rect = Rectangle(width=3, height=7)
        assert rect.area() == 21


class TestTriangle(ShapeTestBase):
    def create_shape(self):
        return Triangle(base=6, height=4)

    def test_triangle_specific_area(self):
        tri = Triangle(base=10, height=5)
        assert tri.area() == 25


# 全てのサブクラスが ShapeTestBase のテストに通る = LSP準拠
```

---

## 6. 実務でのアンチパターンと対処法

### 6.1 「何でもインターフェース」アンチパターン

```typescript
// ❌ アンチパターン: 実装が1つしかないのに無理にインターフェースを作る
interface IUserService {
  getUser(id: string): Promise<User>;
}

class UserService implements IUserService {
  getUser(id: string): Promise<User> { /* ... */ }
}

// → 「I」プレフィックスのインターフェースが大量に...
// → 実装が1つしかないので、ISPの恩恵が薄い
// → コードナビゲーションが困難

// ✅ 改善: 本当に必要な場合だけインターフェースを作る
// 以下の場合にインターフェースが有用:
// 1. 複数の実装がある（本番環境、テスト環境、開発環境）
// 2. 外部サービスへの依存を抽象化したい
// 3. テスト時のモックが必要

// 外部API依存: インターフェースが有用
interface PaymentGateway {
  charge(amount: number, token: string): Promise<ChargeResult>;
}

class StripeGateway implements PaymentGateway {
  async charge(amount: number, token: string): Promise<ChargeResult> {
    // Stripe API 呼び出し
  }
}

class MockPaymentGateway implements PaymentGateway {
  async charge(amount: number, token: string): Promise<ChargeResult> {
    return { success: true, id: "mock_" + Date.now() };
  }
}
```

### 6.2 「Header Interface」アンチパターン

```java
// ❌ Header Interface: クラスの全publicメソッドをそのままインターフェースにする
public interface IOrderService {
    Order createOrder(CreateOrderDto dto);
    Order getOrder(String id);
    List<Order> getOrdersByUser(String userId);
    void cancelOrder(String id);
    void updateOrderStatus(String id, OrderStatus status);
    OrderReport generateReport(DateRange range);
    void sendOrderConfirmation(String orderId);
    List<Order> searchOrders(OrderSearchCriteria criteria);
    void archiveOldOrders(int daysOld);
    OrderStats getStatistics(DateRange range);
}

// → クラスの全メソッドをコピーしただけ
// → ISPの精神に反している（クライアントは全メソッドを使わない）

// ✅ クライアントの視点でインターフェースを設計
// 注文の作成・管理
public interface OrderManagement {
    Order createOrder(CreateOrderDto dto);
    void cancelOrder(String id);
    void updateOrderStatus(String id, OrderStatus status);
}

// 注文の検索・閲覧
public interface OrderQuery {
    Order getOrder(String id);
    List<Order> getOrdersByUser(String userId);
    List<Order> searchOrders(OrderSearchCriteria criteria);
}

// レポート・分析
public interface OrderReporting {
    OrderReport generateReport(DateRange range);
    OrderStats getStatistics(DateRange range);
}

// 管理者操作
public interface OrderAdministration {
    void archiveOldOrders(int daysOld);
}

// 通知
public interface OrderNotification {
    void sendOrderConfirmation(String orderId);
}
```

### 6.3 言語別のISP実装テクニック

```python
# Python: Protocol を使った ISP の実装
from typing import Protocol, runtime_checkable


@runtime_checkable
class Drawable(Protocol):
    """描画可能なオブジェクト"""
    def draw(self, canvas: "Canvas") -> None: ...


@runtime_checkable
class Resizable(Protocol):
    """サイズ変更可能なオブジェクト"""
    def resize(self, factor: float) -> None: ...


@runtime_checkable
class Movable(Protocol):
    """移動可能なオブジェクト"""
    def move(self, dx: float, dy: float) -> None: ...


@runtime_checkable
class Rotatable(Protocol):
    """回転可能なオブジェクト"""
    def rotate(self, angle: float) -> None: ...


class Circle:
    """円: 全ての操作に対応"""
    def __init__(self, x: float, y: float, radius: float):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, canvas: "Canvas") -> None:
        canvas.draw_circle(self.x, self.y, self.radius)

    def resize(self, factor: float) -> None:
        self.radius *= factor

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def rotate(self, angle: float) -> None:
        pass  # 円は回転しても変わらない（これは valid な実装）


class TextLabel:
    """テキストラベル: 描画と移動のみ"""
    def __init__(self, x: float, y: float, text: str):
        self.x = x
        self.y = y
        self.text = text

    def draw(self, canvas: "Canvas") -> None:
        canvas.draw_text(self.x, self.y, self.text)

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    # resize() と rotate() は実装しない → ISP準拠


# 利用側: 必要なプロトコルだけ型ヒントで指定
def draw_all(items: list[Drawable]) -> None:
    for item in items:
        item.draw(canvas)

def resize_all(items: list[Resizable], factor: float) -> None:
    for item in items:
        item.resize(factor)

def move_all(items: list[Movable], dx: float, dy: float) -> None:
    for item in items:
        item.move(dx, dy)


# 型チェック: Protocol は isinstance でも使える
circle = Circle(0, 0, 10)
label = TextLabel(0, 0, "Hello")

assert isinstance(circle, Drawable)   # True
assert isinstance(circle, Resizable)  # True
assert isinstance(label, Drawable)    # True
assert isinstance(label, Resizable)   # False — ISPにより安全
```

```rust
// Rust: トレイトでISPを自然に実現
trait Drawable {
    fn draw(&self, canvas: &mut Canvas);
}

trait Resizable {
    fn resize(&mut self, factor: f64);
}

trait Movable {
    fn move_by(&mut self, dx: f64, dy: f64);
}

trait Rotatable {
    fn rotate(&mut self, angle: f64);
}

struct Circle {
    x: f64,
    y: f64,
    radius: f64,
}

// 必要なトレイトだけ実装
impl Drawable for Circle {
    fn draw(&self, canvas: &mut Canvas) {
        canvas.draw_circle(self.x, self.y, self.radius);
    }
}

impl Resizable for Circle {
    fn resize(&mut self, factor: f64) {
        self.radius *= factor;
    }
}

impl Movable for Circle {
    fn move_by(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

struct TextLabel {
    x: f64,
    y: f64,
    text: String,
}

// TextLabel は Drawable と Movable のみ
impl Drawable for TextLabel {
    fn draw(&self, canvas: &mut Canvas) {
        canvas.draw_text(self.x, self.y, &self.text);
    }
}

impl Movable for TextLabel {
    fn move_by(&mut self, dx: f64, dy: f64) {
        self.x += dx;
        self.y += dy;
    }
}

// トレイト境界で必要な能力を指定
fn draw_all(items: &[&dyn Drawable]) {
    for item in items {
        item.draw(&mut canvas);
    }
}

fn resize_all(items: &mut [&mut dyn Resizable], factor: f64) {
    for item in items {
        item.resize(factor);
    }
}

// 複数のトレイト境界の組み合わせ
fn interactive_element<T: Drawable + Movable + Resizable>(element: &mut T) {
    element.draw(&mut canvas);
    element.move_by(10.0, 20.0);
    element.resize(1.5);
    element.draw(&mut canvas);
}
```

---

## 7. 他の SOLID 原則との関係

```
LSP と他の原則:

  SRP ↔ LSP:
    SRP違反（複数の責務）→ LSP違反しやすい
    例: 「データ保存」と「通知」の責務を持つクラスを継承すると
        片方の責務が不要なサブクラスで LSP 違反が起きる

  OCP ↔ LSP:
    LSP準拠 → OCP準拠しやすい
    新しいサブタイプを追加しても既存コードが壊れない

  ISP ↔ LSP:
    ISP準拠 → LSP違反が起きにくい（前述の通り）

  DIP ↔ ISP:
    ISPで分離されたインターフェースに依存（DIP）
    → 疎結合なアーキテクチャが自然に実現

ISP と他の原則:

  SRP ↔ ISP:
    SRP: クラスレベルでの単一責任
    ISP: インターフェースレベルでの単一責任
    同じ「責務の分離」を異なるレベルで適用

  OCP ↔ ISP:
    ISPで分離 → 変更の影響範囲が限定 → OCPに貢献

  DIP ↔ ISP:
    ISPで適切な粒度のインターフェース → DIPの抽象層が最適化
```

---

## まとめ

| 原則 | 核心 | 違反のサイン | 解決策 |
|------|------|------------|--------|
| LSP | 代替可能性 | 空実装、instanceof、例外追加 | インターフェース再設計 |
| ISP | 適切な粒度 | 不要メソッド、肥大化したIF | インターフェース分割 |

```
LSP + ISP の実践まとめ:

  1. インターフェースを設計するとき:
     → ISPの観点: 小さく、焦点を絞る
     → 「全実装者が全メソッドを有意味に実装できるか？」

  2. 継承関係を設計するとき:
     → LSPの観点: 代替可能性を保証
     → 「親のテストがサブクラスで通るか？」

  3. リファクタリングのとき:
     → LSP違反を発見 → ISPで分割を検討
     → ISP違反を発見 → LSP違反も同時にチェック

  4. テストのとき:
     → LSP: 親クラスのテストをサブクラスで再実行
     → ISP: モック作成が容易かで粒度を確認
```

---

## 次に読むべきガイド
→ [[03-dip.md]] — DIP（依存性逆転の原則）

---

## 参考文献
1. Liskov, B. "Data Abstraction and Hierarchy." OOPSLA, 1987.
2. Liskov, B. and Wing, J. "A Behavioral Notion of Subtyping." ACM Transactions on Programming Languages and Systems, 1994.
3. Martin, R. "The Interface Segregation Principle." 1996.
4. Martin, R. "Agile Software Development: Principles, Patterns, and Practices." Prentice Hall, 2002.
5. Meyer, B. "Object-Oriented Software Construction." Prentice Hall, 1997.
6. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
