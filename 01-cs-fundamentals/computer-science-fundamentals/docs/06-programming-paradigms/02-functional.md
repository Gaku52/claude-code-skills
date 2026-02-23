# 関数型プログラミング

> 関数型プログラミング（FP）は「副作用のない純粋関数の合成」を基本とするパラダイムであり、並行処理への安全性とテスタビリティに優れる。数学的な関数の概念に基づき、状態変化ではなくデータの変換としてプログラムを記述する。

## この章で学ぶこと

- [ ] 純粋関数と副作用の概念を説明できる
- [ ] 参照透過性の意味と利点を理解する
- [ ] map/filter/reduceを使いこなせる
- [ ] 不変性（Immutability）のメリットを理解し適用できる
- [ ] 関数合成、カリー化、部分適用を実装できる
- [ ] モナドの概念を理解し、Option/Result パターンを使える
- [ ] 実務での関数型テクニックを適用できる

---

## 1. 関数型プログラミングの歴史と背景

### 1.1 数学的基礎からプログラミングへ

関数型プログラミングの理論的基礎は、1930年代のアロンゾ・チャーチによるラムダ計算（Lambda Calculus）に遡る。チューリングマシンと計算能力において等価であることが証明されており、計算の数学的モデルとして強力な基盤を持つ。

```
関数型プログラミングの歴史年表:

1930年代 ラムダ計算（Alonzo Church）
          - 関数の抽象化と適用の形式的体系
          - チューリング完全であることが証明
          - 現在のFPの数学的基礎

1958年   LISP（John McCarthy）
          - 最初の関数型プログラミング言語
          - リスト処理、再帰、ガベージコレクション
          - S式による統一的な表現

1973年   ML（Robin Milner）
          - 型推論（Hindley-Milner型システム）
          - パターンマッチング
          - 代数的データ型

1977年   FP（John Backus）
          - Turing Award講演「Can Programming Be Liberated
            from the von Neumann Style?」
          - 関数レベルプログラミングの提唱

1986年   Erlang（Joe Armstrong / Ericsson）
          - アクターモデル + 関数型
          - 耐障害性、ホットスワップ
          - 通信システムで実績

1990年   Haskell
          - 純粋関数型言語の統一
          - 遅延評価、モナド、型クラス
          - 学術研究の標準言語

2003年   Scala（Martin Odersky）
          - OOPとFPの融合
          - JVM上で動作
          - Akka（アクターモデル）

2007年   Clojure（Rich Hickey）
          - JVM上のLISP方言
          - 永続データ構造
          - 並行処理への強い支援

2012年   Elixir（José Valim）
          - Erlang VM（BEAM）上の関数型言語
          - Phoenixフレームワーク
          - Web開発での実用性

2015年   Elm（Evan Czaplicki）
          - Webフロントエンド向け純粋FP
          - The Elm Architecture（TEA）
          - ReactやReduxに大きな影響
```

### 1.2 関数型プログラミングの核心思想

```
関数型プログラミングの核心:

  命令型プログラミング:
  「どうやるか（How）」を手順として記述
  → 変数を宣言し、ループで回し、状態を変更する

  関数型プログラミング:
  「何であるか（What）」を宣言的に記述
  → 入力と出力の関係（写像）を定義する

  数学の関数との対応:
  f(x) = x² + 2x + 1
  → 同じxに対して常に同じ結果
  → 関数の実行が他の何かに影響しない
  → 関数を組み合わせて新しい関数を作れる
    g(x) = f(x) + 3 = x² + 2x + 4

  プログラミングへの応用:
  プログラム = 関数の合成
  データフロー: 入力 → 変換1 → 変換2 → ... → 出力
```

---

## 2. 関数型の核心概念

### 2.1 純粋関数（Pure Functions）

純粋関数は関数型プログラミングの最も基本的な概念である。2つの条件を満たす関数を純粋関数と呼ぶ。

```python
# === 純粋関数の定義 ===
# 条件1: 同じ引数に対して常に同じ戻り値を返す（決定的）
# 条件2: 副作用がない（外部状態を読み書きしない）

# ✅ 純粋関数の例
def add(a: int, b: int) -> int:
    return a + b  # 入力のみに依存、外部を変更しない

def square(x: float) -> float:
    return x ** 2  # 同じxには常に同じ結果

def format_name(first: str, last: str) -> str:
    return f"{last} {first}"  # 副作用なし

def calculate_tax(price: int, tax_rate: float) -> int:
    return int(price * tax_rate)  # 外部状態に依存しない

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)  # 再帰も純粋


# ❌ 不純な関数の例
import random
import datetime

# 不純: 外部状態（グローバル変数）を変更
counter = 0
def increment():
    global counter
    counter += 1  # 副作用: 外部状態を変更
    return counter

# 不純: 同じ引数でも異なる結果
def get_random_number(max_val: int) -> int:
    return random.randint(0, max_val)  # 非決定的

# 不純: 外部状態（現在時刻）に依存
def get_greeting(name: str) -> str:
    hour = datetime.datetime.now().hour  # 外部状態に依存
    if hour < 12:
        return f"おはようございます、{name}さん"
    return f"こんにちは、{name}さん"

# 不純: I/O（副作用）
def log_message(message: str) -> None:
    print(message)  # 副作用: 画面出力

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()  # 副作用: ファイルI/O
```

```python
# === 不純な関数を純粋に近づけるテクニック ===

# 方法1: 依存する外部状態を引数に含める
# ❌ 不純
def get_greeting_impure(name: str) -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return f"おはようございます、{name}さん"
    return f"こんにちは、{name}さん"

# ✅ 純粋（時刻を引数にする）
def get_greeting_pure(name: str, hour: int) -> str:
    if hour < 12:
        return f"おはようございます、{name}さん"
    return f"こんにちは、{name}さん"

# テスト容易
assert get_greeting_pure("太郎", 9) == "おはようございます、太郎さん"
assert get_greeting_pure("太郎", 15) == "こんにちは、太郎さん"


# 方法2: 副作用を関数の外側に押し出す
# ❌ 不純: ビジネスロジックとI/Oが混在
def process_order_impure(order_id: int):
    order = db.fetch_order(order_id)         # I/O
    total = calculate_total(order.items)     # 計算
    tax = total * 0.1                         # 計算
    db.update_order(order_id, total + tax)   # I/O
    send_email(order.customer, "注文確定")    # I/O

# ✅ 純粋なコア + 不純なシェル
def calculate_order_total(items: list[dict]) -> dict:
    """純粋: 計算ロジックのみ"""
    subtotal = sum(item['price'] * item['qty'] for item in items)
    tax = int(subtotal * 0.1)
    return {
        'subtotal': subtotal,
        'tax': tax,
        'total': subtotal + tax
    }

def process_order(order_id: int):
    """不純: I/Oを担当するシェル"""
    order = db.fetch_order(order_id)                     # I/O
    result = calculate_order_total(order.items)          # 純粋関数呼び出し
    db.update_order(order_id, result['total'])            # I/O
    send_email(order.customer, f"注文確定: {result}")    # I/O
```

### 2.2 参照透過性（Referential Transparency）

```python
# === 参照透過性 ===
# 式をその結果の値に置き換えても、プログラムの振る舞いが変わらない性質

# ✅ 参照透過
def add(a: int, b: int) -> int:
    return a + b

# add(3, 4) を 7 に置き換えても同じ
result = add(3, 4) + add(1, 2)
# ↓ と同じ
result = 7 + 3  # = 10

# これにより以下が可能:
# 1. メモ化（キャッシュ）: 同じ引数なら結果を再計算不要
# 2. 遅延評価: 必要になるまで計算を延期
# 3. 並列化: 部分式を並列に安全に評価
# 4. 等式推論: 数学的に正しさを証明


# ❌ 参照透過でない
call_count = 0
def add_with_side_effect(a: int, b: int) -> int:
    global call_count
    call_count += 1  # 副作用があるため参照透過でない
    return a + b

# add_with_side_effect(3, 4) を 7 に置き換えると
# call_count の挙動が変わってしまう


# メモ化の実装例（参照透過性が前提）
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    """重い計算を結果をキャッシュして高速化"""
    print(f"  計算中: {n}")  # 初回のみ出力される
    return sum(i ** 2 for i in range(n))

# 初回は計算される
print(expensive_computation(10000))  # 計算中: 10000 → 結果
# 2回目はキャッシュから返される（純粋関数だから安全）
print(expensive_computation(10000))  # 即座に結果（計算なし）


# フィボナッチ数列: メモ化で指数時間→線形時間に
@lru_cache(maxsize=None)
def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(100))  # 354224848179261915075（メモ化なしでは計算不能）
```

### 2.3 高階関数（Higher-Order Functions）

高階関数は「関数を引数に取る」か「関数を返す」関数のことである。関数型プログラミングの基本ツールとなる。

```python
# === 高階関数の基本 ===

from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')


# 1. map: 各要素に関数を適用して変換
names = ["alice", "bob", "charlie"]
upper_names = list(map(str.upper, names))
# → ['ALICE', 'BOB', 'CHARLIE']

# Python 的にはリスト内包表記も使える
upper_names = [name.upper() for name in names]


# 2. filter: 条件を満たす要素を抽出
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
# → [2, 4, 6, 8, 10]

# Python 的
evens = [x for x in numbers if x % 2 == 0]


# 3. reduce: 畳み込み（左からの累積演算）
from functools import reduce

total = reduce(lambda acc, x: acc + x, numbers, 0)
# → 55

product = reduce(lambda acc, x: acc * x, numbers, 1)
# → 3628800

# reduceの動作を追跡:
# reduce(f, [1,2,3,4], 0)
# → f(f(f(f(0, 1), 2), 3), 4)
# → f(f(f(1, 2), 3), 4)
# → f(f(3, 3), 4)
# → f(6, 4)
# → 10


# 4. sorted: カスタムキーによるソート
students = [
    {"name": "田中", "score": 85},
    {"name": "佐藤", "score": 92},
    {"name": "鈴木", "score": 78},
]
by_score = sorted(students, key=lambda s: s["score"], reverse=True)
# → [{"name": "佐藤", ...}, {"name": "田中", ...}, {"name": "鈴木", ...}]


# 5. any / all: 論理的な集約
scores = [85, 92, 78, 65, 90]
all_passed = all(s >= 60 for s in scores)       # True: 全員60点以上
has_perfect = any(s == 100 for s in scores)     # False: 100点はいない


# 6. 関数を返す高階関数
def make_multiplier(factor: int) -> Callable[[int], int]:
    """乗算関数を生成する高階関数"""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)
print(double(5))   # 10
print(triple(5))   # 15


# 7. デコレータ（Pythonの高階関数パターン）
import time
from functools import wraps

def timing(func: Callable) -> Callable:
    """関数の実行時間を計測するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed:.4f}秒")
        return result
    return wrapper

def retry(max_attempts: int = 3, delay: float = 1.0):
    """リトライ機能を付与するデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"リトライ {attempt}/{max_attempts}: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@timing
@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """データを取得する"""
    # 実際のAPI呼び出し処理
    pass
```

```javascript
// JavaScript での高階関数

// map: 変換
const users = [
    { name: '田中', age: 30 },
    { name: '佐藤', age: 25 },
    { name: '鈴木', age: 35 },
];

const names = users.map(u => u.name);
// → ['田中', '佐藤', '鈴木']

const withGreeting = users.map(u => ({
    ...u,
    greeting: `こんにちは、${u.name}さん（${u.age}歳）`,
}));

// filter: 抽出
const adults = users.filter(u => u.age >= 30);

// reduce: 畳み込み
const totalAge = users.reduce((sum, u) => sum + u.age, 0);
// → 90

// reduce で groupBy を実装
const byAge = users.reduce((groups, user) => {
    const key = user.age >= 30 ? 'senior' : 'junior';
    return {
        ...groups,
        [key]: [...(groups[key] || []), user],
    };
}, {});

// メソッドチェーン（パイプライン的な使い方）
const result = users
    .filter(u => u.age >= 25)
    .map(u => u.name.toUpperCase())
    .sort()
    .join(', ');
// → '佐藤, 田中, 鈴木'

// flatMap: ネストを平坦化しながら変換
const orders = [
    { id: 1, items: ['ペン', 'ノート'] },
    { id: 2, items: ['消しゴム'] },
    { id: 3, items: ['定規', '分度器', 'コンパス'] },
];

const allItems = orders.flatMap(order => order.items);
// → ['ペン', 'ノート', '消しゴム', '定規', '分度器', 'コンパス']
```

### 2.4 不変性（Immutability）

```python
# === 不変性の原則と実装 ===

# 不変性: データを変更するのではなく、新しいデータを生成する

# ❌ ミュータブル（可変）な操作
cart = [{"item": "ノートPC", "qty": 1}]
cart[0]["qty"] = 2        # 元のデータが変更される！
cart.append({"item": "マウス", "qty": 1})  # 元のリストが変更される！


# ✅ イミュータブル（不変）な操作
from dataclasses import dataclass, replace, field
from typing import Tuple


@dataclass(frozen=True)
class CartItem:
    """買い物カゴの商品（不変）"""
    name: str
    price: int
    quantity: int

    def with_quantity(self, qty: int) -> 'CartItem':
        """数量を変更した新しいCartItemを返す"""
        return replace(self, quantity=qty)


@dataclass(frozen=True)
class ShoppingCart:
    """買い物カゴ（不変）"""
    items: tuple[CartItem, ...] = ()

    def add_item(self, item: CartItem) -> 'ShoppingCart':
        """商品を追加した新しいカートを返す"""
        return replace(self, items=self.items + (item,))

    def remove_item(self, index: int) -> 'ShoppingCart':
        """商品を削除した新しいカートを返す"""
        new_items = self.items[:index] + self.items[index + 1:]
        return replace(self, items=new_items)

    def update_quantity(self, index: int, qty: int) -> 'ShoppingCart':
        """数量を変更した新しいカートを返す"""
        items_list = list(self.items)
        items_list[index] = items_list[index].with_quantity(qty)
        return replace(self, items=tuple(items_list))

    @property
    def total(self) -> int:
        return sum(item.price * item.quantity for item in self.items)

    @property
    def item_count(self) -> int:
        return sum(item.quantity for item in self.items)


# 使用例: 不変操作のチェーン
cart = ShoppingCart()
cart = cart.add_item(CartItem("ノートPC", 150000, 1))
cart = cart.add_item(CartItem("マウス", 3000, 2))
cart = cart.update_quantity(0, 2)  # ノートPC を 2 台に

# 元の cart は変更されていない（新しいインスタンスが生成された）
print(f"合計: ¥{cart.total:,}")  # ¥306,000
print(f"商品数: {cart.item_count}")  # 4
```

```python
# === 不変なコレクション操作 ===

# リストの不変操作
def append(lst: tuple, item) -> tuple:
    """要素を追加した新しいタプルを返す"""
    return lst + (item,)

def remove_at(lst: tuple, index: int) -> tuple:
    """要素を削除した新しいタプルを返す"""
    return lst[:index] + lst[index + 1:]

def update_at(lst: tuple, index: int, value) -> tuple:
    """要素を更新した新しいタプルを返す"""
    return lst[:index] + (value,) + lst[index + 1:]


# 辞書の不変操作
def assoc(d: dict, key: str, value) -> dict:
    """キーと値を追加/更新した新しいdictを返す"""
    return {**d, key: value}

def dissoc(d: dict, key: str) -> dict:
    """キーを削除した新しいdictを返す"""
    return {k: v for k, v in d.items() if k != key}

def update_in(d: dict, keys: list[str], func) -> dict:
    """ネストしたキーの値を関数で更新した新しいdictを返す"""
    if len(keys) == 1:
        return {**d, keys[0]: func(d.get(keys[0]))}
    key = keys[0]
    return {**d, key: update_in(d.get(key, {}), keys[1:], func)}


# 使用例
state = {"user": {"name": "田中", "age": 30}, "count": 0}

# 不変的に更新
new_state = update_in(state, ["user", "age"], lambda x: x + 1)
print(state["user"]["age"])       # 30（元は変わらない）
print(new_state["user"]["age"])   # 31（新しい状態）

new_state = assoc(state, "count", state["count"] + 1)
print(state["count"])      # 0
print(new_state["count"])  # 1
```

```typescript
// TypeScript での不変性

// Readonly で不変型を定義
interface User {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly tags: readonly string[];
}

// 更新は常にスプレッド構文で新しいオブジェクトを生成
function updateUserName(user: User, name: string): User {
    return { ...user, name };
}

function addTag(user: User, tag: string): User {
    return { ...user, tags: [...user.tags, tag] };
}

function removeTag(user: User, tag: string): User {
    return { ...user, tags: user.tags.filter(t => t !== tag) };
}

// ReadonlyArray, ReadonlyMap, ReadonlySet
const numbers: readonly number[] = [1, 2, 3];
// numbers.push(4);  // コンパイルエラー

// Utility Types で深い不変性
type DeepReadonly<T> = {
    readonly [P in keyof T]: T[P] extends object
        ? DeepReadonly<T[P]>
        : T[P];
};

interface AppState {
    user: {
        name: string;
        settings: {
            theme: string;
            language: string;
        };
    };
    todos: Array<{ id: number; text: string; done: boolean }>;
}

// DeepReadonly<AppState> で全プロパティが再帰的に readonly になる
type ImmutableAppState = DeepReadonly<AppState>;
```

```javascript
// React/Redux での不変性パターン

// Redux の Reducer は常に新しいstateを返す（不変的更新）
function todosReducer(state = [], action) {
    switch (action.type) {
        case 'ADD_TODO':
            // ❌ state.push(action.todo) → 元のstateを変更してしまう
            // ✅ 新しい配列を返す
            return [...state, action.todo];

        case 'TOGGLE_TODO':
            return state.map(todo =>
                todo.id === action.id
                    ? { ...todo, completed: !todo.completed }
                    : todo
            );

        case 'REMOVE_TODO':
            return state.filter(todo => todo.id !== action.id);

        default:
            return state;
    }
}

// Immer を使うとミュータブル風の記法で不変的更新ができる
// import { produce } from 'immer';
//
// const nextState = produce(state, draft => {
//     draft.todos.push({ id: 3, text: '新しいTodo', done: false });
//     draft.todos[0].done = true;
// });
// → state は変更されない、nextState は新しいオブジェクト
```

---

## 3. 関数合成と変換パイプライン

### 3.1 関数合成（Function Composition）

```python
# === 関数合成 ===
# 小さな関数を組み合わせて複雑な処理を構築する

from typing import Callable, TypeVar
from functools import reduce as functools_reduce

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


# 基本的な関数合成
def compose(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
    """f ∘ g: g を適用してから f を適用する"""
    return lambda x: f(g(x))


def pipe(*functions: Callable) -> Callable:
    """左から右へ関数を順に適用する（パイプライン）"""
    def piped(x):
        result = x
        for func in functions:
            result = func(result)
        return result
    return piped


# 使用例: テキスト処理パイプライン
def strip_whitespace(text: str) -> str:
    return text.strip()

def to_lowercase(text: str) -> str:
    return text.lower()

def remove_punctuation(text: str) -> str:
    import re
    return re.sub(r'[^\w\s]', '', text)

def split_words(text: str) -> list[str]:
    return text.split()

def unique_words(words: list[str]) -> set[str]:
    return set(words)


# 関数合成でパイプラインを構築
normalize_text = pipe(strip_whitespace, to_lowercase, remove_punctuation)
extract_unique_words = pipe(normalize_text, split_words, unique_words)

text = "  Hello, World! Hello, Python!  "
print(normalize_text(text))        # "hello world hello python"
print(extract_unique_words(text))  # {'hello', 'world', 'python'}
```

```python
# === データ変換パイプライン ===

from dataclasses import dataclass
from typing import Callable, Iterable


def pipeline(data, *functions):
    """データを関数のチェーンに通す"""
    result = data
    for func in functions:
        result = func(result)
    return result


# 実務例: ログ解析パイプライン
@dataclass(frozen=True)
class LogEntry:
    timestamp: str
    level: str
    message: str
    source: str


def parse_log_line(line: str) -> LogEntry:
    """ログ行をパースする"""
    parts = line.split(" | ")
    return LogEntry(
        timestamp=parts[0],
        level=parts[1],
        message=parts[2],
        source=parts[3] if len(parts) > 3 else "unknown"
    )


def parse_all(lines: list[str]) -> list[LogEntry]:
    return [parse_log_line(line) for line in lines if line.strip()]


def filter_errors(entries: list[LogEntry]) -> list[LogEntry]:
    return [e for e in entries if e.level in ("ERROR", "CRITICAL")]


def group_by_source(entries: list[LogEntry]) -> dict[str, list[LogEntry]]:
    groups: dict[str, list[LogEntry]] = {}
    for entry in entries:
        groups.setdefault(entry.source, []).append(entry)
    return groups


def count_per_group(groups: dict[str, list]) -> dict[str, int]:
    return {source: len(entries) for source, entries in groups.items()}


def sort_by_count(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# パイプラインで処理
log_lines = [
    "2025-01-15 10:00:00 | ERROR | DB接続失敗 | database",
    "2025-01-15 10:01:00 | INFO | リクエスト処理完了 | api",
    "2025-01-15 10:02:00 | ERROR | タイムアウト | api",
    "2025-01-15 10:03:00 | CRITICAL | OOM発生 | worker",
    "2025-01-15 10:04:00 | ERROR | DB接続失敗 | database",
]

error_ranking = pipeline(
    log_lines,
    parse_all,
    filter_errors,
    group_by_source,
    count_per_group,
    sort_by_count,
)
# → [('database', 2), ('api', 1), ('worker', 1)]
for source, count in error_ranking:
    print(f"  {source}: {count}件のエラー")
```

### 3.2 カリー化と部分適用

```python
# === カリー化（Currying）===
# 複数引数の関数を、1引数の関数のチェーンに変換すること

from functools import partial


# 通常の関数
def add(a: int, b: int) -> int:
    return a + b

# カリー化された関数
def add_curried(a: int) -> Callable[[int], int]:
    def inner(b: int) -> int:
        return a + b
    return inner

add5 = add_curried(5)
print(add5(3))   # 8
print(add5(10))  # 15


# 汎用的なカリー化ヘルパー
def curry(func: Callable) -> Callable:
    """任意の関数をカリー化する"""
    import inspect
    params = inspect.signature(func).parameters
    arity = len(params)

    def curried(*args):
        if len(args) >= arity:
            return func(*args[:arity])
        return lambda *more_args: curried(*args, *more_args)

    return curried


@curry
def multiply(x: int, y: int) -> int:
    return x * y

double = multiply(2)
triple = multiply(3)
print(double(5))   # 10
print(triple(5))   # 15


@curry
def format_log(level: str, source: str, message: str) -> str:
    return f"[{level}] [{source}] {message}"

error_log = format_log("ERROR")
api_error = error_log("API")
print(api_error("接続タイムアウト"))  # [ERROR] [API] 接続タイムアウト

db_error = error_log("DB")
print(db_error("デッドロック検出"))   # [ERROR] [DB] デッドロック検出


# === 部分適用（Partial Application）===
# functools.partial で引数を部分的に固定

from functools import partial

def power(base: int, exponent: int) -> int:
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)
print(square(5))  # 25
print(cube(3))    # 27


# 実務例: 設定済みの関数を生成
import json

def serialize(data: dict, indent: int = None, ensure_ascii: bool = True) -> str:
    return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)

# 日本語対応の整形出力を部分適用
pretty_json = partial(serialize, indent=2, ensure_ascii=False)
compact_json = partial(serialize, indent=None, ensure_ascii=False)

data = {"名前": "田中太郎", "年齢": 30}
print(pretty_json(data))
# {
#   "名前": "田中太郎",
#   "年齢": 30
# }
print(compact_json(data))
# {"名前": "田中太郎", "年齢": 30}
```

---

## 4. モナドとエラーハンドリング

### 4.1 Option/Maybe パターン

```python
# === Option（Maybe）パターン ===
# None チェックの連鎖を安全かつ簡潔に扱う

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Optional
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')


class Option(Generic[T]):
    """Option型: 値が存在するかもしれないし、しないかもしれない"""

    @staticmethod
    def some(value: T) -> 'Option[T]':
        return Some(value)

    @staticmethod
    def none() -> 'Option[T]':
        return _None()

    @staticmethod
    def of(value: Optional[T]) -> 'Option[T]':
        """None なら None_, そうでなければ Some を返す"""
        if value is None:
            return _None()
        return Some(value)

    def map(self, func: Callable[[T], U]) -> 'Option[U]':
        raise NotImplementedError

    def flat_map(self, func: Callable[[T], 'Option[U]']) -> 'Option[U]':
        raise NotImplementedError

    def get_or_else(self, default: T) -> T:
        raise NotImplementedError

    def or_else(self, alternative: Callable[[], 'Option[T]']) -> 'Option[T]':
        raise NotImplementedError

    def filter(self, predicate: Callable[[T], bool]) -> 'Option[T]':
        raise NotImplementedError

    def is_present(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Some(Option[T]):
    value: T

    def map(self, func: Callable[[T], U]) -> Option[U]:
        return Some(func(self.value))

    def flat_map(self, func: Callable[[T], Option[U]]) -> Option[U]:
        return func(self.value)

    def get_or_else(self, default: T) -> T:
        return self.value

    def or_else(self, alternative: Callable[[], Option[T]]) -> Option[T]:
        return self

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self if predicate(self.value) else _None()

    def is_present(self) -> bool:
        return True


class _None(Option[T]):
    def map(self, func: Callable[[T], U]) -> Option[U]:
        return _None()

    def flat_map(self, func: Callable[[T], Option[U]]) -> Option[U]:
        return _None()

    def get_or_else(self, default: T) -> T:
        return default

    def or_else(self, alternative: Callable[[], Option[T]]) -> Option[T]:
        return alternative()

    def filter(self, predicate: Callable[[T], bool]) -> Option[T]:
        return self

    def is_present(self) -> bool:
        return False


# 使用例: ネストした None チェックを排除
def find_user(user_id: int) -> Option[dict]:
    users = {1: {"name": "田中", "department_id": 10}}
    return Option.of(users.get(user_id))

def find_department(dept_id: int) -> Option[dict]:
    departments = {10: {"name": "開発部", "manager_id": 100}}
    return Option.of(departments.get(dept_id))

def find_manager(manager_id: int) -> Option[dict]:
    managers = {100: {"name": "佐藤部長", "email": "sato@example.com"}}
    return Option.of(managers.get(manager_id))


# ❌ 従来の None チェックの連鎖
def get_manager_email_imperative(user_id: int) -> str:
    user = find_user(user_id)
    if user.is_present():
        dept = find_department(user.get_or_else({}).get("department_id"))
        if dept.is_present():
            mgr = find_manager(dept.get_or_else({}).get("manager_id"))
            if mgr.is_present():
                return mgr.get_or_else({}).get("email", "不明")
    return "不明"


# ✅ Option の flat_map でチェーン
def get_manager_email_functional(user_id: int) -> str:
    return (
        find_user(user_id)
        .flat_map(lambda u: find_department(u.get("department_id")))
        .flat_map(lambda d: find_manager(d.get("manager_id")))
        .map(lambda m: m.get("email"))
        .get_or_else("不明")
    )

print(get_manager_email_functional(1))   # sato@example.com
print(get_manager_email_functional(999)) # 不明
```

### 4.2 Result/Either パターン

```python
# === Result（Either）パターン ===
# 成功/失敗を型で表現し、例外なしにエラーを伝播する

from __future__ import annotations
from typing import TypeVar, Generic, Callable, Union
from dataclasses import dataclass

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')


class Result(Generic[T, E]):
    """Result型: 成功（Ok）または失敗（Err）"""

    @staticmethod
    def ok(value: T) -> 'Result[T, E]':
        return Ok(value)

    @staticmethod
    def err(error: E) -> 'Result[T, E]':
        return Err(error)

    def map(self, func: Callable[[T], U]) -> 'Result[U, E]':
        raise NotImplementedError

    def flat_map(self, func: Callable[[T], 'Result[U, E]']) -> 'Result[U, E]':
        raise NotImplementedError

    def map_err(self, func: Callable[[E], Exception]) -> 'Result[T, Exception]':
        raise NotImplementedError

    def unwrap(self) -> T:
        raise NotImplementedError

    def unwrap_or(self, default: T) -> T:
        raise NotImplementedError

    def is_ok(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class Ok(Result[T, E]):
    value: T

    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        return Ok(func(self.value))

    def flat_map(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return func(self.value)

    def map_err(self, func):
        return self

    def unwrap(self) -> T:
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value

    def is_ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err(Result[T, E]):
    error: E

    def map(self, func):
        return self

    def flat_map(self, func):
        return self

    def map_err(self, func):
        return Err(func(self.error))

    def unwrap(self) -> T:
        raise RuntimeError(f"Err を unwrap: {self.error}")

    def unwrap_or(self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False


# 実務例: ユーザー登録のバリデーション
def validate_email(email: str) -> Result[str, str]:
    if "@" not in email:
        return Result.err("メールアドレスに@が含まれていません")
    if len(email) < 5:
        return Result.err("メールアドレスが短すぎます")
    return Result.ok(email)

def validate_password(password: str) -> Result[str, str]:
    if len(password) < 8:
        return Result.err("パスワードは8文字以上必要です")
    if not any(c.isdigit() for c in password):
        return Result.err("パスワードに数字を含めてください")
    return Result.ok(password)

def validate_name(name: str) -> Result[str, str]:
    if not name or len(name) < 2:
        return Result.err("名前は2文字以上必要です")
    return Result.ok(name)

def register_user(name: str, email: str, password: str) -> Result[dict, str]:
    """ユーザー登録: バリデーション → 登録"""
    return (
        validate_name(name)
        .flat_map(lambda n:
            validate_email(email)
            .flat_map(lambda e:
                validate_password(password)
                .map(lambda p: {
                    "name": n,
                    "email": e,
                    "password_hash": f"hashed_{p}"
                })
            )
        )
    )

# 成功パターン
result = register_user("田中太郎", "tanaka@example.com", "password123")
if result.is_ok():
    print(f"登録成功: {result.unwrap()}")

# 失敗パターン
result = register_user("田中太郎", "invalid-email", "pass")
if not result.is_ok():
    print(f"登録失敗: {result.unwrap_or({})}")  # デフォルト値を返す
```

```rust
// Rust の Result と Option（言語組み込み）

use std::fs;
use std::num::ParseIntError;

// Result<T, E> は Rust の標準型
fn read_number_from_file(path: &str) -> Result<i32, String> {
    // ? 演算子でエラーを自動伝播
    let content = fs::read_to_string(path)
        .map_err(|e| format!("ファイル読み込み失敗: {}", e))?;

    let number: i32 = content.trim().parse()
        .map_err(|e: ParseIntError| format!("数値パース失敗: {}", e))?;

    Ok(number)
}

// メソッドチェーンでの変換
fn process_config(path: &str) -> Result<String, String> {
    read_number_from_file(path)
        .map(|n| n * 2)                          // 値を変換
        .and_then(|n| {                           // Result を返す関数でチェーン
            if n > 0 {
                Ok(format!("結果: {}", n))
            } else {
                Err("負の値です".to_string())
            }
        })
}

// Option<T>
fn find_user(id: u32) -> Option<User> {
    // Some(user) または None
    users.iter().find(|u| u.id == id).cloned()
}

fn get_user_email(id: u32) -> Option<String> {
    find_user(id)
        .filter(|u| u.is_active)
        .map(|u| u.email.clone())
}

// unwrap_or_else でデフォルト値
let email = get_user_email(42)
    .unwrap_or_else(|| "unknown@example.com".to_string());
```

---

## 5. パターンマッチング

### 5.1 代数的データ型とパターンマッチ

```python
# Python 3.10+ の構造的パターンマッチング

from dataclasses import dataclass
from typing import Union


# 代数的データ型（直和型）をクラスで表現
@dataclass(frozen=True)
class Circle:
    radius: float

@dataclass(frozen=True)
class Rectangle:
    width: float
    height: float

@dataclass(frozen=True)
class Triangle:
    base: float
    height: float

Shape = Union[Circle, Rectangle, Triangle]


def area(shape: Shape) -> float:
    """パターンマッチで面積を計算"""
    match shape:
        case Circle(radius=r):
            return 3.14159 * r ** 2
        case Rectangle(width=w, height=h):
            return w * h
        case Triangle(base=b, height=h):
            return b * h / 2
        case _:
            raise ValueError(f"未知の図形: {shape}")


def describe(shape: Shape) -> str:
    """パターンマッチで図形を説明"""
    match shape:
        case Circle(radius=r) if r > 100:
            return f"大きな円（半径{r}）"
        case Circle(radius=r):
            return f"円（半径{r}）"
        case Rectangle(width=w, height=h) if w == h:
            return f"正方形（辺{w}）"
        case Rectangle(width=w, height=h):
            return f"長方形（{w}x{h}）"
        case Triangle(base=b, height=h):
            return f"三角形（底辺{b}、高さ{h}）"


# 使用例
shapes = [Circle(5), Rectangle(3, 4), Rectangle(5, 5), Triangle(6, 8)]
for s in shapes:
    print(f"{describe(s)}: 面積 = {area(s):.2f}")


# JSON/API レスポンスのパターンマッチ
def handle_response(response: dict) -> str:
    match response:
        case {"status": "success", "data": data}:
            return f"成功: {data}"
        case {"status": "error", "code": code, "message": msg}:
            return f"エラー({code}): {msg}"
        case {"status": "redirect", "url": url}:
            return f"リダイレクト先: {url}"
        case {"status": status}:
            return f"未知のステータス: {status}"
        case _:
            return "不正なレスポンス形式"


# コマンドパターンのマッチ
def execute_command(command: list[str]) -> str:
    match command:
        case ["quit" | "exit"]:
            return "終了します"
        case ["help", topic]:
            return f"ヘルプ: {topic}"
        case ["search", *keywords]:
            return f"検索: {' '.join(keywords)}"
        case ["add", name, value] if value.isdigit():
            return f"{name} = {value} を追加"
        case [cmd, *args]:
            return f"未知のコマンド: {cmd} (引数: {args})"
        case _:
            return "コマンドが空です"
```

---

## 6. ジェネレータとイテレータ（遅延評価）

```python
# === 遅延評価（Lazy Evaluation）===
# 必要になるまで計算を遅延させる

from typing import Iterator, Iterable


# ジェネレータ: 遅延的にデータを生成
def fibonacci() -> Iterator[int]:
    """無限フィボナッチ数列"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

def take(n: int, iterable: Iterable) -> list:
    """先頭 n 要素を取得"""
    result = []
    for i, item in enumerate(iterable):
        if i >= n:
            break
        result.append(item)
    return result

# 無限列から最初の10個だけ取得（メモリに全部載せない）
print(take(10, fibonacci()))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]


# ジェネレータ式: メモリ効率の良いデータ処理
def process_large_file(filepath: str) -> Iterator[dict]:
    """大きなCSVファイルを1行ずつ遅延処理"""
    with open(filepath) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

# 1億行のファイルでもメモリを圧迫しない
# for record in process_large_file("huge.csv"):
#     if record["status"] == "error":
#         print(record)


# ジェネレータによるパイプライン
def lines(filepath: str) -> Iterator[str]:
    with open(filepath) as f:
        yield from f

def non_empty(lines: Iterator[str]) -> Iterator[str]:
    return (line for line in lines if line.strip())

def parse_json_lines(lines: Iterator[str]) -> Iterator[dict]:
    import json
    for line in lines:
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue

def filter_by(key: str, value, records: Iterator[dict]) -> Iterator[dict]:
    return (r for r in records if r.get(key) == value)


# パイプラインを構築（全て遅延評価、メモリ効率的）
# errors = filter_by(
#     "level", "ERROR",
#     parse_json_lines(
#         non_empty(
#             lines("app.log")
#         )
#     )
# )
# for error in errors:
#     print(error)


# itertools: 標準ライブラリの遅延ユーティリティ
import itertools

# chain: 複数のイテラブルを連結
combined = itertools.chain([1, 2, 3], [4, 5, 6])
# → 1, 2, 3, 4, 5, 6

# islice: スライス
first_5_fibs = list(itertools.islice(fibonacci(), 5))
# → [0, 1, 1, 2, 3]

# groupby: グルーピング
data = [
    ("A", 1), ("A", 2), ("B", 3), ("B", 4), ("A", 5)
]
sorted_data = sorted(data, key=lambda x: x[0])
for key, group in itertools.groupby(sorted_data, key=lambda x: x[0]):
    print(f"{key}: {list(group)}")

# takewhile / dropwhile
numbers = [2, 4, 6, 1, 3, 5]
even_prefix = list(itertools.takewhile(lambda x: x % 2 == 0, numbers))
# → [2, 4, 6]

# accumulate: 累積計算
running_total = list(itertools.accumulate([1, 2, 3, 4, 5]))
# → [1, 3, 6, 10, 15]

# product: 直積（全組み合わせ）
combos = list(itertools.product(['A', 'B'], [1, 2]))
# → [('A', 1), ('A', 2), ('B', 1), ('B', 2)]
```

---

## 7. 実務での関数型テクニック

### 7.1 データ変換パイプライン（ETL）

```python
# === 実務的なデータ変換パイプライン ===

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Callable, Iterator
import json


@dataclass(frozen=True)
class SalesRecord:
    """売上レコード"""
    date: str
    product: str
    category: str
    amount: int
    quantity: int
    region: str


# 各ステップを純粋関数として定義
def parse_records(raw_data: list[dict]) -> list[SalesRecord]:
    """生データをレコードに変換"""
    return [SalesRecord(**record) for record in raw_data]


def filter_date_range(
    start: str, end: str
) -> Callable[[list[SalesRecord]], list[SalesRecord]]:
    """日付範囲でフィルタする関数を返す（クロージャ）"""
    def _filter(records: list[SalesRecord]) -> list[SalesRecord]:
        return [r for r in records if start <= r.date <= end]
    return _filter


def filter_category(
    category: str
) -> Callable[[list[SalesRecord]], list[SalesRecord]]:
    """カテゴリでフィルタする関数を返す"""
    def _filter(records: list[SalesRecord]) -> list[SalesRecord]:
        return [r for r in records if r.category == category]
    return _filter


def group_by_region(records: list[SalesRecord]) -> dict[str, list[SalesRecord]]:
    """地域別にグループ化"""
    groups: dict[str, list[SalesRecord]] = {}
    for record in records:
        groups.setdefault(record.region, []).append(record)
    return groups


def summarize_groups(
    groups: dict[str, list[SalesRecord]]
) -> dict[str, dict]:
    """グループごとに集計"""
    return {
        region: {
            "total_amount": sum(r.amount for r in records),
            "total_quantity": sum(r.quantity for r in records),
            "record_count": len(records),
            "avg_amount": sum(r.amount for r in records) // len(records),
        }
        for region, records in groups.items()
    }


def sort_by_total(summary: dict[str, dict]) -> list[tuple[str, dict]]:
    """合計金額で降順ソート"""
    return sorted(
        summary.items(),
        key=lambda x: x[1]["total_amount"],
        reverse=True
    )


def format_report(sorted_summary: list[tuple[str, dict]]) -> str:
    """レポートをフォーマット"""
    lines = ["=== 地域別売上レポート ==="]
    for region, data in sorted_summary:
        lines.append(
            f"  {region}: ¥{data['total_amount']:,} "
            f"({data['record_count']}件, "
            f"平均 ¥{data['avg_amount']:,})"
        )
    return "\n".join(lines)


# パイプラインを構築して実行
raw_data = [
    {"date": "2025-01-15", "product": "ノートPC", "category": "電子機器",
     "amount": 150000, "quantity": 1, "region": "東京"},
    {"date": "2025-01-16", "product": "マウス", "category": "電子機器",
     "amount": 3000, "quantity": 5, "region": "大阪"},
    {"date": "2025-01-17", "product": "デスク", "category": "家具",
     "amount": 50000, "quantity": 2, "region": "東京"},
    {"date": "2025-01-18", "product": "モニター", "category": "電子機器",
     "amount": 40000, "quantity": 3, "region": "東京"},
    {"date": "2025-01-19", "product": "キーボード", "category": "電子機器",
     "amount": 8000, "quantity": 10, "region": "大阪"},
]

report = pipeline(
    raw_data,
    parse_records,
    filter_date_range("2025-01-15", "2025-01-19"),
    filter_category("電子機器"),
    group_by_region,
    summarize_groups,
    sort_by_total,
    format_report,
)
print(report)
```

### 7.2 Webフレームワークでの関数型パターン

```python
# === FastAPI での関数型パターン ===

from functools import wraps
from typing import Callable


# ミドルウェアとしてのデコレータ（高階関数）
def require_auth(func: Callable) -> Callable:
    """認証必須デコレータ"""
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return {"error": "認証トークンが必要です"}, 401
        user = await verify_token(token)
        if not user:
            return {"error": "無効なトークンです"}, 403
        return await func(request, user=user, *args, **kwargs)
    return wrapper


def validate_body(schema: type) -> Callable:
    """リクエストボディのバリデーションデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            try:
                body = schema(**await request.json())
            except Exception as e:
                return {"error": f"バリデーションエラー: {e}"}, 400
            return await func(request, body=body, *args, **kwargs)
        return wrapper
    return decorator


def rate_limit(max_calls: int, window_seconds: int) -> Callable:
    """レートリミットデコレータ"""
    from collections import defaultdict
    import time
    calls: dict[str, list[float]] = defaultdict(list)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            client_ip = request.client.host
            now = time.time()
            # 古い呼び出し記録を削除
            calls[client_ip] = [
                t for t in calls[client_ip]
                if now - t < window_seconds
            ]
            if len(calls[client_ip]) >= max_calls:
                return {"error": "レートリミット超過"}, 429
            calls[client_ip].append(now)
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

```javascript
// React での関数型パターン

// カスタムフック: 関数型の合成でロジックを再利用
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);

    return debouncedValue;
}

function useLocalStorage(key, initialValue) {
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = window.localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch {
            return initialValue;
        }
    });

    const setValue = (value) => {
        // 関数型更新をサポート
        const valueToStore = value instanceof Function
            ? value(storedValue) : value;
        setStoredValue(valueToStore);
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
    };

    return [storedValue, setValue];
}

// Reducer パターン（関数型の状態管理）
function todoReducer(state, action) {
    // 純粋関数: 同じ state + action → 同じ結果
    switch (action.type) {
        case 'ADD':
            return {
                ...state,
                todos: [...state.todos, {
                    id: Date.now(),
                    text: action.text,
                    done: false
                }],
            };
        case 'TOGGLE':
            return {
                ...state,
                todos: state.todos.map(todo =>
                    todo.id === action.id
                        ? { ...todo, done: !todo.done }
                        : todo
                ),
            };
        case 'FILTER':
            return { ...state, filter: action.filter };
        default:
            return state;
    }
}
```

### 7.3 関数型エラーハンドリングの実務応用

```python
# === Railway Oriented Programming ===
# 処理をレール（成功/失敗）に沿って流す

from typing import Callable, TypeVar

T = TypeVar('T')
U = TypeVar('U')


def railway(*steps: Callable) -> Callable:
    """Result をチェーンする Railway パターン"""
    def run(input_data):
        result = Result.ok(input_data)
        for step in steps:
            if result.is_ok():
                try:
                    result = step(result.unwrap())
                except Exception as e:
                    result = Result.err(str(e))
            # エラーの場合はスキップ（エラーレールを通る）
        return result
    return run


# 各ステップを関数として定義
def validate_input(data: dict) -> Result:
    if not data.get("email"):
        return Result.err("メールアドレスが必要です")
    if not data.get("name"):
        return Result.err("名前が必要です")
    return Result.ok(data)

def normalize_data(data: dict) -> Result:
    return Result.ok({
        **data,
        "email": data["email"].lower().strip(),
        "name": data["name"].strip(),
    })

def check_duplicate(data: dict) -> Result:
    # DBチェック（シミュレーション）
    existing_emails = ["existing@example.com"]
    if data["email"] in existing_emails:
        return Result.err("このメールアドレスは既に登録されています")
    return Result.ok(data)

def save_to_database(data: dict) -> Result:
    # DB保存（シミュレーション）
    return Result.ok({**data, "id": 42})

def send_welcome_email(data: dict) -> Result:
    # メール送信（シミュレーション）
    print(f"ウェルカムメール送信: {data['email']}")
    return Result.ok(data)


# Railway パイプラインを構築
register_user = railway(
    validate_input,
    normalize_data,
    check_duplicate,
    save_to_database,
    send_welcome_email,
)

# 成功パターン
result = register_user({"name": "田中太郎", "email": "Tanaka@Example.COM"})
print(result)  # Ok({"name": "田中太郎", "email": "tanaka@example.com", "id": 42})

# 失敗パターン（バリデーションエラー）
result = register_user({"name": "", "email": "test@example.com"})
print(result)  # Err("名前が必要です")

# 失敗パターン（重複エラー）
result = register_user({"name": "テスト", "email": "existing@example.com"})
print(result)  # Err("このメールアドレスは既に登録されています")
```

---

## 8. 言語別の関数型サポート

```
言語別の関数型プログラミングサポート:

┌──────────────┬─────────┬──────────┬─────────┬──────────┬────────┐
│              │ Python  │ JS/TS    │ Java    │ Rust     │ Haskell│
├──────────────┼─────────┼──────────┼─────────┼──────────┼────────┤
│ 第一級関数   │ ✅      │ ✅       │ ✅(8+)  │ ✅       │ ✅     │
│ ラムダ式     │ ✅(単式)│ ✅       │ ✅      │ ✅       │ ✅     │
│ クロージャ   │ ✅      │ ✅       │ ✅      │ ✅       │ ✅     │
│ map/filter   │ ✅      │ ✅       │ ✅Stream│ ✅iter   │ ✅     │
│ パターンマッチ│ ✅(3.10)│ ❌       │ ✅(21+) │ ✅       │ ✅     │
│ 不変性       │ frozen  │ const    │ final   │ デフォルト│ デフォルト│
│ 代数的データ型│ Union   │ Union    │ sealed  │ enum     │ data   │
│ Option型     │ Optional│ ?./??   │ Optional│ Option   │ Maybe  │
│ Result型     │ なし*   │ なし*    │ なし*   │ Result   │ Either │
│ 遅延評価     │ generator│generator │ Stream  │ iterator │ デフォルト│
│ 型推論       │ mypy    │ TS      │ var     │ 強力     │ 強力   │
│ 純粋性保証   │ ❌      │ ❌       │ ❌      │ 部分的   │ ✅     │
│ 末尾再帰最適化│ ❌      │ 一部    │ ❌      │ ❌       │ ✅     │
│ モナド       │ なし*   │ Promise  │ Optional│ 慣用的   │ ✅     │
│ カリー化     │ partial │ ✅       │ なし*   │ なし*    │ デフォルト│
└──────────────┴─────────┴──────────┴─────────┴──────────┴────────┘

* ライブラリで実現可能

推奨:
- 純粋FPを学ぶ → Haskell, Elm
- 実務でFP → TypeScript, Rust, Scala
- 既存言語にFP要素を導入 → Python, Java, JavaScript
```

---

## 9. 関数型プログラミングのトレードオフ

```
関数型プログラミングの利点:

  1. テスタビリティ
     - 純粋関数は入出力だけをテスト
     - モック不要、セットアップ不要
     - プロパティベーステストとの相性良

  2. 並行安全性
     - 不変データ → ロック不要
     - 共有状態なし → 競合条件なし
     - メッセージパッシングとの親和性

  3. 推論の容易さ
     - 参照透過性 → 部分的に理解可能
     - 副作用の局所化 → バグの特定が容易
     - 等式推論 → リファクタリングが安全

  4. 合成可能性
     - 小さな関数を組み合わせて大きな処理を構築
     - コードの再利用性が高い
     - パイプライン的な記述が可能

関数型プログラミングの課題:

  1. 学習コスト
     - モナド等の抽象概念が難解
     - 命令型に慣れた開発者には直感的でない
     - エラーメッセージが分かりにくい場合がある

  2. パフォーマンス
     - 不変データ構造の更新コスト（コピー）
     - 再帰のスタックオーバーフロー（末尾再帰最適化がない言語）
     - GCへの負荷（短命なオブジェクトの大量生成）

  3. 状態管理
     - 本質的にステートフルな問題（GUI、ゲーム等）への適用が複雑
     - モナドやEffect systemsが必要になる
     - データベースやファイルI/Oとの境界設計

  4. デバッグ
     - 遅延評価のデバッグが困難
     - 高階関数の連鎖でスタックトレースが追いにくい
     - ポイントフリースタイルの可読性問題

  実務での推奨アプローチ:
  ┌─────────────────────────────────────────────────┐
  │ 「純粋関数型」に固執せず、                       │
  │ 「関数型のエッセンス」を適材適所で導入する        │
  │                                                   │
  │ - ビジネスロジック → 純粋関数                    │
  │ - I/O、副作用 → 手続き型で素直に                 │
  │ - データ変換 → map/filter/reduce                 │
  │ - 状態管理 → 不変性 + Reducer パターン           │
  │ - エラー処理 → Result/Option パターン            │
  └─────────────────────────────────────────────────┘
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 純粋関数 | 副作用なし、同じ入力に同じ出力。テスト容易、並行安全 |
| 参照透過性 | 式を値に置換可能。メモ化、遅延評価、並列化を可能にする |
| 高階関数 | map/filter/reduce。データ変換の基本ツール |
| 不変性 | データを変更せず新しいデータを生成。スレッドセーフ |
| 関数合成 | 小さな関数を組み合わせて複雑な処理を構築 |
| カリー化/部分適用 | 引数を固定して特化した関数を生成 |
| Option/Result | null/例外なしにエラーを型で表現 |
| パターンマッチ | 代数的データ型を安全に分解 |
| 遅延評価 | ジェネレータ/イテレータでメモリ効率的な処理 |
| パイプライン | データ変換の連鎖を宣言的に記述 |

---

## 次に読むべきガイド
→ [[03-concurrent.md]] — 並行・並列プログラミング

---

## 参考文献
1. Hutton, G. "Programming in Haskell." 2nd Edition, Cambridge University Press, 2016.
2. Chiusano, P. & Bjarnason, R. "Functional Programming in Scala." Manning, 2014.
3. Bird, R. "Thinking Functionally with Haskell." Cambridge University Press, 2015.
4. Wlaschin, S. "Domain Modeling Made Functional." Pragmatic Bookshelf, 2018.
5. Fogus, M. "Functional JavaScript." O'Reilly, 2013.
6. Lipovaca, M. "Learn You a Haskell for Great Good!" No Starch Press, 2011.
7. Armstrong, J. "Programming Erlang." 2nd Edition, Pragmatic Bookshelf, 2013.
8. Backus, J. "Can Programming Be Liberated from the von Neumann Style?" ACM Turing Award Lecture, 1977.
9. Church, A. "The Calculi of Lambda Conversion." Princeton University Press, 1941.
10. Milner, R. et al. "The Definition of Standard ML." MIT Press, 1997.
