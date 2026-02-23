# 高階関数（Higher-Order Functions）

> 高階関数は「関数を引数に取る、または関数を返す関数」。コードの再利用性と抽象化のレベルを劇的に向上させる。

## この章で学ぶこと

- [ ] map, filter, reduce の本質を理解する
- [ ] 高階関数による抽象化パターンを習得する
- [ ] 関数型プログラミングの実践的な適用を理解する
- [ ] カリー化と部分適用の違いを理解する
- [ ] 関数合成による宣言的プログラミングを習得する
- [ ] 各言語での高階関数の実装パターンを横断的に理解する

---

## 1. 高階関数の基礎概念

### 1.1 高階関数とは何か

高階関数（Higher-Order Function）は、以下のいずれかまたは両方を満たす関数である。

1. **関数を引数として受け取る**（コールバック関数、述語関数など）
2. **関数を戻り値として返す**（クロージャ、ファクトリ関数など）

この概念はラムダ計算（Lambda Calculus）に起源を持ち、Alonzo Church が1930年代に定式化した計算モデルに基づく。高階関数が存在することで、関数は「第一級オブジェクト（First-class citizen）」として扱われ、変数に代入したり、データ構造に格納したり、他の関数に渡したりすることが可能になる。

```
第一級関数の条件:
1. 変数に代入できる           const f = Math.sqrt;
2. 引数として渡せる           arr.map(f);
3. 戻り値として返せる         function make() { return f; }
4. データ構造に格納できる     const fns = [f, Math.abs];
5. 実行時に生成できる         const g = (x) => x * 2;
```

### 1.2 なぜ高階関数が重要なのか

高階関数がプログラミングにおいて重要な理由は以下の通りである。

```
┌─────────────────────────────────────────────────┐
│            高階関数の利点                         │
├─────────────────────────────────────────────────┤
│ 1. 抽象化: 共通パターンを関数として抽出          │
│ 2. 再利用性: 動作をパラメータ化して汎用的に      │
│ 3. 合成可能性: 小さな関数を組み合わせて複雑な    │
│    処理を構築                                    │
│ 4. 宣言的: 「何をするか」を記述、「どうやるか」  │
│    を抽象化                                      │
│ 5. テスト容易性: 純粋関数を個別にテスト可能      │
│ 6. 遅延評価: 計算を必要な時まで遅延可能          │
└─────────────────────────────────────────────────┘
```

```typescript
// 命令的アプローチ（低い抽象度）
const results: string[] = [];
for (let i = 0; i < users.length; i++) {
    if (users[i].active) {
        results.push(users[i].name.toUpperCase());
    }
}

// 宣言的アプローチ（高い抽象度 - 高階関数を使用）
const results = users
    .filter(u => u.active)
    .map(u => u.name.toUpperCase());
```

命令的アプローチでは「どうやって処理するか」（ループ変数の管理、条件分岐、配列への追加）を記述しているが、宣言的アプローチでは「何をしたいか」（アクティブなユーザーをフィルタし、名前を大文字に変換する）だけを記述している。この違いは、コードの可読性と保守性に大きな影響を与える。

### 1.3 歴史的背景

```
1930年代  Alonzo Church がラムダ計算を定式化
1958年    Lisp 誕生 - 初めて高階関数を実用的に実装
1973年    ML 誕生 - 型推論と高階関数の融合
1990年    Haskell 誕生 - 純粋関数型言語
2004年    Scala 誕生 - OOP + FP の融合
2007年    C# 3.0 LINQ - 高階関数が主流言語に浸透
2011年    Java 8 Lambda - 世界で最も使用される言語に高階関数が導入
2015年    ES2015 (JavaScript) - Arrow Function が標準化
2015年    Rust 1.0 - ゼロコスト抽象化としての高階関数
```

---

## 2. 三大高階関数: map, filter, reduce

### 2.1 概念の図解

```
map:    各要素を変換する      [1,2,3] → [2,4,6]
filter: 条件に合う要素を選ぶ  [1,2,3,4,5] → [2,4]
reduce: 全要素を1つの値に集約 [1,2,3,4,5] → 15
```

```
                    map (f)
┌───┬───┬───┬───┐  f(x) = x * 2  ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ ─────────────→ │ 2 │ 4 │ 6 │ 8 │
└───┴───┴───┴───┘                 └───┴───┴───┴───┘
  入力配列（同じ長さ）              出力配列（同じ長さ）

                    filter (p)
┌───┬───┬───┬───┐  p(x) = x > 2  ┌───┬───┐
│ 1 │ 2 │ 3 │ 4 │ ─────────────→ │ 3 │ 4 │
└───┴───┴───┴───┘                 └───┴───┘
  入力配列                          出力配列（要素数 ≤ 入力）

                    reduce (f, init)
┌───┬───┬───┬───┐  f(acc, x) = acc + x  ┌────┐
│ 1 │ 2 │ 3 │ 4 │ ───────────────────→  │ 10 │
└───┴───┴───┴───┘  init = 0              └────┘
  入力配列                                 単一値
```

### 2.2 TypeScript での実装

```typescript
// TypeScript: 三大高階関数
const numbers = [1, 2, 3, 4, 5];

// map: 変換
numbers.map(n => n * 2);              // [2, 4, 6, 8, 10]
numbers.map(n => n.toString());       // ["1", "2", "3", "4", "5"]
numbers.map((n, i) => ({ index: i, value: n }));

// filter: 選択
numbers.filter(n => n > 3);           // [4, 5]
numbers.filter(n => n % 2 === 0);     // [2, 4]

// reduce: 集約
numbers.reduce((acc, n) => acc + n, 0);        // 15（合計）
numbers.reduce((acc, n) => acc * n, 1);        // 120（積）
numbers.reduce((max, n) => Math.max(max, n), -Infinity); // 5（最大）

// reduce で他の高階関数を実装できる
const myMap = <T, U>(arr: T[], fn: (x: T) => U): U[] =>
    arr.reduce<U[]>((acc, x) => [...acc, fn(x)], []);

const myFilter = <T>(arr: T[], pred: (x: T) => boolean): T[] =>
    arr.reduce<T[]>((acc, x) => pred(x) ? [...acc, x] : acc, []);
```

### 2.3 reduce の深い理解

`reduce` は三大高階関数の中で最も強力で汎用的である。実際、`map` と `filter` は `reduce` で実装できることからもわかるように、`reduce` は折りたたみ（fold）操作の特殊なケースである。

```typescript
// reduce の仕組みを段階的に追跡
const trace = [1, 2, 3, 4].reduce((acc, n) => {
    console.log(`acc=${acc}, n=${n}, result=${acc + n}`);
    return acc + n;
}, 0);
// acc=0, n=1, result=1
// acc=1, n=2, result=3
// acc=3, n=3, result=6
// acc=6, n=4, result=10
// → 10

// reduce でさまざまなデータ変換を実現
const items = ["apple", "banana", "apple", "cherry", "banana", "apple"];

// 頻度カウント
const frequency = items.reduce<Record<string, number>>((acc, item) => {
    acc[item] = (acc[item] || 0) + 1;
    return acc;
}, {});
// → { apple: 3, banana: 2, cherry: 1 }

// 配列の平坦化
const nested = [[1, 2], [3, 4], [5, 6]];
const flat = nested.reduce<number[]>((acc, arr) => [...acc, ...arr], []);
// → [1, 2, 3, 4, 5, 6]

// パイプライン構築
type Transform = (s: string) => string;
const transforms: Transform[] = [
    s => s.trim(),
    s => s.toLowerCase(),
    s => s.replace(/\s+/g, "-"),
];
const slugify = (input: string): string =>
    transforms.reduce((result, transform) => transform(result), input);
slugify("  Hello World  ");  // → "hello-world"

// reduceRight: 右から左への集約
const compose = <T>(...fns: Array<(x: T) => T>) =>
    (x: T): T => fns.reduceRight((acc, fn) => fn(acc), x);
```

### 2.4 Python での三大高階関数

```python
from functools import reduce
from typing import List, Dict, Any

numbers = [1, 2, 3, 4, 5]

# map: 変換（遅延評価 - イテレータを返す）
doubled = list(map(lambda n: n * 2, numbers))      # [2, 4, 6, 8, 10]
strings = list(map(str, numbers))                    # ['1', '2', '3', '4', '5']

# filter: 選択（遅延評価）
evens = list(filter(lambda n: n % 2 == 0, numbers)) # [2, 4]
adults = list(filter(lambda u: u['age'] >= 18, users))

# reduce: 集約
total = reduce(lambda acc, n: acc + n, numbers, 0)   # 15
product = reduce(lambda acc, n: acc * n, numbers, 1)  # 120

# Python ではリスト内包表記が推奨される場合が多い
doubled_comp = [n * 2 for n in numbers]               # [2, 4, 6, 8, 10]
evens_comp = [n for n in numbers if n % 2 == 0]       # [2, 4]

# ジェネレータ式（メモリ効率が良い）
sum_of_squares = sum(n ** 2 for n in range(1000000))

# 複雑な変換パイプライン
users: List[Dict[str, Any]] = [
    {"name": "Alice", "age": 30, "active": True},
    {"name": "Bob", "age": 17, "active": True},
    {"name": "Charlie", "age": 25, "active": False},
    {"name": "Diana", "age": 22, "active": True},
]

# 関数型スタイル
active_adult_names = list(
    map(
        lambda u: u["name"],
        filter(
            lambda u: u["active"] and u["age"] >= 18,
            users,
        ),
    )
)

# リスト内包表記スタイル（Python ではこちらが推奨）
active_adult_names = [
    u["name"] for u in users
    if u["active"] and u["age"] >= 18
]

# functools を活用した高度な使い方
from functools import partial, lru_cache

# partial: 部分適用
def multiply(a: int, b: int) -> int:
    return a * b

double = partial(multiply, 2)
triple = partial(multiply, 3)
print(list(map(double, numbers)))  # [2, 4, 6, 8, 10]
print(list(map(triple, numbers)))  # [3, 6, 9, 12, 15]

# itertools との組み合わせ
from itertools import chain, starmap, accumulate

# accumulate: 累積計算（scan 操作）
running_sum = list(accumulate(numbers))        # [1, 3, 6, 10, 15]
running_max = list(accumulate(numbers, max))   # [1, 2, 3, 4, 5]

# chain: 複数のイテラブルを連結
combined = list(chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# starmap: タプルの展開
pairs = [(2, 5), (3, 2), (10, 3)]
results = list(starmap(pow, pairs))  # [32, 9, 1000]
```

### 2.5 Go での高階関数

```go
package main

import (
    "fmt"
    "strings"
)

// Go にはジェネリクス（1.18+）があるが、組み込みの map/filter はない
// 自前で実装する

// Map: スライスの各要素に関数を適用
func Map[T, U any](slice []T, f func(T) U) []U {
    result := make([]U, len(slice))
    for i, v := range slice {
        result[i] = f(v)
    }
    return result
}

// Filter: 述語を満たす要素のみ返す
func Filter[T any](slice []T, pred func(T) bool) []T {
    result := make([]T, 0)
    for _, v := range slice {
        if pred(v) {
            result = append(result, v)
        }
    }
    return result
}

// Reduce: スライスを単一の値に集約
func Reduce[T, U any](slice []T, init U, f func(U, T) U) U {
    acc := init
    for _, v := range slice {
        acc = f(acc, v)
    }
    return acc
}

// ForEach: 各要素に対して副作用を実行
func ForEach[T any](slice []T, f func(T)) {
    for _, v := range slice {
        f(v)
    }
}

// Any: いずれかの要素が条件を満たすか
func Any[T any](slice []T, pred func(T) bool) bool {
    for _, v := range slice {
        if pred(v) {
            return true
        }
    }
    return false
}

// All: すべての要素が条件を満たすか
func All[T any](slice []T, pred func(T) bool) bool {
    for _, v := range slice {
        if !pred(v) {
            return false
        }
    }
    return true
}

func main() {
    numbers := []int{1, 2, 3, 4, 5}

    // Map
    doubled := Map(numbers, func(n int) int { return n * 2 })
    fmt.Println(doubled) // [2 4 6 8 10]

    // Filter
    evens := Filter(numbers, func(n int) bool { return n%2 == 0 })
    fmt.Println(evens) // [2 4]

    // Reduce
    sum := Reduce(numbers, 0, func(acc, n int) int { return acc + n })
    fmt.Println(sum) // 15

    // チェーン（Go では関数呼び出しのネストになる）
    words := []string{"hello", "world", "go", "higher", "order"}
    longUpper := Map(
        Filter(words, func(s string) bool { return len(s) > 3 }),
        func(s string) string { return strings.ToUpper(s) },
    )
    fmt.Println(longUpper) // [HELLO WORLD HIGHER ORDER]
}
```

### 2.6 Rust での三大高階関数

```rust
// Rust: イテレータ + 高階関数（ゼロコスト抽象化）

fn main() {
    let words = vec!["hello", "world", "foo", "bar"];

    // map + collect
    let upper: Vec<String> = words.iter()
        .map(|w| w.to_uppercase())
        .collect();
    // → ["HELLO", "WORLD", "FOO", "BAR"]

    // filter + map + collect（filter_map で一括）
    let lengths: Vec<usize> = words.iter()
        .filter(|w| w.len() > 3)
        .map(|w| w.len())
        .collect();
    // → [5, 5]

    // filter_map: None を除外しつつ変換
    let valid_numbers: Vec<i32> = vec!["1", "abc", "3", "def", "5"]
        .iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect();
    // → [1, 3, 5]

    // fold（reduce に相当）
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().fold(0, |acc, &n| acc + n);
    // → 15

    // 初期値なしの reduce（Rust 1.51+: reduce メソッド）
    let product: Option<i32> = numbers.iter().copied().reduce(|a, b| a * b);
    // → Some(120)

    // scan: 累積計算（遅延イテレータ）
    let running_sum: Vec<i32> = numbers.iter()
        .scan(0, |state, &n| {
            *state += n;
            Some(*state)
        })
        .collect();
    // → [1, 3, 6, 10, 15]

    // enumerate + map: インデックス付き変換
    let indexed: Vec<String> = words.iter()
        .enumerate()
        .map(|(i, w)| format!("{}:{}", i, w))
        .collect();
    // → ["0:hello", "1:world", "2:foo", "3:bar"]

    // zip: 2つのイテレータを結合
    let keys = vec!["a", "b", "c"];
    let values = vec![1, 2, 3];
    let pairs: Vec<(&str, i32)> = keys.iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    // → [("a", 1), ("b", 2), ("c", 3)]

    // chain: 2つのイテレータを連結
    let first = vec![1, 2, 3];
    let second = vec![4, 5, 6];
    let combined: Vec<i32> = first.iter()
        .chain(second.iter())
        .copied()
        .collect();
    // → [1, 2, 3, 4, 5, 6]

    // flat_map: ネストを平坦化
    let sentences = vec!["hello world", "foo bar"];
    let words_flat: Vec<&str> = sentences.iter()
        .flat_map(|s| s.split_whitespace())
        .collect();
    // → ["hello", "world", "foo", "bar"]

    // partition: 条件で二分割
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let (evens, odds): (Vec<i32>, Vec<i32>) = numbers.iter()
        .partition(|&&n| n % 2 == 0);
    // evens → [2, 4, 6, 8, 10], odds → [1, 3, 5, 7, 9]
}
```

---

## 3. 実践的なデータ変換パイプライン

### 3.1 データパイプラインの構築

```typescript
// データ変換パイプライン
interface User { name: string; age: number; active: boolean; }

const users: User[] = [
    { name: "Alice", age: 30, active: true },
    { name: "Bob", age: 17, active: true },
    { name: "Charlie", age: 25, active: false },
    { name: "Diana", age: 22, active: true },
];

// アクティブな成人ユーザーの名前を取得
const result = users
    .filter(u => u.active)
    .filter(u => u.age >= 18)
    .map(u => u.name)
    .sort();
// → ["Alice", "Diana"]

// グループ化（reduce）
const byAge = users.reduce<Record<string, User[]>>((groups, user) => {
    const key = user.age >= 18 ? "adult" : "minor";
    return { ...groups, [key]: [...(groups[key] || []), user] };
}, {});
// → { adult: [Alice, Charlie, Diana], minor: [Bob] }

// Object.groupBy（ES2024）
const grouped = Object.groupBy(users, u => u.age >= 18 ? "adult" : "minor");
```

### 3.2 複雑なビジネスロジック

```typescript
// ECサイトの注文処理パイプライン
interface Order {
    id: string;
    userId: string;
    items: OrderItem[];
    status: "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";
    createdAt: Date;
    shippingAddress: Address;
}

interface OrderItem {
    productId: string;
    name: string;
    price: number;
    quantity: number;
    category: string;
}

interface Address {
    country: string;
    prefecture: string;
    city: string;
}

interface OrderSummary {
    userId: string;
    totalOrders: number;
    totalSpent: number;
    averageOrderValue: number;
    topCategory: string;
    lastOrderDate: Date;
}

// ユーザーごとの注文サマリーを生成
function generateOrderSummaries(orders: Order[]): OrderSummary[] {
    return Object.entries(
        // 1. ユーザーIDでグループ化
        orders
            .filter(o => o.status !== "cancelled")
            .reduce<Record<string, Order[]>>((groups, order) => {
                const key = order.userId;
                return {
                    ...groups,
                    [key]: [...(groups[key] || []), order],
                };
            }, {})
    )
    // 2. 各ユーザーのサマリーを計算
    .map(([userId, userOrders]): OrderSummary => {
        const totalSpent = userOrders
            .flatMap(o => o.items)
            .reduce((sum, item) => sum + item.price * item.quantity, 0);

        // カテゴリ別の購入金額を集計
        const categorySpend = userOrders
            .flatMap(o => o.items)
            .reduce<Record<string, number>>((acc, item) => ({
                ...acc,
                [item.category]: (acc[item.category] || 0) + item.price * item.quantity,
            }), {});

        // 最も金額の大きいカテゴリを取得
        const topCategory = Object.entries(categorySpend)
            .reduce((max, [cat, amount]) =>
                amount > max[1] ? [cat, amount] : max,
                ["", 0]
            )[0];

        return {
            userId,
            totalOrders: userOrders.length,
            totalSpent,
            averageOrderValue: totalSpent / userOrders.length,
            topCategory,
            lastOrderDate: userOrders
                .map(o => o.createdAt)
                .reduce((latest, date) => date > latest ? date : latest),
        };
    })
    // 3. 総購入額の降順でソート
    .sort((a, b) => b.totalSpent - a.totalSpent);
}

// 月別売上レポートの生成
function monthlyRevenue(orders: Order[]): Map<string, number> {
    return orders
        .filter(o => o.status !== "cancelled")
        .reduce((map, order) => {
            const key = `${order.createdAt.getFullYear()}-${String(order.createdAt.getMonth() + 1).padStart(2, "0")}`;
            const orderTotal = order.items.reduce(
                (sum, item) => sum + item.price * item.quantity, 0
            );
            map.set(key, (map.get(key) || 0) + orderTotal);
            return map;
        }, new Map<string, number>());
}
```

### 3.3 CSV / JSON データの変換

```typescript
// CSV パーサー（高階関数を活用）
function parseCSV<T>(
    csv: string,
    transform: (row: Record<string, string>) => T
): T[] {
    const lines = csv.trim().split("\n");
    const headers = lines[0].split(",").map(h => h.trim());

    return lines
        .slice(1)
        .map(line => line.split(",").map(cell => cell.trim()))
        .filter(cells => cells.length === headers.length)
        .map(cells =>
            headers.reduce<Record<string, string>>(
                (obj, header, i) => ({ ...obj, [header]: cells[i] }),
                {}
            )
        )
        .map(transform);
}

// 使用例
const csvData = `
name, age, department, salary
Alice, 30, Engineering, 80000
Bob, 25, Marketing, 60000
Charlie, 35, Engineering, 95000
Diana, 28, Design, 70000
`;

interface Employee {
    name: string;
    age: number;
    department: string;
    salary: number;
}

const employees = parseCSV<Employee>(csvData, row => ({
    name: row.name,
    age: parseInt(row.age, 10),
    department: row.department,
    salary: parseInt(row.salary, 10),
}));

// 部門別の平均給与
const avgSalaryByDept = employees
    .reduce<Record<string, { total: number; count: number }>>((acc, emp) => ({
        ...acc,
        [emp.department]: {
            total: (acc[emp.department]?.total || 0) + emp.salary,
            count: (acc[emp.department]?.count || 0) + 1,
        },
    }), {});

const departmentReport = Object.entries(avgSalaryByDept)
    .map(([dept, { total, count }]) => ({
        department: dept,
        averageSalary: Math.round(total / count),
        employeeCount: count,
    }))
    .sort((a, b) => b.averageSalary - a.averageSalary);
```

---

## 4. 関数を返す高階関数

### 4.1 ファクトリパターン

```typescript
// ファクトリパターン: バリデータ生成
function createValidator(rules: Record<string, (v: any) => boolean>) {
    return function validate(data: Record<string, any>): string[] {
        const errors: string[] = [];
        for (const [field, rule] of Object.entries(rules)) {
            if (!rule(data[field])) {
                errors.push(`Invalid: ${field}`);
            }
        }
        return errors;
    };
}

const validateUser = createValidator({
    name: (v) => typeof v === "string" && v.length > 0,
    age: (v) => typeof v === "number" && v >= 0 && v <= 150,
    email: (v) => typeof v === "string" && v.includes("@"),
});

validateUser({ name: "", age: -1, email: "invalid" });
// → ["Invalid: name", "Invalid: age", "Invalid: email"]
```

### 4.2 ミドルウェアパターン

```typescript
// ミドルウェアパターン
type Middleware = (req: Request, next: () => Response) => Response;

function compose(...middlewares: Middleware[]) {
    return (req: Request): Response => {
        let index = 0;
        function next(): Response {
            const mw = middlewares[index++];
            if (!mw) return new Response("Not Found", { status: 404 });
            return mw(req, next);
        }
        return next();
    };
}

// Express スタイルのミドルウェアチェーン
type ExpressMiddleware<T = any> = (
    req: T,
    res: { body: string; status: number; headers: Record<string, string> },
    next: () => void
) => void;

function createPipeline<T>(...middlewares: ExpressMiddleware<T>[]) {
    return (req: T) => {
        const res = { body: "", status: 200, headers: {} as Record<string, string> };
        let index = 0;

        function next() {
            const mw = middlewares[index++];
            if (mw) {
                mw(req, res, next);
            }
        }

        next();
        return res;
    };
}

// ログ記録ミドルウェア
const logger: ExpressMiddleware = (req, res, next) => {
    console.log(`[${new Date().toISOString()}] Request received`);
    next();
    console.log(`[${new Date().toISOString()}] Response: ${res.status}`);
};

// 認証ミドルウェア
const auth: ExpressMiddleware<{ token?: string }> = (req, res, next) => {
    if (!req.token) {
        res.status = 401;
        res.body = "Unauthorized";
        return;
    }
    next();
};

// レスポンスヘッダーミドルウェア
const cors: ExpressMiddleware = (_req, res, next) => {
    res.headers["Access-Control-Allow-Origin"] = "*";
    next();
};
```

### 4.3 デコレータパターン

```typescript
// 関数デコレータ: 既存の関数に機能を追加する高階関数

// ログ記録デコレータ
function withLogging<Args extends any[], R>(
    fn: (...args: Args) => R,
    label?: string
): (...args: Args) => R {
    const name = label || fn.name || "anonymous";
    return (...args: Args): R => {
        console.log(`[${name}] called with:`, args);
        const start = performance.now();
        const result = fn(...args);
        const elapsed = performance.now() - start;
        console.log(`[${name}] returned:`, result, `(${elapsed.toFixed(2)}ms)`);
        return result;
    };
}

// メモ化デコレータ
function withMemoization<Args extends any[], R>(
    fn: (...args: Args) => R,
    keyFn?: (...args: Args) => string
): (...args: Args) => R {
    const cache = new Map<string, R>();
    return (...args: Args): R => {
        const key = keyFn ? keyFn(...args) : JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key)!;
        }
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
}

// リトライデコレータ
function withRetry<Args extends any[], R>(
    fn: (...args: Args) => Promise<R>,
    maxRetries: number = 3,
    delayMs: number = 1000
): (...args: Args) => Promise<R> {
    return async (...args: Args): Promise<R> => {
        let lastError: Error | undefined;
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await fn(...args);
            } catch (error) {
                lastError = error as Error;
                if (attempt < maxRetries) {
                    const delay = delayMs * Math.pow(2, attempt);
                    console.log(`Retry ${attempt + 1}/${maxRetries} after ${delay}ms`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        throw lastError;
    };
}

// スロットルデコレータ
function withThrottle<Args extends any[]>(
    fn: (...args: Args) => void,
    intervalMs: number
): (...args: Args) => void {
    let lastCallTime = 0;
    return (...args: Args): void => {
        const now = Date.now();
        if (now - lastCallTime >= intervalMs) {
            lastCallTime = now;
            fn(...args);
        }
    };
}

// デバウンスデコレータ
function withDebounce<Args extends any[]>(
    fn: (...args: Args) => void,
    delayMs: number
): (...args: Args) => void {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    return (...args: Args): void => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delayMs);
    };
}

// 使用例
const fetchUserData = async (userId: string) => {
    const response = await fetch(`/api/users/${userId}`);
    return response.json();
};

// デコレータの合成
const resilientFetchUser = withLogging(
    withRetry(fetchUserData, 3, 500),
    "fetchUserData"
);

const memoizedExpensiveCalc = withMemoization(
    withLogging(
        (n: number) => {
            // 重い計算
            let result = 0;
            for (let i = 0; i < n; i++) result += Math.sqrt(i);
            return result;
        },
        "expensiveCalc"
    )
);
```

### 4.4 Python のデコレータ

```python
import functools
import time
import logging
from typing import TypeVar, Callable, Any

F = TypeVar("F", bound=Callable[..., Any])

# Python のデコレータは高階関数の糖衣構文
# @decorator は func = decorator(func) と同等

# タイミングデコレータ
def timing(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper  # type: ignore

# リトライデコレータ（引数付き）
def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        wait = delay * (2 ** attempt)
                        logging.warning(
                            f"Retry {attempt + 1}/{max_attempts} "
                            f"for {func.__name__} after {wait}s: {e}"
                        )
                        time.sleep(wait)
            raise last_error
        return wrapper  # type: ignore
    return decorator

# キャッシュデコレータ（標準ライブラリ）
@functools.lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# デコレータのスタック
@timing
@retry(max_attempts=3, delay=0.5)
def fetch_data(url: str) -> dict:
    """外部APIからデータを取得"""
    import requests
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

# クラスベースのデコレータ
class RateLimiter:
    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0.0

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()
            return func(*args, **kwargs)
        return wrapper  # type: ignore

@RateLimiter(calls_per_second=2.0)
def call_api(endpoint: str) -> dict:
    """レート制限付きAPI呼び出し"""
    pass
```

---

## 5. カリー化と部分適用

### 5.1 概念の違い

```
カリー化 (Currying):
  f(a, b, c) → f(a)(b)(c)
  複数引数の関数を、1引数関数のチェーンに変換

部分適用 (Partial Application):
  f(a, b, c) → g(b, c)  （a を固定）
  一部の引数を固定した新しい関数を生成

┌─────────────────────────────────────────────┐
│  カリー化                                    │
│  add(a, b)  → add(a)(b)                     │
│  add(1, 2)  → add(1)(2) → 3                 │
│                                              │
│  部分適用                                    │
│  add(a, b)  → add1(b)  （a=1 を固定）       │
│  add(1, 2)  → add1(2)  → 3                  │
└─────────────────────────────────────────────┘
```

### 5.2 TypeScript でのカリー化

```typescript
// 手動カリー化
const add = (a: number) => (b: number) => a + b;
add(1)(2);  // → 3

const add5 = add(5);
add5(10);   // → 15
add5(20);   // → 25

// 汎用カリー化関数
function curry<A, B, C>(fn: (a: A, b: B) => C): (a: A) => (b: B) => C {
    return (a: A) => (b: B) => fn(a, b);
}

function curry3<A, B, C, D>(
    fn: (a: A, b: B, c: C) => D
): (a: A) => (b: B) => (c: C) => D {
    return (a: A) => (b: B) => (c: C) => fn(a, b, c);
}

// 使用例
const multiply = (a: number, b: number) => a * b;
const curriedMultiply = curry(multiply);
const double = curriedMultiply(2);
const triple = curriedMultiply(3);

[1, 2, 3, 4, 5].map(double);  // [2, 4, 6, 8, 10]
[1, 2, 3, 4, 5].map(triple);  // [3, 6, 9, 12, 15]

// 自動カリー化（可変長引数対応）
function autoCurry(fn: Function): Function {
    return function curried(...args: any[]): any {
        if (args.length >= fn.length) {
            return fn(...args);
        }
        return (...moreArgs: any[]) => curried(...args, ...moreArgs);
    };
}

const curriedAdd3 = autoCurry((a: number, b: number, c: number) => a + b + c);
curriedAdd3(1)(2)(3);     // → 6
curriedAdd3(1, 2)(3);     // → 6
curriedAdd3(1)(2, 3);     // → 6
curriedAdd3(1, 2, 3);     // → 6

// 実践的なカリー化の活用
const propGetter = <T>(key: keyof T) => (obj: T): T[keyof T] => obj[key];
const getName = propGetter<User>("name");
const getAge = propGetter<User>("age");

users.map(getName);  // ["Alice", "Bob", "Charlie", "Diana"]
users.map(getAge);   // [30, 17, 25, 22]

// 述語関数の生成
const greaterThan = (threshold: number) => (value: number) => value > threshold;
const isAdult = greaterThan(17);
const isSenior = greaterThan(64);

numbers.filter(greaterThan(3));  // [4, 5]
users.filter(u => isAdult(u.age));  // Alice, Charlie, Diana

// 文字列操作のカリー化
const startsWith = (prefix: string) => (str: string) => str.startsWith(prefix);
const endsWith = (suffix: string) => (str: string) => str.endsWith(suffix);
const contains = (substr: string) => (str: string) => str.includes(substr);

const files = ["index.ts", "utils.ts", "style.css", "main.js", "test.ts"];
files.filter(endsWith(".ts"));    // ["index.ts", "utils.ts", "test.ts"]
files.filter(startsWith("main")); // ["main.js"]
```

### 5.3 Rust でのクロージャとカリー化

```rust
// Rust では move クロージャでカリー化を実現
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}

fn make_multiplier(factor: f64) -> impl Fn(f64) -> f64 {
    move |x| x * factor
}

fn make_range_checker(min: i32, max: i32) -> impl Fn(i32) -> bool {
    move |x| x >= min && x <= max
}

fn main() {
    let add5 = make_adder(5);
    let double = make_multiplier(2.0);
    let is_valid_age = make_range_checker(0, 150);

    println!("{}", add5(10));          // 15
    println!("{}", double(3.14));      // 6.28
    println!("{}", is_valid_age(25));   // true
    println!("{}", is_valid_age(200));  // false

    // イテレータとの組み合わせ
    let numbers = vec![1, 2, 3, 4, 5];
    let add10 = make_adder(10);
    let result: Vec<i32> = numbers.iter()
        .map(|&n| add10(n))
        .collect();
    // → [11, 12, 13, 14, 15]

    // Fn, FnMut, FnOnce トレイトの違い
    // Fn:     &self で呼び出し（不変借用、何度でも呼べる）
    // FnMut:  &mut self で呼び出し（可変借用）
    // FnOnce: self で呼び出し（所有権を消費、1回のみ）

    // FnMut の例: 内部状態を変更するクロージャ
    fn make_counter() -> impl FnMut() -> i32 {
        let mut count = 0;
        move || {
            count += 1;
            count
        }
    }

    let mut counter = make_counter();
    println!("{}", counter()); // 1
    println!("{}", counter()); // 2
    println!("{}", counter()); // 3

    // FnOnce の例: 所有権を消費するクロージャ
    fn consume_and_print(f: impl FnOnce() -> String) {
        println!("{}", f());
        // f(); // コンパイルエラー: 2回呼べない
    }

    let name = String::from("Alice");
    consume_and_print(move || format!("Hello, {}!", name));
}
```

---

## 6. 関数合成（Function Composition）

### 6.1 基本的な関数合成

```
合成: (f . g)(x) = f(g(x))

  x → [g] → g(x) → [f] → f(g(x))

例: toUpper . trim
  "  hello  " → trim → "hello" → toUpper → "HELLO"
```

```typescript
// 基本的な合成
const compose2 = <A, B, C>(
    f: (b: B) => C,
    g: (a: A) => B
): ((a: A) => C) => (a: A) => f(g(a));

const pipe2 = <A, B, C>(
    f: (a: A) => B,
    g: (b: B) => C
): ((a: A) => C) => (a: A) => g(f(a));

// pipe: 左から右へ実行（可読性が高い）
function pipe<T>(...fns: Array<(arg: any) => any>) {
    return (initial: T) => fns.reduce((acc, fn) => fn(acc), initial as any);
}

// compose: 右から左へ実行（数学的記法に近い）
function compose<T>(...fns: Array<(arg: any) => any>) {
    return (initial: T) => fns.reduceRight((acc, fn) => fn(acc), initial as any);
}

// 使用例: テキスト処理パイプライン
const processText = pipe<string>(
    (s: string) => s.trim(),
    (s: string) => s.toLowerCase(),
    (s: string) => s.replace(/[^\w\s]/g, ""),
    (s: string) => s.replace(/\s+/g, "-"),
);

processText("  Hello, World!  ");  // → "hello-world"

// 型安全なパイプ（TypeScript 5.0+ のオーバーロード）
function typedPipe<A, B>(f1: (a: A) => B): (a: A) => B;
function typedPipe<A, B, C>(f1: (a: A) => B, f2: (b: B) => C): (a: A) => C;
function typedPipe<A, B, C, D>(
    f1: (a: A) => B, f2: (b: B) => C, f3: (c: C) => D
): (a: A) => D;
function typedPipe<A, B, C, D, E>(
    f1: (a: A) => B, f2: (b: B) => C, f3: (c: C) => D, f4: (d: D) => E
): (a: A) => E;
function typedPipe(...fns: Array<(arg: any) => any>) {
    return (initial: any) => fns.reduce((acc, fn) => fn(acc), initial);
}

// 型安全に合成される
const processUser = typedPipe(
    (user: User) => user.name,          // User → string
    (name: string) => name.toUpperCase(), // string → string
    (name: string) => name.length,       // string → number
);
// processUser の型: (user: User) => number
```

### 6.2 ポイントフリースタイル

```typescript
// ポイントフリー: 引数を明示せずに関数を合成するスタイル

// ポイントあり（通常のスタイル）
const getActiveUserNames1 = (users: User[]) =>
    users
        .filter(u => u.active)
        .map(u => u.name);

// ポイントフリーに近いスタイル
const isActive = (u: User) => u.active;
const getName = (u: User) => u.name;

const getActiveUserNames2 = (users: User[]) =>
    users.filter(isActive).map(getName);

// ヘルパー関数を使ったポイントフリー
const filter = <T>(pred: (item: T) => boolean) => (arr: T[]) =>
    arr.filter(pred);
const map = <T, U>(fn: (item: T) => U) => (arr: T[]) =>
    arr.map(fn);

const getActiveUserNames3 = pipe<User[]>(
    filter(isActive),
    map(getName),
);

// Ramda スタイルの関数合成
// npm install ramda @types/ramda
// import * as R from "ramda";
// const getActiveUserNames = R.pipe(
//     R.filter(R.prop("active")),
//     R.map(R.prop("name")),
// );
```

### 6.3 Haskell での関数合成（参考）

```haskell
-- Haskell: 関数合成の本場

-- (.) 演算子: 関数合成
-- (f . g) x = f (g x)

-- ポイントフリースタイルが自然
toSlug :: String -> String
toSlug = map toLower . filter isAlphaNum . words . unwords

-- パイプ演算子（&）
-- x & f = f x
result = [1,2,3,4,5]
    & filter even     -- [2, 4]
    & map (* 2)       -- [4, 8]
    & sum              -- 12

-- 高階関数の基本
map :: (a -> b) -> [a] -> [b]
filter :: (a -> Bool) -> [a] -> [a]
foldl :: (b -> a -> b) -> b -> [a] -> b
foldr :: (a -> b -> b) -> b -> [a] -> b

-- 高階関数の合成例
wordCount :: String -> [(String, Int)]
wordCount =
    map (\ws -> (head ws, length ws))  -- グループを (単語, 出現回数) に
    . group                             -- 同じ単語をグループ化
    . sort                              -- ソート
    . words                             -- 単語分割
```

---

## 7. flatMap（bind / chain）

### 7.1 基本概念

```typescript
// flatMap: map + flatten（ネストしたコレクションを平坦化）
const sentences = ["hello world", "foo bar baz"];

// map だとネスト
sentences.map(s => s.split(" "));
// → [["hello", "world"], ["foo", "bar", "baz"]]

// flatMap で平坦化
sentences.flatMap(s => s.split(" "));
// → ["hello", "world", "foo", "bar", "baz"]

// Option/Result での flatMap（モナドの bind）
// Promise.then は flatMap に相当
fetch("/api/user")
    .then(res => res.json())        // Response → Promise<JSON>（flatMap）
    .then(user => fetch(`/api/posts/${user.id}`))  // JSON → Promise
    .then(res => res.json());
```

### 7.2 flatMap の実践的活用

```typescript
// 1対多の関係を展開
interface Department {
    name: string;
    members: string[];
}

const departments: Department[] = [
    { name: "Engineering", members: ["Alice", "Bob", "Charlie"] },
    { name: "Design", members: ["Diana", "Eve"] },
    { name: "Marketing", members: ["Frank"] },
];

// 全メンバーのリスト
const allMembers = departments.flatMap(d => d.members);
// → ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

// メンバーと部門名のペア
const memberDepts = departments.flatMap(d =>
    d.members.map(m => ({ member: m, department: d.name }))
);
// → [
//     { member: "Alice", department: "Engineering" },
//     { member: "Bob", department: "Engineering" },
//     ...
// ]

// ネストした配列の完全な平坦化
const deepNested = [[[1, 2], [3]], [[4, 5]], [[6]]];
const fullyFlat = deepNested.flat(Infinity); // [1, 2, 3, 4, 5, 6]
// または再帰的 flatMap
function deepFlatten<T>(arr: (T | T[])[]): T[] {
    return arr.flatMap(item =>
        Array.isArray(item) ? deepFlatten(item) : [item]
    );
}

// 順列の生成
function permutations<T>(items: T[]): T[][] {
    if (items.length <= 1) return [items];
    return items.flatMap((item, i) => {
        const rest = [...items.slice(0, i), ...items.slice(i + 1)];
        return permutations(rest).map(perm => [item, ...perm]);
    });
}
permutations([1, 2, 3]);
// → [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

// 組み合わせの生成
function combinations<T>(items: T[], k: number): T[][] {
    if (k === 0) return [[]];
    if (items.length === 0) return [];
    const [first, ...rest] = items;
    const withFirst = combinations(rest, k - 1).map(c => [first, ...c]);
    const withoutFirst = combinations(rest, k);
    return [...withFirst, ...withoutFirst];
}
```

### 7.3 モナドとしての flatMap

```typescript
// モナドの本質: flatMap（bind / chain）による計算の連鎖

// Maybe モナドの簡易実装
class Maybe<T> {
    private constructor(private value: T | null) {}

    static of<T>(value: T): Maybe<T> {
        return new Maybe(value);
    }

    static nothing<T>(): Maybe<T> {
        return new Maybe<T>(null);
    }

    static fromNullable<T>(value: T | null | undefined): Maybe<T> {
        return value == null ? Maybe.nothing() : Maybe.of(value);
    }

    map<U>(fn: (value: T) => U): Maybe<U> {
        if (this.value === null) return Maybe.nothing();
        return Maybe.of(fn(this.value));
    }

    flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
        if (this.value === null) return Maybe.nothing();
        return fn(this.value);
    }

    getOrElse(defaultValue: T): T {
        return this.value ?? defaultValue;
    }

    toString(): string {
        return this.value === null ? "Nothing" : `Just(${this.value})`;
    }
}

// 使用例: 安全なプロパティアクセスチェーン
interface Config {
    database?: {
        connection?: {
            host?: string;
            port?: number;
        };
    };
}

function getDbHost(config: Config): string {
    return Maybe.fromNullable(config.database)
        .flatMap(db => Maybe.fromNullable(db.connection))
        .flatMap(conn => Maybe.fromNullable(conn.host))
        .getOrElse("localhost");
}

// Result モナドの簡易実装
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
const Err = <E>(error: E): Result<never, E> => ({ ok: false, error });

function mapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => U
): Result<U, E> {
    return result.ok ? Ok(fn(result.value)) : result;
}

function flatMapResult<T, U, E>(
    result: Result<T, E>,
    fn: (value: T) => Result<U, E>
): Result<U, E> {
    return result.ok ? fn(result.value) : result;
}

// 使用例: バリデーションチェーン
function parseAge(input: string): Result<number, string> {
    const age = parseInt(input, 10);
    if (isNaN(age)) return Err(`"${input}" is not a number`);
    if (age < 0 || age > 150) return Err(`Age ${age} is out of range`);
    return Ok(age);
}

function parseName(input: string): Result<string, string> {
    const trimmed = input.trim();
    if (trimmed.length === 0) return Err("Name cannot be empty");
    if (trimmed.length > 100) return Err("Name is too long");
    return Ok(trimmed);
}

interface ValidatedUser {
    name: string;
    age: number;
}

function validateUser(
    nameInput: string,
    ageInput: string
): Result<ValidatedUser, string> {
    const nameResult = parseName(nameInput);
    if (!nameResult.ok) return nameResult;

    const ageResult = parseAge(ageInput);
    if (!ageResult.ok) return ageResult;

    return Ok({ name: nameResult.value, age: ageResult.value });
}
```

---

## 8. 高度な高階関数パターン

### 8.1 トランスデューサー（Transducers）

```typescript
// トランスデューサー: 中間配列を生成せずに変換を合成する

// 通常のチェーン（各ステップで中間配列が生成される）
const result1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .filter(n => n % 2 === 0)  // [2, 4, 6, 8, 10] ← 中間配列
    .map(n => n * 3)           // [6, 12, 18, 24, 30] ← 中間配列
    .filter(n => n > 10);      // [12, 18, 24, 30]

// トランスデューサーの型定義
type Reducer<Acc, T> = (acc: Acc, item: T) => Acc;
type Transducer<T, U> = <Acc>(reducer: Reducer<Acc, U>) => Reducer<Acc, T>;

// map トランスデューサー
function tMap<T, U>(fn: (item: T) => U): Transducer<T, U> {
    return <Acc>(reducer: Reducer<Acc, U>): Reducer<Acc, T> =>
        (acc: Acc, item: T) => reducer(acc, fn(item));
}

// filter トランスデューサー
function tFilter<T>(pred: (item: T) => boolean): Transducer<T, T> {
    return <Acc>(reducer: Reducer<Acc, T>): Reducer<Acc, T> =>
        (acc: Acc, item: T) => pred(item) ? reducer(acc, item) : acc;
}

// トランスデューサーの合成
function tCompose<A, B, C>(
    t1: Transducer<A, B>,
    t2: Transducer<B, C>
): Transducer<A, C> {
    return <Acc>(reducer: Reducer<Acc, C>): Reducer<Acc, A> =>
        t1(t2(reducer));
}

// 使用例
const xform = tCompose(
    tCompose(
        tFilter<number>(n => n % 2 === 0),
        tMap<number, number>(n => n * 3)
    ),
    tFilter<number>(n => n > 10)
);

const arrayAppend: Reducer<number[], number> = (acc, item) => {
    acc.push(item);
    return acc;
};

const result2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    .reduce(xform(arrayAppend), []);
// → [12, 18, 24, 30]（中間配列なし、1パスで完了）
```

### 8.2 レンズ（Lenses）

```typescript
// レンズ: 不変データ構造の深いプロパティへのアクセスと更新を合成可能にする

interface Lens<S, A> {
    get: (s: S) => A;
    set: (a: A, s: S) => S;
}

// レンズの生成
function lens<S, A>(
    get: (s: S) => A,
    set: (a: A, s: S) => S
): Lens<S, A> {
    return { get, set };
}

// レンズ経由の変更
function over<S, A>(l: Lens<S, A>, fn: (a: A) => A, s: S): S {
    return l.set(fn(l.get(s)), s);
}

// レンズの合成
function composeLens<A, B, C>(
    outer: Lens<A, B>,
    inner: Lens<B, C>
): Lens<A, C> {
    return {
        get: (a: A) => inner.get(outer.get(a)),
        set: (c: C, a: A) => outer.set(inner.set(c, outer.get(a)), a),
    };
}

// 使用例
interface Address {
    street: string;
    city: string;
    zipCode: string;
}

interface Person {
    name: string;
    age: number;
    address: Address;
}

// レンズ定義
const addressLens: Lens<Person, Address> = lens(
    p => p.address,
    (a, p) => ({ ...p, address: a })
);

const cityLens: Lens<Address, string> = lens(
    a => a.city,
    (c, a) => ({ ...a, city: c })
);

// レンズの合成
const personCityLens = composeLens(addressLens, cityLens);

const alice: Person = {
    name: "Alice",
    age: 30,
    address: { street: "123 Main St", city: "Tokyo", zipCode: "100-0001" },
};

// 読み取り
personCityLens.get(alice);  // → "Tokyo"

// 更新（不変 - 新しいオブジェクトを返す）
const aliceInOsaka = personCityLens.set("Osaka", alice);
// → { name: "Alice", age: 30, address: { ...address, city: "Osaka" } }

// over: 既存の値に関数を適用
const aliceUpperCity = over(personCityLens, city => city.toUpperCase(), alice);
// → { name: "Alice", age: 30, address: { ...address, city: "TOKYO" } }
```

### 8.3 コンティニュエーション渡しスタイル（CPS）

```typescript
// CPS: 結果をコールバックに渡すスタイル

// 直接スタイル
function addDirect(a: number, b: number): number {
    return a + b;
}

// CPS（コンティニュエーション渡し）
function addCPS(a: number, b: number, k: (result: number) => void): void {
    k(a + b);
}

// CPS での複雑な計算チェーン
function factorialCPS(n: number, k: (result: number) => void): void {
    if (n <= 1) {
        k(1);
    } else {
        factorialCPS(n - 1, (result) => k(n * result));
    }
}

factorialCPS(5, (result) => console.log(result));  // → 120

// CPS の実用例: 非同期操作の連鎖（コールバック地獄の原型）
function readFileCPS(
    path: string,
    onSuccess: (data: string) => void,
    onError: (err: Error) => void
): void {
    // 非同期ファイル読み取り
    try {
        const data = "file content"; // 実際にはFS操作
        onSuccess(data);
    } catch (e) {
        onError(e as Error);
    }
}

// CPS から Promise への変換
function cpsToPromise<T>(
    fn: (onSuccess: (value: T) => void, onError: (err: Error) => void) => void
): Promise<T> {
    return new Promise((resolve, reject) => {
        fn(resolve, reject);
    });
}

// Promise が CPS を構造化・標準化したものであることがわかる
const readFilePromise = (path: string) =>
    cpsToPromise<string>((resolve, reject) =>
        readFileCPS(path, resolve, reject)
    );
```

---

## 9. パフォーマンスと最適化

### 9.1 遅延評価（Lazy Evaluation）

```typescript
// 遅延イテレータ: 必要な要素だけ計算する

function* lazyMap<T, U>(iterable: Iterable<T>, fn: (item: T) => U): Generator<U> {
    for (const item of iterable) {
        yield fn(item);
    }
}

function* lazyFilter<T>(iterable: Iterable<T>, pred: (item: T) => boolean): Generator<T> {
    for (const item of iterable) {
        if (pred(item)) yield item;
    }
}

function* lazyTake<T>(iterable: Iterable<T>, n: number): Generator<T> {
    let count = 0;
    for (const item of iterable) {
        if (count >= n) return;
        yield item;
        count++;
    }
}

// 無限リストから最初の5個の偶数の二乗を取得
function* naturals(): Generator<number> {
    let n = 1;
    while (true) yield n++;
}

const result = [...lazyTake(
    lazyMap(
        lazyFilter(naturals(), n => n % 2 === 0),
        n => n * n
    ),
    5
)];
// → [4, 16, 36, 64, 100]
// 無限リストでも必要な分だけ計算される

// パイプライン API（提案中）
// TC39 Pipeline Operator Proposal
// value |> fn1 |> fn2 |> fn3
// ↓ 現時点での代替
const pipeValue = <T>(value: T) => ({
    pipe: <U>(fn: (v: T) => U) => pipeValue(fn(value)),
    value,
});

const finalResult = pipeValue(10)
    .pipe(n => n * 2)
    .pipe(n => n + 5)
    .pipe(n => n.toString())
    .value;
// → "25"
```

### 9.2 チェーンのパフォーマンス比較

```typescript
// 大量データでの高階関数 vs for ループ

// テストデータ
const largeArray = Array.from({ length: 1_000_000 }, (_, i) => i);

// 方法1: チェーン（各ステップで新しい配列生成）
console.time("chain");
const r1 = largeArray
    .filter(n => n % 2 === 0)    // 500,000要素の配列生成
    .map(n => n * 3)              // 500,000要素の配列生成
    .filter(n => n > 100_000)     // さらに配列生成
    .reduce((sum, n) => sum + n, 0);
console.timeEnd("chain");

// 方法2: for ループ（配列生成なし）
console.time("loop");
let r2 = 0;
for (let i = 0; i < largeArray.length; i++) {
    const n = largeArray[i];
    if (n % 2 === 0) {
        const tripled = n * 3;
        if (tripled > 100_000) {
            r2 += tripled;
        }
    }
}
console.timeEnd("loop");

// 方法3: reduce 一発（中間配列なし）
console.time("single-reduce");
const r3 = largeArray.reduce((sum, n) => {
    if (n % 2 === 0) {
        const tripled = n * 3;
        if (tripled > 100_000) {
            return sum + tripled;
        }
    }
    return sum;
}, 0);
console.timeEnd("single-reduce");

// 方法4: ジェネレータ（遅延評価）
console.time("generator");
const r4 = [...lazyFilter(
    lazyMap(
        lazyFilter(
            largeArray,
            n => n % 2 === 0
        ),
        n => n * 3
    ),
    n => n > 100_000
)].reduce((sum, n) => sum + n, 0);
console.timeEnd("generator");

// 実測値の目安（環境依存）:
// chain:          ~80ms  （可読性: 高、メモリ: 大）
// loop:           ~15ms  （可読性: 低、メモリ: 小）
// single-reduce:  ~20ms  （可読性: 中、メモリ: 小）
// generator:      ~120ms （可読性: 中、メモリ: 小）
//
// 結論:
// - 通常のデータ量（<10,000）ではチェーンで十分
// - 大量データでは reduce 一発 or for ループ
// - メモリ制約がある場合はジェネレータ
```

### 9.3 Rust のゼロコスト抽象化

```rust
// Rust のイテレータチェーンはゼロコスト抽象化
// コンパイル時にfor ループと同等のコードに最適化される

fn benchmark_rust() {
    let numbers: Vec<i32> = (0..1_000_000).collect();

    // 高階関数チェーン（ゼロコスト: for ループと同等性能）
    let result: i64 = numbers.iter()
        .filter(|&&n| n % 2 == 0)
        .map(|&n| n as i64 * 3)
        .filter(|&n| n > 100_000)
        .sum();

    // これは以下の for ループとほぼ同じ機械語に最適化される
    let mut result2: i64 = 0;
    for &n in &numbers {
        if n % 2 == 0 {
            let tripled = n as i64 * 3;
            if tripled > 100_000 {
                result2 += tripled;
            }
        }
    }

    // さらに SIMD 最適化が適用される場合もある
    // Rust のイテレータは:
    // 1. 中間コレクションを生成しない（遅延評価）
    // 2. 各要素に対してパイプライン全体を適用
    // 3. インライン化によりオーバーヘッドなし
    assert_eq!(result, result2);
}
```

---

## 10. 実務でよく使うパターン集

### 10.1 バリデーションの合成

```typescript
// バリデーション関数の合成
type Validator<T> = (value: T) => string | null;

function composeValidators<T>(...validators: Validator<T>[]): Validator<T> {
    return (value: T) => {
        for (const validate of validators) {
            const error = validate(value);
            if (error) return error;
        }
        return null;
    };
}

function collectErrors<T>(...validators: Validator<T>[]): (value: T) => string[] {
    return (value: T) =>
        validators
            .map(v => v(value))
            .filter((err): err is string => err !== null);
}

// バリデータの定義
const required: Validator<string> = (v) =>
    v.trim().length === 0 ? "必須項目です" : null;

const minLength = (min: number): Validator<string> => (v) =>
    v.length < min ? `${min}文字以上で入力してください` : null;

const maxLength = (max: number): Validator<string> => (v) =>
    v.length > max ? `${max}文字以内で入力してください` : null;

const pattern = (regex: RegExp, message: string): Validator<string> => (v) =>
    regex.test(v) ? null : message;

const email = pattern(
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    "有効なメールアドレスを入力してください"
);

// 合成して使用
const validateEmail = composeValidators(
    required,
    email,
    maxLength(254),
);

const validatePassword = composeValidators(
    required,
    minLength(8),
    maxLength(128),
    pattern(/[A-Z]/, "大文字を含めてください"),
    pattern(/[a-z]/, "小文字を含めてください"),
    pattern(/[0-9]/, "数字を含めてください"),
);

// すべてのエラーを収集
const allPasswordErrors = collectErrors(
    required,
    minLength(8),
    pattern(/[A-Z]/, "大文字を含めてください"),
    pattern(/[a-z]/, "小文字を含めてください"),
    pattern(/[0-9]/, "数字を含めてください"),
);

console.log(allPasswordErrors("abc"));
// → ["8文字以上で入力してください", "大文字を含めてください", "数字を含めてください"]
```

### 10.2 イベントハンドラの合成

```typescript
// React でのイベントハンドラ合成
type EventHandler<E> = (event: E) => void;

function combineHandlers<E>(...handlers: EventHandler<E>[]): EventHandler<E> {
    return (event: E) => {
        handlers.forEach(handler => handler(event));
    };
}

function conditionalHandler<E>(
    pred: (event: E) => boolean,
    handler: EventHandler<E>
): EventHandler<E> {
    return (event: E) => {
        if (pred(event)) handler(event);
    };
}

// 使用例
const logClick: EventHandler<MouseEvent> = (e) =>
    console.log(`Clicked at (${e.clientX}, ${e.clientY})`);

const trackAnalytics: EventHandler<MouseEvent> = (e) =>
    console.log("Analytics: button clicked");

const preventAndHandle: EventHandler<MouseEvent> = (e) => {
    e.preventDefault();
    // 処理...
};

const handleClick = combineHandlers(
    logClick,
    trackAnalytics,
    conditionalHandler(
        (e) => e.ctrlKey,
        (e) => console.log("Ctrl+Click detected!")
    ),
);
```

### 10.3 条件分岐の関数化

```typescript
// match パターン: switch 文の関数化
function match<T, R>(value: T, cases: Array<[((v: T) => boolean) | T, R]>, defaultValue: R): R {
    for (const [condition, result] of cases) {
        if (typeof condition === "function") {
            if ((condition as (v: T) => boolean)(value)) return result;
        } else if (condition === value) {
            return result;
        }
    }
    return defaultValue;
}

// 使用例
const httpStatus = (code: number): string =>
    match(code, [
        [200, "OK"],
        [201, "Created"],
        [400, "Bad Request"],
        [401, "Unauthorized"],
        [403, "Forbidden"],
        [404, "Not Found"],
        [(c: number) => c >= 500, "Server Error"],
    ], "Unknown");

// パターンマッチング風の分岐
type Shape =
    | { type: "circle"; radius: number }
    | { type: "rectangle"; width: number; height: number }
    | { type: "triangle"; base: number; height: number };

const area = (shape: Shape): number => {
    const handlers: Record<string, (s: any) => number> = {
        circle: (s) => Math.PI * s.radius ** 2,
        rectangle: (s) => s.width * s.height,
        triangle: (s) => 0.5 * s.base * s.height,
    };
    return handlers[shape.type](shape);
};
```

### 10.4 状態管理（Redux パターン）

```typescript
// Redux: 高階関数による状態管理

type Action = { type: string; payload?: any };
type Reducer<S> = (state: S, action: Action) => S;
type Middleware<S> = (
    store: { getState: () => S; dispatch: (action: Action) => void }
) => (next: (action: Action) => void) => (action: Action) => void;

// ストア作成（高階関数のオンパレード）
function createStore<S>(
    reducer: Reducer<S>,
    initialState: S,
    ...middlewares: Middleware<S>[]
) {
    let state = initialState;
    const listeners: Array<() => void> = [];

    const getState = () => state;

    const subscribe = (listener: () => void) => {
        listeners.push(listener);
        return () => {
            const index = listeners.indexOf(listener);
            if (index > -1) listeners.splice(index, 1);
        };
    };

    let dispatch = (action: Action) => {
        state = reducer(state, action);
        listeners.forEach(l => l());
    };

    // ミドルウェアの適用（右から左に合成）
    const store = { getState, dispatch };
    const chain = middlewares.map(mw => mw(store));
    dispatch = chain.reduceRight(
        (next, mw) => mw(next),
        dispatch
    );

    return { getState, dispatch, subscribe };
}

// Reducer の合成
function combineReducers<S extends Record<string, any>>(
    reducers: { [K in keyof S]: Reducer<S[K]> }
): Reducer<S> {
    return (state: S, action: Action): S => {
        const nextState = {} as S;
        let hasChanged = false;
        for (const key in reducers) {
            const prevStateForKey = state[key];
            nextState[key] = reducers[key](prevStateForKey, action);
            if (nextState[key] !== prevStateForKey) {
                hasChanged = true;
            }
        }
        return hasChanged ? nextState : state;
    };
}

// ログミドルウェア
const loggerMiddleware: Middleware<any> = (store) => (next) => (action) => {
    console.log("dispatching:", action.type);
    console.log("before:", store.getState());
    next(action);
    console.log("after:", store.getState());
};

// Thunk ミドルウェア（非同期アクション対応）
const thunkMiddleware: Middleware<any> = (store) => (next) => (action: any) => {
    if (typeof action === "function") {
        return action(store.dispatch, store.getState);
    }
    return next(action);
};
```

---

## 11. 言語間の比較表

```
┌──────────────┬────────────┬──────────┬──────────┬──────────┬──────────┐
│ 概念          │ TypeScript │ Python   │ Rust     │ Go       │ Haskell  │
├──────────────┼────────────┼──────────┼──────────┼──────────┼──────────┤
│ map          │ .map()     │ map()    │ .map()   │ 自前実装  │ map      │
│ filter       │ .filter()  │ filter() │ .filter()│ 自前実装  │ filter   │
│ reduce/fold  │ .reduce()  │ reduce() │ .fold()  │ 自前実装  │ foldl    │
│ flatMap      │ .flatMap() │ chain()  │ .flat_map│ 自前実装  │ >>=      │
│ 合成         │ 手動       │ 手動     │ 手動     │ 手動      │ (.)      │
│ カリー化     │ 手動       │ partial  │ クロージャ│ クロージャ│ 自動     │
│ パイプ       │ 提案中(|>) │ -        │ -        │ -         │ & / $    │
│ 遅延評価     │ Generator  │ Iterator │ Iterator │ -         │ デフォルト│
│ パターンマッチ│ switch/if  │ match    │ match    │ switch    │ case     │
│ クロージャ   │ Arrow Fn   │ lambda   │ |x| expr │ func lit  │ \x->expr │
│ デコレータ   │ @実験的    │ @標準    │ マクロ    │ -         │ HoF      │
│ ゼロコスト   │ ×          │ ×        │ ○        │ △         │ ×        │
└──────────────┴────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 12. アンチパターンと注意点

### 12.1 避けるべきパターン

```typescript
// アンチパターン 1: 副作用のある map
// BAD: map で副作用を実行
users.map(u => {
    sendEmail(u.email);  // 副作用！
    return u.name;
});

// GOOD: forEach で副作用、map で変換を分離
users.forEach(u => sendEmail(u.email));
const names = users.map(u => u.name);

// アンチパターン 2: 不必要な中間配列
// BAD: 大量データで中間配列を多数生成
const result = hugeArray
    .map(transform1)    // 中間配列 1
    .filter(predicate1) // 中間配列 2
    .map(transform2)    // 中間配列 3
    .filter(predicate2) // 中間配列 4
    .reduce(aggregate, init);

// GOOD: reduce で一括処理
const result2 = hugeArray.reduce((acc, item) => {
    const t1 = transform1(item);
    if (!predicate1(t1)) return acc;
    const t2 = transform2(t1);
    if (!predicate2(t2)) return acc;
    return aggregate(acc, t2);
}, init);

// アンチパターン 3: 過度なポイントフリー
// BAD: 読めない
const processData = compose(
    sortBy(prop("score")),
    map(over(lensProp("name"), toUpper)),
    filter(both(propSatisfies(gt(__, 18), "age"), prop("active"))),
    uniqBy(prop("id")),
);

// GOOD: 適度に名前をつける
const isEligible = (u: User) => u.age > 18 && u.active;
const normalizeName = (u: User) => ({ ...u, name: u.name.toUpperCase() });
const processData = (users: User[]) =>
    users
        .filter(isEligible)
        .map(normalizeName)
        .sort((a, b) => b.score - a.score);

// アンチパターン 4: reduce の乱用
// BAD: find で十分なのに reduce を使う
const firstEven = numbers.reduce<number | null>(
    (found, n) => found !== null ? found : (n % 2 === 0 ? n : null),
    null
);

// GOOD: find を使う
const firstEven2 = numbers.find(n => n % 2 === 0);

// アンチパターン 5: 深いネスト
// BAD: 高階関数のネストが深すぎる
const result3 = arr1.flatMap(a =>
    arr2.filter(b => b.id === a.id).map(b =>
        arr3.filter(c => c.key === b.key).map(c => ({
            ...a, ...b, ...c
        }))
    )
);

// GOOD: 処理を分割して名前をつける
const lookup2 = new Map(arr2.map(b => [b.id, b]));
const lookup3 = new Map(arr3.map(c => [c.key, c]));
const result4 = arr1
    .filter(a => lookup2.has(a.id))
    .map(a => {
        const b = lookup2.get(a.id)!;
        const c = lookup3.get(b.key);
        return c ? { ...a, ...b, ...c } : null;
    })
    .filter(Boolean);
```

### 12.2 高階関数の選択ガイド

```
やりたいこと                    → 使うべき高階関数
─────────────────────────────────────────────────
各要素を変換                    → map
条件に合う要素を選択            → filter
全要素を1つの値に集約           → reduce
最初に条件を満たす要素を取得    → find
条件を満たす要素が存在するか    → some
全要素が条件を満たすか          → every
ネストした配列を平坦化+変換     → flatMap
各要素に副作用を実行            → forEach
条件で二分割                    → partition（自前実装 or lodash）
グループ化                      → groupBy（Object.groupBy / reduce）
重複排除                        → [...new Set(arr)] / reduce
ソート                          → sort（比較関数を渡す）
```

---

## まとめ

| 高階関数 | 型シグネチャ | 用途 |
|---------|------------|------|
| map | (A->B) -> [A] -> [B] | 要素の変換 |
| filter | (A->Bool) -> [A] -> [A] | 要素の選択 |
| reduce | (B,A->B) -> B -> [A] -> B | 集約 |
| flatMap | (A->[B]) -> [A] -> [B] | 変換+平坦化 |
| compose | (B->C, A->B) -> (A->C) | 関数合成 |
| curry | ((A,B)->C) -> A -> B -> C | カリー化 |
| partial | ((A,B)->C, A) -> (B->C) | 部分適用 |

高階関数を効果的に使うための原則:

1. **適切な抽象化レベル**: 単純な処理は map/filter/find で、複雑な集約は reduce で
2. **可読性優先**: ポイントフリーや過度なチェーンより、適切に名前をつけた関数を使う
3. **パフォーマンス意識**: 大量データでは中間配列の生成を抑制する
4. **副作用の分離**: map/filter/reduce は純粋関数で、副作用は forEach で
5. **型安全**: TypeScript のジェネリクスを活用して型推論を効かせる
6. **テスト容易性**: 小さな純粋関数に分割し、個別にテスト可能にする

---

## 次に読むべきガイド
-> [[03-recursion.md]] -- 再帰

---

## 参考文献
1. Bird, R. "Thinking Functionally with Haskell." Cambridge, 2014.
2. Fogus, M. "Functional JavaScript." O'Reilly, 2013.
3. Frisby, B. "Professor Frisby's Mostly Adequate Guide to Functional Programming." 2015.
4. Chiusano, P. & Bjarnason, R. "Functional Programming in Scala." Manning, 2014.
5. Hutton, G. "Programming in Haskell." Cambridge, 2016.
6. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
7. Mozilla Developer Network. "Array.prototype.map/filter/reduce." MDN Web Docs.
8. Rust Documentation. "Iterator trait." doc.rust-lang.org.
