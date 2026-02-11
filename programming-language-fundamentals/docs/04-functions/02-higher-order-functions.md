# 高階関数（Higher-Order Functions）

> 高階関数は「関数を引数に取る、または関数を返す関数」。コードの再利用性と抽象化のレベルを劇的に向上させる。

## この章で学ぶこと

- [ ] map, filter, reduce の本質を理解する
- [ ] 高階関数による抽象化パターンを習得する
- [ ] 関数型プログラミングの実践的な適用を理解する

---

## 1. 三大高階関数: map, filter, reduce

```
map:    各要素を変換する      [1,2,3] → [2,4,6]
filter: 条件に合う要素を選ぶ  [1,2,3,4,5] → [2,4]
reduce: 全要素を1つの値に集約 [1,2,3,4,5] → 15
```

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

### 実践的な活用

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

---

## 2. 関数を返す高階関数

```typescript
// ファクトリパターン
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
```

---

## 3. Rust の高階関数

```rust
// Rust: イテレータ + 高階関数（ゼロコスト抽象化）

let words = vec!["hello", "world", "foo", "bar"];

// map + collect
let upper: Vec<String> = words.iter()
    .map(|w| w.to_uppercase())
    .collect();

// filter + map + collect（filter_map で一括）
let lengths: Vec<usize> = words.iter()
    .filter(|w| w.len() > 3)
    .map(|w| w.len())
    .collect();

// 独自の高階関数
fn apply_all<T, F>(items: &[T], f: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> T,
{
    items.iter().map(|item| f(item)).collect()
}

// 関数を返す
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n
}

let add5 = make_adder(5);
add5(10);  // → 15
```

---

## 4. flatMap（bind / chain）

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

---

## まとめ

| 高階関数 | 型シグネチャ | 用途 |
|---------|------------|------|
| map | (A→B) → [A] → [B] | 要素の変換 |
| filter | (A→Bool) → [A] → [A] | 要素の選択 |
| reduce | (B,A→B) → B → [A] → B | 集約 |
| flatMap | (A→[B]) → [A] → [B] | 変換+平坦化 |
| compose | (B→C, A→B) → (A→C) | 関数合成 |

---

## 次に読むべきガイド
→ [[03-recursion.md]] — 再帰

---

## 参考文献
1. Bird, R. "Thinking Functionally with Haskell." Cambridge, 2014.
