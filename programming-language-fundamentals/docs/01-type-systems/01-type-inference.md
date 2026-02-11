# 型推論（Type Inference）

> 型推論は「型を書かなくても、コンパイラが自動的に型を決定する」仕組み。静的型付けの安全性と動的型付けの簡潔さを両立する。

## この章で学ぶこと

- [ ] 型推論の仕組みと限界を理解する
- [ ] 各言語の型推論の範囲を把握する
- [ ] 型推論と明示的型注釈の使い分けを判断できる

---

## 1. 型推論とは

```
型注釈なし（型推論）:
  let x = 42             // コンパイラが x: int と推論

型注釈あり（明示的）:
  let x: int = 42        // プログラマが型を指定

推論の仕組み（簡略化）:
  1. リテラルの型を確定     42 → int
  2. 変数の型を推論         x = 42 → x: int
  3. 式の型を推論           x + 1 → int + int → int
  4. 関数の型を推論         f(x) = x + 1 → f: int → int
  5. 制約の解決             未決定の型を制約から決定
```

---

## 2. 言語ごとの型推論

### TypeScript

```typescript
// ローカル変数: 推論される
let x = 42;              // x: number
let s = "hello";         // s: string
let arr = [1, 2, 3];     // arr: number[]
let obj = { name: "a" }; // obj: { name: string }

// 関数の戻り値: 推論される
function add(a: number, b: number) {
    return a + b;         // 戻り値: number と推論
}

// 関数の引数: 推論されない（明示が必要）
function add(a, b) {     // ❌ noImplicitAny エラー
    return a + b;
}

// コンテキストからの推論
const names = ["Alice", "Bob"];
names.map(name => name.toUpperCase());
//         ↑ name: string と推論される（配列の型から）

// 型の絞り込み（Type Narrowing）
function process(value: string | number) {
    if (typeof value === "string") {
        // ここでは value: string と推論
        return value.toUpperCase();
    }
    // ここでは value: number と推論
    return value.toFixed(2);
}
```

### Rust

```rust
// Rust: 強力なローカル型推論
let x = 42;              // x: i32（デフォルト整数型）
let y = 3.14;            // y: f64（デフォルト浮動小数点型）
let v = vec![1, 2, 3];   // v: Vec<i32>

// 文脈からの推論
let v: Vec<i64> = vec![1, 2, 3];  // 型注釈で i64 に
let n: u8 = 42;                    // 型注釈で u8 に

// ターボフィッシュ（型パラメータの明示）
let parsed = "42".parse::<i32>().unwrap();
// parse() だけでは戻り値型が決まらないため、::<i32> で指定

// クロージャの引数: 推論される
let add = |a, b| a + b;
let result: i32 = add(1, 2);  // a: i32, b: i32 と推論

// 関数の引数と戻り値: 推論されない（明示が必要）
fn add(a: i32, b: i32) -> i32 {
    a + b  // 関数では常に型注釈が必要
}
```

### Go

```go
// Go: シンプルな型推論（:= で推論）
x := 42              // x: int
s := "hello"         // s: string
arr := []int{1,2,3}  // arr: []int

// var 宣言: 型指定またはゼロ値
var x int             // x = 0
var x = 42            // 推論

// 関数の引数・戻り値: 推論されない
func add(a int, b int) int {
    return a + b
}

// Go の型推論は意図的にシンプル
// → ジェネリクスの型推論は限定的
```

### Haskell（最も強力な型推論）

```haskell
-- Haskell: Hindley-Milner 型推論
-- 型注釈なしで全ての型が推論可能

-- 推論: add :: Num a => a -> a -> a
add x y = x + y

-- 推論: factorial :: (Eq a, Num a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- 推論: map :: (a -> b) -> [a] -> [b]
-- （標準ライブラリの定義そのもの）

-- 推論: compose :: (b -> c) -> (a -> b) -> a -> c
compose f g x = f (g x)

-- Haskell では型注釈はドキュメントとして書くが、技術的には不要
-- ただし、型推論の結果を確認するために書くことが推奨される
```

---

## 3. 型推論のアルゴリズム

### Hindley-Milner 型推論

```
最も有名な型推論アルゴリズム。Haskell, ML, Rust（の一部）で使用。

手順:
  1. 各式に未知の型変数を割り当て
     let f x = x + 1
     f: α → β,  x: α,  (+): γ → δ → ε,  1: Int

  2. 制約を収集
     α = γ     （xが+の第1引数）
     Int = δ   （1が+の第2引数）
     β = ε     （+の結果がfの戻り値）
     γ = δ = ε = Int  （+はInt→Int→Int）

  3. 単一化（Unification）で制約を解く
     α = Int,  β = Int

  4. 結果
     f: Int → Int

特徴:
  - 決定的（常に最も一般的な型を推論）
  - 効率的（ほぼ線形時間）
  - 高階多相（let多相）をサポート
```

### 双方向型チェック（Bidirectional Type Checking）

```
TypeScript, Scala 3, Kotlin などで使用される現代的手法。

推論（Inference）: 式から型を推論する（ボトムアップ）
チェック（Checking）: 期待される型と照合する（トップダウン）

例:
  const f: (x: number) => string = x => x.toString();
                                    ↑
                          期待型 (number) => string から
                          x: number と推論（チェック方向）
                          x.toString(): string と推論（推論方向）
```

---

## 4. 型推論の限界

```
推論できない場面（明示的な型注釈が必要）:

  1. 関数の公開API
     → 引数・戻り値の型はドキュメントとして明示すべき

  2. 空のコレクション
     let arr = [];        // 型が決定できない
     let arr: number[] = [];  // 明示が必要

  3. 複雑なジェネリクス
     let result = parse(input);  // 戻り値型が不定
     let result: User = parse(input);  // 明示で解決

  4. 再帰的な型
     // 自己参照する型は推論が困難な場合がある

  5. オーバーロード
     // 複数の型が候補になる場合
```

### 型注釈のベストプラクティス

```
書くべき場所:
  ✓ 関数の引数と戻り値（公開API）
  ✓ クラス/構造体のフィールド
  ✓ 空のコレクション
  ✓ 型推論の結果が明白でない場合

省略してよい場所:
  ✓ ローカル変数（let x = 42）
  ✓ クロージャの引数（コンテキストから推論可能）
  ✓ 一時的な中間変数
  ✓ 推論結果が明白な場合
```

---

## 実践演習

### 演習1: [基礎] — 型推論の確認
TypeScript で型注釈を一切書かずに関数を定義し、IDE でホバーして推論結果を確認する。

### 演習2: [応用] — 推論の限界を体験
型推論が失敗するケースを TypeScript と Rust で作成し、適切な型注釈で解決する。

### 演習3: [発展] — Hindley-Milner の手動実行
簡単な関数に対して、手動で型変数の割り当て→制約収集→単一化のプロセスを実行する。

---

## まとめ

| 言語 | 推論範囲 | 特徴 |
|------|---------|------|
| Haskell | 全プログラム | Hindley-Milner。最も強力 |
| Rust | ローカル | 関数境界では明示が必要 |
| TypeScript | ローカル + コンテキスト | 双方向型チェック |
| Go | 変数のみ | 意図的にシンプル |
| Java | 限定的 | var（Java 10+）で改善 |

---

## 次に読むべきガイド
→ [[02-generics-and-polymorphism.md]] — ジェネリクスと多態性

---

## 参考文献
1. Pierce, B. "Types and Programming Languages." Ch.22, MIT Press, 2002.
2. Hindley, R. "The Principal Type-Scheme of an Object in Combinatory Logic." 1969.
