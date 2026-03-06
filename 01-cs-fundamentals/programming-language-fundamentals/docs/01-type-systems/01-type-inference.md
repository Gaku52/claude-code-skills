# 型推論（Type Inference）

> **型推論は「型を書かなくても、コンパイラが自動的に型を決定する」仕組みである。静的型付けの安全性と動的型付けの簡潔さを両立させる、プログラミング言語設計の中核技術の一つ。**

## この章で学ぶこと

- [ ] 型推論の数学的基盤と歴史的背景を理解する
- [ ] Hindley-Milner 型推論アルゴリズムの動作原理を説明できる
- [ ] 双方向型チェックの仕組みと利点を把握する
- [ ] 主要言語（TypeScript, Rust, Go, Haskell, Kotlin, Scala）の型推論の範囲と特性を比較できる
- [ ] 型推論と明示的型注釈の適切な使い分けを判断できる
- [ ] 制約ベース型推論の手動実行ができる
- [ ] 型推論が失敗するケースを予測し、適切に対処できる

---

## 目次

1. [型推論の基礎と歴史](#1-型推論の基礎と歴史)
2. [型推論の分類体系](#2-型推論の分類体系)
3. [Hindley-Milner 型推論アルゴリズム](#3-hindley-milner-型推論アルゴリズム)
4. [双方向型チェック](#4-双方向型チェック)
5. [言語別の型推論詳解](#5-言語別の型推論詳解)
6. [型推論の限界と対処法](#6-型推論の限界と対処法)
7. [アンチパターンとベストプラクティス](#7-アンチパターンとベストプラクティス)
8. [実践演習（3段階）](#8-実践演習3段階)
9. [FAQ（よくある質問）](#9-faqよくある質問)
10. [まとめと次のステップ](#10-まとめと次のステップ)
11. [参考文献](#11-参考文献)

---

## 1. 型推論の基礎と歴史

### 1.1 型推論とは何か

型推論（Type Inference）とは、プログラマが明示的に型注釈（type annotation）を記述しなくても、コンパイラやインタプリタが式や変数の型を自動的に決定する機構である。

```
型注釈なし（型推論に委ねる）:
  let x = 42             // コンパイラが x: int と推論

型注釈あり（明示的に指定）:
  let x: int = 42        // プログラマが型を指定

両方とも同じ結果を生む。型推論は「冗長な型注釈」を省略可能にする。
```

型推論の根本的な価値は、**型安全性を犠牲にせずにコードの可読性と簡潔さを向上させること**にある。動的型付け言語のような書きやすさを、静的型付けの安全保証のもとで実現する。

### 1.2 歴史的背景

型推論の歴史は、数理論理学と計算機科学の交差点に位置する。

```
型推論の歴史年表
================================================================

1958  Curry の型割り当て
      └── コンビナトリ論理学における型の自動割り当て

1969  Hindley の主型スキーム（Principal Type Scheme）
      └── 「最も一般的な型」が一意に存在することの証明

1978  Milner の Algorithm W
      └── ML言語のための効率的な型推論アルゴリズム

1982  Damas-Milner 型システム
      └── let多相（let-polymorphism）の形式化

1985  ML言語ファミリの確立
      └── SML, OCaml の基盤

1990  Haskell 1.0
      └── Hindley-Milner の完全実装 + 型クラス

2004  C# 3.0 の var キーワード
      └── 主流言語への型推論の導入開始

2012  TypeScript 0.8
      └── JavaScript エコシステムへの型推論の導入

2014  Swift 1.0
      └── Apple エコシステムでの双方向型推論

2015  Rust 1.0
      └── 所有権システムと統合された型推論

2016  Java の var（JEP 286 提案）
      └── 2018年 Java 10 で正式導入

2021  Go 1.18 ジェネリクス
      └── 型パラメータ推論の追加

================================================================
```

### 1.3 型推論の基本的な動作原理

型推論は一般的に以下の5段階のプロセスで動作する。

```
型推論の基本プロセス
================================================================

Step 1: リテラルの型を確定
  42        --> int
  3.14      --> float
  "hello"   --> string
  true      --> bool

Step 2: 変数の型を推論
  x = 42    --> x: int （右辺の型から左辺を決定）

Step 3: 式の型を推論
  x + 1     --> int + int --> int（演算子の型規則を適用）

Step 4: 関数の型を推論
  f(x) = x + 1  --> f: int -> int（引数と戻り値の型を決定）

Step 5: 制約の解決
  未決定の型変数に対して、収集した制約を解いて型を決定

================================================================

具体例: let result = if condition then 42 else 0

  1. condition: bool（if の条件は bool）
  2. 42: int, 0: int（リテラルの型）
  3. then節と else節の型が一致 → int
  4. result: int（if式全体の型）
```

### 1.4 型推論が解決する問題

型推論が存在しない世界では、プログラマは全ての式に型注釈を記述しなければならない。これは特にジェネリクスやコレクション操作において深刻な冗長性を生む。

```typescript
// 型推論がない場合（Java 5 スタイルの冗長な記述）
Map<String, List<Pair<Integer, String>>> map =
    new HashMap<String, List<Pair<Integer, String>>>();

// 型推論がある場合（Java 10+ / TypeScript スタイル）
const map = new HashMap<String, List<Pair<Integer, String>>>();
// あるいは
let map = new Map<string, [number, string][]>();
```

この差は、コードの行数が増えるほど、そしてジェネリック型がネストするほど顕著になる。

---

## 2. 型推論の分類体系

### 2.1 推論の方向による分類

型推論には、情報がどの方向に流れるかによって複数のアプローチが存在する。

```
型推論の方向分類
================================================================

[1] ボトムアップ推論（Bottom-up / Synthesis）
    ─────────────────────────────────
    葉（リテラル・変数）から根（式全体）へ型情報が伝播

    let x = 42          42: int --> x: int
    let y = x + 1       x: int, 1: int, (+): int->int->int --> y: int

    特徴: 局所的な型情報だけで推論可能
    採用: Go, C（一部）

[2] トップダウン推論（Top-down / Checking）
    ─────────────────────────────────
    期待される型が上位のコンテキストから下位の式へ伝播

    let x: number[] = [1, 2, 3]
                       ↓ number[] を期待
                       各要素が number であることをチェック

    特徴: 文脈依存の推論が可能
    採用: Haskell（一部）, Scala 3

[3] 双方向推論（Bidirectional）
    ─────────────────────────────────
    ボトムアップとトップダウンを組み合わせ

    const f: (x: number) => string = x => x.toString();
         ↑ トップダウン: x: number        ↑ ボトムアップ: string

    特徴: 最も柔軟で実用的
    採用: TypeScript, Kotlin, Swift, Scala 3

================================================================
```

### 2.2 推論の範囲による分類

```
推論スコープの分類と比較表
================================================================

スコープ         | 説明                    | 採用言語
-----------------+-------------------------+------------------
ローカル推論     | 関数本体内のみ          | Go, C++（auto）
関数内推論       | 関数シグネチャを含む     | TypeScript, Kotlin
モジュール推論   | モジュール全体           | Rust（一部）
グローバル推論   | プログラム全体           | Haskell, ML, OCaml

================================================================

ローカル推論:
  関数の境界（引数・戻り値）では型注釈が必須。
  推論は各関数の本体内に閉じる。

  例 (Rust):
    fn add(a: i32, b: i32) -> i32 {  // 境界は明示
        let sum = a + b;              // ローカルは推論
        sum
    }

グローバル推論:
  型注釈が一切なくても、プログラム全体から型を決定。
  Hindley-Milner アルゴリズムがこれを可能にする。

  例 (Haskell):
    add x y = x + y
    -- 推論: add :: Num a => a -> a -> a
    -- 型注釈なしでも完全に型安全
```

### 2.3 推論の強さによる分類

| レベル | 説明 | 具体例 | 言語 |
|--------|------|--------|------|
| Level 0 | 推論なし | 全て明示的型注釈 | C（伝統的）, Java（<10） |
| Level 1 | 変数のみ | `var x = 42` | Go, Java 10+, C++ auto |
| Level 2 | 変数 + 戻り値 | ローカル関数の戻り値推論 | TypeScript, Kotlin |
| Level 3 | 変数 + 戻り値 + 文脈 | コールバック引数の推論 | TypeScript, Swift |
| Level 4 | 全式（型クラス制約付き） | 完全な Hindley-Milner | Haskell, ML, OCaml |

---

## 3. Hindley-Milner 型推論アルゴリズム

### 3.1 概要

Hindley-Milner（HM）型推論は、最も有名かつ理論的に完成された型推論アルゴリズムである。1969年の Hindley の研究と1978年の Milner の Algorithm W を基盤とし、1982年の Damas-Milner 型システムとして形式化された。

**HM型推論の3大特性:**

1. **完全性（Completeness）**: 型が付く全てのプログラムに対して、必ず型を推論できる
2. **主型性（Principal Type Property）**: 推論される型は常に「最も一般的な型」である
3. **決定可能性（Decidability）**: 推論は常に有限時間で終了する

### 3.2 Algorithm W の詳細

Algorithm W は HM 型推論の標準的な実装アルゴリズムである。

```
Algorithm W の動作手順
================================================================

入力: 型環境 Γ, 式 e
出力: 代入 S, 型 τ

W(Γ, e) = case e of

  [変数] x:
    Γ(x) を参照し、型スキームをインスタンス化
    → 新しい型変数で置換した単相型を返す

  [適用] e1 e2:
    (S1, τ1) = W(Γ, e1)
    (S2, τ2) = W(S1(Γ), e2)
    β = 新しい型変数
    S3 = unify(S2(τ1), τ2 → β)
    → (S3 ∘ S2 ∘ S1, S3(β))

  [抽象] λx.e:
    β = 新しい型変数
    (S1, τ1) = W(Γ ∪ {x: β}, e)
    → (S1, S1(β) → τ1)

  [let] let x = e1 in e2:
    (S1, τ1) = W(Γ, e1)
    σ = generalize(S1(Γ), τ1)   ← ここで多相化
    (S2, τ2) = W(S1(Γ) ∪ {x: σ}, e2)
    → (S2 ∘ S1, τ2)

================================================================
```

### 3.3 単一化（Unification）アルゴリズム

単一化は、2つの型表現を等しくする代入（substitution）を見つけるアルゴリズムである。HM型推論の心臓部を成す。

```
単一化アルゴリズム: unify(τ1, τ2)
================================================================

unify(α, τ)           = { α := τ }     （α が τ に出現しない場合）
unify(τ, α)           = { α := τ }     （α が τ に出現しない場合）
unify(Int, Int)        = {}              （同じ基底型）
unify(τ1→τ2, τ3→τ4)  = let S1 = unify(τ1, τ3)
                              S2 = unify(S1(τ2), S1(τ4))
                          in S2 ∘ S1
unify(τ1, τ2)         = エラー          （単一化不可能）

================================================================

出現チェック（Occurs Check）:
  unify(α, List<α>) は無限型を生むため失敗
  α = List<α> = List<List<α>> = List<List<List<α>>> = ...

  ※ 出現チェックを省略すると無限ループの原因になる
```

### 3.4 手動トレースの具体例

以下の関数に対して、HM型推論を手動でトレースする。

```
対象関数: let compose f g x = f (g x)

Step 1: 型変数の割り当て
================================================================
  f: α
  g: β
  x: γ
  compose: δ

Step 2: 式の解析と制約収集
================================================================
  式 (g x):
    g は関数なので β = γ → ε  （ε は新しい型変数）
    g x の型は ε

  式 f (g x):
    f は関数なので α = ε → ζ  （ζ は新しい型変数）
    f (g x) の型は ζ

  compose f g x の型は ζ

Step 3: 制約の解決（単一化）
================================================================
  制約一覧:
    β = γ → ε
    α = ε → ζ

  代入:
    β ↦ γ → ε
    α ↦ ε → ζ

Step 4: 最終的な型の組み立て
================================================================
  compose: α → β → γ → ζ
         = (ε → ζ) → (γ → ε) → γ → ζ

  一般的な型変数名に置換:
  compose :: (b -> c) -> (a -> b) -> a -> c

  これは Haskell の (.) 演算子と同じ型である。
================================================================
```

### 3.5 let多相（Let-Polymorphism）

HM型推論の最も重要な特徴の一つが **let多相** である。これは `let` 束縛で定義された値に多相型（polymorphic type）を付与する機構である。

```haskell
-- let多相の例
let id = \x -> x        -- id :: forall a. a -> a
in (id 42, id "hello")  -- id を int と string の両方で使用可能

-- lambda 内では多相化されない（単相制限）
(\id -> (id 42, id "hello")) (\x -> x)
-- ↑ これは型エラー！
-- id が単相型に制限されるため、int と string の両方では使えない
```

```
let多相と単相の違い
================================================================

let式:
  let id = λx.x in ...
  → id は多相型 ∀α. α → α を持つ
  → id 42 と id "hello" が同時に使える

lambda式:
  (λid. ...) (λx.x)
  → id は単相型 α → α を持つ
  → α は一つの具体型に固定される

この区別が HM 型推論の決定可能性を保証する鍵である。
System F（ランク2以上の多相）では型推論が決定不能になる。

================================================================
```

---

## 4. 双方向型チェック（Bidirectional Type Checking）

### 4.1 概要

双方向型チェックは、HM型推論に代わる現代的な型推論手法である。TypeScript, Kotlin, Swift, Scala 3 などの実用言語で広く採用されている。

**基本原理**: 型情報を2つの方向で伝播させる。

```
双方向型チェックの2つのモード
================================================================

[推論モード（Synthesis / Infer）]
  式から型を合成する（ボトムアップ）

  Γ ⊢ e ⇒ τ   「環境 Γ のもとで、式 e の型は τ と推論される」

  例:
    42 ⇒ number          リテラルの型を合成
    x ⇒ Γ(x)            変数の型を環境から取得
    f(e) ⇒ τ2           f: τ1→τ2 かつ e ⇐ τ1 のとき

[チェックモード（Checking / Check）]
  式が期待される型を持つか検証する（トップダウン）

  Γ ⊢ e ⇐ τ   「環境 Γ のもとで、式 e は型 τ を持つ」

  例:
    λx.e ⇐ τ1→τ2       引数 x に τ1 を割り当て、e ⇐ τ2 をチェック
    if c then e1 else e2 ⇐ τ   e1 ⇐ τ かつ e2 ⇐ τ をチェック

================================================================
```

### 4.2 推論モードとチェックモードの切り替え

```
双方向型チェックの情報フロー図
================================================================

const handler: (event: MouseEvent) => void = (e) => {
  console.log(e.clientX);
};

解析過程:

  1. handler の型注釈を読み取る
     期待型: (event: MouseEvent) => void
            │
  2. ────── チェックモード ──────────────────────
            │
            ▼
     (e) => { console.log(e.clientX); }
     ⇐ (event: MouseEvent) => void

  3. e に MouseEvent を割り当て
     e: MouseEvent
            │
  4. ────── 推論モード ──────────────────────────
            │
            ▼
     e.clientX ⇒ number    （MouseEvent のプロパティ）
     console.log(e.clientX) ⇒ void

  5. 戻り値の型チェック
     void ⇐ void  ✓ 成功

================================================================
```

### 4.3 TypeScript における双方向型チェックの具体例

```typescript
// 【コード例1】コールバックの型推論
// 配列のメソッドチェーンで双方向型チェックが動作する例

interface User {
    id: number;
    name: string;
    age: number;
    active: boolean;
}

const users: User[] = [
    { id: 1, name: "Alice", age: 30, active: true },
    { id: 2, name: "Bob", age: 25, active: false },
    { id: 3, name: "Charlie", age: 35, active: true },
];

// 双方向型チェックの連鎖
const result = users
    .filter(u => u.active)       // u: User（users の要素型から推論）
    .map(u => u.name)            // u: User, 戻り値: string
    .map(name => name.length)    // name: string（前段の戻り値型から推論）
    .reduce((sum, len) => sum + len, 0); // sum: number, len: number

// result: number（全てのチェーンを通じて型推論が伝播）
```

### 4.4 HM型推論との比較

```
HM型推論 vs 双方向型チェック
================================================================

                    HM型推論          双方向型チェック
------------------------------------------------------------
推論の完全性        完全              部分的（注釈が必要な場合あり）
主型性              あり              なし（注釈に依存）
サブタイピング      困難              自然にサポート
オーバーロード      困難              自然にサポート
高階多相            let多相のみ       ランク1多相（注釈で拡張可）
実装の複雑さ        中程度            低い
エラーメッセージ    やや不明確        明確（方向が分かるため）
採用言語            Haskell, ML       TypeScript, Kotlin, Swift

================================================================

HM型推論が優れている点:
  - 型注釈が一切不要でも完全に推論できる
  - 推論結果が常に「最も一般的な型」（主型）

双方向型チェックが優れている点:
  - サブタイピング（継承・共変反変）との相性が良い
  - メソッドオーバーロードを自然に扱える
  - 型エラーのメッセージが分かりやすい
  - 実装がシンプルで段階的に拡張しやすい

================================================================
```

### 4.5 フロー感応型（Flow-Sensitive Typing）

双方向型チェックの発展形として、制御フローに基づく型の絞り込み（narrowing）がある。TypeScript はこの機能を強力にサポートしている。

```typescript
// 【コード例2】フロー感応型の詳細な動作

function processValue(value: string | number | null | undefined) {
    // この時点: value: string | number | null | undefined

    if (value == null) {
        // null チェック後: value: null | undefined
        return "no value";
    }
    // ここでは: value: string | number（null と undefined が除外）

    if (typeof value === "string") {
        // typeof ガード後: value: string
        return value.toUpperCase();
    }
    // ここでは: value: number（string が除外）

    return value.toFixed(2);
}

// 判別共用体（Discriminated Union）による絞り込み
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            // shape: { kind: "circle"; radius: number }
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            // shape: { kind: "rectangle"; width: number; height: number }
            return shape.width * shape.height;
        case "triangle":
            // shape: { kind: "triangle"; base: number; height: number }
            return (shape.base * shape.height) / 2;
    }
}
```

```
フロー感応型の型状態遷移図
================================================================

function example(x: A | B | C) {

  x の型: A | B | C
  │
  ├─ if (isA(x)) ──────────► x: A ──► return
  │
  │  x の型: B | C   （A が除外された）
  │
  ├─ if (isB(x)) ──────────► x: B ──► return
  │
  │  x の型: C       （B も除外された）
  │
  └─ else ──────────────────► x: C ──► return

  各分岐で型が段階的に絞り込まれる
  （型の状態が制御フローに沿って変化する）

================================================================
```

---

## 5. 言語別の型推論詳解

### 5.1 TypeScript

TypeScript は JavaScript に型システムを追加した言語であり、双方向型チェックをベースとした実用的な型推論を提供する。

```typescript
// 【コード例3】TypeScript の型推論の全範囲

// --- ローカル変数の推論 ---
let x = 42;                    // x: number
let s = "hello";               // s: string
let b = true;                  // b: boolean
let arr = [1, 2, 3];           // arr: number[]
let obj = { name: "Alice" };   // obj: { name: string }
let tuple = [1, "a"] as const; // tuple: readonly [1, "a"]

// --- 関数の戻り値の推論 ---
function add(a: number, b: number) {
    return a + b;   // 戻り値: number と推論
}

function greet(name: string) {
    return `Hello, ${name}!`;  // 戻り値: string と推論
}

// --- 関数の引数は推論されない（明示が必要）---
// function add(a, b) { ... }  // noImplicitAny エラー

// --- コンテキストからの推論（Contextual Typing）---
const names = ["Alice", "Bob", "Charlie"];
names.map(name => name.toUpperCase());
//         ↑ name: string（配列の要素型から推論）

// --- const アサーションと推論 ---
const config = {
    host: "localhost",
    port: 3000,
} as const;
// config: { readonly host: "localhost"; readonly port: 3000 }
// リテラル型として推論される

// --- 条件型の推論 ---
type Awaited<T> = T extends Promise<infer U> ? U : T;
type Result = Awaited<Promise<string>>; // Result = string

// --- テンプレートリテラル型の推論 ---
type EventName = `on${Capitalize<"click" | "focus" | "blur">}`;
// EventName = "onClick" | "onFocus" | "onBlur"

// --- satisfies 演算子（TypeScript 4.9+）---
const palette = {
    red: [255, 0, 0],
    green: "#00ff00",
    blue: [0, 0, 255],
} satisfies Record<string, string | number[]>;
// 型チェックしつつ、推論された型を保持
// palette.red は number[] として推論される（string | number[] ではなく）
```

**TypeScript の型推論の限界:**

```typescript
// 推論が失敗する / 不十分なケース

// 1. 空の配列
const arr = [];      // arr: any[] （型が決定できない）
const arr: string[] = []; // 明示が必要

// 2. 複数の型候補がある場合
const x = Math.random() > 0.5 ? 42 : "hello";
// x: string | number （ユニオン型に推論される）

// 3. コールバックのオーバーロード
declare function on(event: "click", handler: (e: MouseEvent) => void): void;
declare function on(event: "focus", handler: (e: FocusEvent) => void): void;
// オーバーロードの解決は型推論だけでは完全でない場合がある

// 4. 再帰的な型
type JSON = string | number | boolean | null | JSON[] | { [key: string]: JSON };
// 推論コンテキストによっては明示的な型注釈が必要
```

### 5.2 Rust

Rust はローカル型推論と所有権（ownership）システムを統合した独自の型推論を持つ。

```rust
// 【コード例4】Rust の型推論の特徴

// --- 基本的なローカル推論 ---
let x = 42;                    // x: i32（デフォルト整数型）
let y = 3.14;                  // y: f64（デフォルト浮動小数点型）
let s = String::from("hello"); // s: String
let v = vec![1, 2, 3];         // v: Vec<i32>

// --- 文脈からの推論（後方参照） ---
let mut v = Vec::new();    // この時点では Vec<_>（型未定）
v.push(42);                // push の引数から Vec<i32> と確定
// Rust は「前方」だけでなく「後方」の使用箇所からも推論する

// --- ターボフィッシュ構文（明示的型パラメータ） ---
let parsed = "42".parse::<i32>().unwrap();
// parse() の戻り値型が文脈から決まらないため、::<i32> で指定

// --- クロージャの型推論 ---
let add = |a, b| a + b;
let result: i32 = add(1, 2);
// クロージャの引数型は使用箇所から推論される

// --- 所有権と型推論の相互作用 ---
let s1 = String::from("hello");
let s2 = s1;       // s1 から s2 へ所有権がムーブ
// s1 はここで無効化される（型推論 + 所有権チェック）
// println!("{}", s1); // コンパイルエラー: value used after move

// --- ライフタイム省略規則と型推論 ---
// ライフタイムもある意味での「型推論」
fn first_word(s: &str) -> &str {
    // ライフタイム省略規則により、
    // fn first_word<'a>(s: &'a str) -> &'a str と推論
    s.split_whitespace().next().unwrap_or("")
}

// --- 関数の引数・戻り値は推論されない ---
fn add(a: i32, b: i32) -> i32 {
    a + b   // 関数シグネチャでは常に型注釈が必要
}

// --- トレイト境界と型推論 ---
fn print_all<T: std::fmt::Display>(items: &[T]) {
    for item in items {
        println!("{}", item); // T: Display であることが保証
    }
}
// print_all(&[1, 2, 3]); // T = i32 と推論
// print_all(&["a", "b"]); // T = &str と推論
```

### 5.3 Go

Go は意図的にシンプルな型推論を採用している。これは Go の設計哲学「シンプルさ（simplicity）」を反映している。

```go
// Go の型推論: := 短縮変数宣言

// --- 基本的な推論 ---
x := 42              // x: int
s := "hello"         // s: string
f := 3.14            // f: float64
b := true            // b: bool
arr := []int{1,2,3}  // arr: []int

// --- var 宣言との比較 ---
var x1 int            // ゼロ値初期化、型明示
var x2 = 42           // 型推論
x3 := 42              // 短縮宣言 + 型推論（最も簡潔）

// --- 複数変数の同時推論 ---
a, b, c := 1, "hello", true
// a: int, b: string, c: bool

// --- 関数の引数・戻り値は推論されない ---
func add(a int, b int) int {
    return a + b
}

// --- Go 1.18+ ジェネリクスの型パラメータ推論 ---
func Map[T any, U any](s []T, f func(T) U) []U {
    result := make([]U, len(s))
    for i, v := range s {
        result[i] = f(v)
    }
    return result
}

// 呼び出し時に型パラメータを省略可能
nums := []int{1, 2, 3}
strs := Map(nums, func(n int) string {  // T=int, U=string と推論
    return fmt.Sprintf("%d", n)
})
```

### 5.4 Haskell

Haskell は HM 型推論を完全に実装しており、型注釈なしでも全てのプログラムに型が付く。

```haskell
-- Haskell: 最も強力な型推論

-- 推論: id :: a -> a
id x = x

-- 推論: const :: a -> b -> a
const x _ = x

-- 推論: flip :: (a -> b -> c) -> b -> a -> c
flip f x y = f y x

-- 推論: map :: (a -> b) -> [a] -> [b]
map _ []     = []
map f (x:xs) = f x : map f xs

-- 推論: foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ z []     = z
foldr f z (x:xs) = f x (foldr f z xs)

-- 推論: (.) :: (b -> c) -> (a -> b) -> a -> c
(.) f g x = f (g x)

-- 型クラス制約の自動推論
-- 推論: sort :: Ord a => [a] -> [a]
sort []     = []
sort (x:xs) = sort [y | y <- xs, y <= x]
           ++ [x]
           ++ sort [y | y <- xs, y > x]
-- Ord 制約は (<=) と (>) の使用から自動的に推論される

-- モナドの型推論
-- 推論: readAndPrint :: IO ()
readAndPrint = do
    line <- getLine        -- line :: String
    putStrLn (map toUpper line)  -- 全体 :: IO ()
```

### 5.5 Kotlin と Scala 3

```kotlin
// Kotlin: スマートキャストと型推論

// 基本的な型推論
val x = 42              // x: Int
val s = "hello"         // s: String
val list = listOf(1, 2) // list: List<Int>

// スマートキャスト（フロー感応型の一種）
fun process(obj: Any) {
    if (obj is String) {
        // obj は自動的に String にキャストされる
        println(obj.length)  // キャスト不要
    }
    // when 式でも同様
    when (obj) {
        is Int -> println(obj + 1)      // obj: Int
        is String -> println(obj.length) // obj: String
        is List<*> -> println(obj.size)  // obj: List<*>
    }
}

// ラムダの型推論
val transform: (String) -> Int = { it.length }
// it: String（期待型から推論）

// ビルダーパターンでの推論
val result = buildList {
    add(1)       // Int を追加
    add(2)       // 型が一致
    addAll(listOf(3, 4))
}
// result: List<Int>
```

```scala
// Scala 3: 高度な型推論

// 基本的な推論
val x = 42              // x: Int
val s = "hello"         // s: String

// コンテキスト関数の推論
given ord: Ordering[Int] = Ordering.Int
def sorted[T](list: List[T])(using Ordering[T]): List[T] =
  list.sorted
// using パラメータは given インスタンスから推論

// マッチ型の推論
type Elem[X] = X match
  case String      => Char
  case Array[t]    => t
  case Iterable[t] => t

// Elem[String] = Char, Elem[Array[Int]] = Int

// ユニオン型とインターセクション型
val x: String | Int = if true then "hello" else 42
// 推論された型のユニオン

// 拡張メソッドの推論
extension (s: String)
  def words: List[String] = s.split("\\s+").toList
// "hello world".words の型: List[String]
```

### 5.6 言語間の型推論能力比較表

```
言語別の型推論能力比較
================================================================

機能                | TS    | Rust  | Go    | Haskell | Kotlin | Scala3
--------------------+-------+-------+-------+---------+--------+-------
ローカル変数推論    | ○     | ○     | ○     | ○       | ○      | ○
関数戻り値推論      | ○     | ×     | ×     | ○       | △      | ○
関数引数推論        | ×     | ×     | ×     | ○       | ×      | ×
コールバック引数推論| ○     | ○     | △     | ○       | ○      | ○
ジェネリクス推論    | ○     | ○     | △     | ○       | ○      | ○
フロー感応型        | ○     | ×*    | ×     | ×       | ○      | △
型クラス/制約推論   | ×     | ○     | ×     | ○       | ×      | ○
ライフタイム推論    | N/A   | ○     | N/A   | N/A     | N/A    | N/A
パターンマッチ推論  | △     | ○     | ×     | ○       | ○      | ○

○: 完全サポート  △: 部分サポート  ×: 非サポート
*: Rust は借用チェッカーが類似の機能を提供

================================================================
```

---

## 6. 型推論の限界と対処法

### 6.1 推論が失敗する5つの典型パターン

```
型推論が失敗するパターン一覧
================================================================

パターン1: 空のコレクション
------------------------------------------------------------
  let arr = [];               // TypeScript: any[]
  let v = Vec::new();         // Rust: Vec<_> 型未定

  対処: 型注釈を追加
  let arr: number[] = [];
  let v: Vec<i32> = Vec::new();

パターン2: 複数の型候補（オーバーロード）
------------------------------------------------------------
  // 複数の解釈が可能な場合
  let result = parse(input);  // parse の戻り値型が複数ある

  対処: 型注釈またはターボフィッシュ
  let result: User = parse(input);
  let result = input.parse::<i32>();

パターン3: 再帰的データ構造
------------------------------------------------------------
  // 自己参照する型は推論が困難
  type Tree = { value: number; children: Tree[] };
  let tree = { value: 1, children: [] };
  // children: never[]（Tree[] と推論されない）

  対処: 明示的型注釈
  let tree: Tree = { value: 1, children: [] };

パターン4: 高階関数の部分適用
------------------------------------------------------------
  // 部分適用時に型が確定しない
  const apply = (f: Function) => f; // 型情報が失われる

  対処: ジェネリクスを使用
  const apply = <T, U>(f: (x: T) => U) => f;

パターン5: 異なるブランチの型不一致
------------------------------------------------------------
  // 条件分岐で異なる型を返す場合
  function getValue(flag: boolean) {
      if (flag) return 42;
      else return "hello";
  }
  // 戻り値: number | string（意図しないユニオン型）

  対処: 戻り値型を明示するか、設計を見直す

================================================================
```

### 6.2 型注釈を書くべき場所・省略すべき場所

```
型注釈の判断基準マトリクス
================================================================

                            推論結果が     推論結果が
                            明白           不明瞭
                        ┌──────────────┬──────────────┐
  公開API               │  書く(*)     │  必ず書く    │
  （関数引数・戻り値）   │              │              │
                        ├──────────────┼──────────────┤
  内部実装               │  省略OK      │  書く        │
  （ローカル変数等）     │              │              │
                        └──────────────┴──────────────┘

  (*) 公開APIは推論が明白でもドキュメントとして型注釈を書くべき

================================================================

具体的ガイドライン:

  必ず書く:
    [1] 関数の引数型（公開・非公開問わず）
    [2] 公開関数の戻り値型
    [3] 空のコレクションの初期化
    [4] any / unknown 型が推論される場合
    [5] 型キャスト（as / type assertion）の結果

  省略してよい:
    [1] リテラルからの代入（let x = 42）
    [2] 明白な関数呼び出しの結果（let len = str.length）
    [3] コールバック引数（names.map(n => ...)）
    [4] 中間変数（パイプラインの途中結果）
    [5] 構造分割代入（const { name, age } = user）

================================================================
```

### 6.3 型推論のパフォーマンスへの影響

型推論は一般的にコンパイル時間に影響を与える。特に大規模プロジェクトでは無視できないこともある。

```
型推論のコンパイル時間への影響
================================================================

言語        | 推論の計算量       | 大規模プロジェクトでの影響
------------+-------------------+---------------------------
Haskell     | ほぼ線形 O(n)     | 型クラス解決で遅くなる場合あり
Rust        | ほぼ線形 O(n)     | トレイト解決 + 借用チェックが支配的
TypeScript  | 最悪指数関数的     | 条件型のネストで遅くなる場合あり
Go          | 線形 O(n)         | 影響は軽微
Scala 3     | 最悪指数関数的     | 暗黙の解決（given/using）が重い場合あり

================================================================

TypeScript で型推論がコンパイルを遅くする例:
  // 深いネストの条件型
  type DeepPartial<T> = {
    [P in keyof T]?: T[P] extends object
      ? DeepPartial<T[P]>
      : T[P];
  };
  // 大きなオブジェクト型に適用すると爆発的に遅くなる可能性

  対策:
  - 型の再帰深度を制限する
  - 中間型に明示的な型注釈を付ける
  - interface を type alias より優先する（構造比較のキャッシュが効く）

================================================================
```

---

## 7. アンチパターンとベストプラクティス

### 7.1 アンチパターン1: 過剰な型注釈（Annotation Overkill）

型推論が正確に動作する場所に冗長な型注釈を追加すると、コードの可読性が低下し、保守性も悪化する。

```typescript
// *** アンチパターン: 過剰な型注釈 ***

// BAD: 全てに型注釈を付ける（冗長）
const name: string = "Alice";
const age: number = 30;
const active: boolean = true;
const scores: number[] = [90, 85, 92];
const user: { name: string; age: number } = { name: "Alice", age: 30 };
const doubled: number[] = scores.map((s: number): number => s * 2);

// GOOD: 推論に任せる（簡潔）
const name = "Alice";
const age = 30;
const active = true;
const scores = [90, 85, 92];
const user = { name: "Alice", age: 30 };
const doubled = scores.map(s => s * 2);
```

```
なぜ問題なのか:
================================================================

1. 可読性の低下
   - 型注釈がノイズになり、ロジックが見えにくくなる
   - 情報の重複（右辺からも型は明らか）

2. 保守性の悪化
   - 値を変更したとき、型注釈も同時に変更が必要
   - 型注釈と実際の型が乖離するリスク

3. リファクタリングの困難化
   - 冗長な型注釈があると、型の変更が広範囲に波及

判断基準:
  「この型注釈を消しても、読む人は型が分かるか？」
  → Yes なら省略してよい
  → No なら書くべき

================================================================
```

### 7.2 アンチパターン2: any への逃避（Any Escape Hatch）

型推論が困難な場面で安易に `any` 型を使用すると、型安全性が破壊される。

```typescript
// *** アンチパターン: any への逃避 ***

// BAD: any で型チェックを無効化
function processData(data: any): any {
    return data.map((item: any) => item.name.toUpperCase());
    // 全ての型情報が失われている
    // data が配列でなくても、item.name が存在しなくてもコンパイルが通る
}

// GOOD: 適切な型を定義する
interface DataItem {
    name: string;
    value: number;
}

function processData(data: DataItem[]): string[] {
    return data.map(item => item.name.toUpperCase());
    // 型安全: data が DataItem[] でなければコンパイルエラー
}

// BETTER: ジェネリクスで汎用化する
function processData<T extends { name: string }>(data: T[]): string[] {
    return data.map(item => item.name.toUpperCase());
    // name プロパティを持つ任意の型の配列を受け付ける
}

// 段階的な改善: any → unknown → 具体型
function safeProcess(data: unknown): string[] {
    if (!Array.isArray(data)) {
        throw new Error("Expected array");
    }
    return data.map(item => {
        if (typeof item === "object" && item !== null && "name" in item) {
            return String((item as { name: unknown }).name).toUpperCase();
        }
        throw new Error("Invalid item");
    });
}
```

```
any vs unknown vs never の使い分け
================================================================

型        | 安全性 | 用途
----------+--------+------------------------------------------
any       | ×      | 型チェックの完全な無効化（極力避ける）
unknown   | ○      | 型が不明だが安全に扱いたい場合
never     | ○      | 到達不能なコードの型（網羅性チェック）
object    | △      | null でないオブジェクト全般

any の正当な用途:
  - 外部ライブラリの型定義がない場合の一時的な措置
  - テストコードでのモック（型安全性より柔軟性を優先）
  - 移行期間中の段階的な型付け

================================================================
```

### 7.3 アンチパターン3: 不適切な型アサーション

```typescript
// *** アンチパターン: 型アサーションの乱用 ***

// BAD: 根拠のない型アサーション
const data = JSON.parse(response) as User;
// JSON.parse は any を返す。User である保証はない。

// BAD: ダブルアサーション（型安全性の完全な破壊）
const x = ("hello" as unknown) as number;
// string を number に強制変換（ランタイムエラーの温床）

// GOOD: 型ガードで安全に絞り込む
function isUser(obj: unknown): obj is User {
    return (
        typeof obj === "object" &&
        obj !== null &&
        "name" in obj &&
        "age" in obj &&
        typeof (obj as User).name === "string" &&
        typeof (obj as User).age === "number"
    );
}

const data: unknown = JSON.parse(response);
if (isUser(data)) {
    // ここでは data: User と推論される（型ガードによる絞り込み）
    console.log(data.name);
}

// BETTER: zod 等のバリデーションライブラリを使用
import { z } from "zod";
const UserSchema = z.object({
    name: z.string(),
    age: z.number(),
});
type User = z.infer<typeof UserSchema>;

const data = UserSchema.parse(JSON.parse(response));
// data: User（バリデーション済み）
```

### 7.4 ベストプラクティスまとめ

```
型推論のベストプラクティス
================================================================

[1] 公開APIの境界では型注釈を必ず書く
    → 関数の引数、戻り値、エクスポートされる型

[2] ローカル変数は推論に任せる
    → let x = 42; で十分。let x: number = 42; は冗長

[3] 空のコレクションには型注釈を付ける
    → const arr: string[] = [];

[4] any を避け、unknown を使う
    → 型が不明な場合は unknown + 型ガード

[5] 型アサーション（as）より型ガードを優先する
    → ランタイムの安全性を確保

[6] 推論結果が不明確な場合は型注釈を追加する
    → 読み手の理解を助ける

[7] const アサーションを活用する
    → as const でリテラル型を保持

[8] IDE の型表示を活用して推論結果を確認する
    → ホバーで推論された型を確認する習慣を付ける

================================================================
```

---

## 8. 実践演習（3段階）

### 演習1: [基礎] 型推論の確認と理解

**目的**: 各言語の型推論がどこまで自動的に型を決定するかを体験する。

**課題 1-1**: TypeScript で以下のコードを記述し、IDE のホバー機能で各変数の推論型を確認せよ。

```typescript
// 各変数の型を IDE で確認し、コメントに記入せよ
const a = 42;                          // a: ???
const b = [1, "hello", true];          // b: ???
const c = { x: 1, y: "hello" };       // c: ???
const d = new Map();                   // d: ???
const e = Promise.resolve(42);         // e: ???
const f = (x: number) => x > 0;       // f: ???

const g = [1, 2, 3].map(x => x * 2);            // g: ???
const h = [1, 2, 3].filter(x => x > 1);         // h: ???
const i = [1, 2, 3].reduce((acc, x) => acc + x); // i: ???
```

**期待される回答:**

```typescript
const a = 42;                          // a: number
const b = [1, "hello", true];          // b: (string | number | boolean)[]
const c = { x: 1, y: "hello" };       // c: { x: number; y: string }
const d = new Map();                   // d: Map<any, any>
const e = Promise.resolve(42);         // e: Promise<number>
const f = (x: number) => x > 0;       // f: (x: number) => boolean

const g = [1, 2, 3].map(x => x * 2);            // g: number[]
const h = [1, 2, 3].filter(x => x > 1);         // h: number[]
const i = [1, 2, 3].reduce((acc, x) => acc + x); // i: number
```

**課題 1-2**: Rust で以下のコードをコンパイルし、コンパイラの型推論を確認せよ。

```rust
fn main() {
    let a = 42;                    // 型は？
    let b = vec![1, 2, 3];        // 型は？
    let c = "hello".to_string();  // 型は？
    let d = (1, "hello", true);   // 型は？

    // 以下の行を追加して、型がどう変わるか確認
    let e: u8 = a;  // これはコンパイルエラーになるか？
    // ヒント: a のデフォルト型は i32 だが、
    //         e の型注釈の影響を受けるか？
}
```

### 演習2: [応用] 型推論の限界を体験し解決する

**目的**: 型推論が失敗するケースを作成し、適切な型注釈で解決する方法を学ぶ。

**課題 2-1**: TypeScript で型推論が失敗する5つのケースを作成し、それぞれ修正せよ。

```typescript
// ケース1: 空の配列
const items = [];  // any[] になる → 修正せよ

// ケース2: 条件式の型拡大
const value = Math.random() > 0.5 ? 42 : "hello";
// string | number になるが、number だけにしたい → 修正せよ

// ケース3: オブジェクトリテラルの余剰プロパティ
interface Config {
    host: string;
    port: number;
}
const config = { host: "localhost", port: 3000, debug: true };
// Config として使いたいが debug が余剰 → 修正せよ

// ケース4: Promise チェーンの型
async function fetchData() {
    const response = await fetch("/api/users");
    const data = await response.json(); // data: any → 修正せよ
    return data;
}

// ケース5: ジェネリクスの型パラメータが推論されない
function identity(x) { return x; }  // 修正せよ
```

**模範解答:**

```typescript
// ケース1: 型注釈を追加
const items: string[] = [];

// ケース2: 条件式を修正、または型注釈
const value: number = Math.random() > 0.5 ? 42 : 0;

// ケース3: satisfies を使用
const config = { host: "localhost", port: 3000, debug: true } satisfies Config;
// または型注釈: const config: Config = { host: "localhost", port: 3000 };

// ケース4: 型パラメータを指定
interface User { id: number; name: string; }
async function fetchData(): Promise<User[]> {
    const response = await fetch("/api/users");
    const data: User[] = await response.json();
    return data;
}

// ケース5: ジェネリクスを使用
function identity<T>(x: T): T { return x; }
```

**課題 2-2**: Rust で型推論が失敗するケースを3つ作成し、それぞれターボフィッシュまたは型注釈で解決せよ。

```rust
fn main() {
    // ケース1: parse の戻り値型が不定
    let n = "42".parse().unwrap(); // 修正せよ

    // ケース2: collect の型が不定
    let v = (0..10).collect(); // 修正せよ

    // ケース3: Default トレイトの型が不定
    let d = Default::default(); // 修正せよ
}
```

**模範解答:**

```rust
fn main() {
    // ケース1: ターボフィッシュで型を指定
    let n = "42".parse::<i32>().unwrap();
    // または型注釈: let n: i32 = "42".parse().unwrap();

    // ケース2: ターボフィッシュまたは型注釈
    let v: Vec<i32> = (0..10).collect();
    // または: let v = (0..10).collect::<Vec<i32>>();

    // ケース3: 型注釈
    let d: i32 = Default::default();
    // または: let d = i32::default();
}
```

### 演習3: [発展] Hindley-Milner 型推論の手動実行

**目的**: HM 型推論アルゴリズムを手動で実行し、型推論の内部動作を理解する。

**課題 3-1**: 以下の関数に対して、HM 型推論を手動でトレースせよ。

```
対象関数: let apply f x = f x
```

**手動トレース手順:**

```
Step 1: 型変数の割り当て
================================================================
  apply: α
  f: β
  x: γ
  式 (f x) の型: δ

Step 2: 制約の収集
================================================================
  f は引数 x に適用されるので、f は関数型:
    β = γ → δ

  apply f x の結果は (f x) なので:
    α = β → γ → δ

Step 3: 単一化
================================================================
  β = γ → δ を α に代入:
    α = (γ → δ) → γ → δ

Step 4: 一般化
================================================================
  自由型変数 γ, δ を全称量化:
    apply :: ∀ γ δ. (γ → δ) → γ → δ

  標準的な変数名に置換:
    apply :: (a -> b) -> a -> b

  これは Haskell の ($) 演算子と同じ型である。
================================================================
```

**課題 3-2**: 以下の関数に対して同様に手動トレースせよ。

```
対象関数: let twice f x = f (f x)
```

**ヒント:**

```
Step 1: f: α, x: β, 内側の (f x): γ, 外側の f γ: δ

Step 2: 制約
  内側の適用: α = β → γ
  外側の適用: α = γ → δ

Step 3: 単一化
  β → γ = γ → δ
  よって β = γ かつ γ = δ
  つまり β = γ = δ

Step 4: 結果
  twice :: (a -> a) -> a -> a
  f は同じ型を受け取って同じ型を返す関数でなければならない
```

**課題 3-3（挑戦）**: 以下の関数をトレースせよ。

```
対象関数: let fix f = f (fix f)
```

```
これは不動点コンビネータである。

Step 1: fix: α, f: β

Step 2:
  fix f の型を τ とする
  f は (fix f) に適用されるので: β = τ → τ'
  fix f の結果は f (fix f) なので: τ = τ'
  よって β = τ → τ

Step 3:
  fix :: (τ → τ) → τ
  標準変数名: fix :: (a -> a) -> a

注意: 出現チェックの観点からは、fix の定義自体が
      再帰的であり、通常の HM では型付けできない。
      Haskell では再帰束縛の特別な規則で対処する。
```

---

## 9. FAQ（よくある質問）

### Q1: 型推論があるなら、なぜ全ての言語が Haskell のように型注釈を不要にしないのか？

**A**: 3つの主要な理由がある。

**理由1: サブタイピングとの非互換性**

Hindley-Milner 型推論は、サブタイピング（継承やインターフェース実装による型の包含関係）がある型システムでは完全には動作しない。Java, TypeScript, Kotlin のようなオブジェクト指向言語では、`Dog extends Animal` のような関係があり、これが主型性を破壊する。

```
例: サブタイピングが主型性を壊す場面
================================================================

class Animal { move() { ... } }
class Dog extends Animal { bark() { ... } }
class Cat extends Animal { meow() { ... } }

function example(x) {
    x.move();  // x の型は？
}

候補:
  - Animal（最も一般的）
  - Dog（bark も使えるかもしれない）
  - Cat（meow も使えるかもしれない）
  - Animal & Serializable（他のインターフェースも？）

HM推論では「最も一般的な型」が一意に決まるが、
サブタイピングがあると「最も一般的」の定義が曖昧になる。

================================================================
```

**理由2: 可読性とドキュメンテーション**

公開APIの型注釈はドキュメントとしての役割を果たす。型注釈がなければ、関数のシグネチャを理解するために実装を読む必要がある。大規模プロジェクトでは、型注釈による明示性が保守性を大きく向上させる。

**理由3: コンパイル時間**

グローバルな型推論は、プログラム全体を解析する必要があるため、コンパイル時間が増大する。ローカル型推論（関数境界で型を明示）は、各関数を独立にコンパイルできるため、差分コンパイルが効率的になる。

### Q2: TypeScript の `as const` と通常の型推論はどう違うのか？

**A**: 通常の型推論は「型の拡大（widening）」を行うが、`as const` はリテラル型を保持する。

```typescript
// 通常の推論（型が拡大される）
const config = {
    method: "GET",        // method: string（リテラル型が string に拡大）
    retries: 3,           // retries: number
    endpoints: ["/a", "/b"] // endpoints: string[]
};
// config: { method: string; retries: number; endpoints: string[] }

// as const（型が拡大されない）
const config = {
    method: "GET",        // method: "GET"（リテラル型を保持）
    retries: 3,           // retries: 3
    endpoints: ["/a", "/b"] // endpoints: readonly ["/a", "/b"]
} as const;
// config: {
//   readonly method: "GET";
//   readonly retries: 3;
//   readonly endpoints: readonly ["/a", "/b"];
// }

// as const が有用な場面: 判別共用体のタグ
const actions = {
    increment: { type: "INCREMENT" as const, payload: 1 },
    decrement: { type: "DECREMENT" as const, payload: 1 },
};
// type フィールドがリテラル型になり、判別共用体として使える
```

```
型の拡大（Widening）と型の絞り込み（Narrowing）
================================================================

拡大（Widening）: リテラル → 一般型
  42        --> number
  "hello"   --> string
  true      --> boolean
  [1,2,3]   --> number[]

  ※ let で宣言した場合に発生
  ※ const で宣言するとリテラル型が保持される

絞り込み（Narrowing）: 一般型 → 具体型
  string | number  --> string  （typeof ガードで）
  Animal           --> Dog     （instanceof ガードで）
  Shape            --> Circle  （判別共用体で）

  ※ 制御フロー分析により自動的に発生

================================================================
```

### Q3: Rust の「ターボフィッシュ」構文 `::<Type>` はなぜ必要なのか？

**A**: Rust のローカル型推論では、型情報が不足する場面が存在する。特に、ジェネリック関数の戻り値型が呼び出し文脈だけでは決定できない場合に、ターボフィッシュが必要になる。

```rust
// ターボフィッシュが必要なケース

// 1. parse(): 戻り値型が複数の型を取りうる
let n = "42".parse::<i32>().unwrap();    // i32 として解析
let n = "42".parse::<f64>().unwrap();    // f64 として解析
let n = "42".parse::<u8>().unwrap();     // u8 として解析

// 2. collect(): イテレータからの変換先が複数ある
let v = (0..10).collect::<Vec<i32>>();       // Vec に変換
let s = (0..10).map(|i| format!("{}", i))
               .collect::<String>();          // String に結合
let hs = vec![1,2,3].into_iter()
                     .collect::<HashSet<_>>(); // HashSet に変換

// 3. Default::default(): 型によって異なるデフォルト値
let x = i32::default();    // 0
let s = String::default(); // ""
let v = Vec::<i32>::default(); // []

// ターボフィッシュの名前の由来:
// ::<> の形が魚（特にターボスネイル）に見えることから
//   ::<>  ← これが魚に見える？
```

### Q4: 型推論とジェネリクスはどのような関係にあるのか？

**A**: 型推論はジェネリクスの型パラメータを自動的に決定する機構として密接に関係している。

```typescript
// ジェネリクスの型パラメータ推論

// 明示的に型パラメータを指定
const result1 = identity<number>(42);

// 型パラメータを推論（引数から推論される）
const result2 = identity(42);  // T = number と推論

// 複数の型パラメータの推論
function merge<A, B>(a: A, b: B): A & B {
    return { ...a, ...b };
}
const merged = merge({ name: "Alice" }, { age: 30 });
// A = { name: string }, B = { age: number } と推論
// 戻り値: { name: string } & { age: number }

// 制約付きジェネリクスの推論
function getLength<T extends { length: number }>(item: T): number {
    return item.length;
}
getLength("hello");     // T = string と推論（string は length を持つ）
getLength([1, 2, 3]);   // T = number[] と推論
// getLength(42);       // エラー: number は length を持たない
```

```
ジェネリクス推論のフロー図
================================================================

function map<T, U>(arr: T[], fn: (item: T) => U): U[]

呼び出し: map([1, 2, 3], x => x.toString())

推論プロセス:
  1. 第1引数 [1, 2, 3] から T = number を推論
     arr: T[]  <-->  [1, 2, 3]: number[]
          │
          ▼
     T = number

  2. T = number をコールバックに伝播
     fn: (item: T) => U  -->  fn: (item: number) => U
                                    │
                                    ▼
     x => x.toString() の x: number

  3. コールバックの戻り値から U を推論
     x.toString(): string  -->  U = string

  4. 最終結果
     map<number, string>([1, 2, 3], x => x.toString()): string[]

================================================================
```

### Q5: なぜ TypeScript では関数の引数型が推論されないのか？

**A**: TypeScript が採用している双方向型チェックでは、関数宣言の引数型は「推論の起点」として使われるため、推論の対象にはならない。これは意図的な設計判断である。

```
関数引数が推論されない理由
================================================================

1. 関数は「型情報の提供者」
   引数型は、関数本体内の式の型推論の起点となる。
   起点自体が推論対象になると、循環依存が生じる。

2. 公開APIの明確性
   関数の引数型はAPIの契約を定義する。
   推論に頼ると、実装の変更でAPIが変わってしまう。

3. エラーメッセージの品質
   引数型が明示されていれば、型エラーの位置が明確になる。
   推論に頼ると「どこが間違っているか」の特定が困難になる。

例外: コールバックの引数は推論される
  names.map(name => name.toUpperCase())
  //        ↑ name: string（配列の要素型から推論）

  これは「コールバックの型」が上位コンテキストから
  確定しているため、チェックモードで推論可能。

================================================================
```

### Q6: 型推論のデバッグ方法は？

**A**: 各言語とツールには、推論された型を確認するための方法が用意されている。

```
型推論のデバッグ手法一覧
================================================================

TypeScript:
  - IDE のホバー表示（VSCode, WebStorm）
  - tsc --noEmit --declaration で .d.ts 生成
  - // @ts-expect-error で意図的にエラーを出し型を確認
  - type Inspect<T> = T; で中間型を可視化

Rust:
  - コンパイラのエラーメッセージに推論された型が表示される
  - let _: () = expr; で expr の型をエラーメッセージで確認
  - rust-analyzer の inlay hints（IDE 内型表示）
  - #[derive(Debug)] + println!("{:?}", x) で型をランタイム確認

Haskell:
  - :type 式  （GHCi で型を確認）
  - :info 名前 （型情報を表示）
  - -fwarn-missing-signatures でトップレベルの型警告
  - _ を型注釈に使い、コンパイラに推論型を表示させる

Go:
  - IDE のホバー表示
  - fmt.Printf("%T\n", x) で型を表示
  - go vet による型チェック

================================================================
```

### Q7: 依存型（Dependent Types）と型推論の関係は？

**A**: 依存型は、値に依存する型を表現する。例えば「長さ n のベクトル」を型レベルで表現できる。依存型を持つ言語（Idris, Agda, Coq）でも型推論は提供されるが、完全ではない。依存型の型推論は一般に決定不能（undecidable）であるため、ユーザーによる型注釈やヒントが必要になる場面が増える。

```
型推論の決定可能性スペクトラム
================================================================

  完全に決定可能             部分的に決定可能        決定不能
  ◄────────────────────────────────────────────────────────►
  │                    │                    │
  HM型推論             双方向型チェック        依存型
  (Haskell, ML)        (TypeScript, Kotlin)   (Idris, Agda)
  │                    │                    │
  型注釈不要            一部型注釈が必要        多くの型注釈が必要

  ※ 表現力と推論能力のトレードオフ
  型システムが表現力を増すほど、自動推論は困難になる

================================================================
```

---

## 10. まとめと次のステップ

### 10.1 型推論の全体像

```
型推論の全体マップ
================================================================

                      型推論（Type Inference）
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         アルゴリズム     推論の範囲       言語の採用
              │               │               │
      ┌───────┼───────┐   ┌───┼───┐     ┌─────┼─────┐
      │       │       │   │   │   │     │     │     │
    HM型   双方向   制約   局所 関数 全域  TS   Rust  Haskell
    推論   型チェック ベース        内          Go   Kotlin
              │                              Scala  Swift
              │
      ┌───────┼───────┐
      │               │
   推論モード     チェックモード
   (Synthesis)    (Checking)
   ボトムアップ    トップダウン

================================================================
```

### 10.2 言語別の型推論の総合比較

| 特性 | Haskell | Rust | TypeScript | Go | Kotlin | Scala 3 |
|------|---------|------|------------|-----|--------|---------|
| 推論アルゴリズム | HM + 型クラス | ローカル HM変種 | 双方向 | ローカル | 双方向 | 双方向 + HM |
| 推論範囲 | グローバル | ローカル | ローカル + 文脈 | 変数のみ | ローカル + 文脈 | ローカル + 文脈 |
| 主型性 | あり | なし | なし | N/A | なし | 部分的 |
| 型注釈の必要度 | 低（推奨） | 中（関数境界） | 中（引数） | 中（引数・戻り値） | 中（引数） | 低〜中 |
| サブタイピング | なし | トレイト | 構造的 | インターフェース | 名前的 | 名前的 + 構造的 |
| フロー感応型 | なし | 借用チェッカー | あり | なし | スマートキャスト | パターンマッチ |
| 学習曲線 | 高 | 高 | 中 | 低 | 中 | 高 |

### 10.3 型推論の判断フローチャート

```
型注釈を書くべきかの判断フローチャート
================================================================

  型を書くべきか？
  │
  ├─ 公開API（export / pub）か？
  │   ├─ Yes → 書く（ドキュメント + 安定性）
  │   └─ No ─┐
  │           │
  │   ├─ 推論結果は正しいか？
  │   │   ├─ No → 書く（推論を上書き）
  │   │   └─ Yes ─┐
  │   │            │
  │   │   ├─ 推論結果は明白か？（読んで分かるか？）
  │   │   │   ├─ Yes → 省略（推論に任せる）
  │   │   │   └─ No → 書く（可読性のため）
  │   │   │
  │   │   └─ 空のコレクションか？
  │   │       ├─ Yes → 書く
  │   │       └─ No → 省略

================================================================
```

### 10.4 学習ロードマップ

```
型推論の学習ロードマップ
================================================================

Level 1（初級）: 型推論の基本を理解
  □ let x = 42 が int と推論される理由を説明できる
  □ IDE で推論された型を確認できる
  □ 型推論が失敗するケースを3つ挙げられる
  □ 型注釈を書くべき場所を判断できる

Level 2（中級）: 言語固有の型推論を使いこなす
  □ TypeScript の型の絞り込み（narrowing）を活用できる
  □ Rust のターボフィッシュをいつ使うか判断できる
  □ ジェネリクスの型パラメータ推論を理解している
  □ コールバックの文脈的型付けを活用できる

Level 3（上級）: 型推論の理論を理解
  □ HM 型推論の手動トレースができる
  □ 単一化アルゴリズムを説明できる
  □ let多相と単相制限の違いを説明できる
  □ 双方向型チェックの推論モード/チェックモードを説明できる
  □ 型推論が決定不能になる条件を理解している

Level 4（エキスパート）: 型システムを設計・拡張できる
  □ 新しいプログラミング言語に型推論を実装できる
  □ 型推論アルゴリズムの計算量を分析できる
  □ 依存型と型推論のトレードオフを議論できる
  □ ランク N 多相と型推論の関係を説明できる

================================================================
```

---

## 11. 参考文献

### 基礎理論

1. **Pierce, Benjamin C.** *Types and Programming Languages.* MIT Press, 2002.
   - 型システムの包括的な教科書。第22章が型推論（型再構築）を詳細に解説。Hindley-Milner アルゴリズムの形式的な定義と正当性の証明を含む。型推論を理論的に学ぶための第一の参考書。

2. **Hindley, J. Roger.** "The Principal Type-Scheme of an Object in Combinatory Logic." *Transactions of the American Mathematical Society*, vol. 146, 1969, pp. 29-60.
   - 主型スキームの存在と一意性を証明した歴史的論文。コンビナトリ論理学の文脈で、型推論の数学的基盤を確立した。

3. **Milner, Robin.** "A Theory of Type Polymorphism in Programming." *Journal of Computer and System Sciences*, vol. 17, no. 3, 1978, pp. 348-375.
   - Algorithm W を提案した画期的な論文。ML 言語のための効率的な型推論アルゴリズムを定義し、その健全性と完全性を証明した。

4. **Damas, Luis, and Robin Milner.** "Principal Type-Schemes for Functional Programs." *Proceedings of the 9th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)*, 1982, pp. 207-212.
   - Damas-Milner 型システムの定義論文。let多相を含む型推論の完全な形式化を行った。

### 双方向型チェック

5. **Pierce, Benjamin C., and David N. Turner.** "Local Type Inference." *ACM Transactions on Programming Languages and Systems (TOPLAS)*, vol. 22, no. 1, 2000, pp. 1-44.
   - 双方向型チェック（ローカル型推論）の基礎を築いた論文。推論モードとチェックモードの概念を導入し、サブタイピングと型推論の統合を実現した。

6. **Dunfield, Jana, and Neelakantan R. Krishnaswami.** "Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism." *Proceedings of the 18th ACM SIGPLAN International Conference on Functional Programming (ICFP)*, 2013, pp. 429-442.
   - 双方向型チェックを高階ランク多相に拡張した論文。実装が比較的容易でありながら完全な型チェックを実現する手法を示した。

### 言語固有の型推論

7. **TypeScript Handbook.** "Type Inference." Microsoft, https://www.typescriptlang.org/docs/handbook/type-inference.html
   - TypeScript における型推論の公式ドキュメント。Best Common Type、Contextual Typing、Type Guards の動作を解説。

8. **The Rust Reference.** "Type Inference." Rust Team, https://doc.rust-lang.org/reference/type-system.html
   - Rust における型推論とライフタイム省略規則の公式リファレンス。ターボフィッシュ構文やトレイト境界と型推論の関係を解説。

9. **Haskell 2010 Language Report.** "Declarations and Bindings, Type Inference." https://www.haskell.org/onlinereport/haskell2010/
   - Haskell における型推論の仕様。型クラス制約の推論、デフォルト規則、単相制限について規定。

### 発展的トピック

10. **Odersky, Martin, Christoph Zenger, and Matthias Zenger.** "Colored Local Type Inference." *Proceedings of the 28th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)*, 2001, pp. 14-26.
    - Scala の型推論の理論的基盤。ローカル型推論をオブジェクト指向言語に適用する手法を提案。

11. **Vytiniotis, Dimitrios, Simon Peyton Jones, Tom Schrijvers, and Martin Sulzmann.** "OutsideIn(X): Modular Type Inference with Local Assumptions." *Journal of Functional Programming*, vol. 21, no. 4-5, 2011, pp. 333-412.
    - GHC（Haskell コンパイラ）の現代的な型推論アルゴリズム。型クラス、GADT、型族との統合を扱う。

---

## 次に読むべきガイド

- [[02-generics-and-polymorphism.md]] --- ジェネリクスと多態性: 型パラメータと型推論の関係を深掘り
- [[03-type-compatibility.md]] --- 型の互換性: 構造的型付けと名前的型付け、サブタイピングの詳細
- [[04-advanced-types.md]] --- 高度な型: 交差型、ユニオン型、条件型、マップ型

---

*最終更新: 2026-03-06*

