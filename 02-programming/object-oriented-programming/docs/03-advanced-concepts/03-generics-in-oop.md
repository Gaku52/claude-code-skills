# OOPにおけるジェネリクス

> ジェネリクスは「型をパラメータ化する」仕組み。型安全性を保ちながら汎用的なコードを書くための必須技術。共変性・反変性・型消去など、深い概念を理解する。

## この章で学ぶこと

- [ ] ジェネリクスの基本と各言語での実装を理解する
- [ ] 共変性・反変性・不変性の違いを把握する
- [ ] 型消去と単相化のトレードオフを学ぶ

---

## 1. ジェネリクスの基本

```
ジェネリクスなし:
  → Object型で汎用化 → キャスト必須 → 実行時エラーのリスク

ジェネリクスあり:
  → 型パラメータで汎用化 → キャスト不要 → コンパイル時に型チェック
```

```typescript
// TypeScript: ジェネリクスの基本
// ❌ any を使う（型安全性なし）
function firstAny(arr: any[]): any {
  return arr[0];
}
const val = firstAny([1, 2, 3]); // val: any（型情報が失われる）

// ✅ ジェネリクスを使う
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}
const num = first([1, 2, 3]);      // num: number
const str = first(["a", "b"]);     // str: string

// ジェネリッククラス
class Stack<T> {
  private items: T[] = [];

  push(item: T): void { this.items.push(item); }
  pop(): T | undefined { return this.items.pop(); }
  peek(): T | undefined { return this.items[this.items.length - 1]; }
  get size(): number { return this.items.length; }
}

const numStack = new Stack<number>();
numStack.push(1);
numStack.push(2);
// numStack.push("hello"); // コンパイルエラー!
```

---

## 2. 制約付きジェネリクス（Bounded）

```typescript
// TypeScript: 型パラメータに制約を付ける
interface HasLength {
  length: number;
}

// T は length プロパティを持つ型に制約
function longest<T extends HasLength>(a: T, b: T): T {
  return a.length >= b.length ? a : b;
}

longest("hello", "world!");     // OK: string は length を持つ
longest([1, 2], [1, 2, 3]);    // OK: Array は length を持つ
// longest(10, 20);             // エラー: number は length を持たない
```

```rust
// Rust: トレイト境界
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// 複数のトレイト境界
fn print_and_compare<T: std::fmt::Display + PartialOrd>(a: T, b: T) {
    if a > b {
        println!("{} is larger", a);
    } else {
        println!("{} is larger", b);
    }
}
```

```java
// Java: 上限境界と下限境界
// 上限境界: T は Number のサブクラス
public <T extends Number> double sum(List<T> list) {
    return list.stream().mapToDouble(Number::doubleValue).sum();
}

// 下限境界: T は Integer のスーパークラス
public void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}
```

---

## 3. 共変性・反変性・不変性

```
共変性（Covariance）: 型パラメータの継承方向が同じ
  Dog extends Animal
  → List<Dog> を List<Animal> として使えるか？

反変性（Contravariance）: 型パラメータの継承方向が逆
  → Consumer<Animal> を Consumer<Dog> として使えるか？

不変性（Invariance）: どちらも不可
  → List<Dog> は List<Animal> ではない

  共変: Producer<Dog> → Producer<Animal>  （生産: 出力のみ）
  反変: Consumer<Animal> → Consumer<Dog>  （消費: 入力のみ）
  不変: Mutable<Dog> ≠ Mutable<Animal>   （両方）
```

```typescript
// TypeScript: 共変性の例（配列は共変）
class Animal { name: string = ""; }
class Dog extends Animal { breed: string = ""; }

// 読み取り専用なら共変で安全
function printNames(animals: readonly Animal[]): void {
  animals.forEach(a => console.log(a.name));
}

const dogs: Dog[] = [{ name: "ポチ", breed: "柴犬" }];
printNames(dogs); // OK: Dog[] を readonly Animal[] として使える
```

```java
// Java: PECS（Producer Extends, Consumer Super）
// Producer: 値を取り出す → extends（共変）
public double sumOfList(List<? extends Number> list) {
    double sum = 0;
    for (Number n : list) { // 読み取りのみ
        sum += n.doubleValue();
    }
    return sum;
}

// Consumer: 値を入れる → super（反変）
public void addNumbers(List<? super Integer> list) {
    list.add(1);  // 書き込みのみ
    list.add(2);
}
```

---

## 4. 型消去 vs 単相化

```
型消去（Type Erasure）: Java, TypeScript
  → コンパイル後にジェネリクスの型情報が消える
  → List<String> と List<Integer> は実行時に同じ List
  → 利点: バイナリサイズが小さい、後方互換性
  → 欠点: 実行時に型情報にアクセスできない

単相化（Monomorphization）: Rust, C++
  → 使用される型ごとに専用のコードを生成
  → Vec<i32> と Vec<String> は別々のコードに
  → 利点: ゼロコスト抽象化、インライン化可能
  → 欠点: バイナリサイズが大きくなる
```

```java
// Java: 型消去の制限
List<String> strings = new ArrayList<>();
List<Integer> ints = new ArrayList<>();

// 実行時には型情報がない
System.out.println(strings.getClass() == ints.getClass()); // true!

// 型消去により不可能な操作
// if (obj instanceof List<String>) {} // コンパイルエラー
// T[] array = new T[10];             // コンパイルエラー
```

---

## 5. 実践パターン

```typescript
// Result型: ジェネリクスの実践例
class Result<T, E> {
  private constructor(
    private readonly value?: T,
    private readonly error?: E,
    private readonly isOk: boolean = true,
  ) {}

  static ok<T, E>(value: T): Result<T, E> {
    return new Result<T, E>(value, undefined, true);
  }

  static err<T, E>(error: E): Result<T, E> {
    return new Result<T, E>(undefined, error, false);
  }

  map<U>(fn: (value: T) => U): Result<U, E> {
    if (this.isOk) return Result.ok(fn(this.value!));
    return Result.err(this.error!);
  }

  unwrapOr(defaultValue: T): T {
    return this.isOk ? this.value! : defaultValue;
  }
}

// 使用例
function parseNumber(s: string): Result<number, string> {
  const n = Number(s);
  if (isNaN(n)) return Result.err(`"${s}" is not a number`);
  return Result.ok(n);
}

const result = parseNumber("42").map(n => n * 2);
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ジェネリクス | 型をパラメータ化して汎用コード |
| 制約 | extends/super で型パラメータを制限 |
| 共変 | 出力（Producer）→ extends |
| 反変 | 入力（Consumer）→ super |
| 型消去 | Java/TS。実行時に型情報なし |
| 単相化 | Rust/C++。型ごとにコード生成 |

---

## 次に読むべきガイド
→ [[../04-practical-patterns/00-creational-patterns.md]] — 生成パターン

---

## 参考文献
1. Wadler, P. "Theorems for Free!" ICFP, 1989.
2. Bloch, J. "Effective Java." Item 31: Use bounded wildcards. 2018.
