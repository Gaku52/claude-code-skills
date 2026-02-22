# OOPにおけるジェネリクス

> ジェネリクスは「型をパラメータ化する」仕組み。型安全性を保ちながら汎用的なコードを書くための必須技術。共変性・反変性・型消去など、深い概念を理解する。

## この章で学ぶこと

- [ ] ジェネリクスの基本と各言語での実装を理解する
- [ ] 共変性・反変性・不変性の違いを把握する
- [ ] 型消去と単相化のトレードオフを学ぶ
- [ ] 高度なジェネリクスパターン（高カインド型、型レベルプログラミング）を知る
- [ ] 実務でジェネリクスを適切に活用する設計判断ができるようになる

---

## 1. ジェネリクスの基本

### 1.1 なぜジェネリクスが必要か

```
ジェネリクスなし:
  → Object型で汎用化 → キャスト必須 → 実行時エラーのリスク

ジェネリクスあり:
  → 型パラメータで汎用化 → キャスト不要 → コンパイル時に型チェック
```

ジェネリクスが解決する問題は「型安全性と再利用性の両立」である。ジェネリクスがない時代、汎用的なコードを書くためには Object 型（Java）や void* ポインタ（C）を使う必要があった。これは実行時にキャストエラーを引き起こすリスクがあり、バグの温床となっていた。

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

### 1.2 各言語でのジェネリクス構文

```java
// Java: ジェネリクスの基本構文
// ジェネリックメソッド
public <T> T firstElement(List<T> list) {
    if (list.isEmpty()) return null;
    return list.get(0);
}

// ジェネリッククラス
public class Pair<A, B> {
    private final A first;
    private final B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public A getFirst() { return first; }
    public B getSecond() { return second; }

    // ジェネリックメソッド内でさらに型パラメータを定義
    public <C> Triple<A, B, C> withThird(C third) {
        return new Triple<>(first, second, third);
    }
}

// ジェネリックインターフェース
public interface Repository<T, ID> {
    T findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
}

// 具象クラスが型を固定
public class UserRepository implements Repository<User, Long> {
    @Override
    public User findById(Long id) { /* ... */ }

    @Override
    public List<User> findAll() { /* ... */ }

    @Override
    public User save(User entity) { /* ... */ }

    @Override
    public void deleteById(Long id) { /* ... */ }
}
```

```csharp
// C#: ジェネリクスの基本構文
// ジェネリッククラス
public class Result<T>
{
    public bool IsSuccess { get; }
    public T Value { get; }
    public string Error { get; }

    private Result(bool isSuccess, T value, string error)
    {
        IsSuccess = isSuccess;
        Value = value;
        Error = error;
    }

    public static Result<T> Success(T value) =>
        new Result<T>(true, value, null);

    public static Result<T> Failure(string error) =>
        new Result<T>(false, default, error);

    // ジェネリックメソッド
    public Result<U> Map<U>(Func<T, U> mapper)
    {
        if (!IsSuccess) return Result<U>.Failure(Error);
        return Result<U>.Success(mapper(Value));
    }
}

// 使用例
Result<int> parsed = Result<int>.Success(42);
Result<string> formatted = parsed.Map(n => $"Value: {n}");
```

```rust
// Rust: ジェネリクスの基本構文
// ジェネリック構造体
struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Self {
        Pair { first, second }
    }

    // 異なる型パラメータを持つメソッド
    fn zip_with<V, W>(self, other: Pair<V, W>) -> Pair<Pair<T, V>, Pair<U, W>> {
        Pair {
            first: Pair::new(self.first, other.first),
            second: Pair::new(self.second, other.second),
        }
    }
}

// ジェネリック列挙型（Option と Result は標準ライブラリ）
enum MyOption<T> {
    Some(T),
    None,
}

impl<T> MyOption<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> MyOption<U> {
        match self {
            MyOption::Some(value) => MyOption::Some(f(value)),
            MyOption::None => MyOption::None,
        }
    }

    fn unwrap_or(self, default: T) -> T {
        match self {
            MyOption::Some(value) => value,
            MyOption::None => default,
        }
    }
}
```

```python
# Python: typing モジュールによるジェネリクス（Python 3.12+）
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')
U = TypeVar('U')

# ジェネリッククラス
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]

    @property
    def size(self) -> int:
        return len(self._items)

# Python 3.12+ の新構文
class Pair[T, U]:
    def __init__(self, first: T, second: U) -> None:
        self.first = first
        self.second = second

    def map_first[V](self, f: Callable[[T], V]) -> 'Pair[V, U]':
        return Pair(f(self.first), self.second)

# 型ヒントによる静的チェック（mypyなど）
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
# stack.push("hello")  # mypy エラー
```

### 1.3 複数の型パラメータ

複数の型パラメータを持つジェネリクスは、異なる型間の関係を表現するのに役立つ。

```typescript
// TypeScript: 複数型パラメータ
// マップ（辞書）の型安全な操作
class TypedMap<K extends string | number, V> {
  private data = new Map<K, V>();

  set(key: K, value: V): void {
    this.data.set(key, value);
  }

  get(key: K): V | undefined {
    return this.data.get(key);
  }

  entries(): [K, V][] {
    return [...this.data.entries()];
  }

  // 値の型を変換する map メソッド
  mapValues<U>(fn: (value: V, key: K) => U): TypedMap<K, U> {
    const result = new TypedMap<K, U>();
    for (const [key, value] of this.data) {
      result.set(key, fn(value, key));
    }
    return result;
  }
}

// Either型: 2つの型のどちらかを保持
class Either<L, R> {
  private constructor(
    private readonly left?: L,
    private readonly right?: R,
    private readonly isRight: boolean = true,
  ) {}

  static left<L, R>(value: L): Either<L, R> {
    return new Either<L, R>(value, undefined, false);
  }

  static right<L, R>(value: R): Either<L, R> {
    return new Either<L, R>(undefined, value, true);
  }

  fold<T>(onLeft: (l: L) => T, onRight: (r: R) => T): T {
    return this.isRight ? onRight(this.right!) : onLeft(this.left!);
  }

  map<U>(fn: (r: R) => U): Either<L, U> {
    return this.isRight
      ? Either.right(fn(this.right!))
      : Either.left(this.left!);
  }

  flatMap<U>(fn: (r: R) => Either<L, U>): Either<L, U> {
    return this.isRight ? fn(this.right!) : Either.left(this.left!);
  }
}

// 使用例: バリデーション
type ValidationError = { field: string; message: string };

function validateAge(age: number): Either<ValidationError, number> {
  if (age < 0 || age > 150) {
    return Either.left({ field: "age", message: "年齢は0-150の範囲" });
  }
  return Either.right(age);
}

function validateName(name: string): Either<ValidationError, string> {
  if (name.length === 0) {
    return Either.left({ field: "name", message: "名前は必須" });
  }
  return Either.right(name);
}
```

---

## 2. 制約付きジェネリクス（Bounded）

### 2.1 上限境界（Upper Bound）

型パラメータに制約を付けることで、特定のインターフェースやクラスの機能を利用できるようになる。

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

```typescript
// TypeScript: オブジェクトの特定プロパティへのアクセスを型安全に
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const person = { name: "太郎", age: 30, email: "taro@example.com" };
const name = getProperty(person, "name");   // string 型
const age = getProperty(person, "age");     // number 型
// getProperty(person, "address");           // エラー: "address" は keyof typeof person に含まれない

// 複数の制約を組み合わせる
interface Serializable {
  serialize(): string;
}

interface Validatable {
  validate(): boolean;
}

// T は Serializable かつ Validatable
function processAndSave<T extends Serializable & Validatable>(entity: T): boolean {
  if (!entity.validate()) {
    console.error("Validation failed");
    return false;
  }
  const data = entity.serialize();
  console.log(`Saving: ${data}`);
  return true;
}
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

// where 句による可読性の向上
fn complex_function<T, U>(t: T, u: U) -> String
where
    T: std::fmt::Display + Clone + Send + 'static,
    U: std::fmt::Debug + Into<String>,
{
    format!("t={}, u={:?}", t, u)
}

// 関連型（Associated Types）を持つトレイト境界
fn sum_collection<C>(collection: &C) -> C::Item
where
    C: IntoIterator,
    C::Item: std::ops::Add<Output = C::Item> + Default + Copy,
{
    let mut total = C::Item::default();
    for item in collection {
        total = total + item;
    }
    total
}
```

### 2.2 下限境界（Lower Bound）

Java では `super` キーワードを使って下限境界を指定できる。これは主にコレクションへの書き込み操作で使用する。

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

// 上限境界の応用: Comparable による制約
public <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) >= 0 ? a : b;
}

// 再帰的型境界（Recursive Type Bounds）
// T は自身を比較できる型でなければならない
public <T extends Comparable<T>> void sort(List<T> list) {
    Collections.sort(list);
}

// 複数の上限境界
public <T extends Serializable & Comparable<T>> void saveOrdered(List<T> list) {
    Collections.sort(list);
    // serialize list...
}
```

### 2.3 条件付き型（Conditional Types）

TypeScript には条件付き型という高度な機能があり、型レベルでの条件分岐が可能になる。

```typescript
// TypeScript: 条件付き型
// T が配列なら要素型を抽出、そうでなければそのまま
type Unwrap<T> = T extends Array<infer U> ? U : T;

type A = Unwrap<string[]>;   // string
type B = Unwrap<number>;     // number
type C = Unwrap<boolean[]>;  // boolean

// Promise のネストを解除
type UnwrapPromise<T> = T extends Promise<infer U> ? UnwrapPromise<U> : T;

type D = UnwrapPromise<Promise<Promise<string>>>;  // string

// 関数の戻り値型を抽出
type ReturnOf<T> = T extends (...args: any[]) => infer R ? R : never;

type E = ReturnOf<() => number>;           // number
type F = ReturnOf<(x: string) => boolean>; // boolean

// 条件付き型を使ったユーティリティ
type NonNullableProperties<T> = {
  [K in keyof T]: T[K] extends null | undefined ? never : K;
}[keyof T];

interface User {
  id: number;
  name: string;
  nickname: string | null;
  avatar: string | undefined;
}

type RequiredUserKeys = NonNullableProperties<User>;
// "id" | "name" のみ

// マッピング型との組み合わせ
type ReadonlyDeep<T> = {
  readonly [K in keyof T]: T[K] extends object
    ? T[K] extends Function
      ? T[K]
      : ReadonlyDeep<T[K]>
    : T[K];
};

interface Config {
  database: {
    host: string;
    port: number;
    credentials: {
      username: string;
      password: string;
    };
  };
  features: string[];
}

type ImmutableConfig = ReadonlyDeep<Config>;
// 全てのネストされたプロパティが readonly になる
```

---

## 3. 共変性・反変性・不変性

### 3.1 基本概念

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

### 3.2 TypeScript での変性

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

```typescript
// TypeScript 4.7+: in/out キーワードで明示的な変性宣言
// out = 共変（出力位置のみ）
interface Producer<out T> {
  produce(): T;
}

// in = 反変（入力位置のみ）
interface Consumer<in T> {
  consume(value: T): void;
}

// in/out = 不変（入力・出力の両方）
interface Transform<in out T> {
  transform(value: T): T;
}

// 共変の具体例
class DogProducer implements Producer<Dog> {
  produce(): Dog {
    return { name: "ポチ", breed: "柴犬" };
  }
}

// Dog は Animal のサブタイプなので、Producer<Dog> は Producer<Animal> のサブタイプ
const animalProducer: Producer<Animal> = new DogProducer(); // OK（共変）

// 反変の具体例
class AnimalConsumer implements Consumer<Animal> {
  consume(value: Animal): void {
    console.log(`Consuming animal: ${value.name}`);
  }
}

// Animal は Dog のスーパータイプなので、Consumer<Animal> は Consumer<Dog> のサブタイプ
const dogConsumer: Consumer<Dog> = new AnimalConsumer(); // OK（反変）
```

### 3.3 Java での変性（PECS原則）

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

// PECS の実践例: Collections.copy
// src は Producer（読み取り） → extends
// dest は Consumer（書き込み） → super
public static <T> void copy(List<? super T> dest, List<? extends T> src) {
    for (int i = 0; i < src.size(); i++) {
        dest.set(i, src.get(i));
    }
}

// 使用例
List<Integer> ints = Arrays.asList(1, 2, 3);
List<Number> nums = new ArrayList<>(Arrays.asList(0.0, 0.0, 0.0));
Collections.copy(nums, ints); // Integer extends Number
```

```java
// Java: ワイルドカードの使い分け詳細
public class WildcardExamples {

    // extends: 読み取り専用（共変）
    // 「少なくとも T 型のものが出てくる」
    public static double average(List<? extends Number> list) {
        double sum = 0;
        for (Number n : list) {
            sum += n.doubleValue();
        }
        return sum / list.size();
    }

    // super: 書き込み用（反変）
    // 「少なくとも T 型のものを入れられる」
    public static void fillWithInts(List<? super Integer> list, int count) {
        for (int i = 0; i < count; i++) {
            list.add(i);
        }
    }

    // 非境界ワイルドカード: 型を気にしない操作
    public static int size(List<?> list) {
        return list.size();
    }

    // 複合的な例: 変換メソッド
    public static <T> void transform(
        List<? extends T> src,      // Producer: T のサブタイプを読む
        List<? super T> dest,       // Consumer: T のスーパータイプに書く
        Function<? super T, ? extends T> mapper  // 関数も PECS
    ) {
        for (T item : src) {
            dest.add(mapper.apply(item));
        }
    }
}
```

### 3.4 Kotlin での変性（宣言サイト変性）

```kotlin
// Kotlin: 宣言サイト変性（declaration-site variance）
// out = 共変（Java の ? extends に相当）
interface Source<out T> {
    fun nextT(): T  // T は出力位置のみ
    // fun consume(t: T)  // コンパイルエラー: T を入力位置で使えない
}

// in = 反変（Java の ? super に相当）
interface Comparable<in T> {
    fun compareTo(other: T): Int  // T は入力位置のみ
    // fun produce(): T  // コンパイルエラー: T を出力位置で使えない
}

// 使用サイト変性（use-site variance）も可能
fun copy(from: Array<out Any>, to: Array<Any>) {
    for (i in from.indices) {
        to[i] = from[i]
    }
}

// Kotlin の具体例: 共変リスト
// List<out E> は共変（読み取り専用）
// MutableList<E> は不変（読み書き両方）
fun printAnimals(animals: List<Animal>) {  // List<out Animal>
    animals.forEach { println(it.name) }
}

val dogs: List<Dog> = listOf(Dog("ポチ"))
printAnimals(dogs)  // OK: List<Dog> は List<Animal> のサブタイプ
```

### 3.5 C# での変性

```csharp
// C#: インターフェースの変性宣言
// out = 共変
public interface IReadOnlyList<out T>
{
    T this[int index] { get; }
    int Count { get; }
}

// in = 反変
public interface IComparer<in T>
{
    int Compare(T x, T y);
}

// .NET 標準ライブラリの例
// IEnumerable<out T>  → 共変
// IComparable<in T>   → 反変
// IList<T>            → 不変

// Func と Action の変性
// Func<in T, out TResult>  → T は反変、TResult は共変
// Action<in T>             → T は反変

// 実践例
IEnumerable<Dog> dogs = new List<Dog>();
IEnumerable<Animal> animals = dogs;  // OK: 共変

IComparer<Animal> animalComparer = new AnimalComparer();
IComparer<Dog> dogComparer = animalComparer;  // OK: 反変
```

### 3.6 変性の安全性と注意点

```typescript
// 変性の安全性に関する注意点

// ❌ 配列の共変性は危険な場合がある（Java の配列）
// Java:
// String[] strings = new String[3];
// Object[] objects = strings;  // コンパイルOK（配列は共変）
// objects[0] = 42;             // 実行時エラー! ArrayStoreException

// TypeScript での安全でない例
class Animal { name = ""; }
class Dog extends Animal { breed = ""; }
class Cat extends Animal { indoor = true; }

// TypeScript は構造的型付けなので、意図しない共変が起きうる
function addCat(animals: Animal[]): void {
  animals.push(new Cat());  // Dog[] に Cat が入る可能性
}

const dogs: Dog[] = [new Dog()];
addCat(dogs);  // TypeScript はこれを許す（構造的型付けのため）
// dogs[1] は実際には Cat だが、Dog[] と信じている

// ✅ readonly にすることで安全に
function safePrint(animals: readonly Animal[]): void {
  // animals.push(new Cat());  // エラー: readonly なので push できない
  animals.forEach(a => console.log(a.name));
}
```

---

## 4. 型消去 vs 単相化

### 4.1 型消去（Type Erasure）

```
型消去（Type Erasure）: Java, TypeScript
  → コンパイル後にジェネリクスの型情報が消える
  → List<String> と List<Integer> は実行時に同じ List
  → 利点: バイナリサイズが小さい、後方互換性
  → 欠点: 実行時に型情報にアクセスできない
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

// 型消去の回避策: Class<T> を渡す
public class TypeSafeContainer<T> {
    private final Class<T> type;
    private final List<T> items = new ArrayList<>();

    public TypeSafeContainer(Class<T> type) {
        this.type = type;
    }

    public void add(Object obj) {
        if (type.isInstance(obj)) {
            items.add(type.cast(obj));
        } else {
            throw new ClassCastException(
                "Expected " + type.getName() + " but got " + obj.getClass().getName()
            );
        }
    }

    public T[] toArray() {
        @SuppressWarnings("unchecked")
        T[] array = (T[]) java.lang.reflect.Array.newInstance(type, items.size());
        return items.toArray(array);
    }

    // 実行時型チェックが可能
    public boolean isTypeOf(Object obj) {
        return type.isInstance(obj);
    }
}

// 使用例
TypeSafeContainer<String> container = new TypeSafeContainer<>(String.class);
container.add("hello");
// container.add(42);  // ClassCastException
```

```java
// Java: 型消去のより詳細な制限と回避策

// 制限1: ジェネリック型の instanceof チェック不可
public <T> boolean isStringList(List<T> list) {
    // ❌ コンパイルエラー
    // return list instanceof List<String>;

    // ✅ ワイルドカードなら OK
    return list instanceof List<?>;
}

// 制限2: ジェネリック配列の生成不可
public <T> T[] createArray(int size) {
    // ❌ コンパイルエラー
    // return new T[size];

    // ✅ 回避策: Object 配列をキャスト（安全ではない）
    @SuppressWarnings("unchecked")
    T[] array = (T[]) new Object[size];
    return array;
}

// 制限3: ジェネリック型の静的フィールド
public class GenericSingleton<T> {
    // ❌ 型消去後、全ての型で同じ静的フィールドになる
    // private static T instance;  // コンパイルエラー

    // ✅ 回避策: Map で管理
    private static final Map<Class<?>, Object> instances = new HashMap<>();

    @SuppressWarnings("unchecked")
    public static <T> T getInstance(Class<T> type) {
        return (T) instances.get(type);
    }
}

// 制限4: オーバーロードの衝突
public class OverloadProblem {
    // ❌ 型消去後、両方とも process(List) になる
    // public void process(List<String> strings) {}
    // public void process(List<Integer> ints) {}

    // ✅ 回避策: メソッド名を変える
    public void processStrings(List<String> strings) {}
    public void processInts(List<Integer> ints) {}
}
```

### 4.2 単相化（Monomorphization）

```
単相化（Monomorphization）: Rust, C++
  → 使用される型ごとに専用のコードを生成
  → Vec<i32> と Vec<String> は別々のコードに
  → 利点: ゼロコスト抽象化、インライン化可能
  → 欠点: バイナリサイズが大きくなる
```

```rust
// Rust: 単相化の仕組み
// このジェネリック関数は...
fn max_of<T: PartialOrd>(a: T, b: T) -> T {
    if a >= b { a } else { b }
}

// 以下のように使うと...
let int_max = max_of(10i32, 20i32);
let float_max = max_of(3.14f64, 2.72f64);
let str_max = max_of("hello", "world");

// コンパイラが以下のような専用関数を生成する:
// fn max_of_i32(a: i32, b: i32) -> i32 { ... }
// fn max_of_f64(a: f64, b: f64) -> f64 { ... }
// fn max_of_str(a: &str, b: &str) -> &str { ... }

// ゼロコスト抽象化の証明
// ジェネリック版とハンドコード版は全く同じ機械語になる
fn generic_sum<I: Iterator<Item = i32>>(iter: I) -> i32 {
    let mut total = 0;
    for item in iter {
        total += item;
    }
    total
}

fn manual_sum(slice: &[i32]) -> i32 {
    let mut total = 0;
    for i in 0..slice.len() {
        total += slice[i];
    }
    total
}

// 両者は同一の最適化された機械語に
```

```rust
// Rust: 動的ディスパッチとの比較
// 静的ディスパッチ（単相化）
fn print_all_static<T: std::fmt::Display>(items: &[T]) {
    for item in items {
        println!("{}", item);
    }
}
// → 型ごとにコードが複製される
// → インライン化可能
// → バイナリサイズ増大

// 動的ディスパッチ（トレイトオブジェクト）
fn print_all_dynamic(items: &[&dyn std::fmt::Display]) {
    for item in items {
        println!("{}", item);
    }
}
// → コードは1つだけ
// → vtable 経由の間接呼び出し
// → バイナリサイズ小さい

// 使い分けの指針
// - パフォーマンス重視 → 静的（単相化）
// - バイナリサイズ重視 → 動的（トレイトオブジェクト）
// - 異なる型の混在コレクション → 動的のみ可能
```

```cpp
// C++: テンプレートの単相化
template<typename T>
class Vector {
    T* data;
    size_t size_;
    size_t capacity_;

public:
    Vector() : data(nullptr), size_(0), capacity_(0) {}

    void push_back(const T& value) {
        if (size_ == capacity_) {
            size_t new_cap = capacity_ == 0 ? 1 : capacity_ * 2;
            T* new_data = new T[new_cap];
            for (size_t i = 0; i < size_; i++) {
                new_data[i] = std::move(data[i]);
            }
            delete[] data;
            data = new_data;
            capacity_ = new_cap;
        }
        data[size_++] = value;
    }

    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }
    size_t size() const { return size_; }

    ~Vector() { delete[] data; }
};

// 使用するとコンパイラが Vector<int>, Vector<string>, Vector<double> を
// それぞれ独立した型として生成する
Vector<int> ints;
Vector<std::string> strings;
Vector<double> doubles;
```

### 4.3 C# の具体化ジェネリクス（Reified Generics）

```csharp
// C#: 実行時に型情報が保持される（Reified Generics）
// Java の型消去とは異なり、実行時にも型情報にアクセス可能

public class TypeAwareContainer<T>
{
    private readonly List<T> items = new();

    public void Add(T item) => items.Add(item);

    // 実行時の型情報を活用
    public Type GetContainedType() => typeof(T);

    public bool IsContaining<U>() => typeof(T) == typeof(U);

    // ジェネリック制約でインスタンス生成
    public static T CreateDefault() where T : new()
    {
        return new T();  // Java では不可能
    }
}

// 使用例
var container = new TypeAwareContainer<string>();
Console.WriteLine(container.GetContainedType());  // System.String
Console.WriteLine(container.IsContaining<string>());  // True
Console.WriteLine(container.IsContaining<int>());     // False

// 値型の最適化
// C# では List<int> は実際に int を直接格納（ボクシングなし）
// Java では List<Integer> はラッパー型を使う必要がある
```

---

## 5. 高度なジェネリクスパターン

### 5.1 再帰的ジェネリクス（F-bounded Polymorphism）

```typescript
// TypeScript: 再帰的型境界
// 自身の型を返すメソッドを持つ基底クラス
abstract class Builder<T extends Builder<T>> {
  protected data: Record<string, unknown> = {};

  abstract getThis(): T;

  set(key: string, value: unknown): T {
    this.data[key] = value;
    return this.getThis();
  }
}

class UserBuilder extends Builder<UserBuilder> {
  getThis(): UserBuilder { return this; }

  setName(name: string): UserBuilder {
    return this.set("name", name);
  }

  setAge(age: number): UserBuilder {
    return this.set("age", age);
  }

  build(): User {
    return this.data as unknown as User;
  }
}

// メソッドチェーンで正しい型が返る
const user = new UserBuilder()
  .setName("太郎")   // UserBuilder が返る（Builder<UserBuilder> ではなく）
  .setAge(30)         // UserBuilder が返る
  .build();
```

```java
// Java: 再帰的型境界（Curiously Recurring Template Pattern に似た手法）
// Comparable の定義
public interface Comparable<T> {
    int compareTo(T o);
}

// 自身の型で比較可能
public class Money implements Comparable<Money> {
    private final BigDecimal amount;
    private final Currency currency;

    @Override
    public int compareTo(Money other) {
        if (!this.currency.equals(other.currency)) {
            throw new IllegalArgumentException("Different currencies");
        }
        return this.amount.compareTo(other.amount);
    }
}

// Enum の定義（Java の Enum は再帰的型境界を使っている）
// public abstract class Enum<E extends Enum<E>> implements Comparable<E>
// これにより各列挙型は自身の型でのみ比較可能
```

### 5.2 型レベルプログラミング

```typescript
// TypeScript: 型レベルプログラミング

// タプル型の操作
type Head<T extends any[]> = T extends [infer H, ...any[]] ? H : never;
type Tail<T extends any[]> = T extends [any, ...infer R] ? R : [];
type Last<T extends any[]> = T extends [...any[], infer L] ? L : never;
type Length<T extends any[]> = T['length'];

type H = Head<[1, 2, 3]>;     // 1
type T2 = Tail<[1, 2, 3]>;    // [2, 3]
type L = Last<[1, 2, 3]>;     // 3
type Len = Length<[1, 2, 3]>;  // 3

// 型レベルでの文字列操作
type Split<S extends string, D extends string> =
  S extends `${infer Head}${D}${infer Tail}`
    ? [Head, ...Split<Tail, D>]
    : [S];

type Parts = Split<"a.b.c", ".">;  // ["a", "b", "c"]

// 型安全な深いプロパティアクセス
type DeepGet<T, Path extends string> =
  Path extends `${infer Key}.${infer Rest}`
    ? Key extends keyof T
      ? DeepGet<T[Key], Rest>
      : never
    : Path extends keyof T
      ? T[Path]
      : never;

interface AppConfig {
  database: {
    host: string;
    port: number;
    credentials: {
      username: string;
      password: string;
    };
  };
  server: {
    port: number;
  };
}

type DBHost = DeepGet<AppConfig, "database.host">;           // string
type DBPort = DeepGet<AppConfig, "database.port">;           // number
type DBUser = DeepGet<AppConfig, "database.credentials.username">; // string

// 型安全な get 関数
function deepGet<T, P extends string>(obj: T, path: P): DeepGet<T, P> {
  const keys = path.split('.');
  let current: any = obj;
  for (const key of keys) {
    current = current[key];
  }
  return current;
}

const config: AppConfig = {
  database: { host: "localhost", port: 5432, credentials: { username: "admin", password: "secret" } },
  server: { port: 3000 },
};

const host = deepGet(config, "database.host");  // string 型
const port = deepGet(config, "server.port");    // number 型
```

### 5.3 ジェネリクスと依存性注入（DI）

```typescript
// TypeScript: ジェネリクスを活用した型安全なDIコンテナ

// サービスキーの型定義
interface ServiceMap {
  logger: Logger;
  database: Database;
  userRepository: UserRepository;
  orderRepository: OrderRepository;
  emailService: EmailService;
}

class TypedContainer {
  private services = new Map<string, unknown>();
  private factories = new Map<string, () => unknown>();

  // 登録: 型安全
  register<K extends keyof ServiceMap>(
    key: K,
    factory: () => ServiceMap[K]
  ): void {
    this.factories.set(key as string, factory);
  }

  // 解決: 型安全
  resolve<K extends keyof ServiceMap>(key: K): ServiceMap[K] {
    if (this.services.has(key as string)) {
      return this.services.get(key as string) as ServiceMap[K];
    }

    const factory = this.factories.get(key as string);
    if (!factory) {
      throw new Error(`Service not registered: ${String(key)}`);
    }

    const instance = factory() as ServiceMap[K];
    this.services.set(key as string, instance);
    return instance;
  }
}

// 使用例
const container = new TypedContainer();

container.register("logger", () => new ConsoleLogger());
container.register("database", () => new PostgresDatabase("localhost:5432"));
container.register("userRepository", () =>
  new UserRepository(container.resolve("database"))
);

const logger = container.resolve("logger");     // Logger 型
const db = container.resolve("database");       // Database 型
const repo = container.resolve("userRepository"); // UserRepository 型
// container.resolve("unknown");                 // コンパイルエラー
```

### 5.4 ジェネリクスとモナドパターン

```typescript
// TypeScript: モナド的なパターンをジェネリクスで実装

// Functor（map を持つ）
interface Functor<T> {
  map<U>(fn: (value: T) => U): Functor<U>;
}

// Monad（flatMap/bind を持つ）
interface Monad<T> extends Functor<T> {
  flatMap<U>(fn: (value: T) => Monad<U>): Monad<U>;
}

// Maybe モナド（null安全な計算チェーン）
class Maybe<T> implements Monad<T> {
  private constructor(private readonly value: T | null) {}

  static of<T>(value: T | null | undefined): Maybe<T> {
    return new Maybe(value ?? null);
  }

  static just<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  isNothing(): boolean {
    return this.value === null;
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

  filter(predicate: (value: T) => boolean): Maybe<T> {
    if (this.value === null) return this;
    return predicate(this.value) ? this : Maybe.nothing();
  }
}

// 使用例: null安全な深いプロパティアクセス
interface Company {
  ceo?: {
    name: string;
    address?: {
      city: string;
      zip?: string;
    };
  };
}

function getCeoCity(company: Company): string {
  return Maybe.of(company.ceo)
    .flatMap(ceo => Maybe.of(ceo.address))
    .map(address => address.city)
    .getOrElse("Unknown");
}

// 使用例: バリデーションチェーン
function validateAndProcess(input: string): Maybe<number> {
  return Maybe.of(input)
    .filter(s => s.length > 0)
    .map(s => parseInt(s, 10))
    .filter(n => !isNaN(n))
    .filter(n => n >= 0 && n <= 100)
    .map(n => n * 2);
}
```

---

## 6. 実践パターン

### 6.1 Result型

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

  mapError<F>(fn: (error: E) => F): Result<T, F> {
    if (!this.isOk) return Result.err(fn(this.error!));
    return Result.ok(this.value!);
  }

  flatMap<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
    if (this.isOk) return fn(this.value!);
    return Result.err(this.error!);
  }

  unwrapOr(defaultValue: T): T {
    return this.isOk ? this.value! : defaultValue;
  }

  match<U>(handlers: { ok: (value: T) => U; err: (error: E) => U }): U {
    return this.isOk ? handlers.ok(this.value!) : handlers.err(this.error!);
  }

  // 複数の Result を合成
  static all<T, E>(results: Result<T, E>[]): Result<T[], E> {
    const values: T[] = [];
    for (const result of results) {
      if (!result.isOk) return Result.err(result.error!);
      values.push(result.value!);
    }
    return Result.ok(values);
  }
}

// 使用例
function parseNumber(s: string): Result<number, string> {
  const n = Number(s);
  if (isNaN(n)) return Result.err(`"${s}" is not a number`);
  return Result.ok(n);
}

function divide(a: number, b: number): Result<number, string> {
  if (b === 0) return Result.err("Division by zero");
  return Result.ok(a / b);
}

// チェーンで合成
const result = parseNumber("42")
  .flatMap(n => divide(n, 7))
  .map(n => n.toFixed(2))
  .match({
    ok: value => `Result: ${value}`,
    err: error => `Error: ${error}`,
  });
// "Result: 6.00"
```

### 6.2 型安全なイベントシステム

```typescript
// ジェネリクスを活用した型安全なイベントバス
interface EventDefinitions {
  "user:created": { userId: string; email: string; createdAt: Date };
  "user:deleted": { userId: string; reason: string };
  "order:placed": { orderId: string; userId: string; total: number };
  "order:cancelled": { orderId: string; reason: string };
  "payment:completed": { paymentId: string; orderId: string; amount: number };
}

class TypedEventBus<Events extends Record<string, any>> {
  private handlers = new Map<keyof Events, Set<Function>>();
  private middlewares: Array<(event: string, data: any) => any> = [];

  on<K extends keyof Events>(
    event: K,
    handler: (data: Events[K]) => void | Promise<void>
  ): () => void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, new Set());
    }
    this.handlers.get(event)!.add(handler);
    return () => this.handlers.get(event)?.delete(handler);
  }

  once<K extends keyof Events>(
    event: K,
    handler: (data: Events[K]) => void | Promise<void>
  ): () => void {
    const unsubscribe = this.on(event, (data) => {
      unsubscribe();
      handler(data);
    });
    return unsubscribe;
  }

  async emit<K extends keyof Events>(event: K, data: Events[K]): Promise<void> {
    // ミドルウェアを適用
    let processedData = data;
    for (const middleware of this.middlewares) {
      processedData = middleware(event as string, processedData) ?? processedData;
    }

    const handlers = this.handlers.get(event);
    if (!handlers) return;

    const promises = [...handlers].map(handler => handler(processedData));
    await Promise.all(promises);
  }

  // 型安全な待機
  waitFor<K extends keyof Events>(
    event: K,
    timeout?: number
  ): Promise<Events[K]> {
    return new Promise((resolve, reject) => {
      const timer = timeout
        ? setTimeout(() => reject(new Error(`Timeout waiting for ${String(event)}`)), timeout)
        : undefined;

      this.once(event, (data) => {
        if (timer) clearTimeout(timer);
        resolve(data);
      });
    });
  }
}

// 使用例
const bus = new TypedEventBus<EventDefinitions>();

bus.on("user:created", (data) => {
  console.log(`New user: ${data.email}`);  // 型安全: email は string
});

bus.on("order:placed", (data) => {
  console.log(`Order ${data.orderId}: ¥${data.total}`);  // total は number
});

await bus.emit("user:created", {
  userId: "u-1",
  email: "tanaka@example.com",
  createdAt: new Date(),
});
```

### 6.3 型安全なビルダーパターン（Phantom Types）

```typescript
// Phantom Types を使った型安全なビルダー
interface BuilderState {
  hasName: boolean;
  hasEmail: boolean;
  hasAge: boolean;
}

class UserBuilder<State extends BuilderState = {
  hasName: false;
  hasEmail: false;
  hasAge: false;
}> {
  private data: Partial<User> = {};

  private constructor(data: Partial<User>) {
    this.data = data;
  }

  static create(): UserBuilder<{ hasName: false; hasEmail: false; hasAge: false }> {
    return new UserBuilder({});
  }

  setName(name: string): UserBuilder<State & { hasName: true }> {
    return new UserBuilder({ ...this.data, name }) as any;
  }

  setEmail(email: string): UserBuilder<State & { hasEmail: true }> {
    return new UserBuilder({ ...this.data, email }) as any;
  }

  setAge(age: number): UserBuilder<State & { hasAge: true }> {
    return new UserBuilder({ ...this.data, age }) as any;
  }

  // build は全てのフィールドが設定された場合のみ呼べる
  build(
    this: UserBuilder<{ hasName: true; hasEmail: true; hasAge: true }>
  ): User {
    return this.data as User;
  }
}

// 使用例
const user = UserBuilder.create()
  .setName("太郎")
  .setEmail("taro@example.com")
  .setAge(30)
  .build();  // OK: 全フィールド設定済み

// const incomplete = UserBuilder.create()
//   .setName("太郎")
//   .build();  // コンパイルエラー: email と age が未設定
```

### 6.4 ジェネリクスとリポジトリパターン

```typescript
// 型安全なジェネリックリポジトリ
interface Entity {
  id: string;
  createdAt: Date;
  updatedAt: Date;
}

interface QueryOptions<T> {
  where?: Partial<T>;
  orderBy?: { field: keyof T; direction: "asc" | "desc" };
  limit?: number;
  offset?: number;
}

interface Repository<T extends Entity> {
  findById(id: string): Promise<T | null>;
  findAll(options?: QueryOptions<T>): Promise<T[]>;
  create(data: Omit<T, "id" | "createdAt" | "updatedAt">): Promise<T>;
  update(id: string, data: Partial<Omit<T, "id" | "createdAt" | "updatedAt">>): Promise<T>;
  delete(id: string): Promise<void>;
  count(where?: Partial<T>): Promise<number>;
}

// 汎用的な実装
class InMemoryRepository<T extends Entity> implements Repository<T> {
  private items = new Map<string, T>();
  private idCounter = 0;

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  async findAll(options?: QueryOptions<T>): Promise<T[]> {
    let results = [...this.items.values()];

    // フィルタリング
    if (options?.where) {
      results = results.filter(item =>
        Object.entries(options.where!).every(
          ([key, value]) => (item as any)[key] === value
        )
      );
    }

    // ソート
    if (options?.orderBy) {
      const { field, direction } = options.orderBy;
      results.sort((a, b) => {
        const aVal = a[field];
        const bVal = b[field];
        const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return direction === "asc" ? cmp : -cmp;
      });
    }

    // ページネーション
    if (options?.offset) results = results.slice(options.offset);
    if (options?.limit) results = results.slice(0, options.limit);

    return results;
  }

  async create(data: Omit<T, "id" | "createdAt" | "updatedAt">): Promise<T> {
    const now = new Date();
    const entity = {
      ...data,
      id: String(++this.idCounter),
      createdAt: now,
      updatedAt: now,
    } as T;
    this.items.set(entity.id, entity);
    return entity;
  }

  async update(
    id: string,
    data: Partial<Omit<T, "id" | "createdAt" | "updatedAt">>
  ): Promise<T> {
    const existing = this.items.get(id);
    if (!existing) throw new Error(`Entity not found: ${id}`);
    const updated = { ...existing, ...data, updatedAt: new Date() };
    this.items.set(id, updated);
    return updated;
  }

  async delete(id: string): Promise<void> {
    if (!this.items.has(id)) throw new Error(`Entity not found: ${id}`);
    this.items.delete(id);
  }

  async count(where?: Partial<T>): Promise<number> {
    if (!where) return this.items.size;
    return (await this.findAll({ where })).length;
  }
}

// 具体的なエンティティ
interface UserEntity extends Entity {
  name: string;
  email: string;
  role: "admin" | "user";
}

interface OrderEntity extends Entity {
  userId: string;
  total: number;
  status: "pending" | "confirmed" | "shipped" | "delivered";
}

// 型安全なリポジトリのインスタンス
const userRepo = new InMemoryRepository<UserEntity>();
const orderRepo = new InMemoryRepository<OrderEntity>();

// 完全に型安全
const users = await userRepo.findAll({
  where: { role: "admin" },
  orderBy: { field: "createdAt", direction: "desc" },
  limit: 10,
});

const orders = await orderRepo.findAll({
  where: { status: "pending" },
  // where: { unknownField: "value" },  // コンパイルエラー
});
```

---

## 7. ジェネリクスのベストプラクティス

### 7.1 命名規約

```
一般的な型パラメータ名:
  T       → Type（一般的な型）
  E       → Element（コレクションの要素）
  K       → Key（マップのキー）
  V       → Value（マップの値）
  R       → Return（戻り値の型）
  S, U    → 2番目、3番目の型パラメータ
  N       → Number
  P       → Parameter / Props

良い命名:
  Repository<Entity>
  Converter<From, To>
  Handler<Request, Response>
  Mapper<Input, Output>
  Validator<T>

避けるべき命名:
  Repository<A>（意味不明）
  Handler<X, Y, Z>（何がなんだかわからない）
```

### 7.2 ジェネリクスの使いすぎに注意

```typescript
// ❌ 過剰なジェネリクス
function add<T extends number>(a: T, b: T): T {
  return (a + b) as T;  // キャスト必要 = 意味がない
}

// ✅ ジェネリクスが不要な場合
function add(a: number, b: number): number {
  return a + b;
}

// ❌ 1回しか使わない型パラメータ
function logValue<T>(value: T): void {
  console.log(value);
}

// ✅ unknown で十分
function logValue(value: unknown): void {
  console.log(value);
}

// ✅ ジェネリクスが必要な場合: 入力と出力の型関係を表現
function identity<T>(value: T): T {
  return value;  // 入力と同じ型が返ることを保証
}

// ✅ ジェネリクスが必要な場合: 複数の引数間の型関係を表現
function merge<T, U>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 };
}
```

### 7.3 制約は最小限に

```typescript
// ❌ 不必要に厳しい制約
function getName<T extends { name: string; age: number; email: string }>(obj: T): string {
  return obj.name;  // age と email は使っていない
}

// ✅ 必要最小限の制約
function getName<T extends { name: string }>(obj: T): string {
  return obj.name;
}

// ❌ 具象クラスで制約
function process<T extends UserService>(service: T): void {
  service.getUsers();
}

// ✅ インターフェースで制約
interface HasGetUsers {
  getUsers(): User[];
}
function process<T extends HasGetUsers>(service: T): void {
  service.getUsers();
}
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
| 具体化 | C#。実行時にも型情報保持 |
| 条件付き型 | TypeScript の型レベルプログラミング |
| F-bounded | 再帰的型境界（自身の型を返す） |
| Phantom Types | 型パラメータで状態を表現 |

### 言語ごとのジェネリクス比較

| 特徴 | Java | TypeScript | Rust | C# | C++ | Python |
|------|------|-----------|------|-----|-----|--------|
| 実装方式 | 型消去 | 型消去 | 単相化 | 具体化 | 単相化 | 型ヒント |
| 実行時型情報 | なし | なし | なし | あり | なし | あり（動的） |
| 値型サポート | なし | N/A | あり | あり | あり | N/A |
| 変性宣言 | 使用サイト | 宣言サイト | なし | 宣言サイト | なし | なし |
| 条件付き型 | なし | あり | なし | なし | あり(C++20) | なし |
| 高カインド型 | なし | 制限的 | なし | なし | あり | なし |
| デフォルト型 | なし | あり | あり | あり | あり | あり |

---

## 次に読むべきガイド
→ [[../04-practical-patterns/00-creational-patterns.md]] — 生成パターン

---

## 参考文献
1. Wadler, P. "Theorems for Free!" ICFP, 1989.
2. Bloch, J. "Effective Java." Item 31: Use bounded wildcards. 2018.
3. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
4. Odersky, M. & Zenger, M. "Scalable Component Abstractions." OOPSLA, 2005.
5. Rust Reference. "Generics." https://doc.rust-lang.org/reference/items/generics.html
6. TypeScript Handbook. "Generics." https://www.typescriptlang.org/docs/handbook/2/generics.html
7. Microsoft Docs. "Generics in C#." https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/generics
