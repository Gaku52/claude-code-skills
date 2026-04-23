# Generics in OOP

> Generics are a mechanism for "parameterizing types." They are an essential technique for writing generic code while maintaining type safety. Understand deep concepts such as covariance, contravariance, and type erasure.

## What You Will Learn in This Chapter

- [ ] Understand the fundamentals of generics and their implementations across languages
- [ ] Grasp the differences between covariance, contravariance, and invariance
- [ ] Learn the trade-offs between type erasure and monomorphization
- [ ] Know advanced generics patterns (higher-kinded types, type-level programming)
- [ ] Be able to make design decisions that appropriately leverage generics in practice


## Prerequisites

Your understanding will be deeper if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of [Mixins and Multiple Inheritance](./02-mixins-and-multiple-inheritance.md)

---

## 1. Fundamentals of Generics

### 1.1 Why Generics Are Necessary

```
Without generics:
  -> Generalize with Object type -> casting required -> risk of runtime errors

With generics:
  -> Generalize with type parameters -> no casting -> compile-time type checking
```

The problem generics solve is "balancing type safety and reusability." Before generics existed, you had to use Object types (Java) or void* pointers (C) to write generic code. This carried the risk of causing cast errors at runtime and was a hotbed for bugs.

```typescript
// TypeScript: basics of generics
// Bad: using any (no type safety)
function firstAny(arr: any[]): any {
  return arr[0];
}
const val = firstAny([1, 2, 3]); // val: any (type information is lost)

// Good: using generics
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}
const num = first([1, 2, 3]);      // num: number
const str = first(["a", "b"]);     // str: string

// Generic class
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
// numStack.push("hello"); // compile error!
```

### 1.2 Generics Syntax in Various Languages

```java
// Java: basic generics syntax
// Generic method
public <T> T firstElement(List<T> list) {
    if (list.isEmpty()) return null;
    return list.get(0);
}

// Generic class
public class Pair<A, B> {
    private final A first;
    private final B second;

    public Pair(A first, B second) {
        this.first = first;
        this.second = second;
    }

    public A getFirst() { return first; }
    public B getSecond() { return second; }

    // Define additional type parameters within a generic method
    public <C> Triple<A, B, C> withThird(C third) {
        return new Triple<>(first, second, third);
    }
}

// Generic interface
public interface Repository<T, ID> {
    T findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
}

// Concrete class fixes the types
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
// C#: basic generics syntax
// Generic class
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

    // Generic method
    public Result<U> Map<U>(Func<T, U> mapper)
    {
        if (!IsSuccess) return Result<U>.Failure(Error);
        return Result<U>.Success(mapper(Value));
    }
}

// Usage example
Result<int> parsed = Result<int>.Success(42);
Result<string> formatted = parsed.Map(n => $"Value: {n}");
```

```rust
// Rust: basic generics syntax
// Generic struct
struct Pair<T, U> {
    first: T,
    second: U,
}

impl<T, U> Pair<T, U> {
    fn new(first: T, second: U) -> Self {
        Pair { first, second }
    }

    // Method with different type parameters
    fn zip_with<V, W>(self, other: Pair<V, W>) -> Pair<Pair<T, V>, Pair<U, W>> {
        Pair {
            first: Pair::new(self.first, other.first),
            second: Pair::new(self.second, other.second),
        }
    }
}

// Generic enum (Option and Result are in the standard library)
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
# Python: generics via the typing module (Python 3.12+)
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')
U = TypeVar('U')

# Generic class
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

# New syntax in Python 3.12+
class Pair[T, U]:
    def __init__(self, first: T, second: U) -> None:
        self.first = first
        self.second = second

    def map_firstV -> 'Pair[V, U]':
        return Pair(f(self.first), self.second)

# Static checking via type hints (mypy, etc.)
stack: Stack[int] = Stack()
stack.push(1)
stack.push(2)
# stack.push("hello")  # mypy error
```

### 1.3 Multiple Type Parameters

Generics with multiple type parameters are useful for expressing relationships between different types.

```typescript
// TypeScript: multiple type parameters
// Type-safe operations on a map (dictionary)
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

  // map method that transforms the value type
  mapValues<U>(fn: (value: V, key: K) => U): TypedMap<K, U> {
    const result = new TypedMap<K, U>();
    for (const [key, value] of this.data) {
      result.set(key, fn(value, key));
    }
    return result;
  }
}

// Either type: holds one of two types
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

// Usage example: validation
type ValidationError = { field: string; message: string };

function validateAge(age: number): Either<ValidationError, number> {
  if (age < 0 || age > 150) {
    return Either.left({ field: "age", message: "Age must be in the range 0-150" });
  }
  return Either.right(age);
}

function validateName(name: string): Either<ValidationError, string> {
  if (name.length === 0) {
    return Either.left({ field: "name", message: "Name is required" });
  }
  return Either.right(name);
}
```

---

## 2. Bounded Generics

### 2.1 Upper Bound

By placing constraints on type parameters, you gain access to the functionality of specific interfaces or classes.

```typescript
// TypeScript: applying constraints to type parameters
interface HasLength {
  length: number;
}

// T is constrained to types that have a length property
function longest<T extends HasLength>(a: T, b: T): T {
  return a.length >= b.length ? a : b;
}

longest("hello", "world!");     // OK: string has length
longest([1, 2], [1, 2, 3]);    // OK: Array has length
// longest(10, 20);             // error: number does not have length
```

```typescript
// TypeScript: type-safe access to specific object properties
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const person = { name: "Taro", age: 30, email: "taro@example.com" };
const name = getProperty(person, "name");   // string type
const age = getProperty(person, "age");     // number type
// getProperty(person, "address");           // error: "address" is not in keyof typeof person

// Combining multiple constraints
interface Serializable {
  serialize(): string;
}

interface Validatable {
  validate(): boolean;
}

// T is both Serializable and Validatable
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
// Rust: trait bounds
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in &list[1..] {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Multiple trait bounds
fn print_and_compare<T: std::fmt::Display + PartialOrd>(a: T, b: T) {
    if a > b {
        println!("{} is larger", a);
    } else {
        println!("{} is larger", b);
    }
}

// Improved readability with where clauses
fn complex_function<T, U>(t: T, u: U) -> String
where
    T: std::fmt::Display + Clone + Send + 'static,
    U: std::fmt::Debug + Into<String>,
{
    format!("t={}, u={:?}", t, u)
}

// Trait bounds with associated types
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

### 2.2 Lower Bound

In Java, you can specify a lower bound using the `super` keyword. This is mainly used in write operations to collections.

```java
// Java: upper bound and lower bound
// Upper bound: T is a subclass of Number
public <T extends Number> double sum(List<T> list) {
    return list.stream().mapToDouble(Number::doubleValue).sum();
}

// Lower bound: T is a superclass of Integer
public void addIntegers(List<? super Integer> list) {
    list.add(1);
    list.add(2);
}

// Applying upper bound: constraint via Comparable
public <T extends Comparable<T>> T max(T a, T b) {
    return a.compareTo(b) >= 0 ? a : b;
}

// Recursive type bounds
// T must be a type that can be compared with itself
public <T extends Comparable<T>> void sort(List<T> list) {
    Collections.sort(list);
}

// Multiple upper bounds
public <T extends Serializable & Comparable<T>> void saveOrdered(List<T> list) {
    Collections.sort(list);
    // serialize list...
}
```

### 2.3 Conditional Types

TypeScript has an advanced feature called conditional types, which enables conditional branching at the type level.

```typescript
// TypeScript: conditional types
// If T is an array, extract the element type; otherwise return it as-is
type Unwrap<T> = T extends Array<infer U> ? U : T;

type A = Unwrap<string[]>;   // string
type B = Unwrap<number>;     // number
type C = Unwrap<boolean[]>;  // boolean

// Unwrap nested Promises
type UnwrapPromise<T> = T extends Promise<infer U> ? UnwrapPromise<U> : T;

type D = UnwrapPromise<Promise<Promise<string>>>;  // string

// Extract the return type of a function
type ReturnOf<T> = T extends (...args: any[]) => infer R ? R : never;

type E = ReturnOf<() => number>;           // number
type F = ReturnOf<(x: string) => boolean>; // boolean

// Utility using conditional types
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
// Only "id" | "name"

// Combined with mapped types
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
// All nested properties become readonly
```

---

## 3. Covariance, Contravariance, and Invariance

### 3.1 Basic Concepts

```
Covariance: type parameters inherit in the same direction
  Dog extends Animal
  -> Can List<Dog> be used as List<Animal>?

Contravariance: type parameters inherit in the opposite direction
  -> Can Consumer<Animal> be used as Consumer<Dog>?

Invariance: neither is allowed
  -> List<Dog> is not List<Animal>

  Covariance: Producer<Dog> -> Producer<Animal>  (produces: output only)
  Contravariance: Consumer<Animal> -> Consumer<Dog>  (consumes: input only)
  Invariance: Mutable<Dog> != Mutable<Animal>   (both)
```

### 3.2 Variance in TypeScript

```typescript
// TypeScript: covariance example (arrays are covariant)
class Animal { name: string = ""; }
class Dog extends Animal { breed: string = ""; }

// Safe with covariance if read-only
function printNames(animals: readonly Animal[]): void {
  animals.forEach(a => console.log(a.name));
}

const dogs: Dog[] = [{ name: "Pochi", breed: "Shiba" }];
printNames(dogs); // OK: Dog[] can be used as readonly Animal[]
```

```typescript
// TypeScript 4.7+: explicit variance declarations with in/out keywords
// out = covariant (output position only)
interface Producer<out T> {
  produce(): T;
}

// in = contravariant (input position only)
interface Consumer<in T> {
  consume(value: T): void;
}

// in/out = invariant (both input and output)
interface Transform<in out T> {
  transform(value: T): T;
}

// Covariance concrete example
class DogProducer implements Producer<Dog> {
  produce(): Dog {
    return { name: "Pochi", breed: "Shiba" };
  }
}

// Since Dog is a subtype of Animal, Producer<Dog> is a subtype of Producer<Animal>
const animalProducer: Producer<Animal> = new DogProducer(); // OK (covariant)

// Contravariance concrete example
class AnimalConsumer implements Consumer<Animal> {
  consume(value: Animal): void {
    console.log(`Consuming animal: ${value.name}`);
  }
}

// Since Animal is a supertype of Dog, Consumer<Animal> is a subtype of Consumer<Dog>
const dogConsumer: Consumer<Dog> = new AnimalConsumer(); // OK (contravariant)
```

### 3.3 Variance in Java (PECS Principle)

```java
// Java: PECS (Producer Extends, Consumer Super)
// Producer: reads values out -> extends (covariant)
public double sumOfList(List<? extends Number> list) {
    double sum = 0;
    for (Number n : list) { // read only
        sum += n.doubleValue();
    }
    return sum;
}

// Consumer: puts values in -> super (contravariant)
public void addNumbers(List<? super Integer> list) {
    list.add(1);  // write only
    list.add(2);
}

// Practical example of PECS: Collections.copy
// src is a Producer (read) -> extends
// dest is a Consumer (write) -> super
public static <T> void copy(List<? super T> dest, List<? extends T> src) {
    for (int i = 0; i < src.size(); i++) {
        dest.set(i, src.get(i));
    }
}

// Usage example
List<Integer> ints = Arrays.asList(1, 2, 3);
List<Number> nums = new ArrayList<>(Arrays.asList(0.0, 0.0, 0.0));
Collections.copy(nums, ints); // Integer extends Number
```

```java
// Java: detailed use cases for wildcards
public class WildcardExamples {

    // extends: read-only (covariant)
    // "At least a T-typed value comes out"
    public static double average(List<? extends Number> list) {
        double sum = 0;
        for (Number n : list) {
            sum += n.doubleValue();
        }
        return sum / list.size();
    }

    // super: for writing (contravariant)
    // "At least a T-typed value can be put in"
    public static void fillWithInts(List<? super Integer> list, int count) {
        for (int i = 0; i < count; i++) {
            list.add(i);
        }
    }

    // Unbounded wildcard: operations indifferent to the type
    public static int size(List<?> list) {
        return list.size();
    }

    // Compound example: transformation method
    public static <T> void transform(
        List<? extends T> src,      // Producer: reads subtypes of T
        List<? super T> dest,       // Consumer: writes supertypes of T
        Function<? super T, ? extends T> mapper  // function is also PECS
    ) {
        for (T item : src) {
            dest.add(mapper.apply(item));
        }
    }
}
```

### 3.4 Variance in Kotlin (Declaration-Site Variance)

```kotlin
// Kotlin: declaration-site variance
// out = covariant (equivalent to Java's ? extends)
interface Source<out T> {
    fun nextT(): T  // T is in output position only
    // fun consume(t: T)  // compile error: T cannot be used in input position
}

// in = contravariant (equivalent to Java's ? super)
interface Comparable<in T> {
    fun compareTo(other: T): Int  // T is in input position only
    // fun produce(): T  // compile error: T cannot be used in output position
}

// Use-site variance is also possible
fun copy(from: Array<out Any>, to: Array<Any>) {
    for (i in from.indices) {
        to[i] = from[i]
    }
}

// Kotlin concrete example: covariant list
// List<out E> is covariant (read-only)
// MutableList<E> is invariant (both read and write)
fun printAnimals(animals: List<Animal>) {  // List<out Animal>
    animals.forEach { println(it.name) }
}

val dogs: List<Dog> = listOf(Dog("Pochi"))
printAnimals(dogs)  // OK: List<Dog> is a subtype of List<Animal>
```

### 3.5 Variance in C#

```csharp
// C#: variance declarations on interfaces
// out = covariant
public interface IReadOnlyList<out T>
{
    T this[int index] { get; }
    int Count { get; }
}

// in = contravariant
public interface IComparer<in T>
{
    int Compare(T x, T y);
}

// Examples from the .NET standard library
// IEnumerable<out T>  -> covariant
// IComparable<in T>   -> contravariant
// IList<T>            -> invariant

// Variance of Func and Action
// Func<in T, out TResult>  -> T is contravariant, TResult is covariant
// Action<in T>             -> T is contravariant

// Practical example
IEnumerable<Dog> dogs = new List<Dog>();
IEnumerable<Animal> animals = dogs;  // OK: covariant

IComparer<Animal> animalComparer = new AnimalComparer();
IComparer<Dog> dogComparer = animalComparer;  // OK: contravariant
```

### 3.6 Variance Safety and Considerations

```typescript
// Notes on variance safety

// Bad: array covariance can be dangerous (Java arrays)
// Java:
// String[] strings = new String[3];
// Object[] objects = strings;  // compiles OK (arrays are covariant)
// objects[0] = 42;             // runtime error! ArrayStoreException

// Unsafe example in TypeScript
class Animal { name = ""; }
class Dog extends Animal { breed = ""; }
class Cat extends Animal { indoor = true; }

// Since TypeScript uses structural typing, unintended covariance can occur
function addCat(animals: Animal[]): void {
  animals.push(new Cat());  // a Cat may end up inside a Dog[]
}

const dogs: Dog[] = [new Dog()];
addCat(dogs);  // TypeScript allows this (due to structural typing)
// dogs[1] is actually a Cat, but believed to be a Dog[]

// Good: make it safe with readonly
function safePrint(animals: readonly Animal[]): void {
  // animals.push(new Cat());  // error: cannot push because it's readonly
  animals.forEach(a => console.log(a.name));
}
```

---

## 4. Type Erasure vs Monomorphization

### 4.1 Type Erasure

```
Type Erasure: Java, TypeScript
  -> Generic type information is erased after compilation
  -> List<String> and List<Integer> are the same List at runtime
  -> Advantages: smaller binary size, backward compatibility
  -> Disadvantages: type information is not accessible at runtime
```

```java
// Java: limitations of type erasure
List<String> strings = new ArrayList<>();
List<Integer> ints = new ArrayList<>();

// No type information at runtime
System.out.println(strings.getClass() == ints.getClass()); // true!

// Operations made impossible by type erasure
// if (obj instanceof List<String>) {} // compile error
// T[] array = new T[10];             // compile error

// Workaround for type erasure: pass Class<T>
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

    // Runtime type check is possible
    public boolean isTypeOf(Object obj) {
        return type.isInstance(obj);
    }
}

// Usage example
TypeSafeContainer<String> container = new TypeSafeContainer<>(String.class);
container.add("hello");
// container.add(42);  // ClassCastException
```

```java
// Java: more detailed restrictions of type erasure and workarounds

// Restriction 1: instanceof checks on generic types are not allowed
public <T> boolean isStringList(List<T> list) {
    // Bad: compile error
    // return list instanceof List<String>;

    // Good: OK with a wildcard
    return list instanceof List<?>;
}

// Restriction 2: creation of generic arrays is not allowed
public <T> T[] createArray(int size) {
    // Bad: compile error
    // return new T[size];

    // Good: workaround: cast an Object array (not safe)
    @SuppressWarnings("unchecked")
    T[] array = (T[]) new Object[size];
    return array;
}

// Restriction 3: static fields of generic types
public class GenericSingleton<T> {
    // Bad: after type erasure, the same static field is shared across all types
    // private static T instance;  // compile error

    // Good: workaround: manage with a Map
    private static final Map<Class<?>, Object> instances = new HashMap<>();

    @SuppressWarnings("unchecked")
    public static <T> T getInstance(Class<T> type) {
        return (T) instances.get(type);
    }
}

// Restriction 4: overload collisions
public class OverloadProblem {
    // Bad: after type erasure, both become process(List)
    // public void process(List<String> strings) {}
    // public void process(List<Integer> ints) {}

    // Good: workaround: change method names
    public void processStrings(List<String> strings) {}
    public void processInts(List<Integer> ints) {}
}
```

### 4.2 Monomorphization

```
Monomorphization: Rust, C++
  -> Generate dedicated code for each type used
  -> Vec<i32> and Vec<String> become separate code
  -> Advantages: zero-cost abstractions, can be inlined
  -> Disadvantages: larger binary size
```

```rust
// Rust: how monomorphization works
// This generic function...
fn max_of<T: PartialOrd>(a: T, b: T) -> T {
    if a >= b { a } else { b }
}

// When used as follows...
let int_max = max_of(10i32, 20i32);
let float_max = max_of(3.14f64, 2.72f64);
let str_max = max_of("hello", "world");

// The compiler generates dedicated functions like the following:
// fn max_of_i32(a: i32, b: i32) -> i32 { ... }
// fn max_of_f64(a: f64, b: f64) -> f64 { ... }
// fn max_of_str(a: &str, b: &str) -> &str { ... }

// Proof of zero-cost abstraction
// The generic version and the hand-written version produce identical machine code
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

// Both produce identical optimized machine code
```

```rust
// Rust: comparison with dynamic dispatch
// Static dispatch (monomorphization)
fn print_all_static<T: std::fmt::Display>(items: &[T]) {
    for item in items {
        println!("{}", item);
    }
}
// -> Code is duplicated per type
// -> Can be inlined
// -> Binary size grows

// Dynamic dispatch (trait object)
fn print_all_dynamic(items: &[&dyn std::fmt::Display]) {
    for item in items {
        println!("{}", item);
    }
}
// -> Only one piece of code
// -> Indirect calls via vtable
// -> Smaller binary size

// Guidelines for choosing between them
// - Performance priority -> static (monomorphization)
// - Binary size priority -> dynamic (trait object)
// - Mixed-type collections -> only dynamic is possible
```

```cpp
// C++: template monomorphization
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

    T& operator { return data[index]; }
    const T& operator const { return data[index]; }
    size_t size() const { return size_; }

    ~Vector() { delete[] data; }
};

// When used, the compiler generates Vector<int>, Vector<string>, and Vector<double>
// as independent types respectively
Vector<int> ints;
Vector<std::string> strings;
Vector<double> doubles;
```

### 4.3 C# Reified Generics

```csharp
// C#: type information is retained at runtime (reified generics)
// Unlike Java's type erasure, type information can be accessed at runtime as well

public class TypeAwareContainer<T>
{
    private readonly List<T> items = new();

    public void Add(T item) => items.Add(item);

    // Leverage runtime type information
    public Type GetContainedType() => typeof(T);

    public bool IsContaining<U>() => typeof(T) == typeof(U);

    // Instance creation with generic constraints
    public static T CreateDefault() where T : new()
    {
        return new T();  // impossible in Java
    }
}

// Usage example
var container = new TypeAwareContainer<string>();
Console.WriteLine(container.GetContainedType());  // System.String
Console.WriteLine(container.IsContaining<string>());  // True
Console.WriteLine(container.IsContaining<int>());     // False

// Value type optimization
// In C#, List<int> actually stores int directly (no boxing)
// In Java, List<Integer> must use a wrapper type
```

---

## 5. Advanced Generics Patterns

### 5.1 Recursive Generics (F-bounded Polymorphism)

```typescript
// TypeScript: recursive type bounds
// Base class with a method that returns its own type
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

// The correct type is returned from method chaining
const user = new UserBuilder()
  .setName("Taro")    // returns UserBuilder (not Builder<UserBuilder>)
  .setAge(30)         // returns UserBuilder
  .build();
```

```java
// Java: recursive type bounds (technique similar to Curiously Recurring Template Pattern)
// Definition of Comparable
public interface Comparable<T> {
    int compareTo(T o);
}

// Comparable with its own type
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

// Enum definition (Java's Enum uses recursive type bounds)
// public abstract class Enum<E extends Enum<E>> implements Comparable<E>
// This allows each enum type to be compared only with its own type
```

### 5.2 Type-Level Programming

```typescript
// TypeScript: type-level programming

// Tuple type operations
type Head<T extends any[]> = T extends [infer H, ...any[]] ? H : never;
type Tail<T extends any[]> = T extends [any, ...infer R] ? R : [];
type Last<T extends any[]> = T extends [...any[], infer L] ? L : never;
type Length<T extends any[]> = T['length'];

type H = Head<[1, 2, 3]>;     // 1
type T2 = Tail<[1, 2, 3]>;    // [2, 3]
type L = Last<[1, 2, 3]>;     // 3
type Len = Length<[1, 2, 3]>;  // 3

// String manipulation at the type level
type Split<S extends string, D extends string> =
  S extends `${infer Head}${D}${infer Tail}`
    ? [Head, ...Split<Tail, D>]
    : [S];

type Parts = Split<"a.b.c", ".">;  // ["a", "b", "c"]

// Type-safe deep property access
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

// Type-safe get function
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

const host = deepGet(config, "database.host");  // string type
const port = deepGet(config, "server.port");    // number type
```

### 5.3 Generics and Dependency Injection (DI)

```typescript
// TypeScript: type-safe DI container leveraging generics

// Service key type definition
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

  // Registration: type-safe
  register<K extends keyof ServiceMap>(
    key: K,
    factory: () => ServiceMap[K]
  ): void {
    this.factories.set(key as string, factory);
  }

  // Resolution: type-safe
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

// Usage example
const container = new TypedContainer();

container.register("logger", () => new ConsoleLogger());
container.register("database", () => new PostgresDatabase("localhost:5432"));
container.register("userRepository", () =>
  new UserRepository(container.resolve("database"))
);

const logger = container.resolve("logger");     // Logger type
const db = container.resolve("database");       // Database type
const repo = container.resolve("userRepository"); // UserRepository type
// container.resolve("unknown");                 // compile error
```

### 5.4 Generics and the Monad Pattern

```typescript
// TypeScript: implementing monad-like patterns with generics

// Functor (has map)
interface Functor<T> {
  map<U>(fn: (value: T) => U): Functor<U>;
}

// Monad (has flatMap/bind)
interface Monad<T> extends Functor<T> {
  flatMap<U>(fn: (value: T) => Monad<U>): Monad<U>;
}

// Maybe monad (null-safe computation chain)
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

// Usage example: null-safe deep property access
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

// Usage example: validation chain
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

## 6. Practical Patterns

### 6.1 Result Type

```typescript
// Result type: a practical example of generics
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

  // Compose multiple Results
  static all<T, E>(results: Result<T, E>[]): Result<T[], E> {
    const values: T[] = [];
    for (const result of results) {
      if (!result.isOk) return Result.err(result.error!);
      values.push(result.value!);
    }
    return Result.ok(values);
  }
}

// Usage example
function parseNumber(s: string): Result<number, string> {
  const n = Number(s);
  if (isNaN(n)) return Result.err(`"${s}" is not a number`);
  return Result.ok(n);
}

function divide(a: number, b: number): Result<number, string> {
  if (b === 0) return Result.err("Division by zero");
  return Result.ok(a / b);
}

// Compose via chaining
const result = parseNumber("42")
  .flatMap(n => divide(n, 7))
  .map(n => n.toFixed(2))
  .match({
    ok: value => `Result: ${value}`,
    err: error => `Error: ${error}`,
  });
// "Result: 6.00"
```

### 6.2 Type-Safe Event System

```typescript
// Type-safe event bus leveraging generics
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
    // Apply middlewares
    let processedData = data;
    for (const middleware of this.middlewares) {
      processedData = middleware(event as string, processedData) ?? processedData;
    }

    const handlers = this.handlers.get(event);
    if (!handlers) return;

    const promises = [...handlers].map(handler => handler(processedData));
    await Promise.all(promises);
  }

  // Type-safe waiting
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

// Usage example
const bus = new TypedEventBus<EventDefinitions>();

bus.on("user:created", (data) => {
  console.log(`New user: ${data.email}`);  // type-safe: email is string
});

bus.on("order:placed", (data) => {
  console.log(`Order ${data.orderId}: ¥${data.total}`);  // total is number
});

await bus.emit("user:created", {
  userId: "u-1",
  email: "tanaka@example.com",
  createdAt: new Date(),
});
```

### 6.3 Type-Safe Builder Pattern (Phantom Types)

```typescript
// Type-safe builder using phantom types
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

  // build can only be called when all fields have been set
  build(
    this: UserBuilder<{ hasName: true; hasEmail: true; hasAge: true }>
  ): User {
    return this.data as User;
  }
}

// Usage example
const user = UserBuilder.create()
  .setName("Taro")
  .setEmail("taro@example.com")
  .setAge(30)
  .build();  // OK: all fields have been set

// const incomplete = UserBuilder.create()
//   .setName("Taro")
//   .build();  // compile error: email and age are not set
```

### 6.4 Generics and the Repository Pattern

```typescript
// Type-safe generic repository
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

// Generic implementation
class InMemoryRepository<T extends Entity> implements Repository<T> {
  private items = new Map<string, T>();
  private idCounter = 0;

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  async findAll(options?: QueryOptions<T>): Promise<T[]> {
    let results = [...this.items.values()];

    // Filtering
    if (options?.where) {
      results = results.filter(item =>
        Object.entries(options.where!).every(
          ([key, value]) => (item as any)[key] === value
        )
      );
    }

    // Sorting
    if (options?.orderBy) {
      const { field, direction } = options.orderBy;
      results.sort((a, b) => {
        const aVal = a[field];
        const bVal = b[field];
        const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        return direction === "asc" ? cmp : -cmp;
      });
    }

    // Pagination
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

// Concrete entities
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

// Type-safe repository instances
const userRepo = new InMemoryRepository<UserEntity>();
const orderRepo = new InMemoryRepository<OrderEntity>();

// Fully type-safe
const users = await userRepo.findAll({
  where: { role: "admin" },
  orderBy: { field: "createdAt", direction: "desc" },
  limit: 10,
});

const orders = await orderRepo.findAll({
  where: { status: "pending" },
  // where: { unknownField: "value" },  // compile error
});
```

---

## 7. Generics Best Practices

### 7.1 Naming Conventions

```
Common type parameter names:
  T       -> Type (generic type)
  E       -> Element (element of a collection)
  K       -> Key (key of a map)
  V       -> Value (value of a map)
  R       -> Return (return type)
  S, U    -> second, third type parameter
  N       -> Number
  P       -> Parameter / Props

Good naming:
  Repository<Entity>
  Converter<From, To>
  Handler<Request, Response>
  Mapper<Input, Output>
  Validator<T>

Naming to avoid:
  Repository<A> (meaningless)
  Handler<X, Y, Z> (unclear what's what)
```

### 7.2 Beware of Overusing Generics

```typescript
// Bad: excessive generics
function add<T extends number>(a: T, b: T): T {
  return (a + b) as T;  // cast required = meaningless
}

// Good: when generics are not needed
function add(a: number, b: number): number {
  return a + b;
}

// Bad: a type parameter that is used only once
function logValue<T>(value: T): void {
  console.log(value);
}

// Good: unknown is sufficient
function logValue(value: unknown): void {
  console.log(value);
}

// Good: when generics are needed: expressing the type relationship between input and output
function identity<T>(value: T): T {
  return value;  // guarantees that the same type as the input is returned
}

// Good: when generics are needed: expressing the type relationship between multiple arguments
function merge<T, U>(obj1: T, obj2: U): T & U {
  return { ...obj1, ...obj2 };
}
```

### 7.3 Keep Constraints Minimal

```typescript
// Bad: unnecessarily strict constraints
function getName<T extends { name: string; age: number; email: string }>(obj: T): string {
  return obj.name;  // age and email are not used
}

// Good: minimal necessary constraints
function getName<T extends { name: string }>(obj: T): string {
  return obj.name;
}

// Bad: constraining with a concrete class
function process<T extends UserService>(service: T): void {
  service.getUsers();
}

// Good: constraining with an interface
interface HasGetUsers {
  getUsers(): User[];
}
function process<T extends HasGetUsers>(service: T): void {
  service.getUsers();
}
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Your understanding deepens through not only theory but also by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping into advanced topics. We recommend firmly understanding the basic concepts described in this guide before moving on to the next step.

### Q3: How is it used in practice?

Knowledge on this topic is frequently leveraged in day-to-day development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Concept | Key Point |
|------|---------|
| Generics | Parameterize types to write generic code |
| Constraints | Restrict type parameters with extends/super |
| Covariance | Output (Producer) -> extends |
| Contravariance | Input (Consumer) -> super |
| Type erasure | Java/TS. No type information at runtime |
| Monomorphization | Rust/C++. Code is generated per type |
| Reification | C#. Type information is retained at runtime as well |
| Conditional types | Type-level programming in TypeScript |
| F-bounded | Recursive type bounds (return one's own type) |
| Phantom Types | Represent state via type parameters |

### Comparison of Generics Across Languages

| Feature | Java | TypeScript | Rust | C# | C++ | Python |
|------|------|-----------|------|-----|-----|--------|
| Implementation method | Type erasure | Type erasure | Monomorphization | Reification | Monomorphization | Type hints |
| Runtime type info | None | None | None | Yes | None | Yes (dynamic) |
| Value type support | None | N/A | Yes | Yes | Yes | N/A |
| Variance declaration | Use-site | Declaration-site | None | Declaration-site | None | None |
| Conditional types | None | Yes | None | None | Yes (C++20) | None |
| Higher-kinded types | None | Limited | None | None | Yes | None |
| Default types | None | Yes | Yes | Yes | Yes | Yes |

---

## Guides to Read Next

---

## References
1. Wadler, P. "Theorems for Free!" ICFP, 1989.
2. Bloch, J. "Effective Java." Item 31: Use bounded wildcards. 2018.
3. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
4. Odersky, M. & Zenger, M. "Scalable Component Abstractions." OOPSLA, 2005.
5. Rust Reference. "Generics." https://doc.rust-lang.org/reference/items/generics.html
6. TypeScript Handbook. "Generics." https://www.typescriptlang.org/docs/handbook/2/generics.html
7. Microsoft Docs. "Generics in C#." https://learn.microsoft.com/en-us/dotnet/csharp/fundamentals/types/generics
