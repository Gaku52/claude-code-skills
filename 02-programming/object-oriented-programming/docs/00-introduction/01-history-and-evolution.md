# History and Evolution of OOP

> OOP began with Simula in the 1960s, evolved through Smalltalk, C++, and Java, and has developed into today's multi-paradigm languages. Knowing the history helps you understand "why it is designed this way."

## What You Will Learn in This Chapter

- [ ] Understand the evolution of OOP from its birth to the present day
- [ ] Grasp the innovations of each era and their impact
- [ ] Understand the problems each language tried to solve and the trade-offs involved
- [ ] Gain a perspective on where modern OOP is headed

## Prerequisite Knowledge

Reading this guide will be more productive if you have the following background:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [What is OOP](./00-what-is-oop.md)

---

## 1. OOP Timeline

```
1960s: Birth
  1967  Simula       — The ancestor of OOP. Introduced classes and inheritance
                       Developed by Dahl and Nygaard in Norway
                       Originally for simulation -> generalized

1970s: Establishment of Pure OOP
  1972  Smalltalk    — Alan Kay (Xerox PARC)
                       "Everything is an object"
                       Message passing, GC, IDE, GUI
                       -> Established most modern OOP concepts

1980s: Practical Adoption
  1983  C++          — Bjarne Stroustrup
                       C + OOP. "Zero-cost abstractions"
                       Static typing + multiple inheritance
  1986  Objective-C  — C + Smalltalk messaging
                       -> Adopted by Apple/NeXT

1990s: Widespread Adoption
  1995  Java         — Sun Microsystems
                       "Write once, run anywhere"
                       Single inheritance + interfaces
                       GC, JVM -> the enterprise standard
  1995  JavaScript   — Prototype-based OOP
                       Object creation without classes
  1993  Ruby         — Yukihiro Matsumoto
                       "Everything is an object," developer happiness

2000s: Reflection and Refinement
  2000  C#           — Microsoft (counter to Java)
                       Properties, delegates, LINQ
  2003  Scala        — Fusion of OOP + FP
                       Runs on the JVM

2010s: Modern OOP
  2011  Kotlin       — A better Java. Null safety, data classes
  2014  Swift        — Protocol-oriented programming
                       Value-type centric, reference counting
  2012  TypeScript   — JavaScript + type safety
                       Structural typing

2020s: Post-OOP
  -> Multi-paradigm (OOP + FP) becomes the standard
  -> From "pure OOP" to "OOP when needed"
  -> Emphasis on composition, shrinking use of inheritance
```

---

## 2. Innovations of Each Era

### Simula (1967): The Invention of Classes and Inheritance

```
Simula's innovations:
  1. Class: a "blueprint" that unifies data and procedures
  2. Object: an "instance" created from a class
  3. Inheritance: extending an existing class
  4. Virtual procedure: prototype of polymorphism

Background:
  -> Developed for discrete-event simulation
  -> The need to "represent real-world things in a program"
  -> Customers, cars, factories... expressed as "objects"
```

The background to Simula's birth lies in simulation research in Norway during the 1960s. Ole-Johan Dahl and Kristen Nygaard built on ALGOL 60 to express at the language level the concepts needed for simulation.

```
Simula's design philosophy:

  Problems with ALGOL 60:
    -> Data structures and procedures were separated
    -> No natural way to represent simulation subjects
      (customers, cars, factories)
    -> Coroutine-like concurrency was needed

  Solutions in Simula 67:
    -> Unify data and procedures in classes
    -> Express commonality and differences through inheritance
    -> Switch runtime behavior via virtual procedures
    -> Achieve pseudo-concurrency with coroutine features

  Example: Bank simulation
    class Customer:
      arrival time, service time, wait time
      -> each customer represented as an object

    class Teller:
      customer queue, customer being served
      -> tellers also represented as objects

    class Bank:
      list of tellers, simulation time
      -> the object managing the whole simulation
```

Simula code example (pseudo-code):

```
! Simula 67-style pseudo-code
Class Vehicle;
  Virtual: Real Procedure fuelConsumption;
Begin
  Real speed, weight;

  Procedure accelerate(delta);
    Real delta;
  Begin
    speed := speed + delta;
  End;
End;

Vehicle Class Car;
Begin
  Integer passengers;

  Real Procedure fuelConsumption;
  Begin
    fuelConsumption := weight * speed * 0.01 + passengers * 0.5;
  End;
End;

Vehicle Class Truck;
Begin
  Real cargo_weight;

  Real Procedure fuelConsumption;
  Begin
    fuelConsumption := (weight + cargo_weight) * speed * 0.02;
  End;
End;
```

### Smalltalk (1972): Establishment of Pure OOP

```
Smalltalk's innovations:
  1. Everything is an object (numbers, booleans, nil too)
  2. Message passing (not method invocation)
  3. Garbage collection
  4. Invention of the Integrated Development Environment (IDE)
  5. Invention of the MVC pattern
  6. Reflection (metaprogramming)

Alan Kay's philosophy:
  "OOP is about messaging.
   The essence is the exchange of messages between objects,
   more than classes or inheritance."

  -> Many modern languages evolved in a different direction than Kay intended
  -> Kay: "C++ and Java are not the OOP I intended."
```

Xerox PARC, where Smalltalk was born, was a research laboratory that produced many of the core concepts of modern computing.

```
Contributions of Xerox PARC (1970s):
  -> GUI (Graphical User Interface)
  -> WYSIWYG editor
  -> Ethernet (LAN)
  -> Laser printer
  -> Smalltalk (OOP + IDE + GUI)

  Alan Kay's vision:
    -> The "Dynabook" concept
    -> A computer that even children could program
    -> Objects should operate independently like "small computers"
      and communicate via messages
    -> Inspiration drawn from biological cells

Smalltalk message passing:
  3 + 4
  -> Send a "+" message with argument 4 to 3
  -> 3 (an Integer object) decides how to add by itself

  "hello" size
  -> Send the "size" message to "hello"
  -> The String object returns the number of characters

  collection do: [:each | each printNl]
  -> Send the "do:" message with a block argument to collection
  -> The collection decides how to iterate by itself

  Important difference:
    C++/Java: compiler resolves method calls
    Smalltalk: objects handle messages dynamically
    -> Can also handle cases where no method corresponds to the message
    -> Metaprogramming via doesNotUnderstand:
```

The concepts Smalltalk invented and established have deeply permeated modern software development.

```
Smalltalk's legacy to the modern world:

  1. MVC pattern (Model-View-Controller)
     -> The basic structure of web frameworks
     -> Rails, Django, Spring MVC, ASP.NET MVC
     -> Also influenced the design philosophy of React/Vue

  2. IDE (Integrated Development Environment)
     -> Integration of code editor + debugger + browser
     -> The ancestor of Eclipse, IntelliJ IDEA, VS Code

  3. Refactoring
     -> Systematic methods for improving code structure
     -> Martin Fowler's book grew out of the Smalltalk community

  4. Test-Driven Development (TDD)
     -> SUnit (Smalltalk's unit testing framework)
     -> The prototype for JUnit, pytest, Jest

  5. Design Patterns
     -> Many of the GoF patterns were discovered in the Smalltalk community
     -> Iterator, Observer, Strategy and others originated in Smalltalk

  6. Agile development
     -> XP (Extreme Programming) came out of a Smalltalk project
     -> Kent Beck came from the Smalltalk community
```

### C++ (1983): Practical Adoption and Static Typing

```
C++'s innovations:
  1. Backward compatibility with C (leveraging existing code)
  2. Compile-time checks via static typing
  3. Multiple inheritance
  4. Templates (prototype of generics)
  5. Operator overloading
  6. RAII (Resource Acquisition Is Initialization)

Impact:
  -> Spread the definition "OOP = classes + inheritance + polymorphism"
  -> The philosophy of zero-cost abstractions
  -> But complexity also grew (C++ is one of the most complex languages)
```

C++'s design philosophy is based on the "zero-overhead principle."

```
Bjarne Stroustrup's design philosophy:

  1. Zero-overhead principle:
     "You don't pay for what you don't use."
     "What you do use costs the same as hand-written code."
     -> The cost of a virtual function table is the same as
       writing a table of function pointers yourself.

  2. C compatibility:
     -> Existing C code can be used as-is
     -> Gradual adoption of OOP is possible
     -> Encouraged adoption in systems programming

  3. Multi-paradigm:
     -> Not just OOP, but also procedural, generic, and functional
     -> "Don't enforce a particular style"

  C++'s impact on OOP:
    Positives:
      -> Brought OOP into practical systems programming
      -> Established the combination of static typing and OOP
      -> Discovery of template metaprogramming

    Negatives:
      -> Diamond problem of multiple inheritance
      -> Excessively complex language specification
      -> The "C with Classes" style of use became widespread
```

```cpp
// C++: A practical example of OOP (RAII pattern)

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <vector>

// RAII: Resources are acquired at initialization and released in the destructor
class FileHandle {
private:
    std::fstream file;
    std::string filename;

public:
    // Open the file in the constructor (resource acquisition)
    explicit FileHandle(const std::string& fname)
        : filename(fname) {
        file.open(fname, std::ios::in | std::ios::out);
        if (!file.is_open()) {
            throw std::runtime_error("ファイルを開けません: " + fname);
        }
        std::cout << "ファイルを開きました: " << fname << std::endl;
    }

    // Close the file in the destructor (resource release)
    ~FileHandle() {
        if (file.is_open()) {
            file.close();
            std::cout << "ファイルを閉じました: " << filename << std::endl;
        }
    }

    // Disable copying (to prevent double release of the resource)
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // Allow moving (transfer of ownership)
    FileHandle(FileHandle&& other) noexcept
        : file(std::move(other.file)), filename(std::move(other.filename)) {}

    void write(const std::string& data) {
        file << data;
    }

    std::string readAll() {
        file.seekg(0);
        return std::string(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );
    }
};

// Smart pointers: the memory-management version of RAII
class ResourceManager {
public:
    // unique_ptr: exclusive ownership
    std::unique_ptr<FileHandle> openFile(const std::string& filename) {
        return std::make_unique<FileHandle>(filename);
    }

    // shared_ptr: shared ownership (reference counting)
    std::shared_ptr<std::vector<int>> createSharedData() {
        return std::make_shared<std::vector<int>>();
    }
};

// C++ templates: compile-time polymorphism
template<typename Shape>
double calculateArea(const Shape& shape) {
    return shape.area(); // Method resolved at compile time
}

class Circle {
    double radius;
public:
    explicit Circle(double r) : radius(r) {}
    double area() const { return 3.14159 * radius * radius; }
};

class Rectangle {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    double area() const { return width * height; }
};

// Templates achieve polymorphism without virtual functions
// -> Zero overhead (no indirect calls through a vtable)
```

### Java (1995): Enterprise Standardization

```
Java's innovations:
  1. Single inheritance + interfaces (avoiding the problems of multiple inheritance)
  2. Garbage collection (relief from C++ memory management)
  3. Portability via the JVM
  4. A rich standard library
  5. Namespace management via packages

Impact:
  -> Became the de facto standard language for enterprise
  -> Popularized design patterns (GoF)
  -> But also criticized as "too verbose" and "too much boilerplate"
  -> The AbstractSingletonProxyFactoryBean problem
```

Java realized the vision of "write once, run anywhere" and became the standard for enterprise development.

```java
// Java: Evolution of enterprise patterns

// === Java 1.0 era (1995): Basic OOP ===
public class Employee {
    private String name;
    private double salary;

    public Employee(String name, double salary) {
        this.name = name;
        this.salary = salary;
    }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public double getSalary() { return salary; }
    public void setSalary(double salary) { this.salary = salary; }

    @Override
    public String toString() {
        return "Employee{name='" + name + "', salary=" + salary + "}";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Employee employee = (Employee) o;
        return Double.compare(employee.salary, salary) == 0
            && Objects.equals(name, employee.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, salary);
    }
}
// -> A mountain of boilerplate (getter/setter/equals/hashCode/toString)


// === Java 5 era (2004): Generics + annotations ===
public interface Repository<T, ID> {
    Optional<T> findById(ID id);
    List<T> findAll();
    T save(T entity);
    void deleteById(ID id);
}

public class EmployeeRepository implements Repository<Employee, Long> {
    private final Map<Long, Employee> store = new HashMap<>();

    @Override
    public Optional<Employee> findById(Long id) {
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public List<Employee> findAll() {
        return new ArrayList<>(store.values());
    }

    @Override
    public Employee save(Employee employee) {
        store.put(employee.getId(), employee);
        return employee;
    }

    @Override
    public void deleteById(Long id) {
        store.remove(id);
    }
}


// === Java 8 era (2014): Lambdas + Stream API ===
public class EmployeeService {
    private final Repository<Employee, Long> repository;

    public EmployeeService(Repository<Employee, Long> repository) {
        this.repository = repository;
    }

    // Stream API: elements of functional programming
    public List<Employee> getHighPaidEmployees(double threshold) {
        return repository.findAll().stream()
            .filter(e -> e.getSalary() > threshold)
            .sorted(Comparator.comparingDouble(Employee::getSalary).reversed())
            .collect(Collectors.toList());
    }

    public Map<String, Double> getAverageSalaryByDepartment() {
        return repository.findAll().stream()
            .collect(Collectors.groupingBy(
                Employee::getDepartment,
                Collectors.averagingDouble(Employee::getSalary)
            ));
    }

    public Optional<Employee> findHighestPaid() {
        return repository.findAll().stream()
            .max(Comparator.comparingDouble(Employee::getSalary));
    }
}


// === Java 17+ era (2021-): Records + Sealed + Pattern Matching ===

// Record: reduced boilerplate (immutable data class)
public record EmployeeRecord(
    long id,
    String name,
    String department,
    double salary
) {
    // Compact constructor for validation
    public EmployeeRecord {
        if (salary < 0) throw new IllegalArgumentException("給与は正の数");
        if (name == null || name.isBlank()) throw new IllegalArgumentException("名前は必須");
    }
}

// Sealed class: algebraic data type
public sealed interface PaymentMethod
    permits CreditCard, BankTransfer, DigitalWallet {
}

public record CreditCard(String number, String expiry) implements PaymentMethod {}
public record BankTransfer(String accountNumber, String bankCode) implements PaymentMethod {}
public record DigitalWallet(String walletId, String provider) implements PaymentMethod {}

// Pattern Matching: type-safe branching in switch expressions
public String processPayment(PaymentMethod method, double amount) {
    return switch (method) {
        case CreditCard cc -> "クレジットカード %s で %.0f円決済".formatted(
            cc.number().substring(cc.number().length() - 4), amount);
        case BankTransfer bt -> "銀行振込 %s へ %.0f円送金".formatted(
            bt.bankCode(), amount);
        case DigitalWallet dw -> "%s ウォレット %s で %.0f円決済".formatted(
            dw.provider(), dw.walletId(), amount);
    };
}
```

### Ruby (1993): Developer Happiness

Ruby was designed by Yukihiro Matsumoto (Matz) with the goal of maximizing "programmer happiness."

```
Ruby's design philosophy:
  -> Syntax natural for humans
  -> Principle of Least Surprise
  -> Everything is an object (influenced by Smalltalk)
  -> Metaprogramming (the power to extend the language)

Ruby's impact on OOP:
  1. Block syntax: concise notation for closures
  2. Open class: extending existing classes after the fact
  3. Mixin: modules as an alternative to multiple inheritance
  4. DSL: easy construction of domain-specific languages
  5. Rails: a revolution for OOP in web development
```

```ruby
# Ruby: pure OOP + metaprogramming

# Everything is an object
42.class          # => Integer
42.even?          # => true
"hello".reverse   # => "olleh"
nil.class         # => NilClass
true.class        # => TrueClass

# Mixin (alternative to multiple inheritance)
module Serializable
  def to_json
    require 'json'
    hash = {}
    instance_variables.each do |var|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    end
    JSON.generate(hash)
  end
end

module Auditable
  def self.included(base)
    base.instance_variable_set(:@audit_log, [])
  end

  def log_change(message)
    self.class.instance_variable_get(:@audit_log) << {
      timestamp: Time.now,
      object_id: object_id,
      message: message
    }
  end
end

class User
  include Serializable
  include Auditable

  attr_reader :name, :email

  def initialize(name, email)
    @name = name
    @email = email
    log_change("User created: #{name}")
  end

  def update_email(new_email)
    old = @email
    @email = new_email
    log_change("Email changed: #{old} -> #{new_email}")
  end
end

# Open class: extending existing classes
class String
  def palindrome?
    self == self.reverse
  end
end

"racecar".palindrome?  # => true
"hello".palindrome?    # => false

# Metaprogramming: dynamic method definition
class ActiveRecordLike
  def self.has_attribute(name, type: :string, default: nil)
    # getter
    define_method(name) do
      instance_variable_get("@#{name}") || default
    end

    # setter
    define_method("#{name}=") do |value|
      instance_variable_set("@#{name}", value)
    end

    # query method
    define_method("#{name}?") do
      !send(name).nil? && send(name) != "" && send(name) != false
    end
  end
end

class Product < ActiveRecordLike
  has_attribute :name, type: :string
  has_attribute :price, type: :float, default: 0
  has_attribute :in_stock, type: :boolean, default: true
end

p = Product.new
p.name = "Ruby本"
p.price = 3000
p.name?      # => true
p.in_stock?  # => true
```

### Python (1991): Practical OOP

Python was designed by Guido van Rossum of the Netherlands. It supports OOP but does not enforce it, resulting in a practical design.

```
Characteristics of Python's OOP:
  1. Multi-paradigm: OOP is just one option
  2. Duck typing: "If it walks like a duck and quacks like a duck, it's a duck"
  3. Convention-based access control: _private is not enforced
  4. Special methods: customize via __init__, __str__, __eq__, etc.
  5. Decorators: a concise means of metaprogramming
  6. Data classes: reduced boilerplate since Python 3.7+
```

```python
# Python: modern OOP in practice

from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable
from functools import total_ordering
from datetime import datetime


# Data class: automatic generation of boilerplate
@dataclass(frozen=True)  # frozen=True makes it immutable
@total_ordering
class Money:
    """金額値オブジェクト"""
    amount: int
    currency: str = "JPY"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("金額は0以上である必要があります")

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"通貨が異なります: {self.currency} vs {other.currency}")
        return Money(self.amount - other.amount, self.currency)

    def __mul__(self, factor: int | float) -> Money:
        return Money(int(self.amount * factor), self.currency)

    def __lt__(self, other: Money) -> bool:
        if self.currency != other.currency:
            raise ValueError("通貨が異なります")
        return self.amount < other.amount

    def __str__(self) -> str:
        if self.currency == "JPY":
            return f"¥{self.amount:,}"
        return f"{self.amount / 100:.2f} {self.currency}"


# Protocol: structural typing (type-safe version of duck typing)
@runtime_checkable
class Discountable(Protocol):
    """割引適用可能なもの"""
    def apply_discount(self, rate: float) -> Money: ...
    @property
    def price(self) -> Money: ...


# Abstract base class
class Product(ABC):
    """商品の抽象基底クラス"""

    def __init__(self, name: str, base_price: Money):
        self._name = name
        self._base_price = base_price

    @property
    def name(self) -> str:
        return self._name

    @property
    def price(self) -> Money:
        return self._base_price

    @abstractmethod
    def description(self) -> str:
        """商品の説明を返す"""
        pass

    def apply_discount(self, rate: float) -> Money:
        """割引後の価格を計算"""
        if not 0 <= rate <= 1:
            raise ValueError("割引率は0〜1の範囲")
        return self._base_price * (1 - rate)


# Concrete classes
@dataclass
class Book(Product):
    """書籍"""
    _name: str = field(init=False)
    _base_price: Money = field(init=False)
    author: str = ""
    isbn: str = ""
    pages: int = 0

    def __init__(self, name: str, price: Money, author: str, isbn: str = "", pages: int = 0):
        super().__init__(name, price)
        self.author = author
        self.isbn = isbn
        self.pages = pages

    def description(self) -> str:
        return f"『{self.name}』{self.author}著 ({self.pages}ページ)"


@dataclass
class Electronics(Product):
    """電子機器"""
    _name: str = field(init=False)
    _base_price: Money = field(init=False)
    brand: str = ""
    warranty_months: int = 12

    def __init__(self, name: str, price: Money, brand: str, warranty_months: int = 12):
        super().__init__(name, price)
        self.brand = brand
        self.warranty_months = warranty_months

    def description(self) -> str:
        return f"{self.brand} {self.name} (保証: {self.warranty_months}ヶ月)"


# Decorator pattern
class DiscountedProduct:
    """割引適用済み商品（デコレータ）"""

    def __init__(self, product: Product, discount_rate: float, reason: str = ""):
        self._product = product
        self._discount_rate = discount_rate
        self._reason = reason

    @property
    def name(self) -> str:
        return f"{self._product.name} [{self._reason}]" if self._reason else self._product.name

    @property
    def price(self) -> Money:
        return self._product.apply_discount(self._discount_rate)

    @property
    def original_price(self) -> Money:
        return self._product.price

    def description(self) -> str:
        return f"{self._product.description()} - {self._discount_rate*100:.0f}%OFF"


# Context manager: Pythonic resource management
class DatabaseConnection:
    """データベース接続（コンテキストマネージャ）"""

    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        self._connected = False

    def __enter__(self):
        print(f"DB接続開始: {self._connection_string}")
        self._connected = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._connected:
            print("DB接続終了")
            self._connected = False
        return False  # Re-raise the exception

    def query(self, sql: str) -> list[dict]:
        if not self._connected:
            raise RuntimeError("接続されていません")
        print(f"SQL実行: {sql}")
        return []


# Example usage
with DatabaseConnection("postgresql://localhost/mydb") as db:
    results = db.query("SELECT * FROM users")
# -> __exit__ is called automatically (even when an exception occurs)
```

---

## 3. Directions in the Evolution of OOP

```
1st generation (1967-1980): Class-based
  Simula, Smalltalk
  -> "Model the world with objects"

2nd generation (1983-1995): Static typing + practical adoption
  C++, Java, C#
  -> "Structure large-scale development"

3rd generation (2000-2015): Lightweight OOP + FP fusion
  Ruby, Scala, Kotlin, Swift
  -> "Reduce boilerplate and bring in the good parts of functional programming"

4th generation (2015-present): Post-OOP
  Rust (traits), Go (interfaces), TypeScript (structural typing)
  -> "Eliminate inheritance, design with composition and interfaces"

Trends in evolution:
  Multiple inheritance -> single inheritance -> composition over inheritance -> no inheritance
  Mutable -> immutable first
  Class-centric -> interface/trait-centric
  Implicit -> explicit
```

### 3.1 The Decline of Inheritance

One of the biggest shifts in OOP history has been the move "from inheritance to composition."

```
History of the decline of inheritance:

  1990s: Inheritance is at the center of OOP
    -> GoF: "Program to an interface"
    -> In practice, however, deep inheritance hierarchies proliferated

  2000s: The problems of inheritance become recognized
    -> Joshua Bloch (Effective Java): "Favor composition over inheritance"
    -> The fragile base class problem
    -> Violations of the Liskov Substitution Principle

  2010s: Languages that eliminate inheritance emerge
    -> Go: no inheritance, implicit interface implementation
    -> Rust: no inheritance, trait-based
    -> Swift: protocol-oriented (value types + protocols)

  2020s: Consensus that inheritance should be "used sparingly"
    -> Modern Java recommends sealed class + record
    -> Kotlin's data class cannot be inherited
    -> Deep inheritance hierarchies are a clear anti-pattern
```

```typescript
// TypeScript: Evolution from inheritance to composition

// === 1990s style: deep inheritance hierarchy ===
// An approach riddled with problems

/*
abstract class Animal {
  abstract speak(): string;
}

class Mammal extends Animal {
  breathe(): string { return "肺で呼吸"; }
  abstract speak(): string;
}

class Pet extends Mammal {
  constructor(public owner: string) { super(); }
  abstract speak(): string;
}

class Dog extends Pet {
  speak(): string { return "ワン！"; }
}

class Cat extends Pet {
  speak(): string { return "ニャー！"; }
}

// Problem: Penguins are birds but can't fly -> inheritance hierarchy breaks
// Problem: Adding new behaviors is difficult
// Problem: Swapping implementations for testing is difficult
*/


// === 2020s style: composition + interfaces ===

// Define behaviors via interfaces
interface CanSpeak {
  speak(): string;
}

interface CanMove {
  move(): string;
}

interface CanSwim {
  swim(): string;
}

interface CanFly {
  fly(): string;
}

// Combine via composition
class DogComposed implements CanSpeak, CanMove, CanSwim {
  constructor(
    public readonly name: string,
    public readonly owner: string,
  ) {}

  speak(): string { return `${this.name}: ワン！`; }
  move(): string { return `${this.name}が走る`; }
  swim(): string { return `${this.name}が泳ぐ`; }
}

class PenguinComposed implements CanSpeak, CanMove, CanSwim {
  constructor(public readonly name: string) {}

  speak(): string { return `${this.name}: ペンペン！`; }
  move(): string { return `${this.name}がよちよち歩く`; }
  swim(): string { return `${this.name}が高速で泳ぐ`; }
  // fly() is not implemented -> the inability to fly is guaranteed at compile time
}

class EagleComposed implements CanSpeak, CanMove, CanFly {
  constructor(public readonly name: string) {}

  speak(): string { return `${this.name}: ピーッ！`; }
  move(): string { return `${this.name}が飛び回る`; }
  fly(): string { return `${this.name}が大空を飛ぶ`; }
  // swim() is not implemented -> the inability to swim is expressed in the type
}

// Require only the interfaces that are actually needed
function makeSwimRace(swimmers: CanSwim[]): void {
  for (const s of swimmers) {
    console.log(s.swim());
  }
}

// Dog and Penguin can swim, but Eagle cannot
// -> caught as a compile-time error
makeSwimRace([
  new DogComposed("ポチ", "田中"),
  new PenguinComposed("ペンタ"),
  // new EagleComposed("タカ"),  // Compile error: does not implement CanSwim
]);
```

### 3.2 Evolution of Type Systems

The type systems of OOP languages have also evolved significantly.

```
Evolution of type systems:

  Nominal typing:
    -> Java, C#, C++
    -> No compatibility unless type names match
    -> You must explicitly declare implements / extends

  Structural typing:
    -> TypeScript, Go
    -> Types are compatible if their structure (method signatures) matches
    -> Even without writing implements, it's OK if methods match

  Duck typing:
    -> Python, Ruby
    -> OK if the required methods exist at runtime
    -> "If it walks like a duck and quacks like a duck, it's a duck"

  Direction of evolution:
    Nominal (too strict)
      -> Structural (flexible + type-safe)
        -> Duck typing + type hints (flexible + documented)
```

```typescript
// TypeScript: Practical example of structural typing

// Types are compatible even without explicitly implementing an interface
interface Printable {
  toString(): string;
}

interface HasLength {
  length: number;
}

// Works with classes
class Document implements Printable {
  constructor(private content: string) {}
  toString(): string { return this.content; }
}

// Works with plain objects too (as long as the structure matches)
const logEntry = {
  toString(): string { return "2024-01-01 INFO: Application started"; }
};

// Arrays satisfy HasLength
const items = [1, 2, 3]; // has { length: number }

function print(item: Printable): void {
  console.log(item.toString());
}

function getLength(item: HasLength): number {
  return item.length;
}

print(new Document("Hello"));   // OK
print(logEntry);                 // OK: structure matches
getLength(items);               // OK: has a length property
getLength("hello");             // OK: strings also have length
```

---

## 4. Modern OOP: Multi-paradigm

```kotlin
// Kotlin: example of modern OOP
// Data class (reduced boilerplate)
data class User(
    val name: String,
    val email: String,
    val age: Int
)

// sealed class (algebraic data type - influence from FP)
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: Throwable) : Result<Nothing>()
}

// Extension functions (adding functionality without modifying the object)
fun String.isValidEmail(): Boolean =
    this.matches(Regex("^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$"))

// Higher-order functions (element of FP)
fun <T> List<T>.filterAndMap(
    predicate: (T) -> Boolean,
    transform: (T) -> String
): List<String> = this.filter(predicate).map(transform)
```

```swift
// Swift: Protocol-oriented programming
protocol Drawable {
    func draw()
}

protocol Resizable {
    func resize(by factor: Double)
}

// Protocol extension (default implementation)
extension Drawable {
    func draw() {
        print("Default drawing")
    }
}

// Value type (struct) + protocol conformance
struct Circle: Drawable, Resizable {
    var radius: Double

    func draw() {
        print("Drawing circle with radius \(radius)")
    }

    func resize(by factor: Double) -> Circle {
        Circle(radius: radius * factor)
    }
}
```

### 4.1 Kotlin: A Better Java

Kotlin is a language JetBrains designed with the goal of being "a better Java," and it has many features characteristic of modern OOP.

```kotlin
// Kotlin: A comprehensive example of modern OOP

// === Null safety ===
fun processUser(name: String?) {
    // The compiler enforces null checks
    val length = name?.length ?: 0
    val upper = name?.uppercase() ?: "UNKNOWN"

    // Smart cast
    if (name != null) {
        // Inside this branch, name is String (non-null)
        println(name.length)
    }
}

// === Sealed class + when expression (exhaustive pattern matching) ===
sealed interface Shape {
    data class Circle(val radius: Double) : Shape
    data class Rectangle(val width: Double, val height: Double) : Shape
    data class Triangle(val base: Double, val height: Double) : Shape
}

fun area(shape: Shape): Double = when (shape) {
    is Shape.Circle -> Math.PI * shape.radius * shape.radius
    is Shape.Rectangle -> shape.width * shape.height
    is Shape.Triangle -> shape.base * shape.height / 2
    // when expressions are exhaustive: adding a new Shape causes a compile error
}

// === Delegation pattern (the by keyword) ===
interface Logger {
    fun log(message: String)
}

class ConsoleLogger : Logger {
    override fun log(message: String) = println("[LOG] $message")
}

class UserService(logger: Logger) : Logger by logger {
    // Delegates the implementation of Logger to ConsoleLogger
    // No need to implement the log() method explicitly

    fun createUser(name: String) {
        log("Creating user: $name")  // Delegated method
        // ... user creation logic
    }
}

// === Coroutines (asynchronous programming) ===
import kotlinx.coroutines.*

class OrderProcessor {
    suspend fun processOrder(orderId: String): Result<Order> {
        return try {
            val order = fetchOrder(orderId)      // async DB fetch
            val validated = validateOrder(order)  // validation
            val charged = chargePayment(validated) // async payment
            Result.success(charged)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    private suspend fun fetchOrder(id: String): Order = withContext(Dispatchers.IO) {
        // Fetch from DB (async)
        delay(100) // simulation
        Order(id, "pending")
    }

    private fun validateOrder(order: Order): Order {
        // Validate business rules
        require(order.status == "pending") { "注文は pending 状態である必要があります" }
        return order
    }

    private suspend fun chargePayment(order: Order): Order = withContext(Dispatchers.IO) {
        // Payment processing (async)
        delay(200) // simulation
        order.copy(status = "confirmed")
    }
}
```

### 4.2 Rust: The Leading Edge of Post-OOP

Rust has neither classes nor inheritance, but it realizes powerful OOP-like patterns via traits and structs.

```rust
// Rust: Trait-based OOP

use std::fmt;

// Trait: interface + default implementation + associated types
trait Animal: fmt::Display {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;

    // Default implementation
    fn introduce(&self) -> String {
        format!("{}は「{}」と鳴きます", self.name(), self.sound())
    }
}

// Struct + trait implementation
struct Dog {
    name: String,
    breed: String,
}

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ワン" }
}

impl fmt::Display for Dog {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "犬「{}」({})", self.name, self.breed)
    }
}

struct Cat {
    name: String,
    indoor: bool,
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ニャー" }
}

impl fmt::Display for Cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let location = if self.indoor { "室内飼い" } else { "外飼い" };
        write!(f, "猫「{}」({})", self.name, location)
    }
}

// Trait objects: runtime polymorphism
fn introduce_all(animals: &[&dyn Animal]) {
    for animal in animals {
        println!("{}", animal.introduce());
    }
}

// Generics + trait bounds: compile-time polymorphism
fn loudest_sound<T: Animal>(animals: &[T]) -> &str {
    // Type determined at compile time -> no vtable -> zero cost
    animals.first().map(|a| a.sound()).unwrap_or("")
}

// Enums: algebraic data types (one of Rust's strengths)
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => base * height / 2.0,
        }
    }

    fn perimeter(&self) -> f64 {
        match self {
            Shape::Circle { radius } => 2.0 * std::f64::consts::PI * radius,
            Shape::Rectangle { width, height } => 2.0 * (width + height),
            Shape::Triangle { base, height } => {
                let hyp = (base * base + height * height).sqrt();
                base + height + hyp
            }
        }
    }
}

// Ownership system: compile-time memory safety guarantees
struct FileProcessor {
    path: String,
    content: Option<String>,
}

impl FileProcessor {
    fn new(path: &str) -> Self {
        FileProcessor {
            path: path.to_string(),
            content: None,
        }
    }

    // &self: read-only borrow
    fn path(&self) -> &str {
        &self.path
    }

    // &mut self: mutable borrow
    fn load(&mut self) -> Result<(), std::io::Error> {
        self.content = Some(std::fs::read_to_string(&self.path)?);
        Ok(())
    }

    // self: consumes ownership (cannot be used after the call)
    fn into_content(self) -> Option<String> {
        self.content
    }
}
```

### 4.3 Go: The Pursuit of Simplicity

Go intentionally omits many OOP features in the pursuit of simplicity.

```go
// Go: Structs + interfaces (implicit implementation)
package main

import (
    "fmt"
    "math"
    "sort"
)

// Interface: implemented implicitly
type Shape interface {
    Area() float64
    Perimeter() float64
    String() string
}

// Struct (in place of class)
type Circle struct {
    Radius float64
}

// Method (a function with a receiver)
func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

func (c Circle) String() string {
    return fmt.Sprintf("Circle(r=%.2f)", c.Radius)
}

type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

func (r Rectangle) String() string {
    return fmt.Sprintf("Rect(%.2fx%.2f)", r.Width, r.Height)
}

// Embedding: the alternative to inheritance
type NamedShape struct {
    Shape // Embeds the Shape interface
    Name  string
}

func (ns NamedShape) Describe() string {
    return fmt.Sprintf("%s: area=%.2f", ns.Name, ns.Area())
}

// Composition of interfaces
type ReadWriter interface {
    Reader
    Writer
}

type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// Functional style: heavy use of small interfaces
type SortByArea []Shape

func (s SortByArea) Len() int           { return len(s) }
func (s SortByArea) Less(i, j int) bool { return s[i].Area() < s[j].Area() }
func (s SortByArea) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func main() {
    shapes := []Shape{
        Circle{Radius: 5},
        Rectangle{Width: 3, Height: 4},
        Circle{Radius: 2},
        Rectangle{Width: 10, Height: 1},
    }

    sort.Sort(SortByArea(shapes))

    for _, s := range shapes {
        fmt.Printf("%s -> Area: %.2f\n", s, s.Area())
    }
}
```

---

## 5. The Future of OOP

```
Direction of OOP in the 2020s-2030s:

  1. Spread of algebraic data types
     -> sealed class (Kotlin, Java 17+)
     -> enum + match (Rust)
     -> union types (TypeScript)
     -> OOP + FP hybrid patterns become standardized

  2. Immutability becomes mainstream
     -> record (Java), data class (Kotlin)
     -> frozen dataclass (Python)
     -> readonly (TypeScript)
     -> "Immutable by default, mutable only when necessary" as a principle

  3. Evolution of type inference
     -> Type inference for local variables becomes standard
     -> Spread of structural typing
     -> Compile-time safety + concise notation

  4. Effect systems
     -> Side effects tracked at the type level
     -> Evolution of async/await
     -> Clear separation of pure functions and side effects

  5. Composition APIs
     -> React Hooks, Vue Composition API
     -> Swift Protocol Extensions
     -> Rust Trait + generics
     -> Spread of "OOP-like design without classes"

  6. AI-assisted code generation
     -> Automatic suggestions for OOP design
     -> Automatic application of design patterns
     -> AI assistance for refactoring
```

### 5.1 Effect Systems and Management of Side Effects

```typescript
// TypeScript: Error handling via the Result type (an FP-style approach)

type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

// Express side effects explicitly in the type
class UserService {
  constructor(
    private readonly db: Database,
    private readonly mailer: Mailer,
  ) {}

  // The return type makes "success or failure" explicit
  async createUser(
    name: string,
    email: string,
  ): Promise<Result<User, CreateUserError>> {
    // Validation (pure function)
    const validationResult = this.validateInput(name, email);
    if (!validationResult.ok) return validationResult;

    // DB operation (side effect)
    const existingUser = await this.db.findUserByEmail(email);
    if (existingUser) {
      return err({ type: "DUPLICATE_EMAIL", email });
    }

    const user = await this.db.createUser({ name, email });

    // Sending email (side effect)
    const mailResult = await this.mailer.sendWelcome(user);
    if (!mailResult.ok) {
      // Log on email failure, but the user creation itself is a success
      console.warn("Welcome email failed:", mailResult.error);
    }

    return ok(user);
  }

  private validateInput(
    name: string,
    email: string,
  ): Result<void, CreateUserError> {
    if (!name.trim()) return err({ type: "INVALID_NAME", name });
    if (!email.includes("@")) return err({ type: "INVALID_EMAIL", email });
    return ok(undefined);
  }
}

type CreateUserError =
  | { type: "INVALID_NAME"; name: string }
  | { type: "INVALID_EMAIL"; email: string }
  | { type: "DUPLICATE_EMAIL"; email: string }
  | { type: "DATABASE_ERROR"; cause: Error };
```

### 5.2 The Rise of the Composition API Pattern

```typescript
// TypeScript: React Hooks style (OOP-like design without classes)

// Encapsulating state (without using a class)
function useCounter(initialValue: number = 0) {
  let count = initialValue;

  return {
    get value() { return count; },
    increment() { count++; },
    decrement() { count--; },
    reset() { count = initialValue; },
  };
}

// Encapsulating business logic
function useShoppingCart() {
  const items: Map<string, { name: string; price: number; qty: number }> = new Map();

  return {
    addItem(id: string, name: string, price: number) {
      const existing = items.get(id);
      if (existing) {
        existing.qty++;
      } else {
        items.set(id, { name, price, qty: 1 });
      }
    },

    removeItem(id: string) {
      items.delete(id);
    },

    get total() {
      let sum = 0;
      for (const item of items.values()) {
        sum += item.price * item.qty;
      }
      return sum;
    },

    get itemCount() {
      let count = 0;
      for (const item of items.values()) {
        count += item.qty;
      }
      return count;
    },

    get isEmpty() {
      return items.size === 0;
    },
  };
}

// Composition: combining multiple features
function useCheckout() {
  const cart = useShoppingCart();
  const step = useCounter(1);

  return {
    cart,
    step,

    get canProceed() {
      if (step.value === 1) return !cart.isEmpty;
      if (step.value === 2) return true; // shipping address entered
      return false;
    },

    nextStep() {
      if (this.canProceed && step.value < 3) {
        step.increment();
      }
    },

    previousStep() {
      if (step.value > 1) {
        step.decrement();
      }
    },
  };
}

// -> Realizing OOP's benefits (encapsulation, composition) without using classes
// -> Pattern where a function returns an object (a closure)
// -> Easy to test (no mocks required, more functional in style)
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate the input data
- Implement appropriate error handling
- Also write test code

```python
# Exercise 1: Template for basic implementation
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation and add the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # size limit reached
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**Points:**
- Be conscious of the computational complexity of algorithms
- Choose appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, writing actual code and confirming its behavior deepens your understanding.

### Q2: What mistakes do beginners often make?

Skipping the fundamentals and jumping into applied topics. We recommend firmly grasping the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is used frequently in day-to-day development work. It becomes especially important during code reviews and when designing architectures.

---

## Summary

| Era | Language | Innovation |
|------|------|------|
| 1967 | Simula | Invention of classes and inheritance |
| 1972 | Smalltalk | Pure OOP, messaging, IDE, MVC |
| 1983 | C++ | Statically typed OOP, practical adoption, RAII |
| 1993 | Ruby | Pure OOP, metaprogramming, developer experience |
| 1995 | Java | Enterprise standard, GC, JVM |
| 2010s | Kotlin/Swift | Modern OOP, FP fusion, null safety |
| 2020s | Rust/Go/TS | Post-OOP, composition, type safety |

---

## Next Guides to Read

---

## References
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN, 1993.
2. Stroustrup, B. "The Design and Evolution of C++." Addison-Wesley, 1994.
3. Bloch, J. "Effective Java." 3rd Ed, Addison-Wesley, 2018.
4. Nygaard, K. and Dahl, O-J. "The Development of the Simula Languages." ACM SIGPLAN, 1978.
5. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
6. Matsakis, N. and Klock, F. "The Rust Language." ACM SIGAda, 2014.
7. Odersky, M. and Zenger, M. "Scalable Component Abstractions." OOPSLA, 2005.
