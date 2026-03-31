# Programming Paradigms Overview

> **A paradigm is a fundamental system of thought regarding "how to decompose problems and how to structure solutions."**
> Understanding paradigms is on a fundamentally different level from merely memorizing syntax.
> It is nothing less than the act of forming one's "worldview as a programmer."

---

## What You Will Learn in This Chapter

- [ ] Understand the essential characteristics of major programming paradigms (procedural, OOP, functional, logic, reactive, actor)
- [ ] Grasp the historical background in which each paradigm was born and the problems it aimed to solve
- [ ] Systematically organize the relationships and mutual influences between paradigms
- [ ] Develop the ability to make appropriate design decisions in the multi-paradigm era
- [ ] Recognize and avoid anti-patterns of each paradigm


## Prerequisites

The following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Compilation vs Interpretation](./01-compilation-vs-interpretation.md)

---

## Table of Contents

1. [What Is a Paradigm -- Conceptual Foundations](#1-what-is-a-paradigm----conceptual-foundations)
2. [Historical Lineage of Paradigms](#2-historical-lineage-of-paradigms)
3. [Procedural Programming](#3-procedural-programming)
4. [Object-Oriented Programming (OOP)](#4-object-oriented-programming-oop)
5. [Functional Programming](#5-functional-programming)
6. [Logic Programming](#6-logic-programming)
7. [Reactive Programming and the Actor Model](#7-reactive-programming-and-the-actor-model)
8. [Multi-Paradigm -- The Modern Mainstream](#8-multi-paradigm----the-modern-mainstream)
9. [Guidelines for Paradigm Selection and Design Decisions](#9-guidelines-for-paradigm-selection-and-design-decisions)
10. [Anti-Pattern Collection](#10-anti-pattern-collection)
11. [Exercises (3 Levels)](#11-exercises-3-levels)
12. [FAQ -- Frequently Asked Questions](#12-faq----frequently-asked-questions)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. What Is a Paradigm -- Conceptual Foundations

### 1.1 Definition of a Paradigm

The term "paradigm" originates from a concept proposed by philosopher of science Thomas Kuhn in his 1962 work *The Structure of Scientific Revolutions*.
In programming, a paradigm refers to a **fundamental system of thought about what abstraction framework to use for understanding and describing the structure and behavior of software**.

Robert Floyd stated the following in his 1978 Turing Award lecture "The Paradigms of Programming":

> "A paradigm is a way of conceptualizing what it means to perform computation,
> and how tasks to be carried out on a computer should be structured and organized."

A paradigm does not prescribe "how to write programs" but rather **"how to think about programs."**
Even for the same problem, if the adopted paradigm differs, the way problems are decomposed, data is handled, control flows, and abstraction boundaries are drawn changes fundamentally.

### 1.2 Paradigm Classification System

Paradigms are broadly classified into two lineages.

```
Programming Paradigm Classification
============================================================

          +-- Imperative
          |     |
          |     +-- Procedural
          |     |     Examples: C, Pascal, BASIC
          |     |
          |     +-- Object-Oriented (OOP)
          |     |     Examples: Java, C#, Smalltalk
          |     |
Paradigm -+     +-- Concurrent-Oriented
          |           Examples: Go (goroutines)
          |
          +-- Declarative
                |
                +-- Functional
                |     Examples: Haskell, Erlang, Clojure
                |
                +-- Logic
                |     Examples: Prolog, Datalog
                |
                +-- Reactive
                      Examples: RxJS, ReactiveX
```

**Imperative** describes "how to compute." A program is a sequence of statements that sequentially modifies the machine's state.

**Declarative** describes "what to compute." A program is a declaration of relationships, constraints, and transformations, delegating the execution method to the processing system.

### 1.3 Three Perspectives for Understanding Paradigms

There are three important perspectives for understanding paradigms.

```
+-----------------+--------------------+--------------------+
| Perspective     | Imperative         | Declarative        |
+=================+====================+====================+
| State handling  | Mutable            | Immutable          |
|                 | Reassignment to    | Generation of new  |
|                 | variables          | values             |
+-----------------+--------------------+--------------------+
| Control flow    | Explicit           | Implicit           |
|                 | if/for/while       | Pattern matching   |
|                 |                    | Recursion/HOFs     |
+-----------------+--------------------+--------------------+
| Abstraction     | Procedures/Objects | Functions/Predicates|
| unit            | Modules            | Monads/Type classes|
+-----------------+--------------------+--------------------+
```

### 1.4 Why Learn Paradigms

There are three reasons to learn paradigms.

**First, it broadens the scope of your thinking.** A developer who knows only procedural programming will try to solve every problem with "state changes and control flow." Learning functional programming adds the perspective of "data transformation," enabling more appropriate solution selection.

**Second, it enables understanding of language design intent.** Why does Rust not have inheritance? Why does Haskell not allow variable reassignment? Why was generics added to Go later? All of these are design decisions based on paradigms.

**Third, it dramatically accelerates the learning of new languages.** If you understand the essence of paradigms, a new language can be approached as "a new implementation of a known paradigm." Learning progresses through conceptual mapping rather than syntax memorization.

---

## 2. Historical Lineage of Paradigms

### 2.1 The Evolution of Paradigms in Timeline

```
Year     Event                                       Impact
=================================================================
1936     Turing Machine / Lambda Calculus             Theoretical foundation for imperative / functional
1957     FORTRAN released                             Practical realization of procedural
1958     LISP released                                Ancestor of functional
1962     Simula released                              Birth of OOP concepts
1970     Prolog released                              Practical realization of logic programming
1972     Smalltalk released                           Establishment of pure OOP
1973     C language released                          Definitive procedural language
1978     CSP paper (Hoare)                            Theory of concurrent computation
1986     Erlang development started                   Practical realization of actor model
1987     Haskell project started                      Unification of purely functional
1990     Haskell 1.0                                  Lazy evaluation + type classes
1995     Java 1.0                                     Popularization of OOP
1995     JavaScript released                          Pioneer of multi-paradigm
2003     Scala released                               Integration of OOP + FP
2007     Clojure released                             Functional on the JVM
2010     Rust released                                Ownership + functional elements
2011     Kotlin released                              Practical integration of OOP + FP
2012     Elixir released                              Modern FP on the Erlang VM
2014     Swift released                               Protocol-oriented
2024--   AI-assisted programming                      Automatic paradigm selection
```

### 2.2 The Confluence of Two Great Currents

The history of programming can be understood as the parallel evolution and convergence of two great currents: imperative (derived from Turing machines) and declarative (derived from lambda calculus).

```
1930s          1960s          1980s          2000s          2020s
  |              |              |              |              |
  | Turing       |   C          |  C++         | Java 8       |
  | Machine ----+---FORTRAN ---+---Pascal ----+---C# --------+-- Functional
  |              |              |              |   (lambdas)  |   elements
  |              |              |              |              |   standard in
  |              |              |  Haskell     | Scala        |   imperative
  | Lambda     --+---LISP -----+---ML --------+---Clojure ---+-- Practical
  | Calculus     |              |   (type inf) |   Elixir     |   functional
  |              |              |              |              |   advancing
  |              +-- Simula     +-- Smalltalk  +-- Kotlin     +-- Multi-
  |              |   (OOP dawn) |   (pure OOP) |   Swift      |   paradigm
  |              |              |              |   Rust       |   becomes
  |              |              |              |              |   mainstream
  v              v              v              v              v
```

Until the 1990s, there was a strong oppositional framing of "procedural vs OOP vs functional."
However, from the 2000s onward, the spread of multi-core processors and the demands of distributed systems drove functional concepts such as **immutability, pure functions, and composability** to be absorbed into imperative languages. Java 8's lambda expressions, C#'s LINQ, and Python's generators and list comprehensions are representative examples.

### 2.3 The Current State of Paradigm Fusion

As of the 2020s, virtually no major language is designed purely around a single paradigm. Even Haskell allows procedural-style description through IO monads, and even Java recommends functional style via the Stream API and Optional.

This fusion is not "the disappearance of paradigms" but rather **"the internalization of paradigms."**
Each paradigm's concepts have been absorbed as language features, and developers now switch between multiple paradigms unconsciously while writing code.

That is precisely why clearly understanding the essence of each paradigm has become more important than ever.

---

## 3. Procedural Programming

### 3.1 Core Philosophy

```
============================================================
 The Essence of Procedural Programming
============================================================

 Philosophy: "Describe processing as step-by-step instructions
              from top to bottom"

 Computational Model: Turing Machine (state transition machine)

 Features:
   - Sequential execution of instructions (sequence of statements)
   - Assignment to variables (state mutation)
   - Control structures (if, for, while, switch)
   - Division by procedures (functions/subroutines)
   - Namespace management via scope

 Central abstraction: Procedure (procedure / function)
============================================================
```

Procedural programming is the most intuitive and historically oldest paradigm.
It describes a program as "a series of instructions to the computer."
It mirrors the same thinking as writing a cooking recipe or assembly instructions:
"first do this, then do that, and depending on conditions, do this instead."

### 3.2 Core Concepts in Detail

**Sequential Execution**

In procedural programming, statements are executed from top to bottom in order.
This is the most fundamental control flow and the foundation of all other paradigms.

**State Mutation**

Values are assigned to variables, and those values can be changed later. This corresponds to writing to the tape of a Turing machine. The essence of procedural programming is "sequential mutation of state."

**Control Structures**

Conditional branching (if/else, switch) and iteration (for, while, do-while) control the flow of execution. Dijkstra's structured program theorem proves that any program can be described using only three structures: sequence, selection, and iteration.

**Procedure Extraction**

Repeatedly occurring processing patterns are extracted as procedures (functions/subroutines), given names, and reused. This is the most fundamental abstraction technique.

### 3.3 Code Example: Procedural Programming in C

```c
/* C: Typical example of procedural programming */
/* Read data from a file and compute statistics */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_SIZE 1000

/* --- Data reading procedure --- */
int read_data(const char *filename, double data[], int max_size) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return -1;
    }

    int count = 0;
    while (count < max_size && fscanf(fp, "%lf", &data[count]) == 1) {
        count++;  /* State mutation: increment counter */
    }

    fclose(fp);
    return count;
}

/* --- Procedure to calculate mean --- */
double calculate_mean(const double data[], int n) {
    double sum = 0.0;          /* State initialization */
    for (int i = 0; i < n; i++) {
        sum += data[i];        /* State mutation: accumulate sum */
    }
    return sum / n;
}

/* --- Procedure to calculate standard deviation --- */
double calculate_stddev(const double data[], int n, double mean) {
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff; /* State mutation: accumulate sum of squares */
    }
    return sqrt(sum_sq / n);
}

/* --- Sort procedure (bubble sort) --- */
void sort_ascending(double data[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (data[j] > data[j + 1]) {
                /* State mutation: swap elements */
                double temp = data[j];
                data[j] = data[j + 1];
                data[j + 1] = temp;
            }
        }
    }
}

/* --- Main procedure --- */
int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <datafile>\n", argv[0]);
        return 1;
    }

    double data[MAX_SIZE];
    /* Step 1: Read data */
    int n = read_data(argv[1], data, MAX_SIZE);
    if (n <= 0) return 1;

    /* Step 2: Compute statistics */
    double mean = calculate_mean(data, n);
    double stddev = calculate_stddev(data, n, mean);

    /* Step 3: Sort and find median */
    sort_ascending(data, n);
    double median = (n % 2 == 0)
        ? (data[n/2 - 1] + data[n/2]) / 2.0
        : data[n/2];

    /* Step 4: Output results */
    printf("Count:  %d\n", n);
    printf("Mean:   %.4f\n", mean);
    printf("Median: %.4f\n", median);
    printf("StdDev: %.4f\n", stddev);
    printf("Min:    %.4f\n", data[0]);
    printf("Max:    %.4f\n", data[n-1]);

    return 0;
}
```

This code exhibits all characteristics of procedural programming:
- Sequential execution: Steps 1->2->3->4 in the main function
- State mutation: Writing to sum, sum_sq, data[]
- Control structures: for loops, if conditional branches
- Procedure extraction: read_data, calculate_mean, sort_ascending

### 3.4 Strengths and Limitations of Procedural Programming

```
+-----------------------------------------------+
|     Strengths of Procedural Programming        |
+-----------------------------------------------+
| [1] Intuitive: Sequential description close    |
|     to human thought                           |
| [2] High efficiency: Low-level control close   |
|     to hardware                                |
| [3] Predictable: Easy to debug via step        |
|     execution                                  |
| [4] Low overhead: Thin abstraction layer       |
| [5] Wide applicability: Applicable to almost   |
|     all domains                                |
+-----------------------------------------------+

+-----------------------------------------------+
|     Limitations of Procedural Programming      |
+-----------------------------------------------+
| [1] Scalability: Global state becomes complex  |
|     at scale, making it hard to understand     |
|     and maintain                               |
| [2] Reusability: Data and processing are       |
|     separate, making code reuse difficult      |
| [3] Concurrency: Shared mutable state breeds   |
|     race conditions                            |
| [4] Testing: Functions with side effects are   |
|     cumbersome to test                         |
| [5] Abstraction: Limited means for expressing  |
|     advanced abstractions                      |
+-----------------------------------------------+
```

### 3.5 Where Procedural Programming Shines

- **Systems programming**: OS kernels, device drivers, bootloaders
- **Embedded systems**: Environments with limited memory and CPU resources
- **Scripting**: Shell scripts, batch processing, automation
- **Numerical computing**: Scientific computing, simulations
- **Prototyping**: The stage of quickly turning ideas into working code

---

## 4. Object-Oriented Programming (OOP)

### 4.1 Core Philosophy

```
============================================================
 The Essence of OOP
============================================================

 Philosophy: "Combine data and behavior as objects, and describe
             the system as interactions between objects"

 Origin: Simula (1962), Smalltalk (1972)

 Four Pillars:
   1. Encapsulation
      -- Hide internal state and expose only a public interface
   2. Inheritance
      -- Inherit and extend functionality from existing classes
   3. Polymorphism
      -- Invoke different implementations through the same interface
   4. Abstraction
      -- Extract essential features and hide unnecessary details

 Central abstraction: Object (state + behavior + identity)
============================================================
```

OOP originated with Simula in the 1960s, was refined into its pure form by Smalltalk in the 1970s, and became the industry standard paradigm through Java and C++ in the 1990s. Its essence lies in the idea of "modeling real-world entities as software objects."

### 4.2 The Four Pillars Explained

#### 4.2.1 Encapsulation

Encapsulation, also called "Information Hiding," is based on the concept proposed by David Parnas in his 1972 paper "On the Criteria To Be Used in Decomposing Systems into Modules."

It hides an object's internal state (fields) from direct external access and allows manipulation only through public methods (interface). This enables internal implementation changes without affecting external code.

#### 4.2.2 Inheritance

A mechanism where a new class (child/subclass) inherits the attributes and methods of an existing class (parent/superclass). It enables code reuse, but excessive use causes the "Fragile Base Class Problem."

Modern best practice recommends **composition over inheritance** ("Favor composition over inheritance" -- GoF, 1994).

#### 4.2.3 Polymorphism

A mechanism for achieving different behaviors through the same interface or method name. Types of polymorphism include:

- **Subtype polymorphism**: Through inheritance/interface implementation (most common)
- **Parametric polymorphism**: Through generics/templates
- **Ad hoc polymorphism**: Through method overloading

#### 4.2.4 Abstraction

Extracting only the essential features of the problem domain and hiding implementation details. Abstract classes and interfaces define "what can be done," while delegating "how to do it" to concrete implementation classes.

### 4.3 Code Example: OOP in Python

```python
"""
Python: Comprehensive OOP Example
-- Shape Calculation System --
"""
from abc import ABC, abstractmethod
from math import pi, sqrt
from typing import Protocol, runtime_checkable


# ---- Abstraction: Defining a common interface ----

class Shape(ABC):
    """Abstract base class for all shapes"""

    @abstractmethod
    def area(self) -> float:
        """Calculate area"""
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """Calculate perimeter"""
        pass

    def describe(self) -> str:
        """Return shape info as a string (Template Method pattern)"""
        return (
            f"{self.__class__.__name__}: "
            f"area={self.area():.2f}, "
            f"perimeter={self.perimeter():.2f}"
        )


# ---- Protocol: Structural subtyping (type-safe version of Duck Typing) ----

@runtime_checkable
class Drawable(Protocol):
    def draw(self, canvas: str) -> None: ...


# ---- Encapsulation: Hiding internal state ----

class Circle(Shape):
    """Circle: Encapsulation guarantees the invariant on radius"""

    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("radius must be positive")
        self._radius = radius  # Private attribute

    @property
    def radius(self) -> float:
        """Read-only property"""
        return self._radius

    def area(self) -> float:
        return pi * self._radius ** 2

    def perimeter(self) -> float:
        return 2 * pi * self._radius

    def draw(self, canvas: str) -> None:
        print(f"Drawing circle (r={self._radius}) on {canvas}")


class Rectangle(Shape):
    """Rectangle"""

    def __init__(self, width: float, height: float):
        if width <= 0 or height <= 0:
            raise ValueError("dimensions must be positive")
        self._width = width
        self._height = height

    def area(self) -> float:
        return self._width * self._height

    def perimeter(self) -> float:
        return 2 * (self._width + self._height)

    def draw(self, canvas: str) -> None:
        print(f"Drawing rectangle ({self._width}x{self._height}) on {canvas}")


# ---- Inheritance: Extending existing classes ----

class Square(Rectangle):
    """Square: Specialization of Rectangle (is-a relationship)"""

    def __init__(self, side: float):
        super().__init__(side, side)  # Reuse parent class constructor


class Triangle(Shape):
    """Triangle: Defined by three side lengths"""

    def __init__(self, a: float, b: float, c: float):
        if not self._is_valid(a, b, c):
            raise ValueError("Invalid triangle sides")
        self._a, self._b, self._c = a, b, c

    @staticmethod
    def _is_valid(a: float, b: float, c: float) -> bool:
        """Triangle inequality validation"""
        return a + b > c and b + c > a and a + c > b

    def area(self) -> float:
        """Heron's formula"""
        s = self.perimeter() / 2
        return sqrt(s * (s - self._a) * (s - self._b) * (s - self._c))

    def perimeter(self) -> float:
        return self._a + self._b + self._c


# ---- Polymorphism: Operating through a unified interface ----

def print_all_shapes(shapes: list[Shape]) -> None:
    """Any shape can be operated on with the same method"""
    for shape in shapes:
        print(shape.describe())


def draw_all(drawables: list[Drawable], canvas: str) -> None:
    """Draw objects that satisfy the Drawable protocol"""
    for d in drawables:
        d.draw(canvas)


# ---- Composition: Prefer composition over inheritance ----

class ShapeCollection:
    """Collection of shapes (example of composition)"""

    def __init__(self):
        self._shapes: list[Shape] = []

    def add(self, shape: Shape) -> None:
        self._shapes.append(shape)

    def total_area(self) -> float:
        return sum(s.area() for s in self._shapes)

    def largest(self) -> Shape:
        return max(self._shapes, key=lambda s: s.area())


# ---- Usage Example ----

if __name__ == "__main__":
    shapes = [
        Circle(5),
        Rectangle(3, 4),
        Square(6),
        Triangle(3, 4, 5),
    ]

    # Polymorphism: Process all shapes uniformly
    print_all_shapes(shapes)
    # => Circle: area=78.54, perimeter=31.42
    # => Rectangle: area=12.00, perimeter=14.00
    # => Square: area=36.00, perimeter=24.00
    # => Triangle: area=6.00, perimeter=12.00

    # Composition
    collection = ShapeCollection()
    for s in shapes:
        collection.add(s)
    print(f"\nTotal area: {collection.total_area():.2f}")
    print(f"Largest: {collection.largest().describe()}")
```

### 4.4 OOP Design Principles -- SOLID

SOLID principles, proposed by Robert C. Martin, are widely known as design principles for effectively leveraging OOP.

```
+---+-----------------------------------+----------------------------------+
| S | Single Responsibility Principle    | A class should have only one     |
|   |                                   | reason to change                 |
+---+-----------------------------------+----------------------------------+
| O | Open-Closed Principle             | Open for extension, closed for   |
|   |                                   | modification. Add features       |
|   |                                   | without changing existing code   |
+---+-----------------------------------+----------------------------------+
| L | Liskov Substitution Principle     | Subclasses should be             |
|   |                                   | substitutable for their parent   |
|   |                                   | classes                          |
+---+-----------------------------------+----------------------------------+
| I | Interface Segregation Principle   | Should not force dependence on   |
|   |                                   | methods that are not used        |
+---+-----------------------------------+----------------------------------+
| D | Dependency Inversion Principle    | Depend on abstractions, not      |
|   |                                   | concretions. Higher modules      |
|   |                                   | should not depend on lower ones  |
+---+-----------------------------------+----------------------------------+
```

### 4.5 Class-Based vs Prototype-Based

There are two implementation approaches to OOP.

```
Class-based OOP (Java, C#, Python, C++)
=============================================
  Class (blueprint) -> Instance (entity)

  class Dog {
      String name;
      void bark() { ... }
  }
  Dog fido = new Dog("Fido");

  Features:
  - Classes define types; instances conform to them
  - Static type hierarchy
  - Compile-time type checking

Prototype-based OOP (JavaScript)
=============================================
  Prototype (archetype) -> Clone (copy)

  const dog = {
      name: "Fido",
      bark() { console.log("Woof!"); }
  };
  const puppy = Object.create(dog);
  puppy.name = "Rex";

  Features:
  - Objects directly inherit from other objects
  - Dynamic delegation chain
  - Flexible modification at runtime
```

### 4.6 Strengths and Limitations of OOP

```
+-----------------------------------------------+
|          Strengths of OOP                      |
+-----------------------------------------------+
| [1] Modeling: Directly represent real-world    |
|     concepts                                   |
| [2] Scalability: Suitable for division of      |
|     labor in large-scale development           |
| [3] Reusability: Reuse through inheritance     |
|     and composition                            |
| [4] Maintainability: Encapsulation limits the  |
|     scope of change impact                     |
| [5] Ecosystem: Rich design patterns            |
+-----------------------------------------------+

+-----------------------------------------------+
|          Limitations of OOP                    |
+-----------------------------------------------+
| [1] Over-engineering: Class explosion, deep    |
|     inheritance hierarchies                    |
| [2] Concurrency: Shared mutable state breeds   |
|     race conditions                            |
| [3] Expressiveness: Forcing "verbs (behavior)" |
|     into "nouns (objects)"                     |
| [4] Boilerplate: Proliferation of              |
|     getter/setter/constructor code             |
| [5] Testing: Tends to require extensive mocking |
+-----------------------------------------------+
```

### 4.7 Where OOP Shines

- **GUI applications**: Widget hierarchies, event handling
- **Game development**: Entity-component systems, scene graphs
- **Enterprise systems**: Domain modeling, business logic
- **Framework design**: Plugin mechanisms, template methods
- **API design**: Resource modeling, versioning

---

## 5. Functional Programming

### 5.1 Core Philosophy

```
============================================================
 The Essence of Functional Programming
============================================================

 Philosophy: "Describe computation as application and
             composition of mathematical functions"

 Origin: Lambda Calculus (Church, 1936), LISP (McCarthy, 1958)

 Principles:
   1. Pure Functions
      -- Same input always produces same output, no side effects
   2. Immutability
      -- Don't modify data; generate new data
   3. First-class Functions
      -- Functions can be assigned to variables, passed as
         arguments, returned as values
   4. Referential Transparency
      -- An expression can be replaced by its value without
         changing program meaning
   5. Higher-order Functions
      -- Functions that take functions as arguments or return
         functions

 Central abstraction: Function (mapping from input to output)
============================================================
```

Functional programming (FP) is a paradigm that directly brings the mathematical concept of functions into programming. While imperative programming focuses on "how to change state," FP focuses on "how to transform data."

This shift in perspective is not merely a difference in style.
**By assuming immutability, reasoning about programs becomes dramatically easier, concurrent processing becomes safe, and tests become easier to write** -- these are practical benefits.

### 5.2 Core Concepts in Detail

#### 5.2.1 Pure Functions and Side Effects

A pure function satisfies the following two properties:

1. **Determinism**: Always returns the same value for the same arguments
2. **No side effects**: Does not modify external state (no log output, DB writes, global variable changes, etc.)

```
Pure Functions vs Impure Functions
============================================================

Pure function:
  f(x) = x * 2 + 1
  - f(3) always returns 7
  - Result is the same regardless of how many times called
  - Testing: assert f(3) == 7 is sufficient

Impure function:
  counter = 0
  def increment():
      global counter
      counter += 1       # Side effect: modifies global state
      return counter
  - increment() returns a different value each time called
  - Testing: requires state reset
============================================================
```

#### 5.2.2 Immutability and Data Structures

In functional programming, once data is created, it is not modified.
When "modification" is needed, new data is generated with only the changed parts differing.

This may seem inefficient, but by using **persistent data structures**, most of the structure is shared, achieving efficiency.

```
Updating an Immutable List (Structural Sharing)
============================================================

Original list: [A] -> [B] -> [C] -> [D]

Prepend E:

  New list: [E] -> [A] -> [B] -> [C] -> [D]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
                   Shared as-is from original list

The original list is not modified (no impact on other code referencing it)
============================================================
```

#### 5.2.3 Function Composition and Pipelines

"Function composition," which builds large functions by combining small ones, is FP's most powerful abstraction technique.

### 5.3 Code Examples: Haskell and JavaScript

```haskell
-- Haskell: Purely Functional Programming
-- ==========================================

-- Pure function: Factorial (recursion + pattern matching)
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Higher-order function: Takes a function as argument
applyTwice :: (a -> a) -> a -> a
applyTwice f x = f (f x)
-- applyTwice (+3) 10  =>  16
-- applyTwice (*2) 3   =>  12

-- Function composition: (.) operator
-- process = sum . map (*2) . filter (>0)
-- Right to left: filter -> map -> sum
process :: [Int] -> Int
process = sum . map (*2) . filter (>0)
-- process [1, -2, 3, -4, 5] => 18

-- List comprehension
pythagoreanTriples :: Int -> [(Int, Int, Int)]
pythagoreanTriples n =
    [(a, b, c) | c <- [1..n],
                 b <- [1..c],
                 a <- [1..b],
                 a*a + b*b == c*c]

-- Algebraic data types + pattern matching
data Tree a = Leaf | Node (Tree a) a (Tree a)

treeSum :: Tree Int -> Int
treeSum Leaf         = 0
treeSum (Node l v r) = treeSum l + v + treeSum r

-- Maybe monad: Safe handling of null
safeDivide :: Double -> Double -> Maybe Double
safeDivide _ 0 = Nothing
safeDivide x y = Just (x / y)

-- Monadic chaining
calculate :: Double -> Double -> Double -> Maybe Double
calculate a b c = do
    x <- safeDivide a b    -- If b is 0, Nothing propagates
    y <- safeDivide x c    -- If c is 0, Nothing propagates
    return (x + y)
```

```javascript
// JavaScript: Functional Style
// ==========================================

// --- Pure functions ---
const add = (a, b) => a + b;
const multiply = (a, b) => a * b;
const isPositive = n => n > 0;
const double = n => n * 2;

// --- Higher-order functions ---
const applyTwice = (fn, value) => fn(fn(value));
// applyTwice(double, 3) => 12

// --- Function composition (right to left) ---
const compose = (...fns) => x =>
    fns.reduceRight((acc, fn) => fn(acc), x);

// --- Pipeline (left to right) ---
const pipe = (...fns) => x =>
    fns.reduce((acc, fn) => fn(acc), x);

// --- Immutable data operations ---
const numbers = [1, -2, 3, -4, 5];

// Imperative (mutates state)
let result = 0;
for (const n of numbers) {
    if (n > 0) result += n * 2;
}

// Functional (transforms data)
const result2 = numbers
    .filter(isPositive)
    .map(double)
    .reduce(add, 0);
// => 18

// --- Currying ---
const curry = fn => {
    const arity = fn.length;
    return function curried(...args) {
        if (args.length >= arity) return fn(...args);
        return (...moreArgs) => curried(...args, ...moreArgs);
    };
};

const curriedAdd = curry(add);
const add5 = curriedAdd(5);
// add5(3) => 8

// --- Immutable object operations ---
const user = { name: "Alice", age: 30, active: true };

// Bad example: Mutation
// user.age = 31;  // Modifies the original object

// Good example: Generate a new object
const updatedUser = { ...user, age: 31 };
// user is not modified
// updatedUser is { name: "Alice", age: 31, active: true }

// --- Recursion as an alternative to loops ---
const sumArray = (arr) => {
    if (arr.length === 0) return 0;
    const [head, ...tail] = arr;
    return head + sumArray(tail);
};
```

### 5.4 Important Functional Patterns

```
Important Functional Programming Patterns
============================================================

1. Map-Filter-Reduce Pipeline
   Data list -> filter(condition) -> map(transform) -> reduce(aggregate)

   [1,-2,3,-4,5]
       |
       v  filter(x > 0)
   [1, 3, 5]
       |
       v  map(x * 2)
   [2, 6, 10]
       |
       v  reduce(+, 0)
   18

2. Maybe / Option Pattern
   Express "no value" via types instead of using null

   safeDivide(10, 0) => Nothing
   safeDivide(10, 2) => Just(5)

3. Either / Result Pattern
   Express errors via types instead of using exceptions

   parseAge("25")     => Right(25)
   parseAge("abc")    => Left("Invalid number")

4. Pattern Matching
   Branch processing based on data structure

   match shape {
       Circle(r)        => pi * r * r,
       Rectangle(w, h)  => w * h,
       Triangle(b, h)   => b * h / 2,
   }
============================================================
```

### 5.5 Strengths and Limitations of Functional Programming

```
+--------------------------------------------------+
|     Strengths of Functional Programming           |
+--------------------------------------------------+
| [1] Testability: Pure functions verify by I/O only|
| [2] Concurrency safety: Immutable data = no races |
| [3] Composability: Build large features from small|
|     function compositions                         |
| [4] Reasoning: Referential transparency enables   |
|     equational reasoning                          |
| [5] Refactoring: Function replacement is safe     |
+--------------------------------------------------+

+--------------------------------------------------+
|     Limitations of Functional Programming         |
+--------------------------------------------------+
| [1] Learning curve: Abstract concepts like monads |
|     and type classes                              |
| [2] Performance prediction: Understanding lazy    |
|     evaluation behavior                           |
| [3] Side effect handling: IO is inherently complex|
| [4] Debugging: Tracing composed function chains   |
| [5] Existing ecosystem: Integration with          |
|     OOP-based libraries                           |
+--------------------------------------------------+
```

### 5.6 Where Functional Programming Shines

- **Data pipelines**: ETL processing, log analysis, data transformation
- **Concurrent/distributed processing**: MapReduce, Spark, stream processing
- **Compilers/interpreters**: Parsing, AST transformation, code generation
- **Financial systems**: Transaction processing, risk calculation (correctness is top priority)
- **Configuration/DSLs**: Declarative configuration, domain-specific languages
- **Frontend**: React (UI = f(state)), Redux

---

## 6. Logic Programming

### 6.1 Core Philosophy

```
============================================================
 The Essence of Logic Programming
============================================================

 Philosophy: "Describe programs as logical propositions and rules,
             and derive solutions through logical inference"

 Origin: First-order predicate logic, Prolog (Colmerauer, 1972)

 Principles:
   1. Facts -- Declare propositions that are true
   2. Rules -- Conditions for deriving new facts from existing facts
   3. Queries -- Request the inference engine to search for solutions

 Central abstraction: Logical relationships and Unification
============================================================
```

Logic programming is fundamentally different from other paradigms.
While imperative describes "how to compute" and functional describes "what to transform," logic programming describes **"what is true."**

### 6.2 Code Example: Prolog

```prolog
% ==========================================
% Prolog: Family Relationship Inference
% ==========================================

% --- Facts: Declare propositions that are true ---
parent(tom, bob).       % tom is a parent of bob
parent(tom, liz).       % tom is a parent of liz
parent(bob, ann).       % bob is a parent of ann
parent(bob, pat).       % bob is a parent of pat
parent(pat, jim).       % pat is a parent of jim

male(tom).
male(bob).
male(jim).
female(liz).
female(ann).
female(pat).

% --- Rules: Derive new facts from existing facts ---
father(X, Y) :- parent(X, Y), male(X).
mother(X, Y) :- parent(X, Y), female(X).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
grandfather(X, Z) :- grandparent(X, Z), male(X).

sibling(X, Y) :- parent(Z, X), parent(Z, Y), X \= Y.

% Transitive ancestor relationship (recursive rule)
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

% --- Queries: Ask the inference engine ---
% ?- grandparent(tom, ann).
% => true (tom -> bob -> ann)
%
% ?- grandfather(tom, X).
% => X = ann ; X = pat (enumerate tom's grandchildren)
%
% ?- ancestor(tom, jim).
% => true (tom -> bob -> pat -> jim)
%
% ?- sibling(ann, pat).
% => true (share the same parent bob)
```

```prolog
% ==========================================
% Prolog: List Operations
% ==========================================

% List length
list_length([], 0).
list_length([_|T], N) :- list_length(T, N1), N is N1 + 1.

% List concatenation
append([], L, L).
append([H|T1], L2, [H|T3]) :- append(T1, L2, T3).

% List reversal
reverse([], []).
reverse([H|T], R) :- reverse(T, R1), append(R1, [H], R).

% List sorting (insertion sort)
insert_sorted(X, [], [X]).
insert_sorted(X, [H|T], [X,H|T]) :- X =< H.
insert_sorted(X, [H|T], [H|R]) :- X > H, insert_sorted(X, T, R).

insertion_sort([], []).
insertion_sort([H|T], Sorted) :-
    insertion_sort(T, SortedTail),
    insert_sorted(H, SortedTail, Sorted).
```

### 6.3 Application Domains of Logic Programming

- **Artificial intelligence**: Expert systems, knowledge representation, natural language processing
- **Databases**: Datalog, SQL (declarative queries are influenced by logic programming)
- **Type checking**: Type inference algorithms are based on unification
- **Theorem proving**: Coq, Isabelle (extensions of logic programming)
- **Configuration management**: Ansible, Terraform (declarative state definitions)

---

## 7. Reactive Programming and the Actor Model

### 7.1 Reactive Programming

```
============================================================
 The Essence of Reactive Programming
============================================================

 Philosophy: "Declaratively process asynchronous events as
             streams that automatically propagate data changes"

 Origin: Functional Reactive Programming (Elliott & Hudak, 1997)

 Principles:
   1. Observable (data stream)
   2. Operators (stream transformations)
   3. Subscription (consuming results)
   4. Backpressure (flow control)

 Central abstraction: Data stream (sequence of values over time)
============================================================
```

```
Conceptual Diagram of Reactive Streams
============================================================

Time ->
               click    click         click
User input:    --[A]------[B]----------[C]------>

                 |         |             |
                 v         v             v
debounce(300ms): --------[A]-----------[B]---[C]->

                          |              |     |
                          v              v     v
filter(len>=2):  --------[AB]-----------[BC]------>

                          |              |
                          v              v
switchMap(API):  --------[result1]------[result2]--->

============================================================
```

```javascript
// RxJS: Reactive Programming Example
// ==========================================

import { fromEvent, interval, merge } from 'rxjs';
import {
    debounceTime, map, filter, switchMap,
    scan, takeUntil, distinctUntilChanged,
    catchError, retry
} from 'rxjs/operators';
import { of } from 'rxjs';

// --- Search Autocomplete ---
const searchInput = document.getElementById('search');

const search$ = fromEvent(searchInput, 'input').pipe(
    debounceTime(300),                          // Wait for 300ms of silence
    map(event => event.target.value.trim()),    // Extract value
    filter(query => query.length >= 2),         // Only 2+ characters
    distinctUntilChanged(),                     // Skip if no change
    switchMap(query =>                          // Only latest request is valid
        fetch(`/api/search?q=${encodeURIComponent(query)}`)
            .then(res => res.json())
    ),
    catchError(err => {                         // Error handling
        console.error('Search error:', err);
        return of([]);
    })
);

search$.subscribe(results => {
    renderResults(results);
});

// --- Counter: Composing multiple streams ---
const increment$ = fromEvent(document.getElementById('inc'), 'click')
    .pipe(map(() => +1));
const decrement$ = fromEvent(document.getElementById('dec'), 'click')
    .pipe(map(() => -1));

const counter$ = merge(increment$, decrement$).pipe(
    scan((count, delta) => count + delta, 0)
);

counter$.subscribe(count => {
    document.getElementById('display').textContent = count;
});
```

### 7.2 The Actor Model

```
============================================================
 The Essence of the Actor Model
============================================================

 Philosophy: "Describe concurrent computation as a collection
             of independent actors (lightweight processes)
             that communicate via message passing"

 Origin: Carl Hewitt (1973), Erlang (1986)

 Principles:
   1. Actor = state + message queue + behavior
   2. Communication is via message passing only (no shared memory)
   3. Each actor cannot directly access another actor's internal state
   4. Actors can create new actors
   5. Actors can change behavior in response to received messages

 Central abstraction: Actor (independent computational entity)
============================================================
```

```
Actor Model Architecture
============================================================

  +-----------------------------------------------------+
  |               Supervisor                             |
  |  +----------+  +----------+  +----------+           |
  |  | Actor A  |  | Actor B  |  | Actor C  |           |
  |  |          |  |          |  |          |           |
  |  | [State]  |  | [State]  |  | [State]  |           |
  |  | [Mail-   |  | [Mail-   |  | [Mail-   |           |
  |  |  box]    |  |  box]    |  |  box]    |           |
  |  +----+-----+  +-----+----+  +----+-----+           |
  |       |   Message     |            |                  |
  |       |<--------------+            |                  |
  |       |                            |                  |
  |       +--------Message------------>|                  |
  |                                                       |
  |  Failure strategies:                                  |
  |    one_for_one  -- Restart only the failed actor      |
  |    one_for_all  -- Restart all child actors           |
  |    rest_for_one -- Restart actors after the failed one|
  +-----------------------------------------------------+

  "Let it crash" philosophy:
    Design for recovery from errors, not prevention of errors
============================================================
```

```elixir
# Elixir: Actor Model (GenServer) Example
# ==========================================

defmodule Counter do
  use GenServer

  # --- Client API ---

  def start_link(initial_value \\ 0) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment do
    GenServer.cast(__MODULE__, :increment)
  end

  def decrement do
    GenServer.cast(__MODULE__, :decrement)
  end

  def get_value do
    GenServer.call(__MODULE__, :get_value)
  end

  # --- Server Callbacks ---

  @impl true
  def init(initial_value) do
    {:ok, initial_value}
  end

  @impl true
  def handle_cast(:increment, count) do
    {:noreply, count + 1}     # Update and return state
  end

  @impl true
  def handle_cast(:decrement, count) do
    {:noreply, count - 1}
  end

  @impl true
  def handle_call(:get_value, _from, count) do
    {:reply, count, count}    # Return the current value
  end
end

# Usage:
# {:ok, _pid} = Counter.start_link(0)
# Counter.increment()
# Counter.increment()
# Counter.increment()
# Counter.decrement()
# Counter.get_value()  # => 2
```

### 7.3 Application Domains of Reactive/Actor Models

| Model | Application Domain | Representative Technologies |
|-------|-------------------|---------------------------|
| Reactive | UI/UX, real-time search, dashboards | RxJS, RxJava, Reactor |
| Actor | Distributed systems, IoT, telecom, chat | Erlang/OTP, Akka, Orleans |
| CQRS/ES | Event-driven architecture, audit logs | EventStoreDB, Axon |

---

## 8. Multi-Paradigm -- The Modern Mainstream

### 8.1 Why Multi-Paradigm Became Mainstream

From the 2000s onward, the following changes accelerated the shift to multi-paradigm:

1. **Proliferation of multi-core**: The need for concurrent processing increased, making functional immutability important
2. **Distributed systems**: The penetration of microservices made mixing different paradigms natural
3. **Developer experience**: Constraints of a single paradigm increasingly hindered productivity
4. **Maturity of language design**: Techniques for integrating multiple paradigms without contradiction were established

### 8.2 Paradigm Mapping of Major Languages

```
Paradigm Support Table for Major Languages
============================================================

Language      Procedural  OOP   FP    Concurrent  Generics  Other
-------------------------------------------------------------------
Python          *         *     ~       ~           ~       Dynamic typing
JavaScript      *         ~     *       ~           -       Prototype
TypeScript      *         *     *       ~           *       Structural typing
Java            *         *     ~       ~           *       -
C#              *         *     ~       ~           *       LINQ
Kotlin          *         *     *       *           *       Coroutines
Swift           *         *     *       *           *       Protocol-oriented
Rust            *         ~     *       *           *       Ownership
Go              *         ~     ~       *           *       goroutine
Scala           *         *     *       *           *       Implicit conversions
Haskell         -         -     *       ~           *       Lazy evaluation
Elixir          -         -     *       *           -       Actor
-------------------------------------------------------------------
* = First-class support  ~ = Partial support  - = Not supported/minimal
```

### 8.3 Code Example: TypeScript Multi-Paradigm

```typescript
// TypeScript: Practical Multi-Paradigm Example
// ==========================================

// ---- Type definition (algebraic data type approach) ----

type Result<T, E = Error> =
    | { ok: true; value: T }
    | { ok: false; error: E };

interface User {
    readonly id: string;
    readonly name: string;
    readonly email: string;
    readonly role: 'admin' | 'user' | 'guest';
    readonly createdAt: Date;
}

// ---- Repository interface (OOP: Abstraction) ----

interface UserRepository {
    findById(id: string): Promise<Result<User, string>>;
    findAll(): Promise<User[]>;
    save(user: User): Promise<Result<void, string>>;
}

// ---- Functional: Pure domain logic ----

// Validation functions (pure functions)
const validateEmail = (email: string): Result<string, string> => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email)
        ? { ok: true, value: email }
        : { ok: false, error: `Invalid email: ${email}` };
};

const validateName = (name: string): Result<string, string> =>
    name.length >= 2 && name.length <= 50
        ? { ok: true, value: name.trim() }
        : { ok: false, error: `Name must be 2-50 chars: ${name}` };

// Function composition: Validation pipeline
const pipe = <T>(...fns: Array<(arg: T) => T>) =>
    (value: T): T => fns.reduce((acc, fn) => fn(acc), value);

// Filtering (functional: higher-order functions)
const filterUsers = (predicate: (u: User) => boolean) =>
    (users: User[]): User[] => users.filter(predicate);

const isAdmin = (u: User): boolean => u.role === 'admin';
const isActive = (u: User): boolean =>
    u.createdAt > new Date(Date.now() - 90 * 24 * 60 * 60 * 1000);

const getActiveAdmins = (users: User[]): User[] =>
    pipe(
        filterUsers(isAdmin),
        filterUsers(isActive),
    )(users);

// ---- OOP: Service layer (state and dependency management) ----

class UserService {
    constructor(
        private readonly repo: UserRepository,
        private readonly logger: { info: (msg: string) => void }
    ) {}

    async createUser(
        name: string,
        email: string,
        role: User['role'] = 'user'
    ): Promise<Result<User, string>> {
        // Functional: Validation
        const nameResult = validateName(name);
        if (!nameResult.ok) return nameResult;

        const emailResult = validateEmail(email);
        if (!emailResult.ok) return emailResult;

        // OOP: Object creation
        const user: User = {
            id: crypto.randomUUID(),
            name: nameResult.value,
            email: emailResult.value,
            role,
            createdAt: new Date(),
        };

        // Procedural: Sequential execution of side effects
        const saveResult = await this.repo.save(user);
        if (!saveResult.ok) return saveResult;

        this.logger.info(`User created: ${user.id}`);
        return { ok: true, value: user };
    }

    async getActiveAdmins(): Promise<User[]> {
        const users = await this.repo.findAll();
        // Functional: Data transformation pipeline
        return getActiveAdmins(users);
    }
}

// ---- Key takeaways for when to use which ----
// Domain logic (validation, transformation)     -> Functional (pure functions)
// Dependency management (repository, logger)    -> OOP (dependency injection)
// Side effect execution (DB save, log output)   -> Procedural (sequential execution)
```

### 8.4 Best Practices for Paradigm Mixing

```
Guidelines for Multi-Paradigm Design
============================================================

Layer               Recommended Paradigm     Reason
----------------------------------------------------------
Presentation        Reactive/Declarative     UI state management
Application         OOP (DI + services)      Dependency management
Domain Logic        Functional (pure funcs)  Testability, reasoning
Infrastructure      Procedural               Explicit side effect mgmt
Data Access         OOP (Repository)         Abstraction and encapsulation
Utilities           Functional               Composability, reusability
----------------------------------------------------------

Principles:
  [1] Maximize the pure part; minimize the side-effect part
  [2] Push side effects to the edges
      = Functional Core, Imperative Shell pattern
  [3] Data transformation in functional; state management in OOP
  [4] Concurrent processing via actor model or functional
============================================================
```

### 8.5 Functional Core, Imperative Shell Pattern

This pattern, advocated by Gary Bernhardt, embodies the most important principle of multi-paradigm design.

```
Functional Core, Imperative Shell
============================================================

  +---------------------------------------------+
  |              Imperative Shell                |
  |    (Outer shell with side effects:           |
  |     DB, API, files)                          |
  |                                              |
  |    +-----------------------------+           |
  |    |     Functional Core         |           |
  |    | (Business logic via pure    |           |
  |    |  functions)                 |           |
  |    |                             |           |
  |    | input -> validate ->        |           |
  |    |   transform -> calculate -> |           |
  |    |   format                    |           |
  |    |                             |           |
  |    | Easy to test / Easy to      |           |
  |    | reason about                |           |
  |    | Concurrency-safe /          |           |
  |    | Composable                  |           |
  |    +-----------------------------+           |
  |                                              |
  |    Shell: Read from DB -> Core -> Write to DB|
  +---------------------------------------------+

  Core testing: Simple input/output verification (no mocks needed)
  Shell testing: Integration tests (can be limited)
============================================================
```

---

## 9. Guidelines for Paradigm Selection and Design Decisions

### 9.1 Mapping Problem Domains to Paradigms

Paradigm selection should be an **engineering decision based on the nature of the problem**, not a matter of preference.
The following mapping table provides guidelines for selecting paradigms from problem characteristics.

```
Problem Characteristics and Paradigm Mapping
============================================================

Problem Characteristic             First Choice      Second Choice
----------------------------------------------------------
Managing stateful entities          OOP               Actor
Data transformation/aggregation     Functional         Procedural
Sequential processing with clear    Procedural         OOP
  steps
Async event processing              Reactive           Functional
Rule-based inference                Logic              Functional
High-concurrency/distributed        Actor              Functional
  systems
UI state management                 Reactive           OOP
Low-level system control            Procedural         (none)
Parsing/compilers                   Functional         Logic
Business rules/workflows            OOP (DDD)          Functional
Numerical/scientific computing      Procedural         Functional
Configuration/infrastructure def    Declarative        Logic
============================================================
```

### 9.2 Decision Flowchart

Organizing the thought process for paradigm selection.

```
Paradigm Selection Flowchart
============================================================

Analyze the problem
     |
     +-- What is the primary concern?
     |
     +-> Data transformation/computation is central
     |     |
     |     +- Are side effects minimal? -> YES -> Functional
     |     +- Are side effects abundant? -> YES -> Procedural + functional elements
     |
     +-> Entity state management is central
     |     |
     |     +- Is there concurrent access?  -> YES -> Actor model
     |     +- Is access sequential?        -> YES -> OOP
     |
     +-> Event/stream processing is central
     |     |
     |     +-> Reactive
     |
     +-> Rule/constraint expression is central
     |     |
     |     +-> Logic / Declarative
     |
     +-> Hardware control/performance is top priority
           |
           +-> Procedural

 * In reality, multiple concerns coexist, so multi-paradigm is the default
============================================================
```

### 9.3 Recommended Patterns by Context

**Startup / Prototype Phase**

Speed and simplicity are the top priority. Start simple with a procedural base, incorporating OOP and functional elements as needed. Avoid excessive abstraction and build "something that works" as quickly as possible.

**Mid-size Team Development (5-20 people)**

Combine OOP-based layered architecture with functional domain logic. SOLID principles and the Functional Core / Imperative Shell pattern are effective. Prioritize testability and increase the proportion of pure functions.

**Large-scale Distributed Systems**

Actor model or event-driven architecture at the center. Within each microservice, select the optimal paradigm based on the situation. Design around immutable messaging and eventual consistency.

**Data-intensive Applications**

Functional pipelines (Map-Filter-Reduce) at the center. Frameworks like Spark and Flink provide functional APIs. Confine side effects to the edges of the pipeline.

---

## 10. Anti-Pattern Collection

### 10.1 Anti-Pattern: God Object

**Category**: OOP anti-pattern

**Symptoms**: A single class takes on most of the system's responsibilities, concentrating all data and logic.

```python
# ===== Anti-pattern: God Object =====
# All responsibilities concentrated in one class

class ApplicationManager:
    """What NOT to do: A class that knows everything and does everything"""

    def __init__(self):
        self.users = []
        self.orders = []
        self.products = []
        self.email_config = {}
        self.db_connection = None
        self.cache = {}
        self.logger = None

    def create_user(self, name, email): ...
    def delete_user(self, user_id): ...
    def authenticate_user(self, email, password): ...
    def create_order(self, user_id, items): ...
    def cancel_order(self, order_id): ...
    def calculate_total(self, order_id): ...
    def apply_discount(self, order_id, code): ...
    def add_product(self, name, price): ...
    def update_inventory(self, product_id, qty): ...
    def send_email(self, to, subject, body): ...
    def send_sms(self, to, message): ...
    def generate_report(self, report_type): ...
    def backup_database(self): ...
    def clear_cache(self): ...
    # ... hundreds of lines continue ...
```

**Problems**:
- Violates the Single Responsibility Principle (SRP)
- Extremely difficult to test (need to mock all dependencies)
- Unpredictable scope of change impact
- Frequent conflicts when multiple developers work simultaneously

**Solution**: Separate classes by responsibility.

```python
# ===== Fix: Separation of responsibilities =====

class UserService:
    """Specialized for user management"""
    def __init__(self, repo: UserRepository):
        self._repo = repo

    def create(self, name: str, email: str) -> User: ...
    def authenticate(self, email: str, password: str) -> bool: ...

class OrderService:
    """Specialized for order management"""
    def __init__(self, repo: OrderRepository, pricing: PricingEngine):
        self._repo = repo
        self._pricing = pricing

    def create(self, user_id: str, items: list[OrderItem]) -> Order: ...
    def cancel(self, order_id: str) -> None: ...

class NotificationService:
    """Specialized for notifications"""
    def __init__(self, email_client: EmailClient, sms_client: SmsClient):
        self._email = email_client
        self._sms = sms_client

    def send_email(self, to: str, subject: str, body: str) -> None: ...
    def send_sms(self, to: str, message: str) -> None: ...
```

### 10.2 Anti-Pattern: Callback Hell

**Category**: Procedural/asynchronous programming anti-pattern

**Symptoms**: Deeply nested asynchronous processing destroys code readability and maintainability.

```javascript
// ===== Anti-pattern: Callback Hell =====

function processOrder(userId, callback) {
    getUser(userId, function(err, user) {
        if (err) return callback(err);
        validateUser(user, function(err, validUser) {
            if (err) return callback(err);
            getCart(validUser.id, function(err, cart) {
                if (err) return callback(err);
                calculateTotal(cart, function(err, total) {
                    if (err) return callback(err);
                    applyDiscount(total, user.discountCode, function(err, finalTotal) {
                        if (err) return callback(err);
                        chargePayment(user.paymentMethod, finalTotal, function(err, payment) {
                            if (err) return callback(err);
                            createOrder(user, cart, payment, function(err, order) {
                                if (err) return callback(err);
                                sendConfirmation(user.email, order, function(err) {
                                    if (err) return callback(err);
                                    callback(null, order);
                                    // 8 levels of nesting. Unreadable. Untestable.
                                });
                            });
                        });
                    });
                });
            });
        });
    });
}
```

**Problems**:
- Destroyed readability (ever-growing indentation to the right)
- Duplicated error handling
- Difficulty testing
- Fear of refactoring

**Solution**: Use async/await (procedural-style) or function composition.

```javascript
// ===== Fix 1: async/await (procedural-style) =====

async function processOrder(userId) {
    const user = await getUser(userId);
    const validUser = await validateUser(user);
    const cart = await getCart(validUser.id);
    const total = await calculateTotal(cart);
    const finalTotal = await applyDiscount(total, user.discountCode);
    const payment = await chargePayment(user.paymentMethod, finalTotal);
    const order = await createOrder(user, cart, payment);
    await sendConfirmation(user.email, order);
    return order;
}

// ===== Fix 2: Function composition (functional-style) =====

const processOrder = pipe(
    getUser,
    validateUser,
    enrichWithCart,
    calculateAndDiscount,
    processPayment,
    createAndConfirmOrder
);
```

### 10.3 Anti-Pattern: Inheritance Abuse

**Category**: OOP anti-pattern

**Symptoms**: Attempting to solve all code reuse through inheritance, producing deep inheritance hierarchies and fragile class structures.

```
Inheritance abuse: Excessively deep inheritance hierarchy
============================================================

  BaseEntity
    +-- User
          +-- PremiumUser
                +-- BusinessUser
                      +-- EnterpriseUser
                            +-- EnterpriseAdminUser
                                  +-- SuperAdminUser
                                        +-- ...

  Problems:
  - Changes to BaseEntity propagate to all subclasses
  - Inherits all unnecessary methods from intermediate classes
  - "is-a" relationship is ambiguous (SuperAdminUser "is-a" User?)
  - Initialization of all parent classes required for testing
============================================================
```

**Solution**: Use composition and interfaces.

```python
# ===== Fix: Composition + Interfaces =====

from dataclasses import dataclass
from typing import Protocol

class Billable(Protocol):
    def calculate_bill(self) -> float: ...

class Authenticatable(Protocol):
    def authenticate(self, credentials: str) -> bool: ...

@dataclass
class User:
    id: str
    name: str
    email: str

@dataclass
class Subscription:
    plan: str
    price: float

    def calculate_bill(self) -> float:
        return self.price

@dataclass
class AdminPrivileges:
    level: int
    permissions: list[str]

# Combine via composition
@dataclass
class PremiumUser:
    user: User                      # has-a User
    subscription: Subscription      # has-a Subscription

@dataclass
class AdminUser:
    user: User                      # has-a User
    privileges: AdminPrivileges     # has-a AdminPrivileges
```

### 10.4 Anti-Pattern: Impure Everywhere

**Category**: Functional programming anti-pattern

**Symptoms**: Claiming a functional style while scattering side effects throughout functions.

```python
# ===== Anti-pattern: Side effects scattered everywhere =====

def calculate_price(items):
    """Appears to be a pure function, but is full of side effects"""
    total = 0
    for item in items:
        price = get_price_from_db(item.id)    # Side effect: DB access
        logging.info(f"Price: {price}")       # Side effect: Log output
        if price > 100:
            send_alert(item.id)               # Side effect: Send notification
        total += price * item.quantity
    cache.set(f"total_{hash(items)}", total)  # Side effect: Cache write
    return total
```

**Solution**: Separate pure functions from IO.

```python
# ===== Fix: Separate pure functions from IO =====

# Pure function: Only performs computation
def calculate_total(priced_items: list[tuple[Item, float]]) -> float:
    """No side effects. Easy to test."""
    return sum(price * item.quantity for item, price in priced_items)

def find_expensive_items(
    priced_items: list[tuple[Item, float]],
    threshold: float = 100
) -> list[Item]:
    """No side effects. Extract items matching the condition."""
    return [item for item, price in priced_items if price > threshold]

# IO layer: Side effects are concentrated here
async def process_order(items: list[Item]) -> float:
    prices = await fetch_prices(items)
    priced_items = list(zip(items, prices))

    total = calculate_total(priced_items)               # Pure
    expensive = find_expensive_items(priced_items)       # Pure

    logger.info(f"Total: {total}")
    for item in expensive:
        await send_alert(item.id)
    await cache.set(f"total_{hash(items)}", total)

    return total
```

---

## 11. Exercises (3 Levels)

### 11.1 Basic Exercises (Comprehension Check)

**Exercise 1-1: Paradigm Classification**

Identify which paradigm the following code fragments most strongly exhibit. Explain your reasoning in 1-2 sentences.

```
(a)  result = items.filter(x => x.active).map(x => x.value).reduce((a,b) => a+b, 0)

(b)  for i in range(len(data)):
         data[i] = data[i] * 2

(c)  ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).

(d)  class Cat extends Animal { meow() { ... } }

(e)  fromEvent(input, 'click').pipe(debounceTime(300), switchMap(fetchData))
```

**Exercise 1-2: Concept Mapping**

Map the following concepts to the correct paradigm(s) (multiple selections allowed).

| Concept | Procedural | OOP | Functional | Logic | Reactive |
|---------|-----------|-----|------------|-------|----------|
| Encapsulation | | | | | |
| Referential transparency | | | | | |
| Unification | | | | | |
| Observable | | | | | |
| for loop | | | | | |
| Pattern matching | | | | | |
| Inheritance | | | | | |
| Currying | | | | | |

**Exercise 1-3: Comparative Essay**

Write code solving the same problem (summing only the even numbers in a list) using both procedural and functional styles, then identify 3 essential differences between the two.

### 11.2 Applied Exercises (Design Decisions)

**Exercise 2-1: Paradigm Selection**

For the following system requirements, state the primary paradigm and your reasoning.

1. A real-time matching engine for a stock exchange
2. A CMS backend for a blog platform
3. A weather data collection and analysis pipeline
4. A chat application server

**Exercise 2-2: Refactoring**

Split the following "God Object" into appropriate classes. Clearly define each class's responsibility and diagram the relationships between classes.

```python
class ECommerceSystem:
    def register_user(self, name, email, password): ...
    def login(self, email, password): ...
    def add_product(self, name, price, stock): ...
    def search_products(self, query): ...
    def add_to_cart(self, user_id, product_id, qty): ...
    def checkout(self, user_id): ...
    def process_payment(self, order_id, card_info): ...
    def ship_order(self, order_id): ...
    def send_receipt(self, order_id): ...
    def generate_sales_report(self, start, end): ...
```

**Exercise 2-3: Multi-Paradigm Design**

Implement a "User Registration -> Email Verification -> Profile Creation" flow in TypeScript. Meet the following requirements:

- Implement validation logic as pure functions
- Define user entities as immutable data
- Abstract DB operations with the Repository pattern (OOP)
- Express errors with a Result type (functional)

### 11.3 Advanced Exercises (Research Topics)

**Exercise 3-1: Paradigm History Research**

Choose one of the following topics and write a report of approximately 2000 words.

- The influence of Smalltalk on OOP and the transformation of OOP by Java
- The influence of Haskell's monads on async processing in modern languages
- Erlang/OTP's "Let it crash" philosophy and modern distributed systems design

**Exercise 3-2: Language Design**

Design your own mini language (DSL). Include the following:

- Target domain (what is the language for?)
- Adopted paradigm (why this paradigm?)
- Syntax examples (at least 3 operations)
- Differences from existing general-purpose languages

**Exercise 3-3: Paradigm Limitations Analysis**

Develop an argument addressing the question "Was object-oriented programming a failure?" including the following perspectives:

- Problems OOP solved and new problems it created
- The background of "Composition over Inheritance"
- The evolution of OOP through incorporation of functional elements
- The position of OOP in the 2020s

---

## 12. FAQ -- Frequently Asked Questions

### Q1: I've heard functional programming is difficult. Is it worth learning?

**A**: The value of learning it is very high. Admittedly, fully understanding Haskell's monads and type classes takes time, but the **fundamental concepts** of FP (pure functions, immutability, higher-order functions, pipelines) can be acquired without learning Haskell.

You can immediately practice in everyday languages like JavaScript, Python, TypeScript, Kotlin, and Swift. Specifically, start with the following:

1. Data transformation using `map`, `filter`, `reduce`
2. Creating pure functions without side effects
3. Ensuring immutability with `const` / `readonly`
4. Passing functions as arguments to functions (higher-order functions)

Even just these will significantly improve code quality and testability.

### Q2: Are OOP and functional programming opposed to each other? Which should I learn?

**A**: They are not opposed but rather in a **complementary relationship**.
Modern best practice is to combine both.

- **Data transformation/computation logic** -> Functional (pure functions, pipelines)
- **State management/dependency injection** -> OOP (classes, interfaces)
- **Side effect execution** -> Procedural (sequential processing)

The "Functional Core, Imperative Shell" pattern most clearly embodies this philosophy. Write pure business logic (Core) in functional style, and manage side effects like DB access and API calls (Shell) in OOP/procedural style.

You should learn both, but the typical order is to understand OOP first and then progress to functional. By experiencing OOP's limitations, you can more deeply understand functional's advantages.

### Q3: I've heard Go doesn't have OOP. Is this a problem for large-scale development?

**A**: Go lacks classes and inheritance, but that doesn't mean OOP is impossible. Go achieves the same level of abstraction as OOP through **interfaces and composition**.

```go
// Go: Interface + Composition
type Reader interface {
    Read(p []byte) (n int, err error)
}

type Writer interface {
    Write(p []byte) (n int, err error)
}

// Interface composition
type ReadWriter interface {
    Reader
    Writer
}

// Struct embedding (composition)
type BufferedWriter struct {
    writer Writer    // has-a relationship
    buffer []byte
}
```

Go's design philosophy is "avoid the complexity of inheritance and provide sufficient abstraction through simple composition." Many large-scale projects including Google, Docker, and Kubernetes have been successfully developed in Go, demonstrating that large-scale development is possible without OOP's classes and inheritance.

In fact, some argue that Go's constraints prevent "excessive abstraction" and contribute to maintaining simple, readable codebases.

### Q4: In what situations should reactive programming be used?

**A**: Reactive programming is particularly effective in the following situations:

1. **Real-time UI**: Search autocomplete, live dashboards, form validation (where async verification runs on every input)
2. **Composing multiple streams**: When you need to combine, transform, and filter events from multiple data sources
3. **Backpressure control**: When the producer generates data faster than the consumer can process it

Conversely, for simple request-response processing (like REST API handlers), async/await is sufficient, and there is little benefit to introducing reactive libraries like RxJS. Excessive reactive adoption unnecessarily increases code complexity, so it is important to identify the appropriate scope of application.

### Q5: How does paradigm knowledge help when learning a new language?

**A**: Paradigm knowledge gives you the ability to instantly grasp a "language's skeleton." When encountering a new language, you can classify it with the following checklist:

1. **State management**: Are variables reassignable? Is the default mutable or immutable?
2. **Functions**: Are they first-class? Are higher-order functions supported?
3. **Type system**: Static or dynamic? Are algebraic data types available?
4. **Concurrency**: Which model is used? (threads/goroutines/actors)
5. **Abstraction**: Which are used -- classes/traits/protocols/type classes?

For example, when first seeing Rust, if you can organize it as "procedural + functional with an ownership system, using traits instead of classes, with pattern matching and algebraic data types," you can read code intent before memorizing individual syntax.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory alone, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping ahead to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

### 13.1 Comprehensive Paradigm Comparison Table

| Paradigm | Central Concept | Strength Domain | Representative Languages | Abstraction Unit | State Handling |
|----------|----------------|-----------------|------------------------|-----------------|----------------|
| Procedural | Sequential execution | Scripting, systems | C, Go, Pascal | Procedure/Function | Mutable |
| OOP | Objects | Large-scale dev, GUI | Java, C#, Python | Class/Object | Encapsulated mutable |
| Functional | Pure functions, immutability | Data transformation, concurrency | Haskell, Elixir, Clojure | Function/Type | Immutable |
| Logic | Logical relationships | AI, inference | Prolog, Datalog | Predicate/Rule | Declarative |
| Reactive | Data streams | UI, event processing | RxJS, Reactor | Observable | Stream |
| Actor | Message passing | High concurrency, distributed | Erlang, Akka | Actor | Isolated mutable |

### 13.2 Five Principles of Paradigm Selection

```
Five Principles of Paradigm Selection
============================================================

Principle 1: Paradigms are tools, not religions
  -> Don't cling to one paradigm; select based on the problem

Principle 2: The nature of the problem determines the paradigm
  -> Decide based on problem structure, not preference or trends

Principle 3: Prioritize simplicity
  -> If procedural is sufficient, that's fine. Avoid unnecessary abstraction

Principle 4: Maximize the pure part
  -> Push side effects to the edges; increase the proportion of testable
     pure functions

Principle 5: Consider the team's capabilities
  -> Even a theoretically optimal paradigm is meaningless if the team
     cannot use it effectively. Balance education cost and productivity
============================================================
```

### 13.3 Learning Roadmap

```
Paradigm Learning Roadmap
============================================================

Phase 1 (Introduction): Solid understanding of procedural
  - Implement basic algorithms in C or Python
  - Master concepts of variables, control structures, functions
  - Recommended duration: 1-3 months

Phase 2 (Foundation): Mastering OOP
  - Learn class design in Java, Python, TypeScript
  - SOLID principles, basics of design patterns
  - Recommended duration: 3-6 months

Phase 3 (Expansion): Functional basics
  - Practice functional style in JavaScript/TypeScript
  - map/filter/reduce, pure functions, immutability
  - Recommended duration: 2-4 months

Phase 4 (Deepening): Deep understanding of functional
  - Learn serious FP in Haskell, Elixir, or Scala
  - Monads, type classes, algebraic data types
  - Recommended duration: 3-6 months

Phase 5 (Integration): Multi-paradigm design
  - Practice Functional Core / Imperative Shell
  - Accumulate experience selecting paradigms based on problems
  - Introduce reactive and actor models
  - Recommended duration: Ongoing
============================================================
```

---

## Recommended Next Reading


---

## 14. References

### Books

1. **Van Roy, P. & Haridi, S.** *Concepts, Techniques, and Models of Computer Programming.* MIT Press, 2004.
   - A comprehensive textbook on paradigms. Using the multi-paradigm language Oz as its vehicle, it provides a unified explanation of major programming models. The most comprehensive reference providing the theoretical foundation for all paradigms covered in this chapter.

2. **Abelson, H. & Sussman, G. J.** *Structure and Interpretation of Computer Programs (SICP).* 2nd Ed, MIT Press, 1996.
   - MIT's legendary textbook. Using Scheme, it deeply explores the essence of procedural, functional, and object-oriented programming. A masterpiece that fundamentally questions "what programming is" through the step-by-step construction of abstractions.

3. **Armstrong, J.** *Programming Erlang: Software for a Concurrent World.* 2nd Ed, Pragmatic Bookshelf, 2013.
   - An explanation by the creator himself of the actor model and the "Let it crash" philosophy. Essential reading for learning practical approaches to concurrent programming.

4. **Martin, R. C.** *Clean Architecture: A Craftsman's Guide to Software Structure and Design.* Prentice Hall, 2017.
   - Explains software design principles that transcend paradigms. Guidelines for architecture-level design decisions by the proponent of the SOLID principles.

5. **Hutton, G.** *Programming in Haskell.* 2nd Ed, Cambridge University Press, 2016.
   - The most highly regarded introductory book for purely functional programming. Progressively explains monads, type classes, and lazy evaluation.

### Papers and Lectures

6. **Floyd, R. W.** "The Paradigms of Programming." *Communications of the ACM*, 22(8), 1979.
   - The Turing Award lecture that clarified the concept of "programming paradigms." A historic document asserting that paradigms are not merely coding styles but frameworks for thinking.

7. **Dijkstra, E. W.** "Go To Statement Considered Harmful." *Communications of the ACM*, 11(3), 1968.
   - The proposal for structured programming. A paper that had a decisive influence on the modernization of the procedural paradigm.

8. **Parnas, D. L.** "On the Criteria To Be Used in Decomposing Systems into Modules." *Communications of the ACM*, 15(12), 1972.
   - The paper that established the concept of information hiding (encapsulation). One of the theoretical foundations of OOP.

### Online Resources

9. **Bernhardt, G.** "Boundaries." RubyConf 2012.
   - The proposal of the "Functional Core, Imperative Shell" pattern. Widely referenced as a practical guideline for multi-paradigm design.

10. **Nystrom, R.** *Crafting Interpreters.* craftinginterpreters.com, 2021.
    - An online book that enables understanding the essence of paradigms through implementing programming languages. The full text is available for free.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
