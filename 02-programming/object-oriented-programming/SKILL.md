[日本語版](../../ja/02-programming/object-oriented-programming/SKILL.md)

# Object-Oriented Programming (OOP) Complete Guide

> A comprehensive guide to understanding and effectively applying object-oriented programming. Covers the four pillars, SOLID principles, design patterns, and anti-patterns with MIT-level quality.

## Target Audience

- Engineers who want to systematically learn OOP from fundamentals to advanced topics
- Developers seeking practical understanding of SOLID principles and design patterns
- Anyone looking to sharpen their judgment on "inheritance vs composition" and "when to use OOP"

## Prerequisites

- Programming basics (variables, functions, control flow)
- Experience with at least one programming language

## Guide Index

### 00-introduction (Introduction)
| File | Topic | Summary |
|------|-------|---------|
| [00-what-is-oop.md](docs/00-introduction/00-what-is-oop.md) | What is OOP | The essence of object-oriented thinking, mental models, and how it fits among other paradigms |
| [01-history-and-evolution.md](docs/00-introduction/01-history-and-evolution.md) | History and Evolution of OOP | Evolution from Simula to Smalltalk to C++ to Java to modern languages |
| [02-oop-vs-other-paradigms.md](docs/00-introduction/02-oop-vs-other-paradigms.md) | OOP vs Other Paradigms | Comparison with procedural, functional, and reactive paradigms; when to use each |
| [03-class-and-object.md](docs/00-introduction/03-class-and-object.md) | Classes and Objects | Internal structure of classes, memory layout, constructors, static members |

### 01-four-pillars (The Four Pillars)
| File | Topic | Summary |
|------|-------|---------|
| [00-encapsulation.md](docs/01-four-pillars/00-encapsulation.md) | Encapsulation | Information hiding, access modifiers, immutable objects, the getter/setter debate |
| [01-inheritance.md](docs/01-four-pillars/01-inheritance.md) | Inheritance | Single/multiple inheritance, the diamond problem, abstract classes, pitfalls of inheritance |
| [02-polymorphism.md](docs/01-four-pillars/02-polymorphism.md) | Polymorphism | Subtype, parametric, and ad-hoc polymorphism; dynamic dispatch; virtual function tables |
| [03-abstraction.md](docs/01-four-pillars/03-abstraction.md) | Abstraction | Interface design, abstract classes vs interfaces, leaky abstractions |

### 02-design-principles (Design Principles)
| File | Topic | Summary |
|------|-------|---------|
| [00-solid-overview.md](docs/02-design-principles/00-solid-overview.md) | SOLID Principles Overview | The big picture of all five principles, why SOLID matters, criteria for applying them |
| [01-srp-and-ocp.md](docs/02-design-principles/01-srp-and-ocp.md) | SRP + OCP | Single Responsibility Principle, Open/Closed Principle, practical examples |
| [02-lsp-and-isp.md](docs/02-design-principles/02-lsp-and-isp.md) | LSP + ISP | Liskov Substitution Principle, Interface Segregation Principle |
| [03-dip.md](docs/02-design-principles/03-dip.md) | DIP | Dependency Inversion Principle, Dependency Injection (DI), IoC Containers |

### 03-advanced-concepts (Advanced Concepts)
| File | Topic | Summary |
|------|-------|---------|
| [00-composition-vs-inheritance.md](docs/03-advanced-concepts/00-composition-vs-inheritance.md) | Composition vs Inheritance | Why "favor composition over inheritance," practical decision criteria |
| [01-interfaces-and-traits.md](docs/03-advanced-concepts/01-interfaces-and-traits.md) | Interfaces and Traits | Implementations in Java/TypeScript/Rust/Go, duck typing |
| [02-mixins-and-multiple-inheritance.md](docs/03-advanced-concepts/02-mixins-and-multiple-inheritance.md) | Mixins and Multiple Inheritance | Python MRO, Ruby modules, TypeScript Mixins |
| [03-generics-in-oop.md](docs/03-advanced-concepts/03-generics-in-oop.md) | Generics in OOP | Type parameters, covariance/contravariance, type erasure vs monomorphization |

### 04-practical-patterns (Practical Patterns)
| File | Topic | Summary |
|------|-------|---------|
| [00-creational-patterns.md](docs/04-practical-patterns/00-creational-patterns.md) | Creational Patterns | Factory, Builder, Singleton, Prototype |
| [01-structural-patterns.md](docs/04-practical-patterns/01-structural-patterns.md) | Structural Patterns | Adapter, Decorator, Facade, Proxy, Composite |
| [02-behavioral-patterns.md](docs/04-practical-patterns/02-behavioral-patterns.md) | Behavioral Patterns | Strategy, Observer, Command, State, Iterator |
| [03-anti-patterns.md](docs/04-practical-patterns/03-anti-patterns.md) | Anti-Patterns | God Object, deep inheritance hierarchies, Anemic Domain Model |

## Learning Path

```
Fundamentals:  00-introduction -> 01-four-pillars
Principles:    02-design-principles (SOLID)
Applied:       03-advanced-concepts -> 04-practical-patterns
```

## Related Skills

