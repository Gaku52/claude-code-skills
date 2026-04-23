# Mixins and Multiple Inheritance

> A technique for combining multiple behaviors while avoiding the problems of multiple inheritance. Compares Python's MRO, Ruby's modules, and TypeScript's mixin pattern.

## What You Will Learn in This Chapter

- [ ] Understand the problems of multiple inheritance (the diamond problem) and how to solve them
- [ ] Grasp how to implement the mixin pattern
- [ ] Learn how approaches differ across languages
- [ ] Understand cooperative multiple inheritance
- [ ] Grasp the design principles and anti-patterns of mixins
- [ ] Master testing strategies


## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content of [Interfaces and Traits](./01-interfaces-and-traits.md)

---

## 1. The Problem of Multiple Inheritance

```
The Diamond Problem:

      ┌─────────┐
      │    A    │  greet() → "Hello from A"
      └────┬────┘
      ┌────┴────┐
      ▼         ▼
  ┌─────┐   ┌─────┐
  │  B  │   │  C  │  B.greet() → "Hello from B"
  └──┬──┘   └──┬──┘  C.greet() → "Hello from C"
     └────┬────┘
          ▼
      ┌─────┐
      │  D  │  D.greet() → ??? (B? C? A?)
      └─────┘

Solutions:
  C++:    Virtual inheritance
  Python: MRO (C3 linearization)
  Java:   Multiple inheritance forbidden (interfaces only)
  Ruby:   Module
  Rust:   Trait (no multiple inheritance)
```

### 1.1 The Essence of the Diamond Problem

```
Three aspects of the diamond problem:

  1. Method Resolution Ambiguity
     → Methods with the same name are inherited through multiple paths
     → Which version should be called is unclear
     → Is D.greet() B.greet() or C.greet()?

  2. Constructor Duplication
     → A's constructor is called twice, once via B and once via C
     → Double initialization of resources, inconsistent state

  3. Shared State Problem
     → A's fields are initialized separately in B and C
     → From D's perspective, which state is correct?
     → Two copies of the fields may exist

  History:
    → 1969: Simula 67 only supports single inheritance
    → 1983: C++ introduces multiple inheritance
    → 1987: The diamond problem becomes widely recognized
    → 1995: Java intentionally excludes multiple inheritance
    → 2000s: Traits / mixins become mainstream
```

### 1.2 C++: Virtual Inheritance

```cpp
// C++: The diamond problem and virtual inheritance

#include <iostream>
#include <string>

// Without virtual inheritance
class A {
public:
    int value;
    A(int v) : value(v) {
        std::cout << "A(" << v << ") constructed" << std::endl;
    }
    virtual std::string greet() { return "Hello from A"; }
};

// Normal inheritance: two copies of A are created
class B_normal : public A {
public:
    B_normal() : A(1) {}
    std::string greet() override { return "Hello from B"; }
};

class C_normal : public A {
public:
    C_normal() : A(2) {}
    std::string greet() override { return "Hello from C"; }
};

// class D_normal : public B_normal, public C_normal {};
// → Compile error: A's members are ambiguous
// → Is d.value B_normal::value or C_normal::value?

// Virtual inheritance: only one instance of A
class B_virtual : virtual public A {
public:
    B_virtual() : A(1) {}
    std::string greet() override { return "Hello from B"; }
};

class C_virtual : virtual public A {
public:
    C_virtual() : A(2) {}
    std::string greet() override { return "Hello from C"; }
};

class D : public B_virtual, public C_virtual {
public:
    // The constructor of the virtual base class A is called by the most derived class
    D() : A(0), B_virtual(), C_virtual() {}

    // greet() must be overridden (because it is ambiguous between B and C)
    std::string greet() override {
        return "Hello from D (B says: " + B_virtual::greet() + ")";
    }
};

int main() {
    D d;
    std::cout << d.greet() << std::endl;
    std::cout << d.value << std::endl;  // 0 (A's constructor is called by D)
    // "A(0) constructed" is printed only once
    return 0;
}
```

### 1.3 Comparison of Approaches Across Languages

```
┌──────────┬──────────────────────┬───────────────────┬──────────────┐
│ Language │ Multiple Inheritance │ Behavior Comp.    │ Shared State │
├──────────┼──────────────────────┼───────────────────┼──────────────┤
│ C++      │ Via virtual inherit. │ Virtual base cls  │ Possible     │
│ Python   │ Linearized with MRO  │ Multi-inh + Mixin │ Possible     │
│ Java     │ No class multi-inh.  │ Interface         │ Not possible │
│ Kotlin   │ No class multi-inh.  │ Interface         │ Not possible │
│ Ruby     │ No class multi-inh.  │ Module            │ No (Note 1)  │
│ Scala    │ No class multi-inh.  │ trait             │ Possible(val)│
│ Rust     │ No classes at all    │ trait             │ Not possible │
│ Swift    │ No class multi-inh.  │ Protocol          │ Not possible │
│ TypeScript│ No class multi-inh. │ Mixin function    │ Possible (N2)│
│ PHP      │ No class multi-inh.  │ trait             │ Possible     │
└──────────┴──────────────────────┴───────────────────┴──────────────┘

Note 1: Ruby's Module can access the instance variables of the class that
        includes it, but the Module itself does not hold state
Note 2: TypeScript mixins are functions that return classes, so they can
        add state (properties)
```

---

## 2. Python: MRO (Method Resolution Order)

### 2.1 The C3 Linearization Algorithm

```python
# Python: MRO via C3 linearization
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    def greet(self):
        return "Hello from C"

class D(B, C):
    pass

d = D()
print(d.greet())  # "Hello from B"

# Check the MRO
print(D.__mro__)
# (D, B, C, A, object)
# → Searched in the order D → B → C → A → object
```

```python
# Details of the C3 linearization algorithm

# C3 linearization formula:
# L(C) = C + merge(L(B1), L(B2), ..., L(Bn), B1 B2 ... Bn)
#
# Rules for merge:
#   1. Take the head element of the first list
#   2. If that element does not appear in the "tail" of any other list, add it to the result
#   3. If it does appear, try the head of the next list
#   4. Repeat until all lists are empty

class O: pass   # stand-in for object

class A(O):
    def method(self):
        return "A"

class B(O):
    def method(self):
        return "B"

class C(O):
    def method(self):
        return "C"

class D(O):
    def method(self):
        return "D"

class E(A, B):
    pass

class F(C, D):
    pass

class G(E, F):
    pass

# MRO computation:
# L(O) = [O]
# L(A) = [A, O]
# L(B) = [B, O]
# L(C) = [C, O]
# L(D) = [D, O]
# L(E) = E + merge(L(A), L(B), [A, B])
#       = E + merge([A, O], [B, O], [A, B])
#       = [E, A] + merge([O], [B, O], [B])
#       = [E, A, B] + merge([O], [O])
#       = [E, A, B, O]
# L(F) = [F, C, D, O]
# L(G) = G + merge(L(E), L(F), [E, F])
#       = G + merge([E, A, B, O], [F, C, D, O], [E, F])
#       = [G, E] + merge([A, B, O], [F, C, D, O], [F])
#       = [G, E, A, B] + merge([O], [F, C, D, O], [F])
#       = [G, E, A, B, F] + merge([O], [C, D, O])
#       = [G, E, A, B, F, C, D] + merge([O], [O])
#       = [G, E, A, B, F, C, D, O]

print(G.__mro__)
# (<class 'G'>, <class 'E'>, <class 'A'>, <class 'B'>,
#  <class 'F'>, <class 'C'>, <class 'D'>, <class 'O'>)


# A case where C3 linearization fails
# Contradictory inheritance orders are detected and raise an error
class X(A, B): pass  # A before B
class Y(B, A): pass  # B before A

# class Z(X, Y): pass
# → TypeError: Cannot create a consistent method resolution order (MRO)
#   for bases A, B
# X requires the order A → B, while Y requires B → A, which is contradictory
```

### 2.2 Cooperative Multiple Inheritance

```python
# Cooperative multiple inheritance using super()

class Base:
    def __init__(self, **kwargs):
        # The final base class ignores any remaining kwargs
        pass

    def process(self, data: str) -> str:
        return data


class LoggingMixin(Base):
    """Mixin that adds logging"""
    def __init__(self, *, log_prefix: str = "LOG", **kwargs):
        super().__init__(**kwargs)  # delegate to the next class
        self.log_prefix = log_prefix
        self._logs: list[str] = []

    def process(self, data: str) -> str:
        self._logs.append(f"[{self.log_prefix}] Processing: {data}")
        # Call the next class's process via super() in MRO order
        result = super().process(data)
        self._logs.append(f"[{self.log_prefix}] Result: {result}")
        return result


class ValidationMixin(Base):
    """Mixin that adds validation"""
    def __init__(self, *, max_length: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def process(self, data: str) -> str:
        if len(data) > self.max_length:
            raise ValueError(
                f"Data too long: {len(data)} > {self.max_length}"
            )
        return super().process(data)


class TransformMixin(Base):
    """Mixin that adds data transformation"""
    def __init__(self, *, uppercase: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.uppercase = uppercase

    def process(self, data: str) -> str:
        if self.uppercase:
            data = data.upper()
        return super().process(data)


class CacheMixin(Base):
    """Mixin that adds caching"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache: dict[str, str] = {}

    def process(self, data: str) -> str:
        if data in self._cache:
            return self._cache[data]
        result = super().process(data)
        self._cache[data] = result
        return result


# Composing mixins
class TextProcessor(
    LoggingMixin,
    ValidationMixin,
    TransformMixin,
    CacheMixin,
    Base,
):
    """Text processor composed from multiple mixins"""
    pass


# Check the MRO
print(TextProcessor.__mro__)
# TextProcessor → LoggingMixin → ValidationMixin →
# TransformMixin → CacheMixin → Base → object

# Usage example
processor = TextProcessor(
    log_prefix="TXT",
    max_length=200,
    uppercase=True,
)

result = processor.process("hello world")
print(result)        # "HELLO WORLD"
print(processor._logs)
# ['[TXT] Processing: hello world', '[TXT] Result: HELLO WORLD']

# The process call chain:
# 1. LoggingMixin.process: log → super()
# 2. ValidationMixin.process: validate → super()
# 3. TransformMixin.process: uppercase → super()
# 4. CacheMixin.process: cache → super()
# 5. Base.process: return the data as-is
```

```python
# Details of the super() mechanism

class A:
    def method(self):
        print("A.method start")
        # A is last in the MRO, so super() is object
        print("A.method end")

class B(A):
    def method(self):
        print("B.method start")
        super().method()  # next in MRO → C (Note: the next class after B is NOT A!)
        print("B.method end")

class C(A):
    def method(self):
        print("C.method start")
        super().method()  # next in MRO → A
        print("C.method end")

class D(B, C):
    def method(self):
        print("D.method start")
        super().method()  # next in MRO → B
        print("D.method end")

d = D()
d.method()
# Output:
# D.method start
# B.method start
# C.method start    ← B's super() calls C! (not A)
# A.method start
# A.method end
# C.method end
# B.method end
# D.method end

# MRO: D → B → C → A → object
# super() calls "the next class in the MRO", not "the parent class"
```

### 2.3 Python: Practical Mixin Patterns

```python
# Mixin = a class that is not used standalone but adds functionality
import json
from datetime import datetime
from typing import Any, TypeVar, Type

T = TypeVar("T")


class JsonMixin:
    """Adds JSON conversion functionality"""
    def to_json(self) -> str:
        return json.dumps(self._to_dict(), default=str, ensure_ascii=False)

    def _to_dict(self) -> dict[str, Any]:
        """Returns the fields to be serialized"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        return result

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls(**data)


class TimestampMixin:
    """Adds timestamp functionality"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.created_at = datetime.now()
            self.updated_at = datetime.now()

        cls.__init__ = new_init

    def touch(self) -> None:
        """Updates updated_at"""
        self.updated_at = datetime.now()


class LoggableMixin:
    """Adds logging functionality"""
    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().isoformat()
        print(f"[{timestamp}] [{level}] [{type(self).__name__}] {message}")

    def log_error(self, message: str) -> None:
        self.log(message, level="ERROR")

    def log_warning(self, message: str) -> None:
        self.log(message, level="WARNING")


class ValidatableMixin:
    """Adds validation functionality"""

    def validate(self) -> list[str]:
        """Returns a list of validation errors"""
        errors = []
        for attr_name in dir(self):
            if attr_name.startswith("validate_"):
                field_name = attr_name[len("validate_"):]
                validator = getattr(self, attr_name)
                error = validator()
                if error:
                    errors.append(f"{field_name}: {error}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class EventEmitterMixin:
    """Adds event emission functionality"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._event_handlers: dict[str, list] = {}

        cls.__init__ = new_init

    def on(self, event: str, handler) -> None:
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def emit(self, event: str, *args, **kwargs) -> None:
        for handler in self._event_handlers.get(event, []):
            handler(*args, **kwargs)

    def off(self, event: str, handler=None) -> None:
        if handler is None:
            self._event_handlers.pop(event, None)
        elif event in self._event_handlers:
            self._event_handlers[event] = [
                h for h in self._event_handlers[event] if h != handler
            ]


# Combining mixins
class User(
    JsonMixin,
    TimestampMixin,
    LoggableMixin,
    ValidatableMixin,
    EventEmitterMixin,
):
    def __init__(self, name: str, email: str, age: int = 0):
        self.name = name
        self.email = email
        self.age = age

    def validate_name(self) -> str | None:
        if not self.name or len(self.name) < 2:
            return "Name must be at least 2 characters"
        return None

    def validate_email(self) -> str | None:
        if "@" not in self.email:
            return "Email address format is invalid"
        return None

    def validate_age(self) -> str | None:
        if self.age < 0 or self.age > 150:
            return "Age must be in the range 0 to 150"
        return None


# Usage example
user = User("Taro Tanaka", "tanaka@example.com", age=30)

# JsonMixin
print(user.to_json())
# {"name": "Taro Tanaka", "email": "tanaka@example.com", "age": 30}

# TimestampMixin
print(user.created_at)

# LoggableMixin
user.log("Logged in")

# ValidatableMixin
print(user.validate())  # [] (no errors)
print(user.is_valid())  # True

# EventEmitterMixin
user.on("login", lambda: print("Login event fired"))
user.emit("login")
```

### 2.4 Python: Metaprogramming with __init_subclass__

```python
# __init_subclass__: a hook that runs automatically when a subclass is defined

class RegisterMixin:
    """Mixin that auto-registers subclasses"""
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, register_name: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        name = register_name or cls.__name__
        RegisterMixin._registry[name] = cls

    @classmethod
    def get_registered(cls, name: str) -> type | None:
        return RegisterMixin._registry.get(name)

    @classmethod
    def list_registered(cls) -> list[str]:
        return list(RegisterMixin._registry.keys())


class Plugin(RegisterMixin):
    def execute(self):
        raise NotImplementedError


class CSVExporter(Plugin, register_name="csv"):
    def execute(self):
        return "Exporting as CSV..."


class JSONExporter(Plugin, register_name="json"):
    def execute(self):
        return "Exporting as JSON..."


class XMLExporter(Plugin, register_name="xml"):
    def execute(self):
        return "Exporting as XML..."


# Use the auto-registered plugins
print(RegisterMixin.list_registered())
# ['Plugin', 'csv', 'json', 'xml']

exporter_cls = RegisterMixin.get_registered("csv")
exporter = exporter_cls()
print(exporter.execute())  # "Exporting as CSV..."


# Descriptor-based mixin using __set_name__
class TypeCheckedMixin:
    """Automatically defines type-checked properties"""

    class TypeCheckedDescriptor:
        def __init__(self, name: str, expected_type: type):
            self.name = name
            self.expected_type = expected_type
            self.private_name = f"_typechecked_{name}"

        def __set_name__(self, owner, name):
            self.name = name
            self.private_name = f"_typechecked_{name}"

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            return getattr(obj, self.private_name, None)

        def __set__(self, obj, value):
            if not isinstance(value, self.expected_type):
                raise TypeError(
                    f"{self.name} must be {self.expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
            setattr(obj, self.private_name, value)


class StrictUser:
    name = TypeCheckedMixin.TypeCheckedDescriptor("name", str)
    age = TypeCheckedMixin.TypeCheckedDescriptor("age", int)
    email = TypeCheckedMixin.TypeCheckedDescriptor("email", str)

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


user = StrictUser("Tanaka", 30, "tanaka@example.com")  # OK
# StrictUser("Tanaka", "30", "tanaka@example.com")  # TypeError
```

---

## 3. Ruby: Modules

### 3.1 The Difference Between include, extend, and prepend

```ruby
# Ruby: Mixins via Module

# Three ways to incorporate a Module
#   include:  added as instance methods
#   extend:   added as class methods
#   prepend:  added as instance methods (found first during method lookup)

module Greetable
  def greet
    "Hello, I'm #{name}"
  end
end

module ClassInfo
  def info
    "Class: #{self.name}, Methods: #{instance_methods(false).count}"
  end
end

module Logging
  def greet
    puts "[LOG] greet called"
    super  # with prepend, this calls the original greet
  end
end

class Person
  include Greetable  # added as instance methods
  extend ClassInfo   # added as class methods
  prepend Logging    # overrides greet (super can call the original method)

  attr_reader :name

  def initialize(name)
    @name = name
  end
end

person = Person.new("Tanaka")
puts person.greet      # [LOG] greet called  →  "Hello, I'm Tanaka"
puts Person.info       # "Class: Person, Methods: 1"

# Method lookup order
puts Person.ancestors
# [Logging, Person, Greetable, Object, Kernel, BasicObject]
# prepend comes before Person
# include comes after Person
```

```ruby
# Ruby: Advanced use of Module

module Serializable
  def to_json
    require 'json'
    JSON.generate(instance_variables.each_with_object({}) { |var, hash|
      hash[var.to_s.delete('@')] = instance_variable_get(var)
    })
  end

  def to_hash
    instance_variables.each_with_object({}) { |var, hash|
      hash[var.to_s.delete('@').to_sym] = instance_variable_get(var)
    }
  end
end

module Loggable
  def log(message, level: :info)
    timestamp = Time.now.strftime('%Y-%m-%d %H:%M:%S')
    puts "[#{timestamp}] [#{level.upcase}] [#{self.class.name}] #{message}"
  end
end

module Cacheable
  def self.included(base)
    # Hook called when the Module is included
    base.instance_variable_set(:@cache, {})
    base.extend(ClassMethods)
  end

  module ClassMethods
    def cache
      @cache
    end

    def cached_method(method_name)
      original = instance_method(method_name)
      define_method(method_name) do |*args|
        cache_key = "#{method_name}:#{args.hash}"
        self.class.cache[cache_key] ||= original.bind(self).call(*args)
      end
    end
  end

  def cache_key
    "#{self.class.name}:#{object_id}"
  end
end

module Comparable
  def <=>(other)
    raise NotImplementedError, "#{self.class} must implement <=>"
  end

  def <(other);  (self <=> other) < 0; end
  def >(other);  (self <=> other) > 0; end
  def <=(other); (self <=> other) <= 0; end
  def >=(other); (self <=> other) >= 0; end
  def ==(other); (self <=> other) == 0; end
end

class Product
  include Serializable
  include Loggable
  include Cacheable
  include Comparable

  attr_accessor :name, :price, :category

  def initialize(name, price, category = "general")
    @name = name
    @price = price
    @category = category
  end

  def <=>(other)
    @price <=> other.price
  end

  def expensive?
    @price > 10000
  end
  cached_method :expensive?  # cached via Cacheable
end

product = Product.new("Laptop", 89800, "electronics")
puts product.to_json        # Serializable
product.log("Inventory added")      # Loggable
puts product.cache_key      # Cacheable
puts product > Product.new("Mouse", 2980)  # Comparable → true
```

### 3.2 Ruby: The concern Pattern (Rails)

```ruby
# The ActiveSupport::Concern pattern
# A widely-used way to structure mixins in Rails

module ActiveSupport
  module Concern
    def self.extended(base)
      base.instance_variable_set(:@_dependencies, [])
    end

    def included(base)
      @_dependencies.each { |dep| base.include(dep) }
      base.class_eval(&@_included_block) if @_included_block
      base.extend(const_get(:ClassMethods)) if const_defined?(:ClassMethods)
    end

    def class_methods(&block)
      mod = const_defined?(:ClassMethods, false) ?
        const_get(:ClassMethods) :
        const_set(:ClassMethods, Module.new)
      mod.module_eval(&block)
    end

    def included_block(&block)
      @_included_block = block
    end
  end
end

# Example usage of Concern
module Searchable
  extend ActiveSupport::Concern

  class_methods do
    def search(query)
      puts "Searching #{self.name} for: #{query}"
      # In actual Rails: where("name LIKE ?", "%#{query}%")
    end

    def search_by_field(field, value)
      puts "Searching #{self.name} by #{field}: #{value}"
    end
  end

  def highlight(query)
    # instance method
    puts "Highlighting '#{query}' in #{self.class.name}"
  end
end

module Taggable
  extend ActiveSupport::Concern

  class_methods do
    def find_by_tag(tag)
      puts "Finding #{self.name} with tag: #{tag}"
    end
  end

  def add_tag(tag)
    @tags ||= []
    @tags << tag
  end

  def tags
    @tags || []
  end
end

module Auditable
  extend ActiveSupport::Concern

  class_methods do
    def audit_log
      @audit_log ||= []
    end
  end

  def audit(action)
    self.class.audit_log << {
      action: action,
      record: self,
      timestamp: Time.now
    }
  end
end

class Article
  include Searchable
  include Taggable
  include Auditable

  attr_accessor :title, :body

  def initialize(title, body)
    @title = title
    @body = body
  end
end

Article.search("Ruby")             # Searchable
Article.find_by_tag("programming") # Taggable

article = Article.new("Intro to Ruby", "Ruby basics...")
article.add_tag("ruby")           # Taggable
article.highlight("Ruby")         # Searchable
article.audit("created")          # Auditable
```

---

## 4. TypeScript: The Mixin Pattern

### 4.1 Class-Expression-Based Mixins

```typescript
// TypeScript: Mixins (pattern using class expressions)
type Constructor<T = {}> = new (...args: any[]) => T;

// Mixin function
function Timestamped<TBase extends Constructor>(Base: TBase) {
  return class Timestamped extends Base {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      this.updatedAt = new Date();
    }
  };
}

function Activatable<TBase extends Constructor>(Base: TBase) {
  return class Activatable extends Base {
    isActive = false;

    activate() { this.isActive = true; }
    deactivate() { this.isActive = false; }
  };
}

function Taggable<TBase extends Constructor>(Base: TBase) {
  return class Taggable extends Base {
    tags: Set<string> = new Set();

    addTag(tag: string) { this.tags.add(tag); }
    removeTag(tag: string) { this.tags.delete(tag); }
    hasTag(tag: string) { return this.tags.has(tag); }
  };
}

// Base class
class User {
  constructor(public name: string, public email: string) {}
}

// Compose the mixins
const EnhancedUser = Taggable(Activatable(Timestamped(User)));

const user = new EnhancedUser("Tanaka", "tanaka@example.com");
user.activate();          // Activatable
user.addTag("premium");   // Taggable
user.touch();             // Timestamped
```

### 4.2 Type-Safe Mixins (Advanced Pattern)

```typescript
// TypeScript: Implementing type-safe mixins

// More strict type definitions
type GConstructor<T = {}> = new (...args: any[]) => T;

// Define the type of each mixin via an interface
interface HasId {
  id: string;
}

interface HasName {
  name: string;
}

// Constrained mixin: applicable only to classes with HasId
function Persistable<TBase extends GConstructor<HasId>>(Base: TBase) {
  return class Persistable extends Base {
    isPersisted = false;

    async save(): Promise<void> {
      console.log(`Saving entity with id: ${this.id}`);
      this.isPersisted = true;
    }

    async delete(): Promise<void> {
      console.log(`Deleting entity with id: ${this.id}`);
      this.isPersisted = false;
    }
  };
}

// Constrained mixin: applicable only to classes with HasName
function Searchable<TBase extends GConstructor<HasName>>(Base: TBase) {
  return class Searchable extends Base {
    matches(query: string): boolean {
      return this.name.toLowerCase().includes(query.toLowerCase());
    }
  };
}

// Serializable: applicable to any class
function Serializable<TBase extends GConstructor>(Base: TBase) {
  return class Serializable extends Base {
    toJSON(): Record<string, unknown> {
      const result: Record<string, unknown> = {};
      for (const key of Object.keys(this)) {
        const value = (this as any)[key];
        if (typeof value !== "function") {
          result[key] = value;
        }
      }
      return result;
    }

    toString(): string {
      return JSON.stringify(this.toJSON(), null, 2);
    }
  };
}

// Validatable: adds validation rules
function Validatable<TBase extends GConstructor>(Base: TBase) {
  return class Validatable extends Base {
    private _validationRules: Map<string, (value: any) => string | null> =
      new Map();

    addRule(field: string, rule: (value: any) => string | null): void {
      this._validationRules.set(field, rule);
    }

    validate(): string[] {
      const errors: string[] = [];
      for (const [field, rule] of this._validationRules) {
        const value = (this as any)[field];
        const error = rule(value);
        if (error) errors.push(`${field}: ${error}`);
      }
      return errors;
    }

    isValid(): boolean {
      return this.validate().length === 0;
    }
  };
}

// Base class
class Entity {
  constructor(public id: string, public name: string) {}
}

// Compose the mixins
const EnhancedEntity = Validatable(
  Serializable(
    Searchable(
      Persistable(Entity)
    )
  )
);

// Usage example
const entity = new EnhancedEntity("1", "Product A");
entity.addRule("name", (v) =>
  v.length < 2 ? "Name must be at least 2 characters" : null,
);

console.log(entity.matches("product"));  // true (Searchable)
console.log(entity.isValid());            // true (Validatable)
console.log(entity.toJSON());             // { id: "1", name: "Product A" }
await entity.save();                      // "Saving entity with id: 1"
```

### 4.3 Decorator-Based Mixins

```typescript
// Mixins using TypeScript 5.0+ decorators

// Method decorator: adds logging
function logged(
  target: any,
  context: ClassMethodDecoratorContext,
) {
  const methodName = String(context.name);
  return function (this: any, ...args: any[]) {
    console.log(`[${methodName}] Called with:`, args);
    const result = target.call(this, ...args);
    console.log(`[${methodName}] Returned:`, result);
    return result;
  };
}

// Method decorator: adds caching
function cached(
  target: any,
  context: ClassMethodDecoratorContext,
) {
  const cache = new Map<string, any>();
  return function (this: any, ...args: any[]) {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = target.call(this, ...args);
    cache.set(key, result);
    return result;
  };
}

// Method decorator: adds retry
function retry(maxAttempts: number) {
  return function (
    target: any,
    context: ClassMethodDecoratorContext,
  ) {
    return async function (this: any, ...args: any[]) {
      let lastError: Error | undefined;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          return await target.call(this, ...args);
        } catch (error) {
          lastError = error as Error;
          console.log(
            `Attempt ${attempt}/${maxAttempts} failed: ${lastError.message}`,
          );
        }
      }
      throw lastError;
    };
  };
}

// Class decorator: adds timestamps
function withTimestamps<T extends Constructor>(Base: T) {
  return class extends Base {
    createdAt = new Date();
    updatedAt = new Date();

    touch() {
      this.updatedAt = new Date();
    }
  };
}

// Usage example
@withTimestamps
class ApiClient {
  constructor(private baseUrl: string) {}

  @logged
  @retry(3)
  async fetchData(endpoint: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}${endpoint}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  @logged
  @cached
  computeExpensiveResult(input: number): number {
    // Simulating an expensive computation
    let result = 0;
    for (let i = 0; i < input * 1000; i++) {
      result += Math.sin(i);
    }
    return result;
  }
}
```

---

## 5. Rust: Trait Composition

```rust
// Rust: Composing behavior with traits

use std::fmt;

// Trait definitions
trait Displayable {
    fn display_name(&self) -> String;
}

trait Serializable {
    fn serialize(&self) -> String;
    fn content_type(&self) -> &str { "application/json" }
}

trait Validatable {
    fn validate(&self) -> Result<(), Vec<String>>;

    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

trait Auditable {
    fn audit_log(&self) -> String;
}

// Implementing multiple traits
struct Product {
    id: u64,
    name: String,
    price: f64,
    category: String,
}

impl Displayable for Product {
    fn display_name(&self) -> String {
        format!("{} (¥{:.0})", self.name, self.price)
    }
}

impl Serializable for Product {
    fn serialize(&self) -> String {
        format!(
            r#"{{"id":{},"name":"{}","price":{},"category":"{}"}}"#,
            self.id, self.name, self.price, self.category
        )
    }
}

impl Validatable for Product {
    fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("Name is required".to_string());
        }
        if self.price < 0.0 {
            errors.push("Price must be non-negative".to_string());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

impl Auditable for Product {
    fn audit_log(&self) -> String {
        format!("Product[{}]: {} - ¥{:.0}", self.id, self.name, self.price)
    }
}

// Require multiple traits via trait bounds
fn process_entity<T>(entity: &T)
where
    T: Displayable + Serializable + Validatable + Auditable,
{
    // Validation
    match entity.validate() {
        Ok(()) => println!("✓ Validation passed"),
        Err(errors) => {
            for err in &errors {
                println!("✗ {}", err);
            }
            return;
        }
    }

    // Display
    println!("Name: {}", entity.display_name());

    // Serialize
    println!("JSON: {}", entity.serialize());

    // Audit log
    println!("Audit: {}", entity.audit_log());
}

// Combine as a trait object
// Note: To use multiple traits as a trait object,
//       define a super-trait.
trait FullyFeatured: Displayable + Serializable + Validatable + Auditable {}

// Blanket implementation: any type that implements all four is automatically FullyFeatured
impl<T> FullyFeatured for T
where
    T: Displayable + Serializable + Validatable + Auditable,
{}

fn process_any(entity: &dyn FullyFeatured) {
    println!("Name: {}", entity.display_name());
    println!("JSON: {}", entity.serialize());
}
```

```rust
// Rust: Automatic mixins via Derive macros

// Standard derive macros
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}

// Example usage of custom derive macros (procedural macros)
// Note: in practice a proc-macro crate is required
// #[derive(Serialize, Deserialize)]  // serde
// #[derive(Builder)]                  // derive_builder
// #[derive(Display)]                  // derive_more

// Newtype pattern with Deref and DerefMut
use std::ops::{Deref, DerefMut};

struct Email(String);

impl Email {
    fn new(value: &str) -> Result<Self, String> {
        if value.contains('@') && value.contains('.') {
            Ok(Email(value.to_string()))
        } else {
            Err(format!("Invalid email: {}", value))
        }
    }
}

impl Deref for Email {
    type Target = String;
    fn deref(&self) -> &String {
        &self.0
    }
}

// Email can use all String methods
let email = Email::new("user@example.com").unwrap();
println!("Length: {}", email.len());       // String::len()
println!("Upper: {}", email.to_uppercase()); // String::to_uppercase()

// However, implicit type conversion does not occur
// let s: String = email;  // ❌ compile error
let s: &str = &email;     // ✅ Deref coercion
```

---

## 6. Java: Default Methods and Interface Composition

```java
// Java: A mixin-like pattern using default methods in interfaces

public interface Identifiable {
    String getId();
}

public interface Timestamped {
    Instant getCreatedAt();
    Instant getUpdatedAt();
    void setUpdatedAt(Instant updatedAt);

    default void touch() {
        setUpdatedAt(Instant.now());
    }

    default Duration age() {
        return Duration.between(getCreatedAt(), Instant.now());
    }
}

public interface Loggable {
    default void logInfo(String message) {
        System.out.printf("[INFO] [%s] %s%n", getClass().getSimpleName(), message);
    }

    default void logError(String message, Throwable error) {
        System.out.printf("[ERROR] [%s] %s: %s%n",
            getClass().getSimpleName(), message, error.getMessage());
    }
}

public interface Validatable {
    List<String> validate();

    default boolean isValid() {
        return validate().isEmpty();
    }

    default void validateOrThrow() {
        List<String> errors = validate();
        if (!errors.isEmpty()) {
            throw new ValidationException(
                String.join("; ", errors)
            );
        }
    }
}

public interface Serializable {
    default String toJson() {
        // Simple implementation using reflection
        var sb = new StringBuilder("{");
        var fields = getClass().getDeclaredFields();
        for (int i = 0; i < fields.length; i++) {
            fields[i].setAccessible(true);
            try {
                sb.append(String.format("\"%s\":\"%s\"",
                    fields[i].getName(),
                    fields[i].get(this)));
            } catch (IllegalAccessException e) {
                // skip
            }
            if (i < fields.length - 1) sb.append(",");
        }
        sb.append("}");
        return sb.toString();
    }
}

// Implement multiple interfaces
public class User implements
        Identifiable, Timestamped, Loggable,
        Validatable, Serializable {

    private final String id;
    private final String name;
    private final String email;
    private final Instant createdAt;
    private Instant updatedAt;

    public User(String id, String name, String email) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createdAt = Instant.now();
        this.updatedAt = Instant.now();
    }

    @Override
    public String getId() { return id; }

    @Override
    public Instant getCreatedAt() { return createdAt; }

    @Override
    public Instant getUpdatedAt() { return updatedAt; }

    @Override
    public void setUpdatedAt(Instant updatedAt) {
        this.updatedAt = updatedAt;
    }

    @Override
    public List<String> validate() {
        var errors = new ArrayList<String>();
        if (name == null || name.length() < 2) {
            errors.add("Name must be at least 2 characters");
        }
        if (email == null || !email.contains("@")) {
            errors.add("Email address is invalid");
        }
        return errors;
    }
}

// Usage example
var user = new User("1", "Tanaka", "tanaka@example.com");
user.logInfo("Logged in");        // Loggable
user.touch();                    // Timestamped
System.out.println(user.isValid());  // Validatable
System.out.println(user.toJson());   // Serializable
```

```java
// Java: Resolving default method conflicts

interface Flyable {
    default String move() { return "Flying"; }
}

interface Swimmable {
    default String move() { return "Swimming"; }
}

interface Walkable {
    default String move() { return "Walking"; }
}

// Three move() methods conflict → explicit override is required
class Duck implements Flyable, Swimmable, Walkable {
    @Override
    public String move() {
        // A specific interface's default implementation can be selected
        return Flyable.super.move();
    }

    // Switch based on context
    public String move(String context) {
        return switch (context) {
            case "air" -> Flyable.super.move();
            case "water" -> Swimmable.super.move();
            case "land" -> Walkable.super.move();
            default -> move();
        };
    }
}

Duck duck = new Duck();
System.out.println(duck.move());          // "Flying"
System.out.println(duck.move("water"));   // "Swimming"
System.out.println(duck.move("land"));    // "Walking"
```

---

## 7. Kotlin: Mixins via Delegation

```kotlin
// Kotlin: Interface delegation with the by keyword

interface Logger {
    fun log(message: String)
    fun logError(message: String, error: Throwable)
}

interface Cache<K, V> {
    fun get(key: K): V?
    fun put(key: K, value: V)
    fun invalidate(key: K)
    fun clear()
}

// Concrete implementations
class ConsoleLogger : Logger {
    override fun log(message: String) {
        println("[INFO] $message")
    }

    override fun logError(message: String, error: Throwable) {
        println("[ERROR] $message: ${error.message}")
    }
}

class InMemoryCache<K, V> : Cache<K, V> {
    private val store = mutableMapOf<K, V>()

    override fun get(key: K): V? = store[key]
    override fun put(key: K, value: V) { store[key] = value }
    override fun invalidate(key: K) { store.remove(key) }
    override fun clear() { store.clear() }
}

// Delegate with the by keyword (composition that satisfies the interfaces)
class UserService(
    private val logger: Logger = ConsoleLogger(),
    private val cache: Cache<String, User> = InMemoryCache(),
) : Logger by logger, Cache<String, User> by cache {

    fun findUser(id: String): User? {
        // Cache.get can be called directly (because it is delegated via `by`)
        val cached = get(id)
        if (cached != null) {
            log("Cache hit for user: $id")
            return cached
        }

        log("Cache miss for user: $id")
        // Fetch from the DB
        val user = fetchFromDb(id) ?: return null
        put(id, user)  // Cache.put can be called directly
        return user
    }

    private fun fetchFromDb(id: String): User? {
        log("Fetching user $id from database")
        return User(id, "Taro Tanaka", "tanaka@example.com")
    }
}

data class User(val id: String, val name: String, val email: String)

// Usage example
val service = UserService()
val user = service.findUser("user-1")

// UserService satisfies both the Logger and Cache interfaces
val logger: Logger = service
val cache: Cache<String, User> = service
```

```kotlin
// Kotlin: A mixin-like pattern using extension functions

// Add cross-cutting functionality via interfaces + extension functions
interface HasTimestamp {
    val createdAt: Long
    val updatedAt: Long
}

// Extension function: applies to any type that implements HasTimestamp
fun HasTimestamp.formatCreatedAt(): String {
    val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    return sdf.format(java.util.Date(createdAt))
}

fun HasTimestamp.isOlderThan(days: Int): Boolean {
    val threshold = System.currentTimeMillis() - days * 86400000L
    return createdAt < threshold
}

interface HasName {
    val name: String
}

fun HasName.initials(): String {
    return name.split(" ")
        .mapNotNull { it.firstOrNull()?.uppercase() }
        .joinToString("")
}

// A data class implementing multiple interfaces
data class Article(
    val id: String,
    override val name: String,
    val content: String,
    override val createdAt: Long = System.currentTimeMillis(),
    override val updatedAt: Long = System.currentTimeMillis(),
) : HasTimestamp, HasName

// Usage example
val article = Article("1", "Intro to Kotlin", "Learn the basics of Kotlin")
println(article.formatCreatedAt())  // HasTimestamp extension
println(article.isOlderThan(30))    // HasTimestamp extension
println(article.initials())          // HasName extension → "I"
```

---

## 8. PHP: Traits

```php
<?php
// PHP: Mixins via trait

trait Timestampable {
    private DateTime $createdAt;
    private DateTime $updatedAt;

    public function initTimestamps(): void {
        $this->createdAt = new DateTime();
        $this->updatedAt = new DateTime();
    }

    public function getCreatedAt(): DateTime {
        return $this->createdAt;
    }

    public function getUpdatedAt(): DateTime {
        return $this->updatedAt;
    }

    public function touch(): void {
        $this->updatedAt = new DateTime();
    }
}

trait SoftDeletable {
    private ?DateTime $deletedAt = null;

    public function softDelete(): void {
        $this->deletedAt = new DateTime();
    }

    public function restore(): void {
        $this->deletedAt = null;
    }

    public function isDeleted(): bool {
        return $this->deletedAt !== null;
    }
}

trait HasSlug {
    private string $slug;

    public function generateSlug(string $title): void {
        $this->slug = strtolower(
            preg_replace('/[^a-zA-Z0-9]+/', '-', $title)
        );
    }

    public function getSlug(): string {
        return $this->slug;
    }
}

// Composing traits
class Article {
    use Timestampable;
    use SoftDeletable;
    use HasSlug;

    public function __construct(
        private string $title,
        private string $body,
    ) {
        $this->initTimestamps();
        $this->generateSlug($title);
    }
}

// Resolving trait conflicts
trait A {
    public function hello(): string {
        return "Hello from A";
    }
}

trait B {
    public function hello(): string {
        return "Hello from B";
    }
}

class C {
    use A, B {
        A::hello insteadof B;  // prefer A's hello
        B::hello as helloB;    // use B's hello under an alias
    }
}

$c = new C();
echo $c->hello();    // "Hello from A"
echo $c->helloB();   // "Hello from B"

// Trait requirements (abstract methods)
trait Loggable {
    abstract protected function getLogPrefix(): string;

    public function log(string $message): void {
        echo "[{$this->getLogPrefix()}] $message\n";
    }
}

class UserService {
    use Loggable;

    protected function getLogPrefix(): string {
        return 'UserService';
    }
}
```

---

## 9. Scala: Trait Linearization

```scala
// Scala: Stackable Modification with traits

trait IntQueue {
  def get(): Int
  def put(x: Int): Unit
}

class BasicIntQueue extends IntQueue {
  import scala.collection.mutable.ArrayBuffer
  private val buf = new ArrayBuffer[Int]

  def get(): Int = buf.remove(0)
  def put(x: Int): Unit = buf += x
}

// Stackable modifications: each trait adds behavior
trait Doubling extends IntQueue {
  abstract override def put(x: Int): Unit = super.put(2 * x)
}

trait Incrementing extends IntQueue {
  abstract override def put(x: Int): Unit = super.put(x + 1)
}

trait Filtering extends IntQueue {
  abstract override def put(x: Int): Unit = {
    if (x >= 0) super.put(x)
    // negative numbers are ignored
  }
}

// Trait composition: applied from right to left
val queue1 = new BasicIntQueue with Incrementing with Filtering
queue1.put(-1)  // Filtering: x >= 0 → ignored
queue1.put(0)   // Filtering: x >= 0 → Incrementing: put(0 + 1) → put(1)
queue1.put(1)   // Filtering: x >= 0 → Incrementing: put(1 + 1) → put(2)
// queue1 contains: [1, 2]

val queue2 = new BasicIntQueue with Filtering with Incrementing
queue2.put(-1)  // Incrementing: put(-1 + 1) → Filtering: 0 >= 0 → put(0)
queue2.put(0)   // Incrementing: put(0 + 1) → Filtering: 1 >= 0 → put(1)
// queue2 contains: [0, 1]

// The linearization order changes the result!
// with Incrementing with Filtering: Filtering first → then Incrementing
// with Filtering with Incrementing: Incrementing first → then Filtering


// Scala: Declaring dependencies with Self-type
trait UserRepository {
  def findUser(id: String): Option[User]
  def saveUser(user: User): Unit
}

trait EmailService {
  def sendEmail(to: String, subject: String, body: String): Unit
}

// Self-type: requires both UserRepository and EmailService
trait UserRegistration {
  self: UserRepository with EmailService =>

  def register(name: String, email: String): User = {
    val user = User(java.util.UUID.randomUUID().toString, name, email)
    saveUser(user)  // method from UserRepository
    sendEmail(email, "Welcome!", s"Welcome, $name!")  // method from EmailService
    user
  }
}

// When implementing, all dependencies must be satisfied
class ProductionApp
    extends UserRegistration
    with UserRepository
    with EmailService {

  private var users = Map[String, User]()

  override def findUser(id: String): Option[User] = users.get(id)
  override def saveUser(user: User): Unit = {
    users += (user.id -> user)
  }
  override def sendEmail(to: String, subject: String, body: String): Unit = {
    println(s"Sending email to $to: $subject")
  }
}

case class User(id: String, name: String, email: String)
```

---

## 10. Swift: Mixins via Protocol Extensions

```swift
// Swift: A mixin-like pattern via Protocol Extension

protocol Identifiable {
    var id: String { get }
}

protocol Timestamped {
    var createdAt: Date { get }
    var updatedAt: Date { get set }
}

// Provide default implementations via Protocol Extension
extension Timestamped {
    mutating func touch() {
        updatedAt = Date()
    }

    var age: TimeInterval {
        return Date().timeIntervalSince(createdAt)
    }

    var isRecent: Bool {
        return age < 86400  // within 24 hours
    }
}

protocol Validatable {
    var validationErrors: [String] { get }
}

extension Validatable {
    var isValid: Bool {
        return validationErrors.isEmpty
    }

    func validateOrThrow() throws {
        guard isValid else {
            throw ValidationError.invalid(validationErrors)
        }
    }
}

enum ValidationError: Error {
    case invalid([String])
}

protocol JSONConvertible: Codable {
    // Just requires Codable
}

extension JSONConvertible {
    func toJSON() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        return String(data: data, encoding: .utf8)!
    }

    static func fromJSON(_ json: String) throws -> Self {
        let decoder = JSONDecoder()
        let data = json.data(using: .utf8)!
        return try decoder.decode(Self.self, from: data)
    }
}

// Protocol composition
struct Article: Identifiable, Timestamped, Validatable, JSONConvertible, Codable {
    let id: String
    var title: String
    var body: String
    let createdAt: Date
    var updatedAt: Date

    var validationErrors: [String] {
        var errors: [String] = []
        if title.isEmpty { errors.append("Title is required") }
        if body.count < 10 { errors.append("Body must be at least 10 characters") }
        return errors
    }
}

// Usage example
var article = Article(
    id: "1",
    title: "Intro to Swift",
    body: "Learn the basics of Swift. Protocols are powerful.",
    createdAt: Date(),
    updatedAt: Date()
)

article.touch()                    // Timestamped
print(article.isRecent)            // Timestamped extension → true
print(article.isValid)             // Validatable extension → true
let json = try article.toJSON()    // JSONConvertible extension
print(json)
```

---

## 11. Mixin Design Principles and Anti-Patterns

### 11.1 Design Principles

```
Mixin design principles:

  1. Single Responsibility Principle (SRP)
     → Each mixin addresses only one cross-cutting concern
     → ❌ LoggingAndCachingMixin → ✅ LoggingMixin + CachingMixin

  2. Prefer Stateless
     → Stateless mixins have fewer side effects and are safer
     → If state is needed, be careful about initialization order
     → Naming convention: mark stateful mixins explicitly (StatefulXxxMixin)

  3. Explicit Dependencies
     → Avoid implicit dependencies (assuming methods of other mixins exist)
     → If needed, declare requirements via abstract methods
     → Python: Protocol, Ruby: abstract method, Rust: trait bounds

  4. Shallow Inheritance Chain
     → Limit mixin inheritance to one level (a mixin should not inherit from another mixin)
     → Consider whether the problem can be solved with composition first

  5. Naming Conventions
     → Python: XxxMixin suffix
     → Ruby: adjective-like names (Serializable, Loggable, Cacheable)
     → TypeScript: verb-like mixin functions (withTimestamp, makeLoggable)
     → PHP: XxxTrait suffix (or Xxxable)
```

### 11.2 Anti-Patterns

```
Anti-pattern 1: God Mixin
  Problem: cramming many features into a single mixin
  Symptoms:
    → mixins over 100 lines
    → multiple unrelated responsibilities
    → many classes that only use part of the mixin
  Solution: split by responsibility

Anti-pattern 2: Implicit Dependencies
  Problem: a mixin assumes other mixins or specific fields exist
  Symptoms:
    → accesses to fields like self.name are not type-checked
    → breaks when mixin order is changed
  Solution: declare dependencies via abstract methods or type-check via protocols

Anti-pattern 3: Mixin Abuse
  Problem: adding every feature as a mixin, obscuring the essence of the class
  Symptoms:
    → class User(A, B, C, D, E, F, G, H, I, J): ...
    → method origins are hard to trace
    → IDEs cannot infer method types
  Solution: limit to 5 or fewer; consider composition

Anti-pattern 4: State Collision
  Problem: multiple mixins add fields with the same name
  Symptoms:
    → self._cache has different meanings across two mixins
    → the result changes based on initialization order
  Solution: use prefixes to namespace them (_logging_cache, _http_cache)

Anti-pattern 5: Diamond Mixin
  Problem: multiple mixins inherit from a common base mixin
  Symptoms:
    → MRO ends up in an unexpected order
    → the super() chain becomes complicated
  Solution: avoid mixin inheritance; keep the structure flat
```

```python
# Concrete examples of anti-patterns and their fixes

# ❌ Anti-pattern: implicit dependencies
class BadLoggingMixin:
    def log(self, message: str) -> None:
        # Implicitly assumes self.name exists
        print(f"[{self.name}] {message}")  # type: ignore


# ✅ Fix: declare the dependency with a Protocol
from typing import Protocol


class HasName(Protocol):
    name: str


class GoodLoggingMixin:
    """Use with a class that implements HasName"""
    def log(self: "HasName", message: str) -> None:
        print(f"[{self.name}] {message}")

    # Or declare the requirement via an abstract method
    # def get_log_prefix(self) -> str:
    #     raise NotImplementedError


# ❌ Anti-pattern: state collision
class CacheMixin1:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = {}  # HTTP cache

class CacheMixin2:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cache = {}  # cache of computation results (collision!)


# ✅ Fix: separate by namespace
class HttpCacheMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._http_cache: dict[str, bytes] = {}

class ComputeCacheMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._compute_cache: dict[str, any] = {}


# ❌ Anti-pattern: mixin abuse
class OverMixedUser(
    JsonMixin, XmlMixin, CsvMixin,       # don't need three serializers
    LoggingMixin, TracingMixin,           # both logging and tracing?
    CacheMixin, MemcacheMixin, RedisMixin,  # three caches?
    ValidatableMixin, SanitizableMixin,
):
    pass  # the essence of the class is completely invisible


# ✅ Fix: minimum necessary mixins + composition
class CleanUser(JsonMixin, LoggableMixin, ValidatableMixin):
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email
        # Caching is handled via composition
        self._cache = CacheService()
```

---

## 12. Testing Strategies

### 12.1 Unit Testing Mixins

```python
# Unit testing strategy for mixins

import pytest
from datetime import datetime


# Mixins under test
class SerializableMixin:
    def to_dict(self) -> dict:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


class ValidatableMixin:
    def validate(self) -> list[str]:
        errors = []
        for attr in dir(self):
            if attr.startswith("validate_"):
                error = getattr(self, attr)()
                if error:
                    errors.append(error)
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# Host classes for testing (minimal classes to exercise the mixin)
class SerializableHost(SerializableMixin):
    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value
        self._internal = "should not be serialized"


class ValidatableHost(ValidatableMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def validate_name(self) -> str | None:
        if not self.name:
            return "Name is required"
        return None

    def validate_age(self) -> str | None:
        if self.age < 0:
            return "Age must be non-negative"
        return None


# Tests
class TestSerializableMixin:
    def test_to_dict_includes_public_attrs(self):
        obj = SerializableHost("test", 42)
        result = obj.to_dict()
        assert result == {"name": "test", "value": 42}

    def test_to_dict_excludes_private_attrs(self):
        obj = SerializableHost("test", 42)
        result = obj.to_dict()
        assert "_internal" not in result

    def test_to_dict_with_empty_object(self):
        class EmptyHost(SerializableMixin):
            pass
        obj = EmptyHost()
        assert obj.to_dict() == {}


class TestValidatableMixin:
    def test_valid_object(self):
        obj = ValidatableHost("Tanaka", 30)
        assert obj.is_valid()
        assert obj.validate() == []

    def test_invalid_name(self):
        obj = ValidatableHost("", 30)
        assert not obj.is_valid()
        errors = obj.validate()
        assert "Name is required" in errors

    def test_invalid_age(self):
        obj = ValidatableHost("Tanaka", -1)
        assert not obj.is_valid()
        errors = obj.validate()
        assert "Age must be non-negative" in errors

    def test_multiple_errors(self):
        obj = ValidatableHost("", -1)
        errors = obj.validate()
        assert len(errors) == 2


# Integration test for multiple mixins
class TestMixinComposition:
    def test_serializable_and_validatable(self):
        class ComposedHost(SerializableMixin, ValidatableMixin):
            def __init__(self, name: str):
                self.name = name

            def validate_name(self) -> str | None:
                if not self.name:
                    return "Name is required"
                return None

        obj = ComposedHost("test")
        assert obj.is_valid()
        assert obj.to_dict() == {"name": "test"}
```

### 12.2 Testing the MRO

```python
# Verify that the MRO order is correct

class TestMROOrder:
    def test_diamond_mro(self):
        """Verify the MRO order for diamond inheritance"""
        class A:
            def method(self): return "A"

        class B(A):
            def method(self): return "B"

        class C(A):
            def method(self): return "C"

        class D(B, C):
            pass

        # Verify the MRO order
        assert D.__mro__ == (D, B, C, A, object)
        # The first one found is B
        assert D().method() == "B"

    def test_cooperative_super_chain(self):
        """Verify the order of the cooperative super() chain"""
        call_order = []

        class Base:
            def process(self):
                call_order.append("Base")

        class MixinA(Base):
            def process(self):
                call_order.append("MixinA")
                super().process()

        class MixinB(Base):
            def process(self):
                call_order.append("MixinB")
                super().process()

        class Combined(MixinA, MixinB):
            def process(self):
                call_order.append("Combined")
                super().process()

        Combined().process()
        assert call_order == ["Combined", "MixinA", "MixinB", "Base"]

    def test_mixin_initialization_order(self):
        """Verify the mixin initialization order"""
        init_order = []

        class Base:
            def __init__(self, **kwargs):
                init_order.append("Base")

        class MixinA(Base):
            def __init__(self, **kwargs):
                init_order.append("MixinA")
                super().__init__(**kwargs)

        class MixinB(Base):
            def __init__(self, **kwargs):
                init_order.append("MixinB")
                super().__init__(**kwargs)

        class Combined(MixinA, MixinB):
            def __init__(self):
                init_order.append("Combined")
                super().__init__()

        Combined()
        assert init_order == ["Combined", "MixinA", "MixinB", "Base"]
```

```typescript
// TypeScript: testing mixins

describe("Timestamped mixin", () => {
  type Constructor<T = {}> = new (...args: any[]) => T;

  function Timestamped<TBase extends Constructor>(Base: TBase) {
    return class extends Base {
      createdAt = new Date();
      updatedAt = new Date();
      touch() { this.updatedAt = new Date(); }
    };
  }

  class BaseEntity {
    constructor(public id: string) {}
  }

  const TimestampedEntity = Timestamped(BaseEntity);

  it("should add createdAt and updatedAt", () => {
    const entity = new TimestampedEntity("1");
    expect(entity.createdAt).toBeInstanceOf(Date);
    expect(entity.updatedAt).toBeInstanceOf(Date);
  });

  it("should update updatedAt on touch", () => {
    const entity = new TimestampedEntity("1");
    const original = entity.updatedAt;

    // Wait a bit, then touch
    jest.advanceTimersByTime(1000);
    entity.touch();

    expect(entity.updatedAt.getTime()).toBeGreaterThanOrEqual(
      original.getTime()
    );
  });

  it("should preserve base class properties", () => {
    const entity = new TimestampedEntity("test-id");
    expect(entity.id).toBe("test-id");
  });

  it("should work with multiple mixins", () => {
    function Activatable<TBase extends Constructor>(Base: TBase) {
      return class extends Base {
        isActive = false;
        activate() { this.isActive = true; }
      };
    }

    const FullEntity = Activatable(Timestamped(BaseEntity));
    const entity = new FullEntity("1");

    // Both mixins work
    expect(entity.createdAt).toBeInstanceOf(Date);
    expect(entity.isActive).toBe(false);
    entity.activate();
    expect(entity.isActive).toBe(true);
  });
});
```

---

## 13. Cross-Language Comparison and Selection Guidelines

```
Summary of mixin / multiple-inheritance comparisons:

  Python:
    Approach: multiple inheritance + MRO + super()
    Pros: flexible, enables cooperative multiple inheritance
    Cons: requires understanding MRO, risk of runtime errors
    Recommendation: 5 or fewer mixins, XxxMixin naming convention

  Ruby:
    Approach: Module include / extend / prepend
    Pros: natural syntax, flexible conflict resolution
    Cons: no type checking, prepend behavior can be non-intuitive
    Recommendation: one responsibility per Module, safe access with respond_to?

  TypeScript:
    Approach: class-expression mixins / decorators
    Pros: type-safe, flexible composition
    Cons: type inference can become complex
    Recommendation: constrained mixins (GConstructor pattern)

  Rust:
    Approach: traits + blanket implementations
    Pros: compile-time safety, zero-cost abstraction
    Cons: less flexible (dynamic composition is limited)
    Recommendation: declare constraints with trait bounds, use Derive for auto-impl

  Java:
    Approach: default methods on interfaces
    Pros: backward compatible, type-safe
    Cons: cannot hold state, conflict resolution is cumbersome
    Recommendation: restrict default methods to utility-style use

  Kotlin:
    Approach: delegation via the `by` keyword
    Pros: concise syntax for composition
    Cons: requires a field for the delegate
    Recommendation: combine delegation + extension functions

  Scala:
    Approach: trait linearization + Self-type
    Pros: can hold state, stackable modification pattern
    Cons: linearization order can be non-intuitive
    Recommendation: declare dependencies with Self-type; the cake pattern can be replaced by DI containers

  PHP:
    Approach: trait + insteadof / as
    Pros: explicit conflict resolution, simple
    Cons: weak type checking
    Recommendation: declare requirements with abstract methods

  Swift:
    Approach: Protocol Extension
    Pros: type-safe, works with value types, conditional conformance
    Cons: cannot hold state (Protocol Extension itself)
    Recommendation: Protocol composition + Extension with default implementations
```

```
Selection flowchart:

  Q1: "Do you need code reuse?"
  │
  ├── No → a regular class design is enough
  │
  └── Yes
      │
      Q2: "Is the functionality you want to reuse a cross-cutting concern?"
      │    (logging, caching, authentication, validation, etc.)
      │
      ├── Yes → mixin / trait
      │         Q2a: "Does it need to hold state?"
      │         ├── Yes → Python Mixin, Scala Trait, PHP Trait
      │         └── No → interface + default implementation
      │
      └── No
          │
          Q3: "Is an is-a relationship appropriate?"
          │
          ├── Yes → inheritance
          └── No → composition (delegation)
```

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Beyond theory, understanding deepens by actually writing code and verifying its behavior.

### Q2: What mistakes do beginners commonly make?

Skipping the fundamentals and jumping to advanced topics. It is recommended that you firmly understand the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in day-to-day development work. It becomes especially important during code reviews and architectural design.

---

## Summary

| Language | Approach | Characteristics |
|------|------|------|
| Python | Multiple inheritance + MRO | Order resolved by C3 linearization |
| Ruby | Module include | The most natural mixin |
| TypeScript | Class-expression composition | Type-safe mixins |
| Rust | Traits | No multiple inheritance. Composition via traits |
| Java | Default methods | Adds implementations to interfaces |
| Kotlin | by delegation | Concise syntax for composition |
| Scala | Trait linearization | Stackable modification pattern |
| PHP | trait | Explicit conflict resolution (insteadof/as) |
| Swift | Protocol Extension | Type-safe protocol-oriented style |

```
Practical guidelines:

  1. Use 5 or fewer mixins
     → If you need more, rethink the design
     → The essential responsibility of the class becomes obscured

  2. Each mixin should have a single responsibility
     → LoggingMixin + CachingMixin (separate) ✅
     → LoggingAndCachingMixin (combined) ❌

  3. Minimize mixins that hold state
     → Stateless mixins are safer and more predictable
     → If state is needed, consider composition

  4. Make dependencies explicit
     → Avoid implicit dependencies
     → Make requirements clear via abstract methods or protocols

  5. Ensure testability
     → Each mixin should be independently testable
     → Prepare host classes for testing
     → Don't forget to test MRO and composition order too

  6. Standardize naming conventions
     → Python: XxxMixin
     → Ruby: -able suffix
     → TypeScript: with/make prefix
     → PHP: XxxTrait or -able
```

---

## Guides to Read Next

---

## References
1. Barrett, S. "C3 Linearization." 1996.
2. Bracha, G. "The Programming Language Jigsaw." 1992.
3. Gamma, E. et al. "Design Patterns." Addison-Wesley, 1994.
4. Bloch, J. "Effective Java." 3rd Edition, Addison-Wesley, 2018.
5. Odersky, M. "Programming in Scala." Artima, 2021.
6. The Rust Programming Language. "Traits." doc.rust-lang.org.
7. Python Documentation. "Multiple Inheritance." docs.python.org.
8. Ruby Documentation. "Modules." ruby-doc.org.
