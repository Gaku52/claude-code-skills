# Branching and Loops

> Control flow is the mechanism that "changes the execution order" of a program. The design of branching and loops reflects the philosophy of a language.

## What You Will Learn in This Chapter

- [ ] Understand the differences and design philosophies of branching syntax across languages
- [ ] Grasp the types of loops and when to use each
- [ ] Understand expression-based control flow
- [ ] Properly use early returns and guard clauses
- [ ] Grasp loop optimization techniques
- [ ] Compare control flow design decisions across languages


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Branching

### 1.1 if Statement vs if Expression

The design of `if` in programming languages differs significantly depending on whether it is treated as a "statement" or an "expression." Statements are executed for their side effects, while expressions return values.

```python
# Python: if is a statement
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

# Ternary operator (expression) — conditional expression
grade = "A" if score >= 90 else "B" if score >= 80 else "C"

# Practical uses of the ternary operator
status = "active" if user.is_verified else "pending"
display_name = user.nickname if user.nickname else user.email
max_val = a if a > b else b

# Avoid ternary operators for complex conditions; use regular if statements
# Bad — hard to read
result = "A" if x > 90 else "B" if x > 80 else "C" if x > 70 else "D" if x > 60 else "F"

# Good — easy to read
if x > 90:
    result = "A"
elif x > 80:
    result = "B"
elif x > 70:
    result = "C"
elif x > 60:
    result = "D"
else:
    result = "F"
```

```rust
// Rust: if is an expression — returns a value
let grade = if score >= 90 {
    "A"
} else if score >= 80 {
    "B"
} else if score >= 70 {
    "C"
} else if score >= 60 {
    "D"
} else {
    "F"
};  // Bound with a semicolon

// Can be used in a single line (for short conditions)
let abs_val = if x >= 0 { x } else { -x };
let min_val = if a < b { a } else { b };

// When a block contains multiple statements, the last expression becomes the value
let description = if score >= 90 {
    let prefix = "Excellent";
    let emoji = "🌟";
    format!("{} {}", prefix, emoji)  // This is the returned value
} else {
    "Keep trying".to_string()
};

// if let — shorthand for pattern matching on Option or Result
let config_value = if let Some(val) = config.get("timeout") {
    val.parse::<u64>().unwrap_or(30)
} else {
    30  // Default value
};
```

```kotlin
// Kotlin: both if and when are expressions
val grade = if (score >= 90) "A" else if (score >= 80) "B" else "C"

// For multi-line blocks, the last expression becomes the value
val result = if (score >= 90) {
    println("Great job!")
    "A"  // This value is returned
} else {
    println("Keep going")
    "B"
}

// when expression (Kotlin's powerful branching construct)
val result = when {
    score >= 90 -> "A"
    score >= 80 -> "B"
    score >= 70 -> "C"
    score >= 60 -> "D"
    else -> "F"
}

// Using when with value-based matching
val typeDescription = when (val day = getDayOfWeek()) {
    "Mon", "Tue", "Wed", "Thu", "Fri" -> "Weekday"
    "Sat", "Sun" -> "Weekend"
    else -> "Unknown"
}

// Type checking with when
fun describe(obj: Any): String = when (obj) {
    is Int -> "Integer: $obj"
    is String -> "String of length ${obj.length}"
    is List<*> -> "List of size ${obj.size}"
    else -> "Unknown type"
}

// Range checking with when
val category = when (age) {
    in 0..12 -> "Child"
    in 13..17 -> "Teen"
    in 18..64 -> "Adult"
    in 65..Int.MAX_VALUE -> "Senior"
    else -> "Invalid"
}
```

```swift
// Swift: if is a statement, but if let / guard let are powerful
let score = 85

// if let (Optional binding)
if let username = optionalUsername {
    print("Hello, \(username)")
} else {
    print("No username")
}

// guard let (designed for early returns)
func processUser(_ user: User?) -> String {
    guard let user = user else {
        return "No user"
    }
    guard user.isActive else {
        return "Inactive user"
    }
    return "Processing \(user.name)"
}

// Swift 5.9+: if can be used as an expression
let grade = if score >= 90 { "A" } else if score >= 80 { "B" } else { "C" }
```

```scala
// Scala: if is an expression (always returns a value)
val grade = if (score >= 90) "A"
            else if (score >= 80) "B"
            else if (score >= 70) "C"
            else "F"

// Block expression
val result = if (condition) {
  val computed = heavyComputation()
  computed * 2  // The last expression is the returned value
} else {
  defaultValue
}
```

```haskell
-- Haskell: if is an expression (else is required)
grade = if score >= 90 then "A"
        else if score >= 80 then "B"
        else if score >= 70 then "C"
        else "F"

-- Guards (conditional branching in function definitions)
bmiCategory bmi
  | bmi < 18.5 = "Underweight"
  | bmi < 25.0 = "Normal"
  | bmi < 30.0 = "Overweight"
  | otherwise   = "Obese"
```

### 1.2 Statement vs Expression Design Philosophy Comparison

```
Statement-based languages:
  C, Java, Python, JavaScript, Go
  → if does not return a value. Results are communicated through side effects (assignment, etc.)
  → The ternary operator (? :) supplements branching as an expression

Expression-based languages:
  Rust, Kotlin, Scala, Haskell, OCaml, F#, Elixir
  → if returns a value. Naturally combines with variable bindings
  → Consistency of "every construct returns a value"

Advantages of expression-based:
  1. Easier to maintain variable immutability (assign once with let x = if ...)
  2. No risk of forgetting initialization (missing else causes a compile error)
  3. High affinity with functional style
  4. Type inference works well

Advantages of statement-based:
  1. Familiar (tradition from C)
  2. Side effects can be explicitly separated
  3. void (no-value) branching feels natural
```

### 1.3 switch / match

```javascript
// JavaScript: switch statement (beware of fall-through)
switch (day) {
    case "Mon": case "Tue": case "Wed":
    case "Thu": case "Fri":
        type = "Weekday";
        break;      // Forgetting break → fall-through (source of bugs)
    case "Sat": case "Sun":
        type = "Weekend";
        break;
    default:
        type = "Unknown";
}

// Practical switch usage: classifying HTTP status codes
function categorizeStatus(code) {
    switch (true) {
        case code >= 200 && code < 300:
            return "Success";
        case code >= 300 && code < 400:
            return "Redirect";
        case code >= 400 && code < 500:
            return "Client Error";
        case code >= 500:
            return "Server Error";
        default:
            return "Informational";
    }
}

// Alternative to switch: object map (recommended pattern)
const statusMessages = {
    200: "OK",
    201: "Created",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    500: "Internal Server Error",
};
const message = statusMessages[code] ?? "Unknown Status";
```

```typescript
// TypeScript: type narrowing with switch
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            // Here shape is { kind: "circle"; radius: number }
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            // Here shape is { kind: "rectangle"; width: number; height: number }
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
        default:
            // Exhaustiveness check (never type)
            const _exhaustive: never = shape;
            return _exhaustive;
    }
}
```

```rust
// Rust: match expression (exhaustiveness check + pattern matching)
let type_str = match day {
    "Mon" | "Tue" | "Wed" | "Thu" | "Fri" => "Weekday",
    "Sat" | "Sun" => "Weekend",
    _ => "Unknown",
};
// Compile error if not all patterns are covered
// No fall-through (safe)

// Complex conditional branching with match
let message = match status_code {
    200 => "OK",
    201 => "Created",
    204 => "No Content",
    301 | 302 => "Redirect",
    400 => "Bad Request",
    401 => "Unauthorized",
    403 => "Forbidden",
    404 => "Not Found",
    405 => "Method Not Allowed",
    500 => "Internal Server Error",
    502 | 503 => "Service Unavailable",
    code @ 100..=199 => "Informational",
    code @ 200..=299 => "Success",
    code @ 300..=399 => "Redirection",
    code @ 400..=499 => "Client Error",
    code @ 500..=599 => "Server Error",
    _ => "Unknown",
};
```

```go
// Go: switch (no break needed, fall-through is explicit)
switch day {
case "Mon", "Tue", "Wed", "Thu", "Fri":
    typeStr = "Weekday"
case "Sat", "Sun":
    typeStr = "Weekend"
default:
    typeStr = "Unknown"
}
// No break needed (exits automatically)
// Use the fallthrough keyword for explicit fall-through

// switch without a condition expression (alternative to if-else chain)
switch {
case score >= 90:
    grade = "A"
case score >= 80:
    grade = "B"
case score >= 70:
    grade = "C"
case score >= 60:
    grade = "D"
default:
    grade = "F"
}

// Type switch (Go's interface type assertion)
func describe(i interface{}) string {
    switch v := i.(type) {
    case int:
        return fmt.Sprintf("Integer: %d", v)
    case string:
        return fmt.Sprintf("String: %s", v)
    case bool:
        return fmt.Sprintf("Boolean: %t", v)
    case []int:
        return fmt.Sprintf("Int slice of length %d", len(v))
    default:
        return fmt.Sprintf("Unknown type: %T", v)
    }
}
```

```c
// C: switch statement (integer types only, fall-through is the default)
switch (command) {
    case CMD_START:
        initialize();
        break;
    case CMD_STOP:
        cleanup();
        break;
    case CMD_PAUSE:
    case CMD_SUSPEND:  // Fall-through (intentional)
        pause();
        break;
    default:
        fprintf(stderr, "Unknown command: %d\n", command);
        break;
}

// Limitations of C's switch
// - Integer types only (int, char, enum)
// - Cannot compare strings
// - Cannot specify ranges (except with GCC extensions)
```

```java
// Java: switch expression (Java 14+)
// Traditional switch statement
String typeStr;
switch (day) {
    case "Mon": case "Tue": case "Wed":
    case "Thu": case "Fri":
        typeStr = "Weekday";
        break;
    case "Sat": case "Sun":
        typeStr = "Weekend";
        break;
    default:
        typeStr = "Unknown";
}

// Java 14+: switch expression (arrow syntax)
String typeStr = switch (day) {
    case "Mon", "Tue", "Wed", "Thu", "Fri" -> "Weekday";
    case "Sat", "Sun" -> "Weekend";
    default -> "Unknown";
};

// Java 14+: switch expression with blocks (return values with yield)
String result = switch (statusCode) {
    case 200 -> "OK";
    case 404 -> "Not Found";
    default -> {
        logger.warn("Unexpected status: " + statusCode);
        yield "Unknown";
    }
};

// Java 21+: pattern matching switch
String describe(Object obj) {
    return switch (obj) {
        case Integer i when i > 0 -> "Positive integer: " + i;
        case Integer i -> "Non-positive integer: " + i;
        case String s -> "String: " + s;
        case null -> "null";
        default -> "Unknown: " + obj;
    };
}
```

### 1.4 Conditional Operators and Conditional Expressions

```python
# Python: conditional expression (equivalent to ternary operator)
result = value_if_true if condition else value_if_false

# Practical examples
display = f"{count} item{'s' if count != 1 else ''}"
log_level = "DEBUG" if is_development else "INFO"
timeout = custom_timeout if custom_timeout is not None else default_timeout

# Python: Walrus operator (:=) — Python 3.8+
# Perform assignment and condition check simultaneously
if (n := len(data)) > 10:
    print(f"Data too long: {n}")

# Usage in while loops
while (line := file.readline()) != "":
    process(line)

# Usage in list comprehensions
results = [y for x in data if (y := expensive_computation(x)) is not None]
```

```c
// C / C++ / JavaScript / Java: ternary operator
int abs_val = (x >= 0) ? x : -x;
const char* msg = (err == 0) ? "Success" : "Error";

// Nested ternary operators (not recommended — hard to read)
const char* grade = (score >= 90) ? "A"
                  : (score >= 80) ? "B"
                  : (score >= 70) ? "C"
                  : "F";
```

```ruby
# Ruby: diverse conditional expressions
# if modifier (postfix if)
puts "Adult" if age >= 18
log.warn("Low memory") if memory_usage > 0.9

# unless (negated condition)
raise "Not found" unless record
send_notification unless user.opted_out?

# Ternary operator
status = active? ? "Active" : "Inactive"

# case-when (pattern-match style)
result = case score
         when 90..100 then "A"
         when 80..89  then "B"
         when 70..79  then "C"
         when 60..69  then "D"
         else              "F"
         end

# case-in (Ruby 3.0+ pattern matching)
case user
in { name: String => name, age: (18..) => age }
  puts "#{name} is an adult (#{age})"
in { name: String => name, age: }
  puts "#{name} is a minor (#{age})"
end
```

---

## 2. Loops

### 2.1 Evolution of for Loops

```c
// C: classic for loop
for (int i = 0; i < 10; i++) {
    printf("%d\n", i);
}

// C: traversing an array (index-based)
int arr[] = {10, 20, 30, 40, 50};
int len = sizeof(arr) / sizeof(arr[0]);
for (int i = 0; i < len; i++) {
    printf("arr[%d] = %d\n", i, arr[i]);
}

// C: reverse traversal
for (int i = len - 1; i >= 0; i--) {
    printf("arr[%d] = %d\n", i, arr[i]);
}

// C: nested loops (matrix processing)
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        matrix[i][j] = i * cols + j;
    }
}
```

```python
# Python: for-in (iterator-based)
for i in range(10):
    print(i)

for item in collection:
    print(item)

# enumerate (with index)
for i, item in enumerate(collection):
    print(f"{i}: {item}")

# Specifying the starting index for enumerate
for i, line in enumerate(lines, start=1):
    print(f"Line {i}: {line}")

# zip (parallel iteration)
for name, age in zip(names, ages):
    print(f"{name}: {age}")

# zip_longest (combining iterables of different lengths)
from itertools import zip_longest
for a, b in zip_longest([1, 2, 3], [10, 20], fillvalue=0):
    print(f"{a}, {b}")  # (1,10), (2,20), (3,0)

# reversed (reverse order)
for item in reversed(collection):
    print(item)

# sorted (sorted order)
for item in sorted(collection, key=lambda x: x.name):
    print(item)

# Leveraging itertools
from itertools import product, combinations, permutations, chain

# Cartesian product (all combinations)
for x, y in product(range(3), range(3)):
    print(f"({x}, {y})")

# Combinations
for a, b in combinations([1, 2, 3, 4], 2):
    print(f"{a}, {b}")  # (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

# Permutations
for a, b in permutations([1, 2, 3], 2):
    print(f"{a}, {b}")  # (1,2), (1,3), (2,1), (2,3), (3,1), (3,2)

# Chain (concatenating multiple iterables)
for item in chain([1, 2], [3, 4], [5, 6]):
    print(item)  # 1, 2, 3, 4, 5, 6

# Iterating over dictionaries
for key, value in config.items():
    print(f"{key} = {value}")

# Dictionary comprehension (loop + transformation)
squared = {x: x**2 for x in range(10)}
filtered = {k: v for k, v in data.items() if v > threshold}
```

```rust
// Rust: for-in (ownership-aware)
for item in &collection {     // Immutable borrow (collection preserved)
    println!("{}", item);
}

for item in &mut collection { // Mutable borrow (modify elements)
    *item += 1;
}

for item in collection {      // Move (collection consumed)
    println!("{}", item);
}
// collection can no longer be used

// Ranges
for i in 0..10 {        // 0 to 9
    println!("{}", i);
}
for i in 0..=10 {       // 0 to 10 (inclusive)
    println!("{}", i);
}

// Stepping (step_by)
for i in (0..100).step_by(5) {
    println!("{}", i);  // 0, 5, 10, ..., 95
}

// Reverse
for i in (0..10).rev() {
    println!("{}", i);  // 9, 8, 7, ..., 0
}

// enumerate (with index)
for (i, item) in collection.iter().enumerate() {
    println!("{}: {}", i, item);
}

// zip (parallel iteration)
for (name, age) in names.iter().zip(ages.iter()) {
    println!("{}: {}", name, age);
}

// windows and chunks
let data = vec![1, 2, 3, 4, 5, 6, 7, 8];

// Sliding window
for window in data.windows(3) {
    println!("{:?}", window);  // [1,2,3], [2,3,4], [3,4,5], ...
}

// Chunk splitting
for chunk in data.chunks(3) {
    println!("{:?}", chunk);  // [1,2,3], [4,5,6], [7,8]
}
```

```go
// Go: only for (while and loop are also expressed with for)
for i := 0; i < 10; i++ {  // C-style for
    fmt.Println(i)
}

for condition {              // Equivalent to while
    // ...
}

for {                        // Infinite loop
    // ...
}

for i, v := range slice {   // for-range (slice)
    fmt.Println(i, v)
}

// for-range (map)
for key, value := range myMap {
    fmt.Printf("%s: %v\n", key, value)
}

// for-range (string) — iterates by rune (Unicode code point)
for i, r := range "Hello, 世界" {
    fmt.Printf("index=%d, rune=%c\n", i, r)
}

// for-range (channel)
for msg := range ch {
    fmt.Println("Received:", msg)
}

// Index only (discard value)
for i := range slice {
    fmt.Println(i)
}

// Value only (discard index)
for _, v := range slice {
    fmt.Println(v)
}
```

```javascript
// JavaScript: 4 types of for loops

// 1. Classic for
for (let i = 0; i < 10; i++) {
    console.log(i);
}

// 2. for...in (iterates over object keys — not recommended for arrays)
for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
        console.log(`${key}: ${obj[key]}`);
    }
}

// 3. for...of (iterates over iterable values — ES6+)
for (const item of array) {
    console.log(item);
}

// 4. forEach method (array-only)
array.forEach((item, index) => {
    console.log(`${index}: ${item}`);
});

// Applications of for...of
// Iterating over a Map
for (const [key, value] of map) {
    console.log(`${key}: ${value}`);
}

// Iterating over a Set
for (const item of set) {
    console.log(item);
}

// entries() (with index)
for (const [i, item] of array.entries()) {
    console.log(`${i}: ${item}`);
}

// Note: for...in vs for...of
// for...in: also enumerates prototype chain properties (dangerous)
// for...of: follows the Symbol.iterator protocol (safe)
```

### 2.2 while and loop

```rust
// Rust: loop (infinite loop that can return a value)
let result = loop {
    let input = get_input();
    if input.is_valid() {
        break input.value();  // Return a value with break
    }
    println!("Invalid input, try again");
};

// Labeled loops (controlling nested loops)
'outer: for i in 0..10 {
    for j in 0..10 {
        if i + j > 15 {
            break 'outer;  // Break out of the outer loop
        }
        if j % 2 == 0 {
            continue;  // Skip to the next iteration of the inner loop
        }
        println!("{}, {}", i, j);
    }
}

// while let (with pattern matching)
while let Some(item) = iterator.next() {
    println!("{}", item);
}

// while let chain (Rust 1.64+)
while let Some(item) = stack.pop() {
    match item {
        Item::Value(v) => results.push(v),
        Item::SubList(list) => {
            for sub_item in list.into_iter().rev() {
                stack.push(sub_item);
            }
        }
    }
}

// loop + match pattern (state machine implementation)
enum State { Init, Running, Paused, Done }

let mut state = State::Init;
loop {
    state = match state {
        State::Init => {
            initialize();
            State::Running
        }
        State::Running => {
            if should_pause() {
                State::Paused
            } else if is_complete() {
                State::Done
            } else {
                process_next();
                State::Running
            }
        }
        State::Paused => {
            wait_for_resume();
            State::Running
        }
        State::Done => break,
    };
}
```

```python
# Python: while loops

# Basic while
count = 0
while count < 10:
    print(count)
    count += 1

# while + else (else executes when the loop completes normally)
def find_item(items, target):
    i = 0
    while i < len(items):
        if items[i] == target:
            print(f"Found at index {i}")
            break
        i += 1
    else:
        # Executes when not exited via break
        print("Not found")

# do-while equivalent (Python has no do-while)
while True:
    user_input = input("Enter a number (0 to quit): ")
    if user_input == "0":
        break
    process(user_input)

# Consuming an iterator with while
it = iter(data)
while (chunk := list(islice(it, 100))):
    process_batch(chunk)
```

```go
// Go: all loop types expressed with for

// Equivalent to while
for condition {
    // ...
}

// Equivalent to do-while
for {
    doSomething()
    if !condition {
        break
    }
}

// Labeled loops
OuterLoop:
    for i := 0; i < 10; i++ {
        for j := 0; j < 10; j++ {
            if i+j > 15 {
                break OuterLoop
            }
        }
    }
```

```java
// Java: do-while (executes at least once)
Scanner scanner = new Scanner(System.in);
String input;
do {
    System.out.print("Enter command: ");
    input = scanner.nextLine();
    processCommand(input);
} while (!input.equals("quit"));

// Java: enhanced for loop (for-each)
for (String item : collection) {
    System.out.println(item);
}

// Java: labeled break/continue
outer:
for (int i = 0; i < matrix.length; i++) {
    for (int j = 0; j < matrix[i].length; j++) {
        if (matrix[i][j] == target) {
            System.out.printf("Found at (%d, %d)%n", i, j);
            break outer;
        }
    }
}
```

### 2.3 Iterator Methods (Functional Style)

```typescript
// TypeScript: method chaining
const result = numbers
    .filter(n => n > 0)
    .map(n => n * 2)
    .reduce((sum, n) => sum + n, 0);

// vs imperative style
let result = 0;
for (const n of numbers) {
    if (n > 0) {
        result += n * 2;
    }
}

// Practical method chaining example
interface User {
    name: string;
    age: number;
    department: string;
    isActive: boolean;
}

// Count active users per department
const departmentCounts = users
    .filter(user => user.isActive)
    .reduce((acc, user) => {
        acc[user.department] = (acc[user.department] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

// Grouping (Object.groupBy — ES2024)
const grouped = Object.groupBy(users, user => user.department);

// Using flatMap
const allTags = articles
    .flatMap(article => article.tags)
    .filter((tag, i, arr) => arr.indexOf(tag) === i);  // Remove duplicates

// find and findIndex
const firstAdmin = users.find(u => u.role === "admin");
const adminIndex = users.findIndex(u => u.role === "admin");

// some and every
const hasAdmin = users.some(u => u.role === "admin");
const allActive = users.every(u => u.isActive);
```

```rust
// Rust: iterators (zero-cost abstraction)
let result: i32 = numbers.iter()
    .filter(|&&n| n > 0)
    .map(|&n| n * 2)
    .sum();
// After compilation, performance is equivalent to a hand-written loop

// Practical iterator example
struct Employee {
    name: String,
    department: String,
    salary: u64,
}

// Average salary per department
let dept_averages: HashMap<String, f64> = employees.iter()
    .fold(HashMap::new(), |mut acc, emp| {
        let entry = acc.entry(emp.department.clone())
            .or_insert((0u64, 0u64));
        entry.0 += emp.salary;
        entry.1 += 1;
        acc
    })
    .into_iter()
    .map(|(dept, (total, count))| {
        (dept, total as f64 / count as f64)
    })
    .collect();

// partition: split into two based on a condition
let (evens, odds): (Vec<i32>, Vec<i32>) = numbers.iter()
    .partition(|&&n| n % 2 == 0);

// unzip: split an iterator of pairs into two collections
let (names, ages): (Vec<&str>, Vec<u32>) = people.iter()
    .map(|p| (p.name.as_str(), p.age))
    .unzip();

// Efficient search leveraging lazy evaluation of chained operations
let first_match = huge_dataset.iter()
    .filter(|item| expensive_check(item))
    .map(|item| transform(item))
    .next();  // Only computes the first one (rest is not evaluated)
```

```python
# Python: comprehensions (Pythonic loops)

# List comprehension
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]

# Nested comprehension
pairs = [(x, y) for x in range(3) for y in range(3) if x != y]

# Dictionary comprehension
word_lengths = {word: len(word) for word in words}

# Set comprehension
unique_lengths = {len(word) for word in words}

# Generator expression (memory-efficient)
total = sum(x**2 for x in range(1000000))

# map, filter, reduce (functional style)
from functools import reduce

squared = list(map(lambda x: x**2, numbers))
positive = list(filter(lambda x: x > 0, numbers))
total = reduce(lambda acc, x: acc + x, numbers, 0)

# In practice, comprehensions are more Pythonic
# List comprehensions are preferred over map/filter
squared = [x**2 for x in numbers]              # Equivalent to map
positive = [x for x in numbers if x > 0]       # Equivalent to filter
```

---

## 3. Early Returns and Guard Clauses

### 3.1 Guard Clause Pattern

```rust
// Guard clause pattern (reduces nesting)
// Bad — deeply nested
fn process(input: Option<&str>) -> Result<String, Error> {
    if let Some(s) = input {
        if !s.is_empty() {
            if s.len() < 100 {
                Ok(s.to_uppercase())
            } else {
                Err(Error::TooLong)
            }
        } else {
            Err(Error::Empty)
        }
    } else {
        Err(Error::Missing)
    }
}

// Good — early return with guard clauses
fn process(input: Option<&str>) -> Result<String, Error> {
    let s = input.ok_or(Error::Missing)?;
    if s.is_empty() { return Err(Error::Empty); }
    if s.len() >= 100 { return Err(Error::TooLong); }
    Ok(s.to_uppercase())
}
```

```python
# Python: guard clauses

# Bad — deeply nested
def process_order(order):
    if order is not None:
        if order.is_valid():
            if order.has_items():
                if order.payment_verified():
                    return ship_order(order)
                else:
                    return {"error": "Payment not verified"}
            else:
                return {"error": "No items"}
        else:
            return {"error": "Invalid order"}
    else:
        return {"error": "No order"}

# Good — early return with guard clauses
def process_order(order):
    if order is None:
        return {"error": "No order"}
    if not order.is_valid():
        return {"error": "Invalid order"}
    if not order.has_items():
        return {"error": "No items"}
    if not order.payment_verified():
        return {"error": "Payment not verified"}
    return ship_order(order)
```

```go
// Go: guard clauses are the standard style
func processUser(userID string) (*User, error) {
    if userID == "" {
        return nil, fmt.Errorf("empty user ID")
    }

    user, err := db.FindUser(userID)
    if err != nil {
        return nil, fmt.Errorf("find user %s: %w", userID, err)
    }

    if !user.IsActive {
        return nil, fmt.Errorf("user %s is not active", userID)
    }

    if user.IsLocked {
        return nil, fmt.Errorf("user %s is locked", userID)
    }

    return user, nil
}
```

```typescript
// TypeScript: guard clauses + type narrowing
function processInput(input: unknown): string {
    if (input === null || input === undefined) {
        return "No input";
    }
    if (typeof input !== "string") {
        return "Not a string";
    }
    // Here input is narrowed to the string type
    if (input.length === 0) {
        return "Empty string";
    }
    if (input.length > 100) {
        return "Too long";
    }
    return input.toUpperCase();
}
```

### 3.2 continue and break Within Loops

```python
# continue: skip the current iteration
for item in items:
    if not item.is_valid():
        continue  # Skip invalid items
    if item.is_deleted():
        continue  # Skip deleted items
    process(item)

# break: exit the loop
for item in items:
    if item.matches(target):
        result = item
        break
else:
    # Executes when not exited via break (i.e., not found)
    result = None
```

```rust
// Rust: labeled break/continue
'search: for row in &matrix {
    for &cell in row {
        if cell == target {
            println!("Found: {}", cell);
            break 'search;  // Break out of the outer loop
        }
    }
}

// Return a value with break (for loop)
let found = 'outer: loop {
    for item in &collection {
        if item.matches(&criteria) {
            break 'outer Some(item);  // Return a value from the outer loop
        }
    }
    break None;  // Not found
};
```

---

## 4. Advanced Control Flow Patterns

### 4.1 Table-Driven Dispatch

```python
# Alternative to if-elif chains: dictionary dispatch
def handle_command(command: str, args: list[str]) -> str:
    handlers = {
        "help": lambda args: show_help(),
        "list": lambda args: list_items(),
        "add": lambda args: add_item(args[0]) if args else "Missing argument",
        "remove": lambda args: remove_item(args[0]) if args else "Missing argument",
        "search": lambda args: search_items(" ".join(args)),
    }

    handler = handlers.get(command)
    if handler is None:
        return f"Unknown command: {command}"
    return handler(args)
```

```typescript
// TypeScript: dispatch map
type Handler = (req: Request) => Response;

const routes: Record<string, Handler> = {
    "/api/users": handleUsers,
    "/api/posts": handlePosts,
    "/api/comments": handleComments,
};

function router(req: Request): Response {
    const handler = routes[req.path];
    if (!handler) {
        return new Response("Not Found", { status: 404 });
    }
    return handler(req);
}
```

```go
// Go: dispatch table
type CommandHandler func(args []string) error

var commands = map[string]CommandHandler{
    "start":   handleStart,
    "stop":    handleStop,
    "status":  handleStatus,
    "restart": handleRestart,
}

func dispatch(name string, args []string) error {
    handler, ok := commands[name]
    if !ok {
        return fmt.Errorf("unknown command: %s", name)
    }
    return handler(args)
}
```

### 4.2 State Machine Pattern

```rust
// Rust: state machine using enums
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32, max_attempts: u32 },
    Connected { session_id: String },
    Disconnecting,
}

fn handle_event(state: ConnectionState, event: Event) -> ConnectionState {
    match (state, event) {
        (ConnectionState::Disconnected, Event::Connect) => {
            ConnectionState::Connecting { attempt: 1, max_attempts: 5 }
        }
        (ConnectionState::Connecting { attempt, max_attempts }, Event::Success(session)) => {
            ConnectionState::Connected { session_id: session }
        }
        (ConnectionState::Connecting { attempt, max_attempts }, Event::Failure) => {
            if attempt < max_attempts {
                ConnectionState::Connecting { attempt: attempt + 1, max_attempts }
            } else {
                ConnectionState::Disconnected
            }
        }
        (ConnectionState::Connected { .. }, Event::Disconnect) => {
            ConnectionState::Disconnecting
        }
        (ConnectionState::Disconnecting, Event::Done) => {
            ConnectionState::Disconnected
        }
        (state, _) => state,  // Ignore irrelevant events
    }
}
```

```python
# Python: state machine implemented with classes
from enum import Enum, auto

class State(Enum):
    IDLE = auto()
    PROCESSING = auto()
    WAITING = auto()
    ERROR = auto()
    DONE = auto()

class StateMachine:
    def __init__(self):
        self.state = State.IDLE
        self._transitions = {
            State.IDLE: {
                "start": self._start_processing,
            },
            State.PROCESSING: {
                "complete": self._complete,
                "error": self._handle_error,
                "wait": self._wait,
            },
            State.WAITING: {
                "resume": self._resume,
                "timeout": self._handle_error,
            },
            State.ERROR: {
                "retry": self._start_processing,
                "abort": self._abort,
            },
        }

    def handle_event(self, event: str):
        transitions = self._transitions.get(self.state, {})
        handler = transitions.get(event)
        if handler is None:
            raise ValueError(
                f"Invalid event '{event}' in state {self.state}"
            )
        handler()

    def _start_processing(self):
        print("Starting...")
        self.state = State.PROCESSING

    def _complete(self):
        print("Done!")
        self.state = State.DONE

    def _handle_error(self):
        print("Error occurred")
        self.state = State.ERROR

    def _wait(self):
        print("Waiting...")
        self.state = State.WAITING

    def _resume(self):
        print("Resuming...")
        self.state = State.PROCESSING

    def _abort(self):
        print("Aborted")
        self.state = State.DONE
```

### 4.3 Recursion vs Loops

```python
# Recursive tree traversal
def tree_sum(node):
    if node is None:
        return 0
    return node.value + tree_sum(node.left) + tree_sum(node.right)

# Loop (explicit traversal using a stack)
def tree_sum_iterative(root):
    if root is None:
        return 0
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        total += node.value
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return total
```

```rust
// Rust: manual tail recursion optimization
// Recursive version
fn factorial(n: u64) -> u64 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

// Loop version (no risk of stack overflow)
fn factorial_iter(n: u64) -> u64 {
    (1..=n).product()
}

// Accumulator pattern (tail-recursive style)
fn factorial_acc(n: u64, acc: u64) -> u64 {
    if n <= 1 { acc } else { factorial_acc(n - 1, n * acc) }
}
```

```haskell
-- Haskell: tail call optimization (TCO)
-- Non-tail recursive (consumes stack)
factorial :: Integer -> Integer
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Tail recursive (with accumulator)
factorial' :: Integer -> Integer
factorial' n = go n 1
  where
    go 0 acc = acc
    go n acc = go (n - 1) (n * acc)

-- Expressed with fold (most concise)
factorial'' :: Integer -> Integer
factorial'' n = foldl' (*) 1 [1..n]
```

---

## 5. Loop Optimization Techniques

### 5.1 Hoisting Loop Invariants

```python
# Bad — computed on every iteration
for i in range(len(data)):
    normalized = data[i] / sum(data)  # sum(data) is computed every iteration
    results.append(normalized)

# Good — computed once outside the loop
total = sum(data)
for i in range(len(data)):
    normalized = data[i] / total
    results.append(normalized)

# Even more Pythonic
total = sum(data)
results = [x / total for x in data]
```

### 5.2 Loop Unrolling and Batch Processing

```python
# Batch processing (for database operations, etc.)
# Bad — one INSERT at a time (slow)
for item in items:
    db.execute("INSERT INTO table VALUES (?)", (item,))

# Good — batch INSERT (fast)
BATCH_SIZE = 1000
for i in range(0, len(items), BATCH_SIZE):
    batch = items[i:i + BATCH_SIZE]
    db.executemany("INSERT INTO table VALUES (?)", [(item,) for item in batch])
```

```rust
// Rust: batch processing with chunks
let data: Vec<Record> = load_records();

for chunk in data.chunks(100) {
    db.bulk_insert(chunk)?;
}

// Parallel batch processing with par_chunks (rayon)
use rayon::prelude::*;
data.par_chunks(100)
    .for_each(|chunk| {
        process_batch(chunk);
    });
```

### 5.3 Leveraging Short-Circuit Evaluation

```python
# Short-circuit evaluation: stops evaluating once the result is determined
# any() — stops at the first True
has_error = any(item.is_error() for item in items)

# all() — stops at the first False
all_valid = all(item.is_valid() for item in items)

# For large datasets, use generator expressions for lazy evaluation
has_match = any(
    expensive_check(item)
    for item in huge_dataset
)  # Stops at the first match; does not check all items
```

---

## 6. Cross-Language Control Flow Design Comparison

### 6.1 Exceptional Control Flow

```python
# Python: for-else (else executes when the loop finishes without break)
def find_prime_factor(n):
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return i
    return n  # If n is prime

# Python: using try-except for control flow (Pythonic)
# EAFP: Easier to Ask Forgiveness than Permission
try:
    value = dictionary[key]
except KeyError:
    value = default_value

# vs LBYL: Look Before You Leap
if key in dictionary:
    value = dictionary[key]
else:
    value = default_value

# Recommended: use dict.get()
value = dictionary.get(key, default_value)
```

```go
// Go: defer (executes when the function exits) — a form of control flow
func processFile(path string) error {
    f, err := os.Open(path)
    if err != nil {
        return err
    }
    defer f.Close()  // Guaranteed to close when the function exits

    // Multiple defers execute in LIFO (last-in, first-out) order
    defer fmt.Println("Step 3")
    defer fmt.Println("Step 2")
    defer fmt.Println("Step 1")
    // Output: Step 1, Step 2, Step 3

    return processData(f)
}

// Go: panic/recover (only for exceptional situations)
func safeDivide(a, b float64) (result float64, err error) {
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic: %v", r)
        }
    }()

    if b == 0 {
        panic("division by zero")
    }
    return a / b, nil
}
```

### 6.2 Coroutines and Control Flow

```python
# Python: control flow with async/await
import asyncio

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

async def fetch_one(session, url):
    async with session.get(url) as response:
        return await response.json()

# Asynchronous iteration
async def process_stream(stream):
    async for chunk in stream:
        await process_chunk(chunk)
```

```rust
// Rust: async/await
async fn fetch_all(urls: Vec<String>) -> Vec<Result<String, Error>> {
    let futures: Vec<_> = urls.iter()
        .map(|url| fetch_one(url))
        .collect();

    futures::future::join_all(futures).await
}

async fn fetch_one(url: &str) -> Result<String, Error> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}
```

---

## 7. Anti-Patterns and Best Practices

### 7.1 Common Anti-Patterns

```python
# Bad — Anti-pattern 1: overuse of flag variables
found = False
for item in items:
    if item == target:
        found = True
        break
if found:
    process(item)

# Good — use early return or built-in functions
if target in items:
    process(target)

# Or
try:
    index = items.index(target)
    process(items[index])
except ValueError:
    pass
```

```python
# Bad — Anti-pattern 2: deep nesting
def validate_and_process(data):
    if data is not None:
        if isinstance(data, dict):
            if "type" in data:
                if data["type"] in VALID_TYPES:
                    if "payload" in data:
                        return process(data["payload"])
                    else:
                        return Error("Missing payload")
                else:
                    return Error("Invalid type")
            else:
                return Error("Missing type")
        else:
            return Error("Not a dict")
    else:
        return Error("No data")

# Good — guard clauses
def validate_and_process(data):
    if data is None:
        return Error("No data")
    if not isinstance(data, dict):
        return Error("Not a dict")
    if "type" not in data:
        return Error("Missing type")
    if data["type"] not in VALID_TYPES:
        return Error("Invalid type")
    if "payload" not in data:
        return Error("Missing payload")
    return process(data["payload"])
```

```javascript
// Bad — Anti-pattern 3: callback hell
getUser(userId, (err, user) => {
    if (err) return handleError(err);
    getOrders(user.id, (err, orders) => {
        if (err) return handleError(err);
        getOrderDetails(orders[0].id, (err, details) => {
            if (err) return handleError(err);
            processDetails(details);
        });
    });
});

// Good — async/await
async function processUserOrder(userId) {
    try {
        const user = await getUser(userId);
        const orders = await getOrders(user.id);
        const details = await getOrderDetails(orders[0].id);
        return processDetails(details);
    } catch (err) {
        handleError(err);
    }
}
```

```python
# Bad — Anti-pattern 4: unnecessary recomputation within loops
for i in range(len(items)):
    for j in range(len(items)):
        distance = compute_distance(items[i], items[j])
        if distance < threshold:
            pairs.append((i, j))

# Good — leverage symmetry to cut computation in half
for i in range(len(items)):
    for j in range(i + 1, len(items)):  # Only compute for j > i
        distance = compute_distance(items[i], items[j])
        if distance < threshold:
            pairs.append((i, j))
            pairs.append((j, i))  # Also add the symmetric pair (if needed)
```

### 7.2 Best Practices Summary

```
1. Keep nesting shallow
   → Use guard clauses (early returns)
   → Aim for a maximum of 3 levels of nesting

2. Leverage expression-based branching
   → Rust/Kotlin: if expressions, match expressions
   → Maintain variable immutability

3. Choose the appropriate loop construct
   → Iterator-based (for-in) is the modern mainstream
   → Leverage functional method chaining (map/filter/reduce)
   → Index-based for is a last resort

4. Be mindful of short-circuit evaluation
   → Leverage short-circuit evaluation with any/all, &&/||
   → Use lazy evaluation for expensive computations

5. Use table-driven approaches
   → Convert long if-elif/switch chains to dispatch tables
   → Improves maintainability and extensibility

6. Guarantee exhaustiveness
   → Explicitly handle all cases in match/switch
   → Avoid careless use of wildcards
   → TypeScript: exhaustiveness checking with the never type

7. Be mindful of loop optimization
   → Hoist loop invariants
   → Leverage batch processing
   → Eliminate unnecessary recomputation
```


---

## Hands-On Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

```python
# Exercise 1: basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
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
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation to add the following features.

```python
# Exercise 2: advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
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
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
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

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup factor: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be mindful of algorithmic time complexity
- Choose the appropriate data structures
- Measure the effect with benchmarks
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but also by actually writing code and verifying its behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend solidly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Syntax | Statement vs Expression | Characteristics |
|--------|------------------------|----------------|
| if (Python, JS) | Statement | Classic; ternary operator is an expression |
| if (Rust, Kotlin) | Expression | Can return a value |
| switch (JS, C) | Statement | Beware of fall-through |
| switch (Java 14+) | Expression | Arrow syntax, yield |
| match (Rust) | Expression | Exhaustiveness check, safe |
| when (Kotlin) | Expression | Supports ranges and type checking |
| for-in | - | Iterator-based (modern mainstream) |
| for-range (Go) | - | Supports slices, maps, and channels |
| .filter().map() | Expression | Functional style |
| loop (Rust) | Expression | Can return a value with break |
| while let (Rust) | - | Loop with pattern matching |

### Loop Syntax Comparison by Language

| Language | C-style for | for-in | while | do-while | Infinite loop | Labels |
|----------|-------------|--------|-------|----------|--------------|--------|
| C | `for(;;)` | - | `while` | `do-while` | `for(;;)` | goto |
| Java | `for(;;)` | `for(:)` | `while` | `do-while` | `while(true)` | label |
| Python | - | `for-in` | `while` | - | `while True` | - |
| JavaScript | `for(;;)` | `for-of` | `while` | `do-while` | `while(true)` | label |
| Rust | - | `for-in` | `while` | - | `loop` | 'label |
| Go | `for` | `for-range` | `for` | `for{...if}` | `for{}` | label |
| Kotlin | `for(;;)` | `for-in` | `while` | `do-while` | `while(true)` | label |

---

## Recommended Next Guides

---

## References
1. Scott, M. "Programming Language Pragmatics." 4th Ed, Ch.6, Morgan Kaufmann, 2015.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.3, 2023.
3. Bloch, J. "Effective Java." 3rd Ed, Item 58-65, Addison-Wesley, 2018.
4. Van Rossum, G. "PEP 20 -- The Zen of Python." python.org, 2004.
5. Donovan, A. & Kernighan, B. "The Go Programming Language." Ch.1, Addison-Wesley, 2015.
6. Jemerov, D. & Isakova, S. "Kotlin in Action." Ch.2, Manning, 2017.
7. Martin, R. "Clean Code." Ch.7 (Error Handling), Prentice Hall, 2008.
8. "Rust By Example: Flow of Control." doc.rust-lang.org.
9. Lipovaca, M. "Learn You a Haskell for Great Good!" Ch.4, No Starch Press, 2011.
10. Odersky, M., Spoon, L. & Venners, B. "Programming in Scala." 5th Ed, Ch.7, Artima, 2021.
