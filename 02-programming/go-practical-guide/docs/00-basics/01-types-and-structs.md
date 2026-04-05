# Types and Structs -- Understanding Go's Type System

> Go adopts static typing and structural subtyping, achieving flexible yet safe data modeling through structs, interfaces, and type assertions.

---

## What You Will Learn in This Chapter

1. **Basic and composite types** -- From int/string/slice to struct
2. **Interfaces and structural subtyping** -- Implicit interface satisfaction
3. **Type assertion and type switch** -- Safe ways to perform dynamic type checks
4. **Generics and type constraints** -- Type parameters in Go 1.18+
5. **Practical data modeling** -- Techniques for designing domain models


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Go Language Overview -- Design Philosophy and Ecosystem](./00-go-overview.md)

---

## 1. Basic Types

### 1.1 Details of Numeric Types

Go gives its integer and floating-point types explicit bit widths. This makes behavior predictable across platforms.

| Type | Size | Range | Use Case |
|----|--------|------|------|
| `int8` | 1 byte | -128 to 127 | When memory conservation is needed |
| `int16` | 2 bytes | -32768 to 32767 | When memory conservation is needed |
| `int32` | 4 bytes | -2^31 to 2^31-1 | Alias for rune |
| `int64` | 8 bytes | -2^63 to 2^63-1 | Timestamps, large integers |
| `int` | 4 or 8 bytes | Platform-dependent | General-purpose integer (default) |
| `uint8` | 1 byte | 0 to 255 | Alias for byte |
| `uint16` | 2 bytes | 0 to 65535 | Port numbers, etc. |
| `uint32` | 4 bytes | 0 to 2^32-1 | IPv4 addresses, etc. |
| `uint64` | 8 bytes | 0 to 2^64-1 | Hash values, etc. |
| `float32` | 4 bytes | IEEE 754 single precision | GPU computation, memory saving |
| `float64` | 8 bytes | IEEE 754 double precision | General-purpose floating-point (default) |
| `complex64` | 8 bytes | float32 real+imaginary | Signal processing, etc. |
| `complex128` | 16 bytes | float64 real+imaginary | Scientific computing, etc. |

### Code Example 1: Declaring and Manipulating Basic Types

```go
package main

import (
    "fmt"
    "math"
    "unicode/utf8"
)

func main() {
    // Declaring basic types
    var (
        i    int     = 42
        f    float64 = 3.14
        s    string  = "hello"
        b    bool    = true
        r    rune    = '日'    // alias for int32
        by   byte    = 0xFF   // alias for uint8
    )

    // Short declaration (type inference)
    x := 100           // int
    pi := 3.14159      // float64
    msg := "Go言語"     // string
    flag := true        // bool

    // Type conversion (implicit conversion is not allowed; explicit conversion is required)
    var n int = 42
    var f64 float64 = float64(n)       // int -> float64
    var n32 int32 = int32(n)           // int -> int32
    var u uint = uint(n)               // int -> uint (beware of negative values)

    // Relationship between strings and byte sequences
    str := "Hello, 世界"
    bytes := []byte(str)               // string -> byte slice
    runes := []rune(str)               // string -> rune slice

    fmt.Printf("len(str)=%d bytes\n", len(str))           // 13 bytes (UTF-8)
    fmt.Printf("rune count=%d\n", utf8.RuneCountInString(str)) // 9 characters
    fmt.Printf("runes=%v\n", runes)                       // Unicode code points

    // Numeric limits
    fmt.Printf("int8 max: %d\n", math.MaxInt8)
    fmt.Printf("int64 max: %d\n", math.MaxInt64)
    fmt.Printf("float64 max: %e\n", math.MaxFloat64)

    // Bitwise operations
    a := 0b1010  // 10
    b2 := 0b1100 // 12
    fmt.Printf("AND: %04b\n", a&b2)   // 1000
    fmt.Printf("OR:  %04b\n", a|b2)   // 1110
    fmt.Printf("XOR: %04b\n", a^b2)   // 0110
    fmt.Printf("NOT: %04b\n", ^a)     // ...0101

    _ = i; _ = f; _ = s; _ = b; _ = r; _ = by
    _ = x; _ = pi; _ = msg; _ = flag
    _ = f64; _ = n32; _ = u; _ = bytes
}
```

### 1.2 Internal Structure of Strings

Strings in Go are immutable byte sequences. Internally, `string` has the following structure:

```go
// runtime/string.go (conceptual structure)
type stringHeader struct {
    Data unsafe.Pointer  // pointer to the byte sequence
    Len  int             // byte length
}
```

Important characteristics of strings:
- Immutable (cannot be changed once created)
- UTF-8 encoding
- `len()` returns the number of bytes (not characters)
- Index access is byte-based
- `range` loops iterate by rune (Unicode code points)

```go
package main

import (
    "fmt"
    "strings"
    "unicode/utf8"
)

func main() {
    s := "Go言語プログラミング"

    // Byte-level operations
    fmt.Printf("len=%d bytes\n", len(s))  // number of bytes

    // Rune-level iteration
    for i, r := range s {
        fmt.Printf("byte_offset=%d, rune=%c, unicode=%U\n", i, r, r)
    }

    // String concatenation
    // For a small number of concatenations, the + operator is sufficient
    greeting := "Hello" + ", " + "World"

    // For many concatenations, use strings.Builder (efficient)
    var builder strings.Builder
    for i := 0; i < 1000; i++ {
        fmt.Fprintf(&builder, "item %d, ", i)
    }
    result := builder.String()
    _ = result

    // String slicing (byte-based)
    sub := s[:2]  // "Go" (ASCII characters, so bytes and runes match)
    fmt.Println(sub)

    // Safely handling multi-byte characters
    runes := []rune(s)
    first3 := string(runes[:3])  // "Go言"
    fmt.Println(first3)

    // Character count
    fmt.Printf("rune count=%d\n", utf8.RuneCountInString(s))

    _ = greeting
}
```

### Code Example 2: Struct Definition and Methods

```go
package main

import (
    "encoding/json"
    "fmt"
    "time"
)

// User is a struct representing user information
type User struct {
    ID        int       `json:"id"`
    Name      string    `json:"name"`
    Email     string    `json:"email"`
    Age       int       `json:"age,omitempty"`       // omit when 0
    IsAdmin   bool      `json:"is_admin"`
    CreatedAt time.Time `json:"created_at"`
    password  string    // private field (starts with lowercase)
}

// NewUser is a factory function for User (equivalent to a constructor)
func NewUser(name, email string) *User {
    return &User{
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
    }
}

// DisplayName returns the display name (value receiver -- does not modify the struct)
func (u User) DisplayName() string {
    return fmt.Sprintf("%s (%d)", u.Name, u.ID)
}

// UpdateEmail updates the email address (pointer receiver -- modifies the struct)
func (u *User) UpdateEmail(email string) {
    u.Email = email
}

// SetPassword sets the password
func (u *User) SetPassword(password string) {
    u.password = password // access to a private field
}

// Validate performs validation
func (u User) Validate() error {
    if u.Name == "" {
        return fmt.Errorf("name is required")
    }
    if u.Email == "" {
        return fmt.Errorf("email is required")
    }
    if u.Age < 0 || u.Age > 200 {
        return fmt.Errorf("invalid age: %d", u.Age)
    }
    return nil
}

// String implements the fmt.Stringer interface
func (u User) String() string {
    return fmt.Sprintf("User{id=%d, name=%q, email=%q}", u.ID, u.Name, u.Email)
}

// MarshalJSON provides custom JSON serialization
func (u User) MarshalJSON() ([]byte, error) {
    type Alias User // avoid infinite recursion
    return json.Marshal(&struct {
        Alias
        CreatedAtStr string `json:"created_at_formatted"`
    }{
        Alias:        Alias(u),
        CreatedAtStr: u.CreatedAt.Format("2006-01-02 15:04:05"),
    })
}

func main() {
    user := NewUser("Alice", "alice@example.com")
    user.ID = 1
    user.Age = 30
    user.SetPassword("secret123")

    fmt.Println(user)
    fmt.Println(user.DisplayName())

    if err := user.Validate(); err != nil {
        fmt.Printf("Validation error: %v\n", err)
    }

    // JSON serialization
    data, _ := json.MarshalIndent(user, "", "  ")
    fmt.Println(string(data))

    // JSON deserialization
    jsonStr := `{"id":2,"name":"Bob","email":"bob@example.com","age":25}`
    var user2 User
    json.Unmarshal([]byte(jsonStr), &user2)
    fmt.Println(user2)
}
```

### Code Example 3: Struct Embedding and Delegation

```go
package main

import (
    "fmt"
    "time"
)

// BaseModel provides common fields
type BaseModel struct {
    ID        int
    CreatedAt time.Time
    UpdatedAt time.Time
}

// BeforeSave updates timestamps before saving
func (b *BaseModel) BeforeSave() {
    now := time.Now()
    if b.CreatedAt.IsZero() {
        b.CreatedAt = now
    }
    b.UpdatedAt = now
}

// Animal holds basic animal information
type Animal struct {
    Name string
    Age  int
}

func (a Animal) Speak() string {
    return fmt.Sprintf("I am %s, %d years old", a.Name, a.Age)
}

func (a Animal) Move() string {
    return "moving..."
}

// Dog embeds Animal (composition, not inheritance)
type Dog struct {
    Animal        // embedding
    Breed  string
}

// Speak overrides Animal's Speak (method shadowing)
func (d Dog) Speak() string {
    return fmt.Sprintf("Woof! I am %s the %s", d.Name, d.Breed)
}

// Cat also embeds Animal
type Cat struct {
    Animal
    Indoor bool
}

func (c Cat) Speak() string {
    return fmt.Sprintf("Meow! I am %s", c.Name)
}

// Product is a practical example that embeds BaseModel
type Product struct {
    BaseModel
    Name     string
    Price    float64
    Category string
    Tags     []string
}

func (p *Product) Save() {
    p.BeforeSave() // BaseModel's method is promoted and usable
    fmt.Printf("Saving product: %s (created=%v, updated=%v)\n",
        p.Name, p.CreatedAt.Format("15:04:05"), p.UpdatedAt.Format("15:04:05"))
}

// Multiple embedding
type Logger struct{}
func (l Logger) Log(msg string) { fmt.Printf("[LOG] %s\n", msg) }

type Metrics struct{}
func (m Metrics) Record(name string, value float64) {
    fmt.Printf("[METRIC] %s=%.2f\n", name, value)
}

type Service struct {
    Logger  // logging capability
    Metrics // metrics capability
    Name string
}

func main() {
    d := Dog{
        Animal: Animal{Name: "Pochi", Age: 3},
        Breed:  "Shiba",
    }
    fmt.Println(d.Speak())  // Dog.Speak() is called
    fmt.Println(d.Move())   // Animal.Move() is promoted and called
    fmt.Println(d.Name)     // shorthand for d.Animal.Name
    fmt.Println(d.Animal.Speak()) // explicitly call the original Animal.Speak()

    p := &Product{
        Name:     "Go Book",
        Price:    4980,
        Category: "Books",
        Tags:     []string{"programming", "golang"},
    }
    p.Save()

    svc := Service{Name: "UserService"}
    svc.Log("service started")        // Logger.Log is promoted
    svc.Record("requests", 42.0)      // Metrics.Record is promoted
}
```

### Code Example 4: Interfaces and Structural Subtyping

```go
package main

import (
    "fmt"
    "io"
    "math"
    "strings"
)

// Small interfaces (the recommended pattern in Go)
type Stringer interface {
    String() string
}

type Area interface {
    Area() float64
}

type Perimeter interface {
    Perimeter() float64
}

// Composing interfaces
type Shape interface {
    Area
    Perimeter
    Stringer
}

// Concrete type: Circle
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return math.Pi * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * math.Pi * c.Radius
}

func (c Circle) String() string {
    return fmt.Sprintf("Circle{radius=%.2f}", c.Radius)
}

// Concrete type: Rectangle
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
    return fmt.Sprintf("Rectangle{%.2f x %.2f}", r.Width, r.Height)
}

// Concrete type: Triangle
type Triangle struct {
    A, B, C float64 // lengths of the three sides
}

func (t Triangle) Area() float64 {
    s := (t.A + t.B + t.C) / 2
    return math.Sqrt(s * (s - t.A) * (s - t.B) * (s - t.C))
}

func (t Triangle) Perimeter() float64 {
    return t.A + t.B + t.C
}

func (t Triangle) String() string {
    return fmt.Sprintf("Triangle{a=%.2f, b=%.2f, c=%.2f}", t.A, t.B, t.C)
}

// Function that takes a Shape (polymorphism)
func printShapeInfo(s Shape) {
    fmt.Printf("%s: area=%.2f, perimeter=%.2f\n", s, s.Area(), s.Perimeter())
}

// Function that requires only Area (minimal interface)
func totalArea(shapes []Area) float64 {
    var total float64
    for _, s := range shapes {
        total += s.Area()
    }
    return total
}

// Example using io.Reader
func countLines(r io.Reader) (int, error) {
    buf := make([]byte, 32*1024)
    count := 0
    for {
        n, err := r.Read(buf)
        for i := 0; i < n; i++ {
            if buf[i] == '\n' {
                count++
            }
        }
        if err != nil {
            if err == io.EOF {
                return count, nil
            }
            return count, err
        }
    }
}

func main() {
    shapes := []Shape{
        Circle{Radius: 5},
        Rectangle{Width: 10, Height: 3},
        Triangle{A: 3, B: 4, C: 5},
    }

    for _, s := range shapes {
        printShapeInfo(s)
    }

    // Use as Area interface
    areas := make([]Area, len(shapes))
    for i, s := range shapes {
        areas[i] = s
    }
    fmt.Printf("Total area: %.2f\n", totalArea(areas))

    // Using io.Reader
    text := "line 1\nline 2\nline 3\n"
    reader := strings.NewReader(text)
    lines, _ := countLines(reader)
    fmt.Printf("Lines: %d\n", lines)
}
```

### Code Example 5: Type Assertion and Type Switch

```go
package main

import (
    "fmt"
    "io"
    "os"
    "strings"
)

// Practical use of type switch
type Result struct {
    Value interface{}
    Err   error
}

func describe(i interface{}) string {
    switch v := i.(type) {
    case nil:
        return "nil"
    case int:
        return fmt.Sprintf("integer: %d", v)
    case int64:
        return fmt.Sprintf("int64: %d", v)
    case float64:
        return fmt.Sprintf("float64: %.2f", v)
    case string:
        return fmt.Sprintf("string: %q (len=%d)", v, len(v))
    case bool:
        return fmt.Sprintf("bool: %t", v)
    case []byte:
        return fmt.Sprintf("bytes: %x (len=%d)", v, len(v))
    case []int:
        return fmt.Sprintf("[]int: %v (len=%d)", v, len(v))
    case map[string]interface{}:
        return fmt.Sprintf("map: %d entries", len(v))
    case error:
        return fmt.Sprintf("error: %v", v)
    case fmt.Stringer:
        return fmt.Sprintf("Stringer: %s", v.String())
    case io.Reader:
        return "io.Reader"
    default:
        return fmt.Sprintf("unknown: %T = %v", v, v)
    }
}

// Safe type assertion (comma-ok idiom)
func toInt(i interface{}) (int, bool) {
    v, ok := i.(int)
    return v, ok
}

// Extending functionality via interface type assertion
type Closer interface {
    Close() error
}

type Flusher interface {
    Flush() error
}

func cleanup(w io.Writer) error {
    // If the writer also implements the Flusher interface, call Flush
    if f, ok := w.(Flusher); ok {
        if err := f.Flush(); err != nil {
            return fmt.Errorf("flush: %w", err)
        }
    }

    // If the writer also implements the Closer interface, call Close
    if c, ok := w.(Closer); ok {
        if err := c.Close(); err != nil {
            return fmt.Errorf("close: %w", err)
        }
    }

    return nil
}

// Type constraint that requires multiple interfaces
type ReadWriteCloser interface {
    io.Reader
    io.Writer
    io.Closer
}

// Type-safe handling of JSON decoded results
func processJSON(data map[string]interface{}) {
    for key, val := range data {
        switch v := val.(type) {
        case string:
            fmt.Printf("  %s: string = %q\n", key, v)
        case float64: // JSON numbers become float64
            if v == float64(int(v)) {
                fmt.Printf("  %s: int = %d\n", key, int(v))
            } else {
                fmt.Printf("  %s: float = %.2f\n", key, v)
            }
        case bool:
            fmt.Printf("  %s: bool = %t\n", key, v)
        case nil:
            fmt.Printf("  %s: null\n", key)
        case []interface{}:
            fmt.Printf("  %s: array (len=%d)\n", key, len(v))
        case map[string]interface{}:
            fmt.Printf("  %s: object (keys=%d)\n", key, len(v))
        }
    }
}

func main() {
    // Usage example for describe
    values := []interface{}{
        42, 3.14, "hello", true, nil,
        []byte{0xDE, 0xAD},
        []int{1, 2, 3},
        os.Stdout,
    }

    for _, v := range values {
        fmt.Println(describe(v))
    }

    // Safe type assertion
    var i interface{} = 42
    if n, ok := toInt(i); ok {
        fmt.Printf("Got int: %d\n", n)
    }

    var s interface{} = "not an int"
    if _, ok := toInt(s); !ok {
        fmt.Println("Not an int")
    }

    // Usage example for cleanup
    reader := strings.NewReader("test")
    cleanup(reader) // Does not have Close/Flush -> does nothing

    // Processing JSON results
    data := map[string]interface{}{
        "name":   "Alice",
        "age":    30.0,
        "active": true,
        "score":  95.5,
        "tags":   []interface{}{"go", "programming"},
    }
    fmt.Println("JSON data:")
    processJSON(data)
}
```

### Code Example 6: Custom Types and Type Conversion

```go
package main

import (
    "fmt"
    "strings"
    "time"
)

// Increase type safety with custom types
type UserID int64
type OrderID int64
type ProductID int64

// Prevent mixing different ID types
// var uid UserID = OrderID(1) // compile error: cannot use OrderID(1) as type UserID

// Temperature type example
type Celsius float64
type Fahrenheit float64
type Kelvin float64

func (c Celsius) ToFahrenheit() Fahrenheit {
    return Fahrenheit(c*9/5 + 32)
}

func (c Celsius) ToKelvin() Kelvin {
    return Kelvin(c + 273.15)
}

func (f Fahrenheit) ToCelsius() Celsius {
    return Celsius((f - 32) * 5 / 9)
}

func (c Celsius) String() string {
    return fmt.Sprintf("%.1f°C", float64(c))
}

func (f Fahrenheit) String() string {
    return fmt.Sprintf("%.1f°F", float64(f))
}

func (k Kelvin) String() string {
    return fmt.Sprintf("%.1fK", float64(k))
}

// Custom string type
type Email string

func (e Email) Validate() error {
    s := string(e)
    if !strings.Contains(s, "@") {
        return fmt.Errorf("invalid email: %q", s)
    }
    parts := strings.Split(s, "@")
    if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
        return fmt.Errorf("invalid email format: %q", s)
    }
    if !strings.Contains(parts[1], ".") {
        return fmt.Errorf("invalid email domain: %q", parts[1])
    }
    return nil
}

func (e Email) Domain() string {
    parts := strings.Split(string(e), "@")
    if len(parts) == 2 {
        return parts[1]
    }
    return ""
}

func (e Email) String() string {
    return string(e)
}

// Custom Duration wrapper
type Timeout time.Duration

func (t Timeout) Duration() time.Duration {
    return time.Duration(t)
}

func (t Timeout) String() string {
    return time.Duration(t).String()
}

// Emulating an enum (using iota)
type Status int

const (
    StatusPending  Status = iota // 0
    StatusActive                  // 1
    StatusInactive                // 2
    StatusDeleted                 // 3
)

func (s Status) String() string {
    names := [...]string{"pending", "active", "inactive", "deleted"}
    if int(s) < len(names) {
        return names[s]
    }
    return fmt.Sprintf("Status(%d)", s)
}

func (s Status) IsValid() bool {
    return s >= StatusPending && s <= StatusDeleted
}

// Bit-flag enum
type Permission uint8

const (
    PermRead    Permission = 1 << iota // 1
    PermWrite                           // 2
    PermExecute                         // 4
    PermAdmin                           // 8
)

func (p Permission) Has(flag Permission) bool {
    return p&flag != 0
}

func (p Permission) String() string {
    var perms []string
    if p.Has(PermRead) {
        perms = append(perms, "read")
    }
    if p.Has(PermWrite) {
        perms = append(perms, "write")
    }
    if p.Has(PermExecute) {
        perms = append(perms, "execute")
    }
    if p.Has(PermAdmin) {
        perms = append(perms, "admin")
    }
    if len(perms) == 0 {
        return "none"
    }
    return strings.Join(perms, "|")
}

func main() {
    // Temperature conversion
    temp := Celsius(100)
    fmt.Printf("%s = %s = %s\n", temp, temp.ToFahrenheit(), temp.ToKelvin())

    // Email validation
    emails := []Email{
        "alice@example.com",
        "invalid-email",
        "bob@",
    }
    for _, e := range emails {
        if err := e.Validate(); err != nil {
            fmt.Printf("  %s: %v\n", e, err)
        } else {
            fmt.Printf("  %s: valid (domain=%s)\n", e, e.Domain())
        }
    }

    // Status
    status := StatusActive
    fmt.Printf("Status: %s (valid=%t)\n", status, status.IsValid())

    // Permissions (bit flags)
    perm := PermRead | PermWrite
    fmt.Printf("Permissions: %s\n", perm)
    fmt.Printf("Has read: %t\n", perm.Has(PermRead))
    fmt.Printf("Has admin: %t\n", perm.Has(PermAdmin))

    perm = perm | PermAdmin
    fmt.Printf("After adding admin: %s\n", perm)
}
```

### Code Example 7: Internal Structure and Operations of Slices

```go
package main

import (
    "fmt"
    "slices" // Go 1.21+
    "sort"
)

// Internal structure of a slice (conceptual)
// type slice struct {
//     array unsafe.Pointer  // pointer to the underlying array
//     len   int             // length
//     cap   int             // capacity
// }

func main() {
    // Ways to create slices
    s1 := []int{1, 2, 3, 4, 5}       // literal
    s2 := make([]int, 5)              // make (length=5, capacity=5)
    s3 := make([]int, 0, 10)          // make (length=0, capacity=10)

    fmt.Printf("s1: len=%d, cap=%d, %v\n", len(s1), cap(s1), s1)
    fmt.Printf("s2: len=%d, cap=%d, %v\n", len(s2), cap(s2), s2)
    fmt.Printf("s3: len=%d, cap=%d, %v\n", len(s3), cap(s3), s3)

    // append and capacity growth
    var growing []int
    prevCap := cap(growing)
    for i := 0; i < 20; i++ {
        growing = append(growing, i)
        if cap(growing) != prevCap {
            fmt.Printf("len=%2d, cap changed: %d -> %d\n",
                len(growing), prevCap, cap(growing))
            prevCap = cap(growing)
        }
    }

    // Slices share the underlying array (be careful)
    original := []int{1, 2, 3, 4, 5}
    sub := original[1:3]         // [2, 3]
    sub[0] = 99                  // original is modified too!
    fmt.Println("original:", original) // [1, 99, 3, 4, 5]

    // Create an independent slice by copying
    independent := make([]int, len(original))
    copy(independent, original)
    independent[0] = 777
    fmt.Println("original:", original)    // unchanged
    fmt.Println("independent:", independent) // [777, 99, 3, 4, 5]

    // Using the slices package (Go 1.21+)
    nums := []int{3, 1, 4, 1, 5, 9, 2, 6}
    slices.Sort(nums)
    fmt.Println("sorted:", nums)

    idx, found := slices.BinarySearch(nums, 5)
    fmt.Printf("BinarySearch(5): idx=%d, found=%t\n", idx, found)

    // Contains
    fmt.Printf("Contains(9): %t\n", slices.Contains(nums, 9))
    fmt.Printf("Contains(7): %t\n", slices.Contains(nums, 7))

    // Filtering (manually)
    evens := make([]int, 0)
    for _, n := range nums {
        if n%2 == 0 {
            evens = append(evens, n)
        }
    }
    fmt.Println("evens:", evens)

    // Custom sort
    type Person struct {
        Name string
        Age  int
    }
    people := []Person{
        {"Alice", 30},
        {"Bob", 25},
        {"Carol", 35},
    }
    sort.Slice(people, func(i, j int) bool {
        return people[i].Age < people[j].Age
    })
    fmt.Println("sorted by age:", people)
}
```

### Code Example 8: Maps in Detail

```go
package main

import (
    "fmt"
    "maps" // Go 1.21+
    "sort"
    "sync"
)

func main() {
    // Creating a map
    m1 := map[string]int{
        "alice": 95,
        "bob":   87,
        "carol": 92,
    }

    // Create an empty map with make
    m2 := make(map[string]int)
    m2["dave"] = 88

    // Retrieving elements (comma-ok idiom)
    score, ok := m1["alice"]
    fmt.Printf("alice: score=%d, exists=%t\n", score, ok)

    score, ok = m1["eve"]
    fmt.Printf("eve: score=%d, exists=%t\n", score, ok) // 0, false

    // Deletion
    delete(m1, "bob")
    fmt.Printf("After delete: %v\n", m1)

    // Iterating the map (order is non-deterministic)
    for key, val := range m1 {
        fmt.Printf("  %s: %d\n", key, val)
    }

    // Iterating with sorted keys
    keys := make([]string, 0, len(m1))
    for k := range m1 {
        keys = append(keys, k)
    }
    sort.Strings(keys)
    fmt.Println("Sorted iteration:")
    for _, k := range keys {
        fmt.Printf("  %s: %d\n", k, m1[k])
    }

    // nil map vs empty map
    var nilMap map[string]int
    emptyMap := map[string]int{}
    fmt.Printf("nil map == nil: %t\n", nilMap == nil)     // true
    fmt.Printf("empty map == nil: %t\n", emptyMap == nil)  // false
    // nilMap["key"] = 1  // panic: assignment to entry in nil map
    _ = nilMap["key"]    // reads are OK (zero value is returned)

    // Nested maps (map of maps)
    graph := map[string]map[string]int{
        "A": {"B": 1, "C": 4},
        "B": {"C": 2, "D": 5},
        "C": {"D": 1},
    }

    // Safe access to nested maps
    if neighbors, ok := graph["A"]; ok {
        for node, weight := range neighbors {
            fmt.Printf("A -> %s: weight=%d\n", node, weight)
        }
    }

    // Using a map as a set
    set := make(map[string]struct{})
    set["apple"] = struct{}{}
    set["banana"] = struct{}{}
    set["cherry"] = struct{}{}

    if _, exists := set["apple"]; exists {
        fmt.Println("apple is in the set")
    }

    // maps package (Go 1.21+)
    clone := maps.Clone(m1)
    fmt.Printf("clone: %v\n", clone)

    // Word count (practical example)
    text := "the quick brown fox jumps over the lazy dog the fox"
    wordCount := make(map[string]int)
    for _, word := range splitWords(text) {
        wordCount[word]++
    }
    fmt.Println("Word counts:", wordCount)

    // Concurrency-safe map operations (see the separate section on sync.Map)
    _ = sync.Map{}
}

func splitWords(s string) []string {
    var words []string
    word := ""
    for _, r := range s {
        if r == ' ' || r == '\n' || r == '\t' {
            if word != "" {
                words = append(words, word)
                word = ""
            }
        } else {
            word += string(r)
        }
    }
    if word != "" {
        words = append(words, word)
    }
    return words
}
```

### Code Example 9: Generics and Type Constraints (Go 1.18+)

```go
package main

import (
    "cmp"
    "fmt"
    "slices"
)

// Type constraint: Ordered constrains types that are comparable
type Ordered interface {
    ~int | ~int8 | ~int16 | ~int32 | ~int64 |
        ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
        ~float32 | ~float64 |
        ~string
}

// Generic Pair type
type Pair[T, U any] struct {
    First  T
    Second U
}

func NewPairT, U any Pair[T, U] {
    return Pair[T, U]{First: first, Second: second}
}

func (p Pair[T, U]) String() string {
    return fmt.Sprintf("(%v, %v)", p.First, p.Second)
}

// Generic Optional type (equivalent to Rust's Option)
type Optional[T any] struct {
    value   T
    present bool
}

func SomeT any Optional[T] {
    return Optional[T]{value: value, present: true}
}

func None[T any]() Optional[T] {
    return Optional[T]{}
}

func (o Optional[T]) Get() (T, bool) {
    return o.value, o.present
}

func (o Optional[T]) GetOrElse(defaultValue T) T {
    if o.present {
        return o.value
    }
    return defaultValue
}

func (o Optional[T]) Map(f func(T) T) Optional[T] {
    if o.present {
        return Some(f(o.value))
    }
    return None[T]()
}

// Generic Set
type Set[T comparable] struct {
    items map[T]struct{}
}

func NewSetT comparable *Set[T] {
    s := &Set[T]{items: make(map[T]struct{})}
    for _, item := range items {
        s.Add(item)
    }
    return s
}

func (s *Set[T]) Add(item T) {
    s.items[item] = struct{}{}
}

func (s *Set[T]) Remove(item T) {
    delete(s.items, item)
}

func (s *Set[T]) Contains(item T) bool {
    _, ok := s.items[item]
    return ok
}

func (s *Set[T]) Len() int {
    return len(s.items)
}

func (s *Set[T]) Union(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        result.Add(item)
    }
    for item := range other.items {
        result.Add(item)
    }
    return result
}

func (s *Set[T]) Intersection(other *Set[T]) *Set[T] {
    result := NewSet[T]()
    for item := range s.items {
        if other.Contains(item) {
            result.Add(item)
        }
    }
    return result
}

// Generic function using cmp.Ordered (Go 1.21+)
func ClampT cmp.Ordered T {
    if value < min {
        return min
    }
    if value > max {
        return max
    }
    return value
}

// Generic LinkedList
type Node[T any] struct {
    Value T
    Next  *Node[T]
}

type LinkedList[T any] struct {
    Head *Node[T]
    Len  int
}

func (ll *LinkedList[T]) Push(value T) {
    ll.Head = &Node[T]{Value: value, Next: ll.Head}
    ll.Len++
}

func (ll *LinkedList[T]) Pop() (T, bool) {
    if ll.Head == nil {
        var zero T
        return zero, false
    }
    value := ll.Head.Value
    ll.Head = ll.Head.Next
    ll.Len--
    return value, true
}

func (ll *LinkedList[T]) ForEach(fn func(T)) {
    for node := ll.Head; node != nil; node = node.Next {
        fn(node.Value)
    }
}

func main() {
    // Pair
    p1 := NewPair("name", 42)
    p2 := NewPair(3.14, true)
    fmt.Println(p1, p2)

    // Optional
    opt1 := Some(42)
    opt2 := None[string]()

    if v, ok := opt1.Get(); ok {
        fmt.Printf("opt1: %d\n", v)
    }
    fmt.Printf("opt2: %q\n", opt2.GetOrElse("default"))

    doubled := opt1.Map(func(n int) int { return n * 2 })
    fmt.Printf("doubled: %v\n", doubled.GetOrElse(0))

    // Set
    s1 := NewSet("go", "rust", "python")
    s2 := NewSet("python", "java", "go")

    union := s1.Union(s2)
    intersection := s1.Intersection(s2)

    fmt.Printf("s1 contains 'go': %t\n", s1.Contains("go"))
    fmt.Printf("union size: %d\n", union.Len())
    fmt.Printf("intersection size: %d\n", intersection.Len())

    // Clamp
    fmt.Printf("Clamp(15, 0, 10): %d\n", Clamp(15, 0, 10))
    fmt.Printf("Clamp(-5, 0, 10): %d\n", Clamp(-5, 0, 10))
    fmt.Printf("Clamp(5, 0, 10): %d\n", Clamp(5, 0, 10))

    // LinkedList
    ll := &LinkedList[int]{}
    ll.Push(1)
    ll.Push(2)
    ll.Push(3)

    ll.ForEach(func(v int) {
        fmt.Printf("%d -> ", v)
    })
    fmt.Println("nil")

    // The slices package with generics
    nums := []int{5, 3, 8, 1, 9}
    slices.Sort(nums)
    fmt.Println("sorted:", nums)
}
```

---

## 2. ASCII Diagrams

### Diagram 1: Go's Type Hierarchy

```
                     ┌──────────┐
                     │   any    │
                     │(interface│
                     │   {})    │
                     └────┬─────┘
           ┌──────────────┼──────────────┐
     ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼──────┐
     │ Basic     │  │ Composite │  │ Reference │
     │ int,float │  │ struct    │  │ slice,map │
     │ string    │  │ array     │  │ channel   │
     │ bool,rune │  │           │  │ pointer   │
     │ complex   │  │           │  │ function  │
     └───────────┘  └──────────┘  └───────────┘

Details of integer types:
  ┌─────────────────────────────────────────┐
  │ Signed                                   │
  │  int8 - int16 - int32 - int64 - int      │
  │  (rune = int32)                          │
  │                                          │
  │ Unsigned                                 │
  │  uint8 - uint16 - uint32 - uint64 - uint │
  │  (byte = uint8)                          │
  │                                          │
  │ Special                                  │
  │  uintptr (pointer-sized unsigned int)    │
  └─────────────────────────────────────────┘
```

### Diagram 2: Memory Layout of Struct Embedding

```
Dog struct:
┌──────────────────────────────────┐
│  Animal (embedded)               │
│  ┌──────────────────────────┐    │
│  │ Name string              │    │
│  │ Age  int                 │    │
│  └──────────────────────────┘    │
│  Breed string                    │
└──────────────────────────────────┘

Access:
  d.Name   ->  d.Animal.Name  (shorthand)
  d.Age    ->  d.Animal.Age   (shorthand)
  d.Speak() -> d.Animal.Speak() (method promotion)

Name collision with multiple embeddings:
┌──────────────────────────────────┐
│  Service                         │
│  ┌─────────────┐ ┌────────────┐ │
│  │ Logger      │ │ Metrics    │ │
│  │  Log()      │ │  Record()  │ │
│  └─────────────┘ └────────────┘ │
│  Name string                     │
│  * If Logger and Metrics have    │
│    methods with the same name,   │
│    it's a compile error          │
│    (explicit access required)    │
└──────────────────────────────────┘
```

### Diagram 3: How Interface Satisfaction Works

```
┌────────────────┐          ┌────────────────┐
│  io.Reader     │          │  *os.File      │
│  ┌────────────┐│          │  ┌────────────┐│
│  │ Read([]byte)│├satisfies┤  │ Read([]byte)││
│  │ (int, error)││(implicit)│  │ (int, error)││
│  └────────────┘│          │  │ Write(...)  ││
└────────────────┘          │  │ Close()     ││
                            │  │ Stat()      ││
┌────────────────┐          │  └────────────┘│
│  io.ReadCloser │          └────────────────┘
│  ┌────────────┐│               ▲
│  │ Read(...)   ││   also        │
│  │ Close()     ││── satisfies ──┘
│  └────────────┘│
└────────────────┘

Internal representation of an interface (iface):
┌──────────────────────────┐
│  interface value          │
│  ┌──────────┐ ┌────────┐│
│  │  type    │ │ value  ││
│  │  *itab   │ │ *data  ││
│  └──────────┘ └────────┘│
└──────────────────────────┘

itab = interface table (cache for the method table)
data = pointer to the actual value

nil interface vs nil pointer:
  var w io.Writer         // type=nil, data=nil -> w == nil (true)
  var f *os.File = nil
  w = f                   // type=*os.File, data=nil -> w == nil (false!)
```

### Diagram 4: Internal Structure of a Slice

```
s := []int{1, 2, 3, 4, 5}

Slice header:
┌─────────┐
│ ptr  ────┼──> ┌───┬───┬───┬───┬───┐
│ len = 5  │    │ 1 │ 2 │ 3 │ 4 │ 5 │  underlying array
│ cap = 5  │    └───┴───┴───┴───┴───┘
└─────────┘

sub := s[1:3]

┌─────────┐          ┌───┬───┬───┬───┬───┐
│ ptr  ────┼──────────┼───┤ 2 │ 3 │ 4 │ 5 │
│ len = 2  │          │ 1 │   │   │   │   │
│ cap = 4  │          └───┴───┴───┴───┴───┘
└─────────┘          ↑       ↑
                     s[0]    sub[0] = s[1]

Reallocation due to append:
s2 := append(s, 6)

cap is insufficient -> allocate a new array (roughly double the capacity)
┌─────────┐
│ ptr  ────┼──> ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ len = 6  │    │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │   │   │   │   │
│ cap = 10 │    └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
└─────────┘
* The underlying array is different from the original slice (independent)
```

### Diagram 5: Conceptual Internal Structure of a Map

```
map[string]int:
┌──────────────────────────────────────┐
│ Hash table                           │
│                                      │
│  Bucket array:                        │
│  ┌─────────┐                          │
│  │bucket[0]│──> [key1:val1, key2:val2]│
│  ├─────────┤                          │
│  │bucket[1]│──> [key3:val3]           │
│  ├─────────┤                          │
│  │bucket[2]│──> (empty)               │
│  ├─────────┤                          │
│  │bucket[3]│──> [key4:val4, ...]      │
│  └─────────┘                          │
│                                      │
│  Each bucket holds up to 8 entries    │
│  On overflow, it chains to            │
│  an overflow bucket                   │
│                                      │
│  Growth: when items/buckets > 6.5,    │
│  the bucket array doubles             │
└──────────────────────────────────────┘
```

---

## 3. Comparison Tables

### Table 1: Value Receiver vs Pointer Receiver

| Item | Value receiver `(t T)` | Pointer receiver `(t *T)` |
|------|-------------------|------------------------|
| Copy | Copied on every call | Only the pointer is copied |
| Modification of original | Not possible | Possible |
| Call on nil | Not possible | Possible (requires nil check) |
| Interface satisfaction | Both T and *T | Only *T |
| Recommended use | Small structs, immutable | Large structs, mutable |
| Contains sync.Mutex | Not allowed (gets copied) | Required (copying is forbidden) |
| Storage in slices | The value is copied | The pointer is stored |

### Table 2: Zero Values for Go Types

| Type | Zero value | Notes |
|-----|-------|------|
| int, float64 | `0`, `0.0` | All numeric types are 0 |
| string | `""` | Empty string |
| bool | `false` | |
| pointer | `nil` | |
| slice | `nil` | `len()=0`, `cap()=0`. append is allowed |
| map | `nil` | Readable; make is required before assignment |
| channel | `nil` | Send/receive blocks forever |
| struct | Zero value of each field | |
| interface | `nil` | Both type and value are nil |
| function | `nil` | |
| array | Zero value of each element | Fixed length |

### Table 3: Comparison of Collection Types

| Characteristic | Array | Slice | Map | sync.Map |
|------|-------|-------|-----|----------|
| Size | Fixed | Dynamic | Dynamic | Dynamic |
| Memory | Can be on stack | Heap | Heap | Heap |
| Key type | int (index) | int (index) | comparable | any |
| Order guarantee | Yes | Yes | No | No |
| Concurrency safe | No | No | No | Yes |
| Zero value | Usable | nil (append OK) | nil (assignment not allowed) | Usable |
| Equality comparison | `==` allowed | Not allowed (reflect) | Not allowed | Not allowed |

### Table 4: Interface Design Guidelines

| Principle | Description | Example |
|------|------|-----|
| Keep them small | 1-3 methods is ideal | io.Reader (1), io.ReadWriter (2) |
| Define on consumer side | The consumer declares the interface | handler.Getter |
| Name as verb + "er" | Name based on the action | Reader, Writer, Closer |
| Abstract only when needed | Avoid pre-defining | When a mock becomes necessary in tests |
| Extend via composition | Embed existing interfaces | io.ReadWriteCloser |

---

## 4. Anti-Patterns

### Anti-Pattern 1: The Giant Interface

```go
// BAD: A giant interface (God Interface)
type Repository interface {
    FindUser(id int) (*User, error)
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
    FindOrder(id int) (*Order, error)
    CreateOrder(o *Order) error
    UpdateOrder(o *Order) error
    DeleteOrder(id int) error
    FindProduct(id int) (*Product, error)
    // ... more than 20 methods
}

// Problems:
// 1. Creating mocks for tests is tedious
// 2. Implementations are forced to implement methods they don't need
// 3. The blast radius of changes is large

// GOOD: Split into small, role-based interfaces
type UserReader interface {
    FindUser(id int) (*User, error)
}

type UserWriter interface {
    CreateUser(u *User) error
    UpdateUser(u *User) error
    DeleteUser(id int) error
}

// Compose as needed
type UserRepository interface {
    UserReader
    UserWriter
}

// In tests, only mock the interfaces you need
type mockUserReader struct {
    users map[int]*User
}

func (m *mockUserReader) FindUser(id int) (*User, error) {
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}
```

### Anti-Pattern 2: The nil Interface Trap

```go
// BAD: Storing a nil pointer in an interface breaks the nil check
type Logger interface {
    Log(msg string)
}

type FileLogger struct {
    Path string
}

func (f *FileLogger) Log(msg string) {
    fmt.Printf("[%s] %s\n", f.Path, msg)
}

func getLogger() *FileLogger {
    // returns nil depending on conditions
    return nil
}

func processWithLogger() {
    var logger Logger = getLogger()  // Assigns *FileLogger(nil)
    if logger != nil {
        // Execution reaches here!
        // interface{type:*FileLogger, value:nil} is not nil
        logger.Log("hello") // nil pointer dereference -> PANIC
    }
}

// GOOD: Explicitly return a nil interface
func getLoggerSafe() Logger {
    f := findLogger()
    if f == nil {
        return nil  // the interface itself becomes nil
    }
    return f
}

// GOOD: nil check using reflect (expensive, not recommended)
import "reflect"

func isNil(v interface{}) bool {
    if v == nil {
        return true
    }
    rv := reflect.ValueOf(v)
    switch rv.Kind() {
    case reflect.Ptr, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func:
        return rv.IsNil()
    }
    return false
}
```

### Anti-Pattern 3: Mistakes in Struct Field Tags

```go
// BAD: Whitespace mistakes in JSON tags
type Config struct {
    Host string `json: "host"` // Contains a space -> tag is ignored
    Port int    `json:"port" `  // Trailing whitespace -> doesn't work properly
}

// BAD: Inconsistencies in validation tags
type Request struct {
    Name  string `json:"name" validate:"required"`
    Email string `json:"email" validate:"required,email"`
    Age   int    `json:"age" validate:"min=0,max=200"`  // OK
    // Age   int    `json:"age" validate:"min=0, max=200"` // BAD: contains a space
}

// GOOD: go vet can detect some of these, but manual checking is also needed
type Config struct {
    Host     string        `json:"host" yaml:"host" env:"APP_HOST"`
    Port     int           `json:"port" yaml:"port" env:"APP_PORT"`
    Timeout  time.Duration `json:"timeout" yaml:"timeout" env:"APP_TIMEOUT"`
    LogLevel string        `json:"log_level" yaml:"log_level" env:"APP_LOG_LEVEL"`
}
```

### Anti-Pattern 4: Slice Capacity Leaks

```go
// BAD: Holding only a part of a large slice -> the underlying array is not GC'd
func getFirstThree(data []byte) []byte {
    return data[:3]  // the entire original data remains in memory
}

// GOOD: Copy to cut off the reference to the underlying array
func getFirstThreeSafe(data []byte) []byte {
    result := make([]byte, 3)
    copy(result, data[:3])
    return result
}

// BAD: append unintentionally modifies the original slice
func appendToSlice(s []int) []int {
    return append(s, 999)  // if there is spare cap, it mutates the original underlying array
}

// GOOD: Use the full slice expression to cap the capacity
func safeSubslice(s []int) []int {
    sub := s[1:3:3]  // s[low:high:max] -> cap = max - low
    return append(sub, 999)  // a new array is guaranteed to be allocated
}
```

### Anti-Pattern 5: Forgetting to Leverage Zero Values

```go
// BAD: Unnecessary initialization
var mu = &sync.Mutex{}     // no need to make this a pointer
var buf = bytes.Buffer{}   // explicit initialization is unnecessary
var wg = &sync.WaitGroup{} // the zero value is sufficient

// GOOD: Leverage the zero value
var mu sync.Mutex           // usable from its zero value
var buf bytes.Buffer         // usable from its zero value
var wg sync.WaitGroup        // usable from its zero value
var once sync.Once           // usable from its zero value

// Examples of types with useful zero values:
// sync.Mutex      -> an unlocked state
// sync.WaitGroup  -> counter of 0
// bytes.Buffer    -> an empty buffer
// strings.Builder -> an empty builder
```

---

## 5. FAQ

### Q1: Do structs have "constructors"?

Go has no constructor syntax, but by convention you use `New<TypeName>`. Example: `func NewUser(name string) *User`. This is the factory function pattern and is well suited for validation and setting default values.

```go
// Basic factory function
func NewUser(name, email string) *User {
    return &User{
        Name:      name,
        Email:     email,
        CreatedAt: time.Now(),
    }
}

// Functional Options pattern (for complex initialization)
type ServerOption func(*Server)

func WithPort(port int) ServerOption {
    return func(s *Server) { s.Port = port }
}

func WithTLS(cert, key string) ServerOption {
    return func(s *Server) {
        s.TLS = true
        s.CertFile = cert
        s.KeyFile = key
    }
}

func NewServer(host string, opts ...ServerOption) *Server {
    s := &Server{
        Host: host,
        Port: 8080, // default value
    }
    for _, opt := range opts {
        opt(s)
    }
    return s
}

// Usage example
srv := NewServer("localhost",
    WithPort(9090),
    WithTLS("cert.pem", "key.pem"),
)
```

### Q2: When should I use value receivers vs pointer receivers?

Rules of thumb: (1) if you mutate the struct, use a pointer receiver; (2) if the struct is large (many fields), use a pointer receiver; (3) for consistency, keep receivers of the same type uniform. When in doubt, prefer a pointer receiver.

Specific criteria:
- **Pointer receiver required**: contains a `sync.Mutex`, mutates fields, large struct (rule of thumb: more than 3 fields)
- **Value receiver appropriate**: small struct (like `time.Time`), immutable methods, custom types over basic types
- **Mixing forbidden**: do not mix value and pointer receivers on the same type (causes issues with interface satisfaction)

### Q3: Is it OK to use the any type (interface{}) a lot?

Avoid it. Since Go 1.18, you can use generics, so limit `any` to situations where you genuinely need to accept arbitrary types (JSON parsing, etc.). It harms type safety, so use concrete types or interfaces whenever possible.

Acceptable uses of `any`:
- Marshal/Unmarshal in `encoding/json`
- Arguments to logging libraries
- Situations that require reflection
- Test helpers

### Q4: Does the field order of a struct affect memory usage?

Yes. The Go compiler inserts padding to align struct fields on memory boundaries. By adjusting the field order, you can reduce padding and improve memory efficiency.

```go
// BAD: Lots of padding (24 bytes)
type BadLayout struct {
    A bool    // 1 byte + 7 bytes padding
    B int64   // 8 bytes
    C bool    // 1 byte + 7 bytes padding
}

// GOOD: Less padding (16 bytes)
type GoodLayout struct {
    B int64   // 8 bytes
    A bool    // 1 byte
    C bool    // 1 byte + 6 bytes padding
}

// You can verify this with the fieldalignment tool
// go install golang.org/x/tools/go/analysis/passes/fieldalignment/cmd/fieldalignment@latest
// fieldalignment -fix ./...
```

### Q5: What is the comparable constraint?

`comparable` is a built-in type constraint introduced in Go 1.18 that constrains types on which the `==` and `!=` operators can be used. It is used for `map` key types and for the element type of generic sets.

Types included in `comparable`: bool, integer types, floating-point types, complex, string, pointer, channel, array (if elements are comparable), struct (if all fields are comparable)

Types not included in `comparable`: slice, map, function

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing and running code to see how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 6. Summary

| Concept | Key Points |
|------|------|
| Basic types | int, float64, string, bool, rune, byte. Explicit bit widths are recommended in some situations |
| struct | A collection of fields. Can have methods. JSON tags control serialization |
| Embedding | Composition, not inheritance. Methods and fields are promoted |
| interface | A method set. Satisfied implicitly. Defining on the consumer side is recommended |
| type assertion | Safely check with `v, ok := i.(Type)` |
| type switch | Branch with `switch v := i.(type)` |
| Zero value | Every type has a zero value. Design for useful zero values |
| Custom types | `type X underlying` improves type safety. Emulate enums with iota |
| Generics | Type parameters in Go 1.18+. Type constraints like `comparable`, `any` |
| Slice | Dynamic array. Be careful about sharing the underlying array. append may reallocate |
| Map | Hash table. Assignment to a nil map panics. Order is non-deterministic |

---

## Recommended Next Reads

- [02-error-handling.md](./02-error-handling.md) -- Error handling
- [03-packages-modules.md](./03-packages-modules.md) -- Packages and modules
- [../03-tools/01-generics.md](../03-tools/01-generics.md) -- Generics

---

## References

1. **The Go Programming Language Specification -- Types** -- https://go.dev/ref/spec#Types
2. **Effective Go -- Interfaces** -- https://go.dev/doc/effective_go#interfaces
3. **Go Blog, "The Laws of Reflection"** -- https://go.dev/blog/laws-of-reflection
4. **Go Blog, "Strings, bytes, runes and characters in Go"** -- https://go.dev/blog/strings
5. **Go Wiki, "SliceTricks"** -- https://go.dev/wiki/SliceTricks
6. **Go Blog, "An Introduction to Generics"** -- https://go.dev/blog/intro-generics
7. **Go Blog, "Maps in Go"** -- https://go.dev/blog/maps
