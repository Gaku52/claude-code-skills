# Compilation vs Interpretation

> The method of transforming source code into an executable program significantly influences the characteristics of a language, the development experience, and runtime performance.

## What You Will Learn in This Chapter

- [ ] Understand the internal workings of compilation and interpretation
- [ ] Understand the mechanisms and benefits of JIT compilation
- [ ] Make informed judgments about the trade-offs of each approach
- [ ] Gain a concrete understanding of each phase of a compiler
- [ ] Grasp runtime optimization techniques
- [ ] Understand modern execution models such as WebAssembly and transpilation
- [ ] Develop the ability to evaluate and select language processing systems


## Prerequisites

The following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [What Is a Programming Language](./00-what-is-programming-language.md)

---

## 1. Compiled Languages

### The Compilation Pipeline

```
Source Code (.c, .rs, .go)
    |
+-----------------------------+
| 1. Lexical Analysis (Lexing) |  Split into tokens
|    int x = 42;               |  -> [int] [x] [=] [42] [;]
+-----------------------------+
| 2. Parsing                   |  Build AST (Abstract Syntax Tree)
|    VariableDecl              |
|    +-- Type: int             |
|    +-- Name: x               |
|    +-- Value: 42             |
+-----------------------------+
| 3. Semantic Analysis         |  Type checking, scope resolution
|    x: int OK                 |
+-----------------------------+
| 4. IR Generation             |  Optimization-friendly intermediate form
|    %x = alloca i32           |
|    store i32 42, i32* %x     |
+-----------------------------+
| 5. Optimization              |  Dead code elimination, inlining
|    Constant folding, loop    |
|    unrolling                 |
+-----------------------------+
| 6. Code Generation           |  Convert to target machine code
|    mov eax, 42               |
+-----------------------------+
| 7. Linking                   |  Combine with libraries
|    Executable binary complete |
+-----------------------------+
    |
Executable File (a.out, .exe)
```

### Detailed Explanation of Each Phase

#### Phase 1: Lexical Analysis (Tokenization)

The lexical analyzer (lexer/tokenizer) splits source code strings into the smallest meaningful units called "tokens."

```python
# Visualization of lexical analysis behavior (pseudo-implementation in Python)

# Input source code
source = 'let total = price * 1.08;'

# Tokenization result
tokens = [
    Token(type='KEYWORD',    value='let',     line=1, col=1),
    Token(type='IDENTIFIER', value='total',   line=1, col=5),
    Token(type='ASSIGN',     value='=',       line=1, col=11),
    Token(type='IDENTIFIER', value='price',   line=1, col=13),
    Token(type='MULTIPLY',   value='*',       line=1, col=19),
    Token(type='FLOAT',      value='1.08',    line=1, col=21),
    Token(type='SEMICOLON',  value=';',       line=1, col=25),
    Token(type='EOF',        value='',        line=1, col=26),
]
```

The lexical analyzer is implemented using regular expressions (finite automata). Each token pattern is defined with regular expressions such as the following.

```python
# Token pattern definition example
import re
from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    KEYWORD = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()
    EQUAL = auto()        # ==
    NOT_EQUAL = auto()    # !=
    LESS_THAN = auto()
    GREATER_THAN = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

# Token patterns (in priority order)
TOKEN_PATTERNS = [
    (r'\s+',                         None),           # Whitespace (skip)
    (r'//[^\n]*',                    None),           # Line comment (skip)
    (r'/\*[\s\S]*?\*/',              None),           # Block comment
    (r'\d+\.\d+',                    TokenType.FLOAT),
    (r'\d+',                         TokenType.INTEGER),
    (r'"[^"]*"',                     TokenType.STRING),
    (r'==',                          TokenType.EQUAL),
    (r'!=',                          TokenType.NOT_EQUAL),
    (r'[a-zA-Z_][a-zA-Z0-9_]*',     None),           # Determined later
    (r'\+',                          TokenType.PLUS),
    (r'-',                           TokenType.MINUS),
    (r'\*',                          TokenType.MULTIPLY),
    (r'/',                           TokenType.DIVIDE),
    (r'=',                           TokenType.ASSIGN),
    (r'\(',                          TokenType.LPAREN),
    (r'\)',                           TokenType.RPAREN),
    (r'\{',                          TokenType.LBRACE),
    (r'\}',                          TokenType.RBRACE),
    (r';',                           TokenType.SEMICOLON),
    (r',',                           TokenType.COMMA),
]

KEYWORDS = {'let', 'fn', 'if', 'else', 'while', 'for', 'return', 'true', 'false'}

class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.source):
            matched = False
            for pattern, token_type in TOKEN_PATTERNS:
                match = re.match(pattern, self.source[self.pos:])
                if match:
                    value = match.group(0)
                    if token_type is not None:
                        tokens.append(Token(token_type, value, self.line, self.column))
                    elif token_type is None and re.match(r'[a-zA-Z_]', value):
                        # Identifier or keyword
                        t = TokenType.KEYWORD if value in KEYWORDS else TokenType.IDENTIFIER
                        tokens.append(Token(t, value, self.line, self.column))

                    # Advance position
                    for ch in value:
                        if ch == '\n':
                            self.line += 1
                            self.column = 1
                        else:
                            self.column += 1
                    self.pos += len(value)
                    matched = True
                    break

            if not matched:
                raise SyntaxError(
                    f"Unexpected character '{self.source[self.pos]}' "
                    f"at line {self.line}, column {self.column}"
                )

        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens
```

#### Phase 2: Syntactic Analysis (Parsing)

The parser converts the token stream into an Abstract Syntax Tree (AST). The main parsing algorithms are as follows.

```
Classification of parsing algorithms:

  Top-down (from top to bottom):
    Recursive descent parser  -- Easiest to implement. Suitable for hand-written parsers
    LL(k) parser              -- k-token lookahead. Generated by ANTLR
    PEG parser                -- Parsing Expression Grammar. No ambiguity
    Pratt parser              -- Easy to handle operator precedence

  Bottom-up (from bottom to top):
    LR(0) parser   -- Simplest LR parser
    SLR parser     -- Simple LR
    LALR(1) parser -- Generated by yacc/bison. Used in many C compilers
    GLR parser     -- Handles ambiguous grammars
```

```python
# Pratt parser implementation example (operator precedence parser)
# Correctly parses "1 + 2 * 3 - 4 / 2"

from dataclasses import dataclass
from typing import Union

@dataclass
class NumberNode:
    value: float

@dataclass
class BinaryOpNode:
    op: str
    left: 'ASTNode'
    right: 'ASTNode'

@dataclass
class UnaryOpNode:
    op: str
    operand: 'ASTNode'

ASTNode = Union[NumberNode, BinaryOpNode, UnaryOpNode]

class PrattParser:
    """Pratt parser: Handles operator precedence and associativity concisely"""

    # Operator precedence (Binding Power)
    PRECEDENCE = {
        '+': (1, 2),    # (left_bp, right_bp)
        '-': (1, 2),
        '*': (3, 4),
        '/': (3, 4),
        '^': (6, 5),    # Right-associative (right_bp < left_bp)
    }

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> ASTNode:
        return self.parse_expression(0)

    def parse_expression(self, min_bp: int) -> ASTNode:
        # Null Denotation (NUD): Parse prefix
        token = self.tokens[self.pos]
        self.pos += 1

        if token.type == TokenType.INTEGER or token.type == TokenType.FLOAT:
            left = NumberNode(float(token.value))
        elif token.type == TokenType.MINUS:
            # Prefix minus (unary operator)
            operand = self.parse_expression(5)  # High precedence
            left = UnaryOpNode('-', operand)
        elif token.type == TokenType.LPAREN:
            left = self.parse_expression(0)
            self.pos += 1  # skip ')'
        else:
            raise SyntaxError(f"Unexpected token: {token}")

        # Left Denotation (LED): Parse infix
        while self.pos < len(self.tokens):
            op_token = self.tokens[self.pos]
            if op_token.value not in self.PRECEDENCE:
                break

            left_bp, right_bp = self.PRECEDENCE[op_token.value]
            if left_bp < min_bp:
                break

            self.pos += 1
            right = self.parse_expression(right_bp)
            left = BinaryOpNode(op_token.value, left, right)

        return left

# Usage example
# Parsing "1 + 2 * 3" yields:
# BinaryOpNode('+', NumberNode(1), BinaryOpNode('*', NumberNode(2), NumberNode(3)))
# That is, 1 + (2 * 3) = 7, correctly parsed
```

#### Phase 3: Semantic Analysis

Semantic analysis detects programs that are syntactically correct but logically invalid.

```
Examples of errors detected in semantic analysis:

  1. Type mismatch
     int x = "hello";       // Assigning string type to int type
     float y = true + 42;   // Adding bool type and int type

  2. Undefined variables/functions
     int result = foo(x);   // foo is undefined
     print(y);              // y is undefined

  3. Scope violation
     {
         int x = 10;
     }
     print(x);              // x is out of scope

  4. Duplicate definition
     int x = 10;
     int x = 20;            // x is defined twice (depends on language)

  5. Access control violation
     private method();       // External access to a private method

  6. Ownership/lifetime violation (Rust-specific)
     let s = String::from("hello");
     let s2 = s;            // Ownership moved to s2
     println!("{}", s);     // s can no longer be used
```

```python
# Symbol table implementation example
class SymbolTable:
    """Manages bindings between variables/functions and their scopes"""

    def __init__(self, parent=None):
        self.symbols: dict[str, dict] = {}
        self.parent: SymbolTable | None = parent

    def define(self, name: str, type_info: str, mutable: bool = True):
        if name in self.symbols:
            raise SemanticError(f"Variable '{name}' is already defined in this scope")
        self.symbols[name] = {
            'type': type_info,
            'mutable': mutable,
            'used': False,
        }

    def lookup(self, name: str) -> dict | None:
        """Search from current scope toward parent scopes"""
        if name in self.symbols:
            self.symbols[name]['used'] = True
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def check_unused(self) -> list[str]:
        """Detect unused variables (for warnings)"""
        return [name for name, info in self.symbols.items() if not info['used']]

# Semantic analyzer
class SemanticAnalyzer:
    def __init__(self):
        self.scope = SymbolTable()  # Global scope
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def enter_scope(self):
        self.scope = SymbolTable(parent=self.scope)

    def exit_scope(self):
        unused = self.scope.check_unused()
        for name in unused:
            self.warnings.append(f"Variable '{name}' is defined but never used")
        self.scope = self.scope.parent

    def analyze_assignment(self, name: str, value_type: str, declared_type: str = None):
        # Type checking
        if declared_type and declared_type != value_type:
            self.errors.append(
                f"Type mismatch: cannot assign {value_type} to {declared_type}"
            )
            return

        # Variable existence check
        existing = self.scope.lookup(name)
        if existing and not existing['mutable']:
            self.errors.append(f"Cannot reassign immutable variable '{name}'")
```

#### Phase 4: Intermediate Representation (IR)

The compiler converts the language-specific AST into an intermediate representation that is easier to optimize.

```llvm
; LLVM IR example: A function that adds two numbers

; Function definition: i32 add(i32 a, i32 b) { return a + b; }
define i32 @add(i32 %a, i32 %b) {
entry:
    %result = add i32 %a, %b
    ret i32 %result
}

; Fibonacci function (recursive version)
define i64 @fib(i64 %n) {
entry:
    %cmp = icmp sle i64 %n, 1
    br i1 %cmp, label %base, label %recursive

base:
    ret i64 %n

recursive:
    %n_minus_1 = sub i64 %n, 1
    %fib1 = call i64 @fib(i64 %n_minus_1)
    %n_minus_2 = sub i64 %n, 2
    %fib2 = call i64 @fib(i64 %n_minus_2)
    %result = add i64 %fib1, %fib2
    ret i64 %result
}

; Characteristics of LLVM IR:
; - SSA form (Static Single Assignment): Each variable is assigned only once
; - Typed: All values have explicit types
; - Target-independent: Can be converted to x86, ARM, RISC-V, etc.
; - Designed to facilitate optimization passes
```

```
Major types of intermediate representations:

  LLVM IR:
    Used by: Clang (C/C++), Rust, Swift, Julia, Zig
    Features: SSA form, rich optimization passes, many backends

  JVM Bytecode:
    Used by: Java, Kotlin, Scala, Clojure, Groovy
    Features: Stack-based VM, platform independent

  .NET IL (CIL):
    Used by: C#, F#, VB.NET
    Features: Similar to JVM bytecode, runs on the .NET runtime

  WebAssembly (Wasm):
    Used by: C, C++, Rust, Go, AssemblyScript
    Features: Executable in both browser and server environments

  GCC GIMPLE / RTL:
    Used by: C, C++, Fortran, Ada (GCC frontends)
    Features: Two-stage process: GIMPLE (high-level IR) -> RTL (low-level IR)
```

#### Phase 5: Optimization

```
Major compiler optimization techniques:

  --- Local Optimization (within a basic block) ---

  1. Constant Folding
     Before: x = 3 + 4
     After:  x = 7

  2. Constant Propagation
     Before: x = 5; y = x * 2
     After:  x = 5; y = 10

  3. Dead Code Elimination
     Before: x = compute(); return 42;
     After:  return 42;  // x is unused, so it's removed

  4. Common Subexpression Elimination
     Before: a = b * c + d; e = b * c + f;
     After:  tmp = b * c; a = tmp + d; e = tmp + f;

  5. Strength Reduction
     Before: x * 2
     After:  x << 1  // Shift is faster than multiplication

  --- Loop Optimization ---

  6. Loop-Invariant Code Motion
     Before: for(i) { x = a * b; arr[i] = x + i; }
     After:  x = a * b; for(i) { arr[i] = x + i; }

  7. Loop Unrolling
     Before: for(i=0; i<100; i++) { process(i); }
     After:  for(i=0; i<100; i+=4) {
                process(i); process(i+1);
                process(i+2); process(i+3);
             }

  8. Loop Vectorization (Auto-Vectorization)
     Before: for(i) { a[i] = b[i] + c[i]; }
     After:  SIMD instructions add 4 elements simultaneously

  --- Interprocedural Optimization ---

  9. Function Inlining
     Before: int square(int x) { return x*x; } ... y = square(5);
     After:  y = 5 * 5;  // -> y = 25; (further constant folding)

  10. Tail Call Optimization
      Before: int factorial(int n, int acc) {
                  if (n <= 1) return acc;
                  return factorial(n-1, n*acc);  // Tail call
              }
      After:  Converted to a loop (prevents stack overflow)

  11. Function Specialization
      Specializes generic functions for specific types

  12. Escape Analysis
      Determines whether heap allocations can be converted to stack allocations
```

```c
// Practical example of optimization: Code changes by GCC optimization levels

// Original C code
int sum_array(const int* arr, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}

// -O0 (no optimization): Straightforward memory access
// All variables are placed in memory
//   mov    DWORD PTR [rbp-4], 0      ; total = 0
//   mov    DWORD PTR [rbp-8], 0      ; i = 0
//   jmp    .L2
// .L3:
//   mov    eax, DWORD PTR [rbp-8]    ; Load i
//   cdqe
//   lea    rdx, [0+rax*4]
//   mov    rax, DWORD PTR [rbp-24]   ; Load arr
//   add    rax, rdx
//   mov    eax, DWORD PTR [rax]      ; Load arr[i]
//   add    DWORD PTR [rbp-4], eax    ; total += arr[i]
//   add    DWORD PTR [rbp-8], 1      ; i++
// .L2:
//   cmp    ... ; i < n

// -O2 (recommended optimization): Register usage, vectorization
// Variables placed in registers, loop unrolling applied
//   xor    eax, eax                  ; total = 0 (register)
//   test   esi, esi                  ; n == 0?
//   jle    .done
// .loop:
//   add    eax, DWORD PTR [rdi]      ; total += *arr
//   add    rdi, 4                    ; arr++
//   dec    esi                       ; n--
//   jnz    .loop
// .done:
//   ret

// -O3 (aggressive optimization): SIMD auto-vectorization
// Uses SSE/AVX to add 4/8 at once
//   vpxor  xmm0, xmm0, xmm0         ; Accumulator register = 0
// .loop:
//   vpaddd xmm0, xmm0, [rdi]        ; Add 4 simultaneously
//   add    rdi, 16
//   dec    ecx
//   jnz    .loop
//   ; Horizontal add for final result
```

#### Phase 6: Code Generation

```
Processing performed during code generation:

  1. Instruction Selection
     Map IR instructions to target machine instructions
     Example: add i32 -> ADD reg, reg (x86)
              add i32 -> ADD Xn, Xn, Xm (ARM64)

  2. Register Allocation
     Assign infinite virtual registers to finite physical registers
     x86-64: 16 general-purpose registers (rax, rbx, rcx, ...)
     ARM64:  31 general-purpose registers (x0-x30)

     When registers are insufficient -> Spill (save to memory)

  3. Instruction Scheduling
     Reorder instructions to maximize pipeline efficiency
     Consider data dependencies

  4. Peephole Optimization
     Replace short instruction sequences with more efficient instructions
     Example: mov rax, 0 -> xor rax, rax (faster)
```

#### Phase 7: Linking

```
Types of linking:

  Static Linking:
    - Embeds library code into the executable file
    - Executable is larger but has fewer dependencies
    - .a files (Unix), .lib files (Windows)
    Example: gcc main.o -static -lm -o program

  Dynamic Linking:
    - Libraries are loaded at runtime
    - Smaller file size, easier library updates
    - .so files (Linux), .dylib (macOS), .dll (Windows)
    Example: gcc main.o -lm -o program

  Link-Time Optimization (LTO):
    - Enables optimization across compilation units
    - Cross-file inlining, dead code elimination
    Example: gcc -flto main.c lib.c -o program
```

```bash
# Verifying the compilation process step by step

# Step 1: Preprocess (macro expansion, includes)
gcc -E main.c -o main.i

# Step 2: Compile (C -> Assembly)
gcc -S main.c -o main.s

# Step 3: Assemble (Assembly -> Object file)
gcc -c main.c -o main.o

# Step 4: Link (Object -> Executable binary)
gcc main.o -o main

# All at once
gcc main.c -o main

# With LLVM/Clang (inspect IR)
clang -S -emit-llvm main.c -o main.ll  # Output LLVM IR
clang -c -emit-llvm main.c -o main.bc  # Output LLVM bitcode
llvm-dis main.bc                        # Bitcode -> text IR
opt -O2 main.ll -o main_opt.ll         # IR-level optimization
llc main_opt.ll -o main.s               # IR -> Assembly
```

### Advantages and Disadvantages of AOT (Ahead-Of-Time) Compilation

```
Advantages:
  + Fast execution speed (pre-optimized)
  + Easy distribution (single binary)
  + Errors detected at compile time
  + Harder to reverse-engineer
  + Short startup time (no warm-up needed)
  + Predictable memory consumption

Disadvantages:
  - Compilation takes time (problematic for large projects)
  - Must compile for each platform
  - Slow incremental development
  - Cannot leverage runtime type information for optimization

Representative Languages:
  C, C++, Rust, Go, Swift, Haskell, Zig
```

### Build Systems for Compiled Languages

```bash
# C/C++: Make / CMake
# Makefile example
CC = gcc
CFLAGS = -Wall -O2
TARGET = myapp
SRCS = main.c util.c parser.c
OBJS = $(SRCS:.c=.o)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
```

```toml
# Rust: Cargo.toml
[package]
name = "myapp"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }

[profile.release]
opt-level = 3      # Maximum optimization
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit (beneficial for optimization)
strip = true        # Strip symbols (reduce binary size)
```

```go
// Go: go.mod
module example.com/myapp

go 1.22

require (
    github.com/gin-gonic/gin v1.9.1
    gorm.io/gorm v1.25.5
)
```

### Techniques for Improving Compilation Speed

```
Improving compilation speed for large projects:

  1. Incremental Compilation
     Only recompile changed files.
     Build systems (Make, cargo, go build) manage this automatically.

  2. Parallel Compilation
     Compile multiple source files simultaneously.
     make -j$(nproc)  # Parallel by number of CPU cores
     cargo build -j8  # 8 parallel jobs

  3. Precompiled Headers (C/C++)
     Pre-compile frequently included headers.
     gcc -x c-header stdafx.h

  4. Module System (C++20 Modules)
     Use modules instead of #include.
     Eliminates redundant header parsing.

  5. Distributed Builds
     Distribute compilation across multiple machines.
     distcc, sccache, icecc

  6. Caching
     Cache compilation results for identical inputs.
     ccache (C/C++), sccache (Rust)

  7. Lower Optimization Levels During Development
     Development: -O0 or -O1 (fast compilation)
     Release: -O2 or -O3 (fast execution)
```

```bash
# Practical examples of improving Rust compilation speed

# Introduce sccache
cargo install sccache
export RUSTC_WRAPPER=sccache

# Cranelift backend (speeds up debug builds)
# .cargo/config.toml
# [unstable]
# codegen-backend = true
# [profile.dev]
# codegen-backend = "cranelift"

# Pre-build dependency crates
cargo build  # First time builds all dependencies (slow)
cargo build  # Second time only diffs (fast)

# Use the mold linker (significantly reduces link time)
# .cargo/config.toml
# [target.x86_64-unknown-linux-gnu]
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

---

## 2. Interpreted Languages

### How Interpreters Work

```
Source Code (.py, .rb)
    |
+-----------------------------+
| 1. Lexing + Parsing          |  Build AST
+-----------------------------+
| 2. Sequential Execution      |  Execute while traversing AST
|    or                        |
|    Bytecode Conversion -> VM |  (CPython, Ruby MRI)
+-----------------------------+
    |
  Execution Result (immediately)
```

### Types of Interpreters

```
1. Tree-Walking Interpreter
   Directly traverses and executes the AST. Simplest but slowest.
   Examples: Early Ruby, Bash, many educational interpreters

   Behavior: Recursively visits AST nodes and executes
             operations corresponding to each node.

2. Bytecode Interpreter
   Compiles source code to bytecode (virtual machine instructions)
   and executes on a virtual machine (VM).

   Examples: CPython, Ruby YARV, Lua, Erlang BEAM

   Behavior: Bytecode is an instruction set for a "virtual CPU."
             Higher abstraction than physical CPU, portable.

3. Register-based vs Stack-based
   Stack-based: JVM, CPython, .NET CLR
     PUSH 3        ; Stack: [3]
     PUSH 4        ; Stack: [3, 4]
     ADD           ; Stack: [7]

   Register-based: Lua VM, Dalvik (Android)
     LOAD  R0, 3   ; R0 = 3
     LOAD  R1, 4   ; R1 = 4
     ADD   R2, R0, R1  ; R2 = 7
```

### CPython's Execution Model

```python
# Python code is internally compiled to bytecode
import dis

def add(a, b):
    return a + b

dis.dis(add)
# Output:
#   LOAD_FAST   0 (a)
#   LOAD_FAST   1 (b)
#   BINARY_ADD
#   RETURN_VALUE

# .pyc files = pre-compiled bytecode
# Automatically cached in __pycache__/
```

```python
# Inspecting bytecode of a more complex function
import dis

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

dis.dis(fibonacci)
# Example output:
#   0 LOAD_FAST                0 (n)
#   2 LOAD_CONST               1 (1)
#   4 COMPARE_OP               1 (<=)
#   6 POP_JUMP_IF_FALSE       12
#   8 LOAD_FAST                0 (n)
#  10 RETURN_VALUE
#  12 LOAD_GLOBAL              0 (fibonacci)
#  14 LOAD_FAST                0 (n)
#  16 LOAD_CONST               1 (1)
#  18 BINARY_SUBTRACT
#  20 CALL_FUNCTION            1
#  22 LOAD_GLOBAL              0 (fibonacci)
#  24 LOAD_FAST                0 (n)
#  26 LOAD_CONST               2 (2)
#  28 BINARY_SUBTRACT
#  30 CALL_FUNCTION            1
#  32 BINARY_ADD
#  34 RETURN_VALUE

# Directly manipulating bytecode (advanced technique)
import types

code = fibonacci.__code__
print(f"Constants: {code.co_consts}")
print(f"Variable names: {code.co_varnames}")
print(f"Stack depth: {code.co_stacksize}")
print(f"Bytecode: {code.co_code.hex()}")
```

### CPython's GIL (Global Interpreter Lock) Problem

```python
# What is CPython's GIL:
# - A lock that protects the entire CPython interpreter
# - Only one thread can execute Python bytecode at a time
# - Multi-threading provides no benefit for CPU-bound tasks

import threading
import time

# CPU-bound task: Multi-threading is ineffective due to the GIL
def cpu_bound_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# Single-threaded
start = time.time()
cpu_bound_task(10_000_000)
cpu_bound_task(10_000_000)
single_time = time.time() - start

# Multi-threaded (not faster due to GIL)
start = time.time()
t1 = threading.Thread(target=cpu_bound_task, args=(10_000_000,))
t2 = threading.Thread(target=cpu_bound_task, args=(10_000_000,))
t1.start(); t2.start()
t1.join(); t2.join()
multi_time = time.time() - start

print(f"Single-threaded: {single_time:.2f}s")
print(f"Multi-threaded:  {multi_time:.2f}s")  # About the same or slower

# Solution 1: multiprocessing (process isolation)
from multiprocessing import Pool

start = time.time()
with Pool(2) as p:
    results = p.map(cpu_bound_task, [10_000_000, 10_000_000])
process_time = time.time() - start
print(f"Multi-process:   {process_time:.2f}s")  # About 2x faster

# Solution 2: For I/O-bound tasks, use asyncio
import asyncio
import aiohttp

async def fetch_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses

# Solution 3: Python 3.13+ free-threaded mode (PEP 703)
# Experimental feature to disable the GIL and enable multi-threading
# python3.13t script.py  # GIL-free mode
```

### Ruby's Execution Model (YARV)

```ruby
# Inspecting Ruby's bytecode
code = RubyVM::InstructionSequence.compile("puts 1 + 2")
puts code.disasm
# Example output:
# == disasm: #<ISeq:<compiled>@<compiled>:1 (1,0)-(1,12)>==========
# 0000 putself                                                          (   1)[Li]
# 0001 putobject_INT2FIX_1_
# 0002 putobject                    2
# 0004 opt_plus                     <calldata!mid:+, argc:1, ARGS_SIMPLE>[CcCr]
# 0006 opt_send_without_block       <calldata!mid:puts, argc:1, FCALL|ARGS_SIMPLE>
# 0008 leave

# Ruby 3.x YJIT (Yet Another JIT)
# JIT compiler is built-in
# ruby --yjit script.rb  # Enable YJIT
# Enabled by default since Ruby 3.3
```

### Lua's Execution Model

```lua
-- Lua: A very lightweight bytecode interpreter
-- Register-based VM (more efficient than stack-based)

-- LuaJIT: Achieves top-tier performance through tracing JIT
-- Achieves execution speeds exceptional for a dynamic language

-- Reasons why Lua is popular for embedded use:
-- 1. Interpreter is very small (~200KB)
-- 2. Excellent C API (easy integration with host languages)
-- 3. Low memory consumption
-- 4. LuaJIT performance approaches C

-- Lua is widely used in game engines, web servers (OpenResty),
-- network devices, Redis scripting, and more
```

### Advantages and Disadvantages of Interpreters

```
Advantages:
  + Immediate execution (REPL, interactive development)
  + Platform independent (runs wherever the interpreter exists)
  + Dynamic language features (eval, metaprogramming)
  + Fast development cycle
  + Easy debugging (source-level error information)
  + Hot reload (instantly reflect code changes)

Disadvantages:
  - Slow execution speed (10-100x difference)
  - Runtime errors are only discovered at execution time
  - Interpreter required in the execution environment
  - High memory consumption (holds AST and bytecode)
  - Complex distribution (dependency management)

Representative Languages:
  Python (CPython), Ruby (MRI), PHP, Perl, Lua
```

### Leveraging REPL (Read-Eval-Print Loop)

```python
# Python's interactive environment is ideal for exploratory programming

# Standard REPL
$ python3
>>> import json
>>> data = {"name": "Alice", "age": 30}
>>> json.dumps(data, indent=2)
'{\n  "name": "Alice",\n  "age": 30\n}'

# IPython: Enhanced REPL
$ ipython
In [1]: import pandas as pd
In [2]: df = pd.read_csv("sales.csv")
In [3]: df.describe()  # Instantly inspect data statistics
In [4]: %timeit df.sort_values("amount")  # Benchmark
In [5]: %debug  # Debug the previous exception


# Jupyter Notebook: Browser-based interactive environment
# Integrates data analysis, visualization, and documentation
```

```javascript
// Node.js REPL
$ node
> const arr = [1, 2, 3, 4, 5]
> arr.filter(x => x % 2 === 0).map(x => x * x)
[ 4, 16 ]
> .help  // Show help
> .exit  // Exit
```

---

## 3. JIT Compilation

### How JIT Works

```
Source Code
    |
Bytecode (pre-compiled)
    |
+-----------------------------------------+
| JIT Compiler (operates during execution) |
|                                          |
| 1. Profiling                             |
|    -> Monitor which code is executed     |
|       frequently                         |
|                                          |
| 2. Hot Spot Detection                    |
|    -> Identify frequently executed code  |
|    (loops, frequently called functions)  |
|                                          |
| 3. Optimizing Compilation                |
|    -> Compile only hot code to machine   |
|       code                               |
|    -> Optimize using runtime type info   |
|                                          |
| 4. Deoptimization                        |
|    -> Fall back to interpreter if        |
|       assumptions are violated           |
+-----------------------------------------+
    |
  Execution (near-native speed after warm-up)
```

### JIT Compiler Optimization Techniques

```
JIT-specific optimizations (impossible with AOT):

  1. Speculative Optimization
     Optimizes based on runtime profiling information.
     Example: "This function's argument is always int" -> skip type check

  2. Type Specialization
     When a dynamically-typed variable actually only holds a specific type,
     generate code specialized for that type.

  3. Inline Cache
     Cache method call resolution results.
     Speed up calls on objects of the same type.

  4. Devirtualization
     Convert virtual method calls to direct calls.
     When it is known at runtime that only one subclass exists.

  5. On-Stack Replacement (OSR)
     Switch a running loop from interpreter to JIT code.
     Allows benefiting from optimization mid-way through long-running loops.
```

```javascript
// V8 (JavaScript) JIT optimization example

// Functions with stable types -> easily optimized
function add(a, b) {
    return a + b;
}

// Always called with number type -> optimized to integer addition machine code
for (let i = 0; i < 1000000; i++) {
    add(i, i + 1);  // -> Optimized to integer addition machine code
}

// If the type changes mid-way, deoptimization occurs
add("hello", " world");  // -> String type! Assumption violated
// JIT compiler discards the optimized code and falls back to interpreter

// === How to write code that V8 can easily optimize ===

// Good: Types are stable
function calculateTotal(items) {
    let total = 0;                    // Always number
    for (let i = 0; i < items.length; i++) {
        total += items[i].price;      // Always number
    }
    return total;
}

// Bad: Types are unstable (Hidden Class changes)
function createUser(name, age) {
    const user = {};
    user.name = name;   // Hidden Class change
    user.age = age;     // Hidden Class change
    if (age > 18) {
        user.adult = true;  // Conditional property -> Hidden Class branch
    }
    return user;
}

// Good: Define all properties from the start
function createUser(name, age) {
    return {
        name: name,
        age: age,
        adult: age > 18,   // Always present -> Hidden Class is stable
    };
}
```

### V8 Engine (JavaScript) Pipeline

```
JavaScript Source Code
    |
  Parser -> AST
    |
  Ignition (Interpreter) -> Bytecode execution
    |  (Hot code detection)
  TurboFan (Optimizing Compiler) -> Machine code
    |  (If assumptions are violated)
  Deoptimize -> Return to Ignition

Performance progression:
  Immediately after startup:  Low speed via interpreter
  After a few seconds:        Sped up by JIT
  After stabilization:        Near-native code performance

V8 Evolution:
  2008: Full-Codegen + Crankshaft
  2017: Ignition + TurboFan (current architecture)
  2023: Maglev (intermediate layer added: Ignition -> Maglev -> TurboFan)
```

### JVM (Java Virtual Machine) Tiered Compilation

```
Java Source Code
    |
  javac -> Bytecode (.class)
    |
+-----------------------------------------+
| JVM Execution Layers                     |
|                                          |
| Level 0: Interpreter                     |
| Level 1: C1 Compiler (basic optimization)|
| Level 2: C1 + Profiling                  |
| Level 3: C1 + Full Profiling             |
| Level 4: C2 Compiler (max optimization)  |
|                                          |
| Optimization progresses gradually        |
| based on execution count                 |
+-----------------------------------------+

GraalVM: Multi-language JIT
  -> Can run Java, JS, Python, Ruby, R on the same VM
  -> AOT compilation (native-image) also available
```

```java
// Flags to inspect JVM JIT optimization

// Display JIT compilation log
// java -XX:+PrintCompilation MyApp

// Example output:
//   42   1       java.lang.String::hashCode (55 bytes)
//   43   2       java.util.HashMap::hash (20 bytes)
//   44   3 %     MyApp::hotLoop @ 5 (30 bytes)
//
// %: OSR (On-Stack Replacement) compilation
// Number: Compilation level

// Change inlining threshold
// java -XX:InlineSmallCode=2000 MyApp

// Enable escape analysis (enabled by default)
// java -XX:+DoEscapeAnalysis MyApp

// GraalVM Native Image (AOT compilation)
// native-image -jar myapp.jar
// -> Startup time: JVM hundreds of ms -> Native Image tens of ms
// -> Memory: JVM hundreds of MB -> Native Image tens of MB
```

### PyPy: Python's JIT Implementation

```python
# PyPy can be 10-100x faster than CPython in some cases

# Example of code slow in CPython that is sped up by PyPy
def matrix_multiply(a, b, n):
    """n x n matrix multiplication (pure Python implementation)"""
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

# CPython: ~10 seconds (n=500)
# PyPy:    ~0.3 seconds (n=500)
# NumPy:   ~0.01 seconds (n=500) <- C extension, fast on either

# Why PyPy is fast:
# 1. Tracing JIT: Detects hot loops and converts to machine code
# 2. Type specialization: Fixes variable types in loops, skips checks
# 3. Unboxing: Handles int/float in registers without heap allocation
# 4. Loop optimization: Guard hoisting, vectorization

# PyPy's limitations:
# - Compatibility issues with C extensions (libraries depending on CPython API)
# - Startup time is slower than CPython (JIT warm-up)
# - Memory consumption can be higher than CPython in some cases
```

### .NET JIT Execution Model

```csharp
// .NET execution pipeline
// C# source code -> Roslyn compiler -> CIL (Common Intermediate Language)
// -> RyuJIT (JIT compiler) -> Native machine code

// CIL example (viewable with IL DASM)
// .method public static int32 Add(int32 a, int32 b) cil managed
// {
//     .maxstack 2
//     ldarg.0      // Push a onto stack
//     ldarg.1      // Push b onto stack
//     add          // Add
//     ret          // Return result
// }

// .NET 8 AOT compilation
// dotnet publish -r linux-x64 -c Release /p:PublishAot=true
// -> Generates native binary (equivalent to JVM's GraalVM native-image)

// .NET tiered compilation
// Tier 0: Minimal optimization (fast startup)
// Tier 1: Full JIT optimization (hot methods)
// R2R (Ready to Run): AOT + JIT hybrid
```

---

## 4. Modern Execution Models

### WebAssembly (Wasm)

```
C/Rust/Go/etc.
    |  Compile
  .wasm (binary format)
    |  Browser / Runtime
  Streaming compilation -> Execution

Features:
  - Near-native execution speed
  - Safe execution in browser (sandbox)
  - Language-agnostic (can be compiled from any language)
  - WASI: Can also run outside the browser (servers, etc.)
```

#### Wasm in Detail

```
WebAssembly Binary Format:

  Magic number: 0x00 0x61 0x73 0x6D ("\0asm")
  Version:      0x01 0x00 0x00 0x00 (version 1)

  Section structure:
    Type Section     - Function signature definitions
    Import Section   - Imports from host environment
    Function Section - Function indices
    Table Section    - Indirect call tables
    Memory Section   - Linear memory definitions
    Global Section   - Global variables
    Export Section   - Exports to host environment
    Start Section    - Entry point
    Element Section  - Table initialization
    Code Section     - Function body bytecode
    Data Section     - Memory initialization data
```

```rust
// Rust -> Wasm compilation example

// lib.rs
#[no_mangle]
pub extern "C" fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

// Build command
// cargo build --target wasm32-unknown-unknown --release

// Integration with JavaScript using wasm-bindgen
// use wasm_bindgen::prelude::*;
//
// #[wasm_bindgen]
// pub fn greet(name: &str) -> String {
//     format!("Hello, {}!", name)
// }
```

```javascript
// Calling Wasm from JavaScript

// Method 1: fetch + instantiate
async function loadWasm() {
    const response = await fetch('fibonacci.wasm');
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);

    const result = instance.exports.fibonacci(40);
    console.log(`fib(40) = ${result}`);
}

// Method 2: Streaming compilation (recommended)
async function loadWasmStreaming() {
    const { instance } = await WebAssembly.instantiateStreaming(
        fetch('fibonacci.wasm')
    );
    return instance.exports;
}

// Data sharing via Wasm's linear memory
async function processArray() {
    const { instance } = await WebAssembly.instantiateStreaming(
        fetch('processor.wasm')
    );

    const memory = instance.exports.memory;
    const buffer = new Float64Array(memory.buffer, 0, 1000);

    // Write data from JavaScript side
    for (let i = 0; i < 1000; i++) {
        buffer[i] = Math.random();
    }

    // Process at high speed on the Wasm side
    const result = instance.exports.process_array(1000);
    console.log(`Result: ${result}`);
}
```

#### WASI (WebAssembly System Interface)

```
WASI: Standard interface for running Wasm outside the browser

  Design Principles:
  - Capability-based Security
  - POSIX-style API (file I/O, networking, etc.)
  - Sandboxed filesystem access

  Runtimes:
  - Wasmtime (Mozilla/Bytecode Alliance)
  - Wasmer
  - WasmEdge
  - wazero (Go implementation)

  Use Cases:
  - Serverless functions (Cloudflare Workers, Fastly Compute@Edge)
  - Plugin systems (Envoy Proxy, Istio)
  - Universal binaries (compile once, run anywhere)
  - Edge computing
```

```bash
# Example of a CLI tool using WASI

# Compile a Rust program for WASI
cargo build --target wasm32-wasi --release

# Run with Wasmtime
wasmtime target/wasm32-wasi/release/myapp.wasm

# Grant file access (sandbox)
wasmtime --dir=/tmp target/wasm32-wasi/release/myapp.wasm

# Grant network access
wasmtime --tcplisten=127.0.0.1:8080 target/wasm32-wasi/release/server.wasm
```

### Transpilation

```
TypeScript -> JavaScript
Kotlin -> JVM Bytecode / JavaScript
Elm -> JavaScript
Sass -> CSS
JSX -> JavaScript

Advantages:
  - Expressiveness of the source language + target's ecosystem
  - Gradual migration possible (introduce TS into a JS project step by step)
```

#### Transpiler Details

```typescript
// TypeScript -> JavaScript transpilation example

// TypeScript source code
interface User {
    name: string;
    age: number;
    email?: string;
}

function greetUser(user: User): string {
    const greeting = `Hello, ${user.name}!`;
    if (user.age >= 18) {
        return `${greeting} Welcome, adult user.`;
    }
    return `${greeting} Welcome, young user.`;
}

const users: User[] = [
    { name: "Alice", age: 30, email: "alice@example.com" },
    { name: "Bob", age: 16 },
];

const messages = users.map(greetUser);

// Transpiled JavaScript (ES2020 target)
"use strict";
function greetUser(user) {
    const greeting = `Hello, ${user.name}!`;
    if (user.age >= 18) {
        return `${greeting} Welcome, adult user.`;
    }
    return `${greeting} Welcome, young user.`;
}
const users = [
    { name: "Alice", age: 30, email: "alice@example.com" },
    { name: "Bob", age: 16 },
];
const messages = users.map(greetUser);

// Key observations:
// - interface completely disappears (type information is not needed at runtime)
// - Function type annotations also disappear
// - Logic is preserved as-is
// - TypeScript's value lies in "compile-time type checking"
```

```
Transpiler target settings:

  TypeScript compile targets:
    ES5:    IE11 support (deprecated). class -> function, expands arrow functions
    ES2015: Outputs class, arrow function, let/const as-is
    ES2020: Supports optional chaining (?.), nullish coalescing (??)
    ES2022: Supports top-level await, class fields
    ESNext: Outputs latest spec as-is

  tsconfig.json example:
    {
      "compilerOptions": {
        "target": "ES2022",
        "module": "NodeNext",
        "strict": true,
        "noUncheckedIndexedAccess": true,
        "noUnusedLocals": true,
        "sourceMap": true,
        "declaration": true,
        "outDir": "./dist"
      }
    }

  Babel presets:
    @babel/preset-env: Automatically transforms based on browser compatibility
    @babel/preset-react: JSX -> React.createElement transformation
    @babel/preset-typescript: TypeScript -> JavaScript
```

### Hybrid Models

Modern languages often combine multiple execution models.

```
Examples of hybrid execution models:

  Java / Kotlin:
    AOT compile (javac) -> Bytecode -> JIT (HotSpot)
    Additionally, AOT is also possible via GraalVM native-image

  C# / F#:
    AOT compile (Roslyn) -> CIL -> JIT (RyuJIT)
    Additionally, native binaries possible via .NET Native AOT

  Python:
    CPython: Source -> Bytecode -> Interpreter
    PyPy:    Source -> Bytecode -> Tracing JIT
    Cython:  Python-like syntax -> C -> AOT
    Mypyc:   Typed Python -> C extension -> AOT
    Nuitka:  Python -> C -> AOT

  JavaScript:
    V8:      Source -> Bytecode (Ignition) -> JIT (TurboFan)
    Bun:     Source -> JavaScriptCore (WebKit) JIT
    Deno:    Source -> V8 JIT
    Static Hermes: Source -> AOT bytecode (for React Native)

  Dart / Flutter:
    Development: JIT (supports hot reload)
    Production:  AOT (fast startup, predictable performance)
```

---

## 5. Performance Comparison

```
Benchmark reference values (Fibonacci recursive n=40):

  Language          Exec Time   Method
  ------------------------------------------
  C (gcc -O2)      0.15s       AOT
  Rust (release)   0.16s       AOT
  Go               0.45s       AOT
  Java             0.55s       JIT
  JavaScript (V8)  0.80s       JIT
  C# (.NET)        0.60s       JIT
  PyPy             1.20s       JIT
  CPython          15.0s       Interpreter
  Ruby (MRI)       12.0s       Interpreter

  * Real application performance varies greatly depending on I/O,
    algorithms, and optimizations. Micro-benchmarks are for reference only.
```

### More Practical Benchmarks

```
Web server throughput comparison (Hello World, wrk benchmark):

  Framework                   req/sec (approx.)    Language
  -----------------------------------------------------------
  actix-web                   500,000+             Rust
  Gin                         200,000+             Go
  Fastify                     70,000+              Node.js (JS)
  Spring Boot (Webflux)       60,000+              Java
  ASP.NET Core                100,000+             C#
  Express                     15,000+              Node.js (JS)
  FastAPI                     10,000+              Python
  Flask                       2,000+               Python
  Rails                       3,000+               Ruby

  * Benchmarks vary significantly based on configuration, hardware, and workload
  * In real applications, DB/external API calls are often the bottleneck
```

```
Memory usage comparison (Hello World web server at startup):

  Language/Runtime          Memory Usage (approx.)
  ------------------------------------------------
  Rust (actix-web)         1-3 MB
  Go (net/http)            5-10 MB
  Node.js (express)        30-50 MB
  Java (Spring Boot)       100-200 MB
  Python (Flask)           20-40 MB
  .NET (ASP.NET Core)      30-60 MB

  * JVM varies greatly depending on initial heap size settings
  * Memory limits are important in container environments
```

### Startup Time Comparison

```
Startup time comparison (CLI tools):

  Language/Runtime          Startup Time (approx.)
  ------------------------------------------------
  C / Rust / Go            1-10 ms
  .NET Native AOT          10-30 ms
  GraalVM Native Image     10-30 ms
  Node.js                  30-100 ms
  Python                   30-50 ms
  JVM (Java)               100-500 ms
  JVM (Spring Boot)        1-5 sec

  Startup time is important for CLI tools and serverless functions:
  - AWS Lambda: Cold start latency
  - Docker: Container startup speed
  - CLI tools: User experience (users want immediate results)
```

### Approaches to Performance Tuning

```
Priority order for performance optimization:

  1. Improve the algorithm (greatest effect)
     O(n^2) -> O(n log n)  Example: Bubble sort -> Merge sort
     Effect: 100x-10000x improvement possible

  2. Data structure selection
     Array vs linked list vs hash table vs B-Tree
     Consider cache efficiency (memory locality)

  3. I/O optimization
     Async I/O, batch processing, connection pooling
     Eliminate N+1 problems (DB)

  4. Parallelization/Concurrency
     Multi-threading, async/await, worker pools
     Be mindful of Amdahl's Law

  5. Language/runtime-level optimization
     Compiler flags, GC tuning, memory pools
     This becomes necessary only after optimizing the above 4

  6. Change the language (last resort)
     Rewrite only hot spots in a faster language
     Example: Replace CPU-bound parts of Python with Rust/C extensions
```

---

## 6. Comparison and Selection of Language Processing Systems

### Multiple Implementations of the Same Language

```
Python implementations:
  CPython    -- Standard implementation. Interpreter written in C
  PyPy       -- Interpreter with JIT. Can be 10-100x faster than CPython
  Jython     -- Python running on the JVM
  IronPython -- Python running on .NET
  MicroPython -- Lightweight implementation for microcontrollers
  Cython     -- Compiler that converts Python syntax to C
  Mypyc      -- Converts typed Python to C extensions

JavaScript implementations:
  V8         -- Used in Chrome, Node.js, Deno (Google)
  SpiderMonkey -- Used in Firefox (Mozilla)
  JavaScriptCore -- Used in Safari, Bun (Apple)
  Hermes     -- For React Native (Meta)
  QuickJS    -- Lightweight embedded JavaScript

Ruby implementations:
  CRuby/MRI  -- Standard implementation (Matz's Ruby Interpreter)
  JRuby      -- Ruby running on the JVM
  TruffleRuby -- Ruby running on GraalVM (fast)
  mruby      -- Lightweight Ruby for embedded use

Scheme implementations:
  Racket, Guile, Chez Scheme, Gambit, Chicken
  -> Performance and features vary greatly even within the same language spec
```

### Compiler Infrastructure

```
LLVM:
  Used by: Clang (C/C++), Rust, Swift, Julia, Zig,
           Kotlin/Native, Crystal, Mojo
  Features: Modular design, rich optimization passes, many backends
            De facto standard for building compilers

GCC:
  Used by: C, C++, Fortran, Ada, Go (gccgo)
  Features: Longest history, many platform support
            Was the de facto standard before LLVM

Cranelift:
  Used by: Wasmtime, Rustc (experimental for debug builds)
  Features: Fast code generation, designed for JIT
            Weaker optimization than LLVM but faster compilation

GraalVM Compiler:
  Used by: Java, JavaScript, Python, Ruby, R
  Features: Partial evaluation-based JIT, multi-language interop
            Easy to add new languages via the Truffle framework
```

---

## Practical Exercises

### Exercise 1: [Basic] -- Inspecting Bytecode

Use Python's `dis` module to examine the bytecode of simple functions.

```python
# Exercise: Inspect the bytecode of the following functions using dis.dis()
import dis

# Function 1: Simple conditional branch
def max_value(a, b):
    if a > b:
        return a
    return b

# Function 2: List comprehension
def square_evens(numbers):
    return [n * n for n in numbers if n % 2 == 0]

# Function 3: Generator
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Inspect the bytecode of each function
print("=== max_value ===")
dis.dis(max_value)
print("\n=== square_evens ===")
dis.dis(square_evens)
print("\n=== fibonacci_gen ===")
dis.dis(fibonacci_gen)

# Discussion points:
# - Behavior of COMPARE_OP, POP_JUMP_IF_FALSE
# - Internal implementation of list comprehension (generated as a separate function)
# - Bytecode representation of yield
```

### Exercise 2: [Applied] -- Measuring the Effect of JIT

Run the same algorithm with CPython and PyPy to compare the effect of JIT.

```python
# benchmark.py: Run with CPython and PyPy and compare
import time

def benchmark_loop(n):
    """Pure CPU computation (benefits most from JIT)"""
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total += i * i
        else:
            total -= i
    return total

def benchmark_string(n):
    """String operations (frequent memory allocation)"""
    result = ""
    for i in range(n):
        result += str(i)
    return len(result)

def benchmark_dict(n):
    """Dictionary operations (hash table)"""
    d = {}
    for i in range(n):
        d[i] = i * i
    total = sum(d.values())
    return total

benchmarks = [
    ("Loop computation", benchmark_loop, 10_000_000),
    ("String concatenation", benchmark_string, 100_000),
    ("Dictionary operations", benchmark_dict, 1_000_000),
]

for name, func, n in benchmarks:
    start = time.time()
    result = func(n)
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s (result={result})")

# How to run:
# python3 benchmark.py     # CPython
# pypy3 benchmark.py       # PyPy
```

### Exercise 3: [Advanced] -- Implementing a Simple Interpreter

Implement a simple interpreter in Python that evaluates arithmetic expressions.

```python
# Exercise: Extend the following interpreter
# 1. Add variable assignment and reference (let x = 10; x + 5)
# 2. Add comparison operators (==, !=, <, >)
# 3. Add if-else statements
# 4. Add function definition and invocation

from dataclasses import dataclass
from typing import Union

# === AST Nodes ===
@dataclass
class Number:
    value: float

@dataclass
class BinaryOp:
    op: str
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnaryOp:
    op: str
    operand: 'Expr'

Expr = Union[Number, BinaryOp, UnaryOp]

# === Lexer ===
def tokenize(source: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(source):
        if source[i].isspace():
            i += 1
        elif source[i].isdigit() or source[i] == '.':
            j = i
            while j < len(source) and (source[j].isdigit() or source[j] == '.'):
                j += 1
            tokens.append(source[i:j])
            i = j
        elif source[i] in '+-*/()':
            tokens.append(source[i])
            i += 1
        else:
            raise SyntaxError(f"Unexpected character: {source[i]}")
    return tokens

# === Parser (Recursive Descent) ===
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self):
        result = self.expression()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"Unexpected token: {self.peek()}")
        return result

    def expression(self):
        left = self.term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.term()
            left = BinaryOp(op, left, right)
        return left

    def term(self):
        left = self.unary()
        while self.peek() in ('*', '/'):
            op = self.consume()
            right = self.unary()
            left = BinaryOp(op, left, right)
        return left

    def unary(self):
        if self.peek() == '-':
            self.consume()
            operand = self.factor()
            return UnaryOp('-', operand)
        return self.factor()

    def factor(self):
        token = self.peek()
        if token == '(':
            self.consume()
            expr = self.expression()
            if self.consume() != ')':
                raise SyntaxError("Expected ')'")
            return expr
        else:
            self.consume()
            return Number(float(token))

# === Evaluator ===
def evaluate(node: Expr) -> float:
    match node:
        case Number(value):
            return value
        case BinaryOp('+', left, right):
            return evaluate(left) + evaluate(right)
        case BinaryOp('-', left, right):
            return evaluate(left) - evaluate(right)
        case BinaryOp('*', left, right):
            return evaluate(left) * evaluate(right)
        case BinaryOp('/', left, right):
            divisor = evaluate(right)
            if divisor == 0:
                raise ZeroDivisionError("Division by zero")
            return evaluate(left) / divisor
        case UnaryOp('-', operand):
            return -evaluate(operand)

# === REPL ===
def repl():
    print("Simple Calculator (type 'quit' to exit)")
    while True:
        try:
            line = input(">>> ")
            if line.strip().lower() == 'quit':
                break
            tokens = tokenize(line)
            ast = Parser(tokens).parse()
            result = evaluate(ast)
            print(f"= {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    repl()
```

### Exercise 4: [Advanced] -- Experiencing Wasm

```bash
# Exercise: Compile Rust code to Wasm and run it in a browser

# 1. Install tools
# rustup target add wasm32-unknown-unknown
# cargo install wasm-pack

# 2. Create project
# cargo new --lib wasm-demo
# cd wasm-demo

# 3. Add to Cargo.toml
# [lib]
# crate-type = ["cdylib"]
# [dependencies]
# wasm-bindgen = "0.2"

# 4. Edit src/lib.rs
# use wasm_bindgen::prelude::*;
#
# #[wasm_bindgen]
# pub fn fibonacci(n: u32) -> u64 {
#     match n {
#         0 => 0,
#         1 => 1,
#         _ => {
#             let mut a: u64 = 0;
#             let mut b: u64 = 1;
#             for _ in 2..=n {
#                 let temp = a + b;
#                 a = b;
#                 b = temp;
#             }
#             b
#         }
#     }
# }

# 5. Build
# wasm-pack build --target web

# 6. Call from HTML (index.html)
# <script type="module">
#     import init, { fibonacci } from './pkg/wasm_demo.js';
#     await init();
#     console.log(fibonacci(50));
# </script>
```

---

## FAQ

### Q1: Is "compiled = fast" always true?
A: Generally true, but there are exceptions. JIT can sometimes achieve better optimization than AOT by using runtime information. In practice, algorithm selection matters far more.

### Q2: Is TypeScript a compiled language?
A: TypeScript is transpiled to JavaScript. Type checking happens at compile time, but at runtime the JavaScript engine (JIT) runs. In terms of classification, it is a transpiled language.

### Q3: Why is Go fast?
A: Static typing + AOT compilation + optimized garbage collector + efficient concurrency via goroutines. The simple language specification makes compiler optimization easier.

### Q4: How do you deal with JIT warm-up problems?
A: There are several approaches:
- **AOT compilation**: Generate native binaries with GraalVM native-image or .NET Native AOT
- **Profile-Guided Optimization (PGO)**: Optimize using pre-collected profile information
- **Tiered compilation**: Gradually increase optimization levels starting from a low level (JVM)
- **Ahead-of-time JIT (AOT JIT)**: Save and reuse past execution profiles
- **Server design**: Gradually increase traffic during the warm-up period

### Q5: Will WebAssembly replace JavaScript?
A: It complements rather than replaces JavaScript. Wasm is suited for CPU-intensive tasks (image processing, encryption, game engines), while JavaScript is suited for DOM manipulation and UI control. Real applications use both in combination.

### Q6: Why is Rust's compilation slow?
A: The main causes are:
- **Ownership checking (borrow checker)**: Computational cost for verifying memory safety
- **Monomorphization**: Generates large amounts of code when instantiating generics
- **LLVM optimization passes**: Powerful but time-consuming
- **Recompilation of dependency crates**: Build time increases with many dependencies
- Mitigations: Use sccache, cranelift backend, mold linker

### Q7: Is an AOT+JIT hybrid the optimal solution?
A: In many cases, yes. Dart/Flutter uses JIT during development (hot reload) and AOT in production (fast startup). GraalVM's Profile-Guided Optimization (PGO) is also an approach that combines insights from both AOT and JIT.

---

## Summary

| Method | Representative Languages | Exec Speed | Dev Speed | Portability |
|--------|------------------------|-----------|-----------|-------------|
| AOT Compilation | C, Rust, Go | Fastest | Slower | Low (recompilation needed) |
| Interpreter | Python, Ruby | Slow | Fastest | High (if interpreter exists) |
| JIT | Java, JS | Fast | Medium | High (runs on VM) |
| Transpilation | TS, Kotlin | Target-dependent | Medium | Target-dependent |
| Wasm | C, Rust->Wasm | Near-native | Medium | Very high |

| Optimization Phase | Content | Effect |
|-------------------|---------|--------|
| Lexical Analysis | Token splitting | Preprocessing for parsing |
| Parsing | AST construction | Structuring the program |
| Semantic Analysis | Type checking, scope resolution | Detecting invalid programs |
| IR Generation | Conversion to intermediate representation | Target-independent optimization |
| Optimization | Constant folding, inlining, vectorization | Improving execution speed |
| Code Generation | Machine code generation | Producing the executable binary |
| Linking | Library combination | Producing the complete executable |

---

## Recommended Next Reading

---

## References
1. Aho, A., Lam, M., Sethi, R. & Ullman, J. "Compilers: Principles, Techniques, and Tools." 2nd Ed, 2006.
2. Nystrom, R. "Crafting Interpreters." 2021.
3. Cooper, K. & Torczon, L. "Engineering a Compiler." 3rd Ed, 2022.
4. Appel, A. "Modern Compiler Implementation in ML." Cambridge University Press, 2004.
5. Aycock, J. "A Brief History of Just-In-Time." ACM Computing Surveys, 2003.
6. Haas, A. et al. "Bringing the Web up to Speed with WebAssembly." PLDI, 2017.
7. Bolz, C. et al. "Tracing the Meta-level: PyPy's Tracing JIT Compiler." ICOOOLPS, 2009.
8. Lattner, C. & Adve, V. "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation." CGO, 2004.
9. Wurthinger, T. et al. "Practical Partial Evaluation for High-Performance Dynamic Language Runtimes." PLDI, 2017.
10. Leroy, X. "The Compcert Verified Compiler." 2009.
