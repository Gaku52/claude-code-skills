# コンパイルとインタプリタ

> ソースコードから実行可能なプログラムへの変換方法は、言語の特性・開発体験・実行性能を大きく左右する。

## この章で学ぶこと

- [ ] コンパイルとインタプリタの内部動作を理解する
- [ ] JIT コンパイルの仕組みと利点を理解する
- [ ] 各方式のトレードオフを判断できる
- [ ] コンパイラの各フェーズを具体的に理解する
- [ ] 実行時最適化の技法を把握する
- [ ] WebAssembly やトランスパイルなど現代の実行モデルを理解する
- [ ] 言語処理系を選定・評価する力を身につける

---

## 1. コンパイル型言語

### コンパイルの流れ

```
ソースコード（.c, .rs, .go）
    ↓
┌─────────────────────────────┐
│ 1. 字句解析（Lexing）        │  トークンに分割
│    int x = 42;               │  → [int] [x] [=] [42] [;]
├─────────────────────────────┤
│ 2. 構文解析（Parsing）       │  AST（抽象構文木）を構築
│    VariableDecl              │
│    ├── Type: int             │
│    ├── Name: x               │
│    └── Value: 42             │
├─────────────────────────────┤
│ 3. 意味解析（Semantic）      │  型チェック、スコープ解決
│    x: int ✓                  │
├─────────────────────────────┤
│ 4. 中間表現（IR）生成        │  最適化しやすい中間形式
│    %x = alloca i32           │
│    store i32 42, i32* %x     │
├─────────────────────────────┤
│ 5. 最適化                    │  デッドコード削除、インライン化
│    定数畳み込み、ループ展開   │
├─────────────────────────────┤
│ 6. コード生成                │  ターゲット機械語に変換
│    mov eax, 42               │
├─────────────────────────────┤
│ 7. リンク                    │  ライブラリと結合
│    実行可能バイナリ完成       │
└─────────────────────────────┘
    ↓
実行可能ファイル（a.out, .exe）
```

### 各フェーズの詳細解説

#### フェーズ1: 字句解析（Lexical Analysis / Tokenization）

字句解析器（レキサー/トークナイザー）はソースコードの文字列を「トークン」と呼ばれる意味のある最小単位に分割する。

```python
# 字句解析の動作イメージ（Pythonで擬似実装）

# 入力ソースコード
source = 'let total = price * 1.08;'

# トークン化結果
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

字句解析器は正規表現（有限オートマトン）で実装される。各トークンパターンは以下のような正規表現で定義される。

```python
# トークンパターンの定義例
import re
from dataclasses import dataclass
from enum import Enum, auto

class TokenType(Enum):
    # リテラル
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()

    # 識別子とキーワード
    IDENTIFIER = auto()
    KEYWORD = auto()

    # 演算子
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    ASSIGN = auto()
    EQUAL = auto()        # ==
    NOT_EQUAL = auto()    # !=
    LESS_THAN = auto()
    GREATER_THAN = auto()

    # 区切り文字
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    SEMICOLON = auto()
    COMMA = auto()

    # 特殊
    EOF = auto()
    NEWLINE = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

# トークンパターン（優先順位順）
TOKEN_PATTERNS = [
    (r'\s+',                         None),           # 空白（スキップ）
    (r'//[^\n]*',                    None),           # 行コメント（スキップ）
    (r'/\*[\s\S]*?\*/',              None),           # ブロックコメント
    (r'\d+\.\d+',                    TokenType.FLOAT),
    (r'\d+',                         TokenType.INTEGER),
    (r'"[^"]*"',                     TokenType.STRING),
    (r'==',                          TokenType.EQUAL),
    (r'!=',                          TokenType.NOT_EQUAL),
    (r'[a-zA-Z_][a-zA-Z0-9_]*',     None),           # 後で判定
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
                        # 識別子 or キーワード
                        t = TokenType.KEYWORD if value in KEYWORDS else TokenType.IDENTIFIER
                        tokens.append(Token(t, value, self.line, self.column))

                    # 位置を進める
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

#### フェーズ2: 構文解析（Syntactic Analysis / Parsing）

構文解析器（パーサ）はトークン列を抽象構文木（AST）に変換する。主要な構文解析アルゴリズムは以下の通り。

```
構文解析アルゴリズムの分類:

  トップダウン（上から下へ）:
    再帰下降パーサ  — 最も実装しやすい。手書きに適する
    LL(k) パーサ    — k トークン先読み。ANTLR が生成
    PEG パーサ      — Parsing Expression Grammar。曖昧さがない
    Pratt パーサ    — 演算子の優先順位を扱いやすい

  ボトムアップ（下から上へ）:
    LR(0) パーサ   — 最も単純なLRパーサ
    SLR パーサ     — Simple LR
    LALR(1) パーサ — yacc/bison が生成。多くのCコンパイラで使用
    GLR パーサ     — 曖昧な文法に対応
```

```python
# Prattパーサの実装例（演算子優先順位パーサ）
# 算術式 "1 + 2 * 3 - 4 / 2" を正しくパースする

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
    """Prattパーサ: 演算子の優先順位と結合性を簡潔に扱える"""

    # 演算子の優先順位（Binding Power）
    PRECEDENCE = {
        '+': (1, 2),    # (left_bp, right_bp)
        '-': (1, 2),
        '*': (3, 4),
        '/': (3, 4),
        '^': (6, 5),    # 右結合（right_bp < left_bp）
    }

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> ASTNode:
        return self.parse_expression(0)

    def parse_expression(self, min_bp: int) -> ASTNode:
        # Null Denotation（NUD）: 前置の解析
        token = self.tokens[self.pos]
        self.pos += 1

        if token.type == TokenType.INTEGER or token.type == TokenType.FLOAT:
            left = NumberNode(float(token.value))
        elif token.type == TokenType.MINUS:
            # 前置マイナス（単項演算子）
            operand = self.parse_expression(5)  # 高い優先順位
            left = UnaryOpNode('-', operand)
        elif token.type == TokenType.LPAREN:
            left = self.parse_expression(0)
            self.pos += 1  # skip ')'
        else:
            raise SyntaxError(f"Unexpected token: {token}")

        # Left Denotation（LED）: 中置の解析
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

# 使用例
# "1 + 2 * 3" をパースすると:
# BinaryOpNode('+', NumberNode(1), BinaryOpNode('*', NumberNode(2), NumberNode(3)))
# つまり 1 + (2 * 3) = 7 と正しく解析される
```

#### フェーズ3: 意味解析（Semantic Analysis）

意味解析では、構文的には正しいが論理的に不正なプログラムを検出する。

```
意味解析で検出されるエラーの例:

  1. 型の不一致
     int x = "hello";       // int型にstring型を代入
     float y = true + 42;   // bool型とint型の加算

  2. 未定義の変数・関数
     int result = foo(x);   // fooが未定義
     print(y);              // yが未定義

  3. スコープ違反
     {
         int x = 10;
     }
     print(x);              // xはスコープ外

  4. 重複定義
     int x = 10;
     int x = 20;            // xが二重定義（言語による）

  5. アクセス権違反
     private method();       // privateメソッドへの外部アクセス

  6. 所有権・ライフタイム違反（Rust固有）
     let s = String::from("hello");
     let s2 = s;            // 所有権がs2に移動
     println!("{}", s);     // sはもう使えない
```

```python
# シンボルテーブルの実装例
class SymbolTable:
    """変数・関数のスコープとの紐づけを管理"""

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
        """現在のスコープから親スコープへ向かって検索"""
        if name in self.symbols:
            self.symbols[name]['used'] = True
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def check_unused(self) -> list[str]:
        """未使用の変数を検出（警告用）"""
        return [name for name, info in self.symbols.items() if not info['used']]

# 意味解析器
class SemanticAnalyzer:
    def __init__(self):
        self.scope = SymbolTable()  # グローバルスコープ
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
        # 型チェック
        if declared_type and declared_type != value_type:
            self.errors.append(
                f"Type mismatch: cannot assign {value_type} to {declared_type}"
            )
            return

        # 変数の存在チェック
        existing = self.scope.lookup(name)
        if existing and not existing['mutable']:
            self.errors.append(f"Cannot reassign immutable variable '{name}'")
```

#### フェーズ4: 中間表現（IR: Intermediate Representation）

コンパイラは言語固有のASTを、最適化しやすい中間表現に変換する。

```llvm
; LLVM IR の例: 2つの数値を加算する関数

; 関数定義: i32 add(i32 a, i32 b) { return a + b; }
define i32 @add(i32 %a, i32 %b) {
entry:
    %result = add i32 %a, %b
    ret i32 %result
}

; フィボナッチ関数（再帰版）
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

; LLVM IRの特徴:
; - SSA形式（Static Single Assignment）: 各変数は一度だけ代入される
; - 型付き: 全ての値が明示的な型を持つ
; - ターゲット非依存: x86, ARM, RISC-V 等に変換可能
; - 最適化パスが適用しやすい設計
```

```
主要な中間表現の種類:

  LLVM IR:
    使用言語: Clang(C/C++), Rust, Swift, Julia, Zig
    特徴: SSA形式、豊富な最適化パス、多数のバックエンド

  JVM バイトコード:
    使用言語: Java, Kotlin, Scala, Clojure, Groovy
    特徴: スタックベースVM、プラットフォーム非依存

  .NET IL (CIL):
    使用言語: C#, F#, VB.NET
    特徴: JVMバイトコードに類似、.NETランタイム上で実行

  WebAssembly (Wasm):
    使用言語: C, C++, Rust, Go, AssemblyScript
    特徴: ブラウザ・サーバー両方で実行可能

  GCC GIMPLE / RTL:
    使用言語: C, C++, Fortran, Ada（GCCフロントエンド）
    特徴: GIMPLE（高レベルIR）→ RTL（低レベルIR）の2段階
```

#### フェーズ5: 最適化（Optimization）

```
コンパイラ最適化の主要技法:

  ─── ローカル最適化（基本ブロック内） ───

  1. 定数畳み込み（Constant Folding）
     変換前: x = 3 + 4
     変換後: x = 7

  2. 定数伝播（Constant Propagation）
     変換前: x = 5; y = x * 2
     変換後: x = 5; y = 10

  3. デッドコード除去（Dead Code Elimination）
     変換前: x = compute(); return 42;
     変換後: return 42;  // xは使われないので削除

  4. 共通部分式の除去（Common Subexpression Elimination）
     変換前: a = b * c + d; e = b * c + f;
     変換後: tmp = b * c; a = tmp + d; e = tmp + f;

  5. 強度削減（Strength Reduction）
     変換前: x * 2
     変換後: x << 1  // 乗算よりシフトの方が高速

  ─── ループ最適化 ───

  6. ループ不変式の移動（Loop-Invariant Code Motion）
     変換前: for(i) { x = a * b; arr[i] = x + i; }
     変換後: x = a * b; for(i) { arr[i] = x + i; }

  7. ループアンローリング（Loop Unrolling）
     変換前: for(i=0; i<100; i++) { process(i); }
     変換後: for(i=0; i<100; i+=4) {
                process(i); process(i+1);
                process(i+2); process(i+3);
             }

  8. ループベクトル化（Auto-Vectorization）
     変換前: for(i) { a[i] = b[i] + c[i]; }
     変換後: SIMD命令で4要素同時加算

  ─── 手続き間最適化 ───

  9. インライン展開（Function Inlining）
     変換前: int square(int x) { return x*x; } ... y = square(5);
     変換後: y = 5 * 5;  // → y = 25;（さらに定数畳み込み）

  10. 末尾呼び出し最適化（Tail Call Optimization）
      変換前: int factorial(int n, int acc) {
                  if (n <= 1) return acc;
                  return factorial(n-1, n*acc);  // 末尾呼び出し
              }
      変換後: ループに変換（スタックオーバーフローを防止）

  11. 関数の特殊化（Function Specialization）
      ジェネリック関数を特定の型に対して特殊化する

  12. エスケープ解析（Escape Analysis）
      ヒープ確保をスタック確保に変換できるか判定する
```

```c
// 最適化の実例: GCCの最適化レベルによるコード変化

// 元のCコード
int sum_array(const int* arr, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}

// -O0（最適化なし）: 素直にメモリアクセス
// 全ての変数がメモリ上に配置される
//   mov    DWORD PTR [rbp-4], 0      ; total = 0
//   mov    DWORD PTR [rbp-8], 0      ; i = 0
//   jmp    .L2
// .L3:
//   mov    eax, DWORD PTR [rbp-8]    ; i をロード
//   cdqe
//   lea    rdx, [0+rax*4]
//   mov    rax, DWORD PTR [rbp-24]   ; arr をロード
//   add    rax, rdx
//   mov    eax, DWORD PTR [rax]      ; arr[i] をロード
//   add    DWORD PTR [rbp-4], eax    ; total += arr[i]
//   add    DWORD PTR [rbp-8], 1      ; i++
// .L2:
//   cmp    ... ; i < n

// -O2（推奨最適化）: レジスタ活用、ベクトル化
// 変数はレジスタに配置、ループアンローリング適用
//   xor    eax, eax                  ; total = 0（レジスタ）
//   test   esi, esi                  ; n == 0?
//   jle    .done
// .loop:
//   add    eax, DWORD PTR [rdi]      ; total += *arr
//   add    rdi, 4                    ; arr++
//   dec    esi                       ; n--
//   jnz    .loop
// .done:
//   ret

// -O3（積極的最適化）: SIMD自動ベクトル化
// SSE/AVXを使って4個/8個を同時加算
//   vpxor  xmm0, xmm0, xmm0         ; 累積レジスタ = 0
// .loop:
//   vpaddd xmm0, xmm0, [rdi]        ; 4個同時加算
//   add    rdi, 16
//   dec    ecx
//   jnz    .loop
//   ; 水平加算で最終結果を得る
```

#### フェーズ6: コード生成（Code Generation）

```
コード生成で行われる処理:

  1. 命令選択（Instruction Selection）
     IR命令をターゲットの機械語命令にマッピング
     例: add i32 → ADD reg, reg（x86）
         add i32 → ADD Xn, Xn, Xm（ARM64）

  2. レジスタ割り当て（Register Allocation）
     無限の仮想レジスタを有限の物理レジスタに割り当て
     x86-64: 16個の汎用レジスタ（rax, rbx, rcx, ...）
     ARM64:  31個の汎用レジスタ（x0〜x30）

     レジスタが足りない場合 → スピル（メモリへ退避）

  3. 命令スケジューリング（Instruction Scheduling）
     パイプラインの効率を最大化するように命令を並べ替え
     データ依存関係を考慮

  4. ピープホール最適化（Peephole Optimization）
     短い命令列パターンをより効率的な命令に置換
     例: mov rax, 0 → xor rax, rax（より高速）
```

#### フェーズ7: リンク（Linking）

```
リンクの種類:

  静的リンク（Static Linking）:
    - ライブラリのコードを実行ファイルに埋め込む
    - 実行ファイルが大きくなるが依存が減る
    - .a ファイル（Unix）、.lib ファイル（Windows）
    例: gcc main.o -static -lm -o program

  動的リンク（Dynamic Linking）:
    - 実行時にライブラリをロードする
    - ファイルサイズが小さい、ライブラリの更新が容易
    - .so ファイル（Linux）、.dylib（macOS）、.dll（Windows）
    例: gcc main.o -lm -o program

  リンク時最適化（LTO: Link-Time Optimization）:
    - コンパイル単位を超えた最適化が可能
    - ファイル間のインライン展開、デッドコード除去
    例: gcc -flto main.c lib.c -o program
```

```bash
# 実際のコンパイルプロセスを段階的に確認する

# ステップ1: プリプロセス（マクロ展開、インクルード）
gcc -E main.c -o main.i

# ステップ2: コンパイル（C → アセンブリ）
gcc -S main.c -o main.s

# ステップ3: アセンブル（アセンブリ → オブジェクトファイル）
gcc -c main.c -o main.o

# ステップ4: リンク（オブジェクト → 実行可能バイナリ）
gcc main.o -o main

# 全て一括で実行
gcc main.c -o main

# LLVM/Clangの場合（IRを確認）
clang -S -emit-llvm main.c -o main.ll  # LLVM IR出力
clang -c -emit-llvm main.c -o main.bc  # LLVM ビットコード出力
llvm-dis main.bc                        # ビットコード → テキストIR
opt -O2 main.ll -o main_opt.ll         # IRレベルの最適化
llc main_opt.ll -o main.s               # IR → アセンブリ
```

### AOT（Ahead-Of-Time）コンパイルの利点と欠点

```
利点:
  ✓ 実行速度が高速（事前に最適化済み）
  ✓ 配布が容易（バイナリ1つ）
  ✓ コンパイル時にエラーを検出
  ✓ リバースエンジニアリングが困難
  ✓ 起動時間が短い（ウォームアップ不要）
  ✓ メモリ消費が予測可能

欠点:
  ✗ コンパイル時間がかかる（大規模プロジェクトで問題）
  ✗ プラットフォームごとにコンパイルが必要
  ✗ インクリメンタルな開発が遅い
  ✗ 実行時の型情報を活用した最適化ができない

代表的な言語:
  C, C++, Rust, Go, Swift, Haskell, Zig
```

### コンパイル型言語のビルドシステム

```bash
# C/C++: Make / CMake
# Makefile の例
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
opt-level = 3      # 最大最適化
lto = true          # リンク時最適化
codegen-units = 1   # 単一コード生成ユニット（最適化に有利）
strip = true        # シンボル除去（バイナリサイズ削減）
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

### コンパイル速度の改善手法

```
大規模プロジェクトでのコンパイル速度改善:

  1. インクリメンタルコンパイル
     変更されたファイルだけを再コンパイルする。
     ビルドシステム（Make, cargo, go build）が自動管理。

  2. 並列コンパイル
     複数のソースファイルを同時にコンパイルする。
     make -j$(nproc)  # CPUコア数分並列
     cargo build -j8  # 8並列

  3. プリコンパイル済みヘッダ（C/C++）
     頻繁にインクルードされるヘッダを事前コンパイル。
     gcc -x c-header stdafx.h

  4. モジュールシステム（C++20 Modules）
     #include の代わりにモジュールを使用。
     ヘッダの重複解析を排除。

  5. 分散ビルド
     複数マシンでコンパイルを分散実行。
     distcc, sccache, icecc

  6. キャッシュ
     同一入力に対するコンパイル結果をキャッシュ。
     ccache（C/C++）, sccache（Rust）

  7. 開発時の最適化レベル低下
     開発中: -O0 または -O1（コンパイル高速）
     リリース: -O2 または -O3（実行高速）
```

```bash
# Rustのコンパイル速度改善の実例

# sccache の導入
cargo install sccache
export RUSTC_WRAPPER=sccache

# cranelift バックエンド（デバッグビルドを高速化）
# .cargo/config.toml
# [unstable]
# codegen-backend = true
# [profile.dev]
# codegen-backend = "cranelift"

# 依存クレートの事前ビルド
cargo build  # 初回は全依存をビルド（遅い）
cargo build  # 2回目は差分のみ（速い）

# mold リンカーの使用（リンク時間を大幅短縮）
# .cargo/config.toml
# [target.x86_64-unknown-linux-gnu]
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

---

## 2. インタプリタ型言語

### インタプリタの動作

```
ソースコード（.py, .rb）
    ↓
┌─────────────────────────────┐
│ 1. 字句解析 + 構文解析       │  AST を構築
├─────────────────────────────┤
│ 2. 逐次実行                  │  AST を辿りながら実行
│    または                    │
│    バイトコード変換 → VM実行  │  （CPython, Ruby MRI）
└─────────────────────────────┘
    ↓
  実行結果（即座に）
```

### インタプリタの種類

```
1. ツリーウォーキングインタプリタ
   ASTを直接辿って実行する。最もシンプルだが最も遅い。
   例: 初期の Ruby、Bash、多くの教育用インタプリタ

   動作: AST のノードを再帰的に訪問し、各ノードに対応する
         操作を実行する。

2. バイトコードインタプリタ
   ソースコードをバイトコード（仮想マシン命令）にコンパイルし、
   仮想マシン（VM）上で実行する。

   例: CPython, Ruby YARV, Lua, Erlang BEAM

   動作: バイトコードは「仮想CPU」の命令セット。
         実機のCPUより抽象度が高く、ポータブル。

3. レジスタベース vs スタックベース
   スタックベース: JVM, CPython, .NET CLR
     PUSH 3        ; スタック: [3]
     PUSH 4        ; スタック: [3, 4]
     ADD           ; スタック: [7]

   レジスタベース: Lua VM, Dalvik (Android)
     LOAD  R0, 3   ; R0 = 3
     LOAD  R1, 4   ; R1 = 4
     ADD   R2, R0, R1  ; R2 = 7
```

### CPython の実行モデル

```python
# Python のコードは内部的にバイトコードにコンパイルされる
import dis

def add(a, b):
    return a + b

dis.dis(add)
# 出力:
#   LOAD_FAST   0 (a)
#   LOAD_FAST   1 (b)
#   BINARY_ADD
#   RETURN_VALUE

# .pyc ファイル = コンパイル済みバイトコード
# __pycache__/ に自動キャッシュされる
```

```python
# より複雑な関数のバイトコードを確認
import dis

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

dis.dis(fibonacci)
# 出力例:
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

# バイトコードを直接操作する（上級テクニック）
import types

code = fibonacci.__code__
print(f"定数: {code.co_consts}")
print(f"変数名: {code.co_varnames}")
print(f"スタック深度: {code.co_stacksize}")
print(f"バイトコード: {code.co_code.hex()}")
```

### CPythonのGIL（Global Interpreter Lock）問題

```python
# CPythonのGILとは:
# - CPythonインタプリタ全体を保護するロック
# - 一度に1つのスレッドしかPythonバイトコードを実行できない
# - CPU バウンドな処理ではマルチスレッドの恩恵が受けられない

import threading
import time

# CPUバウンドタスク: GILによりマルチスレッドが効かない
def cpu_bound_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# シングルスレッド
start = time.time()
cpu_bound_task(10_000_000)
cpu_bound_task(10_000_000)
single_time = time.time() - start

# マルチスレッド（GILのせいで速くならない）
start = time.time()
t1 = threading.Thread(target=cpu_bound_task, args=(10_000_000,))
t2 = threading.Thread(target=cpu_bound_task, args=(10_000_000,))
t1.start(); t2.start()
t1.join(); t2.join()
multi_time = time.time() - start

print(f"シングルスレッド: {single_time:.2f}s")
print(f"マルチスレッド:   {multi_time:.2f}s")  # ほぼ同じか遅い

# 解決策1: multiprocessing（プロセス分離）
from multiprocessing import Pool

start = time.time()
with Pool(2) as p:
    results = p.map(cpu_bound_task, [10_000_000, 10_000_000])
process_time = time.time() - start
print(f"マルチプロセス:   {process_time:.2f}s")  # 約2倍速

# 解決策2: I/Oバウンドタスクなら asyncio
import asyncio
import aiohttp

async def fetch_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return responses

# 解決策3: Python 3.13+ の free-threaded mode (PEP 703)
# GILを無効化してマルチスレッドを活用可能にする実験的機能
# python3.13t script.py  # GILなしモード
```

### Ruby の実行モデル（YARV）

```ruby
# Ruby のバイトコードを確認する
code = RubyVM::InstructionSequence.compile("puts 1 + 2")
puts code.disasm
# 出力例:
# == disasm: #<ISeq:<compiled>@<compiled>:1 (1,0)-(1,12)>==========
# 0000 putself                                                          (   1)[Li]
# 0001 putobject_INT2FIX_1_
# 0002 putobject                    2
# 0004 opt_plus                     <calldata!mid:+, argc:1, ARGS_SIMPLE>[CcCr]
# 0006 opt_send_without_block       <calldata!mid:puts, argc:1, FCALL|ARGS_SIMPLE>
# 0008 leave

# Ruby 3.x の YJIT（Yet Another JIT）
# JIT コンパイラが組み込まれている
# ruby --yjit script.rb  # YJIT有効化
# Ruby 3.3以降はデフォルトで有効
```

### Lua の実行モデル

```lua
-- Lua: 非常に軽量なバイトコードインタプリタ
-- レジスタベースVM（スタックベースより高効率）

-- LuaJIT: トレーシングJITによる最高レベルの性能
-- 動的言語としては異例の実行速度を達成

-- Luaが組み込み用途で人気の理由:
-- 1. インタプリタが非常に小さい（〜200KB）
-- 2. C APIが優秀（ホスト言語との統合が容易）
-- 3. メモリ消費が少ない
-- 4. LuaJITの性能がCに迫る

-- ゲームエンジン、Webサーバー（OpenResty）、
-- ネットワーク機器、RedisのスクリプトなどでLuaは広く使われる
```

### インタプリタの利点と欠点

```
利点:
  ✓ 即座に実行可能（REPL、対話的開発）
  ✓ プラットフォーム非依存（インタプリタがあれば動く）
  ✓ 動的な言語機能（eval, メタプログラミング）
  ✓ 開発サイクルが速い
  ✓ デバッグしやすい（ソースレベルのエラー情報）
  ✓ ホットリロード（コード変更を即反映）

欠点:
  ✗ 実行速度が遅い（10〜100倍の差）
  ✗ ランタイムエラーが実行時まで分からない
  ✗ 実行環境にインタプリタが必要
  ✗ メモリ消費が多い（ASTやバイトコードを保持）
  ✗ 配布が煩雑（依存関係の管理）

代表的な言語:
  Python(CPython), Ruby(MRI), PHP, Perl, Lua
```

### REPL（Read-Eval-Print Loop）の活用

```python
# Pythonの対話型環境は探索的プログラミングに最適

# 標準REPL
$ python3
>>> import json
>>> data = {"name": "Alice", "age": 30}
>>> json.dumps(data, indent=2)
'{\n  "name": "Alice",\n  "age": 30\n}'

# IPython: 強化版REPL
$ ipython
In [1]: import pandas as pd
In [2]: df = pd.read_csv("sales.csv")
In [3]: df.describe()  # データの統計情報を即座に確認
In [4]: %timeit df.sort_values("amount")  # ベンチマーク
In [5]: %debug  # 直前の例外をデバッグ

# Jupyter Notebook: ブラウザベースの対話環境
# データ分析・可視化・ドキュメンテーションを統合
```

```javascript
// Node.jsのREPL
$ node
> const arr = [1, 2, 3, 4, 5]
> arr.filter(x => x % 2 === 0).map(x => x * x)
[ 4, 16 ]
> .help  // ヘルプ表示
> .exit  // 終了
```

---

## 3. JIT コンパイル

### JIT の仕組み

```
ソースコード
    ↓
バイトコード（事前コンパイル）
    ↓
┌─────────────────────────────────────────┐
│ JIT コンパイラ（実行中に動作）            │
│                                          │
│ 1. プロファイリング                       │
│    → どのコードが頻繁に実行されるか監視    │
│                                          │
│ 2. ホットスポット検出                     │
│    → 頻繁に実行されるコードを特定          │
│    （ループ、頻呼び出し関数）             │
│                                          │
│ 3. 最適化コンパイル                       │
│    → ホットコードだけを機械語にコンパイル   │
│    → 実行時の型情報を活用した最適化        │
│                                          │
│ 4. 脱最適化（Deoptimization）             │
│    → 前提が崩れたら再インタプリタ          │
└─────────────────────────────────────────┘
    ↓
  実行（ウォームアップ後はネイティブに近い速度）
```

### JIT コンパイラの最適化技法

```
JIT 固有の最適化（AOTでは不可能なもの）:

  1. 投機的最適化（Speculative Optimization）
     実行時のプロファイル情報に基づいて最適化。
     例: 「この関数の引数は常にint型」→ 型チェックを省略

  2. 型特殊化（Type Specialization）
     動的型の変数が実際には特定の型しか持たない場合、
     その型専用のコードを生成。

  3. インラインキャッシュ（Inline Cache）
     メソッド呼び出しの解決結果をキャッシュ。
     同じ型のオブジェクトに対する呼び出しを高速化。

  4. 脱仮想化（Devirtualization）
     仮想メソッド呼び出しを直接呼び出しに変換。
     実行時にサブクラスが1つしかないことが分かった場合。

  5. オンスタックリプレースメント（OSR）
     実行中のループをインタプリタからJITコードに切り替え。
     長時間ループの途中から最適化の恩恵を受けられる。
```

```javascript
// V8（JavaScript）のJIT最適化の例

// 型が安定している関数 → 最適化されやすい
function add(a, b) {
    return a + b;
}

// 常にnumber型で呼ばれる → 高速な機械語に変換
for (let i = 0; i < 1000000; i++) {
    add(i, i + 1);  // → 整数加算の機械語に最適化
}

// 途中で型が変わると脱最適化（deoptimization）が発生
add("hello", " world");  // → string型！前提が崩れる
// JITコンパイラは最適化されたコードを破棄し、再インタプリタに戻る

// === V8が最適化しやすいコードの書き方 ===

// Good: 型が安定している
function calculateTotal(items) {
    let total = 0;                    // 常にnumber
    for (let i = 0; i < items.length; i++) {
        total += items[i].price;      // 常にnumber
    }
    return total;
}

// Bad: 型が不安定（Hidden Class が変化する）
function createUser(name, age) {
    const user = {};
    user.name = name;   // Hidden Class 変化
    user.age = age;     // Hidden Class 変化
    if (age > 18) {
        user.adult = true;  // 条件付きプロパティ → Hidden Class 分岐
    }
    return user;
}

// Good: プロパティを最初から全て定義
function createUser(name, age) {
    return {
        name: name,
        age: age,
        adult: age > 18,   // 常に存在 → Hidden Class が安定
    };
}
```

### V8 エンジン（JavaScript）のパイプライン

```
JavaScript ソースコード
    ↓
  Parser → AST
    ↓
  Ignition（インタプリタ）→ バイトコード実行
    ↓  （ホットコード検出）
  TurboFan（最適化コンパイラ）→ 機械語
    ↓  （前提が崩れた場合）
  Deoptimize → Ignition に戻る

性能の推移:
  起動直後:  インタプリタで低速
  数秒後:    JIT により高速化
  安定後:    ネイティブコードに近い性能

V8の進化:
  2008: Full-Codegen + Crankshaft
  2017: Ignition + TurboFan（現在のアーキテクチャ）
  2023: Maglev（中間層の追加: Ignition→Maglev→TurboFan）
```

### JVM（Java Virtual Machine）の階層的コンパイル

```
Java ソースコード
    ↓
  javac → バイトコード（.class）
    ↓
┌─────────────────────────────────────┐
│ JVM の実行層                         │
│                                      │
│ Level 0: インタプリタ                │
│ Level 1: C1 コンパイラ（簡易最適化） │
│ Level 2: C1 + プロファイリング       │
│ Level 3: C1 + フルプロファイリング   │
│ Level 4: C2 コンパイラ（最大最適化） │
│                                      │
│ 実行回数に応じて段階的に最適化が進む │
└─────────────────────────────────────┘

GraalVM: 多言語対応JIT
  → Java, JS, Python, Ruby, R を同一VM上で実行可能
  → AOT コンパイル（native-image）も可能
```

```java
// JVMのJIT最適化を確認するフラグ

// JITコンパイルのログを表示
// java -XX:+PrintCompilation MyApp

// 出力例:
//   42   1       java.lang.String::hashCode (55 bytes)
//   43   2       java.util.HashMap::hash (20 bytes)
//   44   3 %     MyApp::hotLoop @ 5 (30 bytes)
//
// %: OSR（On-Stack Replacement）コンパイル
// 数字: コンパイルレベル

// インライン展開の閾値を変更
// java -XX:InlineSmallCode=2000 MyApp

// エスケープ解析の有効化（デフォルトで有効）
// java -XX:+DoEscapeAnalysis MyApp

// GraalVM の Native Image（AOTコンパイル）
// native-image -jar myapp.jar
// → 起動時間: JVM 数百ms → Native Image 数十ms
// → メモリ: JVM 数百MB → Native Image 数十MB
```

### PyPy: Python の JIT 実装

```python
# PyPy は CPython より 10-100 倍高速な場合がある

# CPython で遅いコードが PyPy で高速化される例
def matrix_multiply(a, b, n):
    """n×n 行列の乗算（純粋Python実装）"""
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

# CPython: 約10秒（n=500）
# PyPy:    約0.3秒（n=500）
# NumPy:   約0.01秒（n=500）← C拡張なのでどちらでも高速

# PyPyが高速な理由:
# 1. トレーシングJIT: ホットループを検出して機械語に変換
# 2. 型特殊化: ループ内の変数型を固定してチェックを省略
# 3. ボクシング除去: int/float をヒープ確保せずレジスタで扱う
# 4. ループ最適化: ガードの巻き上げ、ベクトル化

# PyPyの制限:
# - C拡張との互換性問題（CPython API に依存するライブラリ）
# - 起動時間が CPython より遅い（JITのウォームアップ）
# - メモリ消費が CPython より多い場合がある
```

### .NET の JIT 実行モデル

```csharp
// .NET の実行パイプライン
// C# ソースコード → Roslyn コンパイラ → CIL (Common Intermediate Language)
// → RyuJIT (JIT コンパイラ) → ネイティブ機械語

// CILの例（IL DASM で確認可能）
// .method public static int32 Add(int32 a, int32 b) cil managed
// {
//     .maxstack 2
//     ldarg.0      // a をスタックにプッシュ
//     ldarg.1      // b をスタックにプッシュ
//     add          // 加算
//     ret          // 結果を返す
// }

// .NET 8 の AOT コンパイル
// dotnet publish -r linux-x64 -c Release /p:PublishAot=true
// → ネイティブバイナリを生成（JVMのGraalVM native-imageに相当）

// .NET の階層的コンパイル
// Tier 0: 最小限の最適化（高速起動）
// Tier 1: フルJIT最適化（ホットメソッド）
// R2R (Ready to Run): AOT + JIT のハイブリッド
```

---

## 4. 現代の実行モデル

### WebAssembly（Wasm）

```
C/Rust/Go/etc.
    ↓  コンパイル
  .wasm（バイナリ形式）
    ↓  ブラウザ / ランタイム
  ストリーミングコンパイル → 実行

特徴:
  - ほぼネイティブの実行速度
  - ブラウザで安全に実行（サンドボックス）
  - 言語非依存（どの言語からもコンパイル可能）
  - WASI: ブラウザ外（サーバー等）でも実行可能
```

#### Wasm の詳細な仕組み

```
WebAssembly のバイナリフォーマット:

  マジックナンバー: 0x00 0x61 0x73 0x6D ("\0asm")
  バージョン:       0x01 0x00 0x00 0x00 (version 1)

  セクション構成:
    Type Section     - 関数シグネチャの定義
    Import Section   - ホスト環境からのインポート
    Function Section - 関数のインデックス
    Table Section    - 間接呼び出し用テーブル
    Memory Section   - 線形メモリの定義
    Global Section   - グローバル変数
    Export Section   - ホスト環境へのエクスポート
    Start Section    - エントリポイント
    Element Section  - テーブル初期化
    Code Section     - 関数本体のバイトコード
    Data Section     - メモリ初期化データ
```

```rust
// Rust → Wasm のコンパイル例

// lib.rs
#[no_mangle]
pub extern "C" fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        return n;
    }
    fibonacci(n - 1) + fibonacci(n - 2)
}

// ビルドコマンド
// cargo build --target wasm32-unknown-unknown --release

// wasm-bindgen を使った JavaScript との連携
// use wasm_bindgen::prelude::*;
//
// #[wasm_bindgen]
// pub fn greet(name: &str) -> String {
//     format!("Hello, {}!", name)
// }
```

```javascript
// JavaScript から Wasm を呼び出す

// 方法1: fetch + instantiate
async function loadWasm() {
    const response = await fetch('fibonacci.wasm');
    const bytes = await response.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(bytes);

    const result = instance.exports.fibonacci(40);
    console.log(`fib(40) = ${result}`);
}

// 方法2: ストリーミングコンパイル（推奨）
async function loadWasmStreaming() {
    const { instance } = await WebAssembly.instantiateStreaming(
        fetch('fibonacci.wasm')
    );
    return instance.exports;
}

// Wasm の線形メモリを使ったデータ共有
async function processArray() {
    const { instance } = await WebAssembly.instantiateStreaming(
        fetch('processor.wasm')
    );

    const memory = instance.exports.memory;
    const buffer = new Float64Array(memory.buffer, 0, 1000);

    // JavaScript側でデータを書き込み
    for (let i = 0; i < 1000; i++) {
        buffer[i] = Math.random();
    }

    // Wasm側で高速に処理
    const result = instance.exports.process_array(1000);
    console.log(`Result: ${result}`);
}
```

#### WASI（WebAssembly System Interface）

```
WASI: Wasmをブラウザ外で実行するための標準インターフェース

  設計原則:
  - Capability-based Security（権限ベースのセキュリティ）
  - POSIX風のAPI（ファイルI/O、ネットワークなど）
  - サンドボックス化されたファイルシステムアクセス

  ランタイム:
  - Wasmtime（Mozilla/Bytecode Alliance）
  - Wasmer
  - WasmEdge
  - wazero（Go実装）

  ユースケース:
  - サーバーレス関数（Cloudflare Workers, Fastly Compute@Edge）
  - プラグインシステム（Envoy Proxy, Istio）
  - ユニバーサルバイナリ（一度コンパイル、どこでも実行）
  - エッジコンピューティング
```

```bash
# WASIを使ったコマンドラインツールの例

# Rustで書いたプログラムをWASI向けにコンパイル
cargo build --target wasm32-wasi --release

# Wasmtime で実行
wasmtime target/wasm32-wasi/release/myapp.wasm

# ファイルアクセスの許可（サンドボックス）
wasmtime --dir=/tmp target/wasm32-wasi/release/myapp.wasm

# ネットワークアクセスの許可
wasmtime --tcplisten=127.0.0.1:8080 target/wasm32-wasi/release/server.wasm
```

### トランスパイル

```
TypeScript → JavaScript
Kotlin → JVM バイトコード / JavaScript
Elm → JavaScript
Sass → CSS
JSX → JavaScript

利点:
  - 元の言語の表現力 + ターゲットのエコシステム
  - 漸進的な移行が可能（TSをJSプロジェクトに段階的に導入）
```

#### トランスパイラの詳細な仕組み

```typescript
// TypeScript → JavaScript のトランスパイル例

// TypeScript ソースコード
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

// トランスパイル後の JavaScript（ES2020ターゲット）
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

// 注目点:
// - interfaceは完全に消える（型情報は実行時には不要）
// - 関数の型注釈も消える
// - ロジックはそのまま保持される
// - TypeScriptの価値は「コンパイル時の型チェック」にある
```

```
トランスパイラのターゲット設定:

  TypeScript のコンパイルターゲット:
    ES5:    IE11対応（非推奨）。class → function、アロー関数を展開
    ES2015: class, arrow function, let/const をそのまま出力
    ES2020: optional chaining (?.), nullish coalescing (??) に対応
    ES2022: top-level await, class fields に対応
    ESNext: 最新仕様をそのまま出力

  tsconfig.json の設定例:
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

  Babelのプリセット:
    @babel/preset-env: ブラウザ互換性に基づいて自動変換
    @babel/preset-react: JSX → React.createElement 変換
    @babel/preset-typescript: TypeScript → JavaScript
```

### ハイブリッドモデル

現代の言語は複数の実行モデルを組み合わせることが多い。

```
ハイブリッド実行モデルの例:

  Java / Kotlin:
    AOTコンパイル(javac) → バイトコード → JIT(HotSpot)
    さらに GraalVM native-image で AOT も可能

  C# / F#:
    AOTコンパイル(Roslyn) → CIL → JIT(RyuJIT)
    さらに .NET Native AOT でネイティブバイナリも可能

  Python:
    CPython: ソース → バイトコード → インタプリタ
    PyPy:    ソース → バイトコード → トレーシングJIT
    Cython:  Python風構文 → C → AOT
    Mypyc:   型付きPython → C拡張 → AOT
    Nuitka:  Python → C → AOT

  JavaScript:
    V8:      ソース → バイトコード(Ignition) → JIT(TurboFan)
    Bun:     ソース → JavaScriptCore(WebKit) の JIT
    Deno:    ソース → V8 の JIT
    Static Hermes: ソース → AOTバイトコード（React Native向け）

  Dart / Flutter:
    開発時:   JIT（ホットリロード対応）
    本番時:   AOT（高速起動、予測可能な性能）
```

---

## 5. パフォーマンス比較

```
ベンチマーク参考値（フィボナッチ再帰 n=40）:

  言語             実行時間    方式
  ─────────────────────────────────
  C (gcc -O2)      0.15s      AOT
  Rust (release)   0.16s      AOT
  Go               0.45s      AOT
  Java             0.55s      JIT
  JavaScript(V8)   0.80s      JIT
  C# (.NET)        0.60s      JIT
  PyPy             1.20s      JIT
  CPython          15.0s      インタプリタ
  Ruby (MRI)       12.0s      インタプリタ

  ※ 実際のアプリケーション性能はI/O・アルゴリズム・
    最適化で大きく変わる。マイクロベンチマークは参考程度に。
```

### より実践的なベンチマーク

```
Webサーバーのスループット比較（Hello World, wrk ベンチマーク）:

  フレームワーク              req/sec (概算)    言語
  ────────────────────────────────────────────────
  actix-web                   500,000+          Rust
  Gin                         200,000+          Go
  Fastify                     70,000+           Node.js(JS)
  Spring Boot (Webflux)       60,000+           Java
  ASP.NET Core                100,000+          C#
  Express                     15,000+           Node.js(JS)
  FastAPI                     10,000+           Python
  Flask                       2,000+            Python
  Rails                       3,000+            Ruby

  ※ ベンチマークは設定・ハードウェア・ワークロードにより大きく変動
  ※ 実際のアプリケーションではDB/外部API呼び出しがボトルネックになることが多い
```

```
メモリ使用量の比較（Hello World Webサーバー起動時）:

  言語/ランタイム         メモリ使用量 (概算)
  ────────────────────────────────────────
  Rust (actix-web)       1-3 MB
  Go (net/http)          5-10 MB
  Node.js (express)      30-50 MB
  Java (Spring Boot)     100-200 MB
  Python (Flask)         20-40 MB
  .NET (ASP.NET Core)    30-60 MB

  ※ JVMは初期ヒープサイズの設定で大きく変わる
  ※ コンテナ環境ではメモリ制限が重要
```

### 起動時間の比較

```
起動時間の比較（CLI ツールの場合）:

  言語/ランタイム         起動時間 (概算)
  ────────────────────────────────────────
  C / Rust / Go          1-10 ms
  .NET Native AOT        10-30 ms
  GraalVM Native Image   10-30 ms
  Node.js                30-100 ms
  Python                 30-50 ms
  JVM (Java)             100-500 ms
  JVM (Spring Boot)      1-5 sec

  CLIツールやサーバーレス関数では起動時間が重要:
  - AWS Lambda: コールドスタートのレイテンシ
  - Docker: コンテナの起動速度
  - CLIツール: ユーザー体験（即座に結果が欲しい）
```

### パフォーマンスチューニングのアプローチ

```
パフォーマンス最適化の優先順位:

  1. アルゴリズムの改善（最大の効果）
     O(n²) → O(n log n)  例: バブルソート→マージソート
     効果: 100倍-10000倍の改善が可能

  2. データ構造の選択
     配列 vs 連結リスト vs ハッシュテーブル vs B-Tree
     キャッシュ効率を考慮（メモリ局所性）

  3. I/O最適化
     非同期I/O、バッチ処理、接続プール
     N+1問題の解消（DB）

  4. 並列化/並行化
     マルチスレッド、async/await、ワーカープール
     アムダールの法則に注意

  5. 言語/ランタイムレベルの最適化
     コンパイラフラグ、GCチューニング、メモリプール
     これが必要になるのは上位4つを最適化した後

  6. 言語の変更（最後の手段）
     ホットスポットだけを高速言語で書き直す
     例: PythonのCPU bound部分をRust/C拡張に置き換え
```

---

## 6. 言語処理系の比較と選定

### 同一言語の複数実装

```
Python の実装:
  CPython   — 標準実装。C言語で書かれたインタプリタ
  PyPy      — JIT付きインタプリタ。CPythonの10-100倍速い場合も
  Jython    — JVM上で動くPython
  IronPython — .NET上で動くPython
  MicroPython — マイクロコントローラ向けの軽量実装
  Cython    — Pythonの構文をCに変換するコンパイラ
  Mypyc     — 型付きPythonをC拡張に変換

JavaScript の実装:
  V8        — Chrome, Node.js, Deno で使用（Google）
  SpiderMonkey — Firefox で使用（Mozilla）
  JavaScriptCore — Safari, Bun で使用（Apple）
  Hermes    — React Native 向け（Meta）
  QuickJS   — 軽量な組み込み用JavaScript

Ruby の実装:
  CRuby/MRI — 標準実装（Matz's Ruby Interpreter）
  JRuby     — JVM上で動くRuby
  TruffleRuby — GraalVM上で動くRuby（高速）
  mruby     — 組み込み向けの軽量Ruby

Scheme の実装:
  Racket, Guile, Chez Scheme, Gambit, Chicken
  → 同一言語仕様でも実装により性能・機能が大きく異なる
```

### コンパイラ基盤

```
LLVM:
  使用プロジェクト: Clang(C/C++), Rust, Swift, Julia, Zig,
                    Kotlin/Native, Crystal, Mojo
  特徴: モジュラー設計、豊富な最適化パス、多数のバックエンド
        コンパイラを作る際のデファクトスタンダード

GCC:
  使用プロジェクト: C, C++, Fortran, Ada, Go(gccgo)
  特徴: 最も長い歴史、多数のプラットフォームサポート
        LLVM登場前はデファクトスタンダード

Cranelift:
  使用プロジェクト: Wasmtime, Rustc（デバッグビルド実験的）
  特徴: 高速なコード生成、JIT向け設計
        LLVMより最適化は弱いがコンパイルが速い

GraalVM Compiler:
  使用プロジェクト: Java, JavaScript, Python, Ruby, R
  特徴: 部分評価ベースのJIT、多言語相互運用
        Truffle フレームワークで新しい言語の追加が容易
```

---

## 実践演習

### 演習1: [基礎] — バイトコードの確認

Pythonの `dis` モジュールで簡単な関数のバイトコードを確認する。

```python
# 演習: 以下の関数のバイトコードを dis.dis() で確認せよ
import dis

# 関数1: 単純な条件分岐
def max_value(a, b):
    if a > b:
        return a
    return b

# 関数2: リスト内包表記
def square_evens(numbers):
    return [n * n for n in numbers if n % 2 == 0]

# 関数3: ジェネレータ
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# それぞれの関数のバイトコードを確認
print("=== max_value ===")
dis.dis(max_value)
print("\n=== square_evens ===")
dis.dis(square_evens)
print("\n=== fibonacci_gen ===")
dis.dis(fibonacci_gen)

# 考察ポイント:
# - COMPARE_OP, POP_JUMP_IF_FALSE の動作
# - リスト内包表記の内部実装（別関数として生成される）
# - yield のバイトコード表現
```

### 演習2: [応用] — JIT の効果を計測

同じアルゴリズムを CPython と PyPy で実行し、JIT の効果を比較する。

```python
# benchmark.py: CPythonとPyPyで実行して比較せよ
import time

def benchmark_loop(n):
    """純粋なCPU計算（JITの恩恵を最も受けやすい）"""
    total = 0
    for i in range(n):
        if i % 2 == 0:
            total += i * i
        else:
            total -= i
    return total

def benchmark_string(n):
    """文字列操作（メモリ確保が頻繁）"""
    result = ""
    for i in range(n):
        result += str(i)
    return len(result)

def benchmark_dict(n):
    """辞書操作（ハッシュテーブル）"""
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

# 実行方法:
# python3 benchmark.py     # CPython
# pypy3 benchmark.py       # PyPy
```

### 演習3: [発展] — 簡易インタプリタの実装

四則演算を評価する簡易インタプリタを Python で実装する。

```python
# 演習: 以下のインタプリタを拡張せよ
# 1. 変数の代入と参照を追加（let x = 10; x + 5）
# 2. 比較演算子を追加（==, !=, <, >）
# 3. if-else 文を追加
# 4. 関数定義と呼び出しを追加

from dataclasses import dataclass
from typing import Union

# === AST ノード ===
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

# === 字句解析器 ===
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

# === 構文解析器（再帰下降） ===
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

# === 評価器 ===
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

### 演習4: [発展] — Wasm の体験

```bash
# 演習: RustのコードをWasmにコンパイルしてブラウザで実行する

# 1. ツールのインストール
# rustup target add wasm32-unknown-unknown
# cargo install wasm-pack

# 2. プロジェクト作成
# cargo new --lib wasm-demo
# cd wasm-demo

# 3. Cargo.toml に追加
# [lib]
# crate-type = ["cdylib"]
# [dependencies]
# wasm-bindgen = "0.2"

# 4. src/lib.rs を編集
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

# 5. ビルド
# wasm-pack build --target web

# 6. HTMLから呼び出す（index.html）
# <script type="module">
#     import init, { fibonacci } from './pkg/wasm_demo.js';
#     await init();
#     console.log(fibonacci(50));
# </script>
```

---

## FAQ

### Q1: 「コンパイル型 = 速い」は正しい？
A: 概ね正しいが例外もある。JIT は実行時情報を使って AOT より良い最適化が可能な場合がある。実用上はアルゴリズムの選択の方がはるかに重要。

### Q2: TypeScript はコンパイル型？
A: TypeScript は JavaScript にトランスパイルされる。型チェックはコンパイル時だが、実行時は JavaScript エンジン（JIT）が動く。分類としてはトランスパイル型。

### Q3: Go が速い理由は？
A: 静的型付け + AOT コンパイル + ガベージコレクタの最適化 + goroutine による効率的な並行処理。シンプルな言語仕様がコンパイラの最適化を容易にしている。

### Q4: JITのウォームアップ問題にどう対処する？
A: いくつかのアプローチがある:
- **AOTコンパイル**: GraalVM native-image、.NET Native AOT でネイティブバイナリを生成
- **プロファイルガイド最適化（PGO）**: 事前に収集したプロファイル情報で最適化
- **階層的コンパイル**: 低い最適化レベルから段階的に最適化を上げる（JVM）
- **Ahead-of-time JIT（AOT JIT）**: 過去の実行プロファイルを保存して再利用
- **サーバー設計**: ウォームアップ期間中はトラフィックを徐々に増やす

### Q5: WebAssemblyはJavaScriptを置き換える？
A: 置き換えるのではなく補完する関係にある。WasmはCPU集約型タスク（画像処理、暗号化、ゲームエンジン）に適し、JavaScriptはDOM操作やUI制御に適する。実際のアプリケーションでは両者を組み合わせて使う。

### Q6: なぜRustのコンパイルは遅いのか？
A: 主な原因は以下の通り:
- **所有権チェック（borrow checker）**: メモリ安全性の検証に計算コスト
- **モノモーフィゼーション**: ジェネリクスの実体化で大量のコードを生成
- **LLVM最適化パス**: 強力だが時間がかかる
- **依存クレートの再コンパイル**: 依存が多いとビルド時間が増大
- 対策: sccache、cranelift バックエンド、mold リンカーの使用

### Q7: AOTとJITのハイブリッドが最適解？
A: 多くの場合そうだ。Dart/Flutterは開発時にJIT（ホットリロード）、本番でAOT（高速起動）を使い分ける。GraalVMのProfile-Guided Optimization (PGO) もAOTとJITの知見を組み合わせたアプローチである。

---

## まとめ

| 方式 | 代表言語 | 実行速度 | 開発速度 | ポータビリティ |
|------|---------|---------|---------|-------------|
| AOT コンパイル | C, Rust, Go | 最速 | 遅め | 低（再コンパイル必要） |
| インタプリタ | Python, Ruby | 遅い | 最速 | 高（インタプリタがあれば） |
| JIT | Java, JS | 高速 | 中程度 | 高（VM上で実行） |
| トランスパイル | TS, Kotlin | ターゲット依存 | 中程度 | ターゲット依存 |
| Wasm | C, Rust→Wasm | ほぼネイティブ | 中程度 | 非常に高い |

| 最適化段階 | 内容 | 効果 |
|-----------|------|------|
| 字句解析 | トークン分割 | 構文解析の前処理 |
| 構文解析 | AST構築 | プログラムの構造化 |
| 意味解析 | 型チェック・スコープ解決 | 不正なプログラムの検出 |
| IR生成 | 中間表現への変換 | ターゲット非依存な最適化 |
| 最適化 | 定数畳み込み・インライン展開・ベクトル化 | 実行速度の向上 |
| コード生成 | 機械語生成 | 実行可能バイナリの生成 |
| リンク | ライブラリ結合 | 完全な実行ファイルの生成 |

---

## 次に読むべきガイド
→ [[02-paradigms-overview.md]] — プログラミングパラダイム

---

## 参考文献
1. Aho, A., Lam, M., Sethi, R. & Ullman, J. "Compilers: Principles, Techniques, and Tools." 2nd Ed, 2006.
2. Nystrom, R. "Crafting Interpreters." 2021.
3. Cooper, K. & Torczon, L. "Engineering a Compiler." 3rd Ed, 2022.
4. Appel, A. "Modern Compiler Implementation in ML." Cambridge University Press, 2004.
5. Aycock, J. "A Brief History of Just-In-Time." ACM Computing Surveys, 2003.
6. Haas, A. et al. "Bringing the Web up to Speed with WebAssembly." PLDI, 2017.
7. Bolz, C. et al. "Tracing the Meta-level: PyPy's Tracing JIT Compiler." ICOOOLPS, 2009.
8. Lattner, C. & Adve, V. "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation." CGO, 2004.
9. Würthinger, T. et al. "Practical Partial Evaluation for High-Performance Dynamic Language Runtimes." PLDI, 2017.
10. Leroy, X. "The Compcert Verified Compiler." 2009.
