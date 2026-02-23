# プログラミング言語とは何か

> プログラミング言語は「人間の意図をコンピュータに伝える」ための形式言語であり、同時に「人間が思考を構造化する」ための道具でもある。

## この章で学ぶこと

- [ ] プログラミング言語の本質的な役割を理解する
- [ ] 言語の抽象化レベルと設計思想の違いを把握する
- [ ] 言語を構成する要素（構文・意味論・プラグマティクス）を理解する
- [ ] 形式言語理論とプログラミング言語の関係を把握する
- [ ] 言語の歴史的変遷を通じて設計思想の進化を理解する
- [ ] 言語処理系の内部構造を俯瞰する
- [ ] 言語設計のトレードオフを実例を通じて体感する

---

## 1. なぜプログラミング言語が必要か

### コンピュータが理解するのは機械語

```
コンピュータの実行レベル:

  人間の意図:    「Webサーバーを起動して、ポート8080で待ち受けて」
       ↓
  高水準言語:    server.listen(8080)
       ↓
  中間表現:      バイトコード / IR
       ↓
  アセンブリ:    mov eax, 0x1F90 / syscall
       ↓
  機械語:        10111000 10010000 00011111 00000000 00000000
       ↓
  ハードウェア:  電気信号 (High/Low)
```

プログラミング言語は、この**抽象化の階段**を提供する。高水準になるほど人間にとって理解しやすく、低水準になるほどハードウェアに近い制御ができる。

### 抽象化のトレードオフ

```
高水準    Python, Ruby, JavaScript
  ↑      生産性は高いが、実行効率の制御は限定的
  │
  │      Java, C#, Go
  │      バランス型。十分な抽象化と合理的な性能
  │
  │      C, C++, Rust
  │      低レベル制御が可能だが、習得コストが高い
  ↓
低水準    アセンブリ言語
          ハードウェア直接制御。最高の効率だが人間には読みにくい
```

### 抽象化の各レベルの詳細

#### レベル1: 機械語（第1世代言語）

機械語はCPUが直接解釈できる命令の集合である。各命令はオペコード（操作種別）とオペランド（対象データ）で構成される。

```
x86-64 機械語の例（Intel記法）:

  48 89 C7    → mov rdi, rax   （raxの値をrdiにコピー）
  48 83 C0 01 → add rax, 1     （raxに1を加算）
  0F 84 XX XX → jz  label      （ゼロフラグが立っていればジャンプ）
  C3          → ret            （サブルーチンから復帰）

ARM64 機械語の例:
  D2800020    → mov x0, #1     （即値1をx0にセット）
  8B010000    → add x0, x0, x1 （x0 + x1の結果をx0に格納）
  D65F03C0    → ret            （サブルーチンから復帰）
```

機械語は「人間が書くもの」ではなく「ツールが生成するもの」であるべきだ。しかし、組み込みシステムのデバッグやセキュリティ分析（バイナリ解析）では、機械語の理解が不可欠になる場面がある。

#### レベル2: アセンブリ言語（第2世代言語）

機械語を人間が読めるニーモニック（記号名）に置き換えたもの。1対1で機械語に対応する。

```asm
; x86-64 Linux: Hello Worldの出力
section .data
    msg db 'Hello, World!', 0x0a    ; 文字列データ + 改行
    len equ $ - msg                  ; 文字列長を計算

section .text
    global _start

_start:
    ; sys_write(fd=1, buf=msg, count=len)
    mov rax, 1          ; システムコール番号: write
    mov rdi, 1          ; ファイルディスクリプタ: stdout
    mov rsi, msg        ; バッファのアドレス
    mov rdx, len        ; 書き込むバイト数
    syscall             ; カーネル呼び出し

    ; sys_exit(status=0)
    mov rax, 60         ; システムコール番号: exit
    xor rdi, rdi        ; 終了コード: 0
    syscall
```

```asm
; ARM64 macOS: Hello Worldの出力
.global _start
.align 2

_start:
    mov x0, #1              ; stdout
    adrp x1, msg@PAGE       ; 文字列アドレス（ページ）
    add x1, x1, msg@PAGEOFF ; 文字列アドレス（オフセット）
    mov x2, #14             ; 文字列長
    mov x16, #4             ; write システムコール
    svc #0x80               ; カーネル呼び出し

    mov x0, #0              ; 終了コード
    mov x16, #1             ; exit システムコール
    svc #0x80

msg:
    .ascii "Hello, World!\n"
```

アセンブリ言語はCPUアーキテクチャごとに異なる。x86-64用に書いたコードはARM64では動かない。これが「高水準言語」が求められる根本的な理由の1つである。

#### レベル3: 高水準言語（第3世代言語以降）

```python
# Python: 同じ "Hello World" がわずか1行
print("Hello, World!")
```

```go
// Go: クロスコンパイルにより1つのソースから多数のプラットフォーム向けバイナリを生成
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

// ビルド例:
// GOOS=linux   GOARCH=amd64 go build -o hello-linux
// GOOS=darwin  GOARCH=arm64 go build -o hello-mac
// GOOS=windows GOARCH=amd64 go build -o hello.exe
```

高水準言語が提供する主要な抽象化:

| 抽象化 | 低水準での対応物 | 高水準での表現 |
|--------|-----------------|---------------|
| 変数 | メモリアドレス + レジスタ | `name = "Alice"` |
| 関数 | スタックフレーム + call/ret | `def greet():` |
| ループ | 条件ジャンプ + ラベル | `for x in items:` |
| 構造体 | メモリレイアウト + オフセット | `class User:` |
| エラー処理 | ステータスコード + 条件分岐 | `try/except` |
| メモリ管理 | malloc/free | ガベージコレクション |
| 並行処理 | スレッド + ロック + アトミック操作 | `async/await` |

---

## 2. 言語を構成する3つの側面

### 2.1 構文（Syntax）

言語の「文法」。何が正しいプログラムの形か。

```python
# Python: インデントでブロック構造
def greet(name):
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello, World!")
```

```rust
// Rust: 波括弧でブロック構造
fn greet(name: &str) {
    if !name.is_empty() {
        println!("Hello, {}!", name);
    } else {
        println!("Hello, World!");
    }
}
```

```lisp
;; Lisp: S式（全てが括弧）
(defun greet (name)
  (if name
      (format t "Hello, ~a!" name)
      (format t "Hello, World!")))
```

同じロジックでも、構文が異なると**思考のパターン**が変わる。

#### 構文の形式的定義: BNF記法

プログラミング言語の構文はBNF（バッカス・ナウア記法）やEBNF（拡張BNF）で厳密に定義される。

```
# 簡単な算術式の文法（BNF）
<expression>  ::= <term> (('+' | '-') <term>)*
<term>        ::= <factor> (('*' | '/') <factor>)*
<factor>      ::= <number> | '(' <expression> ')' | <identifier>
<number>      ::= <digit>+
<digit>       ::= '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
<identifier>  ::= <letter> (<letter> | <digit>)*
<letter>      ::= 'a' | 'b' | ... | 'z' | 'A' | 'B' | ... | 'Z'
```

```
# if文の文法（EBNF、Python風）
<if_statement> ::= 'if' <expression> ':' NEWLINE INDENT <block> DEDENT
                    ('elif' <expression> ':' NEWLINE INDENT <block> DEDENT)*
                    ['else' ':' NEWLINE INDENT <block> DEDENT]

<block>        ::= <statement>+
```

この形式的定義があるおかげで、パーサジェネレータ（yacc, ANTLR, PEG）を使ってソースコードの構文解析器を自動生成できる。

#### 具象構文 vs 抽象構文

```
ソースコード:  1 + 2 * 3

具象構文木（CST / Parse Tree）:
  expression
  ├── term
  │   └── factor
  │       └── number: 1
  ├── '+'
  └── term
      ├── factor
      │   └── number: 2
      ├── '*'
      └── factor
          └── number: 3

抽象構文木（AST）:
  BinaryOp(+)
  ├── Number(1)
  └── BinaryOp(*)
      ├── Number(2)
      └── Number(3)
```

ASTは不要なトークン（括弧、セミコロンなど）を除去し、プログラムの本質的な構造だけを表現する。コンパイラやインタプリタの内部では、ASTに対して意味解析や最適化を行う。

#### 構文設計の哲学の違い

```python
# Python: "There should be one—and preferably only one—obvious way to do it"
# リストの合計値を求める方法
total = sum(numbers)

# Pythonは意図的に構文を制限し、「正しい方法」を1つに絞る
```

```perl
# Perl: "There's more than one way to do it" (TMTOWTDI)
# 同じことを複数の方法で書ける
my $total = 0;
$total += $_ for @numbers;                    # 方法1
my $total2 = List::Util::sum(@numbers);       # 方法2
my $total3 = eval join '+', @numbers;         # 方法3（非推奨だが可能）
```

```ruby
# Ruby: "Optimized for programmer happiness"
# メソッドチェーンと複数の同義シンタックス
total = numbers.sum                            # 最も簡潔
total = numbers.reduce(:+)                     # 関数型スタイル
total = numbers.inject(0) { |sum, n| sum + n } # ブロック明示
```

### 2.2 意味論（Semantics）

プログラムの「意味」。実行すると何が起こるか。

```javascript
// JavaScriptの意味論の罠
"5" + 3       // → "53"（文字列結合）
"5" - 3       // → 2（数値演算）

[] + []        // → ""
[] + {}        // → "[object Object]"
{} + []        // → 0
```

```python
# Pythonの意味論
"5" + 3       # → TypeError（型の安全性）
```

意味論の設計が言語の「安全性」と「驚きの少なさ」を決める。

#### 意味論の3つの記述方法

言語の意味論を厳密に定義するために、計算機科学では3つの形式的手法が用いられる。

```
1. 操作的意味論（Operational Semantics）
   プログラムの実行手順を抽象機械の状態遷移として定義する。

   例: 「a + b」の評価
   ⟨a + b, σ⟩ → ⟨v₁ + v₂, σ⟩  ただし ⟨a, σ⟩ → v₁ かつ ⟨b, σ⟩ → v₂

   σ は環境（変数束縛の集合）、v は値を表す。

2. 表示的意味論（Denotational Semantics）
   プログラムの意味を数学的関数として定義する。

   例: [[if b then e₁ else e₂]] =
       λσ. if [[b]]σ = true then [[e₁]]σ else [[e₂]]σ

3. 公理的意味論（Axiomatic Semantics）
   事前条件・事後条件の関係として定義する（ホーア論理）。

   例: {x = n ∧ n ≥ 0} y := x * 2 {y = 2n ∧ y ≥ 0}
```

#### 評価戦略の違い

言語によって式の評価タイミングが異なる。これは意味論の重要な一部である。

```python
# Python: 正格評価（Strict Evaluation）
# 引数は関数呼び出しの前に全て評価される
def first(a, b):
    return a

# division_by_zero は呼ばれなくても評価される
# first(1, 1/0)  # → ZeroDivisionError
```

```haskell
-- Haskell: 遅延評価（Lazy Evaluation）
-- 式は必要になるまで評価されない
first a b = a

-- 1/0 は使われないので評価されない
-- first 1 (1/0)  -- → 1（エラーにならない）

-- 無限リストも定義可能（必要な分だけ評価される）
naturals = [1..]           -- 全自然数のリスト
take 5 naturals            -- → [1, 2, 3, 4, 5]

-- フィボナッチ数列を無限リストとして定義
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)
take 10 fibs               -- → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```python
# Python: ジェネレータによる遅延評価の部分的サポート
def naturals():
    n = 1
    while True:
        yield n
        n += 1

import itertools
list(itertools.islice(naturals(), 5))  # → [1, 2, 3, 4, 5]
```

#### スコープと束縛の意味論

```python
# Python: レキシカルスコープ（静的スコープ）
x = 10

def outer():
    x = 20
    def inner():
        print(x)  # 20（定義時の環境を参照）
    inner()

outer()
```

```javascript
// JavaScript: レキシカルスコープ + クロージャ
function makeCounter() {
    let count = 0;                    // クロージャに捕捉される
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count,
    };
}

const counter = makeCounter();
counter.increment(); // 1
counter.increment(); // 2
counter.getCount();  // 2
```

```
# ダイナミックスコープ（歴史的に使われた、現代言語ではほぼ廃止）
# Emacs Lispの例:
(defvar x 10)

(defun show-x ()
  (message "%d" x))  ; 呼び出し時の環境のxを参照

(defun change-x ()
  (let ((x 20))      ; xを動的に束縛
    (show-x)))        ; → 20（レキシカルスコープなら10になる）
```

### 2.3 プラグマティクス（Pragmatics）

言語の「実践的な側面」。エコシステム、ツール、コミュニティ。

```
言語選択に影響する実践的要素:

  ライブラリ:     npm（JS: 200万+パッケージ）
                  PyPI（Python: 50万+パッケージ）
                  crates.io（Rust: 15万+パッケージ）

  ツールチェーン: コンパイラ、リンター、フォーマッタ、デバッガ
  IDE サポート:   補完、型チェック、リファクタリング
  コミュニティ:   ドキュメント、Stack Overflow、GitHub
  採用市場:       求人数、給与水準
  実行環境:       ブラウザ、サーバー、組み込み、モバイル
```

#### エコシステムの成熟度比較

```
言語別ツールチェーン成熟度:

  ┌────────────┬─────────┬──────────┬──────────┬──────────┬──────────┐
  │ 言語        │ パッケージ│ フォーマッタ│ リンター  │ LSP      │ テスト   │
  │            │ 管理     │          │          │          │ FW       │
  ├────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
  │ Rust       │ cargo   │ rustfmt  │ clippy   │ rust-    │ 組込み   │
  │            │         │ (公式)   │ (公式)   │ analyzer │          │
  ├────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
  │ Go         │ go mod  │ gofmt    │ go vet   │ gopls    │ 組込み   │
  │            │         │ (公式)   │ (公式)   │ (公式)   │          │
  ├────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
  │ TypeScript │ npm/yarn│ prettier │ eslint   │ tsserver │ jest/    │
  │            │ pnpm    │ biome    │ biome    │          │ vitest   │
  ├────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
  │ Python     │ pip/uv  │ black    │ ruff     │ pylsp/   │ pytest   │
  │            │ poetry  │ ruff     │ mypy     │ pyright  │          │
  ├────────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
  │ Java       │ maven   │ google-  │ checkstyle│ jdtls   │ junit    │
  │            │ gradle  │ java-fmt │ spotbugs │          │          │
  └────────────┴─────────┴──────────┴──────────┴──────────┴──────────┘

  評価ポイント:
  - 公式ツールの充実度: Rust, Go は公式ツールが最も充実
  - サードパーティの選択肢: JavaScript/TypeScript が最も豊富
  - 一貫性: Rust の cargo は build/test/bench/doc/publish を統一
```

#### パッケージマネージャの設計思想

```bash
# Rust: Cargo — 全てを統一
cargo new my-project    # プロジェクト作成
cargo build             # ビルド
cargo test              # テスト実行
cargo bench             # ベンチマーク
cargo doc --open        # ドキュメント生成+表示
cargo publish           # crates.io へ公開
cargo clippy            # リンター実行
cargo fmt               # フォーマッタ実行
cargo audit             # セキュリティ監査

# Go: go コマンド — シンプルさを追求
go mod init example.com/myproject
go build ./...
go test ./...
go vet ./...
go fmt ./...
go doc fmt              # 標準ライブラリのドキュメント表示

# Node.js: npm — 巨大エコシステム
npm init
npm install express
npm test
npm run build
npm audit               # セキュリティ監査
npx eslint .            # ツールを一時的にインストール＆実行
```

---

## 3. 言語の分類

### 実行方式による分類

```
┌─────────────────────────────────────────────────────┐
│ コンパイル型                                          │
│ ソースコード → コンパイラ → 機械語 → 実行              │
│ 例: C, C++, Rust, Go, Swift                          │
│ 利点: 高速実行、事前エラー検出                         │
│ 欠点: コンパイル時間、プラットフォーム依存              │
├─────────────────────────────────────────────────────┤
│ インタプリタ型                                        │
│ ソースコード → インタプリタ → 逐次実行                 │
│ 例: Python, Ruby, PHP, Perl                          │
│ 利点: 即座に実行、対話的開発                          │
│ 欠点: 実行速度、実行時エラー                          │
├─────────────────────────────────────────────────────┤
│ JIT コンパイル型                                      │
│ ソースコード → バイトコード → JIT → 機械語             │
│ 例: Java, C#, JavaScript(V8), Python(PyPy)           │
│ 利点: ポータビリティ + 実行時最適化                    │
│ 欠点: ウォームアップ時間、メモリ消費                   │
├─────────────────────────────────────────────────────┤
│ トランスパイル型                                      │
│ ソースコード → 別の高水準言語 → 実行                   │
│ 例: TypeScript→JS, Kotlin→JVM/JS, Elm→JS            │
│ 利点: 言語間の良いとこ取り                            │
└─────────────────────────────────────────────────────┘
```

### パラダイムによる分類

```
┌─────────────┬─────────────────────────────────────────┐
│ パラダイム    │ 特徴と代表言語                          │
├─────────────┼─────────────────────────────────────────┤
│ 手続き型     │ 命令の順次実行。C, Pascal, BASIC        │
│ オブジェクト │ データと振る舞いの結合。Java, C#, Ruby  │
│  指向        │                                         │
│ 関数型       │ 純粋関数と不変性。Haskell, Elm, Erlang │
│ 論理型       │ 論理式による記述。Prolog, Datalog       │
│ マルチ       │ 複数パラダイム対応。Python, Scala,      │
│  パラダイム  │ Rust, Kotlin, Swift, TypeScript         │
└─────────────┴─────────────────────────────────────────┘
```

### 型システムによる分類

```
┌────────────────────────────────────────────────────────────────┐
│ 静的型付け × 強い型付け                                         │
│ 型はコンパイル時に確定。暗黙の型変換を制限。                     │
│ 例: Rust, Haskell, Java, TypeScript, Go                       │
│ メリット: コンパイル時にバグを検出、IDEサポートが強力           │
├────────────────────────────────────────────────────────────────┤
│ 動的型付け × 強い型付け                                         │
│ 型は実行時に決定。暗黙の型変換を制限。                          │
│ 例: Python, Ruby, Erlang                                       │
│ メリット: 柔軟だが型安全性を維持                                │
├────────────────────────────────────────────────────────────────┤
│ 静的型付け × 弱い型付け                                         │
│ 型はコンパイル時に決定。暗黙の型変換が多い。                     │
│ 例: C, C++                                                     │
│ リスク: void*やキャストによる型安全性の破壊                      │
├────────────────────────────────────────────────────────────────┤
│ 動的型付け × 弱い型付け                                         │
│ 型は実行時に決定。暗黙の型変換が多い。                          │
│ 例: JavaScript, PHP（歴史的に）, Perl                          │
│ リスク: 予期せぬ型変換によるバグ                                │
└────────────────────────────────────────────────────────────────┘
```

### メモリ管理方式による分類

```
┌──────────────────┬──────────────────┬──────────────────────────┐
│ 管理方式          │ 代表言語          │ 特徴                     │
├──────────────────┼──────────────────┼──────────────────────────┤
│ 手動管理          │ C, C++           │ malloc/free              │
│                  │                  │ 高性能だがメモリリーク危険│
├──────────────────┼──────────────────┼──────────────────────────┤
│ 参照カウント      │ Swift, Python    │ 即座に解放               │
│                  │ (CPython), Perl  │ 循環参照問題あり         │
├──────────────────┼──────────────────┼──────────────────────────┤
│ トレーシングGC    │ Java, Go, C#,   │ 自動回収                 │
│                  │ JavaScript       │ 一時停止（STW）あり      │
├──────────────────┼──────────────────┼──────────────────────────┤
│ 所有権システム    │ Rust             │ コンパイル時にメモリ安全 │
│                  │                  │ を保証。ランタイムコスト0│
├──────────────────┼──────────────────┼──────────────────────────┤
│ リージョンベース  │ Cyclone, Koka   │ メモリ領域単位の管理     │
│                  │                  │ 研究色が強い             │
└──────────────────┴──────────────────┴──────────────────────────┘
```

各方式の実装例:

```c
// C: 手動メモリ管理
#include <stdlib.h>
#include <string.h>

char* create_greeting(const char* name) {
    // メモリを手動で確保
    size_t len = strlen("Hello, ") + strlen(name) + 2;
    char* result = (char*)malloc(len);
    if (result == NULL) {
        return NULL;  // メモリ確保失敗
    }
    snprintf(result, len, "Hello, %s!", name);
    return result;  // 呼び出し側がfreeする責任を負う
}

int main() {
    char* msg = create_greeting("World");
    if (msg) {
        printf("%s\n", msg);
        free(msg);  // ← これを忘れるとメモリリーク
    }
    return 0;
}
```

```rust
// Rust: 所有権システム
fn create_greeting(name: &str) -> String {
    // Stringは所有権を持つ。スコープを抜けると自動解放
    format!("Hello, {}!", name)
}

fn main() {
    let msg = create_greeting("World");
    println!("{}", msg);
    // msg はここでドロップ（自動解放）される
    // free()を呼ぶ必要がない。忘れることもできない
}
```

```go
// Go: ガベージコレクション
func createGreeting(name string) string {
    // GCが管理するヒープに確保される
    return fmt.Sprintf("Hello, %s!", name)
}

func main() {
    msg := createGreeting("World")
    fmt.Println(msg)
    // GCが不要になったメモリを自動回収
    // プログラマはメモリ管理を意識する必要がない
}
```

---

## 4. 言語設計のトレードオフ

### 安全性 vs 自由度

```
最大安全性                                    最大自由度
  ←──────────────────────────────────────→
  Haskell  Rust  Java  Go  Python  C  Assembly

  Haskell: 副作用をモナドで管理。型で不正を排除
  Rust:    所有権システムでメモリ安全を保証
  Java:    GCでメモリ管理。NullPointerExceptionは残る
  Go:      シンプルだが nil パニックの可能性
  Python:  動的型で自由だがランタイムエラーのリスク
  C:       メモリを直接操作。バッファオーバーフロー可能
```

#### 安全性の具体例: Null安全

Null参照は「10億ドルの過ち」（Tony Hoare）と呼ばれる。各言語のアプローチを比較する。

```java
// Java: Null参照が可能（NullPointerException）
String name = null;
int len = name.length();  // → NullPointerException（実行時エラー）

// Java 8+ : Optional型で安全に扱う
Optional<String> maybeName = Optional.ofNullable(getName());
int len = maybeName.map(String::length).orElse(0);
```

```kotlin
// Kotlin: Null安全が言語レベルで組み込み
var name: String = "Alice"   // Non-null型：nullを代入できない
// name = null                // コンパイルエラー

var name2: String? = null    // Nullable型：nullを許容
val len = name2?.length ?: 0 // 安全呼び出し + エルビス演算子
```

```rust
// Rust: Null自体が存在しない。代わりにOption型
fn find_user(id: u64) -> Option<String> {
    if id == 1 {
        Some("Alice".to_string())
    } else {
        None
    }
}

fn main() {
    // パターンマッチで安全に取り出す
    match find_user(1) {
        Some(name) => println!("Found: {}", name),
        None => println!("User not found"),
    }

    // メソッドチェーンでも扱える
    let name = find_user(2).unwrap_or("Unknown".to_string());
}
```

```swift
// Swift: Optional型 + if let / guard let
func findUser(id: Int) -> String? {
    return id == 1 ? "Alice" : nil
}

// if let でアンラップ
if let name = findUser(id: 1) {
    print("Found: \(name)")
}

// guard let で早期リターン
func processUser(id: Int) {
    guard let name = findUser(id: id) else {
        print("User not found")
        return
    }
    print("Processing: \(name)")
}
```

### 表現力 vs 可読性

```python
# Python: 可読性を重視（"There should be one obvious way to do it"）
result = [x * 2 for x in numbers if x > 0]
```

```perl
# Perl: 表現力を重視（"There's more than one way to do it"）
@result = map { $_ * 2 } grep { $_ > 0 } @numbers;
```

```haskell
-- Haskell: 数学的な表現力
result = map (*2) . filter (>0) $ numbers
```

#### 表現力と可読性の実践的トレードオフ

```python
# Python: 同じ処理の書き方の違い

# 方法1: 最も読みやすい（推奨）
active_users = []
for user in users:
    if user.is_active and user.age >= 18:
        active_users.append(user.name.upper())

# 方法2: リスト内包表記（中級者向け、Pythonic）
active_users = [
    user.name.upper()
    for user in users
    if user.is_active and user.age >= 18
]

# 方法3: 過度に関数型（非推奨 — Pythonの文化に合わない）
active_users = list(map(
    lambda u: u.name.upper(),
    filter(lambda u: u.is_active and u.age >= 18, users)
))
```

```rust
// Rust: イテレータチェーン — 表現力と性能を両立
let active_users: Vec<String> = users
    .iter()
    .filter(|u| u.is_active && u.age >= 18)
    .map(|u| u.name.to_uppercase())
    .collect();

// Rustのイテレータは遅延評価 + ゼロコスト抽象化
// コンパイル後はforループと同等の機械語になる
```

### 暗黙 vs 明示

```python
# Python: 型の動作は暗黙的
x = 5           # 型推論（暗黙）
x = "hello"     # 型が変わる（動的型付け）
```

```rust
// Rust: 全てが明示的
let x: i32 = 5;                     // 型を明示
let y = 5;                          // 推論可能な場合は省略可
// let x = "hello";                 // コンパイルエラー（型が異なる）
let x: &str = "hello";             // 再宣言（シャドウイング）
```

#### 暗黙 vs 明示のさらなる例

```go
// Go: エラー処理は明示的（例外機構なし）
file, err := os.Open("data.txt")
if err != nil {
    return fmt.Errorf("failed to open file: %w", err)
}
defer file.Close()

data, err := io.ReadAll(file)
if err != nil {
    return fmt.Errorf("failed to read file: %w", err)
}
```

```python
# Python: エラー処理は例外機構（暗黙的にスタックを巻き戻す）
try:
    with open("data.txt") as f:
        data = f.read()
except FileNotFoundError:
    print("ファイルが見つかりません")
except PermissionError:
    print("アクセス権限がありません")
```

```rust
// Rust: Result型 + ? 演算子（明示的だが簡潔）
fn read_file(path: &str) -> Result<String, std::io::Error> {
    let data = std::fs::read_to_string(path)?;  // エラー時は早期リターン
    Ok(data)
}

// ? 演算子は以下の糖衣構文
fn read_file_verbose(path: &str) -> Result<String, std::io::Error> {
    let data = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => return Err(e),
    };
    Ok(data)
}
```

### コンパイル速度 vs 実行速度

```
コンパイル速度と実行速度のトレードオフ:

  ┌───────────┬──────────────┬──────────────┬─────────────────┐
  │ 言語       │ コンパイル速度│ 実行速度     │ 理由            │
  ├───────────┼──────────────┼──────────────┼─────────────────┤
  │ Go        │ 非常に速い    │ 速い         │ シンプルな型    │
  │           │ (数秒)       │              │ システム        │
  ├───────────┼──────────────┼──────────────┼─────────────────┤
  │ Rust      │ 遅い          │ 非常に速い   │ 所有権チェック  │
  │           │ (分単位)     │              │ + 最適化        │
  ├───────────┼──────────────┼──────────────┼─────────────────┤
  │ C++       │ 遅い          │ 非常に速い   │ テンプレート    │
  │           │ (分単位)     │              │ 展開 + 最適化   │
  ├───────────┼──────────────┼──────────────┼─────────────────┤
  │ Java      │ 普通          │ 速い         │ JITで段階的     │
  │           │              │ (ウォーム後) │ 最適化          │
  ├───────────┼──────────────┼──────────────┼─────────────────┤
  │ TypeScript│ 普通          │ (JSに依存)  │ 型チェック時間  │
  │           │              │              │ が支配的        │
  └───────────┴──────────────┴──────────────┴─────────────────┘
```

### 並行処理モデルの設計

言語の並行処理モデルは、その言語の適用領域を大きく決定する。

```go
// Go: ゴルーチン + チャネル（CSPモデル）
func main() {
    ch := make(chan string)

    go func() {
        // 別のゴルーチンで実行
        result := heavyComputation()
        ch <- result  // チャネルに結果を送信
    }()

    value := <-ch  // チャネルから受信（ブロッキング）
    fmt.Println(value)
}

// 複数のゴルーチンとselect
func fanIn(ch1, ch2 <-chan string) <-chan string {
    merged := make(chan string)
    go func() {
        for {
            select {
            case msg := <-ch1:
                merged <- msg
            case msg := <-ch2:
                merged <- msg
            }
        }
    }()
    return merged
}
```

```rust
// Rust: 所有権による安全な並行処理
use std::thread;
use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel();

    // スレッドに所有権を移動（moveキーワード）
    thread::spawn(move || {
        let result = heavy_computation();
        tx.send(result).unwrap();
    });

    let value = rx.recv().unwrap();
    println!("{}", value);
}

// データ競合はコンパイル時に防止される
// 以下はコンパイルエラーになる:
// let data = vec![1, 2, 3];
// thread::spawn(move || { data.push(4); });
// println!("{:?}", data);  // dataは既にmoveされている
```

```erlang
%% Erlang: アクターモデル
%% 各プロセスは独立したメモリ空間を持つ
-module(example).
-export([start/0, loop/0]).

start() ->
    Pid = spawn(fun loop/0),
    Pid ! {self(), "Hello"},
    receive
        {Pid, Response} -> io:format("~s~n", [Response])
    end.

loop() ->
    receive
        {From, Message} ->
            From ! {self(), "Got: " ++ Message},
            loop()
    end.
```

```javascript
// JavaScript: イベントループ + async/await
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        const user = await response.json();

        // 並行してプロフィールと投稿を取得
        const [profile, posts] = await Promise.all([
            fetch(`/api/profiles/${user.profileId}`).then(r => r.json()),
            fetch(`/api/posts?userId=${userId}`).then(r => r.json()),
        ]);

        return { user, profile, posts };
    } catch (error) {
        console.error("Failed to fetch user data:", error);
        throw error;
    }
}
```

---

## 5. 言語の歴史と系譜

### プログラミング言語の進化

```
1950年代: 最初の高水準言語
  FORTRAN (1957) — 科学技術計算
  LISP    (1958) — 記号処理、関数型の原型
  COBOL   (1959) — 事務処理
  ALGOL   (1958) — アルゴリズム記述、構造化プログラミングの源流

1960-70年代: 構造化プログラミング
  BASIC   (1964) — 教育用
  Pascal  (1970) — 構造化プログラミングの教育用言語
  C       (1972) — UNIXの開発言語。システム記述の標準
  Smalltalk (1972) — 純粋オブジェクト指向の原型
  Prolog  (1972) — 論理プログラミング
  ML      (1973) — 型推論、パターンマッチの原型
  Scheme  (1975) — Lispの簡潔な方言
  SQL     (1974) — 関係データベース問い合わせ

1980年代: オブジェクト指向の台頭
  C++     (1983) — Cにオブジェクト指向を追加
  Objective-C (1984) — C + Smalltalk風メッセージング
  Erlang  (1986) — 並行分散システム、耐障害性
  Perl    (1987) — テキスト処理、「スイスアーミーナイフ」
  Haskell (1990) — 純粋関数型言語の標準

1990年代: インターネットとRAD
  Python  (1991) — 可読性重視の汎用言語
  Ruby    (1995) — 「プログラマの幸福」を追求
  Java    (1995) — "Write Once, Run Anywhere"
  JavaScript (1995) — Webブラウザのスクリプト言語
  PHP     (1995) — Webサーバーサイド
  OCaml   (1996) — MLの実用的な方言

2000年代: 安全性と生産性の向上
  C#      (2000) — Microsoftの.NETプラットフォーム言語
  Scala   (2003) — JVM上の関数型+オブジェクト指向
  Groovy  (2003) — JVM上の動的言語
  F#      (2005) — .NET上の関数型言語
  Clojure (2007) — JVM上のLisp方言
  Go      (2009) — Googleの並行処理重視言語

2010年代: 安全性とパフォーマンスの両立
  Rust    (2010) — メモリ安全 + ゼロコスト抽象化
  Kotlin  (2011) — JVM上のモダン言語、Android公式
  Elixir  (2011) — Erlang VM上の生産性高い言語
  TypeScript (2012) — JavaScriptに静的型付けを追加
  Swift   (2014) — Apple の Objective-C 後継
  Zig     (2015) — C の安全な代替を目指す

2020年代: AI時代の言語
  Mojo    (2023) — Pythonの上位互換 + システム言語の性能
  Carbon  (2022) — C++ の後継を目指す（Google）
  Vale    (2023) — 新しいメモリ管理モデルの実験
```

### 言語の影響関係

```
FORTRAN ─────────────────────────────────────────→ Julia
    │
ALGOL ──→ Pascal ──→ Modula ──→ Oberon
    │         │
    │         └──→ Ada
    │
    └──→ C ──→ C++ ──→ Java ──→ C#
         │         │         │       │
         │         │         │       └──→ TypeScript
         │         │         │
         │         │         └──→ Kotlin
         │         │
         │         └──→ Rust
         │         │
         │         └──→ D
         │
         └──→ Objective-C ──→ Swift
         │
         └──→ Go

LISP ──→ Scheme ──→ Clojure
    │
    └──→ ML ──→ OCaml ──→ Rust（パターンマッチ、型システム）
         │         │
         │         └──→ F#
         │
         └──→ Haskell ──→ Elm
                    │
                    └──→ PureScript

Smalltalk ──→ Ruby
    │
    └──→ Objective-C ──→ Swift

Erlang ──→ Elixir
```

---

## 6. 言語選択の思考フレームワーク

```
プロジェクトに最適な言語を選ぶ判断基準:

  1. ドメイン適合性
     Web フロント    → JavaScript/TypeScript
     Web バック      → Go, Python, Node.js, Java
     システム        → Rust, C++, C
     データ分析      → Python, R, Julia
     モバイル        → Swift(iOS), Kotlin(Android)

  2. チームの習熟度
     チームが得意な言語 > 理論上最適な言語
     学習コスト・採用コストを含めて判断

  3. エコシステムの充実度
     必要なライブラリ・フレームワークがあるか
     ツールチェーン（ビルド・テスト・デプロイ）の成熟度

  4. パフォーマンス要件
     リアルタイム / 低レイテンシ → C++, Rust, Go
     バッチ処理 / スクリプト → Python, Ruby
     高スループット → Go, Java, Rust

  5. 長期保守性
     型安全性、ドキュメント、コミュニティの活発さ
     10年後もサポートされているか
```

### 言語選択マトリクス: 実践的なケーススタディ

```
ケース1: スタートアップのWebアプリケーション
  要件: 迅速な開発、少人数チーム、将来のスケーラビリティ
  推奨: TypeScript (Next.js) + PostgreSQL
  理由: フルスタックTypeScriptで型安全、採用しやすい、
        フロント・バック・DBの型を一貫して管理可能

ケース2: 高頻度取引システム
  要件: マイクロ秒レベルのレイテンシ、予測可能なパフォーマンス
  推奨: C++ または Rust
  理由: GCなし、メモリレイアウトの完全制御、SIMD最適化可能

ケース3: データパイプライン / ML基盤
  要件: 豊富なML/データライブラリ、プロトタイピング速度
  推奨: Python + Rust（パフォーマンスクリティカルな部分）
  理由: PyTorch/TensorFlow/pandas のエコシステム、
        Rustでネイティブ拡張を書いて高速化

ケース4: マイクロサービス基盤
  要件: 高い並行性、低メモリフットプリント、デプロイの容易さ
  推奨: Go
  理由: ゴルーチンによる軽量並行処理、シングルバイナリ、
        高速コンパイル、Docker イメージが小さい

ケース5: 組み込みシステム / IoT
  要件: 限られたメモリ・CPU、リアルタイム制約
  推奨: C または Rust (no_std)
  理由: ヒープ不要、ベアメタル対応、
        割り込みハンドラの精密な制御

ケース6: 分散システム / メッセージブローカー
  要件: 耐障害性、ホットコードリロード、数百万の同時接続
  推奨: Erlang/Elixir
  理由: OTPフレームワーク、let-it-crash哲学、
        BEAM VMの強力な分散機能

ケース7: ゲーム開発
  要件: リアルタイムレンダリング、物理シミュレーション
  推奨: C++ (Unreal) または C# (Unity)
  理由: 成熟したゲームエンジン、GPU制御、
        大規模コミュニティとアセット
```

### 言語移行の判断基準

```
既存言語から新しい言語に移行すべきか？

チェックリスト:
  □ 現在の言語で解決困難な技術的課題があるか？
  □ 移行先の言語で明確な改善が見込めるか？
  □ チームが新言語を学ぶ意欲と余裕があるか？
  □ 段階的な移行が可能か？（ビッグバンリライトは危険）
  □ 移行先のエコシステムに必要なライブラリがあるか？
  □ 採用市場で移行先の言語のエンジニアを確保できるか？

成功事例:
  - Discord: Go → Rust（レイテンシのスパイク解消）
  - Dropbox: Python → Rust（パフォーマンス10x改善）
  - Twitter: Ruby → Scala/Java（スケーラビリティ向上）
  - Microsoft: JavaScript → TypeScript（大規模コードベースの保守性）
  - Meta: PHP → Hack（段階的な型安全の導入）

失敗しやすいパターン:
  - 「流行っているから」で移行（技術的理由がない）
  - 全面リライト（段階的移行のほうが安全）
  - 移行期間の過小見積もり（2倍以上かかることが多い）
  - チームの合意形成不足（言語の好みは宗教戦争になりがち）
```

---

## 7. 言語処理系の内部構造

プログラミング言語を理解するには、言語処理系（コンパイラ/インタプリタ）がどのようにソースコードを処理するかを知ることが有益である。

### コンパイラのパイプライン

```
ソースコード
    │
    ▼
┌──────────────┐
│ 字句解析      │  ソースコードをトークン列に分割
│ (Lexer)      │  "let x = 42;" → [LET, IDENT("x"), EQUAL, NUM(42), SEMI]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 構文解析      │  トークン列を抽象構文木（AST）に変換
│ (Parser)     │  文法規則に基づいて構造を認識
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 意味解析      │  型チェック、スコープ解決、名前解決
│ (Semantic    │  変数の未定義・型不一致を検出
│  Analysis)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 中間表現生成  │  AST → IR（中間表現）への変換
│ (IR Gen)     │  最適化しやすい形式に変換
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 最適化        │  定数畳み込み、デッドコード除去、
│ (Optimizer)  │  ループ最適化、インライン展開
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ コード生成    │  IR → ターゲットの機械語
│ (Code Gen)   │  レジスタ割り当て、命令選択
└──────┬───────┘
       │
       ▼
  実行可能バイナリ
```

### 最適化の具体例

```c
// 最適化前のCコード
int sum_array(int* arr, int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total = total + arr[i];
    }
    return total;
}

// コンパイラが行う最適化:

// 1. ループアンローリング
//    ループ本体を複数回展開して分岐コストを削減
int sum_array_unrolled(int* arr, int n) {
    int total = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        total += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
    }
    for (; i < n; i++) {
        total += arr[i];
    }
    return total;
}

// 2. SIMD化（自動ベクトル化）
//    複数のデータを1命令で処理
//    例: SSE/AVXで4個/8個のintを同時加算

// 3. 定数畳み込み
int x = 3 + 4;      // → int x = 7; （コンパイル時に計算）
int y = x * 2;      // → int y = 14;

// 4. デッドコード除去
int unused = expensive_function();  // 使われない → 削除
```

```
# GCCの最適化レベル
gcc -O0 main.c  # 最適化なし（デバッグ用）
gcc -O1 main.c  # 基本的な最適化
gcc -O2 main.c  # 推奨される最適化レベル
gcc -O3 main.c  # 積極的な最適化（バイナリが大きくなることも）
gcc -Os main.c  # サイズ最適化（組み込み向け）
gcc -Ofast main.c  # 浮動小数点の厳密さを犠牲にして高速化
```

### インタプリタの実装方式

```
方式1: ツリーウォーキングインタプリタ
  ASTを直接辿って実行する。最もシンプルだが最も遅い。

  例: 初期のRuby, 教育用インタプリタ

方式2: バイトコードインタプリタ
  ASTをバイトコードにコンパイルし、仮想マシンで実行する。

  例: CPython, Ruby (YARV), Lua, Erlang (BEAM)

方式3: JITコンパイラ
  実行時にバイトコードをネイティブ機械語にコンパイルする。
  ホットスポット（頻繁に実行される部分）を重点的に最適化。

  例: V8 (JavaScript), HotSpot (Java), PyPy (Python)

方式4: トレーシングJIT
  実際の実行パスを記録し、その情報を基に最適化する。

  例: LuaJIT, PyPy
```

#### 簡単なインタプリタの実装例

```python
# Python で書く簡易的な電卓インタプリタ

from dataclasses import dataclass
from typing import Union

# AST ノード定義
@dataclass
class Number:
    value: float

@dataclass
class BinaryOp:
    op: str
    left: 'Expr'
    right: 'Expr'

Expr = Union[Number, BinaryOp]

# パーサ（簡略版: トークン列からASTを生成）
class Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Expr:
        return self.parse_expression()

    def parse_expression(self) -> Expr:
        left = self.parse_term()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('+', '-'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_term()
            left = BinaryOp(op, left, right)
        return left

    def parse_term(self) -> Expr:
        left = self.parse_factor()
        while self.pos < len(self.tokens) and self.tokens[self.pos] in ('*', '/'):
            op = self.tokens[self.pos]
            self.pos += 1
            right = self.parse_factor()
            left = BinaryOp(op, left, right)
        return left

    def parse_factor(self) -> Expr:
        token = self.tokens[self.pos]
        if token == '(':
            self.pos += 1
            expr = self.parse_expression()
            self.pos += 1  # skip ')'
            return expr
        else:
            self.pos += 1
            return Number(float(token))

# 評価器（ASTを辿って計算）
def evaluate(expr: Expr) -> float:
    match expr:
        case Number(value):
            return value
        case BinaryOp('+', left, right):
            return evaluate(left) + evaluate(right)
        case BinaryOp('-', left, right):
            return evaluate(left) - evaluate(right)
        case BinaryOp('*', left, right):
            return evaluate(left) * evaluate(right)
        case BinaryOp('/', left, right):
            return evaluate(left) / evaluate(right)

# 使用例
tokens = ['(', '1', '+', '2', ')', '*', '3']
ast = Parser(tokens).parse()
result = evaluate(ast)
print(f"Result: {result}")  # → Result: 9.0
```

---

## 8. 形式言語理論との関係

プログラミング言語は形式言語理論（チョムスキー階層）に基づいて分類される。

### チョムスキー階層

```
レベル3: 正規言語（Regular Languages）
  認識器: 有限オートマトン（DFA/NFA）
  表現:   正規表現
  用途:   字句解析（トークナイザ）
  例:     識別子、数値リテラル、キーワードのパターン

  識別子のパターン: [a-zA-Z_][a-zA-Z0-9_]*
  整数リテラル:     [0-9]+
  浮動小数点:       [0-9]+\.[0-9]+([eE][+-]?[0-9]+)?

レベル2: 文脈自由言語（Context-Free Languages）
  認識器: プッシュダウンオートマトン
  表現:   BNF/EBNF/PEG
  用途:   構文解析（パーサ）
  例:     プログラミング言語の大部分の構文

  括弧の対応: { } の入れ子構造
  式の構文:   演算子の優先順位

レベル1: 文脈依存言語（Context-Sensitive Languages）
  認識器: 線形拘束オートマトン
  用途:   型チェック、名前解決
  例:     変数が使用前に宣言されているかのチェック

レベル0: 再帰可能列挙言語（Recursively Enumerable）
  認識器: チューリングマシン
  用途:   完全な意味解析
  例:     プログラムが停止するかの判定（決定不能）
```

### プログラミング言語とチューリング完全性

```
チューリング完全な言語:
  任意の計算可能関数を表現できる言語。

  条件:
  - 条件分岐（if/else）
  - 繰り返し（while/再帰）
  - 任意量のメモリ（制限なし）

  ほぼ全ての汎用プログラミング言語はチューリング完全。

  意外なチューリング完全システム:
  - CSS + HTML（アニメーションを使ったトリック）
  - SQL（再帰CTE）
  - Excel の数式
  - LaTeX のマクロシステム
  - sed（ストリームエディタ）
  - Minecraft のレッドストーン回路
  - Conway's Game of Life

チューリング完全でない言語（意図的に制限）:
  - 正規表現（ループがない）
  - JSON, YAML, TOML（データ記述のみ）
  - HTML（マークアップのみ）
  - Dhall（設定記述言語、停止が保証される）
  - Agda, Coq の全域関数（停止性が保証される）
```

---

## 9. 現代の言語設計トレンド

### ゼロコスト抽象化

```rust
// Rust: ゼロコスト抽象化の例
// 高水準な抽象を使っても、手書きの低水準コードと同等の性能

// 高水準版（イテレータ）
fn sum_of_squares(numbers: &[i32]) -> i32 {
    numbers.iter()
        .filter(|&&x| x > 0)
        .map(|&x| x * x)
        .sum()
}

// 低水準版（手動ループ）
fn sum_of_squares_manual(numbers: &[i32]) -> i32 {
    let mut total = 0;
    for i in 0..numbers.len() {
        if numbers[i] > 0 {
            total += numbers[i] * numbers[i];
        }
    }
    total
}

// 両者は同じ機械語にコンパイルされる（-O2以上）
```

### 代数的データ型（ADT）

```rust
// Rust: enum でデータの全パターンを型で表現
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
        Shape::Rectangle { width, height } => width * height,
        Shape::Triangle { base, height } => 0.5 * base * height,
        // 全パターンを網羅しないとコンパイルエラー
    }
}
```

```typescript
// TypeScript: Union型 + 判別フィールド
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle":
            return 0.5 * shape.base * shape.height;
        // TypeScript の strictNullChecks + noUncheckedIndexedAccess で
        // 網羅チェックが可能
    }
}
```

```haskell
-- Haskell: 代数的データ型の本家
data Shape
    = Circle Double
    | Rectangle Double Double
    | Triangle Double Double

area :: Shape -> Double
area (Circle r)      = pi * r * r
area (Rectangle w h) = w * h
area (Triangle b h)  = 0.5 * b * h
-- パターンマッチの網羅性をコンパイラが検査
```

### エフェクトシステムと型レベルプログラミング

```
最先端の言語設計トレンド:

1. エフェクトシステム
   副作用（IO、例外、非決定性など）を型で追跡する。
   - Koka: algebraic effects
   - Haskell: モナドによる副作用管理
   - Rust: unsafe による危険な操作の明示

2. 依存型（Dependent Types）
   値に依存する型を表現できる。
   - Idris: 型レベルで「長さnのリスト」を表現
   - Agda: 証明支援言語としても使用
   - Coq: 数学的証明の形式化

3. リニア型 / アフィン型
   値の使用回数を型で制限する。
   - Rust: 所有権 = アフィン型（最大1回使用）
   - Linear Haskell: リニア型拡張
   - Clean: 一意型（uniqueness types）

4. マクロとメタプログラミング
   コンパイル時にコードを生成・変換する。
   - Rust: 手続きマクロ、derive マクロ
   - Lisp: マクロ（homoiconicity による強力なメタプログラミング）
   - Zig: comptime（コンパイル時計算）
   - Nim: テンプレートとマクロ
```

---

## 実践演習

### 演習1: [基礎] — 同じアルゴリズムを3言語で書く

FizzBuzzを Python, JavaScript, Rust で実装し、構文の違いを比較する。

```python
# Python
for i in range(1, 101):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

```javascript
// JavaScript
for (let i = 1; i <= 100; i++) {
    if (i % 15 === 0) {
        console.log("FizzBuzz");
    } else if (i % 3 === 0) {
        console.log("Fizz");
    } else if (i % 5 === 0) {
        console.log("Buzz");
    } else {
        console.log(i);
    }
}
```

```rust
// Rust
fn main() {
    for i in 1..=100 {
        match (i % 3, i % 5) {
            (0, 0) => println!("FizzBuzz"),
            (0, _) => println!("Fizz"),
            (_, 0) => println!("Buzz"),
            _      => println!("{}", i),
        }
    }
}
```

比較ポイント:
- Python: インデントベース、range() のイテレータ
- JavaScript: C系の構文、=== による厳密比較
- Rust: パターンマッチによる網羅的な分岐

### 演習2: [応用] — 言語の実行速度比較

フィボナッチ数列の計算を複数言語で実装し、実行時間を計測する。

```python
# Python: 再帰版
import time

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

start = time.time()
result = fib(40)
elapsed = time.time() - start
print(f"fib(40) = {result}, time: {elapsed:.3f}s")
# 典型的な結果: 約30-60秒
```

```python
# Python: メモ化版（動的計画法）
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

result = fib_memo(100)
# → 354224848179261915075（瞬時に計算）
```

```rust
// Rust: 再帰版
use std::time::Instant;

fn fib(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}

fn main() {
    let start = Instant::now();
    let result = fib(40);
    let elapsed = start.elapsed();
    println!("fib(40) = {}, time: {:.3?}", result, elapsed);
    // 典型的な結果: 約0.5-1秒
}
```

```go
// Go: 再帰版
package main

import (
    "fmt"
    "time"
)

func fib(n int) int {
    if n <= 1 {
        return n
    }
    return fib(n-1) + fib(n-2)
}

func main() {
    start := time.Now()
    result := fib(40)
    elapsed := time.Since(start)
    fmt.Printf("fib(40) = %d, time: %s\n", result, elapsed)
    // 典型的な結果: 約0.5-1.5秒
}
```

### 演習3: [発展] — 言語選定レポート

架空のプロジェクト（例: リアルタイムチャットアプリ）に最適な言語を選定し、根拠を述べる。

```
レポートテンプレート:

1. プロジェクト概要
   - アプリケーション種別: リアルタイムチャットアプリ
   - 想定ユーザー数: 10万DAU
   - 主要機能: 1対1チャット、グループチャット、ファイル共有
   - 非機能要件: メッセージ配信レイテンシ < 100ms

2. 技術要件の整理
   - WebSocket 常時接続の管理
   - メッセージの永続化（DB）
   - プッシュ通知との連携
   - 水平スケーラビリティ
   - E2E暗号化

3. 言語候補の評価
   ┌────────┬──────┬──────┬──────┬──────┬──────┐
   │ 基準    │ Go   │ Rust │ Node │ Elixir│ Java │
   ├────────┼──────┼──────┼──────┼──────┼──────┤
   │ 並行性能│ ◎    │ ◎    │ ○    │ ◎    │ ○    │
   │ 開発速度│ ○    │ △    │ ◎    │ ○    │ △    │
   │ 採用容易│ ○    │ △    │ ◎    │ △    │ ◎    │
   │ WS性能 │ ◎    │ ◎    │ ○    │ ◎    │ ○    │
   │ エコ   │ ○    │ ○    │ ◎    │ ○    │ ◎    │
   └────────┴──────┴──────┴──────┴──────┴──────┘

4. 結論と根拠
   推奨: Go（最有力）/ Elixir（代替案）

   Goの根拠:
   - ゴルーチンで10万+の同時接続を軽量に管理
   - gorilla/websocket等の成熟したライブラリ
   - デプロイが容易（シングルバイナリ）
   - 学習曲線が緩やか、採用しやすい

   Elixirの根拠:
   - Phoenix LiveViewのリアルタイム機能
   - BEAM VMの数百万プロセス管理能力
   - let-it-crash による耐障害性
   - ただし採用市場が小さい
```

### 演習4: [発展] — ミニ言語の設計

独自のドメイン特化言語（DSL）を設計し、その構文と意味論を定義する。

```
課題: タスク管理DSLを設計せよ

構文案（EBNF）:
  <program>    ::= <statement>*
  <statement>  ::= <task_def> | <project_def> | <query>
  <task_def>   ::= 'task' <string> '{' <task_body> '}'
  <task_body>  ::= (<field> '=' <value>)*
  <field>      ::= 'priority' | 'due' | 'assignee' | 'status'
  <project_def>::= 'project' <string> '{' <task_def>* '}'
  <query>      ::= 'find' 'tasks' 'where' <condition>

使用例:
  project "Website Redesign" {
      task "Design mockup" {
          priority = high
          due = 2024-03-15
          assignee = "Alice"
          status = in_progress
      }
      task "Implement frontend" {
          priority = medium
          due = 2024-04-01
          assignee = "Bob"
          status = todo
      }
  }

  find tasks where priority = high and status != done
```

---

## FAQ

### Q1: 最初に学ぶべき言語は？
A: Python（読みやすさとエコシステムの広さ）か JavaScript（Webの普遍性）が推奨。重要なのは1つの言語を深く学ぶこと。2つ目以降の言語は格段に習得しやすくなる。

### Q2: 言語は何個覚えるべき？
A: 3〜5言語を異なるパラダイムから選ぶのが理想。例: Python（動的スクリプト） + TypeScript（静的Web） + Rust（システム） + SQL（データ）。

### Q3: 新しい言語はどうやって学ぶ？
A: 「構文→型システム→メモリモデル→並行処理→エコシステム」の順で学ぶと効率的。既知の言語との差分に注目する。

### Q4: プログラミング言語の数はどれくらいある？
A: 記録されているプログラミング言語は約8,000以上ある。ただし実務で広く使われるのは20-30程度。TIOBE Index や Stack Overflow Survey で人気言語のトレンドを確認できる。主要な分類としては、汎用言語（Python, Java, C++など）、ドメイン特化言語（SQL, R, MATLAB, Verilogなど）、教育用言語（Scratch, Logo, Processingなど）がある。

### Q5: 言語の「良し悪し」はあるか？
A: 絶対的な「最良の言語」は存在しない。言語の評価は以下の文脈に依存する:
- 解決する問題のドメイン
- チームの経験とスキル
- パフォーマンス要件
- 開発速度の要件
- 保守性の要件
- エコシステムの成熟度

「銀の弾丸はない」（No Silver Bullet — Fred Brooks, 1986）というソフトウェア工学の原則がここでも当てはまる。

### Q6: AIはプログラミング言語に影響を与えるか？
A: AIの発展により、以下の変化が起きている:
- **コード生成**: LLMによるコード生成が日常的に使われるようになり、言語の構文の覚えやすさの重要性が相対的に低下
- **型システム**: AIが生成したコードの正しさを保証するために、強い型システムの価値がむしろ上昇
- **DSL**: 自然言語に近いDSLをAIが介在することで実用化できる可能性
- **新言語設計**: Mojo のようにAI/MLのワークロードに特化した言語の登場
- **静的解析**: AIによるバグ検出・脆弱性検出が言語のツールチェーンに統合

### Q7: 関数型プログラミングは実務で使えるか？
A: 純粋関数型言語（Haskell, Elm）の採用は限定的だが、関数型の概念は主流言語に広く浸透している:
- **map/filter/reduce**: Python, JavaScript, Java, Rust 全てで利用可能
- **イミュータビリティ**: React の状態管理、Rustの所有権
- **パターンマッチ**: Python 3.10+, Java 21+, C# 8+
- **代数的データ型**: Rust の enum, TypeScript の Union型
- **関数の第一級市民**: ほぼ全ての現代言語で関数を値として扱える

---

## まとめ

| 観点 | ポイント |
|------|---------|
| 言語の役割 | 抽象化の階段を提供する |
| 3つの側面 | 構文・意味論・プラグマティクス |
| 実行方式 | コンパイル / インタプリタ / JIT |
| パラダイム | 手続き / OOP / 関数型 / マルチ |
| 型システム | 静的/動的 × 強い/弱い の2軸 |
| メモリ管理 | 手動 / RC / GC / 所有権 |
| 設計トレードオフ | 安全性 vs 自由度、明示 vs 暗黙 |
| 言語選択 | ドメイン・チーム・エコシステム・性能・保守性 |
| 歴史 | 1950年代の FORTRAN から現代の Rust/Mojo まで |
| 処理系 | 字句解析→構文解析→意味解析→最適化→コード生成 |
| 形式言語理論 | チョムスキー階層とチューリング完全性 |
| 現代トレンド | ゼロコスト抽象化、ADT、エフェクトシステム |

---

## 次に読むべきガイド
→ [[01-compilation-vs-interpretation.md]] — コンパイルとインタプリタ

---

## 参考文献
1. Abelson, H. & Sussman, G. "Structure and Interpretation of Computer Programs." MIT Press, 1996.
2. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
3. Van Roy, P. & Haridi, S. "Concepts, Techniques, and Models of Computer Programming." MIT Press, 2004.
4. Scott, M. "Programming Language Pragmatics." Morgan Kaufmann, 4th Edition, 2015.
5. Harper, R. "Practical Foundations for Programming Languages." Cambridge University Press, 2016.
6. Appel, A. "Modern Compiler Implementation in ML." Cambridge University Press, 2004.
7. Wirth, N. "Compiler Construction." Addison-Wesley, 1996.
8. Sebesta, R. "Concepts of Programming Languages." Pearson, 12th Edition, 2018.
9. Brooks, F. "No Silver Bullet — Essence and Accidents of Software Engineering." IEEE Computer, 1986.
10. Hoare, C.A.R. "Null References: The Billion Dollar Mistake." QCon London, 2009.
11. Backus, J. "Can Programming Be Liberated from the von Neumann Style?" Communications of the ACM, 1978.
12. Chomsky, N. "Three Models for the Description of Language." IRE Transactions on Information Theory, 1956.
