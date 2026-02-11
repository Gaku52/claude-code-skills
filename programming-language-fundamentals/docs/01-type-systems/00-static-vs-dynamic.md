# 静的型付け vs 動的型付け

> 型システムは「プログラムの正しさを保証する」最も基本的な仕組み。

## この章で学ぶこと

- [ ] 静的型付けと動的型付けの本質的な違いを理解する
- [ ] 強い型付けと弱い型付けの区別を理解する
- [ ] 段階的型付け（Gradual Typing）の意義を理解する

---

## 1. 型とは何か

```
型 = 「値の集合」と「その値に対して許される操作の集合」

例:
  int:    {..., -2, -1, 0, 1, 2, ...}  操作: +, -, *, /, %, 比較
  string: {"", "hello", "世界", ...}   操作: 連結, 長さ, 部分文字列
  bool:   {true, false}                操作: AND, OR, NOT

型が保証するもの:
  - メモリ上のデータの解釈方法
  - 値に対する操作の妥当性
  - プログラムの部分的な正しさ
```

---

## 2. 静的型付け（Static Typing）

```
コンパイル時（実行前）に型チェックを行う

タイムライン:
  ソースコード → [型チェック] → コンパイル → 実行
                 ↑ ここでエラー検出
```

```typescript
// TypeScript: 静的型付け
function add(a: number, b: number): number {
    return a + b;
}

add(1, 2);       // ✅ OK
add("1", 2);     // ❌ コンパイルエラー: Argument of type 'string'
                  //    is not assignable to parameter of type 'number'
```

```rust
// Rust: 強い静的型付け
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
// エラーの可能性が型で表現されている
// → 呼び出し側は Result を処理しないとコンパイルエラー
```

```go
// Go: シンプルな静的型付け
func add(a int, b int) int {
    return a + b
}

// var x int = "hello"  // コンパイルエラー
```

### 静的型付けの利点と欠点

```
利点:
  ✓ コンパイル時にバグを検出（実行前に安全性を保証）
  ✓ IDE の補完・リファクタリングが強力
  ✓ ドキュメントとしての機能（型が仕様を表す）
  ✓ パフォーマンス最適化が容易（型情報でメモリ配置を最適化）
  ✓ 大規模開発・チーム開発に適する

欠点:
  ✗ 記述量が増える（型注釈のオーバーヘッド）
  ✗ 柔軟性が低い（動的な構造の扱いが煩雑）
  ✗ コンパイル時間が必要
  ✗ 学習コストが高い場合がある（ジェネリクス等）
```

---

## 3. 動的型付け（Dynamic Typing）

```
実行時に型チェックを行う

タイムライン:
  ソースコード → インタプリタ → 実行 → [型チェック]
                                       ↑ ここでエラー検出
```

```python
# Python: 動的型付け
def add(a, b):
    return a + b

add(1, 2)         # → 3（数値の加算）
add("hello", " ") # → "hello "（文字列の結合）
add(1, "2")       # → TypeError（実行時エラー）
```

```ruby
# Ruby: 動的型付け + ダックタイピング
def process(obj)
  obj.each { |item| puts item }
end

process([1, 2, 3])           # Array
process(1..5)                # Range
process({a: 1, b: 2})       # Hash
# each メソッドがあれば何でも動く（ダックタイピング）
```

### 動的型付けの利点と欠点

```
利点:
  ✓ 記述が簡潔（型注釈不要）
  ✓ 柔軟性が高い（ダックタイピング）
  ✓ プロトタイピングが速い
  ✓ REPL で対話的に開発可能
  ✓ メタプログラミングが容易

欠点:
  ✗ 実行時エラーのリスク（テストでカバーする必要）
  ✗ IDE のサポートが限定的
  ✗ リファクタリングが困難（型情報がない）
  ✗ パフォーマンスのオーバーヘッド（実行時型チェック）
  ✗ 大規模開発で型の不整合が発見しにくい
```

---

## 4. 強い型付け vs 弱い型付け

```
静的/動的とは別の軸。暗黙の型変換をどこまで許すか。

強い型付け（厳格）                 弱い型付け（寛容）
  ←─────────────────────────────────→
  Python  Ruby  Java  C#  Go  C  JavaScript  PHP

強い型付け:
  暗黙の型変換を（ほとんど）許さない
  型が合わなければエラー
```

```python
# Python: 強い型付け + 動的
"5" + 3      # TypeError（暗黙変換しない）
int("5") + 3 # → 8（明示的に変換）
```

```javascript
// JavaScript: 弱い型付け + 動的
"5" + 3      // → "53"（暗黙の型変換: 数値→文字列）
"5" - 3      // → 2（暗黙の型変換: 文字列→数値）
true + 1     // → 2（暗黙の型変換: bool→数値）
[] == false  // → true（暗黙の型変換）
```

```c
// C: 弱い型付け + 静的
int x = 3.14;        // → 3（暗黙のトランケーション）
char c = 65;          // → 'A'（暗黙の変換）
void *p = &x;        // → void* に暗黙キャスト
```

### 分類マトリクス

```
            静的              動的
         ┌──────────────┬──────────────┐
  強い   │ Java, Rust,  │ Python,      │
         │ Go, Haskell, │ Ruby,        │
         │ TypeScript,  │ Elixir       │
         │ Kotlin, Swift│              │
         ├──────────────┼──────────────┤
  弱い   │ C, C++       │ JavaScript,  │
         │              │ PHP, Perl    │
         └──────────────┴──────────────┘
```

---

## 5. 段階的型付け（Gradual Typing）

```
動的型付け言語に段階的に型注釈を追加するアプローチ。
型注釈がない部分は動的型付けとして扱う。

代表例:
  TypeScript  → JavaScript に型を追加
  Python 型ヒント → Python に型注釈を追加
  PHP 型宣言  → PHP に型を追加
  Dart        → 段階的型付けを言語レベルで採用
```

```python
# Python の型ヒント（Gradual Typing）

# 型注釈なし（従来の動的型付け）
def greet(name):
    return f"Hello, {name}!"

# 型注釈あり（段階的に追加可能）
def greet(name: str) -> str:
    return f"Hello, {name}!"

# 型チェッカー（mypy）で静的解析
# mypy script.py → エラーをコンパイル前に検出

# 実行時の動作は変わらない（型ヒントは無視される）
greet(42)  # mypy はエラー、実行は可能
```

```typescript
// TypeScript: JavaScript との段階的な共存

// any 型: 型チェックを一時的に無効化
let data: any = fetchData();  // 移行期間中の一時措置

// unknown 型: any より安全（型ガードが必要）
let data: unknown = fetchData();
if (typeof data === "string") {
    console.log(data.toUpperCase());  // 型ガード後は安全
}

// strict モード: 段階的に厳格化
// tsconfig.json
// "strict": true,
// "noImplicitAny": true,
// "strictNullChecks": true
```

---

## 6. 型システムの健全性

```
健全性（Soundness）:
  「型チェックを通ったプログラムは実行時に型エラーを起こさない」

完全に健全な言語: Haskell, Rust（unsafeを除く）
ほぼ健全な言語:   Java（ジェネリクスの消去が抜け穴）
意図的に不健全:   TypeScript（any型、型アサーション）

TypeScript が完全な健全性を目指さない理由:
  → JavaScript との互換性
  → 既存コードの段階的な移行を可能にするため
  → 実用性（100%の健全性は開発者の負担が大きい）
```

---

## 実践演習

### 演習1: [基礎] — 型エラーの体験
Python, TypeScript, Rust で同じ型エラーを意図的に起こし、エラーメッセージと検出タイミングを比較する。

### 演習2: [応用] — Python への型ヒント追加
既存の Python スクリプトに型ヒントを追加し、mypy で静的解析を実行する。

### 演習3: [発展] — TypeScript の strict モード移行
JavaScript プロジェクトを TypeScript に移行し、strict モードで全ての型エラーを解消する。

---

## FAQ

### Q1: 動的型付け言語でも安全に開発できる？
A: テストカバレッジを高めること（特にユニットテスト）と、型ヒント + 静的解析ツール（mypy, Pyright）の活用で安全性を高められる。ただし大規模プロジェクトでは静的型付けが推奨される。

### Q2: TypeScript の `any` を使うべきでない理由は？
A: `any` は型チェックを完全に無効化するため、TypeScript を使う意味がなくなる。代わりに `unknown`（型ガード必須）や適切な型定義を使う。移行期間中の一時措置としてのみ許容。

### Q3: 型注釈は多いほど良い？
A: 適度が重要。型推論が効く場所では省略し、公開API・関数の引数・戻り値には明示的に書く。過度な型注釈はノイズになる。

---

## まとめ

| 分類 | 特徴 | 代表言語 |
|------|------|---------|
| 静的 + 強い | 最も安全。コンパイル時にエラー検出 | Rust, Haskell, Go |
| 静的 + 弱い | 高速だが暗黙変換に注意 | C, C++ |
| 動的 + 強い | 柔軟だが実行時エラーのリスク | Python, Ruby |
| 動的 + 弱い | 最も柔軟だが予測困難 | JavaScript, PHP |
| 段階的 | 動的から静的へ段階的に移行 | TypeScript, Python型ヒント |

---

## 次に読むべきガイド
→ [[01-type-inference.md]] — 型推論

---

## 参考文献
1. Pierce, B. "Types and Programming Languages." MIT Press, 2002.
2. Siek, J. & Taha, W. "Gradual Typing for Functional Languages." 2006.
