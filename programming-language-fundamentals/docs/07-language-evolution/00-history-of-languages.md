# プログラミング言語の歴史

> 言語の歴史を知ることは、現在の言語設計の「なぜ」を理解し、未来のトレンドを予測する力を与える。

## この章で学ぶこと

- [ ] プログラミング言語の進化の流れを把握する
- [ ] 各時代の革新と影響を理解する

---

## 1. 年表

```
1950s: 黎明期
  1957  FORTRAN   — 初の高水準言語（科学計算）
  1958  LISP      — 関数型の祖。GC、REPL の発明
  1959  COBOL     — ビジネス処理。今も銀行で稼働

1960s: 構造化の時代
  1964  BASIC     — 教育用。パーソナルコンピュータの普及に貢献
  1967  Simula    — OOPの祖。クラス・継承の概念を導入

1970s: システムと理論
  1970  Pascal    — 構造化プログラミングの教育用
  1972  C         — Unix と共に誕生。システムプログラミングの基盤
  1972  Prolog    — 論理型プログラミング
  1973  ML        — 型推論の祖。Hindley-Milner型システム
  1978  SQL       — リレーショナルDB問い合わせ

1980s: OOP の台頭
  1983  C++       — C + OOP。「ゼロコスト抽象化」
  1984  Common Lisp — Lisp の統一規格
  1986  Erlang    — 通信システム向け。アクターモデル・耐障害性
  1987  Perl      — テキスト処理。「duct tape of the internet」

1990s: Web とスクリプトの時代
  1990  Haskell   — 純粋関数型の標準
  1991  Python    — 読みやすさ重視。「Batteries included」
  1993  Ruby      — 開発者の幸福。「全てがオブジェクト」
  1995  Java      — 「Write once, run anywhere」。JVM
  1995  JavaScript— Web ブラウザの言語（10日で作られた）
  1995  PHP       — Web サーバーサイド
  1996  OCaml     — MLファミリー。実用的関数型

2000s: モダン言語の基盤
  2003  Scala     — JVM上のOOP+FP
  2005  F#        — .NET上のFP
  2007  Clojure   — JVM上のLisp

2010s: 安全性と効率の追求
  2010  Rust      — メモリ安全 + ゼロコスト抽象化
  2011  Kotlin    — より良いJava。Android公式
  2011  Elixir    — Erlang VM上のモダン言語
  2012  Go        — シンプル + 並行処理。Google発
  2012  TypeScript— JavaScript + 型安全
  2014  Swift     — Apple のモダン言語

2020s: AI時代の言語
  2023  Mojo      — Python互換 + C性能（AI向け）
  2024  Gleam     — BEAM上の型付き関数型
```

---

## 2. 影響の系譜

```
FORTRAN → BASIC → Visual Basic
       ↘
LISP → Scheme → Clojure, Racket
     ↘ ML → OCaml → F#, Rust(一部)
          ↘ Haskell → Elm, PureScript

Simula → Smalltalk → Ruby, Objective-C
      ↘ C++ → Java → C#, Kotlin
              ↘ JavaScript → TypeScript

C → C++ → Rust, Go, Zig
  ↘ Objective-C → Swift

Erlang → Elixir, Gleam

AWK + sed + sh → Perl → PHP, Ruby, Python
```

---

## 3. 重要な革新

```
GC（1958, LISP）       → Java, Go, Python 等が採用
OOP（1967, Simula）    → Java, C++, Python 等の基盤
型推論（1973, ML）     → Rust, TypeScript, Kotlin が採用
パターンマッチ（ML系）  → Rust, Scala, Python 3.10 が採用
async/await（C# 5.0）  → JS, Python, Rust が採用
所有権（2015, Rust）   → 今後の言語に影響を与える可能性
```

---

## まとめ

| 時代 | キーワード | 代表言語 |
|------|----------|---------|
| 1950-60s | 高水準化 | FORTRAN, LISP, COBOL |
| 1970s | 構造化・型理論 | C, ML, Pascal |
| 1980s | OOP | C++, Erlang |
| 1990s | Web・スクリプト | Java, JS, Python, Ruby |
| 2010s | 安全性・並行 | Rust, Go, TypeScript |
| 2020s | AI・パフォーマンス | Mojo, Gleam |

---

## 次に読むべきガイド
→ [[01-modern-language-features.md]] — モダン言語の機能

---

## 参考文献
1. Sebesta, R. "Concepts of Programming Languages." 12th Ed, Pearson, 2019.
