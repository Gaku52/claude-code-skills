# モダン言語の共通機能

> 2010年代以降の言語は、過去の失敗を学び、共通の「良い機能」を取り入れている。

## この章で学ぶこと

- [ ] モダン言語に共通する機能を把握する
- [ ] 各機能がどの言語に由来するか理解する

---

## 1. モダン言語の標準装備

```
1. 型推論
   明示的な型注釈なしで型を自動推論
   由来: ML(1973) → Haskell → Rust, TS, Kotlin, Go, Swift

2. Null安全
   NullPointerException を型で防止
   由来: Haskell(Maybe) → Rust(Option), Kotlin(?), Swift(?), TS(strictNullChecks)

3. パターンマッチ
   データの構造に基づく分岐
   由来: ML → Haskell → Rust, Scala, Python(3.10), C#

4. async/await
   非同期処理の直感的な記述
   由来: C#(2012) → JS, Python, Rust, Kotlin, Swift

5. 代数的データ型（ADT）
   直和型 + 直積型でデータを正確にモデリング
   由来: ML → Haskell → Rust(enum), TS(union), Swift(enum), Kotlin(sealed)

6. イミュータビリティ優先
   デフォルトで不変、必要時のみ可変
   由来: Haskell → Rust(let vs let mut), Kotlin(val vs var), Swift(let vs var)

7. クロージャ / ラムダ式
   関数を第一級の値として扱う
   由来: LISP → 全モダン言語が採用

8. パッケージマネージャ内蔵
   依存管理の標準化
   Rust(Cargo), Go(go mod), Swift(SPM), Kotlin(Gradle)

9. フォーマッタ内蔵
   コードスタイルの統一
   Go(gofmt), Rust(rustfmt), Python(black), Zig(zig fmt)

10. 充実したエラーメッセージ
    Rust, Elm のコンパイラは教育的なエラーメッセージで有名
```

---

## 2. トレンドの方向性

```
2010年代:
  → 型安全性の強化（TypeScript, Rust, Kotlin）
  → Null安全（「10億ドルの間違い」の修正）
  → 並行処理の言語レベルサポート

2020年代:
  → AI統合（Copilot、コード生成、型推論の強化）
  → エッジ/Wasmへの対応
  → 段階的型付け（動的言語に型を追加）
  → エラー回復（Result型の普及）

全体の流れ:
  「自由」→「安全」→「生産性」→「AI協調」
```

---

## まとめ

| 機能 | 由来 | 採用言語 |
|------|------|---------|
| 型推論 | ML (1973) | Rust, TS, Go, Kotlin, Swift |
| Null安全 | Haskell | Rust, Kotlin, Swift, TS |
| パターンマッチ | ML | Rust, Scala, Python, C# |
| async/await | C# (2012) | JS, Python, Rust, Swift |
| ADT | ML | Rust, TS, Swift, Kotlin |
| 不変デフォルト | Haskell | Rust, Kotlin, Swift |

---

## 次に読むべきガイド
→ [[02-dsl-and-metaprogramming.md]] — DSLとメタプログラミング

---

## 参考文献
1. "Programming Language Evolution." SIGPLAN Notices.
