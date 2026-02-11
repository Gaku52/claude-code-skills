# JVM 言語比較（Java, Kotlin, Scala, Clojure）

> JVM（Java Virtual Machine）上で動く言語群。Javaの巨大なエコシステムを共有しつつ、それぞれが異なる哲学で進化。

## この章で学ぶこと

- [ ] JVM 言語のエコシステムと互換性を理解する
- [ ] 各言語の特徴と使い分けを把握する

---

## 1. 比較表

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Java     │ Kotlin   │ Scala    │ Clojure  │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ パラダイム    │ OOP      │ OOP+FP  │ OOP+FP  │ FP       │
│              │          │ マルチ   │ マルチ   │ Lisp系   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 型付け        │ 静的     │ 静的     │ 静的     │ 動的     │
│              │ nominal  │ 型推論強 │ 型推論最強│          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ Null安全      │ なし     │ あり     │ Option  │ nil      │
│              │ (NPE)   │ (言語組込)│          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主な用途      │ 企業     │ Android │ データ   │ Web      │
│              │ バックエンド│ サーバー │ 分散処理 │ データ   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主要FW       │ Spring   │ Ktor    │ Akka     │ Ring     │
│              │ Quarkus  │ Spring  │ Play     │ Luminus  │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 学習コスト    │ 中程度   │ 低い    │ 高い     │ 高い     │
│              │          │ (Java経験│          │ (Lisp系) │
│              │          │  あれば) │          │          │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. 同じ処理の比較

```java
// Java: 冗長だが明確
public record User(String name, int age) {}

List<String> names = users.stream()
    .filter(u -> u.age() >= 18)
    .map(User::name)
    .sorted()
    .collect(Collectors.toList());
```

```kotlin
// Kotlin: 簡潔で安全
data class User(val name: String, val age: Int)

val names = users
    .filter { it.age >= 18 }
    .map { it.name }
    .sorted()
```

```scala
// Scala: 最も表現力が高い
case class User(name: String, age: Int)

val names = users
  .filter(_.age >= 18)
  .map(_.name)
  .sorted
```

```clojure
;; Clojure: データ中心・不変
(->> users
     (filter #(>= (:age %) 18))
     (map :name)
     (sort))
```

---

## 3. 選択指針

```
エンタープライズ / 大チーム       → Java（安定・人材豊富）
Android ネイティブ               → Kotlin（Google公式）
Java の近代化 / 新規プロジェクト  → Kotlin
大規模データ処理（Spark）        → Scala
関数型プログラミング             → Scala or Clojure
マイクロサービス                 → Kotlin(Ktor) or Java(Quarkus)
```

---

## まとめ

| 言語 | 哲学 | 最適な場面 |
|------|------|----------|
| Java | 安定・エンタープライズ | 大規模業務システム |
| Kotlin | 簡潔・安全・実用的 | Android, サーバーサイド |
| Scala | 表現力・型安全・FP | データ処理, 分散システム |
| Clojure | シンプル・不変・REPL | データ処理, Web |

---

## 次に読むべきガイド
→ [[03-functional-languages.md]] — 関数型言語比較

---

## 参考文献
1. Odersky, M. "Programming in Scala." 5th Ed, Artima, 2023.
2. Jemerov, D. & Isakova, S. "Kotlin in Action." 2nd Ed, Manning, 2024.
