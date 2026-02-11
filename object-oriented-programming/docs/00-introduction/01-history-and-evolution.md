# OOPの歴史と進化

> OOPは1960年代のSimulaに始まり、Smalltalk、C++、Javaを経て、現代のマルチパラダイム言語へと進化してきた。歴史を知ることで「なぜこう設計されているのか」が分かる。

## この章で学ぶこと

- [ ] OOPの誕生から現代までの進化を理解する
- [ ] 各時代の革新とその影響を把握する
- [ ] 現代のOOPがどこに向かっているかを展望する

---

## 1. OOPの年表

```
1960s: 誕生
  1967  Simula       — OOPの祖。クラス・継承の概念を導入
                       ノルウェーの Dahl と Nygaard が開発
                       シミュレーション用途 → 一般化

1970s: 純粋OOPの確立
  1972  Smalltalk    — Alan Kay（Xerox PARC）
                       「すべてはオブジェクト」
                       メッセージパッシング、GC、IDE、GUI
                       → 現代のOOP概念の大部分を確立

1980s: 実用化
  1983  C++          — Bjarne Stroustrup
                       C + OOP。「ゼロコスト抽象化」
                       静的型付け + 多重継承
  1986  Objective-C  — C + Smalltalk のメッセージング
                       → Apple/NeXT で採用

1990s: 普及
  1995  Java         — Sun Microsystems
                       「Write once, run anywhere」
                       単一継承 + インターフェース
                       GC、JVM → エンタープライズの標準
  1995  JavaScript   — プロトタイプベースOOP
                       クラスなしでオブジェクト生成
  1993  Ruby         — まつもとゆきひろ
                       「全てがオブジェクト」、開発者の幸福

2000s: 反省と改良
  2000  C#           — Microsoft（Java への対抗）
                       プロパティ、デリゲート、LINQ
  2003  Scala        — OOP + FP の融合
                       JVM上で動作

2010s: モダンOOP
  2011  Kotlin       — より良いJava。null安全、データクラス
  2014  Swift        — プロトコル指向プログラミング
                       値型中心、参照カウント
  2012  TypeScript   — JavaScript + 型安全
                       構造的型付け

2020s: ポストOOP
  → マルチパラダイム（OOP + FP）が標準
  → 「純粋OOP」から「必要に応じてOOP」へ
  → コンポジション重視、継承縮小
```

---

## 2. 各時代の革新

### Simula（1967）: クラスと継承の発明

```
Simula の革新:
  1. クラス（class）: データと手続きを一体化した「設計図」
  2. オブジェクト: クラスから生成された「実体」
  3. 継承（inheritance）: 既存クラスの拡張
  4. 仮想手続き（virtual procedure）: ポリモーフィズムの原型

背景:
  → 離散事象シミュレーションのために開発
  → 「現実世界のモノをプログラムで表現する」必要性
  → 顧客、車、工場...を「オブジェクト」として表現
```

### Smalltalk（1972）: 純粋OOPの確立

```
Smalltalk の革新:
  1. すべてがオブジェクト（数値、真偽値、nil も）
  2. メッセージパッシング（メソッド呼び出しではない）
  3. ガベージコレクション
  4. 統合開発環境（IDE）の発明
  5. MVC パターンの発明
  6. リフレクション（メタプログラミング）

Alan Kay の思想:
  「OOPとはメッセージングのことだ。
   クラスや継承よりも、オブジェクト間のメッセージ交換が本質」

  → 現代の多くの言語は Kay の意図とは異なる方向に進化
  → Kay: 「C++ や Java は私が意図した OOP ではない」
```

### C++（1983）: 実用化と静的型付け

```
C++ の革新:
  1. C との後方互換性（既存コードの活用）
  2. 静的型付けによるコンパイル時チェック
  3. 多重継承
  4. テンプレート（ジェネリクスの原型）
  5. 演算子オーバーロード
  6. RAII（Resource Acquisition Is Initialization）

影響:
  → 「OOP = クラス + 継承 + ポリモーフィズム」の定義を普及
  → ゼロコスト抽象化の思想
  → ただし複雑さも増大（C++ は最も複雑な言語の一つ）
```

### Java（1995）: エンタープライズ標準化

```
Java の革新:
  1. 単一継承 + インターフェース（多重継承の問題を回避）
  2. ガベージコレクション（C++ のメモリ管理から解放）
  3. JVM による移植性
  4. 豊富な標準ライブラリ
  5. パッケージによる名前空間管理

影響:
  → エンタープライズの標準言語に
  → デザインパターン（GoF）の普及
  → ただし「冗長すぎる」「ボイラープレート多すぎ」批判も
  → AbstractSingletonProxyFactoryBean 問題
```

---

## 3. OOPの進化の方向性

```
第1世代（1967-1980）: クラスベース
  Simula, Smalltalk
  → 「世界をオブジェクトでモデリングする」

第2世代（1983-1995）: 静的型付け + 実用化
  C++, Java, C#
  → 「大規模開発を構造化する」

第3世代（2000-2015）: 軽量OOP + FP融合
  Ruby, Scala, Kotlin, Swift
  → 「ボイラープレートを減らし、関数型の良さを取り入れる」

第4世代（2015-現在）: ポストOOP
  Rust（トレイト）, Go（インターフェース）, TypeScript（構造的型付け）
  → 「継承を排除し、コンポジションとインターフェースで設計する」

進化の傾向:
  多重継承 → 単一継承 → 継承よりコンポジション → 継承なし
  ミュータブル → イミュータブル優先
  クラス中心 → インターフェース/トレイト中心
  暗黙的 → 明示的
```

---

## 4. 現代のOOP: マルチパラダイム

```kotlin
// Kotlin: モダンOOPの例
// データクラス（ボイラープレート削減）
data class User(
    val name: String,
    val email: String,
    val age: Int
)

// sealed class（代数的データ型 — FPからの影響）
sealed class Result<out T> {
    data class Success<T>(val value: T) : Result<T>()
    data class Failure(val error: Throwable) : Result<Nothing>()
}

// 拡張関数（オブジェクトを変更せずに機能追加）
fun String.isValidEmail(): Boolean =
    this.matches(Regex("^[\\w.-]+@[\\w.-]+\\.[a-zA-Z]{2,}$"))

// 高階関数（FPの要素）
fun <T> List<T>.filterAndMap(
    predicate: (T) -> Boolean,
    transform: (T) -> String
): List<String> = this.filter(predicate).map(transform)
```

```swift
// Swift: プロトコル指向プログラミング
protocol Drawable {
    func draw()
}

protocol Resizable {
    func resize(by factor: Double)
}

// プロトコル拡張（デフォルト実装）
extension Drawable {
    func draw() {
        print("Default drawing")
    }
}

// 値型（struct）+ プロトコル準拠
struct Circle: Drawable, Resizable {
    var radius: Double

    func draw() {
        print("Drawing circle with radius \(radius)")
    }

    func resize(by factor: Double) -> Circle {
        Circle(radius: radius * factor)
    }
}
```

---

## まとめ

| 時代 | 言語 | 革新 |
|------|------|------|
| 1967 | Simula | クラス・継承の発明 |
| 1972 | Smalltalk | 純粋OOP・メッセージング |
| 1983 | C++ | 静的型付けOOP・実用化 |
| 1995 | Java | エンタープライズ標準・GC |
| 2010s | Kotlin/Swift | モダンOOP・FP融合 |
| 2020s | Rust/Go/TS | ポストOOP・コンポジション |

---

## 次に読むべきガイド
→ [[02-oop-vs-other-paradigms.md]] — OOP vs 他パラダイム

---

## 参考文献
1. Kay, A. "The Early History of Smalltalk." ACM SIGPLAN, 1993.
2. Stroustrup, B. "The Design and Evolution of C++." Addison-Wesley, 1994.
