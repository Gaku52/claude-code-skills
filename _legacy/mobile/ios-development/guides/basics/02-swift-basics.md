# Swift基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [Swiftとは](#swiftとは)
3. [変数と定数](#変数と定数)
4. [データ型](#データ型)
5. [演算子](#演算子)
6. [制御構文](#制御構文)
7. [関数](#関数)
8. [オプショナル](#オプショナル)
9. [演習問題](#演習問題)
10. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- Swift言語の基本文法
- 変数、定数、データ型
- 制御構文（if、for、while）
- 関数の定義と呼び出し
- オプショナルの概念

### 学習時間：1〜1.5時間

---

## Swiftとは

### Swiftの特徴

**Swift**は、Appleが2014年に発表したモダンなプログラミング言語です。

**特徴**：
- **安全性**：型安全、メモリ安全
- **高速**：C/C++並みのパフォーマンス
- **読みやすい**：シンプルな構文
- **モダン**：最新のプログラミング言語の機能

### Playgroundで試す

**Xcode Playground**は、Swiftコードを即座に実行できる環境です。

```
Xcode → File → New → Playground
```

---

## 変数と定数

### 変数（var）

**変数**は、後から値を変更できます。

```swift
var score = 10
score = 20  // OK：変更可能

var playerName = "太郎"
playerName = "花子"  // OK
```

### 定数（let）

**定数**は、一度設定したら変更できません。

```swift
let maxScore = 100
// maxScore = 200  // エラー：変更不可

let appName = "MyApp"
// appName = "NewApp"  // エラー
```

### 命名規則

```swift
// ✅ 良い命名（キャメルケース）
let userName = "太郎"
let totalScore = 100
let isGameOver = false

// ❌ 悪い命名
let user_name = "太郎"  // スネークケースは避ける
let x = 100            // 意味不明
```

---

## データ型

### 基本的なデータ型

#### Int（整数）

```swift
let age = 25
let year = 2024
let score: Int = 100  // 型を明示
```

#### Double / Float（小数）

```swift
let pi = 3.14159  // Double（デフォルト）
let price: Double = 1980.0
let temperature: Float = 36.5
```

#### String（文字列）

```swift
let name = "太郎"
let message = "こんにちは"

// 文字列結合
let greeting = "こんにちは、" + name + "さん"

// 文字列補間（推奨）
let greeting2 = "こんにちは、\(name)さん"
```

#### Bool（真偽値）

```swift
let isLoggedIn = true
let hasPermission = false

if isLoggedIn {
    print("ログイン済み")
}
```

### 型推論

Swiftは自動的に型を推論します。

```swift
let score = 100        // Int と推論
let name = "太郎"       // String と推論
let price = 1980.0     // Double と推論
let isActive = true    // Bool と推論
```

### 型変換

```swift
let intValue = 10
let doubleValue = Double(intValue)  // 10.0

let priceString = "1980"
let price = Int(priceString)  // 1980（Optional）

let pi = 3.14159
let piInt = Int(pi)  // 3（小数点切り捨て）
```

---

## 演算子

### 算術演算子

```swift
let a = 10
let b = 3

let sum = a + b        // 13（加算）
let diff = a - b       // 7（減算）
let product = a * b    // 30（乗算）
let quotient = a / b   // 3（除算）
let remainder = a % b  // 1（剰余）
```

### 比較演算子

```swift
let x = 10
let y = 20

x == y  // false（等しい）
x != y  // true（等しくない）
x < y   // true（より小さい）
x > y   // false（より大きい）
x <= y  // true（以下）
x >= y  // false（以上）
```

### 論理演算子

```swift
let isLoggedIn = true
let hasPermission = false

// AND（かつ）
isLoggedIn && hasPermission  // false

// OR（または）
isLoggedIn || hasPermission  // true

// NOT（否定）
!isLoggedIn  // false
```

---

## 制御構文

### if文

```swift
let score = 85

if score >= 90 {
    print("優")
} else if score >= 80 {
    print("良")
} else if score >= 70 {
    print("可")
} else {
    print("不可")
}
```

### 三項演算子

```swift
let age = 18
let status = age >= 20 ? "成人" : "未成年"
```

### switch文

```swift
let fruit = "りんご"

switch fruit {
case "りんご":
    print("赤い果物")
case "バナナ":
    print("黄色い果物")
case "ぶどう":
    print("紫の果物")
default:
    print("その他の果物")
}

// 複数のケース
switch score {
case 90...100:
    print("優")
case 80..<90:
    print("良")
case 70..<80:
    print("可")
default:
    print("不可")
}
```

### forループ

```swift
// 範囲
for i in 1...5 {
    print(i)  // 1, 2, 3, 4, 5
}

for i in 1..<5 {
    print(i)  // 1, 2, 3, 4
}

// 配列
let fruits = ["りんご", "バナナ", "ぶどう"]
for fruit in fruits {
    print(fruit)
}

// インデックス付き
for (index, fruit) in fruits.enumerated() {
    print("\(index): \(fruit)")
}
```

### whileループ

```swift
var count = 0

while count < 5 {
    print(count)
    count += 1
}

// repeat-while（do-whileと同じ）
repeat {
    print(count)
    count += 1
} while count < 10
```

---

## 関数

### 関数の定義

```swift
// 基本形
func greet() {
    print("こんにちは！")
}

greet()  // 呼び出し

// 引数あり
func greet(name: String) {
    print("こんにちは、\(name)さん！")
}

greet(name: "太郎")

// 戻り値あり
func add(a: Int, b: Int) -> Int {
    return a + b
}

let result = add(a: 10, b: 20)  // 30
```

### 引数ラベル

```swift
// 外部ラベルと内部ラベル
func greet(to name: String) {
    print("こんにちは、\(name)さん！")
}

greet(to: "太郎")  // 外部ラベル: to

// アンダースコアで省略
func add(_ a: Int, _ b: Int) -> Int {
    return a + b
}

let sum = add(10, 20)  // ラベルなし
```

### デフォルト引数

```swift
func greet(name: String = "ゲスト") {
    print("こんにちは、\(name)さん！")
}

greet()              // "こんにちは、ゲストさん！"
greet(name: "太郎")  // "こんにちは、太郎さん！"
```

---

## オプショナル

### オプショナルとは

**オプショナル**は、「値がある」または「値がない（nil）」を表す型です。

```swift
var name: String? = "太郎"
name = nil  // OK

// var age: Int = nil  // エラー：通常の型にはnilを代入できない
```

### オプショナルバインディング

```swift
let input = "42"
let number = Int(input)  // Optional(42)

// if let
if let unwrapped = number {
    print("数値: \(unwrapped)")
} else {
    print("変換失敗")
}

// guard let（早期リターン）
func process(input: String?) {
    guard let text = input else {
        print("入力がありません")
        return
    }

    print("入力: \(text)")
}
```

### 強制アンラップ（非推奨）

```swift
let number: Int? = 42
print(number!)  // 42

let nilValue: Int? = nil
// print(nilValue!)  // クラッシュ！
```

### nil合体演算子

```swift
let username: String? = nil
let displayName = username ?? "ゲスト"  // "ゲスト"
```

### オプショナルチェイニング

```swift
struct User {
    var address: Address?
}

struct Address {
    var city: String
}

let user = User(address: Address(city: "東京"))
let city = user.address?.city  // Optional("東京")

let userWithoutAddress = User(address: nil)
let noCity = userWithoutAddress.address?.city  // nil
```

---

## 演習問題

### 問題1：FizzBuzz

1から100までの数字を出力してください。ただし：
- 3の倍数のときは「Fizz」
- 5の倍数のときは「Buzz」
- 両方の倍数のときは「FizzBuzz」

**解答例**：

```swift
for i in 1...100 {
    if i % 15 == 0 {
        print("FizzBuzz")
    } else if i % 3 == 0 {
        print("Fizz")
    } else if i % 5 == 0 {
        print("Buzz")
    } else {
        print(i)
    }
}
```

### 問題2：温度変換

華氏（Fahrenheit）を摂氏（Celsius）に変換する関数を作成してください。

計算式：`C = (F - 32) × 5/9`

**解答例**：

```swift
func fahrenheitToCelsius(_ fahrenheit: Double) -> Double {
    return (fahrenheit - 32) * 5 / 9
}

let celsius = fahrenheitToCelsius(100)
print("\(celsius)°C")  // 37.77...°C
```

---

## よくある間違い

### ❌ 間違い1：定数を変更しようとする

```swift
let score = 10
// score = 20  // エラー：letは変更不可
```

**✅ 正しい方法**：

```swift
var score = 10
score = 20  // OK
```

### ❌ 間違い2：型の不一致

```swift
let age = 25
// let message = "年齢: " + age  // エラー：String + Int
```

**✅ 正しい方法**：

```swift
let message = "年齢: \(age)"
// または
let message = "年齢: " + String(age)
```

### ❌ 間違い3：オプショナルの強制アンラップ

```swift
let input: String? = nil
// let text = input!  // クラッシュ！
```

**✅ 正しい方法**：

```swift
if let text = input {
    print(text)
} else {
    print("値がありません")
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Swift言語の基本文法
- ✅ 変数と定数
- ✅ データ型
- ✅ 制御構文
- ✅ 関数
- ✅ オプショナル

### 次に学ぶべきガイド

**次のガイド**：[03-xcode-intro.md](./03-xcode-intro.md) - Xcode入門

---

**前のガイド**：[01-what-is-ios-development.md](./01-what-is-ios-development.md)

**親ガイド**：[iOS Development - SKILL.md](../../SKILL.md)
