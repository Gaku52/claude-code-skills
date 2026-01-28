# JavaScript基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [変数と定数](#変数と定数)
3. [データ型](#データ型)
4. [関数](#関数)
5. [配列とオブジェクト](#配列とオブジェクト)
6. [ES6+の機能](#es6の機能)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- JavaScript の基本文法
- 変数、関数、配列、オブジェクト
- ES6+ のモダンな機能
- Node.js で使う重要な概念

### 学習時間：1〜1.5時間

---

## 変数と定数

### let（変数）

```javascript
let count = 0
count = 10  // 再代入可能

let message = 'Hello'
message = 'Hi'  // OK
```

### const（定数）

```javascript
const PI = 3.14159
// PI = 3.14  // エラー：再代入不可

const user = { name: '太郎' }
user.name = '花子'  // OK：オブジェクトの中身は変更可能
// user = {}  // エラー：変数自体の再代入は不可
```

### var（非推奨）

```javascript
// var は使わない（スコープの問題）
// 代わりに let / const を使う
```

---

## データ型

### プリミティブ型

```javascript
// 数値
const age = 25
const price = 1980.5

// 文字列
const name = '太郎'
const message = "Hello"

// 真偽値
const isActive = true
const hasPermission = false

// null / undefined
const empty = null
const notDefined = undefined
```

### 型変換

```javascript
// 文字列 → 数値
const str = '42'
const num = Number(str)  // 42
const num2 = parseInt(str)  // 42
const num3 = parseFloat('3.14')  // 3.14

// 数値 → 文字列
const n = 123
const s = String(n)  // '123'
const s2 = n.toString()  // '123'

// 真偽値変換
Boolean(1)  // true
Boolean(0)  // false
Boolean('')  // false
Boolean('text')  // true
```

---

## 関数

### 従来の関数宣言

```javascript
function greet(name) {
  return `こんにちは、${name}さん`
}

console.log(greet('太郎'))  // こんにちは、太郎さん
```

### アロー関数（推奨）

```javascript
// 基本形
const add = (a, b) => {
  return a + b
}

// 短縮形（return省略）
const add2 = (a, b) => a + b

// 引数が1つの場合
const double = x => x * 2

// 引数がない場合
const getTime = () => new Date()
```

### デフォルト引数

```javascript
const greet = (name = 'ゲスト') => {
  return `こんにちは、${name}さん`
}

console.log(greet())  // こんにちは、ゲストさん
console.log(greet('太郎'))  // こんにちは、太郎さん
```

---

## 配列とオブジェクト

### 配列

```javascript
const fruits = ['りんご', 'バナナ', 'ぶどう']

// アクセス
console.log(fruits[0])  // りんご

// 長さ
console.log(fruits.length)  // 3

// 追加
fruits.push('いちご')

// 削除
fruits.pop()  // 最後を削除

// map（変換）
const numbers = [1, 2, 3]
const doubled = numbers.map(n => n * 2)  // [2, 4, 6]

// filter（絞り込み）
const ages = [15, 25, 35]
const adults = ages.filter(age => age >= 20)  // [25, 35]

// find（検索）
const users = ['太郎', '花子', '次郎']
const user = users.find(u => u === '花子')  // 花子
```

### オブジェクト

```javascript
const user = {
  name: '太郎',
  age: 25,
  email: 'taro@example.com'
}

// アクセス
console.log(user.name)  // 太郎
console.log(user['age'])  // 25

// 追加・変更
user.city = '東京'
user.age = 26

// メソッド
const calculator = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b
}

console.log(calculator.add(10, 5))  // 15
```

---

## ES6+の機能

### テンプレートリテラル

```javascript
const name = '太郎'
const age = 25

// 従来
const message1 = 'こんにちは、' + name + 'さん（' + age + '歳）'

// テンプレートリテラル（推奨）
const message2 = `こんにちは、${name}さん（${age}歳）`

// 複数行
const html = `
  <div>
    <h1>${name}</h1>
    <p>年齢: ${age}</p>
  </div>
`
```

### 分割代入

```javascript
// 配列
const [a, b, c] = [1, 2, 3]
console.log(a)  // 1

// オブジェクト
const user = { name: '太郎', age: 25 }
const { name, age } = user
console.log(name)  // 太郎

// 関数の引数
const greet = ({ name, age }) => {
  return `${name}さん（${age}歳）`
}

greet({ name: '太郎', age: 25 })
```

### スプレッド構文

```javascript
// 配列の結合
const arr1 = [1, 2, 3]
const arr2 = [4, 5, 6]
const combined = [...arr1, ...arr2]  // [1, 2, 3, 4, 5, 6]

// オブジェクトのマージ
const user = { name: '太郎', age: 25 }
const updated = { ...user, age: 26 }  // { name: '太郎', age: 26 }

// 関数の引数
const numbers = [1, 2, 3]
console.log(Math.max(...numbers))  // 3
```

### モジュール

```javascript
// math.js（エクスポート）
export const add = (a, b) => a + b
export const subtract = (a, b) => a - b

// デフォルトエクスポート
export default class Calculator {}

// main.js（インポート）
import { add, subtract } from './math.js'
import Calculator from './math.js'

console.log(add(10, 5))  // 15
```

---

## Node.js特有の概念

### CommonJS モジュール

```javascript
// math.js（エクスポート）
const add = (a, b) => a + b
const subtract = (a, b) => a - b

module.exports = { add, subtract }

// main.js（インポート）
const { add, subtract } = require('./math')

console.log(add(10, 5))  // 15
```

### process オブジェクト

```javascript
// 環境変数
console.log(process.env.NODE_ENV)

// コマンドライン引数
console.log(process.argv)
// node app.js arg1 arg2
// ['node', '/path/to/app.js', 'arg1', 'arg2']

// 現在のディレクトリ
console.log(process.cwd())

// プロセス終了
process.exit(0)
```

---

## 演習問題

### 問題1：FizzBuzz

```javascript
// 1から100までの数字で：
// 3の倍数: Fizz
// 5の倍数: Buzz
// 両方の倍数: FizzBuzz

for (let i = 1; i <= 100; i++) {
  if (i % 15 === 0) {
    console.log('FizzBuzz')
  } else if (i % 3 === 0) {
    console.log('Fizz')
  } else if (i % 5 === 0) {
    console.log('Buzz')
  } else {
    console.log(i)
  }
}
```

### 問題2：配列操作

```javascript
// ユーザー配列から20歳以上の名前を抽出

const users = [
  { name: '太郎', age: 25 },
  { name: '花子', age: 17 },
  { name: '次郎', age: 30 }
]

const adults = users
  .filter(user => user.age >= 20)
  .map(user => user.name)

console.log(adults)  // ['太郎', '次郎']
```

---

## よくある間違い

### ❌ 間違い1：varを使う

```javascript
var x = 10  // 非推奨
```

**✅ 正しい方法**：

```javascript
const x = 10  // 推奨（再代入しない場合）
let y = 20    // 推奨（再代入する場合）
```

### ❌ 間違い2：==を使う

```javascript
'5' == 5  // true（型変換される）
```

**✅ 正しい方法**：

```javascript
'5' === 5  // false（型も比較）
```

### ❌ 間違い3：thisの誤用

```javascript
const obj = {
  name: '太郎',
  greet: function() {
    setTimeout(function() {
      console.log(this.name)  // undefined
    }, 1000)
  }
}
```

**✅ 正しい方法**：

```javascript
const obj = {
  name: '太郎',
  greet: function() {
    setTimeout(() => {
      console.log(this.name)  // 太郎
    }, 1000)
  }
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ JavaScript の基本文法
- ✅ 変数、関数、配列、オブジェクト
- ✅ ES6+ のモダンな機能
- ✅ Node.js で使う重要な概念

### 次に学ぶべきガイド

**次のガイド**：[03-npm-basics.md](./03-npm-basics.md) - NPMとパッケージ管理

---

**前のガイド**：[01-what-is-nodejs.md](./01-what-is-nodejs.md)

**親ガイド**：[Node.js Development - SKILL.md](../../SKILL.md)
