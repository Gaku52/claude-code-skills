# 非同期プログラミング - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [同期と非同期の違い](#同期と非同期の違い)
3. [コールバック](#コールバック)
4. [Promise](#promise)
5. [async/await](#asyncawait)
6. [エラーハンドリング](#エラーハンドリング)
7. [演習問題](#演習問題)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- 同期と非同期の違い
- コールバック関数
- Promise の使い方
- async/await の使い方

### 学習時間：1〜1.5時間

---

## 同期と非同期の違い

### 同期処理（ブロッキング）

```javascript
const fs = require('fs')

console.log('開始')

// 同期：ファイル読み込みが完了するまで待つ
const data = fs.readFileSync('file.txt', 'utf8')
console.log(data)

console.log('終了')
```

**実行順序**：
```
1. 開始
2. （ファイル読み込み完了まで待機）
3. ファイル内容
4. 終了
```

### 非同期処理（ノンブロッキング）

```javascript
const fs = require('fs')

console.log('開始')

// 非同期：待たずに次へ進む
fs.readFile('file.txt', 'utf8', (err, data) => {
  console.log(data)
})

console.log('終了')
```

**実行順序**：
```
1. 開始
2. 終了
3. ファイル内容
```

---

## コールバック

### コールバック関数とは

**コールバック**は、非同期処理の完了後に実行される関数です。

```javascript
// 基本形
setTimeout(() => {
  console.log('2秒後')
}, 2000)

// コールバック引数
fs.readFile('file.txt', 'utf8', (err, data) => {
  if (err) {
    console.error(err)
    return
  }
  console.log(data)
})
```

### コールバック地獄

```javascript
// ❌ コールバック地獄（避けるべき）
fs.readFile('file1.txt', 'utf8', (err, data1) => {
  if (err) return console.error(err)

  fs.readFile('file2.txt', 'utf8', (err, data2) => {
    if (err) return console.error(err)

    fs.readFile('file3.txt', 'utf8', (err, data3) => {
      if (err) return console.error(err)

      console.log(data1, data2, data3)
    })
  })
})
```

---

## Promise

### Promiseとは

**Promise**は、非同期処理の結果を表すオブジェクトです。

**状態**：
- **Pending**：実行中
- **Fulfilled**：成功
- **Rejected**：失敗

### 基本的な使い方

```javascript
// Promiseを返す関数
const readFileAsync = (path) => {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) {
        reject(err)
      } else {
        resolve(data)
      }
    })
  })
}

// 使用
readFileAsync('file.txt')
  .then(data => {
    console.log(data)
  })
  .catch(err => {
    console.error(err)
  })
```

### Promise チェーン

```javascript
// ✅ Promiseチェーン（読みやすい）
readFileAsync('file1.txt')
  .then(data1 => {
    console.log('File 1:', data1)
    return readFileAsync('file2.txt')
  })
  .then(data2 => {
    console.log('File 2:', data2)
    return readFileAsync('file3.txt')
  })
  .then(data3 => {
    console.log('File 3:', data3)
  })
  .catch(err => {
    console.error(err)
  })
```

### Promise.all（並行実行）

```javascript
const promises = [
  readFileAsync('file1.txt'),
  readFileAsync('file2.txt'),
  readFileAsync('file3.txt')
]

Promise.all(promises)
  .then(([data1, data2, data3]) => {
    console.log(data1, data2, data3)
  })
  .catch(err => {
    console.error(err)
  })
```

---

## async/await

### async/awaitとは

**async/await**は、Promiseをより読みやすく書ける構文です。

### 基本的な使い方

```javascript
// asyncを付けた関数はPromiseを返す
async function readFiles() {
  try {
    const data1 = await readFileAsync('file1.txt')
    console.log('File 1:', data1)

    const data2 = await readFileAsync('file2.txt')
    console.log('File 2:', data2)

    const data3 = await readFileAsync('file3.txt')
    console.log('File 3:', data3)
  } catch (err) {
    console.error(err)
  }
}

readFiles()
```

### アロー関数での使用

```javascript
const readFiles = async () => {
  try {
    const data = await readFileAsync('file.txt')
    console.log(data)
  } catch (err) {
    console.error(err)
  }
}

readFiles()
```

### 並行実行

```javascript
// ❌ 直列実行（遅い）
async function sequential() {
  const data1 = await readFileAsync('file1.txt')  // 1秒
  const data2 = await readFileAsync('file2.txt')  // 1秒
  const data3 = await readFileAsync('file3.txt')  // 1秒
  // 合計: 3秒
}

// ✅ 並行実行（速い）
async function parallel() {
  const [data1, data2, data3] = await Promise.all([
    readFileAsync('file1.txt'),
    readFileAsync('file2.txt'),
    readFileAsync('file3.txt')
  ])
  // 合計: 1秒
}
```

---

## エラーハンドリング

### try-catch

```javascript
async function fetchData() {
  try {
    const response = await fetch('https://api.example.com/data')
    const data = await response.json()
    return data
  } catch (error) {
    console.error('Error:', error.message)
    throw error
  }
}
```

### 複数のtry-catch

```javascript
async function process() {
  let data

  try {
    data = await fetchData()
  } catch (error) {
    console.error('Fetch failed:', error)
    return
  }

  try {
    await saveData(data)
  } catch (error) {
    console.error('Save failed:', error)
  }
}
```

---

## 実践例

### Example 1: API リクエスト

```javascript
const fetch = require('node-fetch')

// Promiseを使う方法
function getUserPromise(id) {
  return fetch(`https://api.example.com/users/${id}`)
    .then(res => res.json())
    .catch(err => {
      console.error(err)
      throw err
    })
}

// async/awaitを使う方法（推奨）
async function getUser(id) {
  try {
    const response = await fetch(`https://api.example.com/users/${id}`)

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const user = await response.json()
    return user
  } catch (error) {
    console.error('Failed to fetch user:', error)
    throw error
  }
}

// 使用
getUser(1)
  .then(user => console.log(user))
  .catch(err => console.error(err))
```

### Example 2: 複数のAPI呼び出し

```javascript
async function fetchMultipleUsers() {
  try {
    // 並行実行
    const [user1, user2, user3] = await Promise.all([
      getUser(1),
      getUser(2),
      getUser(3)
    ])

    console.log('Users:', user1, user2, user3)
  } catch (error) {
    console.error('Failed to fetch users:', error)
  }
}

fetchMultipleUsers()
```

---

## Expressでのasync/await

### ルートハンドラー

```javascript
const express = require('express')
const app = express()

// ❌ エラーがキャッチされない
app.get('/users/:id', async (req, res) => {
  const user = await getUser(req.params.id)
  res.json({ user })
})

// ✅ try-catchでエラーハンドリング
app.get('/users/:id', async (req, res, next) => {
  try {
    const user = await getUser(req.params.id)
    res.json({ user })
  } catch (error) {
    next(error)  // エラーハンドラーに渡す
  }
})

// エラーハンドラー
app.use((err, req, res, next) => {
  console.error(err)
  res.status(500).json({ error: 'Internal Server Error' })
})
```

### ラッパー関数

```javascript
// asyncハンドラーのラッパー
const asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next)
}

// 使用
app.get('/users/:id', asyncHandler(async (req, res) => {
  const user = await getUser(req.params.id)
  res.json({ user })
}))
```

---

## よくある間違い

### ❌ 間違い1：awaitを忘れる

```javascript
async function fetchData() {
  const data = fetch('https://api.example.com')  // Promiseが返る
  console.log(data)  // [Promise] と表示される
}
```

**✅ 正しい方法**：

```javascript
async function fetchData() {
  const data = await fetch('https://api.example.com')
  console.log(data)  // 正しいデータ
}
```

### ❌ 間違い2：async関数の外でawaitを使う

```javascript
// エラー：asyncの外でawait
const data = await fetchData()
```

**✅ 正しい方法**：

```javascript
async function main() {
  const data = await fetchData()
}

main()
```

---

## 演習問題

### 問題：遅延実行関数

`sleep`関数を実装してください。

```javascript
// 実装
const sleep = (ms) => {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// 使用例
async function example() {
  console.log('開始')
  await sleep(2000)  // 2秒待つ
  console.log('2秒後')
}

example()
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ 同期と非同期の違い
- ✅ コールバック関数
- ✅ Promise の使い方
- ✅ async/await の使い方

### 次に学ぶべきガイド

**次のガイド**：[06-first-server-tutorial.md](./06-first-server-tutorial.md) - 初めてのサーバー構築

---

**前のガイド**：[04-express-intro.md](./04-express-intro.md)

**親ガイド**：[Node.js Development - SKILL.md](../../SKILL.md)
