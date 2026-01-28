# Node.jsとは - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [Node.jsとは何か](#nodejsとは何か)
3. [なぜNode.jsが人気なのか](#なぜnodejsが人気なのか)
4. [Node.jsのインストール](#nodejsのインストール)
5. [初めてのNode.jsプログラム](#初めてのnodejsプログラム)
6. [REPLの使い方](#replの使い方)
7. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- Node.jsの基本概念
- JavaScriptのサーバーサイド実行
- Node.jsのインストール方法
- 初めてのプログラム実行

### 学習時間：30〜40分

---

## Node.jsとは何か

### 定義

**Node.js**は、JavaScriptをサーバーサイドで実行するためのランタイム環境です。

```
ブラウザ（フロントエンド）
   JavaScript → HTMLを操作

Node.js（バックエンド）
   JavaScript → サーバー、DB、ファイルを操作
```

### 特徴

1. **JavaScriptを使用**
   - フロントエンドと同じ言語
   - 学習コストが低い

2. **非同期I/O**
   - 高速で効率的
   - 同時接続に強い

3. **NPMエコシステム**
   - 100万以上のパッケージ
   - 豊富なライブラリ

---

## なぜNode.jsが人気なのか

### 1. フルスタック開発

```
フロントエンド: JavaScript (React, Vue)
    ↕
バックエンド: JavaScript (Node.js, Express)
    ↕
データベース: JavaScript (MongoDB)
```

**利点**：
- 1つの言語で全て開発
- コードの再利用が容易
- チーム効率向上

### 2. 高パフォーマンス

**イベントループ**により、非ブロッキングI/Oを実現：

```javascript
// 同期（遅い）
const data1 = readFileSync('file1.txt')
const data2 = readFileSync('file2.txt')  // file1が終わるまで待つ

// 非同期（速い）
readFile('file1.txt', (data1) => {})
readFile('file2.txt', (data2) => {})  // 並行実行
```

### 3. 大企業での採用

- **Netflix**：APIサーバー
- **LinkedIn**：バックエンド
- **Uber**：リアルタイムマッチング
- **PayPal**：決済システム

---

## Node.jsのインストール

### macOS

```bash
# Homebrewでインストール（推奨）
brew install node

# バージョン確認
node --version  # v20.10.0
npm --version   # 10.2.3
```

### Windows

```bash
# 公式サイトからインストーラーをダウンロード
# https://nodejs.org/

# インストール後、確認
node --version
npm --version
```

### Linux (Ubuntu)

```bash
# NodeSource経由でインストール
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# 確認
node --version
npm --version
```

### バージョン管理（推奨）

```bash
# nvmでNode.jsのバージョン管理
# macOS/Linux
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash

# Node.js 20をインストール
nvm install 20
nvm use 20
```

---

## 初めてのNode.jsプログラム

### 1. プロジェクトディレクトリ作成

```bash
mkdir hello-node
cd hello-node
```

### 2. JavaScriptファイル作成

`index.js`を作成：

```javascript
// index.js
console.log('Hello, Node.js!')

// 計算
const sum = (a, b) => a + b
console.log('2 + 3 =', sum(2, 3))

// 現在時刻
const now = new Date()
console.log('現在時刻:', now.toLocaleString('ja-JP'))
```

### 3. 実行

```bash
node index.js
```

**出力**：
```
Hello, Node.js!
2 + 3 = 5
現在時刻: 2024/12/24 10:30:00
```

---

## REPLの使い方

### REPLとは

**REPL（Read-Eval-Print Loop）**は、対話的にコードを実行できる環境です。

```bash
# REPLを起動
node

# プロンプトが表示される
>
```

### 使用例

```javascript
> 2 + 3
5

> const name = '太郎'
undefined

> console.log(`こんにちは、${name}さん`)
こんにちは、太郎さん
undefined

> [1, 2, 3].map(x => x * 2)
[ 2, 4, 6 ]

> .exit  // 終了
```

---

## Node.jsでできること

### 1. Webサーバー

```javascript
const http = require('http')

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' })
  res.end('Hello, World!')
})

server.listen(3000, () => {
  console.log('Server running at http://localhost:3000/')
})
```

### 2. ファイル操作

```javascript
const fs = require('fs')

// ファイル読み込み
fs.readFile('data.txt', 'utf8', (err, data) => {
  if (err) throw err
  console.log(data)
})

// ファイル書き込み
fs.writeFile('output.txt', 'Hello, Node.js!', (err) => {
  if (err) throw err
  console.log('ファイルが保存されました')
})
```

### 3. API開発

```javascript
const express = require('express')
const app = express()

app.get('/api/users', (req, res) => {
  res.json({ users: ['太郎', '花子'] })
})

app.listen(3000)
```

---

## よくある質問

### Q1: JavaScriptを知らないとダメ？

**A**: はい、JavaScript の基礎知識が必要です。

**学習順序**：
1. JavaScript基礎（変数、関数、配列）
2. ES6+の機能（アロー関数、async/await）
3. Node.js

### Q2: Node.jsとブラウザのJavaScriptの違いは？

| 項目 | ブラウザ | Node.js |
|------|---------|---------|
| **DOM操作** | ✅ 可能 | ❌ 不可 |
| **ファイルI/O** | ❌ 不可 | ✅ 可能 |
| **モジュール** | ES Modules | CommonJS/ES Modules |
| **グローバル** | window | global |

### Q3: どんなプロジェクトに向いている？

**向いている**：
- REST API
- リアルタイムアプリ（チャット、ゲーム）
- マイクロサービス
- CLIツール

**向いていない**：
- CPU集約的な処理（動画エンコード等）
- 大規模な数値計算

---

## 次のステップ

### このガイドで学んだこと

- ✅ Node.jsの基本概念
- ✅ インストール方法
- ✅ 初めてのプログラム実行
- ✅ REPLの使い方

### 次に学ぶべきガイド

**次のガイド**：[02-javascript-basics.md](./02-javascript-basics.md) - JavaScript基礎

---

**親ガイド**：[Node.js Development - SKILL.md](../../SKILL.md)
