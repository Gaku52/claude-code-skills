# NPMとパッケージ管理 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [NPMとは](#npmとは)
3. [package.jsonの作成](#packagejsonの作成)
4. [パッケージのインストール](#パッケージのインストール)
5. [依存関係の管理](#依存関係の管理)
6. [NPMスクリプト](#npmスクリプト)
7. [よく使うパッケージ](#よく使うパッケージ)
8. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- NPMの基本概念
- package.jsonの管理
- パッケージのインストールと削除
- NPMスクリプトの使い方

### 学習時間：40〜50分

---

## NPMとは

### 定義

**NPM（Node Package Manager）**は、Node.jsのパッケージ管理ツールです。

**機能**：
- パッケージのインストール
- 依存関係の管理
- スクリプトの実行
- パッケージの公開

### NPMレジストリ

**npmjs.com**には100万以上のパッケージが登録されています。

```bash
# パッケージ検索
npm search express

# パッケージ情報
npm info express
```

---

## package.jsonの作成

### 初期化

```bash
# 新規プロジェクト作成
mkdir myproject
cd myproject

# package.json作成（対話式）
npm init

# デフォルト設定で作成
npm init -y
```

### package.jsonの構造

```json
{
  "name": "myproject",
  "version": "1.0.0",
  "description": "My awesome project",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": ["nodejs", "express"],
  "author": "Your Name",
  "license": "MIT",
  "dependencies": {
    "express": "^4.18.2"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

---

## パッケージのインストール

### 本番依存関係（dependencies）

```bash
# expressをインストール
npm install express

# 短縮形
npm i express

# 複数インストール
npm i express body-parser cors
```

### 開発依存関係（devDependencies）

```bash
# nodemonを開発依存関係としてインストール
npm install --save-dev nodemon

# 短縮形
npm i -D nodemon
```

### グローバルインストール

```bash
# グローバルにインストール
npm install -g typescript

# 確認
npm list -g --depth=0
```

---

## 依存関係の管理

### package-lock.json

**package-lock.json**は、依存関係の正確なバージョンを記録します。

```bash
# 依存関係をインストール
npm install

# package-lock.jsonも生成される
```

**重要**：
- `package-lock.json`もGitにコミットする
- チーム全体で同じバージョンを使用できる

### node_modules

**node_modules**は、インストールされたパッケージが格納されるディレクトリです。

```bash
# インストール済みパッケージの確認
npm list --depth=0

# node_modulesのサイズ確認
du -sh node_modules
```

**.gitignore**に追加：

```
node_modules/
```

### バージョン指定

```json
{
  "dependencies": {
    "express": "4.18.2",      // 完全一致
    "lodash": "^4.17.21",    // マイナーバージョンまで自動更新
    "axios": "~1.6.0"        // パッチバージョンのみ自動更新
  }
}
```

---

## NPMスクリプト

### スクリプトの定義

```json
{
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest",
    "build": "webpack",
    "lint": "eslint ."
  }
}
```

### スクリプトの実行

```bash
# start / test は省略可能
npm start
npm test

# その他はrunが必要
npm run dev
npm run build
npm run lint
```

### 実践例

```json
{
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js",
    "test": "jest --watch",
    "test:ci": "jest --coverage",
    "lint": "eslint src/**/*.js",
    "lint:fix": "eslint src/**/*.js --fix",
    "format": "prettier --write src/**/*.js"
  }
}
```

---

## よく使うパッケージ

### Web フレームワーク

```bash
# Express - 最も人気のフレームワーク
npm i express

# Fastify - 高速なフレームワーク
npm i fastify

# Koa - 軽量フレームワーク
npm i koa
```

### ユーティリティ

```bash
# lodash - ユーティリティ関数集
npm i lodash

# moment - 日時操作（⚠️ day.js推奨）
npm i dayjs

# dotenv - 環境変数管理
npm i dotenv
```

### 開発ツール

```bash
# nodemon - 自動再起動
npm i -D nodemon

# eslint - コード検証
npm i -D eslint

# prettier - コード整形
npm i -D prettier

# jest - テストフレームワーク
npm i -D jest
```

---

## 実践例

### プロジェクトセットアップ

```bash
# 1. プロジェクト作成
mkdir express-app
cd express-app
npm init -y

# 2. 依存関係インストール
npm i express dotenv
npm i -D nodemon

# 3. ディレクトリ構成
mkdir src
touch src/index.js
touch .env
touch .gitignore
```

### package.json設定

```json
{
  "name": "express-app",
  "version": "1.0.0",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

### .gitignore

```
node_modules/
.env
npm-debug.log
.DS_Store
```

---

## パッケージ管理コマンド

### インストール

```bash
# package.jsonから全てインストール
npm install

# 特定のパッケージをインストール
npm install express

# バージョン指定
npm install express@4.18.2
```

### アンインストール

```bash
# パッケージを削除
npm uninstall express

# 短縮形
npm un express
```

### アップデート

```bash
# 全てのパッケージをアップデート
npm update

# 特定のパッケージをアップデート
npm update express

# 最新バージョンの確認
npm outdated
```

### 確認

```bash
# インストール済みパッケージ
npm list

# グローバルパッケージ
npm list -g --depth=0

# パッケージ情報
npm info express
```

---

## よくある問題と解決方法

### 問題1：依存関係エラー

```bash
# エラー
npm ERR! peer dep missing

# 解決
rm -rf node_modules package-lock.json
npm install
```

### 問題2：権限エラー（EACCES）

```bash
# エラー
npm ERR! EACCES: permission denied

# 解決（macOS/Linux）
sudo chown -R $(whoami) ~/.npm
sudo chown -R $(whoami) /usr/local/lib/node_modules
```

### 問題3：古いパッケージ

```bash
# 古いパッケージを確認
npm outdated

# アップデート
npm update

# メジャーバージョンアップ（注意）
npx npm-check-updates -u
npm install
```

---

## 演習問題

### 問題：プロジェクトセットアップ

以下の要件でプロジェクトをセットアップしてください：
1. `todo-app`プロジェクトを作成
2. `express`と`dotenv`をインストール
3. `nodemon`を開発依存関係としてインストール
4. `start`と`dev`スクリプトを追加

**解答例**：

```bash
# プロジェクト作成
mkdir todo-app
cd todo-app
npm init -y

# 依存関係インストール
npm i express dotenv
npm i -D nodemon

# package.json編集
```

```json
{
  "name": "todo-app",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ NPMの基本概念
- ✅ package.jsonの管理
- ✅ パッケージのインストールと削除
- ✅ NPMスクリプトの使い方

### 次に学ぶべきガイド

**次のガイド**：[04-express-intro.md](./04-express-intro.md) - Express基礎

---

**前のガイド**：[02-javascript-basics.md](./02-javascript-basics.md)

**親ガイド**：[Node.js Development - SKILL.md](../../SKILL.md)
