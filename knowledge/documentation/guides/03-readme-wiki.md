# README & Wiki完全ガイド

> **対象読者**: 全ての開発者、プロジェクトリーダー、オープンソースメンテナー
> **難易度**: 初級〜中級
> **推定読了時間**: 50分

---

## 📋 目次

1. [README & Wikiとは](#readme--wikiとは)
2. [なぜ重要か](#なぜ重要か)
3. [READMEの構成](#readmeの構成)
4. [効果的なREADMEの書き方](#効果的なreadmeの書き方)
5. [バッジとビジュアル](#バッジとビジュアル)
6. [インストール・セットアップ手順](#インストールセットアップ手順)
7. [使用例](#使用例)
8. [Wiki構築](#wiki構築)
9. [多言語対応](#多言語対応)
10. [実践例](#実践例)
11. [メンテナンス](#メンテナンス)
12. [FAQ](#faq)

---

## README & Wikiとは

### 定義

**README** は、プロジェクトの「顔」であり、最初に読まれる最も重要なドキュメントです。
**Wiki** は、READMEでは書ききれない詳細なドキュメント群です。

### 役割の違い

```
📄 README.md
├── プロジェクト概要
├── クイックスタート
├── 基本的な使い方
└── リンク集（Wiki、APIドキュメントなど）

📚 Wiki
├── 詳細なチュートリアル
├── アーキテクチャ設計
├── トラブルシューティング
├── コントリビューションガイド
└── FAQ
```

### 読者

```
👥 読者層:
├── 初めて見る開発者（30秒で興味を持つか判断）
├── 新規参加者（セットアップ手順を知りたい）
├── 既存メンバー（特定機能の使い方を確認）
├── 潜在的コントリビューター（貢献方法を知りたい）
└── 採用担当者（技術力を評価）
```

---

## なぜ重要か

### 1. 第一印象が全て

```
GitHubでプロジェクトを見つけた開発者:

0-10秒: READMEをスクロール
10-30秒: 「使えそうか」判断
30秒-3分: セットアップ手順を確認

→ この30秒で使われるか決まる
```

**実例:**

| README品質 | Star獲得率 | フォーク率 | 採用率 |
|------------|-----------|----------|--------|
| 優れている | 80% | 40% | 60% |
| 普通 | 40% | 15% | 25% |
| 粗末 | 10% | 3% | 5% |

### 2. オンボーディング時間の削減

```
❌ READMEなし:
新メンバー: 「どうやって起動するの?」
→ Slackで質問 → 30分待ち → 試行錯誤で2時間

✅ READMEあり:
新メンバー: README読む → 10分でセットアップ完了

効果: 2時間 → 10分（92%削減）
```

### 3. サポートコストの削減

```
良いREADME:
- FAQ記載 → 同じ質問が90%減る
- トラブルシューティング → サポート時間50%削減
- 使用例豊富 → 「使い方わからない」問い合わせ80%減
```

### 4. 採用・評価への影響

```
採用担当者がGitHubをチェック:

✅ 優れたREADME:
→ 「この人は技術力があり、文書化も得意だ」
→ 採用確率UP

❌ 粗末なREADME:
→ 「技術力はあるかもしれないが、伝える力がない」
→ 評価DOWN
```

---

## READMEの構成

### 黄金の構成（推奨テンプレート）

```markdown
# プロジェクト名

> 1行で説明（何をするツールか）

[バッジ群]

## 特徴

- 主要機能1
- 主要機能2
- 主要機能3

## デモ

[GIF/スクリーンショット]

## インストール

```bash
npm install your-package
```

## クイックスタート

```javascript
// 最小限のコード例
```

## ドキュメント

- [詳細ドキュメント](link)
- [API仕様](link)
- [チュートリアル](link)

## コントリビューション

[CONTRIBUTING.md](CONTRIBUTING.md)を参照

## ライセンス

MIT License
```

### セクション別解説

#### 1. プロジェクト名

```markdown
✅ Good:
# React Query

❌ Bad:
# My Project
```

**ポイント:**
- わかりやすい名前
- ブランディングを意識
- 検索しやすい名前

#### 2. 1行説明

```markdown
✅ Good:
> Powerful asynchronous state management for React

❌ Bad:
> A library for React
```

**ポイント:**
- 30文字以内
- 何をするツールか明確
- 誰が使うべきか示す

#### 3. バッジ

```markdown
[![npm version](https://badge.fury.io/js/react-query.svg)](https://www.npmjs.com/package/react-query)
[![Build Status](https://travis-ci.org/tanstack/react-query.svg?branch=main)](https://travis-ci.org/tanstack/react-query)
[![codecov](https://codecov.io/gh/tanstack/react-query/branch/main/graph/badge.svg)](https://codecov.io/gh/tanstack/react-query)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

**効果:**
- 一目で品質がわかる
- 信頼感を与える
- アクティブさを示す

#### 4. 特徴

```markdown
✅ Good:
## 特徴

- 🚀 **高速**: 独自キャッシュ戦略で10倍高速
- 📦 **軽量**: わずか5KB（gzip圧縮後）
- 🎨 **柔軟**: あらゆるUIフレームワークに対応

❌ Bad:
## 特徴

- 良い
- 速い
- 簡単
```

**ポイント:**
- 絵文字で視覚的に
- 具体的な数字
- 競合との差別化

#### 5. デモ

```markdown
✅ Good:
## デモ

![Demo](docs/demo.gif)

[Live Demo](https://demo.example.com)

❌ Bad:
## デモ

デモはこちら
```

**ポイント:**
- GIF/動画が最も効果的
- 静止画よりも動きが伝わる
- CodeSandbox/StackBlitzへのリンク

---

## 効果的なREADMEの書き方

### 1. 読者を意識する

```markdown
❌ 専門用語だらけ:
This library implements a reactive data-fetching paradigm
with automatic request deduplication and intelligent
cache invalidation strategies.

✅ 初心者にも分かりやすい:
このライブラリは、APIからデータを取得するコードを
シンプルにします。自動的にキャッシュし、無駄な
リクエストを防ぎます。

```javascript
// Before
const [data, setData] = useState(null);
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);

useEffect(() => {
  fetch('/api/users')
    .then(res => res.json())
    .then(data => {
      setData(data);
      setLoading(false);
    })
    .catch(err => {
      setError(err);
      setLoading(false);
    });
}, []);

// After
const { data, isLoading, error } = useQuery('users', fetchUsers);
```
```

### 2. 視覚的な要素を活用

```markdown
✅ Good:
## インストール

```bash
npm install awesome-lib
```

## 使い方

```javascript
import { create } from 'awesome-lib';

const app = create({
  name: 'MyApp',
  version: '1.0.0'
});
```

## 比較

| 機能 | awesome-lib | 競合A | 競合B |
|------|-------------|-------|-------|
| 速度 | ⚡ 超高速 | 🐢 普通 | 🐌 遅い |
| サイズ | 📦 5KB | 📦 50KB | 📦 100KB |
| 学習曲線 | 📈 緩やか | 📈 急 | 📈 急 |

❌ Bad:
インストール方法: npm installを使ってください
使い方: importして使ってください
```

### 3. 段階的な情報提供

```markdown
✅ Good:

# Quick Start（30秒）

```bash
npx create-awesome-app my-app
cd my-app
npm start
```

# 詳細セットアップ（5分）

詳しい手順は [Installation Guide](docs/installation.md) を参照

❌ Bad:

# インストール

まずNode.jsをインストールして...
次にnpmのバージョンを確認して...
（延々と続く）
```

**原則:**
1. 最短経路を最初に示す
2. 詳細は別ページへリンク
3. 段階的に深掘りできる構造

### 4. コード例は最小限かつ実用的

```markdown
✅ Good:

```javascript
// 最小限の実用例
import { useAuth } from 'auth-lib';

function LoginButton() {
  const { login, user } = useAuth();

  return user ? (
    <p>Welcome, {user.name}!</p>
  ) : (
    <button onClick={login}>Login</button>
  );
}
```

❌ Bad:

```javascript
// 抽象的すぎる例
import { doSomething } from 'lib';

doSomething(); // これで完了！
```

❌ Bad:

```javascript
// 複雑すぎる例（100行以上）
import { ... } from '...';
// 延々とコードが続く
```
```

**ポイント:**
- 10-20行が理想
- コピペして動く
- 実用的なユースケース

---

## バッジとビジュアル

### バッジの種類

#### 1. ビルド状態

```markdown
[![CI](https://github.com/owner/repo/actions/workflows/ci.yml/badge.svg)](https://github.com/owner/repo/actions/workflows/ci.yml)
```

**メリット:**
- プロジェクトの健全性を示す
- CIが通っていることを保証
- 信頼性の証明

#### 2. カバレッジ

```markdown
[![codecov](https://codecov.io/gh/owner/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/owner/repo)
```

**メリット:**
- テストの充実度を示す
- 品質へのこだわりを示す

#### 3. バージョン

```markdown
[![npm version](https://badge.fury.io/js/package.svg)](https://www.npmjs.com/package/package)
[![GitHub release](https://img.shields.io/github/release/owner/repo.svg)](https://github.com/owner/repo/releases)
```

**メリット:**
- 最新バージョンが一目で分かる
- アクティブに開発されていることを示す

#### 4. ダウンロード数

```markdown
[![npm downloads](https://img.shields.io/npm/dm/package.svg)](https://www.npmjs.com/package/package)
```

**メリット:**
- 人気度を示す
- 信頼性の指標

#### 5. ライセンス

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

**メリット:**
- 利用条件が明確
- 法的リスクの回避

### バッジの配置

```markdown
✅ Good（見やすい）:
# Project Name

[![Build](...)][...]
[![Coverage](...)][...]
[![npm](...)][...]
[![License](...)][...]

❌ Bad（多すぎ）:
# Project Name

[![1](...)][...] [![2](...)][...] [![3](...)][...] [![4](...)][...]
[![5](...)][...] [![6](...)][...] [![7](...)][...] [![8](...)][...]
[![9](...)][...] [![10](...)][...] [![11](...)][...] [![12](...)][...]
```

**推奨:**
- 最大5-6個
- 重要なものだけ
- 1行に収める

### デモGIFの作り方

**ツール:**
- [LICEcap](https://www.cockos.com/licecap/) - Windows/Mac
- [Kap](https://getkap.co/) - Mac専用
- [Peek](https://github.com/phw/peek) - Linux

**ベストプラクティス:**

```
✅ 良いデモGIF:
- 10-30秒
- 主要機能を1つ見せる
- テキストが読める解像度
- 60fps以下（ファイルサイズ小）

❌ 悪いデモGIF:
- 1分以上（ファイルサイズ大）
- 複数機能を詰め込む
- 小さすぎて読めない
- カーソル動きが速すぎる
```

**例:**

```markdown
## デモ

### インストールから起動まで（20秒）

![Quick Start](docs/quick-start.gif)

### 主要機能

![Feature Demo](docs/feature-demo.gif)
```

---

## インストール・セットアップ手順

### 原則: 3ステップ以内

```markdown
✅ Good:

## クイックスタート

```bash
# 1. インストール
npm install awesome-lib

# 2. プロジェクト作成
npx create-awesome-app my-app

# 3. 起動
cd my-app && npm start
```

🎉 http://localhost:3000 で確認！

❌ Bad:

## インストール

まず以下をインストールしてください:
1. Node.js v16以上
2. npm v8以上
3. Git
4. Docker
5. ...（延々と続く）
```

### 前提条件の明記

```markdown
✅ Good:

## 前提条件

- Node.js 16.x 以上
- npm 8.x 以上

<details>
<summary>Node.jsのインストール方法</summary>

### macOS
```bash
brew install node
```

### Windows
[Node.js公式サイト](https://nodejs.org/)からインストーラーをダウンロード

### Linux
```bash
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
```
</details>
```

### 環境別手順

```markdown
## インストール

### npm

```bash
npm install awesome-lib
```

### yarn

```bash
yarn add awesome-lib
```

### pnpm

```bash
pnpm add awesome-lib
```

### CDN

```html
<script src="https://cdn.jsdelivr.net/npm/awesome-lib@1.0.0/dist/index.min.js"></script>
```
```

### トラブルシューティング

```markdown
## よくあるエラー

### `EACCES: permission denied`

**原因:** npm のグローバルインストール権限がない

**解決策:**

```bash
# 方法1: nvm使用（推奨）
nvm install 16

# 方法2: sudoを使わない設定
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### `Module not found: 'awesome-lib'`

**原因:** node_modulesが正しくインストールされていない

**解決策:**

```bash
rm -rf node_modules package-lock.json
npm install
```
```

---

## 使用例

### 基本例

```markdown
## 基本的な使い方

```javascript
import { create } from 'awesome-lib';

// 1. インスタンス作成
const app = create({
  name: 'MyApp'
});

// 2. 機能を使う
app.start();

// 3. 結果を取得
console.log(app.status); // "running"
```
```

### 段階的な例

```markdown
## 使用例

### レベル1: Hello World（30秒）

```javascript
import { greet } from 'awesome-lib';

console.log(greet('World')); // "Hello, World!"
```

### レベル2: カスタマイズ（2分）

```javascript
import { greet } from 'awesome-lib';

const customGreet = greet('World', {
  prefix: 'Hi',
  suffix: '!',
  capitalize: true
});

console.log(customGreet); // "Hi, WORLD!"
```

### レベル3: 実践的な使い方（5分）

```javascript
import { create, middleware } from 'awesome-lib';

const app = create({
  name: 'MyApp',
  plugins: [
    middleware.logger(),
    middleware.cors()
  ]
});

app.on('start', () => {
  console.log('App started!');
});

app.start();
```

詳細は [Advanced Usage](docs/advanced.md) を参照
```

### ユースケース別

```markdown
## ユースケース

### ユースケース1: REST API

```javascript
import { api } from 'awesome-lib';

const users = await api.get('/users');
console.log(users);
```

### ユースケース2: GraphQL

```javascript
import { graphql } from 'awesome-lib';

const result = await graphql(`
  query {
    users {
      id
      name
    }
  }
`);
```

### ユースケース3: WebSocket

```javascript
import { ws } from 'awesome-lib';

const socket = ws.connect('wss://example.com');
socket.on('message', (data) => {
  console.log(data);
});
```
```

---

## Wiki構築

### いつWikiが必要か

```
README: 1ページで完結（5,000文字以下）
Wiki: 複数ページで詳細解説（総計20,000文字以上）

Wiki作成の判断基準:
✅ プロジェクトが成熟している
✅ 複数の機能がある
✅ チュートリアルが複数必要
✅ アーキテクチャ説明が複雑
✅ コントリビューター向けガイドが必要
```

### Wiki構成例

```
📚 Wiki
├── Home.md（目次）
├── Getting Started/
│   ├── Installation.md
│   ├── Quick Start.md
│   └── First App.md
├── Guides/
│   ├── Authentication.md
│   ├── Database.md
│   └── Deployment.md
├── API Reference/
│   ├── Core API.md
│   ├── Plugins.md
│   └── Middleware.md
├── Architecture/
│   ├── Overview.md
│   ├── Design Decisions.md
│   └── Performance.md
└── Contributing/
    ├── Code Style.md
    ├── Testing.md
    └── Pull Request Process.md
```

### Homeページ（目次）

```markdown
# Welcome to Awesome Lib Wiki

## 📚 ドキュメント

### はじめに
- [インストール](Getting-Started/Installation)
- [クイックスタート](Getting-Started/Quick-Start)
- [最初のアプリ](Getting-Started/First-App)

### ガイド
- [認証](Guides/Authentication)
- [データベース](Guides/Database)
- [デプロイ](Guides/Deployment)

### APIリファレンス
- [Core API](API-Reference/Core-API)
- [Plugins](API-Reference/Plugins)
- [Middleware](API-Reference/Middleware)

### アーキテクチャ
- [概要](Architecture/Overview)
- [設計判断](Architecture/Design-Decisions)
- [パフォーマンス](Architecture/Performance)

### コントリビューション
- [コードスタイル](Contributing/Code-Style)
- [テスト](Contributing/Testing)
- [PR プロセス](Contributing/Pull-Request-Process)

## 🔗 リンク

- [GitHub Repository](https://github.com/owner/repo)
- [NPM Package](https://www.npmjs.com/package/awesome-lib)
- [Issue Tracker](https://github.com/owner/repo/issues)
```

### ページテンプレート

```markdown
# ページタイトル

> **対象読者**: 初級者、中級者、上級者
> **推定時間**: 10分
> **前提知識**: JavaScript基礎

---

## 概要

このページでは〇〇について説明します。

## 目次

1. [セクション1](#セクション1)
2. [セクション2](#セクション2)
3. [セクション3](#セクション3)

---

## セクション1

### 説明

...

### コード例

```javascript
// コード例
```

### 注意点

⚠️ 注意事項

---

## 次のステップ

- [次のガイド](link)
- [関連ドキュメント](link)
```

### GitHub vs 独自Wiki

| 観点 | GitHub Wiki | GitBook / Docusaurus |
|------|-------------|----------------------|
| 簡単さ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| カスタマイズ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| バージョン管理 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 検索機能 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| コスト | 無料 | 無料〜有料 |
| ホスティング | GitHub | 独自ドメイン可 |

**推奨:**
- 小規模プロジェクト: GitHub Wiki
- 中〜大規模: Docusaurus / VitePress
- 商用プロダクト: GitBook / ReadTheDocs

---

## 多言語対応

### いつ多言語化するか

```
判断基準:
✅ グローバルなユーザーがいる
✅ 英語圏以外のコントリビューターがいる
✅ プロジェクトが成熟している
✅ メンテナンスリソースがある

優先順位:
1. 英語（必須）
2. 中国語（ユーザー数多い）
3. 日本語、スペイン語、フランス語
```

### ディレクトリ構造

```
docs/
├── README.md（英語版）
├── README.ja.md（日本語版）
├── README.zh.md（中国語版）
└── README.es.md（スペイン語版）
```

### README内にリンク

```markdown
# Awesome Lib

> [日本語](README.ja.md) | [中文](README.zh.md) | [Español](README.es.md)

...
```

### メンテナンス戦略

```markdown
## 翻訳メンテナンス

**方針:**
1. 英語版がマスター
2. 翻訳版は定期的に同期
3. コミュニティに翻訳依頼

**翻訳の同期:**

```bash
# 英語版更新時に翻訳版をマーク
echo "⚠️ This translation is outdated. See [English version](README.md)" >> README.ja.md
```

**コントリビューター募集:**

```markdown
## Contributing

We welcome translations! See [Translation Guide](TRANSLATION.md).

Current translators:
- 🇯🇵 Japanese: @username
- 🇨🇳 Chinese: @username
```
```

---

## 実践例

### 例1: ライブラリREADME（React Query風）

```markdown
# TanStack Query

> Powerful asynchronous state management for TS/JS, React, Solid, Vue, Svelte and Angular

[![npm version](https://badge.fury.io/js/%40tanstack%2Freact-query.svg)](https://www.npmjs.com/package/@tanstack/react-query)
[![Build Status](https://github.com/TanStack/query/actions/workflows/ci.yml/badge.svg)](https://github.com/TanStack/query/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TanStack/query/branch/main/graph/badge.svg)](https://codecov.io/gh/TanStack/query)

## Features

- 🚀 **Backend agnostic** - Works with REST, GraphQL, or any async data source
- 📦 **Tiny** - Only ~13kb gzipped
- 🔄 **Auto Caching** - Intelligent caching and automatic refetching
- 🎯 **Type Safe** - Full TypeScript support

## Quick Start

```bash
npm install @tanstack/react-query
```

```tsx
import { useQuery } from '@tanstack/react-query';

function Users() {
  const { data, isLoading } = useQuery({
    queryKey: ['users'],
    queryFn: () => fetch('/api/users').then(r => r.json())
  });

  if (isLoading) return <div>Loading...</div>;

  return (
    <ul>
      {data.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

## Documentation

- [Official Documentation](https://tanstack.com/query/latest)
- [Quick Start Guide](https://tanstack.com/query/latest/docs/quick-start)
- [API Reference](https://tanstack.com/query/latest/docs/reference)

## Community

- [Discord](https://discord.gg/tanstack)
- [Twitter](https://twitter.com/tanstack)
- [GitHub Discussions](https://github.com/TanStack/query/discussions)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT © [TanStack](https://github.com/TanStack)
```

### 例2: CLIツールREADME（Vite風）

```markdown
# Vite ⚡

> Next Generation Frontend Tooling

[![npm version](https://badge.fury.io/js/vite.svg)](https://www.npmjs.com/package/vite)
[![node compatibility](https://img.shields.io/node/v/vite.svg)](https://nodejs.org/)

## Why Vite?

- 💡 **Instant Server Start** - No bundling required
- ⚡ **Lightning Fast HMR** - Stay fast regardless of app size
- 🛠️ **Rich Features** - TypeScript, JSX, CSS support out of the box
- 📦 **Optimized Build** - Pre-configured Rollup build

## Getting Started

### Scaffolding Your First Vite Project

```bash
npm create vite@latest my-app
cd my-app
npm install
npm run dev
```

### Manual Installation

```bash
npm install -D vite
```

```json
{
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  }
}
```

## Supported Templates

| Template | Command |
|----------|---------|
| Vanilla | `npm create vite@latest my-app -- --template vanilla` |
| Vue | `npm create vite@latest my-app -- --template vue` |
| React | `npm create vite@latest my-app -- --template react` |
| Svelte | `npm create vite@latest my-app -- --template svelte` |

## Documentation

📖 [vitejs.dev](https://vitejs.dev)

## Community

- [Discord](https://discord.gg/vite)
- [Twitter](https://twitter.com/vite_js)
- [Awesome Vite](https://github.com/vitejs/awesome-vite)

## Sponsors

<p align="center">
  <a href="https://github.com/sponsors/yyx990803">
    <img src="https://sponsors.vuejs.org/vite.svg" alt="Sponsors">
  </a>
</p>

## License

MIT
```

### 例3: フルスタックアプリREADME

```markdown
# E-Commerce Platform

> Modern e-commerce platform built with Next.js, TypeScript, and Prisma

![Demo](docs/demo.gif)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/owner/repo)

## Features

### Customer Features
- 🛒 Shopping cart with local storage
- 💳 Stripe payment integration
- 📧 Email notifications
- 📱 Responsive design

### Admin Features
- 📊 Sales dashboard
- 📦 Product management
- 👥 Customer management
- 📈 Analytics

## Tech Stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS
- **Backend:** Next.js API Routes, tRPC
- **Database:** PostgreSQL, Prisma ORM
- **Auth:** NextAuth.js
- **Payment:** Stripe
- **Hosting:** Vercel

## Getting Started

### Prerequisites

- Node.js 18+
- PostgreSQL 14+
- Stripe account

### Installation

1. Clone the repository

```bash
git clone https://github.com/owner/repo.git
cd repo
```

2. Install dependencies

```bash
npm install
```

3. Set up environment variables

```bash
cp .env.example .env.local
```

Edit `.env.local`:

```env
DATABASE_URL="postgresql://user:password@localhost:5432/ecommerce"
NEXTAUTH_SECRET="your-secret"
STRIPE_SECRET_KEY="sk_test_..."
```

4. Set up database

```bash
npx prisma migrate dev
npx prisma db seed
```

5. Start development server

```bash
npm run dev
```

🎉 Open http://localhost:3000

### Test Accounts

```
Admin:
Email: admin@example.com
Password: admin123

Customer:
Email: customer@example.com
Password: customer123
```

## Deployment

### Vercel (Recommended)

1. Click the deploy button above
2. Add environment variables
3. Deploy

### Docker

```bash
docker-compose up -d
```

## Documentation

- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Contributing](CONTRIBUTING.md)

## Roadmap

- [x] Basic product catalog
- [x] Shopping cart
- [x] Stripe integration
- [ ] Multi-currency support
- [ ] Inventory management
- [ ] Reviews & ratings

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT © [Your Name](https://github.com/owner)
```

---

## メンテナンス

### READMEの更新タイミング

```
✅ 必ず更新:
- 新機能追加時
- Breaking change時
- インストール手順変更時
- 依存関係のメジャーアップデート時

✅ 推奨更新:
- リリース時（バッジのバージョン）
- スクリーンショット古くなった時
- 外部リンク切れ時

❌ 更新不要:
- バグ修正（内部実装）
- マイナーな変更
```

### 更新チェックリスト

```markdown
## README更新チェックリスト

- [ ] バージョン番号は正しいか
- [ ] インストール手順は最新か
- [ ] コード例は動作するか
- [ ] リンクは切れていないか
- [ ] バッジは正しいか
- [ ] スクリーンショットは最新か
- [ ] Breaking changeは明記されているか
- [ ] マイグレーションガイドはあるか
```

### 自動化

#### リンクチェック

```yaml
# .github/workflows/check-links.yml
name: Check Links

on:
  schedule:
    - cron: '0 0 * * 0' # 毎週日曜日
  pull_request:
    paths:
      - '**.md'

jobs:
  check-links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: gaurav-nelson/github-action-markdown-link-check@v1
        with:
          use-quiet-mode: 'yes'
```

#### バージョン同期

```json
// package.json
{
  "version": "1.0.0",
  "scripts": {
    "version": "npm run update-readme",
    "update-readme": "sed -i 's/version-[0-9.]\\+-blue/version-'\"$npm_package_version\"'-blue/' README.md"
  }
}
```

#### コード例のテスト

```javascript
// scripts/test-readme-examples.js
const fs = require('fs');
const { exec } = require('child_process');

// README.mdからコードブロックを抽出
const readme = fs.readFileSync('README.md', 'utf-8');
const codeBlocks = readme.match(/```javascript\n([\s\S]*?)\n```/g);

// 各コードブロックを実行
codeBlocks.forEach((block, index) => {
  const code = block.replace(/```javascript\n/, '').replace(/\n```/, '');
  fs.writeFileSync(`test-example-${index}.js`, code);

  exec(`node test-example-${index}.js`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Example ${index} failed:`, error);
      process.exit(1);
    }
  });
});
```

---

## FAQ

### Q1: READMEはどのくらいの長さが適切？

**A:** 原則として **1,000〜3,000文字** が理想。

```
短すぎ（< 500文字）:
❌ 情報不足
❌ 信頼感がない

適切（1,000〜3,000文字）:
✅ 必要な情報がある
✅ スクロールしても読める

長すぎ（> 5,000文字）:
❌ 読まれない
❌ メンテナンスが大変
→ Wikiに分割すべき
```

### Q2: バッジは何個まで？

**A:** **4〜6個** が理想。

```
✅ 推奨:
- ビルド状態
- カバレッジ
- npm バージョン
- ライセンス
- ダウンロード数

❌ 不要:
- 過剰なバッジ（10個以上）
- 意味のないバッジ
- 自作の謎バッジ
```

### Q3: デモGIFのサイズは？

**A:** **2MB以下、10〜30秒** が理想。

```bash
# GIFを最適化
gifsicle -O3 --colors 128 demo.gif -o demo-optimized.gif

# さらに圧縮
ffmpeg -i demo.gif -vf "fps=10,scale=800:-1:flags=lanczos" -c:v gif demo-compressed.gif
```

### Q4: コードブロックの言語指定は必須？

**A:** はい、必須です。

```markdown
❌ Bad:
```
const foo = 'bar';
```

✅ Good:
```javascript
const foo = 'bar';
```
```

**理由:**
- シンタックスハイライトが効く
- 読みやすい
- スクリーンリーダー対応

### Q5: 絵文字は使うべき？

**A:** 適度に使う。

```markdown
✅ Good（見出しやリスト）:
## 🚀 Features

- ⚡ Fast
- 📦 Lightweight

❌ Bad（文中に多用）:
This library is 🔥 awesome 🎉 and super 💪 powerful 🚀!
```

**ガイドライン:**
- 見出しやリストのアクセント: OK
- 文中の多用: NG
- アイコン代わり: OK
- 感情表現: 控えめに

### Q6: README vs Wiki vs Docs、何が違う？

| ドキュメント | 目的 | 長さ | 対象 |
|--------------|------|------|------|
| README | プロジェクト概要、クイックスタート | 1-3ページ | 初見の人 |
| Wiki | 詳細ガイド、チュートリアル | 5-50ページ | 実装する人 |
| Docs | 完全なリファレンス | 50-500ページ | 全ユーザー |

**使い分け:**

```
小規模プロジェクト:
→ READMEのみ

中規模プロジェクト:
→ README + GitHub Wiki

大規模プロジェクト:
→ README + Docusaurus/GitBook
```

### Q7: 英語が苦手でも英語READMEは必要？

**A:** はい、必須です。

**理由:**
- GitHub ユーザーの80%以上が英語を読める
- 日本語のみだと世界に届かない
- Google翻訳でも良いので英語版を用意

**戦略:**

```
1. 日本語で書く
2. DeepL/ChatGPTで翻訳
3. Grammarly で校正
4. ネイティブに簡単なレビュー依頼（オプション）
```

### Q8: README を自動生成できる？

**A:** 一部は可能。

**ツール:**

```bash
# JSDoc から README 生成
npm install -g jsdoc-to-markdown
jsdoc2md src/**/*.js > docs/API.md

# TypeScript から API ドキュメント生成
npm install -g typedoc
typedoc --out docs src

# package.json から README 生成
npm install -g readme-md-generator
npx readme-md-generator
```

**注意:**
- 自動生成は補助的に
- 説明やデモは手動で書く
- メンテナンスは人間が行う

### Q9: 複数プロジェクトで README テンプレートを共有したい

**A:** GitHub Template Repository を使う。

```bash
# テンプレートリポジトリを作成
mkdir readme-template
cd readme-template

# README テンプレート
cat > README.md <<EOF
# {{ PROJECT_NAME }}

> {{ PROJECT_DESCRIPTION }}

## Features

- Feature 1
- Feature 2

## Installation

\`\`\`bash
npm install {{ PACKAGE_NAME }}
\`\`\`
EOF

# GitHub でテンプレートリポジトリに設定
# Settings → Template repository にチェック
```

**使い方:**

```bash
# テンプレートから新規プロジェクト作成
gh repo create my-project --template owner/readme-template

# プレースホルダーを置換
sed -i 's/{{ PROJECT_NAME }}/My Project/' README.md
sed -i 's/{{ PROJECT_DESCRIPTION }}/Awesome project/' README.md
```

### Q10: README が古くなっているかどうか判断する方法は？

**A:** チェックリストで定期確認。

```markdown
## README 鮮度チェック（月次）

- [ ] 最終更新日は3ヶ月以内か？
- [ ] インストール手順は正しいか？
- [ ] コード例は動作するか？
- [ ] バッジは正しいバージョンか？
- [ ] リンクは切れていないか？
- [ ] スクリーンショットは最新UIか？
- [ ] 新機能は反映されているか？

更新が必要な場合:
1. Issue作成（ラベル: documentation）
2. 更新PR作成
3. CI でコード例テスト
```

**自動化:**

```yaml
# .github/workflows/readme-check.yml
name: README Freshness Check

on:
  schedule:
    - cron: '0 0 1 * *' # 毎月1日

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check last update
        run: |
          LAST_UPDATE=$(git log -1 --format=%cd --date=short README.md)
          DAYS_AGO=$(( ($(date +%s) - $(date -d "$LAST_UPDATE" +%s)) / 86400 ))
          if [ $DAYS_AGO -gt 90 ]; then
            echo "::warning::README hasn't been updated in $DAYS_AGO days"
          fi
```

---

## まとめ

### READMEの黄金律

```
1️⃣ 30秒で興味を持たせる
2️⃣ 3分でセットアップ完了
3️⃣ 5分で基本機能を理解
4️⃣ 詳細はWikiやDocsへ誘導
5️⃣ 定期的にメンテナンス
```

### 必須要素チェックリスト

```markdown
✅ README 必須要素:
- [ ] プロジェクト名
- [ ] 1行説明
- [ ] バッジ（3-5個）
- [ ] 特徴（3-5個）
- [ ] インストール手順
- [ ] 最小限のコード例
- [ ] ドキュメントリンク
- [ ] ライセンス

✅ オプション（推奨）:
- [ ] デモGIF
- [ ] ロゴ
- [ ] コントリビューションガイド
- [ ] Changelog
- [ ] FAQ

✅ 高度（大規模プロジェクト）:
- [ ] 多言語対応
- [ ] スポンサー情報
- [ ] ロードマップ
- [ ] コミュニティリンク
```

### 最後に

```
良いREADMEは:
- プロジェクトの顔
- 開発者の名刺
- チームの生産性を上げる資産

投資する価値は十分にあります 🚀
```

---

## 次のステップ

1. **自分のプロジェクトのREADMEを見直す**
   - 上記チェックリストで確認
   - 不足している要素を追加

2. **Wikiを構築する**
   - GitHub Wikiを有効化
   - 詳細ガイドを作成

3. **定期メンテナンスを設定**
   - GitHub Actionsで自動チェック
   - 月次レビューをカレンダーに追加

4. **関連ガイドを読む**
   - [Technical Writing](01-technical-writing.md)
   - [API Documentation](02-api-documentation.md)

---

**📚 このガイドは [documentation](../SKILL.md) スキルの一部です。**
