# React開発環境のセットアップ - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [Node.jsのインストール](#nodejsのインストール)
4. [最初のReactプロジェクトを作成](#最初のreactプロジェクトを作成)
5. [プロジェクト構造の理解](#プロジェクト構造の理解)
6. [開発サーバーの起動と動作確認](#開発サーバーの起動と動作確認)
7. [最初のコンポーネント編集](#最初のコンポーネント編集)
8. [よくあるトラブルと解決策](#よくあるトラブルと解決策)
9. [演習問題](#演習問題)
10. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Node.jsとnpm/pnpmのインストール
- Viteを使った最新のReactプロジェクト作成
- プロジェクトの基本構造
- 開発サーバーの起動と動作確認
- 最初のコンポーネント編集

### なぜ重要か

React開発を始めるには、適切な開発環境が必要です。2024年現在、以下の理由でVite + Reactの組み合わせが推奨されています：

- **高速な起動**：Create React App（CRA）の10倍以上速い
- **高速なHMR**：コード変更が即座に反映（Hot Module Replacement）
- **モダンな構成**：ES Modules、TypeScript、最新のビルドツール
- **公式推奨**：React公式ドキュメントで推奨

### 学習時間の目安

- このガイドの読了：20〜30分
- 実際の環境構築：20〜40分（ダウンロード速度に依存）
- 演習問題を含めた完全理解：1時間

---

## 前提知識

### 必要な知識

1. **コマンドライン（ターミナル）の基礎**：
   - ディレクトリの移動（`cd`）
   - ファイル一覧の表示（`ls` / `dir`）
   - 基本的なコマンド実行

2. **テキストエディタの使用経験**：
   - VS Code、Sublime Text、Atomなど

### 推奨環境

- **OS**：Windows 10/11、macOS 10.15以降、またはLinux
- **メモリ**：最低4GB（8GB以上推奨）
- **ストレージ**：最低2GBの空き容量

---

## Node.jsのインストール

### Node.jsとは

**Node.js**は、JavaScriptをブラウザ外（サーバーやローカル環境）で実行するためのランタイムです。React開発には以下の理由で必要です：

- **npm/pnpm**：パッケージマネージャー（ライブラリ管理ツール）
- **ビルドツール**：ViteやWebpackなどの実行
- **開発サーバー**：ローカルでReactアプリを実行

### インストール手順

#### Windows / macOS / Linux 共通

1. **公式サイトにアクセス**
   - https://nodejs.org/

2. **バージョンを選択**
   - **LTS（推奨）**：Long Term Support（長期サポート版）
   - 2024年1月現在：Node.js 20.x LTS

3. **インストーラーをダウンロード**
   - Windowsの場合：`.msi`ファイル
   - macOSの場合：`.pkg`ファイル
   - Linuxの場合：パッケージマネージャーを使用

4. **インストーラーを実行**
   - デフォルト設定のまま「次へ」をクリック
   - 「Add to PATH」にチェックが入っていることを確認

5. **インストール確認**

```bash
# Node.jsのバージョン確認
node --version
# 出力例：v20.10.0

# npmのバージョン確認
npm --version
# 出力例：10.2.3
```

### macOSの場合：Homebrewを使ったインストール（オプション）

```bash
# Homebrewがインストールされている場合
brew install node

# バージョン確認
node --version
npm --version
```

### pnpmのインストール（推奨）

**pnpm**は、npmより高速で効率的なパッケージマネージャーです。

```bash
# npmを使ってpnpmをインストール
npm install -g pnpm

# バージョン確認
pnpm --version
# 出力例：8.15.0
```

**pnpmの利点**：
- **高速**：npmの2〜3倍速い
- **省スペース**：ディスク使用量が少ない（シンボリックリンクを使用）
- **厳格な依存関係管理**：予期しないバグを防ぐ

---

## 最初のReactプロジェクトを作成

### Viteを使ったプロジェクト作成

Viteは、モダンで高速なビルドツールです。

#### ステップ1：プロジェクトを作成

```bash
# プロジェクトを作成（pnpmの場合）
pnpm create vite my-react-app --template react-ts

# npmの場合
npm create vite@latest my-react-app -- --template react-ts

# プロジェクト名の説明
# my-react-app：好きな名前に変更可能
# --template react-ts：React + TypeScript テンプレート
```

**所要時間**：約10〜30秒

#### ステップ2：プロジェクトディレクトリに移動

```bash
cd my-react-app
```

#### ステップ3：依存関係をインストール

```bash
# pnpmの場合
pnpm install

# npmの場合
npm install
```

**所要時間**：約1〜3分（ネット環境に依存）

#### 実行結果

```
added 212 packages, and audited 213 packages in 45s

52 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities
```

---

## プロジェクト構造の理解

作成されたプロジェクトの構造を確認しましょう。

```bash
# ファイルツリーを表示
tree -L 2 my-react-app
# (treeコマンドがない場合は、エディタで確認)
```

### ディレクトリ構造

```
my-react-app/
├── node_modules/       # インストールされたライブラリ（触らない）
├── public/             # 静的ファイル（画像、favicon など）
│   └── vite.svg
├── src/                # ソースコード（メインの作業場所）
│   ├── assets/         # 画像、CSS などの静的リソース
│   ├── App.tsx         # メインのAppコンポーネント
│   ├── App.css         # Appコンポーネントのスタイル
│   ├── main.tsx        # エントリーポイント（アプリの起動ファイル）
│   └── index.css       # グローバルスタイル
├── index.html          # HTMLテンプレート
├── package.json        # プロジェクト設定・依存関係
├── tsconfig.json       # TypeScript設定
├── vite.config.ts      # Vite設定
└── README.md           # プロジェクトの説明
```

### 重要なファイルの役割

#### 1. `package.json`

プロジェクトの設定ファイルです。

```json
{
  "name": "my-react-app",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",           // 開発サーバー起動
    "build": "tsc && vite build",  // 本番ビルド
    "preview": "vite preview"      // ビルド結果のプレビュー
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.0.8"
  }
}
```

**重要な項目**：
- `scripts`：コマンドのショートカット
- `dependencies`：本番環境で必要なライブラリ
- `devDependencies`：開発環境でのみ必要なライブラリ

#### 2. `src/main.tsx`

Reactアプリのエントリーポイント（起動ファイル）です。

```typescript
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Reactアプリを #root 要素にマウント（描画）
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

**解説**：
- `ReactDOM.createRoot()`：React 18の新しいレンダリングAPI
- `document.getElementById('root')!`：HTMLの `<div id="root">` を取得
- `<React.StrictMode>`：開発モードでの警告を有効化
- `<App />`：メインのAppコンポーネントを描画

#### 3. `src/App.tsx`

メインのコンポーネントです。

```typescript
import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <div>
        <a href="https://vitejs.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Vite + React</h1>
      <div className="card">
        <button onClick={() => setCount((count) => count + 1)}>
          count is {count}
        </button>
        <p>
          Edit <code>src/App.tsx</code> and save to test HMR
        </p>
      </div>
      <p className="read-the-docs">
        Click on the Vite and React logos to learn more
      </p>
    </>
  )
}

export default App
```

**このコードの仕組み**：
1. `useState(0)`：カウンター状態を初期化
2. `<button onClick={...}>`：クリックでカウントを増やす
3. `count is {count}`：現在のカウントを表示

---

## 開発サーバーの起動と動作確認

### 開発サーバーの起動

```bash
# pnpmの場合
pnpm dev

# npmの場合
npm run dev
```

### 実行結果

```
  VITE v5.0.8  ready in 324 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**重要**：
- `http://localhost:5173/`：ローカル開発サーバーのURL
- ポート番号（5173）は環境によって変わる場合があります

### ブラウザで確認

1. ブラウザを開く（Chrome、Firefox、Edge など）
2. アドレスバーに `http://localhost:5173/` を入力
3. Reactアプリが表示される

**表示される内容**：
- ViteとReactのロゴ
- カウンターボタン
- 「count is 0」のテキスト

### Hot Module Replacement (HMR) を体験

HMRは、コードを変更すると**ブラウザをリロードせずに**即座に反映される機能です。

1. `src/App.tsx` を開く
2. `<h1>Vite + React</h1>` を `<h1>Hello React!</h1>` に変更
3. ファイルを保存（Ctrl+S / Cmd+S）
4. ブラウザが自動的に更新される（**リロード不要**）

**所要時間**：約0.1秒（ほぼ瞬時）

---

## 最初のコンポーネント編集

実際にReactコンポーネントを編集してみましょう。

### 演習：シンプルな自己紹介アプリ

#### ステップ1：App.tsxを編集

```typescript
import { useState } from 'react'
import './App.css'

function App() {
  const [name, setName] = useState('')

  return (
    <div className="App">
      <h1>自己紹介アプリ</h1>

      <div>
        <label htmlFor="name-input">あなたの名前:</label>
        <input
          id="name-input"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="名前を入力してください"
        />
      </div>

      {name && (
        <div className="greeting">
          <h2>こんにちは、{name}さん！</h2>
          <p>Reactへようこそ！</p>
        </div>
      )}
    </div>
  )
}

export default App
```

#### ステップ2：スタイルを追加（App.css）

```css
.App {
  max-width: 600px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
}

input {
  width: 100%;
  max-width: 300px;
  padding: 0.5rem;
  margin: 1rem 0;
  font-size: 1rem;
  border: 2px solid #646cff;
  border-radius: 4px;
}

.greeting {
  margin-top: 2rem;
  padding: 1rem;
  background-color: #f0f0f0;
  border-radius: 8px;
}

.greeting h2 {
  color: #646cff;
  margin: 0 0 0.5rem 0;
}

.greeting p {
  margin: 0;
  color: #333;
}
```

#### 実行結果

1. ブラウザに「自己紹介アプリ」と表示される
2. 入力欄に名前を入力すると、リアルタイムで挨拶が表示される
3. 入力を消すと、挨拶も消える

**このコードのポイント**：
- `useState('')`：名前を管理する状態
- `onChange={(e) => setName(e.target.value)}`：入力のたびに状態を更新
- `{name && <div>...</div>}`：名前が入力されている場合のみ表示

---

## よくあるトラブルと解決策

### トラブル1：`node: command not found`

**症状**：
```bash
$ node --version
bash: node: command not found
```

**原因**：
- Node.jsがインストールされていない
- PATHが通っていない

**解決策**：
```bash
# 1. Node.jsを再インストール
# 公式サイトからインストーラーをダウンロード
# https://nodejs.org/

# 2. ターミナルを再起動

# 3. 確認
node --version
```

### トラブル2：`EACCES: permission denied`

**症状**：
```bash
$ npm install -g pnpm
npm ERR! Error: EACCES: permission denied
```

**原因**：
- グローバルインストールに管理者権限が必要

**解決策（macOS / Linux）**：
```bash
# sudoを使う（非推奨）
sudo npm install -g pnpm

# または、npmのグローバルディレクトリを変更（推奨）
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**解決策（Windows）**：
- PowerShellを「管理者として実行」
- コマンドを再実行

### トラブル3：ポート5173が使用中

**症状**：
```bash
$ pnpm dev
Error: Port 5173 is already in use
```

**原因**：
- 別のプロセスがポート5173を使用している

**解決策1：別のポートを使う**
```bash
# vite.config.tsを編集
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000  // 別のポートを指定
  }
})
```

**解決策2：使用中のプロセスを終了**
```bash
# macOS / Linux
lsof -ti:5173 | xargs kill -9

# Windows（PowerShell）
Get-Process -Id (Get-NetTCPConnection -LocalPort 5173).OwningProcess | Stop-Process
```

### トラブル4：依存関係のエラー

**症状**：
```bash
$ pnpm install
ERR_PNPM_PEER_DEP_ISSUES  Unmet peer dependencies
```

**解決策**：
```bash
# node_modulesとロックファイルを削除
rm -rf node_modules pnpm-lock.yaml

# 再インストール
pnpm install

# または、--force オプション
pnpm install --force
```

---

## 演習問題

### 問題1：カウンターの拡張

**難易度**：初級

**課題**：
カウンターに以下の機能を追加してください。
- 「+10」ボタン（10増やす）
- 「リセット」ボタン（0に戻す）
- 現在の値が偶数か奇数かを表示

**ヒント**：
- `setCount(count + 10)` で10増やせる
- `count % 2 === 0` で偶数判定
- 条件演算子 `? :` を使う

**解答例**：
```typescript
import { useState } from 'react'
import './App.css'

function App() {
  const [count, setCount] = useState(0)

  const increment = () => setCount(count + 1)
  const incrementBy10 = () => setCount(count + 10)
  const reset = () => setCount(0)

  const isEven = count % 2 === 0

  return (
    <div className="App">
      <h1>拡張カウンター</h1>

      <p className="count-display">
        現在の値: {count}
      </p>

      <p className="count-type">
        {isEven ? '偶数です' : '奇数です'}
      </p>

      <div className="button-group">
        <button onClick={increment}>+1</button>
        <button onClick={incrementBy10}>+10</button>
        <button onClick={reset}>リセット</button>
      </div>
    </div>
  )
}

export default App
```

### 問題2：独自のプロジェクトを作成

**難易度**：中級

**課題**：
新しいReactプロジェクトを作成し、以下を実装してください。
- プロジェクト名：`todo-app`
- 機能：シンプルなTODOリストアプリ
  - TODOを追加できる
  - TODOの数を表示
  - 入力後、フィールドをクリア

**ヒント**：
- `pnpm create vite todo-app --template react-ts`
- 配列の状態管理：`useState<string[]>([])`
- スプレッド構文：`[...todos, newTodo]`

**解答例**：
```typescript
import { useState } from 'react'
import './App.css'

function App() {
  const [todos, setTodos] = useState<string[]>([])
  const [input, setInput] = useState('')

  const addTodo = () => {
    if (input.trim()) {
      setTodos([...todos, input])
      setInput('')  // 入力欄をクリア
    }
  }

  return (
    <div className="App">
      <h1>TODOリスト</h1>

      <div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addTodo()}
          placeholder="新しいTODOを入力"
        />
        <button onClick={addTodo}>追加</button>
      </div>

      <p>合計: {todos.length}件</p>

      <ul>
        {todos.map((todo, index) => (
          <li key={index}>{todo}</li>
        ))}
      </ul>
    </div>
  )
}

export default App
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Node.jsとpnpmのインストール
- ✅ Viteを使ったReactプロジェクトの作成
- ✅ プロジェクト構造の理解
- ✅ 開発サーバーの起動とHMRの体験
- ✅ 最初のコンポーネント編集
- ✅ よくあるトラブルシューティング

### 次に学ぶべきガイド

1. **[03-jsx-fundamentals.md](./03-jsx-fundamentals.md)**
   - JSXの詳しい文法
   - JavaScriptとの統合
   - 条件分岐とループ

2. **[04-components-intro.md](./04-components-intro.md)**
   - コンポーネントの分割
   - Propsの受け渡し
   - コンポーネント設計

### 関連リソース

#### 公式ドキュメント
- [Vite公式ドキュメント](https://vitejs.dev/)
- [React公式ドキュメント - Installation](https://react.dev/learn/installation)

#### ツール
- [VS Code](https://code.visualstudio.com/)：推奨エディタ
- [ES7+ React/Redux/React-Native snippets](https://marketplace.visualstudio.com/items?itemName=dsznajder.es7-react-js-snippets)：VS Code拡張機能

---

**次のガイド**：[03-jsx-fundamentals.md](./03-jsx-fundamentals.md)

**前のガイド**：[01-what-is-react.md](./01-what-is-react.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
