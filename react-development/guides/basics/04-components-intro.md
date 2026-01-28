# コンポーネント入門 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [コンポーネントとは何か](#コンポーネントとは何か)
4. [関数コンポーネントの書き方](#関数コンポーネントの書き方)
5. [コンポーネントの分割](#コンポーネントの分割)
6. [インポートとエクスポート](#インポートとエクスポート)
7. [コンポーネントの設計原則](#コンポーネントの設計原則)
8. [コンポーネントの合成](#コンポーネントの合成)
9. [よくある間違い](#よくある間違い)
10. [演習問題](#演習問題)
11. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Reactコンポーネントの基本概念
- 関数コンポーネントの書き方
- コンポーネントの分割と再利用
- ファイル構造とインポート/エクスポート
- コンポーネント設計の基本原則

### なぜ重要か

**コンポーネント**は、Reactアプリケーションの構成要素です。コンポーネントを理解することで：
- **再利用性**：一度作ったコンポーネントを何度も使える
- **保守性**：小さな部品に分けることで、変更が容易
- **テスト性**：個別にテストできる
- **チーム開発**：役割を分担しやすい

### 学習時間の目安

- このガイドの読了：30〜40分
- 演習問題を含めた完全理解：1〜2時間

---

## 前提知識

### 必要な知識

1. **JSX基礎**：
   - [03-jsx-fundamentals.md](./03-jsx-fundamentals.md) を完了していること

2. **JavaScript ES6**：
   - アロー関数
   - import/export
   - 分割代入

3. **TypeScript基礎**：
   - 型注釈（`: string`、`: number` など）
   - インターフェース（`interface` または `type`）

---

## コンポーネントとは何か

### 定義

**コンポーネント**は、UIの一部を表す**再利用可能な部品**です。

```typescript
// 最もシンプルなコンポーネント
function Welcome() {
  return <h1>こんにちは！</h1>;
}
```

### レゴブロックのアナロジー

コンポーネントは**レゴブロック**のようなものです：
- 小さな部品（コンポーネント）を組み合わせる
- 同じ部品を何度でも使える
- 部品を入れ替えて違う形を作れる

```typescript
// 小さな部品
function Button() {
  return <button>クリック</button>;
}

// 部品を組み合わせる
function App() {
  return (
    <div>
      <Button />
      <Button />
      <Button />
    </div>
  );
}
```

### コンポーネントの種類

React 16.8以降、**関数コンポーネント**が推奨されています。

```typescript
// ✅ 推奨：関数コンポーネント（2024年現在の標準）
function Greeting() {
  return <h1>こんにちは</h1>;
}

// ❌ 非推奨：クラスコンポーネント（古い書き方）
class Greeting extends React.Component {
  render() {
    return <h1>こんにちは</h1>;
  }
}
```

**このガイドでは関数コンポーネントのみ扱います。**

---

## 関数コンポーネントの書き方

### 基本形

```typescript
function ComponentName() {
  return (
    <div>
      {/* JSXをここに書く */}
    </div>
  );
}
```

### 重要なルール

#### 1. 名前は必ず大文字で始める

```typescript
// ✅ 正しい：大文字で始まる
function MyComponent() {
  return <div>コンテンツ</div>;
}

// ❌ 間違い：小文字で始まる
function myComponent() {
  return <div>コンテンツ</div>;
}
// Reactは <myComponent /> を HTML タグと認識してしまう
```

#### 2. 必ず何かをreturnする

```typescript
// ✅ 正しい：JSXを返す
function ValidComponent() {
  return <div>コンテンツ</div>;
}

// ❌ 間違い：何も返さない
function InvalidComponent() {
  <div>コンテンツ</div>;  // returnがない
}
```

#### 3. 単一のルート要素を返す

```typescript
// ❌ 間違い：複数のルート要素
function InvalidComponent() {
  return (
    <h1>タイトル</h1>
    <p>本文</p>
  );
}

// ✅ 正しい：Fragmentで囲む
function ValidComponent() {
  return (
    <>
      <h1>タイトル</h1>
      <p>本文</p>
    </>
  );
}
```

### 実例：シンプルなコンポーネント

```typescript
// ユーザーカードコンポーネント
function UserCard() {
  return (
    <div className="card">
      <img src="avatar.jpg" alt="ユーザー" />
      <h2>山田太郎</h2>
      <p>Web開発者</p>
    </div>
  );
}

// 使用例
function App() {
  return (
    <div>
      <UserCard />
      <UserCard />
      <UserCard />
    </div>
  );
}
```

---

## コンポーネントの分割

### なぜ分割するのか

大きなコンポーネントは、以下の問題があります：
- **理解しにくい**：コードが長すぎる
- **再利用できない**：特定の場所でしか使えない
- **テストしにくい**：複雑すぎてテストが困難

### 分割の例

#### Before：1つの大きなコンポーネント

```typescript
function BlogPost() {
  return (
    <article>
      <header>
        <h1>React入門</h1>
        <div>
          <img src="author.jpg" alt="著者" />
          <span>山田太郎</span>
          <time>2024年1月28日</time>
        </div>
      </header>

      <div>
        <p>Reactは素晴らしいライブラリです...</p>
        <p>コンポーネントベースで...</p>
      </div>

      <footer>
        <button>いいね</button>
        <button>シェア</button>
        <button>コメント</button>
      </footer>
    </article>
  );
}
```

#### After：複数の小さなコンポーネントに分割

```typescript
// ヘッダーコンポーネント
function PostHeader() {
  return (
    <header>
      <h1>React入門</h1>
      <AuthorInfo />
    </header>
  );
}

// 著者情報コンポーネント
function AuthorInfo() {
  return (
    <div className="author">
      <img src="author.jpg" alt="著者" />
      <span>山田太郎</span>
      <time>2024年1月28日</time>
    </div>
  );
}

// 本文コンポーネント
function PostContent() {
  return (
    <div className="content">
      <p>Reactは素晴らしいライブラリです...</p>
      <p>コンポーネントベースで...</p>
    </div>
  );
}

// フッターコンポーネント
function PostFooter() {
  return (
    <footer>
      <button>いいね</button>
      <button>シェア</button>
      <button>コメント</button>
    </footer>
  );
}

// メインコンポーネント（組み合わせ）
function BlogPost() {
  return (
    <article>
      <PostHeader />
      <PostContent />
      <PostFooter />
    </article>
  );
}
```

**利点**：
- 各コンポーネントが短くて理解しやすい
- `AuthorInfo` や `PostFooter` を他の場所で再利用できる
- 個別にテストできる

---

## インポートとエクスポート

### ファイル分割の必要性

すべてのコンポーネントを1つのファイルに書くと、ファイルが巨大になります。**ファイルを分割**して管理しましょう。

### ディレクトリ構造

```
src/
├── App.tsx
├── components/
│   ├── Button.tsx
│   ├── UserCard.tsx
│   └── Header.tsx
└── main.tsx
```

### エクスポートの種類

#### 1. Default Export（デフォルトエクスポート）

**1つのファイルに1つのコンポーネント**の場合に使います。

```typescript
// components/Button.tsx
function Button() {
  return <button>クリック</button>;
}

export default Button;  // デフォルトエクスポート
```

```typescript
// App.tsx
import Button from './components/Button';  // 任意の名前でインポート可能

function App() {
  return <Button />;
}
```

#### 2. Named Export（名前付きエクスポート）

**1つのファイルに複数のコンポーネント**がある場合に使います。

```typescript
// components/Buttons.tsx
export function PrimaryButton() {
  return <button className="primary">Primary</button>;
}

export function SecondaryButton() {
  return <button className="secondary">Secondary</button>;
}
```

```typescript
// App.tsx
import { PrimaryButton, SecondaryButton } from './components/Buttons';

function App() {
  return (
    <div>
      <PrimaryButton />
      <SecondaryButton />
    </div>
  );
}
```

### ベストプラクティス

```typescript
// ✅ 推奨：1ファイル1コンポーネント + Default Export
// components/UserCard.tsx
function UserCard() {
  return <div className="user-card">...</div>;
}

export default UserCard;
```

**理由**：
- ファイル名とコンポーネント名が一致
- インポート時に名前を自由に変えられる
- 一般的な慣習

---

## コンポーネントの設計原則

### 1. 単一責任の原則（SRP）

1つのコンポーネントは**1つのことだけ**をすべきです。

```typescript
// ❌ 悪い例：複数の責任
function UserDashboard() {
  return (
    <div>
      {/* ユーザー情報の表示 */}
      <div>...</div>

      {/* 投稿一覧の表示 */}
      <div>...</div>

      {/* フレンド一覧の表示 */}
      <div>...</div>

      {/* 通知一覧の表示 */}
      <div>...</div>
    </div>
  );
}

// ✅ 良い例：責任を分割
function UserDashboard() {
  return (
    <div>
      <UserProfile />
      <PostList />
      <FriendList />
      <NotificationList />
    </div>
  );
}
```

### 2. DRY原則（Don't Repeat Yourself）

同じコードを繰り返さない。

```typescript
// ❌ 悪い例：同じコードを繰り返す
function Buttons() {
  return (
    <div>
      <button className="btn btn-primary">保存</button>
      <button className="btn btn-primary">送信</button>
      <button className="btn btn-primary">削除</button>
    </div>
  );
}

// ✅ 良い例：再利用可能なコンポーネントを作る
type ButtonProps = {
  label: string;
};

function PrimaryButton({ label }: ButtonProps) {
  return <button className="btn btn-primary">{label}</button>;
}

function Buttons() {
  return (
    <div>
      <PrimaryButton label="保存" />
      <PrimaryButton label="送信" />
      <PrimaryButton label="削除" />
    </div>
  );
}
```

### 3. 適切な粒度

コンポーネントは**適切なサイズ**に保つべきです。

**目安**：
- 1コンポーネント：50〜100行以内
- 1ファイル：200行以内
- それ以上なら分割を検討

```typescript
// ❌ 大きすぎる
function MassiveComponent() {
  // 500行のコード...
}

// ✅ 適切なサイズ
function Header() { /* 30行 */ }
function Sidebar() { /* 50行 */ }
function MainContent() { /* 80行 */ }
function Footer() { /* 20行 */ }
```

### 4. 命名規則

わかりやすい名前を付けましょう。

```typescript
// ❌ 悪い名前
function A() { }
function Thing() { }
function DoStuff() { }

// ✅ 良い名前
function UserProfile() { }          // ユーザープロフィール
function LoginButton() { }          // ログインボタン
function ProductCard() { }          // 商品カード
function NavigationMenu() { }       // ナビゲーションメニュー
```

**命名のパターン**：
- `UserCard`、`ProductCard`：`...Card`（カード形式）
- `LoginButton`、`SubmitButton`：`...Button`（ボタン）
- `UserList`、`ProductList`：`...List`（リスト）
- `UserForm`、`LoginForm`：`...Form`（フォーム）

---

## コンポーネントの合成

### コンポーネントをネストする

コンポーネントは他のコンポーネントを含めることができます。

```typescript
function Avatar() {
  return <img src="avatar.jpg" alt="ユーザー" />;
}

function UserName() {
  return <h2>山田太郎</h2>;
}

function UserCard() {
  return (
    <div className="card">
      <Avatar />
      <UserName />
      <p>Web開発者</p>
    </div>
  );
}

function App() {
  return (
    <div>
      <UserCard />
    </div>
  );
}
```

### コンポーネントツリー

アプリケーションは**コンポーネントツリー**として表現できます。

```
App
├── Header
│   ├── Logo
│   └── Navigation
│       ├── NavItem
│       ├── NavItem
│       └── NavItem
├── Main
│   ├── Sidebar
│   │   └── Widget
│   └── Content
│       ├── Article
│       └── Article
└── Footer
    ├── Copyright
    └── SocialLinks
```

---

## よくある間違い

### 間違い1：小文字で始まるコンポーネント名

❌ **問題**：
```typescript
function button() {
  return <button>クリック</button>;
}

<button />  // HTMLの<button>として扱われる
```

✅ **解決策**：
```typescript
function Button() {
  return <button>クリック</button>;
}

<Button />  // Reactコンポーネントとして扱われる
```

### 間違い2：コンポーネント内でコンポーネントを定義

❌ **問題**：
```typescript
function ParentComponent() {
  // ❌ コンポーネント内で別のコンポーネントを定義
  function ChildComponent() {
    return <div>子コンポーネント</div>;
  }

  return <ChildComponent />;
}
```

**問題点**：
- 毎回新しいコンポーネントが作成される
- パフォーマンスが悪化
- 状態がリセットされる

✅ **解決策**：
```typescript
// コンポーネントは外で定義
function ChildComponent() {
  return <div>子コンポーネント</div>;
}

function ParentComponent() {
  return <ChildComponent />;
}
```

### 間違い3：returnを忘れる

❌ **問題**：
```typescript
function MyComponent() {
  <div>コンテンツ</div>;  // returnがない
}
```

✅ **解決策**：
```typescript
function MyComponent() {
  return <div>コンテンツ</div>;
}
```

### 間違い4：ファイル名とコンポーネント名が一致しない

❌ **問題**：
```typescript
// ファイル名: UserProfile.tsx
function Profile() {  // 名前が一致しない
  return <div>...</div>;
}
```

✅ **解決策**：
```typescript
// ファイル名: UserProfile.tsx
function UserProfile() {  // 名前が一致
  return <div>...</div>;
}

export default UserProfile;
```

---

## 演習問題

### 問題1：名刺コンポーネント

**難易度**：初級

**課題**：
以下の情報を表示する名刺コンポーネントを作成してください。
- 名前
- 役職
- 会社名
- メールアドレス

**ヒント**：
- シンプルな関数コンポーネントで十分
- 適切なHTML要素を使う

**解答例**：
```typescript
function BusinessCard() {
  return (
    <div className="business-card">
      <h2>山田太郎</h2>
      <p className="title">シニアエンジニア</p>
      <p className="company">株式会社テックソリューション</p>
      <a href="mailto:taro@example.com">taro@example.com</a>
    </div>
  );
}

export default BusinessCard;
```

### 問題2：ブログ記事を分割

**難易度**：中級

**課題**：
以下の大きなコンポーネントを、4つの小さなコンポーネントに分割してください。
- `ArticleHeader`：タイトルと著者情報
- `ArticleContent`：本文
- `ArticleTags`：タグ一覧
- `BlogArticle`：全体をまとめる

**元のコード**：
```typescript
function BlogArticle() {
  return (
    <article>
      <header>
        <h1>Reactの基礎</h1>
        <div>
          <span>著者: 山田太郎</span>
          <time>2024年1月28日</time>
        </div>
      </header>

      <div>
        <p>Reactはコンポーネントベースのライブラリです。</p>
        <p>再利用可能な部品を作ることができます。</p>
      </div>

      <footer>
        <span className="tag">React</span>
        <span className="tag">JavaScript</span>
        <span className="tag">Web開発</span>
      </footer>
    </article>
  );
}
```

**解答例**：
```typescript
// ArticleHeader.tsx
function ArticleHeader() {
  return (
    <header>
      <h1>Reactの基礎</h1>
      <div className="meta">
        <span>著者: 山田太郎</span>
        <time>2024年1月28日</time>
      </div>
    </header>
  );
}

export default ArticleHeader;

// ArticleContent.tsx
function ArticleContent() {
  return (
    <div className="content">
      <p>Reactはコンポーネントベースのライブラリです。</p>
      <p>再利用可能な部品を作ることができます。</p>
    </div>
  );
}

export default ArticleContent;

// ArticleTags.tsx
function ArticleTags() {
  const tags = ['React', 'JavaScript', 'Web開発'];

  return (
    <footer>
      {tags.map(tag => (
        <span key={tag} className="tag">{tag}</span>
      ))}
    </footer>
  );
}

export default ArticleTags;

// BlogArticle.tsx
import ArticleHeader from './ArticleHeader';
import ArticleContent from './ArticleContent';
import ArticleTags from './ArticleTags';

function BlogArticle() {
  return (
    <article>
      <ArticleHeader />
      <ArticleContent />
      <ArticleTags />
    </article>
  );
}

export default BlogArticle;
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ コンポーネントの基本概念
- ✅ 関数コンポーネントの書き方
- ✅ コンポーネントの分割方法
- ✅ インポート/エクスポート
- ✅ コンポーネント設計の原則

### 次に学ぶべきガイド

1. **[05-props-basics.md](./05-props-basics.md)**
   - Propsの詳細
   - データの受け渡し
   - TypeScriptでの型定義

2. **[06-state-basics.md](./06-state-basics.md)**
   - 状態管理の基礎
   - useStateフック
   - 動的なUIの作成

### 関連リソース

#### 公式ドキュメント
- [React: Your First Component](https://react.dev/learn/your-first-component)
- [React: Importing and Exporting Components](https://react.dev/learn/importing-and-exporting-components)

#### 設計パターン
- [Atomic Design](https://bradfrost.com/blog/post/atomic-web-design/)
- [Component Design Patterns in React](https://www.patterns.dev/)

---

**次のガイド**：[05-props-basics.md](./05-props-basics.md)

**前のガイド**：[03-jsx-fundamentals.md](./03-jsx-fundamentals.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
