# Props基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [Propsとは何か](#propsとは何か)
4. [Propsの基本的な使い方](#propsの基本的な使い方)
5. [TypeScriptでの型定義](#typescriptでの型定義)
6. [デフォルト値](#デフォルト値)
7. [childrenプロップ](#childrenプロップ)
8. [Propsの不変性](#propsの不変性)
9. [よくある間違い](#よくある間違い)
10. [演習問題](#演習問題)
11. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Propsの基本概念と仕組み
- 親コンポーネントから子コンポーネントへのデータ渡し
- TypeScriptでの型安全なProps定義
- デフォルト値の設定
- childrenプロップの活用
- Propsの不変性ルール

### なぜ重要か

**Props（プロップス）**は、Reactコンポーネント間でデータを受け渡すための仕組みです。Propsを理解することで：
- **再利用可能なコンポーネント**：同じコンポーネントに異なるデータを渡せる
- **データの流れが明確**：親→子への一方向データフロー
- **型安全性**：TypeScriptで間違いを防げる

### 学習時間の目安

- このガイドの読了：30〜40分
- 演習問題を含めた完全理解：1〜2時間

---

## 前提知識

### 必要な知識

1. **コンポーネント基礎**：
   - [04-components-intro.md](./04-components-intro.md) を完了していること

2. **TypeScript基礎**：
   - 型注釈（`: string`、`: number` など）
   - `type` または `interface` の定義

3. **JavaScript ES6**：
   - オブジェクト（`{ key: value }`）
   - 分割代入（`const { name } = user`）

---

## Propsとは何か

### 定義

**Props**は、"properties"（プロパティ）の略で、**親コンポーネントから子コンポーネントに渡すデータ**です。

```typescript
// 親コンポーネント
function App() {
  return <Greeting name="太郎" />;  // nameプロップを渡す
}

// 子コンポーネント
function Greeting({ name }: { name: string }) {
  return <h1>こんにちは、{name}さん！</h1>;
}

// 出力：こんにちは、太郎さん！
```

### HTMLの属性とのアナロジー

PropsはHTMLの属性に似ています。

```html
<!-- HTML -->
<img src="logo.png" alt="ロゴ" width="100" />

<!-- React -->
<Avatar imageUrl="logo.png" altText="ロゴ" size={100} />
```

### データの流れ（単方向データフロー）

Reactでは、データは**親から子へ**一方向に流れます。

```
App (親)
  ↓ name="太郎"
Greeting (子)
```

**重要**：子コンポーネントは、受け取ったPropsを**読み取り専用**として扱います（後述）。

---

## Propsの基本的な使い方

### 1. シンプルな例

```typescript
// 子コンポーネント
function Welcome({ name }: { name: string }) {
  return <h1>ようこそ、{name}さん！</h1>;
}

// 親コンポーネント
function App() {
  return (
    <div>
      <Welcome name="太郎" />
      <Welcome name="花子" />
      <Welcome name="次郎" />
    </div>
  );
}

// 出力：
// ようこそ、太郎さん！
// ようこそ、花子さん！
// ようこそ、次郎さん！
```

### 2. 複数のPropsを渡す

```typescript
type UserCardProps = {
  name: string;
  age: number;
  occupation: string;
};

function UserCard({ name, age, occupation }: UserCardProps) {
  return (
    <div className="user-card">
      <h2>{name}</h2>
      <p>年齢: {age}歳</p>
      <p>職業: {occupation}</p>
    </div>
  );
}

// 使用例
<UserCard name="山田太郎" age={28} occupation="エンジニア" />
```

### 3. 様々な型のPropsを渡す

```typescript
type ProductProps = {
  name: string;           // 文字列
  price: number;          // 数値
  inStock: boolean;       // 真偽値
  tags: string[];         // 配列
  manufacturer: {         // オブジェクト
    name: string;
    country: string;
  };
  onBuy: () => void;      // 関数
};

function Product({
  name,
  price,
  inStock,
  tags,
  manufacturer,
  onBuy
}: ProductProps) {
  return (
    <div className="product">
      <h2>{name}</h2>
      <p>価格: ¥{price.toLocaleString()}</p>
      <p>{inStock ? '在庫あり' : '在庫なし'}</p>
      <p>タグ: {tags.join(', ')}</p>
      <p>製造元: {manufacturer.name} ({manufacturer.country})</p>
      <button onClick={onBuy} disabled={!inStock}>
        購入する
      </button>
    </div>
  );
}

// 使用例
<Product
  name="ノートパソコン"
  price={89800}
  inStock={true}
  tags={['電子機器', '人気商品']}
  manufacturer={{ name: 'テックコーポ', country: '日本' }}
  onBuy={() => alert('購入しました！')}
/>
```

---

## TypeScriptでの型定義

### 1. インライン型定義（シンプルな場合）

```typescript
function Greeting({ name }: { name: string }) {
  return <h1>こんにちは、{name}さん！</h1>;
}
```

### 2. type エイリアス（推奨）

```typescript
type GreetingProps = {
  name: string;
};

function Greeting({ name }: GreetingProps) {
  return <h1>こんにちは、{name}さん！</h1>;
}
```

### 3. interface（オブジェクト指向的）

```typescript
interface GreetingProps {
  name: string;
}

function Greeting({ name }: GreetingProps) {
  return <h1>こんにちは、{name}さん！</h1>;
}
```

**type vs interface**：
- `type`：より柔軟（Union型、Intersection型など）
- `interface`：拡張可能（`extends`）

**推奨**：Propsの定義には`type`を使うのが一般的です。

### 4. オプショナルなProps

```typescript
type UserCardProps = {
  name: string;
  age: number;
  email?: string;        // オプショナル（?を付ける）
  bio?: string;
};

function UserCard({ name, age, email, bio }: UserCardProps) {
  return (
    <div>
      <h2>{name}</h2>
      <p>年齢: {age}歳</p>
      {email && <p>メール: {email}</p>}
      {bio && <p>自己紹介: {bio}</p>}
    </div>
  );
}

// emailとbioは省略可能
<UserCard name="太郎" age={25} />
<UserCard name="花子" age={30} email="hanako@example.com" />
```

### 5. Union型（複数の型）

```typescript
type ButtonProps = {
  text: string;
  variant: 'primary' | 'secondary' | 'danger';  // 3つのうちいずれか
};

function Button({ text, variant }: ButtonProps) {
  const className = `btn btn-${variant}`;
  return <button className={className}>{text}</button>;
}

// 使用例
<Button text="送信" variant="primary" />
<Button text="キャンセル" variant="secondary" />
<Button text="削除" variant="danger" />
// <Button text="保存" variant="success" />  ← エラー！
```

---

## デフォルト値

### 1. 分割代入でのデフォルト値（推奨）

```typescript
type GreetingProps = {
  name: string;
  greeting?: string;
};

function Greeting({ name, greeting = "こんにちは" }: GreetingProps) {
  return <h1>{greeting}、{name}さん！</h1>;
}

// 使用例
<Greeting name="太郎" />
// 出力：こんにちは、太郎さん！

<Greeting name="花子" greeting="おはよう" />
// 出力：おはよう、花子さん！
```

### 2. 複数のデフォルト値

```typescript
type ButtonProps = {
  text: string;
  variant?: 'primary' | 'secondary' | 'danger';
  disabled?: boolean;
  size?: 'small' | 'medium' | 'large';
};

function Button({
  text,
  variant = 'primary',
  disabled = false,
  size = 'medium'
}: ButtonProps) {
  const className = `btn btn-${variant} btn-${size}`;

  return (
    <button className={className} disabled={disabled}>
      {text}
    </button>
  );
}

// 使用例
<Button text="送信" />
// variant="primary", disabled=false, size="medium" が自動適用
```

### 3. オブジェクトのデフォルト値

```typescript
type UserCardProps = {
  user?: {
    name: string;
    age: number;
  };
};

function UserCard({
  user = { name: 'ゲスト', age: 0 }
}: UserCardProps) {
  return (
    <div>
      <h2>{user.name}</h2>
      <p>{user.age}歳</p>
    </div>
  );
}

// 使用例
<UserCard />
// 出力：ゲスト、0歳

<UserCard user={{ name: '太郎', age: 25 }} />
// 出力：太郎、25歳
```

---

## childrenプロップ

### childrenとは

`children`は特別なプロップで、**コンポーネントのタグの間に挟まれたコンテンツ**を表します。

```typescript
// childrenを受け取るコンポーネント
type CardProps = {
  children: React.ReactNode;
};

function Card({ children }: CardProps) {
  return (
    <div className="card">
      {children}
    </div>
  );
}

// 使用例
<Card>
  <h2>タイトル</h2>
  <p>本文</p>
</Card>
```

### childrenの型

```typescript
import { ReactNode } from 'react';

type ContainerProps = {
  children: ReactNode;  // あらゆるReact要素
};

function Container({ children }: ContainerProps) {
  return <div className="container">{children}</div>;
}
```

**ReactNodeの種類**：
- 文字列（`"テキスト"`）
- 数値（`123`）
- JSX要素（`<div>...</div>`）
- 配列（`[<p>1</p>, <p>2</p>]`）
- `null` / `undefined`

### 実践例：レイアウトコンポーネント

```typescript
type PageLayoutProps = {
  children: ReactNode;
};

function PageLayout({ children }: PageLayoutProps) {
  return (
    <div className="page-layout">
      <header>
        <h1>マイアプリ</h1>
      </header>
      <main>{children}</main>
      <footer>© 2024</footer>
    </div>
  );
}

// 使用例
<PageLayout>
  <h2>ホームページ</h2>
  <p>ようこそ！</p>
</PageLayout>

<PageLayout>
  <h2>プロフィールページ</h2>
  <UserProfile />
</PageLayout>
```

### 複数のchildrenスロット

```typescript
type ModalProps = {
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
};

function Modal({ title, children, footer }: ModalProps) {
  return (
    <div className="modal">
      <header>{title}</header>
      <main>{children}</main>
      {footer && <footer>{footer}</footer>}
    </div>
  );
}

// 使用例
<Modal
  title={<h2>確認</h2>}
  footer={
    <>
      <button>OK</button>
      <button>キャンセル</button>
    </>
  }
>
  <p>本当に削除しますか？</p>
</Modal>
```

---

## Propsの不変性

### 重要なルール：Propsは読み取り専用

**Propsを直接変更してはいけません**。これは非常に重要なルールです。

```typescript
type CounterProps = {
  count: number;
};

function Counter({ count }: CounterProps) {
  // ❌ エラー：Propsを変更している
  count = count + 1;

  return <div>{count}</div>;
}
```

**理由**：
- **予測可能性**：データの流れが一方向で追跡しやすい
- **デバッグの容易さ**：バグの原因を特定しやすい
- **パフォーマンス最適化**：Reactが効率的に再レンダリングできる

### 正しい方法：状態管理を使う

Propsではなく、**State**（状態）を使います。

```typescript
import { useState } from 'react';

function Counter() {
  // ✅ 正しい：useStateを使う
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>
        +1
      </button>
    </div>
  );
}
```

**State**については、次のガイド [06-state-basics.md](./06-state-basics.md) で詳しく学びます。

### オブジェクトや配列のPropsも不変

```typescript
type UserListProps = {
  users: string[];
};

function UserList({ users }: UserListProps) {
  // ❌ エラー：Propsの配列を変更
  users.push('新しいユーザー');

  return (
    <ul>
      {users.map(user => <li key={user}>{user}</li>)}
    </ul>
  );
}
```

---

## よくある間違い

### 間違い1：Propsの型定義を忘れる

❌ **問題**：
```typescript
function Greeting({ name }) {  // 型がない
  return <h1>こんにちは、{name}さん！</h1>;
}
```

✅ **解決策**：
```typescript
type GreetingProps = {
  name: string;
};

function Greeting({ name }: GreetingProps) {
  return <h1>こんにちは、{name}さん！</h1>;
}
```

### 間違い2：Propsを直接変更する

❌ **問題**：
```typescript
function Counter({ count }: { count: number }) {
  count = count + 1;  // Propsを変更
  return <div>{count}</div>;
}
```

✅ **解決策**：
```typescript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

### 間違い3：文字列以外をクォートで囲む

❌ **問題**：
```typescript
<UserCard age="25" />  // 文字列"25"が渡される
```

✅ **解決策**：
```typescript
<UserCard age={25} />  // 数値25が渡される
```

**ルール**：
- 文字列：`<Component name="太郎" />`
- それ以外：`<Component age={25} active={true} items={[]} />`

### 間違い4：必須Propsを省略

❌ **問題**：
```typescript
type UserCardProps = {
  name: string;  // 必須
  age: number;   // 必須
};

<UserCard name="太郎" />  // ageがない！TypeScriptエラー
```

✅ **解決策**：
```typescript
// 方法1：すべてのPropsを渡す
<UserCard name="太郎" age={25} />

// 方法2：オプショナルにする
type UserCardProps = {
  name: string;
  age?: number;  // オプショナル
};
```

### 間違い5：keyをPropsとして使おうとする

❌ **問題**：
```typescript
function Item({ key }: { key: string }) {
  return <li>{key}</li>;
}

// keyはReact内部で使われる特別なプロップ
```

✅ **解決策**：
```typescript
// 別の名前を使う
function Item({ id }: { id: string }) {
  return <li>{id}</li>;
}

{items.map(item => (
  <Item key={item.id} id={item.id} />
))}
```

---

## 演習問題

### 問題1：商品カード

**難易度**：初級

**課題**：
以下のPropsを受け取る商品カードコンポーネントを作成してください。
- `name`：商品名（文字列、必須）
- `price`：価格（数値、必須）
- `imageUrl`：画像URL（文字列、オプション）
- `onSale`：セール中か（真偽値、オプション、デフォルト：false）

**要件**：
- TypeScriptで型を定義
- セール中の場合、「セール中！」と表示

**解答例**：
```typescript
type ProductCardProps = {
  name: string;
  price: number;
  imageUrl?: string;
  onSale?: boolean;
};

function ProductCard({
  name,
  price,
  imageUrl = 'https://via.placeholder.com/150',
  onSale = false
}: ProductCardProps) {
  return (
    <div className="product-card">
      <img src={imageUrl} alt={name} />
      <h3>{name}</h3>
      <p className="price">¥{price.toLocaleString()}</p>
      {onSale && <span className="badge">セール中！</span>}
    </div>
  );
}

// 使用例
<ProductCard name="ノートパソコン" price={89800} onSale={true} />
<ProductCard name="マウス" price={2980} />
```

### 問題2：再利用可能なボタン

**難易度**：中級

**課題**：
以下の機能を持つボタンコンポーネントを作成してください。
- `text`：ボタンのテキスト（必須）
- `variant`：スタイル（'primary' | 'secondary' | 'danger'、デフォルト：'primary'）
- `size`：サイズ（'small' | 'medium' | 'large'、デフォルト：'medium'）
- `disabled`：無効化（デフォルト：false）
- `onClick`：クリックハンドラ（必須）

**要件**：
- Union型を使う
- デフォルト値を設定

**解答例**：
```typescript
type ButtonProps = {
  text: string;
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  onClick: () => void;
};

function Button({
  text,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  onClick
}: ButtonProps) {
  const className = `btn btn-${variant} btn-${size}`;

  return (
    <button
      className={className}
      disabled={disabled}
      onClick={onClick}
    >
      {text}
    </button>
  );
}

// 使用例
function App() {
  return (
    <div>
      <Button
        text="送信"
        variant="primary"
        onClick={() => alert('送信しました')}
      />
      <Button
        text="削除"
        variant="danger"
        size="small"
        onClick={() => alert('削除しました')}
      />
    </div>
  );
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Propsの基本概念と仕組み
- ✅ TypeScriptでの型安全なProps定義
- ✅ デフォルト値の設定方法
- ✅ childrenプロップの活用
- ✅ Propsの不変性ルール

### 次に学ぶべきガイド

1. **[06-state-basics.md](./06-state-basics.md)**
   - 状態管理の基礎
   - useStateフック
   - 動的なUIの作成

2. **[07-events-lists.md](./07-events-lists.md)**
   - イベント処理
   - リストのレンダリング
   - ユーザーインタラクション

### 関連リソース

#### 公式ドキュメント
- [React: Passing Props to a Component](https://react.dev/learn/passing-props-to-a-component)
- [TypeScript: React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app/)

---

**次のガイド**：[06-state-basics.md](./06-state-basics.md)

**前のガイド**：[04-components-intro.md](./04-components-intro.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
