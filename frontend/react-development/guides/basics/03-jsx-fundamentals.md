# JSX基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [JSXとは何か](#jsxとは何か)
4. [JSX基本文法](#jsx基本文法)
5. [JavaScriptの埋め込み](#javascriptの埋め込み)
6. [属性（Attributes）](#属性attributes)
7. [条件分岐](#条件分岐)
8. [リストのレンダリング](#リストのレンダリング)
9. [フラグメント](#フラグメント)
10. [よくある間違い](#よくある間違い)
11. [演習問題](#演習問題)
12. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- JSXの基本概念と仕組み
- HTMLに似た構文でUIを記述する方法
- JavaScriptの式をJSXに埋め込む方法
- 条件分岐とループの実装
- JSXの制約とルール

### なぜ重要か

**JSX（JavaScript XML）**は、Reactの最も特徴的な構文です。JSXを使うことで：
- **直感的なUI記述**：HTMLのような見た目でコンポーネントを書ける
- **型安全性**：TypeScriptと組み合わせて安全なコードを書ける
- **強力な表現力**：JavaScriptの全機能をUI記述に活用できる

### 学習時間の目安

- このガイドの読了：30〜40分
- 演習問題を含めた完全理解：1〜2時間

---

## 前提知識

### 必要な知識

1. **HTML基礎**：
   - タグの構造（`<div>`, `<p>`, `<button>` など）
   - 属性（`class`, `id`, `href` など）

2. **JavaScript基礎**：
   - 変数（`const`, `let`）
   - 関数（アロー関数）
   - 配列メソッド（`map`, `filter`）
   - テンプレートリテラル（`` `文字列` ``）

3. **React環境構築**：
   - [02-setup-environment.md](./02-setup-environment.md) を完了していること

---

## JSXとは何か

### 公式定義

> JSX is a syntax extension for JavaScript that lets you write HTML-like markup inside a JavaScript file.
> （JSXは、JavaScriptファイル内でHTMLのようなマークアップを書けるようにする、JavaScriptの構文拡張です）

### より詳しい説明

JSXは**JavaScriptの構文拡張**であり、以下の特徴があります：

#### 1. HTMLに似ているが、JavaScriptである

```typescript
// これはJSX
const element = <h1>こんにちは、世界！</h1>;

// ブラウザは直接理解できないため、Babelなどで以下に変換される
const element = React.createElement('h1', null, 'こんにちは、世界！');
```

#### 2. JSXはオブジェクトを生成する

JSXは最終的に**Reactエレメント**（JavaScriptオブジェクト）に変換されます。

```typescript
// JSX
<div className="container">Hello</div>

// 変換後（内部表現）
{
  type: 'div',
  props: {
    className: 'container',
    children: 'Hello'
  }
}
```

#### 3. 任意の場所で使える

JSXは式（expression）なので、変数に代入したり、関数の引数にしたり、return文で返したりできます。

```typescript
// 変数に代入
const greeting = <h1>こんにちは</h1>;

// 関数の引数
const element = renderElement(<p>テキスト</p>);

// return文
function Component() {
  return <div>コンテンツ</div>;
}
```

---

## JSX基本文法

### 1. 単一のルート要素

JSXでは、**必ず1つのルート要素**で囲む必要があります。

```typescript
// ❌ エラー：複数のルート要素
function Component() {
  return (
    <h1>タイトル</h1>
    <p>本文</p>
  );
}

// ✅ 正しい：1つのdivで囲む
function Component() {
  return (
    <div>
      <h1>タイトル</h1>
      <p>本文</p>
    </div>
  );
}

// ✅ より良い：Fragmentを使う（後述）
function Component() {
  return (
    <>
      <h1>タイトル</h1>
      <p>本文</p>
    </>
  );
}
```

### 2. すべてのタグを閉じる

HTMLでは閉じタグが不要な要素も、JSXでは必須です。

```typescript
// ❌ HTMLでは許容されるが、JSXではエラー
<img src="image.jpg">
<input type="text">
<br>

// ✅ 自己閉じタグ（self-closing）を使う
<img src="image.jpg" />
<input type="text" />
<br />
```

### 3. キャメルケース（camelCase）を使う

HTML属性はキャメルケースで書きます。

```typescript
// HTML
<div class="container" onclick="handleClick()">

// JSX
<div className="container" onClick={handleClick}>
```

**よく使う変換例**：
| HTML | JSX |
|------|-----|
| `class` | `className` |
| `for` | `htmlFor` |
| `onclick` | `onClick` |
| `onchange` | `onChange` |
| `tabindex` | `tabIndex` |

**理由**：`class`と`for`はJavaScriptの予約語のため

---

## JavaScriptの埋め込み

### 中括弧 `{}` で式を埋め込む

JSX内で `{}` を使うと、JavaScript式を評価できます。

#### 1. 変数の埋め込み

```typescript
function Greeting() {
  const name = "太郎";
  const age = 25;

  return (
    <div>
      <h1>こんにちは、{name}さん！</h1>
      <p>あなたは{age}歳です。</p>
    </div>
  );
}

// 出力：
// こんにちは、太郎さん！
// あなたは25歳です。
```

#### 2. 式の評価

```typescript
function Calculator() {
  const a = 10;
  const b = 20;

  return (
    <div>
      <p>{a} + {b} = {a + b}</p>
      <p>{a} × {b} = {a * b}</p>
    </div>
  );
}

// 出力：
// 10 + 20 = 30
// 10 × 20 = 200
```

#### 3. 関数呼び出し

```typescript
function formatDate(date: Date): string {
  return date.toLocaleDateString('ja-JP');
}

function App() {
  return (
    <div>
      <p>今日は{formatDate(new Date())}です</p>
    </div>
  );
}

// 出力：
// 今日は2024/1/28です
```

#### 4. テンプレートリテラル

```typescript
function UserCard() {
  const firstName = "太郎";
  const lastName = "山田";

  return (
    <div>
      <h2>{`${lastName} ${firstName}`}</h2>
      <p>フルネーム: {`${lastName} ${firstName}さん`}</p>
    </div>
  );
}

// 出力：
// 山田 太郎
// フルネーム: 山田 太郎さん
```

### 重要な制約

`{}` 内には**式（expression）**のみ書けます。**文（statement）**は書けません。

```typescript
// ❌ エラー：if文は式ではない
<div>
  {if (isLoggedIn) { "ログイン済み" }}
</div>

// ✅ 正しい：三項演算子を使う（式）
<div>
  {isLoggedIn ? "ログイン済み" : "未ログイン"}
</div>

// ❌ エラー：for文は式ではない
<ul>
  {for (let i = 0; i < 5; i++) { <li>{i}</li> }}
</ul>

// ✅ 正しい：map関数を使う（式）
<ul>
  {[0, 1, 2, 3, 4].map(i => <li key={i}>{i}</li>)}
</ul>
```

---

## 属性（Attributes）

### 1. 文字列リテラル

```typescript
// ダブルクォート
<img src="logo.png" alt="ロゴ" />

// シングルクォート
<a href='https://example.com'>リンク</a>
```

### 2. JavaScript式

```typescript
function Avatar() {
  const imageUrl = "https://example.com/avatar.jpg";
  const size = 100;
  const userName = "太郎";

  return (
    <img
      src={imageUrl}
      width={size}
      height={size}
      alt={`${userName}のアバター`}
    />
  );
}
```

### 3. 真偽値属性

```typescript
// disabled、checked、readOnly などは真偽値
<button disabled={true}>無効なボタン</button>
<button disabled={false}>有効なボタン</button>

// {true} は省略可能
<button disabled>無効なボタン</button>

// {false} の場合は属性自体を省略
<button>有効なボタン</button>
```

### 4. スタイル属性

スタイルは**オブジェクト**で指定し、プロパティ名はキャメルケースです。

```typescript
function StyledBox() {
  const boxStyle = {
    backgroundColor: 'lightblue',  // background-color → backgroundColor
    fontSize: '20px',               // font-size → fontSize
    padding: '10px',
    borderRadius: '8px'             // border-radius → borderRadius
  };

  return (
    <div style={boxStyle}>
      スタイル付きボックス
    </div>
  );
}

// インラインでも書ける（二重中括弧に注意）
<div style={{ color: 'red', fontSize: '24px' }}>
  赤い文字
</div>
```

**二重中括弧 `{{ }}` の意味**：
- 外側の `{}`：JavaScript式の開始
- 内側の `{}`：オブジェクトリテラル

---

## 条件分岐

### 1. 三項演算子（最も一般的）

```typescript
function LoginButton() {
  const isLoggedIn = false;

  return (
    <div>
      {isLoggedIn ? (
        <button>ログアウト</button>
      ) : (
        <button>ログイン</button>
      )}
    </div>
  );
}
```

### 2. 論理AND演算子（`&&`）

条件が**true**のときだけ表示したい場合に使います。

```typescript
function Notification() {
  const hasNewMessages = true;
  const messageCount = 5;

  return (
    <div>
      <h1>メッセージ</h1>
      {hasNewMessages && (
        <div className="notification">
          {messageCount}件の新着メッセージがあります
        </div>
      )}
    </div>
  );
}
```

**注意**：`false`、`null`、`undefined`はレンダリングされませんが、`0`はレンダリングされます。

```typescript
function Counter() {
  const count = 0;

  return (
    <div>
      {/* ❌ 0が表示されてしまう */}
      {count && <p>カウント: {count}</p>}

      {/* ✅ 正しい：明示的に比較 */}
      {count > 0 && <p>カウント: {count}</p>}
    </div>
  );
}
```

### 3. 複数条件（if-else if-else）

```typescript
function UserStatus({ status }: { status: 'online' | 'offline' | 'away' }) {
  return (
    <div>
      {status === 'online' ? (
        <span className="status-online">オンライン</span>
      ) : status === 'away' ? (
        <span className="status-away">離席中</span>
      ) : (
        <span className="status-offline">オフライン</span>
      )}
    </div>
  );
}
```

### 4. 変数に代入（複雑な条件）

複雑な条件分岐は、変数に代入してからレンダリングすると読みやすくなります。

```typescript
function UserGreeting({ user }: { user: { name: string; isAdmin: boolean } | null }) {
  let content;

  if (!user) {
    content = <p>ゲストユーザー</p>;
  } else if (user.isAdmin) {
    content = <p>管理者: {user.name}さん</p>;
  } else {
    content = <p>ユーザー: {user.name}さん</p>;
  }

  return <div>{content}</div>;
}
```

---

## リストのレンダリング

### 1. map関数を使う

配列を表示するには、`map()`関数を使います。

```typescript
function FruitList() {
  const fruits = ['りんご', 'バナナ', 'オレンジ'];

  return (
    <ul>
      {fruits.map((fruit, index) => (
        <li key={index}>{fruit}</li>
      ))}
    </ul>
  );
}

// 出力：
// • りんご
// • バナナ
// • オレンジ
```

### 2. keyプロパティ（重要）

**key**は、Reactがリストの各要素を識別するために必要です。

```typescript
// ❌ keyがないと警告が出る
<ul>
  {fruits.map(fruit => <li>{fruit}</li>)}
</ul>
// Warning: Each child in a list should have a unique "key" prop.

// ✅ 正しい：keyを指定
<ul>
  {fruits.map((fruit, index) => (
    <li key={index}>{fruit}</li>
  ))}
</ul>
```

**keyの選び方**：
1. **ユニークなID**（最優先）：`<li key={item.id}>`
2. **インデックス**（変更がない場合のみ）：`<li key={index}>`
3. **コンテンツ**（一時的な使用のみ）：`<li key={item.name}>`

```typescript
// ✅ ベストプラクティス：IDをkeyに使う
type Todo = {
  id: string;
  text: string;
  completed: boolean;
};

function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <ul>
      {todos.map(todo => (
        <li key={todo.id}>
          {todo.text}
        </li>
      ))}
    </ul>
  );
}
```

### 3. オブジェクトの配列

```typescript
type User = {
  id: number;
  name: string;
  email: string;
};

function UserList() {
  const users: User[] = [
    { id: 1, name: '太郎', email: 'taro@example.com' },
    { id: 2, name: '花子', email: 'hanako@example.com' },
    { id: 3, name: '次郎', email: 'jiro@example.com' }
  ];

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>
          <strong>{user.name}</strong> - {user.email}
        </li>
      ))}
    </ul>
  );
}
```

### 4. フィルタリングと組み合わせ

```typescript
function ActiveTodoList({ todos }: { todos: Todo[] }) {
  return (
    <ul>
      {todos
        .filter(todo => !todo.completed)  // 未完了のみ
        .map(todo => (
          <li key={todo.id}>{todo.text}</li>
        ))}
    </ul>
  );
}
```

---

## フラグメント

### 問題：余計なdivの増加

複数要素を返す場合、通常は`<div>`で囲みますが、これはDOMに余計な要素を追加します。

```typescript
// ❌ 余計なdivが増える
function Table() {
  return (
    <table>
      <tbody>
        <tr>
          <Columns />
        </tr>
      </tbody>
    </table>
  );
}

function Columns() {
  return (
    <div>  {/* この<div>はtableの構造を壊す */}
      <td>列1</td>
      <td>列2</td>
    </div>
  );
}
```

### 解決策：Fragment

**Fragment**を使うと、余計なDOMノードを追加せずに複数要素をグループ化できます。

```typescript
import { Fragment } from 'react';

// ✅ 方法1：<Fragment>を使う
function Columns() {
  return (
    <Fragment>
      <td>列1</td>
      <td>列2</td>
    </Fragment>
  );
}

// ✅ 方法2：短縮構文 <> を使う（最も一般的）
function Columns() {
  return (
    <>
      <td>列1</td>
      <td>列2</td>
    </>
  );
}
```

### Fragmentにkeyを指定

リスト内でFragmentを使う場合、keyが必要です。この場合は短縮構文は使えません。

```typescript
function DescriptionList({ items }: { items: Array<{ term: string; desc: string }> }) {
  return (
    <dl>
      {items.map(item => (
        <Fragment key={item.term}>
          <dt>{item.term}</dt>
          <dd>{item.desc}</dd>
        </Fragment>
      ))}
    </dl>
  );
}
```

---

## よくある間違い

### 間違い1：複数のルート要素

❌ **問題**：
```typescript
function Component() {
  return (
    <h1>タイトル</h1>
    <p>本文</p>
  );
}
// エラー: Adjacent JSX elements must be wrapped in an enclosing tag.
```

✅ **解決策**：
```typescript
function Component() {
  return (
    <>
      <h1>タイトル</h1>
      <p>本文</p>
    </>
  );
}
```

### 間違い2：classを使う

❌ **問題**：
```typescript
<div class="container">  // class は予約語
```

✅ **解決策**：
```typescript
<div className="container">  // className を使う
```

### 間違い3：スタイルを文字列で指定

❌ **問題**：
```typescript
<div style="color: red; font-size: 20px;">
```

✅ **解決策**：
```typescript
<div style={{ color: 'red', fontSize: '20px' }}>
```

### 間違い4：閉じタグを忘れる

❌ **問題**：
```typescript
<img src="logo.png">
<input type="text">
```

✅ **解決策**：
```typescript
<img src="logo.png" />
<input type="text" />
```

### 間違い5：keyを指定しない

❌ **問題**：
```typescript
{items.map(item => <li>{item}</li>)}
// Warning: Each child in a list should have a unique "key" prop.
```

✅ **解決策**：
```typescript
{items.map((item, index) => <li key={index}>{item}</li>)}
```

### 間違い6：0の条件チェック

❌ **問題**：
```typescript
const count = 0;
return <div>{count && <p>カウント: {count}</p>}</div>;
// 出力: 0 (0が表示されてしまう)
```

✅ **解決策**：
```typescript
const count = 0;
return <div>{count > 0 && <p>カウント: {count}</p>}</div>;
// 出力: (何も表示されない)
```

---

## 演習問題

### 問題1：ユーザープロフィール

**難易度**：初級

**課題**：
以下の情報を表示するコンポーネントを作成してください。
- 名前
- 年齢
- メールアドレス
- 自己紹介（ある場合のみ表示）

**ヒント**：
- オブジェクトで情報を管理
- 条件分岐で自己紹介を表示

**解答例**：
```typescript
type User = {
  name: string;
  age: number;
  email: string;
  bio?: string;  // オプショナル
};

function UserProfile() {
  const user: User = {
    name: '山田太郎',
    age: 28,
    email: 'taro@example.com',
    bio: 'Web開発が好きです。'
  };

  return (
    <div className="profile">
      <h1>{user.name}</h1>
      <p>年齢: {user.age}歳</p>
      <p>メール: {user.email}</p>
      {user.bio && (
        <div className="bio">
          <h2>自己紹介</h2>
          <p>{user.bio}</p>
        </div>
      )}
    </div>
  );
}
```

### 問題2：買い物リスト

**難易度**：中級

**課題**：
以下の機能を持つ買い物リストを作成してください。
- アイテム名と価格を表示
- 合計金額を計算して表示
- 1000円以上のアイテムは赤文字で表示

**ヒント**：
- `reduce()`で合計を計算
- スタイルを条件で変える

**解答例**：
```typescript
type Item = {
  id: number;
  name: string;
  price: number;
};

function ShoppingList() {
  const items: Item[] = [
    { id: 1, name: 'りんご', price: 200 },
    { id: 2, name: '牛乳', price: 150 },
    { id: 3, name: 'パソコン', price: 80000 },
    { id: 4, name: 'パン', price: 120 }
  ];

  const total = items.reduce((sum, item) => sum + item.price, 0);

  return (
    <div>
      <h1>買い物リスト</h1>
      <ul>
        {items.map(item => (
          <li
            key={item.id}
            style={{
              color: item.price >= 1000 ? 'red' : 'black',
              fontWeight: item.price >= 1000 ? 'bold' : 'normal'
            }}
          >
            {item.name}: ¥{item.price.toLocaleString()}
          </li>
        ))}
      </ul>
      <p className="total">合計: ¥{total.toLocaleString()}</p>
    </div>
  );
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ JSXの基本概念と仕組み
- ✅ JavaScript式の埋め込み（`{}`）
- ✅ 属性の書き方（className、style など）
- ✅ 条件分岐（三項演算子、&&演算子）
- ✅ リストのレンダリング（map、key）
- ✅ Fragment の使い方

### 次に学ぶべきガイド

1. **[04-components-intro.md](./04-components-intro.md)**
   - コンポーネントの分割
   - Propsの受け渡し
   - コンポーネント設計の基礎

2. **[05-props-basics.md](./05-props-basics.md)**
   - Propsの詳細
   - TypeScriptでの型定義
   - デフォルト値

### 関連リソース

#### 公式ドキュメント
- [React: Writing Markup with JSX](https://react.dev/learn/writing-markup-with-jsx)
- [React: JavaScript in JSX with Curly Braces](https://react.dev/learn/javascript-in-jsx-with-curly-braces)
- [Babel REPL](https://babeljs.io/repl)：JSXの変換を確認できる

---

**次のガイド**：[04-components-intro.md](./04-components-intro.md)

**前のガイド**：[02-setup-environment.md](./02-setup-environment.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
