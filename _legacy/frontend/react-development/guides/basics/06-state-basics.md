# State（状態）基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [Stateとは何か](#stateとは何か)
4. [useStateフックの基本](#usestateフックの基本)
5. [状態の更新](#状態の更新)
6. [複数の状態管理](#複数の状態管理)
7. [オブジェクトと配列の状態](#オブジェクトと配列の状態)
8. [StateとPropsの違い](#stateとpropsの違い)
9. [よくある間違い](#よくある間違い)
10. [演習問題](#演習問題)
11. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Stateの基本概念と必要性
- `useState`フックの使い方
- 状態の更新方法とルール
- 複数の状態の管理
- オブジェクトと配列の状態管理
- StateとPropsの違い

### なぜ重要か

**State（状態）**は、Reactコンポーネントが**動的に変化するデータ**を保持するための仕組みです。Stateを理解することで：
- **インタラクティブなUI**：ユーザーの操作に応じて画面が変化
- **データの保持**：コンポーネント内でデータを保存・更新
- **再レンダリング**：状態が変わると自動的に画面が更新される

### 学習時間の目安

- このガイドの読了：40〜50分
- 演習問題を含めた完全理解：2〜3時間

---

## 前提知識

### 必要な知識

1. **Props基礎**：
   - [05-props-basics.md](./05-props-basics.md) を完了していること

2. **JavaScript ES6**：
   - 配列の分割代入（`const [a, b] = [1, 2]`）
   - スプレッド構文（`[...array]`、`{...object}`）

3. **非同期処理の概念**：
   - 同期と非同期の違い

---

## Stateとは何か

### 定義

**State（状態）**は、**コンポーネントが保持する動的なデータ**です。

```typescript
import { useState } from 'react';

function Counter() {
  // countが状態（State）
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>カウント: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

### なぜStateが必要なのか

#### Before：通常の変数（動作しない例）

```typescript
function Counter() {
  let count = 0;  // 通常の変数

  const increment = () => {
    count = count + 1;  // 値は変わるが...
    console.log(count);  // コンソールには表示される
  };

  return (
    <div>
      <p>カウント: {count}</p>  {/* 画面は更新されない！ */}
      <button onClick={increment}>+1</button>
    </div>
  );
}
```

**問題**：
- 変数の値は変わるが、**Reactは変更を検知できない**
- 再レンダリングが起こらない
- 画面は更新されない

#### After：useState を使う（正しい例）

```typescript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);  // State

  const increment = () => {
    setCount(count + 1);  // Stateを更新
    // Reactが変更を検知 → 自動的に再レンダリング
  };

  return (
    <div>
      <p>カウント: {count}</p>  {/* 画面が更新される！ */}
      <button onClick={increment}>+1</button>
    </div>
  );
}
```

**解決**：
- `useState` で状態を管理
- `setCount` で状態を更新
- Reactが自動的に再レンダリング
- 画面が更新される

---

## useStateフックの基本

### 構文

```typescript
const [state, setState] = useState(initialValue);
```

- `state`：現在の状態値
- `setState`：状態を更新する関数
- `initialValue`：初期値

### 基本的な例

```typescript
import { useState } from 'react';

function Example() {
  // 数値の状態
  const [count, setCount] = useState(0);

  // 文字列の状態
  const [name, setName] = useState('');

  // 真偽値の状態
  const [isVisible, setIsVisible] = useState(true);

  return <div>...</div>;
}
```

### useState の命名規則

慣習的に、以下のように命名します：

```typescript
const [value, setValue] = useState(initialValue);
```

**具体例**：
- `const [count, setCount] = useState(0);`
- `const [name, setName] = useState('');`
- `const [isOpen, setIsOpen] = useState(false);`
- `const [todos, setTodos] = useState([]);`

**ルール**：
- 状態変数：名詞（`count`、`name`、`isOpen`）
- 更新関数：`set` + 状態変数名（`setCount`、`setName`、`setIsOpen`）

---

## 状態の更新

### 1. 直接値を指定

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
      <button onClick={() => setCount(count - 1)}>-1</button>
      <button onClick={() => setCount(0)}>リセット</button>
    </div>
  );
}
```

### 2. 関数を渡す（推奨）

**現在の状態に基づいて更新する場合**は、関数を渡す方が安全です。

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  return (
    <div>
      <p>{count}</p>
      {/* 関数を渡す（prevCount は現在の値） */}
      <button onClick={() => setCount(prevCount => prevCount + 1)}>
        +1
      </button>
    </div>
  );
}
```

**なぜ関数を渡すのか？**

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  const handleMultipleClicks = () => {
    // ❌ 間違い：3回呼んでも1しか増えない
    setCount(count + 1);  // count = 0 + 1 = 1
    setCount(count + 1);  // count = 0 + 1 = 1
    setCount(count + 1);  // count = 0 + 1 = 1
    // 結果：count = 1

    // ✅ 正しい：3回呼ぶと3増える
    setCount(prev => prev + 1);  // prev = 0, 新しい値 = 1
    setCount(prev => prev + 1);  // prev = 1, 新しい値 = 2
    setCount(prev => prev + 1);  // prev = 2, 新しい値 = 3
    // 結果：count = 3
  };

  return <button onClick={handleMultipleClicks}>+3</button>;
}
```

### 3. 状態の更新は非同期

**重要**：`setState` は**非同期**で実行されます。

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    setCount(count + 1);
    console.log(count);  // まだ古い値が表示される！
  };

  return <button onClick={increment}>+1</button>;
}
```

**解決策**：更新後の値は、次のレンダリング時に反映されます。

```typescript
function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => {
    const newCount = count + 1;
    setCount(newCount);
    console.log(newCount);  // 新しい値を使う
  };

  return <button onClick={increment}>+1</button>;
}
```

---

## 複数の状態管理

### 複数のuseStateを使う

1つのコンポーネントで複数の状態を管理できます。

```typescript
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = () => {
    setIsLoading(true);
    setError('');

    // ログイン処理...
    if (email === '' || password === '') {
      setError('メールアドレスとパスワードを入力してください');
      setIsLoading(false);
      return;
    }

    // API呼び出しなど...
    setIsLoading(false);
  };

  return (
    <form>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="メールアドレス"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="パスワード"
      />
      {error && <p className="error">{error}</p>}
      <button onClick={handleSubmit} disabled={isLoading}>
        {isLoading ? 'ログイン中...' : 'ログイン'}
      </button>
    </form>
  );
}
```

### 関連する状態はオブジェクトにまとめる

```typescript
type FormState = {
  email: string;
  password: string;
  isLoading: boolean;
  error: string;
};

function LoginForm() {
  const [form, setForm] = useState<FormState>({
    email: '',
    password: '',
    isLoading: false,
    error: ''
  });

  const handleSubmit = () => {
    setForm(prev => ({ ...prev, isLoading: true, error: '' }));

    if (form.email === '' || form.password === '') {
      setForm(prev => ({
        ...prev,
        error: 'メールアドレスとパスワードを入力してください',
        isLoading: false
      }));
      return;
    }

    // ログイン処理...
    setForm(prev => ({ ...prev, isLoading: false }));
  };

  return (
    <form>
      <input
        type="email"
        value={form.email}
        onChange={(e) => setForm(prev => ({ ...prev, email: e.target.value }))}
      />
      <input
        type="password"
        value={form.password}
        onChange={(e) => setForm(prev => ({ ...prev, password: e.target.value }))}
      />
      {form.error && <p className="error">{form.error}</p>}
      <button onClick={handleSubmit} disabled={form.isLoading}>
        {form.isLoading ? 'ログイン中...' : 'ログイン'}
      </button>
    </form>
  );
}
```

---

## オブジェクトと配列の状態

### オブジェクトの状態更新

**重要**：オブジェクトは**不変（immutable）に更新**する必要があります。

```typescript
type User = {
  name: string;
  age: number;
  email: string;
};

function UserProfile() {
  const [user, setUser] = useState<User>({
    name: '山田太郎',
    age: 28,
    email: 'taro@example.com'
  });

  // ❌ 間違い：直接変更
  const updateName = (newName: string) => {
    user.name = newName;  // これはNG！
    setUser(user);
  };

  // ✅ 正しい：新しいオブジェクトを作成
  const updateName = (newName: string) => {
    setUser({
      ...user,        // 既存のプロパティをコピー
      name: newName   // nameだけ上書き
    });
  };

  // ✅ 関数形式（推奨）
  const updateAge = (newAge: number) => {
    setUser(prev => ({
      ...prev,
      age: newAge
    }));
  };

  return (
    <div>
      <p>名前: {user.name}</p>
      <p>年齢: {user.age}歳</p>
      <p>メール: {user.email}</p>
      <button onClick={() => updateName('鈴木次郎')}>
        名前を変更
      </button>
      <button onClick={() => updateAge(user.age + 1)}>
        年齢+1
      </button>
    </div>
  );
}
```

### 配列の状態更新

配列も**不変に更新**します。

```typescript
function TodoList() {
  const [todos, setTodos] = useState<string[]>(['買い物', '掃除']);

  // ❌ 間違い：直接変更
  const addTodoWrong = (newTodo: string) => {
    todos.push(newTodo);  // これはNG！
    setTodos(todos);
  };

  // ✅ 正しい：新しい配列を作成
  const addTodo = (newTodo: string) => {
    setTodos([...todos, newTodo]);
  };

  // ✅ 削除
  const removeTodo = (index: number) => {
    setTodos(todos.filter((_, i) => i !== index));
  };

  // ✅ 更新
  const updateTodo = (index: number, newText: string) => {
    setTodos(todos.map((todo, i) =>
      i === index ? newText : todo
    ));
  };

  return (
    <ul>
      {todos.map((todo, index) => (
        <li key={index}>
          {todo}
          <button onClick={() => removeTodo(index)}>削除</button>
        </li>
      ))}
      <button onClick={() => addTodo('新しいTODO')}>
        追加
      </button>
    </ul>
  );
}
```

### 配列の操作パターン

```typescript
// 追加（末尾）
setTodos([...todos, newTodo]);

// 追加（先頭）
setTodos([newTodo, ...todos]);

// 削除
setTodos(todos.filter((_, i) => i !== indexToRemove));

// 更新
setTodos(todos.map((todo, i) =>
  i === indexToUpdate ? newValue : todo
));

// 並び替え
setTodos([...todos].sort());

// クリア
setTodos([]);
```

---

## StateとPropsの違い

### 比較表

| 項目 | State | Props |
|------|-------|-------|
| **定義** | コンポーネント内部のデータ | 親から渡されるデータ |
| **変更** | 変更可能（setStateで） | 読み取り専用 |
| **管理** | コンポーネント自身が管理 | 親コンポーネントが管理 |
| **使用** | `useState`フック | 関数の引数 |
| **再レンダリング** | 変更すると再レンダリング | 変更されると再レンダリング |

### 実例

```typescript
// 親コンポーネント
function Parent() {
  const [count, setCount] = useState(0);  // State

  return (
    <div>
      <p>親のカウント: {count}</p>
      <button onClick={() => setCount(count + 1)}>親で+1</button>
      {/* countをPropsとして子に渡す */}
      <Child count={count} />
    </div>
  );
}

// 子コンポーネント
type ChildProps = {
  count: number;  // Props
};

function Child({ count }: ChildProps) {
  const [localCount, setLocalCount] = useState(0);  // State

  return (
    <div>
      <p>親から受け取ったカウント: {count}</p>
      <p>子のローカルカウント: {localCount}</p>
      <button onClick={() => setLocalCount(localCount + 1)}>
        子で+1
      </button>
    </div>
  );
}
```

---

## よくある間違い

### 間違い1：状態を直接変更

❌ **問題**：
```typescript
const [count, setCount] = useState(0);
count = count + 1;  // 直接変更
```

✅ **解決策**：
```typescript
const [count, setCount] = useState(0);
setCount(count + 1);  // setCountを使う
```

### 間違い2：オブジェクトの直接変更

❌ **問題**：
```typescript
const [user, setUser] = useState({ name: '太郎', age: 25 });
user.age = 26;  // 直接変更
setUser(user);
```

✅ **解決策**：
```typescript
const [user, setUser] = useState({ name: '太郎', age: 25 });
setUser({ ...user, age: 26 });  // 新しいオブジェクト
```

### 間違い3：配列のpush/pop

❌ **問題**：
```typescript
const [todos, setTodos] = useState(['買い物']);
todos.push('掃除');  // 直接変更
setTodos(todos);
```

✅ **解決策**：
```typescript
const [todos, setTodos] = useState(['買い物']);
setTodos([...todos, '掃除']);  // 新しい配列
```

### 間違い4：setState直後に値を参照

❌ **問題**：
```typescript
const [count, setCount] = useState(0);
setCount(count + 1);
console.log(count);  // まだ古い値！
```

✅ **解決策**：
```typescript
const [count, setCount] = useState(0);
const newCount = count + 1;
setCount(newCount);
console.log(newCount);  // 新しい値を参照
```

### 間違い5：初期値に関数実行の結果を直接書く

❌ **問題**：
```typescript
// 毎回expensiveCalculation()が実行される（無駄）
const [value, setValue] = useState(expensiveCalculation());
```

✅ **解決策**：
```typescript
// 初回のみ実行（関数を渡す）
const [value, setValue] = useState(() => expensiveCalculation());
```

---

## 演習問題

### 問題1：カウンターの拡張

**難易度**：初級

**課題**：
以下の機能を持つカウンターを作成してください。
- +1ボタン
- -1ボタン
- +10ボタン
- リセットボタン
- カウントが0未満にならないようにする

**解答例**：
```typescript
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);

  const increment = () => setCount(prev => prev + 1);
  const decrement = () => setCount(prev => Math.max(0, prev - 1));
  const incrementBy10 = () => setCount(prev => prev + 10);
  const reset = () => setCount(0);

  return (
    <div>
      <h1>カウント: {count}</h1>
      <button onClick={increment}>+1</button>
      <button onClick={decrement}>-1</button>
      <button onClick={incrementBy10}>+10</button>
      <button onClick={reset}>リセット</button>
    </div>
  );
}
```

### 問題2：TODOリスト

**難易度**：中級

**課題**：
以下の機能を持つTODOリストを作成してください。
- TODOを追加
- TODOを削除
- 完了/未完了の切り替え
- 完了済みTODOの数を表示

**解答例**：
```typescript
import { useState } from 'react';

type Todo = {
  id: number;
  text: string;
  completed: boolean;
};

function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [input, setInput] = useState('');

  const addTodo = () => {
    if (input.trim()) {
      setTodos([
        ...todos,
        { id: Date.now(), text: input, completed: false }
      ]);
      setInput('');
    }
  };

  const deleteTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  const completedCount = todos.filter(todo => todo.completed).length;

  return (
    <div>
      <h1>TODOリスト</h1>
      <div>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && addTodo()}
          placeholder="新しいTODO"
        />
        <button onClick={addTodo}>追加</button>
      </div>

      <p>完了: {completedCount} / {todos.length}</p>

      <ul>
        {todos.map(todo => (
          <li
            key={todo.id}
            style={{
              textDecoration: todo.completed ? 'line-through' : 'none'
            }}
          >
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span>{todo.text}</span>
            <button onClick={() => deleteTodo(todo.id)}>削除</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ Stateの基本概念と必要性
- ✅ `useState`フックの使い方
- ✅ 状態の更新方法とルール
- ✅ 複数の状態の管理
- ✅ オブジェクトと配列の不変更新
- ✅ StateとPropsの違い

### 次に学ぶべきガイド

1. **[07-events-lists.md](./07-events-lists.md)**
   - イベント処理の詳細
   - フォームの扱い方
   - ユーザーインタラクション

2. **[hooks-mastery.md](../hooks-mastery.md)**
   - useEffectなど他のフック
   - カスタムフック
   - 高度な状態管理

### 関連リソース

#### 公式ドキュメント
- [React: State: A Component's Memory](https://react.dev/learn/state-a-components-memory)
- [React: useState](https://react.dev/reference/react/useState)

---

**次のガイド**：[07-events-lists.md](./07-events-lists.md)

**前のガイド**：[05-props-basics.md](./05-props-basics.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
