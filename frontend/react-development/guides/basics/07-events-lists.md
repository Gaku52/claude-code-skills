# イベント処理とリスト表示 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [前提知識](#前提知識)
3. [イベントハンドリングの基礎](#イベントハンドリングの基礎)
4. [様々なイベントタイプ](#様々なイベントタイプ)
5. [フォームの扱い方](#フォームの扱い方)
6. [イベントオブジェクト](#イベントオブジェクト)
7. [イベントハンドラへの引数渡し](#イベントハンドラへの引数渡し)
8. [リストのレンダリング詳細](#リストのレンダリング詳細)
9. [keyの重要性](#keyの重要性)
10. [よくある間違い](#よくある間違い)
11. [演習問題](#演習問題)
12. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

このガイドでは、以下を学びます：
- Reactでのイベント処理の基本
- クリック、入力、キーボードなど様々なイベント
- フォームの制御とバリデーション
- イベントオブジェクトの活用
- リストの効率的なレンダリング
- keyプロップの重要性

### なぜ重要か

**イベント処理**と**リスト表示**は、インタラクティブなUIを作るための必須スキルです。これらを理解することで：
- **ユーザーインタラクション**：ボタンクリック、入力、ドラッグなどに対応
- **動的なUI**：ユーザーの操作に応じて画面が変化
- **効率的なレンダリング**：大量のデータを高速に表示

### 学習時間の目安

- このガイドの読了：40〜50分
- 演習問題を含めた完全理解：2〜3時間

---

## 前提知識

### 必要な知識

1. **State基礎**：
   - [06-state-basics.md](./06-state-basics.md) を完了していること

2. **JavaScript基礎**：
   - イベント（`addEventListener`、`onClick` など）
   - 配列メソッド（`map`、`filter`、`find`）

---

## イベントハンドリングの基礎

### HTMLとReactの違い

#### HTML（従来の方法）

```html
<!-- HTML -->
<button onclick="handleClick()">クリック</button>

<script>
function handleClick() {
  alert('クリックされました');
}
</script>
```

#### React

```typescript
function Button() {
  const handleClick = () => {
    alert('クリックされました');
  };

  return <button onClick={handleClick}>クリック</button>;
}
```

**違い**：
- React：`onClick`（キャメルケース）
- HTML：`onclick`（小文字）
- React：関数を渡す（`{handleClick}`）
- HTML：文字列を渡す（`"handleClick()"`）

### 基本的なイベントハンドラ

```typescript
function Button() {
  // イベントハンドラ関数を定義
  const handleClick = () => {
    console.log('ボタンがクリックされました');
  };

  return (
    <button onClick={handleClick}>
      クリック
    </button>
  );
}
```

### インライン関数（簡単な処理の場合）

```typescript
function Button() {
  return (
    <button onClick={() => alert('クリック！')}>
      クリック
    </button>
  );
}
```

**注意**：複雑な処理は別途関数を定義する方が読みやすいです。

---

## 様々なイベントタイプ

### 1. クリックイベント

```typescript
function ClickExample() {
  const [count, setCount] = useState(0);

  const handleClick = () => {
    setCount(count + 1);
  };

  const handleDoubleClick = () => {
    setCount(0);
  };

  return (
    <div>
      <p>カウント: {count}</p>
      <button onClick={handleClick}>
        クリック
      </button>
      <button onDoubleClick={handleDoubleClick}>
        ダブルクリックでリセット
      </button>
    </div>
  );
}
```

### 2. 入力イベント

```typescript
function InputExample() {
  const [text, setText] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setText(e.target.value);
  };

  return (
    <div>
      <input
        type="text"
        value={text}
        onChange={handleChange}
        placeholder="入力してください"
      />
      <p>入力内容: {text}</p>
      <p>文字数: {text.length}</p>
    </div>
  );
}
```

### 3. キーボードイベント

```typescript
function KeyboardExample() {
  const [text, setText] = useState('');
  const [message, setMessage] = useState('');

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      setMessage(`メッセージ送信: ${text}`);
      setText('');
    } else if (e.key === 'Escape') {
      setText('');
    }
  };

  return (
    <div>
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Enterで送信、Escでクリア"
      />
      {message && <p>{message}</p>}
    </div>
  );
}
```

### 4. マウスイベント

```typescript
function MouseExample() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    setPosition({ x: e.clientX, y: e.clientY });
  };

  return (
    <div>
      <div
        onMouseMove={handleMouseMove}
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        style={{
          width: '300px',
          height: '300px',
          border: '2px solid black',
          backgroundColor: isHovering ? 'lightblue' : 'white'
        }}
      >
        マウスを動かしてください
      </div>
      <p>位置: X: {position.x}, Y: {position.y}</p>
    </div>
  );
}
```

### 5. フォーカスイベント

```typescript
function FocusExample() {
  const [isFocused, setIsFocused] = useState(false);

  return (
    <div>
      <input
        type="text"
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        placeholder="フォーカスしてください"
        style={{
          borderColor: isFocused ? 'blue' : 'gray',
          borderWidth: '2px'
        }}
      />
      <p>{isFocused ? 'フォーカス中' : 'フォーカスなし'}</p>
    </div>
  );
}
```

---

## フォームの扱い方

### 制御されたコンポーネント（Controlled Components）

Reactでは、フォームの値を**Stateで管理**します。

```typescript
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();  // デフォルトの送信動作を防ぐ
    console.log('Email:', email);
    console.log('Password:', password);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label htmlFor="email">メールアドレス:</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
      </div>

      <div>
        <label htmlFor="password">パスワード:</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
      </div>

      <button type="submit">ログイン</button>
    </form>
  );
}
```

### 複数の入力を扱う

```typescript
type FormData = {
  username: string;
  email: string;
  age: number;
  bio: string;
};

function RegistrationForm() {
  const [formData, setFormData] = useState<FormData>({
    username: '',
    email: '',
    age: 0,
    bio: ''
  });

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('送信データ:', formData);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        name="username"
        value={formData.username}
        onChange={handleChange}
        placeholder="ユーザー名"
      />
      <input
        name="email"
        type="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="メールアドレス"
      />
      <input
        name="age"
        type="number"
        value={formData.age}
        onChange={handleChange}
        placeholder="年齢"
      />
      <textarea
        name="bio"
        value={formData.bio}
        onChange={handleChange}
        placeholder="自己紹介"
      />
      <button type="submit">登録</button>
    </form>
  );
}
```

### バリデーション

```typescript
function ValidatedForm() {
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const validateEmail = (value: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!value) {
      return 'メールアドレスを入力してください';
    }
    if (!emailRegex.test(value)) {
      return '有効なメールアドレスを入力してください';
    }
    return '';
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setEmail(value);
    setError(validateEmail(value));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const validationError = validateEmail(email);
    if (validationError) {
      setError(validationError);
      return;
    }
    console.log('送信:', email);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={handleChange}
        placeholder="メールアドレス"
        style={{ borderColor: error ? 'red' : 'gray' }}
      />
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <button type="submit" disabled={!!error}>
        送信
      </button>
    </form>
  );
}
```

---

## イベントオブジェクト

### 合成イベント（SyntheticEvent）

Reactは、ブラウザのネイティブイベントをラップした**合成イベント（SyntheticEvent）**を提供します。

```typescript
function EventExample() {
  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    console.log('イベントタイプ:', e.type);           // "click"
    console.log('ターゲット要素:', e.currentTarget);   // <button>
    console.log('クリック位置:', e.clientX, e.clientY);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    console.log('押されたキー:', e.key);
    console.log('Shiftキー:', e.shiftKey);
    console.log('Ctrlキー:', e.ctrlKey);
  };

  return (
    <div>
      <button onClick={handleClick}>クリック</button>
      <input onKeyDown={handleKeyDown} placeholder="キーを押してください" />
    </div>
  );
}
```

### デフォルト動作の防止

```typescript
function PreventDefaultExample() {
  const handleLinkClick = (e: React.MouseEvent<HTMLAnchorElement>) => {
    e.preventDefault();  // リンクの遷移を防ぐ
    console.log('リンクがクリックされましたが、遷移しません');
  };

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();  // フォームの送信を防ぐ
    console.log('フォームが送信されましたが、ページはリロードされません');
  };

  return (
    <div>
      <a href="https://example.com" onClick={handleLinkClick}>
        クリック（遷移しない）
      </a>

      <form onSubmit={handleSubmit}>
        <button type="submit">送信</button>
      </form>
    </div>
  );
}
```

### イベントの伝播（バブリング）

```typescript
function BubblingExample() {
  const handleParentClick = () => {
    console.log('親要素がクリックされました');
  };

  const handleChildClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();  // 親へのイベント伝播を止める
    console.log('子要素がクリックされました');
  };

  return (
    <div onClick={handleParentClick} style={{ padding: '20px', border: '1px solid black' }}>
      親要素
      <button onClick={handleChildClick}>
        子要素（クリックしても親には伝わらない）
      </button>
    </div>
  );
}
```

---

## イベントハンドラへの引数渡し

### 方法1：アロー関数を使う

```typescript
function ButtonList() {
  const handleClick = (id: number) => {
    console.log(`ボタン ${id} がクリックされました`);
  };

  return (
    <div>
      <button onClick={() => handleClick(1)}>ボタン 1</button>
      <button onClick={() => handleClick(2)}>ボタン 2</button>
      <button onClick={() => handleClick(3)}>ボタン 3</button>
    </div>
  );
}
```

### 方法2：bindを使う（古い方法）

```typescript
function ButtonList() {
  const handleClick = (id: number) => {
    console.log(`ボタン ${id} がクリックされました`);
  };

  return (
    <div>
      <button onClick={handleClick.bind(null, 1)}>ボタン 1</button>
      <button onClick={handleClick.bind(null, 2)}>ボタン 2</button>
    </div>
  );
}
```

### 方法3：カリー化（高度な方法）

```typescript
function ButtonList() {
  const handleClick = (id: number) => () => {
    console.log(`ボタン ${id} がクリックされました`);
  };

  return (
    <div>
      <button onClick={handleClick(1)}>ボタン 1</button>
      <button onClick={handleClick(2)}>ボタン 2</button>
    </div>
  );
}
```

---

## リストのレンダリング詳細

### 基本的なリスト

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
```

### オブジェクトの配列

```typescript
type Product = {
  id: number;
  name: string;
  price: number;
  inStock: boolean;
};

function ProductList() {
  const products: Product[] = [
    { id: 1, name: 'ノートパソコン', price: 89800, inStock: true },
    { id: 2, name: 'マウス', price: 2980, inStock: true },
    { id: 3, name: 'キーボード', price: 5980, inStock: false }
  ];

  return (
    <div>
      {products.map(product => (
        <div key={product.id} className="product-card">
          <h3>{product.name}</h3>
          <p>¥{product.price.toLocaleString()}</p>
          <p>{product.inStock ? '在庫あり' : '在庫なし'}</p>
          <button disabled={!product.inStock}>
            購入する
          </button>
        </div>
      ))}
    </div>
  );
}
```

### フィルタリングとソート

```typescript
type Task = {
  id: number;
  text: string;
  completed: boolean;
  priority: number;
};

function TaskList() {
  const [tasks, setTasks] = useState<Task[]>([
    { id: 1, text: '買い物', completed: false, priority: 2 },
    { id: 2, text: '掃除', completed: true, priority: 1 },
    { id: 3, text: '料理', completed: false, priority: 3 }
  ]);

  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');

  const filteredTasks = tasks
    .filter(task => {
      if (filter === 'active') return !task.completed;
      if (filter === 'completed') return task.completed;
      return true;
    })
    .sort((a, b) => a.priority - b.priority);

  return (
    <div>
      <div>
        <button onClick={() => setFilter('all')}>すべて</button>
        <button onClick={() => setFilter('active')}>未完了</button>
        <button onClick={() => setFilter('completed')}>完了済み</button>
      </div>

      <ul>
        {filteredTasks.map(task => (
          <li key={task.id}>
            <input
              type="checkbox"
              checked={task.completed}
              onChange={() => {
                setTasks(tasks.map(t =>
                  t.id === task.id ? { ...t, completed: !t.completed } : t
                ));
              }}
            />
            <span style={{
              textDecoration: task.completed ? 'line-through' : 'none'
            }}>
              {task.text} (優先度: {task.priority})
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## keyの重要性

### なぜkeyが必要なのか

**key**は、Reactがリストの各要素を識別するために必要です。keyがないと、Reactは効率的に更新できません。

```typescript
// ❌ keyがない（警告が出る）
<ul>
  {items.map(item => <li>{item}</li>)}
</ul>
// Warning: Each child in a list should have a unique "key" prop.

// ✅ keyがある
<ul>
  {items.map((item, index) => (
    <li key={index}>{item}</li>
  ))}
</ul>
```

### keyの選び方

#### 1. ユニークなID（最優先）

```typescript
type User = {
  id: string;  // ユニークなID
  name: string;
};

const users: User[] = [
  { id: 'u1', name: '太郎' },
  { id: 'u2', name: '花子' }
];

// ✅ ベスト：ユニークなIDを使う
<ul>
  {users.map(user => (
    <li key={user.id}>{user.name}</li>
  ))}
</ul>
```

#### 2. インデックス（変更がない場合のみ）

```typescript
// ✅ OK：静的なリスト
const fruits = ['りんご', 'バナナ', 'オレンジ'];

<ul>
  {fruits.map((fruit, index) => (
    <li key={index}>{fruit}</li>
  ))}
</ul>

// ❌ NG：動的なリスト（追加・削除・並び替えがある場合）
<ul>
  {todos.map((todo, index) => (
    <li key={index}>  {/* インデックスは変わってしまう */}
      {todo.text}
    </li>
  ))}
</ul>
```

#### 3. keyにインデックスを使うと起こる問題

```typescript
function TodoList() {
  const [todos, setTodos] = useState(['買い物', '掃除', '料理']);

  const deleteTodo = (index: number) => {
    setTodos(todos.filter((_, i) => i !== index));
  };

  return (
    <ul>
      {todos.map((todo, index) => (
        // ❌ インデックスをkeyに使うと、削除時に問題が起こる
        <li key={index}>
          {todo}
          <button onClick={() => deleteTodo(index)}>削除</button>
        </li>
      ))}
    </ul>
  );
}
```

**問題**：
- 「掃除」を削除すると、インデックスが再割り当てされる
- Reactは「どの要素が削除されたか」を正しく判断できない
- 予期しない挙動やパフォーマンスの問題

**解決策**：
```typescript
type Todo = {
  id: string;  // ユニークなID
  text: string;
};

function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([
    { id: '1', text: '買い物' },
    { id: '2', text: '掃除' },
    { id: '3', text: '料理' }
  ]);

  const deleteTodo = (id: string) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  return (
    <ul>
      {todos.map(todo => (
        // ✅ IDをkeyに使う
        <li key={todo.id}>
          {todo.text}
          <button onClick={() => deleteTodo(todo.id)}>削除</button>
        </li>
      ))}
    </ul>
  );
}
```

---

## よくある間違い

### 間違い1：イベントハンドラを実行してしまう

❌ **問題**：
```typescript
// 即座に実行されてしまう
<button onClick={handleClick()}>クリック</button>
```

✅ **解決策**：
```typescript
// 関数を渡す（実行しない）
<button onClick={handleClick}>クリック</button>

// または、アロー関数でラップ
<button onClick={() => handleClick()}>クリック</button>
```

### 間違い2：e.preventDefaultを忘れる

❌ **問題**：
```typescript
const handleSubmit = () => {
  // フォームが送信されてページがリロードされる
  console.log('送信');
};

<form onSubmit={handleSubmit}>...</form>
```

✅ **解決策**：
```typescript
const handleSubmit = (e: React.FormEvent) => {
  e.preventDefault();  // デフォルト動作を防ぐ
  console.log('送信');
};
```

### 間違い3：インラインオブジェクトをkeyに使う

❌ **問題**：
```typescript
// 毎回新しいオブジェクトが作成される
{items.map(item => (
  <div key={{ id: item.id }}>  {/* オブジェクトは毎回異なる */}
    {item.name}
  </div>
))}
```

✅ **解決策**：
```typescript
// プリミティブな値を使う
{items.map(item => (
  <div key={item.id}>
    {item.name}
  </div>
))}
```

---

## 演習問題

### 問題1：検索機能付きリスト

**難易度**：中級

**課題**：
以下の機能を持つユーザーリストを作成してください。
- ユーザーのリストを表示
- 検索ボックスで名前をフィルタリング
- リアルタイムで検索結果を表示

**解答例**：
```typescript
import { useState } from 'react';

type User = {
  id: number;
  name: string;
  email: string;
};

function UserList() {
  const [users] = useState<User[]>([
    { id: 1, name: '山田太郎', email: 'taro@example.com' },
    { id: 2, name: '鈴木花子', email: 'hanako@example.com' },
    { id: 3, name: '田中次郎', email: 'jiro@example.com' }
  ]);

  const [searchTerm, setSearchTerm] = useState('');

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="ユーザーを検索..."
      />

      <p>検索結果: {filteredUsers.length}件</p>

      <ul>
        {filteredUsers.map(user => (
          <li key={user.id}>
            <strong>{user.name}</strong> - {user.email}
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### 問題2：フォーム付きTODOリスト

**難易度**：上級

**課題**：
以下の機能を持つ完全なTODOリストを作成してください。
- TODOを追加（フォーム）
- TODOを削除
- 完了/未完了の切り替え
- 優先度の選択
- フィルタリング（すべて/未完了/完了済み）
- 優先度でソート

**解答例**：
```typescript
import { useState } from 'react';

type Todo = {
  id: number;
  text: string;
  completed: boolean;
  priority: 'low' | 'medium' | 'high';
};

function TodoApp() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [input, setInput] = useState('');
  const [priority, setPriority] = useState<'low' | 'medium' | 'high'>('medium');
  const [filter, setFilter] = useState<'all' | 'active' | 'completed'>('all');

  const addTodo = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      setTodos([
        ...todos,
        {
          id: Date.now(),
          text: input,
          completed: false,
          priority
        }
      ]);
      setInput('');
    }
  };

  const toggleTodo = (id: number) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  const deleteTodo = (id: number) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  const priorityOrder = { high: 1, medium: 2, low: 3 };

  const filteredTodos = todos
    .filter(todo => {
      if (filter === 'active') return !todo.completed;
      if (filter === 'completed') return todo.completed;
      return true;
    })
    .sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  return (
    <div>
      <h1>TODOリスト</h1>

      <form onSubmit={addTodo}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="新しいTODO"
        />
        <select
          value={priority}
          onChange={(e) => setPriority(e.target.value as any)}
        >
          <option value="low">低</option>
          <option value="medium">中</option>
          <option value="high">高</option>
        </select>
        <button type="submit">追加</button>
      </form>

      <div>
        <button onClick={() => setFilter('all')}>すべて</button>
        <button onClick={() => setFilter('active')}>未完了</button>
        <button onClick={() => setFilter('completed')}>完了済み</button>
      </div>

      <ul>
        {filteredTodos.map(todo => (
          <li key={todo.id}>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo(todo.id)}
            />
            <span style={{
              textDecoration: todo.completed ? 'line-through' : 'none'
            }}>
              {todo.text}
            </span>
            <span className={`priority-${todo.priority}`}>
              [{todo.priority}]
            </span>
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

- ✅ Reactでのイベントハンドリングの基本
- ✅ 様々なイベントタイプ（クリック、入力、キーボードなど）
- ✅ フォームの制御とバリデーション
- ✅ イベントオブジェクトの活用
- ✅ リストの効率的なレンダリング
- ✅ keyの重要性と正しい使い方

### 次に学ぶべきガイド

**おめでとうございます！** React基礎ガイドを完了しました。

次のステップ：

1. **[Hooks 完全マスターガイド](../hooks-mastery.md)**
   - useEffect、useContext、カスタムフックなど
   - 中級〜上級のReact開発

2. **[TypeScript パターンガイド](../typescript-patterns.md)**
   - 型安全なReact開発
   - ジェネリクス、ユーティリティ型

3. **[パフォーマンス最適化ガイド](../optimization-complete.md)**
   - React.memo、useMemo、useCallback
   - 大規模アプリのパフォーマンスチューニング

### 関連リソース

#### 公式ドキュメント
- [React: Responding to Events](https://react.dev/learn/responding-to-events)
- [React: Rendering Lists](https://react.dev/learn/rendering-lists)

#### 実践プロジェクト
- [React Tutorial: Tic-Tac-Toe](https://react.dev/learn/tutorial-tic-tac-toe)
- [React Challenges](https://react-challenges.vercel.app/)

---

**前のガイド**：[06-state-basics.md](./06-state-basics.md)

**親ガイド**：[React Development - SKILL.md](../../SKILL.md)
