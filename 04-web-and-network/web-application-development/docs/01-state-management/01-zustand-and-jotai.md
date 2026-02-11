# Zustand / Jotai

> Zustandはストアベースの軽量状態管理、Jotaiはアトムベースのボトムアップ状態管理。それぞれのメンタルモデル、実装パターン、使い分けの基準を理解し、プロジェクトに最適なツールを選択する。

## この章で学ぶこと

- [ ] Zustandのストア設計とミドルウェアを理解する
- [ ] Jotaiのアトム設計と派生アトムを把握する
- [ ] 両者の使い分け基準を学ぶ

---

## 1. Zustand

```typescript
// Zustand: 超シンプルなストアベース状態管理
import { create } from 'zustand';

// --- 基本的なストア ---
interface CounterStore {
  count: number;
  increment: () => void;
  decrement: () => void;
  reset: () => void;
}

const useCounterStore = create<CounterStore>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  reset: () => set({ count: 0 }),
}));

// 使用（必要なプロパティだけ選択 → 最小限の再レンダリング）
function Counter() {
  const count = useCounterStore((state) => state.count);
  const increment = useCounterStore((state) => state.increment);
  return <button onClick={increment}>{count}</button>;
}

// --- 実践的なストア（認証）---
interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const useAuthStore = create<AuthStore>((set, get) => ({
  user: null,
  token: null,
  isAuthenticated: false,

  login: async (email, password) => {
    const { user, token } = await api.auth.login(email, password);
    set({ user, token, isAuthenticated: true });
  },

  logout: () => {
    set({ user: null, token: null, isAuthenticated: false });
  },
}));

// --- ミドルウェア ---
import { persist, devtools } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

const useCartStore = create<CartStore>()(
  devtools(
    persist(
      immer((set) => ({
        items: [],
        addItem: (product: Product) =>
          set((state) => {
            const existing = state.items.find(i => i.productId === product.id);
            if (existing) {
              existing.quantity += 1;
            } else {
              state.items.push({ productId: product.id, product, quantity: 1 });
            }
          }),
        removeItem: (productId: string) =>
          set((state) => {
            state.items = state.items.filter(i => i.productId !== productId);
          }),
        clearCart: () => set({ items: [] }),
      })),
      { name: 'cart-storage' } // localStorage に永続化
    ),
    { name: 'CartStore' } // Redux DevTools に表示
  )
);
```

---

## 2. Jotai

```typescript
// Jotai: アトムベースのボトムアップ状態管理
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';

// --- プリミティブアトム ---
const countAtom = atom(0);
const nameAtom = atom('');
const isDarkModeAtom = atom(false);

// --- 派生アトム（Derived Atom）---
const doubleCountAtom = atom((get) => get(countAtom) * 2);

// 読み書き派生アトム
const countWithLimitAtom = atom(
  (get) => get(countAtom),
  (get, set, newValue: number) => {
    set(countAtom, Math.min(Math.max(newValue, 0), 100));
  }
);

// --- 非同期アトム ---
const userAtom = atom(async () => {
  const response = await fetch('/api/user');
  return response.json();
});

// 使用
function Counter() {
  const [count, setCount] = useAtom(countAtom);
  const doubleCount = useAtomValue(doubleCountAtom); // 読み取り専用
  const setName = useSetAtom(nameAtom); // 書き込み専用

  return (
    <div>
      <p>Count: {count}, Double: {doubleCount}</p>
      <button onClick={() => setCount(c => c + 1)}>+1</button>
    </div>
  );
}

// --- 実践的なパターン ---
// フィルタリング
const filterAtom = atom('all');
const todosAtom = atom<Todo[]>([]);
const filteredTodosAtom = atom((get) => {
  const filter = get(filterAtom);
  const todos = get(todosAtom);
  switch (filter) {
    case 'active': return todos.filter(t => !t.completed);
    case 'completed': return todos.filter(t => t.completed);
    default: return todos;
  }
});

// atomWithStorage（永続化）
import { atomWithStorage } from 'jotai/utils';
const themeAtom = atomWithStorage('theme', 'light');

// atomFamily（動的アトム）
import { atomFamily } from 'jotai/utils';
const todoAtomFamily = atomFamily((id: string) =>
  atom(async () => {
    const response = await fetch(`/api/todos/${id}`);
    return response.json();
  })
);

function TodoItem({ id }: { id: string }) {
  const todo = useAtomValue(todoAtomFamily(id));
  return <div>{todo.title}</div>;
}
```

---

## 3. 使い分け

```
Zustand を選ぶ場合:
  ✓ 明確な「ストア」の概念が欲しい
  ✓ React外からも状態にアクセスしたい
  ✓ ミドルウェア（persist, devtools, immer）が必要
  ✓ チームにRedux経験者が多い
  ✓ 状態の構造が事前に決まっている

Jotai を選ぶ場合:
  ✓ コンポーネント単位の細かい再レンダリング制御
  ✓ 派生状態（computed）が多い
  ✓ 状態が動的に増減する（atomFamily）
  ✓ Suspense / Concurrent React との統合
  ✓ ボトムアップで状態を組み立てたい

共通:
  → どちらも TypeScript ファースト
  → どちらも軽量（< 5KB）
  → どちらも React 18+ に最適化

実務での組み合わせ:
  → Zustand: 認証、カート、UI設定等のグローバルストア
  → Jotai: フォーム、フィルタ、ソート等の動的な状態
  → TanStack Query: サーバーデータ
  → useState: ローカルUI状態
```

---

## まとめ

| 特徴 | Zustand | Jotai |
|------|---------|-------|
| モデル | ストアベース（トップダウン） | アトムベース（ボトムアップ） |
| API | create() | atom() + useAtom() |
| 再レンダリング | セレクターで最適化 | アトム単位で自動最適化 |
| ミドルウェア | persist, devtools, immer | atomWithStorage, atomFamily |
| React外 | ✓（getState, subscribe） | ✗ |
| サイズ | ~2KB | ~3KB |

---

## 次に読むべきガイド
→ [[02-server-state.md]] — サーバー状態

---

## 参考文献
1. Zustand. "Bear necessities for state management." github.com/pmndrs/zustand, 2024.
2. Jotai. "Primitive and flexible state management." jotai.org, 2024.
3. Daishi Kato. "When I Use Jotai vs Zustand." blog.axlight.com, 2024.
