# Zustand / Jotai

> Zustandはストアベースの軽量状態管理、Jotaiはアトムベースのボトムアップ状態管理。それぞれのメンタルモデル、実装パターン、使い分けの基準を理解し、プロジェクトに最適なツールを選択する。

## この章で学ぶこと

- [ ] Zustandのストア設計とミドルウェアを理解する
- [ ] Jotaiのアトム設計と派生アトムを把握する
- [ ] 両者の使い分け基準を学ぶ
- [ ] 実務での高度なパターンを習得する
- [ ] テスト戦略を理解する
- [ ] パフォーマンス最適化テクニックを身につける

---

## 1. Zustand の基礎

### 1.1 基本的なストア定義

```typescript
// Zustand: 超シンプルなストアベース状態管理
import { create } from 'zustand';

// --- 基本的なストア ---
interface CounterStore {
  count: number;
  increment: () => void;
  decrement: () => void;
  reset: () => void;
  incrementBy: (amount: number) => void;
}

const useCounterStore = create<CounterStore>((set) => ({
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
  decrement: () => set((state) => ({ count: state.count - 1 })),
  reset: () => set({ count: 0 }),
  incrementBy: (amount) => set((state) => ({ count: state.count + amount })),
}));

// 使用（必要なプロパティだけ選択 → 最小限の再レンダリング）
function Counter() {
  const count = useCounterStore((state) => state.count);
  const increment = useCounterStore((state) => state.increment);
  return <button onClick={increment}>{count}</button>;
}

// 複数の値をまとめて取得する場合
function CounterDisplay() {
  // shallow 比較でオブジェクトの再レンダリングを最適化
  const { count, increment, decrement } = useCounterStore(
    useShallow((state) => ({
      count: state.count,
      increment: state.increment,
      decrement: state.decrement,
    }))
  );

  return (
    <div>
      <span>{count}</span>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
    </div>
  );
}
```

### 1.2 set, get の詳細理解

```typescript
// create の引数関数には set, get, api の3つが渡される
const useStore = create<MyStore>((set, get, api) => ({
  // === set の使い方 ===

  // ① オブジェクトを渡す（部分的なマージ）
  setName: (name: string) => set({ name }),
  // → 他のプロパティは保持される（Object.assign相当）

  // ② 関数を渡す（前の状態に基づく更新）
  increment: () => set((state) => ({ count: state.count + 1 })),

  // ③ replace フラグ（状態全体を置き換え）
  resetAll: () =>
    set(
      { count: 0, name: '', items: [] },
      true // 第2引数: replace = true（マージではなく置き換え）
    ),

  // === get の使い方 ===
  // ストアの現在の状態を同期的に取得
  doubleCount: () => get().count * 2,

  // 他のアクションを呼び出す
  incrementAndLog: () => {
    get().increment();
    console.log('New count:', get().count);
  },

  // 非同期処理での状態参照
  saveToServer: async () => {
    const { items, name } = get();
    await api.save({ items, name });
    set({ lastSaved: new Date() });
  },

  // === api の使い方 ===
  // api.getState() = get と同じ
  // api.setState() = set と同じ
  // api.subscribe() = 状態変更のリスナー登録
  // api.getInitialState() = 初期状態を取得
}));
```

### 1.3 セレクターのベストプラクティス

```typescript
interface TodoStore {
  todos: Todo[];
  filter: 'all' | 'active' | 'completed';
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  setFilter: (filter: 'all' | 'active' | 'completed') => void;
}

const useTodoStore = create<TodoStore>((set) => ({
  todos: [],
  filter: 'all',
  addTodo: (text) =>
    set((state) => ({
      todos: [
        ...state.todos,
        { id: crypto.randomUUID(), text, completed: false },
      ],
    })),
  toggleTodo: (id) =>
    set((state) => ({
      todos: state.todos.map((t) =>
        t.id === id ? { ...t, completed: !t.completed } : t
      ),
    })),
  setFilter: (filter) => set({ filter }),
}));

// NG: オブジェクトを毎回作成 → 毎レンダリングで新しい参照
function TodoListBad() {
  // 毎回新しいオブジェクトが返されるので、常に再レンダリング
  const { todos, filter } = useTodoStore((state) => ({
    todos: state.todos,
    filter: state.filter,
  }));
  // ...
}

// OK: useShallow を使う
import { useShallow } from 'zustand/react/shallow';

function TodoListGood() {
  const { todos, filter } = useTodoStore(
    useShallow((state) => ({
      todos: state.todos,
      filter: state.filter,
    }))
  );
  // → 値が実際に変わった時だけ再レンダリング
}

// OK: 個別にセレクト
function TodoFilter() {
  const filter = useTodoStore((state) => state.filter);
  const setFilter = useTodoStore((state) => state.setFilter);
  // → todos が変わっても再レンダリングされない
  return (
    <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
      <option value="all">すべて</option>
      <option value="active">未完了</option>
      <option value="completed">完了</option>
    </select>
  );
}

// 計算値はストア外でセレクターとして定義
const selectFilteredTodos = (state: TodoStore) => {
  const { todos, filter } = state;
  switch (filter) {
    case 'active':
      return todos.filter((t) => !t.completed);
    case 'completed':
      return todos.filter((t) => t.completed);
    default:
      return todos;
  }
};

function FilteredTodoList() {
  // 注意: この書き方では毎回新しい配列が返されるため再レンダリングされる
  // パフォーマンスが問題になる場合は useMemo と組み合わせる
  const filteredTodos = useTodoStore(selectFilteredTodos);
  return (
    <ul>
      {filteredTodos.map((todo) => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}
```

---

## 2. Zustand の実践パターン

### 2.1 認証ストア

```typescript
// --- 実践的なストア（認証）---
interface AuthStore {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
  clearError: () => void;
}

const useAuthStore = create<AuthStore>((set, get) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  login: async (email, password) => {
    set({ isLoading: true, error: null });
    try {
      const response = await api.auth.login(email, password);
      set({
        user: response.user,
        token: response.token,
        isAuthenticated: true,
        isLoading: false,
      });
      // トークンをHTTPクライアントに設定
      apiClient.setAuthToken(response.token);
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'ログインに失敗しました',
        isLoading: false,
      });
      throw error;
    }
  },

  logout: () => {
    set({
      user: null,
      token: null,
      isAuthenticated: false,
    });
    apiClient.clearAuthToken();
    // ログアウト後のクリーンアップ
    queryClient.clear();
  },

  refreshToken: async () => {
    const currentToken = get().token;
    if (!currentToken) return;

    try {
      const { token } = await api.auth.refresh(currentToken);
      set({ token });
      apiClient.setAuthToken(token);
    } catch {
      // リフレッシュ失敗 → ログアウト
      get().logout();
    }
  },

  clearError: () => set({ error: null }),
}));

// React 外からの使用例（APIインターセプターなど）
apiClient.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.response?.status === 401) {
      const { refreshToken, logout } = useAuthStore.getState();
      try {
        await refreshToken();
        // リトライ
        return apiClient.request(error.config);
      } catch {
        logout();
      }
    }
    return Promise.reject(error);
  }
);
```

### 2.2 カートストア（ミドルウェア活用）

```typescript
// --- ミドルウェア ---
import { persist, devtools, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface CartItem {
  productId: string;
  name: string;
  price: number;
  quantity: number;
  image: string;
}

interface CartStore {
  items: CartItem[];
  addItem: (product: Product) => void;
  removeItem: (productId: string) => void;
  updateQuantity: (productId: string, quantity: number) => void;
  clearCart: () => void;
  // 計算プロパティ（getter的に使う）
  totalItems: () => number;
  totalPrice: () => number;
}

const useCartStore = create<CartStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          items: [],

          addItem: (product: Product) =>
            set((state) => {
              const existing = state.items.find(
                (i) => i.productId === product.id
              );
              if (existing) {
                existing.quantity += 1;
              } else {
                state.items.push({
                  productId: product.id,
                  name: product.name,
                  price: product.price,
                  quantity: 1,
                  image: product.image,
                });
              }
            }),

          removeItem: (productId: string) =>
            set((state) => {
              state.items = state.items.filter(
                (i) => i.productId !== productId
              );
            }),

          updateQuantity: (productId: string, quantity: number) =>
            set((state) => {
              const item = state.items.find(
                (i) => i.productId === productId
              );
              if (item) {
                if (quantity <= 0) {
                  state.items = state.items.filter(
                    (i) => i.productId !== productId
                  );
                } else {
                  item.quantity = quantity;
                }
              }
            }),

          clearCart: () => set({ items: [] }),

          totalItems: () =>
            get().items.reduce((sum, item) => sum + item.quantity, 0),

          totalPrice: () =>
            get().items.reduce(
              (sum, item) => sum + item.price * item.quantity,
              0
            ),
        }))
      ),
      {
        name: 'cart-storage', // localStorage のキー
        // 一部のフィールドのみ永続化
        partialize: (state) => ({ items: state.items }),
        // カスタムストレージ（sessionStorage等）
        // storage: createJSONStorage(() => sessionStorage),
        // バージョン管理（マイグレーション）
        version: 1,
        migrate: (persistedState, version) => {
          if (version === 0) {
            // v0 → v1 のマイグレーション
            return {
              ...(persistedState as any),
              items: (persistedState as any).items.map((item: any) => ({
                ...item,
                image: item.image ?? '/placeholder.png',
              })),
            };
          }
          return persistedState as CartStore;
        },
      }
    ),
    { name: 'CartStore' } // Redux DevTools に表示される名前
  )
);

// subscribeWithSelector で特定の状態変更を監視
useCartStore.subscribe(
  (state) => state.items.length,
  (itemCount, prevItemCount) => {
    if (itemCount > prevItemCount) {
      toast.success('カートに追加しました');
    }
  }
);
```

### 2.3 UIストア

```typescript
// UIに関する状態をまとめたストア
interface UIStore {
  // サイドバー
  sidebarOpen: boolean;
  sidebarWidth: number;
  toggleSidebar: () => void;
  setSidebarWidth: (width: number) => void;

  // モーダル
  activeModal: string | null;
  modalData: Record<string, unknown>;
  openModal: (id: string, data?: Record<string, unknown>) => void;
  closeModal: () => void;

  // トースト通知
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;

  // テーマ
  theme: 'light' | 'dark' | 'system';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;

  // ブレッドクラム
  breadcrumbs: Breadcrumb[];
  setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => void;
}

const useUIStore = create<UIStore>()(
  persist(
    (set, get) => ({
      // サイドバー
      sidebarOpen: true,
      sidebarWidth: 240,
      toggleSidebar: () =>
        set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),

      // モーダル
      activeModal: null,
      modalData: {},
      openModal: (id, data = {}) =>
        set({ activeModal: id, modalData: data }),
      closeModal: () => set({ activeModal: null, modalData: {} }),

      // トースト通知
      toasts: [],
      addToast: (toast) => {
        const id = crypto.randomUUID();
        set((state) => ({
          toasts: [...state.toasts, { ...toast, id }],
        }));
        // 自動削除
        if (toast.duration !== Infinity) {
          setTimeout(() => {
            get().removeToast(id);
          }, toast.duration ?? 5000);
        }
      },
      removeToast: (id) =>
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        })),

      // テーマ
      theme: 'system',
      setTheme: (theme) => set({ theme }),

      // ブレッドクラム
      breadcrumbs: [],
      setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs }),
    }),
    {
      name: 'ui-preferences',
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        sidebarWidth: state.sidebarWidth,
        theme: state.theme,
      }),
    }
  )
);

// モーダルを型安全に使うヘルパー
type ModalType = 'confirm' | 'editUser' | 'createProject';

interface ModalDataMap {
  confirm: { title: string; message: string; onConfirm: () => void };
  editUser: { userId: string };
  createProject: { teamId: string };
}

function useTypedModal<T extends ModalType>(type: T) {
  const activeModal = useUIStore((state) => state.activeModal);
  const modalData = useUIStore((state) => state.modalData);
  const openModal = useUIStore((state) => state.openModal);
  const closeModal = useUIStore((state) => state.closeModal);

  return {
    isOpen: activeModal === type,
    data: modalData as ModalDataMap[T],
    open: (data: ModalDataMap[T]) => openModal(type, data),
    close: closeModal,
  };
}

// 使用例
function UserList() {
  const editModal = useTypedModal('editUser');

  return (
    <div>
      <button onClick={() => editModal.open({ userId: '123' })}>
        編集
      </button>
      {editModal.isOpen && (
        <EditUserModal
          userId={editModal.data.userId}
          onClose={editModal.close}
        />
      )}
    </div>
  );
}
```

### 2.4 スライスパターン（大規模アプリ向け）

```typescript
// 大規模アプリでは、ストアをスライスに分割する

// --- 型定義 ---
interface UserSlice {
  user: User | null;
  setUser: (user: User | null) => void;
  updateProfile: (data: Partial<User>) => Promise<void>;
}

interface CartSlice {
  items: CartItem[];
  addItem: (product: Product) => void;
  removeItem: (productId: string) => void;
  totalPrice: () => number;
}

interface NotificationSlice {
  notifications: Notification[];
  unreadCount: number;
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => void;
  markAsRead: (id: string) => void;
  markAllAsRead: () => void;
  clearAll: () => void;
}

// 結合型
type AppStore = UserSlice & CartSlice & NotificationSlice;

// --- 各スライスの実装 ---
import { StateCreator } from 'zustand';

// UserSlice
const createUserSlice: StateCreator<AppStore, [], [], UserSlice> = (
  set,
  get
) => ({
  user: null,
  setUser: (user) => set({ user }),
  updateProfile: async (data) => {
    const currentUser = get().user;
    if (!currentUser) throw new Error('Not authenticated');

    const updated = await api.users.update(currentUser.id, data);
    set({ user: updated });

    // 他のスライスとの連携: 通知を追加
    get().addNotification({
      type: 'success',
      title: 'プロフィール更新',
      message: 'プロフィールを更新しました',
    });
  },
});

// CartSlice
const createCartSlice: StateCreator<AppStore, [], [], CartSlice> = (
  set,
  get
) => ({
  items: [],
  addItem: (product) =>
    set((state) => {
      const existing = state.items.find(
        (i) => i.productId === product.id
      );
      if (existing) {
        return {
          items: state.items.map((i) =>
            i.productId === product.id
              ? { ...i, quantity: i.quantity + 1 }
              : i
          ),
        };
      }
      return {
        items: [
          ...state.items,
          {
            productId: product.id,
            name: product.name,
            price: product.price,
            quantity: 1,
          },
        ],
      };
    }),
  removeItem: (productId) =>
    set((state) => ({
      items: state.items.filter((i) => i.productId !== productId),
    })),
  totalPrice: () =>
    get().items.reduce((sum, i) => sum + i.price * i.quantity, 0),
});

// NotificationSlice
const createNotificationSlice: StateCreator<
  AppStore,
  [],
  [],
  NotificationSlice
> = (set, get) => ({
  notifications: [],
  unreadCount: 0,
  addNotification: (notification) =>
    set((state) => {
      const newNotification: Notification = {
        ...notification,
        id: crypto.randomUUID(),
        timestamp: new Date(),
        read: false,
      };
      return {
        notifications: [newNotification, ...state.notifications],
        unreadCount: state.unreadCount + 1,
      };
    }),
  markAsRead: (id) =>
    set((state) => ({
      notifications: state.notifications.map((n) =>
        n.id === id ? { ...n, read: true } : n
      ),
      unreadCount: Math.max(0, state.unreadCount - 1),
    })),
  markAllAsRead: () =>
    set((state) => ({
      notifications: state.notifications.map((n) => ({
        ...n,
        read: true,
      })),
      unreadCount: 0,
    })),
  clearAll: () => set({ notifications: [], unreadCount: 0 }),
});

// --- ストア作成 ---
const useAppStore = create<AppStore>()(
  devtools((...a) => ({
    ...createUserSlice(...a),
    ...createCartSlice(...a),
    ...createNotificationSlice(...a),
  }))
);

// 各スライスごとにエクスポートするヘルパー
export const useUser = () => useAppStore((state) => state.user);
export const useCartItems = () => useAppStore((state) => state.items);
export const useNotifications = () =>
  useAppStore((state) => state.notifications);
export const useUnreadCount = () =>
  useAppStore((state) => state.unreadCount);
```

### 2.5 React 外からのアクセス

```typescript
// Zustand の強力な特徴: React コンポーネント外から状態にアクセスできる

// API インターセプターでの使用
import axios from 'axios';

const apiClient = axios.create({ baseURL: '/api' });

apiClient.interceptors.request.use((config) => {
  // React の外から直接 token を取得
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// WebSocket ハンドラでの使用
const socket = new WebSocket('wss://api.example.com/ws');

socket.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'NEW_NOTIFICATION':
      // React の外から直接状態を更新
      useAppStore.getState().addNotification(data.notification);
      break;

    case 'CART_UPDATED':
      useCartStore.setState({ items: data.items });
      break;

    case 'USER_LOGGED_OUT':
      useAuthStore.getState().logout();
      break;
  }
});

// タイマー/スケジューラでの使用
setInterval(() => {
  const { token, refreshToken } = useAuthStore.getState();
  if (token) {
    // トークンの有効期限チェック
    const payload = parseJwt(token);
    const expiresIn = payload.exp * 1000 - Date.now();
    if (expiresIn < 5 * 60 * 1000) {
      // 5分以内に期限切れ
      refreshToken();
    }
  }
}, 60 * 1000); // 毎分チェック

// テストでの使用
describe('CartStore', () => {
  beforeEach(() => {
    // テスト間で状態をリセット
    useCartStore.setState({
      items: [],
    });
  });

  it('should add item', () => {
    useCartStore.getState().addItem({
      id: '1',
      name: 'テスト商品',
      price: 1000,
      image: '/test.png',
    });

    expect(useCartStore.getState().items).toHaveLength(1);
    expect(useCartStore.getState().items[0].quantity).toBe(1);
  });
});

// subscribe で状態変更を監視（React外）
const unsubscribe = useAuthStore.subscribe(
  (state) => state.isAuthenticated,
  (isAuthenticated) => {
    if (!isAuthenticated) {
      // 未認証になったらWebSocket接続を切断
      socket.close();
    }
  }
);
```

---

## 3. Jotai の基礎

### 3.1 プリミティブアトム

```typescript
// Jotai: アトムベースのボトムアップ状態管理
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';

// --- プリミティブアトム ---
// 最も基本的な状態単位
const countAtom = atom(0);
const nameAtom = atom('');
const isDarkModeAtom = atom(false);
const selectedIdsAtom = atom<Set<string>>(new Set());
const formDataAtom = atom<FormData>({
  firstName: '',
  lastName: '',
  email: '',
});

// 使用: useAtom（読み書き両方）
function Counter() {
  const [count, setCount] = useAtom(countAtom);
  return (
    <button onClick={() => setCount((c) => c + 1)}>
      Count: {count}
    </button>
  );
}

// useAtomValue（読み取り専用）
function CountDisplay() {
  const count = useAtomValue(countAtom);
  return <span>現在のカウント: {count}</span>;
}

// useSetAtom（書き込み専用 → このコンポーネントは値の変更で再レンダリングされない）
function IncrementButton() {
  const setCount = useSetAtom(countAtom);
  return <button onClick={() => setCount((c) => c + 1)}>+1</button>;
}
```

### 3.2 派生アトム（Derived Atoms）

```typescript
// --- 派生アトム（Derived Atom）---

// ① 読み取り専用の派生アトム
const doubleCountAtom = atom((get) => get(countAtom) * 2);

const fullNameAtom = atom((get) => {
  const data = get(formDataAtom);
  return `${data.lastName} ${data.firstName}`;
});

// 複数のアトムに依存
const cartSummaryAtom = atom((get) => {
  const items = get(cartItemsAtom);
  const discount = get(discountAtom);
  const subtotal = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );
  const discountAmount = subtotal * discount;
  const total = subtotal - discountAmount;
  const itemCount = items.reduce((sum, item) => sum + item.quantity, 0);

  return {
    subtotal,
    discountAmount,
    total,
    itemCount,
    isEmpty: items.length === 0,
  };
});

function CartSummary() {
  const summary = useAtomValue(cartSummaryAtom);
  // cartItemsAtom か discountAtom が変わった時のみ再計算＆再レンダリング
  return (
    <div>
      <p>小計: {summary.subtotal.toLocaleString()}円</p>
      <p>割引: -{summary.discountAmount.toLocaleString()}円</p>
      <p>合計: {summary.total.toLocaleString()}円</p>
    </div>
  );
}

// ② 読み書き派生アトム（Write-only derived atom）
const countWithLimitAtom = atom(
  (get) => get(countAtom),
  (get, set, newValue: number) => {
    // 0〜100 の範囲に制限
    set(countAtom, Math.min(Math.max(newValue, 0), 100));
  }
);

// 複数のアトムを同時に更新する派生アトム
const resetAllAtom = atom(null, (get, set) => {
  set(countAtom, 0);
  set(nameAtom, '');
  set(isDarkModeAtom, false);
  set(selectedIdsAtom, new Set());
});

function ResetButton() {
  const resetAll = useSetAtom(resetAllAtom);
  return <button onClick={resetAll}>すべてリセット</button>;
}

// ③ 条件付き派生アトム
const currentUserAtom = atom<User | null>(null);
const isAdminAtom = atom((get) => {
  const user = get(currentUserAtom);
  return user?.role === 'admin';
});

const adminMenuItemsAtom = atom((get) => {
  const isAdmin = get(isAdminAtom);
  if (!isAdmin) return [];
  return [
    { label: 'ユーザー管理', path: '/admin/users' },
    { label: 'システム設定', path: '/admin/settings' },
    { label: 'ログ', path: '/admin/logs' },
  ];
});
```

### 3.3 非同期アトム

```typescript
// --- 非同期アトム ---

// 基本: 非同期な初期値
const userAtom = atom(async () => {
  const response = await fetch('/api/user');
  return response.json() as Promise<User>;
});

// Suspense と組み合わせて使用
function UserProfile() {
  const user = useAtomValue(userAtom);
  // Suspense がローディング状態をハンドル
  return <div>{user.name}</div>;
}

function App() {
  return (
    <Suspense fallback={<Spinner />}>
      <UserProfile />
    </Suspense>
  );
}

// 読み書き非同期アトム
const todosAtom = atom<Todo[]>([]);

const fetchTodosAtom = atom(
  (get) => get(todosAtom),
  async (get, set) => {
    const response = await fetch('/api/todos');
    const todos = await response.json();
    set(todosAtom, todos);
  }
);

const addTodoAtom = atom(null, async (get, set, text: string) => {
  const response = await fetch('/api/todos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  const newTodo = await response.json();
  set(todosAtom, [...get(todosAtom), newTodo]);
});

function TodoApp() {
  const [todos, fetchTodos] = useAtom(fetchTodosAtom);
  const addTodo = useSetAtom(addTodoAtom);

  useEffect(() => {
    fetchTodos();
  }, [fetchTodos]);

  return (
    <div>
      <button onClick={() => addTodo('新しいタスク')}>追加</button>
      <ul>
        {todos.map((todo) => (
          <li key={todo.id}>{todo.text}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

## 4. Jotai の実践パターン

### 4.1 atomWithStorage（永続化）

```typescript
import { atomWithStorage, createJSONStorage } from 'jotai/utils';

// localStorage に永続化
const themeAtom = atomWithStorage<'light' | 'dark' | 'system'>(
  'app-theme',
  'system'
);

const languageAtom = atomWithStorage<'ja' | 'en' | 'zh'>('language', 'ja');

// sessionStorage に永続化
const sessionThemeAtom = atomWithStorage(
  'session-theme',
  'light',
  createJSONStorage(() => sessionStorage)
);

// カスタムストレージ（例: AsyncStorage, MMKV）
const customStorage = createJSONStorage<string>(() => ({
  getItem: async (key) => {
    return await AsyncStorage.getItem(key);
  },
  setItem: async (key, value) => {
    await AsyncStorage.setItem(key, value);
  },
  removeItem: async (key) => {
    await AsyncStorage.removeItem(key);
  },
}));

// ユーザー設定を永続化
interface UserPreferences {
  fontSize: number;
  lineHeight: number;
  fontFamily: string;
  sidebarWidth: number;
  showLineNumbers: boolean;
  autoSave: boolean;
  autoSaveInterval: number;
}

const userPreferencesAtom = atomWithStorage<UserPreferences>(
  'user-preferences',
  {
    fontSize: 14,
    lineHeight: 1.6,
    fontFamily: 'Inter',
    sidebarWidth: 240,
    showLineNumbers: true,
    autoSave: true,
    autoSaveInterval: 30000,
  }
);

// 個別のプロパティを更新する派生アトム
const fontSizeAtom = atom(
  (get) => get(userPreferencesAtom).fontSize,
  (get, set, fontSize: number) => {
    set(userPreferencesAtom, {
      ...get(userPreferencesAtom),
      fontSize: Math.min(Math.max(fontSize, 10), 24),
    });
  }
);
```

### 4.2 atomFamily（動的アトム）

```typescript
import { atomFamily, atomWithDefault } from 'jotai/utils';

// 基本的な atomFamily
const todoAtomFamily = atomFamily((id: string) =>
  atom<Todo | null>(null)
);

// 非同期な atomFamily
const userAtomFamily = atomFamily((userId: string) =>
  atom(async () => {
    const response = await fetch(`/api/users/${userId}`);
    if (!response.ok) throw new Error('User not found');
    return response.json() as Promise<User>;
  })
);

function UserCard({ userId }: { userId: string }) {
  const user = useAtomValue(userAtomFamily(userId));
  return (
    <Suspense fallback={<Skeleton />}>
      <div>
        <img src={user.avatar} alt={user.name} />
        <h3>{user.name}</h3>
      </div>
    </Suspense>
  );
}

// 読み書き可能な atomFamily
interface FieldState {
  value: string;
  error: string | null;
  touched: boolean;
}

const fieldAtomFamily = atomFamily((fieldName: string) =>
  atom<FieldState>({
    value: '',
    error: null,
    touched: false,
  })
);

// フォームの各フィールドを独立して管理
function FormField({ name, label }: { name: string; label: string }) {
  const [field, setField] = useAtom(fieldAtomFamily(name));

  const handleChange = (value: string) => {
    setField((prev) => ({
      ...prev,
      value,
      error: null, // 入力時にエラーをクリア
    }));
  };

  const handleBlur = () => {
    setField((prev) => ({ ...prev, touched: true }));
  };

  return (
    <div>
      <label>{label}</label>
      <input
        value={field.value}
        onChange={(e) => handleChange(e.target.value)}
        onBlur={handleBlur}
      />
      {field.touched && field.error && (
        <span className="error">{field.error}</span>
      )}
    </div>
  );
}

// フォーム全体のバリデーション
const formFieldNames = ['firstName', 'lastName', 'email', 'phone'];

const formValidAtom = atom((get) => {
  return formFieldNames.every((name) => {
    const field = get(fieldAtomFamily(name));
    return field.value.length > 0 && field.error === null;
  });
});

const formDataAtom = atom((get) => {
  const data: Record<string, string> = {};
  for (const name of formFieldNames) {
    data[name] = get(fieldAtomFamily(name)).value;
  }
  return data;
});
```

### 4.3 フィルタリングとソートの実践パターン

```typescript
// 状態定義
const filterAtom = atom<'all' | 'active' | 'completed'>('all');
const sortAtom = atom<'newest' | 'oldest' | 'name'>('newest');
const searchQueryAtom = atom('');
const todosAtom = atom<Todo[]>([]);

// 派生アトム: フィルタリング → ソート → 検索の順で適用
const filteredTodosAtom = atom((get) => {
  const todos = get(todosAtom);
  const filter = get(filterAtom);

  switch (filter) {
    case 'active':
      return todos.filter((t) => !t.completed);
    case 'completed':
      return todos.filter((t) => t.completed);
    default:
      return todos;
  }
});

const sortedTodosAtom = atom((get) => {
  const filtered = get(filteredTodosAtom);
  const sortBy = get(sortAtom);

  return [...filtered].sort((a, b) => {
    switch (sortBy) {
      case 'newest':
        return b.createdAt.getTime() - a.createdAt.getTime();
      case 'oldest':
        return a.createdAt.getTime() - b.createdAt.getTime();
      case 'name':
        return a.text.localeCompare(b.text);
      default:
        return 0;
    }
  });
});

const displayTodosAtom = atom((get) => {
  const sorted = get(sortedTodosAtom);
  const query = get(searchQueryAtom).toLowerCase();

  if (!query) return sorted;
  return sorted.filter(
    (t) =>
      t.text.toLowerCase().includes(query) ||
      t.tags?.some((tag) => tag.toLowerCase().includes(query))
  );
});

// 統計アトム
const todoStatsAtom = atom((get) => {
  const todos = get(todosAtom);
  return {
    total: todos.length,
    active: todos.filter((t) => !t.completed).length,
    completed: todos.filter((t) => t.completed).length,
    completionRate:
      todos.length > 0
        ? Math.round(
            (todos.filter((t) => t.completed).length / todos.length) * 100
          )
        : 0,
  };
});

// 使用
function TodoFilters() {
  const [filter, setFilter] = useAtom(filterAtom);
  const [sort, setSort] = useAtom(sortAtom);
  const [query, setQuery] = useAtom(searchQueryAtom);
  const stats = useAtomValue(todoStatsAtom);

  return (
    <div className="filters">
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="検索..."
      />
      <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
        <option value="all">すべて ({stats.total})</option>
        <option value="active">未完了 ({stats.active})</option>
        <option value="completed">完了 ({stats.completed})</option>
      </select>
      <select value={sort} onChange={(e) => setSort(e.target.value as any)}>
        <option value="newest">新しい順</option>
        <option value="oldest">古い順</option>
        <option value="name">名前順</option>
      </select>
      <span>完了率: {stats.completionRate}%</span>
    </div>
  );
}

function TodoList() {
  const displayTodos = useAtomValue(displayTodosAtom);
  return (
    <ul>
      {displayTodos.map((todo) => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}
```

### 4.4 atomWithReducer

```typescript
import { atomWithReducer } from 'jotai/utils';

// useReducer のアトム版
type CountAction =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'reset' }
  | { type: 'set'; value: number };

const countReducerAtom = atomWithReducer(0, (state, action: CountAction) => {
  switch (action.type) {
    case 'increment':
      return state + 1;
    case 'decrement':
      return state - 1;
    case 'reset':
      return 0;
    case 'set':
      return action.value;
  }
});

function Counter() {
  const [count, dispatch] = useAtom(countReducerAtom);
  return (
    <div>
      <span>{count}</span>
      <button onClick={() => dispatch({ type: 'increment' })}>+</button>
      <button onClick={() => dispatch({ type: 'decrement' })}>-</button>
      <button onClick={() => dispatch({ type: 'reset' })}>リセット</button>
    </div>
  );
}

// より複雑な例: ドラッグ&ドロップの状態管理
type DragState = {
  isDragging: boolean;
  draggedItem: string | null;
  dragOverTarget: string | null;
  startPosition: { x: number; y: number } | null;
};

type DragAction =
  | { type: 'START_DRAG'; item: string; position: { x: number; y: number } }
  | { type: 'DRAG_OVER'; target: string }
  | { type: 'DROP' }
  | { type: 'CANCEL' };

const dragAtom = atomWithReducer<DragState, DragAction>(
  {
    isDragging: false,
    draggedItem: null,
    dragOverTarget: null,
    startPosition: null,
  },
  (state, action) => {
    switch (action.type) {
      case 'START_DRAG':
        return {
          isDragging: true,
          draggedItem: action.item,
          dragOverTarget: null,
          startPosition: action.position,
        };
      case 'DRAG_OVER':
        return { ...state, dragOverTarget: action.target };
      case 'DROP':
      case 'CANCEL':
        return {
          isDragging: false,
          draggedItem: null,
          dragOverTarget: null,
          startPosition: null,
        };
    }
  }
);
```

### 4.5 Jotai と TanStack Query の統合

```typescript
import { atomWithQuery, atomWithMutation } from 'jotai-tanstack-query';

// クエリアトム
const usersQueryAtom = atomWithQuery(() => ({
  queryKey: ['users'],
  queryFn: async () => {
    const response = await fetch('/api/users');
    return response.json() as Promise<User[]>;
  },
  staleTime: 5 * 60 * 1000,
}));

// パラメータ付きクエリアトム
const userIdAtom = atom<string | null>(null);

const userQueryAtom = atomWithQuery((get) => {
  const userId = get(userIdAtom);
  return {
    queryKey: ['user', userId],
    queryFn: async () => {
      if (!userId) throw new Error('No user ID');
      const response = await fetch(`/api/users/${userId}`);
      return response.json() as Promise<User>;
    },
    enabled: !!userId,
  };
});

// ミューテーションアトム
const createUserMutationAtom = atomWithMutation(() => ({
  mutationFn: async (data: CreateUserInput) => {
    const response = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return response.json() as Promise<User>;
  },
  onSuccess: () => {
    // キャッシュ無効化
    queryClient.invalidateQueries({ queryKey: ['users'] });
  },
}));

// 使用
function UserList() {
  const [{ data: users, isLoading, error }] = useAtom(usersQueryAtom);
  const [, createUser] = useAtom(createUserMutationAtom);

  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div>
      <button onClick={() => createUser({ name: '新しいユーザー', email: 'new@example.com' })}>
        ユーザー追加
      </button>
      <ul>
        {users?.map((user) => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

---

## 5. Zustand vs Jotai: 詳細比較

### 5.1 メンタルモデルの違い

```
Zustand: トップダウン（ストアベース）
  ┌─────────────────────────────────┐
  │           AppStore              │
  │  ┌───────┐ ┌──────┐ ┌───────┐  │
  │  │ user  │ │ cart │ │ ui    │  │
  │  └───────┘ └──────┘ └───────┘  │
  └─────────────────────────────────┘
           ↓         ↓         ↓
       Component  Component  Component

  → 「何のストアが必要か？」から設計を始める
  → ストアの形を最初に定義し、コンポーネントはそれを参照する
  → Redux のメンタルモデルに近い

Jotai: ボトムアップ（アトムベース）
       atom    atom    atom    atom
        ↓       ↓       ↓       ↓
     ┌──┴──┐ ┌──┴──┐ ┌──┴──────┴──┐
     │     │ │     │ │ derived    │
     └──┬──┘ └──┬──┘ └──────┬─────┘
        ↓       ↓           ↓
    Component Component  Component

  → 「このコンポーネントに何の状態が必要か？」から設計を始める
  → 小さなアトムを組み合わせて必要な状態を構築する
  → Recoil のメンタルモデルに近い
```

### 5.2 パフォーマンス特性

```typescript
// === 再レンダリングの違い ===

// Zustand: セレクターで明示的に最適化
function ZustandExample() {
  // 方法1: 個別セレクター（最も効率的）
  const count = useStore((s) => s.count);
  const name = useStore((s) => s.name);
  // → count or name が変わった時だけ再レンダリング

  // 方法2: useShallow（複数値を一度に取得）
  const { count, name } = useStore(
    useShallow((s) => ({ count: s.count, name: s.name }))
  );
  // → count or name が変わった時だけ再レンダリング
  // → shallow comparison で判定

  // 注意: セレクター内で新しいオブジェクトを作ると毎回再レンダリング
  // NG:
  const state = useStore((s) => ({ count: s.count })); // 毎回新しいオブジェクト
  // → useShallow で解決
}

// Jotai: アトム単位で自動最適化
function JotaiExample() {
  const count = useAtomValue(countAtom);
  // → countAtom が変わった時だけ再レンダリング
  // → nameAtom の変更は影響しない
  // → セレクター不要で自動的に最適化される

  // 派生アトムも自動的に依存関係を追跡
  const displayName = useAtomValue(displayNameAtom);
  // → displayNameAtom が依存するアトムが変わった時のみ再計算
}

// パフォーマンス比較:
// Zustand:
//   ✓ セレクターで精密な制御が可能
//   ✗ セレクターの書き方を間違えると不要な再レンダリング
//   ✓ useShallow で複数値の取得も最適化可能

// Jotai:
//   ✓ アトム単位で自動的に最適化
//   ✓ 派生アトムの依存関係も自動追跡
//   ✗ アトムを細かく分割しすぎるとコードが散乱する可能性
```

### 5.3 DevTools とデバッグ

```typescript
// === Zustand の DevTools ===
import { devtools } from 'zustand/middleware';

const useStore = create<AppStore>()(
  devtools(
    (set) => ({
      count: 0,
      increment: () =>
        set(
          (state) => ({ count: state.count + 1 }),
          false, // replace = false
          'increment' // アクション名（DevToolsに表示）
        ),
    }),
    {
      name: 'AppStore', // DevToolsに表示されるストア名
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// Redux DevTools で:
// - 状態のスナップショットを確認
// - タイムトラベルデバッグ
// - アクションの履歴を確認
// - 状態の差分を確認

// === Jotai の DevTools ===
import { DevTools } from 'jotai-devtools';
import 'jotai-devtools/styles.css';

// アトムにデバッグラベルを付ける
const countAtom = atom(0);
countAtom.debugLabel = 'countAtom';

const nameAtom = atom('');
nameAtom.debugLabel = 'nameAtom';

// DevTools コンポーネントを配置
function App() {
  return (
    <Provider>
      <DevTools />
      <MainContent />
    </Provider>
  );
}

// React DevTools の「Atoms」タブで:
// - 各アトムの現在値を確認
// - アトム間の依存関係を可視化
// - 値の変更をリアルタイムで監視
```

---

## 6. 使い分けガイド

### 6.1 プロジェクト特性別の選定

```
Zustand を選ぶ場合:
  ✓ 明確な「ストア」の概念が欲しい
  ✓ React外からも状態にアクセスしたい（APIインターセプター等）
  ✓ ミドルウェア（persist, devtools, immer）が必要
  ✓ チームにRedux経験者が多い
  ✓ 状態の構造が事前に決まっている
  ✓ WebSocket やタイマーから状態を更新する必要がある
  ✓ テストで状態を直接操作したい

Jotai を選ぶ場合:
  ✓ コンポーネント単位の細かい再レンダリング制御
  ✓ 派生状態（computed）が多い
  ✓ 状態が動的に増減する（atomFamily）
  ✓ Suspense / Concurrent React との統合
  ✓ ボトムアップで状態を組み立てたい
  ✓ フォームの各フィールドを独立して管理したい
  ✓ 複雑なフィルタリング/ソートのロジック

共通:
  → どちらも TypeScript ファースト
  → どちらも軽量（< 5KB）
  → どちらも React 18+ に最適化
  → どちらも pmndrs（Poimandres）が開発

両方使う場合（実務で最も多いパターン）:
  → Zustand: 認証、カート、UI設定等のグローバルストア
  → Jotai: フォーム、フィルタ、ソート等の動的な状態
  → TanStack Query: サーバーデータ
  → useState: ローカルUI状態
```

### 6.2 具体的なシナリオ別選定

```
シナリオ1: Eコマースアプリ
  認証状態 → Zustand（persist + React外からのアクセス）
  カート → Zustand（persist + 複数ページで共有）
  商品データ → TanStack Query
  商品フィルタ → URL状態（nuqs）
  テーマ/言語 → Zustand（persist）
  モーダル状態 → useState（ローカル）

シナリオ2: ダッシュボード/管理画面
  認証状態 → Zustand
  ダッシュボードのウィジェット配置 → Zustand（persist + ドラッグ&ドロップ）
  各ウィジェットのデータ → TanStack Query
  フィルタ/日付範囲 → URL状態
  サイドバー/テーマ → Zustand（persist）
  テーブルの列設定 → Jotai（atomFamily で列ごとに管理）

シナリオ3: リアルタイムコラボツール
  WebSocket接続 → Zustand（React外からのアクセス）
  ドキュメント状態 → Zustand or Jotai（要件による）
  ユーザープレゼンス → Zustand（WebSocketから更新）
  カーソル位置 → Jotai（ユーザーごとにatomFamily）
  エディタ設定 → Jotai（atomWithStorage）
  ファイル一覧 → TanStack Query

シナリオ4: フォーム重視のアプリ（申請システム等）
  認証 → Zustand
  フォームデータ → React Hook Form + Zod
  フォームの動的フィールド → Jotai（atomFamily）
  ウィザード進捗 → useReducer
  申請データ → TanStack Query
  下書き保存 → Zustand（persist）
```

---

## 7. テスト戦略

### 7.1 Zustand のテスト

```typescript
import { renderHook, act } from '@testing-library/react';

// テスト用にストアをリセットするユーティリティ
function resetStore<T extends object>(useStore: any) {
  const initialState = useStore.getInitialState();
  useStore.setState(initialState, true);
}

describe('useCartStore', () => {
  beforeEach(() => {
    resetStore(useCartStore);
  });

  it('should add an item to the cart', () => {
    const { result } = renderHook(() =>
      useCartStore(
        useShallow((state) => ({
          items: state.items,
          addItem: state.addItem,
        }))
      )
    );

    act(() => {
      result.current.addItem({
        id: 'p1',
        name: 'テスト商品',
        price: 1000,
        image: '/test.png',
      });
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]).toEqual({
      productId: 'p1',
      name: 'テスト商品',
      price: 1000,
      quantity: 1,
      image: '/test.png',
    });
  });

  it('should increment quantity for existing item', () => {
    // 直接状態を設定してテストのセットアップを簡潔に
    useCartStore.setState({
      items: [
        {
          productId: 'p1',
          name: 'テスト商品',
          price: 1000,
          quantity: 1,
          image: '/test.png',
        },
      ],
    });

    const { result } = renderHook(() => useCartStore());

    act(() => {
      result.current.addItem({
        id: 'p1',
        name: 'テスト商品',
        price: 1000,
        image: '/test.png',
      });
    });

    expect(result.current.items[0].quantity).toBe(2);
  });

  it('should calculate total price correctly', () => {
    useCartStore.setState({
      items: [
        { productId: 'p1', name: '商品A', price: 1000, quantity: 2, image: '' },
        { productId: 'p2', name: '商品B', price: 500, quantity: 3, image: '' },
      ],
    });

    expect(useCartStore.getState().totalPrice()).toBe(3500);
  });

  it('should remove an item', () => {
    useCartStore.setState({
      items: [
        { productId: 'p1', name: '商品A', price: 1000, quantity: 1, image: '' },
        { productId: 'p2', name: '商品B', price: 500, quantity: 1, image: '' },
      ],
    });

    act(() => {
      useCartStore.getState().removeItem('p1');
    });

    expect(useCartStore.getState().items).toHaveLength(1);
    expect(useCartStore.getState().items[0].productId).toBe('p2');
  });
});

// コンポーネント統合テスト
import { render, screen, fireEvent } from '@testing-library/react';

describe('CartComponent', () => {
  beforeEach(() => {
    resetStore(useCartStore);
  });

  it('should display cart items', () => {
    useCartStore.setState({
      items: [
        { productId: 'p1', name: '商品A', price: 1000, quantity: 2, image: '' },
      ],
    });

    render(<CartComponent />);

    expect(screen.getByText('商品A')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('2,000円')).toBeInTheDocument();
  });

  it('should remove item when delete button is clicked', () => {
    useCartStore.setState({
      items: [
        { productId: 'p1', name: '商品A', price: 1000, quantity: 1, image: '' },
      ],
    });

    render(<CartComponent />);

    fireEvent.click(screen.getByRole('button', { name: '削除' }));

    expect(screen.queryByText('商品A')).not.toBeInTheDocument();
    expect(useCartStore.getState().items).toHaveLength(0);
  });
});
```

### 7.2 Jotai のテスト

```typescript
import { renderHook, act } from '@testing-library/react';
import { Provider, createStore } from 'jotai';
import { useHydrateAtoms } from 'jotai/utils';

// テスト用ラッパー
function TestProvider({
  initialValues,
  children,
}: {
  initialValues: Array<[any, any]>;
  children: React.ReactNode;
}) {
  return (
    <Provider>
      <HydrateAtoms initialValues={initialValues}>
        {children}
      </HydrateAtoms>
    </Provider>
  );
}

function HydrateAtoms({
  initialValues,
  children,
}: {
  initialValues: Array<[any, any]>;
  children: React.ReactNode;
}) {
  useHydrateAtoms(initialValues);
  return children;
}

describe('Todo Atoms', () => {
  it('should filter todos by status', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <TestProvider
        initialValues={[
          [
            todosAtom,
            [
              { id: '1', text: 'Task 1', completed: false },
              { id: '2', text: 'Task 2', completed: true },
              { id: '3', text: 'Task 3', completed: false },
            ],
          ],
          [filterAtom, 'active'],
        ]}
      >
        {children}
      </TestProvider>
    );

    const { result } = renderHook(
      () => useAtomValue(filteredTodosAtom),
      { wrapper }
    );

    expect(result.current).toHaveLength(2);
    expect(result.current.every((t) => !t.completed)).toBe(true);
  });

  it('should calculate stats correctly', () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <TestProvider
        initialValues={[
          [
            todosAtom,
            [
              { id: '1', text: 'Task 1', completed: false },
              { id: '2', text: 'Task 2', completed: true },
              { id: '3', text: 'Task 3', completed: true },
            ],
          ],
        ]}
      >
        {children}
      </TestProvider>
    );

    const { result } = renderHook(
      () => useAtomValue(todoStatsAtom),
      { wrapper }
    );

    expect(result.current).toEqual({
      total: 3,
      active: 1,
      completed: 2,
      completionRate: 67,
    });
  });
});

// createStore を使ったテスト（Provider不要）
describe('Todo Atoms (with createStore)', () => {
  it('should toggle todo', () => {
    const store = createStore();

    store.set(todosAtom, [
      { id: '1', text: 'Task 1', completed: false },
    ]);

    // toggleTodoAtom が書き込みアトムの場合
    store.set(toggleTodoAtom, '1');

    const todos = store.get(todosAtom);
    expect(todos[0].completed).toBe(true);
  });
});
```

---

## 8. 高度なパターン

### 8.1 Zustand: Temporal ミドルウェア（Undo/Redo）

```typescript
import { temporal } from 'zundo';

interface EditorStore {
  content: string;
  fontSize: number;
  setContent: (content: string) => void;
  setFontSize: (size: number) => void;
}

const useEditorStore = create<EditorStore>()(
  temporal(
    (set) => ({
      content: '',
      fontSize: 14,
      setContent: (content) => set({ content }),
      setFontSize: (size) => set({ fontSize: size }),
    }),
    {
      limit: 50, // 履歴の最大数
      // 特定のフィールドのみ履歴に含める
      partialize: (state) => ({
        content: state.content,
      }),
      // デバウンス（タイピング中は毎キーストロークで履歴を作らない）
      handleSet: (handleSet) => {
        let timeoutId: NodeJS.Timeout;
        return (state) => {
          clearTimeout(timeoutId);
          timeoutId = setTimeout(() => {
            handleSet(state);
          }, 500);
        };
      },
    }
  )
);

// Undo/Redo ボタン
function UndoRedoButtons() {
  const { undo, redo, pastStates, futureStates } =
    useEditorStore.temporal.getState();

  return (
    <div>
      <button onClick={undo} disabled={pastStates.length === 0}>
        元に戻す ({pastStates.length})
      </button>
      <button onClick={redo} disabled={futureStates.length === 0}>
        やり直し ({futureStates.length})
      </button>
    </div>
  );
}
```

### 8.2 Jotai: focusAtom（レンズパターン）

```typescript
import { focusAtom } from 'jotai-optics';

// 大きなオブジェクトの特定のフィールドにフォーカスするアトム
interface AppConfig {
  editor: {
    fontSize: number;
    fontFamily: string;
    theme: string;
    lineNumbers: boolean;
  };
  sidebar: {
    width: number;
    collapsed: boolean;
    position: 'left' | 'right';
  };
  notifications: {
    enabled: boolean;
    sound: boolean;
    desktop: boolean;
  };
}

const configAtom = atom<AppConfig>({
  editor: {
    fontSize: 14,
    fontFamily: 'monospace',
    theme: 'vs-dark',
    lineNumbers: true,
  },
  sidebar: {
    width: 240,
    collapsed: false,
    position: 'left',
  },
  notifications: {
    enabled: true,
    sound: true,
    desktop: false,
  },
});

// focusAtom で特定のフィールドにフォーカス
const editorConfigAtom = focusAtom(configAtom, (optic) =>
  optic.prop('editor')
);
const fontSizeAtom = focusAtom(configAtom, (optic) =>
  optic.prop('editor').prop('fontSize')
);
const sidebarWidthAtom = focusAtom(configAtom, (optic) =>
  optic.prop('sidebar').prop('width')
);

// fontSizeAtom を更新すると、configAtom のネストされた値が自動的に更新される
function FontSizeControl() {
  const [fontSize, setFontSize] = useAtom(fontSizeAtom);
  return (
    <input
      type="range"
      min={10}
      max={24}
      value={fontSize}
      onChange={(e) => setFontSize(Number(e.target.value))}
    />
  );
  // → configAtom.editor.fontSize が更新される
  // → sidebar や notifications を使うコンポーネントは再レンダリングされない
}
```

### 8.3 Zustand と Jotai の組み合わせ

```typescript
// 実務では両方を使い分けることが多い

// Zustand: アプリ全体のグローバルストア（React外からもアクセス）
const useAuthStore = create<AuthStore>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      login: async (email, password) => {
        const result = await api.login(email, password);
        set({ user: result.user, token: result.token });
      },
      logout: () => set({ user: null, token: null }),
    }),
    { name: 'auth' }
  )
);

// Jotai: 画面固有の動的な状態（アトムベースで柔軟に）
const searchQueryAtom = atom('');
const filtersAtom = atom<Filter[]>([]);
const sortAtom = atom<SortConfig>({ field: 'createdAt', order: 'desc' });
const pageAtom = atom(1);

// Zustand の状態を Jotai から参照する場合
const currentUserAtom = atom((get) => {
  // Zustand ストアから直接取得
  return useAuthStore.getState().user;
});

// より反応的にする場合: subscribe を使う
const currentUserReactiveAtom = atom<User | null>(null);

// アプリ起動時に同期をセットアップ
useAuthStore.subscribe(
  (state) => state.user,
  (user) => {
    // Jotai の store を通じてアトムを更新
    jotaiStore.set(currentUserReactiveAtom, user);
  }
);
```

---

## 9. マイグレーションガイド

### 9.1 Redux から Zustand へ

```typescript
// === Redux Toolkit ===
// store/todoSlice.ts
const todoSlice = createSlice({
  name: 'todos',
  initialState: {
    items: [] as Todo[],
    filter: 'all' as FilterType,
  },
  reducers: {
    addTodo: (state, action: PayloadAction<string>) => {
      state.items.push({
        id: crypto.randomUUID(),
        text: action.payload,
        completed: false,
      });
    },
    toggleTodo: (state, action: PayloadAction<string>) => {
      const todo = state.items.find((t) => t.id === action.payload);
      if (todo) todo.completed = !todo.completed;
    },
    setFilter: (state, action: PayloadAction<FilterType>) => {
      state.filter = action.payload;
    },
  },
});

// コンポーネント
function TodoList() {
  const todos = useSelector((state: RootState) => state.todos.items);
  const dispatch = useDispatch();
  return (
    <button onClick={() => dispatch(todoSlice.actions.addTodo('New'))}>
      追加
    </button>
  );
}

// === 同じものを Zustand で ===
// stores/useTodoStore.ts
interface TodoStore {
  items: Todo[];
  filter: FilterType;
  addTodo: (text: string) => void;
  toggleTodo: (id: string) => void;
  setFilter: (filter: FilterType) => void;
}

const useTodoStore = create<TodoStore>()(
  immer((set) => ({
    items: [],
    filter: 'all',
    addTodo: (text) =>
      set((state) => {
        state.items.push({
          id: crypto.randomUUID(),
          text,
          completed: false,
        });
      }),
    toggleTodo: (id) =>
      set((state) => {
        const todo = state.items.find((t) => t.id === id);
        if (todo) todo.completed = !todo.completed;
      }),
    setFilter: (filter) => set({ filter }),
  }))
);

// コンポーネント（Provider不要！）
function TodoList() {
  const todos = useTodoStore((state) => state.items);
  const addTodo = useTodoStore((state) => state.addTodo);
  return <button onClick={() => addTodo('New')}>追加</button>;
}

// マイグレーションのポイント:
// 1. Provider/configureStore が不要
// 2. useSelector → useStore(selector)
// 3. dispatch(action) → store.action()
// 4. createSlice → create() 内で直接定義
// 5. immer ミドルウェアで同じ書き方が可能
// 6. Redux DevTools もそのまま使える
```

### 9.2 Context から Jotai へ

```typescript
// === React Context ===
const TodoContext = createContext<{
  todos: Todo[];
  filter: FilterType;
  addTodo: (text: string) => void;
  setFilter: (filter: FilterType) => void;
} | null>(null);

function TodoProvider({ children }: { children: React.ReactNode }) {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [filter, setFilter] = useState<FilterType>('all');

  const addTodo = useCallback((text: string) => {
    setTodos((prev) => [...prev, { id: crypto.randomUUID(), text, completed: false }]);
  }, []);

  const value = useMemo(
    () => ({ todos, filter, addTodo, setFilter }),
    [todos, filter, addTodo]
  );

  return <TodoContext.Provider value={value}>{children}</TodoContext.Provider>;
}

// 問題: filter だけ使うコンポーネントも todos の変更で再レンダリング

// === 同じものを Jotai で ===
const todosAtom = atom<Todo[]>([]);
const filterAtom = atom<FilterType>('all');

const addTodoAtom = atom(null, (get, set, text: string) => {
  set(todosAtom, [
    ...get(todosAtom),
    { id: crypto.randomUUID(), text, completed: false },
  ]);
});

// Provider 不要、各アトムが独立して更新される
function TodoFilters() {
  const [filter, setFilter] = useAtom(filterAtom);
  // todosAtom の変更ではこのコンポーネントは再レンダリングされない！
  return (
    <select value={filter} onChange={(e) => setFilter(e.target.value as any)}>
      <option value="all">すべて</option>
      <option value="active">未完了</option>
    </select>
  );
}

// マイグレーションのポイント:
// 1. Provider が不要（Jotai は React tree に暗黙的にスコープ）
// 2. useMemo/useCallback の手動最適化が不要
// 3. 各アトムが独立 → 再レンダリングが自動的に最適化
// 4. Context の分割（Value/Dispatch分離）が不要
```

---

## 10. ベストプラクティスまとめ

```
Zustand ベストプラクティス:
  1. セレクターを使って必要な値だけ取得する
  2. アクションはストア内で定義する（コンポーネント外でも使えるように）
  3. persist で永続化する際は partialize で必要なフィールドだけ
  4. devtools のアクション名を付けてデバッグしやすく
  5. 大規模アプリではスライスパターンで分割
  6. テスト時は getInitialState() でリセット
  7. immer ミドルウェアでネストした更新を簡潔に

Jotai ベストプラクティス:
  1. アトムは小さく保つ（1つのアトム = 1つの関心事）
  2. 派生アトムを積極的に使う（状態の導出をアトムレベルで）
  3. debugLabel を付けてデバッグしやすく
  4. useAtomValue / useSetAtom を使い分ける（不要な再レンダリング防止）
  5. atomFamily で動的な状態を管理
  6. atomWithStorage で永続化
  7. focusAtom でネストしたオブジェクトの特定フィールドに注目

共通のベストプラクティス:
  1. サーバーデータは TanStack Query に任せる（ストア/アトムに入れない）
  2. ローカルで済む状態は useState で（過度なグローバル化を避ける）
  3. TypeScript の型を正確に定義する
  4. テストを書く（ストア/アトムのロジックは純粋関数として）
  5. パフォーマンス計測してから最適化する（premature optimization を避ける）
```

---

## まとめ

| 特徴 | Zustand | Jotai |
|------|---------|-------|
| モデル | ストアベース（トップダウン） | アトムベース（ボトムアップ） |
| API | create() | atom() + useAtom() |
| 再レンダリング | セレクターで最適化 | アトム単位で自動最適化 |
| ミドルウェア | persist, devtools, immer, temporal | atomWithStorage, atomFamily, focusAtom |
| React外アクセス | getState(), setState(), subscribe() | createStore() 経由で可能 |
| DevTools | Redux DevTools | jotai-devtools |
| バンドルサイズ | ~1.1kB (gzip) | ~3.8kB (gzip) |
| 学習コスト | 低（Redux経験者は特に） | 中（アトムの概念を理解する必要） |
| 適切な規模 | 中〜大規模 | 小〜大規模 |
| 非同期処理 | ストア内で async/await | 非同期アトム or jotai-tanstack-query |
| テスト | getState()/setState() で直接操作 | createStore() or Provider でテスト |
| SSR対応 | hydrate ミドルウェア | Provider + useHydrateAtoms |

---

## 次に読むべきガイド
→ [[02-server-state.md]] — TanStack Query によるサーバー状態管理
→ [[03-url-state.md]] — URL状態管理

---

## 参考文献
1. Zustand. "Bear necessities for state management." github.com/pmndrs/zustand, 2024.
2. Jotai. "Primitive and flexible state management." jotai.org, 2024.
3. Daishi Kato. "When I Use Jotai vs Zustand." blog.axlight.com, 2024.
4. Daishi Kato. "Zustand Internals." blog.axlight.com, 2023.
5. TkDodo. "Working with Zustand." tkdodo.eu, 2024.
6. Jotai. "Comparison with Recoil." jotai.org/docs/basics/comparison, 2024.
7. pmndrs. "Zustand Middleware." github.com/pmndrs/zustand/wiki, 2024.
8. zundo. "Undo/Redo middleware for Zustand." github.com/charkour/zundo, 2024.
9. jotai-optics. "Optics for Jotai." github.com/jotaijs/jotai-optics, 2024.
10. jotai-tanstack-query. "TanStack Query integration." github.com/jotaijs/jotai-tanstack-query, 2024.
