# 状態管理概論

> 状態管理はWebアプリの複雑さの根源。ローカル状態、グローバル状態、サーバー状態、URL状態の分類を理解し、各カテゴリに最適なツールを選択することで、シンプルで保守しやすい状態管理を実現する。

## この章で学ぶこと

- [ ] 状態の4つのカテゴリを理解する
- [ ] 各カテゴリに適したツールの選定基準を把握する
- [ ] 状態管理の設計原則を学ぶ
- [ ] パフォーマンスを考慮した状態設計ができるようになる
- [ ] 実務での状態管理アンチパターンを回避できるようになる
- [ ] 大規模アプリケーションでの状態管理戦略を策定できるようになる

---

## 1. 状態とは何か

Webアプリケーションにおける「状態」とは、アプリケーションが現在どのような振る舞いをすべきかを決定するデータの総体である。ボタンが押されたか、ユーザーがログインしているか、APIからどんなデータが返ってきたか、URLにどんなパラメータが含まれているか。これらすべてが「状態」であり、UIはこの状態の関数として描画される。

```
UI = f(state)

この式が意味すること:
  - 同じ状態が与えられれば、同じUIが描画される
  - 状態が変化するとUIが再描画される
  - UIの問題 = 状態の問題（デバッグの基本方針）

Reactの基本的なメンタルモデル:
  1. 状態を宣言する（useState, useReducer）
  2. 状態に基づいてUIを宣言的に記述する
  3. イベントハンドラで状態を更新する
  4. Reactが差分検出して効率的にDOMを更新する

重要な区別:
  - 状態（State）: 時間とともに変化するデータ
  - 定数（Constant）: 変化しないデータ → 状態にすべきでない
  - 導出値（Derived）: 既存の状態から計算可能 → 状態にすべきでない
  - Props: 親から渡されるデータ → 子コンポーネントの状態にすべきでない
```

### 1.1 状態管理が難しい理由

```
なぜ状態管理が複雑化するのか:

  ① 状態の散在:
     → 同じデータが複数のコンポーネントで必要
     → どこに配置すべきかの判断が難しい
     → Props Drilling vs Context vs 外部ストア

  ② 状態の同期:
     → クライアント側のキャッシュとサーバーのデータのズレ
     → 複数タブ間での状態同期
     → オフライン/オンライン切り替え時の整合性

  ③ 状態の正規化:
     → ネストしたオブジェクトの更新の複雑さ
     → 同じエンティティが複数の場所に存在
     → 部分的な更新と全体の整合性

  ④ 非同期状態:
     → ローディング、エラー、成功の3状態の管理
     → 競合する複数のリクエストの処理
     → 楽観的更新とロールバック

  ⑤ パフォーマンス:
     → 不必要な再レンダリングの発生
     → メモリリーク（適切なクリーンアップの欠如）
     → 巨大な状態ツリーの管理コスト
```

---

## 2. 状態の4つのカテゴリ

```
4つの状態カテゴリ:

  ① ローカル状態（UI State）:
     → コンポーネント固有の一時的な状態
     → モーダルの開閉、フォーム入力値、ホバー状態
     → ツール: useState, useReducer
     → ライフサイクル: コンポーネントのマウント〜アンマウント

  ② グローバル状態（Client State）:
     → 複数コンポーネントで共有する状態
     → テーマ、言語設定、ユーザー認証状態
     → ツール: Zustand, Jotai, Context
     → ライフサイクル: アプリ全体のライフタイム

  ③ サーバー状態（Server State）:
     → APIから取得したデータ
     → ユーザー一覧、商品データ、注文履歴
     → ツール: TanStack Query, SWR
     → ライフサイクル: キャッシュの有効期限に基づく

  ④ URL状態（URL State）:
     → URLに反映される状態
     → 検索クエリ、フィルタ、ページ番号、ソート
     → ツール: useSearchParams, nuqs
     → ライフサイクル: ナビゲーションに連動

よくある間違い:
  ✗ サーバー状態を useState で管理
    → キャッシュ、リトライ、再検証が全て手動に
    → TanStack Query に任せるべき

  ✗ ローカル状態をグローバルに置く
    → 不要な再レンダリング
    → useState で十分

  ✗ URL状態を useState で管理
    → ブックマーク不可、共有不可
    → useSearchParams に

  ✗ 導出値を状態として管理
    → 同期が崩れるバグの温床
    → useMemo で計算すべき

原則:
  「最も局所的な場所で、最も適切なツールで管理する」
```

### 2.1 状態カテゴリの判定フローチャート

```
状態カテゴリ判定フロー:

  Q1: そのデータはAPIから取得するものか？
  │
  ├─ Yes → サーバー状態（TanStack Query / SWR）
  │
  └─ No
     │
     Q2: URLに反映すべきか？（ブックマーク/共有で保持したい？）
     │
     ├─ Yes → URL状態（useSearchParams / nuqs）
     │
     └─ No
        │
        Q3: 複数のコンポーネントで共有するか？
        │
        ├─ Yes
        │  │
        │  Q4: 更新頻度はどの程度か？
        │  │
        │  ├─ 低頻度（テーマ/認証/言語）→ Context
        │  │
        │  └─ 中〜高頻度 → Zustand / Jotai
        │
        └─ No → ローカル状態（useState / useReducer）

実務での判断例:

  「ショッピングカートの中身」
  → 複数ページで参照 → グローバル状態
  → ただしサーバーに永続化するなら → サーバー状態

  「検索結果のフィルタ条件」
  → URLに反映してブックマーク可能にしたい → URL状態

  「フォームの入力途中のデータ」
  → そのページでしか使わない → ローカル状態

  「ログイン中ユーザーの情報」
  → APIから取得 → サーバー状態（TanStack Queryで管理）
  → 認証トークン自体 → グローバル状態
```

---

## 3. ローカル状態の詳細

### 3.1 useState: 最もシンプルな状態管理

```typescript
// useState: 最もシンプル
function ToggleButton() {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <button onClick={() => setIsOpen(!isOpen)}>
      {isOpen ? 'Close' : 'Open'}
    </button>
  );
}

// フォーム入力の管理
function LoginForm() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await login(email, password);
    } catch (err) {
      setError('ログインに失敗しました');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="メールアドレス"
      />
      <div className="password-field">
        <input
          type={showPassword ? 'text' : 'password'}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="パスワード"
        />
        <button
          type="button"
          onClick={() => setShowPassword(!showPassword)}
        >
          {showPassword ? '隠す' : '表示'}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      <button type="submit">ログイン</button>
    </form>
  );
}
```

### 3.2 useState の注意点

```typescript
// ① バッチ更新の理解
function Counter() {
  const [count, setCount] = useState(0);

  // NG: 同じレンダリングサイクル内の値を参照
  const handleClick = () => {
    setCount(count + 1);
    setCount(count + 1); // count は古い値のまま → 結果: +1
  };

  // OK: 関数型アップデートで前の値を参照
  const handleClickCorrect = () => {
    setCount((prev) => prev + 1);
    setCount((prev) => prev + 1); // prev は更新後の値 → 結果: +2
  };

  return <button onClick={handleClickCorrect}>{count}</button>;
}

// ② 初期値の遅延初期化
function ExpensiveComponent() {
  // NG: 毎レンダリングで computeExpensiveValue() が実行される
  // （結果は初回のみ使われるが、関数呼び出し自体は毎回行われる）
  const [value, setValue] = useState(computeExpensiveValue());

  // OK: 関数を渡すと初回のみ実行される
  const [value2, setValue2] = useState(() => computeExpensiveValue());

  return <div>{value2}</div>;
}

// ③ オブジェクト状態の更新
function UserProfile() {
  const [user, setUser] = useState({
    name: 'Taro',
    email: 'taro@example.com',
    preferences: {
      theme: 'dark',
      language: 'ja',
    },
  });

  // NG: 直接変更（Reactが変更を検知できない）
  const updateThemeBad = () => {
    user.preferences.theme = 'light';
    setUser(user); // 同じ参照なので再レンダリングされない
  };

  // OK: イミュータブルに更新
  const updateThemeGood = () => {
    setUser({
      ...user,
      preferences: {
        ...user.preferences,
        theme: 'light',
      },
    });
  };

  return <button onClick={updateThemeGood}>テーマ変更</button>;
}
```

### 3.3 useReducer: 複雑な状態遷移

```typescript
// useReducer: 複雑な状態遷移
type State = { count: number; step: number; history: number[] };
type Action =
  | { type: 'increment' }
  | { type: 'decrement' }
  | { type: 'setStep'; step: number }
  | { type: 'reset' }
  | { type: 'undo' };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'increment':
      return {
        ...state,
        count: state.count + state.step,
        history: [...state.history, state.count],
      };
    case 'decrement':
      return {
        ...state,
        count: state.count - state.step,
        history: [...state.history, state.count],
      };
    case 'setStep':
      return { ...state, step: action.step };
    case 'reset':
      return { count: 0, step: 1, history: [] };
    case 'undo': {
      const previous = state.history[state.history.length - 1];
      if (previous === undefined) return state;
      return {
        ...state,
        count: previous,
        history: state.history.slice(0, -1),
      };
    }
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, {
    count: 0,
    step: 1,
    history: [],
  });

  return (
    <div>
      <span>{state.count}</span>
      <button onClick={() => dispatch({ type: 'increment' })}>
        +{state.step}
      </button>
      <button onClick={() => dispatch({ type: 'decrement' })}>
        -{state.step}
      </button>
      <button onClick={() => dispatch({ type: 'undo' })}>
        元に戻す
      </button>
      <button onClick={() => dispatch({ type: 'reset' })}>
        リセット
      </button>
      <input
        type="number"
        value={state.step}
        onChange={(e) =>
          dispatch({ type: 'setStep', step: Number(e.target.value) })
        }
      />
    </div>
  );
}

// useReducer を使うべき場面:
// → 3つ以上の関連する状態
// → 状態遷移のルールが複雑
// → 次の状態が前の状態に依存
// → Undo/Redo が必要
// → テストしやすくしたい（reducerは純粋関数）
```

### 3.4 実務例: マルチステップフォーム

```typescript
// マルチステップフォームの状態管理
type FormData = {
  // Step 1: 基本情報
  firstName: string;
  lastName: string;
  email: string;
  // Step 2: 住所
  postalCode: string;
  prefecture: string;
  city: string;
  address: string;
  // Step 3: 支払い
  cardNumber: string;
  expiryDate: string;
  cvv: string;
};

type FormState = {
  currentStep: number;
  data: FormData;
  errors: Partial<Record<keyof FormData, string>>;
  isSubmitting: boolean;
  completedSteps: Set<number>;
};

type FormAction =
  | { type: 'UPDATE_FIELD'; field: keyof FormData; value: string }
  | { type: 'SET_ERRORS'; errors: Partial<Record<keyof FormData, string>> }
  | { type: 'NEXT_STEP' }
  | { type: 'PREV_STEP' }
  | { type: 'GO_TO_STEP'; step: number }
  | { type: 'SUBMIT_START' }
  | { type: 'SUBMIT_SUCCESS' }
  | { type: 'SUBMIT_ERROR'; error: string };

const initialFormState: FormState = {
  currentStep: 0,
  data: {
    firstName: '',
    lastName: '',
    email: '',
    postalCode: '',
    prefecture: '',
    city: '',
    address: '',
    cardNumber: '',
    expiryDate: '',
    cvv: '',
  },
  errors: {},
  isSubmitting: false,
  completedSteps: new Set(),
};

function formReducer(state: FormState, action: FormAction): FormState {
  switch (action.type) {
    case 'UPDATE_FIELD':
      return {
        ...state,
        data: { ...state.data, [action.field]: action.value },
        errors: { ...state.errors, [action.field]: undefined },
      };

    case 'SET_ERRORS':
      return { ...state, errors: action.errors };

    case 'NEXT_STEP':
      return {
        ...state,
        currentStep: Math.min(state.currentStep + 1, 2),
        completedSteps: new Set([
          ...state.completedSteps,
          state.currentStep,
        ]),
      };

    case 'PREV_STEP':
      return {
        ...state,
        currentStep: Math.max(state.currentStep - 1, 0),
      };

    case 'GO_TO_STEP':
      if (action.step <= state.currentStep || state.completedSteps.has(action.step - 1)) {
        return { ...state, currentStep: action.step };
      }
      return state;

    case 'SUBMIT_START':
      return { ...state, isSubmitting: true };

    case 'SUBMIT_SUCCESS':
      return { ...state, isSubmitting: false };

    case 'SUBMIT_ERROR':
      return {
        ...state,
        isSubmitting: false,
        errors: { ...state.errors },
      };

    default:
      return state;
  }
}

function MultiStepForm() {
  const [state, dispatch] = useReducer(formReducer, initialFormState);
  const { currentStep, data, errors, isSubmitting } = state;

  const validateStep = (step: number): boolean => {
    const newErrors: Partial<Record<keyof FormData, string>> = {};

    if (step === 0) {
      if (!data.firstName) newErrors.firstName = '名前は必須です';
      if (!data.lastName) newErrors.lastName = '姓は必須です';
      if (!data.email) newErrors.email = 'メールは必須です';
      if (data.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) {
        newErrors.email = '有効なメールアドレスを入力してください';
      }
    }

    if (step === 1) {
      if (!data.postalCode) newErrors.postalCode = '郵便番号は必須です';
      if (!data.prefecture) newErrors.prefecture = '都道府県は必須です';
      if (!data.city) newErrors.city = '市区町村は必須です';
    }

    if (step === 2) {
      if (!data.cardNumber) newErrors.cardNumber = 'カード番号は必須です';
      if (!data.expiryDate) newErrors.expiryDate = '有効期限は必須です';
      if (!data.cvv) newErrors.cvv = 'CVVは必須です';
    }

    dispatch({ type: 'SET_ERRORS', errors: newErrors });
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      dispatch({ type: 'NEXT_STEP' });
    }
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;
    dispatch({ type: 'SUBMIT_START' });
    try {
      await submitOrder(data);
      dispatch({ type: 'SUBMIT_SUCCESS' });
    } catch (err) {
      dispatch({ type: 'SUBMIT_ERROR', error: '送信に失敗しました' });
    }
  };

  return (
    <div>
      <StepIndicator
        currentStep={currentStep}
        completedSteps={state.completedSteps}
        onStepClick={(step) => dispatch({ type: 'GO_TO_STEP', step })}
      />
      {currentStep === 0 && (
        <BasicInfoStep data={data} errors={errors} dispatch={dispatch} />
      )}
      {currentStep === 1 && (
        <AddressStep data={data} errors={errors} dispatch={dispatch} />
      )}
      {currentStep === 2 && (
        <PaymentStep data={data} errors={errors} dispatch={dispatch} />
      )}
      <div className="navigation">
        {currentStep > 0 && (
          <button onClick={() => dispatch({ type: 'PREV_STEP' })}>
            戻る
          </button>
        )}
        {currentStep < 2 ? (
          <button onClick={handleNext}>次へ</button>
        ) : (
          <button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? '送信中...' : '注文を確定'}
          </button>
        )}
      </div>
    </div>
  );
}
```

---

## 4. グローバル状態の選定

### 4.1 ライブラリ比較

```
ライブラリ詳細比較:

  Zustand:
  → シンプル、ボイラープレート最小
  → ストア = 関数（Reduxより直感的）
  → React外からもアクセス可能
  → バンドルサイズ: ~1.1kB (gzip)
  → TypeScript対応: 優秀（型推論が自然）
  → DevTools: Redux DevTools に対応
  → ミドルウェア: persist, devtools, immer, subscribeWithSelector
  → 推奨: 中規模以上のアプリ
  → 学習コスト: 低

  Jotai:
  → アトムベース（Recoilの後継的）
  → コンポーネント単位の細かい再レンダリング制御
  → バンドルサイズ: ~3.8kB (gzip)
  → TypeScript対応: 優秀（ジェネリクス活用）
  → DevTools: React DevTools の Atoms Inspector
  → 拡張: atomWithStorage, atomWithQuery, atomWithMachine
  → 推奨: 複雑なUIの状態管理
  → 学習コスト: 中

  React Context:
  → React組み込み、追加依存なし
  → 頻繁に変化する値には不向き（再レンダリング問題）
  → バンドルサイズ: 0kB（React内蔵）
  → TypeScript対応: 手動の型定義が必要
  → DevTools: React DevTools で確認可能
  → 推奨: テーマ、認証情報等の低頻度更新
  → 学習コスト: 低

  Redux Toolkit:
  → 最も成熟したエコシステム
  → DevTools が優秀
  → ボイラープレートが多い
  → バンドルサイズ: ~12.7kB (gzip)
  → TypeScript対応: 優秀（RTK は型推論が強力）
  → ミドルウェア: RTK Query, Thunk, Saga, Observable
  → 推奨: 大規模エンタープライズ
  → 学習コスト: 高

  Valtio:
  → Proxy ベースの状態管理
  → ミュータブルな書き方が可能
  → バンドルサイズ: ~3.3kB (gzip)
  → 推奨: ミュータブルAPIを好む場合
  → 学習コスト: 低

選定フロー:
  テーマ/認証/言語（低頻度更新）→ Context
  中規模の共有状態 → Zustand
  アトム単位の細かい制御 → Jotai
  大規模 + 厳密なアーキテクチャ → Redux Toolkit
  ミュータブル志向 → Valtio
```

### 4.2 各ライブラリのコード比較

```typescript
// === 同じ機能を各ライブラリで実装（カウンター + テーマ） ===

// --- React Context ---
type ThemeContextType = {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  count: number;
  increment: () => void;
};

const ThemeContext = createContext<ThemeContextType | null>(null);

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [count, setCount] = useState(0);

  // useMemoでvalueをメモ化しないと、毎レンダリングで
  // 新しいオブジェクトが生成され、全消費者が再レンダリングされる
  const value = useMemo(
    () => ({
      theme,
      toggleTheme: () => setTheme((t) => (t === 'light' ? 'dark' : 'light')),
      count,
      increment: () => setCount((c) => c + 1),
    }),
    [theme, count]
  );

  return (
    <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
  );
}

// Context の問題: count が変わるとテーマだけ使うコンポーネントも再レンダリング
function ThemeOnlyComponent() {
  const ctx = useContext(ThemeContext);
  // count の変更でもこのコンポーネントは再レンダリングされる！
  return <div className={ctx?.theme}>テーマのみ使用</div>;
}

// --- Zustand ---
import { create } from 'zustand';

interface AppStore {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  count: number;
  increment: () => void;
}

const useAppStore = create<AppStore>((set) => ({
  theme: 'light',
  toggleTheme: () =>
    set((state) => ({
      theme: state.theme === 'light' ? 'dark' : 'light',
    })),
  count: 0,
  increment: () => set((state) => ({ count: state.count + 1 })),
}));

// Zustand: セレクターで必要な値だけ取得 → 最小限の再レンダリング
function ThemeOnlyComponentZustand() {
  const theme = useAppStore((state) => state.theme);
  // count が変わってもこのコンポーネントは再レンダリングされない！
  return <div className={theme}>テーマのみ使用</div>;
}

// --- Jotai ---
import { atom, useAtom, useAtomValue, useSetAtom } from 'jotai';

const themeAtom = atom<'light' | 'dark'>('light');
const countAtom = atom(0);

// 派生アトム
const themeClassAtom = atom((get) => {
  const theme = get(themeAtom);
  return theme === 'dark' ? 'bg-gray-900 text-white' : 'bg-white text-black';
});

function ThemeOnlyComponentJotai() {
  const themeClass = useAtomValue(themeClassAtom);
  // countAtom の変更でこのコンポーネントは再レンダリングされない！
  return <div className={themeClass}>テーマのみ使用</div>;
}

function CounterJotai() {
  const [count, setCount] = useAtom(countAtom);
  return (
    <button onClick={() => setCount((c) => c + 1)}>
      Count: {count}
    </button>
  );
}
```

### 4.3 Context の再レンダリング問題と対策

```typescript
// Context の再レンダリング問題を理解する

// NG: 1つのContextに全状態を入れる
const AppContext = createContext<{
  user: User | null;
  theme: Theme;
  notifications: Notification[];
  sidebarOpen: boolean;
} | null>(null);

// → notificationsが更新されると、themeだけ使うコンポーネントも再レンダリング

// OK: Contextを分割する
const UserContext = createContext<User | null>(null);
const ThemeContext = createContext<Theme>('light');
const NotificationContext = createContext<Notification[]>([]);
const SidebarContext = createContext<{
  isOpen: boolean;
  toggle: () => void;
}>({
  isOpen: false,
  toggle: () => {},
});

// さらに良い: 状態と更新関数を分離
const ThemeValueContext = createContext<Theme>('light');
const ThemeDispatchContext = createContext<(theme: Theme) => void>(() => {});

function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light');

  return (
    <ThemeValueContext.Provider value={theme}>
      <ThemeDispatchContext.Provider value={setTheme}>
        {children}
      </ThemeDispatchContext.Provider>
    </ThemeValueContext.Provider>
  );
}

// テーマの値だけ必要なコンポーネント
function ThemedComponent() {
  const theme = useContext(ThemeValueContext);
  // setThemeが変わっても再レンダリングされない
  return <div className={theme}>テーマ適用済み</div>;
}

// テーマの変更だけ行うコンポーネント
function ThemeToggle() {
  const setTheme = useContext(ThemeDispatchContext);
  // theme値が変わっても再レンダリングされない
  return <button onClick={() => setTheme('dark')}>ダークモード</button>;
}
```

---

## 5. サーバー状態の概要

```typescript
// サーバー状態を useState で管理する場合の問題点
function UserListBad() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchUsers()
      .then((data) => {
        if (!cancelled) {
          setUsers(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // 問題点:
  // → キャッシュなし（他のコンポーネントで同じデータが必要な場合、再取得が発生）
  // → 自動再検証なし（データが古くなっても気づけない）
  // → リトライロジックなし
  // → ローディング/エラー状態の管理が手動
  // → 重複リクエストの抑制なし
  // → 楽観的更新の実装が困難
  // → Suspense非対応
  // → ウィンドウフォーカス時の再取得なし

  return loading ? <Spinner /> : <UserTable users={users} />;
}

// TanStack Query で同じことを実現
function UserListGood() {
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users'],
    queryFn: fetchUsers,
    staleTime: 5 * 60 * 1000, // 5分間キャッシュを新鮮とみなす
    retry: 3,
    refetchOnWindowFocus: true,
  });

  // 自動で得られる機能:
  // ✓ キャッシュ（他のコンポーネントから同じqueryKeyで取得 → キャッシュから即座に返す）
  // ✓ 自動再検証（staleTime経過後、バックグラウンドで再取得）
  // ✓ リトライ（失敗時に自動リトライ）
  // ✓ ローディング/エラー状態の自動管理
  // ✓ 重複リクエストの自動抑制
  // ✓ ウィンドウフォーカス時の再取得
  // ✓ Suspense対応

  if (isLoading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  return <UserTable users={users!} />;
}
```

### 5.1 サーバー状態の特殊性

```
サーバー状態がクライアント状態と本質的に異なる点:

  ① 所有権がサーバーにある:
     → クライアントが持つのは「スナップショット」にすぎない
     → 別のユーザーがサーバー上のデータを変更する可能性がある
     → 定期的な再検証（revalidation）が必要

  ② 非同期で取得する:
     → ローディング状態が常に存在する
     → ネットワークエラーの可能性
     → レイテンシーの考慮

  ③ キャッシュの概念が必要:
     → 同じデータを何度も取得するのは無駄
     → しかしキャッシュが古くなる問題
     → stale-while-revalidate パターン

  ④ 楽観的更新が有用:
     → ユーザー操作に即座に反映 → UX向上
     → サーバー応答後に整合性を確認
     → エラー時はロールバック

stale-while-revalidate パターン:
  1. キャッシュにデータがあれば即座に返す（stale data）
  2. バックグラウンドで最新データを取得する（revalidate）
  3. 最新データが取得できたらキャッシュを更新してUIに反映

  ユーザー体験:
  → 初回: ローディング → データ表示
  → 2回目以降: 即座にキャッシュ表示 → (バックグラウンド更新) → 最新データに切り替え
  → ユーザーは「一瞬で表示される」と感じる
```

---

## 6. URL状態の概要

```typescript
// URL状態をuseStateで管理した場合の問題
function SearchPageBad() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [page, setPage] = useState(1);

  // 問題:
  // → URLに反映されない → ブックマーク不可
  // → ブラウザの戻る/進むで状態が復元されない
  // → URLをコピーして共有しても検索条件が再現されない
  // → SEO的にも不利

  return (
    <div>
      <SearchInput value={query} onChange={setQuery} />
      <CategoryFilter value={category} onChange={setCategory} />
      <Pagination page={page} onChange={setPage} />
    </div>
  );
}

// URL状態として管理した場合
function SearchPageGood() {
  const [searchParams, setSearchParams] = useSearchParams();
  const query = searchParams.get('q') ?? '';
  const category = searchParams.get('category') ?? 'all';
  const page = Number(searchParams.get('page') ?? '1');

  const updateParams = (updates: Record<string, string>) => {
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      Object.entries(updates).forEach(([key, value]) => {
        if (value) {
          next.set(key, value);
        } else {
          next.delete(key);
        }
      });
      return next;
    });
  };

  // メリット:
  // ✓ URL: /search?q=react&category=books&page=2
  // ✓ ブックマーク可能
  // ✓ ブラウザの戻る/進むで状態が復元される
  // ✓ URLを共有すれば検索条件が再現される
  // ✓ SSR/SSGでの初期値として利用可能

  return (
    <div>
      <SearchInput
        value={query}
        onChange={(q) => updateParams({ q, page: '1' })}
      />
      <CategoryFilter
        value={category}
        onChange={(c) => updateParams({ category: c, page: '1' })}
      />
      <Pagination
        page={page}
        onChange={(p) => updateParams({ page: String(p) })}
      />
    </div>
  );
}

// nuqs を使ったより型安全なURL状態管理
import { useQueryState, parseAsInteger, parseAsString } from 'nuqs';

function SearchPageNuqs() {
  const [query, setQuery] = useQueryState('q', parseAsString.withDefault(''));
  const [category, setCategory] = useQueryState(
    'category',
    parseAsString.withDefault('all')
  );
  const [page, setPage] = useQueryState('page', parseAsInteger.withDefault(1));

  // nuqs のメリット:
  // ✓ 型安全（parseAsInteger は自動的に数値型）
  // ✓ デフォルト値の指定が簡潔
  // ✓ Next.js App Router との深い統合
  // ✓ サーバーコンポーネントからの初期値渡しに対応
  // ✓ 浅いルーティング（ナビゲーションなしでURL更新）

  return (
    <div>
      <SearchInput value={query} onChange={setQuery} />
      <CategoryFilter value={category} onChange={setCategory} />
      <Pagination page={page} onChange={setPage} />
    </div>
  );
}
```

### 6.1 URL状態にすべきもの、すべきでないもの

```
URL状態にすべきもの:
  ✓ 検索クエリ（?q=react）
  ✓ フィルタ条件（?category=books&price=low）
  ✓ ソート順（?sort=price&order=asc）
  ✓ ページ番号（?page=3）
  ✓ 表示モード（?view=grid）
  ✓ タブ選択（?tab=settings）
  ✓ 日付範囲（?from=2024-01-01&to=2024-12-31）
  ✓ 選択中のアイテムID（/items/123）

URL状態にすべきでないもの:
  ✗ フォームの入力途中のデータ
  ✗ モーダルの開閉状態（議論あり、場合による）
  ✗ ホバー状態、ドラッグ状態
  ✗ アニメーション状態
  ✗ 認証トークン
  ✗ 一時的なエラーメッセージ
  ✗ 大量のデータ（URLの長さ制限）

判断基準:
  「そのページをブックマークして後で開いた時、
   その状態が復元されるべきか？」
  → Yes → URL状態
  → No → ローカル or グローバル状態
```

---

## 7. 設計原則

### 7.1 状態の最小化

```typescript
// 原則①: 状態の最小化 — 計算できる値は状態にしない

// NG: 冗長な状態
function CartBad() {
  const [items, setItems] = useState<CartItem[]>([]);
  const [totalPrice, setTotalPrice] = useState(0);     // items から計算可能
  const [itemCount, setItemCount] = useState(0);        // items から計算可能
  const [isEmpty, setIsEmpty] = useState(true);          // items から計算可能

  // items を更新するたびに他の3つも同期する必要がある → バグの温床
  const addItem = (item: CartItem) => {
    const newItems = [...items, item];
    setItems(newItems);
    setTotalPrice(newItems.reduce((sum, i) => sum + i.price * i.quantity, 0));
    setItemCount(newItems.reduce((sum, i) => sum + i.quantity, 0));
    setIsEmpty(false);
    // 1つでも更新を忘れると不整合が発生
  };

  return <div>{totalPrice}</div>;
}

// OK: 1つの状態から導出
function CartGood() {
  const [items, setItems] = useState<CartItem[]>([]);

  // 導出値: useMemo で計算（items が変わった時だけ再計算）
  const totalPrice = useMemo(
    () => items.reduce((sum, i) => sum + i.price * i.quantity, 0),
    [items]
  );
  const itemCount = useMemo(
    () => items.reduce((sum, i) => sum + i.quantity, 0),
    [items]
  );
  const isEmpty = items.length === 0; // 軽い計算は useMemo 不要

  const addItem = (item: CartItem) => {
    setItems((prev) => [...prev, item]);
    // totalPrice, itemCount は自動で再計算される → 不整合が起きない
  };

  return <div>{totalPrice}</div>;
}
```

### 7.2 Derived State（導出状態）

```typescript
// 原則②: Derived State

// NG: 同期が必要な冗長な状態
function ProductListBad() {
  const [products, setProducts] = useState<Product[]>([]);
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([]);
  const [sortedProducts, setSortedProducts] = useState<Product[]>([]);
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'price'>('name');

  // products, filter, sortBy のどれが変わっても
  // filteredProducts と sortedProducts を手動で更新する必要がある
  useEffect(() => {
    const filtered = products.filter((p) =>
      p.name.toLowerCase().includes(filter.toLowerCase())
    );
    setFilteredProducts(filtered);
  }, [products, filter]);

  useEffect(() => {
    const sorted = [...filteredProducts].sort((a, b) =>
      sortBy === 'name'
        ? a.name.localeCompare(b.name)
        : a.price - b.price
    );
    setSortedProducts(sorted);
  }, [filteredProducts, sortBy]);
  // 問題: useEffect の連鎖 → 理解しづらい、バグが生まれやすい

  return <ProductGrid products={sortedProducts} />;
}

// OK: 導出値として計算
function ProductListGood() {
  const [products, setProducts] = useState<Product[]>([]);
  const [filter, setFilter] = useState('');
  const [sortBy, setSortBy] = useState<'name' | 'price'>('name');

  // 状態は3つだけ。表示用データは計算で得る
  const displayProducts = useMemo(() => {
    return products
      .filter((p) => p.name.toLowerCase().includes(filter.toLowerCase()))
      .sort((a, b) =>
        sortBy === 'name'
          ? a.name.localeCompare(b.name)
          : a.price - b.price
      );
  }, [products, filter, sortBy]);

  return <ProductGrid products={displayProducts} />;
}
```

### 7.3 Colocate State

```typescript
// 原則③: Colocate State（状態を使う場所の近くに配置）

// NG: 不必要にグローバルにした状態
// store.ts
const useStore = create<{
  modalOpen: boolean;           // ← 1つのコンポーネントでしか使わない
  tooltipText: string;          // ← 1つのコンポーネントでしか使わない
  dropdownItems: string[];      // ← 1つのコンポーネントでしか使わない
  searchQuery: string;          // ← 実際に共有が必要
  user: User | null;            // ← 実際に共有が必要
}>((set) => ({
  // ...
}));

// OK: ローカルにすべきものはローカルに
function Modal() {
  const [isOpen, setIsOpen] = useState(false); // ローカルで十分
  return (
    <>
      <button onClick={() => setIsOpen(true)}>開く</button>
      {isOpen && <ModalDialog onClose={() => setIsOpen(false)} />}
    </>
  );
}

// グローバルストアには本当に共有が必要なものだけ
const useStore = create<{
  searchQuery: string;
  user: User | null;
}>((set) => ({
  searchQuery: '',
  user: null,
}));
```

### 7.4 Props Drilling とコンポジション

```typescript
// 原則④: Props Drilling の許容範囲と代替手段

// Props Drilling: 2-3階層は許容
function App() {
  const [user, setUser] = useState<User | null>(null);
  return <Dashboard user={user} />;
}

function Dashboard({ user }: { user: User | null }) {
  return <Header user={user} />;
}

function Header({ user }: { user: User | null }) {
  return <UserMenu user={user} />;
}

// 4階層以上の場合 → コンポジションで解決
// コンポジションパターン: children を使って中間コンポーネントを「飛ばす」
function App() {
  const [user, setUser] = useState<User | null>(null);
  return (
    <Dashboard>
      <Header>
        <UserMenu user={user} />
      </Header>
    </Dashboard>
  );
}

function Dashboard({ children }: { children: React.ReactNode }) {
  return <div className="dashboard">{children}</div>;
}

function Header({ children }: { children: React.ReactNode }) {
  return <header>{children}</header>;
}

// → Dashboard と Header は user を知る必要がない
// → UserMenu だけが user を受け取る
// → Props Drilling が解消される
```

### 7.5 Single Source of Truth

```typescript
// 原則⑤: Single Source of Truth

// NG: 同じユーザーデータを複数箇所で管理
function App() {
  // ヘッダー表示用
  const [headerUser, setHeaderUser] = useState<User | null>(null);
  // プロフィールページ用
  const [profileUser, setProfileUser] = useState<User | null>(null);
  // 設定ページ用
  const [settingsUser, setSettingsUser] = useState<User | null>(null);
  // → 1つ更新して他を忘れると不整合

  return <div>...</div>;
}

// OK: TanStack Query でサーバーデータを一元管理
function useCurrentUser() {
  return useQuery({
    queryKey: ['currentUser'],
    queryFn: fetchCurrentUser,
    staleTime: 5 * 60 * 1000,
  });
}

// どのコンポーネントから呼んでも同じキャッシュを参照
function Header() {
  const { data: user } = useCurrentUser();
  return <div>{user?.name}</div>;
}

function ProfilePage() {
  const { data: user } = useCurrentUser();
  return <div>{user?.email}</div>;
}

function SettingsPage() {
  const { data: user } = useCurrentUser();
  const queryClient = useQueryClient();

  const updateUser = async (data: Partial<User>) => {
    await api.updateUser(data);
    // キャッシュを無効化 → 全コンポーネントが最新データに
    queryClient.invalidateQueries({ queryKey: ['currentUser'] });
  };

  return <UserSettingsForm user={user!} onSave={updateUser} />;
}
```

### 7.6 不変性（Immutability）

```typescript
// 原則⑥: 不変性（Immutability）

// NG: 直接変更
function TodoListBad() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const toggleTodo = (id: string) => {
    const todo = todos.find((t) => t.id === id);
    if (todo) {
      todo.completed = !todo.completed; // 直接変更！
      setTodos([...todos]); // スプレッドしても元のオブジェクトは変更済み
    }
  };

  return <div>{/* ... */}</div>;
}

// OK: イミュータブルに更新
function TodoListGood() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const toggleTodo = (id: string) => {
    setTodos((prev) =>
      prev.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    );
  };

  const addTodo = (text: string) => {
    setTodos((prev) => [
      ...prev,
      { id: crypto.randomUUID(), text, completed: false },
    ]);
  };

  const removeTodo = (id: string) => {
    setTodos((prev) => prev.filter((todo) => todo.id !== id));
  };

  return <div>{/* ... */}</div>;
}

// ネストが深い場合は Immer を活用
import { produce } from 'immer';

function NestedStateUpdate() {
  const [state, setState] = useState({
    users: {
      byId: {
        '1': {
          name: 'Taro',
          address: {
            city: 'Tokyo',
            zip: '100-0001',
          },
        },
      },
    },
  });

  // Immer なし: スプレッドの嵐
  const updateCityManual = () => {
    setState({
      ...state,
      users: {
        ...state.users,
        byId: {
          ...state.users.byId,
          '1': {
            ...state.users.byId['1'],
            address: {
              ...state.users.byId['1'].address,
              city: 'Osaka',
            },
          },
        },
      },
    });
  };

  // Immer あり: 直感的な書き方
  const updateCityImmer = () => {
    setState(
      produce((draft) => {
        draft.users.byId['1'].address.city = 'Osaka';
      })
    );
  };

  return <div>{/* ... */}</div>;
}
```

---

## 8. パフォーマンス最適化

### 8.1 再レンダリングの理解

```typescript
// React の再レンダリングが発生する条件
// 1. state が変更された
// 2. props が変更された
// 3. 親コンポーネントが再レンダリングされた
// 4. コンテキストの値が変更された

// 再レンダリングの最適化テクニック

// ① React.memo: props が変わらなければ再レンダリングをスキップ
const ExpensiveList = React.memo(function ExpensiveList({
  items,
}: {
  items: Item[];
}) {
  console.log('ExpensiveList rendered');
  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>{item.name}</li>
      ))}
    </ul>
  );
});

// ② useMemo: 計算結果をメモ化
function Dashboard({ orders }: { orders: Order[] }) {
  // orders が変わった時だけ再計算
  const stats = useMemo(() => {
    return {
      total: orders.length,
      revenue: orders.reduce((sum, o) => sum + o.total, 0),
      averageOrder: orders.reduce((sum, o) => sum + o.total, 0) / orders.length,
      byStatus: orders.reduce(
        (acc, o) => {
          acc[o.status] = (acc[o.status] || 0) + 1;
          return acc;
        },
        {} as Record<string, number>
      ),
    };
  }, [orders]);

  return (
    <div>
      <StatCard title="総注文数" value={stats.total} />
      <StatCard title="売上" value={stats.revenue} />
      <StatCard title="平均注文額" value={stats.averageOrder} />
    </div>
  );
}

// ③ useCallback: コールバックをメモ化
function ParentComponent() {
  const [count, setCount] = useState(0);
  const [text, setText] = useState('');

  // useCallback なし: text が変わるたびに新しい関数が生成
  // → ChildComponent が React.memo でも再レンダリングされる
  const handleClickBad = () => {
    setCount((c) => c + 1);
  };

  // useCallback あり: count の変更時のみ新しい関数
  const handleClickGood = useCallback(() => {
    setCount((c) => c + 1);
  }, []);

  return (
    <div>
      <input value={text} onChange={(e) => setText(e.target.value)} />
      <MemoizedChild onClick={handleClickGood} count={count} />
    </div>
  );
}

const MemoizedChild = React.memo(function Child({
  onClick,
  count,
}: {
  onClick: () => void;
  count: number;
}) {
  console.log('Child rendered');
  return <button onClick={onClick}>Count: {count}</button>;
});
```

### 8.2 状態の構造とパフォーマンス

```typescript
// 状態の構造がパフォーマンスに与える影響

// NG: フラットな巨大配列 → 1つの変更で全体が再レンダリング
function BigListBad() {
  const [items, setItems] = useState<Item[]>(generateItems(10000));

  const toggleItem = (id: string) => {
    setItems((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, selected: !item.selected } : item
      )
    );
    // 10000要素の配列を全部mapして新しい配列を作成
    // → items が新しい参照 → リスト全体が再レンダリング
  };

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id} onClick={() => toggleItem(item.id)}>
          {item.name}
        </li>
      ))}
    </ul>
  );
}

// OK: 正規化されたデータ構造 + React.memo
function BigListGood() {
  const [itemsById, setItemsById] = useState<Record<string, Item>>({});
  const [itemIds, setItemIds] = useState<string[]>([]);

  const toggleItem = useCallback((id: string) => {
    setItemsById((prev) => ({
      ...prev,
      [id]: { ...prev[id], selected: !prev[id].selected },
    }));
    // 変更されたアイテムのみ新しいオブジェクトが生成される
  }, []);

  return (
    <ul>
      {itemIds.map((id) => (
        <MemoizedItem
          key={id}
          id={id}
          item={itemsById[id]}
          onToggle={toggleItem}
        />
      ))}
    </ul>
  );
}

const MemoizedItem = React.memo(function ItemRow({
  id,
  item,
  onToggle,
}: {
  id: string;
  item: Item;
  onToggle: (id: string) => void;
}) {
  return (
    <li onClick={() => onToggle(id)}>
      {item.name} {item.selected ? '✓' : ''}
    </li>
  );
});
// → 変更されたアイテムのみ再レンダリング
```

### 8.3 仮想化（Virtualization）

```typescript
// 大量リストのパフォーマンス対策: 仮想化
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedList({ items }: { items: Item[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 50, // 各行の推定高さ（px）
    overscan: 5, // ビューポート外に余分にレンダリングする行数
  });

  return (
    <div
      ref={parentRef}
      style={{ height: '400px', overflow: 'auto' }}
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => (
          <div
            key={virtualRow.key}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: `${virtualRow.size}px`,
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            {items[virtualRow.index].name}
          </div>
        ))}
      </div>
    </div>
  );
  // 10000アイテムでも、画面に表示される分（+ overscan）だけレンダリング
  // → DOMノード数を大幅に削減
}
```

---

## 9. 実務での状態管理アーキテクチャ

### 9.1 小規模アプリ（〜10ページ）

```
推奨構成:
  - ローカル状態: useState / useReducer
  - サーバー状態: TanStack Query
  - URL状態: useSearchParams
  - グローバル状態: React Context（必要な場合のみ）

ディレクトリ構成:
  src/
  ├── components/
  │   ├── Header.tsx          // ローカル状態のみ
  │   └── SearchForm.tsx      // ローカル + URL状態
  ├── hooks/
  │   ├── useUsers.ts         // TanStack Query
  │   └── useAuth.ts          // TanStack Query + Context
  ├── contexts/
  │   └── AuthContext.tsx      // 認証状態
  └── pages/
      └── UsersPage.tsx       // URL状態 + サーバー状態

特徴:
  → 追加ライブラリは TanStack Query のみ
  → Context は認証やテーマなど1-2個
  → シンプルで学習コストが低い
```

### 9.2 中規模アプリ（10〜50ページ）

```
推奨構成:
  - ローカル状態: useState / useReducer
  - サーバー状態: TanStack Query
  - URL状態: nuqs
  - グローバル状態: Zustand

ディレクトリ構成:
  src/
  ├── components/
  ├── hooks/
  │   ├── queries/            // TanStack Query のカスタムフック
  │   │   ├── useUsers.ts
  │   │   ├── useProducts.ts
  │   │   └── useOrders.ts
  │   └── mutations/          // TanStack Query のミューテーション
  │       ├── useCreateUser.ts
  │       └── useUpdateProduct.ts
  ├── stores/                 // Zustand ストア
  │   ├── useUIStore.ts       // UI状態（サイドバー、モーダル等）
  │   ├── useCartStore.ts     // カート状態
  │   └── usePreferenceStore.ts  // ユーザー設定
  └── pages/

特徴:
  → Zustand でクライアント状態を効率的に管理
  → TanStack Query でサーバー状態を一元管理
  → nuqs で型安全なURL状態管理
  → 明確な責務分離
```

### 9.3 大規模アプリ（50ページ以上）

```
推奨構成:
  - ローカル状態: useState / useReducer
  - サーバー状態: TanStack Query
  - URL状態: nuqs
  - グローバル状態: Zustand（ドメイン分割）
  - フォーム状態: React Hook Form + Zod

ディレクトリ構成:
  src/
  ├── features/               // 機能ベースのモジュール分割
  │   ├── auth/
  │   │   ├── hooks/
  │   │   │   ├── useLogin.ts
  │   │   │   └── useCurrentUser.ts
  │   │   ├── stores/
  │   │   │   └── useAuthStore.ts
  │   │   └── components/
  │   ├── products/
  │   │   ├── hooks/
  │   │   │   ├── queries/
  │   │   │   └── mutations/
  │   │   ├── stores/
  │   │   └── components/
  │   └── orders/
  │       ├── hooks/
  │       ├── stores/
  │       └── components/
  ├── shared/
  │   ├── stores/             // アプリ全体で共有する状態
  │   │   └── useUIStore.ts
  │   └── hooks/
  │       └── useSearchParams.ts
  └── lib/
      ├── queryClient.ts      // TanStack Query の設定
      └── api.ts              // API クライアント

特徴:
  → 機能ベースのモジュール分割で責務を明確化
  → 各機能が独自のストア、フック、コンポーネントを持つ
  → 共有状態は shared/ に集約
  → チーム開発でのコンフリクトを最小化
```

### 9.4 状態管理の実装パターン集

```typescript
// パターン1: カスタムフックで状態ロジックをカプセル化
function useToggle(initialValue = false) {
  const [value, setValue] = useState(initialValue);

  const toggle = useCallback(() => setValue((v) => !v), []);
  const setTrue = useCallback(() => setValue(true), []);
  const setFalse = useCallback(() => setValue(false), []);

  return { value, toggle, setTrue, setFalse } as const;
}

// 使用例
function Sidebar() {
  const { value: isOpen, toggle, setFalse: close } = useToggle(false);
  return (
    <>
      <button onClick={toggle}>メニュー</button>
      {isOpen && <SidebarContent onClose={close} />}
    </>
  );
}

// パターン2: useReducer + Context で Domain-Specific な状態管理
type CartState = {
  items: CartItem[];
  discount: number;
};

type CartAction =
  | { type: 'ADD_ITEM'; item: Product; quantity: number }
  | { type: 'REMOVE_ITEM'; productId: string }
  | { type: 'UPDATE_QUANTITY'; productId: string; quantity: number }
  | { type: 'APPLY_DISCOUNT'; code: string; discount: number }
  | { type: 'CLEAR' };

function cartReducer(state: CartState, action: CartAction): CartState {
  switch (action.type) {
    case 'ADD_ITEM': {
      const existing = state.items.find(
        (i) => i.productId === action.item.id
      );
      if (existing) {
        return {
          ...state,
          items: state.items.map((i) =>
            i.productId === action.item.id
              ? { ...i, quantity: i.quantity + action.quantity }
              : i
          ),
        };
      }
      return {
        ...state,
        items: [
          ...state.items,
          {
            productId: action.item.id,
            name: action.item.name,
            price: action.item.price,
            quantity: action.quantity,
          },
        ],
      };
    }
    case 'REMOVE_ITEM':
      return {
        ...state,
        items: state.items.filter((i) => i.productId !== action.productId),
      };
    case 'UPDATE_QUANTITY':
      return {
        ...state,
        items: state.items.map((i) =>
          i.productId === action.productId
            ? { ...i, quantity: action.quantity }
            : i
        ),
      };
    case 'APPLY_DISCOUNT':
      return { ...state, discount: action.discount };
    case 'CLEAR':
      return { items: [], discount: 0 };
  }
}

// パターン3: Zustand のスライスパターン
interface UserSlice {
  user: User | null;
  setUser: (user: User | null) => void;
  logout: () => void;
}

interface UISlice {
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  theme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark') => void;
}

interface NotificationSlice {
  notifications: Notification[];
  addNotification: (notification: Notification) => void;
  removeNotification: (id: string) => void;
  clearAll: () => void;
}

// スライスを結合
type AppStore = UserSlice & UISlice & NotificationSlice;

const useAppStore = create<AppStore>()((...a) => ({
  ...createUserSlice(...a),
  ...createUISlice(...a),
  ...createNotificationSlice(...a),
}));

// 各スライスは別ファイルで定義
// stores/userSlice.ts
const createUserSlice: StateCreator<AppStore, [], [], UserSlice> = (set) => ({
  user: null,
  setUser: (user) => set({ user }),
  logout: () => set({ user: null }),
});

// stores/uiSlice.ts
const createUISlice: StateCreator<AppStore, [], [], UISlice> = (set) => ({
  sidebarOpen: false,
  toggleSidebar: () =>
    set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  theme: 'light',
  setTheme: (theme) => set({ theme }),
});
```

---

## 10. React 19 と状態管理の進化

### 10.1 useActionState（旧 useFormState）

```typescript
// React 19 の useActionState
import { useActionState } from 'react';

type FormState = {
  error: string | null;
  success: boolean;
};

async function submitAction(
  prevState: FormState,
  formData: FormData
): Promise<FormState> {
  const email = formData.get('email') as string;
  const password = formData.get('password') as string;

  try {
    await login(email, password);
    return { error: null, success: true };
  } catch (err) {
    return { error: 'ログインに失敗しました', success: false };
  }
}

function LoginForm() {
  const [state, action, isPending] = useActionState(submitAction, {
    error: null,
    success: false,
  });

  return (
    <form action={action}>
      <input name="email" type="email" />
      <input name="password" type="password" />
      {state.error && <p className="error">{state.error}</p>}
      <button type="submit" disabled={isPending}>
        {isPending ? 'ログイン中...' : 'ログイン'}
      </button>
    </form>
  );
}
```

### 10.2 useOptimistic

```typescript
// React 19 の useOptimistic
import { useOptimistic, useTransition } from 'react';

function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [optimisticTodos, addOptimisticTodo] = useOptimistic(
    todos,
    (state, newTodo: Todo) => [...state, newTodo]
  );
  const [isPending, startTransition] = useTransition();

  const addTodo = async (text: string) => {
    const newTodo: Todo = {
      id: crypto.randomUUID(),
      text,
      completed: false,
    };

    startTransition(async () => {
      // 楽観的にUIを更新（即座に表示）
      addOptimisticTodo(newTodo);
      // サーバーに送信
      const savedTodo = await api.createTodo(newTodo);
      // サーバーの応答で実際のデータに置き換え
      setTodos((prev) => [...prev, savedTodo]);
    });
  };

  return (
    <div>
      <AddTodoForm onAdd={addTodo} />
      <ul>
        {optimisticTodos.map((todo) => (
          <li key={todo.id}>{todo.text}</li>
        ))}
      </ul>
    </div>
  );
}
```

### 10.3 use() フック

```typescript
// React 19 の use() フック
import { use, Suspense } from 'react';

// Promise を直接読み取る
function UserProfile({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise);
  // Suspense が自動で Loading 状態をハンドル
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}

function App() {
  const userPromise = fetchUser(1); // Promiseを渡す（awaitしない）
  return (
    <Suspense fallback={<Spinner />}>
      <UserProfile userPromise={userPromise} />
    </Suspense>
  );
}

// Context を条件付きで読み取る（use は if 文の中で使える）
function ConditionalTheme({ useTheme }: { useTheme: boolean }) {
  // 従来の useContext はトップレベルでしか呼べなかった
  // use() は条件分岐の中で使える
  if (useTheme) {
    const theme = use(ThemeContext);
    return <div className={theme}>テーマ適用</div>;
  }
  return <div>デフォルト</div>;
}
```

---

## 11. テスト戦略

### 11.1 状態管理のテスト

```typescript
// useReducer のテスト（純粋関数なので簡単）
describe('formReducer', () => {
  const initialState: FormState = {
    currentStep: 0,
    data: { firstName: '', lastName: '', email: '' },
    errors: {},
    isSubmitting: false,
  };

  it('should update a field', () => {
    const result = formReducer(initialState, {
      type: 'UPDATE_FIELD',
      field: 'firstName',
      value: 'Taro',
    });
    expect(result.data.firstName).toBe('Taro');
    // エラーがクリアされることも確認
    expect(result.errors.firstName).toBeUndefined();
  });

  it('should advance to next step', () => {
    const result = formReducer(initialState, { type: 'NEXT_STEP' });
    expect(result.currentStep).toBe(1);
  });

  it('should not go below step 0', () => {
    const result = formReducer(initialState, { type: 'PREV_STEP' });
    expect(result.currentStep).toBe(0);
  });
});

// Zustand ストアのテスト
import { act, renderHook } from '@testing-library/react';

describe('useCartStore', () => {
  beforeEach(() => {
    // テスト間でストアをリセット
    useCartStore.setState({ items: [], discount: 0 });
  });

  it('should add an item to the cart', () => {
    const { result } = renderHook(() => useCartStore());

    act(() => {
      result.current.addItem({
        id: '1',
        name: 'テスト商品',
        price: 1000,
      });
    });

    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0].name).toBe('テスト商品');
  });

  it('should calculate total correctly', () => {
    const { result } = renderHook(() => useCartStore());

    act(() => {
      result.current.addItem({ id: '1', name: '商品A', price: 1000 });
      result.current.addItem({ id: '2', name: '商品B', price: 2000 });
    });

    expect(result.current.total).toBe(3000);
  });
});

// TanStack Query のテスト
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

describe('useUsers', () => {
  it('should fetch users successfully', async () => {
    // MSW でAPIをモック
    server.use(
      http.get('/api/users', () => {
        return HttpResponse.json([
          { id: '1', name: 'Taro' },
          { id: '2', name: 'Hanako' },
        ]);
      })
    );

    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    // 初期状態: ローディング
    expect(result.current.isLoading).toBe(true);

    // データ取得完了を待つ
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data![0].name).toBe('Taro');
  });
});
```

---

## 12. よくあるアンチパターンと解決策

### 12.1 useEffect での状態同期

```typescript
// アンチパターン①: useEffect で状態を同期する

// NG: props を state にコピー
function UserProfile({ user }: { user: User }) {
  const [name, setName] = useState(user.name);

  // props が変わったら state を更新...
  useEffect(() => {
    setName(user.name);
  }, [user.name]);
  // → 1フレーム遅れる、不要な再レンダリング

  return <div>{name}</div>;
}

// OK: key を使ってコンポーネントをリセット
function UserProfilePage({ userId }: { userId: string }) {
  return <EditableUserProfile key={userId} userId={userId} />;
}

function EditableUserProfile({ userId }: { userId: string }) {
  const { data: user } = useUser(userId);
  const [name, setName] = useState(user?.name ?? '');
  // key が変わるとコンポーネント全体がリマウントされ、stateがリセットされる
  return <input value={name} onChange={(e) => setName(e.target.value)} />;
}

// アンチパターン②: useEffect の連鎖
// NG: 「useEffect → setState → 別のuseEffect → setState ...」
function FilteredListBad() {
  const [items, setItems] = useState<Item[]>([]);
  const [filter, setFilter] = useState('');
  const [filtered, setFiltered] = useState<Item[]>([]);
  const [sorted, setSorted] = useState<Item[]>([]);

  useEffect(() => {
    setFiltered(items.filter((i) => i.name.includes(filter)));
  }, [items, filter]);

  useEffect(() => {
    setSorted([...filtered].sort((a, b) => a.name.localeCompare(b.name)));
  }, [filtered]);
  // → 3回のレンダリングが発生（items変更 → filtered変更 → sorted変更）

  return <List items={sorted} />;
}

// OK: useMemo で同期的に計算
function FilteredListGood() {
  const [items, setItems] = useState<Item[]>([]);
  const [filter, setFilter] = useState('');

  const displayItems = useMemo(() => {
    return items
      .filter((i) => i.name.includes(filter))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [items, filter]);
  // → 1回のレンダリングで完結

  return <List items={displayItems} />;
}
```

### 12.2 グローバルストアの肥大化

```typescript
// アンチパターン③: 何でもグローバルストアに入れる

// NG: 巨大な単一ストア
const useMegaStore = create<{
  // UI状態
  sidebarOpen: boolean;
  modalOpen: boolean;
  activeTab: string;
  tooltipText: string;
  dropdownOpen: boolean;
  // ユーザー状態
  user: User | null;
  isAuthenticated: boolean;
  // 商品状態
  products: Product[];
  selectedProduct: Product | null;
  // カート状態
  cartItems: CartItem[];
  cartTotal: number;
  // 検索状態
  searchQuery: string;
  searchResults: Product[];
  // 通知状態
  notifications: Notification[];
  // ... 50以上のプロパティ
}>((set) => ({
  // ... 膨大なアクション定義
}));

// OK: 関心ごとに分割
const useUIStore = create<UIStore>((set) => ({
  sidebarOpen: false,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
}));

const useCartStore = create<CartStore>((set, get) => ({
  items: [],
  addItem: (item) => set((s) => ({ items: [...s.items, item] })),
  get total() {
    return get().items.reduce((sum, i) => sum + i.price * i.quantity, 0);
  },
}));

// サーバーデータは TanStack Query に任せる（ストアに入れない）
function useProducts() {
  return useQuery({ queryKey: ['products'], queryFn: fetchProducts });
}
```

### 12.3 不要な状態の保持

```typescript
// アンチパターン④: propsを状態にコピーする

// NG
function UserCard({ user }: { user: User }) {
  const [name, setName] = useState(user.name);
  const [email, setEmail] = useState(user.email);
  // → user.name や user.email が変わっても反映されない（初期値として1回だけ使われる）

  return (
    <div>
      <p>{name}</p>
      <p>{email}</p>
    </div>
  );
}

// OK: props をそのまま使う
function UserCard({ user }: { user: User }) {
  return (
    <div>
      <p>{user.name}</p>
      <p>{user.email}</p>
    </div>
  );
}

// 編集機能がある場合は、編集中の値だけ状態にする
function EditableUserCard({ user, onSave }: {
  user: User;
  onSave: (data: Partial<User>) => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');

  const startEditing = () => {
    setEditName(user.name); // 編集開始時に初期値をセット
    setIsEditing(true);
  };

  const save = () => {
    onSave({ name: editName });
    setIsEditing(false);
  };

  return (
    <div>
      {isEditing ? (
        <>
          <input value={editName} onChange={(e) => setEditName(e.target.value)} />
          <button onClick={save}>保存</button>
          <button onClick={() => setIsEditing(false)}>キャンセル</button>
        </>
      ) : (
        <>
          <p>{user.name}</p>
          <button onClick={startEditing}>編集</button>
        </>
      )}
    </div>
  );
}
```

---

## 13. 状態管理のチェックリスト

```
プロジェクト開始時の状態管理チェックリスト:

  □ 状態のカテゴリ分類を行ったか
    - ローカル、グローバル、サーバー、URLの4分類
  □ サーバー状態には TanStack Query / SWR を使っているか
    - useState + useEffect でのデータフェッチはNG
  □ URL状態を適切に使っているか
    - ブックマーク/共有可能にすべき状態はURLに
  □ グローバル状態は本当に必要か
    - ローカルで済むものをグローバルにしていないか
  □ 導出値を状態にしていないか
    - 既存の状態から計算可能な値は useMemo で
  □ 再レンダリングの最適化は適切か
    - Context の分割、セレクター、React.memo
  □ テスト可能な設計になっているか
    - Reducer は純粋関数、ストアはリセット可能
  □ TypeScript で型安全か
    - any を使っていないか、discriminated union を活用しているか

コードレビュー時の状態管理チェックポイント:
  □ useEffect で状態を同期していないか → useMemo / 導出値に
  □ props を useState にコピーしていないか → そのまま使う or key でリセット
  □ グローバルストアが肥大化していないか → 分割
  □ 同じデータが複数箇所で管理されていないか → Single Source of Truth
  □ 不変性が守られているか → Immer or スプレッド
  □ 適切なメモ化がされているか → ただし過剰なメモ化も避ける
```

---

## 14. 状態管理ライブラリの歴史と変遷

```
Reactの状態管理ライブラリの歴史:

  2014: Flux（Facebookが提唱）
  → 単方向データフローの概念を広めた
  → 実装は複数（Fluxxor, Alt, Reflux等）

  2015: Redux（Dan Abramov）
  → Fluxの実装を統一
  → 単一ストア、純粋なReducer、不変性
  → React エコシステムの事実上の標準に
  → ボイラープレートの多さが批判の対象に

  2016-2018: MobX
  → Observable パターンで状態変更を自動追跡
  → ボイラープレートが少ない
  → 「magic」が多いという批判も

  2019: Redux Toolkit
  → Redux のボイラープレートを大幅削減
  → createSlice, createAsyncThunk
  → 公式推奨のReduxの書き方に

  2020: Recoil（Facebook実験的）
  → アトムベースの状態管理
  → React の concurrent features との相性を意識
  → 2025年時点でメンテナンス停滞

  2020: React Query（TanStack Query）
  → サーバー状態の管理を革命的に簡素化
  → 「サーバー状態はクライアント状態ではない」という認識を広めた

  2021: Zustand
  → シンプル、軽量、ボイラープレート最小
  → React の外からもアクセス可能
  → 急速にシェアを拡大

  2021: Jotai
  → Recoilのコンセプトをよりシンプルに
  → アトムベース、TypeScript ファースト
  → pmndrs（Zustandと同じ開発者グループ）

  2022-2024: サーバーコンポーネント時代
  → Next.js App Router / React Server Components
  → サーバーでデータを取得 → クライアント状態の必要性が減少
  → 「本当にクライアントで管理すべき状態」の見極めが重要に

  2024-2026: 現在のトレンド
  → 軽量ライブラリ（Zustand, Jotai）が主流
  → TanStack Query がサーバー状態管理のデファクト
  → URL状態管理（nuqs等）への関心の高まり
  → React 19 の新しいフック（useActionState, useOptimistic, use）
  → signals への関心（Preact Signals, Angular Signals）
```

---

## まとめ

| カテゴリ | 例 | 推奨ツール | 選定理由 |
|---------|-----|-----------|---------|
| ローカル | モーダル開閉、入力値 | useState, useReducer | React組み込み、追加依存なし |
| グローバル | テーマ、認証 | Zustand, Context | 軽量、シンプル、型安全 |
| サーバー | API データ | TanStack Query | キャッシュ、再検証、リトライ自動 |
| URL | 検索、フィルタ | useSearchParams, nuqs | ブックマーク、共有、SEO |
| フォーム | バリデーション | React Hook Form + Zod | パフォーマンス、型安全 |

### 状態管理の黄金律

```
1. 「最も局所的な場所で、最も適切なツールで管理する」

2. 「状態は最小限に。計算可能な値は状態にしない」

3. 「サーバーデータはサーバー状態として管理する」

4. 「URLに反映すべきものはURL状態として管理する」

5. 「グローバル状態は最後の手段。まずローカル、次にコンポジション」

6. 「Single Source of Truth を守る」

7. 「不変性を守る。直接変更しない」

8. 「テスト可能な設計にする」
```

---

## 次に読むべきガイド
→ [[01-zustand-and-jotai.md]] — Zustand / Jotai の詳細な使い方
→ [[02-server-state.md]] — TanStack Query によるサーバー状態管理
→ [[03-url-state.md]] — URL状態管理の実践

---

## 参考文献
1. Kent C. Dodds. "Application State Management with React." kentcdodds.com, 2020.
2. TkDodo. "Practical React Query." tkdodo.eu, 2024.
3. Zustand. "Documentation." github.com/pmndrs/zustand, 2024.
4. Jotai. "Documentation." jotai.org, 2024.
5. TanStack Query. "Documentation." tanstack.com/query, 2024.
6. React. "Managing State." react.dev, 2024.
7. Mark Erikson. "Blogged Answers: Why Redux Toolkit Uses Thunks for Async Logic." blog.isquaredsoftware.com, 2023.
8. Daishi Kato. "When I Use Valtio and When I Use Jotai." blog.axlight.com, 2023.
9. nuqs. "Documentation." nuqs.47ng.com, 2024.
10. React. "React 19 Blog Post." react.dev, 2024.
