# フォーム設計

> フォームはユーザーとの主要なインタラクションポイント。React Hook Form、制御/非制御コンポーネント、パフォーマンス最適化、アクセシビリティまで、使いやすく保守しやすいフォーム設計のベストプラクティスを習得する。

## この章で学ぶこと

- [ ] React Hook Formの基本パターンと応用テクニックを理解する
- [ ] 制御/非制御コンポーネントの使い分けと実装パターンを把握する
- [ ] フォームのUXとアクセシビリティのベストプラクティスを学ぶ
- [ ] Server Actionsとの統合パターンを実装できるようになる
- [ ] 複雑なフォーム（マルチステップ、動的フィールド）を設計できる
- [ ] フォームのパフォーマンス最適化手法を理解する
- [ ] テスト戦略とデバッグ手法を身につける

---

## 前提知識

この章を最大限活用するために、以下の知識を事前に習得しておくことを推奨する:

- **コンポーネントアーキテクチャ**: `../00-architecture/02-component-architecture.md` で学ぶ、Reactコンポーネントの設計原則と再利用可能なコンポーネントの構築方法を理解していること
- **React Hooksの基礎**: `useState`, `useRef`, `useEffect` などの基本的なHooksの使い方と、カスタムフックの設計パターンを把握していること
- **HTMLフォーム要素**: `<form>`, `<input>`, `<select>`, `<textarea>` といったネイティブHTML要素の動作、属性、イベントハンドリングの基礎知識を持っていること

---

## 1. フォーム設計の基本原則

フォーム設計において最も重要なのは、ユーザーが目的を最小限の摩擦で達成できるようにすることである。技術的な実装の前に、設計原則を理解しておく必要がある。

### 1.1 フォーム設計の3つの柱

```
フォーム設計の3つの柱:

1. ユーザビリティ (Usability)
   - 直感的なレイアウトとフロー
   - 明確なラベルとプレースホルダー
   - 適切なエラーメッセージとフィードバック
   - モバイルフレンドリーな入力体験

2. アクセシビリティ (Accessibility)
   - スクリーンリーダー対応
   - キーボードナビゲーション
   - 十分なコントラスト比
   - ARIA属性の適切な使用

3. パフォーマンス (Performance)
   - 最小限の再レンダリング
   - 遅延バリデーション
   - 効率的な状態管理
   - バンドルサイズの最適化
```

### 1.2 フォームライブラリの比較

Reactエコシステムにおける主要なフォームライブラリを比較する。

| 特性 | React Hook Form | Formik | React Final Form | 標準useState |
|------|----------------|--------|-----------------|-------------|
| バンドルサイズ | ~9KB | ~13KB | ~5KB | 0KB |
| 再レンダリング | 最小限（非制御ベース） | フィールド変更ごと | 最小限 | フィールド変更ごと |
| TypeScript対応 | 優秀（推論が効く） | 良好 | 良好 | 完全（手動定義） |
| バリデーション | Zod/Yup統合 | Yup統合 | 独自 | 手動実装 |
| 学習コスト | 低〜中 | 中 | 中 | 低 |
| エコシステム | 豊富（DevTools等） | 成熟 | 限定的 | なし |
| メンテナンス状況 | 活発 | やや停滞 | 安定 | - |
| パフォーマンス | 最高 | 普通 | 良好 | 実装次第 |

### 1.3 なぜ React Hook Form を選ぶのか

```typescript
// React Hook Form を選ぶ理由:

// 1. パフォーマンス: 非制御コンポーネントベースで再レンダリングが最小限
//    → フォームフィールドが多いページでも高速

// 2. DX（開発者体験）: register() で簡単にフィールド登録
//    → ボイラープレートコードが少ない

// 3. バリデーション統合: Zod, Yup, Joi など主要なバリデーションライブラリと統合
//    → スキーマファーストのバリデーション

// 4. TypeScript推論: スキーマから型が自動推論される
//    → 型安全なフォーム開発

// 5. DevTools: React Hook Form DevToolsで状態をリアルタイム確認
//    → デバッグが容易

// 6. 軽量: gzipで約9KB
//    → バンドルサイズへの影響が小さい
```

---

## 2. React Hook Form 基本パターン

### 2.1 インストールとセットアップ

```bash
# 基本インストール
npm install react-hook-form

# Zodバリデーション統合
npm install zod @hookform/resolvers

# DevTools（開発環境のみ）
npm install -D @hookform/devtools
```

### 2.2 基本的なフォーム実装

```typescript
// React Hook Form: パフォーマンス最適化されたフォームライブラリ
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// ========================================
// Step 1: Zodスキーマ定義
// ========================================
const userSchema = z.object({
  name: z.string()
    .min(1, '名前は必須です')
    .max(100, '名前は100文字以内で入力してください'),
  email: z.string()
    .email('有効なメールアドレスを入力してください'),
  age: z.coerce
    .number()
    .min(0, '年齢は0以上で入力してください')
    .max(150, '年齢は150以下で入力してください')
    .optional(),
  role: z.enum(['user', 'admin', 'editor'], {
    errorMap: () => ({ message: '有効なロールを選択してください' }),
  }),
  agreed: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),
});

// Step 2: 型の自動推論
type UserFormData = z.infer<typeof userSchema>;
// 推論される型:
// {
//   name: string;
//   email: string;
//   age?: number | undefined;
//   role: "user" | "admin" | "editor";
//   agreed: true;
// }

// ========================================
// Step 3: フォームコンポーネント
// ========================================
function CreateUserForm() {
  const {
    register,       // input を非制御コンポーネントとして登録
    handleSubmit,   // フォーム送信ハンドラ（バリデーション込み）
    formState: {
      errors,        // バリデーションエラーオブジェクト
      isSubmitting,  // 送信中フラグ
      isDirty,       // フォームが変更されたか
      isValid,       // フォームが有効か
      dirtyFields,   // 変更されたフィールド
      touchedFields, // タッチされたフィールド
    },
    reset,          // フォームリセット
    watch,          // 値の監視（再レンダリングを引き起こす）
    setValue,       // プログラマティックに値を設定
    getValues,      // 再レンダリングなしで値を取得
    setError,       // 手動でエラーを設定
    clearErrors,    // エラーをクリア
    trigger,        // バリデーションを手動トリガー
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      name: '',
      email: '',
      role: 'user',
      agreed: false as any,
    },
    mode: 'onBlur',           // バリデーションタイミング
    reValidateMode: 'onChange', // 再バリデーションタイミング
  });

  // 送信ハンドラ
  const onSubmit = async (data: UserFormData) => {
    try {
      await api.users.create(data);
      reset(); // フォームリセット
      toast.success('ユーザーを作成しました');
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setError('email', {
          message: 'このメールアドレスは既に登録されています',
        });
      } else {
        toast.error('ユーザーの作成に失敗しました');
      }
    }
  };

  // エラーハンドラ（バリデーション失敗時）
  const onError = (errors: FieldErrors<UserFormData>) => {
    console.error('Validation errors:', errors);
    // 最初のエラーフィールドにフォーカス（自動で行われる）
  };

  return (
    <form onSubmit={handleSubmit(onSubmit, onError)} noValidate>
      {/* 名前フィールド */}
      <div className="form-group">
        <label htmlFor="name">名前 *</label>
        <input
          id="name"
          type="text"
          {...register('name')}
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
          aria-required="true"
          autoComplete="name"
          placeholder="山田太郎"
        />
        {errors.name && (
          <p id="name-error" className="error-message" role="alert">
            {errors.name.message}
          </p>
        )}
      </div>

      {/* メールフィールド */}
      <div className="form-group">
        <label htmlFor="email">メールアドレス *</label>
        <input
          id="email"
          type="email"
          {...register('email')}
          aria-invalid={!!errors.email}
          aria-describedby={errors.email ? 'email-error' : undefined}
          aria-required="true"
          autoComplete="email"
          placeholder="example@example.com"
        />
        {errors.email && (
          <p id="email-error" className="error-message" role="alert">
            {errors.email.message}
          </p>
        )}
      </div>

      {/* 年齢フィールド（オプション） */}
      <div className="form-group">
        <label htmlFor="age">年齢</label>
        <input
          id="age"
          type="number"
          {...register('age')}
          aria-invalid={!!errors.age}
          aria-describedby={errors.age ? 'age-error' : 'age-hint'}
          min={0}
          max={150}
        />
        <p id="age-hint" className="hint-text">
          任意項目です
        </p>
        {errors.age && (
          <p id="age-error" className="error-message" role="alert">
            {errors.age.message}
          </p>
        )}
      </div>

      {/* ロール選択 */}
      <div className="form-group">
        <label htmlFor="role">ロール *</label>
        <select
          id="role"
          {...register('role')}
          aria-invalid={!!errors.role}
        >
          <option value="user">一般ユーザー</option>
          <option value="editor">編集者</option>
          <option value="admin">管理者</option>
        </select>
        {errors.role && (
          <p className="error-message" role="alert">
            {errors.role.message}
          </p>
        )}
      </div>

      {/* 利用規約同意 */}
      <div className="form-group">
        <label className="checkbox-label">
          <input
            type="checkbox"
            {...register('agreed')}
            aria-invalid={!!errors.agreed}
          />
          <span>利用規約に同意します *</span>
        </label>
        {errors.agreed && (
          <p className="error-message" role="alert">
            {errors.agreed.message}
          </p>
        )}
      </div>

      {/* 送信ボタン */}
      <button
        type="submit"
        disabled={isSubmitting}
        aria-busy={isSubmitting}
      >
        {isSubmitting ? '作成中...' : 'ユーザーを作成'}
      </button>
    </form>
  );
}
```

### 2.3 useForm のオプション詳細

```typescript
// useForm の全オプション解説
const form = useForm<FormData>({
  // バリデーションリゾルバー
  resolver: zodResolver(schema),

  // デフォルト値（非同期も可能）
  defaultValues: {
    name: '',
    email: '',
  },
  // または非同期でデフォルト値を取得
  // defaultValues: async () => {
  //   const user = await fetchUser(userId);
  //   return user;
  // },

  // バリデーションモード
  mode: 'onBlur',
  // 'onSubmit'  - 送信時のみ（デフォルト）
  // 'onBlur'    - フォーカスを外した時
  // 'onChange'  - 値が変わるたび
  // 'onTouched' - 最初のBlur後はonChange
  // 'all'       - onBlur + onChange

  // 再バリデーションモード（最初のエラー後）
  reValidateMode: 'onChange',
  // 'onBlur'    - フォーカスを外した時
  // 'onChange'  - 値が変わるたび（デフォルト）
  // 'onSubmit'  - 送信時のみ

  // 送信時にフォーカスをエラーフィールドに移動
  shouldFocusError: true,

  // フォーム値の比較基準
  criteriaMode: 'firstError',
  // 'firstError' - 最初のエラーのみ（デフォルト）
  // 'all'        - すべてのエラーを収集

  // アンマウント時にフィールドを保持するか
  shouldUnregister: false,

  // ネイティブバリデーションを使用するか
  shouldUseNativeValidation: false,

  // デフォルト値の変更を監視
  resetOptions: {
    keepDirtyValues: true,  // ユーザーが変更した値を保持
    keepErrors: false,
  },
});
```

### 2.4 register の詳細オプション

```typescript
// register() のオプション
<input
  {...register('fieldName', {
    // React Hook Form ネイティブバリデーション（Zod不使用時）
    required: '必須項目です',
    minLength: { value: 3, message: '3文字以上で入力してください' },
    maxLength: { value: 100, message: '100文字以内で入力してください' },
    min: { value: 0, message: '0以上の値を入力してください' },
    max: { value: 150, message: '150以下の値を入力してください' },
    pattern: {
      value: /^[A-Za-z]+$/,
      message: '英字のみ入力可能です',
    },
    validate: {
      // カスタムバリデーション（複数定義可能）
      notAdmin: (v) => v !== 'admin' || '管理者名は使用できません',
      unique: async (v) => {
        const exists = await checkUsername(v);
        return !exists || 'このユーザー名は使用されています';
      },
    },
    // フィールド値の変換
    setValueAs: (v) => v.trim(),
    // または数値変換
    // valueAsNumber: true,
    // または日付変換
    // valueAsDate: true,

    // フィールドが非表示の場合
    disabled: false,

    // onChange / onBlur イベントハンドラ
    onChange: (e) => console.log('Changed:', e.target.value),
    onBlur: (e) => console.log('Blurred:', e.target.value),

    // 依存フィールドのバリデーション
    deps: ['otherField'], // otherField変更時にこのフィールドも再バリデーション
  })}
/>
```

### 2.5 watch の使い方と注意点

```typescript
// watch: フィールドの値をリアクティブに監視
function ConditionalForm() {
  const { register, watch, control } = useForm<FormData>();

  // 1. 特定のフィールドを監視（再レンダリングが発生する）
  const role = watch('role');

  // 2. 複数フィールドを監視
  const [firstName, lastName] = watch(['firstName', 'lastName']);

  // 3. 全フィールドを監視（パフォーマンス注意）
  // const allValues = watch();

  // 4. useWatch: コンポーネントレベルで分離（推奨）
  // → 監視対象のフィールドが変更された時だけ該当コンポーネントが再レンダリング
  return (
    <form>
      <select {...register('role')}>
        <option value="user">User</option>
        <option value="admin">Admin</option>
      </select>

      {/* 条件付きフィールド表示 */}
      {role === 'admin' && (
        <div>
          <label htmlFor="adminCode">管理者コード</label>
          <input
            id="adminCode"
            {...register('adminCode', { required: '管理者コードは必須です' })}
          />
        </div>
      )}

      {/* useWatch を使った分離された監視コンポーネント */}
      <PriceDisplay control={control} />
    </form>
  );
}

// useWatch: 特定コンポーネントだけが再レンダリングされる
import { useWatch } from 'react-hook-form';

function PriceDisplay({ control }: { control: Control<FormData> }) {
  const [quantity, unitPrice] = useWatch({
    control,
    name: ['quantity', 'unitPrice'],
  });

  const total = (quantity || 0) * (unitPrice || 0);

  return (
    <div className="price-display">
      合計金額: {total.toLocaleString()}円
    </div>
  );
}
```

### 2.6 DevTools の活用

```typescript
// React Hook Form DevTools
import { DevTool } from '@hookform/devtools';

function FormWithDevTools() {
  const { control, register, handleSubmit } = useForm();

  return (
    <>
      <form onSubmit={handleSubmit(onSubmit)}>
        <input {...register('name')} />
        <input {...register('email')} />
        <button type="submit">Submit</button>
      </form>

      {/* 開発環境でのみ表示 */}
      {process.env.NODE_ENV === 'development' && (
        <DevTool control={control} placement="top-right" />
      )}
    </>
  );
}

// DevTools で確認できる情報:
// - フォームの現在値
// - バリデーションエラー
// - touched / dirty / valid の状態
// - フィールドの登録状態
// - 送信回数と成否
```

---

## 3. 制御 / 非制御コンポーネント

### 3.1 概念の理解

```
非制御コンポーネント (Uncontrolled):
  ┌────────────────────────────────────────┐
  │ DOM が値を管理                          │
  │ register() で ref を登録               │
  │ パフォーマンスが良い（再レンダリングなし）│
  │                                        │
  │ 適用: ネイティブ HTML要素              │
  │   - <input type="text" />              │
  │   - <input type="email" />             │
  │   - <input type="number" />            │
  │   - <input type="checkbox" />          │
  │   - <input type="radio" />             │
  │   - <select />                         │
  │   - <textarea />                       │
  └────────────────────────────────────────┘

制御コンポーネント (Controlled):
  ┌────────────────────────────────────────┐
  │ React が値を管理                        │
  │ Controller で React Hook Form と統合   │
  │ カスタムUIコンポーネントに必要          │
  │                                        │
  │ 適用: カスタム / サードパーティUI       │
  │   - DatePicker                         │
  │   - Autocomplete                       │
  │   - Rich Text Editor                   │
  │   - shadcn/ui コンポーネント           │
  │   - Material UI コンポーネント         │
  │   - Radix UI Primitives               │
  └────────────────────────────────────────┘
```

### 3.2 非制御コンポーネントのパターン

```typescript
// 非制御コンポーネント: register() を使用
function UncontrolledExample() {
  const { register, handleSubmit, formState: { errors } } = useForm({
    defaultValues: {
      username: '',
      bio: '',
      newsletter: false,
      category: 'general',
      priority: 'medium',
    },
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {/* テキスト入力 */}
      <input type="text" {...register('username')} />

      {/* テキストエリア */}
      <textarea {...register('bio')} rows={5} />

      {/* チェックボックス */}
      <label>
        <input type="checkbox" {...register('newsletter')} />
        ニュースレターを購読する
      </label>

      {/* セレクトボックス */}
      <select {...register('category')}>
        <option value="general">一般</option>
        <option value="tech">テクノロジー</option>
        <option value="design">デザイン</option>
      </select>

      {/* ラジオボタン */}
      <fieldset>
        <legend>優先度</legend>
        <label>
          <input type="radio" value="low" {...register('priority')} />
          低
        </label>
        <label>
          <input type="radio" value="medium" {...register('priority')} />
          中
        </label>
        <label>
          <input type="radio" value="high" {...register('priority')} />
          高
        </label>
      </fieldset>

      <button type="submit">送信</button>
    </form>
  );
}
```

### 3.3 制御コンポーネントのパターン

```typescript
// Controller の使用例: カスタムUIコンポーネント統合
import { Controller, useForm } from 'react-hook-form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/shared/components/ui/select';
import { DatePicker } from '@/shared/components/ui/date-picker';
import { Slider } from '@/shared/components/ui/slider';
import { Switch } from '@/shared/components/ui/switch';
import { Combobox } from '@/shared/components/ui/combobox';

interface ProjectFormData {
  projectName: string;
  category: string;
  startDate: Date;
  budget: number;
  isPublic: boolean;
  assignee: { id: string; name: string } | null;
}

function ProjectForm() {
  const {
    control,
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ProjectFormData>({
    defaultValues: {
      projectName: '',
      category: '',
      startDate: new Date(),
      budget: 50,
      isPublic: false,
      assignee: null,
    },
  });

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {/* 非制御: ネイティブinput */}
      <input {...register('projectName')} />

      {/* 制御: shadcn/ui Select */}
      <Controller
        name="category"
        control={control}
        rules={{ required: 'カテゴリを選択してください' }}
        render={({ field, fieldState: { error } }) => (
          <div>
            <Select
              onValueChange={field.onChange}
              defaultValue={field.value}
            >
              <SelectTrigger aria-invalid={!!error}>
                <SelectValue placeholder="カテゴリを選択" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="web">Web開発</SelectItem>
                <SelectItem value="mobile">モバイル開発</SelectItem>
                <SelectItem value="design">デザイン</SelectItem>
                <SelectItem value="marketing">マーケティング</SelectItem>
              </SelectContent>
            </Select>
            {error && (
              <p className="error-message" role="alert">
                {error.message}
              </p>
            )}
          </div>
        )}
      />

      {/* 制御: DatePicker */}
      <Controller
        name="startDate"
        control={control}
        rules={{ required: '開始日を選択してください' }}
        render={({ field, fieldState: { error } }) => (
          <div>
            <DatePicker
              value={field.value}
              onChange={field.onChange}
              onBlur={field.onBlur}
              aria-invalid={!!error}
            />
            {error && (
              <p className="error-message" role="alert">
                {error.message}
              </p>
            )}
          </div>
        )}
      />

      {/* 制御: Slider */}
      <Controller
        name="budget"
        control={control}
        render={({ field }) => (
          <div>
            <label>予算: {field.value}%</label>
            <Slider
              value={[field.value]}
              onValueChange={(vals) => field.onChange(vals[0])}
              min={0}
              max={100}
              step={10}
            />
          </div>
        )}
      />

      {/* 制御: Switch */}
      <Controller
        name="isPublic"
        control={control}
        render={({ field }) => (
          <div className="flex items-center gap-2">
            <Switch
              checked={field.value}
              onCheckedChange={field.onChange}
              id="is-public"
            />
            <label htmlFor="is-public">公開プロジェクト</label>
          </div>
        )}
      />

      {/* 制御: Combobox（検索付きセレクト） */}
      <Controller
        name="assignee"
        control={control}
        render={({ field }) => (
          <Combobox
            value={field.value}
            onChange={field.onChange}
            onBlur={field.onBlur}
            options={members}
            placeholder="担当者を検索..."
            displayValue={(item) => item?.name || ''}
          />
        )}
      />

      <button type="submit">作成</button>
    </form>
  );
}
```

### 3.4 制御/非制御の使い分け判断フロー

```
フィールドの種類は？
│
├─ ネイティブHTML要素（input, select, textarea）
│  │
│  ├─ 値のリアルタイム表示が必要？
│  │  ├─ YES → watch() または useWatch() を使用（非制御のまま）
│  │  └─ NO  → register() のみ（非制御）
│  │
│  └─ → 非制御コンポーネント: register()
│
├─ サードパーティUIコンポーネント
│  │
│  ├─ ref をサポートしている？
│  │  ├─ YES → register() を試す（非制御で動く場合あり）
│  │  └─ NO  → Controller が必須
│  │
│  └─ → 制御コンポーネント: Controller
│
└─ カスタムコンポーネント
   │
   ├─ forwardRef で ref を転送している？
   │  ├─ YES → register() が使える
   │  └─ NO  → Controller が必要
   │
   └─ → 通常は制御コンポーネント: Controller
```

### 3.5 パフォーマンス比較

```typescript
// 非制御コンポーネントのレンダリング挙動
// input に "hello" と入力した場合:
//
// 非制御（register）:
//   初回レンダリング: 1回
//   入力中: 0回（DOM が直接値を管理）
//   送信時: 1回
//   合計: 2回
//
// 制御（Controller + onChange）:
//   初回レンダリング: 1回
//   入力中: 5回（"h", "he", "hel", "hell", "hello"）
//   送信時: 1回
//   合計: 7回

// → フィールド数が多い場合、非制御の方が圧倒的に有利
// → ただし制御コンポーネントでも React.memo で最適化可能

// パフォーマンス最適化: useWatch をコンポーネント分離で使う
function OptimizedForm() {
  const { register, control, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {/* これらは再レンダリングされない */}
      <input {...register('field1')} />
      <input {...register('field2')} />
      <input {...register('field3')} />
      <input {...register('field4')} />

      {/* この子コンポーネントだけが再レンダリングされる */}
      <WatchedFieldDisplay control={control} />

      <button type="submit">送信</button>
    </form>
  );
}

function WatchedFieldDisplay({ control }: { control: Control }) {
  // field1 が変更された時だけこのコンポーネントが再レンダリング
  const field1Value = useWatch({ control, name: 'field1' });
  return <div>Field 1 の値: {field1Value}</div>;
}
```

---

## 4. 高度なフォームパターン

### 4.1 動的フィールド配列（useFieldArray）

```typescript
import { useForm, useFieldArray, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// スキーマ定義
const orderSchema = z.object({
  customerName: z.string().min(1, '顧客名は必須です'),
  items: z.array(
    z.object({
      productId: z.string().min(1, '商品を選択してください'),
      quantity: z.coerce.number().min(1, '1以上を入力してください'),
      price: z.coerce.number().min(0, '0以上を入力してください'),
      note: z.string().optional(),
    })
  ).min(1, '1つ以上の商品を追加してください'),
  discount: z.coerce.number().min(0).max(100).default(0),
});

type OrderFormData = z.infer<typeof orderSchema>;

function OrderForm() {
  const {
    register,
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<OrderFormData>({
    resolver: zodResolver(orderSchema),
    defaultValues: {
      customerName: '',
      items: [{ productId: '', quantity: 1, price: 0, note: '' }],
      discount: 0,
    },
  });

  const { fields, append, remove, move, swap, insert } = useFieldArray({
    control,
    name: 'items',
  });

  const onSubmit = (data: OrderFormData) => {
    console.log('Order:', data);
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <div>
        <label htmlFor="customerName">顧客名</label>
        <input id="customerName" {...register('customerName')} />
        {errors.customerName && (
          <p className="error-message" role="alert">
            {errors.customerName.message}
          </p>
        )}
      </div>

      <h3>注文商品</h3>

      {fields.map((field, index) => (
        <div key={field.id} className="item-row">
          {/* field.id をキーに使う（indexは不可） */}
          <div>
            <label>商品</label>
            <select {...register(`items.${index}.productId`)}>
              <option value="">選択してください</option>
              <option value="prod-1">商品A - ¥1,000</option>
              <option value="prod-2">商品B - ¥2,000</option>
              <option value="prod-3">商品C - ¥3,000</option>
            </select>
            {errors.items?.[index]?.productId && (
              <p className="error-message" role="alert">
                {errors.items[index]?.productId?.message}
              </p>
            )}
          </div>

          <div>
            <label>数量</label>
            <input
              type="number"
              min={1}
              {...register(`items.${index}.quantity`)}
            />
          </div>

          <div>
            <label>単価</label>
            <input
              type="number"
              min={0}
              {...register(`items.${index}.price`)}
            />
          </div>

          <div>
            <label>備考</label>
            <input {...register(`items.${index}.note`)} />
          </div>

          <button
            type="button"
            onClick={() => remove(index)}
            disabled={fields.length <= 1}
            aria-label={`商品 ${index + 1} を削除`}
          >
            削除
          </button>

          {/* 並び替えボタン */}
          <button
            type="button"
            onClick={() => move(index, Math.max(0, index - 1))}
            disabled={index === 0}
            aria-label="上へ移動"
          >
            上へ
          </button>
          <button
            type="button"
            onClick={() => move(index, Math.min(fields.length - 1, index + 1))}
            disabled={index === fields.length - 1}
            aria-label="下へ移動"
          >
            下へ
          </button>
        </div>
      ))}

      {/* 配列レベルのエラー */}
      {errors.items?.root && (
        <p className="error-message" role="alert">
          {errors.items.root.message}
        </p>
      )}

      <button
        type="button"
        onClick={() => append({ productId: '', quantity: 1, price: 0, note: '' })}
      >
        商品を追加
      </button>

      {/* 合計表示 */}
      <OrderTotal control={control} />

      <button type="submit">注文確定</button>
    </form>
  );
}

// 合計金額コンポーネント（useWatch で分離）
function OrderTotal({ control }: { control: Control<OrderFormData> }) {
  const items = useWatch({ control, name: 'items' });
  const discount = useWatch({ control, name: 'discount' });

  const subtotal = items.reduce((sum, item) => {
    return sum + (item.quantity || 0) * (item.price || 0);
  }, 0);

  const total = subtotal * (1 - (discount || 0) / 100);

  return (
    <div className="order-total">
      <p>小計: ¥{subtotal.toLocaleString()}</p>
      <p>割引: {discount}%</p>
      <p className="total">合計: ¥{Math.floor(total).toLocaleString()}</p>
    </div>
  );
}
```

### 4.2 マルチステップフォーム

```typescript
import { useForm, FormProvider, useFormContext } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';

// 各ステップのスキーマ
const step1Schema = z.object({
  firstName: z.string().min(1, '姓は必須です'),
  lastName: z.string().min(1, '名は必須です'),
  email: z.string().email('有効なメールアドレスを入力してください'),
});

const step2Schema = z.object({
  company: z.string().min(1, '会社名は必須です'),
  position: z.string().min(1, '役職は必須です'),
  department: z.string().optional(),
});

const step3Schema = z.object({
  plan: z.enum(['free', 'pro', 'enterprise']),
  paymentMethod: z.enum(['credit', 'invoice']).optional(),
  agreeToTerms: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意してください' }),
  }),
});

// 全体のスキーマ
const fullSchema = step1Schema.merge(step2Schema).merge(step3Schema);
type FullFormData = z.infer<typeof fullSchema>;

// ステップごとのバリデーションスキーマ
const stepSchemas = [step1Schema, step2Schema, step3Schema];

// ステップごとのフィールド名
const stepFields: (keyof FullFormData)[][] = [
  ['firstName', 'lastName', 'email'],
  ['company', 'position', 'department'],
  ['plan', 'paymentMethod', 'agreeToTerms'],
];

function MultiStepForm() {
  const [currentStep, setCurrentStep] = useState(0);
  const totalSteps = 3;

  const methods = useForm<FullFormData>({
    resolver: zodResolver(fullSchema),
    defaultValues: {
      firstName: '',
      lastName: '',
      email: '',
      company: '',
      position: '',
      department: '',
      plan: 'free',
      agreeToTerms: false as any,
    },
    mode: 'onBlur',
  });

  const { trigger, handleSubmit, formState: { isSubmitting } } = methods;

  // 次のステップへ進む
  const handleNext = async () => {
    // 現在のステップのフィールドのみバリデーション
    const fieldsToValidate = stepFields[currentStep];
    const isValid = await trigger(fieldsToValidate);

    if (isValid) {
      setCurrentStep((prev) => Math.min(prev + 1, totalSteps - 1));
    }
  };

  // 前のステップに戻る
  const handlePrev = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 0));
  };

  const onSubmit = async (data: FullFormData) => {
    try {
      await api.registration.submit(data);
      toast.success('登録が完了しました');
    } catch (error) {
      toast.error('登録に失敗しました');
    }
  };

  return (
    <FormProvider {...methods}>
      <form onSubmit={handleSubmit(onSubmit)}>
        {/* プログレスバー */}
        <div className="progress-bar" role="progressbar"
          aria-valuenow={currentStep + 1}
          aria-valuemin={1}
          aria-valuemax={totalSteps}
        >
          {Array.from({ length: totalSteps }, (_, i) => (
            <div
              key={i}
              className={`step ${i <= currentStep ? 'active' : ''} ${i < currentStep ? 'completed' : ''}`}
              aria-current={i === currentStep ? 'step' : undefined}
            >
              <span className="step-number">{i + 1}</span>
              <span className="step-label">
                {['基本情報', '会社情報', 'プラン選択'][i]}
              </span>
            </div>
          ))}
        </div>

        {/* ステップコンテンツ */}
        <div className="step-content">
          {currentStep === 0 && <Step1BasicInfo />}
          {currentStep === 1 && <Step2CompanyInfo />}
          {currentStep === 2 && <Step3PlanSelection />}
        </div>

        {/* ナビゲーションボタン */}
        <div className="step-navigation">
          <button
            type="button"
            onClick={handlePrev}
            disabled={currentStep === 0}
          >
            戻る
          </button>

          {currentStep < totalSteps - 1 ? (
            <button type="button" onClick={handleNext}>
              次へ
            </button>
          ) : (
            <button type="submit" disabled={isSubmitting}>
              {isSubmitting ? '送信中...' : '登録する'}
            </button>
          )}
        </div>
      </form>
    </FormProvider>
  );
}

// Step 1: 基本情報
function Step1BasicInfo() {
  const { register, formState: { errors } } = useFormContext<FullFormData>();

  return (
    <div>
      <h2>基本情報</h2>
      <div>
        <label htmlFor="firstName">姓 *</label>
        <input id="firstName" {...register('firstName')} />
        {errors.firstName && (
          <p className="error-message" role="alert">
            {errors.firstName.message}
          </p>
        )}
      </div>
      <div>
        <label htmlFor="lastName">名 *</label>
        <input id="lastName" {...register('lastName')} />
        {errors.lastName && (
          <p className="error-message" role="alert">
            {errors.lastName.message}
          </p>
        )}
      </div>
      <div>
        <label htmlFor="email">メールアドレス *</label>
        <input id="email" type="email" {...register('email')} />
        {errors.email && (
          <p className="error-message" role="alert">
            {errors.email.message}
          </p>
        )}
      </div>
    </div>
  );
}

// Step 2: 会社情報
function Step2CompanyInfo() {
  const { register, formState: { errors } } = useFormContext<FullFormData>();

  return (
    <div>
      <h2>会社情報</h2>
      <div>
        <label htmlFor="company">会社名 *</label>
        <input id="company" {...register('company')} />
        {errors.company && (
          <p className="error-message" role="alert">
            {errors.company.message}
          </p>
        )}
      </div>
      <div>
        <label htmlFor="position">役職 *</label>
        <input id="position" {...register('position')} />
        {errors.position && (
          <p className="error-message" role="alert">
            {errors.position.message}
          </p>
        )}
      </div>
      <div>
        <label htmlFor="department">部署</label>
        <input id="department" {...register('department')} />
      </div>
    </div>
  );
}

// Step 3: プラン選択
function Step3PlanSelection() {
  const { register, watch, formState: { errors } } = useFormContext<FullFormData>();
  const plan = watch('plan');

  return (
    <div>
      <h2>プラン選択</h2>
      <fieldset>
        <legend>プランを選択 *</legend>
        <label className="plan-option">
          <input type="radio" value="free" {...register('plan')} />
          <span>Free - 無料</span>
        </label>
        <label className="plan-option">
          <input type="radio" value="pro" {...register('plan')} />
          <span>Pro - ¥980/月</span>
        </label>
        <label className="plan-option">
          <input type="radio" value="enterprise" {...register('plan')} />
          <span>Enterprise - お問い合わせ</span>
        </label>
      </fieldset>

      {plan !== 'free' && (
        <div>
          <label htmlFor="paymentMethod">支払い方法</label>
          <select id="paymentMethod" {...register('paymentMethod')}>
            <option value="credit">クレジットカード</option>
            <option value="invoice">請求書払い</option>
          </select>
        </div>
      )}

      <label className="checkbox-label">
        <input type="checkbox" {...register('agreeToTerms')} />
        <span>利用規約に同意します *</span>
      </label>
      {errors.agreeToTerms && (
        <p className="error-message" role="alert">
          {errors.agreeToTerms.message}
        </p>
      )}
    </div>
  );
}
```

### 4.3 ネストされたフォーム（FormProvider）

```typescript
// FormProvider を使ったコンポーネント分割
import { FormProvider, useForm, useFormContext } from 'react-hook-form';

// 親コンポーネント
function ParentForm() {
  const methods = useForm<ProfileFormData>({
    resolver: zodResolver(profileSchema),
    defaultValues: {
      personal: { name: '', email: '' },
      address: { zip: '', prefecture: '', city: '', street: '' },
      preferences: { theme: 'light', language: 'ja', notifications: true },
    },
  });

  return (
    <FormProvider {...methods}>
      <form onSubmit={methods.handleSubmit(onSubmit)}>
        <PersonalInfoSection />
        <AddressSection />
        <PreferencesSection />
        <button type="submit">保存</button>
      </form>
    </FormProvider>
  );
}

// 子コンポーネント: useFormContext で親のフォームにアクセス
function PersonalInfoSection() {
  const { register, formState: { errors } } = useFormContext<ProfileFormData>();

  return (
    <fieldset>
      <legend>個人情報</legend>
      <input {...register('personal.name')} />
      {errors.personal?.name && (
        <p className="error-message">{errors.personal.name.message}</p>
      )}
      <input {...register('personal.email')} />
      {errors.personal?.email && (
        <p className="error-message">{errors.personal.email.message}</p>
      )}
    </fieldset>
  );
}

// 住所セクション（郵便番号から住所自動入力の例）
function AddressSection() {
  const { register, setValue, formState: { errors } } = useFormContext<ProfileFormData>();

  const handleZipCodeChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const zip = e.target.value.replace(/[^0-9]/g, '');
    if (zip.length === 7) {
      try {
        const address = await fetchAddressFromZipCode(zip);
        setValue('address.prefecture', address.prefecture, { shouldValidate: true });
        setValue('address.city', address.city, { shouldValidate: true });
      } catch {
        // 住所取得失敗 - ユーザーに手動入力を促す
      }
    }
  };

  return (
    <fieldset>
      <legend>住所</legend>
      <input
        {...register('address.zip')}
        onChange={(e) => {
          register('address.zip').onChange(e); // RHF のイベントも発火
          handleZipCodeChange(e);
        }}
        placeholder="1234567"
        inputMode="numeric"
      />
      <input {...register('address.prefecture')} placeholder="東京都" />
      <input {...register('address.city')} placeholder="渋谷区" />
      <input {...register('address.street')} placeholder="1-2-3" />
    </fieldset>
  );
}
```

---

## 5. Server Actions との統合

### 5.1 基本的な Server Actions 統合

```typescript
// React Hook Form + Server Actions
'use client';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { createUser } from './actions';
import { useTransition } from 'react';

// Server Action 側（actions.ts）
// 'use server';
// import { z } from 'zod';
// import { userSchema } from './schema';
//
// export async function createUser(formData: FormData) {
//   const rawData = Object.fromEntries(formData.entries());
//   const result = userSchema.safeParse(rawData);
//
//   if (!result.success) {
//     return {
//       success: false,
//       errors: result.error.flatten().fieldErrors,
//     };
//   }
//
//   try {
//     await db.user.create({ data: result.data });
//     return { success: true };
//   } catch (error) {
//     return {
//       success: false,
//       errors: { _form: ['ユーザーの作成に失敗しました'] },
//     };
//   }
// }

function CreateUserForm() {
  const [isPending, startTransition] = useTransition();

  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      name: '',
      email: '',
    },
  });

  return (
    <form
      action={async (formData) => {
        // Step 1: クライアントサイドバリデーション
        const valid = await form.trigger();
        if (!valid) return;

        // Step 2: Server Action 実行
        startTransition(async () => {
          const result = await createUser(formData);

          if (result?.errors) {
            // Step 3: サーバーサイドエラーをフォームに反映
            for (const [field, messages] of Object.entries(result.errors)) {
              if (field === '_form') {
                // フォーム全体のエラー
                toast.error(messages[0]);
              } else {
                form.setError(field as any, { message: messages[0] });
              }
            }
          } else if (result?.success) {
            form.reset();
            toast.success('ユーザーを作成しました');
          }
        });
      }}
    >
      <div>
        <label htmlFor="name">名前 *</label>
        <input
          id="name"
          {...form.register('name')}
          aria-invalid={!!form.formState.errors.name}
        />
        {form.formState.errors.name && (
          <p className="error-message" role="alert">
            {form.formState.errors.name.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="email">メールアドレス *</label>
        <input
          id="email"
          type="email"
          {...form.register('email')}
          aria-invalid={!!form.formState.errors.email}
        />
        {form.formState.errors.email && (
          <p className="error-message" role="alert">
            {form.formState.errors.email.message}
          </p>
        )}
      </div>

      <button type="submit" disabled={isPending}>
        {isPending ? '作成中...' : 'ユーザーを作成'}
      </button>
    </form>
  );
}
```

### 5.2 useActionState との統合（React 19）

```typescript
// React 19 の useActionState を使ったパターン
'use client';
import { useActionState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

// Server Action の戻り値型
interface ActionState {
  success: boolean;
  errors?: Record<string, string[]>;
  message?: string;
}

function CreateUserFormWithActionState() {
  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
  });

  // useActionState でサーバーアクションの状態を管理
  const [state, formAction, isPending] = useActionState<ActionState, FormData>(
    async (prevState, formData) => {
      // クライアントバリデーション
      const valid = await form.trigger();
      if (!valid) {
        return { success: false, message: 'バリデーションエラー' };
      }

      // Server Action 呼び出し
      const result = await createUser(formData);

      if (!result.success && result.errors) {
        // サーバーエラーをフォームに反映
        for (const [field, messages] of Object.entries(result.errors)) {
          form.setError(field as keyof UserFormData, {
            message: messages[0],
          });
        }
      }

      return result;
    },
    { success: false }
  );

  return (
    <form action={formAction}>
      {/* フォームフィールド */}
      <input {...form.register('name')} />
      <input {...form.register('email')} />

      {/* サーバーからの成功メッセージ */}
      {state.success && (
        <div className="success-message" role="status">
          {state.message || 'ユーザーを作成しました'}
        </div>
      )}

      <button type="submit" disabled={isPending}>
        {isPending ? '作成中...' : '作成'}
      </button>
    </form>
  );
}
```

### 5.3 プログレッシブエンハンスメント

```typescript
// JavaScript が無効でも動作するフォーム設計
// Server Actions はプログレッシブエンハンスメントをネイティブにサポート

// Pattern 1: Server Action のみ（JS不要で動作）
async function submitForm(formData: FormData) {
  'use server';
  const name = formData.get('name') as string;
  const email = formData.get('email') as string;

  // サーバーサイドバリデーション
  if (!name || !email) {
    redirect('/form?error=validation');
  }

  await db.user.create({ data: { name, email } });
  redirect('/users');
}

// Pattern 2: React Hook Form + Server Actions（JS有効時はクライアントバリデーション付き）
function ProgressiveForm() {
  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
  });

  return (
    <form
      action={submitForm}                    // JS無効時: Server Action直接実行
      onSubmit={form.handleSubmit(           // JS有効時: クライアントバリデーション
        async (data) => {
          const formData = new FormData();
          Object.entries(data).forEach(([key, value]) => {
            formData.append(key, String(value));
          });
          await submitForm(formData);
        }
      )}
    >
      <input name="name" {...form.register('name')} />
      <input name="email" type="email" {...form.register('email')} />

      {/* JS無効時はnoscriptでメッセージを表示 */}
      <noscript>
        <p className="info-text">
          JavaScriptが無効の場合、サーバーサイドでバリデーションが行われます。
        </p>
      </noscript>

      <button type="submit">送信</button>
    </form>
  );
}
```

### 5.4 Optimistic Updates パターン

```typescript
// useOptimistic を使った楽観的更新
'use client';
import { useOptimistic } from 'react';
import { useForm } from 'react-hook-form';

interface Comment {
  id: string;
  text: string;
  author: string;
  createdAt: string;
  isPending?: boolean;
}

function CommentForm({ comments }: { comments: Comment[] }) {
  const form = useForm<{ text: string }>({
    defaultValues: { text: '' },
  });

  const [optimisticComments, addOptimisticComment] = useOptimistic<
    Comment[],
    Comment
  >(
    comments,
    (state, newComment) => [...state, newComment]
  );

  return (
    <div>
      {/* コメント一覧 */}
      <ul>
        {optimisticComments.map((comment) => (
          <li key={comment.id} className={comment.isPending ? 'opacity-50' : ''}>
            <p>{comment.text}</p>
            <span>{comment.author}</span>
            {comment.isPending && <span className="badge">送信中...</span>}
          </li>
        ))}
      </ul>

      {/* コメント投稿フォーム */}
      <form
        action={async (formData) => {
          const text = formData.get('text') as string;

          // 楽観的にUIを更新（即座に表示）
          addOptimisticComment({
            id: `temp-${Date.now()}`,
            text,
            author: currentUser.name,
            createdAt: new Date().toISOString(),
            isPending: true,
          });

          form.reset();

          // Server Action で実際に保存
          await addComment(formData);
        }}
      >
        <textarea {...form.register('text')} name="text" required />
        <button type="submit">コメントを投稿</button>
      </form>
    </div>
  );
}
```

---

## 6. フォームUXのベストプラクティス

### 6.1 バリデーションタイミングの設計

```
バリデーションタイミング戦略:

推奨パターン: 「送信時 → 以降リアルタイム」

1. 初回入力時:
   ✗ エラーを表示しない
   ✗ フォームを開いた瞬間にバリデーションしない
   → ユーザーがまだ入力を完了していない

2. 最初の送信時:
   ✓ 全フィールドをバリデーション
   ✓ エラーがあれば該当フィールドにフォーカス
   → ユーザーに入力完了の意思がある

3. 送信後の入力中:
   ✓ onChange / onBlur でリアルタイムバリデーション
   ✓ エラー修正時は即座にエラー表示を消す
   → ユーザーがエラーを修正する際のフィードバック

React Hook Form での設定:
  mode: 'onSubmit'         → 送信時のみバリデーション
  reValidateMode: 'onChange' → 送信後はリアルタイムバリデーション
  または
  mode: 'onBlur'           → フォーカスを外した時にバリデーション
  reValidateMode: 'onChange' → エラー後はリアルタイムバリデーション
```

### 6.2 エラーメッセージの設計

```typescript
// エラーメッセージの設計原則

// 良いエラーメッセージ:
// 1. 何が問題かを具体的に説明する
// 2. どうすれば解決できるかを示す
// 3. ユーザーを責めない

const goodMessages = {
  required: '名前を入力してください',           // 何をすべきか
  email: '有効なメールアドレスの形式で入力してください（例: user@example.com）',
  minLength: '8文字以上のパスワードを入力してください',
  pattern: '半角英数字のみ使用できます',
  unique: 'このメールアドレスは既に登録されています。ログインしますか？',
};

// 悪いエラーメッセージ:
const badMessages = {
  required: '入力エラー',            // 何が問題か不明
  email: 'Invalid email',           // 英語のまま / 解決方法不明
  minLength: 'Error: too short',    // 技術的すぎる
  pattern: '不正な入力です',         // 漠然としている
  unique: 'エラー: 409 Conflict',   // HTTP ステータスコードをそのまま表示
};

// エラーメッセージコンポーネント
function FieldError({ error }: { error?: FieldError }) {
  if (!error) return null;

  return (
    <div className="field-error" role="alert" aria-live="assertive">
      <svg className="error-icon" aria-hidden="true" /* ... */ />
      <span>{error.message}</span>
    </div>
  );
}
```

### 6.3 送信状態の管理

```typescript
// ダブルサブミット防止とローディング表示
function SubmitButton({
  isSubmitting,
  isDirty,
  isValid,
  label = '送信',
}: {
  isSubmitting: boolean;
  isDirty: boolean;
  isValid: boolean;
  label?: string;
}) {
  return (
    <button
      type="submit"
      disabled={isSubmitting || !isDirty}
      aria-busy={isSubmitting}
      aria-disabled={isSubmitting || !isDirty}
      className={cn(
        'submit-button',
        isSubmitting && 'loading',
        !isDirty && 'disabled',
      )}
    >
      {isSubmitting ? (
        <>
          <Spinner aria-hidden="true" />
          <span>送信中...</span>
        </>
      ) : (
        label
      )}
    </button>
  );
}
```

### 6.4 未保存変更の警告

```typescript
// ページ離脱時の警告
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

function useUnsavedChangesWarning(isDirty: boolean) {
  const router = useRouter();

  useEffect(() => {
    // ブラウザのネイティブ離脱警告
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, [isDirty]);

  // Next.js App Router でのページ遷移警告
  useEffect(() => {
    if (!isDirty) return;

    const originalPush = router.push;

    router.push = (...args) => {
      const confirmed = window.confirm(
        '未保存の変更があります。ページを離れますか？'
      );
      if (confirmed) {
        originalPush.apply(router, args);
      }
    };

    return () => {
      router.push = originalPush;
    };
  }, [isDirty, router]);
}

// フォームでの使用
function EditForm() {
  const { register, handleSubmit, formState: { isDirty } } = useForm();

  // ページ離脱警告を有効化
  useUnsavedChangesWarning(isDirty);

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      {isDirty && (
        <div className="unsaved-banner" role="status">
          未保存の変更があります
        </div>
      )}
      {/* フォームフィールド */}
    </form>
  );
}
```

### 6.5 フォームレイアウトのベストプラクティス

```
フォームレイアウト原則:

1. 単一カラムレイアウトを基本とする
   ✓ 上から下への自然な視線の流れ
   ✓ モバイルでもそのまま表示可能
   ✗ 2カラムは関連フィールドのみ（姓名、市区町村など）

2. ラベルの配置
   ✓ フィールドの上（推奨）: 読みやすく、モバイル対応しやすい
   △ フィールドの左: デスクトップでは整列されるが、モバイルで崩れやすい
   ✗ フィールドの中（プレースホルダーのみ）: 入力後に消える

3. 必須/任意の表示
   ✓ 必須フィールドが多い場合: 任意フィールドに「(任意)」と表示
   ✓ 任意フィールドが多い場合: 必須フィールドに「*」をつける
   ✗ 「*」だけで意味が不明

4. グルーピング
   ✓ 関連フィールドを fieldset + legend でグループ化
   ✓ 視覚的にも余白やボーダーで分離
   ✓ セクション見出しを付ける

5. モバイル対応
   ✓ inputMode を適切に設定（numeric, tel, email, url）
   ✓ autoComplete を設定（ブラウザの自動入力サポート）
   ✓ タッチターゲットのサイズは最低44x44px
   ✓ ズーム防止: font-size を16px以上にする
```

```css
/* モバイルフレンドリーなフォームCSS */
.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
}

.form-group input,
.form-group select,
.form-group textarea {
  width: 100%;
  padding: 0.75rem 1rem;
  font-size: 1rem; /* 16px以上でiOSのズーム防止 */
  border: 1px solid #d1d5db;
  border-radius: 0.375rem;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.form-group input:focus,
.form-group select:focus,
.form-group textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-group input[aria-invalid="true"] {
  border-color: #ef4444;
}

.form-group input[aria-invalid="true"]:focus {
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
}

.error-message {
  margin-top: 0.25rem;
  font-size: 0.875rem;
  color: #ef4444;
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.hint-text {
  margin-top: 0.25rem;
  font-size: 0.75rem;
  color: #6b7280;
}

/* タッチターゲットの最小サイズ */
.checkbox-label,
.radio-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-height: 44px;
  cursor: pointer;
}

/* モバイル最適化 */
@media (max-width: 640px) {
  .form-group input,
  .form-group select,
  .form-group textarea {
    padding: 0.875rem 1rem;
  }
}
```

---

## 7. アクセシビリティ（a11y）

### 7.1 ARIA属性の正しい使い方

```typescript
// アクセシブルなフォームフィールドコンポーネント
interface FormFieldProps {
  name: string;
  label: string;
  type?: string;
  required?: boolean;
  hint?: string;
  error?: string;
  register: UseFormRegister<any>;
}

function FormField({
  name,
  label,
  type = 'text',
  required = false,
  hint,
  error,
  register,
}: FormFieldProps) {
  const fieldId = `field-${name}`;
  const errorId = `${fieldId}-error`;
  const hintId = `${fieldId}-hint`;

  // aria-describedby の値を構築
  const describedBy = [
    hint ? hintId : null,
    error ? errorId : null,
  ].filter(Boolean).join(' ') || undefined;

  return (
    <div className="form-field">
      {/* ラベル */}
      <label htmlFor={fieldId}>
        {label}
        {required && <span className="required-mark" aria-hidden="true"> *</span>}
        {required && <span className="sr-only">（必須）</span>}
      </label>

      {/* ヒントテキスト */}
      {hint && (
        <p id={hintId} className="hint-text">
          {hint}
        </p>
      )}

      {/* 入力フィールド */}
      <input
        id={fieldId}
        type={type}
        {...register(name)}
        aria-invalid={!!error}
        aria-required={required}
        aria-describedby={describedBy}
      />

      {/* エラーメッセージ */}
      {error && (
        <p id={errorId} className="error-message" role="alert" aria-live="assertive">
          {error}
        </p>
      )}
    </div>
  );
}
```

### 7.2 キーボードナビゲーション

```typescript
// キーボードナビゲーション対応
// 重要な原則:
// 1. Tab キーで全フィールドにアクセス可能
// 2. Enter キーでフォーム送信
// 3. Escape キーでモーダルフォームを閉じる
// 4. Space キーでチェックボックス/ラジオボタンを切り替え

function AccessibleForm() {
  const formRef = useRef<HTMLFormElement>(null);

  // カスタムキーボードイベント
  const handleKeyDown = (e: React.KeyboardEvent<HTMLFormElement>) => {
    // Ctrl+Enter で送信（テキストエリアがある場合に便利）
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      formRef.current?.requestSubmit();
    }

    // Escape でフォームリセット
    if (e.key === 'Escape') {
      const confirmed = window.confirm('入力内容をリセットしますか？');
      if (confirmed) {
        form.reset();
      }
    }
  };

  return (
    <form
      ref={formRef}
      onKeyDown={handleKeyDown}
      onSubmit={form.handleSubmit(onSubmit)}
    >
      {/* tabIndex の順序に注意 */}
      <input {...form.register('name')} tabIndex={1} />
      <input {...form.register('email')} tabIndex={2} />
      <textarea {...form.register('message')} tabIndex={3} />

      {/* スキップリンク: 長いフォームの場合 */}
      <a href="#form-actions" className="sr-only focus:not-sr-only">
        送信ボタンへスキップ
      </a>

      <div id="form-actions">
        <button type="submit" tabIndex={4}>送信</button>
        <button type="button" tabIndex={5} onClick={() => form.reset()}>
          リセット
        </button>
      </div>
    </form>
  );
}
```

### 7.3 スクリーンリーダー対応

```typescript
// スクリーンリーダー向けの最適化

// 1. ライブリージョン: エラーメッセージの動的通知
function LiveErrorSummary({ errors }: { errors: FieldErrors }) {
  const errorMessages = Object.entries(errors)
    .map(([field, error]) => `${field}: ${error?.message}`)
    .join('. ');

  return (
    <div
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
      className="sr-only"
    >
      {errorMessages && `フォームに${Object.keys(errors).length}件のエラーがあります。${errorMessages}`}
    </div>
  );
}

// 2. エラーサマリー: フォーム上部にエラー一覧を表示
function ErrorSummary({ errors }: { errors: FieldErrors }) {
  const errorList = Object.entries(errors);
  if (errorList.length === 0) return null;

  return (
    <div
      role="alert"
      aria-labelledby="error-summary-title"
      className="error-summary"
      tabIndex={-1}
      ref={(el) => el?.focus()} // エラー発生時にフォーカス
    >
      <h3 id="error-summary-title">
        {errorList.length}件の入力エラーがあります
      </h3>
      <ul>
        {errorList.map(([field, error]) => (
          <li key={field}>
            <a href={`#field-${field}`}>
              {error?.message as string}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

// 3. フォーム完了通知
function FormSuccessMessage({ show }: { show: boolean }) {
  if (!show) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="success-message"
      tabIndex={-1}
      ref={(el) => el?.focus()}
    >
      フォームが正常に送信されました
    </div>
  );
}
```

### 7.4 autoComplete 属性の完全ガイド

```html
<!-- autoComplete 属性一覧 -->
<!-- ブラウザの自動入力を正しく動作させるために重要 -->

<!-- 名前 -->
<input autoComplete="name" />           <!-- フルネーム -->
<input autoComplete="given-name" />     <!-- 名 -->
<input autoComplete="family-name" />    <!-- 姓 -->
<input autoComplete="honorific-prefix" /> <!-- 敬称 -->

<!-- 連絡先 -->
<input autoComplete="email" />
<input autoComplete="tel" />            <!-- 電話番号 -->
<input autoComplete="tel-national" />   <!-- 国内電話番号 -->

<!-- 住所 -->
<input autoComplete="postal-code" />    <!-- 郵便番号 -->
<input autoComplete="address-level1" /> <!-- 都道府県 -->
<input autoComplete="address-level2" /> <!-- 市区町村 -->
<input autoComplete="street-address" /> <!-- 番地 -->
<input autoComplete="country" />        <!-- 国 -->

<!-- アカウント -->
<input autoComplete="username" />
<input autoComplete="new-password" />   <!-- 新しいパスワード -->
<input autoComplete="current-password" /> <!-- 現在のパスワード -->

<!-- 支払い -->
<input autoComplete="cc-name" />        <!-- カード名義 -->
<input autoComplete="cc-number" />      <!-- カード番号 -->
<input autoComplete="cc-exp" />         <!-- 有効期限 -->
<input autoComplete="cc-csc" />         <!-- セキュリティコード -->

<!-- その他 -->
<input autoComplete="organization" />   <!-- 組織名 -->
<input autoComplete="organization-title" /> <!-- 役職 -->
<input autoComplete="bday" />           <!-- 生年月日 -->
<input autoComplete="sex" />            <!-- 性別 -->
<input autoComplete="url" />            <!-- URL -->

<!-- 自動入力を無効化 -->
<input autoComplete="off" />            <!-- 標準的な方法 -->
<!-- 注意: ブラウザによっては "off" が無視される場合がある -->
<!-- その場合は一意な値を使う: -->
<input autoComplete="nope" />
```

### 7.5 WCAG 2.1 準拠チェックリスト

```
フォームの WCAG 2.1 準拠チェックリスト:

レベル A（必須）:
  [x] 全フォームコントロールにラベルが紐付いている (1.3.1)
  [x] エラーが発生した場合テキストで説明される (3.3.1)
  [x] フォームコントロールの目的が特定できる (1.3.5)
  [x] キーボードだけで全操作が可能 (2.1.1)
  [x] フォーカスが見える (2.4.7)
  [x] コンテキストの変化は予測可能 (3.2.1, 3.2.2)

レベル AA（推奨）:
  [x] エラーの修正方法が提案される (3.3.3)
  [x] 法的・金銭的データは確認/取消可能 (3.3.4)
  [x] テキストのコントラスト比が4.5:1以上 (1.4.3)
  [x] ターゲットサイズが24x24px以上 (2.5.8)
  [x] フォーカス表示が十分に目立つ (2.4.11)

レベル AAA（理想）:
  [x] ヘルプが利用可能 (3.3.5)
  [x] ターゲットサイズが44x44px以上 (2.5.5)
  [x] テキストのコントラスト比が7:1以上 (1.4.6)
```

---

## 8. パフォーマンス最適化

### 8.1 再レンダリングの最小化

```typescript
// パフォーマンス最適化パターン集

// Pattern 1: useWatch でコンポーネントを分離
// → 監視対象のフィールドが変更された時だけ該当コンポーネントが再レンダリング
import { useWatch } from 'react-hook-form';

// 悪い例: 親コンポーネント全体が再レンダリング
function BadForm() {
  const { register, watch, handleSubmit } = useForm();
  const name = watch('name'); // 親全体が再レンダリング

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('name')} />
      <input {...register('email')} />    {/* name変更でこれも再レンダリング */}
      <input {...register('phone')} />    {/* name変更でこれも再レンダリング */}
      <input {...register('address')} />  {/* name変更でこれも再レンダリング */}
      <p>プレビュー: {name}</p>
    </form>
  );
}

// 良い例: 子コンポーネントだけが再レンダリング
function GoodForm() {
  const { register, control, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input {...register('name')} />
      <input {...register('email')} />    {/* name変更で再レンダリングされない */}
      <input {...register('phone')} />    {/* name変更で再レンダリングされない */}
      <input {...register('address')} />  {/* name変更で再レンダリングされない */}
      <NamePreview control={control} />   {/* これだけが再レンダリング */}
    </form>
  );
}

function NamePreview({ control }: { control: Control }) {
  const name = useWatch({ control, name: 'name' });
  return <p>プレビュー: {name}</p>;
}

// Pattern 2: React.memo でフィールドコンポーネントをメモ化
const MemoizedField = React.memo(function MemoizedField({
  name,
  register,
  error,
}: {
  name: string;
  register: UseFormRegister<any>;
  error?: FieldError;
}) {
  console.log(`${name} rendered`); // デバッグ用

  return (
    <div>
      <label htmlFor={name}>{name}</label>
      <input id={name} {...register(name)} />
      {error && <p className="error-message">{error.message}</p>}
    </div>
  );
});

// Pattern 3: useFormState で必要な状態だけを購読
import { useFormState } from 'react-hook-form';

function SubmitButtonOptimized({ control }: { control: Control }) {
  // isSubmitting が変わった時だけ再レンダリング
  const { isSubmitting, isDirty } = useFormState({
    control,
    name: ['isSubmitting', 'isDirty'], // 購読する状態を限定
  });

  return (
    <button type="submit" disabled={isSubmitting || !isDirty}>
      {isSubmitting ? '送信中...' : '送信'}
    </button>
  );
}
```

### 8.2 大量フィールドの最適化

```typescript
// 大量のフィールドがある場合の最適化戦略

// Strategy 1: 仮想化（Virtualization）
// → 画面に表示されているフィールドだけをレンダリング
import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedFieldList() {
  const { register, control } = useForm();
  const parentRef = useRef<HTMLDivElement>(null);

  const fieldNames = Array.from({ length: 1000 }, (_, i) => `field_${i}`);

  const virtualizer = useVirtualizer({
    count: fieldNames.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 60, // 各フィールドの推定高さ
    overscan: 5, // 画面外に5フィールド分を先読み
  });

  return (
    <div ref={parentRef} style={{ height: '600px', overflow: 'auto' }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px`, position: 'relative' }}>
        {virtualizer.getVirtualItems().map((virtualItem) => {
          const fieldName = fieldNames[virtualItem.index];
          return (
            <div
              key={virtualItem.key}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: `${virtualItem.size}px`,
                transform: `translateY(${virtualItem.start}px)`,
              }}
            >
              <input {...register(fieldName)} placeholder={fieldName} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Strategy 2: セクション分割と遅延ロード
function SectionedForm() {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['basic'])
  );

  const toggleSection = (id: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <form>
      {/* 各セクションは折りたたみ可能 */}
      <details open={expandedSections.has('basic')}>
        <summary onClick={() => toggleSection('basic')}>基本情報</summary>
        <BasicInfoFields />
      </details>

      <details open={expandedSections.has('contact')}>
        <summary onClick={() => toggleSection('contact')}>連絡先</summary>
        {/* 展開時のみレンダリング */}
        {expandedSections.has('contact') && <ContactFields />}
      </details>

      <details open={expandedSections.has('preferences')}>
        <summary onClick={() => toggleSection('preferences')}>設定</summary>
        {expandedSections.has('preferences') && <PreferenceFields />}
      </details>
    </form>
  );
}

// Strategy 3: デバウンスバリデーション
function DebouncedValidation() {
  const { register, trigger } = useForm({
    mode: 'onChange',
  });

  const debouncedValidate = useMemo(
    () =>
      debounce((fieldName: string) => {
        trigger(fieldName);
      }, 300),
    [trigger]
  );

  return (
    <input
      {...register('search', {
        onChange: (e) => {
          debouncedValidate('search');
        },
      })}
    />
  );
}
```

### 8.3 バンドルサイズの最適化

```typescript
// バンドルサイズ最適化

// 1. 動的インポートでフォームを遅延ロード
const EditProfileForm = lazy(() => import('./EditProfileForm'));

function ProfilePage() {
  const [isEditing, setIsEditing] = useState(false);

  return (
    <div>
      {isEditing ? (
        <Suspense fallback={<FormSkeleton />}>
          <EditProfileForm onCancel={() => setIsEditing(false)} />
        </Suspense>
      ) : (
        <ProfileDisplay onEdit={() => setIsEditing(true)} />
      )}
    </div>
  );
}

// 2. Zodスキーマの分割
// 大きなスキーマを分割してツリーシェイキングを効かせる
// schema/user.ts
export const userBasicSchema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

// schema/user-extended.ts
// 必要な時だけインポート
export const userExtendedSchema = userBasicSchema.extend({
  address: addressSchema,
  preferences: preferencesSchema,
});

// 3. resolver の動的インポート
async function loadResolver() {
  const { zodResolver } = await import('@hookform/resolvers/zod');
  return zodResolver;
}
```

---

## 9. テスト戦略

### 9.1 React Testing Library でのフォームテスト

```typescript
// フォームのユニットテスト
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CreateUserForm } from './CreateUserForm';

describe('CreateUserForm', () => {
  const user = userEvent.setup();

  it('正常にフォームを送信できる', async () => {
    const onSubmit = vi.fn();
    render(<CreateUserForm onSubmit={onSubmit} />);

    // フィールドに値を入力
    await user.type(screen.getByLabelText('名前 *'), '山田太郎');
    await user.type(screen.getByLabelText('メールアドレス *'), 'yamada@example.com');
    await user.selectOptions(screen.getByLabelText('ロール *'), 'admin');
    await user.click(screen.getByLabelText('利用規約に同意します *'));

    // 送信
    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    // onSubmit が正しい値で呼ばれたことを確認
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith({
        name: '山田太郎',
        email: 'yamada@example.com',
        role: 'admin',
        agreed: true,
      });
    });
  });

  it('必須フィールドが空の場合エラーが表示される', async () => {
    render(<CreateUserForm />);

    // 何も入力せずに送信
    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    // エラーメッセージが表示されることを確認
    await waitFor(() => {
      expect(screen.getByText('名前は必須です')).toBeInTheDocument();
      expect(screen.getByText('有効なメールアドレスを入力してください')).toBeInTheDocument();
    });
  });

  it('メールアドレスが不正な場合エラーが表示される', async () => {
    render(<CreateUserForm />);

    await user.type(screen.getByLabelText('メールアドレス *'), 'invalid-email');
    await user.tab(); // フォーカスを外す（onBlur バリデーション）

    await waitFor(() => {
      expect(screen.getByText('有効なメールアドレスを入力してください')).toBeInTheDocument();
    });
  });

  it('送信中はボタンが無効になる', async () => {
    const onSubmit = vi.fn(() => new Promise((resolve) => setTimeout(resolve, 1000)));
    render(<CreateUserForm onSubmit={onSubmit} />);

    // フォームを埋める
    await user.type(screen.getByLabelText('名前 *'), '山田太郎');
    await user.type(screen.getByLabelText('メールアドレス *'), 'yamada@example.com');
    await user.click(screen.getByLabelText('利用規約に同意します *'));

    // 送信
    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    // ボタンが無効になっている
    expect(screen.getByRole('button', { name: /作成中/ })).toBeDisabled();
  });

  it('エラー修正後にエラーメッセージが消える', async () => {
    render(<CreateUserForm />);

    // 不正なメールを入力して送信
    await user.type(screen.getByLabelText('メールアドレス *'), 'invalid');
    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    await waitFor(() => {
      expect(screen.getByText('有効なメールアドレスを入力してください')).toBeInTheDocument();
    });

    // 正しいメールに修正
    const emailInput = screen.getByLabelText('メールアドレス *');
    await user.clear(emailInput);
    await user.type(emailInput, 'valid@example.com');

    // エラーメッセージが消える（reValidateMode: 'onChange'）
    await waitFor(() => {
      expect(screen.queryByText('有効なメールアドレスを入力してください')).not.toBeInTheDocument();
    });
  });
});
```

### 9.2 アクセシビリティテスト

```typescript
// axe-core を使ったアクセシビリティテスト
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

describe('CreateUserForm Accessibility', () => {
  it('アクセシビリティ違反がない', async () => {
    const { container } = render(<CreateUserForm />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('エラー状態でもアクセシビリティ違反がない', async () => {
    const { container } = render(<CreateUserForm />);
    const user = userEvent.setup();

    // エラーを発生させる
    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    await waitFor(async () => {
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  it('全フィールドにラベルが紐付いている', () => {
    render(<CreateUserForm />);

    // 全inputにラベルがあることを確認
    const inputs = screen.getAllByRole('textbox');
    inputs.forEach((input) => {
      expect(input).toHaveAccessibleName();
    });
  });

  it('キーボードでフォーカス移動できる', async () => {
    render(<CreateUserForm />);
    const user = userEvent.setup();

    // Tab キーでフォーカス移動
    await user.tab();
    expect(screen.getByLabelText('名前 *')).toHaveFocus();

    await user.tab();
    expect(screen.getByLabelText('メールアドレス *')).toHaveFocus();

    await user.tab();
    expect(screen.getByLabelText('年齢')).toHaveFocus();
  });

  it('エラーメッセージが aria-describedby で紐付いている', async () => {
    render(<CreateUserForm />);
    const user = userEvent.setup();

    await user.click(screen.getByRole('button', { name: /ユーザーを作成/ }));

    await waitFor(() => {
      const nameInput = screen.getByLabelText('名前 *');
      const errorId = nameInput.getAttribute('aria-describedby');
      expect(errorId).toBeTruthy();

      const errorElement = document.getElementById(errorId!);
      expect(errorElement).toHaveTextContent('名前は必須です');
    });
  });
});
```

### 9.3 E2E テスト（Playwright）

```typescript
// Playwright でのフォーム E2E テスト
import { test, expect } from '@playwright/test';

test.describe('ユーザー作成フォーム', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/users/new');
  });

  test('正常なフォーム送信', async ({ page }) => {
    // フィールドに入力
    await page.getByLabel('名前').fill('山田太郎');
    await page.getByLabel('メールアドレス').fill('yamada@example.com');
    await page.getByLabel('ロール').selectOption('admin');
    await page.getByLabel('利用規約に同意します').check();

    // API レスポンスを待機
    const responsePromise = page.waitForResponse(
      (response) => response.url().includes('/api/users') && response.status() === 201
    );

    // 送信
    await page.getByRole('button', { name: 'ユーザーを作成' }).click();

    // API レスポンスを確認
    const response = await responsePromise;
    expect(response.status()).toBe(201);

    // 成功メッセージの確認
    await expect(page.getByText('ユーザーを作成しました')).toBeVisible();

    // フォームがリセットされていることを確認
    await expect(page.getByLabel('名前')).toHaveValue('');
  });

  test('バリデーションエラーの表示と修正', async ({ page }) => {
    // 空のまま送信
    await page.getByRole('button', { name: 'ユーザーを作成' }).click();

    // エラーメッセージの確認
    await expect(page.getByText('名前は必須です')).toBeVisible();
    await expect(page.getByText('有効なメールアドレスを入力してください')).toBeVisible();

    // 最初のエラーフィールドにフォーカスが移動している
    await expect(page.getByLabel('名前')).toBeFocused();

    // エラーを修正
    await page.getByLabel('名前').fill('山田太郎');

    // エラーメッセージが消える
    await expect(page.getByText('名前は必須です')).not.toBeVisible();
  });

  test('重複メールアドレスのサーバーエラー', async ({ page }) => {
    // サーバーエラーをモック
    await page.route('**/api/users', (route) => {
      route.fulfill({
        status: 409,
        contentType: 'application/json',
        body: JSON.stringify({
          errors: {
            email: ['このメールアドレスは既に登録されています'],
          },
        }),
      });
    });

    // フォームを入力して送信
    await page.getByLabel('名前').fill('山田太郎');
    await page.getByLabel('メールアドレス').fill('existing@example.com');
    await page.getByLabel('利用規約に同意します').check();
    await page.getByRole('button', { name: 'ユーザーを作成' }).click();

    // サーバーエラーメッセージの表示
    await expect(page.getByText('このメールアドレスは既に登録されています')).toBeVisible();
  });

  test('モバイルでのフォーム操作', async ({ page }) => {
    // モバイルビューポートに変更
    await page.setViewportSize({ width: 375, height: 812 });

    // タッチ操作でフォームを入力
    await page.getByLabel('名前').tap();
    await page.getByLabel('名前').fill('山田太郎');

    // 画面下部の送信ボタンまでスクロール
    await page.getByRole('button', { name: 'ユーザーを作成' }).scrollIntoViewIfNeeded();
    await page.getByRole('button', { name: 'ユーザーを作成' }).tap();
  });
});
```

---

## 10. アンチパターンとトラブルシューティング

### 10.1 よくあるアンチパターン

```typescript
// アンチパターン 1: register と state の二重管理
// Bad: React Hook Form と useState で二重管理
function BadDoubleState() {
  const { register } = useForm();
  const [name, setName] = useState(''); // 不要！

  return (
    <input
      {...register('name')}
      value={name}                       // register の値を上書きしてしまう
      onChange={(e) => setName(e.target.value)} // register の onChange を上書き
    />
  );
}

// Good: React Hook Form に任せる
function GoodSingleSource() {
  const { register, watch } = useForm();
  const name = watch('name'); // 必要な時だけ watch で値を取得

  return <input {...register('name')} />;
}


// アンチパターン 2: useFieldArray で index を key に使用
// Bad: index を key に使うと削除・並び替えでバグ
function BadFieldArray() {
  const { fields } = useFieldArray({ control, name: 'items' });

  return fields.map((field, index) => (
    <div key={index}>  {/* 削除時にフィールドの値がずれる */}
      <input {...register(`items.${index}.name`)} />
    </div>
  ));
}

// Good: field.id を key に使う
function GoodFieldArray() {
  const { fields } = useFieldArray({ control, name: 'items' });

  return fields.map((field, index) => (
    <div key={field.id}>  {/* 安定したキー */}
      <input {...register(`items.${index}.name`)} />
    </div>
  ));
}


// アンチパターン 3: defaultValues の参照が変わる
// Bad: レンダリングごとに新しいオブジェクトが作られる
function BadDefaultValues() {
  const form = useForm({
    defaultValues: {  // レンダリングごとに新しい参照
      items: [],
    },
  });
}

// Good: コンポーネント外に定義するか useMemo を使う
const defaultValues = { items: [] };

function GoodDefaultValues() {
  const form = useForm({ defaultValues });
}


// アンチパターン 4: バリデーションモードの選択ミス
// Bad: mode: 'onChange' + 重い非同期バリデーション
function BadValidationMode() {
  const form = useForm({
    mode: 'onChange', // キー入力ごとにAPIコール！
  });

  return (
    <input
      {...form.register('username', {
        validate: async (value) => {
          const exists = await checkUsername(value); // 毎キー入力で実行
          return !exists || 'このユーザー名は使用されています';
        },
      })}
    />
  );
}

// Good: mode: 'onBlur' + デバウンス
function GoodValidationMode() {
  const form = useForm({
    mode: 'onBlur', // フォーカスを外した時だけバリデーション
  });

  return (
    <input
      {...form.register('username', {
        validate: async (value) => {
          const exists = await checkUsername(value);
          return !exists || 'このユーザー名は使用されています';
        },
      })}
    />
  );
}


// アンチパターン 5: エラー表示のタイミングミス
// Bad: フォーム初期表示でエラーを出す
function BadInitialErrors() {
  const { register, formState: { errors } } = useForm({
    mode: 'all',       // 全モードでバリデーション
    defaultValues: {
      name: '',        // 初期値が空 → 即座にエラー表示
    },
  });

  return (
    <div>
      <input {...register('name', { required: '名前は必須です' })} />
      {errors.name && <p>{errors.name.message}</p>} {/* 初回表示でエラー！ */}
    </div>
  );
}

// Good: touchedFields を考慮
function GoodInitialDisplay() {
  const { register, formState: { errors, touchedFields } } = useForm({
    mode: 'onTouched',
  });

  return (
    <div>
      <input {...register('name', { required: '名前は必須です' })} />
      {touchedFields.name && errors.name && (
        <p>{errors.name.message}</p>
      )}
    </div>
  );
}
```

### 10.2 トラブルシューティングガイド

```
トラブルシューティング:

Q: register した input の値が取得できない
A: 考えられる原因:
   1. defaultValues を設定していない
      → useForm({ defaultValues: { fieldName: '' } })
   2. Controller を使うべきカスタムコンポーネントに register を使用
      → Controller に切り替える
   3. コンポーネントのアンマウント/リマウント
      → shouldUnregister: false を設定

Q: バリデーションが実行されない
A: 考えられる原因:
   1. resolver のインポートが間違っている
      → import { zodResolver } from '@hookform/resolvers/zod'
   2. スキーマとフィールド名が一致していない
      → register('name') なら schema に name プロパティが必要
   3. mode の設定
      → mode: 'onSubmit'（デフォルト）は送信時のみ実行

Q: TypeScript の型エラー
A: 考えられる原因:
   1. z.infer の型とフォームの型が不一致
      → type FormData = z.infer<typeof schema> を使用
   2. register の名前がスキーマに存在しない
      → スキーマのプロパティ名と一致させる
   3. optional フィールドの型
      → z.coerce.number().optional() で undefined を許容

Q: useFieldArray で削除後に値がおかしくなる
A: 考えられる原因:
   1. key に index を使用している
      → key={field.id} に変更
   2. defaultValues にフィールドが含まれていない
      → defaultValues: { items: [{ name: '' }] }

Q: フォームリセット後も値が残る
A: 考えられる原因:
   1. reset() にオプションを渡していない
      → reset({ name: '', email: '' }) で明示的にリセット
   2. Controller のコンポーネントが内部状態を持っている
      → key を変更して強制リマウント

Q: Server Action で formData が空
A: 考えられる原因:
   1. input に name 属性がない
      → register('name') は自動で name を設定するが、
         handleSubmit を使う場合は FormData ではなく parsed data が渡される
   2. action と onSubmit を同時に使っている
      → どちらか一方に統一する

Q: 大量のフィールドでパフォーマンスが悪い
A: 対策:
   1. watch の使用を最小限にする → useWatch で分離
   2. mode: 'onChange' を避ける → mode: 'onBlur' に変更
   3. 仮想スクロールを導入する → @tanstack/react-virtual
   4. フォームをセクションに分割する
   5. React.memo でフィールドコンポーネントをメモ化
```

### 10.3 エラーハンドリングの完全パターン

```typescript
// エラーハンドリングの包括的実装
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

// カスタムエラークラス
class FormSubmitError extends Error {
  constructor(
    message: string,
    public fieldErrors?: Record<string, string[]>,
    public statusCode?: number,
  ) {
    super(message);
    this.name = 'FormSubmitError';
  }
}

// エラーハンドリング付きフォーム
function RobustForm() {
  const [globalError, setGlobalError] = useState<string | null>(null);

  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: { name: '', email: '' },
  });

  const onSubmit = async (data: UserFormData) => {
    setGlobalError(null);

    try {
      const response = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);

        switch (response.status) {
          case 400:
            // バリデーションエラー: フィールドごとにエラーを設定
            if (errorBody?.fieldErrors) {
              for (const [field, messages] of Object.entries(errorBody.fieldErrors)) {
                form.setError(field as keyof UserFormData, {
                  type: 'server',
                  message: (messages as string[])[0],
                });
              }
            }
            break;

          case 409:
            // 重複エラー
            form.setError('email', {
              type: 'server',
              message: 'このメールアドレスは既に使用されています',
            });
            break;

          case 422:
            // 処理不能エンティティ
            setGlobalError('入力データの処理に失敗しました。内容を確認してください。');
            break;

          case 429:
            // レート制限
            setGlobalError('リクエストが多すぎます。しばらく待ってから再度お試しください。');
            break;

          case 500:
            // サーバーエラー
            setGlobalError('サーバーエラーが発生しました。時間をおいて再度お試しください。');
            break;

          default:
            setGlobalError(`エラーが発生しました（${response.status}）`);
        }
        return;
      }

      // 成功
      form.reset();
      toast.success('ユーザーを作成しました');

    } catch (error) {
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        // ネットワークエラー
        setGlobalError('ネットワークエラー: インターネット接続を確認してください。');
      } else if (error instanceof DOMException && error.name === 'AbortError') {
        // リクエストタイムアウト
        setGlobalError('リクエストがタイムアウトしました。再度お試しください。');
      } else {
        // 予期しないエラー
        console.error('Unexpected error:', error);
        setGlobalError('予期しないエラーが発生しました。');
      }
    }
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {/* グローバルエラー表示 */}
      {globalError && (
        <div className="global-error" role="alert" aria-live="assertive">
          <p>{globalError}</p>
          <button
            type="button"
            onClick={() => setGlobalError(null)}
            aria-label="エラーメッセージを閉じる"
          >
            閉じる
          </button>
        </div>
      )}

      {/* エラーサマリー */}
      {Object.keys(form.formState.errors).length > 0 && (
        <ErrorSummary errors={form.formState.errors} />
      )}

      {/* フォームフィールド */}
      <div>
        <label htmlFor="name">名前 *</label>
        <input id="name" {...form.register('name')} />
        {form.formState.errors.name && (
          <p className="error-message" role="alert">
            {form.formState.errors.name.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="email">メールアドレス *</label>
        <input id="email" type="email" {...form.register('email')} />
        {form.formState.errors.email && (
          <p className="error-message" role="alert">
            {form.formState.errors.email.message}
          </p>
        )}
      </div>

      <button
        type="submit"
        disabled={form.formState.isSubmitting}
        aria-busy={form.formState.isSubmitting}
      >
        {form.formState.isSubmitting ? '送信中...' : 'ユーザーを作成'}
      </button>
    </form>
  );
}
```

---

## 11. 再利用可能なフォームコンポーネント設計

### 11.1 汎用フォームフィールドコンポーネント

```typescript
// 再利用可能なフォームフィールドコンポーネント
import { type FieldValues, type Path, type UseFormReturn } from 'react-hook-form';

interface FormInputProps<T extends FieldValues> {
  form: UseFormReturn<T>;
  name: Path<T>;
  label: string;
  type?: string;
  placeholder?: string;
  required?: boolean;
  hint?: string;
  autoComplete?: string;
  className?: string;
}

function FormInput<T extends FieldValues>({
  form,
  name,
  label,
  type = 'text',
  placeholder,
  required = false,
  hint,
  autoComplete,
  className,
}: FormInputProps<T>) {
  const fieldId = `field-${String(name)}`;
  const errorId = `${fieldId}-error`;
  const hintId = `${fieldId}-hint`;
  const error = form.formState.errors[name];

  const describedBy = [
    hint ? hintId : null,
    error ? errorId : null,
  ].filter(Boolean).join(' ') || undefined;

  return (
    <div className={cn('form-group', className)}>
      <label htmlFor={fieldId}>
        {label}
        {required && <span aria-hidden="true"> *</span>}
      </label>

      {hint && (
        <p id={hintId} className="hint-text">{hint}</p>
      )}

      <input
        id={fieldId}
        type={type}
        placeholder={placeholder}
        autoComplete={autoComplete}
        aria-invalid={!!error}
        aria-required={required}
        aria-describedby={describedBy}
        {...form.register(name)}
      />

      {error && (
        <p id={errorId} className="error-message" role="alert">
          {error.message as string}
        </p>
      )}
    </div>
  );
}

// 使用例
function UserForm() {
  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <FormInput
        form={form}
        name="name"
        label="名前"
        required
        autoComplete="name"
        placeholder="山田太郎"
      />
      <FormInput
        form={form}
        name="email"
        label="メールアドレス"
        type="email"
        required
        autoComplete="email"
        placeholder="user@example.com"
      />
      <FormInput
        form={form}
        name="age"
        label="年齢"
        type="number"
        hint="任意項目です"
      />
      <button type="submit">送信</button>
    </form>
  );
}
```

### 11.2 フォームラッパーコンポーネント

```typescript
// 汎用フォームラッパー
import { type FieldValues, type DefaultValues, FormProvider, useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { type ZodSchema } from 'zod';

interface FormWrapperProps<T extends FieldValues> {
  schema: ZodSchema<T>;
  defaultValues: DefaultValues<T>;
  onSubmit: (data: T) => Promise<void> | void;
  children: React.ReactNode;
  className?: string;
  mode?: 'onSubmit' | 'onBlur' | 'onChange' | 'onTouched' | 'all';
}

function FormWrapper<T extends FieldValues>({
  schema,
  defaultValues,
  onSubmit,
  children,
  className,
  mode = 'onBlur',
}: FormWrapperProps<T>) {
  const methods = useForm<T>({
    resolver: zodResolver(schema),
    defaultValues,
    mode,
    reValidateMode: 'onChange',
  });

  const [globalError, setGlobalError] = useState<string | null>(null);

  const handleSubmit = async (data: T) => {
    setGlobalError(null);
    try {
      await onSubmit(data);
    } catch (error) {
      if (error instanceof Error) {
        setGlobalError(error.message);
      } else {
        setGlobalError('予期しないエラーが発生しました');
      }
    }
  };

  return (
    <FormProvider {...methods}>
      <form
        onSubmit={methods.handleSubmit(handleSubmit)}
        className={className}
        noValidate
      >
        {globalError && (
          <div className="global-error" role="alert">
            {globalError}
          </div>
        )}
        {children}
      </form>
    </FormProvider>
  );
}

// 使用例
function CreateUserPage() {
  return (
    <FormWrapper
      schema={userSchema}
      defaultValues={{ name: '', email: '', role: 'user' }}
      onSubmit={async (data) => {
        await api.users.create(data);
        router.push('/users');
      }}
    >
      <FormInput name="name" label="名前" required />
      <FormInput name="email" label="メールアドレス" type="email" required />
      <SubmitButton label="ユーザーを作成" />
    </FormWrapper>
  );
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| React Hook Form | register + zodResolver で型安全かつ高パフォーマンスなフォームを実現 |
| 非制御コンポーネント | ネイティブHTML要素に使用、再レンダリングなしで最高パフォーマンス |
| 制御コンポーネント | Controller を使ってカスタムUI/サードパーティコンポーネントと統合 |
| useFieldArray | 動的フィールド配列を効率的に管理、key には必ず field.id を使用 |
| マルチステップ | FormProvider + trigger で部分バリデーション、状態を一元管理 |
| Server Actions | プログレッシブエンハンスメント対応、楽観的更新パターン |
| UX | 送信後リアルタイム検証、ダブルサブミット防止、未保存警告 |
| アクセシビリティ | ARIA属性、キーボードナビゲーション、スクリーンリーダー対応 |
| パフォーマンス | useWatch で分離、React.memo、仮想化、デバウンス |
| テスト | RTL でユーザー操作をテスト、axe で a11y テスト、Playwright で E2E |
| エラーハンドリング | クライアント/サーバーエラーの統合、グローバルエラー表示 |
| 再利用性 | ジェネリック型でフォームフィールドとラッパーを汎用化 |

---

## よくある質問（FAQ）

### Q1. Controlled Components と Uncontrolled Components はどう使い分けるべきですか？

**A:** 基本的には **Controlled Components を優先** する。理由は以下の通り:

- **即座のバリデーション**: 入力値の変化をリアルタイムで検証できる
- **条件付きUI**: 入力内容に応じて動的にフォームを変更できる
- **デバッグのしやすさ**: 状態がReactの管理下にあるため、開発ツールでの追跡が容易

ただし、以下の場合は **Uncontrolled Components の方が適している**:

- **大量のフォームフィールド**: 数百個のフィールドがあり、パフォーマンスが懸念される場合
- **レガシーコードとの統合**: 既存の非Reactコードとの互換性が必要な場合
- **ファイル入力**: `<input type="file">` は常にUncontrolledである必要がある

React Hook Form は内部的にUncontrolledな仕組みを使いつつ、Controlledライクなインターフェースを提供しているため、パフォーマンスとDXを両立できる。

### Q2. React Hook Form と Formik、どちらを選ぶべきですか？

**A:** 現在のプロジェクトでは **React Hook Form を推奨** する:

| 項目 | React Hook Form | Formik |
|------|----------------|---------|
| パフォーマンス | 優秀（Uncontrolled方式で再レンダリング最小） | 可（Controlled方式で再レンダリング多） |
| バンドルサイズ | 8.5KB（gzip） | 15KB（gzip） |
| TypeScript対応 | 完全対応、型推論が強力 | 対応しているが弱め |
| エコシステム | Zod/Yup統合が簡単 | Yup推奨 |
| 学習曲線 | やや急（useForm APIに慣れが必要） | 緩やか（Formikコンポーネントが直感的） |
| メンテナンス | 活発 | やや減速傾向 |

Formikが優れている点:

- **直感的なAPI**: `<Formik>`, `<Field>` コンポーネントでReactらしく書ける
- **ドキュメントが豊富**: 歴史が長く、学習リソースが充実

React Hook Formが優れている点:

- **パフォーマンス**: 大規模フォームでも高速
- **TypeScript統合**: Zodスキーマから型を自動推論
- **最新のReact哲学**: HooksベースでモダンなReact開発に適合

### Q3. マルチステップフォームはどのように設計すべきですか？

**A:** マルチステップフォームの設計では、以下の4つのアプローチがある:

**1. ステップごとに独立したフォーム（推奨）**

```typescript
// 各ステップで独自のuseFormを持つ
const Step1 = () => {
  const { register, handleSubmit } = useForm<Step1Data>();
  const onSubmit = (data) => saveToContext(data);
  // ...
};
```

利点: 各ステップが独立、バリデーションが分離、戻るボタンの実装が簡単
欠点: ステップ間のデータ共有に Context か状態管理ライブラリが必要

**2. 単一フォームで条件付き表示**

```typescript
const MultiStepForm = () => {
  const { register, handleSubmit } = useForm<AllStepsData>();
  const [currentStep, setCurrentStep] = useState(1);
  // 全フィールドを保持しつつ、表示だけ切り替え
};
```

利点: 実装がシンプル、全データが1つのフォームに統合
欠点: 大規模になると管理が複雑、バリデーションの制御が難しい

**3. ステップごとのスキーマ + マージ戦略**

```typescript
const step1Schema = z.object({ name: z.string() });
const step2Schema = z.object({ email: z.string().email() });
const finalSchema = step1Schema.merge(step2Schema);
```

利点: Zodの型推論を活用、各ステップで段階的にバリデーション
欠点: スキーママージの複雑さ

**推奨する設計パターン:**

- ステップが3〜5個程度: **ステップごとに独立したフォーム + Contextで状態共有**
- ステップが2個: **単一フォームで条件付き表示**
- ステップが6個以上: **フォームビルダーライブラリ**（react-multi-step-form）の導入を検討

**UXの考慮事項:**

- プログレスバーを常に表示
- 前のステップに戻れること
- 各ステップ完了時にlocalStorageに保存（離脱対策）
- 最終確認画面で全入力内容を表示

---

## 次に読むべきガイド
→ [[01-validation-patterns.md]] -- バリデーションパターンの詳細

---

## 参考文献
1. React Hook Form. "Documentation." react-hook-form.com, 2024.
2. shadcn/ui. "Form." ui.shadcn.com, 2024.
3. web.dev. "Form Best Practices." web.dev, 2024.
4. W3C. "WCAG 2.1 - Web Content Accessibility Guidelines." w3.org, 2018.
5. MDN Web Docs. "Web forms - Working with user data." developer.mozilla.org, 2024.
6. Zod. "TypeScript-first schema validation." zod.dev, 2024.
7. React. "React 19 - useActionState, useOptimistic." react.dev, 2024.
8. Testing Library. "React Testing Library." testing-library.com, 2024.
9. Playwright. "End-to-end testing." playwright.dev, 2024.
10. Deque Systems. "axe-core - Accessibility Testing." deque.com, 2024.
