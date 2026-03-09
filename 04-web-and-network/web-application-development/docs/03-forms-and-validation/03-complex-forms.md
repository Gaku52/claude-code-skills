# 複雑なフォーム

> 複雑なフォームは実務で避けて通れない課題。マルチステップフォーム、動的フィールド、条件分岐、配列フィールド、ネストしたフォームまで、あらゆる複雑なフォーム要件に対応するパターンを習得する。React Hook Form + Zod を中心に、スケーラブルな複雑フォームの設計から実装、テスト、パフォーマンス最適化までを網羅的に解説する。

## この章で学ぶこと

- [ ] マルチステップフォームの設計と実装を理解する
- [ ] useFieldArray による動的フィールドの管理を把握する
- [ ] 条件分岐フォームのバリデーション設計を学ぶ
- [ ] フォームの自動保存とドラフト管理を実装できる
- [ ] ページ離脱防止と未保存データの保護を実現する
- [ ] ネストしたフォーム構造の設計パターンを習得する
- [ ] 複雑フォームのパフォーマンス最適化手法を理解する
- [ ] アクセシビリティ対応の複雑フォームを構築できる
- [ ] フォームのテスト戦略と実装手法を身につける

---

## 前提知識

この章を最大限活用するために、以下の知識を事前に習得しておくことを推奨する:

- **ファイルアップロード**: `./02-file-upload.md` で学ぶ、ファイルのバリデーション、プレビュー表示、アップロード処理の実装パターンを理解していること
- **状態管理**: `../01-state-management/00-state-management-overview.md` で学ぶ、React Context、Zustand、またはReduxによるグローバル状態管理の基礎を把握していること
- **フォームバリデーション**: `./01-validation-patterns.md` で学ぶ、Zodスキーマ設計、条件付きバリデーション、エラーハンドリングのパターンを理解していること

---

## 1. マルチステップフォーム

### 1.1 設計原則

マルチステップフォーム（ウィザードフォーム）は、ユーザーが一度に処理する情報量を制限し、認知負荷を軽減するためのUIパターンである。以下の原則に従って設計する。

**ステップ分割の基準:**
- 論理的にグループ化できる情報を1ステップにまとめる
- 1ステップあたりのフィールド数は3〜7個が目安
- ユーザーが離脱しやすいステップ（支払い情報など）は後半に配置する
- 必須情報を前半に、オプション情報を後半に配置する

**UXの考慮事項:**
- 進捗インジケーターを常に表示する
- 前のステップに戻れることを保証する
- 各ステップ完了時にデータを保存する（離脱対策）
- 最終確認画面で入力内容を一覧表示する

### 1.2 基本実装: ステップ別スキーマ

```typescript
import { z } from 'zod';
import { useForm, UseFormReturn } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useState, useCallback } from 'react';

// ステップ1: アカウント情報
const step1Schema = z.object({
  name: z.string()
    .min(1, '名前は必須です')
    .max(50, '名前は50文字以内で入力してください'),
  email: z.string()
    .email('有効なメールアドレスを入力してください'),
  password: z.string()
    .min(8, 'パスワードは8文字以上で入力してください')
    .regex(
      /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
      'パスワードは大文字、小文字、数字を含む必要があります'
    ),
  confirmPassword: z.string(),
}).refine(data => data.password === data.confirmPassword, {
  message: 'パスワードが一致しません',
  path: ['confirmPassword'],
});

// ステップ2: プロフィール情報
const step2Schema = z.object({
  company: z.string().min(1, '会社名は必須です'),
  role: z.enum(['developer', 'designer', 'manager', 'other'], {
    errorMap: () => ({ message: '役割を選択してください' }),
  }),
  experience: z.coerce
    .number()
    .min(0, '経験年数は0以上で入力してください')
    .max(50, '経験年数は50以下で入力してください'),
  bio: z.string()
    .max(500, '自己紹介は500文字以内で入力してください')
    .optional(),
});

// ステップ3: プラン選択
const step3Schema = z.object({
  plan: z.enum(['free', 'pro', 'enterprise'], {
    errorMap: () => ({ message: 'プランを選択してください' }),
  }),
  billingCycle: z.enum(['monthly', 'yearly']).optional(),
  agreed: z.literal(true, {
    errorMap: () => ({ message: '利用規約に同意する必要があります' }),
  }),
});

// 全体スキーマ（最終バリデーション用）
const fullSchema = step1Schema
  .merge(step2Schema)
  .merge(step3Schema);

type FormData = z.infer<typeof fullSchema>;

// ステップ設定の型定義
interface StepConfig {
  title: string;
  description: string;
  schema: z.ZodType;
  fields: (keyof FormData)[];
}

const STEPS: StepConfig[] = [
  {
    title: 'アカウント情報',
    description: 'ログインに使用する情報を入力してください',
    schema: step1Schema,
    fields: ['name', 'email', 'password', 'confirmPassword'],
  },
  {
    title: 'プロフィール',
    description: 'あなたの情報を教えてください',
    schema: step2Schema,
    fields: ['company', 'role', 'experience', 'bio'],
  },
  {
    title: 'プラン選択',
    description: 'ご利用プランを選択してください',
    schema: step3Schema,
    fields: ['plan', 'billingCycle', 'agreed'],
  },
];
```

### 1.3 ステップ管理フック

```typescript
// カスタムフック: マルチステップフォームのロジック管理
function useMultiStepForm<T extends Record<string, any>>(
  steps: StepConfig[],
  defaultValues: Partial<T>
) {
  const [currentStep, setCurrentStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());
  const [stepData, setStepData] = useState<Partial<T>[]>(
    steps.map(() => ({}))
  );

  const form = useForm<T>({
    resolver: zodResolver(steps[currentStep].schema),
    mode: 'onTouched',
    defaultValues: defaultValues as any,
  });

  const isFirstStep = currentStep === 0;
  const isLastStep = currentStep === steps.length - 1;
  const progress = ((currentStep + 1) / steps.length) * 100;

  const goToNext = useCallback(async () => {
    // 現在のステップのバリデーション
    const fieldsToValidate = steps[currentStep].fields;
    const isValid = await form.trigger(fieldsToValidate as any);

    if (!isValid) return false;

    // ステップデータの保存
    const currentValues = form.getValues();
    setStepData(prev => {
      const updated = [...prev];
      updated[currentStep] = currentValues;
      return updated;
    });

    // ステップ完了マーク
    setCompletedSteps(prev => new Set(prev).add(currentStep));

    if (!isLastStep) {
      setCurrentStep(s => s + 1);
    }

    return true;
  }, [currentStep, form, steps, isLastStep]);

  const goToPrevious = useCallback(() => {
    if (!isFirstStep) {
      setCurrentStep(s => s - 1);
    }
  }, [isFirstStep]);

  const goToStep = useCallback((step: number) => {
    // 完了済みステップか現在のステップの次まで遷移可能
    if (step <= currentStep || completedSteps.has(step - 1)) {
      setCurrentStep(step);
    }
  }, [currentStep, completedSteps]);

  const getMergedData = useCallback((): Partial<T> => {
    return stepData.reduce((acc, data) => ({ ...acc, ...data }), {} as Partial<T>);
  }, [stepData]);

  return {
    form,
    currentStep,
    isFirstStep,
    isLastStep,
    progress,
    completedSteps,
    goToNext,
    goToPrevious,
    goToStep,
    getMergedData,
    totalSteps: steps.length,
  };
}
```

### 1.4 完全なマルチステップフォームコンポーネント

```tsx
function MultiStepForm() {
  const {
    form,
    currentStep,
    isFirstStep,
    isLastStep,
    progress,
    completedSteps,
    goToNext,
    goToPrevious,
    goToStep,
    getMergedData,
    totalSteps,
  } = useMultiStepForm<FormData>(STEPS, {
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    company: '',
    role: undefined,
    experience: 0,
    bio: '',
    plan: undefined,
    billingCycle: 'monthly',
    agreed: false as any,
  });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleNext = async () => {
    const success = await goToNext();
    if (success && !isLastStep) {
      // ドラフト保存
      const data = getMergedData();
      await saveDraft(data);
    }
  };

  const onSubmit = async (data: FormData) => {
    setIsSubmitting(true);
    setSubmitError(null);

    try {
      // 全ステップのデータをマージ
      const mergedData = { ...getMergedData(), ...data };

      // 最終バリデーション
      const result = fullSchema.safeParse(mergedData);
      if (!result.success) {
        const firstError = result.error.errors[0];
        setSubmitError(`入力エラー: ${firstError.message}`);
        return;
      }

      await api.register(result.data);
      router.push('/registration/complete');
    } catch (error) {
      setSubmitError(
        error instanceof Error
          ? error.message
          : '登録に失敗しました。もう一度お試しください。'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      {/* プログレスバー */}
      <div className="mb-8">
        <div className="flex justify-between mb-2">
          {STEPS.map((step, i) => (
            <button
              key={i}
              type="button"
              onClick={() => goToStep(i)}
              className={`flex items-center gap-2 text-sm font-medium
                ${i === currentStep ? 'text-blue-600' : ''}
                ${completedSteps.has(i) ? 'text-green-600 cursor-pointer' : 'text-gray-400'}
              `}
              disabled={!completedSteps.has(i) && i !== currentStep}
            >
              <span className={`w-8 h-8 rounded-full flex items-center justify-center text-xs
                ${i === currentStep ? 'bg-blue-600 text-white' : ''}
                ${completedSteps.has(i) ? 'bg-green-600 text-white' : 'bg-gray-200'}
              `}>
                {completedSteps.has(i) ? '✓' : i + 1}
              </span>
              <span className="hidden sm:inline">{step.title}</span>
            </button>
          ))}
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* ステップヘッダー */}
      <div className="mb-6">
        <h2 className="text-xl font-bold">{STEPS[currentStep].title}</h2>
        <p className="text-gray-500 mt-1">{STEPS[currentStep].description}</p>
      </div>

      {/* フォーム本体 */}
      <form onSubmit={form.handleSubmit(onSubmit)}>
        {currentStep === 0 && <Step1Fields form={form} />}
        {currentStep === 1 && <Step2Fields form={form} />}
        {currentStep === 2 && <Step3Fields form={form} />}

        {/* エラーメッセージ */}
        {submitError && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600 text-sm">{submitError}</p>
          </div>
        )}

        {/* ナビゲーション */}
        <div className="flex justify-between mt-8">
          <button
            type="button"
            onClick={goToPrevious}
            disabled={isFirstStep}
            className={`px-6 py-2 rounded-lg border
              ${isFirstStep ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-50'}
            `}
          >
            戻る
          </button>

          {isLastStep ? (
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg
                hover:bg-blue-700 disabled:opacity-50"
            >
              {isSubmitting ? '送信中...' : '登録する'}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleNext}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              次へ
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
```

### 1.5 ステップコンポーネントの実装

```tsx
// ステップ1: アカウント情報
function Step1Fields({ form }: { form: UseFormReturn<FormData> }) {
  const { register, formState: { errors } } = form;

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="name" className="block text-sm font-medium mb-1">
          名前 <span className="text-red-500">*</span>
        </label>
        <input
          id="name"
          {...register('name')}
          className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          placeholder="山田 太郎"
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
        />
        {errors.name && (
          <p id="name-error" className="mt-1 text-sm text-red-500" role="alert">
            {errors.name.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="email" className="block text-sm font-medium mb-1">
          メールアドレス <span className="text-red-500">*</span>
        </label>
        <input
          id="email"
          type="email"
          {...register('email')}
          className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          placeholder="taro@example.com"
          aria-invalid={!!errors.email}
          aria-describedby={errors.email ? 'email-error' : undefined}
        />
        {errors.email && (
          <p id="email-error" className="mt-1 text-sm text-red-500" role="alert">
            {errors.email.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="password" className="block text-sm font-medium mb-1">
          パスワード <span className="text-red-500">*</span>
        </label>
        <input
          id="password"
          type="password"
          {...register('password')}
          className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          placeholder="8文字以上"
          aria-invalid={!!errors.password}
          aria-describedby={errors.password ? 'password-error' : 'password-hint'}
        />
        <p id="password-hint" className="mt-1 text-xs text-gray-400">
          大文字、小文字、数字を含む8文字以上
        </p>
        {errors.password && (
          <p id="password-error" className="mt-1 text-sm text-red-500" role="alert">
            {errors.password.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="confirmPassword" className="block text-sm font-medium mb-1">
          パスワード確認 <span className="text-red-500">*</span>
        </label>
        <input
          id="confirmPassword"
          type="password"
          {...register('confirmPassword')}
          className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500"
          placeholder="パスワードを再入力"
          aria-invalid={!!errors.confirmPassword}
        />
        {errors.confirmPassword && (
          <p className="mt-1 text-sm text-red-500" role="alert">
            {errors.confirmPassword.message}
          </p>
        )}
      </div>
    </div>
  );
}

// ステップ2: プロフィール情報
function Step2Fields({ form }: { form: UseFormReturn<FormData> }) {
  const { register, formState: { errors } } = form;

  return (
    <div className="space-y-4">
      <div>
        <label htmlFor="company" className="block text-sm font-medium mb-1">
          会社名 <span className="text-red-500">*</span>
        </label>
        <input
          id="company"
          {...register('company')}
          className="w-full px-3 py-2 border rounded-lg"
          placeholder="株式会社サンプル"
        />
        {errors.company && (
          <p className="mt-1 text-sm text-red-500">{errors.company.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="role" className="block text-sm font-medium mb-1">
          役割 <span className="text-red-500">*</span>
        </label>
        <select
          id="role"
          {...register('role')}
          className="w-full px-3 py-2 border rounded-lg"
        >
          <option value="">選択してください</option>
          <option value="developer">開発者</option>
          <option value="designer">デザイナー</option>
          <option value="manager">マネージャー</option>
          <option value="other">その他</option>
        </select>
        {errors.role && (
          <p className="mt-1 text-sm text-red-500">{errors.role.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="experience" className="block text-sm font-medium mb-1">
          経験年数
        </label>
        <input
          id="experience"
          type="number"
          {...register('experience')}
          className="w-full px-3 py-2 border rounded-lg"
          min={0}
          max={50}
        />
        {errors.experience && (
          <p className="mt-1 text-sm text-red-500">{errors.experience.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="bio" className="block text-sm font-medium mb-1">
          自己紹介
        </label>
        <textarea
          id="bio"
          {...register('bio')}
          className="w-full px-3 py-2 border rounded-lg"
          rows={4}
          placeholder="自由に記入してください（500文字以内）"
        />
        {errors.bio && (
          <p className="mt-1 text-sm text-red-500">{errors.bio.message}</p>
        )}
      </div>
    </div>
  );
}
```

### 1.6 マルチステップフォームのアンチパターン

| アンチパターン | 問題点 | 正しいアプローチ |
|-------------|--------|--------------|
| 全フィールドを一度にバリデーション | ユーザーが見えないエラーに困惑 | ステップ単位でバリデーション |
| ステップ間でフォームをリセット | データが失われる | 共通の form インスタンスを使用 |
| 戻るボタンで入力データが消える | UX劣化 | defaultValues を適切に管理 |
| 最終ステップのみでAPI送信 | 途中離脱でデータ消失 | ステップ完了時にドラフト保存 |
| プログレスバーなし | ユーザーが進捗を把握できない | 常に進捗を表示 |
| ステップ間のアニメーションなし | 遷移が分かりにくい | 適切なトランジション |

### 1.7 ステップ間のアニメーション

```tsx
import { AnimatePresence, motion } from 'framer-motion';

const stepVariants = {
  enter: (direction: number) => ({
    x: direction > 0 ? 300 : -300,
    opacity: 0,
  }),
  center: {
    x: 0,
    opacity: 1,
  },
  exit: (direction: number) => ({
    x: direction < 0 ? 300 : -300,
    opacity: 0,
  }),
};

function AnimatedStep({
  children,
  direction,
  stepKey,
}: {
  children: React.ReactNode;
  direction: number;
  stepKey: number;
}) {
  return (
    <AnimatePresence mode="wait" custom={direction}>
      <motion.div
        key={stepKey}
        custom={direction}
        variants={stepVariants}
        initial="enter"
        animate="center"
        exit="exit"
        transition={{
          x: { type: 'spring', stiffness: 300, damping: 30 },
          opacity: { duration: 0.2 },
        }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}

// 使用例
function MultiStepFormWithAnimation() {
  const [direction, setDirection] = useState(0);
  // ... useMultiStepForm

  const handleNext = async () => {
    setDirection(1);
    await goToNext();
  };

  const handlePrev = () => {
    setDirection(-1);
    goToPrevious();
  };

  return (
    <form>
      <AnimatedStep direction={direction} stepKey={currentStep}>
        {currentStep === 0 && <Step1Fields form={form} />}
        {currentStep === 1 && <Step2Fields form={form} />}
        {currentStep === 2 && <Step3Fields form={form} />}
      </AnimatedStep>
    </form>
  );
}
```

### 1.8 確認画面の実装

```tsx
// 最終確認ステップ
function ConfirmationStep({ data }: { data: FormData }) {
  const sections = [
    {
      title: 'アカウント情報',
      items: [
        { label: '名前', value: data.name },
        { label: 'メールアドレス', value: data.email },
        { label: 'パスワード', value: '********' },
      ],
    },
    {
      title: 'プロフィール',
      items: [
        { label: '会社名', value: data.company },
        { label: '役割', value: ROLE_LABELS[data.role] },
        { label: '経験年数', value: `${data.experience}年` },
        { label: '自己紹介', value: data.bio || '未入力' },
      ],
    },
    {
      title: 'プラン',
      items: [
        { label: 'プラン', value: PLAN_LABELS[data.plan] },
        { label: '請求サイクル', value: data.billingCycle === 'monthly' ? '月払い' : '年払い' },
      ],
    },
  ];

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-bold">入力内容の確認</h3>
      {sections.map((section) => (
        <div key={section.title} className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-700 mb-3">{section.title}</h4>
          <dl className="space-y-2">
            {section.items.map((item) => (
              <div key={item.label} className="flex">
                <dt className="w-32 text-gray-500 text-sm">{item.label}</dt>
                <dd className="text-sm font-medium">{item.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}

const ROLE_LABELS: Record<string, string> = {
  developer: '開発者',
  designer: 'デザイナー',
  manager: 'マネージャー',
  other: 'その他',
};

const PLAN_LABELS: Record<string, string> = {
  free: 'フリープラン',
  pro: 'プロプラン',
  enterprise: 'エンタープライズ',
};
```

---

## 2. 動的フィールド（useFieldArray）

### 2.1 useFieldArray の基本概念

`useFieldArray` は React Hook Form が提供する、配列形式のフィールドを効率的に管理するためのフックである。動的に行を追加・削除・並び替えする必要があるフォームで威力を発揮する。

**主なユースケース:**
- 注文フォームの商品行
- 請求書の明細行
- アンケートの選択肢
- チームメンバーの招待リスト
- タグやカテゴリの管理
- 住所の複数登録

**useFieldArray が提供するメソッド:**

| メソッド | 説明 | 使用例 |
|---------|------|-------|
| `append` | 末尾に追加 | 新しい行を追加 |
| `prepend` | 先頭に追加 | 先頭に行を挿入 |
| `insert` | 指定位置に挿入 | 特定位置に行を挿入 |
| `remove` | 指定位置を削除 | 行の削除 |
| `swap` | 2つの要素を交換 | 行の入れ替え |
| `move` | 要素を移動 | ドラッグ&ドロップ |
| `update` | 要素を更新 | 行の値を直接更新 |
| `replace` | 全要素を置換 | リスト全体のリセット |

### 2.2 注文フォームの完全実装

```typescript
import { useFieldArray, useForm, useWatch } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

// 商品マスタの型定義
interface Product {
  id: string;
  name: string;
  price: number;
  stock: number;
  category: string;
}

// 注文スキーマ
const orderItemSchema = z.object({
  productId: z.string().min(1, '商品を選択してください'),
  quantity: z.coerce
    .number()
    .min(1, '数量は1以上で入力してください')
    .max(999, '数量は999以下で入力してください'),
  unitPrice: z.coerce.number().min(0),
  discount: z.coerce
    .number()
    .min(0, '割引率は0以上で入力してください')
    .max(100, '割引率は100以下で入力してください')
    .optional()
    .default(0),
  note: z.string().max(200, 'メモは200文字以内で入力してください').optional(),
});

const orderSchema = z.object({
  customerName: z.string().min(1, '顧客名は必須です'),
  customerEmail: z.string().email('有効なメールアドレスを入力してください'),
  shippingAddress: z.string().min(1, '配送先住所は必須です'),
  items: z
    .array(orderItemSchema)
    .min(1, '1つ以上の商品を追加してください')
    .max(50, '一度に注文できるのは50商品までです'),
  notes: z.string().optional(),
  priority: z.enum(['normal', 'urgent', 'express']).default('normal'),
});

type OrderFormData = z.infer<typeof orderSchema>;

// 合計金額計算コンポーネント
function OrderSummary({ control }: { control: any }) {
  const items = useWatch({ control, name: 'items' });

  const summary = useMemo(() => {
    if (!items || !Array.isArray(items)) {
      return { subtotal: 0, discountTotal: 0, total: 0, itemCount: 0 };
    }

    return items.reduce(
      (acc, item) => {
        const lineTotal = (item.unitPrice || 0) * (item.quantity || 0);
        const discountAmount = lineTotal * ((item.discount || 0) / 100);
        return {
          subtotal: acc.subtotal + lineTotal,
          discountTotal: acc.discountTotal + discountAmount,
          total: acc.total + (lineTotal - discountAmount),
          itemCount: acc.itemCount + (item.quantity || 0),
        };
      },
      { subtotal: 0, discountTotal: 0, total: 0, itemCount: 0 }
    );
  }, [items]);

  return (
    <div className="bg-gray-50 rounded-lg p-4 mt-4">
      <h4 className="font-medium mb-3">注文サマリー</h4>
      <dl className="space-y-1 text-sm">
        <div className="flex justify-between">
          <dt className="text-gray-500">商品数</dt>
          <dd>{summary.itemCount}点</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-gray-500">小計</dt>
          <dd>{summary.subtotal.toLocaleString()}円</dd>
        </div>
        <div className="flex justify-between text-red-500">
          <dt>割引合計</dt>
          <dd>-{summary.discountTotal.toLocaleString()}円</dd>
        </div>
        <div className="flex justify-between font-bold text-lg border-t pt-2 mt-2">
          <dt>合計</dt>
          <dd>{summary.total.toLocaleString()}円</dd>
        </div>
      </dl>
    </div>
  );
}

// 注文フォーム本体
function OrderForm({ products }: { products: Product[] }) {
  const form = useForm<OrderFormData>({
    resolver: zodResolver(orderSchema),
    defaultValues: {
      customerName: '',
      customerEmail: '',
      shippingAddress: '',
      items: [{ productId: '', quantity: 1, unitPrice: 0, discount: 0, note: '' }],
      notes: '',
      priority: 'normal',
    },
  });

  const { fields, append, remove, move, insert } = useFieldArray({
    control: form.control,
    name: 'items',
  });

  const handleProductChange = (index: number, productId: string) => {
    const product = products.find(p => p.id === productId);
    if (product) {
      form.setValue(`items.${index}.unitPrice`, product.price);
      form.setValue(`items.${index}.productId`, productId);
    }
  };

  const handleDuplicate = (index: number) => {
    const currentItem = form.getValues(`items.${index}`);
    insert(index + 1, { ...currentItem });
  };

  const onSubmit = async (data: OrderFormData) => {
    try {
      await api.orders.create(data);
      toast.success('注文を作成しました');
      form.reset();
    } catch (error) {
      toast.error('注文の作成に失敗しました');
    }
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
      {/* 顧客情報 */}
      <fieldset className="border rounded-lg p-4">
        <legend className="text-lg font-medium px-2">顧客情報</legend>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
          <div>
            <label className="block text-sm font-medium mb-1">顧客名</label>
            <input {...form.register('customerName')} className="w-full border rounded px-3 py-2" />
            {form.formState.errors.customerName && (
              <p className="text-red-500 text-sm mt-1">
                {form.formState.errors.customerName.message}
              </p>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">メールアドレス</label>
            <input
              type="email"
              {...form.register('customerEmail')}
              className="w-full border rounded px-3 py-2"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-sm font-medium mb-1">配送先住所</label>
            <input
              {...form.register('shippingAddress')}
              className="w-full border rounded px-3 py-2"
            />
          </div>
        </div>
      </fieldset>

      {/* 商品明細 */}
      <fieldset className="border rounded-lg p-4">
        <legend className="text-lg font-medium px-2">
          商品明細 ({fields.length}件)
        </legend>

        {form.formState.errors.items?.message && (
          <p className="text-red-500 text-sm mb-2">
            {form.formState.errors.items.message}
          </p>
        )}

        <div className="space-y-3 mt-2">
          {fields.map((field, index) => (
            <div
              key={field.id}
              className="flex flex-wrap gap-2 items-start p-3 bg-gray-50 rounded-lg"
            >
              <span className="text-gray-400 text-sm self-center w-6">
                {index + 1}.
              </span>

              <div className="flex-1 min-w-[200px]">
                <select
                  {...form.register(`items.${index}.productId`)}
                  onChange={(e) => handleProductChange(index, e.target.value)}
                  className="w-full border rounded px-2 py-1.5 text-sm"
                >
                  <option value="">商品を選択</option>
                  {products.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.name} ({p.price.toLocaleString()}円)
                    </option>
                  ))}
                </select>
              </div>

              <div className="w-24">
                <input
                  type="number"
                  {...form.register(`items.${index}.quantity`)}
                  className="w-full border rounded px-2 py-1.5 text-sm"
                  placeholder="数量"
                  min={1}
                />
              </div>

              <div className="w-24">
                <input
                  type="number"
                  {...form.register(`items.${index}.discount`)}
                  className="w-full border rounded px-2 py-1.5 text-sm"
                  placeholder="割引%"
                  min={0}
                  max={100}
                />
              </div>

              <div className="flex-1 min-w-[150px]">
                <input
                  {...form.register(`items.${index}.note`)}
                  className="w-full border rounded px-2 py-1.5 text-sm"
                  placeholder="メモ"
                />
              </div>

              <div className="flex gap-1">
                {index > 0 && (
                  <button
                    type="button"
                    onClick={() => move(index, index - 1)}
                    className="p-1.5 text-gray-400 hover:text-gray-600"
                    title="上に移動"
                  >
                    ↑
                  </button>
                )}
                {index < fields.length - 1 && (
                  <button
                    type="button"
                    onClick={() => move(index, index + 1)}
                    className="p-1.5 text-gray-400 hover:text-gray-600"
                    title="下に移動"
                  >
                    ↓
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => handleDuplicate(index)}
                  className="p-1.5 text-gray-400 hover:text-blue-600"
                  title="複製"
                >
                  複製
                </button>
                {fields.length > 1 && (
                  <button
                    type="button"
                    onClick={() => remove(index)}
                    className="p-1.5 text-gray-400 hover:text-red-600"
                    title="削除"
                  >
                    削除
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        <button
          type="button"
          onClick={() => append({ productId: '', quantity: 1, unitPrice: 0, discount: 0, note: '' })}
          className="mt-3 px-4 py-2 border-2 border-dashed border-gray-300
            rounded-lg text-gray-500 hover:border-blue-400 hover:text-blue-500
            transition-colors w-full"
          disabled={fields.length >= 50}
        >
          + 商品を追加
        </button>
      </fieldset>

      {/* 注文サマリー */}
      <OrderSummary control={form.control} />

      {/* 送信ボタン */}
      <button
        type="submit"
        disabled={form.formState.isSubmitting}
        className="w-full py-3 bg-blue-600 text-white rounded-lg
          hover:bg-blue-700 disabled:opacity-50 font-medium"
      >
        {form.formState.isSubmitting ? '注文処理中...' : '注文を確定する'}
      </button>
    </form>
  );
}
```

### 2.3 ネストした useFieldArray

```typescript
// 請求書フォーム: セクション > 明細行 のネスト構造
const invoiceSectionSchema = z.object({
  sectionTitle: z.string().min(1, 'セクション名は必須です'),
  items: z.array(z.object({
    description: z.string().min(1, '説明は必須です'),
    quantity: z.coerce.number().min(1),
    unitPrice: z.coerce.number().min(0),
    taxRate: z.coerce.number().min(0).max(100).default(10),
  })).min(1, '1つ以上の明細を追加してください'),
});

const invoiceSchema = z.object({
  invoiceNumber: z.string().min(1),
  clientName: z.string().min(1),
  issueDate: z.string().min(1),
  dueDate: z.string().min(1),
  sections: z.array(invoiceSectionSchema).min(1),
});

type InvoiceFormData = z.infer<typeof invoiceSchema>;

function InvoiceForm() {
  const form = useForm<InvoiceFormData>({
    resolver: zodResolver(invoiceSchema),
    defaultValues: {
      invoiceNumber: '',
      clientName: '',
      issueDate: new Date().toISOString().split('T')[0],
      dueDate: '',
      sections: [
        {
          sectionTitle: 'サービス',
          items: [{ description: '', quantity: 1, unitPrice: 0, taxRate: 10 }],
        },
      ],
    },
  });

  const { fields: sectionFields, append: appendSection, remove: removeSection } =
    useFieldArray({
      control: form.control,
      name: 'sections',
    });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {sectionFields.map((section, sectionIndex) => (
        <InvoiceSection
          key={section.id}
          form={form}
          sectionIndex={sectionIndex}
          onRemove={() => removeSection(sectionIndex)}
          canRemove={sectionFields.length > 1}
        />
      ))}

      <button
        type="button"
        onClick={() =>
          appendSection({
            sectionTitle: '',
            items: [{ description: '', quantity: 1, unitPrice: 0, taxRate: 10 }],
          })
        }
      >
        + セクションを追加
      </button>
    </form>
  );
}

// ネストした明細行コンポーネント
function InvoiceSection({
  form,
  sectionIndex,
  onRemove,
  canRemove,
}: {
  form: UseFormReturn<InvoiceFormData>;
  sectionIndex: number;
  onRemove: () => void;
  canRemove: boolean;
}) {
  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: `sections.${sectionIndex}.items`,
  });

  return (
    <div className="border rounded-lg p-4 mb-4">
      <div className="flex justify-between items-center mb-4">
        <input
          {...form.register(`sections.${sectionIndex}.sectionTitle`)}
          className="text-lg font-medium border-b border-transparent
            hover:border-gray-300 focus:border-blue-500 outline-none"
          placeholder="セクション名"
        />
        {canRemove && (
          <button type="button" onClick={onRemove} className="text-red-500">
            セクション削除
          </button>
        )}
      </div>

      <table className="w-full">
        <thead>
          <tr className="text-left text-sm text-gray-500">
            <th className="pb-2">説明</th>
            <th className="pb-2 w-24">数量</th>
            <th className="pb-2 w-32">単価</th>
            <th className="pb-2 w-24">税率</th>
            <th className="pb-2 w-32">小計</th>
            <th className="pb-2 w-16"></th>
          </tr>
        </thead>
        <tbody>
          {fields.map((field, itemIndex) => (
            <InvoiceLineItem
              key={field.id}
              form={form}
              sectionIndex={sectionIndex}
              itemIndex={itemIndex}
              onRemove={() => remove(itemIndex)}
              canRemove={fields.length > 1}
            />
          ))}
        </tbody>
      </table>

      <button
        type="button"
        onClick={() => append({ description: '', quantity: 1, unitPrice: 0, taxRate: 10 })}
        className="mt-2 text-sm text-blue-500 hover:text-blue-700"
      >
        + 明細を追加
      </button>
    </div>
  );
}
```

### 2.4 ドラッグ&ドロップによる並び替え

```typescript
import { DndContext, closestCenter, DragEndEvent } from '@dnd-kit/core';
import {
  SortableContext,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

// ソート可能な行コンポーネント
function SortableItem({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
    zIndex: isDragging ? 1 : 0,
  };

  return (
    <div ref={setNodeRef} style={style} className="relative">
      <div
        {...attributes}
        {...listeners}
        className="absolute left-0 top-0 bottom-0 w-8 flex items-center
          justify-center cursor-grab active:cursor-grabbing text-gray-400
          hover:text-gray-600"
      >
        ⋮⋮
      </div>
      <div className="pl-8">{children}</div>
    </div>
  );
}

// ドラッグ&ドロップ対応の動的フィールド
function DraggableFieldArray() {
  const form = useForm({ /* ... */ });
  const { fields, move } = useFieldArray({
    control: form.control,
    name: 'items',
  });

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    const oldIndex = fields.findIndex(f => f.id === active.id);
    const newIndex = fields.findIndex(f => f.id === over.id);

    if (oldIndex !== -1 && newIndex !== -1) {
      move(oldIndex, newIndex);
    }
  };

  return (
    <DndContext
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <SortableContext
        items={fields.map(f => f.id)}
        strategy={verticalListSortingStrategy}
      >
        {fields.map((field, index) => (
          <SortableItem key={field.id} id={field.id}>
            {/* フィールドの内容 */}
            <input {...form.register(`items.${index}.name`)} />
          </SortableItem>
        ))}
      </SortableContext>
    </DndContext>
  );
}
```

### 2.5 useFieldArray のパフォーマンス最適化

```typescript
// アンチパターン: 全フィールドの再レンダリング
function BadExample() {
  const { fields } = useFieldArray({ control, name: 'items' });
  // watch() で全体を監視すると、1フィールドの変更で全行が再レンダリング
  const allValues = form.watch('items'); // 非推奨

  return fields.map((field, i) => (
    <div key={field.id}>
      <input {...form.register(`items.${i}.name`)} />
      <span>合計: {allValues[i].price * allValues[i].quantity}</span>
    </div>
  ));
}

// 推奨パターン: 行ごとに useWatch を使用
function OptimizedRow({
  index,
  control,
  register,
}: {
  index: number;
  control: any;
  register: any;
}) {
  // この行のデータのみを監視 → この行だけが再レンダリング
  const item = useWatch({ control, name: `items.${index}` });

  const lineTotal = useMemo(
    () => (item.price || 0) * (item.quantity || 0),
    [item.price, item.quantity]
  );

  return (
    <div>
      <input {...register(`items.${index}.name`)} />
      <input type="number" {...register(`items.${index}.price`)} />
      <input type="number" {...register(`items.${index}.quantity`)} />
      <span>小計: {lineTotal.toLocaleString()}円</span>
    </div>
  );
}

function GoodExample() {
  const { fields } = useFieldArray({ control: form.control, name: 'items' });

  return fields.map((field, i) => (
    <OptimizedRow
      key={field.id}
      index={i}
      control={form.control}
      register={form.register}
    />
  ));
}
```

### 2.6 useFieldArray のよくある落とし穴

| 問題 | 原因 | 解決策 |
|------|------|--------|
| key に index を使用してデータがずれる | React の再レンダリング最適化が誤動作 | `field.id` を key に使用する |
| append 後に新しいフィールドにフォーカスしない | デフォルトではフォーカス制御されない | `shouldFocus: true` オプションを使用 |
| remove 後にバリデーションエラーが残る | エラーの再評価が走らない | `form.clearErrors()` を呼ぶ |
| 大量のフィールドでパフォーマンス劣化 | 全フィールドが同時にレンダリング | 仮想化（react-window）を使用 |
| ネストした配列で型エラーが出る | TypeScript の型推論の限界 | `as const` や明示的な型注釈を使用 |

---

## 3. 条件分岐フォーム

### 3.1 discriminatedUnion パターン

条件分岐フォームは、ユーザーの選択に応じて表示するフィールドとバリデーションルールを動的に切り替えるフォームである。Zod の `discriminatedUnion` が最も直感的に実装できるパターンである。

```typescript
// 通知設定: タイプによって異なるフィールドとバリデーション
const emailNotificationSchema = z.object({
  type: z.literal('email'),
  email: z.string().email('有効なメールアドレスを入力してください'),
  frequency: z.enum(['daily', 'weekly', 'monthly'], {
    errorMap: () => ({ message: '配信頻度を選択してください' }),
  }),
  format: z.enum(['html', 'text']).default('html'),
  categories: z.array(z.string()).min(1, '1つ以上のカテゴリを選択してください'),
});

const smsNotificationSchema = z.object({
  type: z.literal('sms'),
  phone: z.string()
    .regex(/^\+?\d{10,15}$/, '有効な電話番号を入力してください'),
  maxMessages: z.coerce
    .number()
    .min(1)
    .max(100)
    .default(10),
});

const webhookNotificationSchema = z.object({
  type: z.literal('webhook'),
  url: z.string().url('有効なURLを入力してください'),
  secret: z.string()
    .min(16, 'シークレットキーは16文字以上で入力してください'),
  events: z.array(z.string()).min(1, '1つ以上のイベントを選択してください'),
  retryCount: z.coerce.number().min(0).max(5).default(3),
  timeout: z.coerce.number().min(1000).max(30000).default(5000),
});

const slackNotificationSchema = z.object({
  type: z.literal('slack'),
  webhookUrl: z.string().url('有効なWebhook URLを入力してください'),
  channel: z.string().min(1, 'チャンネル名は必須です'),
  username: z.string().optional(),
  iconEmoji: z.string().optional(),
});

const notificationSchema = z.discriminatedUnion('type', [
  emailNotificationSchema,
  smsNotificationSchema,
  webhookNotificationSchema,
  slackNotificationSchema,
]);

type NotificationFormData = z.infer<typeof notificationSchema>;
```

### 3.2 条件分岐フォームの完全実装

```tsx
function NotificationForm() {
  const form = useForm<NotificationFormData>({
    resolver: zodResolver(notificationSchema),
    defaultValues: { type: 'email' as const },
  });

  const notificationType = form.watch('type');

  // タイプ変更時にフィールドをリセット
  const handleTypeChange = (newType: NotificationFormData['type']) => {
    // 現在のタイプ固有のフィールドをクリア
    form.reset({ type: newType } as any);
  };

  const onSubmit = async (data: NotificationFormData) => {
    try {
      await api.notifications.create(data);
      toast.success('通知設定を保存しました');
    } catch (error) {
      toast.error('保存に失敗しました');
    }
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
      {/* 通知タイプ選択 */}
      <div>
        <label className="block text-sm font-medium mb-2">通知タイプ</label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {NOTIFICATION_TYPES.map((option) => (
            <label
              key={option.value}
              className={`flex items-center gap-2 p-3 border rounded-lg cursor-pointer
                transition-colors
                ${notificationType === option.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
                }
              `}
            >
              <input
                type="radio"
                value={option.value}
                checked={notificationType === option.value}
                onChange={() => handleTypeChange(option.value)}
                className="sr-only"
              />
              <span className="text-xl">{option.icon}</span>
              <span className="text-sm font-medium">{option.label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* 条件分岐フィールド */}
      {notificationType === 'email' && <EmailFields form={form} />}
      {notificationType === 'sms' && <SmsFields form={form} />}
      {notificationType === 'webhook' && <WebhookFields form={form} />}
      {notificationType === 'slack' && <SlackFields form={form} />}

      <button
        type="submit"
        disabled={form.formState.isSubmitting}
        className="w-full py-2 bg-blue-600 text-white rounded-lg"
      >
        保存する
      </button>
    </form>
  );
}

const NOTIFICATION_TYPES = [
  { value: 'email' as const, label: 'メール', icon: 'M' },
  { value: 'sms' as const, label: 'SMS', icon: 'S' },
  { value: 'webhook' as const, label: 'Webhook', icon: 'W' },
  { value: 'slack' as const, label: 'Slack', icon: 'K' },
];

// メール通知フィールド
function EmailFields({ form }: { form: UseFormReturn<any> }) {
  return (
    <div className="space-y-4 p-4 bg-blue-50 rounded-lg">
      <div>
        <label className="block text-sm font-medium mb-1">メールアドレス</label>
        <input
          type="email"
          {...form.register('email')}
          className="w-full border rounded px-3 py-2"
          placeholder="example@company.com"
        />
        {form.formState.errors.email && (
          <p className="text-red-500 text-sm mt-1">
            {form.formState.errors.email.message as string}
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">配信頻度</label>
        <select {...form.register('frequency')} className="w-full border rounded px-3 py-2">
          <option value="">選択してください</option>
          <option value="daily">毎日</option>
          <option value="weekly">毎週</option>
          <option value="monthly">毎月</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">フォーマット</label>
        <div className="flex gap-4">
          <label className="flex items-center gap-2">
            <input type="radio" value="html" {...form.register('format')} />
            HTML
          </label>
          <label className="flex items-center gap-2">
            <input type="radio" value="text" {...form.register('format')} />
            テキスト
          </label>
        </div>
      </div>
    </div>
  );
}

// Webhook フィールド
function WebhookFields({ form }: { form: UseFormReturn<any> }) {
  const [showSecret, setShowSecret] = useState(false);

  return (
    <div className="space-y-4 p-4 bg-purple-50 rounded-lg">
      <div>
        <label className="block text-sm font-medium mb-1">Webhook URL</label>
        <input
          type="url"
          {...form.register('url')}
          className="w-full border rounded px-3 py-2"
          placeholder="https://api.example.com/webhooks"
        />
        {form.formState.errors.url && (
          <p className="text-red-500 text-sm mt-1">
            {form.formState.errors.url.message as string}
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">シークレットキー</label>
        <div className="relative">
          <input
            type={showSecret ? 'text' : 'password'}
            {...form.register('secret')}
            className="w-full border rounded px-3 py-2 pr-20"
            placeholder="16文字以上のシークレットキー"
          />
          <button
            type="button"
            onClick={() => setShowSecret(s => !s)}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-sm text-gray-500"
          >
            {showSecret ? '隠す' : '表示'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">リトライ回数</label>
          <input
            type="number"
            {...form.register('retryCount')}
            className="w-full border rounded px-3 py-2"
            min={0}
            max={5}
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">タイムアウト (ms)</label>
          <input
            type="number"
            {...form.register('timeout')}
            className="w-full border rounded px-3 py-2"
            min={1000}
            max={30000}
            step={1000}
          />
        </div>
      </div>
    </div>
  );
}
```

### 3.3 superRefine を使った複雑な条件分岐バリデーション

```typescript
// discriminatedUnion では対応できない複雑な条件分岐
const paymentSchema = z.object({
  paymentMethod: z.enum(['credit_card', 'bank_transfer', 'invoice']),
  // クレジットカード情報
  cardNumber: z.string().optional(),
  cardExpiry: z.string().optional(),
  cardCvc: z.string().optional(),
  // 銀行振込情報
  bankName: z.string().optional(),
  branchName: z.string().optional(),
  accountNumber: z.string().optional(),
  // 請求書情報
  companyName: z.string().optional(),
  billingAddress: z.string().optional(),
  taxId: z.string().optional(),
}).superRefine((data, ctx) => {
  switch (data.paymentMethod) {
    case 'credit_card':
      if (!data.cardNumber || !/^\d{16}$/.test(data.cardNumber)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '有効なカード番号（16桁）を入力してください',
          path: ['cardNumber'],
        });
      }
      if (!data.cardExpiry || !/^\d{2}\/\d{2}$/.test(data.cardExpiry)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '有効期限をMM/YY形式で入力してください',
          path: ['cardExpiry'],
        });
      }
      if (!data.cardCvc || !/^\d{3,4}$/.test(data.cardCvc)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'CVCは3〜4桁の数字で入力してください',
          path: ['cardCvc'],
        });
      }
      break;

    case 'bank_transfer':
      if (!data.bankName?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '銀行名は必須です',
          path: ['bankName'],
        });
      }
      if (!data.accountNumber || !/^\d{7}$/.test(data.accountNumber)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '口座番号は7桁の数字で入力してください',
          path: ['accountNumber'],
        });
      }
      break;

    case 'invoice':
      if (!data.companyName?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '会社名は必須です',
          path: ['companyName'],
        });
      }
      if (!data.billingAddress?.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: '請求先住所は必須です',
          path: ['billingAddress'],
        });
      }
      break;
  }
});
```

### 3.4 依存関係のあるフィールド

```typescript
// 都道府県 → 市区町村 → 住所 のカスケード選択
function CascadeSelectForm() {
  const form = useForm<AddressFormData>();

  const selectedPrefecture = form.watch('prefecture');
  const selectedCity = form.watch('city');

  // 都道府県に応じた市区町村リストを取得
  const { data: cities, isLoading: citiesLoading } = useQuery({
    queryKey: ['cities', selectedPrefecture],
    queryFn: () => api.getCities(selectedPrefecture),
    enabled: !!selectedPrefecture,
  });

  // 市区町村に応じた町名リストを取得
  const { data: towns, isLoading: townsLoading } = useQuery({
    queryKey: ['towns', selectedCity],
    queryFn: () => api.getTowns(selectedCity),
    enabled: !!selectedCity,
  });

  // 都道府県が変わったら市区町村をリセット
  useEffect(() => {
    form.setValue('city', '');
    form.setValue('town', '');
  }, [selectedPrefecture]);

  // 市区町村が変わったら町名をリセット
  useEffect(() => {
    form.setValue('town', '');
  }, [selectedCity]);

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">都道府県</label>
          <select {...form.register('prefecture')} className="w-full border rounded px-3 py-2">
            <option value="">選択してください</option>
            {PREFECTURES.map(pref => (
              <option key={pref.code} value={pref.code}>{pref.name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">市区町村</label>
          <select
            {...form.register('city')}
            className="w-full border rounded px-3 py-2"
            disabled={!selectedPrefecture || citiesLoading}
          >
            <option value="">
              {citiesLoading ? '読み込み中...' : '選択してください'}
            </option>
            {cities?.map(city => (
              <option key={city.code} value={city.code}>{city.name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">町名</label>
          <select
            {...form.register('town')}
            className="w-full border rounded px-3 py-2"
            disabled={!selectedCity || townsLoading}
          >
            <option value="">
              {townsLoading ? '読み込み中...' : '選択してください'}
            </option>
            {towns?.map(town => (
              <option key={town.code} value={town.code}>{town.name}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="mt-4">
        <label className="block text-sm font-medium mb-1">番地・建物名</label>
        <input
          {...form.register('address')}
          className="w-full border rounded px-3 py-2"
          placeholder="1-2-3 サンプルビル 4F"
        />
      </div>
    </form>
  );
}
```

### 3.5 条件分岐フォームの比較表

| パターン | 適用場面 | メリット | デメリット |
|---------|---------|---------|----------|
| `discriminatedUnion` | 選択肢ごとに完全に異なるフィールド | 型安全、簡潔 | 共通フィールドの扱いが冗長 |
| `superRefine` | 共通フィールド + 条件付きバリデーション | 柔軟性が高い | 型推論が弱い |
| `watch` + 動的レンダリング | UIの出し分けのみ | シンプル | バリデーション側の制御が別途必要 |
| カスケード選択 | 親子関係のあるデータ | UXが良い | API呼び出しが増える |

---

## 4. フォームの自動保存

### 4.1 デバウンス付き自動保存

フォームの自動保存は、ユーザーが入力中のデータを定期的にサーバーやローカルストレージに保存し、ブラウザクラッシュや誤操作によるデータ消失を防止する機能である。

```typescript
import { useEffect, useRef, useMemo, useCallback } from 'react';
import { useForm, useWatch } from 'react-hook-form';

// デバウンスユーティリティ
function useDebouncedCallback<T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T {
  const timeoutRef = useRef<NodeJS.Timeout>();
  const callbackRef = useRef(callback);

  // コールバックの最新版を保持
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);

  // クリーンアップ
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return useMemo(
    () =>
      ((...args: any[]) => {
        if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
        }
        timeoutRef.current = setTimeout(() => {
          callbackRef.current(...args);
        }, delay);
      }) as T,
    [delay]
  );
}

// 自動保存のステータス型
type AutoSaveStatus = 'idle' | 'saving' | 'saved' | 'error';

// 自動保存フック
function useAutoSave<T extends Record<string, any>>({
  form,
  onSave,
  debounceMs = 1500,
  enabled = true,
}: {
  form: ReturnType<typeof useForm<T>>;
  onSave: (data: Partial<T>) => Promise<void>;
  debounceMs?: number;
  enabled?: boolean;
}) {
  const [status, setStatus] = useState<AutoSaveStatus>('idle');
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const save = useCallback(async (data: Partial<T>) => {
    setStatus('saving');
    setError(null);

    try {
      await onSave(data);
      setStatus('saved');
      setLastSaved(new Date());

      // 3秒後に idle に戻す
      setTimeout(() => setStatus('idle'), 3000);
    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err : new Error('保存に失敗しました'));
    }
  }, [onSave]);

  const debouncedSave = useDebouncedCallback(save, debounceMs);

  // フォーム値の変更を監視
  useEffect(() => {
    if (!enabled) return;

    const subscription = form.watch((value) => {
      if (form.formState.isDirty) {
        debouncedSave(value as Partial<T>);
      }
    });

    return () => subscription.unsubscribe();
  }, [form, debouncedSave, enabled]);

  return { status, lastSaved, error };
}

// 自動保存ステータス表示コンポーネント
function AutoSaveIndicator({
  status,
  lastSaved,
  error,
}: {
  status: AutoSaveStatus;
  lastSaved: Date | null;
  error: Error | null;
}) {
  const statusConfig = {
    idle: { text: '', className: 'text-gray-400' },
    saving: { text: '保存中...', className: 'text-blue-500' },
    saved: { text: '保存しました', className: 'text-green-500' },
    error: { text: '保存に失敗しました', className: 'text-red-500' },
  };

  const config = statusConfig[status];

  return (
    <div className={`flex items-center gap-2 text-xs ${config.className}`}>
      {status === 'saving' && (
        <span className="animate-spin inline-block w-3 h-3 border-2
          border-current border-t-transparent rounded-full" />
      )}
      <span>{config.text}</span>
      {lastSaved && status === 'idle' && (
        <span className="text-gray-400">
          最終保存: {lastSaved.toLocaleTimeString()}
        </span>
      )}
      {error && (
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="text-red-500 underline ml-2"
        >
          再読み込み
        </button>
      )}
    </div>
  );
}
```

### 4.2 localStorage を使ったドラフト保存

```typescript
// localStorage ベースのドラフト管理
function useFormDraft<T extends Record<string, any>>(
  key: string,
  defaultValues: T,
  options?: {
    debounceMs?: number;
    expiresInMs?: number; // ドラフトの有効期限
  }
) {
  const { debounceMs = 1000, expiresInMs = 24 * 60 * 60 * 1000 } = options ?? {};

  // ドラフトの読み込み
  const loadDraft = useCallback((): T | null => {
    try {
      const stored = localStorage.getItem(`draft:${key}`);
      if (!stored) return null;

      const { data, timestamp } = JSON.parse(stored);
      const isExpired = Date.now() - timestamp > expiresInMs;

      if (isExpired) {
        localStorage.removeItem(`draft:${key}`);
        return null;
      }

      return data as T;
    } catch {
      return null;
    }
  }, [key, expiresInMs]);

  // ドラフトの保存
  const saveDraft = useCallback((data: Partial<T>) => {
    try {
      localStorage.setItem(`draft:${key}`, JSON.stringify({
        data,
        timestamp: Date.now(),
      }));
    } catch (error) {
      console.warn('ドラフトの保存に失敗しました:', error);
    }
  }, [key]);

  // ドラフトの削除
  const clearDraft = useCallback(() => {
    localStorage.removeItem(`draft:${key}`);
  }, [key]);

  // ドラフトの存在確認
  const hasDraft = useMemo(() => {
    return loadDraft() !== null;
  }, [loadDraft]);

  // 初期値（ドラフトがあればそちらを優先）
  const initialValues = useMemo(() => {
    const draft = loadDraft();
    return draft ?? defaultValues;
  }, [loadDraft, defaultValues]);

  return {
    initialValues,
    hasDraft,
    saveDraft,
    clearDraft,
    loadDraft,
  };
}

// 使用例: ドラフト復元ダイアログ付きフォーム
function ArticleForm() {
  const {
    initialValues,
    hasDraft,
    saveDraft,
    clearDraft,
  } = useFormDraft('article-editor', {
    title: '',
    content: '',
    tags: [],
    status: 'draft',
  });

  const [showDraftDialog, setShowDraftDialog] = useState(hasDraft);

  const form = useForm({
    defaultValues: showDraftDialog ? initialValues : {
      title: '',
      content: '',
      tags: [],
      status: 'draft',
    },
  });

  const { status } = useAutoSave({
    form,
    onSave: async (data) => {
      saveDraft(data);
    },
    debounceMs: 2000,
  });

  const handleSubmit = async (data: any) => {
    await api.articles.create(data);
    clearDraft(); // 送信成功時にドラフトを削除
  };

  return (
    <>
      {/* ドラフト復元ダイアログ */}
      {showDraftDialog && (
        <div className="mb-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg
          flex items-center justify-between">
          <p className="text-sm text-yellow-700">
            前回の下書きが見つかりました。復元しますか?
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                form.reset(initialValues);
                setShowDraftDialog(false);
              }}
              className="px-3 py-1 bg-yellow-500 text-white rounded text-sm"
            >
              復元する
            </button>
            <button
              type="button"
              onClick={() => {
                clearDraft();
                setShowDraftDialog(false);
              }}
              className="px-3 py-1 border rounded text-sm"
            >
              破棄する
            </button>
          </div>
        </div>
      )}

      <form onSubmit={form.handleSubmit(handleSubmit)}>
        <AutoSaveIndicator status={status} lastSaved={null} error={null} />
        {/* フォームフィールド */}
      </form>
    </>
  );
}
```

### 4.3 サーバーサイドドラフト保存

```typescript
// サーバーサイドのドラフト保存（TanStack Query 連携）
function useServerDraft<T>({
  draftId,
  defaultValues,
  form,
}: {
  draftId: string | null;
  defaultValues: T;
  form: ReturnType<typeof useForm<T>>;
}) {
  // ドラフトの読み込み
  const { data: savedDraft, isLoading } = useQuery({
    queryKey: ['draft', draftId],
    queryFn: () => api.drafts.get(draftId!),
    enabled: !!draftId,
    staleTime: 0,
  });

  // ドラフトの保存
  const saveMutation = useMutation({
    mutationFn: (data: Partial<T>) =>
      draftId
        ? api.drafts.update(draftId, data)
        : api.drafts.create(data),
    onSuccess: (response) => {
      if (!draftId) {
        // 新規作成時はURLにドラフトIDを追加
        router.replace(`/editor?draftId=${response.id}`);
      }
    },
  });

  // ドラフトの自動保存
  const { status } = useAutoSave({
    form,
    onSave: async (data) => {
      await saveMutation.mutateAsync(data);
    },
    debounceMs: 3000,
    enabled: !isLoading,
  });

  // ドラフト読み込み時にフォームを更新
  useEffect(() => {
    if (savedDraft) {
      form.reset(savedDraft.data as T);
    }
  }, [savedDraft, form]);

  return {
    status,
    isLoading,
    isDraftSaving: saveMutation.isPending,
  };
}
```

---

## 5. ページ離脱防止

### 5.1 BeforeUnload イベントによる離脱防止

```typescript
// 未保存の変更がある場合にページ離脱を防止するフック
function useUnsavedChangesWarning(isDirty: boolean, message?: string) {
  const defaultMessage = '変更が保存されていません。ページを離れますか?';

  useEffect(() => {
    if (!isDirty) return;

    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      // 最新のブラウザでは returnValue の設定は不要だが、互換性のため残す
      e.returnValue = message ?? defaultMessage;
      return message ?? defaultMessage;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty, message]);
}

// 使用例
function EditForm() {
  const form = useForm({ defaultValues: { title: '', content: '' } });

  useUnsavedChangesWarning(form.formState.isDirty);

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input {...form.register('title')} />
      <textarea {...form.register('content')} />
      <button type="submit">保存</button>
    </form>
  );
}
```

### 5.2 React Router での離脱防止

```typescript
// React Router v6 でのルート遷移防止
import { useBlocker, useNavigate } from 'react-router-dom';

function useNavigationBlocker(isDirty: boolean) {
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      isDirty && currentLocation.pathname !== nextLocation.pathname
  );

  return blocker;
}

// 確認ダイアログコンポーネント
function UnsavedChangesDialog({
  blocker,
}: {
  blocker: ReturnType<typeof useBlocker>;
}) {
  if (blocker.state !== 'blocked') return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md mx-4">
        <h3 className="text-lg font-bold mb-2">変更が保存されていません</h3>
        <p className="text-gray-600 mb-6">
          保存されていない変更があります。このページを離れると変更は失われます。
        </p>
        <div className="flex justify-end gap-3">
          <button
            onClick={() => blocker.reset()}
            className="px-4 py-2 border rounded-lg hover:bg-gray-50"
          >
            このページにとどまる
          </button>
          <button
            onClick={() => blocker.proceed()}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            変更を破棄して移動
          </button>
        </div>
      </div>
    </div>
  );
}

// 統合した使用例
function ProtectedForm() {
  const form = useForm({ defaultValues: { title: '', content: '' } });
  const isDirty = form.formState.isDirty;

  // ブラウザバック・リロード防止
  useUnsavedChangesWarning(isDirty);

  // React Router でのルート遷移防止
  const blocker = useNavigationBlocker(isDirty);

  return (
    <>
      <form onSubmit={form.handleSubmit(async (data) => {
        await api.save(data);
        form.reset(data); // isDirty を false にする
      })}>
        <input {...form.register('title')} />
        <textarea {...form.register('content')} />
        <button type="submit">保存</button>
      </form>

      <UnsavedChangesDialog blocker={blocker} />
    </>
  );
}
```

### 5.3 Next.js App Router での離脱防止

```typescript
// Next.js App Router での離脱防止
// App Router ではネイティブの useBlocker が使えないため、独自実装が必要

'use client';

import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useRef, useCallback } from 'react';

function useNextNavigationGuard(isDirty: boolean) {
  const pathname = usePathname();
  const router = useRouter();
  const isDirtyRef = useRef(isDirty);

  useEffect(() => {
    isDirtyRef.current = isDirty;
  }, [isDirty]);

  // popstate（ブラウザバック）の処理
  useEffect(() => {
    if (!isDirty) return;

    const handlePopState = (e: PopStateEvent) => {
      if (isDirtyRef.current) {
        const confirmed = window.confirm(
          '変更が保存されていません。ページを離れますか?'
        );
        if (!confirmed) {
          // 元のURLに戻す
          window.history.pushState(null, '', pathname);
        }
      }
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [isDirty, pathname]);

  // Link コンポーネントのクリックをインターセプト
  useEffect(() => {
    if (!isDirty) return;

    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const anchor = target.closest('a');

      if (
        anchor &&
        anchor.href &&
        !anchor.href.startsWith('#') &&
        anchor.target !== '_blank' &&
        isDirtyRef.current
      ) {
        const confirmed = window.confirm(
          '変更が保存されていません。ページを離れますか?'
        );
        if (!confirmed) {
          e.preventDefault();
          e.stopPropagation();
        }
      }
    };

    document.addEventListener('click', handleClick, true);
    return () => document.removeEventListener('click', handleClick, true);
  }, [isDirty]);

  return { isDirty };
}
```

---

## 6. 複雑なフォームのパフォーマンス最適化

### 6.1 再レンダリングの最小化

複雑なフォームでは、フィールド数が増えるにつれてパフォーマンスが劣化しやすい。React Hook Form は非制御コンポーネントベースのため基本的にパフォーマンスが良いが、`watch` や `useWatch` の使い方を誤ると不要な再レンダリングが発生する。

```typescript
// アンチパターン: フォーム全体を watch
function BadPerformanceForm() {
  const form = useForm<LargeFormData>();

  // フォーム全体の値が変わるたびに、このコンポーネント全体が再レンダリング
  const allValues = form.watch(); // 非推奨

  return (
    <div>
      {/* 100個のフィールド全てが不要に再レンダリングされる */}
      {Array.from({ length: 100 }, (_, i) => (
        <input key={i} {...form.register(`field_${i}`)} />
      ))}
      <pre>{JSON.stringify(allValues, null, 2)}</pre>
    </div>
  );
}

// 推奨パターン: 必要な値だけを個別に watch
function GoodPerformanceForm() {
  const form = useForm<LargeFormData>();

  return (
    <div>
      {Array.from({ length: 100 }, (_, i) => (
        <input key={i} {...form.register(`field_${i}`)} />
      ))}
      {/* 値の表示は別コンポーネントに分離 */}
      <FormDebugger control={form.control} />
    </div>
  );
}

// 分離されたデバッガーコンポーネント
function FormDebugger({ control }: { control: any }) {
  // このコンポーネントだけが再レンダリングされる
  const values = useWatch({ control });
  return <pre className="text-xs">{JSON.stringify(values, null, 2)}</pre>;
}
```

### 6.2 React.memo による最適化

```typescript
// 個別フィールドコンポーネントをメモ化
const MemoizedField = React.memo(function MemoizedField({
  name,
  label,
  register,
  error,
  type = 'text',
}: {
  name: string;
  label: string;
  register: any;
  error?: string;
  type?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1">{label}</label>
      <input
        type={type}
        {...register(name)}
        className={`w-full border rounded px-3 py-2 ${error ? 'border-red-500' : ''}`}
      />
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
});

// Controller を使った制御コンポーネントのメモ化
const MemoizedSelect = React.memo(function MemoizedSelect({
  name,
  label,
  control,
  options,
}: {
  name: string;
  label: string;
  control: any;
  options: { value: string; label: string }[];
}) {
  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState }) => (
        <div>
          <label className="block text-sm font-medium mb-1">{label}</label>
          <select
            {...field}
            className={`w-full border rounded px-3 py-2
              ${fieldState.error ? 'border-red-500' : ''}`}
          >
            <option value="">選択してください</option>
            {options.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          {fieldState.error && (
            <p className="text-red-500 text-sm mt-1">{fieldState.error.message}</p>
          )}
        </div>
      )}
    />
  );
});
```

### 6.3 大量データの仮想化

```typescript
import { FixedSizeList as List } from 'react-window';

// 大量の行を持つ動的フォームの仮想化
function VirtualizedFieldArray() {
  const form = useForm<{ items: Array<{ name: string; value: string }> }>({
    defaultValues: {
      items: Array.from({ length: 1000 }, (_, i) => ({
        name: `Item ${i + 1}`,
        value: '',
      })),
    },
  });

  const { fields } = useFieldArray({
    control: form.control,
    name: 'items',
  });

  const Row = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => (
      <div style={style} className="flex gap-2 items-center px-2">
        <span className="text-gray-400 w-12 text-right text-sm">{index + 1}.</span>
        <input
          {...form.register(`items.${index}.name`)}
          className="flex-1 border rounded px-2 py-1 text-sm"
        />
        <input
          {...form.register(`items.${index}.value`)}
          className="flex-1 border rounded px-2 py-1 text-sm"
        />
      </div>
    ),
    [form.register]
  );

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <div className="border rounded-lg overflow-hidden">
        <div className="flex gap-2 px-2 py-2 bg-gray-100 text-sm font-medium">
          <span className="w-12 text-right">#</span>
          <span className="flex-1">名前</span>
          <span className="flex-1">値</span>
        </div>
        <List
          height={400}
          itemCount={fields.length}
          itemSize={40}
          width="100%"
        >
          {Row}
        </List>
      </div>

      <div className="mt-4 text-sm text-gray-500">
        {fields.length}件のアイテム
      </div>

      <button type="submit" className="mt-4 px-4 py-2 bg-blue-600 text-white rounded">
        保存
      </button>
    </form>
  );
}
```

### 6.4 パフォーマンス比較表

| 手法 | 対象 | 効果 | 注意点 |
|------|------|------|--------|
| `useWatch` で個別監視 | 特定フィールドの値参照 | 再レンダリング範囲を限定 | フィールド名の指定が必要 |
| `React.memo` | フィールドコンポーネント | 不要な再レンダリング防止 | props の比較コストに注意 |
| `Controller` の分離 | 制御コンポーネント | レンダリング分離 | コンポーネント数が増える |
| `react-window` | 大量フィールド（100+） | DOM ノード数の削減 | スクロール時の入力に注意 |
| `shouldUnregister: false` | 条件表示フィールド | アンマウント時のデータ保持 | メモリ消費が増える |
| `mode: 'onSubmit'` | バリデーション | 入力中のバリデーションコスト削減 | リアルタイムフィードバックなし |

---

## 7. アクセシビリティ対応

### 7.1 フォームのアクセシビリティ基本原則

複雑なフォームにおけるアクセシビリティは、特にスクリーンリーダーユーザーやキーボードナビゲーションユーザーにとって重要である。以下の原則を遵守する。

**WCAG 2.1 AA準拠のチェックリスト:**
- すべての入力フィールドに適切な `label` が関連付けられている
- エラーメッセージが `aria-describedby` で関連付けられている
- 必須フィールドが `aria-required` で示されている
- フォーカス管理が適切に行われている
- エラー発生時にフォーカスが最初のエラーフィールドに移動する
- 色だけでなくテキストやアイコンでもエラー状態を表現する

```tsx
// アクセシブルなフォームフィールドコンポーネント
function AccessibleField({
  id,
  label,
  required = false,
  error,
  description,
  children,
}: {
  id: string;
  label: string;
  required?: boolean;
  error?: string;
  description?: string;
  children: (props: {
    id: string;
    'aria-invalid': boolean;
    'aria-required': boolean;
    'aria-describedby': string;
  }) => React.ReactNode;
}) {
  const describedByIds = [
    description ? `${id}-description` : null,
    error ? `${id}-error` : null,
  ].filter(Boolean).join(' ');

  return (
    <div className="space-y-1">
      <label htmlFor={id} className="block text-sm font-medium">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-label="必須">
            *
          </span>
        )}
      </label>

      {description && (
        <p id={`${id}-description`} className="text-xs text-gray-500">
          {description}
        </p>
      )}

      {children({
        id,
        'aria-invalid': !!error,
        'aria-required': required,
        'aria-describedby': describedByIds,
      })}

      {error && (
        <p
          id={`${id}-error`}
          className="text-sm text-red-500 flex items-center gap-1"
          role="alert"
          aria-live="polite"
        >
          <span aria-hidden="true">[!]</span>
          {error}
        </p>
      )}
    </div>
  );
}

// 使用例
function AccessibleForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    mode: 'onTouched',
  });

  // エラー発生時に最初のエラーフィールドにフォーカス
  useEffect(() => {
    const errors = form.formState.errors;
    const firstErrorKey = Object.keys(errors)[0];
    if (firstErrorKey) {
      const element = document.getElementById(firstErrorKey);
      element?.focus();
    }
  }, [form.formState.errors]);

  return (
    <form
      onSubmit={form.handleSubmit(onSubmit)}
      noValidate // ブラウザのデフォルトバリデーションを無効化
      aria-label="ユーザー登録フォーム"
    >
      <AccessibleField
        id="name"
        label="名前"
        required
        error={form.formState.errors.name?.message}
      >
        {(ariaProps) => (
          <input
            type="text"
            {...form.register('name')}
            {...ariaProps}
            className="w-full border rounded px-3 py-2"
            autoComplete="name"
          />
        )}
      </AccessibleField>

      <AccessibleField
        id="email"
        label="メールアドレス"
        required
        description="確認メールを送信します"
        error={form.formState.errors.email?.message}
      >
        {(ariaProps) => (
          <input
            type="email"
            {...form.register('email')}
            {...ariaProps}
            className="w-full border rounded px-3 py-2"
            autoComplete="email"
          />
        )}
      </AccessibleField>
    </form>
  );
}
```

### 7.2 マルチステップフォームのアクセシビリティ

```tsx
// アクセシブルなステッパー
function AccessibleStepper({
  steps,
  currentStep,
  completedSteps,
}: {
  steps: StepConfig[];
  currentStep: number;
  completedSteps: Set<number>;
}) {
  return (
    <nav aria-label="フォームの進捗">
      <ol className="flex gap-2" role="list">
        {steps.map((step, i) => {
          const status = completedSteps.has(i)
            ? 'completed'
            : i === currentStep
            ? 'current'
            : 'upcoming';

          return (
            <li
              key={i}
              className="flex items-center gap-2"
              aria-current={i === currentStep ? 'step' : undefined}
            >
              <span
                className={`w-8 h-8 rounded-full flex items-center justify-center text-sm
                  ${status === 'current' ? 'bg-blue-600 text-white' : ''}
                  ${status === 'completed' ? 'bg-green-600 text-white' : ''}
                  ${status === 'upcoming' ? 'bg-gray-200 text-gray-500' : ''}
                `}
                aria-hidden="true"
              >
                {status === 'completed' ? '✓' : i + 1}
              </span>
              <span className="sr-only">
                ステップ {i + 1}: {step.title}
                {status === 'completed' && '（完了）'}
                {status === 'current' && '（現在）'}
              </span>
              <span className="hidden sm:inline text-sm" aria-hidden="true">
                {step.title}
              </span>
            </li>
          );
        })}
      </ol>

      {/* ライブリージョンでステップ変更を通知 */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        ステップ {currentStep + 1} / {steps.length}: {steps[currentStep].title}
      </div>
    </nav>
  );
}
```

### 7.3 動的フィールドのアクセシビリティ

```tsx
// アクセシブルな動的フィールド
function AccessibleFieldArray() {
  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: 'items',
  });

  const [announcement, setAnnouncement] = useState('');

  const handleAppend = () => {
    append({ name: '', value: '' });
    setAnnouncement(`アイテムを追加しました。合計 ${fields.length + 1} 件です。`);
    // 新しいフィールドにフォーカス
    setTimeout(() => {
      const newField = document.getElementById(`items-${fields.length}-name`);
      newField?.focus();
    }, 100);
  };

  const handleRemove = (index: number) => {
    const itemName = form.getValues(`items.${index}.name`) || `アイテム ${index + 1}`;
    remove(index);
    setAnnouncement(`${itemName} を削除しました。合計 ${fields.length - 1} 件です。`);
  };

  return (
    <fieldset>
      <legend className="text-lg font-medium mb-4">アイテムリスト</legend>

      {/* ライブリージョン: 操作結果を通知 */}
      <div className="sr-only" aria-live="assertive" aria-atomic="true">
        {announcement}
      </div>

      <div role="list" aria-label="アイテム一覧">
        {fields.map((field, index) => (
          <div
            key={field.id}
            role="listitem"
            className="flex gap-2 items-center mb-2"
            aria-label={`アイテム ${index + 1}`}
          >
            <label htmlFor={`items-${index}-name`} className="sr-only">
              アイテム {index + 1} の名前
            </label>
            <input
              id={`items-${index}-name`}
              {...form.register(`items.${index}.name`)}
              className="flex-1 border rounded px-3 py-2"
              placeholder={`アイテム ${index + 1}`}
            />

            <button
              type="button"
              onClick={() => handleRemove(index)}
              aria-label={`アイテム ${index + 1} を削除`}
              className="p-2 text-red-500 hover:bg-red-50 rounded"
              disabled={fields.length <= 1}
            >
              削除
            </button>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={handleAppend}
        className="mt-2 px-4 py-2 border-2 border-dashed rounded-lg w-full"
        aria-label="新しいアイテムを追加"
      >
        + アイテムを追加
      </button>
    </fieldset>
  );
}
```

### 7.4 キーボードナビゲーション対応

```typescript
// キーボードショートカットの実装
function useFormKeyboardShortcuts({
  onSave,
  onCancel,
  onNextStep,
  onPrevStep,
}: {
  onSave?: () => void;
  onCancel?: () => void;
  onNextStep?: () => void;
  onPrevStep?: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+S / Cmd+S: 保存
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        onSave?.();
      }

      // Escape: キャンセル
      if (e.key === 'Escape') {
        onCancel?.();
      }

      // Ctrl+ArrowRight / Cmd+ArrowRight: 次のステップ
      if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowRight') {
        e.preventDefault();
        onNextStep?.();
      }

      // Ctrl+ArrowLeft / Cmd+ArrowLeft: 前のステップ
      if ((e.ctrlKey || e.metaKey) && e.key === 'ArrowLeft') {
        e.preventDefault();
        onPrevStep?.();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSave, onCancel, onNextStep, onPrevStep]);
}

// フォーカストラップ（モーダルフォーム向け）
function useFocusTrap(containerRef: React.RefObject<HTMLElement>) {
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    firstElement?.focus();

    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [containerRef]);
}
```

---

## 8. フォームのテスト戦略

### 8.1 テストピラミッド

複雑なフォームのテストは、以下のレイヤーに分けて実施する。

| レイヤー | ツール | テスト対象 | 比率 |
|---------|--------|----------|------|
| Unit Test | Vitest / Jest | スキーマ、バリデーションロジック | 50% |
| Integration Test | Testing Library | フォームコンポーネントの振る舞い | 35% |
| E2E Test | Playwright / Cypress | ユーザーフロー全体 | 15% |

### 8.2 スキーマのユニットテスト

```typescript
import { describe, it, expect } from 'vitest';

describe('step1Schema', () => {
  it('有効なデータを受け付ける', () => {
    const validData = {
      name: '山田太郎',
      email: 'taro@example.com',
      password: 'Password1',
      confirmPassword: 'Password1',
    };

    const result = step1Schema.safeParse(validData);
    expect(result.success).toBe(true);
  });

  it('名前が空の場合エラーになる', () => {
    const data = {
      name: '',
      email: 'taro@example.com',
      password: 'Password1',
      confirmPassword: 'Password1',
    };

    const result = step1Schema.safeParse(data);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.errors[0].path).toContain('name');
      expect(result.error.errors[0].message).toBe('名前は必須です');
    }
  });

  it('パスワードが一致しない場合エラーになる', () => {
    const data = {
      name: '山田太郎',
      email: 'taro@example.com',
      password: 'Password1',
      confirmPassword: 'Password2',
    };

    const result = step1Schema.safeParse(data);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.errors[0].message).toBe('パスワードが一致しません');
    }
  });

  it('パスワードの強度要件を検証する', () => {
    const weakPasswords = [
      { pw: 'short', reason: '8文字未満' },
      { pw: 'alllowercase1', reason: '大文字なし' },
      { pw: 'ALLUPPERCASE1', reason: '小文字なし' },
      { pw: 'NoDigitsHere', reason: '数字なし' },
    ];

    weakPasswords.forEach(({ pw, reason }) => {
      const data = {
        name: '山田太郎',
        email: 'taro@example.com',
        password: pw,
        confirmPassword: pw,
      };

      const result = step1Schema.safeParse(data);
      expect(result.success, `${reason}: "${pw}" は拒否されるべき`).toBe(false);
    });
  });

  it('メールアドレスの形式を検証する', () => {
    const invalidEmails = [
      'not-an-email',
      '@no-local.com',
      'no-domain@',
      'spaces in@email.com',
    ];

    invalidEmails.forEach((email) => {
      const data = {
        name: '山田太郎',
        email,
        password: 'Password1',
        confirmPassword: 'Password1',
      };

      const result = step1Schema.safeParse(data);
      expect(result.success, `"${email}" は拒否されるべき`).toBe(false);
    });
  });
});

describe('orderSchema', () => {
  it('空の商品リストを拒否する', () => {
    const data = {
      customerName: 'テスト顧客',
      customerEmail: 'test@example.com',
      shippingAddress: '東京都',
      items: [],
    };

    const result = orderSchema.safeParse(data);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.errors[0].message).toBe('1つ以上の商品を追加してください');
    }
  });

  it('50商品を超える注文を拒否する', () => {
    const items = Array.from({ length: 51 }, (_, i) => ({
      productId: `prod_${i}`,
      quantity: 1,
      unitPrice: 100,
    }));

    const data = {
      customerName: 'テスト顧客',
      customerEmail: 'test@example.com',
      shippingAddress: '東京都',
      items,
    };

    const result = orderSchema.safeParse(data);
    expect(result.success).toBe(false);
  });
});

describe('notificationSchema (discriminatedUnion)', () => {
  it('email タイプの通知を受け付ける', () => {
    const data = {
      type: 'email' as const,
      email: 'test@example.com',
      frequency: 'daily' as const,
      format: 'html' as const,
      categories: ['news'],
    };

    const result = notificationSchema.safeParse(data);
    expect(result.success).toBe(true);
  });

  it('webhook タイプで短いシークレットを拒否する', () => {
    const data = {
      type: 'webhook' as const,
      url: 'https://example.com/webhook',
      secret: 'short',
      events: ['order.created'],
    };

    const result = notificationSchema.safeParse(data);
    expect(result.success).toBe(false);
  });

  it('未知のタイプを拒否する', () => {
    const data = {
      type: 'unknown',
      email: 'test@example.com',
    };

    const result = notificationSchema.safeParse(data);
    expect(result.success).toBe(false);
  });
});
```

### 8.3 フォームコンポーネントの統合テスト

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

describe('MultiStepForm', () => {
  it('ステップ1からステップ2に進める', async () => {
    const user = userEvent.setup();
    render(<MultiStepForm />);

    // ステップ1のフィールドに入力
    await user.type(screen.getByLabelText('名前'), '山田太郎');
    await user.type(screen.getByLabelText('メールアドレス'), 'taro@example.com');
    await user.type(screen.getByLabelText('パスワード'), 'Password1');
    await user.type(screen.getByLabelText('パスワード確認'), 'Password1');

    // 次へボタンをクリック
    await user.click(screen.getByText('次へ'));

    // ステップ2が表示される
    await waitFor(() => {
      expect(screen.getByLabelText('会社名')).toBeInTheDocument();
    });
  });

  it('バリデーションエラーがあるとステップを進めない', async () => {
    const user = userEvent.setup();
    render(<MultiStepForm />);

    // 何も入力せずに次へをクリック
    await user.click(screen.getByText('次へ'));

    // エラーメッセージが表示される
    await waitFor(() => {
      expect(screen.getByText('名前は必須です')).toBeInTheDocument();
    });

    // ステップ1のままである
    expect(screen.getByLabelText('名前')).toBeInTheDocument();
  });

  it('戻るボタンで前のステップに戻れる', async () => {
    const user = userEvent.setup();
    render(<MultiStepForm />);

    // ステップ1を入力して進む
    await user.type(screen.getByLabelText('名前'), '山田太郎');
    await user.type(screen.getByLabelText('メールアドレス'), 'taro@example.com');
    await user.type(screen.getByLabelText('パスワード'), 'Password1');
    await user.type(screen.getByLabelText('パスワード確認'), 'Password1');
    await user.click(screen.getByText('次へ'));

    // ステップ2に到達
    await waitFor(() => {
      expect(screen.getByLabelText('会社名')).toBeInTheDocument();
    });

    // 戻るボタンをクリック
    await user.click(screen.getByText('戻る'));

    // ステップ1に戻り、入力値が保持されている
    await waitFor(() => {
      expect(screen.getByLabelText('名前')).toHaveValue('山田太郎');
    });
  });
});

describe('OrderForm', () => {
  const mockProducts: Product[] = [
    { id: 'prod_1', name: 'Widget A', price: 1000, stock: 100, category: 'widget' },
    { id: 'prod_2', name: 'Widget B', price: 2000, stock: 50, category: 'widget' },
  ];

  it('商品行を追加できる', async () => {
    const user = userEvent.setup();
    render(<OrderForm products={mockProducts} />);

    // 追加ボタンをクリック
    await user.click(screen.getByText('+ 商品を追加'));

    // 2行に増える
    const productSelects = screen.getAllByRole('combobox');
    expect(productSelects.length).toBeGreaterThan(1);
  });

  it('商品行を削除できる', async () => {
    const user = userEvent.setup();
    render(<OrderForm products={mockProducts} />);

    // 2行追加
    await user.click(screen.getByText('+ 商品を追加'));

    // 削除ボタンをクリック
    const deleteButtons = screen.getAllByTitle('削除');
    await user.click(deleteButtons[0]);

    // 1行に戻る
    await waitFor(() => {
      expect(screen.queryAllByTitle('削除')).toHaveLength(0);
    });
  });

  it('最後の1行は削除できない', () => {
    render(<OrderForm products={mockProducts} />);

    // 削除ボタンが表示されない
    expect(screen.queryByTitle('削除')).not.toBeInTheDocument();
  });
});
```

### 8.4 E2E テスト

```typescript
import { test, expect } from '@playwright/test';

test.describe('ユーザー登録フロー', () => {
  test('全ステップを完了して登録できる', async ({ page }) => {
    await page.goto('/register');

    // ステップ1: アカウント情報
    await page.fill('[name="name"]', '山田太郎');
    await page.fill('[name="email"]', 'taro@example.com');
    await page.fill('[name="password"]', 'Password1');
    await page.fill('[name="confirmPassword"]', 'Password1');
    await page.click('button:text("次へ")');

    // ステップ2: プロフィール
    await expect(page.locator('[name="company"]')).toBeVisible();
    await page.fill('[name="company"]', '株式会社テスト');
    await page.selectOption('[name="role"]', 'developer');
    await page.fill('[name="experience"]', '5');
    await page.click('button:text("次へ")');

    // ステップ3: プラン選択
    await expect(page.locator('[name="plan"]')).toBeVisible();
    await page.click('label:text("プロプラン")');
    await page.check('[name="agreed"]');
    await page.click('button:text("登録する")');

    // 完了ページにリダイレクト
    await expect(page).toHaveURL('/registration/complete');
    await expect(page.locator('h1')).toContainText('登録完了');
  });

  test('バリデーションエラー時にエラーメッセージが表示される', async ({ page }) => {
    await page.goto('/register');

    // 何も入力せずに次へ
    await page.click('button:text("次へ")');

    // エラーメッセージが表示される
    await expect(page.locator('text=名前は必須です')).toBeVisible();
    await expect(page.locator('text=有効なメールアドレスを入力してください')).toBeVisible();
  });

  test('ページ離脱時に確認ダイアログが表示される', async ({ page }) => {
    await page.goto('/register');

    // フォームに入力
    await page.fill('[name="name"]', '山田太郎');

    // ページを離れようとする
    page.on('dialog', async (dialog) => {
      expect(dialog.type()).toBe('beforeunload');
      await dialog.accept();
    });

    await page.goto('/');
  });
});
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `watch` の値が undefined になる | `defaultValues` が設定されていない | `useForm` に `defaultValues` を必ず渡す |
| `useFieldArray` の行が重複する | `key` に `index` を使用している | `field.id` を `key` に使用する |
| 条件分岐フォームで古い値が残る | `shouldUnregister` の設定不備 | `shouldUnregister: true` を設定するか、タイプ変更時に `reset` する |
| `resolver` の変更が反映されない | `resolver` がステップ変更時に再評価されない | ステップごとに `resolver` を動的に切り替える |
| フォーム送信後にバリデーションが走らない | `mode` が `onSubmit` のまま | 送信後は `mode: 'onChange'` に切り替えるか、`trigger()` を手動呼出し |
| 非制御コンポーネントで値が反映されない | `register` を使用していない | `Controller` を使用するか、`setValue` で手動設定 |
| `defaultValues` の変更が反映されない | フォーム初期化後に `defaultValues` が変わった | `reset(newDefaultValues)` を呼ぶ |
| 大量フィールドでフォームが遅い | 全フィールドが再レンダリング | `React.memo` と `useWatch` で最適化 |

### 9.2 デバッグツール

```typescript
// フォーム状態のデバッグコンポーネント（開発時のみ使用）
function FormDevTools<T extends Record<string, any>>({
  form,
}: {
  form: ReturnType<typeof useForm<T>>;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const values = useWatch({ control: form.control });
  const { errors, dirtyFields, touchedFields, isValid, isDirty, isSubmitting } =
    form.formState;

  if (process.env.NODE_ENV === 'production') return null;

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setIsOpen(o => !o)}
        className="bg-gray-900 text-white px-3 py-1 rounded-full text-xs"
      >
        {isOpen ? 'DevTools [x]' : 'DevTools [o]'}
      </button>

      {isOpen && (
        <div className="absolute bottom-10 right-0 w-96 max-h-96 overflow-auto
          bg-gray-900 text-green-400 rounded-lg p-4 text-xs font-mono shadow-xl">
          <h4 className="text-white font-bold mb-2">Form State</h4>

          <div className="space-y-2">
            <div>
              <span className="text-gray-400">isValid:</span>{' '}
              <span className={isValid ? 'text-green-400' : 'text-red-400'}>
                {String(isValid)}
              </span>
            </div>
            <div>
              <span className="text-gray-400">isDirty:</span>{' '}
              {String(isDirty)}
            </div>
            <div>
              <span className="text-gray-400">isSubmitting:</span>{' '}
              {String(isSubmitting)}
            </div>
          </div>

          <h4 className="text-white font-bold mt-4 mb-2">Values</h4>
          <pre className="whitespace-pre-wrap">
            {JSON.stringify(values, null, 2)}
          </pre>

          {Object.keys(errors).length > 0 && (
            <>
              <h4 className="text-red-400 font-bold mt-4 mb-2">Errors</h4>
              <pre className="whitespace-pre-wrap text-red-400">
                {JSON.stringify(
                  Object.fromEntries(
                    Object.entries(errors).map(([key, val]: [string, any]) => [
                      key,
                      val?.message ?? val,
                    ])
                  ),
                  null,
                  2
                )}
              </pre>
            </>
          )}

          <h4 className="text-white font-bold mt-4 mb-2">Dirty Fields</h4>
          <pre className="whitespace-pre-wrap text-yellow-400">
            {JSON.stringify(dirtyFields, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
```

### 9.3 エラーハンドリングのベストプラクティス

```typescript
// グローバルエラーハンドリング
function useFormErrorHandler() {
  const handleFormError = useCallback((error: unknown) => {
    // Zod バリデーションエラー
    if (error instanceof z.ZodError) {
      const messages = error.errors.map(e =>
        `${e.path.join('.')}: ${e.message}`
      );
      toast.error(`バリデーションエラー:\n${messages.join('\n')}`);
      return;
    }

    // API エラー
    if (error instanceof ApiError) {
      switch (error.status) {
        case 400:
          toast.error('入力内容に誤りがあります。確認してください。');
          break;
        case 409:
          toast.error('このメールアドレスは既に登録されています。');
          break;
        case 422:
          // サーバーサイドバリデーションエラー
          if (error.fieldErrors) {
            // フォームにエラーを反映
            Object.entries(error.fieldErrors).forEach(([field, message]) => {
              form.setError(field as any, {
                type: 'server',
                message: message as string,
              });
            });
          }
          break;
        case 429:
          toast.error('リクエストが多すぎます。しばらくしてから再試行してください。');
          break;
        case 500:
          toast.error('サーバーエラーが発生しました。しばらくしてから再試行してください。');
          break;
        default:
          toast.error('予期しないエラーが発生しました。');
      }
      return;
    }

    // ネットワークエラー
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
      toast.error('ネットワークエラー: インターネット接続を確認してください。');
      return;
    }

    // その他のエラー
    console.error('Unhandled form error:', error);
    toast.error('予期しないエラーが発生しました。');
  }, []);

  return { handleFormError };
}

// サーバーサイドバリデーションエラーの統合
async function submitWithServerValidation<T>(
  form: ReturnType<typeof useForm<T>>,
  data: T,
  submitFn: (data: T) => Promise<void>
) {
  try {
    await submitFn(data);
  } catch (error) {
    if (error instanceof ApiError && error.fieldErrors) {
      // サーバーからのフィールドエラーをフォームに反映
      Object.entries(error.fieldErrors).forEach(([field, message]) => {
        form.setError(field as any, {
          type: 'server',
          message: message as string,
        });
      });

      // 最初のエラーフィールドにフォーカス
      const firstErrorField = Object.keys(error.fieldErrors)[0];
      const element = document.querySelector(`[name="${firstErrorField}"]`);
      if (element instanceof HTMLElement) {
        element.focus();
      }
    } else {
      throw error;
    }
  }
}
```

---

## まとめ

### 複雑フォームパターンの全体像

| パターン | 用途 | 主要ツール | 難易度 |
|---------|------|----------|--------|
| マルチステップ | ウィザード形式の登録フロー | useForm + useState | 中 |
| useFieldArray | 動的な配列フィールド | useFieldArray | 中 |
| ネスト配列 | 請求書の明細行（セクション構造） | ネストした useFieldArray | 高 |
| discriminatedUnion | 条件分岐バリデーション | Zod discriminatedUnion | 中 |
| superRefine | 複雑な条件バリデーション | Zod superRefine | 高 |
| カスケード選択 | 親子関係の連動セレクト | watch + useEffect | 中 |
| 自動保存 | 下書き保存 | useWatch + debounce | 低〜中 |
| ドラフト復元 | 途中離脱からの復帰 | localStorage / API | 中 |
| 離脱防止 | 未保存データの保護 | beforeunload / useBlocker | 低 |
| 仮想化 | 大量フィールドの最適化 | react-window | 高 |
| ドラッグ&ドロップ | フィールドの並び替え | dnd-kit + useFieldArray.move | 高 |

### 設計判断のフローチャート

```
フォームにどのパターンが必要か?

1. フィールドが多い（10個以上）?
   → YES: マルチステップフォームを検討
   → NO: シングルページフォーム

2. 動的にフィールドを追加/削除する?
   → YES: useFieldArray を使用
   → NO: 静的なフォーム

3. 選択に応じてフィールドが変わる?
   → YES: discriminatedUnion または superRefine
   → NO: 固定フィールド

4. 入力途中のデータを保護する必要がある?
   → YES: 自動保存 + 離脱防止
   → NO: 送信時のみ処理

5. フィールド数が100を超える?
   → YES: 仮想化（react-window）を検討
   → NO: 通常のレンダリング
```

### ベストプラクティスチェックリスト

- [ ] 全フィールドに `defaultValues` を設定している
- [ ] `useFieldArray` で `field.id` を key に使用している
- [ ] エラーメッセージが日本語で分かりやすく設定されている
- [ ] 必須フィールドが視覚的に区別できる
- [ ] フォーカス管理（エラー時の自動フォーカス）が実装されている
- [ ] `aria-invalid` と `aria-describedby` が設定されている
- [ ] 大量フィールドの場合、パフォーマンス最適化が行われている
- [ ] マルチステップの場合、進捗インジケーターがある
- [ ] 未保存データの離脱防止が実装されている
- [ ] サーバーサイドバリデーションエラーがフォームに反映される
- [ ] テスト（ユニット・統合・E2E）が適切に書かれている
- [ ] TypeScript の型安全性が確保されている

---

## よくある質問（FAQ）

### Q1. 動的フィールドの追加・削除はどう実装すべきですか？

**A:** React Hook Form の **`useFieldArray`** を使うのが最も堅牢である。手動で配列を管理すると、キー管理やバリデーションの同期が複雑になる。

**基本パターン:**

```typescript
import { useForm, useFieldArray } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';

const schema = z.object({
  members: z.array(
    z.object({
      name: z.string().min(1, '名前は必須です'),
      email: z.string().email('有効なメールアドレスを入力してください'),
    })
  ).min(1, '少なくとも1人のメンバーが必要です'),
});

type FormData = z.infer<typeof schema>;

function DynamicFieldForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: {
      members: [{ name: '', email: '' }],
    },
  });

  const { fields, append, remove } = useFieldArray({
    control: form.control,
    name: 'members',
  });

  return (
    <form onSubmit={form.handleSubmit((data) => console.log(data))}>
      {fields.map((field, index) => (
        <div key={field.id}>
          <input {...form.register(`members.${index}.name`)} placeholder="名前" />
          <input {...form.register(`members.${index}.email`)} placeholder="メール" />
          <button type="button" onClick={() => remove(index)}>削除</button>
        </div>
      ))}

      <button type="button" onClick={() => append({ name: '', email: '' })}>
        メンバーを追加
      </button>

      <button type="submit">送信</button>
    </form>
  );
}
```

**重要なポイント:**

1. **`field.id` をキーに使う**: `fields.map((field, index) => <div key={field.id}>)` とすること。`index` をキーにすると、削除時に誤ったフィールドがバリデーションされる
2. **デフォルト値を設定**: `defaultValues` で初期配列を指定する
3. **Zodスキーマで最小・最大数を制御**: `.min(1)` や `.max(10)` で配列の長さを検証
4. **削除前に確認**: ユーザーが誤って削除しないよう、確認ダイアログを表示する

**複雑なケース: ネストした動的フィールド**

```typescript
const schema = z.object({
  teams: z.array(
    z.object({
      teamName: z.string(),
      members: z.array(
        z.object({
          name: z.string(),
          role: z.string(),
        })
      ),
    })
  ),
});

function NestedDynamicFields() {
  const form = useForm<FormData>({ resolver: zodResolver(schema) });
  const { fields: teamFields, append: appendTeam } = useFieldArray({
    control: form.control,
    name: 'teams',
  });

  return (
    <form>
      {teamFields.map((team, teamIndex) => (
        <div key={team.id}>
          <input {...form.register(`teams.${teamIndex}.teamName`)} />

          <NestedMembers teamIndex={teamIndex} control={form.control} />

          <button type="button" onClick={() => appendTeam({ teamName: '', members: [] })}>
            チームを追加
          </button>
        </div>
      ))}
    </form>
  );
}

function NestedMembers({ teamIndex, control }) {
  const { fields, append, remove } = useFieldArray({
    control,
    name: `teams.${teamIndex}.members`,
  });

  return (
    <>
      {fields.map((member, memberIndex) => (
        <div key={member.id}>
          <input {...form.register(`teams.${teamIndex}.members.${memberIndex}.name`)} />
          <button onClick={() => remove(memberIndex)}>削除</button>
        </div>
      ))}
      <button onClick={() => append({ name: '', role: '' })}>メンバー追加</button>
    </>
  );
}
```

### Q2. フォームのパフォーマンス最適化はどうすべきですか？

**A:** 大規模フォーム（50フィールド以上）では、以下の最適化が必須:

**1. React Hook Form の mode 設定**

```typescript
const form = useForm({
  mode: 'onBlur', // デフォルトは 'onChange'（入力中に毎回バリデーション）
  // onBlur: フォーカスアウト時にバリデーション（推奨）
  // onSubmit: 送信時のみバリデーション（最もパフォーマンス良好）
});
```

**2. 不要な再レンダリングを防ぐ**

```typescript
// ❌ 悪い例: フォーム全体が再レンダリングされる
const { watch } = useForm();
const allValues = watch(); // 全フィールドを監視

// ✅ 良い例: 必要なフィールドだけ監視
const email = watch('email');
```

**3. コンポーネント分割**

```typescript
// ❌ 悪い例: 1つの巨大なコンポーネント
function MassiveForm() {
  const form = useForm();
  return (
    <form>
      {/* 100個のinputが全て再レンダリング */}
      <input {...form.register('field1')} />
      <input {...form.register('field2')} />
      {/* ... */}
    </form>
  );
}

// ✅ 良い例: セクションごとに分割
function OptimizedForm() {
  const form = useForm();
  return (
    <FormProvider {...form}>
      <PersonalInfoSection />
      <AddressSection />
      <PaymentSection />
    </FormProvider>
  );
}

function PersonalInfoSection() {
  const { register } = useFormContext();
  return (
    <div>
      <input {...register('name')} />
      <input {...register('email')} />
    </div>
  );
}
```

**4. React.memo で不要な再レンダリングを防止**

```typescript
const FormField = React.memo(({ name, label }: { name: string; label: string }) => {
  const { register } = useFormContext();
  return (
    <div>
      <label>{label}</label>
      <input {...register(name)} />
    </div>
  );
});
```

**5. useFieldArray のパフォーマンス最適化**

```typescript
// 大量の動的フィールド（100個以上）がある場合
const { fields } = useFieldArray({ name: 'items' });

// react-window で仮想スクロール
import { FixedSizeList } from 'react-window';

<FixedSizeList
  height={600}
  itemCount={fields.length}
  itemSize={50}
>
  {({ index, style }) => (
    <div style={style}>
      <input {...register(`items.${index}.name`)} />
    </div>
  )}
</FixedSizeList>
```

**6. Zod スキーマの最適化**

```typescript
// ❌ 悪い例: 複雑な正規表現やカスタムバリデーション
const schema = z.object({
  email: z.string().refine(async (val) => {
    // 非同期バリデーションが毎回走る
    const exists = await checkEmailExists(val);
    return !exists;
  }),
});

// ✅ 良い例: 基本的なバリデーションはZod、非同期はonBlurで
const schema = z.object({
  email: z.string().email(), // シンプルなバリデーション
});

<input
  {...form.register('email', {
    onBlur: async (e) => {
      // フォーカスアウト時のみ非同期チェック
      const exists = await checkEmailExists(e.target.value);
      if (exists) form.setError('email', { message: '既に使用されています' });
    },
  })}
/>
```

### Q3. Server Actions でのフォーム処理はどうすべきですか？

**A:** Next.js 14以降では **Server Actions** がフォーム送信の標準パターンである:

**基本パターン:**

```typescript
// app/actions/user.ts
'use server';

import { z } from 'zod';

const userSchema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

export async function createUser(formData: FormData) {
  const rawData = {
    name: formData.get('name'),
    email: formData.get('email'),
  };

  // バリデーション
  const result = userSchema.safeParse(rawData);
  if (!result.success) {
    return { errors: result.error.flatten() };
  }

  // データベース保存
  await db.user.create({ data: result.data });

  return { success: true };
}

// app/page.tsx
import { createUser } from './actions/user';

export default function Page() {
  return (
    <form action={createUser}>
      <input name="name" />
      <input name="email" />
      <button type="submit">送信</button>
    </form>
  );
}
```

**React Hook Form + Server Actions の統合（推奨）:**

```typescript
'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { createUser } from './actions/user';
import { z } from 'zod';

const schema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

type FormData = z.infer<typeof schema>;

export default function UserForm() {
  const form = useForm<FormData>({
    resolver: zodResolver(schema),
  });

  const onSubmit = async (data: FormData) => {
    const formData = new FormData();
    formData.append('name', data.name);
    formData.append('email', data.email);

    const result = await createUser(formData);

    if (result.errors) {
      // サーバー側のバリデーションエラーをフォームに反映
      Object.entries(result.errors.fieldErrors).forEach(([field, errors]) => {
        form.setError(field as keyof FormData, {
          message: errors?.[0],
        });
      });
    }
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input {...form.register('name')} />
      {form.formState.errors.name && <p>{form.formState.errors.name.message}</p>}

      <input {...form.register('email')} />
      {form.formState.errors.email && <p>{form.formState.errors.email.message}</p>}

      <button type="submit">送信</button>
    </form>
  );
}
```

**useActionState を使った楽観的更新:**

```typescript
'use client';

import { useActionState } from 'react';
import { createUser } from './actions/user';

export default function UserForm() {
  const [state, formAction, isPending] = useActionState(createUser, null);

  return (
    <form action={formAction}>
      <input name="name" />
      {state?.errors?.name && <p>{state.errors.name[0]}</p>}

      <input name="email" />
      {state?.errors?.email && <p>{state.errors.email[0]}</p>}

      <button type="submit" disabled={isPending}>
        {isPending ? '送信中...' : '送信'}
      </button>
    </form>
  );
}
```

**ベストプラクティス:**

1. **クライアント・サーバー二重バリデーション**: クライアントはUX向上、サーバーはセキュリティ保証
2. **同じZodスキーマを共有**: モノレポやパッケージ共有でスキーマを一元管理
3. **楽観的更新**: `useOptimistic` でUI即座に更新、エラー時にロールバック
4. **revalidatePath**: Server Actionsでデータ更新後、キャッシュを無効化

---

## 次に読むべきガイド
→ [[00-deployment-platforms.md]] -- デプロイ先

---

## 参考文献
1. React Hook Form. "useFieldArray." react-hook-form.com, 2024.
2. React Hook Form. "Performance Optimization." react-hook-form.com, 2024.
3. Zod. "Discriminated Unions." zod.dev, 2024.
4. Zod. "superRefine." zod.dev, 2024.
5. W3C. "Web Content Accessibility Guidelines (WCAG) 2.1." w3.org, 2018.
6. WAI-ARIA. "ARIA Authoring Practices Guide - Forms." w3.org, 2024.
7. Testing Library. "React Testing Library - User Event." testing-library.com, 2024.
8. Playwright. "Test Generator." playwright.dev, 2024.
9. @dnd-kit. "Sortable." dndkit.com, 2024.
10. react-window. "Windowed Rendering." react-window.vercel.app, 2024.
