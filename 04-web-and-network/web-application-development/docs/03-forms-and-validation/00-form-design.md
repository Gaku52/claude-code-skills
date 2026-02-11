# フォーム設計

> フォームはユーザーとの主要なインタラクションポイント。React Hook Form、制御/非制御コンポーネント、パフォーマンス最適化、アクセシビリティまで、使いやすく保守しやすいフォーム設計のベストプラクティスを習得する。

## この章で学ぶこと

- [ ] React Hook Formの基本パターンを理解する
- [ ] 制御/非制御コンポーネントの使い分けを把握する
- [ ] フォームのUXとアクセシビリティを学ぶ

---

## 1. React Hook Form

```typescript
// React Hook Form: パフォーマンス最適化されたフォームライブラリ
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// スキーマ定義
const userSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  email: z.string().email('Invalid email'),
  age: z.coerce.number().min(0).max(150).optional(),
  role: z.enum(['user', 'admin']),
  agreed: z.literal(true, {
    errorMap: () => ({ message: 'You must agree to the terms' }),
  }),
});

type UserFormData = z.infer<typeof userSchema>;

function CreateUserForm() {
  const {
    register,       // input を登録
    handleSubmit,   // フォーム送信ハンドラ
    formState: { errors, isSubmitting, isDirty },
    reset,          // フォームリセット
    watch,          // 値の監視
    setValue,       // プログラマティックに値を設定
  } = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
    defaultValues: {
      name: '',
      email: '',
      role: 'user',
      agreed: false as any,
    },
  });

  const onSubmit = async (data: UserFormData) => {
    await api.users.create(data);
    reset(); // フォームリセット
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      <div>
        <label htmlFor="name">Name</label>
        <input
          id="name"
          {...register('name')}
          aria-invalid={!!errors.name}
          aria-describedby={errors.name ? 'name-error' : undefined}
        />
        {errors.name && (
          <p id="name-error" className="text-red-500" role="alert">
            {errors.name.message}
          </p>
        )}
      </div>

      <div>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" {...register('email')} />
        {errors.email && <p className="text-red-500">{errors.email.message}</p>}
      </div>

      <div>
        <label htmlFor="role">Role</label>
        <select id="role" {...register('role')}>
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
      </div>

      <div>
        <label>
          <input type="checkbox" {...register('agreed')} />
          I agree to the terms
        </label>
        {errors.agreed && <p className="text-red-500">{errors.agreed.message}</p>}
      </div>

      <button type="submit" disabled={isSubmitting || !isDirty}>
        {isSubmitting ? 'Creating...' : 'Create User'}
      </button>
    </form>
  );
}
```

---

## 2. 制御 / 非制御コンポーネント

```
非制御コンポーネント（Uncontrolled）:
  → DOMが値を管理
  → register() で ref を登録
  → パフォーマンスが良い（再レンダリングなし）

  <input {...register('name')} />

制御コンポーネント（Controlled）:
  → React が値を管理
  → Controller で React Hook Form と統合
  → カスタムUIコンポーネントに必要

  <Controller
    name="date"
    control={control}
    render={({ field }) => (
      <DatePicker
        value={field.value}
        onChange={field.onChange}
        onBlur={field.onBlur}
      />
    )}
  />

使い分け:
  非制御: ネイティブ input, select, textarea
  制御: カスタムUI、日付ピッカー、リッチエディタ、shadcn/ui コンポーネント
```

```typescript
// Controller の使用例
import { Controller, useForm } from 'react-hook-form';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/shared/components/ui/select';

function UserForm() {
  const { control, handleSubmit } = useForm();

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <Controller
        name="role"
        control={control}
        render={({ field }) => (
          <Select onValueChange={field.onChange} defaultValue={field.value}>
            <SelectTrigger>
              <SelectValue placeholder="Select role" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="user">User</SelectItem>
              <SelectItem value="admin">Admin</SelectItem>
            </SelectContent>
          </Select>
        )}
      />
    </form>
  );
}
```

---

## 3. Server Actions との統合

```typescript
// React Hook Form + Server Actions
'use client';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { createUser } from './actions';

function CreateUserForm() {
  const form = useForm<UserFormData>({
    resolver: zodResolver(userSchema),
  });

  return (
    <form
      action={async (formData) => {
        // クライアントサイドバリデーション
        const valid = await form.trigger();
        if (!valid) return;

        // Server Action 実行
        const result = await createUser(formData);

        if (result?.errors) {
          // サーバーサイドエラーをフォームに反映
          for (const [field, message] of Object.entries(result.errors)) {
            form.setError(field as any, { message: message[0] });
          }
        }
      }}
    >
      {/* フォームフィールド */}
    </form>
  );
}
```

---

## 4. フォームUXのベストプラクティス

```
UX原則:
  ✓ エラーは送信時 + その後はリアルタイム（onBlur / onChange）
  ✓ 送信ボタンは isDirty かつ isValid の時のみ有効
  ✓ 送信中はローディング表示 + ダブルサブミット防止
  ✓ 成功時は明確なフィードバック（トースト、リダイレクト）
  ✓ 未保存の変更がある場合はページ離脱時に警告

  ✗ フォームを開いた瞬間にエラーを表示しない
  ✗ エラーメッセージを赤字だけに頼らない（アイコンも）
  ✗ 送信ボタンを隠さない（disabled にする）

アクセシビリティ:
  ✓ label と input を htmlFor / id で紐付け
  ✓ aria-invalid で invalid 状態を伝える
  ✓ aria-describedby でエラーメッセージを紐付け
  ✓ role="alert" でエラーをスクリーンリーダーに通知
  ✓ Tab キーでフィールド間を移動可能
  ✓ Enter キーでフォーム送信
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| React Hook Form | register + zodResolver |
| 非制御 | ネイティブ要素、高パフォーマンス |
| 制御 | Controller、カスタムUI |
| Server Actions | プログレッシブエンハンスメント |
| UX | 送信後リアルタイム検証、a11y対応 |

---

## 次に読むべきガイド
→ [[01-validation-patterns.md]] — バリデーション

---

## 参考文献
1. React Hook Form. "Documentation." react-hook-form.com, 2024.
2. shadcn/ui. "Form." ui.shadcn.com, 2024.
3. web.dev. "Form Best Practices." web.dev, 2024.
