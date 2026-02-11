# バリデーションパターン

> バリデーションはフォームの品質を決定づける。Zod統合、リアルタイム検証、非同期バリデーション、クライアント/サーバー二重検証まで、堅牢で使いやすいバリデーション設計の全パターンを習得する。

## この章で学ぶこと

- [ ] Zod + React Hook Formの統合パターンを理解する
- [ ] リアルタイム・非同期バリデーションの実装を把握する
- [ ] クライアント/サーバー二重検証の設計を学ぶ

---

## 1. Zodスキーマ設計

```typescript
import { z } from 'zod';

// ユーザー登録フォーム
const registerSchema = z.object({
  username: z.string()
    .min(3, '3文字以上で入力してください')
    .max(20, '20文字以下で入力してください')
    .regex(/^[a-zA-Z0-9_]+$/, '英数字とアンダースコアのみ使用可能'),

  email: z.string()
    .email('有効なメールアドレスを入力してください'),

  password: z.string()
    .min(8, '8文字以上で入力してください')
    .regex(/[A-Z]/, '大文字を1文字以上含めてください')
    .regex(/[a-z]/, '小文字を1文字以上含めてください')
    .regex(/[0-9]/, '数字を1文字以上含めてください'),

  confirmPassword: z.string(),

  birthDate: z.coerce.date()
    .max(new Date(), '未来の日付は指定できません')
    .refine(
      (date) => {
        const age = Math.floor((Date.now() - date.getTime()) / (365.25 * 24 * 60 * 60 * 1000));
        return age >= 13;
      },
      '13歳以上である必要があります'
    ),

  website: z.string().url('有効なURLを入力してください').optional().or(z.literal('')),

}).refine(
  (data) => data.password === data.confirmPassword,
  { message: 'パスワードが一致しません', path: ['confirmPassword'] }
);

// スキーマの再利用
const baseUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
});

const createUserSchema = baseUserSchema.extend({
  password: z.string().min(8),
});

const updateUserSchema = baseUserSchema.partial(); // 全フィールドがオプショナルに
```

---

## 2. バリデーションのタイミング

```
バリデーション戦略:

  ① Submit時のみ（デフォルト）:
     mode: 'onSubmit'
     → 送信ボタン押下時にバリデーション
     → UX: 最もシンプル

  ② Submit後はリアルタイム（推奨）:
     mode: 'onSubmit' + reValidateMode: 'onChange'
     → 初回は送信時にバリデーション
     → エラーが出たフィールドはリアルタイムでバリデーション
     → UX: バランスが良い

  ③ フォーカスアウト時:
     mode: 'onBlur'
     → フィールドを離れた時にバリデーション
     → UX: 入力中は邪魔しない

  ④ リアルタイム:
     mode: 'onChange'
     → 入力するたびにバリデーション
     → UX: 即座にフィードバック、パフォーマンス注意
```

```typescript
const form = useForm<RegisterData>({
  resolver: zodResolver(registerSchema),
  mode: 'onSubmit',           // 初回は送信時
  reValidateMode: 'onChange', // 再検証はリアルタイム
  defaultValues: {
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  },
});
```

---

## 3. 非同期バリデーション

```typescript
// メールアドレスの重複チェック（非同期）
const schema = z.object({
  email: z.string().email(),
});

function RegisterForm() {
  const form = useForm({
    resolver: zodResolver(schema),
  });

  // 非同期バリデーション（register のバリデーション）
  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input
        {...form.register('email', {
          validate: async (value) => {
            if (!value) return true;
            // デバウンスは useCallback + setTimeout で実装
            const exists = await checkEmailExists(value);
            return exists ? 'このメールアドレスは既に使用されています' : true;
          },
        })}
      />
    </form>
  );
}

// デバウンス付き非同期バリデーション
function useAsyncValidation(
  validateFn: (value: string) => Promise<string | true>,
  delay = 500
) {
  const [error, setError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout>();

  const validate = useCallback((value: string) => {
    clearTimeout(timeoutRef.current);
    setIsValidating(true);

    timeoutRef.current = setTimeout(async () => {
      const result = await validateFn(value);
      setError(result === true ? null : result);
      setIsValidating(false);
    }, delay);
  }, [validateFn, delay]);

  return { error, isValidating, validate };
}
```

---

## 4. クライアント/サーバー二重検証

```typescript
// 共有スキーマ（クライアント & サーバーで同じスキーマ）
// shared/schemas/user.ts
export const createUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(['user', 'admin']),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;

// クライアント側: React Hook Form + Zod
'use client';
import { createUserSchema } from '@shared/schemas/user';

function CreateUserForm() {
  const form = useForm<CreateUserInput>({
    resolver: zodResolver(createUserSchema),
  });
  // ...
}

// サーバー側: Server Action + 同じZodスキーマ
'use server';
import { createUserSchema } from '@shared/schemas/user';

export async function createUser(formData: FormData) {
  // サーバーでも同じスキーマでバリデーション
  const parsed = createUserSchema.safeParse({
    name: formData.get('name'),
    email: formData.get('email'),
    role: formData.get('role'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  // サーバー固有のバリデーション（DB制約チェック等）
  const exists = await prisma.user.findUnique({
    where: { email: parsed.data.email },
  });
  if (exists) {
    return { errors: { email: ['このメールアドレスは既に使用されています'] } };
  }

  await prisma.user.create({ data: parsed.data });
  revalidatePath('/users');
  redirect('/users');
}
```

---

## 5. パスワード強度インジケーター

```typescript
// パスワード強度の計算
function getPasswordStrength(password: string): {
  score: number; // 0-4
  label: string;
  color: string;
} {
  let score = 0;
  if (password.length >= 8) score++;
  if (password.length >= 12) score++;
  if (/[A-Z]/.test(password) && /[a-z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;
  score = Math.min(score, 4);

  const labels = ['Very Weak', 'Weak', 'Fair', 'Strong', 'Very Strong'];
  const colors = ['bg-red-500', 'bg-orange-500', 'bg-yellow-500', 'bg-green-400', 'bg-green-600'];

  return { score, label: labels[score], color: colors[score] };
}

function PasswordStrengthBar({ password }: { password: string }) {
  const { score, label, color } = getPasswordStrength(password);

  return (
    <div>
      <div className="flex gap-1 h-1">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className={`flex-1 rounded ${i <= score - 1 ? color : 'bg-gray-200'}`}
          />
        ))}
      </div>
      <p className="text-xs mt-1 text-gray-500">{label}</p>
    </div>
  );
}
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| Zod + React Hook Form | 型安全なバリデーション |
| 二重検証 | クライアント + サーバーで同じスキーマ |
| 非同期検証 | メール重複チェック等 |
| Submit後リアルタイム | 推奨するUXバランス |

---

## 次に読むべきガイド
→ [[02-file-upload.md]] — ファイルアップロード

---

## 参考文献
1. React Hook Form. "Resolvers." react-hook-form.com, 2024.
2. Zod. "Documentation." zod.dev, 2024.
