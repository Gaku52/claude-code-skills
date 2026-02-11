# 複雑なフォーム

> 複雑なフォームは実務で避けて通れない課題。マルチステップフォーム、動的フィールド、条件分岐、配列フィールド、ネストしたフォームまで、あらゆる複雑なフォーム要件に対応するパターンを習得する。

## この章で学ぶこと

- [ ] マルチステップフォームの設計と実装を理解する
- [ ] useFieldArray による動的フィールドの管理を把握する
- [ ] 条件分岐フォームのバリデーション設計を学ぶ

---

## 1. マルチステップフォーム

```typescript
// ステップ別スキーマ
const step1Schema = z.object({
  name: z.string().min(1),
  email: z.string().email(),
});

const step2Schema = z.object({
  company: z.string().min(1),
  role: z.enum(['developer', 'designer', 'manager']),
});

const step3Schema = z.object({
  plan: z.enum(['free', 'pro', 'enterprise']),
  agreed: z.literal(true),
});

const fullSchema = step1Schema.merge(step2Schema).merge(step3Schema);
type FormData = z.infer<typeof fullSchema>;

const schemas = [step1Schema, step2Schema, step3Schema];

function MultiStepForm() {
  const [step, setStep] = useState(0);
  const form = useForm<FormData>({
    resolver: zodResolver(schemas[step]),
    mode: 'onSubmit',
  });

  const next = async () => {
    const valid = await form.trigger();
    if (valid) setStep(s => s + 1);
  };

  const prev = () => setStep(s => s - 1);

  const onSubmit = async (data: FormData) => {
    // 最終送信時は全スキーマでバリデーション
    const result = fullSchema.safeParse(data);
    if (!result.success) return;
    await api.register(result.data);
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {/* ステップインジケーター */}
      <div className="flex gap-2 mb-8">
        {['Account', 'Profile', 'Plan'].map((label, i) => (
          <div key={i} className={`flex-1 h-1 rounded ${i <= step ? 'bg-blue-500' : 'bg-gray-200'}`} />
        ))}
      </div>

      {step === 0 && <Step1 form={form} />}
      {step === 1 && <Step2 form={form} />}
      {step === 2 && <Step3 form={form} />}

      <div className="flex justify-between mt-6">
        {step > 0 && <button type="button" onClick={prev}>Back</button>}
        {step < 2 ? (
          <button type="button" onClick={next}>Next</button>
        ) : (
          <button type="submit">Submit</button>
        )}
      </div>
    </form>
  );
}
```

---

## 2. 動的フィールド（useFieldArray）

```typescript
// 注文フォーム: 商品の追加/削除
import { useFieldArray, useForm } from 'react-hook-form';

const orderSchema = z.object({
  customerName: z.string().min(1),
  items: z.array(z.object({
    productId: z.string().min(1),
    quantity: z.coerce.number().min(1).max(999),
    note: z.string().optional(),
  })).min(1, '1つ以上の商品を追加してください'),
});

function OrderForm() {
  const form = useForm({
    resolver: zodResolver(orderSchema),
    defaultValues: {
      customerName: '',
      items: [{ productId: '', quantity: 1, note: '' }],
    },
  });

  const { fields, append, remove, move } = useFieldArray({
    control: form.control,
    name: 'items',
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <input {...form.register('customerName')} placeholder="Customer Name" />

      {fields.map((field, index) => (
        <div key={field.id} className="flex gap-2 items-start">
          <select {...form.register(`items.${index}.productId`)}>
            <option value="">Select product</option>
            <option value="prod_1">Widget A</option>
            <option value="prod_2">Widget B</option>
          </select>

          <input
            type="number"
            {...form.register(`items.${index}.quantity`)}
            className="w-20"
          />

          <input
            {...form.register(`items.${index}.note`)}
            placeholder="Note"
          />

          {fields.length > 1 && (
            <button type="button" onClick={() => remove(index)}>Remove</button>
          )}
        </div>
      ))}

      <button
        type="button"
        onClick={() => append({ productId: '', quantity: 1, note: '' })}
      >
        + Add Item
      </button>

      <button type="submit">Place Order</button>
    </form>
  );
}
```

---

## 3. 条件分岐フォーム

```typescript
// 通知設定: タイプによって異なるフィールド
const notificationSchema = z.discriminatedUnion('type', [
  z.object({
    type: z.literal('email'),
    email: z.string().email(),
    frequency: z.enum(['daily', 'weekly', 'monthly']),
  }),
  z.object({
    type: z.literal('sms'),
    phone: z.string().regex(/^\+?\d{10,15}$/),
  }),
  z.object({
    type: z.literal('webhook'),
    url: z.string().url(),
    secret: z.string().min(16),
  }),
]);

function NotificationForm() {
  const form = useForm({
    resolver: zodResolver(notificationSchema),
    defaultValues: { type: 'email' as const },
  });

  const type = form.watch('type');

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <select {...form.register('type')}>
        <option value="email">Email</option>
        <option value="sms">SMS</option>
        <option value="webhook">Webhook</option>
      </select>

      {type === 'email' && (
        <>
          <input type="email" {...form.register('email')} placeholder="Email" />
          <select {...form.register('frequency')}>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </>
      )}

      {type === 'sms' && (
        <input {...form.register('phone')} placeholder="+81..." />
      )}

      {type === 'webhook' && (
        <>
          <input {...form.register('url')} placeholder="https://..." />
          <input {...form.register('secret')} placeholder="Secret key" />
        </>
      )}

      <button type="submit">Save</button>
    </form>
  );
}
```

---

## 4. フォームの自動保存

```typescript
// 自動保存（デバウンス付き）
function AutoSaveForm({ defaultValues }: { defaultValues: FormData }) {
  const form = useForm({ defaultValues });

  // 値の変更を監視して自動保存
  useEffect(() => {
    const subscription = form.watch((value) => {
      debouncedSave(value);
    });
    return () => subscription.unsubscribe();
  }, [form.watch]);

  const debouncedSave = useMemo(
    () => debounce(async (data: any) => {
      try {
        await api.drafts.save(data);
      } catch {
        // サイレントに失敗（ユーザーに通知しない）
      }
    }, 1000),
    []
  );

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {/* フィールド */}
      <p className="text-xs text-gray-400">
        {form.formState.isDirty ? 'Saving...' : 'Saved'}
      </p>
    </form>
  );
}
```

---

## 5. ページ離脱防止

```typescript
// 未保存の変更がある場合にページ離脱を防止
function useUnsavedChangesWarning(isDirty: boolean) {
  useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
      }
    };
    window.addEventListener('beforeunload', handler);
    return () => window.removeEventListener('beforeunload', handler);
  }, [isDirty]);
}

// Next.js でのルート遷移防止
// → useRouter の beforePopState（App Router では未サポート）
// → 代替: フォーム状態を Context で管理し、Link に確認ダイアログを追加
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| マルチステップ | ウィザード形式の登録フロー |
| useFieldArray | 動的な配列フィールド |
| discriminatedUnion | 条件分岐バリデーション |
| 自動保存 | 下書き保存 |
| 離脱防止 | 未保存データの保護 |

---

## 次に読むべきガイド
→ [[00-deployment-platforms.md]] — デプロイ先

---

## 参考文献
1. React Hook Form. "useFieldArray." react-hook-form.com, 2024.
2. Zod. "Discriminated Unions." zod.dev, 2024.
