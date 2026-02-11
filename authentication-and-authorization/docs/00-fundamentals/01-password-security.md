# パスワードセキュリティ

> パスワードは最も広く使われる認証手段だが、最も攻撃されやすい。bcrypt/Argon2によるハッシュ化、安全なパスワードポリシー、漏洩検知、パスワードリセットフローまで、パスワード管理の全ベストプラクティスを解説する。

## この章で学ぶこと

- [ ] パスワードハッシュの仕組みと適切なアルゴリズム選択を理解する
- [ ] 安全なパスワードポリシーを設計できるようになる
- [ ] パスワードリセットとアカウントリカバリーを安全に実装する

---

## 1. パスワードハッシュの基礎

```
なぜハッシュが必要か:

  平文保存のリスク:
    DB漏洩 → 全ユーザーのパスワード即座に判明
    内部犯行 → 開発者がパスワードを閲覧可能
    ログ混入 → パスワードがログファイルに記録

  ハッシュの役割:
    パスワード → ハッシュ関数 → ハッシュ値（不可逆）
    "password123" → bcrypt → "$2b$12$LJ3m4ys..."
    ハッシュ値からパスワードを復元不可能

ハッシュ vs 暗号化:
  ハッシュ: 一方向（復元不可）→ パスワードに使用
  暗号化:  双方向（復号可能）→ パスワードには使わない

  ✗ AES暗号化 → 鍵があれば復号できるため不適切
  ✗ MD5 / SHA-256 → 高速すぎてブルートフォースに弱い
  ✓ bcrypt / Argon2 → 意図的に低速化された専用ハッシュ
```

```
ソルトの重要性:

  ソルトなし:
    "password" → SHA-256 → "5e884..."（全ユーザー同じ）
    → レインボーテーブルで一括解読可能

  ソルト付き:
    "password" + "a3f8e2..." → SHA-256 → "8b2c1..."
    "password" + "7d4b9c..." → SHA-256 → "f1e3a..."
    → ユーザーごとに異なるハッシュ値
    → レインボーテーブル攻撃を無効化

  bcrypt / Argon2 はソルトを自動生成・埋込み
```

---

## 2. 推奨アルゴリズム

```
アルゴリズム比較:

  アルゴリズム │ 推奨度  │ 特徴
  ──────────┼────────┼──────────────────────────
  Argon2id  │ ◎ 最良  │ メモリハード、GPU耐性最強
  bcrypt    │ ○ 良好  │ 実績豊富、広くサポート
  scrypt    │ ○ 良好  │ メモリハード
  PBKDF2    │ △ 可    │ FIPS準拠が必要な場合のみ
  SHA-256   │ ✗ 不可  │ 高速すぎる
  MD5       │ ✗ 不可  │ 高速 + 衝突脆弱性

推奨:
  新規プロジェクト → Argon2id
  既存プロジェクト → bcrypt（十分安全）
  FIPS準拠が必要 → PBKDF2
```

```typescript
// bcrypt 実装
import bcrypt from 'bcrypt';

const SALT_ROUNDS = 12; // コスト係数（2^12 = 4096回のイテレーション）

// パスワードのハッシュ化
async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, SALT_ROUNDS);
}

// パスワードの検証
async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// 使用例
const hash = await hashPassword('mySecurePassword123!');
// "$2b$12$LJ3m4ys3Gk8v0f2xKb2I4O..."

const isValid = await verifyPassword('mySecurePassword123!', hash);
// true
```

```typescript
// Argon2 実装（推奨）
import argon2 from 'argon2';

// Argon2id（推奨バリアント）
async function hashPassword(password: string): Promise<string> {
  return argon2.hash(password, {
    type: argon2.argon2id,  // Argon2id: サイドチャネル + GPU 両対策
    memoryCost: 65536,       // 64MB のメモリ使用
    timeCost: 3,             // 3回のイテレーション
    parallelism: 4,          // 4つの並列レーン
  });
}

async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return argon2.verify(hash, password);
}

// ハッシュ結果例:
// "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$..."
// ↑ アルゴリズム、パラメータ、ソルト、ハッシュが全て含まれる
```

```
コスト係数の選定ガイドライン:

  bcrypt:
    目標: ハッシュ計算に 250ms〜1秒
    cost=10: ~100ms（最低限）
    cost=12: ~300ms（推奨）
    cost=14: ~1s（高セキュリティ）

  Argon2id:
    OWASP 推奨（2024）:
    → memoryCost: 19456 (19MB) 以上
    → timeCost: 2 以上
    → parallelism: 1

    高セキュリティ:
    → memoryCost: 65536 (64MB)
    → timeCost: 3
    → parallelism: 4

  チューニング方法:
    → サーバーで実際に計測
    → ログイン時の許容レイテンシに合わせる
    → 250ms〜1秒が一般的な目標
```

---

## 3. パスワードポリシー

```
NIST SP 800-63B（2020）推奨:

  ✓ 推奨:
    → 最小8文字（できれば15文字以上推奨）
    → 最大64文字以上を許容
    → Unicode の全文字を許容（日本語OK）
    → 漏洩パスワードリストとの照合
    → パスワード強度メーターの表示
    → ペーストを許可（パスワードマネージャー対応）

  ✗ 非推奨（NIST が廃止した古い慣習）:
    → 大文字小文字数字記号の強制（✗ 廃止）
    → 定期的な変更の強制（✗ 廃止）
    → 秘密の質問（✗ 廃止）
    → パスワードヒント（✗ 廃止）

  理由:
    → 複雑性ルール → ユーザーが "P@ssw0rd!" のような予測可能な置換
    → 定期変更 → "password1", "password2"... のインクリメント
    → 長さ重視 → "correct horse battery staple" のような長いフレーズが強力
```

```typescript
// モダンなパスワードバリデーション
import { z } from 'zod';

const passwordSchema = z.string()
  .min(8, 'パスワードは8文字以上必要です')
  .max(128, 'パスワードは128文字以下にしてください')
  .refine(
    (password) => !isCommonPassword(password),
    'このパスワードはよく使われるため安全ではありません'
  )
  .refine(
    async (password) => !(await isBreachedPassword(password)),
    'このパスワードは過去のデータ漏洩で確認されています'
  );

// Have I Been Pwned API でチェック
async function isBreachedPassword(password: string): Promise<boolean> {
  const hash = await sha1(password);
  const prefix = hash.substring(0, 5);
  const suffix = hash.substring(5).toUpperCase();

  // k-Anonymity: ハッシュの先頭5文字のみ送信
  const res = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`);
  const text = await res.text();

  return text.split('\n').some((line) => line.startsWith(suffix));
}

// よく使われるパスワードリスト（上位10万件）
function isCommonPassword(password: string): boolean {
  return commonPasswords.has(password.toLowerCase());
}
```

```typescript
// パスワード強度メーター（zxcvbn）
import zxcvbn from 'zxcvbn';

function checkPasswordStrength(password: string) {
  const result = zxcvbn(password);

  return {
    score: result.score,           // 0-4（0=最弱, 4=最強）
    crackTime: result.crack_times_display.offline_slow_hashing_1e4_per_second,
    feedback: result.feedback,     // 改善提案
    warning: result.feedback.warning,
  };
}

// 結果例:
// "password" → score: 0, crackTime: "less than a second"
// "correcthorsebatterystaple" → score: 4, crackTime: "centuries"

// React コンポーネント
function PasswordStrengthMeter({ password }: { password: string }) {
  const { score, feedback } = checkPasswordStrength(password);
  const labels = ['非常に弱い', '弱い', '普通', '強い', '非常に強い'];
  const colors = ['red', 'orange', 'yellow', 'lime', 'green'];

  return (
    <div>
      <div className="flex gap-1">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-1 flex-1 rounded"
            style={{ backgroundColor: i <= score ? colors[score] : '#e5e7eb' }}
          />
        ))}
      </div>
      <p className="text-xs mt-1">{labels[score]}</p>
      {feedback.warning && <p className="text-xs text-red-500">{feedback.warning}</p>}
    </div>
  );
}
```

---

## 4. パスワードリセット

```
安全なパスワードリセットフロー:

  ユーザー        フロントエンド      バックエンド         メールサーバー
    │               │                 │                   │
    │ リセット要求   │                 │                   │
    │──────────────>│                 │                   │
    │               │ POST /reset     │                   │
    │               │────────────────>│                   │
    │               │                 │ トークン生成        │
    │               │                 │ （ランダム、有効期限付き）
    │               │                 │────────────────────>│
    │               │                 │                   │ メール送信
    │               │  「メールを確認   │                   │
    │               │   してください」  │                   │
    │               │<────────────────│                   │
    │               │                 │                   │
    │ メール内リンク  │                 │                   │
    │ をクリック      │                 │                   │
    │──────────────>│                 │                   │
    │               │ トークン検証      │                   │
    │               │────────────────>│                   │
    │               │                 │ トークン有効性確認   │
    │ 新パスワード入力│                 │                   │
    │──────────────>│                 │                   │
    │               │ POST /reset/confirm                 │
    │               │────────────────>│                   │
    │               │                 │ パスワード更新      │
    │               │                 │ 全セッション無効化   │
    │ 完了          │                 │ トークン無効化       │
    │<──────────────│                 │                   │
```

```typescript
// パスワードリセットの実装
import crypto from 'crypto';

// リセットトークン生成
async function createResetToken(email: string): Promise<void> {
  const user = await db.user.findUnique({ where: { email } });

  // ユーザーが存在しなくても同じレスポンスを返す（ユーザー列挙攻撃対策）
  if (!user) return;

  // 安全なランダムトークン生成
  const token = crypto.randomBytes(32).toString('hex');
  const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

  await db.resetToken.create({
    data: {
      userId: user.id,
      token: hashedToken,            // ハッシュ化して保存
      expiresAt: new Date(Date.now() + 60 * 60 * 1000), // 1時間有効
    },
  });

  // リセットリンクを送信（平文トークンをURL に含める）
  await sendEmail({
    to: email,
    subject: 'パスワードリセット',
    html: `
      <p>以下のリンクからパスワードをリセットしてください（1時間有効）:</p>
      <a href="${process.env.APP_URL}/reset-password?token=${token}">
        パスワードをリセット
      </a>
    `,
  });
}

// パスワードリセット実行
async function resetPassword(token: string, newPassword: string): Promise<void> {
  const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

  const resetToken = await db.resetToken.findFirst({
    where: {
      token: hashedToken,
      expiresAt: { gt: new Date() },   // 有効期限チェック
      usedAt: null,                     // 未使用チェック
    },
  });

  if (!resetToken) {
    throw new Error('Invalid or expired reset token');
  }

  // パスワード更新
  const hashedPassword = await hashPassword(newPassword);
  await db.$transaction([
    db.user.update({
      where: { id: resetToken.userId },
      data: { password: hashedPassword },
    }),
    // トークンを使用済みに
    db.resetToken.update({
      where: { id: resetToken.id },
      data: { usedAt: new Date() },
    }),
    // 全セッション無効化（パスワード変更時は全デバイスからログアウト）
    db.session.deleteMany({
      where: { userId: resetToken.userId },
    }),
  ]);
}
```

```
パスワードリセットのセキュリティ要件:

  トークン:
  ✓ 暗号的に安全なランダム値（crypto.randomBytes）
  ✓ DB にはハッシュ化して保存
  ✓ 有効期限を設定（1時間以内推奨）
  ✓ 使用後は即座に無効化
  ✓ 1ユーザー1トークン（新規発行時に旧トークン削除）

  レスポンス:
  ✓ ユーザー存在有無に関わらず同じレスポンス
    → 「メールアドレスが登録されていればメールを送信しました」
    → ユーザー列挙攻撃を防止

  追加対策:
  ✓ レート制限（同一メールへの連続リクエスト制限）
  ✓ パスワード変更後の全セッション無効化
  ✓ パスワード変更通知メールの送信
  ✓ 旧パスワードと同じ新パスワードを拒否
```

---

## 5. アンチパターン

```
パスワード管理のアンチパターン:

  ✗ 平文保存:
    → DB漏洩で全パスワードが即座に判明
    → 法的責任を問われる可能性

  ✗ 可逆暗号化:
    → AES等で暗号化 → 鍵があれば復号可能
    → 鍵管理の問題が発生

  ✗ MD5/SHA-256（ソルトなし）:
    → レインボーテーブルで解読可能
    → GPU で毎秒数十億回のハッシュ計算

  ✗ 独自のハッシュアルゴリズム:
    → 暗号の専門家でない限り脆弱性がある
    → 検証済みのライブラリを使用すべき

  ✗ パスワードの最大長制限（例: 16文字）:
    → パスフレーズの使用を妨げる
    → ハッシュ化すれば長さは関係ない

  ✗ パスワードをログに出力:
    → 平文パスワードがログファイルに残る
    → リクエストボディのログ記録時に特に注意

  ✗ エラーメッセージで情報漏洩:
    → 「パスワードが間違っています」→ ユーザー存在が判明
    → 「メールアドレスまたはパスワードが間違っています」が正しい
```

---

## まとめ

| 項目 | ベストプラクティス |
|------|-----------------|
| ハッシュ | Argon2id（推奨）または bcrypt |
| ソルト | アルゴリズムが自動生成（手動不要） |
| コスト | 250ms〜1秒のハッシュ計算時間 |
| ポリシー | 8文字以上、漏洩チェック、強度メーター |
| リセット | 暗号ランダムトークン、1時間有効、ハッシュ保存 |
| エラー | 「メールまたはパスワードが違います」（曖昧に） |

---

## 次に読むべきガイド
→ [[02-multi-factor-authentication.md]] — 多要素認証

---

## 参考文献
1. NIST. "SP 800-63B: Digital Identity Guidelines." nist.gov, 2020.
2. OWASP. "Password Storage Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. Troy Hunt. "Have I Been Pwned." haveibeenpwned.com, 2024.
4. Dropbox. "zxcvbn: Realistic Password Strength Estimation." github.com/dropbox/zxcvbn.
