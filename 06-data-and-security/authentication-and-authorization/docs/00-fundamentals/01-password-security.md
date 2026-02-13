# パスワードセキュリティ

> パスワードは最も広く使われる認証手段だが、最も攻撃されやすい。bcrypt/Argon2によるハッシュ化、安全なパスワードポリシー、漏洩検知、パスワードリセットフローまで、パスワード管理の全ベストプラクティスを解説する。NIST SP 800-63B、OWASP Password Storage Cheat Sheet に基づき、内部アルゴリズムレベルの理解から実運用のセキュリティ対策までを網羅する。

## 前提知識

- [[00-authentication-vs-authorization.md]] — 認証と認可の基礎
- ハッシュ関数の基本概念（一方向性、衝突耐性）
- 対称鍵暗号と非対称鍵暗号の違い
- HTTP リクエスト/レスポンスの基本
- データベース操作の基本

## この章で学ぶこと

- [ ] パスワードハッシュの仕組みと適切なアルゴリズム選択を理解する
- [ ] bcrypt と Argon2id の内部動作を把握する
- [ ] 安全なパスワードポリシーを設計できるようになる
- [ ] パスワードリセットとアカウントリカバリーを安全に実装する
- [ ] ブルートフォース攻撃とクレデンシャルスタッフィングへの防御策を習得する
- [ ] パスワードマイグレーション戦略を理解する

---

## 1. パスワードハッシュの基礎

```
なぜハッシュが必要か:

  平文保存のリスク:
    DB漏洩 → 全ユーザーのパスワード即座に判明
    内部犯行 → 開発者がパスワードを閲覧可能
    ログ混入 → パスワードがログファイルに記録
    バックアップ → バックアップファイルから読取可能

  ハッシュの役割:
    パスワード → ハッシュ関数 → ハッシュ値（不可逆）
    "password123" → bcrypt → "$2b$12$LJ3m4ys..."
    ハッシュ値からパスワードを復元不可能

ハッシュ vs 暗号化:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ハッシュ（一方向関数）:                            │
  │  → 入力 → ハッシュ値（復元不可能）                  │
  │  → パスワード保存に使用                            │
  │  → 同じ入力 → 常に同じ出力                         │
  │  → 例: bcrypt, Argon2, SHA-256                   │
  │                                                  │
  │  暗号化（双方向関数）:                              │
  │  → 平文 → 暗号文（復号可能）                       │
  │  → データの保護に使用（パスワードには使わない）       │
  │  → 鍵があれば復元可能                              │
  │  → 例: AES, ChaCha20                             │
  │                                                  │
  └──────────────────────────────────────────────────┘

  ✗ AES暗号化 → 鍵があれば復号できるため不適切
  ✗ MD5 / SHA-256 → 高速すぎてブルートフォースに弱い
  ✓ bcrypt / Argon2 → 意図的に低速化された専用ハッシュ
```

### 1.1 ソルトの重要性

```
ソルトの仕組み:

  ソルトなし:
    "password" → SHA-256 → "5e884..."（全ユーザー同じ）
    → レインボーテーブルで一括解読可能

  ソルト付き:
    "password" + "a3f8e2..." → SHA-256 → "8b2c1..."
    "password" + "7d4b9c..." → SHA-256 → "f1e3a..."
    → ユーザーごとに異なるハッシュ値
    → レインボーテーブル攻撃を無効化

  ソルトの要件:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ① 暗号的に安全な乱数生成器で生成                   │
  │     → crypto.randomBytes(16) 以上                │
  │     → Math.random() は不可                       │
  │                                                  │
  │  ② ユーザーごとに一意                              │
  │     → 同じパスワードでも異なるハッシュ値              │
  │                                                  │
  │  ③ ハッシュ値と共に保存                             │
  │     → bcrypt は自動的にハッシュ値にソルトを埋め込む  │
  │     → 手動管理は不要                               │
  │                                                  │
  │  ④ 十分な長さ（16バイト / 128ビット以上）            │
  │     → ソルト空間が広いほどレインボーテーブルが困難    │
  │                                                  │
  └──────────────────────────────────────────────────┘

  bcrypt / Argon2 はソルトを自動生成・埋込み → 手動管理不要
```

### 1.2 ペッパー（Secret Salt）

```
ペッパーの概念:

  ソルト: ハッシュ値と共にDBに保存（公開情報）
  ペッパー: DB外に保存される秘密値（秘密情報）

  目的:
  → DB漏洩だけではハッシュを攻撃できない
  → ペッパーも入手しないとオフライン攻撃不可

  実装パターン:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  方法1: HMAC ラッピング                            │
  │  hash = bcrypt(HMAC-SHA256(pepper, password))    │
  │  → ペッパーは環境変数 or HSM/KMS に保存            │
  │                                                  │
  │  方法2: 暗号化ラッピング                            │
  │  stored = AES-256-GCM(key, bcrypt(password))     │
  │  → ハッシュ値を暗号化して保存                       │
  │  → 鍵ローテーションが容易                           │
  │                                                  │
  └──────────────────────────────────────────────────┘

  推奨: 方法2（暗号化ラッピング）
  → ペッパーの更新時にパスワード再設定が不要
  → 鍵ローテーション = 暗号化し直すだけ
```

```typescript
// ペッパー（暗号化ラッピング）の実装
import crypto from 'crypto';
import argon2 from 'argon2';

const PEPPER_KEY = Buffer.from(process.env.PEPPER_KEY!, 'hex'); // 32バイト

// パスワードハッシュ + ペッパー暗号化
async function hashPasswordWithPepper(password: string): Promise<string> {
  // Step 1: Argon2id でハッシュ化
  const hash = await argon2.hash(password, {
    type: argon2.argon2id,
    memoryCost: 65536,
    timeCost: 3,
    parallelism: 4,
  });

  // Step 2: AES-256-GCM で暗号化
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv('aes-256-gcm', PEPPER_KEY, iv);
  const encrypted = Buffer.concat([
    cipher.update(hash, 'utf8'),
    cipher.final(),
  ]);
  const authTag = cipher.getAuthTag();

  // iv + authTag + encrypted を結合して返す
  return Buffer.concat([iv, authTag, encrypted]).toString('base64');
}

// パスワード検証
async function verifyPasswordWithPepper(
  password: string,
  stored: string
): Promise<boolean> {
  const data = Buffer.from(stored, 'base64');

  // 分離: iv(16) + authTag(16) + encrypted(残り)
  const iv = data.subarray(0, 16);
  const authTag = data.subarray(16, 32);
  const encrypted = data.subarray(32);

  // Step 1: AES-256-GCM で復号
  const decipher = crypto.createDecipheriv('aes-256-gcm', PEPPER_KEY, iv);
  decipher.setAuthTag(authTag);
  const hash = decipher.update(encrypted) + decipher.final('utf8');

  // Step 2: Argon2 で検証
  return argon2.verify(hash, password);
}
```

---

## 2. 推奨アルゴリズム

```
アルゴリズム比較:

  アルゴリズム │ 推奨度  │ 特徴                    │ GPU耐性
  ──────────┼────────┼────────────────────────┼────────
  Argon2id  │ ◎ 最良  │ メモリハード、GPU耐性最強  │ ◎
  bcrypt    │ ○ 良好  │ 実績豊富、広くサポート     │ ○
  scrypt    │ ○ 良好  │ メモリハード              │ ○
  PBKDF2    │ △ 可    │ FIPS準拠が必要な場合のみ  │ △
  SHA-256   │ ✗ 不可  │ 高速すぎる               │ ✗
  MD5       │ ✗ 不可  │ 高速 + 衝突脆弱性         │ ✗

推奨:
  新規プロジェクト → Argon2id
  既存プロジェクト → bcrypt（十分安全）
  FIPS準拠が必要 → PBKDF2（HMAC-SHA256, 600,000回以上）
```

### 2.1 bcrypt の内部動作

```
bcrypt の構造:

  ハッシュ値の形式:
  $2b$12$LJ3m4ys3Gk8v0f2xKb2I4OXYiDkG0...
  │  │ │  └──────────────────────────────── ハッシュ + ソルト
  │  │ └─── コスト係数（2^12 = 4096回）
  │  └───── バージョン（2b が最新）
  └──────── アルゴリズム識別子

  内部アルゴリズム:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  bcrypt(password, cost, salt):                    │
  │                                                  │
  │  ① state = EksBlowfishSetup(cost, salt, password)│
  │                                                  │
  │  ② ctext = "OrpheanBeholderScryDoubt"            │
  │     → 24バイトの固定マジック文字列                  │
  │                                                  │
  │  ③ for i = 0 to 63:                              │
  │       ctext = EncryptECB(state, ctext)            │
  │     → Blowfish ECB 暗号化を 64 回繰り返す         │
  │                                                  │
  │  ④ return concat(cost, salt, ctext)              │
  │                                                  │
  │  EksBlowfishSetup:                               │
  │  → Blowfish の鍵スケジュールを2^cost回繰り返す     │
  │  → cost=12 の場合: 2^12 = 4,096 回               │
  │  → 各イテレーションでパスワードとソルトを交互に使用  │
  │  → これが「意図的な遅さ」の源                       │
  │                                                  │
  └──────────────────────────────────────────────────┘

  bcrypt の制限:
  → パスワード長: 最大72バイト（超過分は無視）
  → UTF-8 の場合、日本語は1文字3バイト → 24文字が上限
  → 対策: SHA-256 プレハッシュ
    bcrypt(SHA256(password).base64())
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

// bcrypt の72バイト制限に対するプレハッシュ
import crypto from 'crypto';

async function hashLongPassword(password: string): Promise<string> {
  // SHA-256 でプレハッシュ（Base64で44文字、72バイト以内）
  const preHash = crypto.createHash('sha256').update(password).digest('base64');
  return bcrypt.hash(preHash, SALT_ROUNDS);
}

async function verifyLongPassword(password: string, hash: string): Promise<boolean> {
  const preHash = crypto.createHash('sha256').update(password).digest('base64');
  return bcrypt.compare(preHash, hash);
}
```

### 2.2 Argon2id の内部動作

```
Argon2 の3つのバリアント:

  Argon2d: サイドチャネル攻撃に弱いが、GPU 攻撃に最強
  Argon2i: サイドチャネル攻撃に強いが、GPU 耐性がやや低い
  Argon2id: Argon2d + Argon2i のハイブリッド（推奨）

Argon2id の内部動作:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  パラメータ:                                       │
  │  → memoryCost (m): 使用メモリ量（KB）              │
  │  → timeCost (t): イテレーション回数                 │
  │  → parallelism (p): 並列レーン数                   │
  │  → saltLength: ソルト長（16バイト推奨）             │
  │  → hashLength: 出力ハッシュ長（32バイト推奨）       │
  │                                                  │
  │  アルゴリズム:                                     │
  │  ① メモリを m KB 確保                              │
  │  ② メモリを p 個のレーンに分割                      │
  │  ③ 各レーンで独立にメモリフィリング                  │
  │  ④ t 回のパス実行:                                 │
  │     → 最初のパス: Argon2i モード                   │
  │       （データ独立アクセス → サイドチャネル耐性）     │
  │     → 2回目以降: Argon2d モード                    │
  │       （データ依存アクセス → GPU 攻撃耐性）          │
  │  ⑤ 各レーンの最終ブロックを XOR して出力            │
  │                                                  │
  │  なぜメモリハードが重要か:                           │
  │  → GPU は計算は速いがメモリが限定的                 │
  │  → 64MB のメモリが必要 → GPU の並列実行数が激減     │
  │  → ASIC での攻撃コストも大幅に増加                  │
  │                                                  │
  └──────────────────────────────────────────────────┘
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
    saltLength: 16,          // 16バイトのソルト
    hashLength: 32,          // 32バイトのハッシュ出力
  });
}

async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return argon2.verify(hash, password);
}

// ハッシュ結果例:
// "$argon2id$v=19$m=65536,t=3,p=4$c29tZXNhbHQ$..."
// ↑ アルゴリズム、パラメータ、ソルト、ハッシュが全て含まれる

// パスワードが変更された時のリハッシュ確認
async function needsRehash(hash: string): Promise<boolean> {
  return argon2.needsRehash(hash, {
    type: argon2.argon2id,
    memoryCost: 65536,
    timeCost: 3,
    parallelism: 4,
  });
}

// ログイン時の透過的リハッシュ
async function loginWithRehash(
  password: string,
  storedHash: string,
  userId: string
): Promise<boolean> {
  const isValid = await argon2.verify(storedHash, password);

  if (isValid && await needsRehash(storedHash)) {
    // パラメータが古い場合、新しいパラメータでリハッシュ
    const newHash = await hashPassword(password);
    await db.user.update({
      where: { id: userId },
      data: { password: newHash },
    });
    console.log(`Rehashed password for user ${userId}`);
  }

  return isValid;
}
```

### 2.3 コスト係数の選定

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

    最高セキュリティ（余裕がある場合）:
    → memoryCost: 131072 (128MB)
    → timeCost: 4
    → parallelism: 4

  チューニング方法:
    → サーバーで実際に計測
    → ログイン時の許容レイテンシに合わせる
    → 250ms〜1秒が一般的な目標
    → 定期的にパラメータを見直す（ハードウェアの進化に合わせる）

  サーバーリソースへの影響:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  同時ログインユーザー数 × ハッシュ計算時間            │
  │  → 100 req/s × 300ms = 30 CPU コア分              │
  │                                                  │
  │  対策:                                            │
  │  → ハッシュ計算をワーカースレッドで実行              │
  │  → 同時実行数を制限（Semaphore）                   │
  │  → 急激なスパイク時はキューイング                    │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

```typescript
// コスト係数のベンチマーク
async function benchmarkHashParameters() {
  const password = 'test-password-for-benchmarking';

  // bcrypt ベンチマーク
  for (const cost of [10, 11, 12, 13, 14]) {
    const start = performance.now();
    await bcrypt.hash(password, cost);
    const duration = performance.now() - start;
    console.log(`bcrypt cost=${cost}: ${duration.toFixed(0)}ms`);
  }

  // Argon2id ベンチマーク
  const configs = [
    { memoryCost: 19456, timeCost: 2, parallelism: 1 },
    { memoryCost: 47104, timeCost: 1, parallelism: 1 },
    { memoryCost: 65536, timeCost: 3, parallelism: 4 },
    { memoryCost: 131072, timeCost: 4, parallelism: 4 },
  ];

  for (const config of configs) {
    const start = performance.now();
    await argon2.hash(password, { type: argon2.argon2id, ...config });
    const duration = performance.now() - start;
    console.log(
      `argon2id m=${config.memoryCost} t=${config.timeCost} p=${config.parallelism}: ${duration.toFixed(0)}ms`
    );
  }
}

// ハッシュ計算の並行制御
class HashService {
  private semaphore: number = 0;
  private readonly maxConcurrent: number;
  private queue: Array<() => void> = [];

  constructor(maxConcurrent: number = 10) {
    this.maxConcurrent = maxConcurrent;
  }

  async hash(password: string): Promise<string> {
    await this.acquire();
    try {
      return await argon2.hash(password, {
        type: argon2.argon2id,
        memoryCost: 65536,
        timeCost: 3,
        parallelism: 4,
      });
    } finally {
      this.release();
    }
  }

  async verify(password: string, hash: string): Promise<boolean> {
    await this.acquire();
    try {
      return await argon2.verify(hash, password);
    } finally {
      this.release();
    }
  }

  private async acquire(): Promise<void> {
    if (this.semaphore < this.maxConcurrent) {
      this.semaphore++;
      return;
    }
    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }

  private release(): void {
    this.semaphore--;
    if (this.queue.length > 0) {
      this.semaphore++;
      const next = this.queue.shift()!;
      next();
    }
  }
}
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
    → 過去パスワードとの類似性チェック（✗ 過度なものは廃止）

  理由:
    → 複雑性ルール → ユーザーが "P@ssw0rd!" のような予測可能な置換
    → 定期変更 → "password1", "password2"... のインクリメント
    → 長さ重視 → "correct horse battery staple" のような長いフレーズが強力

パスワード強度とエントロピー:

  ┌───────────────────────────────┬─────────┬──────────────┐
  │ パスワードの例                 │ エントロピー│ オフライン攻撃  │
  ├───────────────────────────────┼─────────┼──────────────┤
  │ "password"                   │ ~0 bit  │ 即座に解読     │
  │ "P@ssw0rd!"                  │ ~15 bit │ 数秒          │
  │ "7kX#mP2q"                   │ ~50 bit │ 数時間        │
  │ "correct horse battery staple"│ ~44 bit │ 数日          │
  │ "dWp8#kL2$mN9xQ4@"          │ ~95 bit │ 数十億年       │
  │ ランダム20文字（全文字種）       │ ~130 bit│ 宇宙の寿命超   │
  └───────────────────────────────┴─────────┴──────────────┘

  ※ オフライン攻撃は bcrypt cost=12 前提
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

// Have I Been Pwned API でチェック（k-Anonymity モデル）
async function isBreachedPassword(password: string): Promise<boolean> {
  const hash = await sha1(password);
  const prefix = hash.substring(0, 5);
  const suffix = hash.substring(5).toUpperCase();

  // k-Anonymity: ハッシュの先頭5文字のみ送信
  // → サーバーにパスワードの情報を漏らさない
  const res = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`, {
    headers: { 'Add-Padding': 'true' }, // タイミング攻撃防止
  });
  const text = await res.text();

  // レスポンス例:
  // "1E4C9B93F3F0682250B6CF8331B7EE68FD8:3"
  // → suffix: 一致するハッシュ, count: 漏洩回数
  return text.split('\n').some((line) => {
    const [hashSuffix] = line.split(':');
    return hashSuffix === suffix;
  });
}

// SHA-1 ハッシュ（HIBP API 用）
async function sha1(input: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(input);
  const hashBuffer = await crypto.subtle.digest('SHA-1', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('').toUpperCase();
}

// よく使われるパスワードリスト（上位10万件）
const commonPasswordSet = new Set<string>();
// 起動時にファイルから読み込み
function isCommonPassword(password: string): boolean {
  return commonPasswordSet.has(password.toLowerCase());
}

// コンテキスト依存チェック
function containsUserInfo(password: string, userInfo: {
  email: string;
  name?: string;
  username?: string;
}): boolean {
  const lowerPassword = password.toLowerCase();
  const checks = [
    userInfo.email.split('@')[0],
    userInfo.name,
    userInfo.username,
  ].filter(Boolean).map((s) => s!.toLowerCase());

  return checks.some((info) => lowerPassword.includes(info));
}
```

### 3.1 パスワード強度メーター

```typescript
// パスワード強度メーター（zxcvbn）
import zxcvbn from 'zxcvbn';

function checkPasswordStrength(password: string, userInputs: string[] = []) {
  const result = zxcvbn(password, userInputs);

  return {
    score: result.score,           // 0-4（0=最弱, 4=最強）
    crackTime: result.crack_times_display.offline_slow_hashing_1e4_per_second,
    feedback: result.feedback,     // 改善提案
    warning: result.feedback.warning,
    guesses: result.guesses,       // 推定試行回数
    guessesLog10: result.guesses_log10,
  };
}

// 結果例:
// "password" → score: 0, crackTime: "less than a second"
// "correcthorsebatterystaple" → score: 4, crackTime: "centuries"

// React コンポーネント
function PasswordStrengthMeter({ password, email }: {
  password: string;
  email: string;
}) {
  const { score, feedback, crackTime } = checkPasswordStrength(
    password,
    [email.split('@')[0]] // ユーザー固有の入力をペナルティ対象に
  );

  const labels = ['非常に弱い', '弱い', '普通', '強い', '非常に強い'];
  const colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e'];

  if (!password) return null;

  return (
    <div className="mt-2">
      {/* 強度バー */}
      <div className="flex gap-1">
        {[0, 1, 2, 3].map((i) => (
          <div
            key={i}
            className="h-1.5 flex-1 rounded-full transition-colors"
            style={{
              backgroundColor: i <= score ? colors[score] : '#e5e7eb',
            }}
          />
        ))}
      </div>

      {/* ラベルと推定解読時間 */}
      <div className="flex justify-between mt-1">
        <span className="text-xs" style={{ color: colors[score] }}>
          {labels[score]}
        </span>
        <span className="text-xs text-gray-500">
          解読推定: {crackTime}
        </span>
      </div>

      {/* フィードバック */}
      {feedback.warning && (
        <p className="text-xs text-amber-600 mt-1">{feedback.warning}</p>
      )}
      {feedback.suggestions.map((suggestion: string, i: number) => (
        <p key={i} className="text-xs text-gray-500 mt-0.5">{suggestion}</p>
      ))}
    </div>
  );
}
```

---

## 4. ブルートフォース対策

```
攻撃手法と対策:

  ① オンラインブルートフォース:
     → ログインエンドポイントへの連続試行
     → 対策: レート制限、アカウントロックアウト

  ② オフラインブルートフォース:
     → DB漏洩後のハッシュに対する攻撃
     → 対策: 強力なハッシュアルゴリズム（Argon2id）

  ③ クレデンシャルスタッフィング:
     → 他サービスで漏洩した認証情報の流用
     → 対策: 漏洩チェック（HIBP）、MFA

  ④ パスワードスプレー:
     → 少数の一般的なパスワードで多数アカウントを試行
     → 対策: よく使われるパスワードの禁止、IP ベースの制限

攻撃速度の比較（GPU クラスター想定）:

  アルゴリズム     │ 試行速度 / 秒    │ 8文字ランダム解読
  ──────────────┼────────────────┼──────────────
  MD5            │ ~300 億         │ 数秒
  SHA-256        │ ~30 億          │ 数分
  bcrypt (12)    │ ~10万           │ 数十年
  Argon2id (64MB)│ ~1,000          │ 数億年
```

```typescript
// レート制限の実装（Redis ベース）
class LoginRateLimiter {
  constructor(private redis: Redis) {}

  // IP ベースの制限
  async checkIPLimit(ip: string): Promise<{ allowed: boolean; retryAfter?: number }> {
    const key = `login:ip:${ip}`;
    const attempts = await this.redis.incr(key);

    if (attempts === 1) {
      await this.redis.expire(key, 900); // 15分
    }

    if (attempts > 100) { // IP あたり15分に100回まで
      const ttl = await this.redis.ttl(key);
      return { allowed: false, retryAfter: ttl };
    }

    return { allowed: true };
  }

  // アカウントベースの制限
  async checkAccountLimit(email: string): Promise<{
    allowed: boolean;
    retryAfter?: number;
    remainingAttempts?: number;
  }> {
    const key = `login:account:${email.toLowerCase()}`;
    const attempts = await this.redis.incr(key);

    if (attempts === 1) {
      await this.redis.expire(key, 3600); // 1時間
    }

    const maxAttempts = 10;

    if (attempts > maxAttempts) {
      const ttl = await this.redis.ttl(key);
      // プログレッシブロックアウト: 失敗が増えるほど長くロック
      const lockoutTime = Math.min(
        Math.pow(2, attempts - maxAttempts) * 60, // 指数バックオフ
        3600 // 最大1時間
      );
      await this.redis.expire(key, lockoutTime);

      return {
        allowed: false,
        retryAfter: lockoutTime,
        remainingAttempts: 0,
      };
    }

    return {
      allowed: true,
      remainingAttempts: maxAttempts - attempts,
    };
  }

  // ログイン成功時にカウンターをリセット
  async onLoginSuccess(email: string): Promise<void> {
    await this.redis.del(`login:account:${email.toLowerCase()}`);
  }
}

// ログインエンドポイント
app.post('/auth/login', async (req, res) => {
  const { email, password } = req.body;
  const ip = req.ip!;

  // IP 制限チェック
  const ipCheck = await rateLimiter.checkIPLimit(ip);
  if (!ipCheck.allowed) {
    return res.status(429).json({
      error: 'Too many requests',
      retryAfter: ipCheck.retryAfter,
    });
  }

  // アカウント制限チェック
  const accountCheck = await rateLimiter.checkAccountLimit(email);
  if (!accountCheck.allowed) {
    return res.status(429).json({
      error: 'Account temporarily locked',
      retryAfter: accountCheck.retryAfter,
    });
  }

  // 認証処理
  const user = await db.user.findUnique({ where: { email: email.toLowerCase() } });

  // タイミング攻撃防止: ユーザーが存在しなくてもハッシュ計算
  if (!user) {
    await argon2.hash('dummy-password-for-timing', {
      type: argon2.argon2id, memoryCost: 65536, timeCost: 3, parallelism: 4,
    });
    return res.status(401).json({ error: 'Invalid email or password' });
  }

  const isValid = await loginWithRehash(password, user.password, user.id);

  if (!isValid) {
    return res.status(401).json({
      error: 'Invalid email or password',
      remainingAttempts: accountCheck.remainingAttempts,
    });
  }

  // ログイン成功
  await rateLimiter.onLoginSuccess(email);

  // 異常検知: 新しい IP / デバイスからのログイン
  await notifyUnusualLogin(user, req);

  const tokens = await issueTokens(user);
  res.json(tokens);
});
```

---

## 5. パスワードリセット

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

class PasswordResetService {
  constructor(
    private db: Database,
    private redis: Redis,
    private emailService: EmailService,
    private hashService: HashService
  ) {}

  // リセットトークン生成
  async createResetToken(email: string): Promise<void> {
    const user = await this.db.user.findUnique({ where: { email } });

    // ユーザーが存在しなくても同じレスポンスを返す（ユーザー列挙攻撃対策）
    if (!user) {
      // タイミング攻撃防止
      await new Promise((resolve) => setTimeout(resolve, 200));
      return;
    }

    // レート制限: 同一メールへの連続リクエスト制限
    const rateLimitKey = `reset:ratelimit:${email}`;
    const isLimited = await this.redis.exists(rateLimitKey);
    if (isLimited) return;
    await this.redis.setex(rateLimitKey, 300, '1'); // 5分

    // 既存のトークンを無効化
    await this.db.resetToken.updateMany({
      where: { userId: user.id, usedAt: null },
      data: { usedAt: new Date() },
    });

    // 安全なランダムトークン生成
    const token = crypto.randomBytes(32).toString('hex');
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    await this.db.resetToken.create({
      data: {
        userId: user.id,
        token: hashedToken,            // ハッシュ化して保存
        expiresAt: new Date(Date.now() + 60 * 60 * 1000), // 1時間有効
      },
    });

    // リセットリンクを送信（平文トークンをURL に含める）
    await this.emailService.send(email, {
      subject: 'パスワードリセット',
      html: `
        <p>以下のリンクからパスワードをリセットしてください（1時間有効）:</p>
        <a href="${process.env.APP_URL}/reset-password?token=${token}">
          パスワードをリセット
        </a>
        <p>このリクエストに心当たりがない場合は、このメールを無視してください。</p>
        <p>リンクの有効期限: 1時間</p>
      `,
    });
  }

  // パスワードリセット実行
  async resetPassword(token: string, newPassword: string): Promise<void> {
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    const resetToken = await this.db.resetToken.findFirst({
      where: {
        token: hashedToken,
        expiresAt: { gt: new Date() },   // 有効期限チェック
        usedAt: null,                     // 未使用チェック
      },
      include: { user: true },
    });

    if (!resetToken) {
      throw new Error('Invalid or expired reset token');
    }

    // 旧パスワードと同じでないか確認
    const isSameAsOld = await argon2.verify(resetToken.user.password, newPassword);
    if (isSameAsOld) {
      throw new Error('New password must be different from the current password');
    }

    // パスワード更新
    const hashedPassword = await this.hashService.hash(newPassword);
    await this.db.$transaction([
      this.db.user.update({
        where: { id: resetToken.userId },
        data: { password: hashedPassword },
      }),
      // トークンを使用済みに
      this.db.resetToken.update({
        where: { id: resetToken.id },
        data: { usedAt: new Date() },
      }),
      // 全セッション無効化（パスワード変更時は全デバイスからログアウト）
      this.db.session.deleteMany({
        where: { userId: resetToken.userId },
      }),
    ]);

    // パスワード変更通知メールを送信
    await this.emailService.send(resetToken.user.email, {
      subject: 'パスワードが変更されました',
      html: `
        <p>パスワードが正常に変更されました。</p>
        <p>変更日時: ${new Date().toISOString()}</p>
        <p>この変更に心当たりがない場合は、直ちにサポートに連絡してください。</p>
      `,
    });
  }
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
  ✓ CAPTCHA（ボット対策）
```

---

## 6. パスワードマイグレーション

```
既存のハッシュアルゴリズムから移行する戦略:

  シナリオ: MD5/SHA-256 → Argon2id への移行

  方法1: 透過的リハッシュ（推奨）
  ┌──────────────────────────────────────────────┐
  │  ① ユーザーがログイン                          │
  │  ② 旧アルゴリズムでパスワードを検証              │
  │  ③ 検証成功 → 新アルゴリズムでリハッシュ         │
  │  ④ DB のハッシュ値を更新                        │
  │                                              │
  │  利点: ユーザーの操作不要、段階的に移行           │
  │  欠点: 全ユーザーがログインするまで移行完了しない  │
  └──────────────────────────────────────────────┘

  方法2: ラッピング（即座に移行）
  ┌──────────────────────────────────────────────┐
  │  旧: MD5(password)                            │
  │  新: Argon2id(MD5(password))                   │
  │                                              │
  │  ① 既存の MD5 ハッシュを Argon2id で再ハッシュ   │
  │  ② 全ユーザーを即座に移行                       │
  │  ③ ログイン時: Argon2id(MD5(input)) で検証      │
  │  ④ 次回ログイン時に Argon2id(password) に更新    │
  │                                              │
  │  利点: 即座に全ユーザーの保護を強化              │
  │  欠点: 実装が複雑                              │
  └──────────────────────────────────────────────┘
```

```typescript
// パスワードマイグレーション実装
import crypto from 'crypto';
import argon2 from 'argon2';

class PasswordMigration {
  // ハッシュ形式の判定
  detectHashType(hash: string): 'md5' | 'sha256' | 'bcrypt' | 'argon2' {
    if (hash.startsWith('$argon2')) return 'argon2';
    if (hash.startsWith('$2b$') || hash.startsWith('$2a$')) return 'bcrypt';
    if (hash.length === 32) return 'md5';    // 32文字のhex
    if (hash.length === 64) return 'sha256'; // 64文字のhex
    throw new Error(`Unknown hash format: ${hash.substring(0, 10)}...`);
  }

  // 旧ハッシュで検証
  async verifyLegacy(password: string, hash: string, type: string): Promise<boolean> {
    switch (type) {
      case 'md5':
        return crypto.createHash('md5').update(password).digest('hex') === hash;
      case 'sha256':
        return crypto.createHash('sha256').update(password).digest('hex') === hash;
      case 'bcrypt':
        return bcrypt.compare(password, hash);
      case 'argon2':
        return argon2.verify(hash, password);
      default:
        return false;
    }
  }

  // ログイン時の透過的マイグレーション
  async loginWithMigration(
    email: string,
    password: string
  ): Promise<{ success: boolean; user?: any }> {
    const user = await db.user.findUnique({ where: { email } });
    if (!user) return { success: false };

    const hashType = this.detectHashType(user.password);
    const isValid = await this.verifyLegacy(password, user.password, hashType);

    if (!isValid) return { success: false };

    // 旧アルゴリズムの場合、Argon2id にリハッシュ
    if (hashType !== 'argon2') {
      const newHash = await argon2.hash(password, {
        type: argon2.argon2id,
        memoryCost: 65536,
        timeCost: 3,
        parallelism: 4,
      });

      await db.user.update({
        where: { id: user.id },
        data: { password: newHash },
      });

      console.log(`Migrated password hash for user ${user.id}: ${hashType} → argon2id`);
    }

    return { success: true, user };
  }

  // 一括ラッピングマイグレーション
  async wrapAllHashes(): Promise<{ migrated: number; errors: number }> {
    let migrated = 0;
    let errors = 0;

    const users = await db.user.findMany({
      where: {
        NOT: { password: { startsWith: '$argon2' } },
      },
    });

    for (const user of users) {
      try {
        // 旧ハッシュを Argon2id でラッピング
        const wrappedHash = await argon2.hash(user.password, {
          type: argon2.argon2id,
          memoryCost: 65536,
          timeCost: 3,
          parallelism: 4,
          raw: false,
        });

        await db.user.update({
          where: { id: user.id },
          data: {
            password: wrappedHash,
            passwordWrapped: true, // ラッピング済みフラグ
          },
        });

        migrated++;
      } catch (error) {
        errors++;
        console.error(`Failed to migrate user ${user.id}:`, error);
      }
    }

    return { migrated, errors };
  }
}
```

---

## 7. アンチパターン

```
パスワード管理のアンチパターン:

  ✗ 平文保存:
    → DB漏洩で全パスワードが即座に判明
    → 法的責任を問われる可能性
    → GDPR、個人情報保護法に違反

  ✗ 可逆暗号化:
    → AES等で暗号化 → 鍵があれば復号可能
    → 鍵管理の問題が発生
    → 鍵漏洩 = 全パスワード漏洩

  ✗ MD5/SHA-256（ソルトなし）:
    → レインボーテーブルで解読可能
    → GPU で毎秒数十億回のハッシュ計算
    → 現在のハードウェアでは防御力ゼロ

  ✗ 独自のハッシュアルゴリズム:
    → 暗号の専門家でない限り脆弱性がある
    → 検証済みのライブラリを使用すべき
    → 「シンプルすぎて破られない」は幻想

  ✗ パスワードの最大長制限（例: 16文字）:
    → パスフレーズの使用を妨げる
    → ハッシュ化すれば長さは関係ない
    → bcrypt の72バイト制限はプレハッシュで対応

  ✗ パスワードをログに出力:
    → 平文パスワードがログファイルに残る
    → リクエストボディのログ記録時に特に注意
    → ログフレームワークでフィルタリング

  ✗ エラーメッセージで情報漏洩:
    → 「パスワードが間違っています」→ ユーザー存在が判明
    → 「メールアドレスまたはパスワードが間違っています」が正しい

  ✗ タイミング攻撃への無防備:
    → ユーザーが存在しない場合、即座にエラー応答
    → ハッシュ検証のない高速応答でユーザー存在が判明
    → 対策: 常にハッシュ計算を行う（ダミーでも）

  ✗ パスワード変更時に旧セッションを残す:
    → パスワード変更後も旧セッションが有効
    → アカウント侵害時に攻撃者のセッションが残る
    → パスワード変更時は全セッション無効化
```

---

## 8. 演習

### 演習 1: パスワードハッシュの比較実験（基礎）

```
課題:
  各ハッシュアルゴリズムの速度を計測し、なぜ bcrypt/Argon2 が
  パスワードに適しているかを体感する。

  要件:
  1. MD5, SHA-256, bcrypt, Argon2id でそれぞれハッシュ化
  2. 各アルゴリズムで10万回のハッシュ計算時間を計測
  3. 結果を表にまとめる
  4. なぜ「遅い」ことが利点なのかを説明

  期待される結果:
  MD5:        ~1秒 / 10万回
  SHA-256:    ~2秒 / 10万回
  bcrypt(12): ~30分 / 10万回
  Argon2id:   ~8時間 / 10万回
```

### 演習 2: 安全なパスワードリセットフローの実装（応用）

```
課題:
  Express + Prisma を使って、OWASP に準拠した
  パスワードリセットフローを実装せよ。

  要件:
  1. POST /auth/reset-request: メールアドレスでリセット要求
  2. GET /auth/reset-verify/:token: トークン有効性確認
  3. POST /auth/reset-confirm: 新パスワード設定
  4. セキュリティ要件:
     → トークンは crypto.randomBytes(32) で生成
     → DB にはハッシュ化して保存
     → 有効期限: 1時間
     → 使用済みトークンの無効化
     → ユーザー列挙攻撃の防止
     → レート制限
     → 全セッション無効化
     → 変更通知メール
```

### 演習 3: パスワードマイグレーションの実装（発展）

```
課題:
  MD5 → bcrypt → Argon2id の段階的マイグレーション機構を
  実装せよ。

  要件:
  1. ハッシュ形式の自動判定
  2. 透過的リハッシュ（ログイン時に自動移行）
  3. 一括ラッピング（MD5 hash を Argon2id で包む）
  4. マイグレーション進捗のモニタリング
  5. ロールバック可能な設計

  テストシナリオ:
  → MD5 ハッシュのユーザーがログイン → Argon2id に移行
  → bcrypt ハッシュのユーザーがログイン → Argon2id に移行
  → Argon2id の古いパラメータ → 新しいパラメータでリハッシュ
```

---

## 9. FAQ・トラブルシューティング

```
Q1: bcrypt と Argon2 どちらを使うべきか
A1: → 新規プロジェクト: Argon2id（メモリハードで GPU 耐性最強）
    → 既存プロジェクト: bcrypt のままでも十分安全
    → FIPS 準拠が必要: PBKDF2-HMAC-SHA256
    → どちらも使えない環境: scrypt

Q2: bcrypt の72バイト制限が心配
A2: → SHA-256 プレハッシュで対応: bcrypt(SHA256(password).base64())
    → Base64 出力は44文字 → 72バイト以内
    → または Argon2id に移行（長さ制限なし）

Q3: パスワードの最大長はどうすべきか
A3: → 少なくとも64文字を許容（NIST 推奨）
    → 128〜256文字を上限に（DoS 対策）
    → ハッシュ前に長さチェック（巨大な入力での計算負荷を防止）

Q4: パスワードの保存場所はどこか
A4: → ハッシュ化してメインDBに保存
    → 暗号化が必要なら「暗号化ラッピング」を追加
    → 暗号化鍵はKMS/HSMで管理
    → バックアップにも同じ保護を適用

Q5: ユーザーがパスワードを忘れた場合のフロー
A5: → リセットトークンをメールで送信
    → トークンはランダムで暗号的に安全
    → 1時間有効、使用後は無効化
    → ユーザー存在を漏らさない
    → MFA が有効ならリセット後も MFA 要求

Q6: パスワードの平文がログに残ってしまった
A6: → 即座にログファイルを安全に削除
    → 影響を受けたユーザーのパスワードリセットを強制
    → ログフレームワークにリクエストボディのフィルタリングを追加
    → 監査ログにインシデントを記録
    → セキュリティチームに報告

Q7: Argon2 のメモリ使用量でサーバーが不安定になる
A7: → parallelism を下げる（4 → 1）
    → memoryCost を下げる（65536 → 19456）
    → 同時ハッシュ計算数を制限（Semaphore）
    → ワーカースレッドで実行
    → ハッシュ計算専用のサービスに分離
```

---

## まとめ

| 項目 | ベストプラクティス |
|------|-----------------|
| ハッシュ | Argon2id（推奨）または bcrypt |
| ソルト | アルゴリズムが自動生成（手動不要） |
| ペッパー | 暗号化ラッピング（KMS で鍵管理） |
| コスト | 250ms〜1秒のハッシュ計算時間 |
| ポリシー | 8文字以上、漏洩チェック、強度メーター |
| 禁止ルール | よく使われるパスワード、ユーザー情報を含むもの |
| リセット | 暗号ランダムトークン、1時間有効、ハッシュ保存 |
| エラー | 「メールまたはパスワードが違います」（曖昧に） |
| セッション | パスワード変更時に全セッション無効化 |
| マイグレーション | 透過的リハッシュ or ラッピング |

---

## 次に読むべきガイド
→ [[02-multi-factor-authentication.md]] — 多要素認証
→ [[../02-token-auth/00-jwt-deep-dive.md]] — JWT 詳解

---

## 参考文献
1. NIST. "SP 800-63B: Digital Identity Guidelines." nist.gov, 2020.
2. OWASP. "Password Storage Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. Troy Hunt. "Have I Been Pwned." haveibeenpwned.com, 2024.
4. Dropbox. "zxcvbn: Realistic Password Strength Estimation." github.com/dropbox/zxcvbn.
5. RFC 9106. "Argon2 Memory-Hard Function for Password Hashing and Proof-of-Work Applications." IETF, 2021.
6. Niels Provos, David Mazieres. "A Future-Adaptable Password Scheme." USENIX, 1999.
7. OWASP. "Credential Stuffing Prevention Cheat Sheet." cheatsheetseries.owasp.org, 2024.
8. OWASP. "Forgot Password Cheat Sheet." cheatsheetseries.owasp.org, 2024.
