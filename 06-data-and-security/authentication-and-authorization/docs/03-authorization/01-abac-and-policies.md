# ABAC とポリシーエンジン

> RBAC では表現しきれない複雑なアクセス制御には ABAC（属性ベースアクセス制御）が必要。ユーザー属性、リソース属性、環境条件を組み合わせた動的な認可ポリシーの設計と、CASL / Oso / Cedar 等のポリシーエンジンを解説する。

## 前提知識

- [[../01-session-auth/00-cookie-and-session.md]] — Cookie とセッション管理
- [[../02-token-auth/01-jwt-basics.md]] — JWT の基礎
- RBAC の基本概念（ロール、パーミッション）
- TypeScript / JavaScript の基礎

## この章で学ぶこと

- [ ] ABAC の概念と RBAC との本質的な違いを理解する
- [ ] NIST SP 800-162 に基づく ABAC アーキテクチャを把握する
- [ ] ポリシーの設計パターンを実践的に適用できる
- [ ] CASL を使った実践的なポリシー実装を学ぶ
- [ ] OPA/Rego、Cedar などの外部ポリシーエンジンの特徴を比較できる
- [ ] ポリシーのテスト・デバッグ・運用手法を身につける

---

## 1. ABAC の基本概念

### 1.1 ABAC とは何か

ABAC（Attribute-Based Access Control）は、アクセス制御の決定を「属性」の評価に基づいて行うモデルである。RBAC（Role-Based Access Control）がユーザーに割り当てられた「ロール」によってアクセス権を決定するのに対し、ABAC はユーザー、リソース、アクション、環境の各属性を動的に評価してアクセス可否を判断する。

ABAC の最大の利点は、事前にロールを定義する必要がなく、属性の組み合わせによって極めて細かい粒度のアクセス制御が可能になる点にある。例えば「東京オフィスの部長以上が、営業時間内に、自部署の機密文書を閲覧できる」といった複合的な条件を自然に表現できる。

```
ABAC（Attribute-Based Access Control）:

  RBAC: 「ロール」に基づくアクセス制御
    → admin は全記事を編集できる
    → 粒度が粗い、ロール爆発の問題

  ABAC: 「属性」に基づくアクセス制御
    → 記事の作者は自分の記事を編集できる
    → 部署が同じマネージャーは部下の評価を閲覧できる
    → 営業時間内のみデータをエクスポートできる
    → 条件の組み合わせで細かい制御が可能

  4つの属性カテゴリ:

  ┌──────────────────────────────────────────┐
  │                                          │
  │  (1) Subject（主体）属性:                  │
  │     → ユーザーID、ロール、部署、役職         │
  │     → メール、入社日、資格、クリアランス      │
  │     → 所属グループ、マネージャーフラグ        │
  │                                          │
  │  (2) Resource（リソース）属性:              │
  │     → リソースID、タイプ、作成者             │
  │     → ステータス、分類、公開フラグ            │
  │     → 機密レベル、所有組織、作成日            │
  │                                          │
  │  (3) Action（操作）属性:                    │
  │     → read、create、update、delete         │
  │     → publish、approve、export             │
  │     → archive、share、transfer             │
  │                                          │
  │  (4) Environment（環境）属性:              │
  │     → 時刻、IPアドレス、デバイス             │
  │     → 地域、ネットワーク種別                 │
  │     → リスクスコア、MFA ステータス            │
  │                                          │
  └──────────────────────────────────────────┘

  ポリシー例:
    IF subject.role == "editor"
    AND resource.type == "article"
    AND resource.author == subject.id
    AND action == "update"
    THEN ALLOW

    IF subject.department == resource.department
    AND subject.role == "manager"
    AND action == "read"
    AND environment.time BETWEEN "09:00" AND "18:00"
    THEN ALLOW

    IF subject.clearance >= resource.classification
    AND environment.network == "corporate"
    AND environment.mfa_verified == true
    THEN ALLOW
```

### 1.2 NIST SP 800-162 の ABAC アーキテクチャ

NIST（米国国立標準技術研究所）が定義する ABAC の参照アーキテクチャは、以下のコンポーネントで構成される。

```
NIST ABAC 参照アーキテクチャ:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  ┌───────────┐     ┌───────────┐               │
  │  │   PEP     │────>│   PDP     │               │
  │  │ (Policy   │<────│ (Policy   │               │
  │  │ Enforce-  │     │ Decision  │               │
  │  │ ment      │     │ Point)    │               │
  │  │ Point)    │     └─────┬─────┘               │
  │  └───────────┘           │                     │
  │       │            ┌─────┴─────┐               │
  │       │            │   PAP     │               │
  │  アプリ層で         │ (Policy   │               │
  │  アクセス制御       │ Admin     │               │
  │  を実施            │ Point)    │               │
  │                    └─────┬─────┘               │
  │                          │                     │
  │                    ┌─────┴─────┐               │
  │                    │   PIP     │               │
  │                    │ (Policy   │               │
  │                    │ Info      │               │
  │                    │ Point)    │               │
  │                    └───────────┘               │
  │                                                 │
  └─────────────────────────────────────────────────┘

  PEP (Policy Enforcement Point):
    → アクセス要求をインターセプトする
    → PDP の決定を実施する（許可/拒否）
    → アプリケーションのミドルウェアやゲートウェイ

  PDP (Policy Decision Point):
    → ポリシーを評価してアクセス可否を決定する
    → 属性情報を PIP から取得する
    → ポリシーエンジン（OPA、Cedar、CASL 等）

  PAP (Policy Administration Point):
    → ポリシーの作成・管理・配布を行う
    → 管理者がポリシーを定義するインターフェース

  PIP (Policy Information Point):
    → 属性情報を提供する
    → ユーザーDB、リソースDB、外部サービス等
    → 環境情報（時刻、IP、デバイス）の取得
```

### 1.3 RBAC vs ABAC vs ReBAC の比較

```
アクセス制御モデルの比較:

  項目         │ RBAC           │ ABAC              │ ReBAC
  ────────────┼───────────────┼──────────────────┼──────────────────
  制御の基盤   │ ロール          │ 属性の組合せ       │ 関係（リレーション）
  制御の粒度   │ ロール単位      │ 属性の組合せ       │ オブジェクト間関係
  柔軟性       │ 低〜中          │ 高                │ 高
  複雑性       │ 低              │ 中〜高            │ 中〜高
  管理コスト   │ 低              │ 中                │ 中
  スケーラ     │ ロール爆発の     │ ポリシー複雑化の   │ グラフ探索の
  ビリティ     │ リスク          │ リスク             │ パフォーマンス
  ユースケース │ 組織の役割が     │ リソース所有者、   │ ドキュメント共有、
              │ 明確な場合       │ 条件付きアクセス   │ 階層的権限
  実装例       │ admin/editor/  │ 「自分の記事のみ   │ 「共有された
              │ viewer          │ 編集可能」         │ フォルダの中身」
  代表的       │ express-roles  │ CASL, OPA,        │ Google Zanzibar,
  ツール       │                │ Cedar             │ SpiceDB, Ory Keto

  実際のプロジェクト:
  → RBAC + ABAC のハイブリッドが一般的
  → 基本はRBAC、細かい制御にABACを追加
  → 大規模SaaS ではReBAC を併用するケースも
```

### 1.4 RBAC のロール爆発問題

RBAC が ABAC を必要とする主な理由の一つが「ロール爆発」問題である。

```
ロール爆発（Role Explosion）:

  単純な RBAC:
    admin, editor, viewer → 3 ロール

  部署を追加:
    admin_sales, admin_engineering, admin_hr,
    editor_sales, editor_engineering, editor_hr,
    viewer_sales, viewer_engineering, viewer_hr
    → 3 x 3 = 9 ロール

  地域を追加:
    admin_sales_tokyo, admin_sales_osaka, ...
    → 3 x 3 x 3 = 27 ロール

  プロジェクトを追加:
    → 3 x 3 x 3 x N = 指数関数的増加

  ABAC なら:
    ポリシー: "同じ部署 AND 同じ地域 AND ロールが editor 以上"
    → ロール数は増えない
    → 属性の組み合わせで動的に判定
```

---

## 2. CASL によるポリシー実装

### 2.1 CASL の基本

CASL（pronounced "castle"）は JavaScript/TypeScript 向けのポリシーライブラリで、ABAC スタイルのアクセス制御を宣言的に定義できる。MongoDB のクエリ構文と互換性があり、フロントエンド・バックエンドの両方で同じポリシー定義を共有できる点が大きな特徴である。

```typescript
// CASL: JavaScript/TypeScript のポリシーライブラリ
// npm install @casl/ability

import {
  AbilityBuilder,
  createMongoAbility,
  MongoAbility,
  subject,
  ForbiddenError,
} from '@casl/ability';

// アクションとサブジェクトの型定義
type Actions = 'read' | 'create' | 'update' | 'delete' | 'publish' | 'manage';
type Subjects = 'Article' | 'User' | 'Comment' | 'Organization' | 'all';
type AppAbility = MongoAbility<[Actions, Subjects]>;

// ユーザー型
interface User {
  id: string;
  role: string;
  orgId: string;
  department?: string;
  permissions?: string[];
}

// ユーザーのロールに基づくAbility定義
function defineAbilityFor(user: User): AppAbility {
  const { can, cannot, build } = new AbilityBuilder<AppAbility>(createMongoAbility);

  switch (user.role) {
    case 'super_admin':
      can('manage', 'all'); // 全権限
      break;

    case 'admin':
      can('manage', 'Article');
      can('read', 'User');
      can('create', 'User');
      can('update', 'User', { orgId: user.orgId }); // 同じ組織のみ
      cannot('delete', 'User'); // ユーザー削除は super_admin のみ
      can('manage', 'Comment');
      can('read', 'Organization', { id: user.orgId });
      can('update', 'Organization', { id: user.orgId });
      break;

    case 'editor':
      can('read', 'Article');
      can('create', 'Article');
      can('update', 'Article', { authorId: user.id }); // 自分の記事のみ
      can('delete', 'Article', { authorId: user.id }); // 自分の記事のみ
      can('read', 'Comment');
      can('create', 'Comment');
      can('update', 'Comment', { authorId: user.id });
      can('delete', 'Comment', { authorId: user.id });
      break;

    case 'viewer':
      can('read', 'Article', { status: 'published' }); // 公開記事のみ
      can('read', 'Comment');
      can('create', 'Comment');
      can('update', 'Comment', { authorId: user.id }); // 自分のコメントのみ
      can('delete', 'Comment', { authorId: user.id });
      break;
  }

  return build();
}

// 使用例
const ability = defineAbilityFor({ id: 'user_1', role: 'editor', orgId: 'org_1' });

ability.can('read', 'Article');    // true
ability.can('create', 'Article');  // true
ability.can('update', subject('Article', { authorId: 'user_1' })); // true（自分の記事）
ability.can('update', subject('Article', { authorId: 'user_2' })); // false（他人の記事）
ability.can('publish', 'Article'); // false（editor には publish 権限なし）
```

### 2.2 CASL の条件構文の詳細

CASL は MongoDB のクエリ構文をサポートしており、複雑な条件を表現できる。

```typescript
// CASL の MongoDB 互換条件構文
function defineAdvancedAbility(user: User): AppAbility {
  const { can, cannot, build } = new AbilityBuilder<AppAbility>(createMongoAbility);

  // $eq: 等値比較（デフォルト）
  can('read', 'Article', { status: 'published' });

  // $ne: 否定
  can('update', 'Article', { status: { $ne: 'archived' } });

  // $in: 配列内のいずれかに一致
  can('read', 'Article', { category: { $in: ['tech', 'science'] } });

  // $nin: 配列内のいずれにも一致しない
  can('read', 'Article', { sensitivity: { $nin: ['confidential', 'top_secret'] } });

  // $gt, $gte, $lt, $lte: 比較演算
  can('read', 'Article', { priority: { $gte: 1, $lte: 5 } });

  // $exists: フィールドの存在チェック
  can('read', 'Article', { deletedAt: { $exists: false } });

  // $regex: 正規表現マッチ
  can('read', 'Article', { title: { $regex: /^公開/ } });

  // $all: 配列の全要素を含む
  can('read', 'Article', { tags: { $all: ['approved', 'reviewed'] } });

  // $elemMatch: 配列要素の条件
  can('read', 'Article', {
    collaborators: { $elemMatch: { userId: user.id, role: 'editor' } },
  });

  // 複数条件の組み合わせ（AND）
  can('update', 'Article', {
    authorId: user.id,
    status: { $ne: 'archived' },
    orgId: user.orgId,
  });

  // cannot で明示的に拒否（can よりも優先）
  cannot('delete', 'Article', { status: 'published' });

  return build();
}
```

### 2.3 カスタムフィールドマッチャー

デフォルトの MongoDB 構文では表現できない条件にはカスタムマッチャーを使用する。

```typescript
import { createMongoAbility, MongoAbility, AbilityBuilder } from '@casl/ability';
import { buildMongoQueryMatcher } from '@casl/ability/extra';

// カスタム演算子: $today（今日かどうか）
const customConditionsMatcher = buildMongoQueryMatcher({
  $today: (value: boolean, fieldValue: Date) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const isToday = fieldValue >= today && fieldValue < tomorrow;
    return value ? isToday : !isToday;
  },
  $withinHours: (hours: number, fieldValue: Date) => {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    return fieldValue >= cutoff;
  },
});

// カスタムマッチャーを使った Ability
const ability = createMongoAbility<[Actions, Subjects]>(
  [
    {
      action: 'update',
      subject: 'Article',
      conditions: { createdAt: { $withinHours: 24 } }, // 作成24時間以内のみ編集可
    },
    {
      action: 'delete',
      subject: 'Comment',
      conditions: { createdAt: { $today: true } }, // 今日作成したコメントのみ削除可
    },
  ],
  { conditionsMatcher: customConditionsMatcher }
);
```

---

## 3. API での権限チェック

### 3.1 Express ミドルウェアでの実装

```typescript
// CASL + Express ミドルウェア
import { ForbiddenError, subject } from '@casl/ability';

// 汎用権限チェックミドルウェア
function authorize(action: Actions, subjectType: Subjects) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const ability = defineAbilityFor(req.user);

    try {
      // サブジェクトタイプのみのチェック（条件なし）
      ForbiddenError.from(ability).throwUnlessCan(action, subjectType);
      next();
    } catch (error) {
      if (error instanceof ForbiddenError) {
        return res.status(403).json({
          error: 'Forbidden',
          message: `Cannot ${error.action} ${error.subjectType}`,
          reason: error.message,
        });
      }
      next(error);
    }
  };
}

// リソースベースの権限チェック（属性付き）
function authorizeResource<T>(
  action: Actions,
  subjectType: Subjects,
  getResource: (req: Request) => Promise<T | null>
) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const ability = defineAbilityFor(req.user);

    // リソースの取得
    const resource = await getResource(req);
    if (!resource) {
      return res.status(404).json({ error: 'Resource not found' });
    }

    try {
      // 属性ベースのチェック
      ForbiddenError.from(ability).throwUnlessCan(
        action,
        subject(subjectType, resource as Record<string, unknown>)
      );
      (req as any).resource = resource;
      next();
    } catch (error) {
      if (error instanceof ForbiddenError) {
        return res.status(403).json({
          error: 'Forbidden',
          message: `Cannot ${error.action} ${error.subjectType}`,
        });
      }
      next(error);
    }
  };
}

// 記事の更新API
app.put(
  '/api/articles/:id',
  authorizeResource('update', 'Article', async (req) =>
    prisma.article.findUnique({ where: { id: req.params.id } })
  ),
  async (req, res) => {
    const article = (req as any).resource;

    const updated = await prisma.article.update({
      where: { id: req.params.id },
      data: req.body,
    });

    res.json(updated);
  }
);

// 記事一覧API（ポリシーに基づくフィルタリング）
app.get('/api/articles', async (req, res) => {
  const ability = defineAbilityFor(req.user);

  // CASL のルールから Prisma の where 条件を構築
  const articles = await prisma.article.findMany({
    where: accessibleBy(ability, 'read').Article,
    orderBy: { createdAt: 'desc' },
    take: 20,
  });

  res.json(articles);
});

// グローバルエラーハンドラー
app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
  if (error instanceof ForbiddenError) {
    return res.status(403).json({
      error: 'Forbidden',
      message: `Cannot ${error.action} ${error.subjectType}`,
    });
  }
  next(error);
});
```

### 3.2 Prisma との統合（accessibleBy）

CASL は Prisma と統合して、ポリシーに基づいたデータベースクエリのフィルタリングを自動生成できる。

```typescript
// npm install @casl/prisma
import { accessibleBy } from '@casl/prisma';

// CASL のルールから Prisma の where 条件を自動生成
async function getAccessibleArticles(user: User) {
  const ability = defineAbilityFor(user);

  // ability の 'read' ルールから where 条件を自動構築
  // editor の場合:
  //   can('read', 'Article')
  //   → where: {} （制限なし）
  //
  // viewer の場合:
  //   can('read', 'Article', { status: 'published' })
  //   → where: { status: 'published' }
  const articles = await prisma.article.findMany({
    where: accessibleBy(ability, 'read').Article,
  });

  return articles;
}

// 更新可能な記事のみ取得
async function getEditableArticles(user: User) {
  const ability = defineAbilityFor(user);

  // editor: can('update', 'Article', { authorId: user.id })
  // → where: { authorId: user.id }
  const articles = await prisma.article.findMany({
    where: accessibleBy(ability, 'update').Article,
  });

  return articles;
}

// 安全な更新（権限チェック + データ取得を統合）
async function safeUpdateArticle(user: User, articleId: string, data: any) {
  const ability = defineAbilityFor(user);

  // accessibleBy で権限のある記事のみを対象にする
  const article = await prisma.article.findFirst({
    where: {
      id: articleId,
      ...accessibleBy(ability, 'update').Article,
    },
  });

  if (!article) {
    throw new Error('Article not found or no permission');
  }

  return prisma.article.update({
    where: { id: articleId },
    data,
  });
}
```

---

## 4. フロントエンドでの権限制御

### 4.1 React + CASL の統合

```typescript
// React + CASL
import { createContext, useContext, useMemo } from 'react';
import { createContextualCan } from '@casl/react';
import { createMongoAbility, MongoAbility } from '@casl/ability';

// Ability コンテキスト
const AbilityContext = createContext<AppAbility>(undefined!);
const Can = createContextualCan(AbilityContext.Consumer);

// プロバイダー
function AbilityProvider({ children }: { children: React.ReactNode }) {
  const { data: user } = useUser();
  const ability = useMemo(() => {
    if (!user) return createMongoAbility<[Actions, Subjects]>([]);
    return defineAbilityFor(user);
  }, [user]);

  return (
    <AbilityContext.Provider value={ability}>
      {children}
    </AbilityContext.Provider>
  );
}

// Can コンポーネントで UI を条件表示
function ArticleActions({ article }: { article: Article }) {
  return (
    <div className="flex gap-2">
      <Can I="read" this={subject('Article', article)}>
        <button>View</button>
      </Can>

      <Can I="update" this={subject('Article', article)}>
        <button>Edit</button>
      </Can>

      <Can I="delete" this={subject('Article', article)}>
        <button className="text-red-500">Delete</button>
      </Can>

      <Can I="publish" a="Article">
        <button className="text-green-500">Publish</button>
      </Can>

      {/* not プロパティで否定条件 */}
      <Can not I="update" this={subject('Article', article)}>
        <span className="text-gray-400">編集権限なし</span>
      </Can>
    </div>
  );
}

// フック
function useAbility() {
  return useContext(AbilityContext);
}

// useAbility フックを使った条件分岐
function ArticlePage({ article }: { article: Article }) {
  const ability = useAbility();

  if (ability.cannot('read', subject('Article', article))) {
    return <div>この記事を閲覧する権限がありません</div>;
  }

  return (
    <div>
      <ArticleContent article={article} />
      {ability.can('update', subject('Article', article)) && (
        <EditButton articleId={article.id} />
      )}
    </div>
  );
}
```

### 4.2 サーバーからのポリシー同期

フロントエンドとバックエンドでポリシーを一致させるためのアプローチ。

```typescript
// バックエンド: ポリシールールを API で公開
// GET /api/auth/permissions
app.get('/api/auth/permissions', async (req, res) => {
  const ability = defineAbilityFor(req.user);

  // CASL のルールを JSON シリアライズ可能な形式で返す
  res.json({
    rules: ability.rules,
  });
});

// フロントエンド: サーバーのルールで Ability を再構築
import { createMongoAbility } from '@casl/ability';
import { unpackRules } from '@casl/ability/extra';

function useServerAbility() {
  const { data, error } = useSWR('/api/auth/permissions', fetcher);

  const ability = useMemo(() => {
    if (!data?.rules) {
      return createMongoAbility<[Actions, Subjects]>([]);
    }

    return createMongoAbility<[Actions, Subjects]>(data.rules);
  }, [data]);

  return { ability, isLoading: !data && !error };
}

// Ability の更新を検知して再レンダリング
function useAbilityUpdate(ability: AppAbility) {
  const [, forceUpdate] = useState(0);

  useEffect(() => {
    // ability.rules が変更されたら再レンダリング
    const unsubscribe = ability.on('updated', () => {
      forceUpdate((n) => n + 1);
    });

    return () => unsubscribe();
  }, [ability]);
}
```

### 4.3 フロントエンド権限制御の注意点

```
フロントエンドでの権限制御の注意点:

  ┌────────────────────────────────────────────────┐
  │                                                │
  │  重要: フロントエンドの権限制御は UX のため       │
  │        セキュリティのためではない                  │
  │                                                │
  │  フロントエンド:                                 │
  │    → ボタンの表示/非表示                         │
  │    → メニュー項目のフィルタリング                  │
  │    → 権限のないページへのリダイレクト               │
  │    → DevTools で簡単に回避可能                    │
  │                                                │
  │  バックエンド（必須）:                            │
  │    → API レベルでの権限チェック                    │
  │    → データベースクエリのフィルタリング              │
  │    → 回避不可能なセキュリティ境界                   │
  │                                                │
  │  結論: 両方で同じポリシーを適用する                 │
  │        CASL のルール共有が効果的                    │
  │                                                │
  └────────────────────────────────────────────────┘
```

---

## 5. ポリシーの設計パターン

### 5.1 基本パターン

```
ポリシー設計のパターン:

  (1) リソースオーナーシップ:
     → 作成者は自分のリソースを操作可能
     → can('update', 'Article', { authorId: user.id })
     → 最も一般的なパターン

  (2) 組織スコープ:
     → 同じ組織内のリソースのみアクセス可能
     → can('read', 'Article', { orgId: user.orgId })
     → マルチテナント SaaS の基本

  (3) ステータスベース:
     → ステータスに応じた操作制限
     → can('update', 'Article', { status: { $ne: 'archived' } })
     → 公開済みは管理者のみ編集可能

  (4) 時間ベース:
     → 特定時間帯のみ許可
     → ポリシー定義時に現在時刻をチェック
     → 環境属性として実装

  (5) 承認フロー:
     → 下書き → レビュー → 承認 → 公開
     → 各ステージで操作可能なロールが異なる
     → ステートマシンと組み合わせる

  (6) 階層的権限:
     → 部門の長は部門配下の全リソースにアクセス可能
     → 組織ツリーを辿って権限を解決
     → ReBAC 的なアプローチが必要になることもある

  (7) 委任権限:
     → 権限の一時的な委任（代理承認等）
     → 有効期限付きの一時的な権限拡張
     → 監査ログとの連携が重要

  設計の原則:
  → デフォルトは拒否（明示的に許可したもののみ）
  → 最小権限の原則
  → ポリシーをコードとして管理（バージョン管理可能）
  → テスト可能（ポリシーのユニットテスト）
  → 監査可能（誰が何にアクセスしたか記録）
```

### 5.2 実践的な承認フローの実装

```typescript
// 承認フローを ABAC で実装
interface WorkflowStage {
  status: string;
  allowedActions: Record<string, string[]>; // ロール → 許可アクション
}

const articleWorkflow: WorkflowStage[] = [
  {
    status: 'draft',
    allowedActions: {
      author: ['read', 'update', 'delete', 'submit'],
      editor: ['read'],
      admin: ['read', 'update', 'delete'],
    },
  },
  {
    status: 'review',
    allowedActions: {
      author: ['read'],
      editor: ['read', 'approve', 'reject'],
      admin: ['read', 'approve', 'reject', 'update'],
    },
  },
  {
    status: 'approved',
    allowedActions: {
      author: ['read'],
      editor: ['read', 'publish'],
      admin: ['read', 'publish', 'update', 'reject'],
    },
  },
  {
    status: 'published',
    allowedActions: {
      author: ['read'],
      editor: ['read'],
      admin: ['read', 'update', 'unpublish', 'archive'],
    },
  },
];

// ワークフロー対応の Ability 定義
function defineWorkflowAbility(user: User, article: Article): AppAbility {
  const { can, build } = new AbilityBuilder<AppAbility>(createMongoAbility);

  const stage = articleWorkflow.find((s) => s.status === article.status);
  if (!stage) return build();

  // ロールに基づく権限
  const roleActions = stage.allowedActions[user.role] || [];
  for (const action of roleActions) {
    can(action as Actions, 'Article', { id: article.id });
  }

  // 作者の場合の追加権限
  if (article.authorId === user.id) {
    const authorActions = stage.allowedActions['author'] || [];
    for (const action of authorActions) {
      can(action as Actions, 'Article', { id: article.id });
    }
  }

  return build();
}
```

### 5.3 環境属性を使ったポリシー

```typescript
// 環境情報を含むポリシー定義
interface EnvironmentContext {
  time: Date;
  ipAddress: string;
  userAgent: string;
  isMfaVerified: boolean;
  riskScore: number;
  networkType: 'corporate' | 'vpn' | 'public';
}

function defineAbilityWithEnvironment(
  user: User,
  env: EnvironmentContext
): AppAbility {
  const { can, cannot, build } = new AbilityBuilder<AppAbility>(createMongoAbility);

  // 基本的な読み取り権限
  can('read', 'Article', { orgId: user.orgId });

  // 営業時間内のみデータエクスポートを許可
  const hour = env.time.getHours();
  const isBusinessHours = hour >= 9 && hour < 18;
  const isWeekday = env.time.getDay() >= 1 && env.time.getDay() <= 5;

  if (isBusinessHours && isWeekday) {
    can('export', 'Article' as any, { orgId: user.orgId });
  }

  // 社内ネットワークからのみ機密データへのアクセスを許可
  if (env.networkType === 'corporate' || env.networkType === 'vpn') {
    can('read', 'Article', { classification: 'confidential', orgId: user.orgId });
  }

  // MFA 認証済みの場合のみ管理操作を許可
  if (env.isMfaVerified) {
    can('update', 'User', { orgId: user.orgId });
    can('delete', 'Article', { authorId: user.id });
  }

  // 高リスクスコアの場合は機密操作を拒否
  if (env.riskScore > 70) {
    cannot('export', 'Article' as any);
    cannot('delete', 'Article');
    cannot('update', 'User');
  }

  return build();
}

// 環境情報の取得
function getEnvironmentContext(req: Request): EnvironmentContext {
  return {
    time: new Date(),
    ipAddress: getClientIP(req),
    userAgent: req.headers.get('user-agent') || '',
    isMfaVerified: req.session?.mfaVerified ?? false,
    riskScore: calculateRiskScore(req),
    networkType: classifyNetwork(getClientIP(req)),
  };
}
```

---

## 6. ポリシーのテスト

### 6.1 ユニットテスト

```typescript
// ポリシーのテスト
import { describe, it, expect } from 'vitest';
import { subject } from '@casl/ability';

describe('Editor permissions', () => {
  const ability = defineAbilityFor({
    id: 'user_1',
    role: 'editor',
    orgId: 'org_1',
  });

  it('can read articles', () => {
    expect(ability.can('read', 'Article')).toBe(true);
  });

  it('can create articles', () => {
    expect(ability.can('create', 'Article')).toBe(true);
  });

  it('can update own articles', () => {
    expect(
      ability.can('update', subject('Article', { authorId: 'user_1' }))
    ).toBe(true);
  });

  it('cannot update others articles', () => {
    expect(
      ability.can('update', subject('Article', { authorId: 'user_2' }))
    ).toBe(false);
  });

  it('cannot publish articles', () => {
    expect(ability.can('publish', 'Article')).toBe(false);
  });

  it('cannot manage users', () => {
    expect(ability.can('update', 'User')).toBe(false);
  });

  it('can manage own comments', () => {
    expect(
      ability.can('update', subject('Comment', { authorId: 'user_1' }))
    ).toBe(true);
    expect(
      ability.can('delete', subject('Comment', { authorId: 'user_1' }))
    ).toBe(true);
  });

  it('cannot manage others comments', () => {
    expect(
      ability.can('update', subject('Comment', { authorId: 'user_2' }))
    ).toBe(false);
  });
});

describe('Viewer permissions', () => {
  const ability = defineAbilityFor({
    id: 'user_2',
    role: 'viewer',
    orgId: 'org_1',
  });

  it('can only read published articles', () => {
    expect(
      ability.can('read', subject('Article', { status: 'published' }))
    ).toBe(true);
    expect(
      ability.can('read', subject('Article', { status: 'draft' }))
    ).toBe(false);
  });

  it('cannot create articles', () => {
    expect(ability.can('create', 'Article')).toBe(false);
  });
});

describe('Admin permissions', () => {
  const ability = defineAbilityFor({
    id: 'admin_1',
    role: 'admin',
    orgId: 'org_1',
  });

  it('can manage articles', () => {
    expect(ability.can('manage', 'Article')).toBe(true);
  });

  it('can update users in same org', () => {
    expect(
      ability.can('update', subject('User', { orgId: 'org_1' }))
    ).toBe(true);
  });

  it('cannot update users in different org', () => {
    expect(
      ability.can('update', subject('User', { orgId: 'org_2' }))
    ).toBe(false);
  });

  it('cannot delete users', () => {
    expect(ability.can('delete', 'User')).toBe(false);
  });
});

describe('Super admin permissions', () => {
  const ability = defineAbilityFor({
    id: 'super_1',
    role: 'super_admin',
    orgId: 'org_1',
  });

  it('can manage everything', () => {
    expect(ability.can('manage', 'all')).toBe(true);
    expect(ability.can('delete', 'User')).toBe(true);
    expect(ability.can('update', subject('User', { orgId: 'org_99' }))).toBe(true);
  });
});
```

### 6.2 ポリシーマトリクステスト

包括的なテストを行うためのマトリクスベースのアプローチ。

```typescript
// ポリシーマトリクステスト — 全ロール x 全アクション の組み合わせを検証
describe('Permission matrix', () => {
  type PermissionMatrix = {
    [role: string]: {
      [action: string]: {
        [subject: string]: boolean | 'conditional';
      };
    };
  };

  const expectedMatrix: PermissionMatrix = {
    super_admin: {
      manage: { all: true },
    },
    admin: {
      read: { Article: true, User: true, Comment: true, Organization: 'conditional' },
      create: { Article: true, User: true, Comment: true },
      update: { Article: true, User: 'conditional', Comment: true, Organization: 'conditional' },
      delete: { Article: true, User: false, Comment: true },
      publish: { Article: true },
    },
    editor: {
      read: { Article: true, User: false, Comment: true },
      create: { Article: true, Comment: true },
      update: { Article: 'conditional', Comment: 'conditional' },
      delete: { Article: 'conditional', Comment: 'conditional' },
      publish: { Article: false },
    },
    viewer: {
      read: { Article: 'conditional', Comment: true },
      create: { Article: false, Comment: true },
      update: { Article: false, Comment: 'conditional' },
      delete: { Article: false },
    },
  };

  for (const [role, actions] of Object.entries(expectedMatrix)) {
    describe(`${role} role`, () => {
      const ability = defineAbilityFor({
        id: `${role}_user`,
        role,
        orgId: 'org_1',
      });

      for (const [action, subjects] of Object.entries(actions)) {
        for (const [subjectType, expected] of Object.entries(subjects)) {
          if (expected === true) {
            it(`can ${action} ${subjectType}`, () => {
              expect(ability.can(action as Actions, subjectType as Subjects)).toBe(true);
            });
          } else if (expected === false) {
            it(`cannot ${action} ${subjectType}`, () => {
              expect(ability.can(action as Actions, subjectType as Subjects)).toBe(false);
            });
          }
          // 'conditional' は個別テストで検証
        }
      }
    });
  }
});
```

---

## 7. RBAC + ABAC ハイブリッド

### 7.1 ハイブリッドアーキテクチャ

```
実践的なハイブリッドアプローチ:

  第1層: RBAC（粗い制御）
    → ロールで基本的なアクセス範囲を決定
    → admin, editor, viewer
    → ミドルウェアでチェック
    → 「この API エンドポイントにアクセスできるか？」

  第2層: ABAC（細かい制御）
    → リソースの属性で追加の制約
    → 自分の記事のみ編集可能
    → 同じ組織のデータのみ
    → サービス層でチェック
    → 「この特定のリソースを操作できるか？」

  第3層: ビジネスルール
    → 時間帯制限、承認フロー
    → APIレート制限、データ量制限
    → ドメインロジック内でチェック
    → 「この操作はビジネス上許可されているか？」

  リクエスト処理の流れ:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  Request                                         │
  │    │                                             │
  │    ▼                                             │
  │  ┌────────────────────┐                          │
  │  │ 認証ミドルウェア      │ ← ユーザーの特定        │
  │  └────────┬───────────┘                          │
  │           ▼                                      │
  │  ┌────────────────────┐                          │
  │  │ RBAC チェック        │ ← ロールベースの制御     │
  │  │ (ミドルウェア)       │    requireRole('editor') │
  │  └────────┬───────────┘                          │
  │           ▼                                      │
  │  ┌────────────────────┐                          │
  │  │ リソース取得          │ ← DB からリソースを取得   │
  │  └────────┬───────────┘                          │
  │           ▼                                      │
  │  ┌────────────────────┐                          │
  │  │ ABAC チェック        │ ← 属性ベースの制御       │
  │  │ (サービス層)         │    ability.can('update', │
  │  │                    │    subject('Article',    │
  │  │                    │    article))             │
  │  └────────┬───────────┘                          │
  │           ▼                                      │
  │  ┌────────────────────┐                          │
  │  │ ビジネスルール       │ ← 業務ロジックの制約      │
  │  │ (ドメイン層)         │                         │
  │  └────────┬───────────┘                          │
  │           ▼                                      │
  │  Response                                        │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 7.2 ハイブリッド実装

```typescript
// ハイブリッドアクセス制御の完全な実装例

// 第1層: RBAC ミドルウェア
function requireRole(...allowedRoles: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Authentication required' });
    }
    if (!allowedRoles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient role' });
    }
    next();
  };
}

// 第2層: ABAC サービス
class ArticleService {
  async update(user: User, articleId: string, data: UpdateArticleInput) {
    const ability = defineAbilityFor(user);
    const article = await prisma.article.findUnique({ where: { id: articleId } });

    if (!article) throw new NotFoundError('Article not found');

    // ABAC チェック
    ForbiddenError.from(ability).throwUnlessCan(
      'update',
      subject('Article', article)
    );

    // 第3層: ビジネスルール
    if (article.status === 'published' && !user.permissions?.includes('edit_published')) {
      throw new BusinessRuleError('Published articles require special permission to edit');
    }

    return prisma.article.update({ where: { id: articleId }, data });
  }
}

// API ルート（3層を組み合わせ）
app.put(
  '/api/articles/:id',
  requireRole('editor', 'admin', 'super_admin'),  // 第1層: RBAC
  async (req, res, next) => {
    try {
      const result = await articleService.update(  // 第2層 + 第3層
        req.user,
        req.params.id,
        req.body
      );
      res.json(result);
    } catch (error) {
      next(error);
    }
  }
);
```

---

## 8. 外部ポリシーエンジンとの比較

### 8.1 OPA (Open Policy Agent) / Rego

```
OPA (Open Policy Agent):

  概要: CNCF 卒業プロジェクトの汎用ポリシーエンジン
  言語: Rego（独自のポリシー言語）
  デプロイ: サイドカー、ライブラリ、REST API
  用途: Kubernetes、API ゲートウェイ、マイクロサービス

  Rego ポリシー例:

    package authz

    default allow = false

    allow {
      input.action == "read"
      input.resource.type == "article"
      input.resource.status == "published"
    }

    allow {
      input.action == "update"
      input.resource.type == "article"
      input.resource.author_id == input.subject.id
    }

    allow {
      input.subject.role == "admin"
      input.resource.org_id == input.subject.org_id
    }
```

### 8.2 Cedar (AWS)

```
Cedar (Amazon Verified Permissions):

  概要: AWS が開発したポリシー言語・エンジン
  言語: Cedar（型安全なポリシー言語）
  特徴: 形式検証可能、高速な評価、SDK 提供
  用途: AWS Verified Permissions、アプリケーション

  Cedar ポリシー例:

    permit(
      principal in Group::"editors",
      action == Action::"update",
      resource
    )
    when {
      resource.author == principal &&
      resource.status != "archived"
    };

    forbid(
      principal,
      action == Action::"delete",
      resource
    )
    when {
      resource.status == "published"
    };
```

### 8.3 ポリシーエンジンの比較

```
ポリシーエンジン比較表:

  項目           │ CASL          │ OPA/Rego      │ Cedar         │ Casbin
  ──────────────┼──────────────┼──────────────┼──────────────┼──────────────
  言語           │ JavaScript/TS│ Rego（独自）   │ Cedar（独自） │ 設定ファイル
  実行環境       │ ブラウザ/Node │ Go バイナリ   │ Rust バイナリ │ 多言語 SDK
  学習コスト     │ 低            │ 中〜高        │ 中            │ 低〜中
  型安全性       │ TypeScript    │ なし          │ あり          │ なし
  フロント対応   │ ✓（React等）  │ ✗             │ ✗            │ ✗
  DB 統合       │ ✓（Prisma等） │ ✗             │ ✗            │ 限定的
  パフォーマンス │ 高速（同一    │ 高速（Go      │ 非常に高速   │ 高速
               │ プロセス）     │ サイドカー）   │ （Rust）     │
  形式検証       │ ✗             │ 限定的        │ ✓            │ ✗
  ユースケース   │ Web アプリ    │ K8s/マイクロ  │ AWS 連携     │ 汎用
               │               │ サービス      │              │
  ライセンス     │ MIT           │ Apache 2.0   │ Apache 2.0   │ Apache 2.0

  選定ガイド:
    単一 Web アプリ → CASL（フロント共有可能）
    マイクロサービス → OPA（サイドカーパターン）
    AWS 環境 → Cedar（Verified Permissions）
    多言語環境 → Casbin（SDK が豊富）
```

---

## 9. アンチパターン

### 9.1 よくある間違い

```
ABAC アンチパターン:

  (1) フロントエンドのみの権限制御
     ✗ ボタンを非表示にしただけ
     → API 直接叩けば操作可能
     → 必ずバックエンドでも検証する

     ✗ 悪い例:
       // フロントでボタン非表示
       {user.role === 'admin' && <DeleteButton />}
       // しかし API に権限チェックがない
       app.delete('/api/articles/:id', async (req, res) => {
         await prisma.article.delete({ where: { id: req.params.id } });
       });

     ✓ 良い例:
       // フロントでボタン非表示 + API でも権限チェック
       app.delete('/api/articles/:id', async (req, res) => {
         const ability = defineAbilityFor(req.user);
         ForbiddenError.from(ability).throwUnlessCan('delete', ...);
         await prisma.article.delete({ where: { id: req.params.id } });
       });

  (2) ハードコードされた権限チェック
     ✗ 悪い例:
       if (user.role === 'admin' || user.role === 'editor') {
         // 操作を許可
       }
     → ロールが増えるたびにコード修正が必要
     → 散在するチェックで不整合が発生

     ✓ 良い例:
       if (ability.can('update', subject('Article', article))) {
         // 操作を許可
       }
     → ポリシーを一箇所で管理

  (3) 過度に複雑なポリシー
     ✗ 100個以上の条件が絡み合うポリシー
     → デバッグが困難
     → パフォーマンスに影響
     → ポリシーの階層化・モジュール化で対処

  (4) テスト不足
     ✗ ポリシーのユニットテストがない
     → 変更時に意図しない権限漏れが発生
     → 必ずポリシーマトリクステストを書く

  (5) 監査ログの欠如
     ✗ 誰がいつ何にアクセスしたか記録されていない
     → セキュリティインシデント時の調査が不可能
     → アクセス制御の決定を必ずログに記録する
```

### 9.2 パフォーマンスの落とし穴

```
パフォーマンス上の注意:

  (1) N+1 問題
     ✗ リスト表示で各アイテムに個別に権限チェック
       for (const article of articles) {
         if (ability.can('read', subject('Article', article))) { ... }
       }
     → アイテム数に比例して処理時間が増加

     ✓ DB レベルでフィルタリング
       const articles = await prisma.article.findMany({
         where: accessibleBy(ability, 'read').Article,
       });

  (2) 属性取得のコスト
     ✗ 毎回 DB から全属性を取得
     → 必要な属性のみ select する
     → キャッシュを活用する

  (3) ポリシー評価のキャッシュ
     → 同じユーザー・同じリソースタイプの評価結果をキャッシュ
     → TTL は短めに設定（権限変更の即時反映のため）
     → ユーザーの権限変更時にキャッシュ無効化
```

---

## 10. 監査ログとコンプライアンス

```typescript
// ABAC アクセス制御の監査ログ
interface AuditLogEntry {
  timestamp: Date;
  userId: string;
  action: string;
  resourceType: string;
  resourceId: string;
  decision: 'allow' | 'deny';
  reason?: string;
  attributes: {
    subject: Record<string, any>;
    resource: Record<string, any>;
    environment: Record<string, any>;
  };
  ipAddress: string;
  userAgent: string;
}

class AuditLogger {
  async log(entry: AuditLogEntry) {
    // 構造化ログとして出力
    console.log(JSON.stringify({
      level: entry.decision === 'deny' ? 'warn' : 'info',
      message: `ABAC ${entry.decision}: ${entry.userId} ${entry.action} ${entry.resourceType}/${entry.resourceId}`,
      ...entry,
    }));

    // DB に永続化
    await prisma.auditLog.create({
      data: {
        timestamp: entry.timestamp,
        userId: entry.userId,
        action: entry.action,
        resourceType: entry.resourceType,
        resourceId: entry.resourceId,
        decision: entry.decision,
        reason: entry.reason,
        metadata: entry.attributes,
        ipAddress: entry.ipAddress,
        userAgent: entry.userAgent,
      },
    });
  }
}

// 監査付き権限チェック
async function authorizeWithAudit(
  user: User,
  action: Actions,
  resourceType: Subjects,
  resource: Record<string, any>,
  req: Request
): Promise<boolean> {
  const ability = defineAbilityFor(user);
  const allowed = ability.can(action, subject(resourceType, resource));

  await auditLogger.log({
    timestamp: new Date(),
    userId: user.id,
    action,
    resourceType,
    resourceId: resource.id,
    decision: allowed ? 'allow' : 'deny',
    attributes: {
      subject: { id: user.id, role: user.role, orgId: user.orgId },
      resource: { id: resource.id, authorId: resource.authorId, status: resource.status },
      environment: { time: new Date().toISOString(), ip: getClientIP(req) },
    },
    ipAddress: getClientIP(req),
    userAgent: req.headers.get('user-agent') || '',
  });

  return allowed;
}
```

---

## 11. 演習問題

### 演習 1: 基本的な ABAC ポリシー（基礎）

以下の要件を CASL で実装せよ。

```
要件:
- viewer: 公開済み記事のみ読める
- editor: 全記事を読める、自分の記事を編集・削除できる
- admin: 全記事を管理でき、同じ組織のユーザーを管理できる
- 全ロール: 自分のコメントのみ編集・削除できる
- アーカイブ済み記事は誰も編集できない

テスト:
- 各ロールで期待通りの権限があることを vitest で検証
- 最低 10 テストケースを作成
```

### 演習 2: 承認ワークフロー（応用）

以下の承認フローを ABAC で実装せよ。

```
要件:
- 記事のステータス: draft → submitted → reviewed → approved → published
- 各ステータスで操作可能なロールが異なる
- 作者は draft のみ編集可能
- レビュアーは submitted を approved/rejected にできる
- 管理者はどのステータスでも操作可能
- ステータス遷移は一方向のみ（戻しは reject で draft に戻す）

実装:
- ステートマシンと CASL を組み合わせる
- API エンドポイント: POST /api/articles/:id/transition
- テスト: 全ステータス x 全ロール の遷移マトリクスを検証
```

### 演習 3: マルチテナント ABAC（発展）

マルチテナント SaaS における ABAC を設計・実装せよ。

```
要件:
- 組織ごとに異なるポリシーを定義可能
- カスタムロールの作成（組織管理者が定義）
- リソースの組織スコープ（他組織のデータにアクセス不可）
- 組織間コラボレーション（招待されたリソースのみアクセス可能）
- ポリシーの監査ログ

設計:
- ポリシーの保存方法（DB スキーマ）
- カスタムロールの Ability への変換
- パフォーマンス最適化（キャッシュ戦略）

実装:
- Prisma + CASL で完全に動作するプロトタイプ
- 組織管理画面のモックAPI
- テスト: 組織間のデータ分離を検証
```

---

## 12. FAQ・トラブルシューティング

### Q1: CASL で「Cannot read properties of undefined」エラーが出る

**原因**: `subject()` ヘルパーを使わずにオブジェクトを直接渡している場合に発生する。CASL はサブジェクトの種類を特定するために `__caslSubjectType__` プロパティを必要とする。

```typescript
// ✗ エラーが出る
ability.can('update', { authorId: 'user_1' }); // subject type が不明

// ✓ 正しい方法
import { subject } from '@casl/ability';
ability.can('update', subject('Article', { authorId: 'user_1' }));
```

### Q2: CASL のルールが期待通りに動かない

**原因**: `cannot` は `can` より後に評価されるため、順序に注意。また、条件付きの `can` は条件なしの `can` を上書きしない。

```typescript
// ✗ 意図しない動作
can('read', 'Article');                          // 全記事を読める
can('read', 'Article', { status: 'published' }); // これは上のルールに追加されるだけ

// ✓ 正しいアプローチ（deny-first）
can('read', 'Article', { status: 'published' }); // まず限定的なルール
// admin の場合のみ全記事
if (user.role === 'admin') {
  can('read', 'Article'); // 条件なしで全許可
}
```

### Q3: パフォーマンスが悪い場合の対処法

```
パフォーマンス改善のステップ:

  1. Ability のキャッシュ
     → リクエストごとに再生成しない
     → ユーザーごとに TTL 付きキャッシュ

  2. DB クエリの最適化
     → accessibleBy で DB レベルフィルタリング
     → 必要な属性のみ select

  3. ルール数の削減
     → 冗長なルールを統合
     → ワイルドカード（manage, all）を適切に使用

  4. 条件の簡素化
     → 複雑な $elemMatch は避ける
     → 可能なら事前計算した属性を使用
```

### Q4: CASL と Prisma の where 条件が一致しない

**原因**: CASL の条件構文と Prisma のクエリ構文には一部互換性のない部分がある。`@casl/prisma` パッケージはこの変換を行うが、全ての MongoDB 演算子がサポートされているわけではない。

```typescript
// サポートされている条件
can('read', 'Article', { status: 'published' });        // ✓
can('read', 'Article', { status: { $ne: 'archived' } }); // ✓
can('read', 'Article', { status: { $in: ['a', 'b'] } }); // ✓

// サポートされていない可能性のある条件
can('read', 'Article', { tags: { $all: ['a', 'b'] } });  // △
can('read', 'Article', { title: { $regex: /^test/ } });   // ✗
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| ABAC | 属性（主体・リソース・操作・環境）ベースの制御 |
| NIST アーキテクチャ | PEP, PDP, PAP, PIP の分離 |
| CASL | JS/TS のポリシーライブラリ。条件付き権限定義 |
| ハイブリッド | RBAC（基本）+ ABAC（細かい制御） |
| フロント | Can コンポーネントで UI の条件表示（UX目的） |
| バックエンド | API + DB レベルの権限チェック（セキュリティ目的） |
| テスト | ポリシーマトリクステストで全ロール x 全アクションを検証 |
| 監査 | アクセス制御の決定を監査ログに記録 |
| ポリシーエンジン | CASL（Web）、OPA（K8s）、Cedar（AWS） |

---

## 次に読むべきガイド
→ [[02-api-authorization.md]] — API 認可

---

## 参考文献
1. NIST. "Guide to Attribute Based Access Control (ABAC) Definition and Considerations." SP 800-162, 2014.
2. NIST. "Attribute-Based Access Control." SP 800-205, 2019.
3. CASL. "Documentation." casl.js.org, 2024.
4. Oso. "Authorization Academy." osohq.com, 2024.
5. Open Policy Agent. "Documentation." openpolicyagent.org, 2024.
6. AWS. "Cedar Policy Language." docs.cedarpolicy.com, 2024.
7. OWASP. "Access Control Cheat Sheet." cheatsheetseries.owasp.org, 2024.
8. Casbin. "Documentation." casbin.org, 2024.
