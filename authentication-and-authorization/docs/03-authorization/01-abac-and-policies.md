# ABAC とポリシーエンジン

> RBAC では表現しきれない複雑なアクセス制御には ABAC（属性ベースアクセス制御）が必要。ユーザー属性、リソース属性、環境条件を組み合わせた動的な認可ポリシーの設計と、CASL / Oso 等のポリシーエンジンを解説する。

## この章で学ぶこと

- [ ] ABAC の概念と RBAC との違いを理解する
- [ ] ポリシーの設計パターンを把握する
- [ ] CASL を使った実践的なポリシー実装を学ぶ

---

## 1. ABAC の基本概念

```
ABAC（Attribute-Based Access Control）:

  RBAC: 「ロール」に基づくアクセス制御
    → admin は全記事を編集できる

  ABAC: 「属性」に基づくアクセス制御
    → 記事の作者は自分の記事を編集できる
    → 部署が同じマネージャーは部下の評価を閲覧できる
    → 営業時間内のみデータをエクスポートできる

  4つの属性カテゴリ:

  ┌──────────────────────────────────────────┐
  │                                          │
  │  ① Subject（主体）属性:                    │
  │     → ユーザーID、ロール、部署、役職         │
  │     → メール、入社日、資格                  │
  │                                          │
  │  ② Resource（リソース）属性:                │
  │     → リソースID、タイプ、作成者             │
  │     → ステータス、分類、公開フラグ            │
  │                                          │
  │  ③ Action（操作）属性:                      │
  │     → read、create、update、delete         │
  │     → publish、approve、export             │
  │                                          │
  │  ④ Environment（環境）属性:                 │
  │     → 時刻、IPアドレス、デバイス             │
  │     → 地域、ネットワーク                    │
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
```

```
RBAC vs ABAC:

  項目         │ RBAC           │ ABAC
  ────────────┼───────────────┼──────────────────
  制御の粒度   │ ロール単位      │ 属性の組合せ
  柔軟性       │ 低〜中          │ 高
  複雑性       │ 低              │ 中〜高
  管理コスト   │ 低              │ 中
  ユースケース │ 組織の役割       │ リソース所有者、
              │ が明確な場合     │ 条件付きアクセス
  実装例       │ admin/editor/  │ 「自分の記事のみ
              │ viewer          │ 編集可能」

  実際のプロジェクト:
  → RBAC + ABAC のハイブリッドが一般的
  → 基本はRBAC、細かい制御にABACを追加
```

---

## 2. CASL によるポリシー実装

```typescript
// CASL: JavaScript/TypeScript のポリシーライブラリ
// npm install @casl/ability

import { AbilityBuilder, createMongoAbility, MongoAbility } from '@casl/ability';

// アクションとサブジェクトの型定義
type Actions = 'read' | 'create' | 'update' | 'delete' | 'publish' | 'manage';
type Subjects = 'Article' | 'User' | 'Comment' | 'Organization' | 'all';
type AppAbility = MongoAbility<[Actions, Subjects]>;

// ユーザーのロールに基づくAbility定義
function defineAbilityFor(user: { id: string; role: string; orgId: string }): AppAbility {
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
ability.can('update', 'Article', { authorId: 'user_1' }); // true（自分の記事）
ability.can('update', 'Article', { authorId: 'user_2' }); // false（他人の記事）
ability.can('publish', 'Article'); // false（editor には publish 権限なし）
```

---

## 3. API での権限チェック

```typescript
// CASL + Express ミドルウェア
import { ForbiddenError } from '@casl/ability';

// 記事の更新API
app.put('/api/articles/:id', async (req, res) => {
  const user = req.user;
  const ability = defineAbilityFor(user);

  const article = await prisma.article.findUnique({
    where: { id: req.params.id },
  });

  if (!article) {
    return res.status(404).json({ error: 'Article not found' });
  }

  // CASL の権限チェック（属性ベース）
  ForbiddenError.from(ability).throwUnlessCan('update', {
    ...article,
    kind: 'Article', // CASL の subject 型
  });

  const updated = await prisma.article.update({
    where: { id: req.params.id },
    data: req.body,
  });

  res.json(updated);
});

// エラーハンドラー
app.use((error: Error, req: Request, res: Response, next: Function) => {
  if (error instanceof ForbiddenError) {
    return res.status(403).json({
      error: 'Forbidden',
      message: `Cannot ${error.action} ${error.subjectType}`,
    });
  }
  next(error);
});
```

---

## 4. フロントエンドでの権限制御

```typescript
// React + CASL
import { createContext, useContext } from 'react';
import { createContextualCan } from '@casl/react';

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
      <Can I="update" this={{ ...article, kind: 'Article' }}>
        <button>Edit</button>
      </Can>

      <Can I="delete" this={{ ...article, kind: 'Article' }}>
        <button className="text-red-500">Delete</button>
      </Can>

      <Can I="publish" a="Article">
        <button className="text-green-500">Publish</button>
      </Can>
    </div>
  );
}

// フック
function useAbility() {
  return useContext(AbilityContext);
}

function ArticlePage({ article }: { article: Article }) {
  const ability = useAbility();

  if (ability.cannot('read', { ...article, kind: 'Article' })) {
    return <div>この記事を閲覧する権限がありません</div>;
  }

  return <ArticleContent article={article} />;
}
```

---

## 5. ポリシーの設計パターン

```
ポリシー設計のパターン:

  ① リソースオーナーシップ:
     → 作成者は自分のリソースを操作可能
     → can('update', 'Article', { authorId: user.id })

  ② 組織スコープ:
     → 同じ組織内のリソースのみアクセス可能
     → can('read', 'Article', { orgId: user.orgId })

  ③ ステータスベース:
     → ステータスに応じた操作制限
     → can('update', 'Article', { status: { $ne: 'archived' } })
     → 公開済みは管理者のみ編集可能

  ④ 時間ベース:
     → 特定時間帯のみ許可
     → ポリシー定義時に現在時刻をチェック

  ⑤ 承認フロー:
     → 下書き → レビュー → 承認 → 公開
     → 各ステージで操作可能なロールが異なる

  設計の原則:
  → デフォルトは拒否（明示的に許可したもののみ）
  → 最小権限の原則
  → ポリシーをコードとして管理（バージョン管理可能）
  → テスト可能（ポリシーのユニットテスト）
```

```typescript
// ポリシーのテスト
import { describe, it, expect } from 'vitest';

describe('Editor permissions', () => {
  const ability = defineAbilityFor({
    id: 'user_1',
    role: 'editor',
    orgId: 'org_1',
  });

  it('can read articles', () => {
    expect(ability.can('read', 'Article')).toBe(true);
  });

  it('can update own articles', () => {
    expect(ability.can('update', {
      kind: 'Article',
      authorId: 'user_1',
    })).toBe(true);
  });

  it('cannot update others articles', () => {
    expect(ability.can('update', {
      kind: 'Article',
      authorId: 'user_2',
    })).toBe(false);
  });

  it('cannot publish articles', () => {
    expect(ability.can('publish', 'Article')).toBe(false);
  });

  it('cannot manage users', () => {
    expect(ability.can('update', 'User')).toBe(false);
  });
});
```

---

## 6. RBAC + ABAC ハイブリッド

```
実践的なハイブリッドアプローチ:

  第1層: RBAC（粗い制御）
    → ロールで基本的なアクセス範囲を決定
    → admin, editor, viewer

  第2層: ABAC（細かい制御）
    → リソースの属性で追加の制約
    → 自分の記事のみ編集可能
    → 同じ組織のデータのみ

  第3層: ビジネスルール
    → 時間帯制限、承認フロー
    → APIレート制限

  実装:
    RBAC: ミドルウェアでロールチェック
    ABAC: サービス層でリソースベースのチェック
    ビジネスルール: ドメインロジック内
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| ABAC | 属性（主体・リソース・操作・環境）ベースの制御 |
| CASL | JS/TS のポリシーライブラリ。条件付き権限定義 |
| ハイブリッド | RBAC（基本）+ ABAC（細かい制御） |
| フロント | Can コンポーネントで UI の条件表示 |
| テスト | ポリシーのユニットテストが重要 |

---

## 次に読むべきガイド
→ [[02-api-authorization.md]] — API 認可

---

## 参考文献
1. NIST. "Guide to Attribute Based Access Control." SP 800-162, 2014.
2. CASL. "Documentation." casl.js.org, 2024.
3. Oso. "Authorization Academy." osohq.com, 2024.
