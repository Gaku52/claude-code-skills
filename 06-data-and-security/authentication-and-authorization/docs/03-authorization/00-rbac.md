# RBAC（ロールベースアクセス制御）

> RBAC はユーザーに「ロール」を割り当て、ロールに「権限」を紐付ける最も普及したアクセス制御モデル。ロール設計、権限モデル、階層ロール、マルチテナント対応、キャッシングまで、実践的な RBAC の設計と実装を解説する。

## この章で学ぶこと

- [ ] RBAC の基本概念（RBAC0〜RBAC3）とモデルの違いを理解する
- [ ] 権限の命名規則とロール設計パターンを実務レベルで設計できる
- [ ] 階層ロール（ロール継承）を再帰的に解決するアルゴリズムを実装できる
- [ ] マルチテナント環境での組織別 RBAC を設計・実装できる
- [ ] Redis を使った権限キャッシングでパフォーマンスを最適化できる

## 前提知識

- 認証と認可の違い → [00-fundamentals/](../00-fundamentals/)
- トークン管理の基本 → [02-token-auth/03-token-management.md](../02-token-auth/03-token-management.md)
- RDB の基本（多対多リレーション）→ [sql-and-query-mastery: 02-design/](../../sql-and-query-mastery/docs/02-design/)
- セキュリティの基礎 → [security-fundamentals: 00-basics/](../../security-fundamentals/docs/00-basics/)

---

## 1. RBAC の基本モデル

### 1.1 RBAC の構成要素

```
RBAC の構成要素:

  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │  User    │───→│  Role    │───→│  Permission  │
  │ ユーザー  │ N:M│  ロール   │ N:M│  権限         │
  └──────────┘    └──────────┘    └──────────────┘

  User → Role:      ユーザーは1つ以上のロールを持つ
  Role → Permission: ロールは1つ以上の権限を持つ
  User → Permission: ユーザーはロール経由で間接的に権限を得る

  なぜ直接 User → Permission にしないのか（WHY）:
    → ユーザー数が増えると権限の個別管理が破綻する
    → 「編集者」という役割を定義すれば、新ユーザーにロールを割当てるだけ
    → 権限変更時もロールを変えれば全ユーザーに反映
    → 監査が容易（「誰がどのロールを持つか」で説明可能）
```

### 1.2 RBAC のレベル（NIST 定義）

```
RBAC のレベル（NIST SP 359 に基づく）:

  ┌──────────────────────────────────────────────────────────────┐
  │                                                              │
  │  RBAC0（基本 RBAC）:                                          │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │ ユーザー ←→ ロール → 権限                              │    │
  │  │ 最もシンプルな形。ロールに権限を紐付ける。              │    │
  │  │ 例: admin, editor, viewer の3ロール                    │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                              │
  │  RBAC1（階層ロール）:                                         │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │ ロール間の継承関係を追加                                │    │
  │  │ admin は editor の権限を「継承」する                     │    │
  │  │ editor は viewer の権限を「継承」する                    │    │
  │  │ → 権限の重複定義を排除できる                            │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                              │
  │  RBAC2（制約付き RBAC）:                                      │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │ ロールに制約を追加                                      │    │
  │  │ ① 相互排他: 「承認者」と「申請者」を同時に持てない       │    │
  │  │ ② 最大ロール数: 1ユーザーにつき3ロールまで              │    │
  │  │ ③ 職務分離（SoD）: 利益相反を防止                       │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                              │
  │  RBAC3（統合）:                                               │
  │  ┌──────────────────────────────────────────────────────┐    │
  │  │ RBAC1（階層）+ RBAC2（制約）の両方を統合                 │    │
  │  │ 最も柔軟で完全なモデル                                  │    │
  │  │ エンタープライズ向け                                    │    │
  │  └──────────────────────────────────────────────────────┘    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘

  選択ガイド:
    小規模アプリ → RBAC0 で十分
    中規模アプリ → RBAC1（階層ロール）推奨
    エンタープライズ → RBAC3（階層 + 制約）
```

---

## 2. ロールと権限の設計

### 2.1 権限の命名規則

```
権限の命名規則: resource:action

  基本パターン:
    articles:read     — 記事の閲覧
    articles:create   — 記事の作成
    articles:update   — 記事の編集
    articles:delete   — 記事の削除
    articles:publish  — 記事の公開

  ユーザー管理:
    users:read        — ユーザー一覧閲覧
    users:create      — ユーザー作成
    users:update      — ユーザー編集
    users:delete      — ユーザー削除
    users:invite      — ユーザー招待

  組織管理:
    org:settings      — 組織設定の変更
    org:billing       — 課金管理
    org:members       — メンバー管理

  命名のベストプラクティス:
    ✓ resource:action 形式で統一
    ✓ リソース名は複数形（articles, users）
    ✓ アクションは CRUD ベース + ドメイン固有（publish, approve）
    ✓ ワイルドカード: articles:* で全権限

  避けるべき命名:
    ✗ can_edit_articles（verbose すぎる）
    ✗ admin（リソースが不明）
    ✗ 1, 2, 3（数値 ID は意味不明）
```

### 2.2 ロール設計パターン

```
ロール設計例（CMS アプリケーション）:

  viewer:
    → articles:read

  editor:
    → viewer の全権限 +
    → articles:create, articles:update

  publisher:
    → editor の全権限 +
    → articles:publish, articles:delete

  admin:
    → publisher の全権限 +
    → users:*, org:settings

  super_admin:
    → admin の全権限 +
    → org:billing, システム設定

  ┌─────────────────────────────────────────────────┐
  │  ロール階層（RBAC1）:                             │
  │                                                   │
  │  super_admin                                      │
  │    └── admin                                      │
  │          └── publisher                            │
  │                └── editor                         │
  │                      └── viewer                   │
  │                                                   │
  │  各ロールは親ロールの権限を全て継承                  │
  └─────────────────────────────────────────────────┘

  ロール数の目安:
    → 小規模: 3〜5 ロール
    → 中規模: 5〜10 ロール
    → 大規模: 10〜20 ロール（それ以上は ABAC の併用を検討）
```

### 2.3 権限とロールの型定義

```typescript
// 権限定義（型安全）
const PERMISSIONS = {
  // 記事
  'articles:read': '記事の閲覧',
  'articles:create': '記事の作成',
  'articles:update': '記事の編集',
  'articles:delete': '記事の削除',
  'articles:publish': '記事の公開',
  // ユーザー
  'users:read': 'ユーザー一覧',
  'users:create': 'ユーザー作成',
  'users:update': 'ユーザー編集',
  'users:delete': 'ユーザー削除',
  'users:invite': 'ユーザー招待',
  // 組織
  'org:settings': '組織設定',
  'org:billing': '課金管理',
  'org:members': 'メンバー管理',
} as const;

type Permission = keyof typeof PERMISSIONS;

// ロール定義（階層付き）
interface RoleConfig {
  description: string;
  permissions: Permission[];
  inherits: string[];
}

const ROLES: Record<string, RoleConfig> = {
  viewer: {
    description: '閲覧のみ',
    permissions: ['articles:read'],
    inherits: [],
  },
  editor: {
    description: '記事の作成・編集',
    permissions: ['articles:create', 'articles:update'],
    inherits: ['viewer'],
  },
  publisher: {
    description: '記事の公開・削除',
    permissions: ['articles:publish', 'articles:delete'],
    inherits: ['editor'],
  },
  admin: {
    description: 'ユーザー・組織管理',
    permissions: [
      'users:read', 'users:create', 'users:update',
      'users:delete', 'users:invite', 'org:settings', 'org:members',
    ],
    inherits: ['publisher'],
  },
  super_admin: {
    description: '全権限（課金含む）',
    permissions: ['org:billing'],
    inherits: ['admin'],
  },
};

type Role = keyof typeof ROLES;
```

### 2.4 ロールの全権限を解決（継承込み）

```typescript
// ロールの全権限を再帰的に解決
function resolvePermissions(role: string, visited = new Set<string>()): Set<Permission> {
  // 循環参照の防止
  if (visited.has(role)) return new Set();
  visited.add(role);

  const roleConfig = ROLES[role];
  if (!roleConfig) return new Set();

  const permissions = new Set<Permission>();

  // 直接の権限を追加
  roleConfig.permissions.forEach((p) => permissions.add(p));

  // 継承された権限を再帰的に解決
  roleConfig.inherits.forEach((parentRole) => {
    resolvePermissions(parentRole, visited).forEach((p) => permissions.add(p));
  });

  return permissions;
}

// 権限チェック
function hasPermission(userRole: string, permission: Permission): boolean {
  const permissions = resolvePermissions(userRole);
  return permissions.has(permission);
}

// ワイルドカード対応の権限チェック
function hasPermissionWithWildcard(
  userRole: string,
  requiredPermission: string
): boolean {
  const permissions = resolvePermissions(userRole);

  // 完全一致
  if (permissions.has(requiredPermission as Permission)) return true;

  // ワイルドカードチェック: users:* → users:read, users:create, ...
  const [resource] = requiredPermission.split(':');
  const wildcardPerm = `${resource}:*` as Permission;
  if (permissions.has(wildcardPerm)) return true;

  // 全権限ワイルドカード
  if (permissions.has('*:*' as Permission)) return true;

  return false;
}

// 使用例
console.log(hasPermission('editor', 'articles:read'));     // true（viewer から継承）
console.log(hasPermission('editor', 'articles:create'));   // true（直接の権限）
console.log(hasPermission('editor', 'articles:publish'));  // false（publisher 以上）
console.log(hasPermission('admin', 'articles:publish'));   // true（publisher から継承）
console.log(hasPermission('super_admin', 'org:billing'));  // true（直接の権限）
```

---

## 3. データベース設計

### 3.1 テーブル構造

```
RBAC のテーブル設計:

  ┌─────────┐   ┌──────────────┐   ┌──────────┐
  │  users  │──→│ user_roles   │←──│  roles   │
  │         │   │ (user_id,    │   │          │
  │         │   │  role_id)    │   │          │
  └─────────┘   └──────────────┘   └──────────┘
                                       │
                                   ┌───┴────────────┐
                                   │ role_permissions│
                                   │ (role_id,       │
                                   │  permission_id) │
                                   └───┬────────────┘
                                       │
                                   ┌───┴──────────┐
                                   │ permissions   │
                                   └──────────────┘

  正規化 vs 非正規化:

  正規化（上記の5テーブル構成）:
    ✓ 柔軟性が高い（権限の動的追加・削除）
    ✓ 管理画面からの権限設定変更に対応
    △ JOIN が多くてクエリが複雑

  非正規化（ロール定義をコードに持つ）:
    ✓ クエリがシンプル（user テーブルに role カラム）
    ✓ パフォーマンスが良い
    △ 権限変更にはコード変更 + デプロイが必要

  推奨:
    → 小〜中規模: 非正規化（role カラム + コード定義）
    → 大規模: 正規化（DB テーブル + 管理画面）
```

### 3.2 Prisma スキーマ（正規化版）

```typescript
// schema.prisma

// model User {
//   id        String     @id @default(cuid())
//   email     String     @unique
//   name      String
//   password  String?
//   roles     UserRole[]
//   createdAt DateTime   @default(now())
//   updatedAt DateTime   @updatedAt
// }
//
// model Role {
//   id          String           @id @default(cuid())
//   name        String           @unique  // "admin", "editor", etc.
//   description String?
//   permissions RolePermission[]
//   users       UserRole[]
//   parentId    String?          // 階層ロール
//   parent      Role?            @relation("RoleHierarchy", fields: [parentId], references: [id])
//   children    Role[]           @relation("RoleHierarchy")
//   createdAt   DateTime         @default(now())
// }
//
// model Permission {
//   id          String           @id @default(cuid())
//   name        String           @unique  // "articles:read"
//   description String?
//   roles       RolePermission[]
//   createdAt   DateTime         @default(now())
// }
//
// model UserRole {
//   userId    String
//   roleId    String
//   user      User     @relation(fields: [userId], references: [id], onDelete: Cascade)
//   role      Role     @relation(fields: [roleId], references: [id], onDelete: Cascade)
//   assignedAt DateTime @default(now())
//   assignedBy String?  // 誰がこのロールを割当てたか
//   @@id([userId, roleId])
//   @@index([userId])
//   @@index([roleId])
// }
//
// model RolePermission {
//   roleId       String
//   permissionId String
//   role         Role       @relation(fields: [roleId], references: [id], onDelete: Cascade)
//   permission   Permission @relation(fields: [permissionId], references: [id], onDelete: Cascade)
//   @@id([roleId, permissionId])
// }
```

### 3.3 DB からユーザーの全権限を取得

```typescript
// DB からユーザーの全権限を取得（階層ロール対応）
async function getUserPermissions(userId: string): Promise<Set<string>> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      roles: {
        include: {
          role: {
            include: {
              permissions: {
                include: { permission: true },
              },
              parent: {
                include: {
                  permissions: {
                    include: { permission: true },
                  },
                  // 2段階まで（深い階層は再帰で解決）
                  parent: {
                    include: {
                      permissions: {
                        include: { permission: true },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
  });

  const permissions = new Set<string>();

  // 全ロールの権限を収集（階層込み）
  function collectPermissions(role: any) {
    role.permissions.forEach(({ permission }: any) => {
      permissions.add(permission.name);
    });

    // 親ロールの権限も収集（再帰）
    if (role.parent) {
      collectPermissions(role.parent);
    }
  }

  user?.roles.forEach(({ role }) => collectPermissions(role));

  return permissions;
}

// 非正規化版（シンプル・高速）
async function getUserPermissionsSimple(userId: string): Promise<Set<string>> {
  const user = await prisma.user.findUnique({
    where: { id: userId },
    select: { role: true }, // role カラムが1つの場合
  });

  if (!user) return new Set();

  // コード定義のロールから権限を解決
  return resolvePermissions(user.role);
}
```

---

## 4. ミドルウェアでの権限チェック

### 4.1 Express ミドルウェア

```typescript
// Express ミドルウェア（汎用的な権限チェック）
import { Request, Response, NextFunction } from 'express';

// 権限チェックミドルウェア
function requirePermission(...requiredPermissions: string[]) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const user = req.user; // 認証ミドルウェアで設定済み
    if (!user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    // ユーザーの権限を取得（キャッシュ付き）
    const userPermissions = await getCachedPermissions(user.id);

    // 全ての必要権限を持っているか確認
    const hasAll = requiredPermissions.every((p) => userPermissions.has(p));
    if (!hasAll) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required: requiredPermissions,
        hint: 'Contact your administrator to request access',
      });
    }

    next();
  };
}

// いずれかの権限を持っていれば OK
function requireAnyPermission(...requiredPermissions: string[]) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const user = req.user;
    if (!user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const userPermissions = await getCachedPermissions(user.id);
    const hasAny = requiredPermissions.some((p) => userPermissions.has(p));

    if (!hasAny) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required_any: requiredPermissions,
      });
    }

    next();
  };
}

// 使用例
app.get('/api/articles', requirePermission('articles:read'), getArticles);
app.post('/api/articles', requirePermission('articles:create'), createArticle);
app.put('/api/articles/:id', requirePermission('articles:update'), updateArticle);
app.delete('/api/articles/:id', requirePermission('articles:delete'), deleteArticle);
app.post('/api/articles/:id/publish', requirePermission('articles:publish'), publishArticle);

// 複数権限が必要な場合
app.delete('/api/users/:id',
  requirePermission('users:delete', 'users:read'),
  deleteUser
);

// いずれかの権限でアクセス可能
app.get('/api/reports',
  requireAnyPermission('reports:read', 'admin'),
  getReports
);
```

### 4.2 Next.js Server Actions での権限チェック

```typescript
// Next.js Server Actions での権限チェック
import { auth } from '@/auth';
import { redirect } from 'next/navigation';

// 権限チェックユーティリティ
async function authorize(...requiredPermissions: string[]) {
  const session = await auth();

  if (!session) {
    redirect('/login');
  }

  const userPermissions = resolvePermissions(session.user.role);
  const hasAll = requiredPermissions.every((p) =>
    userPermissions.has(p as Permission)
  );

  if (!hasAll) {
    throw new Error(
      `Insufficient permissions. Required: ${requiredPermissions.join(', ')}`
    );
  }

  return session;
}

// Server Action での使用例
'use server';

async function createArticle(formData: FormData) {
  const session = await authorize('articles:create');

  const article = await prisma.article.create({
    data: {
      title: formData.get('title') as string,
      content: formData.get('content') as string,
      authorId: session.user.id,
    },
  });

  revalidatePath('/articles');
  return article;
}

async function deleteArticle(articleId: string) {
  const session = await authorize('articles:delete');

  await prisma.article.delete({
    where: { id: articleId },
  });

  revalidatePath('/articles');
}

async function inviteUser(email: string, role: string) {
  const session = await authorize('users:invite', 'users:create');

  // 招待するロールが自分のロールより高くないことを確認
  const inviterPermissions = resolvePermissions(session.user.role);
  const inviteePermissions = resolvePermissions(role);

  // 招待先のロールが自分にない権限を持っている場合はエラー
  for (const perm of inviteePermissions) {
    if (!inviterPermissions.has(perm)) {
      throw new Error('Cannot assign a role with higher permissions than your own');
    }
  }

  await sendInvitation(email, role);
}
```

---

## 5. マルチテナント RBAC

### 5.1 テナント分離の設計

```
マルチテナント RBAC の設計:

  課題: 同じユーザーが組織ごとに異なるロールを持つ
    Alice: Organization A → admin
    Alice: Organization B → viewer
    Bob:   Organization A → editor

  テーブル設計:

  ┌─────────┐   ┌─────────────────────────┐   ┌───────────┐
  │  users  │──→│ organization_members     │←──│ orgs      │
  │         │   │ (user_id, org_id, role)  │   │           │
  └─────────┘   └─────────────────────────┘   └───────────┘

  設計の選択肢:

  ① role カラム（シンプル）:
     organization_members テーブルに role カラムを追加
     → 小〜中規模向け

  ② 別テーブル（柔軟）:
     org_member_roles (member_id, role_id)
     → ユーザーが組織内で複数ロールを持てる
     → 大規模向け
```

### 5.2 マルチテナント RBAC の実装

```typescript
// マルチテナント RBAC の実装
interface OrgMembership {
  userId: string;
  orgId: string;
  role: string;
  joinedAt: Date;
}

// 組織内の権限チェック
async function checkOrgPermission(
  userId: string,
  orgId: string,
  permission: string
): Promise<boolean> {
  const membership = await prisma.organizationMember.findUnique({
    where: {
      userId_orgId: { userId, orgId },
    },
  });

  if (!membership) return false;

  // ロールの権限を解決
  return hasPermission(membership.role, permission as Permission);
}

// 組織コンテキスト付きミドルウェア
function requireOrgPermission(...permissions: string[]) {
  return async (req: Request, res: Response, next: NextFunction) => {
    const user = req.user;
    if (!user) {
      return res.status(401).json({ error: 'Unauthorized' });
    }

    // URL から orgId を取得（/org/:orgId/...）
    const orgId = req.params.orgId;
    if (!orgId) {
      return res.status(400).json({ error: 'Organization ID required' });
    }

    const hasAll = await Promise.all(
      permissions.map((p) => checkOrgPermission(user.id, orgId, p))
    );

    if (hasAll.some((v) => !v)) {
      return res.status(403).json({
        error: 'Insufficient permissions in this organization',
      });
    }

    // 組織情報をリクエストに付与
    req.orgId = orgId;
    next();
  };
}

// Next.js Server Component での組織コンテキスト
async function OrgDashboard({ params }: { params: { orgId: string } }) {
  const session = await auth();
  if (!session) redirect('/login');

  const membership = await prisma.organizationMember.findUnique({
    where: {
      userId_orgId: {
        userId: session.user.id,
        orgId: params.orgId,
      },
    },
    include: { organization: true },
  });

  if (!membership) {
    notFound();
  }

  const permissions = resolvePermissions(membership.role);

  return (
    <div>
      <h1>{membership.organization.name}</h1>
      <p>Your role: {membership.role}</p>

      {permissions.has('articles:create' as Permission) && (
        <Link href={`/org/${params.orgId}/articles/new`}>New Article</Link>
      )}

      {permissions.has('users:read' as Permission) && (
        <Link href={`/org/${params.orgId}/members`}>Members</Link>
      )}

      {permissions.has('org:settings' as Permission) && (
        <Link href={`/org/${params.orgId}/settings`}>Settings</Link>
      )}
    </div>
  );
}
```

### 5.3 組織メンバーの招待と管理

```typescript
// 組織メンバー管理 API
class OrgMemberService {
  // メンバー招待
  async inviteMember(
    inviterId: string,
    orgId: string,
    email: string,
    role: string
  ) {
    // 招待者の権限チェック
    const inviterMembership = await prisma.organizationMember.findUnique({
      where: { userId_orgId: { userId: inviterId, orgId } },
    });

    if (!inviterMembership) {
      throw new Error('You are not a member of this organization');
    }

    if (!hasPermission(inviterMembership.role, 'users:invite' as Permission)) {
      throw new Error('You do not have permission to invite members');
    }

    // 招待先のロールが招待者以下であることを確認
    const inviterPerms = resolvePermissions(inviterMembership.role);
    const inviteePerms = resolvePermissions(role);

    for (const perm of inviteePerms) {
      if (!inviterPerms.has(perm)) {
        throw new Error('Cannot assign a role with higher permissions');
      }
    }

    // 招待を作成
    const invitation = await prisma.orgInvitation.create({
      data: {
        orgId,
        email,
        role,
        invitedBy: inviterId,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
        token: crypto.randomBytes(32).toString('hex'),
      },
    });

    await sendInvitationEmail(email, invitation);
    return invitation;
  }

  // ロール変更
  async changeRole(
    adminId: string,
    orgId: string,
    targetUserId: string,
    newRole: string
  ) {
    // 管理者権限チェック
    const adminMembership = await prisma.organizationMember.findUnique({
      where: { userId_orgId: { userId: adminId, orgId } },
    });

    if (!hasPermission(adminMembership!.role, 'users:update' as Permission)) {
      throw new Error('Permission denied');
    }

    // 自分自身のロール変更は不可（別の管理者が必要）
    if (adminId === targetUserId) {
      throw new Error('Cannot change your own role');
    }

    await prisma.organizationMember.update({
      where: { userId_orgId: { userId: targetUserId, orgId } },
      data: { role: newRole },
    });
  }
}
```

---

## 6. 権限のキャッシング

### 6.1 Redis キャッシュ実装

```typescript
// Redis キャッシュで権限チェックを高速化
import Redis from 'ioredis';

class PermissionCache {
  private redis: Redis;
  private ttl = 300; // 5分

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }

  private key(userId: string, orgId?: string): string {
    return orgId ? `perms:${userId}:${orgId}` : `perms:${userId}`;
  }

  // キャッシュから権限を取得
  async get(userId: string, orgId?: string): Promise<Set<string> | null> {
    const cached = await this.redis.smembers(this.key(userId, orgId));
    return cached.length > 0 ? new Set(cached) : null;
  }

  // キャッシュに権限を保存
  async set(
    userId: string,
    permissions: Set<string>,
    orgId?: string
  ): Promise<void> {
    const key = this.key(userId, orgId);
    const pipeline = this.redis.pipeline();
    pipeline.del(key);
    if (permissions.size > 0) {
      pipeline.sadd(key, ...permissions);
    }
    pipeline.expire(key, this.ttl);
    await pipeline.exec();
  }

  // キャッシュを無効化（ロール変更時）
  async invalidate(userId: string, orgId?: string): Promise<void> {
    if (orgId) {
      await this.redis.del(this.key(userId, orgId));
    } else {
      // 全組織のキャッシュを削除
      const keys = await this.redis.keys(`perms:${userId}:*`);
      if (keys.length > 0) await this.redis.del(...keys);
      await this.redis.del(this.key(userId));
    }
  }

  // ロール変更時のイベントハンドラ
  async onRoleChanged(userId: string, orgId?: string): Promise<void> {
    await this.invalidate(userId, orgId);
  }
}

// キャッシュ付き権限取得
const permissionCache = new PermissionCache(process.env.REDIS_URL!);

async function getCachedPermissions(
  userId: string,
  orgId?: string
): Promise<Set<string>> {
  // キャッシュをチェック
  const cached = await permissionCache.get(userId, orgId);
  if (cached) return cached;

  // キャッシュミス → DB から取得
  const permissions = orgId
    ? await getOrgPermissions(userId, orgId)
    : await getUserPermissions(userId);

  // キャッシュに保存
  await permissionCache.set(userId, permissions, orgId);

  return permissions;
}
```

### 6.2 インメモリキャッシュ（Redis 不要な場合）

```typescript
// LRU キャッシュ（小規模アプリ向け）
class InMemoryPermissionCache {
  private cache = new Map<string, {
    permissions: Set<string>;
    expiresAt: number;
  }>();
  private maxSize = 1000;
  private ttl = 5 * 60 * 1000; // 5分

  get(userId: string): Set<string> | null {
    const entry = this.cache.get(userId);
    if (!entry) return null;
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(userId);
      return null;
    }
    return entry.permissions;
  }

  set(userId: string, permissions: Set<string>): void {
    // サイズ制限チェック
    if (this.cache.size >= this.maxSize) {
      // 最も古いエントリを削除
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }

    this.cache.set(userId, {
      permissions,
      expiresAt: Date.now() + this.ttl,
    });
  }

  invalidate(userId: string): void {
    this.cache.delete(userId);
  }

  clear(): void {
    this.cache.clear();
  }
}
```

---

## 7. RBAC の制約（RBAC2）

### 7.1 相互排他ロールと職務分離

```typescript
// RBAC2: 制約の実装

// 相互排他ロール（同時に持てないロール）
const MUTUALLY_EXCLUSIVE_ROLES: [string, string][] = [
  ['approver', 'requester'],     // 承認者と申請者は兼任不可
  ['auditor', 'admin'],          // 監査役と管理者は兼任不可
];

// ロール割当時のバリデーション
async function assignRole(userId: string, newRole: string, orgId?: string) {
  // 現在のロールを取得
  const currentRoles = await getUserRoles(userId, orgId);

  // 相互排他チェック
  for (const [roleA, roleB] of MUTUALLY_EXCLUSIVE_ROLES) {
    if (newRole === roleA && currentRoles.includes(roleB)) {
      throw new Error(
        `Cannot assign "${newRole}": conflicts with existing role "${roleB}"`
      );
    }
    if (newRole === roleB && currentRoles.includes(roleA)) {
      throw new Error(
        `Cannot assign "${newRole}": conflicts with existing role "${roleA}"`
      );
    }
  }

  // 最大ロール数チェック
  const MAX_ROLES = 5;
  if (currentRoles.length >= MAX_ROLES) {
    throw new Error(`Maximum of ${MAX_ROLES} roles per user`);
  }

  // ロールを割当
  await prisma.userRole.create({
    data: { userId, roleId: newRole },
  });
}

// 職務分離（SoD）チェック
function checkSeparationOfDuties(
  userRoles: string[],
  operation: string
): boolean {
  // 例: 支払い承認は、支払い作成者とは別の人が行う必要がある
  const sodRules: Record<string, string[]> = {
    'payment:approve': ['payment:create'], // 承認者は作成者になれない
    'audit:sign': ['accounting:post'],      // 監査署名者は記帳者になれない
  };

  const conflictingOps = sodRules[operation];
  if (!conflictingOps) return true;

  const userPermissions = new Set<string>();
  userRoles.forEach((role) => {
    resolvePermissions(role).forEach((p) => userPermissions.add(p));
  });

  // 利益相反するオペレーションを持っていないかチェック
  return !conflictingOps.some((op) => userPermissions.has(op as Permission));
}
```

---

## 8. アンチパターン

### 8.1 ロール名をハードコーディングする

```typescript
// NG: ロール名のハードコーディング
async function deleteArticle(userId: string, articleId: string) {
  const user = await prisma.user.findUnique({ where: { id: userId } });

  // ✗ ロール名をコードに直接書く
  if (user?.role !== 'admin' && user?.role !== 'super_admin') {
    throw new Error('Permission denied');
  }

  await prisma.article.delete({ where: { id: articleId } });
}
// 問題: ロールが変わるたびにコード修正が必要

// ✓ OK: 権限ベースでチェック
async function deleteArticleGood(userId: string, articleId: string) {
  const permissions = await getCachedPermissions(userId);

  // 権限名でチェック（ロール名に依存しない）
  if (!permissions.has('articles:delete')) {
    throw new Error('Permission denied');
  }

  await prisma.article.delete({ where: { id: articleId } });
}
// ロールを変更しても、権限の割当を変えるだけでコード修正不要
```

### 8.2 権限チェックを省略する

```typescript
// NG: フロントエンドの表示制御のみで権限管理
// フロントエンド
function AdminPanel() {
  const { user } = useAuth();
  if (user.role !== 'admin') return null; // 非表示にするだけ
  return <AdminDashboard />;
}

// バックエンド API に権限チェックがない
app.delete('/api/users/:id', async (req, res) => {
  // ✗ 誰でもユーザーを削除できてしまう
  await prisma.user.delete({ where: { id: req.params.id } });
  res.json({ success: true });
});

// ✓ OK: バックエンドで必ず権限チェック
app.delete('/api/users/:id',
  requirePermission('users:delete'), // ミドルウェアでチェック
  async (req, res) => {
    await prisma.user.delete({ where: { id: req.params.id } });
    res.json({ success: true });
  }
);
```

### 8.3 デフォルトで許可する

```typescript
// NG: デフォルト許可（明示的に拒否しないとアクセス可能）
function checkAccess(userRole: string, resource: string): boolean {
  const deniedResources: Record<string, string[]> = {
    viewer: ['admin-panel', 'billing'],
    editor: ['admin-panel'],
  };

  // ✗ リストにないリソースは許可（新しいリソース追加時に漏れる）
  return !(deniedResources[userRole]?.includes(resource));
}

// ✓ OK: デフォルト拒否（明示的に許可したもののみアクセス可能）
function checkAccessGood(userRole: string, permission: string): boolean {
  const permissions = resolvePermissions(userRole);
  // 明示的に許可されていなければ拒否
  return permissions.has(permission as Permission);
}
```

---

## 実践演習

### 演習1: 基礎 - ロール階層と権限解決の実装

**課題**: 以下のロール階層を定義し、`resolvePermissions` と `hasPermission` を実装してください。

- guest: articles:read
- member: guest + comments:create, comments:read
- editor: member + articles:create, articles:update
- admin: editor + users:read, users:update, articles:delete

```typescript
// テンプレート
function resolvePermissions(role: string): Set<string> {
  // TODO: 実装してください
  return new Set();
}

// テスト
console.assert(resolvePermissions('guest').size === 1);
console.assert(resolvePermissions('admin').has('articles:read')); // 継承
console.assert(!resolvePermissions('editor').has('users:read')); // admin のみ
```

<details>
<summary>模範解答</summary>

```typescript
const ROLE_DEFINITIONS: Record<string, {
  permissions: string[];
  inherits: string[];
}> = {
  guest: {
    permissions: ['articles:read'],
    inherits: [],
  },
  member: {
    permissions: ['comments:create', 'comments:read'],
    inherits: ['guest'],
  },
  editor: {
    permissions: ['articles:create', 'articles:update'],
    inherits: ['member'],
  },
  admin: {
    permissions: ['users:read', 'users:update', 'articles:delete'],
    inherits: ['editor'],
  },
};

function resolvePermissions(role: string, visited = new Set<string>()): Set<string> {
  if (visited.has(role)) return new Set(); // 循環参照防止
  visited.add(role);

  const config = ROLE_DEFINITIONS[role];
  if (!config) return new Set();

  const perms = new Set<string>(config.permissions);

  for (const parent of config.inherits) {
    for (const p of resolvePermissions(parent, visited)) {
      perms.add(p);
    }
  }

  return perms;
}

function hasPermission(role: string, permission: string): boolean {
  return resolvePermissions(role).has(permission);
}

// テスト
const guestPerms = resolvePermissions('guest');
console.log('guest permissions:', [...guestPerms]);
console.assert(guestPerms.size === 1, 'guest should have 1 permission');
console.assert(guestPerms.has('articles:read'), 'guest should have articles:read');

const memberPerms = resolvePermissions('member');
console.log('member permissions:', [...memberPerms]);
console.assert(memberPerms.size === 3, 'member should have 3 permissions');
console.assert(memberPerms.has('articles:read'), 'member inherits articles:read');

const editorPerms = resolvePermissions('editor');
console.log('editor permissions:', [...editorPerms]);
console.assert(editorPerms.size === 5, 'editor should have 5 permissions');
console.assert(!editorPerms.has('users:read'), 'editor should not have users:read');

const adminPerms = resolvePermissions('admin');
console.log('admin permissions:', [...adminPerms]);
console.assert(adminPerms.size === 8, 'admin should have 8 permissions');
console.assert(adminPerms.has('articles:read'), 'admin inherits articles:read');
console.assert(adminPerms.has('users:read'), 'admin has users:read');

console.log('All tests passed!');
```

</details>

### 演習2: 応用 - マルチテナント RBAC ミドルウェア

**課題**: Express のマルチテナント RBAC ミドルウェアを実装してください。ユーザーは組織ごとに異なるロールを持ち、`/org/:orgId/...` のルートで権限チェックを行います。

<details>
<summary>模範解答</summary>

```typescript
import express, { Request, Response, NextFunction } from 'express';

// インメモリデータ（本番では DB）
const memberships = new Map<string, { userId: string; orgId: string; role: string }>();
memberships.set('user1:org1', { userId: 'user1', orgId: 'org1', role: 'admin' });
memberships.set('user1:org2', { userId: 'user1', orgId: 'org2', role: 'viewer' });
memberships.set('user2:org1', { userId: 'user2', orgId: 'org1', role: 'editor' });

// ロール定義（演習1のものを再利用）
const ROLES: Record<string, { permissions: string[]; inherits: string[] }> = {
  viewer: { permissions: ['articles:read'], inherits: [] },
  editor: {
    permissions: ['articles:create', 'articles:update'],
    inherits: ['viewer'],
  },
  admin: {
    permissions: ['users:read', 'users:update', 'articles:delete', 'org:settings'],
    inherits: ['editor'],
  },
};

function resolvePerms(role: string, visited = new Set<string>()): Set<string> {
  if (visited.has(role)) return new Set();
  visited.add(role);
  const config = ROLES[role];
  if (!config) return new Set();
  const perms = new Set<string>(config.permissions);
  for (const parent of config.inherits) {
    for (const p of resolvePerms(parent, visited)) perms.add(p);
  }
  return perms;
}

// ミドルウェア
function requireOrgPermission(...permissions: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    const userId = (req as any).userId; // 認証済みと仮定
    const orgId = req.params.orgId;

    if (!userId) return res.status(401).json({ error: 'Unauthorized' });
    if (!orgId) return res.status(400).json({ error: 'Organization ID required' });

    const membership = memberships.get(`${userId}:${orgId}`);
    if (!membership) {
      return res.status(403).json({ error: 'Not a member of this organization' });
    }

    const userPerms = resolvePerms(membership.role);
    const hasAll = permissions.every((p) => userPerms.has(p));

    if (!hasAll) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        role: membership.role,
        required: permissions,
        granted: [...userPerms],
      });
    }

    (req as any).orgId = orgId;
    (req as any).orgRole = membership.role;
    next();
  };
}

// テスト用の簡易サーバー
const app = express();

// 認証をシミュレート
app.use((req, res, next) => {
  (req as any).userId = req.headers['x-user-id'] as string;
  next();
});

app.get('/org/:orgId/articles', requireOrgPermission('articles:read'), (req, res) => {
  res.json({ articles: [], role: (req as any).orgRole });
});

app.delete('/org/:orgId/articles/:id', requireOrgPermission('articles:delete'), (req, res) => {
  res.json({ deleted: req.params.id });
});

app.get('/org/:orgId/settings', requireOrgPermission('org:settings'), (req, res) => {
  res.json({ settings: {} });
});

// テスト実行例:
// user1 (admin in org1): GET /org/org1/articles → 200
// user1 (viewer in org2): DELETE /org/org2/articles/1 → 403
// user2 (editor in org1): GET /org/org1/settings → 403
```

</details>

### 演習3: 発展 - Redis キャッシュ付き権限チェッカーの設計

**課題**: 以下の要件を満たす権限チェッカークラスを設計してください。

1. DB からの権限取得結果を Redis にキャッシュ（TTL: 5分）
2. ロール変更時にキャッシュを即座に無効化
3. キャッシュミス時のみ DB アクセス
4. マルチテナント対応（orgId 別にキャッシュ）

<details>
<summary>模範解答</summary>

```typescript
import Redis from 'ioredis';

class CachedPermissionChecker {
  private redis: Redis;
  private ttl = 300; // 5分

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }

  private cacheKey(userId: string, orgId?: string): string {
    return orgId ? `perms:v1:${userId}:${orgId}` : `perms:v1:${userId}`;
  }

  // 権限チェック（キャッシュ優先）
  async hasPermission(
    userId: string,
    permission: string,
    orgId?: string
  ): Promise<boolean> {
    const permissions = await this.getPermissions(userId, orgId);
    return permissions.has(permission);
  }

  // 権限セットを取得（キャッシュ優先）
  async getPermissions(userId: string, orgId?: string): Promise<Set<string>> {
    const key = this.cacheKey(userId, orgId);

    // 1. キャッシュチェック
    const cached = await this.redis.smembers(key);
    if (cached.length > 0) {
      return new Set(cached);
    }

    // 2. DB から取得
    const permissions = orgId
      ? await this.fetchOrgPermissionsFromDB(userId, orgId)
      : await this.fetchPermissionsFromDB(userId);

    // 3. キャッシュに保存
    if (permissions.size > 0) {
      const pipeline = this.redis.pipeline();
      pipeline.del(key);
      pipeline.sadd(key, ...permissions);
      pipeline.expire(key, this.ttl);
      await pipeline.exec();
    }

    return permissions;
  }

  // キャッシュ無効化
  async invalidateUser(userId: string): Promise<void> {
    const pattern = `perms:v1:${userId}*`;
    const keys = await this.redis.keys(pattern);
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
  }

  async invalidateUserOrg(userId: string, orgId: string): Promise<void> {
    await this.redis.del(this.cacheKey(userId, orgId));
  }

  // ロール変更時のフック
  async onRoleChanged(event: {
    userId: string;
    orgId?: string;
    oldRole: string;
    newRole: string;
  }): Promise<void> {
    if (event.orgId) {
      await this.invalidateUserOrg(event.userId, event.orgId);
    } else {
      await this.invalidateUser(event.userId);
    }
    console.log(
      `Cache invalidated for user ${event.userId}: ${event.oldRole} → ${event.newRole}`
    );
  }

  // DB取得（実装は省略、Prisma等で実装）
  private async fetchPermissionsFromDB(userId: string): Promise<Set<string>> {
    // getUserPermissions(userId) を呼び出す
    return new Set(['articles:read']);
  }

  private async fetchOrgPermissionsFromDB(
    userId: string,
    orgId: string
  ): Promise<Set<string>> {
    // getOrgPermissions(userId, orgId) を呼び出す
    return new Set(['articles:read']);
  }
}

// 使用例
async function example() {
  const checker = new CachedPermissionChecker('redis://localhost:6379');

  // 権限チェック（初回はDB、以降はキャッシュ）
  const canRead = await checker.hasPermission('user1', 'articles:read', 'org1');
  console.log('Can read:', canRead);

  // ロール変更時にキャッシュを無効化
  await checker.onRoleChanged({
    userId: 'user1',
    orgId: 'org1',
    oldRole: 'viewer',
    newRole: 'editor',
  });

  // 次回アクセス時はDBから再取得してキャッシュ
}
```

</details>

---

## FAQ

### Q1: RBAC と ABAC はどちらを選ぶべきですか？

基本的には RBAC から始めることを推奨します。多くのアプリケーションでは RBAC で十分です。ABAC が必要になるのは「自分が作成したリソースのみ編集可能」「同じ部署のメンバーのみ閲覧可能」のような属性ベースの制御が必要な場合です。実務では RBAC + ABAC のハイブリッドが一般的で、粗い制御を RBAC、細かい制御を ABAC で行います。詳しくは [ABAC とポリシー](./01-abac-and-policies.md) を参照してください。

### Q2: ロールの数はどのくらいが適切ですか？

3〜10 ロールが管理しやすい範囲です。ロールが 20 を超えると管理が困難になり、ABAC の導入を検討すべきです。「ロール爆発」（Role Explosion）は RBAC の典型的な問題で、リソースや条件ごとにロールを作ると組合せ爆発が起きます。ロールはあくまで「役割」であり、細かい条件分岐は ABAC（属性ベース）で処理すべきです。

### Q3: 権限チェックを API の各エンドポイントで書くのは面倒です。自動化できますか？

ミドルウェアパターンを使えば、ルート定義時に宣言的に権限を指定できます（本章セクション4参照）。さらに、OpenAPI スキーマに権限情報を含めて自動生成するアプローチや、デコレーターパターン（NestJS の `@UseGuards`）を使う方法もあります。

### Q4: 権限のキャッシュを使う場合、ロール変更の即時反映はどうしますか？

Redis キャッシュを使う場合、ロール変更時にキャッシュを明示的に無効化します（セクション6参照）。キャッシュの TTL を 5分程度にしておけば、無効化に失敗しても最大5分で反映されます。即時性が重要な場合は、Token Version と組み合わせてトークンレベルで失効させます。

### Q5: フロントエンドでもロールチェックが必要ですか？

フロントエンドでのロールチェックは UX の最適化（権限のないボタンを非表示にする等）のために行いますが、セキュリティの保証はバックエンド API で行うべきです。DevTools でフロントエンドのチェックを回避できるため、フロントエンドのみの権限チェックは危険です。詳しくは [フロントエンド認可](./03-frontend-authorization.md) を参照してください。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 権限命名 | `resource:action` 形式で統一 |
| ロール階層 | 継承で権限の重複定義を排除（RBAC1） |
| DB設計 | 小規模: role カラム、大規模: 5テーブル正規化 |
| ミドルウェア | `requirePermission()` で宣言的にチェック |
| マルチテナント | 組織ごとにロール割当（organization_members） |
| キャッシュ | Redis で権限チェックを高速化（TTL: 5分） |
| 制約 | 相互排他ロール、職務分離で不正を防止 |
| デフォルト | 拒否がデフォルト（最小権限の原則） |

---

## 次に読むべきガイド

- [ABAC とポリシーエンジン](./01-abac-and-policies.md) - RBAC で表現しきれない細かい制御
- [API 認可](./02-api-authorization.md) - スコープベースの API アクセス制御
- [フロントエンド認可](./03-frontend-authorization.md) - UI での権限制御パターン

---

## 参考文献

1. NIST. "Role Based Access Control." NIST SP 359, csrc.nist.gov, 2004.
2. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. Sandhu, R. et al. "Role-Based Access Control Models." IEEE Computer, 1996.
4. CASL. "RBAC with CASL." casl.js.org, 2024.
5. Prisma. "Modeling Relations." prisma.io/docs, 2024.
