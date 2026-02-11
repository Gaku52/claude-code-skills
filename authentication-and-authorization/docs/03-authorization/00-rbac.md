# RBAC（ロールベースアクセス制御）

> RBAC はユーザーに「ロール」を割り当て、ロールに「権限」を紐付ける最も普及したアクセス制御モデル。ロール設計、権限モデル、階層ロール、マルチテナント対応まで、実践的な RBAC の設計と実装を解説する。

## この章で学ぶこと

- [ ] RBAC の基本概念とモデルを理解する
- [ ] 権限モデルの設計パターンを把握する
- [ ] 階層ロールとマルチテナント RBAC を実装できるようになる

---

## 1. RBAC の基本モデル

```
RBAC の構成要素:

  ┌──────────┐    ┌──────────┐    ┌──────────────┐
  │  User    │───→│  Role    │───→│  Permission  │
  │ ユーザー  │ N:M│  ロール   │ N:M│  権限         │
  └──────────┘    └──────────┘    └──────────────┘

  User → Role:    ユーザーは1つ以上のロールを持つ
  Role → Permission: ロールは1つ以上の権限を持つ
  User → Permission: ユーザーはロール経由で権限を得る

RBAC のレベル:

  RBAC0（基本）:
  → ユーザー ← ロール → 権限
  → 最もシンプルな形

  RBAC1（階層ロール）:
  → ロール間の継承関係
  → admin は editor の権限を含む

  RBAC2（制約付き）:
  → 相互排他ロール（同時に持てないロール）
  → 最大ロール数の制限
  → 職務分離（Separation of Duties）

  RBAC3（RBAC1 + RBAC2）:
  → 階層 + 制約の両方
```

---

## 2. ロールと権限の設計

```
権限の命名規則: resource:action

  記事管理:
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

ロール設計例（CMS）:

  viewer:
    → articles:read

  editor:
    → articles:read, articles:create, articles:update

  publisher:
    → editor の全権限 + articles:publish, articles:delete

  admin:
    → publisher の全権限 + users:*, org:settings

  super_admin:
    → 全権限
```

```typescript
// 権限定義
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
} as const;

type Permission = keyof typeof PERMISSIONS;

// ロール定義（階層付き）
const ROLES = {
  viewer: {
    permissions: ['articles:read'] as Permission[],
    inherits: [],
  },
  editor: {
    permissions: ['articles:create', 'articles:update'] as Permission[],
    inherits: ['viewer'],
  },
  publisher: {
    permissions: ['articles:publish', 'articles:delete'] as Permission[],
    inherits: ['editor'],
  },
  admin: {
    permissions: [
      'users:read', 'users:create', 'users:update',
      'users:delete', 'users:invite', 'org:settings',
    ] as Permission[],
    inherits: ['publisher'],
  },
  super_admin: {
    permissions: ['org:billing'] as Permission[],
    inherits: ['admin'],
  },
} as const;

type Role = keyof typeof ROLES;

// ロールの全権限を解決（継承込み）
function resolvePermissions(role: Role): Set<Permission> {
  const permissions = new Set<Permission>();
  const roleConfig = ROLES[role];

  // 直接の権限
  roleConfig.permissions.forEach((p) => permissions.add(p));

  // 継承された権限（再帰的に解決）
  roleConfig.inherits.forEach((parentRole) => {
    resolvePermissions(parentRole as Role).forEach((p) => permissions.add(p));
  });

  return permissions;
}

// 権限チェック
function hasPermission(userRole: Role, permission: Permission): boolean {
  const permissions = resolvePermissions(userRole);
  return permissions.has(permission);
}

// 使用例
hasPermission('editor', 'articles:read');     // true（viewer から継承）
hasPermission('editor', 'articles:create');   // true（直接の権限）
hasPermission('editor', 'articles:publish');  // false（publisher 以上）
hasPermission('admin', 'articles:publish');   // true（publisher から継承）
```

---

## 3. データベース設計

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
                                   │               │
                                   └──────────────┘
```

```typescript
// Prisma スキーマ
// schema.prisma

// model User {
//   id        String     @id @default(cuid())
//   email     String     @unique
//   name      String
//   roles     UserRole[]
// }
//
// model Role {
//   id          String           @id @default(cuid())
//   name        String           @unique
//   description String?
//   permissions RolePermission[]
//   users       UserRole[]
//   parentId    String?          // 階層ロール
//   parent      Role?            @relation("RoleHierarchy", fields: [parentId], references: [id])
//   children    Role[]           @relation("RoleHierarchy")
// }
//
// model Permission {
//   id          String           @id @default(cuid())
//   name        String           @unique  // "articles:read"
//   description String?
//   roles       RolePermission[]
// }
//
// model UserRole {
//   userId String
//   roleId String
//   user   User @relation(fields: [userId], references: [id])
//   role   Role @relation(fields: [roleId], references: [id])
//   @@id([userId, roleId])
// }
//
// model RolePermission {
//   roleId       String
//   permissionId String
//   role         Role       @relation(fields: [roleId], references: [id])
//   permission   Permission @relation(fields: [permissionId], references: [id])
//   @@id([roleId, permissionId])
// }

// DB からユーザーの全権限を取得
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
            },
          },
        },
      },
    },
  });

  const permissions = new Set<string>();
  user?.roles.forEach(({ role }) => {
    role.permissions.forEach(({ permission }) => {
      permissions.add(permission.name);
    });
  });

  return permissions;
}
```

---

## 4. ミドルウェアでの権限チェック

```typescript
// Express ミドルウェア
function requirePermission(...requiredPermissions: string[]) {
  return async (req: Request, res: Response, next: Function) => {
    const user = req.user; // 認証ミドルウェアで設定済み
    if (!user) {
      return res.status(401).json({ error: 'Authentication required' });
    }

    const userPermissions = await getUserPermissions(user.id);

    const hasAll = requiredPermissions.every((p) => userPermissions.has(p));
    if (!hasAll) {
      return res.status(403).json({
        error: 'Insufficient permissions',
        required: requiredPermissions,
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
```

---

## 5. マルチテナント RBAC

```
マルチテナント RBAC:

  組織ごとにロールが異なる:
    Alice: Organization A → admin
    Alice: Organization B → viewer

  テーブル設計:
  ┌─────────┐   ┌─────────────────────┐   ┌───────────┐
  │  users  │──→│ organization_members │←──│ orgs      │
  │         │   │ (user_id, org_id,   │   │           │
  │         │   │  role)              │   │           │
  └─────────┘   └─────────────────────┘   └───────────┘
```

```typescript
// マルチテナント RBAC
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
  return hasPermission(membership.role as Role, permission as Permission);
}

// Next.js ミドルウェアでの組織コンテキスト
async function withOrgAuth(
  req: Request,
  permission: string
): Promise<{ user: User; org: Organization }> {
  const user = await getAuthenticatedUser(req);
  if (!user) throw new AuthError('Unauthorized', 401);

  // URL から orgId を取得（/org/[orgId]/...）
  const orgId = getOrgIdFromUrl(req.url);
  if (!orgId) throw new AuthError('Organization not specified', 400);

  const hasAccess = await checkOrgPermission(user.id, orgId, permission);
  if (!hasAccess) throw new AuthError('Forbidden', 403);

  const org = await prisma.organization.findUnique({ where: { id: orgId } });
  return { user, org: org! };
}
```

---

## 6. 権限のキャッシング

```typescript
// Redis キャッシュで権限チェックを高速化
class PermissionCache {
  constructor(private redis: Redis) {}

  private key(userId: string, orgId?: string): string {
    return orgId ? `perms:${userId}:${orgId}` : `perms:${userId}`;
  }

  async get(userId: string, orgId?: string): Promise<Set<string> | null> {
    const cached = await this.redis.smembers(this.key(userId, orgId));
    return cached.length > 0 ? new Set(cached) : null;
  }

  async set(userId: string, permissions: Set<string>, orgId?: string): Promise<void> {
    const key = this.key(userId, orgId);
    const pipeline = this.redis.pipeline();
    pipeline.del(key);
    if (permissions.size > 0) {
      pipeline.sadd(key, ...permissions);
    }
    pipeline.expire(key, 300); // 5分キャッシュ
    await pipeline.exec();
  }

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
}
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 権限命名 | resource:action 形式 |
| ロール階層 | 継承で権限の重複を排除 |
| DB設計 | User ← UserRole → Role ← RolePermission → Permission |
| マルチテナント | 組織ごとにロール割当 |
| キャッシュ | Redis で権限チェックを高速化 |
| デフォルト | 拒否がデフォルト（最小権限） |

---

## 次に読むべきガイド
→ [[01-abac-and-policies.md]] — ABAC とポリシー

---

## 参考文献
1. NIST. "Role Based Access Control." csrc.nist.gov, 2004.
2. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
