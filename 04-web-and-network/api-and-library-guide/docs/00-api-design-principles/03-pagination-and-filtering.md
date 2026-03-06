# ページネーションとフィルタリング

> 大量データを効率的に返すためのページネーション、フィルタリング、ソート、検索の設計パターン。Offset / Cursor / Keyset 方式の比較、GraphQL Relay Connection 仕様、フィルタ構文、全文検索まで、データ取得 API の全技法を網羅する。

## この章で学ぶこと

- [ ] Offset 方式・Cursor 方式・Keyset 方式の違いと選定基準を理解する
- [ ] GraphQL Relay Connection 仕様によるページネーションを実装できる
- [ ] フィルタリングとソートの API 設計を把握する
- [ ] 全文検索・ファセット検索の設計を学ぶ
- [ ] ページネーションに関するパフォーマンス最適化を習得する
- [ ] 各方式のエッジケースとアンチパターンを理解する

---

## 1. ページネーション方式の全体像

API が返すデータセットが大きくなるにつれ、一度のレスポンスで全件を返すことは
ネットワーク帯域・メモリ・レスポンスタイムの観点から現実的でなくなる。
ページネーション（Pagination）は、データセットを小さなチャンク（ページ）に分割し、
クライアントが必要な部分だけを取得できるようにする手法である。

### 1.1 三大ページネーション方式の比較

```
┌─────────────────────────────────────────────────────────────────┐
│              ページネーション方式の分類                           │
├──────────────┬──────────────────┬───────────────────────────────┤
│  Offset 方式  │   Cursor 方式    │       Keyset 方式             │
│  (ページ番号)  │   (不透明トークン) │    (ソートキー直接指定)        │
├──────────────┼──────────────────┼───────────────────────────────┤
│  page=3      │  cursor=abc123   │  created_at_gt=2024-01-15     │
│  per_page=20 │  limit=20        │  id_gt=100&limit=20           │
├──────────────┼──────────────────┼───────────────────────────────┤
│  SQL:        │  SQL:            │  SQL:                         │
│  OFFSET 40   │  WHERE (col,id)  │  WHERE created_at > ?         │
│  LIMIT 20    │    < (?,?)       │    AND id > ?                 │
│              │  LIMIT 20        │  LIMIT 20                     │
├──────────────┼──────────────────┼───────────────────────────────┤
│  O(n) skip   │  O(log n) seek   │  O(log n) seek               │
│  ページジャンプ○│  ページジャンプ× │  ページジャンプ×              │
│  位置ずれ有り  │  位置ずれ無し    │  位置ずれ無し                 │
└──────────────┴──────────────────┴───────────────────────────────┘
```

> **Cursor 方式と Keyset 方式の違い**: Cursor 方式はソートキーをBase64等で
> エンコードした「不透明トークン」を使い、クライアントはその中身を知る必要がない。
> Keyset 方式はソートキーの値をそのままクエリパラメータに露出する。
> 本質的なSQL実行計画は同じだが、APIの抽象度が異なる。

---

### 1.2 Offset 方式（ページ番号ベース）

最も直感的なページネーション方式。SQL の `OFFSET` / `LIMIT` に直接対応する。

```
GET /api/v1/users?page=3&per_page=20

レスポンス:
{
  "data": [...],
  "meta": {
    "total": 1500,
    "page": 3,
    "perPage": 20,
    "totalPages": 75,
    "hasNextPage": true,
    "hasPrevPage": true
  },
  "links": {
    "self":  "/api/v1/users?page=3&per_page=20",
    "first": "/api/v1/users?page=1&per_page=20",
    "prev":  "/api/v1/users?page=2&per_page=20",
    "next":  "/api/v1/users?page=4&per_page=20",
    "last":  "/api/v1/users?page=75&per_page=20"
  }
}

内部SQL:
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 20 OFFSET 40;  -- (page - 1) * per_page

利点:
  直感的（「3ページ目」が明確）
  任意のページにジャンプ可能
  UIにページ番号を表示しやすい
  既存のSQLと親和性が高い

欠点:
  OFFSET が大きいとパフォーマンス劣化
    → OFFSET 100000 は10万行スキップ（O(n)）
  データの追加/削除で位置ずれ
    → ページ2を見ている間にデータが挿入されると重複表示
  totalのCOUNTクエリが重い（大規模テーブル）
```

#### Offset 方式の内部動作を図解する

```
データベースのスキャン動作（page=5001, per_page=20 の場合）:

  Row 1      ─┐
  Row 2       │
  Row 3       │
  ...         │  ← OFFSET 100000: これらの行を全てスキャン
  Row 99999   │     してからスキップする（O(n) コスト）
  Row 100000 ─┘
  Row 100001 ─┐
  Row 100002  │
  ...         │  ← LIMIT 20: この20行だけを返す
  Row 100020 ─┘
  Row 100021
  ...

  つまり page が大きくなるほどスキャン量が増え、
  レスポンスタイムは線形的に悪化する:

  page=1    →  ~2ms
  page=100  →  ~15ms
  page=1000 →  ~120ms
  page=5000 →  ~600ms
  page=10000→  ~1200ms （テーブルサイズ依存）
```

#### Offset 方式の位置ずれ問題

```
Timeline:
  T1: Client が page=2 (id=21〜40) を取得
  T2: 別ユーザーが id=25 のデータを削除
  T3: Client が page=3 を取得
      → 本来 id=41〜60 だが、削除により id=42〜61 になる
      → id=41 が page=2 の末尾と page=3 の先頭のどちらにも含まれない
         （データの「穴」が発生）

  逆に挿入の場合:
  T1: Client が page=1 (id=1〜20) を取得
  T2: 新しいデータが先頭に挿入される（id=0 相当）
  T3: Client が page=2 を取得
      → id=20 が page=1 にも page=2 にも含まれる（重複）

  この問題は「ページドリフト」と呼ばれる。
```

---

### 1.3 Cursor 方式（不透明トークンベース）

ソートキーの値をエンコードした不透明なトークン（cursor）を使い、
「この位置の次から N 件」を取得する方式。

```
GET /api/v1/users?cursor=eyJpZCI6MTAwfQ&limit=20

レスポンス:
{
  "data": [...],
  "meta": {
    "hasNextPage": true,
    "nextCursor": "eyJpZCI6MTIwfQ",
    "hasPrevPage": true,
    "prevCursor": "eyJpZCI6MTAxfQ"
  }
}

cursorの中身（Base64エンコード）:
  {"id": 100, "createdAt": "2024-01-15T10:00:00Z"}

内部SQL:
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15T10:00:00Z', 100)
ORDER BY created_at DESC, id DESC
LIMIT 20;

利点:
  一定のパフォーマンス（WHERE句でインデックス利用、O(log n)）
  データの追加/削除で位置ずれしない
  リアルタイムフィードに最適

欠点:
  任意のページにジャンプ不可
  ページ番号の表示が困難
  cursor の生成・解析が複雑
  ソート順の変更で既存cursorが無効になる
```

---

### 1.4 Keyset 方式（ソートキー直接指定）

Cursor 方式の変種で、ソートキーの値をクエリパラメータに直接露出する。

```
GET /api/v1/users?created_at_lt=2024-01-15T10:00:00Z&id_lt=100&limit=20

レスポンス:
{
  "data": [...],
  "meta": {
    "hasNextPage": true,
    "nextCreatedAt": "2024-01-14T08:30:00Z",
    "nextId": 80
  }
}

内部SQL（Cursor方式と同一）:
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15T10:00:00Z', 100)
ORDER BY created_at DESC, id DESC
LIMIT 20;

利点:
  cursorのエンコード/デコードが不要
  デバッグしやすい（パラメータが可読）
  クライアントが自由にソートキーを指定可能

欠点:
  内部のソートキーが外部に露出する（API契約が脆い）
  複合ソートキーのパラメータが冗長になる
  ソートキーの型やフォーマットをクライアントが知る必要がある
```

---

### 1.5 方式選定のデシジョンツリー

```
                    ページネーション方式の選定
                           │
                    ページジャンプが必要？
                    ┌──────┴──────┐
                   Yes           No
                    │             │
             データ量 < 10万件？   リアルタイム性が必要？
             ┌──────┴──────┐   ┌──────┴──────┐
            Yes           No  Yes           No
             │             │   │             │
        ┌────┘        ┌───┘   │        データ量 > 100万件？
        │             │       │        ┌──────┴──────┐
   Offset方式    Offset方式   Cursor   Yes           No
   （推奨）     ＋推定total    方式     │             │
                               │    Cursor方式   どちらでも可
                               │   （推奨）     （Cursor推奨）
                               │
                         Cursor方式
                         （推奨）

  具体的なユースケース:
  ┌───────────────────────┬──────────────┐
  │ ユースケース           │ 推奨方式      │
  ├───────────────────────┼──────────────┤
  │ 管理画面のテーブル      │ Offset       │
  │ 検索結果一覧           │ Offset       │
  │ SNSタイムライン        │ Cursor       │
  │ チャット履歴           │ Cursor       │
  │ 通知一覧              │ Cursor       │
  │ 無限スクロール         │ Cursor       │
  │ データエクスポート      │ Keyset       │
  │ バッチ処理             │ Keyset       │
  │ GraphQL API           │ Cursor       │
  │ 公開API（サードパーティ）│ Cursor       │
  └───────────────────────┴──────────────┘
```

---

## 2. Cursor 実装の詳細

### 2.1 基本実装（Node.js + Prisma）

```javascript
// --- Cursor エンコード/デコード ---

/**
 * カーソルデータをBase64urlエンコードする。
 * Base64url を使う理由:
 *   - URL safe（+, /, = を使わない）
 *   - クエリパラメータにそのまま渡せる
 *   - クライアントにとって不透明（opaque）
 */
function encodeCursor(data) {
  return Buffer.from(JSON.stringify(data)).toString('base64url');
}

/**
 * Base64urlエンコードされたカーソルをデコードする。
 * 不正なカーソルに対してはエラーを投げる。
 */
function decodeCursor(cursor) {
  try {
    const decoded = JSON.parse(
      Buffer.from(cursor, 'base64url').toString()
    );
    // バリデーション: 必要なフィールドが存在するか
    if (!decoded.id) {
      throw new Error('Invalid cursor: missing id');
    }
    return decoded;
  } catch (err) {
    throw new ApiError(400, 'Invalid cursor format');
  }
}

// --- Cursorページネーション ---
async function listUsers(params) {
  const {
    cursor,
    limit = 20,
    sort = 'createdAt',
    order = 'desc',
  } = params;

  // limit の上限を設ける（DoS防止）
  const take = Math.min(Math.max(limit, 1), 100);

  // ソートフィールドのホワイトリスト検証
  const allowedSortFields = ['createdAt', 'updatedAt', 'name', 'email'];
  if (!allowedSortFields.includes(sort)) {
    throw new ApiError(400, `Invalid sort field: ${sort}`);
  }

  let where = {};
  if (cursor) {
    const decoded = decodeCursor(cursor);
    // 複合カーソル: ソートキー + ID でタイブレーク
    // (created_at, id) の複合比較で一意性を保証
    where = {
      OR: [
        {
          [sort]: order === 'desc'
            ? { lt: decoded[sort] }
            : { gt: decoded[sort] },
        },
        {
          [sort]: decoded[sort],
          id: order === 'desc'
            ? { lt: decoded.id }
            : { gt: decoded.id },
        },
      ],
    };
  }

  // take + 1 件取得して hasNextPage を判定する技法
  // 余分な1件が取れたら「次のページが存在する」
  const items = await prisma.user.findMany({
    where,
    orderBy: [{ [sort]: order }, { id: order }],
    take: take + 1,
  });

  const hasNextPage = items.length > take;
  const data = hasNextPage ? items.slice(0, take) : items;

  return {
    data,
    meta: {
      hasNextPage,
      nextCursor: hasNextPage
        ? encodeCursor({
            [sort]: data[data.length - 1][sort],
            id: data[data.length - 1].id,
          })
        : null,
      hasPrevPage: !!cursor,
      prevCursor: data.length > 0
        ? encodeCursor({
            [sort]: data[0][sort],
            id: data[0].id,
          })
        : null,
      limit: take,
    },
  };
}
```

### 2.2 複合ソートキーでのカーソル実装

カーソルが単一キーでなく複合キー（例: `(priority, created_at, id)`）の場合、
SQL の `WHERE` 句が複雑になる。これを「行値比較（Row Value Comparison）」で解決する。

```sql
-- 複合ソート: priority DESC, created_at DESC, id DESC
-- カーソル位置: priority=3, created_at='2024-06-01', id=500

-- 方法1: OR条件の展開（全DBで動作）
SELECT * FROM tasks
WHERE
  (priority < 3)
  OR (priority = 3 AND created_at < '2024-06-01')
  OR (priority = 3 AND created_at = '2024-06-01' AND id < 500)
ORDER BY priority DESC, created_at DESC, id DESC
LIMIT 20;

-- 方法2: 行値比較（PostgreSQL, MySQL 8.0+ で動作）
SELECT * FROM tasks
WHERE (priority, created_at, id) < (3, '2024-06-01', 500)
ORDER BY priority DESC, created_at DESC, id DESC
LIMIT 20;

-- 方法2 は簡潔だが、混合ソート順（ASC/DESC混在）には使えない。
-- 混合ソート順の場合は方法1のOR展開が必須。
```

```javascript
// 複合ソートキーのカーソル実装（Node.js）
function buildCursorWhere(sortKeys, cursorData, orders) {
  // sortKeys: ['priority', 'createdAt', 'id']
  // cursorData: { priority: 3, createdAt: '2024-06-01', id: 500 }
  // orders: ['desc', 'desc', 'desc']

  const conditions = [];

  for (let i = 0; i < sortKeys.length; i++) {
    const condition = {};

    // 前のキーが全て等しい
    for (let j = 0; j < i; j++) {
      condition[sortKeys[j]] = cursorData[sortKeys[j]];
    }

    // 現在のキーが比較条件を満たす
    const op = orders[i] === 'desc' ? 'lt' : 'gt';
    condition[sortKeys[i]] = { [op]: cursorData[sortKeys[i]] };

    conditions.push(condition);
  }

  return { OR: conditions };
}

// 使用例
const where = buildCursorWhere(
  ['priority', 'createdAt', 'id'],
  { priority: 3, createdAt: '2024-06-01T00:00:00Z', id: 500 },
  ['desc', 'desc', 'desc']
);
// → { OR: [
//      { priority: { lt: 3 } },
//      { priority: 3, createdAt: { lt: '2024-06-01T00:00:00Z' } },
//      { priority: 3, createdAt: '2024-06-01T00:00:00Z', id: { lt: 500 } },
//   ]}
```

### 2.3 暗号化カーソルとセキュリティ

カーソルの内容がBase64でエンコードされているだけの場合、
クライアントがデコードして改ざんできる。これを防ぐには
HMAC署名や暗号化を施す。

```javascript
const crypto = require('crypto');

const CURSOR_SECRET = process.env.CURSOR_SECRET; // 十分な長さのランダム文字列

/**
 * 署名付きカーソルを生成する。
 * フォーマット: base64url(JSON) + "." + hmac_signature
 */
function encodeSecureCursor(data) {
  const payload = Buffer.from(JSON.stringify(data)).toString('base64url');
  const hmac = crypto
    .createHmac('sha256', CURSOR_SECRET)
    .update(payload)
    .digest('base64url');
  return `${payload}.${hmac}`;
}

/**
 * 署名付きカーソルを検証・デコードする。
 * 署名が不正な場合は例外を投げる。
 */
function decodeSecureCursor(cursor) {
  const [payload, signature] = cursor.split('.');
  if (!payload || !signature) {
    throw new ApiError(400, 'Invalid cursor format');
  }

  // HMAC検証（タイミング攻撃を防ぐため timingSafeEqual を使用）
  const expected = crypto
    .createHmac('sha256', CURSOR_SECRET)
    .update(payload)
    .digest('base64url');

  if (!crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  )) {
    throw new ApiError(400, 'Invalid cursor signature');
  }

  return JSON.parse(Buffer.from(payload, 'base64url').toString());
}
```

---

## 3. GraphQL Relay Connection 仕様

GraphQL における標準的なページネーション仕様として、
Relay の「Connection」パターンがある。
Facebook が策定し、GitHub・Shopify・Stripe 等の主要 GraphQL API で採用されている。

### 3.1 Connection 仕様の構造

```
Connection 仕様の概念モデル:

  Query
    │
    ├── users(first: 10, after: "cursor_abc")
    │     │
    │     └── UsersConnection
    │           │
    │           ├── edges: [UserEdge]
    │           │     ├── edge[0]
    │           │     │     ├── cursor: "cursor_abc1"
    │           │     │     └── node: User { id, name, ... }
    │           │     ├── edge[1]
    │           │     │     ├── cursor: "cursor_abc2"
    │           │     │     └── node: User { id, name, ... }
    │           │     └── ...
    │           │
    │           ├── pageInfo: PageInfo
    │           │     ├── hasNextPage: true
    │           │     ├── hasPreviousPage: false
    │           │     ├── startCursor: "cursor_abc1"
    │           │     └── endCursor: "cursor_abc10"
    │           │
    │           └── totalCount: 1500 （拡張フィールド）
    │
    └── ...

  用語の定義:
  - Connection: ページネーション対応のコレクション型
  - Edge: ノードとカーソルのペア
  - Node: 実際のデータオブジェクト
  - PageInfo: ページネーションメタデータ
  - Cursor: 各エッジの位置を示す不透明な文字列
```

### 3.2 GraphQL スキーマ定義

```graphql
# Connection 仕様に準拠した GraphQL スキーマ

type Query {
  # 前方ページネーション: first + after
  # 後方ページネーション: last + before
  users(
    first: Int
    after: String
    last: Int
    before: String
    filter: UserFilter
    orderBy: UserOrderBy
  ): UserConnection!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int        # 拡張: 総件数
}

type UserEdge {
  cursor: String!
  node: User!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type User {
  id: ID!
  name: String!
  email: String!
  role: UserRole!
  createdAt: DateTime!
}

input UserFilter {
  role: UserRole
  status: UserStatus
  createdAfter: DateTime
  createdBefore: DateTime
  search: String
}

input UserOrderBy {
  field: UserSortField!
  direction: SortDirection!
}

enum UserSortField {
  CREATED_AT
  NAME
  EMAIL
}

enum SortDirection {
  ASC
  DESC
}

enum UserRole {
  ADMIN
  EDITOR
  VIEWER
}

enum UserStatus {
  ACTIVE
  INACTIVE
  SUSPENDED
}
```

### 3.3 Connection リゾルバの実装

```javascript
// GraphQL Relay Connection リゾルバ実装（Node.js + Prisma）

const resolvers = {
  Query: {
    users: async (_, args, context) => {
      const {
        first, after,
        last, before,
        filter, orderBy,
      } = args;

      // first と last の同時指定は禁止
      if (first != null && last != null) {
        throw new UserInputError(
          'Cannot specify both "first" and "last"'
        );
      }

      // どちらも指定なしの場合はデフォルト
      const limit = first ?? last ?? 20;
      const clampedLimit = Math.min(Math.max(limit, 1), 100);

      // フィルタ条件の構築
      const where = buildFilterWhere(filter);

      // ソート条件の構築
      const sort = orderBy
        ? { field: orderBy.field, dir: orderBy.direction }
        : { field: 'CREATED_AT', dir: 'DESC' };

      const sortField = sortFieldMap[sort.field]; // CREATED_AT → createdAt
      const sortDir = sort.dir.toLowerCase();

      // カーソルのデコードと WHERE 条件の追加
      if (after) {
        const cursorData = decodeCursor(after);
        const cursorWhere = buildCursorWhere(
          [sortField, 'id'],
          cursorData,
          [sortDir, sortDir]
        );
        Object.assign(where, cursorWhere);
      }

      if (before) {
        const cursorData = decodeCursor(before);
        // before の場合は逆方向
        const reverseDir = sortDir === 'desc' ? 'asc' : 'desc';
        const cursorWhere = buildCursorWhere(
          [sortField, 'id'],
          cursorData,
          [reverseDir, reverseDir]
        );
        Object.assign(where, cursorWhere);
      }

      // クエリ実行（take + 1 で hasMore 判定）
      let items = await context.prisma.user.findMany({
        where,
        orderBy: [
          { [sortField]: last ? reverseSortDir(sortDir) : sortDir },
          { id: last ? reverseSortDir(sortDir) : sortDir },
        ],
        take: clampedLimit + 1,
      });

      // last の場合は結果を反転
      if (last) {
        items = items.reverse();
      }

      const hasMore = items.length > clampedLimit;
      const nodes = hasMore ? items.slice(0, clampedLimit) : items;

      // totalCount の取得（オプション）
      const totalCount = await context.prisma.user.count({ where });

      // Connection オブジェクトの構築
      const edges = nodes.map(node => ({
        cursor: encodeCursor({
          [sortField]: node[sortField],
          id: node.id,
        }),
        node,
      }));

      return {
        edges,
        pageInfo: {
          hasNextPage: first != null ? hasMore : !!before,
          hasPreviousPage: last != null ? hasMore : !!after,
          startCursor: edges.length > 0 ? edges[0].cursor : null,
          endCursor: edges.length > 0
            ? edges[edges.length - 1].cursor
            : null,
        },
        totalCount,
      };
    },
  },
};

function reverseSortDir(dir) {
  return dir === 'desc' ? 'asc' : 'desc';
}

const sortFieldMap = {
  CREATED_AT: 'createdAt',
  NAME: 'name',
  EMAIL: 'email',
};
```

### 3.4 Connection 仕様のクエリ例

```graphql
# 前方ページネーション（最初の10件、その後続き）
query GetUsers {
  users(first: 10, filter: { role: ADMIN }) {
    edges {
      cursor
      node {
        id
        name
        email
        role
        createdAt
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
    totalCount
  }
}

# 次のページを取得（endCursor を after に渡す）
query GetNextPage {
  users(first: 10, after: "eyJjcmVhdGVkQXQiOiIyMDI0LTAxLTE1IiwiaWQiOjEwMH0") {
    edges {
      cursor
      node {
        id
        name
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}

# 後方ページネーション（最後の5件を取得）
query GetLastUsers {
  users(last: 5) {
    edges {
      cursor
      node {
        id
        name
      }
    }
    pageInfo {
      hasPreviousPage
      startCursor
    }
  }
}
```

---

## 4. 方式別パフォーマンス比較表

### 4.1 基本特性の比較

| 特性 | Offset 方式 | Cursor 方式 | Keyset 方式 | GraphQL Connection |
|------|------------|------------|------------|-------------------|
| ページジャンプ | 可能 | 不可 | 不可 | 不可 |
| 総件数の表示 | 容易 | 別途COUNT必要 | 別途COUNT必要 | totalCount拡張 |
| パフォーマンス | O(n) offset | O(log n) | O(log n) | O(log n) |
| 位置安定性 | ずれる | ずれない | ずれない | ずれない |
| ソート変更 | 容易 | cursor無効化 | パラメータ変更 | cursor無効化 |
| 双方向ナビ | 可能 | 可能 | 可能 | first/last対応 |
| 実装の複雑さ | 低 | 中 | 低〜中 | 高 |
| API抽象度 | 低（SQL漏出） | 高（不透明） | 低（キー露出） | 高（仕様準拠） |
| モバイル適性 | 中 | 高 | 中 | 高 |
| キャッシュ | しやすい | しにくい | しにくい | しにくい |

### 4.2 データ規模別パフォーマンス目安

| データ件数 | Offset (page=末尾) | Cursor | 備考 |
|-----------|-------------------|--------|------|
| 1,000件 | ~1ms | ~1ms | 差は無視できる |
| 10,000件 | ~5ms | ~1ms | Offsetでもまだ許容範囲 |
| 100,000件 | ~50ms | ~2ms | Offsetの劣化が顕在化 |
| 1,000,000件 | ~500ms | ~3ms | Offsetは本番で問題になる |
| 10,000,000件 | ~5000ms | ~5ms | Offsetは事実上使用不可 |
| 100,000,000件 | timeout | ~8ms | Cursorのみ現実的 |

> 上記の値はインデックスが適切に設定された PostgreSQL 環境での参考値であり、
> ハードウェア・データ分布・同時接続数によって大きく変動する。

---

## 5. フィルタリング設計

### 5.1 フィルタリングパターンの全体像

```
フィルタリングのAPI設計パターン:

  (1) シンプルなクエリパラメータ（推奨・小規模API向け）:
    GET /api/v1/users?status=active&role=admin&age_min=18&age_max=65

    → 単純なフィルタに最適
    → フィールド名がそのままパラメータ名

  (2) フィルタ演算子パターン（中規模API向け）:
    GET /api/v1/users?filter[status]=active
    GET /api/v1/users?filter[age][gte]=18&filter[age][lte]=65
    GET /api/v1/users?filter[name][contains]=taro

    演算子一覧:
    ┌──────────┬────────────────┬────────────────────────────────┐
    │ 演算子    │ 意味           │ 例                              │
    ├──────────┼────────────────┼────────────────────────────────┤
    │ eq       │ 等しい          │ filter[status][eq]=active      │
    │ ne       │ 等しくない      │ filter[status][ne]=deleted     │
    │ gt       │ より大きい      │ filter[age][gt]=18             │
    │ gte      │ 以上           │ filter[age][gte]=18            │
    │ lt       │ より小さい      │ filter[age][lt]=65             │
    │ lte      │ 以下           │ filter[age][lte]=65            │
    │ in       │ 含まれる        │ filter[role][in]=admin,editor  │
    │ nin      │ 含まれない      │ filter[role][nin]=guest        │
    │ contains │ 部分一致        │ filter[name][contains]=taro    │
    │ starts   │ 前方一致        │ filter[name][starts]=ta        │
    │ exists   │ 存在する        │ filter[avatar][exists]=true    │
    │ between  │ 範囲           │ filter[age][between]=18,65     │
    └──────────┴────────────────┴────────────────────────────────┘

  (3) JSON API 仕様:
    GET /api/v1/users?filter[status]=active&filter[role]=admin

  (4) RHS Colon:
    GET /api/v1/users?status=eq:active&age=gte:18&age=lte:65

  (5) LHS Brackets:
    GET /api/v1/users?status[eq]=active&age[gte]=18

  推奨:
    → 小規模API: (1) シンプルパターン
    → 中規模API: (2) フィルタ演算子
    → 複雑な検索: 専用の検索エンドポイント（POST /search）
```

### 5.2 フィルタパーサーの実装

```javascript
// フィルタパーサーの実装例（セキュリティ考慮済み）

/**
 * クエリパラメータからフィルタ条件を抽出する。
 * filter[field][operator] 形式をパースする。
 *
 * セキュリティ上の重要ポイント:
 * - 許可されたフィールドのみ受け付ける（ホワイトリスト）
 * - 許可された演算子のみ受け付ける
 * - 値のサニタイゼーション
 */
function parseFilters(query, schema) {
  const filters = {};

  // スキーマ定義（許可フィールドと型情報）
  const allowedFields = schema || {
    status:    { type: 'enum',    values: ['active', 'inactive', 'suspended'] },
    role:      { type: 'enum',    values: ['admin', 'editor', 'viewer'] },
    age:       { type: 'integer', min: 0, max: 200 },
    name:      { type: 'string',  maxLength: 100 },
    email:     { type: 'string',  maxLength: 254 },
    createdAt: { type: 'datetime' },
  };

  const allowedOperators = [
    'eq', 'ne', 'gt', 'gte', 'lt', 'lte',
    'in', 'nin', 'contains', 'starts', 'exists', 'between',
  ];

  for (const [key, value] of Object.entries(query)) {
    // filter[field][operator] パターンのパース
    const match = key.match(/^filter\[(\w+)\](?:\[(\w+)\])?$/);
    if (!match) continue;

    const field = match[1];
    const operator = match[2] || 'eq';

    // フィールド検証
    if (!allowedFields[field]) {
      continue; // 未知のフィールドは無視（エラーにしてもよい）
    }

    // 演算子検証
    if (!allowedOperators.includes(operator)) {
      continue;
    }

    // 値のバリデーション
    const validated = validateFilterValue(
      value, allowedFields[field], operator
    );
    if (validated === null) continue;

    if (!filters[field]) filters[field] = {};

    if (operator === 'in' || operator === 'nin') {
      filters[field][operator] = value.split(',').map(v => v.trim());
    } else if (operator === 'between') {
      const [min, max] = value.split(',').map(v => v.trim());
      filters[field]['gte'] = min;
      filters[field]['lte'] = max;
    } else {
      filters[field][operator] = validated;
    }
  }

  return filters;
}

/**
 * フィルタ値のバリデーション
 */
function validateFilterValue(value, fieldSchema, operator) {
  switch (fieldSchema.type) {
    case 'enum':
      if (operator === 'in' || operator === 'nin') {
        const values = value.split(',');
        return values.every(v => fieldSchema.values.includes(v.trim()))
          ? value : null;
      }
      return fieldSchema.values.includes(value) ? value : null;

    case 'integer': {
      const num = parseInt(value, 10);
      if (isNaN(num)) return null;
      if (fieldSchema.min != null && num < fieldSchema.min) return null;
      if (fieldSchema.max != null && num > fieldSchema.max) return null;
      return num;
    }

    case 'string':
      if (value.length > (fieldSchema.maxLength || 1000)) return null;
      // SQLインジェクション対策: パラメータバインディングで処理するため
      // ここでのエスケープは不要だが、長さは制限する
      return value;

    case 'datetime': {
      const date = new Date(value);
      return isNaN(date.getTime()) ? null : value;
    }

    default:
      return value;
  }
}

// Prisma WHERE句への変換
function filtersToPrismaWhere(filters) {
  const where = {};
  const operatorMap = {
    eq: 'equals', ne: 'not', gt: 'gt', gte: 'gte',
    lt: 'lt', lte: 'lte', in: 'in', nin: 'notIn',
    contains: 'contains', starts: 'startsWith',
    exists: (v) => v === 'true' ? { not: null } : null,
  };

  for (const [field, ops] of Object.entries(filters)) {
    where[field] = {};
    for (const [op, value] of Object.entries(ops)) {
      const prismaOp = operatorMap[op];
      if (typeof prismaOp === 'function') {
        where[field] = prismaOp(value);
      } else {
        where[field][prismaOp] = value;
      }
    }
  }

  return where;
}
```

---

## 6. ソート設計

### 6.1 ソートパラメータの設計パターン

```
ソートのAPI設計:

  (1) シンプルなパラメータ:
    GET /api/v1/users?sort=created_at&order=desc
    GET /api/v1/users?sort=-created_at        ← -プレフィックスで降順

  (2) 複数フィールドソート:
    GET /api/v1/users?sort=-created_at,name
    → created_at降順 → name昇順

  (3) JSON API 仕様:
    GET /api/v1/users?sort=-created_at,name

ソートの注意点:
  [推奨] ソート可能なフィールドをホワイトリストで制限
  [推奨] デフォルトソートを必ず定義（例: -created_at）
  [推奨] ソートフィールドにインデックスを張る
  [推奨] Cursor方式ではソートキーをcursorに含める
  [推奨] ソートの最後に必ず一意キー（id）を追加する（安定ソート）
  [禁止] ユーザー入力をそのままORDER BYに渡さない
```

### 6.2 ソートパーサーの実装

```javascript
// ソートパーサー（安定ソート保証付き）
function parseSort(sortParam, allowedFields) {
  const DEFAULT_SORT = [{ createdAt: 'desc' }, { id: 'desc' }];

  if (!sortParam) return DEFAULT_SORT;

  const orderBy = sortParam.split(',').map(field => {
    const desc = field.startsWith('-');
    const name = desc ? field.slice(1) : field;

    // ホワイトリスト検証
    if (!allowedFields.includes(name)) {
      throw new ApiError(400, `Invalid sort field: ${name}`);
    }

    // snake_case → camelCase 変換
    const camelName = name.replace(/_([a-z])/g, (_, c) => c.toUpperCase());

    return { [camelName]: desc ? 'desc' : 'asc' };
  });

  // 安定ソートのため、最後に id を追加（重複がなければ）
  const hasId = orderBy.some(o => 'id' in o);
  if (!hasId) {
    // 最初のソートの方向に合わせる
    const firstDir = Object.values(orderBy[0])[0];
    orderBy.push({ id: firstDir });
  }

  return orderBy;
}

// 使用例
const orderBy = parseSort(
  req.query.sort,    // "-created_at,name"
  ['created_at', 'name', 'email', 'updated_at']
);
// → [{ createdAt: 'desc' }, { name: 'asc' }, { id: 'desc' }]
```

---

## 7. フィールド選択（Sparse Fieldsets）

```
不要なフィールドを除外してレスポンスサイズを削減:

  GET /api/v1/users?fields=id,name,email
  GET /api/v1/users?fields[users]=id,name&fields[orders]=id,total

レスポンス（指定フィールドのみ）:
  {
    "data": [
      { "id": "1", "name": "Taro", "email": "taro@example.com" },
      { "id": "2", "name": "Hanako", "email": "hanako@example.com" }
    ]
  }

利点:
  レスポンスサイズの削減
  ネットワーク帯域の節約
  モバイルアプリで特に有効
  DBクエリのSELECT最適化

注意:
  → id は常に含める（クライアントの参照整合性のため）
  → セキュリティ上返してはいけないフィールドのチェック
  → GraphQL はスキーマレベルでこの機能を本質的に備える
  → フィールド選択はキャッシュキーに含める必要がある
```

```javascript
// フィールド選択の実装
function parseFields(fieldsParam, allowedFields) {
  if (!fieldsParam) return undefined; // 全フィールド返却

  const requested = fieldsParam.split(',').map(f => f.trim());

  // ホワイトリスト検証
  const valid = requested.filter(f => allowedFields.includes(f));

  // id は常に含める
  if (!valid.includes('id')) {
    valid.unshift('id');
  }

  // Prisma の select に変換
  const select = {};
  for (const field of valid) {
    select[field] = true;
  }

  return select;
}

// 使用例
const select = parseFields(
  req.query.fields,
  ['id', 'name', 'email', 'role', 'createdAt', 'updatedAt']
);
// fields=name,email → { id: true, name: true, email: true }
```

---

## 8. 検索設計

### 8.1 検索の API 設計パターン

```
検索のAPI設計:

  (1) シンプル検索（全文検索）:
    GET /api/v1/users?q=taro
    → name, email 等の複数フィールドを横断検索

  (2) 詳細検索（フィルタ + 検索の組み合わせ）:
    GET /api/v1/users?q=taro&filter[role]=admin&sort=-relevance

  (3) 専用検索エンドポイント:
    POST /api/v1/search
    {
      "query": "taro",
      "filters": {
        "role": ["admin", "editor"],
        "createdAt": { "gte": "2024-01-01" }
      },
      "sort": ["-_score", "name"],
      "page": { "limit": 20, "offset": 0 },
      "facets": ["role", "department"],
      "highlight": {
        "fields": ["name", "bio"],
        "preTag": "<mark>",
        "postTag": "</mark>"
      }
    }

    レスポンス:
    {
      "data": [
        {
          "id": "42",
          "name": "Yamada Taro",
          "_score": 15.3,
          "_highlight": {
            "name": "Yamada <mark>Taro</mark>"
          }
        }
      ],
      "meta": {
        "total": 42,
        "maxScore": 15.3,
        "took": 12
      },
      "facets": {
        "role": [
          { "value": "admin", "count": 15 },
          { "value": "editor", "count": 27 }
        ],
        "department": [
          { "value": "engineering", "count": 30 },
          { "value": "design", "count": 12 }
        ]
      }
    }
```

### 8.2 検索バックエンドの選定

```
検索バックエンドの比較:

┌──────────────┬────────────┬───────────┬──────────┬───────────────┐
│ バックエンド   │ 全文検索    │ ファセット │ 運用コスト │ 適したケース    │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ PostgreSQL   │ tsvector   │ GROUP BY  │ 低       │ 〜100万件      │
│ (pg_trgm)    │ tsquery    │           │          │ 既にPG使用中   │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ Elasticsearch│ BM25       │ Aggs      │ 高       │ 100万件〜      │
│              │ アナライザ  │ Bucket    │          │ 高度な検索要件  │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ OpenSearch   │ BM25       │ Aggs      │ 中〜高   │ AWS環境        │
│              │ アナライザ  │ Bucket    │          │ ES互換が必要   │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ Meilisearch  │ 組み込み    │ 組み込み   │ 低       │ 〜1000万件     │
│              │ Typo耐性   │           │          │ 簡易セットアップ │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ Typesense    │ 組み込み    │ 組み込み   │ 低       │ 〜1000万件     │
│              │ 型付き     │           │          │ 型安全性重視   │
├──────────────┼────────────┼───────────┼──────────┼───────────────┤
│ Algolia      │ ホスティッド│ 組み込み   │ 高       │ 規模問わず     │
│              │ SaaS       │           │          │ 即座に導入したい │
└──────────────┴────────────┴───────────┴──────────┴───────────────┘
```

### 8.3 PostgreSQL 全文検索の実装

```sql
-- PostgreSQL での全文検索セットアップ

-- 1. tsvector カラムの追加
ALTER TABLE users ADD COLUMN search_vector tsvector;

-- 2. トリガーで自動更新
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS trigger AS $$
BEGIN
  NEW.search_vector :=
    setweight(to_tsvector('simple', COALESCE(NEW.name, '')), 'A') ||
    setweight(to_tsvector('simple', COALESCE(NEW.email, '')), 'B') ||
    setweight(to_tsvector('simple', COALESCE(NEW.bio, '')), 'C');
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_search_vector_trigger
  BEFORE INSERT OR UPDATE ON users
  FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- 3. GIN インデックスの作成
CREATE INDEX idx_users_search_vector ON users USING GIN (search_vector);

-- 4. 検索クエリ
SELECT
  id, name, email,
  ts_rank(search_vector, query) AS relevance
FROM users,
  to_tsquery('simple', 'taro') AS query
WHERE search_vector @@ query
ORDER BY relevance DESC
LIMIT 20;

-- 5. 前方一致（オートコンプリート用）
SELECT id, name
FROM users
WHERE name ILIKE 'tar%'
ORDER BY name
LIMIT 10;

-- 6. pg_trgm による類似検索（typo耐性）
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_users_name_trgm ON users USING GIN (name gin_trgm_ops);

SELECT id, name, similarity(name, 'trao') AS sim
FROM users
WHERE name % 'trao'  -- similarity threshold (default 0.3)
ORDER BY sim DESC
LIMIT 10;
```

---

## 9. ページネーション + フィルタ + ソートの統合実装

ここまで個別に解説した各機能を統合した、本番品質の API エンドポイント実装を示す。

### 9.1 統合コントローラ（Express.js + Prisma）

```javascript
// routes/users.js - 統合的なリスト取得エンドポイント

const express = require('express');
const router = express.Router();

/**
 * GET /api/v1/users
 *
 * クエリパラメータ:
 *   - page / per_page     : Offset方式ページネーション
 *   - cursor / limit      : Cursor方式ページネーション
 *   - filter[field][op]   : フィルタリング
 *   - sort                : ソート（-prefix で降順）
 *   - fields              : フィールド選択
 *   - q                   : 全文検索
 *   - include_total       : 総件数を含めるか
 */
router.get('/users', async (req, res, next) => {
  try {
    const {
      page, per_page, cursor, limit,
      sort, fields, q, include_total,
    } = req.query;

    // --- ページネーション方式の判定 ---
    const useCursor = cursor != null || (page == null && cursor == null);
    // cursor パラメータがある、または何も指定なしの場合は Cursor 方式

    // --- フィルタの解析 ---
    const filters = parseFilters(req.query, USERS_FILTER_SCHEMA);
    const where = filtersToPrismaWhere(filters);

    // --- 全文検索の統合 ---
    if (q) {
      // PostgreSQL 全文検索を WHERE に統合
      where.searchVector = {
        search: q.split(/\s+/).join(' & '),
      };
    }

    // --- ソートの解析 ---
    const orderBy = parseSort(sort, USERS_SORT_FIELDS);

    // --- フィールド選択 ---
    const select = parseFields(fields, USERS_ALLOWED_FIELDS);

    let result;

    if (useCursor) {
      // --- Cursor 方式 ---
      result = await cursorPaginate({
        model: prisma.user,
        where,
        orderBy,
        select,
        cursor,
        limit: limit ? parseInt(limit, 10) : 20,
      });
    } else {
      // --- Offset 方式 ---
      const pageNum = Math.max(parseInt(page, 10) || 1, 1);
      const perPage = Math.min(
        Math.max(parseInt(per_page, 10) || 20, 1),
        100
      );

      result = await offsetPaginate({
        model: prisma.user,
        where,
        orderBy,
        select,
        page: pageNum,
        perPage,
        includeTotal: include_total === 'true',
      });
    }

    // --- レスポンスヘッダの設定 ---
    if (result.meta.total != null) {
      res.set('X-Total-Count', result.meta.total.toString());
    }

    // Link ヘッダ（RFC 8288）
    if (result.links) {
      const linkParts = Object.entries(result.links)
        .filter(([, url]) => url != null)
        .map(([rel, url]) => `<${url}>; rel="${rel}"`);
      if (linkParts.length > 0) {
        res.set('Link', linkParts.join(', '));
      }
    }

    res.json(result);
  } catch (err) {
    next(err);
  }
});

// --- Cursor ページネーション関数 ---
async function cursorPaginate({ model, where, orderBy, select, cursor, limit }) {
  const take = Math.min(Math.max(limit, 1), 100);

  if (cursor) {
    const decoded = decodeCursor(cursor);
    const sortField = Object.keys(orderBy[0])[0];
    const sortDir = Object.values(orderBy[0])[0];
    const cursorWhere = buildCursorWhere(
      [sortField, 'id'],
      decoded,
      [sortDir, sortDir]
    );
    // 既存の where と AND 結合
    if (cursorWhere.OR) {
      where.AND = where.AND || [];
      where.AND.push(cursorWhere);
    }
  }

  const items = await model.findMany({
    where,
    orderBy,
    select,
    take: take + 1,
  });

  const hasNextPage = items.length > take;
  const data = hasNextPage ? items.slice(0, take) : items;

  const sortField = Object.keys(orderBy[0])[0];

  return {
    data,
    meta: {
      hasNextPage,
      nextCursor: hasNextPage
        ? encodeCursor({
            [sortField]: data[data.length - 1][sortField],
            id: data[data.length - 1].id,
          })
        : null,
      hasPrevPage: !!cursor,
      prevCursor: data.length > 0
        ? encodeCursor({
            [sortField]: data[0][sortField],
            id: data[0].id,
          })
        : null,
      limit: take,
    },
  };
}

// --- Offset ページネーション関数 ---
async function offsetPaginate({
  model, where, orderBy, select, page, perPage, includeTotal,
}) {
  const skip = (page - 1) * perPage;

  const [data, total] = await Promise.all([
    model.findMany({
      where,
      orderBy,
      select,
      skip,
      take: perPage,
    }),
    includeTotal ? model.count({ where }) : Promise.resolve(null),
  ]);

  const totalPages = total != null ? Math.ceil(total / perPage) : null;
  const baseUrl = '/api/v1/users'; // 実際にはリクエストから構築

  return {
    data,
    meta: {
      total,
      page,
      perPage,
      totalPages,
      hasNextPage: totalPages != null ? page < totalPages : data.length === perPage,
      hasPrevPage: page > 1,
    },
    links: {
      self:  `${baseUrl}?page=${page}&per_page=${perPage}`,
      first: `${baseUrl}?page=1&per_page=${perPage}`,
      prev:  page > 1 ? `${baseUrl}?page=${page - 1}&per_page=${perPage}` : null,
      next:  (totalPages == null || page < totalPages)
               ? `${baseUrl}?page=${page + 1}&per_page=${perPage}`
               : null,
      last:  totalPages != null
               ? `${baseUrl}?page=${totalPages}&per_page=${perPage}`
               : null,
    },
  };
}

module.exports = router;
```

---

## 10. アンチパターンと対策

### 10.1 アンチパターン 1: limit 無制限の API

```
[問題]
  GET /api/v1/users?limit=999999999

  クライアントが巨大な limit を指定できる場合、
  以下の問題が発生する:

  (a) メモリ枯渇:
      100万件のユーザーオブジェクトをメモリ上に展開
      → JSON シリアライゼーションで更にメモリ使用量が倍増
      → OOM Kill によるプロセスクラッシュ

  (b) レスポンスタイム超過:
      巨大な JSON レスポンスの生成と転送に時間がかかる
      → タイムアウト → リトライ → さらに負荷増大

  (c) DoS 攻撃ベクトル:
      悪意あるクライアントが繰り返し巨大リクエストを送信
      → サーバーリソースの枯渇

[対策]
  // limit のクランプ（必須）
  const MAX_LIMIT = 100;
  const DEFAULT_LIMIT = 20;

  function clampLimit(requestedLimit) {
    if (requestedLimit == null) return DEFAULT_LIMIT;
    const parsed = parseInt(requestedLimit, 10);
    if (isNaN(parsed) || parsed < 1) return DEFAULT_LIMIT;
    return Math.min(parsed, MAX_LIMIT);
  }

  // API ドキュメントに上限を明記:
  // "limit: 1〜100の整数（デフォルト: 20、最大: 100）"

  // レスポンスヘッダで上限を通知:
  // X-Max-Limit: 100
```

### 10.2 アンチパターン 2: COUNT(*) の無条件実行

```
[問題]
  毎回のリストAPIリクエストで COUNT(*) を実行:

  SELECT COUNT(*) FROM users WHERE status = 'active';
  -- 1000万行テーブルの場合、~200ms

  さらに、フィルタ条件が複雑な場合:
  SELECT COUNT(*) FROM users
  WHERE status = 'active'
    AND role IN ('admin', 'editor')
    AND created_at > '2023-01-01';
  -- インデックスが効かないケースでは ~2000ms

  全てのリクエストでこのクエリが走ると、
  DB の CPU 使用率が常に高い状態になる。

[対策]

  (1) 総件数をオプトイン方式にする:
      GET /api/v1/users?include_total=true
      → デフォルトでは total を返さない

  (2) 推定値を返す（PostgreSQL）:
      -- 正確な COUNT の代わりに推定行数を使用
      SELECT reltuples::bigint AS estimate
      FROM pg_class
      WHERE relname = 'users';
      -- 定期的に ANALYZE で更新される

  (3) カウントキャッシュを使う:
      -- Redis にカウントをキャッシュ（TTL 60秒）
      const cacheKey = `count:users:${filterHash}`;
      let total = await redis.get(cacheKey);
      if (total == null) {
        total = await prisma.user.count({ where });
        await redis.set(cacheKey, total, 'EX', 60);
      }

  (4) 「もっと見る」パターン:
      → total を返さず、hasNextPage のみ返す
      → "全 XXX 件" の表示を避け、"もっと見る" ボタンのみ
      → モバイルアプリでは主流のパターン
```

### 10.3 アンチパターン 3: フィルタフィールドのブラックリスト方式

```
[問題]
  // 「禁止フィールド以外は全て許可」という設計
  const blockedFields = ['password', 'secret'];

  function isAllowedFilter(field) {
    return !blockedFields.includes(field);
  }

  // 新しいフィールド（例: internal_notes）が追加されたとき、
  // blocklist の更新を忘れると機密情報がフィルタ可能になる。

[対策]
  // ホワイトリスト方式を使う（許可フィールドのみ明示）
  const ALLOWED_FILTER_FIELDS = [
    'status', 'role', 'name', 'email', 'createdAt'
  ];

  function isAllowedFilter(field) {
    return ALLOWED_FILTER_FIELDS.includes(field);
  }

  // 新しいフィールドは意図的に追加するまでフィルタ不可
  // → デフォルト拒否（Deny by default）の原則
```

---

## 11. エッジケース分析

### 11.1 エッジケース 1: ソートキーの値が重複する場合

```
[状況]
  100人のユーザーが同じ created_at を持つ場合:

  id=1,  created_at='2024-01-15'
  id=2,  created_at='2024-01-15'
  id=3,  created_at='2024-01-15'
  ...
  id=100, created_at='2024-01-15'

  Cursor方式で created_at のみをカーソルに使うと:

  1ページ目: WHERE created_at <= '2024-01-15' LIMIT 20
  → id=1〜20 を取得（created_at で順序が不定）

  2ページ目: WHERE created_at < '2024-01-15' LIMIT 20
  → 0件（全て同じ created_at なので条件に合う行がない）

  結果: 2ページ目以降が取得できない。

[解決策]
  必ずタイブレーカーとして一意キー（id）を複合キーに含める:

  -- 正しい SQL
  WHERE (created_at, id) < ('2024-01-15', 20)
  ORDER BY created_at DESC, id DESC
  LIMIT 20;

  これにより created_at が同じでも id で順序が一意に定まる。

  カーソルデータ:
  { "createdAt": "2024-01-15", "id": 20 }

  [教訓]
  カーソルには常に一意キー（id）を含めること。
  これは「安定カーソル（Stable Cursor）」の基本原則である。
```

### 11.2 エッジケース 2: NULL 値を含むソートキー

```
[状況]
  一部のユーザーの deleted_at が NULL:

  id=1, deleted_at=NULL        （未削除）
  id=2, deleted_at='2024-03-01' （削除済み）
  id=3, deleted_at=NULL        （未削除）
  id=4, deleted_at='2024-01-15' （削除済み）

  deleted_at DESC でソートすると:
  → NULL の位置はDBMSによって異なる
    PostgreSQL: NULL が最初（NULLS FIRST がデフォルト for DESC）
    MySQL: NULL が最後（DESC の場合）

  カーソルに NULL が含まれると比較が正しく動作しない:
  WHERE deleted_at < NULL → 常に FALSE（NULL との比較は UNKNOWN）

[解決策]

  (1) NULL を含むフィールドでのソートを避ける
      → ソート可能フィールドは NOT NULL のものに限定

  (2) NULL を含む場合は COALESCE で置換:
      ORDER BY COALESCE(deleted_at, '9999-12-31') DESC, id DESC

  (3) カーソル内で NULL を特別扱い:
      function buildCursorWhereWithNull(sortField, cursorValue, id) {
        if (cursorValue === null) {
          // NULL の位置（NULLS FIRST / LAST）に応じて条件を変える
          return {
            OR: [
              { [sortField]: { not: null } }, // NULLでない行は全て「後」
              { [sortField]: null, id: { lt: id } }, // 同じNULLならidで比較
            ],
          };
        }
        return {
          OR: [
            { [sortField]: { lt: cursorValue } },
            { [sortField]: cursorValue, id: { lt: id } },
          ],
        };
      }

  [教訓]
  ソートキーに NULL を許容する場合は、NULL の順序を明示的に制御し、
  カーソル比較で NULL を特別扱いする必要がある。
  可能であれば NOT NULL 制約のあるフィールドのみソート対象とする。
```

### 11.3 エッジケース 3: 並行書き込みとカーソルの整合性

```
[状況]
  Time T1: クライアントが1ページ目を取得
           cursor = { createdAt: '2024-06-10', id: 20 }

  Time T2: 管理者が id=15 のユーザーの createdAt を
           '2024-06-10' → '2024-06-11' に更新

  Time T3: クライアントが2ページ目を取得（cursor を使用）
           WHERE (created_at, id) < ('2024-06-10', 20)

  結果:
  - id=15 は createdAt が '2024-06-11' に変わったため、
    1ページ目にも2ページ目にも含まれない（消失）
  - ソートキーが変更可能なフィールドの場合、
    カーソル方式でもデータの消失や重複が起こりうる

[対策]
  (1) ソートキーは不変フィールドを使う:
      → created_at（作成日は変更されない）
      → id（主キーは変更されない）
      → sequence_number（連番は変更されない）

  (2) ソートキーの変更を禁止する:
      → updated_at でソートする場合、ページング中の更新は
        ビジネスルール上許容するかどうかを判断する

  (3) スナップショット方式:
      → ページング開始時のスナップショットIDを発行し、
        全ページ取得が完了するまで同じスナップショットを参照
      → 実装が複雑だが、一貫性は最も高い
```

---

## 12. インデックス戦略

ページネーション・フィルタリング・ソートのパフォーマンスは、
適切なインデックス設計に大きく依存する。

### 12.1 インデックス設計の原則

```
ページネーション関連のインデックス戦略:

  (1) Offset 方式:
      -- ソートキーにインデックスを張る
      CREATE INDEX idx_users_created_at ON users (created_at DESC);

      -- フィルタ + ソートの複合インデックス
      CREATE INDEX idx_users_status_created
        ON users (status, created_at DESC);

  (2) Cursor / Keyset 方式:
      -- ソートキー + ID の複合インデックス（必須）
      CREATE INDEX idx_users_created_id
        ON users (created_at DESC, id DESC);

      -- フィルタ + ソートキー + ID の複合インデックス
      CREATE INDEX idx_users_status_created_id
        ON users (status, created_at DESC, id DESC);

  (3) カバリングインデックス:
      -- SELECT するカラムも含めることでテーブルスキャン不要
      CREATE INDEX idx_users_list_covering
        ON users (status, created_at DESC, id DESC)
        INCLUDE (name, email, role);
      -- PostgreSQL 11+ で INCLUDE が利用可能

  インデックス設計のフローチャート:

  フィルタ条件 → 等価条件のカラムを先頭に
       ↓
  ソート条件 → ソートキーを次に
       ↓
  ページネーション → id を末尾に
       ↓
  SELECT対象 → INCLUDE で追加（カバリング）

  例: status = 'active' AND role = 'admin' ORDER BY created_at DESC

  CREATE INDEX idx_users_optimal
    ON users (status, role, created_at DESC, id DESC)
    INCLUDE (name, email);

  → WHERE status = 'active' AND role = 'admin'
    がインデックスの先頭2列で絞り込み、
    ORDER BY created_at DESC, id DESC
    がインデックスの後続列でカバーされ、
    name, email は INCLUDE でテーブルアクセス不要。
```

### 12.2 EXPLAIN ANALYZE による検証

```sql
-- Offset 方式の実行計画（問題のあるケース）
EXPLAIN ANALYZE
SELECT * FROM users
ORDER BY created_at DESC
LIMIT 20 OFFSET 100000;

-- 結果（例）:
-- Limit  (cost=12345.67..12346.00 rows=20)
--   -> Sort  (cost=12345.67..15000.00 rows=1000000)
--         Sort Key: created_at DESC
--         Sort Method: top-N heapsort  Memory: 30kB
--         -> Seq Scan on users  (cost=0.00..10000.00 rows=1000000)
-- Planning Time: 0.5ms
-- Execution Time: 580ms  ← 遅い

-- Cursor 方式の実行計画（改善後）
EXPLAIN ANALYZE
SELECT * FROM users
WHERE (created_at, id) < ('2024-01-15', 100)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- 結果（例）:
-- Limit  (cost=0.56..1.80 rows=20)
--   -> Index Scan using idx_users_created_id on users
--         (cost=0.56..5000.00 rows=100000)
--         Index Cond: (created_at, id) < ('2024-01-15', 100)
-- Planning Time: 0.3ms
-- Execution Time: 1.2ms  ← 高速
```
