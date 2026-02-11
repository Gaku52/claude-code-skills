# ページネーションとフィルタリング

> 大量データを効率的に返すためのページネーション、フィルタリング、ソート、検索の設計パターン。Offset/Cursor方式の比較、フィルタ構文、全文検索まで、データ取得APIの全技法を網羅する。

## この章で学ぶこと

- [ ] Offset方式とCursor方式の違いと選定基準を理解する
- [ ] フィルタリングとソートのAPI設計を把握する
- [ ] 全文検索・ファセット検索の設計を学ぶ

---

## 1. ページネーション方式

```
① Offset方式（ページ番号ベース）:

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
    }
  }

  内部SQL:
  SELECT * FROM users
  ORDER BY created_at DESC
  LIMIT 20 OFFSET 40;  -- (page - 1) * per_page

  利点:
  ✓ 直感的（「3ページ目」が明確）
  ✓ 任意のページにジャンプ可能
  ✓ UIにページ番号を表示しやすい

  欠点:
  ✗ OFFSET が大きいとパフォーマンス劣化
    → OFFSET 100000 は10万行スキップ（O(n)）
  ✗ データの追加/削除で位置ずれ
    → ページ2を見ている間にデータが挿入されると重複表示
  ✗ totalのCOUNTクエリが重い（大規模テーブル）

② Cursor方式（キーセットベース）:

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
  ✓ 一定のパフォーマンス（WHERE句でインデックス利用）
  ✓ データの追加/削除で位置ずれしない
  ✓ リアルタイムフィードに最適

  欠点:
  ✗ 任意のページにジャンプ不可
  ✗ ページ番号の表示が困難
  ✗ cursor の生成・解析が複雑

③ 方式の選定基準:

  Offset方式を選ぶ場合:
  → 管理画面（ページジャンプが必要）
  → 検索結果（total件数の表示が必要）
  → データ量が少ない（< 10万件）

  Cursor方式を選ぶ場合:
  → SNSフィード（無限スクロール）
  → リアルタイムデータ（チャット、通知）
  → データ量が多い（> 100万件）
  → モバイルアプリ（次へ/前へのみ）
```

---

## 2. Cursor実装の詳細

```javascript
// Node.js + Prisma でのCursor実装

// --- Cursor エンコード/デコード ---
function encodeCursor(data) {
  return Buffer.from(JSON.stringify(data)).toString('base64url');
}

function decodeCursor(cursor) {
  return JSON.parse(Buffer.from(cursor, 'base64url').toString());
}

// --- Cursorページネーション ---
async function listUsers(params) {
  const { cursor, limit = 20, sort = 'createdAt', order = 'desc' } = params;
  const take = Math.min(limit, 100); // 上限制限

  let where = {};
  if (cursor) {
    const decoded = decodeCursor(cursor);
    // 複合カーソル: ソートキー + ID
    where = {
      OR: [
        { [sort]: order === 'desc' ? { lt: decoded[sort] } : { gt: decoded[sort] } },
        {
          [sort]: decoded[sort],
          id: order === 'desc' ? { lt: decoded.id } : { gt: decoded.id },
        },
      ],
    };
  }

  // 1件多く取得して hasNextPage を判定
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
    },
  };
}
```

---

## 3. フィルタリング設計

```
フィルタリングのAPI設計パターン:

① シンプルなクエリパラメータ（推奨）:
  GET /api/v1/users?status=active&role=admin&age_min=18&age_max=65

  → 単純なフィルタに最適
  → フィールド名がそのままパラメータ名

② フィルタ演算子パターン:
  GET /api/v1/users?filter[status]=active
  GET /api/v1/users?filter[age][gte]=18&filter[age][lte]=65
  GET /api/v1/users?filter[name][contains]=taro

  演算子:
  eq     等しい          filter[status][eq]=active
  ne     等しくない      filter[status][ne]=deleted
  gt     より大きい      filter[age][gt]=18
  gte    以上            filter[age][gte]=18
  lt     より小さい      filter[age][lt]=65
  lte    以下            filter[age][lte]=65
  in     含まれる        filter[role][in]=admin,editor
  nin    含まれない      filter[role][nin]=guest
  contains 部分一致      filter[name][contains]=taro
  starts   前方一致      filter[name][starts]=ta

③ JSON API仕様:
  GET /api/v1/users?filter[status]=active&filter[role]=admin

④ RHS Colon:
  GET /api/v1/users?status=eq:active&age=gte:18&age=lte:65

⑤ LHS Brackets:
  GET /api/v1/users?status[eq]=active&age[gte]=18

推奨:
  → 小規模API: ①シンプルパターン
  → 中規模API: ②フィルタ演算子
  → 複雑な検索: 専用の検索エンドポイント（POST /search）
```

```javascript
// フィルタパーサーの実装例
function parseFilters(query) {
  const filters = {};

  for (const [key, value] of Object.entries(query)) {
    // filter[field][operator] パターン
    const match = key.match(/^filter\[(\w+)\](?:\[(\w+)\])?$/);
    if (!match) continue;

    const field = match[1];
    const operator = match[2] || 'eq';

    // ホワイトリスト検証（SQLインジェクション防止）
    const allowedFields = ['status', 'role', 'age', 'name', 'createdAt'];
    const allowedOperators = ['eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in', 'contains'];

    if (!allowedFields.includes(field)) continue;
    if (!allowedOperators.includes(operator)) continue;

    if (!filters[field]) filters[field] = {};

    if (operator === 'in') {
      filters[field][operator] = value.split(',');
    } else {
      filters[field][operator] = value;
    }
  }

  return filters;
}

// Prisma WHERE句への変換
function filtersToPrismaWhere(filters) {
  const where = {};
  const operatorMap = {
    eq: 'equals', ne: 'not', gt: 'gt', gte: 'gte',
    lt: 'lt', lte: 'lte', in: 'in', contains: 'contains',
  };

  for (const [field, ops] of Object.entries(filters)) {
    where[field] = {};
    for (const [op, value] of Object.entries(ops)) {
      where[field][operatorMap[op]] = value;
    }
  }

  return where;
}
```

---

## 4. ソート設計

```
ソートのAPI設計:

① シンプルなパラメータ:
  GET /api/v1/users?sort=created_at&order=desc
  GET /api/v1/users?sort=-created_at        ← -プレフィックスで降順

② 複数フィールドソート:
  GET /api/v1/users?sort=-created_at,name
  → created_at降順 → name昇順

③ JSON API仕様:
  GET /api/v1/users?sort=-created_at,name

ソートの注意点:
  ✓ ソート可能なフィールドをホワイトリストで制限
  ✓ デフォルトソートを必ず定義（例: -created_at）
  ✓ ソートフィールドにインデックスを張る
  ✓ Cursor方式ではソートキーをcursorに含める
  ✗ ユーザー入力をそのままORDER BYに渡さない
```

```javascript
// ソートパーサー
function parseSort(sortParam, allowedFields) {
  if (!sortParam) return [{ createdAt: 'desc' }]; // デフォルト

  return sortParam.split(',').map(field => {
    const desc = field.startsWith('-');
    const name = desc ? field.slice(1) : field;

    // ホワイトリスト検証
    if (!allowedFields.includes(name)) {
      throw new Error(`Invalid sort field: ${name}`);
    }

    return { [name]: desc ? 'desc' : 'asc' };
  });
}

// 使用例
const orderBy = parseSort(
  req.query.sort,
  ['createdAt', 'name', 'email', 'updatedAt']
);
```

---

## 5. フィールド選択（Sparse Fieldsets）

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
  ✓ レスポンスサイズの削減
  ✓ ネットワーク帯域の節約
  ✓ モバイルアプリで特に有効
  ✓ DBクエリのSELECT最適化

注意:
  → id は常に含める（クライアントの参照整合性のため）
  → セキュリティ上返してはいけないフィールドのチェック
  → GraphQLの方が本質的にこの機能を持つ
```

---

## 6. 検索設計

```
検索のAPI設計:

① シンプル検索（全文検索）:
  GET /api/v1/users?q=taro
  → name, email 等の複数フィールドを横断検索

② 詳細検索（フィルタ + 検索の組み合わせ）:
  GET /api/v1/users?q=taro&filter[role]=admin&sort=-relevance

③ 専用検索エンドポイント:
  POST /api/v1/search
  {
    "query": "taro",
    "filters": {
      "role": ["admin", "editor"],
      "createdAt": { "gte": "2024-01-01" }
    },
    "sort": ["-_score", "name"],
    "page": { "limit": 20, "offset": 0 },
    "facets": ["role", "department"]
  }

  レスポンス:
  {
    "data": [...],
    "meta": {
      "total": 42,
      "maxScore": 15.3,
      "took": 12  // ms
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

検索バックエンド:
  PostgreSQL:  全文検索（tsvector / tsquery）
  Elasticsearch: 高度な全文検索、ファセット、集計
  Algolia:     ホスティッド検索サービス
  Meilisearch: 軽量な全文検索エンジン
  Typesense:   型付き全文検索
```

---

## 7. ベストプラクティス

```
設計原則:
  ✓ デフォルト値を必ず設定（limit=20, sort=-createdAt）
  ✓ 上限を設定（limit最大100）
  ✓ total件数は任意取得可能に（include_total=true）
  ✓ 空のコレクションは200 + 空配列（404ではない）
  ✓ フィルタ可能フィールドはドキュメントに明記

パフォーマンス:
  ✓ ソート/フィルタフィールドにインデックスを張る
  ✓ COUNT(*)の代わりに推定値を使う（大規模テーブル）
  ✓ 複合インデックス: (ソートキー, フィルタキー)
  ✓ カバリングインデックスの活用
  ✗ SELECT * を避け、必要なカラムのみ取得

セキュリティ:
  ✓ フィルタ/ソートフィールドのホワイトリスト
  ✓ limit の上限チェック（DoS防止）
  ✓ 検索クエリのサニタイゼーション
  ✗ ユーザー入力をSQLに直接埋め込まない
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Offset方式 | 直感的だが大規模でパフォーマンス劣化 |
| Cursor方式 | 一定性能だがページジャンプ不可 |
| フィルタリング | ホワイトリスト＋演算子パターン |
| ソート | -プレフィックスで降順、インデックス必須 |
| 検索 | 簡易はGET ?q=、複雑はPOST /search |

---

## 次に読むべきガイド
→ [[00-rest-best-practices.md]] — RESTベストプラクティス

---

## 参考文献
1. Stripe. "Pagination." stripe.com/docs, 2024.
2. JSON:API. "Fetching Data." jsonapi.org, 2024.
3. Slack. "Pagination." api.slack.com, 2024.
