# REST vs GraphQL

> RESTとGraphQLの本質的な違い、それぞれの強みと弱み、選定基準を体系的に比較。プロジェクトの要件に応じた適切な選択と、ハイブリッドアプローチまで、実践的な判断基準を提供する。

## この章で学ぶこと

- [ ] RESTとGraphQLの技術的な違いを理解する
- [ ] プロジェクト要件に基づく選定基準を把握する
- [ ] ハイブリッドアプローチの設計を学ぶ

---

## 1. 基本比較

```
                REST                    GraphQL
─────────────────────────────────────────────────
エンドポイント  複数                    単一（/graphql）
データ取得      サーバーが決定          クライアントが決定
型システム      なし（OpenAPIで補完）    組み込み（SDL）
キャッシュ      HTTPキャッシュ活用      独自のキャッシュが必要
学習コスト      低い                    中程度
エコシステム    非常に成熟              成長中
ファイルアップ  容易（multipart）       複雑（別途対応）
リアルタイム    WebSocket/SSE別途       Subscription組み込み
エラー          HTTPステータスコード    常に200 + errors配列
ツール          curl等で即テスト        専用クライアント必要
```

---

## 2. データ取得の違い

```
ユーザーとその注文一覧を取得する場合:

REST:
  # リクエスト1: ユーザー取得
  GET /api/v1/users/123
  → { "id": "123", "name": "Taro", "email": "...", "avatar": "...",
      "address": "...", "phone": "...", ... }
      # 不要なフィールドも含む（Over-fetching）

  # リクエスト2: 注文取得
  GET /api/v1/users/123/orders
  → [{ "id": "1", "total": 5000, ... }, ...]
      # 別リクエストが必要（Under-fetching）

  問題:
  ✗ Over-fetching: 不要なデータ（avatar, address, phone）も取得
  ✗ Under-fetching: 関連データに追加リクエストが必要
  ✗ 3つの関連リソース = 3リクエスト

GraphQL:
  # リクエスト1回で必要なデータのみ
  POST /graphql
  query {
    user(id: "123") {
      name
      email
      orders(first: 5) {
        edges {
          node {
            id
            total
            items { productName, price }
          }
        }
      }
    }
  }

  利点:
  ✓ 必要なフィールドのみ（Over-fetching解消）
  ✓ 1リクエストで関連データも取得（Under-fetching解消）
  ✓ モバイルで通信量削減

RESTでの対策:
  ① フィールド選択: GET /users/123?fields=name,email
  ② インクルード: GET /users/123?include=orders
  ③ 専用エンドポイント: GET /users/123/summary
  → いずれも「GraphQLの部分的な再発明」
```

---

## 3. 選定基準

```
RESTを選ぶべき場合:
  ✓ シンプルなCRUD操作が中心
  ✓ HTTPキャッシュを最大限活用したい
  ✓ ファイルアップロード/ダウンロードが多い
  ✓ サードパーティ向けの公開API
  ✓ チームにGraphQL経験者がいない
  ✓ マイクロサービス間の通信（gRPCも検討）
  ✓ 低レイテンシが最優先（CDNキャッシュ）

GraphQLを選ぶべき場合:
  ✓ 複雑なデータ関係がある（ソーシャルグラフ等）
  ✓ 多様なクライアント（Web、iOS、Android）
  ✓ フロントエンドの柔軟性が重要
  ✓ 1画面で多くの関連データを表示
  ✓ リアルタイム更新が必要
  ✓ マイクロフロントエンドでBFF（Backend for Frontend）
  ✓ スキーマ駆動開発を行いたい

gRPCを選ぶべき場合:
  ✓ マイクロサービス間の高速通信
  ✓ ストリーミング通信
  ✓ 多言語環境でのサービス間通信
  ✓ パフォーマンスが最優先

判断フローチャート:

  公開API？ → YES → REST
    ↓ NO
  マイクロサービス間？ → YES → gRPC
    ↓ NO
  データ関係が複雑？ → YES → GraphQL
    ↓ NO
  多様なクライアント？ → YES → GraphQL
    ↓ NO
  REST
```

---

## 4. パフォーマンス比較

```
レイテンシ:
  REST:
  → CDNキャッシュ: 数ms（最速）
  → サーバー応答: 普通
  → 複数リクエスト: N回のラウンドトリップ

  GraphQL:
  → CDNキャッシュ: 困難（Persisted Queriesで可能）
  → サーバー応答: クエリ解析のオーバーヘッド
  → 1リクエスト: 1回のラウンドトリップ

ペイロードサイズ:
  REST:
  → Over-fetchingで大きくなりがち
  → 圧縮で軽減可能

  GraphQL:
  → 必要最小限のデータのみ
  → モバイルで特に有利

サーバー負荷:
  REST:
  → 予測しやすい（エンドポイントごとに最適化）
  → スケーリングが容易

  GraphQL:
  → クエリの複雑度が予測困難
  → 悪意のあるクエリへの対策が必要
  → N+1問題（DataLoaderで解決）

  結論:
  → 単純なケース: RESTの方が高速（キャッシュ活用）
  → 複雑なケース: GraphQLの方が効率的（リクエスト削減）
```

---

## 5. 開発体験の比較

```
API設計:
  REST:
  → エンドポイント設計に時間がかかる
  → OpenAPIで仕様書を別途作成
  → バージョニングの管理が必要

  GraphQL:
  → スキーマ定義 = 仕様書
  → 型安全なコード生成
  → スキーマの進化が容易（非破壊的追加）

フロントエンド開発:
  REST:
  → 型定義を手動で作成（OpenAPI生成で改善）
  → 複数エンドポイントの呼び出しを管理
  → データの結合はクライアント側で

  GraphQL:
  → codegen で型を自動生成
  → 1クエリで必要なデータを取得
  → Apollo Client のキャッシュが便利

テスト:
  REST:
  → curl で手軽にテスト
  → Postman, Insomnia が豊富
  → 各エンドポイントを個別にテスト

  GraphQL:
  → GraphiQL, Apollo Studio でインタラクティブテスト
  → スキーマ変更の影響分析が容易
  → クエリ単位のテスト

ドキュメント:
  REST:
  → Swagger UI / Redoc（OpenAPIから生成）
  → 追加の説明が必要

  GraphQL:
  → スキーマ自体がドキュメント
  → GraphiQL でインタラクティブに探索
  → 型情報が自動的にドキュメント化
```

---

## 6. ハイブリッドアプローチ

```
実務では REST と GraphQL を共存させるパターンが有効:

パターン1: REST + GraphQL BFF
  クライアント → GraphQL BFF → REST マイクロサービス群

  → フロントエンドは GraphQL の柔軟性を享受
  → バックエンドは REST の安定性を維持
  → BFF がデータの集約・変換を担当

パターン2: メインは REST、特定機能は GraphQL
  → CRUD操作: REST
  → ダッシュボード（複雑なデータ集約）: GraphQL
  → ファイル操作: REST
  → リアルタイム通知: GraphQL Subscription

パターン3: 公開API は REST、内部は GraphQL
  → サードパーティ向け: REST（標準的、キャッシュ可能）
  → 自社フロントエンド: GraphQL（柔軟、型安全）

パターン4: GraphQL Gateway
  GraphQL Gateway → REST Service A
                  → gRPC Service B
                  → GraphQL Service C

  → Apollo Federation / GraphQL Mesh
  → 統一的なGraphQLインターフェース
  → 各サービスは最適な技術を選択
```

```javascript
// GraphQL Gateway（Apollo Federation）の例

// ユーザーサービスのスキーマ
const userSchema = gql`
  type User @key(fields: "id") {
    id: ID!
    name: String!
    email: String!
  }

  type Query {
    user(id: ID!): User
  }
`;

// 注文サービスのスキーマ
const orderSchema = gql`
  type Order @key(fields: "id") {
    id: ID!
    total: Int!
    user: User!
  }

  extend type User @key(fields: "id") {
    id: ID! @external
    orders: [Order!]!
  }
`;

// Gateway が自動的に統合
// query {
//   user(id: "1") {
//     name          ← ユーザーサービスから
//     orders {      ← 注文サービスから
//       total
//     }
//   }
// }
```

---

## 7. 移行戦略

```
REST → GraphQL への段階的移行:

  Phase 1: GraphQL Layer の追加
  → 既存のREST APIの上にGraphQLレイヤーを構築
  → GraphQLのリゾルバーが内部でREST APIを呼ぶ
  → クライアントは徐々にGraphQLに移行

  Phase 2: 新機能はGraphQLで開発
  → 新しい画面/機能からGraphQLを使用
  → 既存画面は引き続きRESTを使用

  Phase 3: 段階的なREST廃止
  → 使用量の少ないRESTエンドポイントから廃止
  → GraphQLリゾルバーを直接DBアクセスに変更

注意点:
  → 一度に全て移行しない
  → チームの学習期間を確保
  → モニタリングで性能を比較
  → ロールバック可能な状態を維持
```

---

## まとめ

| 観点 | REST | GraphQL | gRPC |
|------|------|---------|------|
| 適用場面 | CRUD、公開API | 複雑なデータ、多クライアント | サービス間通信 |
| キャッシュ | HTTP標準 | 独自実装 | なし |
| 型安全 | OpenAPIで補完 | 組み込み | Protocol Buffers |
| 学習コスト | 低 | 中 | 中〜高 |
| エコシステム | 最も成熟 | 成長中 | バックエンド中心 |

---

## 次に読むべきガイド
→ [[00-sdk-design.md]] — SDK設計

---

## 参考文献
1. Buna, S. "GraphQL in Action." Manning, 2021.
2. Sturgeon, P. "Build APIs You Won't Hate." Leanpub, 2023.
3. Netflix. "Beyond REST." netflixtechblog.com, 2023.
