---
name: documentation
description: 技術ドキュメント、README、API仕様書、アーキテクチャ図の作成方法。読みやすく、保守しやすく、チームに役立つドキュメント作成のベストプラクティス。
---

# Documentation Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [ドキュメント種別](#ドキュメント種別)
4. [作成ガイドライン](#作成ガイドライン)
5. [実践例](#実践例)
6. [アンチパターン](#アンチパターン)
7. [Agent連携](#agent連携)

---

## 概要

このSkillは、以下のドキュメント作成をサポートします：

- **README.md** - プロジェクト概要、セットアップ手順
- **API仕様書** - エンドポイント、リクエスト/レスポンス
- **アーキテクチャドキュメント** - 設計思想、技術選定理由
- **コードコメント** - 複雑なロジックの説明
- **変更履歴** - CHANGELOG.md

### 原則

1. **読者視点で書く** - 初めて見る人が理解できるか
2. **鮮度を保つ** - コードと同期させる
3. **必要最小限** - 過剰なドキュメントは逆効果
4. **実例を示す** - コード例、スクリーンショット

## 📚 公式ドキュメント・参考リソース

**このガイドで学べること**: READMEの構造、API仕様書の書き方、ADRの作成方法、コメント規約
**公式で確認すべきこと**: ドキュメント生成ツールの最新機能、Markdown仕様、ドキュメント標準

### 主要な公式ドキュメント

- **[Markdown Guide](https://www.markdownguide.org/)** - Markdown記法の包括的ガイド
  - [Basic Syntax](https://www.markdownguide.org/basic-syntax/)
  - [Extended Syntax](https://www.markdownguide.org/extended-syntax/)

- **[OpenAPI Specification](https://swagger.io/specification/)** - API仕様書の標準フォーマット
  - [Swagger Editor](https://editor.swagger.io/)

- **[JSDoc Documentation](https://jsdoc.app/)** - JavaScriptドキュメントコメント
  - [Getting Started](https://jsdoc.app/about-getting-started.html)

- **[ADR (Architecture Decision Records)](https://adr.github.io/)** - アーキテクチャ決定記録

### 関連リソース

- **[Write the Docs](https://www.writethedocs.org/)** - ドキュメント作成者のコミュニティ
- **[The Documentation System](https://documentation.divio.com/)** - ドキュメント体系化フレームワーク
- **[readme.so](https://readme.so/)** - README生成ツール

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規プロジェクト作成時（README）
- [ ] 公開API追加時（API仕様書）
- [ ] 複雑な設計決定時（ADR: Architecture Decision Record）
- [ ] オンボーディング資料作成時

### 🔄 定期的に

- [ ] リリース前（CHANGELOG更新）
- [ ] 大きなリファクタリング後（アーキテクチャ図更新）
- [ ] 四半期ごと（ドキュメント棚卸し）

---

## ドキュメント種別

### 1. README.md

#### 必須セクション

```markdown
# プロジェクト名

## 概要
何を解決するプロジェクトか（1-2文）

## 特徴
- 主要機能1
- 主要機能2
- 主要機能3

## セットアップ

### 必要環境
- Node.js 18+
- PostgreSQL 14+

### インストール
\`\`\`bash
npm install
cp .env.example .env
npm run db:migrate
\`\`\`

### 起動
\`\`\`bash
npm run dev
\`\`\`

## 使い方
基本的な使用例

## 開発
貢献方法、テストの実行方法

## ライセンス
MIT
```

### 2. API仕様書

OpenAPI (Swagger) 形式推奨：

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
paths:
  /users:
    get:
      summary: ユーザー一覧取得
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/User'
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:
          type: string
```

### 3. アーキテクチャ決定記録 (ADR)

```markdown
# ADR-001: Next.jsでApp Routerを採用

## ステータス
採用

## コンテキスト
新規Webアプリケーション開発にあたり、ルーティング方式を選択する必要がある。

## 決定
Next.js App Router（React Server Components）を採用する。

## 理由
1. サーバーサイドレンダリングが標準で高速
2. レイアウト共有が容易
3. データフェッチングがシンプル

## 代替案
- Pages Router: 安定しているが、新機能が少ない
- Remix: 優れているが、エコシステムが小さい

## 影響
- チームメンバーはRSCの学習が必要
- 一部のライブラリは'use client'が必要
```

### 4. CHANGELOG.md

```markdown
# Changelog

## [Unreleased]

## [1.2.0] - 2025-01-15
### Added
- ダークモード対応
- エクスポート機能

### Changed
- ユーザー設定画面のUI改善

### Fixed
- ログアウト時のメモリリーク修正

## [1.1.0] - 2025-01-01
...
```

---

## 作成ガイドライン

### ✅ 良いドキュメント

```markdown
## 環境変数設定

以下の環境変数を `.env` に設定してください：

\`\`\`bash
# データベース接続（必須）
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# API認証（本番環境のみ必須）
API_KEY=your_api_key_here
\`\`\`

**開発環境の例：**
\`\`\`bash
DATABASE_URL=postgresql://localhost:5432/mydb_dev
\`\`\`
```

**良い点：**
- 実例がある
- 必須/オプションが明確
- 開発環境の例も提示

### ❌ 悪いドキュメント

```markdown
## Setup

Set environment variables.
```

**悪い点：**
- 何を設定するか不明
- 例がない
- 必須かどうか不明

---

## 実践例

### Example 1: 複雑な関数のドキュメント

```typescript
/**
 * ユーザーの権限を階層的にチェックする
 *
 * @param user - チェック対象のユーザー
 * @param resource - アクセス対象のリソース
 * @param action - 実行したいアクション（'read' | 'write' | 'delete'）
 * @returns 権限がある場合true
 *
 * @example
 * ```ts
 * const canEdit = checkPermission(user, 'posts/123', 'write');
 * if (!canEdit) throw new ForbiddenError();
 * ```
 *
 * 注意: 管理者は常にtrueを返します
 */
function checkPermission(
  user: User,
  resource: string,
  action: Action
): boolean {
  // ...
}
```

### Example 2: プロジェクトREADME（MCP Server）

```markdown
# Weather MCP Server

Claude Desktop用の天気情報MCPサーバー

## 機能

- 現在の天気取得
- 5日間の天気予報
- 都市名検索

## インストール

\`\`\`bash
npm install
npm run build
\`\`\`

## Claude Desktopへの設定

\`claude_desktop_config.json\` に追加：

\`\`\`json
{
  "mcpServers": {
    "weather": {
      "command": "node",
      "args": ["/path/to/weather-mcp/build/index.js"],
      "env": {
        "WEATHER_API_KEY": "your_key"
      }
    }
  }
}
\`\`\`

## 使い方

Claude Desktopで：
\`\`\`
東京の今日の天気を教えて
\`\`\`
```

---

## アンチパターン

### ❌ 1. コードの繰り返し

```typescript
// ❌ 悪い例
// この関数はユーザーIDを受け取って、ユーザー情報を返す
function getUser(id: string): User {
  // データベースからユーザーを取得する
  return db.users.findOne(id);
}
```

→ **コードを見れば分かることは書かない**

### ❌ 2. 時代遅れのドキュメント

```markdown
## セットアップ

Node.js 12以上が必要です
```

→ **実際はNode.js 18が必要だった（更新忘れ）**

### ❌ 3. 過剰なドキュメント

小さなユーティリティ関数に50行のドキュメントを書く

→ **コードが自明なら、簡潔に**

---

## Agent連携

### 📖 Agentへの指示例

**新規プロジェクトのREADME作成**
```
新規Next.jsプロジェクトのREADMEを作成してください。
以下を含めてください：
- プロジェクト概要
- セットアップ手順
- 開発サーバー起動方法
- ビルド方法
```

**API仕様書の生成**
```
`/api/users` エンドポイントのOpenAPI仕様書を作成してください。
GET, POST, PUT, DELETEをサポートします。
```

**CHANGELOG更新**
```
今回のリリースで追加した機能をCHANGELOG.mdに追記してください：
- ダークモード対応
- エクスポート機能
- バグ修正3件
```

### 🤖 Agentからの提案例

```
プロジェクトにREADME.mdがありません。
基本的なREADMEを作成しますか？

含める内容：
- プロジェクト名と概要
- インストール手順
- 起動方法
- 開発方法
```

---

## まとめ

### ドキュメント作成のベストプラクティス

1. **読者視点** - 初めて見る人が理解できるか
2. **実例重視** - コード例、スクリーンショット
3. **鮮度維持** - コード変更時にドキュメントも更新
4. **必要最小限** - 過剰なドキュメントは逆効果

### 次のステップ

- [ ] プロジェクトREADME作成
- [ ] API仕様書作成（公開APIがある場合）
- [ ] アーキテクチャ決定記録（重要な設計決定時）
- [ ] CHANGELOG.md（リリース管理時）

---

_Last updated: 2025-12-24_
