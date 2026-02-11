# Technical Writing完全ガイド

> **対象読者**: 全ての開発者、テクニカルライター
> **難易度**: 初級〜中級
> **推定読了時間**: 45分

---

## 📋 目次

1. [Technical Writingとは](#technical-writingとは)
2. [なぜ重要か](#なぜ重要か)
3. [基本原則](#基本原則)
4. [文章構成](#文章構成)
5. [スタイルガイド](#スタイルガイド)
6. [コード例の書き方](#コード例の書き方)
7. [図・画像の活用](#図画像の活用)
8. [レビューと改善](#レビューと改善)
9. [ツールとテンプレート](#ツールとテンプレート)
10. [実践例](#実践例)
11. [よくある失敗](#よくある失敗)
12. [FAQ](#faq)

---

## Technical Writingとは

### 定義

**Technical Writing（技術文書）** は、技術的な情報を分かりやすく、正確に、効率的に伝えるための文書作成技術です。

### 技術文書の種類

```
📚 ドキュメント
├── README.md（プロジェクト概要）
├── API仕様書（エンドポイント定義）
├── チュートリアル（使い方ガイド）
├── リファレンス（関数・クラス詳細）
├── アーキテクチャドキュメント（設計思想）
├── CHANGELOG.md（変更履歴）
└── コードコメント（実装の説明）
```

---

## なぜ重要か

### 1. チーム生産性の向上

```
良いドキュメント → 質問が減る → 生産性UP

例:
- READMEがない → セットアップで2時間迷う
- READMEがある → 10分でセットアップ完了
```

**ROI（投資対効果）**:
- ドキュメント作成: 1時間
- 節約される時間: 10人 × 2時間 = 20時間
- **効果: 20倍のリターン**

### 2. オンボーディング時間の短縮

```
新メンバー加入時:

ドキュメントなし:
- 1週間かけて質問しながら理解
- ベテランメンバーの時間を大量消費

ドキュメントあり:
- 1日でプロジェクト理解
- 自律的に作業開始可能
```

### 3. 知識の属人化防止

```
❌ ドキュメントなし
→ 特定の人しか分からない
→ その人が休むと進まない

✅ ドキュメントあり
→ 誰でも理解可能
→ チーム全体で知識共有
```

### 4. 意思決定の記録

```
なぜこの技術を選んだのか？
なぜこの設計にしたのか？

→ ADR（Architecture Decision Record）で記録
→ 将来の変更時に参照できる
```

---

## 基本原則

### 1. 読者視点で書く

**NG: 書き手の視点**
```markdown
この機能は非同期処理で実装されており、
Promiseを返すのでawaitで待つ必要がある。
```

**OK: 読者の視点**
```markdown
## ユーザー情報の取得

この関数は非同期処理のため、`await`を使って呼び出してください：

\`\`\`typescript
const user = await getUser('user-123');
console.log(user.name);
\`\`\`
```

**ポイント**:
- 読者が何をしたいかを考える
- 具体的な使用例を示す
- 「なぜ」そうするのかを説明

### 2. シンプルで明確に

**NG: 複雑な表現**
```markdown
本システムにおいては、データベースへのアクセスに際して、
ORM（Object-Relational Mapping）ツールであるPrismaを
採用することにより、型安全性を確保しつつ、
生産性の向上を実現している。
```

**OK: シンプルな表現**
```markdown
## データベースアクセス

PrismaというORMツールを使っています。

メリット:
- 型安全（TypeScript対応）
- 開発が速い
- SQLを直接書く必要がない
```

**ポイント**:
- 1文は短く（30文字以内推奨）
- 専門用語は説明する
- 箇条書きを活用

### 3. 具体例を示す

**NG: 抽象的な説明**
```markdown
環境変数を設定してください。
```

**OK: 具体的な例**
```markdown
## 環境変数の設定

`.env`ファイルを作成し、以下を設定してください：

\`\`\`bash
# データベース接続
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# API認証キー
API_KEY=your_api_key_here
\`\`\`

**開発環境の例:**
\`\`\`bash
DATABASE_URL=postgresql://localhost:5432/mydb_dev
API_KEY=dev_key_12345
\`\`\`
```

**ポイント**:
- コード例を必ず付ける
- 開発・本番の両方を示す
- 実際に動くコードを提示

### 4. 最新の状態を保つ

```
ドキュメントは腐る

コード変更 → ドキュメント更新忘れ
→ ドキュメントが信用されなくなる
→ 誰も読まなくなる
```

**対策**:
```markdown
✅ コード変更時にドキュメントも同時更新
✅ PRレビューでドキュメント更新を確認
✅ 四半期ごとにドキュメント棚卸し
✅ 最終更新日を記載
```

---

## 文章構成

### 逆ピラミッド構造

```
最も重要な情報（結論）
  ↓
重要な情報（詳細）
  ↓
補足情報（参考）
```

**例:**

```markdown
## ユーザー認証

**結論:** JWTトークンを使った認証を実装しています。

**詳細:** ログイン時にJWTトークンを発行し、以降のリクエストで
Authorizationヘッダーに含めて送信します。

**補足:** トークンの有効期限は24時間です。期限切れの場合は
401エラーが返ります。
```

**なぜこの構造か**:
- 読者は忙しい（全部読まない）
- 最初の数行で判断する
- 必要なら詳細を読む

### 見出しの階層

```markdown
# H1: ドキュメントタイトル（1つだけ）

## H2: 主要セクション

### H3: サブセクション

#### H4: 詳細項目

##### H5: さらに細かい項目（あまり使わない）
```

**ルール**:
- H1は1つだけ
- 階層をスキップしない（H2の次にH4はNG）
- 3階層まで（H3まで）が理想

### セクションの順序

**基本テンプレート:**

```markdown
# プロジェクト名

## 概要（何のプロジェクトか）

## クイックスタート（すぐ試したい人向け）

## インストール（詳細手順）

## 使い方（基本的な使い方）

## 設定（カスタマイズ方法）

## API リファレンス（詳細仕様）

## トラブルシューティング（よくある問題）

## 貢献方法（コントリビューション）

## ライセンス
```

---

## スタイルガイド

### 1. 文体

#### 敬体 vs 常体

**敬体（です・ます調）:**
```markdown
このコマンドを実行してください。
インストールが完了します。
```

**常体（だ・である調）:**
```markdown
このコマンドを実行する。
インストールが完了する。
```

**推奨:**
- **敬体** - チュートリアル、README（読者に優しい）
- **常体** - リファレンス、仕様書（簡潔）

#### 一貫性を保つ

```markdown
❌ NG: 混在
インストールしてください。
次にサーバーを起動する。

✅ OK: 統一
インストールしてください。
次にサーバーを起動してください。
```

### 2. 用語の統一

**用語集を作る:**

| 使う | 使わない | 理由 |
|------|---------|------|
| ユーザー | ユーザ | JIS規格に準拠 |
| データベース | DB, database | 統一 |
| ログイン | ログオン | 一般的 |
| クリック | 押下 | 分かりやすい |

### 3. 箇条書きのルール

**順序なしリスト（`-` or `*`）:**
```markdown
✅ 良い例:
- 項目は短く
- 1行1項目
- 並列な内容

❌ 悪い例:
- 長い説明が続いて、項目が見づらくなる。複数の文がある。
- 短い項目
```

**順序ありリスト（`1.`）:**
```markdown
✅ 手順を示す時:
1. リポジトリをクローン
2. 依存関係をインストール
3. 開発サーバーを起動

❌ 順序が重要でない時:
1. TypeScript対応（順序関係ない）
2. 高速（順序関係ない）
```

### 4. 強調の使い分け

```markdown
**太字（重要な用語・強調）**
*イタリック（軽い強調・外来語）*
`コード・コマンド・ファイル名`
> 引用・注意事項
```

**例:**
```markdown
**重要:** 本番環境では必ず`HTTPS`を使用してください。

*Node.js*のバージョンは`18.0.0`以上が必要です。

> 注意: この設定は開発環境専用です。
```

---

## コード例の書き方

### 1. 言語を明示

**NG:**
````markdown
```
npm install
```
````

**OK:**
````markdown
```bash
npm install
```
````

**対応言語:**
- `bash` / `sh` / `shell`
- `typescript` / `javascript`
- `python`
- `swift`
- `yaml` / `json`
- `markdown`

### 2. コメントで説明

**NG: コメントなし**
```typescript
const user = await prisma.user.findUnique({
  where: { id: userId },
  include: { posts: true },
});
```

**OK: コメント付き**
```typescript
// ユーザー情報と関連する投稿を取得
const user = await prisma.user.findUnique({
  where: { id: userId },
  include: { posts: true }, // 投稿も一緒に取得
});
```

### 3. 完全なコード vs 抜粋

**チュートリアル: 完全なコード**
```typescript
// ファイル: src/services/user.service.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function getUser(id: string) {
  return await prisma.user.findUnique({
    where: { id },
  });
}
```

**リファレンス: 抜粋**
```typescript
// 使用例（抜粋）
const user = await getUser('user-123');
console.log(user.name);
```

### 4. 実行結果を示す

**NG: 結果なし**
```bash
npm run test
```

**OK: 結果付き**
```bash
$ npm run test

> my-app@1.0.0 test
> jest

 PASS  src/user.test.ts
  ✓ should get user (15ms)
  ✓ should create user (8ms)

Tests: 2 passed, 2 total
Time: 1.234s
```

### 5. エラー例も示す

```markdown
## よくあるエラー

### エラー: `MODULE_NOT_FOUND`

\`\`\`
Error: Cannot find module 'dotenv'
\`\`\`

**原因:** 依存関係がインストールされていません。

**解決方法:**
\`\`\`bash
npm install
\`\`\`
```

---

## 図・画像の活用

### 1. いつ図を使うか

```
✅ 図が有効:
- アーキテクチャ概要
- データフロー
- 関係図（ER図、クラス図）
- UI/UXの説明

❌ 図が不要:
- 簡単なコード例
- テキストで十分な説明
```

### 2. 図の種類

#### アーキテクチャ図

```mermaid
graph TB
    A[クライアント] --> B[API Gateway]
    B --> C[認証サービス]
    B --> D[ユーザーサービス]
    C --> E[(認証DB)]
    D --> F[(ユーザーDB)]
```

#### シーケンス図

```mermaid
sequenceDiagram
    participant U as ユーザー
    participant C as クライアント
    participant S as サーバー
    participant DB as データベース

    U->>C: ログイン
    C->>S: POST /auth/login
    S->>DB: ユーザー検証
    DB-->>S: ユーザー情報
    S-->>C: JWTトークン
    C-->>U: ログイン成功
```

#### フローチャート

```mermaid
graph TD
    A[開始] --> B{認証済み?}
    B -->|Yes| C[データ取得]
    B -->|No| D[ログイン画面]
    D --> E[認証]
    E --> C
    C --> F[終了]
```

### 3. スクリーンショット

**ベストプラクティス:**

```markdown
## インストール確認

インストールが成功すると、以下のような画面が表示されます：

![インストール成功画面](./images/install-success.png)

> 注: バージョン番号は異なる場合があります
```

**注意点:**
- 画像サイズは適切に（大きすぎない）
- altテキストを必ず付ける
- 画像は`images/`フォルダに整理

### 4. Mermaid vs 画像

**Mermaid（コードで書く図）:**
```
✅ メリット:
- テキストで管理（Git diff可能）
- 修正が簡単
- ダークモード対応

❌ デメリット:
- 複雑な図は難しい
- レンダリングが必要
```

**画像（PNG/SVG）:**
```
✅ メリット:
- 複雑な図も表現可能
- デザイン自由度が高い

❌ デメリット:
- 修正が面倒
- バージョン管理しにくい
```

**推奨:**
- シンプルな図 → Mermaid
- 複雑な図 → 画像（できればSVG）

---

## レビューと改善

### セルフレビューチェックリスト

```markdown
□ 読者視点で書けているか
  □ 初めて見る人が理解できるか
  □ 前提知識は明示されているか

□ 構成は適切か
  □ 重要な情報が最初にあるか
  □ 見出しの階層は適切か

□ コード例は十分か
  □ 全てのコード例に言語指定があるか
  □ 実際に動くコードか
  □ 説明コメントがあるか

□ スタイルは統一されているか
  □ 文体は統一されているか（です・ます）
  □ 用語は統一されているか

□ 最新の情報か
  □ 最終更新日は記載されているか
  □ 古い情報は削除されているか

□ リンクは有効か
  □ 全てのリンクが機能するか
  □ 外部リンクは適切か
```

### 他者レビューのポイント

**レビュワーの視点:**

```markdown
## ドキュメントレビューチェックリスト

### 正確性
- [ ] 技術的に正しいか
- [ ] コード例は動作するか
- [ ] バージョン情報は正しいか

### 完全性
- [ ] 必要な情報が全て含まれているか
- [ ] 手順に漏れはないか
- [ ] エッジケースが説明されているか

### 明確性
- [ ] 曖昧な表現はないか
- [ ] 専門用語は説明されているか
- [ ] 図や例は分かりやすいか

### 一貫性
- [ ] 用語は統一されているか
- [ ] 文体は統一されているか
- [ ] フォーマットは統一されているか
```

### 改善のサイクル

```
1. 執筆
   ↓
2. セルフレビュー
   ↓
3. 他者レビュー
   ↓
4. フィードバック反映
   ↓
5. 公開
   ↓
6. ユーザーフィードバック収集
   ↓
7. 改善
   ↓
（繰り返し）
```

---

## ツールとテンプレート

### 1. Markdownエディタ

**推奨ツール:**

| ツール | 特徴 | 推奨用途 |
|--------|------|---------|
| VSCode | プログラマー向け | 開発者のメイン環境 |
| Typora | WYSIWYG | 非エンジニア向け |
| Obsidian | ナレッジベース | 個人的なメモ |
| Notion | コラボレーション | チームドキュメント |

**VSCodeの推奨拡張機能:**
- Markdown All in One
- Markdown Preview Enhanced
- markdownlint
- Paste Image

### 2. Linter（文書チェック）

**textlint:**

```bash
# インストール
npm install --save-dev textlint textlint-rule-preset-ja-technical-writing

# .textlintrc
{
  "rules": {
    "preset-ja-technical-writing": true
  }
}

# 実行
npx textlint README.md
```

**チェック項目:**
- 文の長さ
- 連続した助詞
- 冗長な表現
- 誤字脱字

### 3. テンプレート集

#### README.mdテンプレート

```markdown
# プロジェクト名

<!-- バッジ（オプション） -->
![Build Status](https://img.shields.io/github/workflow/status/user/repo/CI)
![License](https://img.shields.io/github/license/user/repo)

<!-- 1行説明 -->
何を解決するプロジェクトか、1-2行で説明

## 特徴

- 主要機能1
- 主要機能2
- 主要機能3

## デモ

<!-- スクリーンショットまたはGIF -->
![デモ](./demo.gif)

## クイックスタート

\`\`\`bash
# クローン
git clone https://github.com/user/repo.git

# インストール
npm install

# 起動
npm run dev
\`\`\`

## インストール

### 必要環境

- Node.js 18+
- PostgreSQL 14+

### 詳細手順

1. リポジトリをクローン
2. 依存関係をインストール
3. 環境変数を設定
4. データベースをマイグレート
5. 開発サーバーを起動

## 使い方

### 基本的な使い方

\`\`\`typescript
import { example } from 'my-package';

const result = example();
\`\`\`

### 高度な使い方

...

## API リファレンス

詳細は [API.md](./API.md) を参照

## 設定

設定ファイル例:

\`\`\`json
{
  "option1": "value1",
  "option2": "value2"
}
\`\`\`

## 開発

### テスト

\`\`\`bash
npm test
\`\`\`

### ビルド

\`\`\`bash
npm run build
\`\`\`

## トラブルシューティング

### よくある問題

#### Q: エラーXXXが出る
A: ...

## 貢献

プルリクエストを歓迎します。
詳細は [CONTRIBUTING.md](./CONTRIBUTING.md) を参照。

## ライセンス

MIT © [Your Name]

## リンク

- [ドキュメント](https://example.com/docs)
- [デモサイト](https://example.com)
- [Issue Tracker](https://github.com/user/repo/issues)
```

#### CHANGELOGテンプレート

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 追加された新機能

### Changed
- 既存機能の変更

### Deprecated
- 非推奨になった機能

### Removed
- 削除された機能

### Fixed
- バグ修正

### Security
- セキュリティ修正

## [1.0.0] - 2025-01-15

### Added
- 初回リリース
- ユーザー認証機能
- データベース連携

### Fixed
- メモリリーク修正

## [0.1.0] - 2025-01-01

### Added
- プロトタイプ版リリース
```

---

## 実践例

### 例1: 関数のドキュメント

**NG: 説明不足**
```typescript
// ユーザーを取得
function getUser(id: string) {
  return db.users.find(id);
}
```

**OK: 完全なドキュメント**
```typescript
/**
 * ユーザー情報を取得する
 *
 * @param id - ユーザーID（UUID形式）
 * @returns ユーザー情報。見つからない場合はnull
 * @throws {DatabaseError} データベース接続エラー時
 *
 * @example
 * ```ts
 * const user = await getUser('123e4567-e89b-12d3-a456-426614174000');
 * if (user) {
 *   console.log(user.name);
 * }
 * ```
 *
 * @remarks
 * このメソッドはキャッシュを使用します。
 * リアルタイムなデータが必要な場合は`getUserFresh()`を使用してください。
 */
async function getUser(id: string): Promise<User | null> {
  return await db.users.findUnique({ where: { id } });
}
```

### 例2: APIエンドポイントのドキュメント

```markdown
## POST /api/users

新しいユーザーを作成します。

### リクエスト

**ヘッダー:**
\`\`\`
Content-Type: application/json
Authorization: Bearer {token}
\`\`\`

**ボディ:**
\`\`\`json
{
  "name": "山田太郎",
  "email": "yamada@example.com",
  "password": "securePassword123"
}
\`\`\`

**フィールド説明:**

| フィールド | 型 | 必須 | 説明 |
|-----------|-----|------|------|
| name | string | ✅ | ユーザー名（2-50文字） |
| email | string | ✅ | メールアドレス（ユニーク） |
| password | string | ✅ | パスワード（8文字以上） |

### レスポンス

**成功時（201 Created）:**
\`\`\`json
{
  "id": "user-123",
  "name": "山田太郎",
  "email": "yamada@example.com",
  "createdAt": "2025-01-15T10:00:00Z"
}
\`\`\`

**エラー時（400 Bad Request）:**
\`\`\`json
{
  "error": "VALIDATION_ERROR",
  "message": "メールアドレスは既に使用されています",
  "field": "email"
}
\`\`\`

### エラーコード

| コード | 説明 |
|--------|------|
| 400 | バリデーションエラー |
| 401 | 認証エラー |
| 409 | メールアドレス重複 |
| 500 | サーバーエラー |

### コード例

\`\`\`typescript
const response = await fetch('/api/users', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    name: '山田太郎',
    email: 'yamada@example.com',
    password: 'securePassword123'
  })
});

const user = await response.json();
console.log('作成されたユーザーID:', user.id);
\`\`\`
```

### 例3: 環境構築ガイド

```markdown
## 開発環境のセットアップ

### 1. 必要なソフトウェア

以下をインストールしてください：

| ソフトウェア | バージョン | インストール方法 |
|------------|-----------|----------------|
| Node.js | 18.0.0+ | [nodejs.org](https://nodejs.org/) |
| PostgreSQL | 14.0+ | [postgresql.org](https://www.postgresql.org/) |
| Git | 最新版 | [git-scm.com](https://git-scm.com/) |

**バージョン確認:**
\`\`\`bash
node --version  # v18.0.0以上
npm --version   # 9.0.0以上
psql --version  # 14.0以上
\`\`\`

### 2. リポジトリのクローン

\`\`\`bash
git clone https://github.com/yourname/project.git
cd project
\`\`\`

### 3. 依存関係のインストール

\`\`\`bash
npm install
\`\`\`

**所要時間:** 約2-3分

**よくあるエラー:**
- `EACCES`エラー → `sudo`は使わず、Node.jsをユーザー権限でインストール
- `network timeout`エラー → `npm config set registry https://registry.npmjs.org/`

### 4. 環境変数の設定

\`\`\`bash
cp .env.example .env
\`\`\`

`.env`ファイルを編集：

\`\`\`bash
# データベース接続
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# JWT秘密鍵（32文字以上のランダム文字列）
JWT_SECRET=your-secret-key-here-32-chars-minimum

# APIキー（開発環境用）
API_KEY=dev_key_12345
\`\`\`

**注意:**
- `JWT_SECRET`は必ず変更してください
- 本番環境では`.env`をGitにコミットしないでください

### 5. データベースのセットアップ

\`\`\`bash
# データベース作成
createdb mydb

# マイグレーション実行
npm run db:migrate

# 初期データ投入（オプション）
npm run db:seed
\`\`\`

**確認:**
\`\`\`bash
psql mydb
\d users  # usersテーブルが存在するか確認
\q
\`\`\`

### 6. 開発サーバーの起動

\`\`\`bash
npm run dev
\`\`\`

ブラウザで http://localhost:3000 を開いてください。

**成功時の画面:**

![開発サーバー起動成功](./images/dev-server-success.png)

### トラブルシューティング

#### ポート3000が既に使用されている

\`\`\`bash
# 別のポートで起動
PORT=3001 npm run dev
\`\`\`

#### データベース接続エラー

1. PostgreSQLが起動しているか確認
   \`\`\`bash
   brew services list  # macOS
   systemctl status postgresql  # Linux
   \`\`\`

2. 接続情報が正しいか確認
   \`\`\`bash
   psql $DATABASE_URL
   \`\`\`

### 次のステップ

- [開発ガイド](./DEVELOPMENT.md)を読む
- [コーディング規約](./CODING_STYLE.md)を確認
- [初めてのPR](./FIRST_PR.md)を作成
```

---

## よくある失敗

### 1. コードと乖離したドキュメント

**問題:**
```markdown
## インストール

\`\`\`bash
npm install my-package
\`\`\`
```

**実際:**
```bash
# パッケージ名が変わっていた
npm install @company/my-package
```

**対策:**
```markdown
✅ コード変更時にドキュメントも同時更新
✅ PRレビューでドキュメント確認
✅ CI/CDでドキュメントのリンク切れチェック
```

### 2. 専門用語だらけ

**NG:**
```markdown
本システムはREST APIでCRUD操作を提供し、
JWTベースのステートレス認証を採用している。
```

**OK:**
```markdown
本システムはWeb API（REST API）を提供しており、
以下の操作が可能です：
- 作成（Create）
- 読み取り（Read）
- 更新（Update）
- 削除（Delete）

認証にはJWTトークン方式を使用しています。
（トークンはサーバー側で状態を保持しないため、
スケーラビリティに優れています）
```

### 3. 手順の抜け

**NG:**
```markdown
1. リポジトリをクローン
2. npm install
3. npm run dev
```

**実際の手順:**
```markdown
1. リポジトリをクローン
2. npm install
3. .envファイル作成 ← 抜けている！
4. データベースマイグレーション ← 抜けている！
5. npm run dev
```

**対策:**
```markdown
✅ 実際に手順を試す（新しい環境で）
✅ 新メンバーに試してもらう
✅ 動画で手順を記録
```

### 4. 画像・リンク切れ

**問題:**
```markdown
![設定画面](./images/settings.png)
```

**実際:**
- ファイルが存在しない
- パスが間違っている
- リネームされた

**対策:**
```bash
# リンクチェックツール
npm install -g markdown-link-check
markdown-link-check README.md
```

### 5. 更新日がない

**NG:**
```markdown
# README

このプロジェクトは...
```

**OK:**
```markdown
# README

> 最終更新: 2025-01-15

このプロジェクトは...
```

**さらに良い:**
```markdown
# README

> 最終更新: 2025-01-15
> バージョン: 1.2.0
> 対応環境: Node.js 18+

このプロジェクトは...
```

---

## FAQ

### Q1: どのくらいの頻度でドキュメントを更新すべき？

**A**: コード変更と同時に更新が基本です。

```
コード変更 = ドキュメント変更

具体的には:
- API変更時: 必須
- 環境設定変更時: 必須
- リファクタリング: 影響があれば更新
- バグ修正: 通常は不要（ユーザーに影響なければ）
```

### Q2: README.mdはどのくらいの長さが適切？

**A**: 目的によります。

```
✅ 適切な長さ:
- ライブラリ: 200-500行
- アプリケーション: 100-300行
- 小さなツール: 50-100行

❌ 長すぎ:
- 1000行以上 → 分割を検討
```

**分割例:**
```
README.md（概要・クイックスタート）
├── docs/
│   ├── INSTALLATION.md（詳細インストール）
│   ├── USAGE.md（詳細な使い方）
│   ├── API.md（API リファレンス）
│   ├── CONTRIBUTING.md（貢献方法）
│   └── FAQ.md（よくある質問）
```

### Q3: コードコメントとドキュメントの使い分けは？

**A**:

| 種類 | 用途 | 例 |
|------|------|-----|
| **コードコメント** | 実装の詳細説明 | `// XXXのバグ対応のため、ここで処理` |
| **JSDoc/docstring** | 関数・クラスの仕様 | `@param`, `@returns` |
| **README** | プロジェクト全体の説明 | セットアップ、使い方 |
| **Wiki/ドキュメント** | 詳細ガイド | チュートリアル、アーキテクチャ |

**原則:**
```
コードで分かることは書かない
「なぜ」そうしたかを書く
```

### Q4: 英語と日本語、どちらで書くべき？

**A**: チーム・プロジェクトの状況によります。

```
✅ 日本語推奨:
- 社内プロジェクト（日本人のみ）
- 読みやすさ優先

✅ 英語推奨:
- OSSプロジェクト
- 国際チーム
- グローバル展開を視野に

✅ 両方:
- README.md（英語）+ README.ja.md（日本語）
```

### Q5: ドキュメント作成に時間がかかりすぎる

**A**: テンプレートと段階的アプローチを活用してください。

**段階的アプローチ:**
```
Phase 1: 最低限
- README基本（概要・セットアップ）
- 10分で作成

Phase 2: 基本機能
- 使い方セクション追加
- 30分で作成

Phase 3: 詳細
- API リファレンス
- トラブルシューティング
- 1-2時間で作成

Phase 4: 継続的改善
- ユーザーフィードバック反映
- 定期的に更新
```

### Q6: 技術選定の理由をどこに書くべき？

**A**: ADR（Architecture Decision Record）を使いましょう。

```markdown
# docs/adr/001-use-typescript.md

# ADR-001: TypeScriptを採用

## ステータス
採用

## コンテキスト
新規プロジェクトで使用する言語を選定する必要がある。
候補: JavaScript, TypeScript, Flow

## 決定
TypeScriptを採用する。

## 理由
1. 型安全性によるバグ削減
2. IDEサポートが充実
3. チーム全員がTypeScript経験あり
4. エコシステムが大きい

## 代替案
- JavaScript: 型がないためバグが多くなる懸念
- Flow: エコシステムが小さい

## 影響
- 学習コスト: 低（全員経験あり）
- ビルド時間: やや増加（許容範囲）
- 開発生産性: 向上が期待できる
```

### Q7: ドキュメントの品質をどう測る？

**A**: 以下の指標を使います。

```markdown
## ドキュメント品質メトリクス

### 定量指標
- [ ] カバレッジ: 主要機能の90%以上をドキュメント化
- [ ] 鮮度: 最終更新が3ヶ月以内
- [ ] リンク切れ: 0件

### 定性指標
- [ ] 新メンバーが1日でセットアップ完了できる
- [ ] よくある質問が減った（週5件 → 週1件）
- [ ] ドキュメント由来のバグ報告: 0件
```

### Q8: Mermaidが表示されない

**A**: 環境によって対応が異なります。

```markdown
✅ GitHub: 自動レンダリング
✅ GitLab: 自動レンダリング
✅ VSCode: Markdown Preview Mermaid拡張が必要
❌ 古いビューアー: 画像として別途生成が必要

対策:
1. Mermaid Live Editorで画像生成
2. SVGとして保存
3. 画像とコードを両方掲載
```

---

## まとめ

### Technical Writingの5原則

```
1. 読者視点で書く
   → 初めて見る人が理解できるか

2. シンプルで明確に
   → 1文は短く、専門用語は説明

3. 具体例を示す
   → コード例、画像、手順

4. 最新の状態を保つ
   → コード変更時に同時更新

5. 継続的に改善
   → フィードバックを反映
```

### 次のステップ

```markdown
□ 既存ドキュメントをレビュー
  □ 古い情報を削除
  □ 不足している情報を追加
  □ リンク切れを修正

□ テンプレートを準備
  □ README.md
  □ CHANGELOG.md
  □ ADR

□ ドキュメント文化を作る
  □ PR時にドキュメント更新を確認
  □ 定期的なドキュメント棚卸し
  □ 新メンバーにフィードバック依頼
```

### 参考リンク

- [Write the Docs](https://www.writethedocs.org/) - Technical Writingコミュニティ
- [Google Developer Documentation Style Guide](https://developers.google.com/style)
- [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
- [GitLab Documentation Guide](https://docs.gitlab.com/ee/development/documentation/)

---

## 📚 Appendix

### A. ドキュメントタイプ別テンプレート集

#### A-1. チュートリアルテンプレート

```markdown
# [機能名]のチュートリアル

> **所要時間:** 10分
> **難易度:** 初級
> **前提知識:** JavaScript基礎

## 学べること

このチュートリアルでは以下を学びます：

- [ ] 基本的な使い方
- [ ] よくあるユースケース
- [ ] トラブルシューティング

## 必要な準備

- Node.js 18+
- テキストエディタ

## ステップ1: セットアップ

まず、プロジェクトをセットアップします。

\`\`\`bash
mkdir my-project
cd my-project
npm init -y
\`\`\`

**確認:**
\`package.json\`が作成されていればOKです。

## ステップ2: パッケージのインストール

\`\`\`bash
npm install example-package
\`\`\`

**所要時間:** 約30秒

## ステップ3: 最初のコード

\`index.js\`を作成します：

\`\`\`javascript
const example = require('example-package');

// 基本的な使い方
const result = example.doSomething('Hello');
console.log(result); // => "Hello, World!"
\`\`\`

## ステップ4: 実行

\`\`\`bash
node index.js
\`\`\`

**期待される出力:**
\`\`\`
Hello, World!
\`\`\`

## 次のステップ

- [高度な使い方](./advanced.md)
- [API リファレンス](./api.md)
- [実践例集](./examples.md)

## トラブルシューティング

### エラー: Cannot find module

\`\`\`bash
npm install
\`\`\`

を実行してください。
```

#### A-2. ハウツーガイドテンプレート

```markdown
# [タスク名]の方法

## 概要

この手順では、[達成したいこと]を実現します。

**所要時間:** 15分

## 前提条件

- [ ] [必要な環境1]
- [ ] [必要な環境2]

## 手順

### 1. [最初のステップ]

\`\`\`bash
command1
\`\`\`

**説明:** このコマンドは[何をするか]を実行します。

### 2. [次のステップ]

\`\`\`bash
command2
\`\`\`

**注意:** [気をつけるべき点]

### 3. [最後のステップ]

\`\`\`bash
command3
\`\`\`

**確認方法:**
\`\`\`bash
# 正しく設定されているか確認
command-to-verify
\`\`\`

## 確認

以下が表示されれば成功です：

\`\`\`
Success message
\`\`\`

## トラブルシューティング

### 問題: [よくある問題1]

**症状:**
\`\`\`
Error message
\`\`\`

**解決方法:**
1. [手順1]
2. [手順2]

### 問題: [よくある問題2]

...

## 関連ドキュメント

- [関連するガイド1]
- [関連するガイド2]
```

#### A-3. リファレンステンプレート

```markdown
# [API名] リファレンス

## 概要

[APIの簡単な説明]

## インポート

\`\`\`typescript
import { functionName } from 'package-name';
\`\`\`

## 関数

### functionName()

[関数の説明]

**シグネチャ:**
\`\`\`typescript
function functionName(
  param1: string,
  param2?: number,
  options?: Options
): Promise<Result>
\`\`\`

**パラメータ:**

| 名前 | 型 | 必須 | デフォルト | 説明 |
|------|-----|------|-----------|------|
| param1 | string | ✅ | - | [説明] |
| param2 | number | ❌ | 0 | [説明] |
| options | Options | ❌ | {} | [説明] |

**戻り値:**

| 型 | 説明 |
|----|------|
| Promise<Result> | [説明] |

**例外:**

| 例外 | 条件 |
|------|------|
| ValidationError | [条件] |
| NetworkError | [条件] |

**使用例:**

\`\`\`typescript
// 基本的な使い方
const result = await functionName('value');

// オプション付き
const result = await functionName('value', 10, {
  timeout: 5000
});

// エラーハンドリング
try {
  const result = await functionName('value');
} catch (error) {
  if (error instanceof ValidationError) {
    console.error('Invalid input');
  }
}
\`\`\`

**参照:**
- [関連する関数]
- [関連するタイプ]

## タイプ定義

### Options

\`\`\`typescript
interface Options {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}
\`\`\`

| プロパティ | 型 | デフォルト | 説明 |
|----------|-----|-----------|------|
| timeout | number | 3000 | タイムアウト（ミリ秒） |
| retries | number | 3 | リトライ回数 |
| headers | Record<string, string> | {} | カスタムヘッダー |
```

### B. ライティングチェックリスト

#### B-1. 公開前チェックリスト

```markdown
## ドキュメント公開前チェックリスト

### 内容
- [ ] タイトルは明確か
- [ ] 目次は適切か
- [ ] 対象読者は明記されているか
- [ ] 前提条件は明記されているか
- [ ] 全セクションに内容があるか

### 正確性
- [ ] 全てのコマンドは動作するか
- [ ] 全てのコード例は動作するか
- [ ] バージョン情報は正しいか
- [ ] リンクは全て有効か
- [ ] スクリーンショットは最新か

### 完全性
- [ ] セットアップ手順は完全か
- [ ] エラーケースは説明されているか
- [ ] トラブルシューティングがあるか
- [ ] 次のステップが示されているか

### 読みやすさ
- [ ] 文章は簡潔か
- [ ] 専門用語は説明されているか
- [ ] コード例にコメントがあるか
- [ ] 適切に段落分けされているか
- [ ] 見出しの階層は適切か

### フォーマット
- [ ] Markdown構文は正しいか
- [ ] コードブロックに言語指定があるか
- [ ] 表は適切にフォーマットされているか
- [ ] 画像のaltテキストがあるか

### メタ情報
- [ ] 最終更新日が記載されているか
- [ ] 対象バージョンが記載されているか
- [ ] 著者/メンテナーが明記されているか

### アクセシビリティ
- [ ] 画像に代替テキストがあるか
- [ ] 色だけに依存していないか
- [ ] リンクテキストは明確か
```

#### B-2. レビュー観点チェックリスト

```markdown
## レビュワー用チェックリスト

### 第一印象（5分）
- [ ] タイトルで内容が分かるか
- [ ] 目次で構成が分かるか
- [ ] ざっと見て理解できそうか

### 正確性（15分）
- [ ] 技術的に正しいか
- [ ] コード例は動作するか
- [ ] 最新の情報か

### 完全性（10分）
- [ ] 必要な情報が全て含まれているか
- [ ] 手順に漏れはないか
- [ ] エッジケースが説明されているか

### 明確性（10分）
- [ ] 曖昧な表現はないか
- [ ] 専門用語は説明されているか
- [ ] 図や例は分かりやすいか

### ユーザビリティ（10分）
- [ ] 目的の情報が見つけやすいか
- [ ] 手順は実行しやすいか
- [ ] トラブル時の対処法があるか

### 総合評価
- [ ] 推奨（そのまま公開OK）
- [ ] 条件付き推奨（軽微な修正後に公開）
- [ ] 要改善（大幅な修正が必要）
```

### C. ベストプラクティス集

#### C-1. コマンドの書き方

**基本ルール:**

```markdown
✅ Good: シェル指定とコメント
\`\`\`bash
# プロジェクトをクローン
git clone https://github.com/user/repo.git

# ディレクトリに移動
cd repo
\`\`\`

❌ Bad: 言語指定なし、コメントなし
\`\`\`
git clone https://github.com/user/repo.git
cd repo
\`\`\`
```

**プロンプト表示:**

```markdown
✅ インタラクティブなコマンド:
\`\`\`bash
$ npm install
added 142 packages in 3s

$ npm run dev
Server running on http://localhost:3000
\`\`\`

❌ コピペできないコマンド:
\`\`\`bash
$ npm install
$ npm run dev
\`\`\`
(プロンプト記号のせいでコピペできない)
```

**環境依存コマンド:**

```markdown
✅ 環境別に分ける:

**macOS:**
\`\`\`bash
brew install postgresql
\`\`\`

**Ubuntu/Debian:**
\`\`\`bash
sudo apt-get install postgresql
\`\`\`

**Windows:**
\`\`\`bash
choco install postgresql
\`\`\`

❌ 一つだけ:
\`\`\`bash
brew install postgresql
\`\`\`
(macOSユーザーしか使えない)
```

#### C-2. エラーメッセージの扱い方

**完全なエラー表示:**

```markdown
✅ Good:
### エラー: MODULE_NOT_FOUND

\`\`\`
Error: Cannot find module 'dotenv'
Require stack:
- /Users/name/project/src/index.js
- /Users/name/project/node_modules/jest/index.js
    at Function.Module._resolveFilename (internal/modules/cjs/loader.js:880:15)
    at Function.Module._load (internal/modules/cjs/loader.js:725:27)
\`\`\`

**原因:**
`dotenv`パッケージがインストールされていません。

**解決方法:**
\`\`\`bash
npm install dotenv
\`\`\`

❌ Bad:
エラーが出たらnpm installしてください。
```

#### C-3. バージョン表記

```markdown
✅ Good: 具体的なバージョン
- Node.js 18.0.0以上
- PostgreSQL 14.0以上
- React 18.2.0以上

✅ Good: 範囲指定
- Node.js 18.x or 20.x
- PostgreSQL 14.x - 15.x

❌ Bad: 曖昧
- Node.js最新版
- PostgreSQL（バージョン不明）
```

#### C-4. コード例の完全性

```markdown
✅ Complete example:
\`\`\`typescript
// src/services/user.service.ts
import { PrismaClient } from '@prisma/client';
import { User } from '../types';

const prisma = new PrismaClient();

export async function getUser(id: string): Promise<User | null> {
  try {
    return await prisma.user.findUnique({
      where: { id },
    });
  } catch (error) {
    console.error('Failed to fetch user:', error);
    throw error;
  }
}
\`\`\`

❌ Incomplete example:
\`\`\`typescript
const user = await getUser(id);
\`\`\`
(どこからimportするのか、エラーハンドリングは？)
```

### D. アンチパターン詳細

#### D-1. 説明しすぎ

**問題:**
```typescript
/**
 * この関数はユーザーIDを受け取ります。
 * そのユーザーIDを使ってデータベースにクエリを投げます。
 * データベースからユーザー情報が返ってきます。
 * そのユーザー情報を呼び出し元に返します。
 * ユーザーが見つからない場合はnullを返します。
 * エラーが発生した場合は例外をスローします。
 */
function getUser(id: string): Promise<User | null> {
  return db.users.findUnique({ where: { id } });
}
```

**改善:**
```typescript
/**
 * ユーザー情報を取得
 *
 * @param id - ユーザーID
 * @returns ユーザー情報。見つからない場合はnull
 * @throws {DatabaseError} DB接続エラー時
 */
function getUser(id: string): Promise<User | null> {
  return db.users.findUnique({ where: { id } });
}
```

#### D-2. 前提知識の未記載

**問題:**
```markdown
## セットアップ

\`\`\`bash
npm install
npm run dev
\`\`\`
```

**改善:**
```markdown
## セットアップ

### 前提条件

以下がインストールされている必要があります：
- Node.js 18.0.0以上
- npm 9.0.0以上

**確認方法:**
\`\`\`bash
node --version  # v18.0.0以上
npm --version   # 9.0.0以上
\`\`\`

### インストール

\`\`\`bash
npm install
npm run dev
\`\`\`
```

#### D-3. 手順の曖昧さ

**問題:**
```markdown
1. 環境変数を設定する
2. データベースをセットアップする
3. アプリを起動する
```

**改善:**
```markdown
### 1. 環境変数を設定

\`.env\`ファイルを作成し、以下を設定：

\`\`\`bash
DATABASE_URL=postgresql://localhost:5432/mydb
API_KEY=your_key_here
\`\`\`

### 2. データベースをセットアップ

\`\`\`bash
# データベース作成
createdb mydb

# マイグレーション実行
npm run db:migrate
\`\`\`

**確認:**
\`\`\`bash
psql mydb -c "\\dt"
# テーブル一覧が表示されればOK
\`\`\`

### 3. アプリを起動

\`\`\`bash
npm run dev
\`\`\`

ブラウザで http://localhost:3000 を開いてください。
```

### E. 多言語対応

#### E-1. 英語版と日本語版の管理

**ディレクトリ構成:**

```
docs/
├── README.md (英語)
├── README.ja.md (日本語)
├── en/
│   ├── guide.md
│   └── api.md
└── ja/
    ├── guide.md
    └── api.md
```

**言語切り替えリンク:**

```markdown
<!-- README.md -->
# Project Name

[🇯🇵 日本語](./README.ja.md)

...
```

```markdown
<!-- README.ja.md -->
# プロジェクト名

[🇬🇧 English](./README.md)

...
```

#### E-2. i18n対応のREADME

```markdown
# Project Name

> [!NOTE]
> **Language:** [English](#) | [日本語](./README.ja.md) | [中文](./README.zh.md)

## Overview

...

## Installation

...
```

### F. 継続的な改善

#### F-1. ドキュメントメトリクス

```markdown
## ドキュメント品質ダッシュボード

### カバレッジ

| カテゴリ | ドキュメント化率 | 目標 | 状態 |
|---------|----------------|------|------|
| API | 85% (17/20) | 90% | ⚠️ |
| チュートリアル | 100% (5/5) | 100% | ✅ |
| リファレンス | 60% (12/20) | 80% | ❌ |

### 鮮度

| ドキュメント | 最終更新 | 状態 |
|------------|---------|------|
| README.md | 2日前 | ✅ |
| API.md | 15日前 | ⚠️ |
| SETUP.md | 90日前 | ❌ |

### ユーザーフィードバック

| 月 | 質問数 | ドキュメント改善数 |
|----|-------|------------------|
| 1月 | 45 | 12 |
| 2月 | 38 | 8 |
| 3月 | 25 | 15 |

トレンド: 📉 質問減少（ドキュメント改善の効果）
```

#### F-2. 定期レビュー

```markdown
## 四半期ドキュメントレビュー

### チェックリスト

#### Q1 2025 レビュー（1-3月）

**対象ドキュメント:**
- [ ] README.md
- [ ] API.md
- [ ] チュートリアル全般
- [ ] トラブルシューティング

**レビュー観点:**
- [ ] 情報は最新か（バージョン、スクリーンショット）
- [ ] リンク切れはないか
- [ ] 新機能は反映されているか
- [ ] ユーザーからの質問で多いものは追加されているか

**実施日:** 2025-03-31
**担当:** ドキュメントチーム

**改善アクション:**
1. API.mdのバージョン更新
2. 新機能（ダークモード）のチュートリアル追加
3. よくある質問（FAQ）セクション追加
```

---

**このガイドは実際のプロジェクトで進化していきます。**
**フィードバックをお待ちしています！**
