# Documentation Updates Log

このファイルは週次自動チェックで検知されたバージョン更新情報を記録します。

**更新頻度:** 毎週日曜 深夜2:00（JST）
**ワークフロー:** `.github/workflows/weekly-update-check.yml`
**最終チェック:** 初回未実行

---

## 使い方

1. 週次で自動実行され、このファイルに更新情報が追記されます
2. 差分があればDraft PRが自動作成されます
3. **あなたがレビュー・マージ判断**をしてください
4. マージされて初めてドキュメントが正式に更新されます

---

## 更新ログ

<!-- 自動生成エリア（この下に週次ログが追記されます） -->

### 📝 初回セットアップ (2025-01-27)

自動更新システムを構築しました。

**監視対象パッケージ:**
- Testing: Jest, Vitest, Playwright, Cypress, Supertest, Testcontainers
- Database: Prisma, TypeORM, Knex, PostgreSQL, MySQL
- Backend: Express, Fastify, NestJS
- Language/Runtime: TypeScript, Node.js

次回チェック予定: 2025-02-02（日曜）

---

<!-- 以下、自動生成ログが追記されます -->

---

## 2025-12-26 (自動検出)

### 🔄 バージョン更新検知


#### Jest
- **現在のドキュメント:** 29.0.0
- **最新バージョン:** 30.2.0
- **リリースノート:** https://github.com/jestjs/jest/releases/tag/v30.2.0
- **影響範囲:**
  - testing-strategy/guides/unit/unit-testing-complete.md
  - ci-cd-automation/guides/quality/quality-automation-complete.md

**Claude分析結果:**
```
null
```


#### Playwright
- **現在のドキュメント:** 1.40.0
- **最新バージョン:** 1.57.0
- **リリースノート:** https://github.com/microsoft/playwright/releases/tag/v1.57.0
- **影響範囲:**
  - testing-strategy/guides/e2e/e2e-testing-complete.md

**Claude分析結果:**
```
null
```


#### TypeScript
- **現在のドキュメント:** 5.0.0
- **最新バージョン:** 5.9.3
- **リリースノート:** https://github.com/microsoft/TypeScript/releases/tag/v5.9.3
- **影響範囲:**
  - backend-development/guides/typescript/typescript-backend-complete.md
  - nodejs-development/guides/typescript/typescript-patterns-complete.md


#### Next.js
- **現在のドキュメント:** null
- **最新バージョン:** 16.1.1
- **リリースノート:** https://github.com/vercel/next.js/releases
- **影響範囲:**
  - nextjs-development/guides/* (全ファイル)


#### React
- **現在のドキュメント:** null
- **最新バージョン:** 19.2.3
- **リリースノート:** https://github.com/facebook/react/releases
- **影響範囲:**
  - react-development/guides/* (全ファイル)


#### Node.js
- **現在のドキュメント:** null
- **最新バージョン:** 25.2.1
- **リリースノート:** https://nodejs.org/en/blog/release/
- **影響範囲:**
  - nodejs-development/guides/* (全ファイル)
  - backend-development/guides/* (全ファイル)


### 📋 推奨アクション
- [ ] 上記リリースノートを確認
- [ ] 影響範囲のファイルを更新
- [ ] .doc-versions.json を更新
- [ ] このPRをマージ

