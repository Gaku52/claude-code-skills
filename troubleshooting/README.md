# トラブルシューティングデータベース

## 概要

開発中に遭遇する頻出エラーと解決策をまとめたデータベースです。

**収録エラー数:** 125個（各技術25個）

**対象技術:**
- React
- Node.js
- Python
- Backend（API/Database/Deploy）
- iOS

---

## トラブルシューティングガイド一覧

### 1. [React トラブルシューティング](./react-troubleshooting.md)

**収録エラー:** 25個 | **行数:** 768行

**主なカテゴリ:**
- セットアップ・環境構築エラー
- Hooks関連エラー（useEffect, useState, useCallback）
- コンポーネントエラー（key prop, display name）
- State・Props関連エラー
- イベント処理エラー
- ルーティングエラー（React Router）
- ビルド・デプロイエラー
- パフォーマンス問題（メモリリーク、再レンダリング）

**頻出エラー例:**
- `Hooks can only be called inside the body of a function component`
- `Too many re-renders`
- `Each child in a list should have a unique "key" prop`
- `Cannot read property 'map' of undefined`

---

### 2. [Node.js トラブルシューティング](./nodejs-troubleshooting.md)

**収録エラー:** 25個 | **行数:** 1,332行

**主なカテゴリ:**
- インストール・環境構築エラー
- モジュール・依存関係エラー
- サーバー・ネットワークエラー
- データベース接続エラー（Prisma, MongoDB, MySQL）
- 認証・認可エラー（JWT, bcrypt）
- パフォーマンス・メモリエラー
- ビルド・デプロイエラー

**頻出エラー例:**
- `Port 3000 is already in use`
- `Cannot find module 'express'`
- `EACCES: permission denied`
- `Prisma Client did not initialize yet`
- `JWT token expired`

---

### 3. [Python トラブルシューティング](./python-troubleshooting.md)

**収録エラー:** 25個 | **行数:** 1,233行

**主なカテゴリ:**
- インストール・環境構築エラー
- 構文・インデントエラー
- 型・値エラー（TypeError, ValueError）
- モジュール・インポートエラー
- ファイル・パスエラー
- Webフレームワークエラー（Flask, FastAPI, Django）
- データベースエラー（SQLAlchemy）

**頻出エラー例:**
- `ModuleNotFoundError: No module named 'flask'`
- `IndentationError: expected an indented block`
- `TypeError: 'NoneType' object is not subscriptable`
- `ImportError: cannot import name`

---

### 4. [Backend トラブルシューティング](./backend-troubleshooting.md)

**収録エラー:** 25個 | **行数:** 1,866行

**主なカテゴリ:**
- HTTP・ネットワークエラー（404, 500, 401, 403）
- 認証・認可エラー
- CORS・セキュリティエラー
- データベースエラー（Connection pool, N+1 query, Deadlock）
- パフォーマンス問題
- サーバー設定エラー
- デプロイエラー（Render, Vercel, AWS）

**頻出エラー例:**
- `Access to fetch has been blocked by CORS policy`
- `401 Unauthorized - Invalid credentials`
- `Database connection pool exhausted`
- `N+1 query detected`

---

### 5. [iOS トラブルシューティング](./ios-troubleshooting.md)

**収録エラー:** 25個 | **行数:** 1,557行

**主なカテゴリ:**
- ビルド・Xcodeエラー
- SwiftUIエラー（Preview, Binding, View protocol）
- 状態管理エラー（@State, @Published, @EnvironmentObject）
- ナビゲーションエラー
- データ永続化エラー（UserDefaults, Keychain, Core Data）
- ネットワークエラー（URLSession, ATS）
- パフォーマンス問題

**頻出エラー例:**
- `No such module 'SwiftUI'`
- `SwiftUI preview crashed`
- `@State property not updating UI`
- `Keychain access denied`

---

## 使い方

### エラーメッセージから検索

1. **エラーメッセージをコピー**
   ```
   Error: Cannot find module 'express'
   ```

2. **該当するトラブルシューティングガイドを開く**
   - Node.js関連 → `nodejs-troubleshooting.md`
   - React関連 → `react-troubleshooting.md`
   - など

3. **ページ内検索（Ctrl+F / Cmd+F）**
   - エラーメッセージの一部で検索
   - 例：`Cannot find module`

4. **解決策を実行**
   - コード例をコピー&ペースト
   - 段階的に解決

### カテゴリから探す

各トラブルシューティングガイドは目次でカテゴリ分けされています：

```markdown
## 目次
1. セットアップエラー
2. 構文エラー
3. ネットワークエラー
4. データベースエラー
...
```

該当するカテゴリをクリックして、関連エラーを確認してください。

---

## エラー解決の基本手順

### 1. エラーメッセージを読む

**最初の1-2行が最重要:**
```
Error: Cannot find module 'express'
Require stack:
- /Users/project/src/server.js
```

### 2. スタックトレースを確認

**エラー発生箇所を特定:**
```
at Object.<anonymous> (/Users/project/src/server.js:1:15)
```
→ `server.js` の 1行目、15文字目

### 3. このガイドで検索

**キーワードで検索:**
- `Cannot find module`
- `CORS policy`
- `IndentationError`

### 4. 解決策を試す

**段階的に実行:**
1. 簡単な解決策から試す（再起動、再インストール）
2. 設定ファイルを確認
3. コードを修正
4. 動作確認

### 5. 再発防止

**解決後にすべきこと:**
- エラー原因をメモ
- `.gitignore`に追加（環境ファイル等）
- チェックリストに追加

---

## よくある質問

### Q1: エラーメッセージが長すぎて何が問題かわからない

**A:** 最初の1-2行に注目してください。多くの場合、エラーの核心部分が最初に表示されます。

**例:**
```
Error: listen EADDRINUSE: address already in use :::3000  ← これが重要
    at Server.setupListenHandle [as _listen2] (net.js:1318:16)
    at listenInCluster (net.js:1366:12)
    at Server.listen (net.js:1452:7)
    ...（以下、スタックトレース）
```

### Q2: 解決策を試しても直らない

**A:** 以下を確認してください：
1. **バージョンの違い** - 使用しているバージョンを確認
2. **環境の違い** - macOS、Windows、Linuxで解決策が異なる場合あり
3. **複数のエラー** - 1つ解決しても、次のエラーが出ることがある
4. **キャッシュ** - `node_modules`削除、ブラウザキャッシュクリア

### Q3: このガイドにないエラーが出た

**A:** 以下の手順で解決してください：
1. **公式ドキュメント** - 各技術の公式サイトを確認
2. **Google検索** - エラーメッセージをそのまま検索
3. **Stack Overflow** - 同じエラーに遭遇した人の質問
4. **GitHub Issues** - ライブラリの既知の問題

### Q4: エラーは解決したが、なぜそうなったかわからない

**A:** エラーの原因を理解することが重要です：
- **このガイドの「原因」セクションを読む**
- **公式ドキュメントで概念を学ぶ**
- **同じエラーを意図的に再現してみる**

---

## デバッグツール

### React
- **[React DevTools](https://react.dev/learn/react-developer-tools)** - コンポーネント・State確認
- **[Redux DevTools](https://github.com/reduxjs/redux-devtools)** - Redux State確認
- **Chrome DevTools** - パフォーマンス測定

### Node.js
- **[Node.js Inspector](https://nodejs.org/en/docs/guides/debugging-getting-started/)** - デバッグ
- **[Clinic.js](https://clinicjs.org/)** - パフォーマンス分析
- **[PM2](https://pm2.keymetrics.io/)** - プロセス管理

### Python
- **[pdb](https://docs.python.org/3/library/pdb.html)** - デバッガー
- **[ipdb](https://pypi.org/project/ipdb/)** - IPython デバッガー
- **[Python Tutor](https://pythontutor.com/)** - コード可視化

### Backend
- **[Postman](https://www.postman.com/)** - API テスト
- **[curl](https://curl.se/)** - HTTP リクエスト
- **[pgAdmin](https://www.pgadmin.org/)** - PostgreSQL 管理

### iOS
- **[Xcode Debugger](https://developer.apple.com/documentation/xcode/debugging)** - デバッグ
- **[Instruments](https://developer.apple.com/documentation/xcode/improving-your-app-s-performance)** - パフォーマンス測定
- **[SwiftUI Inspector](https://developer.apple.com/documentation/swiftui/view-inspector)** - View 階層確認

---

## 統計データ

### 収録エラー数

| 技術 | エラー数 | 行数 |
|------|---------|------|
| React | 25個 | 768行 |
| Node.js | 25個 | 1,332行 |
| Python | 25個 | 1,233行 |
| Backend | 25個 | 1,866行 |
| iOS | 25個 | 1,557行 |
| **合計** | **125個** | **6,756行** |

### カテゴリ別内訳

| カテゴリ | エラー数 |
|---------|---------|
| セットアップ・環境構築 | 20個 |
| 構文・型エラー | 18個 |
| モジュール・依存関係 | 15個 |
| ネットワーク・HTTP | 15個 |
| データベース | 12個 |
| 認証・認可 | 10個 |
| ビルド・デプロイ | 10個 |
| パフォーマンス | 10個 |
| その他 | 15個 |

---

## 貢献

新しいエラーや解決策を見つけた場合は、以下の方法で共有してください：

1. **GitHub Issues** - [新しいエラーを報告](https://github.com/Gaku52/claude-code-skills/issues)
2. **Pull Request** - 直接編集して提案

**フォーマット:**
```markdown
### ❌ エラー: [エラーメッセージ]

\```
[エラーメッセージの例]
\```

**原因:**
- [原因1]
- [原因2]

**解決策:**

\```language
[解決策のコード]
\```
```

---

## まとめ

**このトラブルシューティングDBで学べること:**
- ✅ 125個の頻出エラーと解決策
- ✅ 実行可能なコード例
- ✅ エラーの原因と根本解決
- ✅ デバッグツールの使い方
- ✅ 再発防止策

**エラー解決能力を向上させるコツ:**
1. エラーメッセージを恐れない
2. スタックトレースを読む習慣をつける
3. 公式ドキュメントを確認する
4. 同じエラーを2度繰り返さない（メモを取る）
5. デバッグツールを活用する

---

## 関連ガイド

### 基礎ガイド
- [React Development](../react-development/SKILL.md)
- [Node.js Development](../nodejs-development/SKILL.md)
- [Python Development](../python-development/SKILL.md)
- [Backend Development](../backend-development/SKILL.md)
- [iOS Development](../ios-development/SKILL.md)

### 統合プロジェクト
- [フルスタックタスク管理アプリ](../integrated-projects/fullstack-task-app/)

---

**エラーに遭遇したら、このガイドを活用してください！** 頻出エラーの90%はここで解決できます。
