# 🔍 Root Cause Analysis - 根本原因分析ガイド

> **目的**: インシデントの表面的な症状ではなく、真の根本原因を特定し、効果的な再発防止策を導く

## 📚 目次

1. [根本原因分析とは](#根本原因分析とは)
2. [5 Whys 分析手法](#5-whys-分析手法)
3. [Fishbone Diagram（特性要因図）](#fishbone-diagram特性要因図)
4. [Fault Tree Analysis（故障の木解析）](#fault-tree-analysis故障の木解析)
5. [Timeline Analysis（時系列分析）](#timeline-analysis時系列分析)
6. [実践例](#実践例)
7. [RCA テンプレート](#rca-テンプレート)

---

## 根本原因分析とは

### なぜ RCA が必要か

**表面的な対処 vs 根本原因の解決**:
```
❌ 表面的な対処
問題: ログイン失敗
対処: サーバー再起動
結果: 一時的に解決するが、再発する

✅ 根本原因の解決
問題: ログイン失敗
RCA: メモリリークが原因でプロセスが異常終了
対処: メモリリークを修正 + 監視アラート追加
結果: 再発しない
```

### RCA の基本原則

**1. 人を責めず、システムを改善する**
```markdown
❌ 悪い例:
「担当者が環境変数を設定し忘れた」
→ 個人の責任に帰結

✅ 良い例:
「環境変数の設定チェックリストがなく、手動設定に依存していた」
→ プロセスの改善に繋がる
```

**2. 複数の原因を考慮する**
- 直接原因（Immediate Cause）
- 間接原因（Contributing Cause）
- 根本原因（Root Cause）

```
例: 本番環境でデータ削除

直接原因: DELETE クエリの WHERE 句がなかった
間接原因: コードレビューで気づかれなかった
根本原因: テスト環境でのデータ検証が不十分だった
```

**3. データに基づく分析**
- ログ、メトリクス、トレースを確認
- 仮説検証を繰り返す
- 主観ではなく事実を重視

---

## 5 Whys 分析手法

### 概要

**5回「なぜ？」を繰り返して根本原因に到達する**

```
問題 → なぜ？ → なぜ？ → なぜ？ → なぜ？ → なぜ？ → 根本原因
```

### 実践例1: 本番環境でのデータ削除

**問題**: 本番環境で誤って全ユーザーデータを削除してしまった

```markdown
## 5 Whys Analysis

### 問題
本番環境で全ユーザーデータが削除された

### Why #1: なぜデータが削除されたか？
→ DELETE クエリに WHERE 句がなかったため

### Why #2: なぜ WHERE 句がなかったか？
→ 開発環境でのテストコードをそのまま本番に投入したため

### Why #3: なぜテストコードが本番に投入されたか？
→ コードレビューで気づかれなかったため

### Why #4: なぜコードレビューで気づかれなかったか？
→ データベース操作のチェックリストがなかったため

### Why #5: なぜチェックリストがなかったか？
→ データベース操作のリスクが軽視されていたため

## 根本原因
**データベース操作に対するリスク認識とプロセスの欠如**

## 対策

### 直接対策（即時実施）
- [x] バックアップから復旧
- [x] DELETE/UPDATE クエリには必ず WHERE 句を必須化
- [x] 本番環境でのクエリ実行前に DRY RUN を必須化

### 根本対策（1週間以内）
- [ ] データベース操作チェックリスト作成
- [ ] コードレビュー時の必須確認項目に追加
- [ ] 本番環境でのデータベース操作には承認フロー導入

### 予防対策（1ヶ月以内）
- [ ] 本番環境へのクエリ実行を制限（読み取り専用ユーザー）
- [ ] データベース操作は CI/CD 経由のみに限定
- [ ] 定期的なバックアップ検証
```

### 実践例2: iOS アプリクラッシュ

**問題**: iOS アプリが起動時にクラッシュする

```markdown
## 5 Whys Analysis

### 問題
iOS アプリが起動時にクラッシュする（iOS 15 のみ）

### Why #1: なぜクラッシュするか？
→ `NavigationStack` が iOS 15 で未対応のため

### Why #2: なぜ iOS 15 未対応の API を使用したか？
→ Xcode の Preview で正常動作していたため気づかなかった

### Why #3: なぜ Preview で気づかなかったか？
→ Preview は iOS 16+ で動作しており、下位 OS を確認していなかった

### Why #4: なぜ下位 OS を確認していなかったか？
→ CI/CD に iOS 15 実機テストがなかった

### Why #5: なぜ CI/CD に iOS 15 テストがなかったか？
→ 最小サポートバージョンのテスト戦略が明確でなかった

## 根本原因
**最小サポートバージョンに対するテスト戦略の欠如**

## 対策

### 直接対策（即時実施）
- [x] iOS 15 対応コードに修正
- [x] 緊急リリース（v2.5.1）

```swift
// 修正コード
if #available(iOS 16.0, *) {
    NavigationStack {
        MainTabView()
    }
} else {
    NavigationView {
        MainTabView()
    }
    .navigationViewStyle(.stack)
}
```

### 根本対策（1週間以内）
- [ ] CI/CD に最小サポートバージョン（iOS 15）の実機テスト追加
- [ ] デプロイ前チェックリストに下位 OS 動作確認を追加

### 予防対策（1ヶ月以内）
- [ ] Xcode Preview 設定をサポート最小バージョンに統一
- [ ] 新 API 使用時には `@available` チェックを必須化
- [ ] 定期的な下位 OS 互換性レビュー
```

### 5 Whys の注意点

**❌ 避けるべきパターン**:

```markdown
## 悪い例

### Why #1: なぜバグが発生したか？
→ エンジニアがミスしたため

### Why #2: なぜミスしたか？
→ 注意力が足りなかったため

### Why #3: なぜ注意力が足りなかったか？
→ 忙しかったため

### Why #4: なぜ忙しかったか？
→ 人手不足だったため

### Why #5: なぜ人手不足だったか？
→ 予算がなかったため

## 問題点
- 個人を責める方向に進んでいる
- システム・プロセスの改善に繋がらない
- 「予算」という解決困難な問題に行き着く
```

**✅ 良いパターン**:

```markdown
## 良い例

### Why #1: なぜバグが発生したか？
→ テストケースに含まれていなかったため

### Why #2: なぜテストケースに含まれていなかったか？
→ エッジケースの洗い出しが不十分だったため

### Why #3: なぜエッジケースの洗い出しが不十分だったか？
→ テスト設計のガイドラインがなかったため

### Why #4: なぜガイドラインがなかったか？
→ テスト設計の重要性が共有されていなかったため

### Why #5: なぜ重要性が共有されていなかったか？
→ テスト設計のトレーニングやレビュー文化がなかったため

## 根本原因
**テスト設計に関する知識共有とレビュー文化の欠如**

## 対策
- テスト設計ガイドライン作成
- テストコードレビューの実施
- テスト設計勉強会の開催
```

---

## Fishbone Diagram（特性要因図）

### 概要

**魚の骨のような図で、原因をカテゴリ別に整理する**

```
                           問題
                             │
        ┌────────────┬────────┼────────┬────────────┐
        │            │        │        │            │
      人材        方法      設備    環境         材料
   (People)    (Method)  (Machine) (Environment) (Material)
```

### 実践例: API レスポンス遅延

**問題**: API のレスポンスタイムが 3秒以上（目標: 300ms 以内）

```markdown
## Fishbone Diagram

### 問題
API レスポンスタイムが 3秒以上

### 原因分析

#### 1. People（人材）
- [ ] バックエンドチームの経験不足
- [x] **パフォーマンスチューニングの知識不足**
- [ ] レビュー時にパフォーマンスを考慮していない

#### 2. Method（方法）
- [x] **N+1 クエリ問題**
- [x] **インデックスが設定されていない**
- [ ] キャッシュ戦略がない
- [x] **不要なデータまで取得している**

#### 3. Machine（設備）
- [ ] サーバースペック不足
- [ ] ネットワーク帯域幅不足
- [x] **データベース接続プール設定が不適切**

#### 4. Environment（環境）
- [ ] 本番環境の負荷が想定以上
- [x] **ステージング環境で負荷テストしていない**
- [ ] 監視が不十分

#### 5. Material（材料・データ）
- [x] **データベーステーブルのサイズが大きい（100万レコード）**
- [ ] データの正規化が不十分
- [x] **古いデータが削除されていない**

## 主要な根本原因（[x] の項目）

1. **N+1 クエリ問題**
   - ユーザー一覧取得時に、各ユーザーごとに投稿を個別取得
2. **インデックスがない**
   - `users.email`、`posts.user_id` にインデックスなし
3. **不要なデータまで取得**
   - `SELECT *` で全カラム取得、必要なのは一部のみ
4. **データベーステーブルが肥大化**
   - 削除されたユーザーのデータも残っている
5. **負荷テスト不足**
   - 本番相当のデータ量でテストしていない

## 対策

### 即時対応
```sql
-- インデックス追加
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_posts_user_id ON posts(user_id);

-- 古いデータ削除
DELETE FROM users WHERE deleted_at < NOW() - INTERVAL 1 YEAR;
```

```javascript
// N+1 クエリ解消
// Before
const users = await User.findAll();
for (const user of users) {
  user.posts = await Post.findAll({ where: { userId: user.id } });
}

// After
const users = await User.findAll({
  include: [{ model: Post }]  // JOIN で一括取得
});

// 必要なカラムのみ取得
const users = await User.findAll({
  attributes: ['id', 'name', 'email'],  // SELECT id, name, email のみ
  include: [{
    model: Post,
    attributes: ['id', 'title']
  }]
});
```

### 根本対策
- [x] データベース設計レビューを必須化
- [x] パフォーマンステストを CI/CD に統合
- [ ] 定期的なインデックス最適化
- [ ] データアーカイブ戦略の策定
```

---

## Fault Tree Analysis（故障の木解析）

### 概要

**トップダウンで障害の要因を論理的に分解**

```
              障害（トップイベント）
                      │
              ┌───────┴───────┐
            OR/AND           OR/AND
              │               │
        ┌─────┴─────┐   ┌─────┴─────┐
      要因1        要因2  要因3      要因4
```

### 実践例: デプロイ失敗

**問題**: 本番環境へのデプロイが失敗する

```markdown
## Fault Tree Analysis

### トップイベント
**本番環境へのデプロイ失敗**

### レベル1（OR: いずれかが発生すれば失敗）
1. ビルド失敗
2. テスト失敗
3. デプロイスクリプトエラー
4. 環境設定エラー

### レベル2（各要因の詳細）

#### 1. ビルド失敗（OR）
- [ ] 依存関係の解決失敗
- [x] **型エラー**
- [ ] メモリ不足

##### 1.1 型エラー（AND: 両方が必要）
- [x] **TypeScript の strict モードが有効**
- [x] **型定義が不完全**

#### 2. テスト失敗（OR）
- [ ] ユニットテスト失敗
- [x] **統合テスト失敗**
- [ ] E2E テスト失敗

##### 2.1 統合テスト失敗（AND）
- [x] **API エンドポイントの変更**
- [x] **テストコードの更新漏れ**

#### 3. デプロイスクリプトエラー（OR）
- [x] **権限エラー**
- [ ] ネットワークエラー
- [ ] タイムアウト

##### 3.1 権限エラー（AND）
- [x] **デプロイ用 SSH キーの有効期限切れ**
- [ ] IAM ロールの変更

#### 4. 環境設定エラー（OR）
- [x] **環境変数未設定**
- [ ] データベース接続失敗
- [ ] 外部サービス連携失敗

##### 4.1 環境変数未設定（AND）
- [x] **新しい環境変数が追加された**
- [x] **.env ファイルが更新されていない**

## 根本原因の特定（[x] の項目）

### 優先度1: 型エラー
**原因**:
- TypeScript の `strict: true` が有効
- API レスポンスの型定義が不完全

**対策**:
```typescript
// Before（型定義不完全）
interface User {
  id: number;
  name: string;
  // email が抜けている
}

// After（完全な型定義）
interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
}

// API レスポンスの型チェック
const response = await fetch('/api/users');
const users: User[] = await response.json();
```

### 優先度2: 統合テスト失敗
**原因**:
- API エンドポイントを `/users` → `/api/v2/users` に変更
- テストコードが古いエンドポイントを参照

**対策**:
```javascript
// Before
test('fetch users', async () => {
  const response = await request(app).get('/users');  // 古い
  expect(response.status).toBe(200);
});

// After
test('fetch users', async () => {
  const response = await request(app).get('/api/v2/users');  // 新しい
  expect(response.status).toBe(200);
});
```

### 優先度3: デプロイ権限エラー
**原因**:
- SSH キーの有効期限が切れていた

**対策**:
```bash
# 新しいキーを生成
ssh-keygen -t ed25519 -C "deploy@example.com"

# GitHub Actions の Secrets を更新
gh secret set DEPLOY_SSH_KEY < ~/.ssh/id_ed25519

# 有効期限アラートを設定
echo "SSH キーの有効期限: $(date -d '+1 year')" >> REMINDERS.md
```

### 優先度4: 環境変数未設定
**原因**:
- 新機能で `STRIPE_API_KEY` が必要になったが、`.env` に追加し忘れ

**対策**:
```bash
# .env.example を更新
echo "STRIPE_API_KEY=sk_test_..." >> .env.example

# 環境変数チェックスクリプト
# scripts/check-env.sh
#!/bin/bash

REQUIRED_VARS=("DATABASE_URL" "JWT_SECRET" "STRIPE_API_KEY")

for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set"
    exit 1
  fi
done

echo "All required environment variables are set"
```

## 予防策

### プロセス改善
- [x] デプロイ前チェックリスト作成
- [x] CI/CD に型チェック・テスト・環境変数チェックを統合
- [ ] 定期的な SSH キー更新アラート

### 自動化
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # 1. 型チェック
      - name: TypeScript Type Check
        run: npm run type-check

      # 2. テスト
      - name: Run Tests
        run: npm test

      # 3. 環境変数チェック
      - name: Check Environment Variables
        run: ./scripts/check-env.sh

      # 4. ビルド
      - name: Build
        run: npm run build

      # 5. デプロイ
      - name: Deploy
        run: ./scripts/deploy.sh
```
```

---

## Timeline Analysis（時系列分析）

### 概要

**時系列でイベントを整理し、因果関係を明確にする**

### 実践例: 本番環境での障害

**問題**: ユーザーがログインできなくなった

```markdown
## Timeline Analysis

### 障害タイムライン

| 時刻 | イベント | 詳細 | 影響 |
|------|---------|------|------|
| 14:00 | デプロイ開始 | v3.2.0 を本番環境にデプロイ | なし |
| 14:05 | デプロイ完了 | ヘルスチェック OK | なし |
| 14:07 | 監視アラート発火 | ログインエラー率 5% | 軽微 |
| 14:10 | エラー率上昇 | ログインエラー率 50% | **重大** |
| 14:12 | 緊急対策チーム招集 | Slack でアラート | - |
| 14:15 | ログ調査開始 | "Invalid token" エラー大量発生 | **全ユーザー影響** |
| 14:20 | 原因特定 | `JWT_PUBLIC_KEY` 環境変数未設定 | - |
| 14:22 | 環境変数設定 | `export JWT_PUBLIC_KEY="..."` | - |
| 14:25 | サービス再起動 | PM2 restart | - |
| 14:27 | 復旧確認 | エラー率 0% に低下 | **復旧** |
| 14:30 | ポストモーテム開始 | RCA セッション | - |

### 時系列チャート

```
14:00 ─┬─ デプロイ開始
       │
14:05 ─┼─ デプロイ完了 ✅
       │
14:07 ─┼─ アラート発火 ⚠️（エラー率 5%）
       │    ↓
       │    ├─ なぜ即座に気づかなかった？
       │    └─ → 閾値が 10% に設定されていた
       │
14:10 ─┼─ エラー率急上昇 🚨（50%）
       │    ↓
       │    ├─ なぜ急激に悪化した？
       │    └─ → ユーザーのトークンが次々と期限切れ
       │
14:12 ─┼─ チーム招集
       │
14:15 ─┼─ ログ調査
       │    ↓
       │    ├─ "Invalid token" エラー大量発生
       │    └─ → JWT 検証が失敗
       │
14:20 ─┼─ 原因特定 🔍
       │    ↓
       │    ├─ `JWT_PUBLIC_KEY` 未設定
       │    └─ → 環境変数チェック漏れ
       │
14:22 ─┼─ 環境変数設定
       │
14:25 ─┼─ サービス再起動
       │
14:27 ─┼─ 復旧 ✅
       │
14:30 ─┴─ ポストモーテム
```

### 根本原因

**直接原因**:
- `JWT_PUBLIC_KEY` 環境変数が未設定

**間接原因**:
- デプロイ時に環境変数チェックがなかった
- ステージング環境と本番環境で設定が異なった

**根本原因**:
- **環境変数管理プロセスの欠如**
- **デプロイ前チェックリストの不備**

### タイムライン分析から得られる洞察

#### 1. 検知の遅れ（14:05 → 14:07）
**問題**:
- デプロイ完了後 2分間、エラーに気づかなかった

**原因**:
- ヘルスチェックが `/health` エンドポイントのみ
- ログイン機能はチェック対象外

**対策**:
```yaml
# デプロイ後のスモークテスト追加
smoke-test:
  steps:
    - name: Health Check
      run: curl https://api.example.com/health

    - name: Login Test
      run: |
        curl -X POST https://api.example.com/auth/login \
          -d '{"email": "test@example.com", "password": "test123"}' \
          | jq -e '.token'
```

#### 2. エラー率の急上昇（14:07 → 14:10）
**問題**:
- 3分間で 5% → 50% に急増

**原因**:
- ユーザーのトークンが次々と期限切れ
- 新規トークン発行が全て失敗

**対策**:
- エラー率閾値を 10% → 5% に引き下げ
- エラー率の変化率も監視（5分間で 2倍以上 → アラート）

#### 3. 原因特定まで 15分（14:07 → 14:20）
**問題**:
- ログ調査に時間がかかった

**原因**:
- エラーメッセージが不明瞭（"Invalid token"）
- どの環境変数が原因か特定が困難

**対策**:
```javascript
// 改善後のエラーメッセージ
if (!process.env.JWT_PUBLIC_KEY) {
  throw new Error('JWT_PUBLIC_KEY environment variable is not set');
}

try {
  jwt.verify(token, publicKey, { algorithms: ['RS256'] });
} catch (error) {
  logger.error('JWT verification failed', {
    error: error.message,
    tokenPreview: token.substring(0, 20) + '...',
    algorithm: 'RS256',
    publicKeySet: !!process.env.JWT_PUBLIC_KEY
  });
  throw error;
}
```

## まとめ

### 主要な根本原因
1. **環境変数管理プロセスの欠如**
2. **デプロイ前チェックリストの不備**
3. **ステージング環境と本番環境の差異**
4. **監視・アラートの閾値設定不適切**

### 改善策
- [x] 環境変数チェックスクリプト追加
- [x] デプロイ前スモークテスト追加
- [x] エラーメッセージの改善
- [ ] ステージング環境を本番と完全一致させる
- [ ] 監視閾値の見直し
```

---

## 実践例

### ケース1: メモリリーク

**問題**: Node.js サーバーが定期的にクラッシュする

```markdown
## RCA: Node.js サーバークラッシュ

### 5 Whys

#### Why #1: なぜサーバーがクラッシュするか？
→ メモリ不足（OOM: Out of Memory）

#### Why #2: なぜメモリ不足になるか？
→ メモリ使用量が時間経過とともに増加（メモリリーク）

#### Why #3: なぜメモリリークが発生するか？
→ イベントリスナーが削除されずに蓄積

#### Why #4: なぜイベントリスナーが削除されないか？
→ WebSocket 接続終了時に `removeListener` が呼ばれていない

#### Why #5: なぜ `removeListener` が呼ばれないか？
→ エラーハンドリングが不十分で、正常終了フローのみ考慮

### 根本原因
**エラーケースを考慮していないリソース管理**

### 問題のコード
```javascript
// Before（メモリリーク）
class WebSocketManager {
  constructor() {
    this.connections = new Map();
  }

  handleConnection(ws) {
    const connectionId = generateId();

    const onMessage = (data) => {
      this.processMessage(data);
    };

    const onClose = () => {
      this.connections.delete(connectionId);
      // ❌ リスナー削除を忘れている
    };

    ws.on('message', onMessage);
    ws.on('close', onClose);

    this.connections.set(connectionId, ws);
  }

  processMessage(data) {
    // メッセージ処理
  }
}
```

### 修正後のコード
```javascript
// After（メモリリーク解消）
class WebSocketManager {
  constructor() {
    this.connections = new Map();
  }

  handleConnection(ws) {
    const connectionId = generateId();

    const onMessage = (data) => {
      this.processMessage(data);
    };

    const onClose = () => {
      // ✅ リスナーを明示的に削除
      ws.removeListener('message', onMessage);
      ws.removeListener('error', onError);
      this.connections.delete(connectionId);
    };

    const onError = (error) => {
      console.error('WebSocket error:', error);
      // ✅ エラー時もクリーンアップ
      onClose();
    };

    ws.on('message', onMessage);
    ws.on('close', onClose);
    ws.on('error', onError);

    this.connections.set(connectionId, ws);
  }

  processMessage(data) {
    // メッセージ処理
  }

  // ✅ 明示的なクリーンアップメソッド
  shutdown() {
    for (const [id, ws] of this.connections) {
      ws.close();
    }
    this.connections.clear();
  }
}
```

### 検証
```javascript
// メモリリークテスト
const manager = new WebSocketManager();

// 1000接続を開いて閉じる
for (let i = 0; i < 1000; i++) {
  const ws = new WebSocket('ws://localhost:3000');
  manager.handleConnection(ws);
  ws.close();
}

// メモリ使用量を確認
const used = process.memoryUsage();
console.log(`Memory usage: ${(used.heapUsed / 1024 / 1024).toFixed(2)} MB`);

// Before: 約 500MB（リスナーが蓄積）
// After:  約 50MB（正常にクリーンアップ）
```

### 予防策
- [x] リソース管理のコードレビューチェックリスト作成
- [x] メモリリークテストを CI/CD に追加
- [x] 定期的なメモリプロファイリング
```

### ケース2: パフォーマンス劣化

**問題**: API レスポンスが徐々に遅くなる

```markdown
## RCA: API レスポンス劣化

### Fishbone Diagram

#### People（人材）
- [ ] パフォーマンスチューニング経験不足
- [x] **データベース設計の知識不足**

#### Method（方法）
- [x] **N+1 クエリ**
- [x] **フルテーブルスキャン**
- [ ] キャッシュ戦略なし

#### Machine（設備）
- [ ] サーバースペック不足
- [x] **データベース接続プール枯渇**

#### Environment（環境）
- [x] **本番データ量が想定の10倍**
- [ ] ネットワーク遅延

#### Material（データ）
- [x] **データベースインデックス欠如**
- [x] **古いデータ削除されていない**

### 根本原因

1. **N+1 クエリ問題**
2. **データベースインデックス欠如**
3. **データ増加への対策不足**

### 問題のコード
```javascript
// Before: N+1 クエリ（遅い）
async function getUsersWithPosts() {
  const users = await db.query('SELECT * FROM users');

  for (const user of users) {
    // ❌ ループ内でクエリ発行（N+1）
    user.posts = await db.query(
      'SELECT * FROM posts WHERE user_id = ?',
      [user.id]
    );
  }

  return users;
}

// 100ユーザーの場合:
// - users 取得: 1クエリ
// - posts 取得: 100クエリ
// → 合計 101クエリ 🐌
```

### 修正後のコード
```javascript
// After: JOIN で一括取得（速い）
async function getUsersWithPosts() {
  const query = `
    SELECT
      users.id, users.name, users.email,
      posts.id AS post_id, posts.title, posts.content
    FROM users
    LEFT JOIN posts ON posts.user_id = users.id
    WHERE users.deleted_at IS NULL
    ORDER BY users.id, posts.created_at DESC
  `;

  const rows = await db.query(query);

  // グルーピング処理
  const usersMap = new Map();

  for (const row of rows) {
    if (!usersMap.has(row.id)) {
      usersMap.set(row.id, {
        id: row.id,
        name: row.name,
        email: row.email,
        posts: []
      });
    }

    if (row.post_id) {
      usersMap.get(row.id).posts.push({
        id: row.post_id,
        title: row.title,
        content: row.content
      });
    }
  }

  return Array.from(usersMap.values());
}

// 100ユーザーの場合:
// → 1クエリのみ ⚡
```

### インデックス追加
```sql
-- Before: インデックスなし → フルテーブルスキャン
EXPLAIN SELECT * FROM posts WHERE user_id = 123;
-- type: ALL, rows: 1000000 (全行スキャン) 🐌

-- After: インデックス追加
CREATE INDEX idx_posts_user_id ON posts(user_id);

EXPLAIN SELECT * FROM posts WHERE user_id = 123;
-- type: ref, rows: 10 (インデックス使用) ⚡
```

### データアーカイブ
```sql
-- 古いデータを削除（論理削除）
UPDATE posts
SET deleted_at = NOW()
WHERE created_at < NOW() - INTERVAL 2 YEAR;

-- 物理削除（定期実行）
DELETE FROM posts
WHERE deleted_at < NOW() - INTERVAL 1 YEAR;
```

### 性能比較

| 対策 | Before | After | 改善率 |
|------|--------|-------|--------|
| N+1 解消 | 3.5秒 | 0.3秒 | **91.4%** |
| インデックス追加 | 2.1秒 | 0.1秒 | **95.2%** |
| データアーカイブ | 1.5秒 | 0.5秒 | **66.7%** |
| **総合** | **3.5秒** | **0.08秒** | **97.7%** |

### 予防策
- [x] クエリ実行計画の定期レビュー
- [x] スロークエリログ監視
- [x] データ増加を考慮した設計レビュー
- [ ] 定期的なデータアーカイブ処理
```

---

## RCA テンプレート

### テンプレート1: 基本版

```markdown
# Root Cause Analysis: [問題タイトル]

## 概要

**発生日時**: YYYY-MM-DD HH:MM:SS
**影響範囲**: [誰が、何が影響を受けたか]
**深刻度**: [CRITICAL / HIGH / MEDIUM / LOW]

## 問題の詳細

### 発生状況
[何が起きたか]

### 影響
- ユーザー数:
- ダウンタイム:
- 売上損失:

## 根本原因分析

### 5 Whys

#### Why #1: [質問]
→ [回答]

#### Why #2: [質問]
→ [回答]

#### Why #3: [質問]
→ [回答]

#### Why #4: [質問]
→ [回答]

#### Why #5: [質問]
→ [回答]

### 根本原因
[特定された根本原因]

## 対策

### 即時対策（完了済み）
- [x] [対策1]
- [x] [対策2]

### 短期対策（1週間以内）
- [ ] [対策1]
- [ ] [対策2]

### 長期対策（1ヶ月以内）
- [ ] [対策1]
- [ ] [対策2]

## 学んだ教訓

1. [教訓1]
2. [教訓2]
3. [教訓3]

## 担当者

- 調査: @
- 対応: @
- レビュー: @
```

### テンプレート2: 詳細版

```markdown
# Root Cause Analysis: [問題タイトル]

## Executive Summary

| 項目 | 内容 |
|------|------|
| **発生日時** | YYYY-MM-DD HH:MM:SS |
| **検知日時** | YYYY-MM-DD HH:MM:SS |
| **復旧日時** | YYYY-MM-DD HH:MM:SS |
| **MTTR** | XX分 |
| **影響範囲** | [範囲] |
| **影響ユーザー数** | XX,XXX 名 |
| **深刻度** | [CRITICAL / HIGH / MEDIUM / LOW] |

## タイムライン

| 時刻 | イベント | 担当者 | アクション |
|------|---------|--------|-----------|
| HH:MM | | | |
| HH:MM | | | |

## 問題の詳細

### 発生状況
[詳細な発生状況]

### 影響
- **ユーザー影響**:
- **ビジネス影響**:
- **技術的影響**:

### 検知方法
- [ ] 監視アラート
- [ ] ユーザー報告
- [ ] 定期確認
- [ ] その他:

## 根本原因分析

### 分析手法
- [ ] 5 Whys
- [ ] Fishbone Diagram
- [ ] Fault Tree Analysis
- [ ] Timeline Analysis

### 5 Whys Analysis

[5 Whys の結果]

### Fishbone Diagram

[Fishbone Diagram の結果]

### 根本原因の特定

**直接原因**:
[直接的な原因]

**間接原因**:
[間接的な原因]

**根本原因**:
[真の根本原因]

## 技術詳細

### 問題のコード
```[language]
[コード例]
```

### 修正後のコード
```[language]
[修正後のコード]
```

### 検証方法
```[language]
[検証コード]
```

## 対策

### 即時対策（完了済み）
- [x] [対策1] - [担当者] - [完了日時]
- [x] [対策2] - [担当者] - [完了日時]

### 短期対策（1週間以内）
- [ ] [対策1] - [担当者] - [期限]
- [ ] [対策2] - [担当者] - [期限]

### 長期対策（1ヶ月以内）
- [ ] [対策1] - [担当者] - [期限]
- [ ] [対策2] - [担当者] - [期限]

## メトリクス

| 指標 | Before | After | 改善率 |
|------|--------|-------|--------|
| | | | |

## 学んだ教訓

### 技術的な学び
1. [学び1]
2. [学び2]

### プロセスの学び
1. [学び1]
2. [学び2]

### チームの学び
1. [学び1]
2. [学び2]

## 参考情報

- **関連インシデント**: [リンク]
- **関連 PR**: [リンク]
- **関連ドキュメント**: [リンク]
- **参考資料**: [リンク]

## レビュー

| レビュアー | 承認日 | コメント |
|-----------|--------|----------|
| @tech-lead | YYYY-MM-DD | |
| @sre-lead | YYYY-MM-DD | |
| @manager | YYYY-MM-DD | |

---

**作成者**: @
**作成日**: YYYY-MM-DD
**最終更新**: YYYY-MM-DD
```

---

## まとめ

### RCA の重要ポイント

1. **人を責めない**: システム・プロセスの改善に焦点
2. **データに基づく**: ログ、メトリクス、証拠を重視
3. **深掘りする**: 表面的な症状ではなく真の原因を特定
4. **具体的な対策**: 実行可能で測定可能な改善策
5. **学びを共有**: チーム全体の知識として蓄積

### RCA 手法の使い分け

| 手法 | 適している場合 | 例 |
|------|--------------|-----|
| **5 Whys** | シンプルな問題、単一の原因系統 | 設定ミス、手順漏れ |
| **Fishbone** | 複数の要因が絡む問題 | パフォーマンス劣化、品質問題 |
| **Fault Tree** | 論理的な要因分解が必要 | デプロイ失敗、システム障害 |
| **Timeline** | 時系列の因果関係が重要 | 本番障害、連鎖的な問題 |

### チェックリスト

**RCA 実施時**:
- [ ] 問題の事実を正確に記録
- [ ] タイムラインを整理
- [ ] 複数の分析手法を試す
- [ ] 根本原因を特定（表面的な症状で終わらない）
- [ ] 具体的な対策を立案
- [ ] 対策の優先順位付け
- [ ] 責任者・期限を明確化
- [ ] チームでレビュー

---

## 次のステップ

1. **03-prevention-strategy.md**: 再発防止と継続的改善
2. **lessons-learned スキル**: 教訓の体系化と共有

**関連スキル**:
- `testing-strategy`: テストで早期発見
- `code-review`: レビューで予防
- `quality-assurance`: 品質保証プロセス

---

*根本原因を見つけることは、同じ問題を繰り返さないための第一歩です。*
