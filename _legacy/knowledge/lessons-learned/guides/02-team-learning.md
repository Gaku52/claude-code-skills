# 🤝 Team Learning - チーム学習プロセスガイド

> **目的**: 個人の学びをチーム全体の知識に変換し、組織全体の継続的な成長を実現する

## 📚 目次

1. [チーム学習の重要性](#チーム学習の重要性)
2. [学習文化の醸成](#学習文化の醸成)
3. [定期的な学習活動](#定期的な学習活動)
4. [ナレッジ共有の仕組み](#ナレッジ共有の仕組み)
5. [オンボーディング活用](#オンボーディング活用)
6. [実践例](#実践例)

---

## チーム学習の重要性

### 個人学習 vs チーム学習

```
個人学習（個別最適）
├─ 個人の経験・失敗から学ぶ
├─ 知識が個人に閉じる
├─ 同じ失敗を他のメンバーも繰り返す
└─ 成長速度: 線形

チーム学習（全体最適）
├─ チーム全体の経験・失敗から学ぶ
├─ 知識がチームに蓄積される
├─ 失敗は一度だけ（学びは永続）
└─ 成長速度: 指数関数的
```

### チーム学習のROI

**導入前**:
```
エンジニアA: 問題Xで2時間悩む
エンジニアB: 同じ問題Xで2時間悩む（翌週）
エンジニアC: 同じ問題Xで2時間悩む（翌月）
→ 合計6時間の無駄
```

**導入後**:
```
エンジニアA: 問題Xで2時間悩む → 教訓化（30分）
エンジニアB: ナレッジベースで5分で解決 ✨
エンジニアC: ナレッジベースで5分で解決 ✨
→ 合計4.5時間削減（75%改善）
```

**年間効果**:
```
5人チーム × 週1回の学び × 2時間節約 = 年間520時間削減
= エンジニア1人分の約25%の工数削減
```

---

## 学習文化の醸成

### 1. 心理的安全性の確保

**失敗を学びに変える文化**:

```markdown
# チーム学習の基本原則

## 1. 人を責めない（Blame-Free Culture）

❌ 「誰がミスしたのか？」
✅ 「なぜシステムがミスを防げなかったのか？」

❌ 「なぜテストを書かなかったのか？」
✅ 「テストを書きやすい環境はどうすれば作れるか？」

## 2. 失敗を共有する（Fail Fast, Learn Faster）

❌ 問題を隠す、報告を遅らせる
✅ 即座に共有し、チーム全体で学ぶ

## 3. 学びを称賛する（Celebrate Learning）

❌ 完璧主義を求める
✅ 失敗からの学びと改善を評価する

毎週の「今週の学び賞」🏆で最も価値ある学びを共有したメンバーを称賛
```

### 2. オープンコミュニケーション

**失敗共有会（Weekly Failure Sharing）**:

```markdown
# 毎週金曜 30分

## ルール
1. 各メンバーが今週の失敗を1つ共有
2. 「どう失敗したか」より「何を学んだか」に焦点
3. 批判・非難は一切禁止
4. 質問・提案は歓迎

## フォーマット
### 失敗したこと
[何が起きたか]

### 学んだこと
[どう改善すべきか]

### 次のアクション
[具体的な予防策]

## 例
### 失敗したこと
- git pull せずにビルド → 古いコードで TestFlight 配布

### 学んだこと
- ビルド前の git pull を習慣化
- CI/CD で自動的に最新を取得すべき

### 次のアクション
- [ ] fastlane スクリプトに git pull 追加
- [ ] ビルド前チェックリスト作成
```

### 3. 学習時間の確保

**20%ルール**:
```markdown
# 学習時間の確保

## 金曜午後は「学習タイム」
- 14:00-17:00 は新技術学習・ドキュメント整備に充てる
- 緊急対応以外は会議を入れない
- 成果は週次レトロで共有

## 月次勉強会（必須参加）
- 毎月最終金曜 15:00-17:00
- 持ち回りで発表（1人30分）
- トピック例:
  - 今月のインシデント深掘り
  - 新技術・ツール紹介
  - 外部カンファレンスの知見共有
```

---

## 定期的な学習活動

### 1. 日次スタンドアップ

**学びの共有を組み込む**:

```markdown
# Daily Standup (15分)

## フォーマット
### 昨日やったこと
- [タスク1]
- [タスク2]

### 今日やること
- [タスク1]
- [タスク2]

### ブロッカー
- [問題1]

### 昨日の学び（NEW!）✨
- [1文で学んだこと]

## 例
### 昨日の学び
「async/await で非同期処理がシンプルに書けることを学んだ」
→ 興味ある人は後で詳細を聞く
→ 価値あるものは教訓化
```

### 2. 週次レトロスペクティブ

**KPT形式**:

```markdown
# Weekly Retrospective (45分)

## Keep（続けること）
- うまくいったこと
- 継続すべきプラクティス

## Problem（問題）
- うまくいかなかったこと
- 改善が必要なこと

## Try（試すこと）
- 次週試してみること
- 新しいアプローチ

## 学びの抽出
各Problemから教訓を抽出:
- この問題から何を学んだか？
- 他に適用できるか？
- パターン化できるか？

→ lessons/ に記録
```

**テンプレート**:
```markdown
# Weekly Retro - 2025 Week 3

## Keep
- ✅ ペアプログラミングでコード品質向上
- ✅ 毎朝のコードレビュー時間確保

## Problem
- ❌ デプロイ前のチェックリスト忘れ → 本番エラー
- ❌ テストカバレッジが低下（75% → 68%）

## Try
- [ ] デプロイスクリプトにチェックリスト自動確認を追加
- [ ] CI でカバレッジ閾値 80% 未満はブロック

## 教訓の抽出

### Problem 1 から
**教訓**: デプロイ前チェックリストを自動化
**記録先**: lessons/best-practices/deployment-checklist.md

### Problem 2 から
**教訓**: カバレッジ閾値を CI に組み込む
**記録先**: lessons/best-practices/test-coverage.md
```

### 3. 月次勉強会

**Deep Dive形式**:

```markdown
# Monthly Learning Session (2時間)

## アジェンダ

### Part 1: 今月のインシデント深掘り（60分）
- 重大インシデント 1-2件をピックアップ
- 担当者が詳細プレゼン
- Q&A、ディスカッション
- 教訓の抽出

### Part 2: 新技術・ツール紹介（30分）
- 持ち回りで担当者が発表
- トピック例:
  - SwiftUI の新機能
  - GitHub Actions の活用
  - 新しいテストフレームワーク

### Part 3: 知識共有（30分）
- 外部カンファレンス参加報告
- 書籍・記事の紹介
- 他チームのベストプラクティス
```

**発表テンプレート**:
```markdown
# [タイトル]

## 概要（1分）
何について話すか

## 背景・動機（3分）
なぜこれを学んだか

## 内容（20分）
詳細な説明、デモ、コード例

## 学んだこと（3分）
重要なポイント、教訓

## 質疑応答（3分）

## 次のアクション
チームでどう活用するか
```

### 4. 四半期レトロスペクティブ

**大きな振り返り**:

```markdown
# Quarterly Retrospective (3時間)

## アジェンダ

### 1. データ分析（30分）
- インシデント統計
- テストカバレッジ推移
- デプロイ頻度・成功率
- チームベロシティ

### 2. 主要イベントの振り返り（60分）
- 成功したこと
- 失敗したこと
- 予期しなかったこと

### 3. 教訓レビュー（30分）
- 今四半期の教訓一覧
- 適用状況の確認
- 陳腐化した教訓の削除

### 4. 次四半期の改善計画（60分）
- 優先度の高い課題
- 新しい取り組み
- 目標設定
```

**テンプレート**:
```markdown
# Q1 2025 Retrospective

## 📊 データ分析

### インシデント
| 月 | 総数 | Critical | High | Medium | Low |
|----|------|---------|------|--------|-----|
| Jan | 15 | 1 | 4 | 7 | 3 |
| Feb | 12 | 0 | 3 | 6 | 3 |
| Mar | 8 | 0 | 2 | 4 | 2 |
| **傾向** | ▼ 47% | ▼ 100% | ▼ 50% | ▼ 43% | ▼ 33% |

### テストカバレッジ
- Jan: 68%
- Feb: 75%
- Mar: 82%
- **傾向**: ▲ 14pt

## 🎉 成功したこと

1. **テストカバレッジ向上**
   - 目標: 80% → 達成: 82%
   - 施策: CI でカバレッジ閾値強制

2. **デプロイ成功率向上**
   - Before: 85% → After: 97%
   - 施策: 自動チェックリスト導入

## ❌ 失敗したこと

1. **新機能のリリース遅延**
   - 予定: 2月 → 実際: 3月
   - 原因: 要件の頻繁な変更

2. **ドキュメント整備不足**
   - API ドキュメントが古い
   - 新メンバーのオンボーディングに支障

## 📚 教訓レビュー

### 新規追加（今四半期）
- 15件の教訓を追加
- カテゴリ: Best Practices(6), Anti-Patterns(3), Patterns(4), Optimizations(2)

### よく参照された教訓 Top 5
1. optional-handling.md - 48回
2. mvvm-repository.md - 32回
3. instruments.md - 28回
4. lazy-loading.md - 24回
5. async-await.md - 22回

### 適用状況
- Optional handling: 95% ✅
- MVVM pattern: 100% ✅
- Instruments 定期実行: 60% ⚠️ 改善必要

### 陳腐化
- 2件削除（古い iOS バージョン関連）

## 🎯 次四半期の目標

### 品質
- [ ] インシデント数を50%削減（8件 → 4件/月）
- [ ] Critical インシデント ゼロ維持
- [ ] テストカバレッジ 85% 達成

### プロセス
- [ ] デプロイ頻度を週2回 → 毎日に
- [ ] ドキュメント整備（API、アーキテクチャ）
- [ ] 新メンバーオンボーディングを1週間 → 3日に短縮

### 学習
- [ ] 月次勉強会 100%実施
- [ ] 外部カンファレンス参加（2名）
- [ ] 教訓適用率 90% 達成
```

---

## ナレッジ共有の仕組み

### 1. Slack 統合

**自動通知**:

```javascript
// Slack Bot: 新しい教訓が追加されたら通知
const { App } = require('@slack/bolt');

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET
});

// GitHub Webhook から新しい教訓を検知
app.event('lesson_added', async ({ event, client }) => {
  await client.chat.postMessage({
    channel: '#engineering',
    text: '📚 新しい教訓が追加されました',
    blocks: [
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*${event.lesson.title}*\n${event.lesson.summary}`
        }
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `*カテゴリ*: ${event.lesson.category}\n*根拠*: ${event.lesson.source}`
        }
      },
      {
        type: 'actions',
        elements: [
          {
            type: 'button',
            text: { type: 'plain_text', text: '詳細を見る' },
            url: event.lesson.url
          }
        ]
      }
    ]
  });
});
```

**週次サマリー**:

```javascript
// 毎週金曜17:00に今週の教訓サマリーを投稿
cron.schedule('0 17 * * 5', async () => {
  const thisWeekLessons = await getLessonsThisWeek();

  await client.chat.postMessage({
    channel: '#engineering',
    text: '📊 今週の教訓サマリー',
    blocks: [
      {
        type: 'header',
        text: {
          type: 'plain_text',
          text: '📊 今週の教訓サマリー'
        }
      },
      {
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `今週は *${thisWeekLessons.length}件* の教訓が追加されました`
        }
      },
      ...thisWeekLessons.map(lesson => ({
        type: 'section',
        text: {
          type: 'mrkdwn',
          text: `• <${lesson.url}|${lesson.title}>`
        }
      }))
    ]
  });
});
```

### 2. Wiki / Notion 統合

**Notion データベース**:

```
Lessons Database
├─ Title (text)
├─ Category (select: BP, AP, Pattern, Optimization, Tool)
├─ Summary (text)
├─ Tags (multi-select)
├─ Source Incident (relation)
├─ Date Added (date)
├─ Contributors (person)
├─ Reference Count (number) ← 参照回数
└─ Status (select: Active, Archived)
```

**ビューの活用**:
- **Table View**: 全教訓一覧
- **Board View**: カテゴリ別
- **Gallery View**: カード形式
- **Calendar View**: 追加日時順
- **Most Referenced**: 参照回数順

### 3. Pull Request コメントでの活用

**GitHub Actions で自動コメント**:

```yaml
# .github/workflows/lesson-suggestion.yml
name: Suggest Relevant Lessons

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  suggest-lessons:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Analyze PR Changes
        id: analyze
        run: |
          # PR の変更内容から関連キーワード抽出
          git diff origin/main...HEAD > changes.diff
          keywords=$(python scripts/extract-keywords.py changes.diff)
          echo "keywords=$keywords" >> $GITHUB_OUTPUT

      - name: Search Related Lessons
        id: search
        run: |
          # キーワードに関連する教訓を検索
          lessons=$(./scripts/search-lessons.sh "${{ steps.analyze.outputs.keywords }}")
          echo "lessons=$lessons" >> $GITHUB_OUTPUT

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const lessons = `${{ steps.search.outputs.lessons }}`;
            if (lessons) {
              github.rest.issues.createComment({
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: `## 📚 関連する教訓\n\nこのPRに関連する教訓が見つかりました:\n\n${lessons}`
              });
            }
```

---

## オンボーディング活用

### 1. 新メンバー向けオンボーディングパス

**Week 1: ナレッジベースの理解**:

```markdown
# 新メンバーオンボーディング - Week 1

## Day 1: ナレッジベースの紹介
- [ ] `lessons/README.md` を読む
- [ ] よく参照される教訓 Top 10 を確認
- [ ] カテゴリ構造を理解

## Day 2: Best Practices
- [ ] `lessons/best-practices/` 全体を読む
- [ ] 特に重要:
  - optional-handling.md
  - error-handling.md
  - async-await.md
- [ ] コード例を手元で試す

## Day 3: Anti-Patterns
- [ ] `lessons/anti-patterns/` を読む
- [ ] 既存コードで該当箇所を探す
- [ ] なぜ問題なのか理解する

## Day 4: Patterns & Optimizations
- [ ] `lessons/patterns/` を読む
- [ ] プロジェクトのアーキテクチャと照合
- [ ] `lessons/optimizations/` で性能改善手法を学ぶ

## Day 5: Tools
- [ ] `lessons/tools/` を読む
- [ ] 各ツールを実際に使ってみる
- [ ] Instruments でメモリプロファイリング実践
```

### 2. チェックリスト

**新メンバー理解度チェック**:

```markdown
# オンボーディング理解度チェック

## ナレッジベース
- [ ] ナレッジベースの場所と構造を理解している
- [ ] 教訓の検索方法を知っている
- [ ] 新しい教訓の追加方法を知っている

## Best Practices
- [ ] Optional handling のベストプラクティスを説明できる
- [ ] Error handling の重要性を理解している
- [ ] async/await の使い方を知っている

## アーキテクチャ
- [ ] MVVM パターンを理解している
- [ ] Repository パターンの役割を説明できる
- [ ] Coordinator パターンのメリットを知っている

## ツール
- [ ] Instruments でメモリリークを検出できる
- [ ] SwiftLint の設定を理解している
- [ ] Charles Proxy でネットワークをデバッグできる

## プロセス
- [ ] コードレビューの観点を理解している
- [ ] デプロイ前チェックリストを知っている
- [ ] インシデント記録の方法を知っている
```

### 3. ペアプログラミング

**教訓の実践**:

```markdown
# ペアプログラミングセッション

## 目的
新メンバーが教訓を実践に活かす

## セッション構成（2時間）

### Part 1: タスク説明（10分）
- 実装するタスクの説明
- 関連する教訓の確認

### Part 2: 実装（80分）
- 新メンバーがドライバー
- 既存メンバーがナビゲーター
- 教訓を参照しながら実装
- リアルタイムでフィードバック

### Part 3: 振り返り（30分）
- 何を学んだか
- どの教訓が役立ったか
- 新たな気づき

## 例
### タスク
「ユーザー一覧画面の実装」

### 関連する教訓
- lessons/patterns/mvvm-repository.md
- lessons/optimizations/lazy-loading.md
- lessons/best-practices/error-handling.md

### 実装中の気づき
- LazyVStack で1000件表示がスムーズに
- Repository パターンでテストが書きやすい
- do-catch でエラーハンドリングが明確に

### 振り返り
新メンバー: 「教訓を見ながら実装したら、迷わず最初から正しく書けた」
```

---

## 実践例

### ケース1: 失敗共有会の実施

**毎週金曜 16:30-17:00**:

```markdown
# Weekly Failure Sharing - 2025-01-17

## 参加者
@alice, @bob, @charlie, @diana

---

## @alice の失敗

### 失敗したこと
force unwrap でクラッシュを本番環境に出してしまった

```swift
let user = optionalUser!  // nil でクラッシュ
```

### 学んだこと
- force unwrap は絶対に使わない
- guard let で安全に処理すべき

### 次のアクション
- [ ] SwiftLint で force unwrap を警告に
- [ ] コードレビューチェックリストに追加

### 教訓化
→ lessons/best-practices/optional-handling.md に追加 ✅

---

## @bob の失敗

### 失敗したこと
git pull せずにビルド → 古いコードで TestFlight 配布

### 学んだこと
- ビルド前に必ず git pull
- 手動は忘れる → 自動化すべき

### 次のアクション
- [ ] fastlane スクリプトに git pull 追加
- [ ] CI/CD で自動化

### 教訓化
→ lessons/best-practices/build-process.md に追加 ✅

---

## @charlie の失敗

### 失敗したこと
LazyVStack を使わず、1000件の画像を一度に表示 → メモリクラッシュ

### 学んだこと
- 大量データは Lazy コンポーネント必須
- Instruments で事前にメモリプロファイリング

### 次のアクション
- [ ] 既存の VStack を LazyVStack に置き換え
- [ ] デプロイ前に Instruments チェック

### 教訓化
→ lessons/optimizations/lazy-loading.md に追加 ✅

---

## @diana の失敗

### 失敗したこと
API エラーを try! で処理 → ネットワークエラー時にクラッシュ

### 学んだこと
- try! は使わない
- do-catch で適切にエラーハンドリング

### 次のアクション
- [ ] 全ての try! を do-catch に置き換え
- [ ] SwiftLint で try! を警告に

### 教訓化
→ lessons/best-practices/error-handling.md に追加 ✅

---

## 今週のまとめ

### 共通パターン
force unwrap / try! などの「!」の危険性

### チーム全体のアクション
- [ ] SwiftLint で force unwrap / try! を警告に設定
- [ ] コードレビューで「!」を重点チェック

### 今週の学び賞 🏆
@charlie - Lazy Loading の重要性を実践的に学んだ
```

### ケース2: 月次勉強会

**2025年1月度勉強会**:

```markdown
# Monthly Learning Session - 2025-01

## 📅 日時
2025-01-26 15:00-17:00

## 参加者
全エンジニア（8名）

---

## Part 1: インシデント深掘り（60分）

### インシデント: iOS 15 で画面が真っ白

**発表者**: @alice

#### 問題
iOS 15 で NavigationStack 未対応 → クラッシュ

#### 原因
最小サポートバージョンでのテスト不足

#### 解決策
```swift
if #available(iOS 16.0, *) {
    NavigationStack { ... }
} else {
    NavigationView { ... }
}
```

#### 学んだこと
- 新 API 使用時は必ず最小バージョンでテスト
- CI/CD に iOS 15 実機テスト追加
- Preview の iOS バージョンを最小に設定

#### Q&A
Q: CI での実機テストは時間がかかる？
A: 約5分増。critical なテストのみ iOS 15 で実行

#### 教訓化
→ lessons/best-practices/ios-version-support.md ✅

---

## Part 2: 新技術紹介（30分）

### SwiftUI の新機能: Observation Framework

**発表者**: @bob

#### 概要
iOS 17 の新しい Observation Framework で @Published が不要に

#### Before (iOS 16)
```swift
class ViewModel: ObservableObject {
    @Published var text: String = ""
}
```

#### After (iOS 17+)
```swift
@Observable
class ViewModel {
    var text: String = ""
}
```

#### メリット
- ボイラープレート削減
- パフォーマンス向上
- より Swift らしいコード

#### デモ
[実際のコードでデモ]

#### 導入提案
- iOS 17+ サポート時に順次移行
- 新規機能から試験的に導入

---

## Part 3: 知識共有（30分）

### カンファレンス参加報告

**発表者**: @charlie

#### iOSDC 2024 参加報告

##### セッション1: SwiftUI のパフォーマンス最適化
- Lazy コンポーネントの重要性（既知）
- .id() modifier で強制再描画回避
- onAppear の注意点

##### セッション2: Testing Best Practices
- Snapshot Testing の活用
- UI Test の高速化技術
- Mock の効果的な使い方

##### 持ち帰り
- [ ] Snapshot Testing を試験導入
- [ ] UI Test の実行時間を50%削減する施策

#### 書籍紹介: "iOS アプリ開発 実践ガイド"
- MVVM の詳細解説
- Repository パターンの実装例
- チームで輪読を提案

---

## アクションアイテム

| アクション | 担当 | 期限 |
|-----------|------|------|
| iOS 15 実機テスト CI 追加 | @alice | 2/2 |
| Observation Framework 試験導入 | @bob | 2/15 |
| Snapshot Testing 調査 | @charlie | 2/9 |
| 書籍輪読会の企画 | @diana | 2/1 |

---

## 次回予告（2025-02）

### 発表予定
- @diana: GitHub Actions の活用事例
- @eve: API 設計ベストプラクティス

### トピック募集
Slack #engineering で投票
```

---

## まとめ

### チーム学習成功の鍵

1. **心理的安全性**: 失敗を共有しやすい文化
2. **定期的な活動**: 日次・週次・月次の学習機会
3. **実践との結合**: 学びを即座に実践に活かす
4. **ツール活用**: Slack, Notion, GitHub で知識共有
5. **オンボーディング**: 新メンバーへの知識伝達

### チェックリスト

**チーム学習の実践**:
- [ ] 失敗共有会を週次で実施
- [ ] レトロスペクティブで教訓抽出
- [ ] 月次勉強会を開催
- [ ] 四半期で大きな振り返り
- [ ] Slack で新しい教訓を自動通知

**文化醸成**:
- [ ] 人を責めない文化を徹底
- [ ] 失敗を共有することを称賛
- [ ] 学習時間を確保（20%ルール）
- [ ] 新メンバーのオンボーディングに教訓を活用

---

## 次のステップ

1. **03-continuous-improvement.md**: 継続的改善サイクルの実践

**関連スキル**:
- `incident-logger`: インシデント記録から学ぶ
- すべてのSkills: 教訓を各Skillにフィードバック

---

*個人の学びをチーム全体の資産に。継続的な成長を実現しましょう。*
