# 📚 Knowledge Base - 教訓データベース構築ガイド

> **目的**: インシデントから抽出された教訓・ベストプラクティスを体系的に管理し、検索可能で再利用しやすいナレッジベースを構築する

## 📚 目次

1. [ナレッジベースの価値](#ナレッジベースの価値)
2. [教訓の種類と分類](#教訓の種類と分類)
3. [ディレクトリ構造](#ディレクトリ構造)
4. [教訓の記録フォーマット](#教訓の記録フォーマット)
5. [抽出プロセス](#抽出プロセス)
6. [検索性の向上](#検索性の向上)
7. [実践例](#実践例)

---

## ナレッジベースの価値

### インシデント記録 vs 教訓ナレッジ

```
┌─────────────────────────────────────────┐
│ incident-logger (生のインシデント記録)    │
├─────────────────────────────────────────┤
│ ・時系列の事実記録                        │
│ ・個別事例の詳細                          │
│ ・特定の状況・環境                        │
│ ・What happened? (何が起きたか)          │
└─────────────────────────────────────────┘
                  ↓ 抽出・一般化
┌─────────────────────────────────────────┐
│ lessons-learned (教訓ナレッジベース)      │
├─────────────────────────────────────────┤
│ ・一般化されたベストプラクティス            │
│ ・再利用可能なパターン                    │
│ ・普遍的な原則                           │
│ ・What to do? (どうすべきか)             │
└─────────────────────────────────────────┘
```

### 具体例の比較

**インシデント記録**:
```markdown
# [HIGH] iOS 15でログイン後に画面が真っ白

## 発生日時
2025-12-30 10:15:00 JST

## 原因
iOS 15では `NavigationStack` が未対応

## 解決策
iOS 16+ と iOS 15 で分岐処理
```

**↓ 抽出・一般化**

**教訓ナレッジ**:
```markdown
# 最小サポートバージョンでの動作確認

## 教訓
新しいAPIを使用する際は、必ず最小サポートバージョンで動作確認する

## ベストプラクティス
- `#available` で分岐
- CI/CDに最小バージョンの実機テスト追加
- Preview の iOS バージョンを最小に設定

## 適用箇所
- iOS の新API使用時全般
- NavigationStack, SwiftUI Charts, etc.

## 根拠
- インシデント#002: iOS 15 で NavigationStack クラッシュ
- インシデント#015: iOS 14 で Charts API 未対応
```

### ナレッジベースのROI

**導入前（3ヶ月平均）**:
```
同じ問題の再発: 月5回
新規問題解決時間: 平均 4.2h
ベストプラクティス適用: 不明瞭
```

**導入後（3ヶ月平均）**:
```
同じ問題の再発: 月0.5回（90%削減）
新規問題解決時間: 平均 1.5h（64%短縮）
ベストプラクティス適用: 85%
```

**効果**:
- **月40時間の削減** = 開発者1人分の10%
- **品質向上**: 再発率90%削減
- **オンボーディング短縮**: 新メンバーが過去の知見を即座に活用

---

## 教訓の種類と分類

### 1. Best Practices（ベストプラクティス）

**定義**: 「こうすべき」という推奨される方法

**例**:
```markdown
# Optional を安全に扱う

## 教訓
force unwrap (!) は使わず、guard/if let/nil coalescing を使用

## 理由
- force unwrap は実行時クラッシュのリスク
- デバッグが困難（スタックトレースが不明瞭）

## ベストプラクティス
```swift
// ❌ Bad: force unwrap
let user = optionalUser!
print(user.name)

// ✅ Good: guard let
guard let user = optionalUser else {
    logger.error("User is nil")
    return
}
print(user.name)

// ✅ Good: if let
if let user = optionalUser {
    print(user.name)
}

// ✅ Good: nil coalescing
let userName = optionalUser?.name ?? "Guest"
```

## 適用箇所
- すべての Optional 処理
- 特にユーザー入力、API レスポンス、データベースクエリ

## 根拠
- インシデント#042: force unwrap で本番クラッシュ（1000ユーザー影響）
- インシデント#089: nil チェック漏れでデータ損失
```

### 2. Anti-Patterns（アンチパターン）

**定義**: 「避けるべき」悪いパターン

**例**:
```markdown
# Massive View Controller を避ける

## 教訓
ViewController に全てのロジックを詰め込まない

## 問題
- テストが困難
- 可読性が低い
- 再利用性がない
- 責務が不明確

## アンチパターン
```swift
// ❌ Bad: 1000行の ViewController
class UserViewController: UIViewController {
    // UI setup
    // API call
    // Data parsing
    // Business logic
    // Validation
    // Navigation
    // Error handling
    // Analytics
    // ...
}
```

## ベストプラクティス
```swift
// ✅ Good: MVVM で責務分離
class UserViewController: UIViewController {
    private let viewModel: UserViewModel

    override func viewDidLoad() {
        super.viewDidLoad()
        bindViewModel()
    }

    private func bindViewModel() {
        viewModel.user
            .sink { [weak self] user in
                self?.updateUI(user)
            }
    }
}

class UserViewModel {
    @Published var user: User?
    private let repository: UserRepository

    func loadUser() {
        // Business logic
    }
}
```

## 根拠
- インシデント#023: 1500行のVCでバグ修正に3日
- プロジェクトA: MVVM 導入後、テストカバレッジ 45% → 85%
```

### 3. Patterns（成功パターン）

**定義**: 実際に成功した設計・実装パターン

**例**:
```markdown
# Repository パターンでデータ層を抽象化

## 教訓
データソース（API、DB、キャッシュ）を Repository で抽象化

## メリット
- データソースの切り替えが容易
- テストが簡単（Mock Repository）
- ビジネスロジックとデータ層の分離

## パターン
```
View → ViewModel → UseCase → Repository → DataSource
                                   ↓
                              API / DB / Cache
```

## 実装例
```swift
// Protocol で抽象化
protocol UserRepository {
    func fetchUsers() async throws -> [User]
    func saveUser(_ user: User) async throws
}

// 本番実装
class UserRepositoryImpl: UserRepository {
    private let api: APIClient
    private let db: Database

    func fetchUsers() async throws -> [User] {
        // API から取得、DB にキャッシュ
        let users = try await api.fetchUsers()
        try await db.save(users)
        return users
    }
}

// テスト用 Mock
class MockUserRepository: UserRepository {
    var mockUsers: [User] = []

    func fetchUsers() async throws -> [User] {
        return mockUsers
    }
}
```

## いつ使うか
- 中規模以上のプロジェクト
- 複数のデータソース（API + DB）
- テストを重視する場合

## 成功事例
- プロジェクトA: テストカバレッジ 85%達成
- プロジェクトB: API切り替えが2時間で完了

## 根拠
- Clean Architecture 原則
- DDD (Domain-Driven Design)
```

### 4. Optimizations（最適化手法）

**定義**: パフォーマンス改善の実績ある手法

**例**:
```markdown
# LazyVGrid で大量データ表示を最適化

## 教訓
大量データの表示には Lazy コンポーネント + キャッシュ

## 問題
- 1000件の画像を一度に表示 → メモリ不足でクラッシュ
- スクロールがカクカク

## 解決策
```swift
// ❌ Bad: すべて一度にロード
ScrollView {
    VStack {
        ForEach(items) { item in  // 1000件
            AsyncImage(url: item.imageURL)
                .frame(width: 100, height: 100)
        }
    }
}
// メモリ使用量: 800MB 🔥

// ✅ Good: Lazy + キャッシュ
ScrollView {
    LazyVGrid(columns: columns, spacing: 10) {
        ForEach(items) { item in
            KFImage(URL(string: item.imageURL))
                .placeholder {
                    ProgressView()
                }
                .resizable()
                .frame(width: 100, height: 100)
        }
    }
}
// メモリ使用量: 150MB ✨
```

## 効果
- メモリ使用量: 800MB → 150MB（▼ 81%）
- スクロール: カクカク → スムーズ
- クラッシュ: なくなった

## いつ使うか
- リスト表示が100件以上
- 画像・動画などの重いコンテンツ
- スクロールパフォーマンスが重要

## ツール
- Kingfisher (画像キャッシュ)
- LazyVStack/LazyVGrid
- Instruments (メモリプロファイリング)

## 根拠
- インシデント#067: メモリクラッシュ
- プロジェクトB: ユーザー満足度 3.2 → 4.5
```

### 5. Tools（ツール・手法）

**定義**: 問題解決に有効なツール・手法

**例**:
```markdown
# Instruments でメモリリーク検出

## 教訓
メモリリークは Instruments の Leaks テンプレートで早期発見

## 使い方
1. Xcode メニュー: `Product` > `Profile`（Cmd+I）
2. `Leaks` テンプレートを選択
3. アプリを操作（問題の画面を開く・閉じるを繰り返す）
4. 赤いバー = リーク発見 🚨
5. Details で原因特定

## 成功事例
```
インシデント#067: メモリリーク問題
- 3日間悩んだ → Instruments で10分で発見
- 原因: WebSocket の removeListener 漏れ
```

## Tips
- **定期的に実行**: リリース前必須
- **自動化**: CI/CD に組み込む（fastlane）
- **ベースライン確保**: 正常時のメモリ使用量を記録

## 自動化例
```ruby
# fastlane/Fastfile
lane :profile_memory do
  scan(
    scheme: "MyApp",
    destination: "platform=iOS Simulator,name=iPhone 15 Pro"
  )

  instruments(
    template: "Leaks",
    app: "MyApp.app",
    devices: ["iPhone 15 Pro"]
  )
end
```

## いつ使うか
- リリース前のチェック
- メモリ使用量が増加している時
- クラッシュレポートで OOM (Out of Memory)

## 参考
- [Apple: Instruments User Guide](https://developer.apple.com/library/archive/documentation/DeveloperTools/Conceptual/InstrumentsUserGuide/)
```

---

## ディレクトリ構造

### 推奨構造

```
lessons/
├── index.md                     # 全教訓一覧（自動生成）
├── README.md                    # ナレッジベースの使い方
│
├── best-practices/              # ベストプラクティス
│   ├── optional-handling.md
│   ├── error-handling.md
│   ├── async-programming.md
│   ├── code-review.md
│   └── testing.md
│
├── anti-patterns/               # アンチパターン
│   ├── massive-view-controller.md
│   ├── force-unwrap.md
│   ├── god-object.md
│   └── copy-paste-code.md
│
├── patterns/                    # 成功パターン
│   ├── mvvm-repository.md
│   ├── coordinator.md
│   ├── dependency-injection.md
│   └── state-management.md
│
├── optimizations/               # 最適化手法
│   ├── lazy-loading.md
│   ├── caching-strategy.md
│   ├── image-optimization.md
│   └── database-indexing.md
│
├── tools/                       # ツール・手法
│   ├── instruments.md
│   ├── swiftlint.md
│   ├── charles-proxy.md
│   └── fastlane.md
│
├── by-category/                 # カテゴリ別（クロスリファレンス）
│   ├── architecture/
│   │   ├── mvvm.md → ../../patterns/mvvm-repository.md
│   │   └── coordinator.md → ../../patterns/coordinator.md
│   ├── testing/
│   │   ├── unit-testing.md
│   │   └── ui-testing.md
│   ├── performance/
│   │   ├── lazy-loading.md → ../../optimizations/lazy-loading.md
│   │   └── caching.md → ../../optimizations/caching-strategy.md
│   ├── security/
│   │   ├── keychain.md
│   │   └── api-security.md
│   └── deployment/
│       ├── ci-cd.md
│       └── app-store.md
│
├── templates/                   # テンプレート
│   ├── lesson-template.md
│   ├── monthly-review.md
│   └── quarterly-review.md
│
└── scripts/                     # 自動化スクリプト
    ├── add-lesson.sh
    ├── search-lessons.sh
    ├── update-index.sh
    └── lesson-stats.sh
```

### index.md の自動生成

**スクリプト** (`scripts/update-index.sh`):
```bash
#!/bin/bash

OUTPUT_FILE="lessons/index.md"

cat > "$OUTPUT_FILE" <<'EOF'
# 📚 Lessons Learned Index

> 自動生成されたファイルです。`./scripts/update-index.sh` で更新。

## 統計

EOF

# 統計情報
TOTAL=$(find lessons -name "*.md" -not -name "index.md" -not -name "README.md" | wc -l)
BP=$(find lessons/best-practices -name "*.md" | wc -l)
AP=$(find lessons/anti-patterns -name "*.md" | wc -l)
PT=$(find lessons/patterns -name "*.md" | wc -l)
OP=$(find lessons/optimizations -name "*.md" | wc -l)
TL=$(find lessons/tools -name "*.md" | wc -l)

cat >> "$OUTPUT_FILE" <<EOF
| カテゴリ | 件数 |
|---------|------|
| **Best Practices** | $BP |
| **Anti-Patterns** | $AP |
| **Patterns** | $PT |
| **Optimizations** | $OP |
| **Tools** | $TL |
| **合計** | $TOTAL |

---

## Best Practices

EOF

# Best Practices のリスト生成
for file in lessons/best-practices/*.md; do
    if [ -f "$file" ]; then
        title=$(head -n 1 "$file" | sed 's/^# //')
        path=$(echo "$file" | sed 's|lessons/||')
        echo "- [$title]($path)" >> "$OUTPUT_FILE"
    fi
done

cat >> "$OUTPUT_FILE" <<EOF

---

## Anti-Patterns

EOF

# Anti-Patterns のリスト生成
for file in lessons/anti-patterns/*.md; do
    if [ -f "$file" ]; then
        title=$(head -n 1 "$file" | sed 's/^# //')
        path=$(echo "$file" | sed 's|lessons/||')
        echo "- [$title]($path)" >> "$OUTPUT_FILE"
    fi
done

# 以下同様に Patterns, Optimizations, Tools を追加...

echo "✅ index.md generated: $TOTAL lessons"
```

---

## 教訓の記録フォーマット

### テンプレート

**`templates/lesson-template.md`**:
```markdown
# [教訓のタイトル]

## 教訓（一文で）

[何を学んだか？どうすべきか？]

## カテゴリ

- [ ] Best Practice
- [ ] Anti-Pattern
- [ ] Pattern
- [ ] Optimization
- [ ] Tool

## 問題・背景

[なぜこの教訓が必要か？何が問題だったか？]

## ベストプラクティス / 解決策

```[language]
[コード例]
```

## 効果・メリット

- [効果1]
- [効果2]

## いつ使うか

- [ユースケース1]
- [ユースケース2]

## 注意点

- [注意点1]
- [注意点2]

## 根拠・事例

- インシデント#XXX: [概要]
- プロジェクトY: [成功事例]

## 関連

- [関連する教訓]
- [関連するSkill]

## タグ

`tag1`, `tag2`, `tag3`

## 更新履歴

- YYYY-MM-DD: 初版作成
- YYYY-MM-DD: [更新内容]
```

### 実際の記録例

**`lessons/best-practices/async-await.md`**:
```markdown
# async/await で非同期処理を簡潔に書く

## 教訓

Swift の async/await を使って非同期処理を同期処理のように書く

## カテゴリ

- [x] Best Practice
- [ ] Anti-Pattern
- [ ] Pattern
- [ ] Optimization
- [ ] Tool

## 問題・背景

Callback hell（コールバック地獄）や completion handler の入れ子で可読性が低下

```swift
// ❌ Bad: Callback hell
fetchUser(id: userId) { user in
    fetchPosts(userId: user.id) { posts in
        fetchComments(postId: posts[0].id) { comments in
            updateUI(user, posts, comments)
        }
    }
}
```

## ベストプラクティス

```swift
// ✅ Good: async/await
func loadUserData() async throws {
    let user = try await fetchUser(id: userId)
    let posts = try await fetchPosts(userId: user.id)
    let comments = try await fetchComments(postId: posts[0].id)
    updateUI(user, posts, comments)
}

// 使用例
Task {
    do {
        try await loadUserData()
    } catch {
        handleError(error)
    }
}
```

## 効果・メリット

- 可読性向上: 同期処理のように読める
- エラーハンドリングが簡潔: try-catch で統一
- デバッグが容易: スタックトレースが明確

## いつ使うか

- iOS 15+ / macOS 12+ をサポートする場合
- 非同期処理が複数連鎖する場合
- API 呼び出し、データベースアクセス

## 注意点

- `await` はメインスレッドをブロックしない（安全）
- UI 更新は `@MainActor` で
- キャンセル処理には `Task.isCancelled` を使用

```swift
@MainActor
func updateUI() async {
    let data = try await fetchData()
    self.label.text = data.text  // メインスレッドで実行
}
```

## 根拠・事例

- インシデント#034: Callback hell でバグ混入（デバッグに2日）
- プロジェクトC: async/await 導入後、非同期バグ 70%削減

## 関連

- [error-handling.md](error-handling.md)
- [async-programming.md](async-programming.md)
- ios-development/guides/async-programming.md

## タグ

`swift`, `async`, `concurrency`, `ios15+`

## 更新履歴

- 2025-01-15: 初版作成
- 2025-02-01: Task.isCancelled の注意点追加
```

---

## 抽出プロセス

### インシデントから教訓へのフロー

```
┌────────────────────────────────────┐
│ 1. インシデント発生                 │
│    incidents/2025/042-crash.md     │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ 2. インシデント記録（incident-logger）│
│    - 発生状況                       │
│    - 原因                          │
│    - 解決策                        │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ 3. 教訓抽出（lessons-learned）      │
│    問いかけ:                       │
│    - 何を学んだか？                 │
│    - 他に適用できるか？             │
│    - パターン化できるか？           │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ 4. 教訓記録                        │
│    lessons/best-practices/xxx.md   │
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ 5. Skillへフィードバック            │
│    各Skillのガイド・チェックリストに反映│
└────────────────────────────────────┘
              ↓
┌────────────────────────────────────┐
│ 6. インデックス更新                 │
│    lessons/index.md 自動生成       │
└────────────────────────────────────┘
```

### 具体例: force unwrap クラッシュ

**1. インシデント記録** (`incidents/2025/042-force-unwrap-crash.md`):
```markdown
# [HIGH] force unwrap でアプリクラッシュ

## 発生日時
2025-01-10 14:23

## 原因
API レスポンスの Optional フィールドを force unwrap

```swift
let userName = user.name!  // nil でクラッシュ
```

## 影響
- 1000ユーザーがクラッシュ
- App Store レビューで低評価増加

## 解決策
```swift
guard let userName = user.name else {
    return "Guest"
}
```
```

**2. 教訓抽出の問いかけ**:
```
Q1: 何を学んだか？
→ force unwrap は危険、guard/if let を使うべき

Q2: 他に適用できるか？
→ すべての Optional 処理に適用可能

Q3: パターン化できるか？
→ ベストプラクティスとして一般化可能
```

**3. 教訓記録** (`lessons/best-practices/optional-handling.md`):
```markdown
# Optional を安全に扱う

## 教訓
force unwrap (!) は使わず、guard/if let を使用

## ベストプラクティス
```swift
// ❌ Bad
let user = optionalUser!

// ✅ Good
guard let user = optionalUser else { return }
```

## 根拠
- インシデント#042: force unwrap でクラッシュ（1000ユーザー影響）
```

**4. Skillへフィードバック**:
```markdown
# ios-development/guides/best-practices.md に追加

## Optional Handling
force unwrap を避け、guard/if let を使用する。
詳細: lessons/best-practices/optional-handling.md

# ios-development/checklists/code-review.md に追加

- [ ] force unwrap (!) を使用していないか？
```

**5. インデックス更新**:
```bash
./scripts/update-index.sh
# lessons/index.md に自動追加
```

---

## 検索性の向上

### 1. タグシステム

**各教訓にタグを付与**:
```markdown
## タグ
`swift`, `optional`, `safety`, `crash-prevention`
```

**タグ検索スクリプト** (`scripts/search-lessons.sh`):
```bash
#!/bin/bash

TAG=$1

if [ -z "$TAG" ]; then
    echo "Usage: ./search-lessons.sh <tag>"
    exit 1
fi

echo "🔍 Searching lessons with tag: $TAG"
echo ""

grep -r "^\`.*$TAG.*\`" lessons --include="*.md" -l | while read file; do
    title=$(head -n 1 "$file" | sed 's/^# //')
    echo "- $title"
    echo "  $file"
    echo ""
done
```

**使用例**:
```bash
$ ./scripts/search-lessons.sh optional

🔍 Searching lessons with tag: optional

- Optional を安全に扱う
  lessons/best-practices/optional-handling.md

- Nil Coalescing でデフォルト値を提供
  lessons/best-practices/nil-coalescing.md
```

### 2. 全文検索

**検索スクリプト** (`scripts/search-content.sh`):
```bash
#!/bin/bash

KEYWORD=$1

echo "🔍 Searching for: $KEYWORD"
echo ""

grep -r "$KEYWORD" lessons --include="*.md" -n | while IFS=: read file line content; do
    title=$(head -n 1 "$file" | sed 's/^# //')
    echo "📄 $title"
    echo "   $file:$line"
    echo "   $content"
    echo ""
done
```

### 3. カテゴリ別インデックス

**`lessons/by-category/architecture/index.md`**:
```markdown
# Architecture 関連の教訓

## Patterns
- [MVVM Repository](../../patterns/mvvm-repository.md)
- [Coordinator Pattern](../../patterns/coordinator.md)
- [Dependency Injection](../../patterns/dependency-injection.md)

## Anti-Patterns
- [Massive View Controller](../../anti-patterns/massive-view-controller.md)
- [God Object](../../anti-patterns/god-object.md)

## Best Practices
- [SOLID Principles](../../best-practices/solid-principles.md)
```

### 4. README.md でのクイックアクセス

**`lessons/README.md`**:
```markdown
# Lessons Learned - 教訓ナレッジベース

## クイックアクセス

### よく参照される教訓 Top 10

1. [Optional Handling](best-practices/optional-handling.md) - 48回参照
2. [MVVM Repository](patterns/mvvm-repository.md) - 32回参照
3. [Instruments 使い方](tools/instruments.md) - 28回参照
4. [Lazy Loading](optimizations/lazy-loading.md) - 24回参照
5. [Async/Await](best-practices/async-await.md) - 22回参照
...

### 最近追加された教訓

- 2025-01-15: [SwiftUI PreferenceKey](best-practices/preference-key.md)
- 2025-01-10: [LazyVStack Performance](optimizations/lazy-vstack.md)
- 2025-01-05: [Keychain Security](best-practices/keychain.md)

### カテゴリ別

- [Architecture](by-category/architecture/)
- [Testing](by-category/testing/)
- [Performance](by-category/performance/)
- [Security](by-category/security/)
- [Deployment](by-category/deployment/)
```

---

## 実践例

### ケース1: API エラーハンドリング

**インシデント** (`incidents/2025/015-api-error.md`):
```markdown
# [MEDIUM] API エラー時にアプリがフリーズ

## 原因
try! を使用していたため、エラー時にクラッシュ

```swift
let data = try! await api.fetchData()  // エラーでクラッシュ
```

## 解決策
```swift
do {
    let data = try await api.fetchData()
} catch {
    showErrorAlert(error)
}
```
```

**↓ 教訓抽出**

**教訓** (`lessons/best-practices/error-handling.md`):
```markdown
# エラーハンドリングは必須

## 教訓
try! は使わず、do-catch で適切にエラーハンドリングする

## 問題・背景
try! を使うと、エラー発生時にアプリがクラッシュする

```swift
// ❌ Bad: try! でクラッシュ
let data = try! await api.fetchData()

// ❌ Bad: try? でエラーを無視
let data = try? await api.fetchData()  // nil になるだけ
```

## ベストプラクティス

```swift
// ✅ Good: do-catch で適切に処理
func loadData() async {
    do {
        let data = try await api.fetchData()
        updateUI(data)
    } catch APIError.unauthorized {
        showLoginScreen()
    } catch APIError.networkError {
        showRetryButton()
    } catch {
        showErrorAlert(error.localizedDescription)
    }
}
```

## カスタムエラー型

```swift
enum APIError: Error {
    case unauthorized
    case networkError
    case serverError(Int)
    case decodingError

    var message: String {
        switch self {
        case .unauthorized:
            return "ログインが必要です"
        case .networkError:
            return "ネットワークエラー。接続を確認してください"
        case .serverError(let code):
            return "サーバーエラー（\(code)）"
        case .decodingError:
            return "データ形式が不正です"
        }
    }
}
```

## 効果・メリット

- クラッシュ防止
- ユーザーへの適切なフィードバック
- デバッグが容易

## いつ使うか

- すべての API 呼び出し
- データベースアクセス
- ファイル I/O
- 外部ライブラリの使用

## 注意点

- try? は本当にエラーを無視して良い場合のみ使用
- エラーログは必ず記録（Analytics、Crashlytics）

```swift
} catch {
    logger.error("API fetch failed: \(error)")
    analytics.logError(error)
    throw error
}
```

## 根拠・事例

- インシデント#015: try! でクラッシュ
- インシデント#089: try? でエラー見逃し

## 関連

- [async-await.md](async-await.md)
- [logging.md](../tools/logging.md)

## タグ

`swift`, `error-handling`, `api`, `crash-prevention`

## 更新履歴

- 2025-01-15: 初版作成
```

**↓ Skillへフィードバック**

**`ios-development/checklists/code-review.md`**:
```markdown
## エラーハンドリング
- [ ] try! を使用していないか？（クラッシュリスク）
- [ ] try? の使用は適切か？（エラーを無視していないか）
- [ ] すべての throws 関数は do-catch されているか？
- [ ] カスタムエラー型を定義しているか？
- [ ] エラーログを記録しているか？

参考: lessons/best-practices/error-handling.md
```

### ケース2: SwiftUI パフォーマンス最適化

**インシデント** (`incidents/2025/023-slow-scroll.md`):
```markdown
# [MEDIUM] リストのスクロールが重い

## 原因
ForEach で1000件のデータを一度に描画

```swift
ScrollView {
    VStack {
        ForEach(items) { item in
            ItemView(item: item)  // 1000個
        }
    }
}
```

## 解決策
LazyVStack で必要なものだけ描画

```swift
ScrollView {
    LazyVStack {
        ForEach(items) { item in
            ItemView(item: item)
        }
    }
}
```
```

**↓ 教訓抽出**

**教訓** (`lessons/optimizations/lazy-loading.md`):
```markdown
# Lazy Loading で大量データ表示を最適化

## 教訓
大量データの表示には LazyVStack/LazyVGrid を使用

## 問題・背景

通常の VStack/HStack は、すべての子 View を即座に生成・描画する。
大量データの場合、メモリ消費とパフォーマンスが問題になる。

```swift
// ❌ Bad: すべて一度に描画
ScrollView {
    VStack {
        ForEach(0..<1000) { i in
            Text("Item \(i)")
        }
    }
}
// → 1000個の View を即座に生成
```

## ベストプラクティス

```swift
// ✅ Good: 必要なものだけ描画
ScrollView {
    LazyVStack {
        ForEach(0..<1000) { i in
            Text("Item \(i)")
        }
    }
}
// → 画面に表示される分だけ生成
```

## LazyVGrid での画像表示

```swift
ScrollView {
    LazyVGrid(columns: [
        GridItem(.flexible()),
        GridItem(.flexible()),
        GridItem(.flexible())
    ], spacing: 10) {
        ForEach(items) { item in
            AsyncImage(url: URL(string: item.imageURL)) { image in
                image
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } placeholder: {
                ProgressView()
            }
            .frame(width: 100, height: 100)
            .clipped()
        }
    }
    .padding()
}
```

## 効果・メリット

| 指標 | VStack | LazyVStack | 改善 |
|------|--------|-----------|------|
| 初期ロード時間 | 3.5秒 | 0.3秒 | ▼ 91% |
| メモリ使用量 | 400MB | 80MB | ▼ 80% |
| スクロール FPS | 20fps | 60fps | 3倍 |

## いつ使うか

- リスト表示が50件以上
- 画像・動画などの重いコンテンツ
- スクロールパフォーマンスが重要

## 注意点

### 1. onAppear の挙動が異なる

```swift
LazyVStack {
    ForEach(items) { item in
        ItemView(item: item)
            .onAppear {
                // 画面に表示された時に呼ばれる
                print("Appeared: \(item.id)")
            }
    }
}
```

### 2. View のサイズを事前に指定

```swift
// パフォーマンス向上
LazyVStack(spacing: 0) {
    ForEach(items) { item in
        ItemView(item: item)
            .frame(height: 100)  // 固定高さ
    }
}
```

## 根拠・事例

- インシデント#023: スクロールが重い → LazyVStack で解決
- インシデント#067: メモリクラッシュ → Lazy + キャッシュで解決
- プロジェクトB: ユーザー満足度 3.2 → 4.5

## 関連

- [caching-strategy.md](caching-strategy.md)
- [image-optimization.md](image-optimization.md)
- swiftui-patterns/guides/03-performance-best-practices.md

## タグ

`swiftui`, `performance`, `lazy`, `optimization`

## 更新履歴

- 2025-01-20: 初版作成
- 2025-01-25: LazyVGrid の例を追加
```

---

## まとめ

### ナレッジベース構築のポイント

1. **インシデントから学ぶ**: すべてのインシデントから教訓を抽出
2. **一般化する**: 特定の事例を普遍的な原則に昇華
3. **体系的に分類**: 探しやすく、再利用しやすい構造
4. **継続的に更新**: 陳腐化した教訓は削除・更新
5. **チームで活用**: 個人の学びをチームの資産に

### チェックリスト

**教訓記録時**:
- [ ] 一文で教訓を要約できているか
- [ ] コード例を含めているか
- [ ] 根拠（インシデント番号）を記載しているか
- [ ] いつ使うかを明記しているか
- [ ] タグを付与しているか

**ナレッジベース運用**:
- [ ] 週次でインシデントから教訓抽出
- [ ] 月次でindex.md更新
- [ ] 四半期で教訓の見直し（陳腐化チェック）
- [ ] 新メンバーのオンボーディングに活用

---

## 次のステップ

1. **02-team-learning.md**: チーム学習プロセスの実践
2. **03-continuous-improvement.md**: 継続的改善サイクル

**関連スキル**:
- `incident-logger`: 教訓の元となるインシデント記録
- すべてのSkills: 教訓を各Skillにフィードバック

---

*知識は共有することで価値が倍増します。チームの学びを資産に変えましょう。*
