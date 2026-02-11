# PRレビュー完全ガイド

> **対象読者**: レビュワー、PR作成者、チームリーダー
> **難易度**: 中級〜上級
> **推定読了時間**: 45分

---

## 📋 目次

1. [PRレビューとは](#prレビューとは)
2. [レビューの目的と価値](#レビューの目的と価値)
3. [レビュー前の準備](#レビュー前の準備)
4. [レビューの7つの観点](#レビューの7つの観点)
5. [効果的なレビューコメント](#効果的なレビューコメント)
6. [レビュープロセス](#レビュープロセス)
7. [セルフレビューの実践](#セルフレビューの実践)
8. [チームレビュー文化の構築](#チームレビュー文化の構築)
9. [レビュー効率化テクニック](#レビュー効率化テクニック)
10. [よくある問題と対策](#よくある問題と対策)
11. [実践例](#実践例)
12. [FAQ](#faq)

---

## PRレビューとは

### 定義

**Pull Request Review（PRレビュー）** は、コード変更をメインブランチにマージする前に、他の開発者が品質・設計・バグを確認するプロセスです。

### 基本原則

```
✅ 全てのPRはレビュー必須
✅ レビューなしでマージしない
✅ 建設的なフィードバック
✅ 学習の機会として活用
✅ チーム全体の品質向上
```

---

## レビューの目的と価値

### 1. バグの早期発見

```
コードが本番環境に到達する前に問題を発見
→ 修正コストを最小化
```

**効果**:
- 本番バグの80%以上を防止
- デバッグ時間の大幅削減
- ユーザー影響の最小化

### 2. コード品質の向上

```
複数の目で確認することで品質を担保
→ 可読性・保守性の向上
```

**効果**:
- より良い設計パターンの適用
- 一貫性のあるコードベース
- 技術的負債の削減

### 3. 知識の共有

```
コードレビューは最高の学習機会
→ チーム全体のスキル向上
```

**効果**:
- コードベースの理解促進
- ベストプラクティスの普及
- 新メンバーの早期戦力化

### 4. アーキテクチャの維持

```
設計方針に沿っているか確認
→ 長期的な保守性確保
```

**効果**:
- 一貫したアーキテクチャ
- スパゲッティコードの防止
- リファクタリングコストの削減

---

## レビュー前の準備

### レビュワーの準備

#### 1. コンテキストの理解

```markdown
✅ 関連Issueを読む
✅ PR説明を熟読
✅ 何を解決しようとしているか把握
✅ なぜこの変更が必要か理解
```

**実践例**:

```bash
# 1. PRページを開く
# 2. Issue番号をクリック（例: Closes #123）
# 3. 背景・要件を確認
# 4. PR説明の「なぜ」を理解
```

#### 2. 環境の準備

```bash
# ローカルでブランチをチェックアウト
git fetch origin
git checkout feature/PROJ-123-add-login

# 依存関係のインストール
npm install  # または pod install, など

# ビルド確認
npm run build

# テスト実行
npm test
```

#### 3. 変更規模の確認

| 変更規模 | 行数目安 | レビュー時間 | 分割推奨 |
|---------|---------|------------|---------|
| Small | 〜100行 | 15分 | 不要 |
| Medium | 100〜400行 | 30分 | 任意 |
| Large | 400〜1000行 | 60分 | 推奨 |
| XLarge | 1000行〜 | 2時間+ | **必須** |

**大きすぎるPRの場合**:

```markdown
[推奨] PRの分割

このPRは1500行の変更があり、レビューが困難です。
以下のように分割することを推奨します：

1. PR1: モデル層の変更
2. PR2: ViewModel層の変更
3. PR3: View層の変更

それぞれを順次マージすることで、
レビュー品質を向上できます。
```

---

## レビューの7つの観点

### 1. 機能性（Functionality）

**確認事項**:

```
✅ 要件を満たしているか
✅ エッジケースを考慮しているか
✅ エラーハンドリングは適切か
✅ 意図した動作をするか
```

**チェックポイント**:

```swift
// ❌ Bad: エッジケース未考慮
func getFirstItem<T>(_ array: [T]) -> T {
    return array[0]  // 空配列でクラッシュ
}

// ✅ Good: エッジケース考慮
func getFirstItem<T>(_ array: [T]) -> T? {
    guard !array.isEmpty else { return nil }
    return array[0]
}
```

**レビューコメント例**:

```markdown
[必須] エッジケースの考慮

空配列の場合にクラッシュする可能性があります。

提案:
- Optional返却に変更
- または guard で空チェック

テストケースの追加も推奨します：
- 空配列
- 1要素
- 複数要素
```

### 2. 設計（Design）

**確認事項**:

```
✅ アーキテクチャに従っているか
✅ 責務が適切に分離されているか
✅ 再利用性があるか
✅ 拡張性があるか
```

**チェックポイント**:

```swift
// ❌ Bad: ViewControllerに全てのロジック
class LoginViewController: UIViewController {
    func loginButtonTapped() {
        // APIリクエスト
        // データ保存
        // 画面遷移
        // 全て混在（200行）
    }
}

// ✅ Good: 責務分離
class LoginViewController: UIViewController {
    private let viewModel: LoginViewModel

    func loginButtonTapped() {
        viewModel.login()  // ロジックはViewModelに
    }
}

class LoginViewModel {
    private let authService: AuthService
    private let userRepository: UserRepository

    func login() {
        // ロジックはここで実装
    }
}
```

**レビューコメント例**:

```markdown
[推奨] MVVMパターンの適用

現在ViewControllerに全てのロジックがありますが、
ViewModelに移動することでテスタビリティが向上します。

メリット:
- ユニットテストが容易
- 再利用性向上
- 関心の分離による保守性向上

参考: `UserProfileViewController`で同様のパターンを使用
```

### 3. 可読性（Readability）

**確認事項**:

```
✅ 変数・関数名が明確か
✅ コメントは適切か
✅ 複雑なロジックは説明されているか
✅ コーディング規約に従っているか
```

**チェックポイント**:

```swift
// ❌ Bad: 読みにくい
func p(u: User) -> Bool {
    return u.a && u.s == 1
}

// ✅ Good: 読みやすい
func canAccessPremiumContent(user: User) -> Bool {
    return user.isPremiumMember && user.subscriptionStatus == .active
}
```

```swift
// ❌ Bad: Magic Number
if users.count > 100 {
    showPagination()
}

// ✅ Good: 定数で意味を明確に
private let paginationThreshold = 100

if users.count > paginationThreshold {
    showPagination()
}
```

**レビューコメント例**:

```markdown
[推奨] 変数名の改善

`p`, `u`, `a`, `s` といった略語は読みづらいです。

提案:
- `p` → `canAccessPremiumContent`
- `u` → `user`
- `a` → `isPremiumMember`
- `s` → `subscriptionStatus`

Swiftでは型推論があるため、変数名を省略せず
明確にすることがベストプラクティスです。
```

### 4. テスト（Testing）

**確認事項**:

```
✅ ユニットテストが追加されているか
✅ カバレッジは十分か
✅ エッジケースのテストがあるか
✅ テストが意味のあるテストか
```

**チェックポイント**:

```swift
// ❌ Bad: テストなし
func calculateDiscount(price: Double, discountRate: Double) -> Double {
    return price * (1 - discountRate)
}

// ✅ Good: 十分なテスト
class DiscountCalculatorTests: XCTestCase {
    func testCalculateDiscount_normalCase() {
        let result = calculateDiscount(price: 100, discountRate: 0.2)
        XCTAssertEqual(result, 80)
    }

    func testCalculateDiscount_zeroDiscount() {
        let result = calculateDiscount(price: 100, discountRate: 0)
        XCTAssertEqual(result, 100)
    }

    func testCalculateDiscount_fullDiscount() {
        let result = calculateDiscount(price: 100, discountRate: 1.0)
        XCTAssertEqual(result, 0)
    }

    func testCalculateDiscount_negativePrice() {
        // エッジケースのテスト
    }
}
```

**レビューコメント例**:

```markdown
[必須] ユニットテストの追加

新しいロジックが追加されていますが、
対応するユニットテストがありません。

以下のテストケースを推奨します：
- 正常系: 通常の割引計算
- 異常系: 負の価格、割引率が1超
- 境界値: 0円、割引率0、割引率100%

参考: `PriceCalculatorTests.swift`
```

### 5. セキュリティ（Security）

**確認事項**:

```
✅ 機密情報がハードコードされていないか
✅ 入力値の検証があるか
✅ SQLインジェクション対策があるか
✅ 認証・認可は適切か
```

**チェックポイント**:

```swift
// ❌ Bad: API Keyハードコード
let apiKey = "sk_live_abc123xyz"

// ✅ Good: 環境変数から読み込み
let apiKey = ProcessInfo.processInfo.environment["API_KEY"]

// または.xcconfig + Info.plistで管理
```

```swift
// ❌ Bad: SQL Injection脆弱性
let query = "SELECT * FROM users WHERE name = '\(userName)'"

// ✅ Good: プリペアドステートメント
let query = "SELECT * FROM users WHERE name = ?"
db.execute(query, parameters: [userName])
```

**レビューコメント例**:

```markdown
[必須] セキュリティ: API Keyの保護

API Keyがソースコードにハードコードされています。
Gitにコミットすると漏洩リスクがあります。

対策:
1. `.xcconfig`ファイルに移動
2. `.gitignore`に追加
3. 環境変数から読み込む

参考: `Config/Development.xcconfig`
```

### 6. パフォーマンス（Performance）

**確認事項**:

```
✅ 不要な計算が繰り返されていないか
✅ メモリリークの可能性はないか
✅ 非同期処理は適切か
✅ N+1問題はないか
```

**チェックポイント**:

```swift
// ❌ Bad: 毎回計算
for item in items {
    if item.price < items.map { $0.price }.max() {  // O(n²)
        // ...
    }
}

// ✅ Good: 一度だけ計算
let maxPrice = items.map { $0.price }.max()
for item in items {
    if item.price < maxPrice {  // O(n)
        // ...
    }
}
```

```swift
// ❌ Bad: メインスレッドで重い処理
func loadImage() {
    let image = processLargeImage()  // 重い処理
    imageView.image = image
}

// ✅ Good: 非同期処理
func loadImage() {
    DispatchQueue.global(qos: .userInitiated).async {
        let image = self.processLargeImage()
        DispatchQueue.main.async {
            self.imageView.image = image
        }
    }
}
```

**レビューコメント例**:

```markdown
[推奨] パフォーマンス改善

ループ内で`max()`を呼び出すと、O(n²)になります。

提案:
```swift
let maxPrice = items.map { $0.price }.max()
for item in items {
    if item.price < maxPrice {
        // ...
    }
}
```

これでO(n)に改善されます。
```

### 7. 保守性（Maintainability）

**確認事項**:

```
✅ 将来の変更が容易か
✅ 依存関係は適切か
✅ ドキュメント・コメントは十分か
✅ 技術的負債を増やしていないか
```

**チェックポイント**:

```swift
// ❌ Bad: ハードコードされたロジック
func getGreeting() -> String {
    let hour = Calendar.current.component(.hour, from: Date())
    if hour < 12 {
        return "Good morning"
    } else if hour < 18 {
        return "Good afternoon"
    } else {
        return "Good evening"
    }
}

// ✅ Good: 設定可能で拡張しやすい
struct GreetingConfig {
    let morningEnd: Int = 12
    let afternoonEnd: Int = 18
}

func getGreeting(config: GreetingConfig = GreetingConfig()) -> String {
    let hour = Calendar.current.component(.hour, from: Date())
    if hour < config.morningEnd {
        return "Good morning"
    } else if hour < config.afternoonEnd {
        return "Good afternoon"
    } else {
        return "Good evening"
    }
}
```

---

## 効果的なレビューコメント

### コメントの4つの要素

```
1. 重要度の明示
2. 問題点の指摘
3. 理由の説明
4. 解決策の提案
```

### コメントテンプレート

#### パターン1: 必須の修正

```markdown
[必須] <問題の概要>

<現状の問題点>

理由:
<なぜ問題か>

提案:
<具体的な修正案>

参考:
<関連ドキュメント・コード>
```

**例**:

```markdown
[必須] Optional強制アンラップ

`user!.name`で強制アンラップされています。

理由:
userがnilの場合、アプリがクラッシュします。

提案:
```swift
guard let user = user else {
    logger.error("User is nil")
    return
}
let name = user.name
```

参考: `UserProfileViewController.swift:45`で同様の処理
```

#### パターン2: 推奨の改善

```markdown
[推奨] <改善提案の概要>

<現状の説明>

メリット:
- <改善のメリット1>
- <改善のメリット2>

提案:
<具体的なコード例>
```

**例**:

```markdown
[推奨] Combineの活用

現在NotificationCenterを使用していますが、
Combineを使うとより宣言的で読みやすくなります。

メリット:
- メモリ管理が自動
- チェーン可能
- テストが容易

提案:
```swift
viewModel.$isLoading
    .receive(on: DispatchQueue.main)
    .sink { [weak self] isLoading in
        self?.updateLoadingState(isLoading)
    }
    .store(in: &cancellables)
```
```

#### パターン3: 質問・確認

```markdown
[質問] <質問内容>

<現状の理解>

確認したい点:
- <質問1>
- <質問2>
```

**例**:

```markdown
[質問] キャッシュの有効期限

キャッシュが無期限に保持されているように見えます。

確認したい点:
- 意図的に無期限ですか？
- それとも有効期限を設定すべきですか？
- メモリ使用量は考慮されていますか？
```

#### パターン4: 学習機会の提供

```markdown
[学習] <トピック>

<補足説明>

参考:
<リンク・ドキュメント>
```

**例**:

```markdown
[学習] weak/unownedの使い分け

現在`unowned`を使用していますが、`weak`の方が安全です。

解説:
- `weak`: nilになる可能性がある場合
- `unowned`: 絶対にnilにならないと保証できる場合

参考:
https://docs.swift.org/swift-book/LanguageGuide/AutomaticReferenceCounting.html
```

#### パターン5: 賞賛

```markdown
[賞賛] <良かった点>

<具体的な理由>
```

**例**:

```markdown
[賞賛] エラーハンドリング

全てのエラーケースを考慮し、ユーザーに
わかりやすいメッセージを表示している点が素晴らしいです！

特に、ネットワークエラーとサーバーエラーを
区別している点が良いですね。
```

### コメントの優先度

| プレフィックス | 意味 | マージ前の対応 |
|--------------|------|--------------|
| `[必須]` | 修正必須 | 必ず対応 |
| `[推奨]` | 修正推奨 | 判断に委ねる |
| `[質問]` | 質問・確認 | 回答必要 |
| `[提案]` | 代替案 | 検討推奨 |
| `[学習]` | 学習機会 | 対応不要 |
| `[賞賛]` | 良い点 | - |

---

## レビュープロセス

### ステップ1: 初見でざっと確認（5分）

```
目的: 全体像の把握
```

**チェック項目**:

```markdown
✅ PR説明を読む
✅ 変更ファイル数・行数を確認
✅ CIステータス確認（テスト・ビルド）
✅ 変更の種類を把握（feature/bugfix/refactor）
```

**判断基準**:

| 状況 | 対応 |
|------|------|
| CIが失敗 | コメント: "CIを通してください" |
| 変更が大きすぎる | コメント: "PRの分割を推奨" |
| 説明不足 | コメント: "詳細を追記してください" |
| 問題なし | 詳細レビューへ |

### ステップ2: 構造的なレビュー（10-15分）

```
目的: アーキテクチャ・設計の確認
```

**確認項目**:

```markdown
1. ファイル構成
   - 適切なディレクトリに配置されているか
   - 新規ファイル名は命名規則に従っているか

2. クラス/モジュール構成
   - 責務が適切に分離されているか
   - 既存のパターンに従っているか

3. 依存関係
   - 循環依存はないか
   - 不要な依存を増やしていないか

4. インターフェース
   - public/private/internalは適切か
   - APIが使いやすいか
```

### ステップ3: 詳細なコードレビュー（20-30分）

```
目的: 7つの観点での詳細確認
```

**レビュー順序**:

```
1. テストコードを先に読む
   → 意図を理解しやすい

2. 公開インターフェースを読む
   → 使い方を理解

3. 実装の詳細を読む
   → ロジックの確認

4. 7つの観点でチェック
   → 機能性、設計、可読性...
```

**レビューテクニック**:

```markdown
✅ 一度に全てレビューせず、観点ごとに分ける
✅ コードを頭の中で実行してみる
✅ エッジケースを想像する
✅ "なぜこう書いたのか"を考える
✅ 自分ならどう書くか考える
```

### ステップ4: 実際に動かして確認（10-15分）

```
目的: 動作確認
```

**手順**:

```bash
# 1. ブランチをチェックアウト
git fetch origin
git checkout feature/PROJ-123

# 2. 依存関係インストール
npm install  # または pod install

# 3. ビルド
npm run build

# 4. テスト実行
npm test

# 5. 実際にアプリを起動して動作確認
npm run dev
```

**確認ポイント**:

```markdown
✅ 正常系の動作
✅ エラーケースの挙動
✅ UIの表示（該当する場合）
✅ パフォーマンス（体感）
```

### ステップ5: コメント作成と判断（5-10分）

```
目的: フィードバックの整理と最終判断
```

**コメント整理**:

```markdown
1. 必須コメントを先に書く
2. 推奨コメントを追加
3. 質問を追加
4. 賞賛を忘れずに
```

**判断基準**:

| 判断 | 条件 | 次のアクション |
|------|------|--------------|
| **Approve** | 必須の問題なし | マージ可能 |
| **Comment** | 質問のみ | 回答待ち |
| **Request Changes** | 必須の問題あり | 修正が必要 |

---

## セルフレビューの実践

### セルフレビューとは

```
自分のコードをPR作成前に自分で確認する
→ レビュー時間を大幅削減
→ レビュー品質向上
```

### セルフレビューチェックリスト

#### 基本チェック

```markdown
□ コミットメッセージはConventional Commitsに従っているか
□ デバッグコード・console.logは削除したか
□ コメントアウトされたコードは削除したか
□ TODO/FIXMEコメントは適切か
□ ビルドが通るか
□ 全テストが通るか
```

#### コード品質チェック

```markdown
□ 変数・関数名は明確か
□ マジックナンバーは定数化したか
□ 重複コードはないか
□ 関数が長すぎないか（50行以内推奨）
□ 1ファイルが長すぎないか（300行以内推奨）
□ エラーハンドリングは適切か
```

#### テストチェック

```markdown
□ 新機能にユニットテストを追加したか
□ エッジケースのテストを追加したか
□ テストは意味のあるテストか（形だけでないか）
□ テストカバレッジは十分か
```

#### ドキュメントチェック

```markdown
□ PR説明を書いたか
□ 複雑なロジックにコメントを追加したか
□ README更新が必要なら更新したか
□ API変更があればドキュメント更新したか
```

### セルフレビューの実践方法

#### 方法1: GitHubでdiffを確認

```bash
# PRを作成（まだマージしない）
# GitHub上で自分のPRを開く
# Files changedタブで全ての変更を確認
# 問題があれば追加コミットで修正
```

**メリット**:
- レビュワーと同じ視点で確認できる
- コメントを自分に付けられる

#### 方法2: ローカルでdiff確認

```bash
# mainとの差分を確認
git diff main...HEAD

# または特定のファイルだけ
git diff main...HEAD -- path/to/file.swift

# 変更されたファイル一覧
git diff --name-only main...HEAD
```

#### 方法3: レビューチェックリスト使用

```markdown
セルフレビューシートを作成し、確認項目をチェック

例: `.github/PULL_REQUEST_TEMPLATE.md`に埋め込む

## セルフレビュー完了確認

- [ ] デバッグコード削除
- [ ] テスト追加
- [ ] ドキュメント更新
- [ ] ビルド・テスト成功
```

---

## チームレビュー文化の構築

### レビュー文化の5原則

#### 1. 心理的安全性

```
✅ レビューは人ではなくコードを対象にする
✅ 初心者の質問も歓迎
✅ 失敗から学ぶ文化
```

**実践例**:

```markdown
❌ "こんな書き方初心者だね"
✅ "このコードは○○の理由で改善できます"

❌ "なんでテスト書いてないの？"
✅ "テストを追加すると品質が向上します"
```

#### 2. 建設的なフィードバック

```
✅ 問題だけでなく解決策も提示
✅ 理由を説明する
✅ 良い点も積極的に褒める
```

**実践例**:

```markdown
❌ "これは遅い"
✅ "このループはO(n²)です。
    以下のようにすることでO(n)に改善できます:
    [コード例]"

✅ "このエラーハンドリング、完璧ですね！"
```

#### 3. 迅速なレビュー

```
✅ 24時間以内にレビュー開始
✅ 小さいPRは即座にレビュー
✅ レビュー時間を確保
```

**目標SLA**:

| PR規模 | レビュー開始 | レビュー完了 |
|--------|------------|------------|
| Small | 4時間以内 | 当日中 |
| Medium | 8時間以内 | 1営業日以内 |
| Large | 24時間以内 | 2営業日以内 |

#### 4. 継続的な学習

```
✅ レビューから学ぶ
✅ 定期的なレビューガイドライン見直し
✅ レトロスペクティブ実施
```

**実践例**:

```markdown
月1回: レビュー振り返り会
- 良かったレビュー事例を共有
- 見逃したバグの分析
- レビュープロセス改善
```

#### 5. 自動化の活用

```
✅ 機械的なチェックは自動化
✅ 人は設計・ロジックに集中
✅ CI/CDでレビュー負荷を軽減
```

**自動化例**:

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on: pull_request

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Linter
        run: npm run lint

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Tests
        run: npm test

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Coverage
        run: npm run coverage
      - name: Comment Coverage
        uses: romeovs/lcov-reporter-action@v0.3.1
```

---

## レビュー効率化テクニック

### テクニック1: レビュー依頼の最適化

```markdown
PR説明に以下を含める:

## レビュー観点
特に以下の点を確認してください:
- [ ] ViewModelのロジックが正しいか
- [ ] エラーハンドリングが適切か

## 確認不要な点
- UI実装は既にデザイナーレビュー済み
- パフォーマンステストは別途実施予定
```

**効果**: レビュワーが集中すべき点が明確になる

### テクニック2: 段階的レビュー

```markdown
大きな変更の場合:

1. Draft PRで設計レビュー
2. 実装完了後に詳細レビュー
3. 最終確認レビュー

→ 手戻りを最小化
```

### テクニック3: ペアプログラミングとの併用

```markdown
複雑な変更:
- ペアプログラミングで実装
- 既にレビュー済みと同等
- PR時は形式的確認のみ

→ レビュー時間を大幅削減
```

### テクニック4: レビューテンプレート

```markdown
チーム共通のレビューテンプレートを用意

## レビュー観点チェックリスト
- [ ] 機能性: 要件を満たしているか
- [ ] 設計: アーキテクチャに従っているか
- [ ] 可読性: 理解しやすいか
- [ ] テスト: 十分なテストがあるか
- [ ] セキュリティ: 脆弱性はないか
- [ ] パフォーマンス: 効率的か
- [ ] 保守性: 将来の変更が容易か

## 総合評価
[Approve / Comment / Request Changes]

## コメント
...
```

### テクニック5: 非同期レビュー vs 同期レビュー

| タイプ | 適用ケース | メリット |
|--------|-----------|---------|
| **非同期** | 通常のPR | 時間に縛られない |
| **同期**（画面共有） | 複雑な変更、議論が必要 | 迅速な合意形成 |

**同期レビューの進め方**:

```
1. 15分のレビュー会議を設定
2. 画面共有でコードを一緒に確認
3. その場で議論・合意
4. 必要な修正をその場で実施

→ 大幅な時間短縮
```

---

## よくある問題と対策

### 問題1: レビューが遅い

**原因**:
- レビュワーの時間不足
- PRが大きすぎる
- 優先度が不明確

**対策**:

```markdown
1. PR サイズを小さく保つ（400行以内）
2. レビュー依頼時に緊急度を明示
3. レビュータイムを設定（例: 毎日14:00-15:00）
4. 自動リマインダーを設定

GitHub Actions例:
- 24時間レビューされていないPRに自動通知
```

### 問題2: レビューコメントが攻撃的

**原因**:
- テキストコミュニケーションの難しさ
- プレッシャー・ストレス

**対策**:

```markdown
1. プレフィックスを使う（[必須][推奨]など）
2. 理由を説明する
3. 解決策を提示する
4. 良い点も褒める
5. 絵文字を適度に使う（😊、👍、💡）

定期的なレビュー文化の振り返りも有効
```

### 問題3: レビューで合意できない

**原因**:
- 主観的な意見の対立
- 技術選択の議論

**対策**:

```markdown
1. データで議論（パフォーマンス測定、事例など）
2. チーム基準を明文化（コーディング規約、設計原則）
3. 第三者（テックリードなど）の意見を求める
4. 時間を区切って決定（長時間議論しない）
5. Agree to Disagree（実験的に試す）

エスカレーションフロー:
1. 5分議論しても合意できない
2. テックリードに相談
3. 技術選択会議で決定
```

### 問題4: 細かすぎるレビュー

**原因**:
- コーディングスタイルの細かい指摘
- 本質的でない議論

**対策**:

```markdown
1. Linter/Formatterで自動化
2. スタイルガイドを明文化
3. "nit"プレフィックスで任意の指摘を明示

例:
[nit] インデントが1つずれています（自動整形で解決）
```

### 問題5: レビュー疲れ

**原因**:
- レビュー量が多すぎる
- 精神的な負担

**対策**:

```markdown
1. レビュー時間を限定（1日2時間まで）
2. レビュワーをローテーション
3. ペアプログラミングと併用
4. 自動レビューツールで負荷軽減
5. 小さいPRを推奨（レビューしやすい）
```

---

## 実践例

### 例1: 新機能追加PR

**PR概要**:

```markdown
## 概要
ユーザーがログイン時に生体認証を使用できる機能を追加

## 変更内容
- BiometricAuthServiceの実装
- LoginViewModelへの統合
- ユニットテスト追加
- 設定画面にトグル追加

## 動作確認
- [ ] Face IDでログイン成功
- [ ] Touch IDでログイン成功
- [ ] 生体認証失敗時にパスワード入力に戻る
- [ ] 設定でON/OFF切り替え可能

## スクリーンショット
[画像]

関連Issue: Closes #234
```

**レビューコメント例**:

```markdown
[必須] エラーハンドリング

BiometricAuthService.swift:45で、生体認証が利用できない
デバイスの場合のハンドリングがありません。

提案:
```swift
guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
    throw BiometricError.notAvailable
}
```

テストケース:
- シミュレーター（生体認証なし）での動作確認
- 設定で生体認証OFFの場合

---

[推奨] ViewModelのテスト強化

現在、正常系のテストのみですが、以下も追加を推奨:
- 生体認証キャンセル時
- 生体認証失敗時（3回）
- デバイス非対応時

---

[質問] 設定の保存先

生体認証ON/OFFの設定はUserDefaultsに保存されていますが、
Keychainの方が安全ではないでしょうか？

---

[賞賛] エラーメッセージ

各エラーケースで適切なメッセージを表示していて素晴らしいです！
ユーザーフレンドリーですね👍
```

### 例2: バグ修正PR

**PR概要**:

```markdown
## 概要
iPad横画面でレイアウトが崩れる問題を修正

## 問題
iPad横画面時、画像が画面外にはみ出す

## 原因
AutoLayoutの制約が固定値になっていた

## 修正内容
- 画像の幅を固定値から比率ベースに変更
- iPad横画面用のレイアウトテスト追加

## 動作確認
- [x] iPhone 縦画面 ✅
- [x] iPhone 横画面 ✅
- [x] iPad 縦画面 ✅
- [x] iPad 横画面 ✅

関連Issue: Fixes #456
```

**レビューコメント例**:

```markdown
[Approve] 修正内容を確認しました

実際にiPad Proシミュレーターで動作確認し、
問題なく表示されることを確認しました。

テストもエッジケースを含めて網羅的で良いですね！

マージして問題ありません✅
```

### 例3: リファクタリングPR

**PR概要**:

```markdown
## 概要
UserViewModelのリファクタリング

## 背景
UserViewModelが500行を超え、保守が困難になっていた

## 変更内容
- 責務ごとに分割
  - UserProfileViewModel
  - UserSettingsViewModel
  - UserAuthViewModel
- 共通ロジックをUserServiceに抽出
- 全てのユニットテストを維持

## 動作確認
- [x] 全テスト成功（既存テストに変更なし）
- [x] アプリ動作確認（全画面）

## メトリクス
- コード行数: 500行 → 150行 + 120行 + 100行 + 80行（Service）
- テストカバレッジ: 75% → 85%
```

**レビューコメント例**:

```markdown
[Approve] 素晴らしいリファクタリングです！

## 良い点
- 責務が明確に分離されている
- テストカバレッジが向上している
- 既存の動作を壊していない

## 質問（マージは問題なし）
UserServiceは新規作成ですが、今後他の画面でも
再利用できそうでしょうか？

もしそうなら、SharedServicesディレクトリへの移動を
検討しても良いかもしれません。

---

[学習] 参考情報

このようなリファクタリングは"Extract Class"パターンと呼ばれ、
Martin Fowlerの"Refactoring"で詳しく解説されています。

参考: https://refactoring.com/catalog/extractClass.html
```

---

## FAQ

### Q1: どのくらいの時間をレビューに使うべき？

**A**: プロジェクトの10-20%をレビューに充てることを推奨します。

```
例: 1週間（40時間）の開発
→ レビューに4-8時間

内訳:
- 他人のPRレビュー: 2-4時間
- セルフレビュー: 2-4時間
```

### Q2: 何行以下にPRを分割すべき？

**A**: 400行以下を推奨します。

| 行数 | 推奨 | 理由 |
|------|------|------|
| 〜100行 | ✅ | 理想的 |
| 100〜400行 | ✅ | 適切 |
| 400〜1000行 | ⚠️ | 分割検討 |
| 1000行〜 | ❌ | 分割必須 |

### Q3: Approve後に小さな修正を見つけた場合は？

**A**: 規模によります。

```
小さい修正（タイポなど）:
→ Approveのまま、後で修正してもらう

大きい修正（ロジック変更）:
→ Approveを取り消し、Request Changesに変更
```

### Q4: レビューで意見が対立したらどうする？

**A**: エスカレーションフローに従います。

```
1. 5分議論しても合意できない場合
2. 第三者（テックリードなど）に相談
3. それでも決まらない場合は技術選択会議
4. 時間制限を設ける（長時間議論しない）
```

### Q5: 自分よりベテランのコードをレビューする場合は？

**A**: 遠慮せずレビューしてください。

```
✅ 質問として投げる
   "[質問] この実装の意図を教えてください"

✅ 学習機会として活用
   "このパターン勉強になりました！"

✅ 新しい視点を提供できる
   → ベテランでも見落としはある
```

### Q6: 毎回同じ指摘をしている場合は？

**A**: プロセスを改善します。

```
1. Linter/Formatterで自動化
2. コーディング規約を明文化
3. ドキュメント化して共有
4. オンボーディング資料に追加
```

### Q7: レビューが遅くてブロックされる場合は？

**A**: 以下を試してください。

```
1. レビュワーに直接連絡（Slack等）
2. 別のレビュワーをアサイン
3. 緊急度をPRタイトルに明記
   例: [Urgent] Fix critical bug
4. レトロスペクティブでプロセス改善を議論
```

### Q8: 細かすぎる指摘をどう扱うべき？

**A**: プレフィックスで区別します。

```
[nit] インデントが1つずれています
→ 任意の修正、マージはブロックしない

自動フォーマッターの導入も検討
```

### Q9: セキュリティの専門知識がない場合は？

**A**: 専門家レビューを依頼します。

```
1. セキュリティ関連の変更は専門家もアサイン
2. 自動セキュリティスキャンツールを導入
   - Snyk
   - GitHub Advanced Security
3. わからないことは質問する
   "[質問] このコードにセキュリティリスクはありますか？"
```

### Q10: レビューで学んだことをチームで共有するには？

**A**: 以下の方法があります。

```
1. 週次レビュー振り返り会
   - 良かったレビュー事例を共有
   - 見逃したバグの分析

2. ドキュメント化
   - よくある指摘事項をWikiに追加
   - ベストプラクティス集を更新

3. 社内勉強会
   - レビューテクニックを共有

4. レビューコメントに[学習]タグを使用
   - 学習機会として明示的に共有
```

---

## まとめ

### PRレビューの成功の鍵

```
1. 迅速なレビュー（24時間以内）
2. 建設的なフィードバック
3. 7つの観点で包括的に確認
4. 自動化できることは自動化
5. 学習の機会として活用
6. 心理的安全性の確保
7. 継続的な改善
```

### 次のステップ

```
□ チームのレビューガイドライン作成
□ レビューテンプレート導入
□ 自動レビューツール導入
□ レビュー文化の振り返り会実施
□ セルフレビューの習慣化
```

### 関連リソース

- [GitHub Flow完全ガイド](01-github-flow.md)
- [コミットメッセージ規約](05-commit-messages.md)
- [コンフリクト解決](07-conflict-resolution.md)

---

## 📚 Appendix

### A. レビューコメントテンプレート集

#### A-1. セキュリティ関連

```markdown
[必須] 機密情報の漏洩リスク

API Keyがソースコードにハードコードされています。

リスク:
- Gitにコミットされると永続的に履歴に残る
- 第三者に漏洩する可能性

対策:
1. 環境変数に移動
2. .env.exampleを作成（サンプル値）
3. .envは.gitignoreに追加

参考: 12-Factorアプリの設定管理
https://12factor.net/ja/config
```

#### A-2. パフォーマンス関連

```markdown
[推奨] メモリ使用量の最適化

大量の画像を一度にメモリに読み込んでいます。

影響:
- メモリ使用量が大きい（約100MB）
- 低スペック端末でクラッシュの可能性

提案:
- ページングの実装
- 画像の遅延読み込み
- サムネイル使用

コード例:
```swift
// ページングの実装
func loadImages(page: Int, pageSize: Int = 20) {
    let start = page * pageSize
    let end = min(start + pageSize, totalCount)
    // ...
}
```
```

#### A-3. テスト関連

```markdown
[必須] エッジケースのテスト不足

現在、正常系のテストのみです。

不足しているテストケース:
- 空配列の場合
- nil値の場合
- 境界値（0, 最大値）
- 異常系（ネットワークエラー、タイムアウト）

推奨テスト:
```swift
func testEmptyArray() {
    let result = processor.process([])
    XCTAssertEqual(result, [])
}

func testNilValue() {
    let result = processor.process(nil)
    XCTAssertNil(result)
}

func testNetworkError() {
    // モックでエラーを発生させる
}
```
```

### B. PR説明テンプレート（拡張版）

```markdown
## 📝 概要
<!-- 何を変更したか1-2行で -->

## 🎯 目的
<!-- なぜこの変更が必要か -->

## 🔄 変更内容
<!-- 主な変更点を箇条書き -->
### 追加
-

### 変更
-

### 削除
-

## 🧪 テスト方法
<!-- 動作確認の手順 -->
1.
2.
3.

## ✅ チェックリスト
### コード品質
- [ ] ビルドが通る
- [ ] 全テストが通る
- [ ] Lintエラーなし
- [ ] デバッグコード削除
- [ ] コメントアウトコード削除

### テスト
- [ ] ユニットテスト追加
- [ ] エッジケースのテスト追加
- [ ] カバレッジ確認

### ドキュメント
- [ ] PR説明記載
- [ ] 複雑なロジックにコメント追加
- [ ] README更新（必要な場合）

### セキュリティ
- [ ] 機密情報のハードコード確認
- [ ] 入力値バリデーション確認
- [ ] 認証・認可確認

## 📸 スクリーンショット
<!-- UI変更がある場合 -->
### Before
<!-- 変更前 -->

### After
<!-- 変更後 -->

## 🔗 関連
<!-- 関連するIssue、PR、ドキュメント -->
- Closes #XXX
- Related to #YYY
- Depends on #ZZZ

## 📊 影響範囲
<!-- この変更が影響する範囲 -->
- [ ] iOS
- [ ] Android
- [ ] Web
- [ ] Backend
- [ ] Database

## ⚠️ 注意事項
<!-- レビュワーに特に確認してほしい点 -->
-

## 🚀 デプロイ時の注意
<!-- デプロイ時に必要な作業 -->
- [ ] マイグレーション実行
- [ ] 環境変数追加
- [ ] 設定ファイル更新
```

### C. レビュー観点別チェックリスト（詳細版）

#### 機能性チェックリスト

```markdown
□ 要件を満たしているか
  □ 仕様書と照合
  □ Acceptanceクライテリアを満たす
  □ エッジケースを考慮

□ エラーハンドリング
  □ 全てのエラーケースをカバー
  □ ユーザーにわかりやすいエラーメッセージ
  □ リトライ処理（必要に応じて）
  □ ログ出力

□ 入力値検証
  □ nil/null チェック
  □ 空文字/空配列チェック
  □ 範囲チェック
  □ 型チェック
  □ フォーマットチェック

□ 境界値テスト
  □ 最小値
  □ 最大値
  □ 0
  □ 負の値
  □ 空
```

#### 設計チェックリスト

```markdown
□ アーキテクチャ
  □ MVVMパターンに従っているか
  □ レイヤー間の依存が正しいか
  □ 循環依存がないか

□ SOLID原則
  □ 単一責任の原則（SRP）
  □ 開放閉鎖の原則（OCP）
  □ リスコフの置換原則（LSP）
  □ インターフェース分離の原則（ISP）
  □ 依存性逆転の原則（DIP）

□ DRY原則
  □ コードの重複がないか
  □ 共通処理は抽出されているか

□ 拡張性
  □ 新機能追加が容易か
  □ 設定変更が容易か
  □ テストが容易か
```

#### 可読性チェックリスト

```markdown
□ 命名
  □ 変数名が明確か
  □ 関数名が動作を表しているか
  □ クラス名が責務を表しているか
  □ 定数名が意味を表しているか

□ コード構造
  □ 関数が短い（50行以内推奨）
  □ クラスが短い（300行以内推奨）
  □ ネストが深すぎない（3階層以内推奨）
  □ 1行が長すぎない（100文字以内推奨）

□ コメント
  □ 必要な箇所にコメントがある
  □ コメントが最新の状態か
  □ コメントアウトコードが削除されているか
  □ TODOは適切に管理されているか

□ フォーマット
  □ インデントが統一されているか
  □ 空行の使い方が適切か
  □ コーディング規約に従っているか
```

### D. 自動レビューツール設定例

#### D-1. ESLint設定（JavaScript/TypeScript）

```json
// .eslintrc.json
{
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react/recommended",
    "prettier"
  ],
  "rules": {
    "no-console": "warn",
    "no-unused-vars": "error",
    "max-len": ["error", { "code": 100 }],
    "complexity": ["error", 10],
    "max-depth": ["error", 3],
    "max-lines-per-function": ["warn", 50]
  }
}
```

#### D-2. SwiftLint設定（iOS）

```yaml
# .swiftlint.yml
disabled_rules:
  - trailing_whitespace

opt_in_rules:
  - empty_count
  - closure_spacing
  - explicit_init

excluded:
  - Pods
  - Generated

line_length:
  warning: 100
  error: 120

function_body_length:
  warning: 40
  error: 50

type_body_length:
  warning: 250
  error: 300

file_length:
  warning: 400
  error: 500

cyclomatic_complexity:
  warning: 10
  error: 15
```

#### D-3. Prettier設定（フォーマッター）

```json
// .prettierrc
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "arrowParens": "always",
  "endOfLine": "lf"
}
```

#### D-4. Danger設定（PR自動チェック）

```javascript
// dangerfile.js
import { danger, warn, fail, message } from 'danger';

// PRサイズチェック
const bigPRThreshold = 400;
const changedLines = danger.github.pr.additions + danger.github.pr.deletions;

if (changedLines > bigPRThreshold) {
  warn(`このPRは${changedLines}行の変更があります。400行以下への分割を推奨します。`);
}

// PR説明チェック
if (danger.github.pr.body.length < 50) {
  fail('PR説明が短すぎます。変更内容を詳しく記載してください。');
}

// テストファイルチェック
const hasAppChanges = danger.git.modified_files.some(f => f.startsWith('src/'));
const hasTestChanges = danger.git.modified_files.some(f => f.includes('.test.'));

if (hasAppChanges && !hasTestChanges) {
  warn('実装の変更がありますが、テストの変更がありません。');
}

// TODO/FIXMEチェック
const newTodos = danger.git.created_files.concat(danger.git.modified_files)
  .filter(file => file.endsWith('.ts') || file.endsWith('.tsx'))
  .map(file => {
    const content = fs.readFileSync(file, 'utf8');
    return { file, todos: (content.match(/TODO|FIXME/g) || []).length };
  })
  .filter(item => item.todos > 0);

if (newTodos.length > 0) {
  message(`以下のファイルにTODO/FIXMEが追加されました:\n${newTodos.map(t => `- ${t.file}: ${t.todos}件`).join('\n')}`);
}
```

### E. レビュー効率化のためのGitHub Actions

```yaml
# .github/workflows/pr-automation.yml
name: PR Automation

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  auto-label:
    runs-on: ubuntu-latest
    steps:
      - name: Label based on files changed
        uses: actions/labeler@v4
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}

  size-label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Add size label
        uses: codelytv/pr-size-labeler@v1
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          xs_label: 'size/XS'
          xs_max_size: 10
          s_label: 'size/S'
          s_max_size: 100
          m_label: 'size/M'
          m_max_size: 400
          l_label: 'size/L'
          l_max_size: 1000
          xl_label: 'size/XL'

  auto-assign-reviewer:
    runs-on: ubuntu-latest
    steps:
      - name: Auto assign reviewer
        uses: kentaro-m/auto-assign-action@v1.2.1
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}

  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      - name: Install dependencies
        run: npm ci
      - name: Run linter
        run: npm run lint
      - name: Run tests
        run: npm test
      - name: Coverage
        run: npm run coverage
      - name: Comment coverage
        uses: romeovs/lcov-reporter-action@v0.3.1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          lcov-file: ./coverage/lcov.info

  stale-pr-reminder:
    runs-on: ubuntu-latest
    steps:
      - name: Stale PR reminder
        uses: actions/stale@v8
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
          stale-pr-message: 'このPRは7日間更新がありません。レビュー・対応をお願いします。'
          days-before-stale: 7
          days-before-close: -1
```

### F. レビューメトリクスの測定

#### 測定すべき指標

| メトリクス | 目標値 | 測定方法 |
|-----------|--------|---------|
| **レビュー開始時間** | 24時間以内 | PR作成〜初回コメントまで |
| **レビュー完了時間** | 2営業日以内 | PR作成〜Approveまで |
| **PRサイズ** | 400行以下 | 変更行数 |
| **レビューコメント数** | 5-15件 | PR当たりの平均コメント数 |
| **修正回数** | 2回以内 | Request Changes回数 |
| **マージ後バグ率** | 5%以下 | マージ後1週間以内のバグ数 |

#### メトリクスの可視化

```javascript
// GitHub API でメトリクス収集
const metrics = {
  avgReviewTime: 0,
  avgPRSize: 0,
  avgComments: 0,
  mergedPRs: 0,
  bugsAfterMerge: 0
};

// レビュー時間の計算
async function calculateReviewTime(pr) {
  const created = new Date(pr.created_at);
  const firstReview = await getFirstReviewTime(pr.number);
  return (firstReview - created) / (1000 * 60 * 60); // 時間単位
}

// ダッシュボード生成
function generateDashboard(metrics) {
  return `
## レビューメトリクス（直近30日）

| 指標 | 実績 | 目標 | 状態 |
|------|------|------|------|
| 平均レビュー時間 | ${metrics.avgReviewTime}h | 24h | ${metrics.avgReviewTime < 24 ? '✅' : '❌'} |
| 平均PRサイズ | ${metrics.avgPRSize}行 | 400行 | ${metrics.avgPRSize < 400 ? '✅' : '❌'} |
| 平均コメント数 | ${metrics.avgComments}件 | 5-15件 | ✅ |
| マージ後バグ率 | ${metrics.bugsAfterMerge}% | 5% | ${metrics.bugsAfterMerge < 5 ? '✅' : '❌'} |
  `;
}
```

#### 改善アクションの例

```markdown
## レビュープロセス改善施策

### レビュー時間が長い場合
- [ ] レビュータイムの設定（毎日14:00-15:00）
- [ ] 自動リマインダーの設定
- [ ] レビュワーの増員
- [ ] ペアプログラミングの活用

### PRサイズが大きい場合
- [ ] PR分割ガイドラインの作成
- [ ] Draft PRでの早期レビュー推奨
- [ ] Feature Flagの活用

### バグ率が高い場合
- [ ] レビュー観点の見直し
- [ ] テスト強化
- [ ] ペアプログラミング導入
- [ ] レビュアートレーニング
```

### G. 参考リンク集

#### 公式ドキュメント・ガイドライン

- [Google's Code Review Guidelines](https://google.github.io/eng-practices/review/)
- [Microsoft's Code Review Best Practices](https://docs.microsoft.com/en-us/azure/devops/repos/git/git-branching-guidance)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)

#### 書籍

- "The Art of Readable Code" - Dustin Boswell, Trevor Foucher
- "Clean Code" - Robert C. Martin
- "Refactoring" - Martin Fowler
- "Code Complete" - Steve McConnell

#### ツール

- [Danger](https://danger.systems/) - PR自動チェック
- [ESLint](https://eslint.org/) - JavaScript Linter
- [SwiftLint](https://github.com/realm/SwiftLint) - Swift Linter
- [Prettier](https://prettier.io/) - コードフォーマッター
- [SonarQube](https://www.sonarqube.org/) - 静的解析

#### コミュニティ

- [Stack Overflow - Code Review](https://codereview.stackexchange.com/)
- [Reddit - r/codereview](https://www.reddit.com/r/codereview/)

---

**このガイドは実際のプロジェクトで進化していきます。**
**フィードバックをお待ちしています！**
