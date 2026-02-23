# バージョン管理

> Git を使いこなすことは現代のソフトウェアエンジニアにとって「読み書き」と同じレベルの基本スキルである。

## この章で学ぶこと

- [ ] Gitの内部モデル（DAG）を理解する
- [ ] ブランチ戦略を説明できる
- [ ] 実務でのGitワークフローを知る
- [ ] コミットメッセージの書き方を習得する
- [ ] マージとリベースの使い分けを理解する
- [ ] コンフリクト解決の手順を身につける
- [ ] Git Hooksによる自動化を学ぶ
- [ ] CI/CDとの連携パターンを理解する
- [ ] 大規模リポジトリの運用ノウハウを知る
- [ ] トラブルシューティングの実践的テクニックを習得する

---

## 1. Gitの内部モデル

### 1.1 DAGとオブジェクトモデル

```
Git = 有向非巡回グラフ（DAG）+ コンテンツアドレスストレージ

  コミット: スナップショット（ファイル全体のツリー）
  ブランチ: コミットへのポインタ（ただの参照）
  HEAD: 現在のブランチ/コミットへのポインタ

  DAG構造:
  C1 ← C2 ← C3 ← C4      main
                 ↖
                  C5 ← C6  feature

  各コミットは親コミットのハッシュを保持
  → 改ざん不可能な履歴チェーン（ブロックチェーンと同原理）

  オブジェクトモデル:
  - blob: ファイルの内容（SHA-1ハッシュ）
  - tree: ディレクトリ（blobとtreeへの参照）
  - commit: ツリー + 親コミット + メタデータ
  - tag: コミットへの名前付き参照
```

### 1.2 オブジェクトモデルの詳細

```
Gitの4つのオブジェクト型:

  1. blob（Binary Large Object）
     ┌──────────────────────────────────┐
     │ blob 15\0Hello, World!\n         │
     │ SHA-1: af5626b4a114abcb82d63db   │
     └──────────────────────────────────┘
     - ファイルの中身そのもの
     - ファイル名を含まない（名前はtreeが管理）
     - 同じ内容のファイルは1つのblobを共有

  2. tree（ディレクトリに相当）
     ┌──────────────────────────────────────────────┐
     │ tree                                          │
     │ 100644 blob a1b2c3... README.md               │
     │ 100644 blob d4e5f6... main.py                 │
     │ 040000 tree 7a8b9c... src/                    │
     └──────────────────────────────────────────────┘
     - ファイル名とモード（パーミッション）を保持
     - 子のblobやtreeへの参照

  3. commit
     ┌──────────────────────────────────────────────┐
     │ commit                                        │
     │ tree    d8329fc...                             │
     │ parent  a0b1c2d...                             │
     │ author  Alice <alice@example.com> 1234567890   │
     │ committer Alice <alice@example.com> 1234567890 │
     │                                                │
     │ Add user authentication feature                │
     └──────────────────────────────────────────────┘
     - ルートtreeへの参照
     - 親コミットへの参照（マージコミットは2つ以上）
     - 作者とコミッター（別人の場合がある）
     - コミットメッセージ

  4. tag（annotated tag）
     ┌──────────────────────────────────────────────┐
     │ tag                                           │
     │ object  e3f4a5b...                            │
     │ type    commit                                │
     │ tag     v1.0.0                                │
     │ tagger  Alice <alice@example.com> 1234567890  │
     │                                               │
     │ Release version 1.0.0                         │
     └──────────────────────────────────────────────┘
```

```bash
# オブジェクトモデルを実際に確認するコマンド
git cat-file -t HEAD        # オブジェクトの型を表示
# commit

git cat-file -p HEAD        # オブジェクトの中身を表示
# tree d8329fc...
# parent a0b1c2d...
# author Alice <alice@example.com> 1704067200 +0900
# committer Alice <alice@example.com> 1704067200 +0900
#
# Add user authentication feature

git cat-file -p HEAD^{tree}  # ルートtreeの中身
# 100644 blob af5626b... README.md
# 040000 tree 7a8b9c0... src

git cat-file -p af5626b      # blobの中身
# （ファイルの内容が表示される）

# オブジェクトの格納場所
ls .git/objects/
# af/  7a/  d8/  ...
# 最初の2文字がディレクトリ名、残りがファイル名
```

### 1.3 参照（References）の仕組み

```
参照の種類:

  1. ブランチ参照
     .git/refs/heads/main          → コミットハッシュ
     .git/refs/heads/feature/login → コミットハッシュ

  2. リモート追跡参照
     .git/refs/remotes/origin/main → コミットハッシュ

  3. タグ参照
     .git/refs/tags/v1.0.0         → タグオブジェクトのハッシュ

  4. HEAD参照
     .git/HEAD → ref: refs/heads/main（通常）
     .git/HEAD → a0b1c2d...（detached HEAD）

  5. 特殊参照
     ORIG_HEAD → merge/rebase/reset前のHEAD
     MERGE_HEAD → マージ中の相手コミット
     FETCH_HEAD → 最後のfetchの結果

  参照の相対指定:
  HEAD~1     → HEADの1つ前の親（= HEAD~）
  HEAD~2     → HEADの2つ前の親
  HEAD^1     → HEADの最初の親（通常コミットではHEAD~と同じ）
  HEAD^2     → HEADの2番目の親（マージコミットの場合のみ意味がある）

  図で理解する:
  C1 ← C2 ← C3 ← M（マージコミット、親はC3とC5）
                 ↖
                  C4 ← C5

  M~1 = C3     （最初の親を辿る）
  M~2 = C2     （最初の親をさらに辿る）
  M^1 = C3     （最初の親）
  M^2 = C5     （2番目の親 = マージ元ブランチ）
```

### 1.4 ステージングエリア（インデックス）

```
Gitの3つの状態:

  Working Directory    Staging Area (Index)    Repository
  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
  │ 編集中のファイル │    │ コミット予定の変更 │    │ 確定した履歴  │
  │              │    │                  │    │              │
  │ file.py      │───→│ file.py          │───→│ commit abc   │
  │ (modified)   │add │ (staged)         │commit│              │
  └──────────────┘    └──────────────────┘    └──────────────┘
        ↑                                          │
        └──────────────────────────────────────────┘
                        checkout / restore

  なぜステージングエリアがあるのか:
  1. コミットの粒度を制御できる
     - 1つのファイルの一部だけをコミット可能（git add -p）
     - 複数の変更を論理的なコミットに分割

  2. コミット前のレビューができる
     - git diff --staged で確定予定の変更を確認
     - 誤った変更を含まないことを保証

  3. マージコンフリクト解決の作業スペース
     - コンフリクトをステージングで解決してからコミット
```

```bash
# ステージングの実践的な使い方

# ファイル全体をステージング
git add file.py

# ハンク（変更の塊）単位で対話的にステージング
git add -p file.py
# y: このハンクをステージング
# n: このハンクをスキップ
# s: ハンクをさらに分割
# e: ハンクを手動編集

# ステージング内容の確認
git diff --staged     # ステージングされた変更
git diff              # ステージングされていない変更
git diff HEAD         # 全ての変更（ステージング + 未ステージング）

# ステージングの取り消し
git restore --staged file.py  # ファイルをアンステージ（変更は保持）
git restore file.py           # ファイルの変更自体を取り消し（注意!）
```

---

## 2. ブランチ戦略

### 2.1 主要なブランチ戦略

```
主要なブランチ戦略:

  1. GitHub Flow（シンプル）:
     main ──●──●──●──●──●──●──
                ↖     ↗
     feature    ●──●──●

     → mainは常にデプロイ可能
     → featureブランチからPRを作成
     → マージ後にデプロイ

  2. Git Flow（厳格）:
     main    ──●────────────●──
     develop ──●──●──●──●──●──
                ↖  ↗     ↖  ↗
     feature    ●──●      ●──●
     release              ●──●──●

     → 開発用(develop), 安定版(main), リリース用(release)
     → 大規模プロジェクト向け

  3. Trunk-Based Development（モダン）:
     main ──●──●──●──●──●──●──●──
               ↖↗  ↖↗
     short     ●    ●    ← 短命ブランチ（1日以内）

     → mainに頻繁にマージ（1日数回）
     → フィーチャーフラグで未完成機能を隠蔽
     → Google, Facebook が採用
```

### 2.2 各戦略の詳細比較

```
ブランチ戦略の比較表:

  ┌──────────────────────────────────────────────────────────┐
  │ 特性           │ GitHub Flow │ Git Flow   │ Trunk-Based │
  ├──────────────────────────────────────────────────────────┤
  │ ブランチ数      │ 少ない      │ 多い       │ 最小限      │
  │ 複雑さ         │ 低い        │ 高い       │ 低い        │
  │ リリース頻度    │ 高い        │ 低い       │ 最も高い    │
  │ CI/CD適性      │ 高い        │ 中程度     │ 最も高い    │
  │ チームサイズ    │ 小〜中      │ 大         │ 小〜大      │
  │ 学習コスト      │ 低い        │ 高い       │ 低い        │
  │ コンフリクト    │ 中程度      │ 多い       │ 少ない      │
  │ ロールバック    │ 容易        │ 容易       │ フラグ切替  │
  │ 適用環境       │ Web/SaaS   │ パッケージ  │ Web/SaaS   │
  │ 代表的採用企業  │ GitHub     │ 大企業     │ Google     │
  └──────────────────────────────────────────────────────────┘

  選択の指針:
  - Webアプリ・SaaS → GitHub Flow or Trunk-Based
  - モバイルアプリ（定期リリース）→ Git Flow
  - CI/CDを完全に構築済み → Trunk-Based
  - チームがGitに不慣れ → GitHub Flow（最もシンプル）
  - 複数バージョンの同時サポート → Git Flow
```

### 2.3 GitHub Flow の実践

```bash
# --- GitHub Flow のワークフロー ---

# 1. mainブランチから新しいブランチを作成
git switch main
git pull origin main
git switch -c feature/add-user-auth

# 2. 開発とコミット
# ... コードを編集 ...
git add src/auth.py tests/test_auth.py
git commit -m "feat: ログイン認証機能を実装"

git add src/middleware.py
git commit -m "feat: 認証ミドルウェアを追加"

git add tests/test_middleware.py
git commit -m "test: 認証ミドルウェアのテストを追加"

# 3. リモートにプッシュ
git push -u origin feature/add-user-auth

# 4. Pull Request を作成
gh pr create --title "feat: ユーザー認証機能を追加" --body "
## 概要
ログイン認証とミドルウェアを実装しました。

## 変更内容
- JWT ベースの認証
- 認証ミドルウェア
- テスト

## テスト方法
\`\`\`bash
pytest tests/test_auth.py tests/test_middleware.py
\`\`\`
"

# 5. レビュー後にマージ
# （GitHub UI または CLI で実行）
gh pr merge --squash

# 6. ローカルの片付け
git switch main
git pull origin main
git branch -d feature/add-user-auth
```

### 2.4 Trunk-Based Development の実践

```bash
# --- Trunk-Based Development のワークフロー ---

# 1. 短命ブランチを作成（1日以内にマージ予定）
git switch main
git pull origin main
git switch -c short-lived/add-button

# 2. 小さな変更をコミット
git add src/components/Button.tsx
git commit -m "feat: 新しいボタンコンポーネントを追加（フラグ付き）"

# 3. フィーチャーフラグで未完成機能を隠蔽
# config/features.py
# FEATURE_FLAGS = {
#     "new_button": False,  # 本番ではオフ
# }

# 4. すぐにマージ
git push -u origin short-lived/add-button
gh pr create --title "feat: ボタンコンポーネント追加" --body "フラグ: new_button"
gh pr merge --squash

# 5. mainを最新にして次の作業へ
git switch main
git pull origin main

# フィーチャーフラグの段階的ロールアウト:
# 1. 開発環境: フラグON → テスト
# 2. ステージング: フラグON → QA
# 3. 本番: カナリアリリース（1%のユーザーにON）
# 4. 段階的にONの割合を増加（10% → 50% → 100%）
# 5. 全ユーザーにON → フラグのコードを削除
```

---

## 3. コミットメッセージ

### 3.1 Conventional Commits

```
Conventional Commits 仕様:

  フォーマット:
  <type>(<scope>): <description>

  <body>

  <footer>

  type の種類:
  ┌──────────────────────────────────────────────────┐
  │ type     │ 説明                    │ 例           │
  ├──────────────────────────────────────────────────┤
  │ feat     │ 新機能                  │ ログイン追加  │
  │ fix      │ バグ修正                │ NPE修正      │
  │ docs     │ ドキュメント            │ README更新   │
  │ style    │ コードスタイル          │ フォーマット   │
  │ refactor │ リファクタリング        │ 関数分割      │
  │ perf     │ パフォーマンス改善      │ クエリ最適化   │
  │ test     │ テスト                  │ テスト追加    │
  │ build    │ ビルドシステム          │ webpack設定   │
  │ ci       │ CI設定                  │ GitHub Actions│
  │ chore    │ その他の雑務            │ 依存関係更新   │
  │ revert   │ コミットの取り消し      │ 前回の変更戻し │
  └──────────────────────────────────────────────────┘

  scope（任意）: 変更の影響範囲
  → auth, api, ui, db, config 等

  BREAKING CHANGE の明示:
  → type の後に ! を付ける: feat!: ...
  → フッターに BREAKING CHANGE: ... を記載
```

### 3.2 良いコミットメッセージの書き方

```bash
# --- 良いコミットメッセージの例 ---

# ✅ 短い1行のメッセージ（シンプルな変更向け）
git commit -m "fix: ユーザー登録時のメールバリデーションエラーを修正"

# ✅ 本文付きのメッセージ（複雑な変更向け）
git commit -m "$(cat <<'EOF'
feat(auth): JWT ベースの認証機能を実装

パスワード認証に加え、JWTトークンによる認証を追加した。
トークンの有効期限は24時間で、リフレッシュトークンによる
自動更新をサポートする。

- アクセストークンの発行と検証
- リフレッシュトークンによるローテーション
- トークン無効化のためのブラックリスト

Closes #123
EOF
)"

# --- 悪いコミットメッセージの例 ---
# ❌ 何をしたか分からない
git commit -m "fix"
git commit -m "update"
git commit -m "作業中"
git commit -m "WIP"
git commit -m "misc changes"

# ❌ 大きすぎるコミット
git commit -m "認証機能追加、バグ修正、UIリファクタリング、テスト追加"
# → 複数のコミットに分割すべき

# ❌ 実装詳細だけで意図が分からない
git commit -m "if文を追加"
git commit -m "変数名をxからuserIdに変更"
```

### 3.3 コミットの粒度

```
適切なコミット粒度:

  原則: 1コミット = 1つの論理的な変更

  良い粒度の例:
  1. "feat: ユーザー登録APIを実装"
  2. "test: ユーザー登録APIのテストを追加"
  3. "fix: メールアドレスのバリデーションを修正"
  4. "refactor: UserServiceをリポジトリパターンに変更"

  悪い粒度の例:
  ❌ 粒度が細かすぎる:
  1. "関数のシグネチャを定義"
  2. "関数の本体を実装"
  3. "テストを1つ追加"
  4. "テストをもう1つ追加"

  ❌ 粒度が粗すぎる:
  1. "ユーザー管理機能一式を実装（認証、CRUD、API、テスト、ドキュメント）"

  コミットの分割テクニック:

  1. 変更を分けてステージング
     git add -p            # ハンク単位で選択
     git add src/auth.py   # ファイル単位で選択

  2. 作業後に分割
     git reset HEAD~1      # 最後のコミットを取り消し（変更は保持）
     git add -p            # 改めてハンク単位でコミット

  3. fixup/squash で整理
     git commit --fixup HEAD~2   # 2つ前のコミットの修正として
     git rebase -i HEAD~5        # 対話的リベースで整理
     # fixup を pick の後に配置して統合
```

---

## 4. マージとリベース

### 4.1 マージの種類

```
マージの3つの方法:

  1. Fast-Forward マージ
     Before:
     main    A ← B ← C
     feature           ↖ D ← E

     After (git merge feature):
     main    A ← B ← C ← D ← E

     → mainが遅れている場合に自動的に適用
     → マージコミットが作られない
     → 履歴が直線的

  2. 3-Way マージ（通常のマージ）
     Before:
     main    A ← B ← C ← F
     feature           ↖ D ← E

     After (git merge feature):
     main    A ← B ← C ← F ← M
                       ↖ D ← E ↗
     → M がマージコミット（2つの親を持つ）
     → 分岐の履歴が残る

  3. Squash マージ
     Before:
     main    A ← B ← C
     feature           ↖ D ← E ← F

     After (git merge --squash feature && git commit):
     main    A ← B ← C ← S
     → S は D+E+F を1つにまとめたコミット
     → 機能ブランチの細かい履歴を残さない
     → PRのマージでよく使われる
```

### 4.2 リベースの仕組み

```
リベース:

  Before:
  main    A ← B ← C ← D
  feature       ↖ E ← F ← G

  git switch feature
  git rebase main

  After:
  main    A ← B ← C ← D
  feature                ↖ E' ← F' ← G'

  → E, F, G を D の後に「貼り直す」
  → E', F', G' は新しいコミット（ハッシュが変わる）
  → 履歴が直線的になる

  リベースの利点:
  - 履歴がきれい（分岐がない直線）
  - git log が読みやすい
  - bisect がやりやすい

  リベースの注意点:
  - 公開済みのコミットをリベースしない!
    （他の人が参照しているコミットが消える）
  - プッシュ済みブランチのリベース後は force push が必要
    git push --force-with-lease origin feature
    （--force-with-lease は安全な force push）
```

### 4.3 マージ vs リベースの使い分け

```
マージとリベースの使い分け:

  マージを使う場合:
  - mainブランチへの統合
  - 共有ブランチの統合
  - 履歴を正確に残したい場合
  - マージの記録（いつ、何を統合したか）を残したい場合

  リベースを使う場合:
  - 個人のfeatureブランチをmainの最新に追従させる
  - コミット履歴をきれいに整理したい場合
  - PRマージ前のブランチ整理

  推奨ワークフロー:
  1. feature ブランチで開発
  2. main の変更を取り込む: git rebase main
  3. PR をレビュー
  4. main にマージ: squash merge（GitHub UI）

  ゴールデンルール:
  「公開済みの履歴をリベースしない」
  → 自分だけのブランチ: リベースOK
  → 共有ブランチ: マージを使う
```

### 4.4 対話的リベース

```bash
# --- 対話的リベース（コミット履歴の整理） ---

# 最新5コミットを整理
git rebase -i HEAD~5

# エディタが開く:
# pick a1b2c3d feat: ユーザーモデルを追加
# pick d4e5f6g feat: ユーザーAPIを実装
# pick 7a8b9c0 fix: typo修正
# pick 1d2e3f4 feat: バリデーション追加
# pick 5a6b7c8 fix: バリデーションのバグ修正

# 以下のように編集:
# pick a1b2c3d feat: ユーザーモデルを追加
# pick d4e5f6g feat: ユーザーAPIを実装
# fixup 7a8b9c0 fix: typo修正                 ← 上のコミットに統合
# pick 1d2e3f4 feat: バリデーション追加
# fixup 5a6b7c8 fix: バリデーションのバグ修正   ← 上のコミットに統合

# リベースコマンド:
# pick   = そのまま使う
# reword = コミットメッセージを変更
# edit   = コミットの内容を編集
# squash = 上のコミットに統合（メッセージも統合）
# fixup  = 上のコミットに統合（メッセージは捨てる）
# drop   = コミットを削除

# コミットの順序を入れ替えることも可能（行を移動するだけ）
```

---

## 5. コンフリクト解決

### 5.1 コンフリクトの発生原因

```
コンフリクトが発生するケース:

  1. 同じ行の変更
     main:    line = "Hello"  → line = "Hello, World!"
     feature: line = "Hello"  → line = "Hi there!"

  2. 隣接行の変更（場合による）
     一方がその行を削除し、他方が編集

  3. ファイルの削除と変更
     main:    file.py を削除
     feature: file.py を編集

  4. ファイルのリネーム
     main:    old.py → new.py
     feature: old.py を編集

  コンフリクトが発生しないケース:
  - 異なるファイルの変更
  - 同じファイルの異なる部分の変更
  - 一方がファイルを追加（既存ファイルと衝突しない）
```

### 5.2 コンフリクト解決の手順

```bash
# --- コンフリクト解決の手順 ---

# 1. マージ（またはリベース）を実行
git merge feature
# Auto-merging src/user.py
# CONFLICT (content): Merge conflict in src/user.py
# Automatic merge failed; fix conflicts and then commit the result.

# 2. コンフリクトの状態を確認
git status
# Unmerged paths:
#   both modified:   src/user.py

# 3. コンフリクトマーカーを確認
# ファイル内のコンフリクト表示:
# <<<<<<< HEAD
# name = "Hello, World!"
# =======
# name = "Hi there!"
# >>>>>>> feature

# 4. 手動で解決
# マーカーを削除し、正しい内容に編集:
# name = "Hello, World!"

# 5. 解決をステージング
git add src/user.py

# 6. マージコミットを作成
git commit  # マージの場合はメッセージが自動生成される

# --- リベース中のコンフリクト解決 ---
git rebase main
# CONFLICT in src/user.py

# 解決後:
git add src/user.py
git rebase --continue  # 次のコミットの適用に進む

# リベースを中断して元に戻す:
git rebase --abort
```

### 5.3 コンフリクトを減らす工夫

```
コンフリクトを最小限にする方法:

  1. 小さいPR を心がける
     - 変更ファイル数を少なく
     - 変更行数を少なく（目安: 200-400行以下）
     - PR のマージを早めに行う

  2. main を頻繁に取り込む
     git switch feature
     git rebase main  # 毎日実行

  3. ファイル構造を適切に設計
     - 1ファイルに多くのコードを詰め込まない
     - 関心の分離を意識する

  4. チーム内のコミュニケーション
     - 同じファイルを編集する場合は事前に共有
     - ペアプログラミングの活用

  5. フォーマッターの統一
     - .editorconfig で設定を共有
     - Prettier, Black 等を CI で強制
     - 自動フォーマットによる無意味な差分を排除

  6. ロックファイルのコンフリクト対策
     # .gitattributes でマージ戦略を指定
     package-lock.json merge=ours
     yarn.lock merge=ours
```

---

## 6. 実務のGitコマンド

### 6.1 日常コマンド

```bash
# --- 基本操作 ---
git status                  # 状態確認
git add -p                  # 対話的にステージング
git commit -m "msg"         # コミット
git push origin branch      # プッシュ
git pull --rebase           # プル（リベースマージ）

# --- ブランチ操作 ---
git switch -c feature       # ブランチ作成+切替
git switch main             # mainに切替
git branch -d feature       # マージ済みブランチを削除
git branch -D feature       # 未マージブランチを強制削除

# --- 履歴の確認 ---
git log --oneline --graph   # グラフ表示
git log --oneline -20       # 直近20件
git log --since="2024-01-01" --until="2024-01-31"  # 期間指定
git log --author="Alice"    # 作者で絞り込み
git log -p -- path/to/file  # ファイルの変更履歴を差分付きで
git blame file.py           # 各行の最終変更者

# --- 差分の確認 ---
git diff                    # ワーキングツリーの変更
git diff --staged           # ステージングされた変更
git diff main..feature      # ブランチ間の差分
git diff HEAD~3..HEAD       # 直近3コミットの差分
git diff --stat             # 変更の統計情報
```

### 6.2 便利なコマンド

```bash
# --- stash（一時退避）---
git stash                   # 変更を退避
git stash push -m "WIP: ログイン機能"  # メッセージ付き退避
git stash list              # 退避リスト
git stash pop               # 最新の退避を復元して削除
git stash apply stash@{1}   # 特定の退避を復元（削除しない）
git stash drop stash@{0}    # 特定の退避を削除

# --- cherry-pick（特定のコミットを適用）---
git cherry-pick abc1234     # 特定のコミットを現在のブランチに適用
git cherry-pick abc1234..def5678  # 範囲指定
git cherry-pick --no-commit abc1234  # コミットせずに変更だけ適用

# --- 特定のファイルを別ブランチから取得 ---
git checkout main -- path/to/file.py  # mainのファイルを取得
git restore --source main -- path/to/file.py  # 同上（新しい書式）

# --- コミットの修正 ---
git commit --amend          # 直前のコミットを修正（メッセージ変更）
git commit --amend --no-edit  # メッセージを変えずにファイルを追加

# --- タグ ---
git tag v1.0.0              # 軽量タグ
git tag -a v1.0.0 -m "Release 1.0.0"  # 注釈付きタグ
git push origin v1.0.0      # タグをプッシュ
git push origin --tags       # 全タグをプッシュ
```

### 6.3 トラブルシューティング

```bash
# --- reflog（HEADの移動履歴）---
git reflog
# abc1234 HEAD@{0}: commit: feat: 新機能追加
# def5678 HEAD@{1}: rebase finished: ...
# 7a8b9c0 HEAD@{2}: rebase: starting

# reflog を使った復旧
git reset --hard HEAD@{2}   # 2操作前の状態に戻る

# --- bisect（バグ導入コミットの特定）---
git bisect start
git bisect bad              # 現在のコミットはバグあり
git bisect good v1.0.0      # v1.0.0にはバグなし
# → Gitが中間のコミットをcheckout
# テストして good/bad を判定
git bisect good  # or  git bisect bad
# → 二分探索で絞り込み
git bisect reset            # 完了、元のブランチに戻る

# 自動化（テストスクリプトで）
git bisect start HEAD v1.0.0
git bisect run python -m pytest tests/test_bug.py

# --- 間違ったコミットの取り消し ---
# 方法1: revert（新しいコミットで取り消し）= 安全
git revert abc1234
git revert HEAD~3..HEAD     # 直近3コミットを取り消し

# 方法2: reset（履歴を巻き戻し）= 注意が必要
git reset --soft HEAD~1     # コミットを取消（変更はステージに残る）
git reset --mixed HEAD~1    # コミットを取消（変更はワーキングツリーに残る）
git reset --hard HEAD~1     # コミットを取消（変更も全て削除!）

# --- 消してしまったブランチの復旧 ---
git reflog | grep "feature/important"
# abc1234 HEAD@{5}: checkout: moving from feature/important to main
git branch feature/important abc1234  # ブランチを復活

# --- 大きなファイルの誤コミットの修正 ---
# 履歴から完全に削除（注意: 履歴を書き換える）
git filter-branch --force --tree-filter \
  'rm -f path/to/large-file.zip' HEAD

# より新しいツール: git-filter-repo
pip install git-filter-repo
git filter-repo --path path/to/large-file.zip --invert-paths
```

### 6.4 エイリアス設定

```bash
# --- 便利なGitエイリアス ---
git config --global alias.st "status"
git config --global alias.co "checkout"
git config --global alias.sw "switch"
git config --global alias.br "branch"
git config --global alias.ci "commit"
git config --global alias.lg "log --oneline --graph --decorate --all"
git config --global alias.last "log -1 HEAD --format='%H %s'"
git config --global alias.unstage "restore --staged"
git config --global alias.undo "reset --soft HEAD~1"
git config --global alias.amend "commit --amend --no-edit"
git config --global alias.wip "commit -am 'WIP'"
git config --global alias.branches "branch -a -v"
git config --global alias.tags "tag -l -n1"
git config --global alias.stashes "stash list"
git config --global alias.cleanup "!git branch --merged | grep -v '\\*\\|main\\|master\\|develop' | xargs -n 1 git branch -d"

# 使用例
git st           # git status
git lg           # きれいなグラフ表示
git undo         # 直前のコミットを取り消し
git cleanup      # マージ済みブランチを一括削除
```

---

## 7. Git Hooks

### 7.1 Git Hooks の種類

```
Git Hooksの種類:

  クライアントサイドフック（ローカル）:
  ┌──────────────────────────────────────────────────┐
  │ フック名             │ タイミング                  │
  ├──────────────────────────────────────────────────┤
  │ pre-commit           │ コミット前                  │
  │ prepare-commit-msg   │ コミットメッセージ編集前      │
  │ commit-msg           │ コミットメッセージ確定後      │
  │ post-commit          │ コミット後                  │
  │ pre-push             │ プッシュ前                  │
  │ pre-rebase           │ リベース前                  │
  │ post-checkout        │ チェックアウト後             │
  │ post-merge           │ マージ後                   │
  └──────────────────────────────────────────────────┘

  サーバーサイドフック:
  ┌──────────────────────────────────────────────────┐
  │ フック名             │ タイミング                  │
  ├──────────────────────────────────────────────────┤
  │ pre-receive          │ プッシュ受信前               │
  │ update               │ 各ブランチ更新前             │
  │ post-receive         │ プッシュ受信後               │
  └──────────────────────────────────────────────────┘
```

### 7.2 実用的な Git Hooks の例

```bash
#!/bin/bash
# .git/hooks/pre-commit（または husky で管理）

# --- Lintチェック ---
echo "Running lint..."
npm run lint
if [ $? -ne 0 ]; then
    echo "❌ Lint errors found. Please fix before committing."
    exit 1
fi

# --- テスト実行 ---
echo "Running tests..."
npm run test -- --bail
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix before committing."
    exit 1
fi

# --- 機密情報の検出 ---
echo "Checking for secrets..."
PATTERNS="(password|secret|api_key|token)\s*=\s*['\"][^'\"]+['\"]"
if git diff --cached --name-only | xargs grep -lE "$PATTERNS" 2>/dev/null; then
    echo "❌ Potential secrets detected. Please review."
    exit 1
fi

# --- 大きなファイルの検出 ---
MAX_SIZE=5242880  # 5MB
for file in $(git diff --cached --name-only); do
    if [ -f "$file" ]; then
        size=$(wc -c < "$file")
        if [ "$size" -gt "$MAX_SIZE" ]; then
            echo "❌ File $file exceeds 5MB. Use Git LFS instead."
            exit 1
        fi
    fi
done

echo "✅ All checks passed."
exit 0
```

```bash
#!/bin/bash
# .git/hooks/commit-msg
# Conventional Commits のフォーマットを強制

commit_msg=$(cat "$1")
pattern="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?(!)?: .{1,72}"

if ! echo "$commit_msg" | head -1 | grep -qE "$pattern"; then
    echo "❌ Invalid commit message format."
    echo ""
    echo "Expected format: <type>(<scope>): <description>"
    echo "  type: feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
    echo "  scope: optional, e.g., (auth), (api)"
    echo "  description: max 72 characters"
    echo ""
    echo "Example: feat(auth): ログイン認証を実装"
    exit 1
fi

exit 0
```

### 7.3 husky + lint-staged による自動化

```json
// package.json
{
  "scripts": {
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss}": [
      "stylelint --fix",
      "prettier --write"
    ],
    "*.{json,md}": [
      "prettier --write"
    ]
  }
}
```

```bash
# husky のセットアップ
npm install --save-dev husky lint-staged
npx husky install

# pre-commit フック
npx husky add .husky/pre-commit "npx lint-staged"

# commit-msg フック（Conventional Commits チェック）
npm install --save-dev @commitlint/cli @commitlint/config-conventional
npx husky add .husky/commit-msg 'npx commitlint --edit "$1"'
```

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2, 'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'perf',
       'test', 'build', 'ci', 'chore', 'revert']
    ],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
  },
};
```

---

## 8. CI/CDとの連携

### 8.1 GitHub Actions の基本

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  test:
    runs-on: ubuntu-latest
    needs: lint
    strategy:
      matrix:
        node-version: [18, 20, 22]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm run test -- --coverage
      - uses: actions/upload-artifact@v4
        with:
          name: coverage-${{ matrix.node-version }}
          path: coverage/

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
      - run: echo "Deploying to production..."
      # 実際のデプロイコマンド
```

### 8.2 Pull Requestの自動化

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  # PR のサイズチェック
  pr-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Check PR size
        run: |
          LINES_CHANGED=$(git diff --stat origin/${{ github.base_ref }}...HEAD | tail -1 | awk '{print $4+$6}')
          echo "Lines changed: $LINES_CHANGED"
          if [ "$LINES_CHANGED" -gt 1000 ]; then
            echo "::warning::PR is too large ($LINES_CHANGED lines). Consider splitting."
          fi

  # 自動レビューアサイン
  auto-assign:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v2
        with:
          configuration-path: '.github/auto-assign.yml'

  # ラベル自動付与
  labeler:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}
```

### 8.3 セマンティックバージョニングとリリース

```
セマンティックバージョニング（SemVer）:
  MAJOR.MINOR.PATCH

  MAJOR: 互換性のない変更
  MINOR: 後方互換性のある新機能
  PATCH: 後方互換性のあるバグ修正

  例:
  1.0.0 → 1.0.1  パッチ（バグ修正）
  1.0.0 → 1.1.0  マイナー（新機能追加）
  1.0.0 → 2.0.0  メジャー（破壊的変更）

  プレリリース:
  2.0.0-alpha.1
  2.0.0-beta.1
  2.0.0-rc.1（Release Candidate）

  Conventional Commits との対応:
  feat: → MINOR バージョンアップ
  fix:  → PATCH バージョンアップ
  feat!: または BREAKING CHANGE: → MAJOR バージョンアップ
```

```bash
# semantic-release による自動リリース
# Conventional Commits からバージョンを自動決定

# インストール
npm install --save-dev semantic-release @semantic-release/git @semantic-release/changelog

# .releaserc.json
# {
#   "branches": ["main"],
#   "plugins": [
#     "@semantic-release/commit-analyzer",
#     "@semantic-release/release-notes-generator",
#     "@semantic-release/changelog",
#     "@semantic-release/npm",
#     "@semantic-release/github",
#     ["@semantic-release/git", {
#       "assets": ["CHANGELOG.md", "package.json"],
#       "message": "chore(release): ${nextRelease.version}"
#     }]
#   ]
# }
```

---

## 9. 大規模リポジトリの運用

### 9.1 モノレポ vs マルチレポ

```
モノレポ（Monorepo）:
  1つのリポジトリに全プロジェクト

  ┌─────────────────────────────────────┐
  │ monorepo/                           │
  │ ├── packages/                       │
  │ │   ├── web/        (フロントエンド)  │
  │ │   ├── api/        (バックエンド)    │
  │ │   ├── shared/     (共通ライブラリ)  │
  │ │   └── mobile/     (モバイルアプリ)  │
  │ ├── tools/                          │
  │ ├── package.json                    │
  │ └── turbo.json (or nx.json)         │
  └─────────────────────────────────────┘

  利点:
  - コード共有が容易
  - 原子的な変更（APIとフロントを同時に変更）
  - 統一されたCI/CD
  - 依存関係の一元管理

  欠点:
  - リポジトリが巨大化
  - ビルド時間の増加
  - アクセス制御が難しい
  - Git操作が遅くなる可能性

  代表ツール: Turborepo, Nx, Lerna, Bazel

マルチレポ（Multirepo）:
  プロジェクトごとに別リポジトリ

  利点:
  - リポジトリが軽量
  - 独立したCI/CD
  - 細かいアクセス制御
  - チームの自律性

  欠点:
  - コード共有が困難
  - 横断的な変更が面倒
  - 依存関係の管理が複雑
  - バージョンの不整合
```

### 9.2 大きなファイルの管理

```bash
# --- Git LFS（Large File Storage）---

# インストール
git lfs install

# 追跡するファイルパターンを指定
git lfs track "*.psd"
git lfs track "*.zip"
git lfs track "*.mp4"
git lfs track "assets/images/*.png"

# .gitattributes が自動生成される
# *.psd filter=lfs diff=lfs merge=lfs -text
# *.zip filter=lfs diff=lfs merge=lfs -text

# 通常通りadd/commit/push
git add .gitattributes
git add large-file.psd
git commit -m "feat: デザインファイルを追加"
git push origin main

# LFS の状態確認
git lfs ls-files     # LFS管理ファイル一覧
git lfs status       # LFS の状態
```

### 9.3 .gitignore のベストプラクティス

```bash
# --- .gitignore の推奨設定 ---

# --- OS ---
.DS_Store
Thumbs.db
*.swp
*~

# --- エディタ/IDE ---
.vscode/settings.json
.idea/
*.sublime-workspace
*.code-workspace

# --- 言語/フレームワーク ---
# Node.js
node_modules/
npm-debug.log*
yarn-error.log*
.pnpm-debug.log*

# Python
__pycache__/
*.py[cod]
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# Java
*.class
target/
*.jar
*.war

# Go
/vendor/

# --- ビルド成果物 ---
dist/
build/
out/
.next/
.nuxt/

# --- 環境変数・機密情報 ---
.env
.env.local
.env.production
*.pem
*.key
credentials.json

# --- テスト/カバレッジ ---
coverage/
.nyc_output/
*.lcov
htmlcov/

# --- ログ ---
*.log
logs/

# --- データベース ---
*.sqlite3
*.db

# グローバル .gitignore（全リポジトリ共通）
git config --global core.excludesfile ~/.gitignore_global
```

---

## 10. セキュリティ

### 10.1 機密情報の管理

```
Gitリポジトリでの機密情報管理:

  絶対にコミットしてはいけないもの:
  - パスワード、APIキー
  - 秘密鍵（.pem, .key）
  - .env ファイル（本番環境の値）
  - credentials.json
  - データベース接続文字列

  対策:
  1. .gitignore に追加
  2. .env.example をコミット（値は空またはダミー）
  3. git-secrets で事前検出
  4. 万が一コミットした場合は即座にキーをローテーション

  git-secrets のセットアップ:
```

```bash
# git-secrets のインストールと設定
brew install git-secrets  # macOS

# リポジトリに設定
cd my-project
git secrets --install
git secrets --register-aws  # AWSキーパターンを登録

# カスタムパターンの追加
git secrets --add 'password\s*=\s*["\047][^"\047]+["\047]'
git secrets --add 'PRIVATE[_-]?KEY'

# コミット前に自動チェック（pre-commit hook に追加される）
git secrets --scan         # リポジトリ全体をスキャン
git secrets --scan-history # 全履歴をスキャン
```

### 10.2 GPG署名

```bash
# --- コミット署名（GPG）---

# GPGキーの生成
gpg --full-generate-key

# キーIDの確認
gpg --list-secret-keys --keyid-format=long
# sec   rsa4096/ABC1234567890DEF 2024-01-01 [SC]

# Gitに設定
git config --global user.signingkey ABC1234567890DEF
git config --global commit.gpgsign true  # 全コミットに署名

# 署名付きコミット
git commit -S -m "feat: 署名付きコミット"

# 署名の検証
git log --show-signature
# gpg: Good signature from "Alice <alice@example.com>"

# GitHubに公開鍵を登録
gpg --armor --export ABC1234567890DEF
# → GitHubの Settings → SSH and GPG keys に追加
```

---

## 11. Gitの高度なテクニック

### 11.1 worktree（複数作業ディレクトリ）

```bash
# --- git worktree ---
# 1つのリポジトリで複数のブランチを同時にチェックアウト

# 新しいworktreeを作成
git worktree add ../project-hotfix hotfix/critical-bug
git worktree add ../project-review feature/new-ui

# worktreeの一覧
git worktree list
# /path/to/project          abc1234 [main]
# /path/to/project-hotfix   def5678 [hotfix/critical-bug]
# /path/to/project-review   7a8b9c0 [feature/new-ui]

# worktreeの削除
git worktree remove ../project-hotfix

# ユースケース:
# - 長時間のビルドを別ディレクトリで実行しつつ開発を続ける
# - コードレビュー用に別ブランチをチェックアウト
# - 緊急のhotfixを本作業を中断せずに対応
```

### 11.2 サブモジュール

```bash
# --- git submodule ---
# 外部リポジトリをサブディレクトリとして含める

# サブモジュールの追加
git submodule add https://github.com/example/library.git libs/library
git commit -m "feat: libraryをサブモジュールとして追加"

# サブモジュール付きリポジトリのクローン
git clone --recurse-submodules https://github.com/example/project.git
# または
git clone https://github.com/example/project.git
cd project
git submodule init
git submodule update

# サブモジュールの更新
git submodule update --remote  # リモートの最新を取得
cd libs/library
git checkout v2.0.0            # 特定のバージョンに固定
cd ../..
git add libs/library
git commit -m "chore: libraryをv2.0.0に更新"

# サブモジュールの注意点:
# - 複雑さが増す（クローン時に --recurse-submodules が必要）
# - CIの設定が複雑になる
# - 代替手段: npm/pip等のパッケージマネージャーを検討
```

### 11.3 パフォーマンス最適化

```bash
# --- 大規模リポジトリのパフォーマンス改善 ---

# シャロークローン（履歴を制限）
git clone --depth 1 https://github.com/large/repo.git
git clone --depth 100 https://github.com/large/repo.git
git clone --shallow-since="2024-01-01" https://github.com/large/repo.git

# 部分クローン（blobを遅延取得）
git clone --filter=blob:none https://github.com/large/repo.git

# スパースチェックアウト（特定のディレクトリのみ）
git clone --no-checkout https://github.com/large/repo.git
cd repo
git sparse-checkout init --cone
git sparse-checkout set packages/web packages/shared
git checkout main
# → packages/web と packages/shared のみがチェックアウトされる

# fsmonitor（ファイル監視デーモン）で高速化
git config core.fsmonitor true
git config core.untrackedcache true

# gc（ガベージコレクション）
git gc --aggressive  # リポジトリの最適化
git prune            # 不要オブジェクトの削除

# パフォーマンス確認
git count-objects -vH  # オブジェクト数とサイズ
```

---

## 12. チーム開発のベストプラクティス

### 12.1 CODEOWNERS

```bash
# --- .github/CODEOWNERS ---
# ファイル/ディレクトリごとにレビューアを自動アサイン

# デフォルトのオーナー
* @team-lead

# フロントエンド
/src/components/ @frontend-team
/src/pages/ @frontend-team
*.tsx @frontend-team
*.css @frontend-team

# バックエンド
/src/api/ @backend-team
/src/models/ @backend-team
/src/services/ @backend-team

# インフラ
/terraform/ @infra-team
/docker/ @infra-team
/.github/workflows/ @infra-team
Dockerfile @infra-team

# ドキュメント
/docs/ @tech-writers
*.md @tech-writers

# セキュリティ関連
/src/auth/ @security-team
/src/middleware/auth* @security-team
```

### 12.2 Pull Request テンプレート

```markdown
<!-- .github/pull_request_template.md -->

## 概要
<!-- この PR で何を変更しましたか？ -->

## 変更の種類
- [ ] 新機能（feat）
- [ ] バグ修正（fix）
- [ ] リファクタリング（refactor）
- [ ] ドキュメント（docs）
- [ ] テスト（test）
- [ ] その他

## 変更内容
<!-- 変更の詳細を記載 -->

## テスト方法
<!-- テスト手順を記載 -->

## スクリーンショット
<!-- UI変更がある場合はスクリーンショットを添付 -->

## チェックリスト
- [ ] テストを追加/更新した
- [ ] ドキュメントを更新した
- [ ] 破壊的変更はない
- [ ] パフォーマンスへの影響を確認した
- [ ] セキュリティへの影響を確認した

## 関連Issue
Closes #
```

### 12.3 ブランチ保護ルール

```
推奨するブランチ保護設定（main ブランチ）:

  ┌──────────────────────────────────────────────────┐
  │ 設定項目                   │ 推奨値              │
  ├──────────────────────────────────────────────────┤
  │ Require pull request       │ ON                  │
  │ Required reviewers         │ 1-2人               │
  │ Dismiss stale reviews      │ ON                  │
  │ Require status checks      │ ON                  │
  │ Require branches up to date│ ON                  │
  │ Require conversation       │ ON                  │
  │ resolution                 │                     │
  │ Require signed commits     │ 必要に応じて         │
  │ Include administrators     │ ON                  │
  │ Allow force pushes         │ OFF                 │
  │ Allow deletions            │ OFF                 │
  └──────────────────────────────────────────────────┘

  必須のステータスチェック:
  - CI（lint, test, build）
  - セキュリティスキャン
  - コードカバレッジ閾値
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Gitの内部 | DAG + コンテンツアドレス。改ざん不可能 |
| ブランチ | ただのポインタ。軽量に作成・削除 |
| 戦略 | GitHub Flow(シンプル), Trunk-Based(モダン) |
| コミット | Conventional Commits、1コミット1論理変更 |
| マージ vs リベース | 共有ブランチはマージ、個人ブランチはリベース |
| コンフリクト | 小さいPR + 頻繁なリベースで最小化 |
| Hooks | pre-commit でリント・テスト・機密情報チェック |
| CI/CD | GitHub Actions で自動化 |
| セキュリティ | 機密情報は .gitignore + git-secrets |
| 必須スキル | rebase, stash, bisect, reflog |

---

## 次に読むべきガイド
→ [[../08-advanced-topics/00-distributed-systems.md]] — 分散システム

---

## 参考文献
1. Chacon, S. & Straub, B. "Pro Git." 2nd Edition, Apress, 2014.
2. Driessen, V. "A Successful Git Branching Model." 2010.
3. Conventional Commits Specification. https://www.conventionalcommits.org/
4. GitHub Flow Guide. https://docs.github.com/en/get-started/quickstart/github-flow
5. Atlassian Git Tutorials. https://www.atlassian.com/git/tutorials
6. Trunk Based Development. https://trunkbaseddevelopment.com/
7. Semantic Versioning. https://semver.org/
