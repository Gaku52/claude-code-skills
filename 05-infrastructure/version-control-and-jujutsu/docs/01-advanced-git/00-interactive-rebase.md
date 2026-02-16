# インタラクティブRebase

> `git rebase -i`を使いこなし、コミット履歴の整理（squash、fixup、reword、edit、drop）を安全に行うためのテクニックと運用ルールを解説する。

## この章で学ぶこと

1. **インタラクティブRebaseの基本操作** -- 各コマンド（pick, squash, fixup, reword, edit, drop）の動作と使い分け
2. **安全なコミット履歴の書き換え** -- autosquash、fixupコミット、`--update-refs`による効率的なワークフロー
3. **トラブル発生時の復旧方法** -- rebase中のコンフリクト対処、`--abort`、reflogからの救出
4. **高度なrebaseテクニック** -- `--rebase-merges`、`exec`コマンド、スタックドブランチ運用
5. **チーム開発でのrebase運用ルール** -- 安全なforce-push、レビュー前の履歴整理フロー

---

## 1. インタラクティブRebaseの基本

### 1.1 起動方法

```bash
# 直近N件のコミットを対象にする
$ git rebase -i HEAD~5

# 特定のコミット以降を対象にする
$ git rebase -i abc123

# ルートコミットから全てを対象にする
$ git rebase -i --root

# 上流ブランチとの分岐点から対象にする
$ git rebase -i main

# upstream..HEADの範囲を明示的に指定
$ git rebase -i --onto main feature-base feature-branch
```

### 1.2 エディタの設定

rebase時に開くエディタはGitの設定で制御できる。

```bash
# エディタの設定
$ git config --global core.editor "vim"
$ git config --global core.editor "code --wait"
$ git config --global core.editor "nano"

# 環境変数でも指定可能（優先度: GIT_SEQUENCE_EDITOR > GIT_EDITOR > core.editor）
$ export GIT_SEQUENCE_EDITOR="vim"

# todoリスト編集時のみ別のエディタを使う
$ GIT_SEQUENCE_EDITOR="code --wait" git rebase -i HEAD~5
# → todoリストはVS Codeで編集、reword時のメッセージ編集はvimなど
```

### 1.3 todoリストの構造

```bash
$ git rebase -i HEAD~4
# エディタが開き、以下のようなtodoリストが表示される:

pick a1b2c3d feat: ユーザー認証の基本実装
pick d4e5f6a fix: パスワードバリデーションの修正
pick b7c8d9e feat: ログイン画面のUI実装
pick e0f1a2b fix: typo in login form

# Rebase abc123..e0f1a2b onto abc123 (4 commands)
#
# Commands:
# p, pick   = コミットをそのまま使用
# r, reword = コミットを使用するが、メッセージを変更
# e, edit   = コミットを使用するが、修正のために停止
# s, squash = 直前のコミットに統合（メッセージも統合）
# f, fixup  = squashと同じだが、このコミットのメッセージは破棄
# x, exec   = シェルコマンドを実行
# b, break  = ここで停止（後で git rebase --continue で再開）
# d, drop   = コミットを削除
# l, label  = 現在のHEADにラベルを付ける
# t, reset  = HEADをラベルにリセット
# m, merge  = マージコミットを作成
```

```
┌─────────────────────────────────────────────────────┐
│  todoリストの実行順序                                │
│                                                     │
│  ※ 上から下へ順番に適用される（古い→新しい）       │
│                                                     │
│  pick a1b2c3d  ──→ 1番目に適用                     │
│  pick d4e5f6a  ──→ 2番目に適用                     │
│  pick b7c8d9e  ──→ 3番目に適用                     │
│  pick e0f1a2b  ──→ 4番目に適用                     │
│                                                     │
│  行の入れ替え = コミット順序の変更                   │
│  行の削除     = コミットの削除（dropと同じ）         │
└─────────────────────────────────────────────────────┘
```

### 1.4 rebaseの内部動作

rebaseは内部的に以下の手順で実行される。この動作を理解するとトラブル時の対応が容易になる。

```
┌──────────────────────────────────────────────────────┐
│  rebase の内部動作フロー                              │
│                                                      │
│  1. ORIG_HEAD に現在のHEADを保存                     │
│  2. HEADを onto（rebase先）に移動                    │
│  3. todoリストを上から順に処理                        │
│     ├── pick: cherry-pick で適用                     │
│     ├── squash/fixup: 前のコミットに統合             │
│     ├── reword: cherry-pick後メッセージ編集          │
│     ├── edit: cherry-pick後に停止                    │
│     └── drop: スキップ（何もしない）                 │
│  4. 全て完了したらブランチポインタを更新              │
│                                                      │
│  ※ 各ステップでコンフリクトが発生すると停止          │
│  ※ .git/rebase-merge/ にtodoリスト等の状態を保存    │
└──────────────────────────────────────────────────────┘
```

```bash
# rebase中の内部状態を確認
$ ls .git/rebase-merge/
git-rebase-todo       # 残りのtodoリスト
done                  # 完了したコマンド
head-name             # rebase開始時のブランチ名
onto                  # rebase先のコミット
orig-head             # rebase開始時のHEAD
interactive           # インタラクティブモードのフラグ
```

---

## 2. 各コマンドの詳細

### 2.1 squash -- コミットの統合（メッセージ編集あり）

```bash
# Before: 4つの細かいコミット
pick a1b2c3d feat: 認証機能の骨格を作成
squash d4e5f6a feat: パスワードハッシュを実装
squash b7c8d9e feat: ログインエンドポイントを追加
squash e0f1a2b feat: 認証ミドルウェアを実装

# → エディタが開き、4つのコミットメッセージが表示される
# → 統合後のメッセージを編集して保存
```

squash時のメッセージ編集画面の例:

```
# This is a combination of 4 commits.
# This is the 1st commit message:

feat: 認証機能の骨格を作成

- auth.jsの雛形作成
- ルーティング設定

# This is the commit message #2:

feat: パスワードハッシュを実装

- bcryptを導入
- ハッシュ化・検証関数を追加

# This is the commit message #3:

feat: ログインエンドポイントを追加

# This is the commit message #4:

feat: 認証ミドルウェアを実装

# ↑ これらを編集して、以下のような統合メッセージにする:

feat: ユーザー認証機能を実装

- 認証ロジックの骨格作成
- bcryptによるパスワードハッシュ化
- ログインエンドポイント (/api/login)
- JWT認証ミドルウェア
```

```
Before:                        After:
a1b2c3d feat: 骨格            xyz789 feat: ユーザー認証機能を実装
d4e5f6a feat: ハッシュ          (4つのコミットが1つに統合)
b7c8d9e feat: エンドポイント
e0f1a2b feat: ミドルウェア
```

### 2.2 fixup -- コミットの統合（メッセージ破棄）

```bash
pick a1b2c3d feat: ユーザー認証機能を実装
fixup d4e5f6a fix: テストの修正
fixup b7c8d9e fix: lint エラーの修正

# → d4e5f6a と b7c8d9e の内容は a1b2c3d に統合されるが、
#   コミットメッセージは a1b2c3d のものだけが残る
```

#### fixup -C オプション（Git 2.32+）

```bash
# fixup -C: fixupだがメッセージは「fixupコミット側」のものを使用
pick a1b2c3d feat: 認証機能（仮メッセージ）
fixup -C d4e5f6a feat: ユーザー認証機能を完全実装

# → 変更は統合されるが、メッセージはd4e5f6aのものが使われる
# → メッセージを後から改善したい場合に便利

# fixup -c: fixup -C と同じだが、エディタでメッセージ編集も可能
pick a1b2c3d feat: 認証機能（仮）
fixup -c d4e5f6a feat: 改善メッセージ
# → エディタが開き、d4e5f6aのメッセージをベースに編集できる
```

### 2.3 reword -- メッセージのみ変更

```bash
reword a1b2c3d feat: auht functon   # typo!
pick d4e5f6a fix: バグ修正

# → エディタが開き、a1b2c3d のメッセージだけを編集できる
# → ファイルの内容は変更されない
```

#### rewordの実務的ユースケース

```bash
# ユースケース1: Conventional Commitsの修正
reword a1b2c3d update: ログイン画面   # ← typeが不正
# → "feat: ログイン画面のUI実装" に修正

# ユースケース2: チケット番号の追加
reword d4e5f6a fix: メモリリークを修正
# → "fix(JIRA-1234): メモリリークを修正" に修正

# ユースケース3: 複数コミットのメッセージを一括修正
reword a1b2c3d feat: 機能A
reword d4e5f6a feat: 機能B
reword b7c8d9e feat: 機能C
# → 3つのコミットメッセージを順番に編集
```

### 2.4 edit -- コミットを修正するために停止

```bash
pick a1b2c3d feat: 認証機能
edit d4e5f6a feat: API実装    # ← ここで停止
pick b7c8d9e feat: UI実装

# rebase実行後、d4e5f6a の時点で停止
$ vim src/api.js              # ファイルを修正
$ git add src/api.js
$ git commit --amend          # コミットを修正
$ git rebase --continue       # rebase再開
```

#### editで追加のコミットを挿入する

```bash
# editで停止した後、新しいコミットを挿入することも可能
$ git rebase -i HEAD~3

pick a1b2c3d feat: 認証機能
edit d4e5f6a feat: API実装
pick b7c8d9e feat: UI実装

# d4e5f6a で停止後:
$ vim src/middleware.js
$ git add src/middleware.js
$ git commit -m "feat: ミドルウェアを追加"   # 新しいコミットを挿入
$ git rebase --continue
# → 結果: a1b2c3d → d4e5f6a → [新コミット] → b7c8d9e
```

### 2.5 exec -- シェルコマンドの実行

```bash
pick a1b2c3d feat: 認証機能
exec npm test                  # テストを実行
pick d4e5f6a feat: API実装
exec npm test                  # テストを実行

# → 各コミットの後でテストを実行し、失敗したら停止
# → 全コミットでテストが通ることを保証
```

#### execの応用パターン

```bash
# 全コミットにexecを自動挿入（--exec オプション）
$ git rebase -i --exec "npm test" main
# → 各コミットの後に自動で "exec npm test" が挿入される

# 複数コマンドの実行
$ git rebase -i --exec "npm run build && npm test" main

# 各コミットでビルド・テスト・lint全てが通ることを確認
pick a1b2c3d feat: 認証機能
exec npm run build && npm test && npm run lint
pick d4e5f6a feat: API実装
exec npm run build && npm test && npm run lint

# exec が失敗した場合
# → rebaseが停止する
# → 修正して git rebase --continue するか
# → git rebase --abort で全て取り消し
```

### 2.6 break -- 一時停止

```bash
pick a1b2c3d feat: 認証機能
break                          # ← ここで一時停止
pick d4e5f6a feat: API実装
pick b7c8d9e feat: UI実装

# breakで停止後、自由に作業可能
$ git log --oneline -3         # 現在の状態を確認
$ git diff HEAD~1              # 直前のコミットの差分を確認
$ git rebase --continue        # 確認後に続行
```

### 2.7 drop -- コミットの削除

```bash
pick a1b2c3d feat: 認証機能
drop d4e5f6a WIP: 一時保存      # ← このコミットを削除
pick b7c8d9e feat: API実装

# ※ todoリストから行を削除するのと同じ効果
# ※ dropは明示的に意図を示すので、行削除より安全
```

### 2.8 label / reset / merge -- マージ構造の再構築

```bash
# --rebase-merges 使用時に現れるコマンド
label onto

# Branch: feature-a
reset onto
pick a1b2c3d feat: 機能A
label feature-a

# Branch: feature-b
reset onto
pick d4e5f6a feat: 機能B
label feature-b

# Merge
reset feature-a
merge -C e0f1a2b feature-b  # マージコミットの再作成
```

---

## 3. autosquashワークフロー

### 3.1 fixup!とsquash!コミット

```bash
# 通常の開発フロー
$ git commit -m "feat: ユーザー認証"   # a1b2c3d

# 後から修正が必要になった場合
$ git commit --fixup=a1b2c3d           # fixup! feat: ユーザー認証
# または
$ git commit --squash=a1b2c3d          # squash! feat: ユーザー認証

# autosquashでrebase
$ git rebase -i --autosquash main
# → fixup!/squash! コミットが自動的に対象コミットの直下に移動
```

```
┌─────────────────────────────────────────────────────┐
│  autosquash の自動並べ替え                           │
│                                                     │
│  コミット履歴:                                       │
│  1. a1b2c3d feat: ユーザー認証                      │
│  2. d4e5f6a feat: API実装                           │
│  3. b7c8d9e fixup! feat: ユーザー認証               │
│  4. e0f1a2b feat: UI実装                            │
│                                                     │
│  git rebase -i --autosquash 後のtodoリスト:         │
│  pick   a1b2c3d feat: ユーザー認証                  │
│  fixup  b7c8d9e fixup! feat: ユーザー認証  ← 移動! │
│  pick   d4e5f6a feat: API実装                       │
│  pick   e0f1a2b feat: UI実装                        │
└─────────────────────────────────────────────────────┘
```

### 3.2 autosquashの自動有効化

```bash
# グローバル設定で常にautosquashを有効にする
$ git config --global rebase.autosquash true

# 個別にautosquashを無効にしたい場合
$ git rebase -i --no-autosquash main
```

### 3.3 --fixup=amend: と --fixup=reword:（Git 2.32+）

```bash
# --fixup=amend: コードの変更とメッセージの変更を同時に行う
$ git commit --fixup=amend:a1b2c3d
# → "amend! feat: ユーザー認証" というコミットが作成される
# → autosquash時にコードの変更を統合し、メッセージ編集画面が開く

# --fixup=reword: メッセージの変更のみ（コードの変更なし）
$ git commit --allow-empty --fixup=reword:a1b2c3d
# → autosquash時にメッセージ編集画面が開く（コードは変更なし）
```

### 3.4 実践: fixupワークフローの全体像

```bash
# Step 1: 機能を実装してコミット
$ git commit -m "feat: ユーザープロフィール画面"

# Step 2: コードレビューで指摘を受ける
# 「バリデーションが不足している」→ 修正
$ vim src/profile.js
$ git add src/profile.js
$ git log --oneline -5  # 対象コミットのSHA-1を確認
# a1b2c3d feat: ユーザープロフィール画面
$ git commit --fixup=a1b2c3d

# Step 3: さらに別の指摘
# 「エラーハンドリングを追加して」→ 修正
$ vim src/profile.js
$ git add src/profile.js
$ git commit --fixup=a1b2c3d

# Step 4: レビュー対応完了、コミットを整理
$ git rebase -i --autosquash main
# → fixup!コミットが自動的に a1b2c3d の直下に配置される
# → 保存してエディタを閉じると、3つのコミットが1つに統合される

# Step 5: force-pushでPRを更新
$ git push --force-with-lease origin feature/profile
```

---

## 4. --update-refs（Git 2.38+）

複数のブランチが積み重なっている場合、rebaseで全てのrefを同時に更新できる。

```bash
# スタックドブランチ構成
# main → feature/base → feature/api → feature/ui

$ git rebase -i --update-refs main
# → feature/base, feature/api のrefも自動的に更新される
```

```
┌────────────────────────────────────────────────────┐
│  --update-refs なし                                 │
│                                                    │
│  Before rebase:                                    │
│  main ── A ── B(feature/base) ── C ── D(feature/api)│
│                                                    │
│  After rebase (feature/api のみ):                  │
│  main ── A ── B(feature/base) ── C ── D            │
│            \                                       │
│             B'── C'── D'(feature/api)              │
│  → feature/base が古い位置のまま！                  │
│                                                    │
├────────────────────────────────────────────────────┤
│  --update-refs あり                                 │
│                                                    │
│  After rebase:                                     │
│  main ── A ── B'(feature/base) ── C'── D'(feature/api)│
│  → 全てのrefが正しく更新される                      │
└────────────────────────────────────────────────────┘
```

```bash
# デフォルトで有効にする
$ git config --global rebase.updateRefs true
```

### 4.1 スタックドブランチの実践ワークフロー

```bash
# Step 1: ベース機能のブランチ
$ git checkout -b feature/auth main
# ... 実装 ...
$ git commit -m "feat: 認証基盤"

# Step 2: 認証の上にAPI機能を積む
$ git checkout -b feature/api feature/auth
# ... 実装 ...
$ git commit -m "feat: API実装"

# Step 3: APIの上にUI機能を積む
$ git checkout -b feature/ui feature/api
# ... 実装 ...
$ git commit -m "feat: UI実装"

# Step 4: mainが進んだのでrebase
$ git checkout feature/ui
$ git rebase -i --update-refs main
# → feature/auth, feature/api のポインタも自動更新

# Step 5: 各ブランチのPRが正しい差分になる
$ git log --oneline main..feature/auth    # 認証のみ
$ git log --oneline feature/auth..feature/api  # APIのみ
$ git log --oneline feature/api..feature/ui    # UIのみ
```

### 4.2 todoリストでのupdate-ref

`--update-refs`を使用すると、todoリストに`update-ref`コマンドが自動挿入される。

```bash
pick a1b2c3d feat: 認証基盤
update-ref refs/heads/feature/auth    # ← ここでブランチポインタ更新
pick d4e5f6a feat: API実装
update-ref refs/heads/feature/api     # ← ここでブランチポインタ更新
pick b7c8d9e feat: UI実装

# update-refの行を削除すると、そのブランチは更新されない
# update-refの位置を移動させることも可能
```

---

## 5. --rebase-merges によるマージ構造の保持

### 5.1 基本的な使い方

```bash
# マージコミットを含む範囲をrebase
$ git rebase -i --rebase-merges main

# todoリストの例:
label onto

# Branch: feature-auth
reset onto
pick a1b2c3d feat: 認証基盤
pick d4e5f6a feat: ログイン画面
label feature-auth

# Branch: feature-api
reset onto
pick b7c8d9e feat: API基盤
pick e0f1a2b feat: エンドポイント
label feature-api

reset feature-auth
merge -C f2a3b4c feature-api  # feature-apiをマージ
pick 1234567 feat: 統合テスト
```

### 5.2 マージ構造の編集

```bash
# マージ戦略を変更
reset feature-auth
merge -C f2a3b4c feature-api   # 通常のマージ

# ↓ --no-ff を明示的に指定
reset feature-auth
merge -C f2a3b4c --no-ff feature-api

# マージコミットのメッセージを変更
reset feature-auth
merge -c f2a3b4c feature-api   # -c (小文字) でメッセージ編集画面が開く
```

### 5.3 --rebase-merges の廃止された前身

```bash
# 旧オプション（非推奨、Git 2.22で廃止）
$ git rebase -i --preserve-merges main  # ← 使わないこと

# 新オプション（Git 2.18+、推奨）
$ git rebase -i --rebase-merges main
```

---

## 6. 実践的なワークフロー例

### 6.1 PR用のコミット整理

```bash
# 開発中の雑多なコミット履歴
$ git log --oneline main..HEAD
e0f1a2b fix: lint
b7c8d9e wip
d4e5f6a fix: typo
a1b2c3d feat: ユーザー登録
9876543 feat: メール送信
1234567 feat: バリデーション

# 整理のためのtodoリスト
$ git rebase -i main

pick 1234567 feat: バリデーション
pick 9876543 feat: メール送信
pick a1b2c3d feat: ユーザー登録
fixup d4e5f6a fix: typo            # typo修正をユーザー登録に統合
fixup b7c8d9e wip                  # wipもユーザー登録に統合
fixup e0f1a2b fix: lint            # lint修正もユーザー登録に統合

# 結果: 3つのクリーンなコミットになる
# 1234567' feat: バリデーション
# 9876543' feat: メール送信
# xxxxxxx  feat: ユーザー登録（typo/wip/lint修正を含む）
```

### 6.2 コミットの分割

```bash
# 1つの大きなコミットを複数に分割
$ git rebase -i HEAD~3

edit a1b2c3d feat: 認証とUI（分割したい）
pick d4e5f6a feat: テスト

# a1b2c3d で停止後:
$ git reset HEAD~1                    # コミットを取り消し（変更は保持）
$ git add src/auth.js src/middleware.js
$ git commit -m "feat: 認証ロジック"
$ git add src/components/Login.jsx
$ git commit -m "feat: ログインUI"
$ git rebase --continue
```

### 6.3 コミットの順序変更

```bash
# todoリスト内で行を入れ替えるだけ
$ git rebase -i HEAD~4

# Before:
pick a1b2c3d feat: テスト追加
pick d4e5f6a feat: 実装
pick b7c8d9e feat: 型定義

# After (テストを最後に移動):
pick d4e5f6a feat: 実装
pick b7c8d9e feat: 型定義
pick a1b2c3d feat: テスト追加

# ※ 順序変更はコンフリクトの原因になりやすい
# ※ 依存関係がないコミット同士で行うのが安全
```

### 6.4 複数のPRにコミットを振り分ける

```bash
# 1つのブランチに混在した変更を2つのPRに分ける

# Step 1: 現在のブランチの状態を確認
$ git log --oneline main..HEAD
e0f1a2b feat: プロフィール画面のUI
b7c8d9e feat: プロフィール画面のAPI
d4e5f6a feat: ダッシュボードのUI
a1b2c3d feat: ダッシュボードのAPI

# Step 2: ダッシュボード用のブランチを作成
$ git checkout -b feature/dashboard main
$ git cherry-pick a1b2c3d d4e5f6a

# Step 3: プロフィール用のブランチを作成
$ git checkout -b feature/profile main
$ git cherry-pick b7c8d9e e0f1a2b

# Step 4: 元のブランチを削除
$ git branch -D feature/mixed
```

### 6.5 テストが全コミットで通ることを検証

```bash
# 全てのコミットでCIが通ることを確認してからPRを出す
$ git rebase -i --exec "npm run build && npm test" main

# ↓ 自動生成されるtodoリスト
pick a1b2c3d feat: 認証機能
exec npm run build && npm test
pick d4e5f6a feat: API実装
exec npm run build && npm test
pick b7c8d9e feat: UI実装
exec npm run build && npm test

# いずれかのコミットでテストが失敗した場合:
# → rebaseが停止
# → editで修正してから続行
```

### 6.6 機密情報を含むコミットの修正

```bash
# 誤ってAPIキーをコミットしてしまった場合
$ git rebase -i HEAD~5

edit a1b2c3d feat: API連携を追加    # ← APIキーが含まれるコミット
pick d4e5f6a feat: テスト追加

# a1b2c3d で停止後:
$ vim .env                            # APIキーを.envに移動
$ vim src/api.js                      # 環境変数から読むように修正
$ echo ".env" >> .gitignore           # .gitignoreに追加
$ git add .gitignore src/api.js
$ git rm --cached .env 2>/dev/null    # .envをトラッキングから除外
$ git commit --amend                  # コミットを修正
$ git rebase --continue

# ※ 既にpush済みの場合、force-pushしてもGitHubのキャッシュに残る可能性がある
# ※ 漏洩したAPIキーは必ずローテーション（再発行）すること
```

---

## 7. コンフリクト解決

### 7.1 rebase中のコンフリクト

```bash
# コンフリクトが発生した場合
$ git rebase -i main
# CONFLICT (content): Merge conflict in src/auth.js
# error: could not apply d4e5f6a... feat: API実装

# 現在の状態を確認
$ git status
# interactive rebase in progress; onto abc123
# You are currently rebasing branch 'feature' on 'abc123'.
#   (fix conflicts and then run "git rebase --continue")
#   (use "git rebase --skip" to skip this patch)
#   (use "git rebase --abort" to cancel the rebase)
#
# Unmerged paths:
#   both modified:   src/auth.js

# コンフリクトを解決
$ vim src/auth.js
# <<<<<<< HEAD
# ... rebase先の内容 ...
# =======
# ... 適用しようとしたコミットの内容 ...
# >>>>>>> d4e5f6a (feat: API実装)

# 解決後
$ git add src/auth.js
$ git rebase --continue
# → 次のコミットの処理に進む
```

### 7.2 コンフリクト解決の選択肢

```bash
# 選択肢1: コンフリクトを解決して続行
$ git add <resolved-files>
$ git rebase --continue

# 選択肢2: このコミットをスキップ
$ git rebase --skip
# → このコミットの変更は適用されない

# 選択肢3: rebase全体を中止
$ git rebase --abort
# → rebase開始前の状態に完全復帰

# 選択肢4: 特定のファイルでどちらかの内容を採用
$ git checkout --ours src/auth.js    # rebase先（onto）の内容を採用
$ git checkout --theirs src/auth.js  # 適用中のコミットの内容を採用
$ git add src/auth.js
$ git rebase --continue
```

### 7.3 rerere -- コンフリクト解決の記憶

```bash
# rerere（reuse recorded resolution）を有効化
$ git config --global rerere.enabled true

# 仕組み:
# 1. コンフリクトが発生 → 解決方法を記録
# 2. 同じコンフリクトが再発 → 自動的に同じ解決方法を適用
# → rebaseのやり直し時に同じコンフリクトを再度解決する必要がなくなる

# 記録された解決方法の確認
$ git rerere status
$ git rerere diff

# 記録をクリア
$ git rerere forget <pathspec>
$ git rerere gc  # 古い記録を削除
```

```
┌──────────────────────────────────────────────────────┐
│  rerere の動作フロー                                  │
│                                                      │
│  1回目のrebase:                                      │
│  コンフリクト発生 → 手動解決 → rerereが解決を記録    │
│                                                      │
│  2回目のrebase（やり直し時）:                        │
│  同じコンフリクト発生 → rerereが自動解決             │
│                                                      │
│  ※ rebase → abort → 修正 → 再rebase の繰り返しで威力発揮  │
│  ※ マージとrebaseを行き来する場合にも有効            │
└──────────────────────────────────────────────────────┘
```

---

## 8. リカバリー手法

### 8.1 rebase --abort

```bash
# rebase中に問題が発生した場合、いつでも中止できる
$ git rebase --abort
# → ORIG_HEADの状態に完全復帰
# → 作業ディレクトリ、インデックス、HEADの全てが元に戻る
```

### 8.2 reflogからの救出

```bash
# rebase完了後に「間違えた」と気づいた場合
$ git reflog
abc123 HEAD@{0}: rebase (finish): returning to refs/heads/feature
def456 HEAD@{1}: rebase (squash): feat: ...
789abc HEAD@{2}: rebase (start): checkout main
fedcba HEAD@{3}: commit: feat: ...    # ← rebase前の最後のcommit

# rebase前の状態に戻す
$ git reset --hard fedcba
# → rebase前の状態に完全復帰
```

### 8.3 ORIG_HEADの活用

```bash
# rebase直後なら ORIG_HEAD で簡単に戻れる
$ git rebase -i main
# ... rebase完了 ...

# 「やっぱりやめたい」
$ git reset --hard ORIG_HEAD
# → rebase開始前のHEADに戻る

# ※ ORIG_HEAD は destructive な操作（rebase, merge, reset）の前に自動保存される
# ※ 次の destructive 操作で上書きされるので注意
```

### 8.4 バックアップブランチの作成

```bash
# rebase前にバックアップブランチを作成する習慣
$ git branch backup/feature-before-rebase
$ git rebase -i main
# ... 作業 ...

# 問題があれば:
$ git reset --hard backup/feature-before-rebase
$ git branch -D backup/feature-before-rebase  # 不要になったら削除
```

---

## 9. アンチパターン

### アンチパターン1: push済みコミットのrebase

```bash
# NG: リモートにpush済みのコミットをrebase
$ git push origin feature
# ... 後からrebase ...
$ git rebase -i HEAD~5
$ git push --force origin feature
# → 他の開発者がpullできなくなる

# OK: force-with-leaseを使い、push前に確認
$ git push --force-with-lease origin feature
# → リモートが予期しない更新をされていたら拒否

# より安全: ローカル専用コミットのみrebase
$ git rebase -i origin/feature  # originにないコミットのみ対象
```

**理由**: rebaseはコミットのSHA-1を変更する。push済みコミットのSHA-1を変更すると、そのブランチを追跡している全ての開発者に影響する。

### アンチパターン2: マージコミットを含む範囲のrebase

```bash
# NG: マージコミットを含む範囲を通常のrebaseで処理
$ git rebase -i HEAD~10
# → マージコミットが消えて、線形履歴に変わってしまう

# OK: --rebase-merges でマージ構造を保持
$ git rebase -i --rebase-merges HEAD~10
# → todoリストにlabel, reset, mergeコマンドが追加される
```

**理由**: 通常のrebaseはマージコミットをスキップまたは線形化する。`--rebase-merges`を使うことで、マージの分岐・合流構造を保ったままrebaseが可能。

### アンチパターン3: 大量のコミットを一度にrebase

```bash
# NG: 100以上のコミットを一度にインタラクティブrebase
$ git rebase -i HEAD~150
# → コンフリクトが連鎖して収拾がつかなくなる

# OK: 段階的にrebaseする
$ git rebase -i HEAD~20    # まず直近20件を整理
$ git rebase -i HEAD~20    # 次の20件を整理
# → 小さな単位で段階的に処理

# OK: 機能単位でブランチを分割してからrebase
$ git rebase -i feature/base  # ベースブランチからの差分のみ
```

**理由**: 大量のコミットのrebaseはコンフリクトの連鎖を引き起こしやすい。段階的に処理することでリスクを最小化できる。

### アンチパターン4: 他人のコミットをsquashで潰す

```bash
# NG: 別の開発者のコミットを自分のコミットにsquash
pick a1b2c3d feat: Tanakaさんの実装
squash d4e5f6a feat: 自分の修正
# → Tanakaさんの貢献が履歴から消える

# OK: Co-authored-byを使う or 別コミットとして保持
pick a1b2c3d feat: Tanakaさんの実装
pick d4e5f6a feat: 自分の修正
# → 両方の貢献が履歴に残る
```

**理由**: オープンソースでは特に、各開発者の貢献を正確に記録することが重要。他人のコミットを自分のものに統合するのは不適切。

---

## 10. パフォーマンスとGit設定

### 10.1 rebase高速化の設定

```bash
# rebase時のmerge-backendを設定（Git 2.33+）
$ git config --global rebase.backend merge
# → 'apply' バックエンドより高速で、rename検出も優れている

# rebase時のstat表示を無効化（大規模リポジトリで有効）
$ git config --global rebase.stat false

# autostashを有効にして手動stashの手間を省く
$ git config --global rebase.autostash true
```

### 10.2 推奨するGit設定まとめ

```bash
# rebase関連の推奨設定
$ git config --global rebase.autosquash true     # fixup!/squash!の自動並べ替え
$ git config --global rebase.updateRefs true     # スタックドブランチの自動更新
$ git config --global rebase.autostash true      # 未コミット変更の自動stash
$ git config --global rerere.enabled true        # コンフリクト解決の記憶
$ git config --global pull.rebase true           # pull時にmergeではなくrebase
$ git config --global fetch.prune true           # fetch時に削除済みブランチを自動削除
```

### 10.3 エイリアスの設定

```bash
# rebase関連の便利エイリアス
$ git config --global alias.ri "rebase -i"
$ git config --global alias.rc "rebase --continue"
$ git config --global alias.ra "rebase --abort"
$ git config --global alias.rs "rebase --skip"
$ git config --global alias.rim "rebase -i main"
$ git config --global alias.fixup "commit --fixup"

# 使用例
$ git ri HEAD~5          # インタラクティブrebase
$ git ri main            # mainからのrebase
$ git fixup abc123       # fixupコミット作成
$ git rc                 # rebase続行
$ git ra                 # rebase中止
```

---

## 11. チーム開発でのrebase運用

### 11.1 rebase vs merge 戦略

```
┌──────────────────────────────────────────────────────┐
│  rebase戦略                                          │
│  main ── A ── B ── C ── D ── E ── F                  │
│  → 直線的で読みやすい履歴                            │
│  → コミットの因果関係が明確                          │
│  → bisectが効率的                                    │
│                                                      │
│  merge戦略                                           │
│  main ── A ── B ──────── M1 ──────── M2              │
│            \           /     \      /                │
│             C ── D ──/       E ── F                  │
│  → ブランチの分岐・合流が明確                        │
│  → 変更の文脈が保持される                            │
│  → コンフリクト解決が一度で済む                      │
│                                                      │
│  推奨: 個人作業はrebase、チーム統合はmerge            │
└──────────────────────────────────────────────────────┘
```

### 11.2 PRマージ前のコミット整理ルール

```bash
# チームで合意すべきルール:

# ルール1: PRマージ前にsquash or rebaseで整理
# → WIP、fixup、typoなどの一時コミットを統合
# → 論理的に意味のある単位にまとめる

# ルール2: force-pushのルール
# → force-push-with-lease のみ許可
# → main/developへのforce-push は禁止
# → レビュー後のforce-pushはレビュアーに通知

# ルール3: コミットメッセージの形式
# → Conventional Commits準拠
# → 日本語/英語の統一
```

### 11.3 GitHub/GitLabでのSquash Mergeとの使い分け

```bash
# GitHubの「Squash and merge」ボタン:
# → PRの全コミットを1つにまとめてmainにマージ
# → 個別のコミット履歴はmainに残らない
# → PRのタイトルがコミットメッセージになる

# ローカルでの手動rebase + 通常のmerge:
# → コミットを整理しつつ、複数のコミットとしてmainに残す
# → 論理的に意味のある複数コミットを保持できる
# → より細かい制御が可能

# 判断基準:
# - 小さなPR（1-2コミット程度）: Squash Merge で十分
# - 大きなPR（複数の論理的変更）: 手動rebaseで整理後、通常のmerge
```

---

## 12. FAQ

### Q1. rebase中にコンフリクトが発生して収拾がつかなくなった場合は？

**A1.** `git rebase --abort`で**rebase開始前の状態に完全に復帰**できます。作業ディレクトリ、インデックス、HEADの全てが元に戻ります。rebaseで途中まで処理したコミットも全て巻き戻されます。

```bash
$ git rebase --abort    # 全てを元に戻す
$ git reflog            # 念のため元の状態を確認
```

### Q2. rebase後に「間違えた」と気づいた場合、元に戻せるか？

**A2.** はい、reflogを使って復元できます。

```bash
# rebase前のHEADの位置をreflogで確認
$ git reflog
abc123 HEAD@{0}: rebase (finish): returning to refs/heads/feature
def456 HEAD@{1}: rebase (squash): feat: ...
789abc HEAD@{2}: rebase (start): checkout main
fedcba HEAD@{3}: commit: feat: ...    # ← rebase前の最後のcommit

# rebase前の状態に戻す
$ git reset --hard fedcba
```

### Q3. `--autosquash`と`--autostash`の違いは何か？

**A3.** 全く異なる機能です。

| 機能            | 説明                                                   |
|-----------------|--------------------------------------------------------|
| `--autosquash`  | `fixup!`/`squash!`コミットをtodoリスト内で自動並べ替え |
| `--autostash`   | rebase開始前に未コミットの変更を自動stash、終了後にpop |

```bash
# 未コミットの変更がある状態でrebase
$ git rebase -i --autostash main
# → 自動で stash → rebase → stash pop
```

### Q4. rebase中にeditで停止した際、どのコミットの状態にいるのか？

**A4.** editで停止した時点では、**そのコミットが既に適用された状態**です。つまりHEADはeditに指定したコミットを指しています。`git commit --amend`でそのコミットを修正するか、新しいコミットを追加して`git rebase --continue`で続行します。

```bash
$ git rebase -i HEAD~3
# edit a1b2c3d feat: 何かの実装

# 停止後の状態:
$ git log --oneline -1
# a1b2c3d feat: 何かの実装    ← このコミットが適用済み
$ git status
# interactive rebase in progress
```

### Q5. squashとfixupはどう使い分けるべきか？

**A5.** 以下の基準で使い分けます。

| 状況 | 推奨コマンド | 理由 |
|------|-------------|------|
| 複数の実装を1つのコミットにまとめたい | squash | メッセージを統合・編集できる |
| typo修正、lint修正など小さな修正 | fixup | メッセージは親コミットのものを維持 |
| コードレビュー後の修正 | fixup + autosquash | 修正を元のコミットに自動統合 |
| メッセージを後から改善したい | fixup -C | fixupコミット側のメッセージを採用 |

### Q6. `git rebase --onto`の使い方は？

**A6.** `--onto`は「コミットの移植先」を指定するオプションで、3つの引数を取ります。

```bash
# 構文: git rebase --onto <新しいベース> <古いベース> <ブランチ>

# ユースケース1: ブランチの付け替え
# Before: main → feature-a → feature-b
# After:  main → feature-b（feature-aを飛ばす）
$ git rebase --onto main feature-a feature-b

# ユースケース2: 特定範囲のコミットだけを移植
# feature ブランチの最新3コミットだけをmainに移植
$ git rebase --onto main HEAD~3 feature

# ユースケース3: 古いベースブランチからの切り替え
# develop → feature だったのを main → feature に変更
$ git rebase --onto main develop feature
```

```
┌──────────────────────────────────────────────────────┐
│  --onto の動作イメージ                                │
│                                                      │
│  Before:                                             │
│  main ── A ── B(feature-a) ── C ── D(feature-b)     │
│                                                      │
│  $ git rebase --onto main feature-a feature-b        │
│                                                      │
│  After:                                              │
│  main ── A ── C'── D'(feature-b)                     │
│            \                                         │
│             B(feature-a)                              │
│  → feature-a以降のコミット(C,D)をmainに直接移植      │
└──────────────────────────────────────────────────────┘
```

### Q7. インタラクティブrebaseとnon-interactiveなrebaseの違いは？

**A7.** 動作原理は同じですが、制御の粒度が異なります。

```bash
# non-interactive rebase
$ git rebase main
# → mainからHEADまでの全コミットを自動的にpick
# → コンフリクト以外は介入なしで完了

# interactive rebase
$ git rebase -i main
# → todoリストが表示され、各コミットの処理を個別に指定可能
# → squash, fixup, reword, edit, drop, exec が使える

# ※ non-interactive でもautosquashは適用される（設定次第）
# ※ --exec はnon-interactiveでも使用可能
$ git rebase --exec "npm test" main
```

---

## まとめ

| 概念           | 要点                                                          |
|----------------|---------------------------------------------------------------|
| pick           | コミットをそのまま使用                                        |
| squash         | 直前のコミットに統合、メッセージ編集可能                      |
| fixup          | 直前のコミットに統合、メッセージは直前のものを維持            |
| fixup -C       | fixupだがメッセージはfixupコミット側を採用（Git 2.32+）       |
| reword         | コミットメッセージのみ変更                                    |
| edit           | コミットの時点で停止、修正や分割が可能                        |
| exec           | 任意のシェルコマンドを実行                                    |
| break          | 一時停止、確認後にcontinueで続行                              |
| drop           | コミットを明示的に削除                                        |
| autosquash     | `fixup!`/`squash!`プレフィックスで自動並べ替え               |
| --update-refs  | スタックドブランチのrefを自動更新（Git 2.38+）               |
| --rebase-merges| マージ構造を保持したままrebase                                |
| --onto         | コミットの移植先を明示的に指定                                |
| rerere         | コンフリクト解決方法を記憶し再利用                            |
| ORIG_HEAD      | rebase前のHEAD位置を自動保存                                  |

---

## 次に読むべきガイド

- [マージアルゴリズム](../00-git-internals/02-merge-algorithms.md) -- rebase時のコンフリクト解決の原理
- [bisect/blame](./02-bisect-blame.md) -- 整理された履歴でのバグ特定
- [Git Hooks](./03-hooks-automation.md) -- rebase時のhook連携

---

## 参考文献

1. **Pro Git Book** -- "Rewriting History" https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History
2. **Git公式ドキュメント** -- `git-rebase` https://git-scm.com/docs/git-rebase
3. **GitHub Blog** -- "Git Tips: `--update-refs`" https://github.blog/2022-10-03-highlights-from-git-2-38/#rebase-update-refs
4. **Git Release Notes 2.32** -- fixup amend/reword https://github.com/git/git/blob/master/Documentation/RelNotes/2.32.0.txt
5. **Git Release Notes 2.38** -- update-refs https://github.com/git/git/blob/master/Documentation/RelNotes/2.38.0.txt
