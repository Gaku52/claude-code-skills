# インタラクティブRebase

> `git rebase -i`を使いこなし、コミット履歴の整理（squash、fixup、reword、edit、drop）を安全に行うためのテクニックと運用ルールを解説する。

## この章で学ぶこと

1. **インタラクティブRebaseの基本操作** -- 各コマンド（pick, squash, fixup, reword, edit, drop）の動作と使い分け
2. **安全なコミット履歴の書き換え** -- autosquash、fixupコミット、`--update-refs`による効率的なワークフロー
3. **トラブル発生時の復旧方法** -- rebase中のコンフリクト対処、`--abort`、reflogからの救出
4. **高度なrebaseテクニック** -- `--rebase-merges`、`exec`コマンド、スタックドブランチ運用
5. **チーム開発でのrebase運用ルール** -- 安全なforce-push、レビュー前の履歴整理フロー


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
