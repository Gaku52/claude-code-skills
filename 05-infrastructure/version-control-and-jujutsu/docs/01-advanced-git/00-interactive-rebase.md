# インタラクティブRebase

> `git rebase -i`を使いこなし、コミット履歴の整理（squash、fixup、reword、edit、drop）を安全に行うためのテクニックと運用ルールを解説する。

## この章で学ぶこと

1. **インタラクティブRebaseの基本操作** — 各コマンド（pick, squash, fixup, reword, edit, drop）の動作と使い分け
2. **安全なコミット履歴の書き換え** — autosquash、fixupコミット、`--update-refs`による効率的なワークフロー
3. **トラブル発生時の復旧方法** — rebase中のコンフリクト対処、`--abort`、reflogからの救出

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
```

### 1.2 todoリストの構造

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

---

## 2. 各コマンドの詳細

### 2.1 squash — コミットの統合（メッセージ編集あり）

```bash
# Before: 4つの細かいコミット
pick a1b2c3d feat: 認証機能の骨格を作成
squash d4e5f6a feat: パスワードハッシュを実装
squash b7c8d9e feat: ログインエンドポイントを追加
squash e0f1a2b feat: 認証ミドルウェアを実装

# → エディタが開き、4つのコミットメッセージが表示される
# → 統合後のメッセージを編集して保存
```

```
Before:                        After:
a1b2c3d feat: 骨格            xyz789 feat: ユーザー認証機能を実装
d4e5f6a feat: ハッシュ          (4つのコミットが1つに統合)
b7c8d9e feat: エンドポイント
e0f1a2b feat: ミドルウェア
```

### 2.2 fixup — コミットの統合（メッセージ破棄）

```bash
pick a1b2c3d feat: ユーザー認証機能を実装
fixup d4e5f6a fix: テストの修正
fixup b7c8d9e fix: lint エラーの修正

# → d4e5f6a と b7c8d9e の内容は a1b2c3d に統合されるが、
#   コミットメッセージは a1b2c3d のものだけが残る
```

### 2.3 reword — メッセージのみ変更

```bash
reword a1b2c3d feat: auht functon   # typo!
pick d4e5f6a fix: バグ修正

# → エディタが開き、a1b2c3d のメッセージだけを編集できる
# → ファイルの内容は変更されない
```

### 2.4 edit — コミットを修正するために停止

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

### 2.5 exec — シェルコマンドの実行

```bash
pick a1b2c3d feat: 認証機能
exec npm test                  # テストを実行
pick d4e5f6a feat: API実装
exec npm test                  # テストを実行

# → 各コミットの後でテストを実行し、失敗したら停止
# → 全コミットでテストが通ることを保証
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

---

## 5. 実践的なワークフロー例

### 5.1 PR用のコミット整理

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

### 5.2 コミットの分割

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

---

## 6. アンチパターン

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

---

## 7. FAQ

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

---

## まとめ

| 概念           | 要点                                                          |
|----------------|---------------------------------------------------------------|
| pick           | コミットをそのまま使用                                        |
| squash         | 直前のコミットに統合、メッセージ編集可能                      |
| fixup          | 直前のコミットに統合、メッセージは直前のものを維持            |
| reword         | コミットメッセージのみ変更                                    |
| edit           | コミットの時点で停止、修正や分割が可能                        |
| exec           | 任意のシェルコマンドを実行                                    |
| autosquash     | `fixup!`/`squash!`プレフィックスで自動並べ替え               |
| --update-refs  | スタックドブランチのrefを自動更新（Git 2.38+）               |
| --rebase-merges| マージ構造を保持したままrebase                                |

---

## 次に読むべきガイド

- [マージアルゴリズム](../00-git-internals/02-merge-algorithms.md) — rebase時のコンフリクト解決の原理
- [bisect/blame](./02-bisect-blame.md) — 整理された履歴でのバグ特定
- [Git Hooks](./03-hooks-automation.md) — rebase時のhook連携

---

## 参考文献

1. **Pro Git Book** — "Rewriting History" https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History
2. **Git公式ドキュメント** — `git-rebase` https://git-scm.com/docs/git-rebase
3. **GitHub Blog** — "Git Tips: `--update-refs`" https://github.blog/2022-10-03-highlights-from-git-2-38/#rebase-update-refs
