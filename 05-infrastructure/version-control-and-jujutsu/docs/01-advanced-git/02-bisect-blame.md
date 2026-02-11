# bisect/blame

> `git bisect`によるバグ原因コミットの二分探索と、`git blame`によるコード変更の追跡手法を解説し、大規模プロジェクトでのデバッグ効率を飛躍的に向上させる。

## この章で学ぶこと

1. **git bisectの二分探索アルゴリズム** — 手動・自動bisectの使い方と効率的なバグ特定手順
2. **git blameの高度な活用** — 行単位の変更追跡、コード移動の検出、ignore-revs
3. **bisectとblameの組み合わせ戦略** — 実践的なデバッグワークフロー

---

## 1. git bisect — 二分探索でバグを特定

### 1.1 二分探索の原理

```
┌────────────────────────────────────────────────────┐
│  二分探索（Binary Search）の原理                    │
│                                                    │
│  1000コミットの中からバグ混入コミットを特定する場合 │
│                                                    │
│  線形探索: 最悪 1000 回のテスト                     │
│  二分探索: 最悪 ceil(log2(1000)) = 10 回のテスト   │
│                                                    │
│  good                              bad             │
│  v                                  v              │
│  o---o---o---o---o---o---o---o---o---o              │
│  1                                  1000           │
│                                                    │
│  Step 1: 500番目をテスト → bad                     │
│  good              bad                             │
│  v                  v                              │
│  o---o---o---o---o---o                             │
│  1                  500                            │
│                                                    │
│  Step 2: 250番目をテスト → good                    │
│           good     bad                             │
│            v        v                              │
│  o---o---o---o---o---o                             │
│          250       500                             │
│                                                    │
│  ... 10ステップで特定完了                           │
└────────────────────────────────────────────────────┘
```

### 1.2 手動bisect

```bash
# 1. bisect開始
$ git bisect start

# 2. 現在のHEAD（バグあり）をbadとマーク
$ git bisect bad

# 3. 正常だったコミットをgoodとマーク
$ git bisect good v1.0.0
# Bisecting: 512 revisions left to test after this (roughly 9 steps)
# [abc123...] feat: some commit message

# 4. テストして結果をマーク（繰り返し）
$ npm test
# テスト失敗
$ git bisect bad
# Bisecting: 256 revisions left to test after this (roughly 8 steps)

$ npm test
# テスト成功
$ git bisect good
# Bisecting: 128 revisions left to test after this (roughly 7 steps)

# ... 繰り返し ...

# 5. 原因コミットが特定される
# abc123def456789abcdef is the first bad commit
# commit abc123def456789abcdef
# Author: Developer <dev@example.com>
# Date:   Mon Feb 10 15:30:00 2025 +0900
#
#     feat: add caching layer

# 6. bisect終了（元のHEADに戻る）
$ git bisect reset
```

### 1.3 自動bisect

```bash
# テストスクリプトを指定して自動実行
$ git bisect start HEAD v1.0.0
$ git bisect run npm test
# → 自動的にテストを実行し、exit code 0=good, 非0=bad として判定

# カスタムスクリプトでの自動bisect
$ git bisect run ./test-specific-bug.sh

# test-specific-bug.sh の例:
#!/bin/bash
npm run build 2>/dev/null || exit 125  # ビルド失敗はスキップ
node -e "
  const result = require('./dist/auth').validate('test@example.com');
  process.exit(result ? 0 : 1);
"
```

**exit codeの意味**:

| exit code | 意味                                          |
|-----------|-----------------------------------------------|
| 0         | good（このコミットにバグなし）                |
| 1-124, 126, 127 | bad（このコミットにバグあり）           |
| 125       | skip（このコミットはテスト不可能）            |
| 128-      | bisectを中断                                  |

### 1.4 bisectの高度な使い方

```bash
# 特定のパスに限定してbisect
$ git bisect start -- src/auth/ tests/auth/
# → 指定パスに変更があるコミットのみを対象にする

# 用語のカスタマイズ（新旧で使う場合）
$ git bisect start --term-old=slow --term-new=fast
$ git bisect slow v1.0.0
$ git bisect fast HEAD
# → パフォーマンス改善コミットの特定にも使える

# bisectログの保存と再実行
$ git bisect log > bisect-log.txt
$ git bisect replay bisect-log.txt

# 特定のコミットをスキップ
$ git bisect skip
# → ビルドできないコミットなどをスキップ
```

---

## 2. git blame — 行単位の変更追跡

### 2.1 基本的な使い方

```bash
# ファイル全体のblame
$ git blame src/auth.js
a1b2c3d4 (Gaku    2025-01-15 10:30:00 +0900  1) const bcrypt = require('bcrypt');
d4e5f6a7 (Tanaka  2025-02-01 14:20:00 +0900  2) const jwt = require('jsonwebtoken');
a1b2c3d4 (Gaku    2025-01-15 10:30:00 +0900  3)
b7c8d9e0 (Suzuki  2025-02-10 09:15:00 +0900  4) async function login(email, password) {
d4e5f6a7 (Tanaka  2025-02-01 14:20:00 +0900  5)   const user = await User.findByEmail(email);

# 行範囲を指定
$ git blame -L 10,20 src/auth.js
$ git blame -L '/function login/,/^}/' src/auth.js  # 正規表現で範囲指定

# 詳細表示（コミットメッセージの1行目も表示）
$ git blame --show-description src/auth.js
```

### 2.2 コード移動・コピーの検出

```bash
# -M: 同一ファイル内のコード移動を検出
$ git blame -M src/auth.js
# → 同じファイル内で移動された行の元のコミットを表示

# -C: ファイル間のコード移動を検出
$ git blame -C src/auth.js
# → 別ファイルからコピーされた行の元のコミットを表示

# -C -C: さらに広範囲のコピー検出（同一コミット内）
$ git blame -C -C src/auth.js

# -C -C -C: 全コミットにわたるコピー検出
$ git blame -C -C -C src/auth.js
```

```
┌────────────────────────────────────────────────────┐
│  -C オプションのレベル                              │
│                                                    │
│  -C (1回):                                         │
│    同じコミットで変更されたファイルからのコピー検出 │
│                                                    │
│  -C -C (2回):                                      │
│    任意のコミットでのファイル作成時のコピー検出     │
│                                                    │
│  -C -C -C (3回):                                   │
│    全コミットにわたるコピー検出（最も遅いが完全）   │
│                                                    │
│  処理時間: -C < -C -C < -C -C -C                   │
│  検出範囲: -C < -C -C < -C -C -C                   │
└────────────────────────────────────────────────────┘
```

### 2.3 ignore-revs — フォーマット変更の除外

```bash
# 大規模なコードフォーマット変更のコミットを除外
$ git blame --ignore-rev abc123def456
$ git blame --ignore-revs-file .git-blame-ignore-revs

# .git-blame-ignore-revs ファイルの作成
$ cat .git-blame-ignore-revs
# Prettier導入による全ファイルフォーマット
abc123def456789abcdef1234567890abcdef1234

# ESLint auto-fix
def456789abcdef1234567890abcdef1234567890

# gitの設定で自動的に読み込む
$ git config blame.ignoreRevsFile .git-blame-ignore-revs

# GitHub上でも自動的に認識される（リポジトリルートに配置）
```

### 2.4 時間を遡るblame

```bash
# 特定のコミット時点でのblame
$ git blame abc123 -- src/auth.js

# 特定の行が変更される前のコミットを追跡
$ git log -p -L 15,25:src/auth.js
# → 指定行範囲の変更履歴を全て表示（logベースの追跡）

# 特定の行の変更履歴を1つずつ遡る
$ git blame src/auth.js     # → 最新のblameでcommit Xを発見
$ git blame X~1 -- src/auth.js  # → X以前のblameでcommit Yを発見
$ git blame Y~1 -- src/auth.js  # → Y以前の変更を確認
```

---

## 3. pickaxeとlog — 変更内容での検索

```bash
# 特定の文字列を追加または削除したコミットを検索
$ git log -S "bcrypt" --oneline
# → "bcrypt"という文字列の出現回数が変化したコミット一覧

# 正規表現での検索
$ git log -G "function\s+login" --oneline
# → 正規表現にマッチする行が変更されたコミット一覧

# 差分も表示
$ git log -S "bcrypt" -p -- src/auth.js
```

```
┌────────────────────────────────────────────────────┐
│  -S と -G の違い                                    │
│                                                    │
│  -S "text" (pickaxe):                              │
│    "text" の出現回数が変化したコミットを検索        │
│    → 追加・削除を検出（移動は検出しない）          │
│                                                    │
│  -G "regex":                                       │
│    差分に regex がマッチするコミットを検索          │
│    → 移動・修正も検出する（範囲が広い）            │
│                                                    │
│  例: "x = 1" を "x = 2" に変更した場合             │
│    -S "x = 1" → 検出する（出現回数が減少）        │
│    -S "x = 2" → 検出する（出現回数が増加）        │
│    -G "x = " → 検出する（差分にマッチ）           │
└────────────────────────────────────────────────────┘
```

---

## 4. 実践的なデバッグワークフロー

### 4.1 bisect + blame の組み合わせ

```bash
# Step 1: bisectでバグ混入コミットを特定
$ git bisect start HEAD v1.0.0
$ git bisect run npm test
# → commit abc123 が原因と判明

# Step 2: 原因コミットの詳細を確認
$ git show abc123 --stat
# → 変更されたファイルの一覧

$ git show abc123 -p
# → 具体的な変更内容

# Step 3: blameで関連するコードの履歴を確認
$ git blame -L '/function validate/,/^}/' src/auth.js
# → validate関数の各行がいつ・誰に変更されたか

# Step 4: pickaxeで関連する変更を全て洗い出す
$ git log -S "validate" --oneline -- src/auth.js
# → validate関連の全変更履歴
```

### 4.2 自動bisectスクリプトのパターン

```bash
#!/bin/bash
# bisect-test.sh - 特定の条件をテストするスクリプト

# ビルドできないコミットはスキップ
npm run build 2>/dev/null || exit 125

# 特定のテストケースを実行
npm test -- --testPathPattern="auth.test" 2>/dev/null
exit $?
```

```bash
# 実行
$ git bisect start HEAD v1.0.0
$ git bisect run ./bisect-test.sh
```

---

## 5. git annotate と git log --follow

```bash
# annotateはblameのエイリアス（出力形式が若干異なる）
$ git annotate src/auth.js

# ファイル名の変更を追跡するblame
$ git log --follow -p -- src/auth.js
# → ファイル名が変更されていても変更履歴を追跡

# blameでの--follow相当
$ git log --follow --diff-filter=R -- src/auth.js
# → リネームを検出して元のファイル名を特定
$ git blame <旧ファイル名のcommit> -- <旧ファイル名>
```

---

## 6. アンチパターン

### アンチパターン1: bisect中にコードを手動修正する

```bash
# NG: bisect中に作業ディレクトリのファイルを修正
$ git bisect start HEAD v1.0.0
$ vim src/auth.js         # ← ファイルを修正してしまう
$ git bisect good         # ← 修正した状態でテストした結果を報告
# → 結果が不正確になり、誤ったコミットが原因として報告される

# OK: bisect中はコードを変更しない。テストのみ行う
$ git bisect start HEAD v1.0.0
$ npm test                # テストのみ実行
$ git bisect bad          # 結果を正確に報告
```

**理由**: bisectは各コミットの「そのままの状態」でテストすることが前提。手動修正を加えるとテスト条件が変わり、二分探索の前提が崩れる。

### アンチパターン2: blameの結果だけで犯人を決めつける

```bash
# NG: blameの表示コミットが必ずしもバグの原因とは限らない
$ git blame src/auth.js
# Line 15: abc123 (Tanaka) ... ← "Tanakaがバグを入れた"と判断
# → 実際にはTanakaはフォーマット変更しただけ。本当の原因は別のコミット

# OK: --ignore-revs-fileとlog -Lで深掘りする
$ git blame --ignore-revs-file .git-blame-ignore-revs src/auth.js
$ git log -p -L 15,15:src/auth.js
# → フォーマット変更を除外し、実質的な変更履歴を追跡
```

**理由**: blameは「最後にその行を変更したコミット」を表示するだけ。空白調整、リネーム、自動フォーマットのコミットが表示されることが多い。

---

## 7. FAQ

### Q1. bisectはマージコミットを正しく扱えるか？

**A1.** はい、bisectは**DAG（有向非巡回グラフ）上の二分探索**を行うため、マージコミットが含まれる複雑な履歴でも正しく動作します。ただし、マージコミット自体のテストが困難な場合は`git bisect skip`でスキップできます。直線的な履歴と比較すると、マージが多い場合はステップ数がやや増える可能性があります。

### Q2. blameで削除された行の履歴を追跡するには？

**A2.** 直接的な方法はありませんが、以下のアプローチで追跡できます。

```bash
# 方法1: pickaxeで文字列の追加・削除コミットを検索
$ git log -S "削除された行の内容" --all -p

# 方法2: 特定行範囲の変更履歴をlog -Lで追跡
$ git log -p -L '15,20:src/auth.js'
# → 過去に存在した行を含む変更履歴が表示される

# 方法3: 削除前のコミットでblame
$ git blame <削除直前のcommit>~1 -- src/auth.js
```

### Q3. 数千コミットの範囲でbisectする場合、効率化する方法はあるか？

**A3.** いくつかの戦略があります。

1. **パス限定**: `git bisect start HEAD v1.0.0 -- src/auth/`で関連パスに変更があるコミットのみを対象にする
2. **自動bisect**: `git bisect run`でテストスクリプトを自動実行する
3. **ビルド不能コミットのスキップ**: テストスクリプトでexit 125を返す
4. **first-parentのみ**: `git bisect start --first-parent`でマージの第一親のみを辿る（Git 2.29+）

```bash
$ git bisect start --first-parent HEAD v1.0.0 -- src/auth/
$ git bisect run ./test-auth-bug.sh
# → 対象を絞り込み、自動実行で高速に特定
```

---

## まとめ

| 概念                | 要点                                                          |
|---------------------|---------------------------------------------------------------|
| git bisect          | 二分探索でバグ混入コミットを O(log n) で特定                  |
| bisect run          | テストスクリプトで自動bisect、exit codeで判定                 |
| git blame           | 各行の最終変更コミット・著者・日時を表示                      |
| blame -M -C         | コード移動・コピーを検出して元のコミットを表示                |
| ignore-revs-file    | フォーマット変更等のノイズをblameから除外                     |
| git log -S          | 特定文字列の出現回数が変化したコミットを検索（pickaxe）       |
| git log -L          | 特定行範囲の変更履歴を追跡                                    |

---

## 次に読むべきガイド

- [インタラクティブRebase](./00-interactive-rebase.md) — bisectで見つけたコミットの修正
- [Git Hooks](./03-hooks-automation.md) — bisectテストの自動化との連携
- [マージアルゴリズム](../00-git-internals/02-merge-algorithms.md) — マージ履歴上でのbisect

---

## 参考文献

1. **Pro Git Book** — "Git Tools - Debugging with Git" https://git-scm.com/book/en/v2/Git-Tools-Debugging-with-Git
2. **Git公式ドキュメント** — `git-bisect`, `git-blame`, `git-log` https://git-scm.com/docs
3. **GitHub Docs** — "Using git blame to trace changes in a file" https://docs.github.com/en/repositories/working-with-files/using-files/viewing-a-file#viewing-the-line-by-line-revision-history-for-a-file
