# bisect/blame

> `git bisect`によるバグ原因コミットの二分探索と、`git blame`によるコード変更の追跡手法を解説し、大規模プロジェクトでのデバッグ効率を飛躍的に向上させる。

## この章で学ぶこと

1. **git bisectの二分探索アルゴリズム** — 手動・自動bisectの使い方と効率的なバグ特定手順
2. **git blameの高度な活用** — 行単位の変更追跡、コード移動の検出、ignore-revs
3. **bisectとblameの組み合わせ戦略** — 実践的なデバッグワークフロー
4. **pickaxeとlog -L** — 変更内容での検索と行範囲の履歴追跡
5. **大規模プロジェクトでの効率化テクニック** — first-parent、パス限定、自動化パターン

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

### 1.2 bisectの計算量

bisectの最大テスト回数は `ceil(log2(n))` で計算できる。具体的な目安は以下の通り。

| コミット数 | 最大テスト回数 | 備考                         |
|-----------|---------------|------------------------------|
| 10        | 4             | 小さなfeatureブランチ        |
| 100       | 7             | 中規模のプロジェクト         |
| 1,000     | 10            | 大規模プロジェクト           |
| 10,000    | 14            | 非常に長い履歴               |
| 100,000   | 17            | Linux kernelレベル           |
| 1,000,000 | 20            | 超大規模monorepo             |

このように、コミット数が増えても必要なテスト回数は対数的にしか増加しない。100万コミットでもわずか20回のテストで原因を特定できる。

### 1.3 手動bisect

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

### 1.4 bisect中の状態確認

```bash
# bisectの現在の状態を確認
$ git bisect log
# git bisect start
# # bad: [abc123...] feat: latest
# git bisect bad abc123...
# # good: [def456...] v1.0.0
# git bisect good def456...
# # bad: [789abc...] feat: caching
# git bisect bad 789abc...

# 残りのコミット数と予想ステップ数
$ git bisect visualize
# → gitk等でbisect範囲のコミットを視覚化

# bisect範囲のコミットを一覧表示
$ git bisect visualize --oneline
# → テキスト形式で残りのコミット一覧を表示

# 現在のbisect位置の確認
$ git log --oneline -1
```

### 1.5 自動bisect

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

### 1.6 自動bisectスクリプトのパターン集

```bash
# パターン1: 特定のテストケースのみ実行
#!/bin/bash
# bisect-specific-test.sh
npm run build 2>/dev/null || exit 125
npm test -- --testPathPattern="auth.test" --bail 2>/dev/null
exit $?
```

```bash
# パターン2: コンパイルエラーの検出
#!/bin/bash
# bisect-compile.sh
make clean 2>/dev/null
make 2>/dev/null
exit $?
# → コンパイルが失敗するコミットを特定
```

```bash
# パターン3: パフォーマンスリグレッションの検出
#!/bin/bash
# bisect-performance.sh
npm run build 2>/dev/null || exit 125
RESULT=$(node -e "
  const start = Date.now();
  require('./dist/app').processData(testData);
  const elapsed = Date.now() - start;
  console.log(elapsed);
")
# 500ms以上かかるようになったら bad
if [ "$RESULT" -gt 500 ]; then
  exit 1
else
  exit 0
fi
```

```bash
# パターン4: 特定の文字列が存在するかチェック
#!/bin/bash
# bisect-string-check.sh
# 特定のファイルに特定の文字列が含まれているか確認
grep -q "deprecated_function" src/auth.js
if [ $? -eq 0 ]; then
  exit 1  # deprecated_functionが存在 → bad
else
  exit 0  # 存在しない → good
fi
```

```bash
# パターン5: Dockerを使った環境構築付きテスト
#!/bin/bash
# bisect-docker.sh
docker build -t bisect-test . 2>/dev/null || exit 125
docker run --rm bisect-test npm test 2>/dev/null
EXIT_CODE=$?
docker rmi bisect-test 2>/dev/null
exit $EXIT_CODE
```

### 1.7 bisectの高度な使い方

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

# 範囲指定でスキップ
$ git bisect skip abc123..def456
# → 指定範囲のコミットを全てスキップ

# first-parent のみをbisect（Git 2.29+）
$ git bisect start --first-parent HEAD v1.0.0
# → マージコミットの第一親のみを辿る
# → featureブランチのコミットをスキップ
# → マージ単位での二分探索が可能
```

### 1.8 bisectとDAG（マージ履歴）

```
┌────────────────────────────────────────────────────────┐
│  マージ履歴でのbisect                                    │
│                                                        │
│  直線的な履歴:                                         │
│  o---o---o---o---o---o---o---o---o---o                  │
│  good                              bad                 │
│  → 単純な二分探索                                      │
│                                                        │
│  マージを含む履歴:                                      │
│  o---o---o---M---o---M---o---M---o                      │
│       \     / \     / \     /                          │
│        o---o   o---o   o---o                           │
│  good                         bad                      │
│                                                        │
│  → bisectはDAG上で二分探索を行う                       │
│  → マージコミット自体もテスト対象になる                │
│  → --first-parent で第一親のみに限定可能               │
│                                                        │
│  --first-parent の動作:                                 │
│  o---o---o---M---o---M---o---M---o                      │
│  ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑                  │
│  これらのcommitのみが対象                               │
│  → マージ単位で「どのマージがバグを導入したか」を特定  │
└────────────────────────────────────────────────────────┘
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

### 2.2 blameの出力フォーマット

```bash
# porcelain形式（スクリプト処理用）
$ git blame --porcelain src/auth.js
# 出力:
# a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 1 1 3
# author Gaku
# author-mail <gaku@example.com>
# author-time 1705282200
# author-tz +0900
# committer Gaku
# committer-mail <gaku@example.com>
# committer-time 1705282200
# committer-tz +0900
# summary feat: initial auth module
# filename src/auth.js
# 	const bcrypt = require('bcrypt');

# line-porcelain形式（各行に完全なcommit情報）
$ git blame --line-porcelain src/auth.js

# 最小限の出力
$ git blame -s src/auth.js
# → 著者名と日付を省略、SHA-1と行番号のみ

# メールアドレスも表示
$ git blame -e src/auth.js
```

### 2.3 コード移動・コピーの検出

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

### 2.4 -Mと-Cの閾値設定

```bash
# -M のデフォルト閾値は 20文字
# 移動とみなす最小文字数を変更
$ git blame -M40 src/auth.js
# → 40文字以上の連続する同一テキストを移動として検出

# -C のデフォルト閾値も 40文字
$ git blame -C40 src/auth.js
# → 40文字以上のコピーを検出

# 閾値を小さくすると:
# - より多くの移動/コピーを検出できる
# - 偽陽性（実際にはコピーでない部分を検出）が増える
# - 処理時間が長くなる
```

### 2.5 ignore-revs — フォーマット変更の除外

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

# タブ→スペース変換
789abcdef1234567890abcdef1234567890abcdef

# gitの設定で自動的に読み込む
$ git config blame.ignoreRevsFile .git-blame-ignore-revs

# GitHub上でも自動的に認識される（リポジトリルートに配置）
```

```
┌────────────────────────────────────────────────────────┐
│  --ignore-revs の動作原理                               │
│                                                        │
│  通常のblame:                                          │
│  行 15: abc123 (Prettier) → "  const x = 1;"          │
│  → Prettierのコミットが最終変更として表示              │
│                                                        │
│  --ignore-revs abc123:                                 │
│  行 15: def456 (Gaku) → "  const x = 1;"              │
│  → Prettierを無視して、実質的な変更のコミットを表示    │
│                                                        │
│  動作:                                                 │
│  1. abc123の変更前の状態を確認                         │
│  2. abc123の変更後の状態を確認                         │
│  3. 行の内容が変わっていても、abc123を「透過」する     │
│  4. abc123以前のコミットを「最終変更」として表示       │
│                                                        │
│  注意: 行の追加・削除があると正確に追跡できない場合あり │
└────────────────────────────────────────────────────────┘
```

### 2.6 時間を遡るblame

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

# 特定の日時以降の変更のみ表示
$ git blame --since="2025-01-01" src/auth.js
# → 2025-01-01以降に変更された行のみblame表示
# → それ以前の行は ^abc123 のように ^ プレフィックス付きで表示
```

### 2.7 blameの視覚化ツール

```bash
# VS Code のGitLens拡張
# → エディタ内でインラインblameを表示
# → カーソル行のblame情報が自動表示される

# GitHub上のblame
# URLパターン: https://github.com/user/repo/blame/main/src/auth.js
# → Webブラウザで対話的にblameを確認
# → 各行のコミットリンクから詳細を辿れる

# git gui blame
$ git gui blame src/auth.js
# → GUIでblameを表示（Git標準のGUIツール）

# tig（コンソールUIツール）
$ tig blame src/auth.js
# → コンソール上で対話的にblameを操作
# → Enterキーでコミット詳細に移動
```

---

## 3. pickaxeとlog — 変更内容での検索

### 3.1 -S（pickaxe）での検索

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
│                                                    │
│  例: 行の移動のみ（内容は同じ）                    │
│    -S "function login" → 検出しない（回数不変）    │
│    -G "function login" → 検出する（差分に出現）    │
└────────────────────────────────────────────────────┘
```

### 3.2 -Sの高度なオプション

```bash
# -S に正規表現を使用（--pickaxe-regex と組み合わせ）
$ git log -S "validate[A-Z]\w+" --pickaxe-regex --oneline
# → validateEmail, validatePassword等のパターンにマッチ

# 全ブランチにわたって検索
$ git log --all -S "deprecated_function" --oneline
# → 全ブランチの全コミットから検索

# 特定のファイルに限定
$ git log -S "bcrypt" -- src/auth/ lib/security/
# → 指定パス内のファイルのみ対象

# 差分のコンテキストも表示
$ git log -S "bcrypt" -p --word-diff
# → 変更箇所を単語単位でハイライト表示
```

### 3.3 log -L による行範囲の追跡

```bash
# 特定の行範囲の変更履歴
$ git log -L 10,20:src/auth.js
# → 10行目から20行目の変更を含む全コミットを表示

# 関数定義の変更履歴
$ git log -L ':function login:src/auth.js'
# → login関数の定義全体の変更履歴を追跡
# → Git が関数の開始と終了を自動検出

# 正規表現で範囲指定
$ git log -L '/^async function login/,/^}/':src/auth.js
# → 指定パターンで範囲を指定

# -p オプションと組み合わせてパッチ表示
$ git log -L 10,20:src/auth.js -p
# → 各コミットの具体的な差分を表示
```

```
┌────────────────────────────────────────────────────────┐
│  git log -L の動作                                      │
│                                                        │
│  git log -L 10,20:src/auth.js                          │
│                                                        │
│  最新 → 古い の順でcommitを辿り:                       │
│                                                        │
│  commit C3 (最新):                                     │
│    10: const salt = 10;                                │
│    11: async function login(email, password) {         │
│    ...                                                 │
│    20: }                                               │
│    ← C3で11行目が変更された → 表示する                 │
│                                                        │
│  commit C2:                                            │
│    10: const salt = 10;                                │
│    11: function login(email, password) {               │
│    ...                                                 │
│    18: }                                               │
│    ← C2では変更なし → スキップ                         │
│    ← ただし行番号の対応関係は追跡                      │
│                                                        │
│  commit C1:                                            │
│    8: function login(email, password) {                │
│    ...                                                 │
│    15: }                                               │
│    ← C1で関数が追加された → 表示する                   │
│    ← 行番号がずれても内容を追跡                        │
│                                                        │
│  → -Lは行番号のずれ（前の行の追加/削除）を考慮して    │
│    「同じ論理的位置」の変更履歴を正確に追跡する        │
└────────────────────────────────────────────────────────┘
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

### 4.2 バグの原因調査の完全なフロー

```bash
# シナリオ: ログイン機能が壊れている

# Phase 1: いつから壊れたかを特定
$ git bisect start HEAD v2.0.0 -- src/auth/
$ git bisect run ./test-login.sh
# → commit def456 が最初の bad commit

# Phase 2: 何が変わったかを確認
$ git show def456 --stat
# src/auth/login.js  | 15 ++++++-----
# src/auth/session.js | 8 ++++----
# 2 files changed, 11 insertions(+), 12 deletions(-)

$ git show def456 -p
# → 具体的なコード変更を確認

# Phase 3: 変更の背景を理解
$ git log --oneline def456~5..def456
# → 前後のコミットのコンテキスト

$ git blame -L '/function createSession/,/^}/' src/auth/session.js
# → セッション関連コードの変更履歴

# Phase 4: 関連する変更を全て洗い出し
$ git log -S "createSession" --oneline
# → createSession関数に関わる全コミット

$ git log -G "session.*expire" --oneline -- src/auth/
# → セッションの有効期限に関わる変更

# Phase 5: 修正方針の決定
$ git diff def456~1 def456 -- src/auth/
# → バグを導入した変更の具体的な差分
# → この差分を元に修正方法を決定
```

### 4.3 パフォーマンスリグレッションの調査

```bash
# Step 1: パフォーマンスが低下した時期を特定
$ git bisect start HEAD v2.0.0
$ git bisect run ./benchmark.sh
# benchmark.shの中身:
#!/bin/bash
npm run build 2>/dev/null || exit 125
TIME=$(node -e "
  const start = Date.now();
  require('./dist/app').processLargeDataset();
  console.log(Date.now() - start);
")
[ "$TIME" -lt 1000 ] && exit 0 || exit 1

# Step 2: 原因コミットの分析
$ git show <first-bad-commit> --stat
# → どのファイルが変更されたか

$ git diff <first-bad-commit>~1 <first-bad-commit>
# → 具体的な変更内容

# Step 3: 変更された関数の履歴を確認
$ git log -L ':function processData:src/data-processor.js'
# → processData関数の変更履歴
```

### 4.4 削除されたコードの追跡

```bash
# Step 1: 特定の関数が削除されたコミットを特定
$ git log -S "function deprecatedAuth" --oneline
# abc123 feat: remove deprecated auth (← 削除)
# def456 feat: initial auth module    (← 追加)

# Step 2: 削除直前のバージョンを確認
$ git show abc123~1:src/auth.js
# → 削除される前のファイル全体

$ git blame abc123~1 -- src/auth.js
# → 削除直前の各行のblame

# Step 3: 関連する変更を追跡
$ git log -S "deprecatedAuth" -p
# → 追加と削除の両方のコミットの差分を表示
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

```
┌────────────────────────────────────────────────────────┐
│  ファイルリネーム時のblame追跡                           │
│                                                        │
│  commit C1: src/authentication.js を作成               │
│  commit C2: src/authentication.js を修正               │
│  commit C3: src/authentication.js → src/auth.js にリネーム │
│  commit C4: src/auth.js を修正                         │
│                                                        │
│  git blame src/auth.js:                                │
│  → C3, C4 の変更のみ表示                               │
│  → C1, C2 の情報は表示されない                         │
│                                                        │
│  git log --follow -p -- src/auth.js:                   │
│  → C1〜C4 の全変更を表示（リネームを追跡）             │
│                                                        │
│  リネーム前の blame を見るには:                          │
│  $ git log --follow --diff-filter=R -- src/auth.js     │
│  # → リネームコミット C3 を発見                        │
│  $ git blame C3~1 -- src/authentication.js             │
│  # → リネーム前のファイルの blame                      │
└────────────────────────────────────────────────────────┘
```

---

## 6. git shortlog と統計分析

```bash
# 著者別のコミット数
$ git shortlog -sn
    145  Gaku
     87  Tanaka
     53  Suzuki

# 特定期間の著者別統計
$ git shortlog -sn --since="2025-01-01" --until="2025-03-01"

# ファイル別の変更回数ランキング
$ git log --name-only --pretty=format: | sort | uniq -c | sort -rn | head -20

# 著者別の変更行数
$ git log --author="Gaku" --numstat --pretty=format: | \
  awk '{added+=$1; deleted+=$2} END {print "Added:", added, "Deleted:", deleted}'

# 月別のコミット数推移
$ git log --format="%ai" | cut -d'-' -f1,2 | uniq -c
```

---

## 7. 大規模プロジェクトでの効率化

### 7.1 bisectの効率化戦略

```bash
# 戦略1: パス限定
$ git bisect start HEAD v1.0.0 -- src/auth/ tests/auth/
# → 関連パスの変更があるコミットのみテスト

# 戦略2: first-parent
$ git bisect start --first-parent HEAD v1.0.0
# → マージコミットの第一親のみ辿る
# → featureブランチのコミットをスキップ

# 戦略3: 自動bisect + スキップ
$ git bisect run ./smart-test.sh
# smart-test.sh:
#!/bin/bash
# 依存関係のインストール（キャッシュ使用）
npm ci --cache /tmp/npm-cache 2>/dev/null || exit 125
npm run build 2>/dev/null || exit 125
npm test -- --bail --testPathPattern="auth" 2>/dev/null
exit $?

# 戦略4: 範囲の事前絞り込み
$ git log --oneline --first-parent v1.0.0..HEAD | wc -l
# 500 コミット
$ git log --oneline --first-parent v1.0.0..HEAD -- src/auth/ | wc -l
# 30 コミット → パス限定でテスト回数を大幅削減
```

### 7.2 blameの効率化

```bash
# 大規模ファイルのblameを高速化
$ git blame --incremental src/auth.js
# → 結果をインクリメンタルに出力（パイプラインに適す）

# 特定の行のみblame（全行をblameしない）
$ git blame -L 100,120 src/auth.js
# → 必要な行範囲のみ処理（高速）

# .git-blame-ignore-revs で不要なコミットを除外
# → フォーマット変更のコミットをスキップして高速化

# diff.renameLimit を調整
$ git config diff.renameLimit 10000
# → リネーム検出の精度と速度のバランスを調整
```

---

## 8. アンチパターン

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

### アンチパターン3: bisect runのスクリプトでexit 125を使わない

```bash
# NG: ビルドエラーをbadとして報告
#!/bin/bash
npm run build
npm test
exit $?
# → ビルドが壊れているコミットもbadとして報告
# → 実際のバグ混入コミットと区別できない
# → bisectの結果が不正確になる

# OK: ビルドエラーはexit 125（skip）にする
#!/bin/bash
npm run build 2>/dev/null || exit 125  # ビルド失敗はスキップ
npm test
exit $?
# → ビルドできないコミットはスキップ
# → テスト結果のみで正確にgood/badを判定
```

**理由**: exit 125はbisectに「このコミットはテスト不可能」と伝えるための特別なexit code。ビルドエラーやテスト環境の問題で正確にgood/bad判定できないコミットはスキップすべき。

### アンチパターン4: 広い範囲でbisectを開始する

```bash
# NG: プロジェクトの最初のコミットからbisect
$ git bisect start HEAD $(git rev-list --max-parents=0 HEAD)
# → 数千〜数万コミットが対象になり、テスト環境の変化も大きい
# → 古いコミットではビルドすらできない可能性が高い

# OK: 範囲を適切に絞り込んでから開始
$ git log --oneline --since="2025-01-01" | tail -1
# def456 最古のコミット
$ git bisect start HEAD def456
# → 直近の変更に限定してbisect

# さらに良い: パス限定も追加
$ git bisect start HEAD def456 -- src/auth/
# → 関連ファイルの変更のみを対象
```

---

## 9. FAQ

### Q1. bisectはマージコミットを正しく扱えるか？

**A1.** はい、bisectは**DAG（有向非巡回グラフ）上の二分探索**を行うため、マージコミットが含まれる複雑な履歴でも正しく動作します。ただし、マージコミット自体のテストが困難な場合は`git bisect skip`でスキップできます。直線的な履歴と比較すると、マージが多い場合はステップ数がやや増える可能性があります。

`--first-parent`オプション（Git 2.29+）を使うと、マージの第一親のみを辿るため、マージ単位での二分探索が可能です。

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

# 方法4: git log --diff-filter=D で削除されたファイル自体を検索
$ git log --diff-filter=D --summary -- src/deprecated/
# → 削除されたファイルの一覧
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

### Q4. blameの-Mと-Cはどのような場面で使うべきか？

**A4.** 以下のような場面で使用します。

| オプション | 場面                                              | 例                                    |
|-----------|---------------------------------------------------|---------------------------------------|
| -M        | 同一ファイル内でコードが移動された場合             | 関数の並べ替え                        |
| -C        | 別ファイルからコピーされたコードの原点を知りたい場合| リファクタリングでファイル分割         |
| -C -C     | ファイル作成時にコピーされたコードの原点            | テンプレートからの新規ファイル作成     |
| -C -C -C  | 全履歴にわたるコピーの完全な検出                   | コードの出自の完全な追跡（低速）      |

### Q5. git log -L は関数の境界をどのように検出するか？

**A5.** Gitは`.gitattributes`で定義された言語ごとの関数パターンを使用します。デフォルトでは多くの言語をサポートしていますが、カスタマイズも可能です。

```bash
# デフォルトの関数検出パターンの確認
$ git config diff.javascript.xfuncname
# → JavaScriptの関数定義を検出する正規表現

# カスタムパターンの設定
$ cat .gitattributes
*.js diff=javascript
*.py diff=python
*.rs diff=rust

# カスタム言語の関数パターンを定義
$ git config diff.myLang.xfuncname "^\\s*(function|class|def)\\s+.*$"
```

### Q6. bisectの結果が間違っている（偽の原因コミットが報告される）場合の対処法は？

**A6.** 以下の原因と対処法があります。

1. **テストが不安定（flaky test）**: テストスクリプトで複数回テストを実行し、多数決で判定する
2. **ビルドエラーをbadと報告**: exit 125でスキップするようにスクリプトを修正
3. **環境依存**: テストスクリプト内で環境を初期化（node_modules再インストール等）
4. **bisect中にコードを変更**: `git bisect reset`でやり直す

```bash
# 安定したテストスクリプトの例
#!/bin/bash
npm ci 2>/dev/null || exit 125           # 依存関係を確実にインストール
npm run build 2>/dev/null || exit 125     # ビルド失敗はスキップ
# 3回テストして2回以上成功ならgood
PASS=0
for i in 1 2 3; do
  npm test -- --bail --testPathPattern="login" 2>/dev/null && PASS=$((PASS+1))
done
[ "$PASS" -ge 2 ] && exit 0 || exit 1
```

---

## まとめ

| 概念                | 要点                                                          |
|---------------------|---------------------------------------------------------------|
| git bisect          | 二分探索でバグ混入コミットを O(log n) で特定                  |
| bisect run          | テストスクリプトで自動bisect、exit codeで判定                 |
| exit 125            | bisect runでテスト不可能なコミットをスキップ                  |
| --first-parent      | マージの第一親のみ辿り、マージ単位でbisect                   |
| git blame           | 各行の最終変更コミット・著者・日時を表示                      |
| blame -M -C         | コード移動・コピーを検出して元のコミットを表示                |
| ignore-revs-file    | フォーマット変更等のノイズをblameから除外                     |
| git log -S          | 特定文字列の出現回数が変化したコミットを検索（pickaxe）       |
| git log -G          | 差分に正規表現がマッチするコミットを検索                      |
| git log -L          | 特定行範囲の変更履歴を追跡                                    |
| --follow            | ファイルリネームを追跡してログを表示                          |

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
4. **Christian Couder** — "Fighting regressions with git bisect" https://git-scm.com/docs/git-bisect-lk2009
