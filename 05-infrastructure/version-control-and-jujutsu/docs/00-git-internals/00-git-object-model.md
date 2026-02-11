# Gitオブジェクトモデル

> Gitの内部構造を支える4種類のオブジェクト（blob, tree, commit, tag）とSHA-1ハッシュによるコンテンツアドレッシングの仕組みを徹底解説する。

## この章で学ぶこと

1. **Gitの4つのオブジェクト型**（blob, tree, commit, tag）の役割と相互関係
2. **SHA-1ハッシュによるコンテンツアドレッシング**の仕組みと不変性の保証
3. **オブジェクトデータベース**（`.git/objects`）の内部構造と操作方法

---

## 1. Gitはスナップショットベースである

多くのVCSは「差分（delta）」を保存するが、Gitは**各時点のファイルツリー全体のスナップショット**を保存する。この設計が高速なブランチ切り替えとマージを可能にしている。

```
┌─────────────────────────────────────────────────────┐
│          従来のVCS（差分ベース）                       │
│                                                     │
│  v1 ──── Δ1 ──── Δ2 ──── Δ3 ──── Δ4               │
│  (全体)  (差分)  (差分)  (差分)  (差分)              │
│                                                     │
│  → v4を得るには v1 + Δ1 + Δ2 + Δ3 + Δ4 を計算      │
├─────────────────────────────────────────────────────┤
│          Git（スナップショットベース）                 │
│                                                     │
│  S1 ──── S2 ──── S3 ──── S4 ──── S5               │
│  (全体)  (全体)  (全体)  (全体)  (全体)              │
│                                                     │
│  → 任意のバージョンに O(1) でアクセス可能            │
└─────────────────────────────────────────────────────┘
```

ただし、Gitも内部的にはpackfileで差分圧縮を行う（後述の「Packfile/GC」を参照）。

---

## 2. 4つのオブジェクト型

### 2.1 blob（Binary Large Object）

ファイルの**中身そのもの**を保存する。ファイル名やパーミッションは含まない。

```bash
# ファイルの内容からblobオブジェクトを作成
$ echo "Hello, Git!" | git hash-object -w --stdin
557db03de997c86a4a028e1ebd3a1ceb225be238

# blobの中身を確認
$ git cat-file -p 557db03
Hello, Git!

# オブジェクトの型を確認
$ git cat-file -t 557db03
blob
```

**重要な特性**: 同じ内容のファイルは、ファイル名が異なっても**同一のblobオブジェクト**として共有される。

```
┌──────────────────────────────────────────────┐
│  src/utils.js  ──┐                           │
│                  ├──→  blob: abc123...        │
│  lib/utils.js  ──┘     (同一内容なら同一blob)  │
│                                              │
│  README.md     ────→  blob: def456...        │
└──────────────────────────────────────────────┘
```

### 2.2 tree（ツリー）

ディレクトリ構造を表現する。各エントリは**モード、型、SHA-1、ファイル名**を持つ。

```bash
# 最新コミットのtreeを確認
$ git cat-file -p HEAD^{tree}
100644 blob 557db03de997c86a4a028e1ebd3a1ceb225be238    README.md
040000 tree 8f94139338f9404f26296befa88755fc2598c289    src
100755 blob a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0    run.sh
```

**モードの意味**:

| モード   | 意味                          |
|----------|-------------------------------|
| `100644` | 通常のファイル                |
| `100755` | 実行可能ファイル              |
| `120000` | シンボリックリンク            |
| `040000` | サブディレクトリ（tree）      |
| `160000` | サブモジュール（commit参照）  |

### 2.3 commit（コミット）

スナップショットとメタデータを結びつける。

```bash
$ git cat-file -p HEAD
tree 8f94139338f9404f26296befa88755fc2598c289
parent a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
author Gaku <gaku@example.com> 1707600000 +0900
committer Gaku <gaku@example.com> 1707600000 +0900

feat: ユーザー認証機能を追加
```

commitオブジェクトの構成要素:

```
┌─────────────────────────────────────┐
│         commit object               │
│                                     │
│  tree     → ルートtreeのSHA-1       │
│  parent   → 親commitのSHA-1        │
│            （マージなら複数parent）   │
│  author   → 作成者 + タイムスタンプ  │
│  committer→ 適用者 + タイムスタンプ  │
│  message  → コミットメッセージ      │
└─────────────────────────────────────┘
```

### 2.4 tag（タグ / 注釈付きタグ）

特定のオブジェクト（通常はcommit）に名前とメタデータを付与する。

```bash
# 注釈付きタグの作成
$ git tag -a v1.0.0 -m "Release version 1.0.0"

# タグオブジェクトの中身を確認
$ git cat-file -p v1.0.0
object a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
type commit
tag v1.0.0
tagger Gaku <gaku@example.com> 1707600000 +0900

Release version 1.0.0
```

**軽量タグ vs 注釈付きタグ**:

| 特性             | 軽量タグ (lightweight) | 注釈付きタグ (annotated) |
|------------------|------------------------|--------------------------|
| オブジェクト作成 | なし（refのみ）        | tagオブジェクトを作成    |
| メッセージ       | なし                   | あり                     |
| 署名             | 不可                   | GPG署名可能              |
| 推奨用途         | 一時的なマーキング     | リリースタグ             |

---

## 3. SHA-1ハッシュとコンテンツアドレッシング

### 3.1 ハッシュの計算方法

Gitオブジェクトのハッシュは以下の形式で計算される:

```
SHA-1( "<型> <サイズ>\0<内容>" )
```

```bash
# 手動でblobのハッシュを計算
$ echo -n "Hello, Git!" | python3 -c "
import hashlib, sys
content = sys.stdin.buffer.read()
header = f'blob {len(content)}\0'.encode()
print(hashlib.sha1(header + content).hexdigest())
"
557db03de997c86a4a028e1ebd3a1ceb225be238

# git hash-objectと同じ結果になる
$ echo -n "Hello, Git!" | git hash-object --stdin
557db03de997c86a4a028e1ebd3a1ceb225be238
```

### 3.2 SHA-256への移行

Git 2.42以降、SHA-256がオプションとして利用可能になっている。

```bash
# SHA-256を使用するリポジトリの作成
$ git init --object-format=sha256 my-repo
```

| 項目         | SHA-1                  | SHA-256                |
|--------------|------------------------|------------------------|
| ハッシュ長   | 40文字（160bit）       | 64文字（256bit）       |
| 衝突耐性     | 理論的に破られている   | 安全                   |
| 互換性       | 全Gitツール対応        | 一部未対応             |
| デフォルト   | Yes                    | No（オプトイン）       |

---

## 4. オブジェクトの格納構造

### 4.1 .git/objects ディレクトリ

```
.git/objects/
├── 55/
│   └── 7db03de997c86a4a028e1ebd3a1ceb225be238   ← loose object
├── 8f/
│   └── 94139338f9404f26296befa88755fc2598c289
├── info/
│   └── packs
└── pack/
    ├── pack-abc123...def456.idx    ← packfileインデックス
    └── pack-abc123...def456.pack   ← packfile本体
```

**loose object**はzlib圧縮されて個別ファイルとして保存される。`git gc`実行後にpackfileへまとめられる。

### 4.2 オブジェクト間の参照関係

```
                    ┌──────────┐
                    │  tag     │
                    │  v1.0.0  │
                    └────┬─────┘
                         │ object
                         ▼
┌──────────┐      ┌──────────┐      ┌──────────┐
│ commit   │◄─────│ commit   │◄─────│ commit   │
│ abc123   │parent│ def456   │parent│ 789abc   │
└────┬─────┘      └────┬─────┘      └────┬─────┘
     │ tree             │ tree            │ tree
     ▼                  ▼                 ▼
┌──────────┐      ┌──────────┐      ┌──────────┐
│  tree    │      │  tree    │      │  tree    │
│ (root)   │      │ (root)   │      │ (root)   │
├──────────┤      ├──────────┤      ├──────────┤
│ README   │──┐   │ README   │──┐   │ README   │──→ blob
│ src/     │  │   │ src/     │  │   │ src/     │──→ tree
└──────────┘  │   └──────────┘  │   └──────────┘
              │                 │
              ▼                 ▼
           blob(同一内容なら共有される)
```

---

## 5. 実践: 低レベルコマンドでオブジェクトを操作する

### 5.1 blobからcommitまで手動で構築

```bash
# 1. blobを作成
$ echo "console.log('hello');" | git hash-object -w --stdin
# => aabbcc...

# 2. treeを構築
$ git update-index --add --cacheinfo 100644,aabbcc...,main.js
$ git write-tree
# => ddeeff...

# 3. commitを作成
$ echo "Initial commit" | git commit-tree ddeeff...
# => 112233...

# 4. ブランチを向ける
$ git update-ref refs/heads/main 112233...
```

### 5.2 オブジェクトの検査

```bash
# 全オブジェクトの一覧（loose + packed）
$ git rev-list --all --objects

# 特定オブジェクトのサイズと型
$ git cat-file -s abc123    # サイズ（バイト）
$ git cat-file -t abc123    # 型

# オブジェクトのダンプ（デバッグ用）
$ git cat-file --batch-check --batch-all-objects
```

---

## 6. アンチパターン

### アンチパターン1: 巨大バイナリファイルのコミット

```bash
# NG: 巨大ファイルを直接コミット
$ git add dataset-5gb.csv
$ git commit -m "Add dataset"
# → blobが5GB消費、gc後もpackfileが肥大化
# → clone時に全履歴をダウンロードする必要がある

# OK: Git LFS を使用する
$ git lfs install
$ git lfs track "*.csv"
$ git add .gitattributes dataset-5gb.csv
$ git commit -m "Add dataset via LFS"
```

**理由**: Gitのオブジェクトモデルはテキストファイルに最適化されている。バイナリの差分圧縮効率が悪く、リポジトリサイズが指数的に増大する。

### アンチパターン2: SHA-1の短縮形を固定値として使用

```bash
# NG: スクリプトに短縮ハッシュをハードコード
DEPLOY_COMMIT="abc123"
git checkout $DEPLOY_COMMIT

# OK: タグやブランチ名を使う、または十分な長さのハッシュを使用
DEPLOY_TAG="v1.0.0"
git checkout $DEPLOY_TAG
```

**理由**: リポジトリが大きくなると短縮ハッシュが衝突する可能性がある。Git 2.11以降ではデフォルトの短縮長が7から動的に調整されるようになったが、固定値としての使用は危険。

---

## 7. FAQ

### Q1. 同じ内容のファイルを10個コミットすると、blobは10個作られるのか？

**A1.** いいえ、**1つだけ**です。Gitはコンテンツアドレッシングを採用しているため、同じ内容は同じSHA-1ハッシュを持ち、1つのblobオブジェクトが共有されます。treeオブジェクトが異なるファイル名で同じblobのSHA-1を参照します。

### Q2. コミットを`git commit --amend`で修正すると、元のコミットはどうなるのか？

**A2.** 元のコミットオブジェクトは**削除されずにオブジェクトデータベースに残り続けます**。新しいコミットオブジェクトが作成され、ブランチのrefが新しいコミットを指すように更新されます。元のコミットは`reflog`から参照可能で、`git gc`が実行されるまで（デフォルト90日間）保持されます。

### Q3. SHA-1の衝突が発生したらどうなるのか？

**A3.** 理論的には異なる内容が同じハッシュを持つ可能性がありますが、実用上の確率は天文学的に低い（2^80回の試行で50%）。2017年にGoogleがSHA-1衝突を実証しましたが、Gitは`sha1dc`（衝突検出付きSHA-1）を採用しており、既知の攻撃パターンを検出・拒否します。将来的にはSHA-256への完全移行が計画されています。

---

## まとめ

| 概念                     | 要点                                                        |
|--------------------------|-------------------------------------------------------------|
| blob                     | ファイル内容のみ保存、名前やパーミッションは含まない        |
| tree                     | ディレクトリ構造を表現、blob/treeへの参照を保持             |
| commit                   | tree + parent + author/committer + message                  |
| tag                      | オブジェクトへの名前付き参照（注釈付きならオブジェクト作成）|
| SHA-1                    | コンテンツアドレッシングの基盤、衝突検出付き実装を使用      |
| コンテンツアドレッシング | 同一内容 → 同一ハッシュ → 自動重複排除                      |
| .git/objects             | loose objectとpackfileの2つの格納形式                       |

---

## 次に読むべきガイド

- [Ref・ブランチ](./01-refs-and-branches.md) — HEAD、reflog、detached HEADの仕組み
- [Packfile/GC](./03-packfile-gc.md) — delta圧縮とリポジトリ最適化
- [マージアルゴリズム](./02-merge-algorithms.md) — 3-way mergeとortの内部動作

---

## 参考文献

1. **Pro Git Book** — Scott Chacon, Ben Straub "Git Internals - Git Objects" https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
2. **Git公式ドキュメント** — `git-cat-file`, `git-hash-object` manpage https://git-scm.com/docs
3. **SHA-1衝突問題とGitの対応** — "How does Git handle SHA-1 collisions on blobs?" https://git-scm.com/docs/hash-function-transition
4. **Git Source Code** — `sha1dc` (SHA-1 collision detection) https://github.com/git/git
