# Gitオブジェクトモデル

> Gitの内部構造を支える4種類のオブジェクト（blob, tree, commit, tag）とSHA-1ハッシュによるコンテンツアドレッシングの仕組みを徹底解説する。

## この章で学ぶこと

1. **Gitの4つのオブジェクト型**（blob, tree, commit, tag）の役割と相互関係
2. **SHA-1ハッシュによるコンテンツアドレッシング**の仕組みと不変性の保証
3. **オブジェクトデータベース**（`.git/objects`）の内部構造と操作方法
4. **低レベルplumbingコマンド**を使ったオブジェクト操作の実践
5. **大規模リポジトリ**におけるオブジェクトモデルの挙動と最適化
6. **SHA-256移行**の背景と実務への影響

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

### 1.1 スナップショット方式の詳細な動作原理

Gitがスナップショットを効率的に保存できる理由を理解するために、具体的なシナリオを考えてみよう。

```
プロジェクト構造:
├── README.md        (1KB)
├── src/
│   ├── main.js      (5KB)
│   ├── utils.js     (3KB)
│   └── config.js    (2KB)
└── package.json     (1KB)
```

ここで `src/main.js` だけを変更してコミットした場合:

```
コミット1のtree:
  README.md   → blob:aaa111  (1KB)
  src/        → tree:bbb222
    main.js   → blob:ccc333  (5KB)  ← 変更前
    utils.js  → blob:ddd444  (3KB)
    config.js → blob:eee555  (2KB)
  package.json → blob:fff666 (1KB)

コミット2のtree:
  README.md   → blob:aaa111  (1KB)  ← 同じblob再利用
  src/        → tree:ggg777           ← 新しいtree（中身が変わったため）
    main.js   → blob:hhh888  (5KB)  ← 新しいblob
    utils.js  → blob:ddd444  (3KB)  ← 同じblob再利用
    config.js → blob:eee555  (2KB)  ← 同じblob再利用
  package.json → blob:fff666 (1KB)  ← 同じblob再利用
```

新しく作成されたオブジェクトは**2つだけ**:
- 変更された `main.js` の新しいblob
- `src/` ディレクトリの新しいtree（main.jsへの参照が変わったため）
- ルートtree（src/への参照が変わったため）

変更されていないファイルのblobは完全に再利用される。これがGitのスナップショット方式が効率的な理由である。

### 1.2 差分ベースVCSとの性能比較

```
操作                    | 差分ベースVCS  | Git（スナップショット）
─────────────────────────────────────────────────────────────
特定バージョンの取得     | O(n)          | O(1)
ブランチの切り替え       | O(n)          | O(変更ファイル数)
2つのバージョンの差分    | O(1)          | O(ファイル数)
マージ                  | O(n)          | O(変更ファイル数)
リポジトリサイズ（論理） | 小さい        | 大きい
リポジトリサイズ（実際） | 同程度        | 同程度（packfile圧縮後）
```

※ nはバージョン数。差分ベースはv1からの再構築が必要なため。

### 1.3 変更されていないファイルの扱い

よくある誤解として「Gitはコミットごとに全ファイルのコピーを作る」というものがあるが、これは正確ではない。

```bash
# 実験: 同一内容のblob共有を確認する
$ mkdir /tmp/git-snapshot-test && cd /tmp/git-snapshot-test
$ git init

# 最初のコミット
$ echo "unchanged content" > stable.txt
$ echo "version 1" > changing.txt
$ git add -A && git commit -m "v1"

# 2回目のコミット（changing.txtだけ変更）
$ echo "version 2" > changing.txt
$ git add -A && git commit -m "v2"

# stable.txt のblobハッシュを両方のコミットで比較
$ git ls-tree HEAD~1 stable.txt
100644 blob 8c4e7a1b2c3d... stable.txt

$ git ls-tree HEAD stable.txt
100644 blob 8c4e7a1b2c3d... stable.txt
# → 同じハッシュ = 同じオブジェクト（コピーは作られていない）
```

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

#### blobの内部バイナリ構造

blobオブジェクトがディスク上でどのように保存されているかを詳しく見てみよう。

```
┌──────────────────────────────────────────────┐
│           blob オブジェクトの格納形式           │
├──────────────────────────────────────────────┤
│                                              │
│  zlib_compress(                              │
│    "blob"                    ← 型名          │
│    " "                       ← スペース      │
│    "12"                      ← バイトサイズ   │
│    "\0"                      ← NULLバイト    │
│    "Hello, Git!\n"           ← 実際の内容    │
│  )                                           │
│                                              │
│  → .git/objects/55/7db03de997c86...          │
│    ファイル名 = SHA-1ハッシュ                 │
│    先頭2文字がディレクトリ名                   │
└──────────────────────────────────────────────┘
```

```bash
# blobオブジェクトの生データを確認する
$ python3 -c "
import zlib, sys
with open('.git/objects/55/7db03de997c86a4a028e1ebd3a1ceb225be238', 'rb') as f:
    raw = zlib.decompress(f.read())
    print(repr(raw))
"
# b'blob 12\x00Hello, Git!\n'
```

#### blobとファイルモードの分離

blobにはファイルの実行権限やファイル名が含まれない。この設計の重要性を示す例:

```bash
# 同じ内容のファイルに異なる権限を設定
$ echo "#!/bin/bash" > script.sh
$ chmod +x script.sh
$ cp script.sh library.sh
$ chmod -x library.sh

# 両方のファイルのblobハッシュを確認
$ git hash-object script.sh
# => abc123...
$ git hash-object library.sh
# => abc123...  ← 同じハッシュ！内容が同じだから

# tree内では異なるモードで参照される
$ git add -A && git commit -m "test"
$ git ls-tree HEAD
100755 blob abc123... script.sh    ← 実行可能
100644 blob abc123... library.sh   ← 通常ファイル
# 同じblobオブジェクトが異なるモードで参照されている
```

#### 空ファイルのblob

```bash
# 空ファイルにもblobは作られる
$ touch empty.txt
$ git hash-object empty.txt
e69de29bb2d1d6434b8b29ae775ad8c2e48c5391

# このハッシュは全Gitリポジトリで共通
# 「空の内容」のSHA-1は常に同じ値になる
$ git cat-file -s e69de29
0
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

| モード   | 意味                          | 用途                          |
|----------|-------------------------------|-------------------------------|
| `100644` | 通常のファイル                | テキスト、設定ファイル等      |
| `100755` | 実行可能ファイル              | スクリプト、バイナリ          |
| `120000` | シンボリックリンク            | リンクファイル                |
| `040000` | サブディレクトリ（tree）      | フォルダ構造                  |
| `160000` | サブモジュール（commit参照）  | 外部リポジトリ参照            |

#### treeオブジェクトのバイナリ形式

treeオブジェクトは `git cat-file -p` で人間が読める形式で表示されるが、内部的にはバイナリ形式で保存されている。

```
┌──────────────────────────────────────────────────────┐
│           tree オブジェクトの内部バイナリ形式           │
├──────────────────────────────────────────────────────┤
│                                                      │
│  "tree <size>\0"                                     │
│  ┌───────────────────────────────────────────────┐   │
│  │ "100644 README.md\0" + <20バイトSHA-1バイナリ> │   │
│  │ "040000 src\0"       + <20バイトSHA-1バイナリ> │   │
│  │ "100755 run.sh\0"    + <20バイトSHA-1バイナリ> │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  ※ エントリはファイル名のASCIIソート順で並ぶ          │
│  ※ SHA-1はhex文字列ではなく20バイトのバイナリ        │
└──────────────────────────────────────────────────────┘
```

```bash
# treeオブジェクトの生バイナリデータを解析
$ python3 -c "
import zlib, binascii
with open('.git/objects/8f/94139338f9404f26296befa88755fc2598c289', 'rb') as f:
    raw = zlib.decompress(f.read())
    # ヘッダーを除去
    null_idx = raw.index(b'\x00')
    header = raw[:null_idx].decode()
    print(f'Header: {header}')

    data = raw[null_idx+1:]
    pos = 0
    while pos < len(data):
        # モードとファイル名を読む
        space_idx = data.index(b' ', pos)
        mode = data[pos:space_idx].decode()
        null_idx = data.index(b'\x00', space_idx)
        name = data[space_idx+1:null_idx].decode()
        sha1 = binascii.hexlify(data[null_idx+1:null_idx+21]).decode()
        pos = null_idx + 21
        print(f'{mode} {sha1} {name}')
"
```

#### ネストしたtreeの構造

実際のプロジェクトでは、treeは再帰的にネストする:

```
プロジェクト構造:
my-app/
├── package.json
├── src/
│   ├── index.ts
│   ├── components/
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   └── utils/
│       └── format.ts
└── tests/
    └── format.test.ts

Gitオブジェクトの関係:

root tree (aaa111)
├── 100644 blob bbb222  package.json
├── 040000 tree ccc333  src
│   ├── 100644 blob ddd444  index.ts
│   ├── 040000 tree eee555  components
│   │   ├── 100644 blob fff666  Header.tsx
│   │   └── 100644 blob ggg777  Footer.tsx
│   └── 040000 tree hhh888  utils
│       └── 100644 blob iii999  format.ts
└── 040000 tree jjj000  tests
    └── 100644 blob kkk111  format.test.ts

合計: 5つのtreeオブジェクト + 6つのblobオブジェクト = 11オブジェクト
```

```bash
# 再帰的にtreeを展開して確認
$ git ls-tree -r HEAD
100644 blob bbb222... package.json
100644 blob ddd444... src/index.ts
100644 blob fff666... src/components/Header.tsx
100644 blob ggg777... src/components/Footer.tsx
100644 blob iii999... src/utils/format.ts
100644 blob kkk111... tests/format.test.ts

# treeも含めて表示
$ git ls-tree -r -t HEAD
040000 tree ccc333... src
040000 tree eee555... src/components
040000 tree hhh888... src/utils
040000 tree jjj000... tests
100644 blob bbb222... package.json
100644 blob ddd444... src/index.ts
# ... (以下略)
```

#### 空ディレクトリとGit

Gitのtreeオブジェクトは**空のtreeを参照できない**わけではないが、`git add`コマンドが空ディレクトリを追跡しない設計になっている。

```bash
# 空ディレクトリはgit addできない
$ mkdir empty-dir
$ git add empty-dir
# → 何も追加されない

# 慣例的な解決策: .gitkeepファイルを配置
$ touch empty-dir/.gitkeep
$ git add empty-dir/.gitkeep
# → empty-dirがtreeとして追跡される

# 別の解決策: .gitignoreを使う
$ echo "*" > logs/.gitignore
$ echo "!.gitignore" >> logs/.gitignore
$ git add logs/.gitignore
```

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

#### authorとcommitterの違い

多くの場合authorとcommitterは同一人物だが、`git am`や`git cherry-pick`では異なることがある:

```bash
# パッチを適用した場合のcommitオブジェクト
$ git cat-file -p abc123
tree 8f94139338f9404f26296befa88755fc2598c289
parent def456789...
author Alice <alice@example.com> 1707500000 +0900
committer Bob <bob@example.com> 1707600000 +0900

fix: メモリリークを修正

# Alice がパッチを作成（author）
# Bob がそのパッチを適用（committer）
```

```bash
# cherry-pickの場合
$ git cherry-pick abc123
# → 新しいcommitが作成される
#   author = 元のcommitのauthor（Alice）
#   committer = cherry-pickを実行した人（Bob）

# rebaseの場合
$ git rebase main
# → 各commitが再作成される
#   author = 元のcommitのauthor（変更なし）
#   committer = rebaseを実行した人 + 現在の時刻
```

#### タイムスタンプの詳細

Gitのタイムスタンプには2種類の形式がある:

```bash
# author date: 元のコードが書かれた日時
# committer date: commitオブジェクトが作成された日時

# 両方のタイムスタンプを確認
$ git log --format='author:    %ai%ncommitter: %ci%n' -3
author:    2024-02-11 10:30:00 +0900
committer: 2024-02-11 10:30:00 +0900

author:    2024-02-10 15:00:00 +0900
committer: 2024-02-11 09:00:00 +0900    ← rebase等で異なる

# author dateを指定してcommitする
$ GIT_AUTHOR_DATE="2024-01-01T00:00:00+0900" git commit -m "New Year commit"

# committer dateも指定する場合
$ GIT_AUTHOR_DATE="2024-01-01T00:00:00+0900" \
  GIT_COMMITTER_DATE="2024-01-01T00:00:00+0900" \
  git commit -m "New Year commit"
```

#### 親コミットの種類

```
初回コミット（parentなし）:
┌──────────────┐
│ commit: aaa  │
│ tree: xxx    │
│ parent: なし │  ← ルートコミット
│ msg: "init"  │
└──────────────┘

通常コミット（parent 1つ）:
┌──────────────┐     ┌──────────────┐
│ commit: bbb  │────→│ commit: aaa  │
│ tree: yyy    │     │ tree: xxx    │
│ parent: aaa  │     │ parent: なし │
│ msg: "feat"  │     │ msg: "init"  │
└──────────────┘     └──────────────┘

マージコミット（parent 2つ）:
┌──────────────┐
│ commit: ddd  │
│ tree: zzz    │
│ parent: bbb  │────→ 1st parent（マージ先）
│ parent: ccc  │────→ 2nd parent（マージ元）
│ msg: "Merge" │
└──────────────┘

オクトパスマージ（parent 3つ以上）:
┌──────────────┐
│ commit: fff  │
│ tree: www    │
│ parent: ccc  │────→ 1st parent
│ parent: ddd  │────→ 2nd parent
│ parent: eee  │────→ 3rd parent
│ msg: "Merge" │
└──────────────┘
```

```bash
# マージコミットの親を確認
$ git cat-file -p HEAD
tree 8f94139...
parent abc123...    ← 1st parent（マージ先ブランチの先頭）
parent def456...    ← 2nd parent（マージ元ブランチの先頭）

Merge branch 'feature/auth' into main

# 1st parentだけをたどる（マージ元を無視）
$ git log --first-parent

# オクトパスマージ（3つ以上のブランチを同時マージ）
$ git merge feature/a feature/b feature/c
# → parentが3つのcommitが作成される
```

#### GPG署名付きコミット

```bash
# 署名付きコミットの作成
$ git commit -S -m "Signed commit"

# 署名付きコミットのオブジェクト内容
$ git cat-file -p HEAD
tree abc123...
parent def456...
author Gaku <gaku@example.com> 1707600000 +0900
committer Gaku <gaku@example.com> 1707600000 +0900
gpgsig -----BEGIN PGP SIGNATURE-----

 iQIzBAABCAAdFiEE...
 ...
 -----END PGP SIGNATURE-----

feat: 署名付きリリース

# 署名の検証
$ git verify-commit HEAD
gpg: Signature made Mon Feb 12 10:00:00 2024 JST
gpg:                using RSA key ABC123...
gpg: Good signature from "Gaku <gaku@example.com>" [ultimate]
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
| `git describe`   | デフォルトで無視       | 認識される               |
| `git push`       | 明示的に指定が必要     | 同様                     |

#### タグが参照できるオブジェクト

tagオブジェクトは通常commitを参照するが、任意のオブジェクト型を参照できる:

```bash
# commitを参照するタグ（最も一般的）
$ git tag -a v1.0.0 -m "Release v1.0.0" HEAD

# treeを参照するタグ（特定のディレクトリ状態をマーク）
$ git tag -a tree-snapshot -m "Snapshot of src/" HEAD^{tree}

# blobを参照するタグ（特定ファイルの特定バージョンをマーク）
$ BLOB_HASH=$(git rev-parse HEAD:README.md)
$ git tag -a readme-v1 -m "README v1" $BLOB_HASH

# 別のタグを参照するタグ（tag-of-tag、珍しい）
$ git tag -a meta-tag -m "Meta tag" v1.0.0
```

#### タグの内部表現の詳細

```
軽量タグ:
  .git/refs/tags/v1.0.0-light → "abc123def456..."（commitのSHA-1が直接書かれる）

注釈付きタグ:
  .git/refs/tags/v1.0.0 → "xyz789..."（tagオブジェクトのSHA-1）

  tagオブジェクト (xyz789...):
    object abc123def456...    ← 参照先commit
    type commit
    tag v1.0.0
    tagger Gaku <gaku@example.com> 1707600000 +0900

    Release version 1.0.0
```

```bash
# 軽量タグの中身（直接commitを指す）
$ git rev-parse v1.0.0-light
abc123def456...  ← commitのSHA-1

$ git cat-file -t v1.0.0-light
commit  ← 直接commitを指している

# 注釈付きタグの中身（tagオブジェクトを指す）
$ git rev-parse v1.0.0
xyz789...  ← tagオブジェクトのSHA-1

$ git cat-file -t v1.0.0
tag  ← tagオブジェクトを指している

# tagオブジェクトの先のcommitを取得
$ git rev-parse v1.0.0^{commit}
abc123def456...

# タグ一覧をオブジェクト型付きで表示
$ git for-each-ref --format='%(refname:short) %(objecttype) %(objectname:short)' refs/tags/
v1.0.0        tag    xyz789
v1.0.0-light  commit abc123
```

#### GPG署名付きタグ

```bash
# GPG署名付きタグの作成
$ git tag -s v1.0.0 -m "Signed release v1.0.0"

# 署名の検証
$ git verify-tag v1.0.0
gpg: Signature made Mon Feb 12 10:00:00 2024 JST
gpg: Good signature from "Gaku <gaku@example.com>"

# SSH鍵での署名（Git 2.34以降）
$ git config --global gpg.format ssh
$ git config --global user.signingkey ~/.ssh/id_ed25519.pub
$ git tag -s v2.0.0 -m "SSH-signed release"
```

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

#### 各オブジェクト型のハッシュ計算

```bash
# blobのハッシュ計算
# SHA-1("blob <size>\0<file-content>")
$ echo -n "Hello, Git!" | git hash-object --stdin
557db03...

# treeのハッシュ計算
# SHA-1("tree <size>\0<binary-tree-entries>")
# treeのバイナリ形式は直接構築が複雑なため、git mktreeを使う
$ echo -e "100644 blob 557db03de997c86a4a028e1ebd3a1ceb225be238\tREADME.md" | git mktree
# => <tree-hash>

# commitのハッシュ計算
# SHA-1("commit <size>\0tree ...\nparent ...\nauthor ...\ncommitter ...\n\n<message>")
$ python3 -c "
import hashlib

commit_content = '''tree 8f94139338f9404f26296befa88755fc2598c289
parent a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
author Gaku <gaku@example.com> 1707600000 +0900
committer Gaku <gaku@example.com> 1707600000 +0900

feat: add user authentication
'''.encode()

header = f'commit {len(commit_content)}\0'.encode()
sha1 = hashlib.sha1(header + commit_content).hexdigest()
print(sha1)
"
```

### 3.2 コンテンツアドレッシングの利点

コンテンツアドレッシングがもたらす具体的なメリットを整理する:

```
┌─────────────────────────────────────────────────────────┐
│         コンテンツアドレッシングの5つの利点               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 1. 自動重複排除                                         │
│    同じ内容 → 同じハッシュ → 1つのオブジェクトで済む      │
│                                                         │
│ 2. データ完全性の保証                                    │
│    ハッシュが内容から導出されるため、改竄検知が自動的      │
│    ストレージ障害やネットワークエラーも検出可能            │
│                                                         │
│ 3. 効率的な比較                                          │
│    2つのtreeの差分 = ハッシュの比較だけで判定可能          │
│    ハッシュが同じ → 中身も同じ（比較不要）                │
│                                                         │
│ 4. 不変性（Immutability）                                │
│    オブジェクトは一度作成されたら変更不可能                │
│    「変更」= 新しいオブジェクトの作成                     │
│                                                         │
│ 5. 分散処理との親和性                                    │
│    同じ内容は誰が計算しても同じハッシュ                   │
│    → リポジトリ間のデータ交換が効率的                     │
└─────────────────────────────────────────────────────────┘
```

```bash
# 実験: データ完全性の検証
$ git fsck
Checking object directories: 100%
Checking objects: 100%

# 手動でオブジェクトを破損させてみる
$ echo "corrupted" > .git/objects/55/7db03de997c86a4a028e1ebd3a1ceb225be238

$ git fsck
error: object file .git/objects/55/7db03... is empty or corrupted
missing blob 557db03de997c86a4a028e1ebd3a1ceb225be238
# → 即座に検出される
```

### 3.3 SHA-256への移行

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
| パフォーマンス| 高速                  | やや遅い（約10-20%）   |

#### SHA-256移行の詳細

```bash
# SHA-256リポジトリの作成と確認
$ git init --object-format=sha256 sha256-test
$ cd sha256-test

# ハッシュの長さを確認
$ echo "Hello" | git hash-object --stdin
# => 64文字のハッシュ（SHA-256）

# SHA-1リポジトリとの互換性に関する注意点
# 現時点では以下の制限がある:
# - SHA-1とSHA-256リポジトリ間のpush/pullは不可
# - GitHub, GitLabなどのホスティングサービスは未対応（2024年時点）
# - submoduleの参照に互換性の問題がある

# SHA-256リポジトリのオブジェクト形式を確認
$ git rev-parse --show-object-format
sha256
```

#### SHA-1衝突検出（sha1dc）

```bash
# Gitが使用しているSHA-1実装を確認
$ git version
git version 2.44.0

# Git 2.13以降、sha1dc（SHA-1 Collision Detection）が標準
# SHAttered攻撃パターンを検出して拒否する

# 衝突検出のデモンストレーション
# （実際の攻撃ファイルは配布されていないが、仕組みを理解する）
# sha1dcは計算中に衝突攻撃の特徴的なパターンを検出し、
# 検出した場合はハッシュ値を意図的に変更して衝突を回避する
```

### 3.4 ハッシュの短縮表現と曖昧性

```bash
# 短縮ハッシュの解決
$ git rev-parse --short HEAD
abc1234

# 短縮長の制御（デフォルトは動的）
$ git rev-parse --short=12 HEAD
abc1234def56

# 曖昧なハッシュの検出
$ git rev-parse --disambiguate=abc
abc1234def567890...
abc1235678901234...
# → 複数のオブジェクトがマッチする場合がある

# リポジトリ内のオブジェクト数と推奨短縮長の関係
# オブジェクト数     推奨短縮長
# 1,000              7文字
# 100,000            8-9文字
# 1,000,000          10文字
# 10,000,000         11-12文字

# Linuxカーネルリポジトリの場合
$ git -C /path/to/linux log --format='%h' -1
# => 12文字程度が使われる
```

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
│   └── packs                                      ← packfile一覧
└── pack/
    ├── pack-abc123...def456.idx    ← packfileインデックス
    └── pack-abc123...def456.pack   ← packfile本体
```

**loose object**はzlib圧縮されて個別ファイルとして保存される。`git gc`実行後にpackfileへまとめられる。

#### loose objectの詳細な保存プロセス

```bash
# 1. 内容をzlib圧縮
$ python3 -c "
import zlib, hashlib

content = b'Hello, Git!\n'
header = f'blob {len(content)}\0'.encode()
store = header + content

# SHA-1ハッシュを計算
sha1 = hashlib.sha1(store).hexdigest()
print(f'SHA-1: {sha1}')
print(f'ディレクトリ: {sha1[:2]}/')
print(f'ファイル名: {sha1[2:]}')

# zlib圧縮
compressed = zlib.compress(store)
print(f'元のサイズ: {len(store)} bytes')
print(f'圧縮後サイズ: {len(compressed)} bytes')
print(f'圧縮率: {len(compressed)/len(store)*100:.1f}%')
"

# 2. .git/objects/<先頭2文字>/<残り38文字> に保存
# 先頭2文字をディレクトリ名にする理由:
# - ファイルシステムの性能（1ディレクトリに大量のファイルがあると遅い）
# - 256個のサブディレクトリに分散される
```

#### infoディレクトリとalternates

```bash
# alternatesファイル: 他のリポジトリのオブジェクトを参照する
$ cat .git/objects/info/alternates
/path/to/other/repo/.git/objects

# 使用例: CIでの共有オブジェクトストア
# 同じプロジェクトの複数ブランチをビルドする場合、
# 共通のオブジェクトを共有してディスク使用量を削減

$ git clone --reference /path/to/cached-repo https://github.com/org/repo.git
# → cachedリポジトリのオブジェクトを参照し、ネットワーク転送を削減
```

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

### 4.3 オブジェクトの到達可能性（Reachability）

ガベージコレクションにおいて、オブジェクトの「到達可能性」は重要な概念:

```
到達可能なオブジェクト（GCで保持される）:
  refs/heads/main → commit → tree → blob
  refs/tags/v1.0  → tag → commit → tree → blob
  refs/remotes/origin/main → commit → ...
  refs/stash → commit → ...

到達不可能なオブジェクト（GCで削除される可能性）:
  - amend前の古いcommit（reflogの期限切れ後）
  - resetで捨てられたcommit
  - filter-branchで書き換えられた古いオブジェクト
  - abortされたmergeの中間オブジェクト

到達可能性の確認:
$ git fsck --unreachable
unreachable blob abc123...
unreachable commit def456...
unreachable tree ghi789...

# 到達不可能なオブジェクトの詳細
$ git fsck --unreachable --no-reflogs
# → reflogからも参照されないオブジェクトのみ表示
```

```bash
# 実験: オブジェクトの到達可能性を確認する

# 1. コミットを作成
$ echo "test" > file.txt
$ git add file.txt && git commit -m "test commit"

# 2. commitのハッシュを記録
$ OLD_COMMIT=$(git rev-parse HEAD)

# 3. 新しいコミットでamend
$ echo "test2" > file.txt
$ git add file.txt && git commit --amend -m "amended commit"

# 4. 古いコミットは到達不可能になる（reflogからは参照可能）
$ git cat-file -t $OLD_COMMIT
commit  # → まだ存在する

$ git fsck --unreachable
# → reflogがあるため「unreachable」とは表示されない

$ git fsck --unreachable --no-reflogs
unreachable commit $OLD_COMMIT
# → reflogを無視すると到達不可能

# 5. reflogの期限切れ後にGCで削除される
$ git reflog expire --expire=now --all
$ git gc --prune=now
$ git cat-file -t $OLD_COMMIT
fatal: Not a valid object name  # → 削除された
```

### 4.4 オブジェクトの圧縮効率

```bash
# リポジトリのオブジェクト統計を確認
$ git count-objects -v
count: 43          ← loose objectの数
size: 128          ← loose objectの合計サイズ（KB）
in-pack: 12345     ← packfile内のオブジェクト数
packs: 1           ← packfileの数
size-pack: 4567    ← packfileの合計サイズ（KB）
prune-packable: 0  ← packfileに含まれているloose objectの数
garbage: 0         ← 不正なファイルの数
size-garbage: 0    ← 不正なファイルのサイズ（KB）

# 大きなオブジェクトを特定する
$ git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  sort -k3 -n -r |
  head -20
blob abc123... 5242880 assets/large-image.png
blob def456... 2097152 data/sample.csv
# ...
```

---

## 5. 実践: 低レベルコマンドでオブジェクトを操作する

### 5.1 blobからcommitまで手動で構築

```bash
# 1. 空のリポジトリを作成
$ git init /tmp/manual-git-test && cd /tmp/manual-git-test

# 2. blobを作成
$ echo "console.log('hello');" | git hash-object -w --stdin
# => aabbcc11...

# 3. インデックスにエントリを追加
$ git update-index --add --cacheinfo 100644,aabbcc11...,main.js

# 4. treeを書き出す
$ git write-tree
# => ddeeff22...

# 5. commitを作成（親なし = 初回コミット）
$ echo "Initial commit" | git commit-tree ddeeff22...
# => 112233aa...

# 6. ブランチをcommitに向ける
$ git update-ref refs/heads/main 112233aa...

# 7. HEADをmainに向ける
$ git symbolic-ref HEAD refs/heads/main

# 8. 確認
$ git log --oneline
112233a Initial commit

$ git show HEAD:main.js
console.log('hello');
```

### 5.2 複数ファイル・ディレクトリ構造の手動構築

```bash
# より複雑な構造を手動で構築する

# 1. 複数のblobを作成
$ echo '{ "name": "my-app" }' | git hash-object -w --stdin
# => pkg_hash...

$ echo 'export function add(a, b) { return a + b; }' | git hash-object -w --stdin
# => utils_hash...

$ echo 'import { add } from "./utils";' | git hash-object -w --stdin
# => main_hash...

$ echo '# My App' | git hash-object -w --stdin
# => readme_hash...

# 2. src/ ディレクトリのtreeを作成
$ printf '100644 blob %s\t%s\n' utils_hash utils.js main_hash main.js | git mktree
# => src_tree_hash...

# 3. ルートtreeを作成
$ printf '100644 blob %s\t%s\n040000 tree %s\t%s\n100644 blob %s\t%s\n' \
    pkg_hash package.json \
    src_tree_hash src \
    readme_hash README.md | git mktree
# => root_tree_hash...

# 4. commitを作成
$ echo "feat: initial project structure" | \
    GIT_AUTHOR_NAME="Gaku" GIT_AUTHOR_EMAIL="gaku@example.com" \
    GIT_COMMITTER_NAME="Gaku" GIT_COMMITTER_EMAIL="gaku@example.com" \
    git commit-tree root_tree_hash
# => commit_hash...

# 5. 確認
$ git cat-file -p root_tree_hash
100644 blob readme_hash    README.md
100644 blob pkg_hash       package.json
040000 tree src_tree_hash  src

$ git cat-file -p src_tree_hash
100644 blob main_hash      main.js
100644 blob utils_hash     utils.js
```

### 5.3 treeの差分を手動で解析する

```bash
# 2つのtreeの差分を確認（git diff-treeの内部動作を理解する）
$ git diff-tree tree_hash_1 tree_hash_2
:100644 100644 old_blob new_blob M  src/main.js
:000000 100644 0000000 new_blob A  src/config.js
:100644 000000 old_blob 0000000 D  src/legacy.js

# 出力形式の解説:
# :旧モード 新モード 旧ハッシュ 新ハッシュ ステータス パス
# ステータス:
#   A = Added（追加）
#   M = Modified（変更）
#   D = Deleted（削除）
#   R = Renamed（リネーム）
#   C = Copied（コピー）
#   T = Type changed（型変更、例: ファイル→シンボリックリンク）

# リネーム検出付き
$ git diff-tree -M tree_hash_1 tree_hash_2
:100644 100644 abc123 abc123 R100  old-name.js  new-name.js
# R100 = 100%一致のリネーム（内容が完全に同じ）
# R075 = 75%一致のリネーム（内容が75%同じ）
```

### 5.4 オブジェクトの検査

```bash
# 全オブジェクトの一覧（loose + packed）
$ git rev-list --all --objects

# 特定オブジェクトのサイズと型
$ git cat-file -s abc123    # サイズ（バイト）
$ git cat-file -t abc123    # 型

# オブジェクトのダンプ（デバッグ用）
$ git cat-file --batch-check --batch-all-objects

# 全オブジェクトの型別カウント
$ git cat-file --batch-check --batch-all-objects | \
    awk '{print $2}' | sort | uniq -c | sort -rn
  12345 blob
   3456 tree
   1234 commit
      5 tag
```

### 5.5 オブジェクトの存在確認と整合性チェック

```bash
# 特定のオブジェクトが存在するか確認
$ git cat-file -e abc123def456 && echo "exists" || echo "not found"

# リポジトリ全体の整合性チェック
$ git fsck --full
Checking object directories: 100%
Checking objects: 100%
Checking connectivity: 12345 objects reachable

# 厳密なチェック（より多くの問題を検出）
$ git fsck --strict
# 通常はwarningとなる問題もerrorとして報告

# dangling objectの確認
$ git fsck --no-reflogs
dangling commit abc123...
dangling blob def456...
# dangling = どの参照からも到達できないオブジェクト

# 修復手順（破損したリポジトリ）
$ git fsck --full 2>&1 | grep "missing"
missing blob abc123...
# → 他のクローンからオブジェクトをコピーして修復
$ cp /path/to/backup/.git/objects/ab/c123... .git/objects/ab/c123...
```

---

## 6. 実務シナリオ: オブジェクトモデルの応用

### 6.1 リポジトリの容量分析

```bash
# リポジトリサイズの詳細分析スクリプト
#!/bin/bash
echo "=== Repository Object Analysis ==="
echo ""

# 全体統計
echo "--- General Statistics ---"
git count-objects -vH
echo ""

# オブジェクト型別の統計
echo "--- Object Type Distribution ---"
git cat-file --batch-check --batch-all-objects 2>/dev/null | \
    awk '{
        type[$2]++
        size[$2] += $3
    }
    END {
        for (t in type) {
            printf "%-10s count: %6d  total_size: %s\n", t, type[t], size[t]
        }
    }'
echo ""

# 最大のblobオブジェクトTOP10
echo "--- Largest Blobs (TOP 10) ---"
git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    grep '^blob' | \
    sort -k3 -n -r | \
    head -10 | \
    awk '{printf "%s  %10d bytes  %s\n", $2, $3, $4}'
echo ""

# コミット数の統計
echo "--- Commit Statistics ---"
echo "Total commits: $(git rev-list --all --count)"
echo "Merge commits: $(git rev-list --all --merges --count)"
echo "Authors: $(git shortlog -sn --all | wc -l)"
```

### 6.2 特定ファイルの全履歴をオブジェクトレベルで追跡

```bash
# ファイルの各バージョンのblobハッシュを一覧表示
$ git log --follow --format="%H" -- src/config.ts | while read commit; do
    blob=$(git rev-parse "$commit:src/config.ts" 2>/dev/null)
    if [ $? -eq 0 ]; then
        size=$(git cat-file -s "$blob")
        date=$(git log -1 --format="%ai" "$commit")
        echo "$date  $blob  ${size}bytes"
    fi
done

# 出力例:
# 2024-02-11 10:30:00 +0900  abc123...  2048bytes
# 2024-02-10 15:00:00 +0900  def456...  1856bytes
# 2024-02-09 09:00:00 +0900  ghi789...  1024bytes

# 特定のバージョン間の差分を確認
$ git diff blob_hash_1 blob_hash_2
```

### 6.3 サブモジュールとオブジェクトモデル

```bash
# サブモジュールはtree内でモード160000として記録される
$ git ls-tree HEAD
100644 blob abc123... .gitmodules
160000 commit def456... libs/external-lib    ← サブモジュール

# サブモジュールのcommitハッシュを確認
$ git ls-tree HEAD libs/external-lib
160000 commit def456... libs/external-lib
# → def456... はサブモジュールリポジトリのcommitハッシュ

# .gitmodulesファイルの内容
$ git cat-file -p HEAD:.gitmodules
[submodule "libs/external-lib"]
    path = libs/external-lib
    url = https://github.com/org/external-lib.git
```

### 6.4 shallow cloneとオブジェクトモデル

```bash
# shallow clone: 履歴を制限してクローン
$ git clone --depth=1 https://github.com/org/repo.git
# → 最新のcommitとそのtree/blobのみ取得

# shallow cloneのオブジェクト状態
$ git cat-file -p HEAD
tree abc123...
parent def456...     ← 存在するが、このcommitオブジェクトは取得されていない
author ...

# shallow boundary（浅いクローンの境界）を確認
$ cat .git/shallow
def456789...    ← この先の履歴は持っていない

# 深さを追加で取得
$ git fetch --deepen=10
# → 10コミット分追加で取得

# 完全な履歴を取得
$ git fetch --unshallow
# → 全コミットを取得（.git/shallowファイルが削除される）
```

### 6.5 replace objectによるオブジェクトの差し替え

```bash
# git replaceを使ってオブジェクトを「差し替え」る
# （元のオブジェクトは変更せず、参照時に別のオブジェクトを返す）

# ユースケース1: コミットメッセージの修正（歴史を書き換えずに）
$ git replace --edit HEAD
# → エディタが開き、commitオブジェクトの内容を編集できる
# → .git/refs/replace/<original-hash> に新しいハッシュが記録される

# ユースケース2: 大きな歴史の接合（graft point）
# 別々のリポジトリの歴史を接合する
$ git replace --graft <commit> <new-parent>

# replaceオブジェクトの一覧
$ git replace -l

# replaceを無視してオリジナルを参照
$ git --no-replace-objects cat-file -p HEAD

# replaceの削除
$ git replace -d <original-hash>
```

---

## 7. 大規模リポジトリとオブジェクトモデル

### 7.1 モノレポにおけるオブジェクト数の爆発

```
大規模モノレポの典型的なオブジェクト数:

リポジトリ例           | オブジェクト数  | サイズ
─────────────────────────────────────────────────
小規模OSS             | 1,000 - 10,000     | 1-10 MB
中規模Webアプリ        | 10,000 - 100,000   | 10-100 MB
大規模モノレポ         | 1,000,000+         | 1-10 GB
Linuxカーネル          | 8,000,000+         | 3+ GB
Chromium               | 15,000,000+        | 10+ GB
```

```bash
# 大規模リポジトリの最適化設定
$ git config core.commitGraph true        # commit-graphを有効化
$ git config gc.writeCommitGraph true      # GC時にcommit-graphを更新
$ git config feature.manyFiles true        # 大量ファイル向け最適化
$ git config core.untrackedCache true      # untracked fileのキャッシュ
$ git config core.fsmonitor true           # ファイルシステム監視

# commit-graphの生成
$ git commit-graph write --reachable
# → .git/objects/info/commit-graphs/ にバイナリファイルが作成
# → git logの高速化に大きく寄与する

# commit-graphの内容確認
$ git commit-graph verify
```

### 7.2 partial cloneとオブジェクトの遅延取得

```bash
# blobless clone: blobを取得しない
$ git clone --filter=blob:none https://github.com/org/large-repo.git
# → commit + treeのみ取得、blobはcheckout時にオンデマンド取得

# treeless clone: tree + blobを取得しない
$ git clone --filter=tree:0 https://github.com/org/large-repo.git
# → commitのみ取得、tree/blobは必要時に取得

# サイズ制限付きclone: 指定サイズ以上のblobを除外
$ git clone --filter=blob:limit=1m https://github.com/org/large-repo.git
# → 1MB以上のblobは取得しない

# 遅延取得されたオブジェクトの確認
$ git rev-list --objects --all --missing=print | grep "^?"
?abc123...    ← 未取得のオブジェクト
?def456...

# 明示的にオブジェクトを取得
$ git fetch origin --filter=blob:none
```

### 7.3 sparse-checkout とオブジェクトの関係

```bash
# sparse-checkoutの設定
$ git sparse-checkout init --cone
$ git sparse-checkout set src/frontend

# sparse-checkout時のオブジェクト取得
# → treeオブジェクトは全て取得されるが、
#   blobはsparse-checkoutのパターンに一致するファイルのみcheckoutされる
# → partial cloneと組み合わせると、不要なblobは全く取得されない

$ git clone --filter=blob:none https://github.com/org/large-repo.git
$ cd large-repo
$ git sparse-checkout init --cone
$ git sparse-checkout set src/frontend
# → src/frontend/ 配下のblobのみオンデマンド取得される
```

---

## 8. アンチパターンと解決策

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

```bash
# 既にコミットされた巨大ファイルの影響を確認
$ git rev-list --objects --all | \
    git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
    grep '^blob' | sort -k3 -n -r | head -5

# 巨大ファイルを履歴から完全に除去する
$ git filter-repo --path dataset-5gb.csv --invert-paths
# ※ git filter-branchは非推奨、git filter-repoを使う

# LFS移行ツール
$ git lfs migrate import --include="*.csv" --everything
# → 全ブランチの全履歴でCSVファイルをLFSに移行
```

### アンチパターン2: SHA-1の短縮形を固定値として使用

```bash
# NG: スクリプトに短縮ハッシュをハードコード
DEPLOY_COMMIT="abc123"
git checkout $DEPLOY_COMMIT

# OK: タグやブランチ名を使う、または十分な長さのハッシュを使用
DEPLOY_TAG="v1.0.0"
git checkout $DEPLOY_TAG

# OK: フルハッシュを使う（自動化スクリプト）
DEPLOY_COMMIT=$(git rev-parse v1.0.0)
git checkout $DEPLOY_COMMIT
```

**理由**: リポジトリが大きくなると短縮ハッシュが衝突する可能性がある。Git 2.11以降ではデフォルトの短縮長が7から動的に調整されるようになったが、固定値としての使用は危険。

### アンチパターン3: オブジェクトデータベースの直接操作

```bash
# NG: .git/objects/ を手動で操作
$ rm .git/objects/ab/c123def456...
# → リポジトリが破損する

# NG: .git/objects/ をコピーしてバックアップ
$ cp -r .git/objects/ /backup/
# → packfileのロック状態が不整合になる可能性

# OK: Gitコマンドを使う
$ git gc                    # オブジェクトの整理
$ git prune                 # 到達不可能なオブジェクトの削除
$ git bundle create backup.bundle --all  # バックアップ
```

### アンチパターン4: 機密情報のコミット

```bash
# NG: 機密情報をコミット
$ echo "API_KEY=sk-abc123" > .env
$ git add .env && git commit -m "Add config"
# → blobオブジェクトとして永続的に保存される
# → git rmしても過去のcommitからアクセス可能

# 機密情報を履歴から完全に除去する
$ git filter-repo --path .env --invert-paths --force
# → 全commitが書き換えられ、新しいSHA-1が割り当てられる
# → すべてのフォーク・クローンに影響するため注意

# OK: .gitignoreで最初から除外
$ echo ".env" >> .gitignore
$ git add .gitignore && git commit -m "Ignore .env"
```

### アンチパターン5: 頻繁なforce pushによるオブジェクトの散乱

```bash
# NG: 頻繁にrebase + force push
$ git rebase -i HEAD~10
$ git push --force
# → リモートリポジトリに到達不可能なオブジェクトが蓄積
# → 他の開発者のローカルリポジトリとの整合性が崩れる

# OK: force-with-leaseを使用し、影響を最小限に
$ git push --force-with-lease
# → リモートの状態が想定と異なる場合は拒否される
```

---

## 9. FAQ

### Q1. 同じ内容のファイルを10個コミットすると、blobは10個作られるのか？

**A1.** いいえ、**1つだけ**です。Gitはコンテンツアドレッシングを採用しているため、同じ内容は同じSHA-1ハッシュを持ち、1つのblobオブジェクトが共有されます。treeオブジェクトが異なるファイル名で同じblobのSHA-1を参照します。

```bash
# 検証
$ for i in $(seq 1 10); do cp template.txt "file_$i.txt"; done
$ git add -A && git commit -m "Add 10 identical files"
$ git ls-tree HEAD | awk '{print $3}' | sort -u | wc -l
# → 1（blobは1つだけ）
```

### Q2. コミットを`git commit --amend`で修正すると、元のコミットはどうなるのか？

**A2.** 元のコミットオブジェクトは**削除されずにオブジェクトデータベースに残り続けます**。新しいコミットオブジェクトが作成され、ブランチのrefが新しいコミットを指すように更新されます。元のコミットは`reflog`から参照可能で、`git gc`が実行されるまで（デフォルト90日間）保持されます。

```bash
# amend前のcommitを復元する
$ git reflog
abc123 HEAD@{0}: commit (amend): fixed message
def456 HEAD@{1}: commit: original message

$ git checkout def456
# → amend前の状態を確認できる

$ git branch recover-amend def456
# → amend前のcommitをブランチとして保存
```

### Q3. SHA-1の衝突が発生したらどうなるのか？

**A3.** 理論的には異なる内容が同じハッシュを持つ可能性がありますが、実用上の確率は天文学的に低い（2^80回の試行で50%）。2017年にGoogleがSHA-1衝突を実証しましたが、Gitは`sha1dc`（衝突検出付きSHA-1）を採用しており、既知の攻撃パターンを検出・拒否します。将来的にはSHA-256への完全移行が計画されています。

```
衝突の確率（バースデーパラドックス）:
  オブジェクト数    衝突確率
  10^6             約 10^-36（事実上ゼロ）
  10^9             約 10^-30
  10^12            約 10^-24
  10^15            約 10^-18

  参考: Linuxカーネルのオブジェクト数は約 10^7
  → 衝突確率は宇宙的にゼロ
```

### Q4. git gcはいつ自動的に実行されるのか？

**A4.** 以下の条件で自動的に実行されます:

```bash
# 自動GCのトリガー条件
$ git config gc.auto
6700    # loose objectがこの数を超えると自動GC（デフォルト: 6700）

$ git config gc.autoPackLimit
50      # packfileがこの数を超えると自動GC（デフォルト: 50）

# 自動GCを無効化
$ git config gc.auto 0

# 手動GC
$ git gc
$ git gc --aggressive    # より積極的な圧縮（時間がかかる）
```

### Q5. blobの内容が1バイトだけ変わった場合、新しいblobが作られるのか？

**A5.** はい、**完全に新しいblobオブジェクト**が作成されます。loose objectの時点ではそれぞれ独立したzlib圧縮ファイルです。しかし、`git gc`でpackfileにまとめられる際に**delta圧縮**が適用され、類似したblobは差分のみが保存されます。

```bash
# 実験
$ echo "version 1" > test.txt
$ git add test.txt && git commit -m "v1"
$ BLOB_V1=$(git rev-parse HEAD:test.txt)

$ echo "version 2" > test.txt
$ git add test.txt && git commit -m "v2"
$ BLOB_V2=$(git rev-parse HEAD:test.txt)

# 異なるハッシュ = 異なるオブジェクト
$ echo "$BLOB_V1"
$ echo "$BLOB_V2"
# → 全く異なるハッシュ

# packfile内ではdelta圧縮される
$ git gc
$ git verify-pack -v .git/objects/pack/*.idx | grep "$BLOB_V2"
# → deltaとして表示される（基準blobからの差分のみ保存）
```

### Q6. commitオブジェクトのtreeが同じになることはあるのか？

**A6.** はい、あり得ます。例えば、ある変更をcommitした後にrevertすると、revertコミットのtreeは元のcommitのtreeと同じになります。

```bash
# 実験
$ git log --format="%H %T" -5
commit1 tree_A    ← 現在
commit2 tree_B    ← revertされる変更
commit3 tree_A    ← revert後（tree_Aと同じ！）

# treeが同じでもcommitは別オブジェクト
# （parent, author, committer, messageが異なるため）
```

### Q7. Gitオブジェクトは暗号化されているのか？

**A7.** いいえ、**暗号化されていません**。zlib圧縮はされていますが、これはサイズ削減のためであり、暗号化ではありません。リポジトリにアクセスできる人は全てのオブジェクトの内容を読めます。

```bash
# リポジトリの暗号化が必要な場合のオプション
# 1. git-crypt: 特定ファイルを暗号化
$ git-crypt init
$ echo "secrets/** filter=git-crypt diff=git-crypt" >> .gitattributes

# 2. ファイルシステムレベルの暗号化
# → LUKS, FileVault, BitLockerなどを使用

# 3. リポジトリホスティングのアクセス制御
# → GitHub Private Repository, GitLab Privateなど
```

---

## 10. デバッグとトラブルシューティング

### 10.1 壊れたリポジトリの診断

```bash
# 1. 整合性チェック
$ git fsck --full --strict 2>&1 | tee fsck-report.txt

# 典型的なエラーと対処法:

# エラー: missing object
# → オブジェクトファイルが削除された or 破損した
$ git fsck 2>&1 | grep "missing"
missing blob abc123...
# 対処: バックアップまたは他のクローンからオブジェクトを取得
$ git fetch origin  # リモートから不足オブジェクトを取得

# エラー: corrupt object
# → zlib圧縮データが破損している
$ git fsck 2>&1 | grep "corrupt"
error: corrupt loose object 'abc123...'
# 対処: 破損ファイルを削除し、リモートから再取得
$ rm .git/objects/ab/c123...
$ git fetch origin

# エラー: broken link
# → commitやtreeが参照するオブジェクトが存在しない
$ git fsck 2>&1 | grep "broken"
broken link from commit abc123...
# 対処: git reflogから正常な状態に復帰
$ git reflog
$ git reset --hard HEAD@{n}
```

### 10.2 オブジェクトの手動復元

```bash
# シナリオ: 誤ってgit reset --hardした後のファイル復元

# 1. reflogから元のcommitを特定
$ git reflog
abc123 HEAD@{0}: reset: moving to HEAD~5
def456 HEAD@{1}: commit: important work

# 2. danglingオブジェクトを確認
$ git fsck --lost-found
dangling commit def456...
dangling blob ghi789...

# 3. danglingオブジェクトの内容を確認
$ git show def456
# → commitの内容が表示される

# 4. 復元
$ git checkout -b recovery def456

# 5. .git/lost-found/ に復元されたオブジェクト
$ ls .git/lost-found/
other/    ← blob, treeなど
commit/   ← danglingなcommit
```

### 10.3 パフォーマンスデバッグ

```bash
# Git操作のトレース（何が遅いか特定する）
$ GIT_TRACE=1 git status
$ GIT_TRACE_PERFORMANCE=1 git log --oneline -100

# オブジェクトアクセスのトレース
$ GIT_TRACE_PACK_ACCESS=1 git log --oneline -10

# packfileのインデックス再構築（破損時）
$ git index-pack .git/objects/pack/pack-abc123.pack

# loose objectの最適化
$ git repack -a -d
# -a: 全オブジェクトを1つのpackfileにまとめる
# -d: 不要なloose objectを削除

# より積極的な最適化
$ git repack -a -d --depth=250 --window=250
# depth: deltaチェーンの最大深度
# window: delta計算時の比較ウィンドウサイズ
```

---

## まとめ

| 概念                     | 要点                                                        |
|--------------------------|-------------------------------------------------------------|
| blob                     | ファイル内容のみ保存、名前やパーミッションは含まない        |
| tree                     | ディレクトリ構造を表現、blob/treeへの参照を保持             |
| commit                   | tree + parent + author/committer + message                  |
| tag                      | オブジェクトへの名前付き参照（注釈付きならオブジェクト作成）|
| SHA-1                    | コンテンツアドレッシングの基盤、衝突検出付き実装を使用      |
| SHA-256                  | SHA-1の後継、Git 2.42以降でオプション利用可能               |
| コンテンツアドレッシング | 同一内容 → 同一ハッシュ → 自動重複排除                      |
| .git/objects             | loose objectとpackfileの2つの格納形式                       |
| 到達可能性               | GCでの削除判定の基準、refs + reflogから辿れるか             |
| partial clone            | オブジェクトの遅延取得で大規模リポジトリに対応              |

---

## 次に読むべきガイド

- [Ref・ブランチ](./01-refs-and-branches.md) -- HEAD、reflog、detached HEADの仕組み
- [Packfile/GC](./03-packfile-gc.md) -- delta圧縮とリポジトリ最適化
- [マージアルゴリズム](./02-merge-algorithms.md) -- 3-way mergeとortの内部動作

---

## 参考文献

1. **Pro Git Book** -- Scott Chacon, Ben Straub "Git Internals - Git Objects" https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
2. **Git公式ドキュメント** -- `git-cat-file`, `git-hash-object` manpage https://git-scm.com/docs
3. **SHA-1衝突問題とGitの対応** -- "How does Git handle SHA-1 collisions on blobs?" https://git-scm.com/docs/hash-function-transition
4. **Git Source Code** -- `sha1dc` (SHA-1 collision detection) https://github.com/git/git
5. **Git Internals - Plumbing and Porcelain** -- https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain
6. **Technical FAQ** -- https://git-scm.com/docs/technical
7. **commit-graph design document** -- https://git-scm.com/docs/commit-graph
8. **partial clone design** -- https://git-scm.com/docs/partial-clone
