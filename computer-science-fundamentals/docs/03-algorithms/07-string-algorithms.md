# 文字列アルゴリズム

> テキスト検索は最も身近なアルゴリズムの1つ。ブラウザのCtrl+F、grepコマンド、IDEの検索——全てが文字列マッチングに依存している。

## この章で学ぶこと

- [ ] ナイーブな文字列検索の問題点を理解する
- [ ] KMP法の原理を説明できる
- [ ] 正規表現とオートマトンの関係を理解する

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]
- 文字コード → 参照: [[../02-data-representation/01-character-encoding.md]]

---

## 1. 文字列マッチング

### 1.1 ナイーブ法

```python
def naive_search(text, pattern):
    """テキスト中のパターンの出現位置を全て返す"""
    n, m = len(text), len(pattern)
    positions = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            positions.append(i)
    return positions

# 計算量: O(n × m) — 最悪ケース
# 例: text="AAAAAB", pattern="AAB"
# 毎回ほぼm文字比較してから不一致を検出
```

### 1.2 KMP法（Knuth-Morris-Pratt）

```python
def kmp_search(text, pattern):
    """失敗関数を使って無駄な比較を省く"""
    # 失敗関数（部分一致テーブル）の構築
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    # 検索
    positions = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = failure[j - 1]  # 失敗関数でジャンプ
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = failure[j - 1]

    return positions

# 計算量: O(n + m) — 線形時間！
# 失敗関数の構築: O(m)
# 検索: O(n)

# KMPの核心:
# 不一致が起きた時、パターンの「接頭辞と接尾辞の一致」を利用して
# テキストのポインタを戻さずにパターンのポインタだけを調整する
```

### 1.3 Rabin-Karp法

```python
def rabin_karp(text, pattern):
    """ローリングハッシュで高速なパターンマッチング"""
    n, m = len(text), len(pattern)
    base, mod = 256, 10**9 + 7
    positions = []

    # パターンのハッシュ値
    pattern_hash = 0
    text_hash = 0
    h = pow(base, m - 1, mod)

    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod

    for i in range(n - m + 1):
        if text_hash == pattern_hash:
            if text[i:i+m] == pattern:  # ハッシュ衝突チェック
                positions.append(i)
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * h) * base
                        + ord(text[i + m])) % mod

    return positions

# 計算量: O(n + m) 期待 / O(nm) 最悪（ハッシュ衝突時）
# 利点: 複数パターンの同時検索に強い
```

---

## 2. 文字列データ構造

### 2.1 Trie（トライ木）

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    """前置木 — 文字列の集合を効率的に管理"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# 用途:
# - オートコンプリート（検索候補の表示）
# - スペルチェック
# - IPルーティング（最長前置一致）
# - 辞書の高速検索
# 計算量: 挿入/検索 O(m) — mは文字列長
```

### 2.2 サフィックス配列

```
サフィックス配列: 全ての接尾辞をソートした配列

  文字列: "banana"
  接尾辞:           ソート後:
  0: banana         5: a
  1: anana          3: ana
  2: nana           1: anana
  3: ana            0: banana
  4: na             4: na
  5: a              2: nana

  サフィックス配列: [5, 3, 1, 0, 4, 2]

  パターン検索: 二分探索で O(m log n)
  構築: O(n) (SA-IS アルゴリズム)

  用途:
  - 全文検索エンジン
  - DNA配列の解析
  - データ圧縮（BWT: Burrows-Wheeler変換）
```

---

## 3. 正規表現とオートマトン

### 3.1 正規表現の内部

```
正規表現の実行方式:

  1. NFA（非決定性有限オートマトン）方式
     → 正規表現 → NFA → 文字列をシミュレート
     → 計算量: O(n × m) — nは文字列長、mは正規表現長
     → Go, Rust, RE2 が採用

  2. バックトラック方式
     → 正規表現をバックトラックで直接実行
     → 計算量: O(2^n) 最悪（指数時間！）
     → Python, JavaScript, Java, Ruby, Perl が採用
     → 「壊滅的バックトラック」の危険

  壊滅的バックトラック (ReDoS) の例:
  パターン: (a+)+b
  入力: "aaaaaaaaaaaaaaaaaaaac"
  → バックトラック方式: 指数時間で応答なし！
  → NFA方式: 線形時間で不一致を判定

  対策:
  - 正規表現のタイムアウト設定
  - RE2/Go正規表現エンジンの使用
  - 正規表現の静的解析ツール
```

### 3.2 実務での文字列処理

```python
# 実務でよく使う文字列操作の計算量

s = "Hello, World!"

# O(n) 操作:
s.find("World")      # 線形検索
s.replace("o", "0")  # 全置換
s.split(",")          # 分割
"".join(parts)        # 結合

# 注意: 文字列の連結
# ❌ O(n²): ループ内で文字列連結
result = ""
for word in words:
    result += word  # 毎回新しい文字列を生成 → O(n²)

# ✅ O(n): joinを使用
result = "".join(words)

# ✅ O(n): io.StringIO
from io import StringIO
buf = StringIO()
for word in words:
    buf.write(word)
result = buf.getvalue()
```

---

## 4. 実践演習

### 演習1: パターンマッチング（基礎）
KMP法の失敗関数を手計算で求めよ（パターン: "ABABC"）。

### 演習2: Trie（応用）
Trieを使ってオートコンプリート機能を実装せよ。

### 演習3: 正規表現エンジン（発展）
NFA方式の簡易正規表現エンジン（., *, | のみサポート）を実装せよ。

---

## FAQ

### Q1: grep は内部的に何を使っていますか？
**A**: GNU grep は Boyer-Moore アルゴリズムがベース。長いパターンに強く、パターン末尾から比較してスキップ量を最大化する。正規表現モードでは DFA/NFA を使用。

### Q2: 全文検索エンジン（Elasticsearch等）の内部は？
**A**: 転置インデックス + BM25スコアリング。各単語の出現位置をインデックス化し、検索時はインデックスのマージで候補を絞る。日本語はMeCab/kuromoji等で形態素解析してからインデックス構築。

### Q3: 文字列処理で最も重要な最適化は？
**A**: (1)文字列連結の回避（joinを使う） (2)正規表現のプリコンパイル(re.compile) (3)不要なコピーの回避（スライスは新しい文字列を生成する）

---

## まとめ

| アルゴリズム | 計算量 | 用途 |
|------------|--------|------|
| ナイーブ | O(nm) | 短いテキスト |
| KMP | O(n+m) | 単一パターン検索 |
| Rabin-Karp | O(n+m)期待 | 複数パターン |
| Trie | O(m) | 辞書、オートコンプリート |
| サフィックス配列 | O(m log n) | 全文検索 |

---

## 次に読むべきガイド
→ [[../04-data-structures/00-arrays-and-strings.md]] — 配列と文字列

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 32: String Matching.
2. Knuth, D. E., Morris, J. H., Pratt, V. R. "Fast Pattern Matching in Strings." 1977.
3. Cox, R. "Regular Expression Matching Can Be Simple And Fast." 2007.
