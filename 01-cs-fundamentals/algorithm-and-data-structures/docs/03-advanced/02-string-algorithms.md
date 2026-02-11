# 文字列アルゴリズム

> 文字列のパターン検索・照合を効率的に行うKMP・Rabin-Karp・Z-algorithm・Trie・Aho-Corasickを体系的に理解する

## この章で学ぶこと

1. **KMP法とZ-algorithm**で単一パターン検索を O(n+m) で実行できる
2. **Rabin-Karp法**のローリングハッシュによる確率的パターン検索を理解する
3. **Trie と Aho-Corasick** で複数パターンの同時検索を効率的に実装できる

---

## 1. 文字列検索アルゴリズムの全体像

```
┌──────────────────────────────────────────────────────┐
│              文字列検索アルゴリズム                     │
├──────────────────┬───────────────────────────────────┤
│  単一パターン     │  複数パターン                      │
├──────────────────┼───────────────────────────────────┤
│ ナイーブ O(nm)   │ Aho-Corasick O(n+m+z)             │
│ KMP O(n+m)       │ (Trie + 失敗関数)                  │
│ Rabin-Karp O(n+m)│                                    │
│ Z-algorithm O(n+m)│                                   │
│ Boyer-Moore O(n/m)│                                   │
└──────────────────┴───────────────────────────────────┘

  n = テキスト長, m = パターン長, z = マッチ数
```

---

## 2. ナイーブ法（Brute Force）

```
テキスト:  ABCABCABD
パターン:  ABCABD

位置0: ABCAB[C]ABD   ← C≠D 不一致
       ABCAB[D]

位置1: ABCABCABD     ← B≠A 不一致
        A

位置2: ABCABCABD     ← C≠A 不一致
         A

位置3: ABCABCABD     ← 全一致!
          ABCABD
```

```python
def naive_search(text: str, pattern: str) -> list:
    """ナイーブ法 - O(nm)"""
    n, m = len(text), len(pattern)
    matches = []
    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)
    return matches

print(naive_search("ABCABCABD", "ABCABD"))  # [3]
```

---

## 3. KMP法（Knuth-Morris-Pratt）

パターンの「接頭辞と接尾辞の一致」情報（失敗関数/部分一致テーブル）を事前計算し、不一致時にパターンを効率的にシフトする。

```
失敗関数（LPS配列）の構築:
パターン: A B C A B D
LPS:     [0, 0, 0, 1, 2, 0]

  "A"      → 接頭辞=接尾辞なし → 0
  "AB"     → 接頭辞=接尾辞なし → 0
  "ABC"    → 接頭辞=接尾辞なし → 0
  "ABCA"   → "A" = "A"        → 1
  "ABCAB"  → "AB" = "AB"      → 2
  "ABCABD" → 接頭辞=接尾辞なし → 0

不一致時の利用:
テキスト: ...ABCAB|C|ABD...
パターン:    ABCAB|D|
             ↑ LPS[4]=2 → 2文字分は一致済み

シフト後:
テキスト: ...ABCAB|C|ABD...
パターン:       AB|C|ABD
                ↑ 位置2から再開
```

```python
def kmp_search(text: str, pattern: str) -> list:
    """KMP法 - O(n + m)"""

    def build_lps(pattern: str) -> list:
        """失敗関数（LPS配列）の構築 - O(m)"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n, m = len(text), len(pattern)
    lps = build_lps(pattern)
    matches = []
    i = j = 0  # i: テキスト, j: パターン

    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1

        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j - 1]  # パターンをシフト
            else:
                i += 1

    return matches

text = "AABAACAADAABAABA"
pattern = "AABA"
print(kmp_search(text, pattern))  # [0, 9, 12]
```

---

## 4. Rabin-Karp法

ローリングハッシュで文字列の部分文字列をハッシュ値で比較する。ハッシュ衝突時のみ文字列比較。

```
テキスト: "ABCDE"  パターン: "BCD" (ハッシュ=h)

ウィンドウのスライド:
  "ABC" → hash("ABC") ≠ h → スキップ
  "BCD" → hash("BCD") = h → 文字列比較 → 一致!
  "CDE" → hash("CDE") ≠ h → スキップ

ローリングハッシュ:
  hash("BCD") = (hash("ABC") - 'A' * base²) * base + 'D'
  → O(1) で次のハッシュを計算
```

```python
def rabin_karp(text: str, pattern: str, prime: int = 101) -> list:
    """Rabin-Karp法 - 平均 O(n+m), 最悪 O(nm)"""
    n, m = len(text), len(pattern)
    base = 256
    matches = []

    # パターンのハッシュと最初のウィンドウのハッシュ
    p_hash = 0
    t_hash = 0
    h = pow(base, m - 1, prime)  # base^(m-1) mod prime

    for i in range(m):
        p_hash = (base * p_hash + ord(pattern[i])) % prime
        t_hash = (base * t_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if p_hash == t_hash:
            # ハッシュ一致 → 実際に文字列比較（偽陽性を排除）
            if text[i:i + m] == pattern:
                matches.append(i)

        if i < n - m:
            # ローリングハッシュ: 先頭を除去、末尾を追加
            t_hash = (base * (t_hash - ord(text[i]) * h) +
                      ord(text[i + m])) % prime
            if t_hash < 0:
                t_hash += prime

    return matches

print(rabin_karp("AABAACAADAABAABA", "AABA"))  # [0, 9, 12]
```

---

## 5. Z-algorithm

Z配列 Z[i] = 「文字列の位置 i から始まる部分文字列と、文字列全体の接頭辞が一致する最大長」を O(n) で計算する。

```
文字列:  a a b x a a b
Z配列: [-, 1, 0, 0, 3, 1, 0]

Z[1]=1: "abxaab" vs "aabxaab" → "a" 一致 → 1
Z[4]=3: "aab" vs "aabxaab" → "aab" 一致 → 3

パターン検索への応用:
  S = pattern + "$" + text
  Z配列で Z[i] == len(pattern) の位置がマッチ位置
```

```python
def z_function(s: str) -> list:
    """Z配列の計算 - O(n)"""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0  # Z-box [l, r)

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def z_search(text: str, pattern: str) -> list:
    """Z-algorithm によるパターン検索 - O(n+m)"""
    combined = pattern + "$" + text
    z = z_function(combined)
    m = len(pattern)

    matches = []
    for i in range(m + 1, len(combined)):
        if z[i] == m:
            matches.append(i - m - 1)

    return matches

print(z_search("AABAACAADAABAABA", "AABA"))  # [0, 9, 12]
```

---

## 6. Trie（トライ木）

文字列の集合を効率的に格納・検索する木構造。各辺が1文字に対応する。

```
単語集合: {"app", "apple", "apply", "apt", "bat", "bath"}

Trie:
        root
       /    \
      a      b
      |      |
      p      a
     / \     |
    p   t    t
   / \       |
  l   [e]   h
  |         |
 [e] [y]  [NULL]

  [] = 単語の終端

検索 "app" → root→a→p→p → 終端マークあり → 発見
検索 "api" → root→a→p→? (iの子がない) → 不在
接頭辞検索 "ap" → root→a→p → この部分木の全単語を返す
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # この接頭辞を持つ単語数

class Trie:
    """Trie（トライ木）"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """単語を挿入 - O(m)"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        node.is_end = True

    def search(self, word: str) -> bool:
        """完全一致検索 - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """接頭辞検索 - O(m)"""
        return self._find_node(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """接頭辞を持つ単語の数"""
        node = self._find_node(prefix)
        return node.count if node else 0

    def _find_node(self, prefix: str) -> TrieNode:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def auto_complete(self, prefix: str) -> list:
        """オートコンプリート"""
        node = self._find_node(prefix)
        if not node:
            return []

        results = []
        def dfs(node, path):
            if node.is_end:
                results.append(prefix + path)
            for char, child in sorted(node.children.items()):
                dfs(child, path + char)

        dfs(node, "")
        return results

# 使用例
trie = Trie()
for word in ["app", "apple", "apply", "apt", "bat", "bath"]:
    trie.insert(word)

print(trie.search("apple"))       # True
print(trie.search("api"))         # False
print(trie.starts_with("ap"))     # True
print(trie.count_prefix("ap"))    # 4
print(trie.auto_complete("ap"))   # ['app', 'apple', 'apply', 'apt']
```

---

## 7. Aho-Corasick法

Trie に KMP の失敗関数を組み合わせ、複数パターンを同時に検索する。

```
パターン: {"he", "she", "his", "hers"}
テキスト: "ahishers"

Trie + 失敗リンク:
     root ──→ h ──→ e ──→ r ──→ s
      │        │
      │        └──→ i ──→ s
      │
      └──→ s ──→ h ──→ e
               (失敗リンク → root.h)

走査結果:
  位置1: "his" (i=1)
  位置3: "she" (i=3), "he" (i=4)
  位置4: "hers" (i=4)
```

```python
from collections import deque, defaultdict

class AhoCorasick:
    """Aho-Corasick法 - 複数パターン同時検索"""

    def __init__(self):
        self.goto = [{}]         # goto関数
        self.fail = [0]          # 失敗関数
        self.output = [[]]       # 出力関数
        self.state_count = 1

    def _add_state(self):
        self.goto.append({})
        self.fail.append(0)
        self.output.append([])
        state = self.state_count
        self.state_count += 1
        return state

    def add_pattern(self, pattern: str, pattern_id: int = None):
        """パターンを追加"""
        if pattern_id is None:
            pattern_id = pattern

        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = self._add_state()
            state = self.goto[state][char]
        self.output[state].append(pattern_id)

    def build(self):
        """失敗関数を構築 - O(パターン文字数の合計)"""
        queue = deque()

        # 深さ1の状態の失敗関数を設定
        for char, state in self.goto[0].items():
            self.fail[state] = 0
            queue.append(state)

        # BFS で構築
        while queue:
            r = queue.popleft()
            for char, s in self.goto[r].items():
                queue.append(s)

                state = self.fail[r]
                while state != 0 and char not in self.goto[state]:
                    state = self.fail[state]

                self.fail[s] = self.goto[state].get(char, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0

                self.output[s] = self.output[s] + self.output[self.fail[s]]

    def search(self, text: str) -> list:
        """テキスト中の全パターンを検索 - O(n + m + z)"""
        state = 0
        results = []

        for i, char in enumerate(text):
            while state != 0 and char not in self.goto[state]:
                state = self.fail[state]

            state = self.goto[state].get(char, 0)

            for pattern in self.output[state]:
                if isinstance(pattern, str):
                    results.append((i - len(pattern) + 1, pattern))
                else:
                    results.append((i, pattern))

        return results

# 使用例
ac = AhoCorasick()
patterns = ["he", "she", "his", "hers"]
for p in patterns:
    ac.add_pattern(p)
ac.build()

results = ac.search("ahishers")
print(results)
# [(1, 'his'), (3, 'she'), (4, 'he'), (4, 'hers')]
```

---

## 8. アルゴリズム比較表

| アルゴリズム | 前処理 | 検索 | 空間 | 特徴 |
|:---|:---|:---|:---|:---|
| ナイーブ | O(1) | O(nm) | O(1) | 実装最簡 |
| KMP | O(m) | O(n) | O(m) | 決定的、最悪O(n) |
| Rabin-Karp | O(m) | 平均O(n) | O(1) | 複数パターン対応 |
| Z-algorithm | O(n+m) | O(n+m) | O(n+m) | Z配列の汎用性 |
| Aho-Corasick | O(Σm) | O(n+z) | O(Σm) | 複数パターン最速 |

## 用途別選択ガイド

| 用途 | 推奨 | 理由 |
|:---|:---|:---|
| 単一パターン（確実） | KMP | 最悪O(n+m)保証 |
| 単一パターン（簡易） | Rabin-Karp | 実装が簡潔 |
| 複数パターン同時検索 | Aho-Corasick | Trie+失敗関数で最適 |
| 接頭辞マッチ | Trie | オートコンプリートに最適 |
| テキストエディタ検索 | Boyer-Moore | 実用上最速（省略） |
| DNA配列検索 | Suffix Array | 大量検索に強い |

---

## 9. アンチパターン

### アンチパターン1: ナイーブ法の多用

```python
# BAD: 大きなテキストでナイーブ法を繰り返す
text = "A" * 1000000
pattern = "A" * 999
# O(n*m) = O(10^12) → タイムアウト

# GOOD: KMP or Z-algorithm を使う
matches = kmp_search(text, pattern)  # O(n+m) = O(10^6)
```

### アンチパターン2: 複数パターンを個別に検索

```python
# BAD: k 個のパターンを個別に KMP で検索 → O(k * (n+m))
patterns = ["pattern1", "pattern2", ..., "patternK"]
for p in patterns:
    kmp_search(text, p)  # k回繰り返し

# GOOD: Aho-Corasick で一括検索 → O(n + Σm + z)
ac = AhoCorasick()
for p in patterns:
    ac.add_pattern(p)
ac.build()
results = ac.search(text)  # 1回の走査
```

---

## 10. FAQ

### Q1: KMPとZ-algorithmはどちらが良いか？

**A:** 計算量は同じ O(n+m)。KMP は失敗関数でオンライン処理（ストリーム対応）が可能。Z-algorithm は Z 配列の構築が直感的で、文字列問題全般に応用しやすい。競技プログラミングでは Z-algorithm のほうが汎用性が高いと人気。

### Q2: Rabin-Karp法のハッシュ衝突の対策は？

**A:** (1) 大きな素数 p を使う。(2) 複数のハッシュ関数を併用（ダブルハッシュ）。(3) ハッシュ一致時に文字列比較で確認。理論上の最悪は O(nm) だが、良いハッシュ関数なら実用上ほぼ O(n+m)。

### Q3: Suffix Array/Suffix Tree とは何か？

**A:** Suffix Array は文字列の全接尾辞をソートした配列で、O(n log n) で構築、O(m log n) でパターン検索できる。Suffix Tree は全接尾辞を Trie に格納した構造で O(n) で構築可能だが、メモリ消費が大きい。大量の検索クエリを処理する場合に有効。

---

## 11. まとめ

| 項目 | 要点 |
|:---|:---|
| KMP | 失敗関数でシフト量を最適化。O(n+m) 保証 |
| Rabin-Karp | ローリングハッシュで高速比較。複数パターン対応 |
| Z-algorithm | Z 配列で接頭辞一致長を計算。汎用性が高い |
| Trie | 文字列集合の木構造。接頭辞検索・オートコンプリート |
| Aho-Corasick | Trie + 失敗関数で複数パターンを O(n+Σm+z) で検索 |
| 使い分け | 単一→KMP/Z、複数→Aho-Corasick、接頭辞→Trie |

---

## 次に読むべきガイド

- [セグメント木](./01-segment-tree.md) -- 別の高度データ構造
- [ネットワークフロー](./03-network-flow.md) -- グラフ上の最適化
- [競技プログラミング](../04-practice/01-competitive-programming.md) -- 文字列問題の実戦

---

## 参考文献

1. Knuth, D. E., Morris, J. H., & Pratt, V. R. (1977). "Fast Pattern Matching in Strings." *SIAM Journal on Computing*.
2. Karp, R. M. & Rabin, M. O. (1987). "Efficient Randomized Pattern-Matching Algorithms." *IBM Journal of Research and Development*.
3. Aho, A. V. & Corasick, M. J. (1975). "Efficient String Matching." *Communications of the ACM*.
4. Gusfield, D. (1997). *Algorithms on Strings, Trees, and Sequences*. Cambridge University Press.
