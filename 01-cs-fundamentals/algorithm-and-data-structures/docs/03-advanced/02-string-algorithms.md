# 文字列アルゴリズム

> 文字列のパターン検索・照合を効率的に行うKMP・Rabin-Karp・Z-algorithm・Trie・Aho-Corasick・Suffix Arrayを体系的に理解する

## この章で学ぶこと

1. **KMP法とZ-algorithm**で単一パターン検索を O(n+m) で実行できる
2. **Rabin-Karp法**のローリングハッシュによる確率的パターン検索を理解する
3. **Trie と Aho-Corasick** で複数パターンの同時検索を効率的に実装できる
4. **Suffix Array と LCP Array** で文字列の高度な解析を行える
5. **Manacher法・ローリングハッシュ**による回文検出・文字列比較の手法を実装できる

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

┌──────────────────────────────────────────────────────┐
│            文字列の構造解析                            │
├──────────────────┬───────────────────────────────────┤
│  索引構造         │  特殊アルゴリズム                  │
├──────────────────┼───────────────────────────────────┤
│ Trie             │ Manacher（回文検出）                │
│ Suffix Array     │ ローリングハッシュ（文字列比較）     │
│ Suffix Tree      │ Z-algorithm（汎用文字列処理）       │
│ Suffix Automaton │ LCP（最長共通接頭辞）               │
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

### ナイーブ法が最悪になるケース

```python
# 最悪ケース: テキスト="AAA...A"(n個), パターン="AA...AB"(m個)
# 各位置で m-1 回一致してから不一致 → O(nm)
text = "A" * 10000
pattern = "A" * 999 + "B"
# ナイーブ法: 9001 * 999 ≈ 9 * 10^6 回の比較

# 実用上の改善: Python の `in` 演算子や str.find() は
# Boyer-Moore-Horspool の変種を使っており、ナイーブ法より高速
idx = text.find(pattern)  # 内部で最適化されている
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
    if m == 0:
        return []
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

### KMP法の応用: 文字列の最小周期

```python
def min_period(s: str) -> int:
    """文字列の最小周期をKMPの失敗関数で求める
    例: "abcabc" の最小周期は 3 ("abc")
    例: "aaaaaa" の最小周期は 1 ("a")
    """
    n = len(s)
    lps = [0] * n
    length = 0
    i = 1

    while i < n:
        if s[i] == s[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                i += 1

    # 最小周期 = n - lps[n-1]
    period = n - lps[n - 1]

    # n が period で割り切れる場合のみ完全な周期
    if n % period == 0:
        return period
    else:
        return n  # 周期構造なし（自身が最小周期）

print(min_period("abcabc"))    # 3
print(min_period("aaaaaa"))    # 1
print(min_period("abcab"))     # 5 (完全な周期ではない)
print(min_period("abababab"))  # 2
```

### KMP法の応用: 文字列中の全接頭辞の出現回数

```python
def count_prefix_occurrences(s: str) -> list:
    """文字列s中で、各接頭辞 s[0..k] が何回出現するか
    返り値: count[k] = s[0..k] の出現回数
    """
    n = len(s)
    lps = [0] * n
    length = 0
    i = 1

    while i < n:
        if s[i] == s[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                i += 1

    # 各接頭辞の出現回数を数える
    count = [0] * (n + 1)
    for i in range(n):
        count[lps[i]] += 1

    # lps の連鎖を辿って累積
    for i in range(n - 1, 0, -1):
        count[lps[i - 1]] += count[i]

    # 各接頭辞は少なくとも自分自身として1回出現
    for i in range(n + 1):
        count[i] += 1

    return count[1:]  # count[1] ~ count[n]

print(count_prefix_occurrences("ababa"))
# "a":3, "ab":2, "aba":2, "abab":1, "ababa":1
# → [3, 2, 2, 1, 1]
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
    if m > n:
        return []
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

### ローリングハッシュ（汎用版）

```python
class RollingHash:
    """汎用ローリングハッシュ（ダブルハッシュ対応）
    文字列の任意の部分文字列のハッシュを O(1) で計算
    """

    def __init__(self, s: str, base1: int = 131, mod1: int = 10**9 + 7,
                 base2: int = 137, mod2: int = 10**9 + 9):
        self.n = len(s)
        self.mod1, self.mod2 = mod1, mod2

        # ハッシュの前計算
        self.hash1 = [0] * (self.n + 1)
        self.hash2 = [0] * (self.n + 1)
        self.pow1 = [1] * (self.n + 1)
        self.pow2 = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash1[i + 1] = (self.hash1[i] * base1 + ord(s[i])) % mod1
            self.hash2[i + 1] = (self.hash2[i] * base2 + ord(s[i])) % mod2
            self.pow1[i + 1] = self.pow1[i] * base1 % mod1
            self.pow2[i + 1] = self.pow2[i] * base2 % mod2

    def get_hash(self, l: int, r: int) -> tuple:
        """部分文字列 s[l:r] のハッシュを返す（半開区間）- O(1)"""
        h1 = (self.hash1[r] - self.hash1[l] * self.pow1[r - l]) % self.mod1
        h2 = (self.hash2[r] - self.hash2[l] * self.pow2[r - l]) % self.mod2
        return (h1, h2)

    def lcp(self, i: int, j: int) -> int:
        """位置 i と位置 j から始まる最長共通接頭辞の長さ - O(log n)"""
        lo, hi = 0, min(self.n - i, self.n - j)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.get_hash(i, i + mid) == self.get_hash(j, j + mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

# 使用例
rh = RollingHash("abcabc")
print(rh.get_hash(0, 3) == rh.get_hash(3, 6))  # True ("abc" == "abc")
print(rh.get_hash(0, 3) == rh.get_hash(1, 4))  # False ("abc" != "bca")
print(rh.lcp(0, 3))  # 3 ("abc" と "abc" の共通接頭辞)

# 応用: 回文判定
def is_palindrome_hash(s: str, l: int, r: int) -> bool:
    """s[l:r] が回文かどうかをハッシュで判定"""
    rh_forward = RollingHash(s)
    rh_reverse = RollingHash(s[::-1])
    n = len(s)
    h_forward = rh_forward.get_hash(l, r)
    h_reverse = rh_reverse.get_hash(n - r, n - l)
    return h_forward == h_reverse
```

### Rabin-Karp の応用: 文字列のユニーク部分文字列の数

```python
def count_unique_substrings(s: str) -> int:
    """文字列のユニーク部分文字列の数をローリングハッシュで計算
    O(n^2) で全部分文字列のハッシュを集合に追加
    """
    rh = RollingHash(s)
    unique = set()
    n = len(s)

    for length in range(1, n + 1):
        for i in range(n - length + 1):
            h = rh.get_hash(i, i + length)
            unique.add(h)

    return len(unique)

print(count_unique_substrings("abc"))   # 6: a,b,c,ab,bc,abc
print(count_unique_substrings("aaa"))   # 3: a,aa,aaa
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

### Z-algorithm の応用

```python
def z_min_period(s: str) -> int:
    """Z配列を使って最小周期を求める"""
    n = len(s)
    z = z_function(s)

    for i in range(1, n):
        if i + z[i] == n and n % i == 0:
            return i
    return n

print(z_min_period("abcabc"))    # 3
print(z_min_period("abababab"))  # 2

def z_count_distinct_substrings(s: str) -> int:
    """Z配列で異なる部分文字列の数を数える - O(n^2)
    各接尾辞の Z 配列を使って「新しく追加される部分文字列」を数える
    """
    n = len(s)
    count = 0

    # 右から1文字ずつ追加していく
    current = ""
    for i in range(n - 1, -1, -1):
        current = s[i] + current
        z = z_function(current)
        # 最大の Z[j] (j >= 1) を求める
        max_z = max(z[1:]) if len(z) > 1 else 0
        # 新しい部分文字列の数 = 現在の長さ - 最大 Z 値
        count += len(current) - max_z

    return count

print(z_count_distinct_substrings("abc"))   # 6
print(z_count_distinct_substrings("aaa"))   # 3
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
    __slots__ = ['children', 'is_end', 'count', 'word']

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # この接頭辞を持つ単語数
        self.word = None  # 終端の場合、元の単語を保持

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
        node.word = word

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

    def auto_complete(self, prefix: str, limit: int = 10) -> list:
        """オートコンプリート"""
        node = self._find_node(prefix)
        if not node:
            return []

        results = []
        def dfs(node, path):
            if len(results) >= limit:
                return
            if node.is_end:
                results.append(prefix + path)
            for char, child in sorted(node.children.items()):
                dfs(child, path + char)

        dfs(node, "")
        return results

    def delete(self, word: str) -> bool:
        """単語を削除 - O(m)"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                node.word = None
                return len(node.children) == 0
            char = word[depth]
            if char not in node.children:
                return False

            child = node.children[char]
            child.count -= 1
            should_delete = _delete(child, word, depth + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end

            return False

        return _delete(self.root, word, 0)

    def longest_common_prefix(self) -> str:
        """Trie に格納された全単語の最長共通接頭辞"""
        prefix = []
        node = self.root
        while len(node.children) == 1 and not node.is_end:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]
        return ''.join(prefix)

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

### 配列ベースの高速 Trie

```python
class ArrayTrie:
    """配列ベースの Trie（高速版、小文字英字限定）
    競技プログラミング向け
    """

    def __init__(self, max_nodes: int = 500001):
        self.ALPHA = 26
        self.children = [[0] * self.ALPHA for _ in range(max_nodes)]
        self.is_end = [False] * max_nodes
        self.node_count = 1  # 0 がルート

    def _char_to_idx(self, c: str) -> int:
        return ord(c) - ord('a')

    def insert(self, word: str) -> None:
        node = 0
        for c in word:
            idx = self._char_to_idx(c)
            if self.children[node][idx] == 0:
                self.children[node][idx] = self.node_count
                self.node_count += 1
            node = self.children[node][idx]
        self.is_end[node] = True

    def search(self, word: str) -> bool:
        node = 0
        for c in word:
            idx = self._char_to_idx(c)
            if self.children[node][idx] == 0:
                return False
            node = self.children[node][idx]
        return self.is_end[node]
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

    def add_pattern(self, pattern: str, pattern_id=None):
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

### Aho-Corasick の応用: 禁止語フィルタリング

```python
def censor_text(text: str, forbidden_words: list) -> str:
    """テキストから禁止語を '*' で置換する
    Aho-Corasick で一括検出して効率的に処理
    """
    ac = AhoCorasick()
    for word in forbidden_words:
        ac.add_pattern(word.lower())
    ac.build()

    text_lower = text.lower()
    matches = ac.search(text_lower)

    # 置換する範囲を計算
    censored = list(text)
    for pos, pattern in matches:
        for i in range(pos, pos + len(pattern)):
            censored[i] = '*'

    return ''.join(censored)

text = "This is a bad example with some ugly words"
forbidden = ["bad", "ugly"]
print(censor_text(text, forbidden))
# "This is a *** example with some **** words"
```

---

## 8. Suffix Array（接尾辞配列）

文字列の全接尾辞をソートした配列。パターン検索、最長共通部分文字列など多くの問題に適用できる。

```
文字列: "banana"

接尾辞一覧:
  0: banana
  1: anana
  2: nana
  3: ana
  4: na
  5: a

ソート後:
  5: a
  3: ana
  1: anana
  0: banana
  4: na
  2: nana

Suffix Array: [5, 3, 1, 0, 4, 2]
```

```python
def build_suffix_array(s: str) -> list:
    """Suffix Array の構築 - O(n log^2 n)"""
    n = len(s)
    # (ランク, 次ランク, インデックス) でソート
    sa = list(range(n))
    rank = [ord(c) for c in s]
    tmp = [0] * n

    k = 1
    while k < n:
        def compare(i, j):
            if rank[i] != rank[j]:
                return -1 if rank[i] < rank[j] else 1
            ri = rank[i + k] if i + k < n else -1
            rj = rank[j + k] if j + k < n else -1
            return -1 if ri < rj else (0 if ri == rj else 1)

        from functools import cmp_to_key
        sa.sort(key=cmp_to_key(compare))

        # 新しいランクを計算
        tmp[sa[0]] = 0
        for i in range(1, n):
            tmp[sa[i]] = tmp[sa[i-1]]
            if compare(sa[i-1], sa[i]) < 0:
                tmp[sa[i]] += 1

        rank = tmp[:]
        if rank[sa[n-1]] == n - 1:
            break
        k *= 1 if k == 0 else k  # k を2倍に
        k *= 2

    return sa

def build_suffix_array_simple(s: str) -> list:
    """Suffix Array の構築（簡易版）- O(n log n * m)
    Pythonのソートを利用した簡潔な実装
    """
    n = len(s)
    return sorted(range(n), key=lambda i: s[i:])

# LCP Array（Kasai's Algorithm）
def build_lcp_array(s: str, sa: list) -> list:
    """LCP Array の構築（Kasai's Algorithm）- O(n)
    lcp[i] = sa[i] と sa[i+1] の最長共通接頭辞の長さ
    """
    n = len(s)
    rank = [0] * n
    for i in range(n):
        rank[sa[i]] = i

    lcp = [0] * (n - 1)
    h = 0

    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            lcp[rank[i] - 1] = h
            if h > 0:
                h -= 1
        else:
            h = 0

    return lcp

# 使用例
s = "banana"
sa = build_suffix_array_simple(s)
print(f"SA: {sa}")  # [5, 3, 1, 0, 4, 2]

lcp = build_lcp_array(s, sa)
print(f"LCP: {lcp}")  # [1, 3, 0, 0, 2]

# 接尾辞の表示
for i, idx in enumerate(sa):
    prefix = f"lcp={lcp[i]}" if i < len(lcp) else ""
    print(f"  sa[{i}]={idx}: {s[idx:]:10s}  {prefix}")
```

### Suffix Array の応用

```python
def search_with_suffix_array(text: str, pattern: str) -> list:
    """Suffix Array を使ったパターン検索 - O(m log n)"""
    sa = build_suffix_array_simple(text)
    n = len(text)
    m = len(pattern)

    # 二分探索で下界を見つける
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if text[sa[mid]:sa[mid] + m] < pattern:
            lo = mid + 1
        else:
            hi = mid
    left = lo

    # 二分探索で上界を見つける
    hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if text[sa[mid]:sa[mid] + m] <= pattern:
            lo = mid + 1
        else:
            hi = mid
    right = lo

    return sorted(sa[left:right])

text = "banana"
print(search_with_suffix_array(text, "ana"))  # [1, 3]
print(search_with_suffix_array(text, "ban"))  # [0]

def longest_repeated_substring(s: str) -> str:
    """最長反復部分文字列（文字列中に2回以上出現する最長部分文字列）
    LCP Array の最大値を使う
    """
    if len(s) <= 1:
        return ""

    sa = build_suffix_array_simple(s)
    lcp = build_lcp_array(s, sa)

    if not lcp:
        return ""

    max_lcp = max(lcp)
    max_idx = lcp.index(max_lcp)

    return s[sa[max_idx]:sa[max_idx] + max_lcp]

print(longest_repeated_substring("banana"))    # "ana"
print(longest_repeated_substring("abcabc"))    # "abc"
print(longest_repeated_substring("aabaaab"))   # "aab"

def count_distinct_substrings_sa(s: str) -> int:
    """Suffix Array + LCP で異なる部分文字列の数を計算 - O(n log n)
    全部分文字列数 n*(n+1)/2 から重複（LCPの総和）を引く
    """
    n = len(s)
    sa = build_suffix_array_simple(s)
    lcp = build_lcp_array(s, sa)

    total = n * (n + 1) // 2
    duplicates = sum(lcp)
    return total - duplicates

print(count_distinct_substrings_sa("abc"))    # 6
print(count_distinct_substrings_sa("aaa"))    # 3
print(count_distinct_substrings_sa("banana")) # 15
```

---

## 9. Manacher法（回文検出）

最長回文部分文字列を O(n) で求めるアルゴリズム。

```python
def manacher(s: str) -> list:
    """Manacher法 - 各位置を中心とする最長回文の半径を計算
    O(n)

    入力文字列を加工して偶数長回文も統一的に扱う:
    "abba" → "#a#b#b#a#"
    """
    # 文字間にダミー文字を挿入
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = 位置 i を中心とする回文の半径

    c = r = 0  # 最も右に到達した回文の中心と右端

    for i in range(n):
        # ミラーの位置
        mirror = 2 * c - i

        if i < r:
            p[i] = min(r - i, p[mirror])

        # 拡張を試みる
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0
               and t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1

        # 右端を超えたら更新
        if i + p[i] > r:
            c, r = i, i + p[i]

    return p

def longest_palindrome_substring(s: str) -> str:
    """最長回文部分文字列 - O(n)"""
    if not s:
        return ""

    t = '#' + '#'.join(s) + '#'
    p = manacher(s)

    # 加工後文字列での最長回文を見つける
    # p 配列から直接計算
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p2 = [0] * n
    c = r = 0

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p2[i] = min(r - i, p2[mirror])
        while (i + p2[i] + 1 < n and i - p2[i] - 1 >= 0
               and t[i + p2[i] + 1] == t[i - p2[i] - 1]):
            p2[i] += 1
        if i + p2[i] > r:
            c, r = i, i + p2[i]

    # 最大半径の位置
    max_len = max(p2)
    center = p2.index(max_len)

    # 元の文字列での開始位置と長さ
    start = (center - max_len) // 2
    return s[start:start + max_len]

print(longest_palindrome_substring("babad"))     # "bab" or "aba"
print(longest_palindrome_substring("cbbd"))      # "bb"
print(longest_palindrome_substring("abacaba"))   # "abacaba"

def count_palindromic_substrings(s: str) -> int:
    """回文部分文字列の総数 - O(n)"""
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0

    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0
               and t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]

    # # 位置の回文は偶数長回文に対応
    # 元文字位置の回文は奇数長回文に対応
    count = 0
    for i in range(n):
        # p[i] の中に含まれる回文の数
        if t[i] == '#':
            count += p[i] // 2  # 偶数長
        else:
            count += (p[i] + 1) // 2  # 奇数長

    return count

print(count_palindromic_substrings("abc"))   # 3 (a, b, c)
print(count_palindromic_substrings("aaa"))   # 6 (a,a,a,aa,aa,aaa)
```

---

## 10. Boyer-Moore法の概要

実用上最速の文字列検索アルゴリズム。パターンを右から左に比較し、不一致時に大きくシフトする。

```python
def boyer_moore_horspool(text: str, pattern: str) -> list:
    """Boyer-Moore-Horspool法（簡易版）
    平均 O(n/m), 最悪 O(nm)
    """
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Bad Character テーブルの構築
    bad_char = {}
    for i in range(m - 1):
        bad_char[pattern[i]] = m - 1 - i

    matches = []
    i = 0

    while i <= n - m:
        j = m - 1
        while j >= 0 and text[i + j] == pattern[j]:
            j -= 1

        if j < 0:
            matches.append(i)
            i += 1
        else:
            # Bad Character Rule でシフト
            skip = bad_char.get(text[i + m - 1], m)
            i += max(1, skip)

    return matches

print(boyer_moore_horspool("ABCABCABD", "ABCABD"))  # [3]

# Boyer-Moore が実用上最速な理由:
# - パターンを右から左に比較 → 不一致が早期に検出される
# - Bad Character Rule: 不一致文字がパターンに存在しなければ m 文字シフト
# - Good Suffix Rule: パターン内部の一致部分を活用してシフト
# - 英語テキストなどでは平均的に O(n/m) の比較回数
```

---

## 11. アルゴリズム比較表

| アルゴリズム | 前処理 | 検索 | 空間 | 特徴 |
|:---|:---|:---|:---|:---|
| ナイーブ | O(1) | O(nm) | O(1) | 実装最簡 |
| KMP | O(m) | O(n) | O(m) | 決定的、最悪O(n) |
| Rabin-Karp | O(m) | 平均O(n) | O(1) | 複数パターン対応 |
| Z-algorithm | O(n+m) | O(n+m) | O(n+m) | Z配列の汎用性 |
| Boyer-Moore | O(m+|Σ|) | 平均O(n/m) | O(m+|Σ|) | 実用上最速 |
| Aho-Corasick | O(Σm) | O(n+z) | O(Σm) | 複数パターン最速 |
| Suffix Array | O(n log n) | O(m log n) | O(n) | 複数検索に強い |

## 用途別選択ガイド

| 用途 | 推奨 | 理由 |
|:---|:---|:---|
| 単一パターン（確実） | KMP | 最悪O(n+m)保証 |
| 単一パターン（簡易） | Rabin-Karp | 実装が簡潔 |
| 単一パターン（高速） | Boyer-Moore | 平均的に最速 |
| 複数パターン同時検索 | Aho-Corasick | Trie+失敗関数で最適 |
| 接頭辞マッチ | Trie | オートコンプリートに最適 |
| テキストエディタ検索 | Boyer-Moore | 実用上最速 |
| DNA配列検索 | Suffix Array | 大量検索に強い |
| 回文検出 | Manacher | O(n)で全回文を検出 |
| 文字列比較（多数） | ローリングハッシュ | O(1)で部分文字列比較 |

---

## 12. 実務応用パターン

### ログ解析（複数パターンの検出）

```python
def analyze_logs(log_lines: list, error_patterns: list) -> dict:
    """ログファイルからエラーパターンを一括検出"""
    ac = AhoCorasick()
    for i, pattern in enumerate(error_patterns):
        ac.add_pattern(pattern.lower(), i)
    ac.build()

    results = {p: [] for p in error_patterns}

    for line_num, line in enumerate(log_lines, 1):
        matches = ac.search(line.lower())
        for _, pattern_id in matches:
            results[error_patterns[pattern_id]].append(line_num)

    return results

logs = [
    "2024-01-15 INFO Server started",
    "2024-01-15 ERROR Connection timeout",
    "2024-01-15 WARN Memory usage high",
    "2024-01-15 ERROR Disk full",
    "2024-01-15 ERROR Connection refused",
]
patterns = ["error", "timeout", "memory"]
result = analyze_logs(logs, patterns)
print(result)
# {'error': [2, 4, 5], 'timeout': [2], 'memory': [3]}
```

### DNA配列解析

```python
def find_motifs(dna: str, motifs: list) -> dict:
    """DNA配列中のモチーフ（特定パターン）を一括検索"""
    ac = AhoCorasick()
    for motif in motifs:
        ac.add_pattern(motif)
    ac.build()

    results = {m: [] for m in motifs}
    matches = ac.search(dna)

    for pos, motif in matches:
        results[motif].append(pos)

    return results

dna = "ATCGATCGATCGATCG"
motifs = ["ATC", "GAT", "ATCG"]
print(find_motifs(dna, motifs))
# {'ATC': [0, 4, 8, 12], 'GAT': [3, 7, 11], 'ATCG': [0, 4, 8, 12]}
```

### スペルチェッカー

```python
def spell_checker(dictionary: list, word: str, max_distance: int = 1) -> list:
    """Trieを使ったスペルチェッカー（編集距離ベース）"""
    trie = Trie()
    for w in dictionary:
        trie.insert(w)

    suggestions = []

    def search_recursive(node, char_idx, word, current_row, results):
        columns = len(word) + 1
        if node.is_end and current_row[-1] <= max_distance:
            results.append((node.word, current_row[-1]))

        for char, child_node in node.children.items():
            new_row = [current_row[0] + 1]
            for col in range(1, columns):
                insert_cost = new_row[col - 1] + 1
                delete_cost = current_row[col] + 1
                replace_cost = current_row[col - 1]
                if word[col - 1] != char:
                    replace_cost += 1
                new_row.append(min(insert_cost, delete_cost, replace_cost))

            if min(new_row) <= max_distance:
                search_recursive(child_node, char_idx, word, new_row, results)

    current_row = list(range(len(word) + 1))
    for char, node in trie.root.children.items():
        new_row = [1]
        for col in range(1, len(word) + 1):
            insert_cost = new_row[col - 1] + 1
            delete_cost = current_row[col] + 1
            replace_cost = current_row[col - 1]
            if word[col - 1] != char:
                replace_cost += 1
            new_row.append(min(insert_cost, delete_cost, replace_cost))

        if min(new_row) <= max_distance:
            search_recursive(node, 0, word, new_row, suggestions)

    return sorted(suggestions, key=lambda x: x[1])

dictionary = ["apple", "apply", "app", "ape", "maple", "application"]
print(spell_checker(dictionary, "aple"))
# [('ape', 1), ('apple', 1)]
```

---

## 13. アンチパターン

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
patterns = ["pattern1", "pattern2", "patternK"]
for p in patterns:
    kmp_search(text, p)  # k回繰り返し

# GOOD: Aho-Corasick で一括検索 → O(n + Σm + z)
ac = AhoCorasick()
for p in patterns:
    ac.add_pattern(p)
ac.build()
results = ac.search(text)  # 1回の走査
```

### アンチパターン3: ハッシュの衝突を無視

```python
# BAD: 単一ハッシュで偽陽性を検証しない
def bad_rabin_karp(text, pattern):
    # ハッシュが一致したら即マッチと判定
    if hash_match:
        matches.append(i)  # 偽陽性の可能性!

# GOOD: ダブルハッシュまたはハッシュ一致時に文字列比較
def good_rabin_karp(text, pattern):
    if hash_match:
        if text[i:i+m] == pattern:  # 検証
            matches.append(i)
```

### アンチパターン4: Trie のメモリ消費を考慮しない

```python
# BAD: 大量の長い文字列を辞書型 Trie に格納
# 各ノードが dict を持つ → メモリ消費が大きい

# GOOD:
# 1. 配列ベースの Trie（文字種が限定的な場合）
# 2. ダブル配列 Trie（メモリ効率が良い）
# 3. Patricia Trie / Radix Tree（共通接頭辞を圧縮）
```

---

## 14. FAQ

### Q1: KMPとZ-algorithmはどちらが良いか？

**A:** 計算量は同じ O(n+m)。KMP は失敗関数でオンライン処理（ストリーム対応）が可能。Z-algorithm は Z 配列の構築が直感的で、文字列問題全般に応用しやすい。競技プログラミングでは Z-algorithm のほうが汎用性が高いと人気。

### Q2: Rabin-Karp法のハッシュ衝突の対策は？

**A:** (1) 大きな素数 p を使う。(2) 複数のハッシュ関数を併用（ダブルハッシュ）。(3) ハッシュ一致時に文字列比較で確認。理論上の最悪は O(nm) だが、良いハッシュ関数なら実用上ほぼ O(n+m)。

### Q3: Suffix Array/Suffix Tree とは何か？

**A:** Suffix Array は文字列の全接尾辞をソートした配列で、O(n log n) で構築、O(m log n) でパターン検索できる。Suffix Tree は全接尾辞を Trie に格納した構造で O(n) で構築可能だが、メモリ消費が大きい。大量の検索クエリを処理する場合に有効。

### Q4: ローリングハッシュで注意すべき点は？

**A:** (1) MOD が小さいと衝突率が高い（10^9+7 以上推奨）。(2) base は入力に依存しない乱数が理想的。(3) ダブルハッシュで衝突確率を 1/(mod1 * mod2) に下げる。(4) 負のハッシュ値に注意（Python は % で正になるが他言語では注意）。

### Q5: 実務で最も使われる文字列検索アルゴリズムは？

**A:** ほとんどのプログラミング言語の標準ライブラリでは Boyer-Moore の変種が使われている（Python の `str.find()`、grep のデフォルトアルゴリズムなど）。セキュリティ用途（IDSのパターンマッチ等）では Aho-Corasick が広く使われている。

---

## 15. まとめ

| 項目 | 要点 |
|:---|:---|
| KMP | 失敗関数でシフト量を最適化。O(n+m) 保証 |
| Rabin-Karp | ローリングハッシュで高速比較。複数パターン対応 |
| Z-algorithm | Z 配列で接頭辞一致長を計算。汎用性が高い |
| Boyer-Moore | 右から左に比較。実用上最速 |
| Trie | 文字列集合の木構造。接頭辞検索・オートコンプリート |
| Aho-Corasick | Trie + 失敗関数で複数パターンを O(n+Σm+z) で検索 |
| Suffix Array | 全接尾辞のソート配列。LCPと組み合わせて強力 |
| Manacher | 回文検出を O(n) で実行 |
| ローリングハッシュ | O(1) で部分文字列のハッシュ比較 |

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
5. Manacher, G. (1975). "A New Linear-Time On-Line Algorithm for Finding the Smallest Initial Palindrome of a String." *JACM*.
6. Kasai, T. et al. (2001). "Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications." *CPM*.
