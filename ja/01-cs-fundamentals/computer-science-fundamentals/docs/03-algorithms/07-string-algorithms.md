# 文字列アルゴリズム

> テキスト検索は最も身近なアルゴリズムの1つ。ブラウザのCtrl+F、grepコマンド、IDEの検索——全てが文字列マッチングに依存している。

## この章で学ぶこと

- [ ] ナイーブな文字列検索の問題点を理解する
- [ ] KMP法の原理を説明できる
- [ ] Rabin-Karp法のローリングハッシュを理解する
- [ ] Boyer-Moore法のスキップ戦略を理解する
- [ ] Trie、サフィックス配列などの文字列データ構造を実装できる
- [ ] 正規表現とオートマトンの関係を理解する
- [ ] 実務での文字列処理の最適化手法を習得する

## 前提知識


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

# 最悪ケースの例:
# text = "A" * 1000000 + "B"
# pattern = "A" * 999 + "B"
# → 約100万回 × 1000文字比較 = 10億回の比較
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

#### KMPの失敗関数の詳細解説

```python
def build_failure_function(pattern):
    """失敗関数を構築し、各ステップを可視化"""
    m = len(pattern)
    failure = [0] * m
    j = 0

    print(f"パターン: {pattern}")
    print(f"{'i':>3} {'pattern[i]':>10} {'j':>3} {'一致?':>5} {'failure[i]':>10}")
    print("-" * 40)

    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]

        match = pattern[i] == pattern[j]
        if match:
            j += 1

        failure[i] = j
        print(f"{i:>3} {pattern[i]:>10} {j:>3} {'YES' if match else 'NO':>5} {failure[i]:>10}")

    return failure

# 例: パターン "ABABAC"
# failure = [0, 0, 1, 2, 3, 0]
#
# 解釈:
# failure[i] = pattern[0:i+1] の最長の「一致する接頭辞と接尾辞」の長さ
#
# i=0: "A"       → 0 (自明)
# i=1: "AB"      → 0 (接頭辞 "A" ≠ 接尾辞 "B")
# i=2: "ABA"     → 1 (接頭辞 "A" = 接尾辞 "A")
# i=3: "ABAB"    → 2 (接頭辞 "AB" = 接尾辞 "AB")
# i=4: "ABABA"   → 3 (接頭辞 "ABA" = 接尾辞 "ABA")
# i=5: "ABABAC"  → 0 (一致なし)
```

#### KMP法の動作可視化

```python
def kmp_search_verbose(text, pattern):
    """KMP法の動作を可視化"""
    m = len(pattern)
    failure = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = failure[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        failure[i] = j

    print(f"失敗関数: {failure}")
    print(f"テキスト:   {text}")
    print()

    positions = []
    j = 0
    comparisons = 0

    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            print(f"  不一致 text[{i}]='{text[i]}' vs pattern[{j}]='{pattern[j]}'"
                  f" → j を {failure[j-1]} に巻き戻し")
            j = failure[j - 1]
            comparisons += 1

        comparisons += 1
        if text[i] == pattern[j]:
            j += 1
            if j == m:
                positions.append(i - m + 1)
                print(f"  ★ マッチ！ 位置 {i - m + 1}")
                j = failure[j - 1]
        else:
            pass

    print(f"\n比較回数: {comparisons}")
    print(f"ナイーブ法の最悪: {len(text) * len(pattern)}")
    return positions

# 使用例
kmp_search_verbose("ABABDABABABABAC", "ABABAC")
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

#### 複数パターンの同時検索

```python
def rabin_karp_multi(text, patterns):
    """複数パターンを同時に検索"""
    n = len(text)
    base, mod = 256, 10**9 + 7
    results = {p: [] for p in patterns}

    # パターンを長さごとにグループ化
    by_length = {}
    for p in patterns:
        m = len(p)
        if m not in by_length:
            by_length[m] = {}
        # ハッシュ値を計算
        h = 0
        for c in p:
            h = (h * base + ord(c)) % mod
        by_length[m][h] = by_length[m].get(h, []) + [p]

    # 各長さグループごとに検索
    for m, hash_to_patterns in by_length.items():
        if m > n:
            continue

        h_pow = pow(base, m - 1, mod)

        # テキストのハッシュ値
        text_hash = 0
        for i in range(m):
            text_hash = (text_hash * base + ord(text[i])) % mod

        for i in range(n - m + 1):
            if text_hash in hash_to_patterns:
                for p in hash_to_patterns[text_hash]:
                    if text[i:i+m] == p:
                        results[p].append(i)

            if i < n - m:
                text_hash = ((text_hash - ord(text[i]) * h_pow) * base
                            + ord(text[i + m])) % mod

    return results

# 使用例
text = "she sells seashells by the seashore"
patterns = ["she", "sea", "sell", "shore"]
result = rabin_karp_multi(text, patterns)
# {'she': [0, 15], 'sea': [10, 27], 'sell': [4], 'shore': [30]}
```

### 1.4 Boyer-Moore法

```python
def boyer_moore(text, pattern):
    """Boyer-Moore法: パターン末尾から比較し、不一致時に大きくスキップ"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Bad Character テーブルの構築
    bad_char = {}
    for i in range(m):
        bad_char[pattern[i]] = i  # 各文字のパターン内での最後の出現位置

    positions = []
    i = 0  # テキスト上の位置

    while i <= n - m:
        j = m - 1  # パターン末尾から比較

        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            # マッチ！
            positions.append(i)
            # 次のマッチを探す
            i += (m - bad_char.get(text[i + m], -1) if i + m < n else 1)
        else:
            # Bad Characterルール
            bad_pos = bad_char.get(text[i + j], -1)
            shift = max(1, j - bad_pos)
            i += shift

    return positions

# Boyer-Mooreの特徴:
# - パターン末尾から比較（右→左）
# - 不一致文字がパターンに含まれなければ、パターン長分スキップ
# - 最善ケース: O(n/m) — パターンが長いほど高速！
# - 最悪ケース: O(nm) — Good Suffixルールを追加するとO(n)に改善
# - 実用上最も高速な文字列検索アルゴリズム（GNU grepが採用）

# Boyer-Mooreの動作例:
# text = "HERE IS A SIMPLE EXAMPLE"
# pattern = "EXAMPLE"
#
# ステップ1: E と S を比較 → 不一致、Sはパターンに無い → 7文字スキップ
# ステップ2: 次の位置で比較 → ...
# → ナイーブ法より大幅に少ない比較回数で済む
```

#### Good Suffix ルールの実装

```python
def boyer_moore_full(text, pattern):
    """Boyer-Moore法のフル実装（Bad Character + Good Suffix）"""
    n, m = len(text), len(pattern)
    if m > n:
        return []

    # Bad Character テーブル
    bad_char = [-1] * 256
    for i in range(m):
        bad_char[ord(pattern[i])] = i

    # Good Suffix テーブル
    good_suffix = [0] * (m + 1)
    border = [0] * (m + 1)

    # Case 1: パターン内にgood suffixの一致がある場合
    i = m
    j = m + 1
    border[i] = j
    while i > 0:
        while j <= m and pattern[i - 1] != pattern[j - 1]:
            if good_suffix[j] == 0:
                good_suffix[j] = j - i
            j = border[j]
        i -= 1
        j -= 1
        border[i] = j

    # Case 2: good suffixの接頭辞がパターンの接頭辞に一致する場合
    j = border[0]
    for i in range(m + 1):
        if good_suffix[i] == 0:
            good_suffix[i] = j
        if i == j:
            j = border[j]

    # 検索
    positions = []
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            positions.append(i)
            i += good_suffix[0]
        else:
            bc_shift = j - bad_char[ord(text[i + j])]
            gs_shift = good_suffix[j + 1]
            i += max(bc_shift, gs_shift)

    return positions

# 計算量: O(n + m) 前処理 + O(n) 検索
# 最善: O(n/m) — パターンが長いほど高速
```

### 1.5 Aho-Corasick法

```python
from collections import deque

class AhoCorasick:
    """Aho-Corasick法: 複数パターンの同時検索（線形時間）"""

    def __init__(self):
        self.goto = [{}]
        self.failure = [0]
        self.output = [[]]

    def add_pattern(self, pattern, idx):
        """パターンを追加"""
        state = 0
        for char in pattern:
            if char not in self.goto[state]:
                self.goto[state][char] = len(self.goto)
                self.goto.append({})
                self.failure.append(0)
                self.output.append([])
            state = self.goto[state][char]
        self.output[state].append(idx)

    def build(self):
        """失敗関数を構築（BFS）"""
        queue = deque()

        # 深さ1のノードの失敗関数は0（ルート）
        for char, state in self.goto[0].items():
            queue.append(state)

        while queue:
            u = queue.popleft()
            for char, v in self.goto[u].items():
                queue.append(v)

                # 失敗関数の計算
                state = self.failure[u]
                while state != 0 and char not in self.goto[state]:
                    state = self.failure[state]
                self.failure[v] = self.goto[state].get(char, 0)
                if self.failure[v] == v:
                    self.failure[v] = 0

                # 出力関数の更新
                self.output[v] = self.output[v] + self.output[self.failure[v]]

    def search(self, text):
        """テキスト内のパターンを全て検索"""
        results = []
        state = 0

        for i, char in enumerate(text):
            while state != 0 and char not in self.goto[state]:
                state = self.failure[state]
            state = self.goto[state].get(char, 0)

            for pattern_idx in self.output[state]:
                results.append((i, pattern_idx))

        return results

# 使用例
ac = AhoCorasick()
patterns = ["he", "she", "his", "hers"]
for i, p in enumerate(patterns):
    ac.add_pattern(p, i)
ac.build()

text = "ahishers"
results = ac.search(text)
for pos, idx in results:
    p = patterns[idx]
    print(f"パターン '{p}' が位置 {pos - len(p) + 1} で発見")

# 計算量: O(n + m + z)
#   n: テキスト長、m: 全パターンの総文字数、z: マッチ数
# 用途:
# - ウイルススキャナ（シグネチャの検索）
# - ネットワークIDSのパケット検査
# - テキストフィルタリング（NGワード検出）
# - DNA配列の複数モチーフ検索
```

### 1.6 Z-Algorithm

```python
def z_function(s):
    """Z配列を計算: z[i] = s[i:]とs[0:]の最長共通接頭辞の長さ"""
    n = len(s)
    z = [0] * n
    z[0] = n
    l, r = 0, 0

    for i in range(1, n):
        if i < r:
            z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > r:
            l, r = i, i + z[i]

    return z

def z_search(text, pattern):
    """Z-Algorithmを使ったパターンマッチング"""
    # パターン$テキスト を連結
    concat = pattern + "$" + text
    z = z_function(concat)
    m = len(pattern)

    positions = []
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            positions.append(i - m - 1)

    return positions

# 計算量: O(n + m)
# KMPと同じ線形時間だが、実装が直感的
# Z配列は文字列の繰り返しパターンの検出にも有用

# Z配列の使用例: 最小周期の検出
def min_period(s):
    """文字列の最小周期を求める"""
    z = z_function(s)
    n = len(s)
    for period in range(1, n + 1):
        if n % period == 0 and z[period] == n - period:
            return period
    return n

# 例: "abcabc" → 最小周期 3 ("abc")
# 例: "abab" → 最小周期 2 ("ab")
# 例: "abcde" → 最小周期 5 (全体)
```

---

## 2. 文字列データ構造

### 2.1 Trie（トライ木）

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # この接頭辞を持つ単語の数

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
            node.count += 1
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

    def count_prefix(self, prefix):
        """prefixで始まる単語の数"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count

    def autocomplete(self, prefix, limit=10):
        """オートコンプリート: prefixで始まる単語を返す"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect_words(node, prefix, results, limit)
        return results

    def _collect_words(self, node, current, results, limit):
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(current)
        for char in sorted(node.children):
            self._collect_words(node.children[char], current + char, results, limit)

    def delete(self, word):
        """単語を削除"""
        def _delete(node, word, depth):
            if depth == len(word):
                if not node.is_end:
                    return False
                node.is_end = False
                return len(node.children) == 0

            char = word[depth]
            if char not in node.children:
                return False

            should_delete = _delete(node.children[char], word, depth + 1)

            if should_delete:
                del node.children[char]
                return not node.is_end and len(node.children) == 0

            node.children[char].count -= 1
            return False

        _delete(self.root, word, 0)

# 用途:
# - オートコンプリート（検索候補の表示）
# - スペルチェック
# - IPルーティング（最長前置一致）
# - 辞書の高速検索
# - 電話番号検索
# 計算量: 挿入/検索 O(m) — mは文字列長
# 空間: O(Σ × N) — Σはアルファベットサイズ、Nはノード数

# 使用例: オートコンプリート
trie = Trie()
words = ["apple", "application", "apply", "ape", "banana", "band", "bank"]
for w in words:
    trie.insert(w)

print(trie.autocomplete("app"))  # ["apple", "application", "apply"]
print(trie.count_prefix("app"))  # 3
print(trie.count_prefix("ban"))  # 3
```

#### 圧縮Trie（Patricia Trie / Radix Tree）

```python
class CompressedTrieNode:
    def __init__(self, label=""):
        self.label = label       # 辺のラベル（複数文字可能）
        self.children = {}
        self.is_end = False

class CompressedTrie:
    """圧縮Trie: 1文字ずつでなく、共通接頭辞をまとめる"""

    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word):
        node = self.root
        i = 0

        while i < len(word):
            first_char = word[i]

            if first_char not in node.children:
                new_node = CompressedTrieNode(word[i:])
                new_node.is_end = True
                node.children[first_char] = new_node
                return

            child = node.children[first_char]
            label = child.label

            # 共通接頭辞の長さを求める
            j = 0
            while j < len(label) and i + j < len(word) and label[j] == word[i + j]:
                j += 1

            if j == len(label):
                # ラベル全体が一致 → 子ノードへ
                i += j
                node = child
            else:
                # 途中で分岐 → ノードを分割
                # 共通部分の新ノード
                split = CompressedTrieNode(label[:j])
                split.children[label[j]] = child
                child.label = label[j:]

                if i + j < len(word):
                    new_node = CompressedTrieNode(word[i + j:])
                    new_node.is_end = True
                    split.children[word[i + j]] = new_node
                else:
                    split.is_end = True

                node.children[first_char] = split
                return

        node.is_end = True

    def search(self, word):
        node = self.root
        i = 0

        while i < len(word):
            first_char = word[i]
            if first_char not in node.children:
                return False

            child = node.children[first_char]
            label = child.label

            if not word[i:].startswith(label):
                return False

            i += len(label)
            node = child

        return node.is_end

# 通常のTrie vs 圧縮Trie:
# 通常: "application" → a → p → p → l → i → c → a → t → i → o → n (11ノード)
# 圧縮: "application" → "application" (1ノード)
# 空間効率が大幅に改善される（特にキーが長い場合）
```

### 2.2 サフィックス配列

```python
def build_suffix_array(s):
    """サフィックス配列を構築（O(n log^2 n) 版）"""
    n = len(s)
    # 各接尾辞を(ランク, 次のランク, 位置)でソート
    suffixes = [(ord(s[i]), ord(s[i + 1]) if i + 1 < n else -1, i)
                for i in range(n)]
    suffixes.sort()

    # ランクの更新
    rank = [0] * n
    rank[suffixes[0][2]] = 0
    for i in range(1, n):
        rank[suffixes[i][2]] = rank[suffixes[i-1][2]]
        if suffixes[i][:2] != suffixes[i-1][:2]:
            rank[suffixes[i][2]] += 1

    k = 2
    while k < n:
        # (rank[i], rank[i+k]) でソート
        suffixes = [(rank[i], rank[i + k] if i + k < n else -1, i)
                    for i in range(n)]
        suffixes.sort()

        new_rank = [0] * n
        new_rank[suffixes[0][2]] = 0
        for i in range(1, n):
            new_rank[suffixes[i][2]] = new_rank[suffixes[i-1][2]]
            if suffixes[i][:2] != suffixes[i-1][:2]:
                new_rank[suffixes[i][2]] += 1
        rank = new_rank
        k *= 2

    return [s[2] for s in suffixes]

# サフィックス配列の使用例
s = "banana"
sa = build_suffix_array(s)
print(f"サフィックス配列: {sa}")
# [5, 3, 1, 0, 4, 2] → a, ana, anana, banana, na, nana

# パターン検索: 二分探索で O(m log n)
def search_with_suffix_array(text, sa, pattern):
    """サフィックス配列でパターンを検索"""
    n = len(text)
    m = len(pattern)
    lo, hi = 0, n - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = text[sa[mid]:sa[mid] + m]

        if suffix == pattern:
            # マッチ → 全ての出現を探す
            start = end = mid
            while start > 0 and text[sa[start-1]:sa[start-1]+m] == pattern:
                start -= 1
            while end < n - 1 and text[sa[end+1]:sa[end+1]+m] == pattern:
                end += 1
            return [sa[i] for i in range(start, end + 1)]
        elif suffix < pattern:
            lo = mid + 1
        else:
            hi = mid - 1

    return []

# LCP配列（Longest Common Prefix）
def build_lcp_array(text, sa):
    """Kasaiのアルゴリズムで LCP 配列を構築 O(n)"""
    n = len(text)
    rank = [0] * n
    lcp = [0] * n

    for i in range(n):
        rank[sa[i]] = i

    k = 0
    for i in range(n):
        if rank[i] == 0:
            k = 0
            continue
        j = sa[rank[i] - 1]
        while i + k < n and j + k < n and text[i + k] == text[j + k]:
            k += 1
        lcp[rank[i]] = k
        if k > 0:
            k -= 1

    return lcp

# LCP配列の用途:
# - 最長反復部分文字列の検出: max(lcp)
# - 異なる部分文字列の数: n*(n+1)/2 - sum(lcp)
# - 最長共通部分文字列（2つの文字列を結合してSA+LCPを構築）
```

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
  - 最長反復部分文字列
  - 異なる部分文字列のカウント
```

### 2.3 サフィックスツリー（概要）

```
サフィックスツリー: 全ての接尾辞を圧縮Trieに格納

  文字列 "banana$" のサフィックスツリー:

  root
  ├── "a" ── "na" ── "na$"
  │          └── "$"
  ├── "banana$"
  ├── "na" ── "na$"
  │          └── "$"
  └── "$"

  特徴:
  - 構築: O(n) (Ukkonenのアルゴリズム)
  - パターン検索: O(m)
  - 最長反復部分文字列: O(n)
  - 最長共通部分文字列: O(n + m)

  サフィックスツリー vs サフィックス配列:
  ┌───────────────┬──────────────────┬──────────────────┐
  │ 特性           │ サフィックスツリー │ サフィックス配列 │
  ├───────────────┼──────────────────┼──────────────────┤
  │ 空間           │ O(n) だが定数大   │ O(n) で効率的    │
  │ 構築           │ O(n) Ukkonen     │ O(n) SA-IS      │
  │ パターン検索   │ O(m)             │ O(m log n)       │
  │ 実装の複雑さ   │ 高い              │ 中程度           │
  │ キャッシュ効率 │ 低い              │ 高い             │
  └───────────────┴──────────────────┴──────────────────┘

  実務的にはサフィックス配列 + LCP配列が主流
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

### 3.2 有限オートマトンの基礎

```python
# DFA（決定性有限オートマトン）の実装
class DFA:
    """決定性有限オートマトン"""
    def __init__(self, states, alphabet, transitions, start, accepts):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # {(state, char): next_state}
        self.start = start
        self.accepts = accepts

    def accepts_string(self, s):
        state = self.start
        for char in s:
            key = (state, char)
            if key not in self.transitions:
                return False
            state = self.transitions[key]
        return state in self.accepts

# 例: "ab" を含む文字列を受理するDFA
dfa = DFA(
    states={0, 1, 2},
    alphabet={'a', 'b'},
    transitions={
        (0, 'a'): 1, (0, 'b'): 0,
        (1, 'a'): 1, (1, 'b'): 2,
        (2, 'a'): 2, (2, 'b'): 2,
    },
    start=0,
    accepts={2}
)

print(dfa.accepts_string("aab"))     # True
print(dfa.accepts_string("ba"))      # False
print(dfa.accepts_string("bab"))     # True

# NFA → DFA変換（部分集合構成法）
# 理論的には状態数が指数的に増加する可能性がある
# 実用的にはほとんどの場合、管理可能なサイズ
```

### 3.3 簡易正規表現エンジン

```python
class SimpleRegex:
    """簡易正規表現エンジン（Thompson NFA構成法）
    サポート: リテラル文字, '.', '*', '+', '?', '|', '(', ')'
    """

    class State:
        def __init__(self):
            self.transitions = {}  # char -> [State]
            self.epsilon = []      # epsilon遷移
            self.is_accept = False

    class NFA:
        def __init__(self, start, accept):
            self.start = start
            self.accept = accept

    @staticmethod
    def char_nfa(c):
        """単一文字のNFA"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True
        start.transitions[c] = [accept]
        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def concat(nfa1, nfa2):
        """連結"""
        nfa1.accept.is_accept = False
        nfa1.accept.epsilon.append(nfa2.start)
        return SimpleRegex.NFA(nfa1.start, nfa2.accept)

    @staticmethod
    def union(nfa1, nfa2):
        """選択（|）"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True

        start.epsilon.extend([nfa1.start, nfa2.start])
        nfa1.accept.is_accept = False
        nfa2.accept.is_accept = False
        nfa1.accept.epsilon.append(accept)
        nfa2.accept.epsilon.append(accept)

        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def star(nfa):
        """クリーネ閉包（*）"""
        start = SimpleRegex.State()
        accept = SimpleRegex.State()
        accept.is_accept = True

        start.epsilon.extend([nfa.start, accept])
        nfa.accept.is_accept = False
        nfa.accept.epsilon.extend([nfa.start, accept])

        return SimpleRegex.NFA(start, accept)

    @staticmethod
    def match(nfa, text):
        """NFAシミュレーションで文字列をマッチ"""
        def epsilon_closure(states):
            stack = list(states)
            closure = set(states)
            while stack:
                state = stack.pop()
                for next_state in state.epsilon:
                    if next_state not in closure:
                        closure.add(next_state)
                        stack.append(next_state)
            return closure

        current = epsilon_closure({nfa.start})

        for char in text:
            next_states = set()
            for state in current:
                if char in state.transitions:
                    next_states.update(state.transitions[char])
                if '.' in state.transitions:  # ワイルドカード
                    next_states.update(state.transitions['.'])
            current = epsilon_closure(next_states)

        return any(state.is_accept for state in current)

# Thompson NFA の利点:
# - 常に O(n × m) の計算量（バックトラックなし）
# - 壊滅的バックトラックが起きない
# - RE2, Go の正規表現エンジンが採用
```

### 3.4 危険な正規表現パターン（ReDoS）

```python
import re
import time

# ReDoS（正規表現サービス拒否攻撃）の例

# 危険なパターン: ネストした量詞
dangerous_patterns = [
    r"(a+)+b",           # ネストした繰り返し
    r"(a|aa)+b",         # 選択 + 繰り返し
    r"(.*a){x}",         # .* + 繰り返し
    r"([a-zA-Z]+)*@",    # 文字クラス + ネスト
]

# 安全なパターンへの書き換え例:
# 危険: (a+)+b   → 安全: a+b
# 危険: (a|aa)+b → 安全: a+b
# 危険: ([a-zA-Z]+)*@ → 安全: [a-zA-Z]+@

# ReDoSの検出チェックリスト:
# 1. ネストした量詞 (X+)+ や (X*)*
# 2. 重なりのある選択肢 (a|a)+ や (a|ab)+
# 3. 量詞の後に失敗しやすいパターン
# 4. バックリファレンスの多用

# 対策:
# 1. 正規表現のタイムアウト設定
# 2. 入力の長さ制限
# 3. RE2 や Go の正規表現エンジンを使用
# 4. 正規表現の静的解析ツール（redos-checker等）
# 5. 可能であれば正規表現を使わない文字列処理に置き換え

# Pythonでのタイムアウト付き正規表現
import signal

def regex_with_timeout(pattern, text, timeout=1):
    """タイムアウト付き正規表現マッチ（Unix系のみ）"""
    def handler(signum, frame):
        raise TimeoutError("正規表現の実行がタイムアウトしました")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)

    try:
        result = re.match(pattern, text)
        signal.alarm(0)
        return result
    except TimeoutError:
        return None
```

### 3.5 実務での文字列処理

```python
# 実務でよく使う文字列操作の計算量

s = "Hello, World!"

# O(n) 操作:
s.find("World")      # 線形検索
s.replace("o", "0")  # 全置換
s.split(",")          # 分割
"".join(parts)        # 結合

# 注意: 文字列の連結
# ❌ O(n^2): ループ内で文字列連結
result = ""
for word in words:
    result += word  # 毎回新しい文字列を生成 → O(n^2)

# ✅ O(n): joinを使用
result = "".join(words)

# ✅ O(n): io.StringIO
from io import StringIO
buf = StringIO()
for word in words:
    buf.write(word)
result = buf.getvalue()
```

#### 文字列処理のベストプラクティス

```python
import re

# 1. 正規表現のプリコンパイル
# ❌ 毎回コンパイル
for line in lines:
    if re.match(r"\d{3}-\d{4}", line):
        pass

# ✅ プリコンパイル
pattern = re.compile(r"\d{3}-\d{4}")
for line in lines:
    if pattern.match(line):
        pass

# 2. 文字列のフォーマット
name = "World"
# ❌ 遅い
result = "Hello, " + name + "!"
# ✅ f-string（Python 3.6+、最も高速）
result = f"Hello, {name}!"

# 3. 大量の文字列操作
# ❌ 文字列の繰り返し連結
s = ""
for i in range(10000):
    s += str(i)

# ✅ リストに溜めてjoin
parts = []
for i in range(10000):
    parts.append(str(i))
s = "".join(parts)

# ✅ ジェネレータ式
s = "".join(str(i) for i in range(10000))

# 4. 文字列のスライスは新しいオブジェクトを生成する
# Python: s[i:j] はコピー O(j-i)
# 大量のスライスを避ける → memoryview や index で対処

# 5. 各言語での文字列ビルダー
# Python: "".join(list) or io.StringIO
# Java: StringBuilder
# C#: StringBuilder
# Go: strings.Builder
# Rust: String::push_str or format!
# JavaScript: Array.join() or template literals
```

#### Unicode文字列の扱い

```python
# Unicodeの注意点

# 1. 文字数 vs バイト数
s = "こんにちは"
print(len(s))                    # 5（文字数）
print(len(s.encode('utf-8')))    # 15（UTF-8バイト数）
print(len(s.encode('utf-16')))   # 12（UTF-16バイト数、BOM含む）

# 2. サロゲートペア
emoji = "😀"
print(len(emoji))                        # 1（Python）
# JavaScriptでは: "😀".length = 2（UTF-16サロゲートペア）

# 3. 結合文字（Combining Characters）
# "が" = "か" + "゛" の場合がある
import unicodedata
s1 = "が"                              # 1文字
s2 = "か\u3099"                        # か + 濁点 = が
print(s1 == s2)                         # False!
print(unicodedata.normalize('NFC', s1) ==
      unicodedata.normalize('NFC', s2)) # True

# 4. 文字列の比較は正規化後に行う
def safe_compare(s1, s2):
    return unicodedata.normalize('NFC', s1) == unicodedata.normalize('NFC', s2)

# 5. 書記素クラスタ（Grapheme Cluster）
# "👨‍👩‍👧‍👦" は1つの絵文字だが、内部的には複数のコードポイント
family = "👨‍👩‍👧‍👦"
print(len(family))  # 7（Python）— ZWJ (Zero Width Joiner) を含む
# 正しく1文字として扱うには grapheme ライブラリが必要
```

---

## 4. 高度な文字列アルゴリズム

### 4.1 Manacher法（最長回文部分文字列）

```python
def manacher(s):
    """全ての位置を中心とする最長回文を O(n) で求める"""
    # 文字間に'#'を挿入: "abc" → "#a#b#c#"
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = 位置iを中心とする回文の半径

    center = right = 0  # 最も右まで届いている回文
    for i in range(n):
        # ミラーを利用
        mirror = 2 * center - i
        if i < right:
            p[i] = min(right - i, p[mirror])

        # 拡張を試みる
        while (i + p[i] + 1 < n and i - p[i] - 1 >= 0 and
               t[i + p[i] + 1] == t[i - p[i] - 1]):
            p[i] += 1

        # 右端を更新
        if i + p[i] > right:
            center = i
            right = i + p[i]

    # 最長回文を見つける
    max_len = max(p)
    max_center = p.index(max_len)
    start = (max_center - max_len) // 2

    return s[start:start + max_len]

# 例:
print(manacher("babad"))    # "bab" or "aba"
print(manacher("cbbd"))     # "bb"
print(manacher("racecar"))  # "racecar"

# 計算量: O(n) — ナイーブ法の O(n^2) から大幅改善
# ミラー利用が計算量削減の鍵
```

### 4.2 Burrows-Wheeler変換（BWT）

```python
def bwt_encode(s):
    """Burrows-Wheeler変換（圧縮のための前処理）"""
    s = s + '$'  # 終端文字を追加
    n = len(s)

    # 全ての巡回シフトをソート
    rotations = sorted(range(n), key=lambda i: s[i:] + s[:i])

    # 最後の列を取得
    bwt = ''.join(s[(i - 1) % n] for i in rotations)
    # 元の文字列の位置
    original_idx = rotations.index(0)

    return bwt, original_idx

def bwt_decode(bwt, idx):
    """BWT逆変換"""
    n = len(bwt)
    table = [''] * n

    for _ in range(n):
        table = sorted(bwt[i] + table[i] for i in range(n))

    for row in table:
        if row.endswith('$'):
            return row[:-1]

# 使用例
text = "banana"
encoded, idx = bwt_encode(text)
print(f"BWT: {encoded}")  # "annb$aa"
decoded = bwt_decode(encoded, idx)
print(f"復元: {decoded}")  # "banana"

# BWTの用途:
# - bzip2 圧縮の核心技術
# - FM-Index（全文検索のインデックス）
# - DNA配列のアラインメント（BWA, Bowtie）
# BWTは同じ文字を近くに集める効果がある → Run-Length Encoding で効率的に圧縮
```

### 4.3 文字列ハッシュの応用

```python
class RollingHash:
    """ローリングハッシュ: 部分文字列のハッシュを O(1) で計算"""

    def __init__(self, s, base=131, mod=10**18 + 9):
        self.n = len(s)
        self.base = base
        self.mod = mod

        # 前処理: ハッシュ値と基数のべき乗を計算
        self.hash = [0] * (self.n + 1)
        self.power = [1] * (self.n + 1)

        for i in range(self.n):
            self.hash[i + 1] = (self.hash[i] * base + ord(s[i])) % mod
            self.power[i + 1] = (self.power[i] * base) % mod

    def get_hash(self, l, r):
        """部分文字列 s[l:r] のハッシュ値を O(1) で返す"""
        return (self.hash[r] - self.hash[l] * self.power[r - l]) % self.mod

    def is_equal(self, l1, r1, l2, r2):
        """s[l1:r1] == s[l2:r2] を O(1) で判定（確率的）"""
        return self.get_hash(l1, r1) == self.get_hash(l2, r2)

# 使用例: 最長反復部分文字列
def longest_repeated_substring(s):
    """二分探索 + ローリングハッシュで最長反復部分文字列を求める"""
    rh = RollingHash(s)
    n = len(s)

    def has_repeat(length):
        """長さlengthの反復部分文字列が存在するか"""
        seen = set()
        for i in range(n - length + 1):
            h = rh.get_hash(i, i + length)
            if h in seen:
                return i
            seen.add(h)
        return -1

    lo, hi = 0, n - 1
    best = ""

    while lo <= hi:
        mid = (lo + hi) // 2
        pos = has_repeat(mid)
        if pos != -1:
            best = s[pos:pos + mid]
            lo = mid + 1
        else:
            hi = mid - 1

    return best

# 計算量: O(n log n) — 期待値
# 使用例
print(longest_repeated_substring("banana"))  # "ana"
print(longest_repeated_substring("abcabc"))  # "abc"
```

---

## 5. 実務での文字列処理

### 5.1 テキスト処理のパイプライン

```python
# 実務的なテキスト処理パイプラインの例

import re
from collections import Counter

def text_processing_pipeline(text):
    """テキストを前処理して単語頻度を計算"""
    # 1. 正規化
    text = text.lower()

    # 2. 特殊文字の除去
    text = re.sub(r'[^\w\s]', '', text)

    # 3. トークン化
    words = text.split()

    # 4. ストップワードの除去
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'of', 'and', 'or', 'for'}
    words = [w for w in words if w not in stop_words]

    # 5. 頻度カウント
    freq = Counter(words)

    return freq.most_common(10)

# ファジーマッチング（あいまい検索）
def fuzzy_search(query, candidates, max_distance=2):
    """編集距離がmax_distance以内の候補を返す"""
    results = []

    for candidate in candidates:
        dist = edit_distance(query, candidate)
        if dist <= max_distance:
            results.append((candidate, dist))

    return sorted(results, key=lambda x: x[1])

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp

    return dp[n]

# 使用例
words = ["apple", "application", "apply", "maple", "ape",
         "banana", "bandana", "band"]
results = fuzzy_search("aple", words, max_distance=2)
# [('apple', 1), ('ape', 2), ('maple', 2)]
```

### 5.2 全文検索エンジンの仕組み

```python
# 転置インデックス（Inverted Index）の簡易実装

class InvertedIndex:
    """全文検索のための転置インデックス"""

    def __init__(self):
        self.index = {}        # 単語 → [(doc_id, position), ...]
        self.documents = {}    # doc_id → 原文

    def add_document(self, doc_id, text):
        """ドキュメントをインデックスに追加"""
        self.documents[doc_id] = text
        words = text.lower().split()

        for pos, word in enumerate(words):
            # ステミング（簡易版: 末尾のsを除去）
            word = word.strip('.,!?;:')
            if word not in self.index:
                self.index[word] = []
            self.index[word].append((doc_id, pos))

    def search(self, query):
        """単語を含むドキュメントを検索"""
        query = query.lower()
        if query not in self.index:
            return []

        # TF-IDF的なスコアリング（簡易版）
        import math
        total_docs = len(self.documents)
        results = {}

        postings = self.index[query]
        df = len(set(doc_id for doc_id, _ in postings))
        idf = math.log(total_docs / df) if df > 0 else 0

        for doc_id, pos in postings:
            if doc_id not in results:
                results[doc_id] = 0
            results[doc_id] += 1  # TF

        scored = [(doc_id, tf * idf) for doc_id, tf in results.items()]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def phrase_search(self, phrase):
        """フレーズ検索（連続する単語の検索）"""
        words = phrase.lower().split()
        if not words or words[0] not in self.index:
            return []

        # 最初の単語の出現位置
        candidates = {}
        for doc_id, pos in self.index[words[0]]:
            if doc_id not in candidates:
                candidates[doc_id] = []
            candidates[doc_id].append(pos)

        # 後続の単語が連続しているか確認
        for i, word in enumerate(words[1:], 1):
            if word not in self.index:
                return []

            next_positions = {}
            for doc_id, pos in self.index[word]:
                if doc_id in candidates:
                    next_positions[doc_id] = next_positions.get(doc_id, [])
                    next_positions[doc_id].append(pos)

            new_candidates = {}
            for doc_id, starts in candidates.items():
                if doc_id in next_positions:
                    valid = [s for s in starts
                            if s + i in next_positions[doc_id]]
                    if valid:
                        new_candidates[doc_id] = valid

            candidates = new_candidates

        return list(candidates.keys())

# 使用例
idx = InvertedIndex()
idx.add_document(1, "The quick brown fox jumps over the lazy dog")
idx.add_document(2, "A quick brown dog outpaces the fox")
idx.add_document(3, "The fox and the dog are friends")

print(idx.search("fox"))           # [(1, ...), (2, ...), (3, ...)]
print(idx.phrase_search("quick brown"))  # [1, 2]
```

---

## 6. 実践演習

### 演習1: パターンマッチング（基礎）
KMP法の失敗関数を手計算で求めよ（パターン: "ABABC"）。

### 演習2: Trie（応用）
Trieを使ってオートコンプリート機能を実装せよ。入力された接頭辞に対して、辞書内の候補を最大10個返すようにせよ。

### 演習3: Boyer-Moore（応用）
Boyer-Moore法の Bad Character ルールを実装し、ナイーブ法との比較回数の差を計測せよ。

### 演習4: 全文検索（応用）
転置インデックスを実装し、AND/OR検索に対応させよ。TF-IDFによるランキングも実装せよ。

### 演習5: 正規表現エンジン（発展）
NFA方式の簡易正規表現エンジン（., *, | のみサポート）を実装せよ。

### 演習6: 回文検出（発展）
Manacher法を実装し、文字列中の全ての回文部分文字列を列挙せよ。

### 演習7: サフィックス配列（発展）
サフィックス配列 + LCP配列を構築し、テキスト中の最長反復部分文字列を見つけよ。

### 演習8: 文字列ハッシュ（発展）
ローリングハッシュを使って、2つの文字列の最長共通部分文字列を O(n log n) で求めよ。


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

## FAQ

### Q1: grep は内部的に何を使っていますか？
**A**: GNU grep は Boyer-Moore アルゴリズムがベース。長いパターンに強く、パターン末尾から比較してスキップ量を最大化する。正規表現モードでは DFA/NFA を使用。fgrep（固定文字列検索）では Aho-Corasick が使われることもある。

### Q2: 全文検索エンジン（Elasticsearch等）の内部は？
**A**: 転置インデックス + BM25スコアリング。各単語の出現位置をインデックス化し、検索時はインデックスのマージで候補を絞る。日本語はMeCab/kuromoji等で形態素解析してからインデックス構築。N-gramインデックスを併用する場合もある。

### Q3: 文字列処理で最も重要な最適化は？
**A**: (1)文字列連結の回避（joinを使う） (2)正規表現のプリコンパイル(re.compile) (3)不要なコピーの回避（スライスは新しい文字列を生成する） (4)適切なデータ構造の選択（Trie、ハッシュマップ等） (5)Unicode正規化の一貫性

### Q4: KMP法とBoyer-Moore法はどちらが速いですか？
**A**: 一般的にBoyer-Moore法の方が実用上高速。特にアルファベットが大きく（ASCII等）パターンが長い場合に有利。最善ケースでは O(n/m) と、テキスト全体を読まずに済む。一方、KMP法は最悪ケースでも O(n+m) が保証され、実装も比較的単純。バイナリデータや小さいアルファベット（DNA等）ではKMP法が有利な場合もある。

### Q5: Aho-Corasick法はどういう場面で使いますか？
**A**: 複数のパターンを同時に検索する場面。例: ウイルススキャナ（数万のシグネチャを同時検索）、ネットワークIDS、テキストフィルタリング（NGワードリスト）、辞書ベースの形態素解析。単一パターンならKMPやBoyer-Mooreの方が効率的。

### Q6: サフィックス配列とサフィックスツリーの使い分けは？
**A**: 実務的にはほぼサフィックス配列 + LCP配列で十分。サフィックスツリーは理論的には強力だが、メモリ消費が大きく（20n バイト程度）、実装が複雑。サフィックス配列は 5n バイト程度で、キャッシュ効率も良い。ただし、オンラインで文字を追加しながらインデックスを構築する必要がある場合はサフィックスツリー（Ukkonen法）が適している。

---

## まとめ

| アルゴリズム | 計算量 | 用途 |
|------------|--------|------|
| ナイーブ | O(nm) | 短いテキスト |
| KMP | O(n+m) | 単一パターン検索 |
| Boyer-Moore | O(n/m)最善, O(n+m) | 実用最速の単一パターン検索 |
| Rabin-Karp | O(n+m)期待 | 複数パターン |
| Aho-Corasick | O(n+m+z) | 複数パターン同時検索 |
| Z-Algorithm | O(n+m) | パターン検索、周期検出 |
| Trie | O(m) | 辞書、オートコンプリート |
| サフィックス配列 | O(m log n) | 全文検索 |
| Manacher | O(n) | 回文検出 |
| ローリングハッシュ | O(1) per query | 部分文字列の比較 |

---

## 次に読むべきガイド

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 32: String Matching.
2. Knuth, D. E., Morris, J. H., Pratt, V. R. "Fast Pattern Matching in Strings." 1977.
3. Cox, R. "Regular Expression Matching Can Be Simple And Fast." 2007.
4. Gusfield, D. "Algorithms on Strings, Trees, and Sequences." Cambridge University Press, 1997.
5. Aho, A. V., Corasick, M. J. "Efficient string matching: an aid to bibliographic search." 1975.
6. Crochemore, M., Rytter, W. "Jewels of Stringology." World Scientific, 2002.
