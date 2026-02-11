# 連結リスト — 単方向・双方向・循環・フロイドのアルゴリズム

> 配列と対をなす線形データ構造「連結リスト」の各種バリエーションと、サイクル検出の古典的アルゴリズムを学ぶ。

---

## この章で学ぶこと

1. **単方向・双方向・循環リスト** の構造と使い分け
2. **基本操作** の実装（挿入・削除・反転・マージ）
3. **フロイドの循環検出** アルゴリズムとその応用

---

## 1. リストの種類

```
単方向リスト (Singly Linked List):
  head → [A|→] → [B|→] → [C|→] → null

双方向リスト (Doubly Linked List):
  null ← [←|A|→] ⇄ [←|B|→] ⇄ [←|C|→] → null

循環リスト (Circular Linked List):
  head → [A|→] → [B|→] → [C|→] ─┐
         ▲                        │
         └────────────────────────┘
```

---

## 2. 基本実装

### 2.1 ノードクラス

```python
class ListNode:
    """単方向リストのノード"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class DoublyListNode:
    """双方向リストのノード"""
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
```

### 2.2 単方向リストの基本操作

```python
class SinglyLinkedList:
    def __init__(self):
        self.head = None

    def prepend(self, val):
        """先頭に挿入 — O(1)"""
        self.head = ListNode(val, self.head)

    def append(self, val):
        """末尾に挿入 — O(n)"""
        if not self.head:
            self.head = ListNode(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(val)

    def delete(self, val):
        """値で削除 — O(n)"""
        dummy = ListNode(0, self.head)
        prev, curr = dummy, self.head
        while curr:
            if curr.val == val:
                prev.next = curr.next
                break
            prev, curr = curr, curr.next
        self.head = dummy.next

    def search(self, val):
        """値の探索 — O(n)"""
        curr = self.head
        while curr:
            if curr.val == val:
                return True
            curr = curr.next
        return False
```

### 2.3 リストの反転

```python
def reverse_list(head):
    """単方向リストを反転 — O(n) 時間, O(1) 空間"""
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

```
反転の過程:

初期: null ← prev   curr → [1] → [2] → [3] → null

Step1: null ← [1]   curr → [2] → [3] → null
       prev

Step2: null ← [1] ← [2]   curr → [3] → null
                    prev

Step3: null ← [1] ← [2] ← [3]   curr → null
                           prev (= new head)
```

---

## 3. フロイドの循環検出アルゴリズム

### 3.1 亀と兎のアルゴリズム

```python
def has_cycle(head):
    """サイクルの有無を判定 — O(n) 時間, O(1) 空間"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next        # 1歩ずつ
        fast = fast.next.next   # 2歩ずつ
        if slow == fast:
            return True
    return False
```

```
サイクル検出の動作:

  [1] → [2] → [3] → [4] → [5]
                ▲              │
                └──────────────┘

  Step 0: S=1, F=1
  Step 1: S=2, F=3
  Step 2: S=3, F=5
  Step 3: S=4, F=4  ← 一致! サイクルあり

  S: slow (亀)、F: fast (兎)
```

### 3.2 サイクル開始点の特定

```python
def detect_cycle_start(head):
    """サイクルの開始ノードを返す — O(n) 時間, O(1) 空間"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # フェーズ2: 先頭から再出発
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

---

## 4. 応用アルゴリズム

### 例1: ソート済みリストのマージ

```python
def merge_sorted_lists(l1, l2):
    """2つのソート済みリストをマージ — O(n+m)"""
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### 例2: 中間ノードの取得

```python
def find_middle(head):
    """リストの中間ノード — O(n)"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

---

## 5. 比較表

### 表1: リスト種別の比較

| 特性 | 単方向 | 双方向 | 循環 |
|------|--------|--------|------|
| メモリ/ノード | ポインタ1つ | ポインタ2つ | ポインタ1つ |
| 前方走査 | O(1) | O(1) | O(1) |
| 後方走査 | O(n) | O(1) | O(n) |
| 先頭挿入 | O(1) | O(1) | O(1) |
| 末尾挿入 | O(n)* | O(1)** | O(1)** |
| ノード削除 | O(n)*** | O(1) | O(n)*** |
| 使用例 | スタック | LRU キャッシュ | ラウンドロビン |

*tail ポインタなし **tail/head ポインタあり ***前のノード不明の場合

### 表2: 配列 vs 連結リスト

| 操作 | 配列 | 連結リスト |
|------|------|-----------|
| ランダムアクセス | O(1) | O(n) |
| 先頭挿入 | O(n) | O(1) |
| 末尾挿入 | O(1) 償却 | O(n) or O(1) |
| 中間挿入 | O(n) | O(1)* |
| メモリ局所性 | 高い | 低い |
| メモリオーバーヘッド | 低い | ポインタ分 |

*位置が既知の場合

---

## 6. アンチパターン

### アンチパターン1: ダミーヘッドを使わない

```python
# BAD: 先頭ノードを特別扱い → コードが複雑
def delete_bad(head, val):
    if head and head.val == val:
        return head.next
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
            return head
        curr = curr.next
    return head

# GOOD: ダミーヘッドで統一的に処理
def delete_good(head, val):
    dummy = ListNode(0, head)
    prev = dummy
    curr = head
    while curr:
        if curr.val == val:
            prev.next = curr.next
        else:
            prev = curr
        curr = curr.next
    return dummy.next
```

### アンチパターン2: リストの長さを毎回数える

```python
# BAD: 操作のたびに O(n) で長さを数える
class BadList:
    def __init__(self):
        self.head = None

    def length(self):  # O(n) 毎回
        count = 0
        curr = self.head
        while curr:
            count += 1
            curr = curr.next
        return count

# GOOD: 長さをフィールドとして管理
class GoodList:
    def __init__(self):
        self.head = None
        self._size = 0

    def length(self):  # O(1)
        return self._size
```

---

## 7. FAQ

### Q1: 連結リストは実務でどこに使われるか？

**A:** OS のプロセス管理、LRU キャッシュ（双方向リスト + ハッシュマップ）、ブロックチェーン、テキストエディタのギャップバッファ代替、多項式の表現などに使われる。

### Q2: Python には組み込みの連結リストはあるか？

**A:** `collections.deque` が内部的に双方向連結リスト（ブロック単位）で実装されている。純粋な連結リストは標準ライブラリにないが、deque が多くのユースケースをカバーする。

### Q3: フロイドのアルゴリズムはなぜ O(1) 空間でサイクルを検出できるのか？

**A:** 速いポインタが遅いポインタに追いつくとき、サイクル内で相対速度 1 で接近するため、サイクル長以内のステップで必ず出会う。ハッシュセットを使う方法（O(n) 空間）と比べて空間効率が優れている。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| 単方向リスト | 最もシンプル。スタック実装に最適 |
| 双方向リスト | O(1) 削除。LRU キャッシュに使用 |
| 循環リスト | ラウンドロビンや循環バッファに使用 |
| ダミーヘッド | エッジケースを簡潔に処理 |
| フロイドの検出 | O(1) 空間でサイクル検出・開始点特定 |
| slow/fast ポインタ | 中間点取得にも応用可能 |

---

## 次に読むべきガイド

- [スタック/キュー — 連結リストによる実装](./02-stacks-queues.md)
- [グラフ — 隣接リスト表現](./06-graphs.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第10章「Elementary Data Structures」
2. Floyd, R.W. (1967). "Nondeterministic Algorithms." *JACM*, 14(4), 636-644.
3. Skiena, S.S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. — リストの実践的ガイド
