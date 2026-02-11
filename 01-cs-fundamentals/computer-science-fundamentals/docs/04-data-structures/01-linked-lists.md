# 連結リスト

> 連結リストは「ポインタ」の概念を学ぶ最良の教材であり、多くのデータ構造の基盤となる。

## この章で学ぶこと

- [ ] 単方向・双方向連結リストの仕組みを理解する
- [ ] 配列との使い分けを判断できる
- [ ] 連結リストの典型問題を解ける

## 前提知識

- 配列 → 参照: [[00-arrays-and-strings.md]]

---

## 1. 連結リストの構造

### 1.1 単方向連結リスト

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None

    def prepend(self, val):
        """先頭に追加: O(1)"""
        self.head = ListNode(val, self.head)

    def append(self, val):
        """末尾に追加: O(n)"""
        if not self.head:
            self.head = ListNode(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = ListNode(val)

    def delete(self, val):
        """値の削除: O(n)"""
        dummy = ListNode(0, self.head)
        curr = dummy
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                break
            curr = curr.next
        self.head = dummy.next

# メモリレイアウト:
# head → [val|next] → [val|next] → [val|next] → None
# 各ノードがメモリ上の別の場所に存在
# → キャッシュ効率が悪い（配列と逆）
```

### 1.2 操作の計算量

```
連結リスト vs 配列:

  ┌──────────────────┬────────────┬────────────┐
  │ 操作             │ 連結リスト  │ 配列       │
  ├──────────────────┼────────────┼────────────┤
  │ 先頭に挿入       │ O(1) ✅   │ O(n)       │
  │ 末尾に挿入       │ O(n)*     │ O(1) 償却  │
  │ 中間に挿入       │ O(1)**    │ O(n)       │
  │ インデックスアクセス│ O(n)     │ O(1) ✅   │
  │ 検索             │ O(n)      │ O(n)/O(logn)│
  │ メモリ効率       │ 悪い       │ 良い ✅    │
  │ キャッシュ効率   │ 悪い       │ 良い ✅    │
  └──────────────────┴────────────┴────────────┘

  * 末尾ポインタを持てばO(1)
  ** 挿入位置のポインタが既知の場合
```

### 1.3 典型テクニック

```python
# 1. 連結リストの反転
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev

# 2. サイクル検出（Floyd's Cycle Detection）
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# 3. 中間ノードの検出
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

---

## 2. 実務での連結リスト

```
連結リストが実際に使われる場面:

  1. OS: プロセスリスト、メモリのフリーリスト
  2. LRUキャッシュ: 双方向連結リスト + ハッシュマップ
  3. テキストエディタ: 行のリスト（挿入/削除が頻繁）
  4. ブロックチェーン: 各ブロックが前のハッシュを保持
  5. 関数型言語: 不変リスト（Haskellのリスト、Clojureのcons）

  実務でのアドバイス:
  → ほとんどの場合、配列/動的配列で十分
  → 連結リストを選ぶのは特殊な理由がある場合のみ
```

---

## 3. 実践演習

### 演習1: 基本操作（基礎）
単方向連結リストのprepend, append, delete, searchを実装せよ。

### 演習2: マージ（応用）
2つのソート済み連結リストを1つのソート済みリストにマージせよ。

### 演習3: LRUキャッシュ（発展）
双方向連結リスト + ハッシュマップで O(1) のLRUキャッシュを実装せよ。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 単方向リスト | 先頭挿入O(1)。ランダムアクセス不可 |
| 双方向リスト | 両端操作O(1)。LRUキャッシュの基盤 |
| vs 配列 | キャッシュ効率で配列が圧勝。特殊な場合のみリスト |
| テクニック | 反転、サイクル検出、Fast/Slow ポインタ |

---

## 次に読むべきガイド
→ [[02-stacks-and-queues.md]] — スタックとキュー

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapter 10.2.
2. Sedgewick, R. "Algorithms." Chapter 1.3.
