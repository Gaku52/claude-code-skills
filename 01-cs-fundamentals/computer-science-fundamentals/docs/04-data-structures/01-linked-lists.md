# 連結リスト（Linked List）

> 連結リストは「ポインタ」の概念を学ぶ最良の教材であり、多くのデータ構造の基盤となる。
> 配列とは対照的なメモリモデルを理解することで、データ構造設計の本質が見えてくる。

## この章で学ぶこと

- [ ] 単方向・双方向・循環連結リストの仕組みを理解する
- [ ] 各種操作の計算量を正確に把握する
- [ ] 配列との使い分けを判断できる
- [ ] 連結リストの典型問題（反転、サイクル検出、マージ等）を解ける
- [ ] LRUキャッシュなど実践的なデータ構造を実装できる

## 前提知識

- 配列とその計算量 → 参照: [[00-arrays-and-strings.md]]
- Python の基本的なクラス定義（`__init__`, `self`）
- ポインタ／参照の概念（C言語の経験があると理解が深まるが必須ではない）

---

## 1. なぜ連結リストが必要か — 配列の限界

### 1.1 配列が抱える構造的制約

配列（array）は最も基本的なデータ構造であり、多くの場面で優れた性能を発揮する。しかし、配列には本質的な制約がいくつか存在する。

**制約1: 連続メモリの必要性**

配列は要素をメモリ上に連続して配置する。これはランダムアクセスを O(1) で実現するために不可欠な性質だが、大きな配列を確保するには同じサイズの連続した空き領域が必要になる。メモリが断片化（フラグメンテーション）している環境では、十分な合計空き容量があっても配列を確保できない場合がある。

```
配列のメモリレイアウト（連続領域が必須）:

アドレス: 0x100  0x104  0x108  0x10C  0x110  0x114
         ┌──────┬──────┬──────┬──────┬──────┬──────┐
         │  10  │  20  │  30  │  40  │  50  │  60  │
         └──────┴──────┴──────┴──────┴──────┴──────┘
         ←────────── 連続した24バイト必要 ──────────→

断片化されたメモリでは確保に失敗する可能性:

         ┌──────┐      ┌──────┐      ┌──────┐
  空き:  │ 8B   │ 使用 │ 12B  │ 使用 │ 8B   │ ...
         └──────┘      └──────┘      └──────┘
         合計28Bの空きがあるが、連続24Bは取れない
```

**制約2: 挿入と削除のコスト**

配列の中間に要素を挿入したり、中間の要素を削除したりするには、後続の全要素をシフトする必要がある。要素数を n とすると、最悪ケースで O(n) の時間がかかる。

```
配列の中間挿入（インデックス2に「25」を挿入する場合）:

Before: [10, 20, 30, 40, 50]
                ↑ ここに25を入れたい

Step 1: 後ろの要素を1つずつ右にシフト
        [10, 20, 30, 30, 40, 50]  ← 50を右へ
        [10, 20, 30, 30, 40, 50]  ← 40を右へ
        [10, 20, 30, 30, 40, 50]  ← 30を右へ（空きができる）

Step 2: 空いた位置に挿入
        [10, 20, 25, 30, 40, 50]

→ n=5 の配列に対して最大 n-1 回のコピーが発生
```

**制約3: サイズ変更のオーバーヘッド**

静的配列はサイズが固定されている。動的配列（Python の `list`、Java の `ArrayList`）はサイズを自動拡張するが、内部的には新しいメモリ領域を確保してデータ全体をコピーする処理が発生する。償却計算量は O(1) だが、リサイズが発生するタイミングでは O(n) の遅延が生じる。リアルタイム性が要求されるシステムでは、この一時的な遅延が問題になることがある。

### 1.2 連結リストという解決策

連結リストは、これらの制約を根本的に異なるアプローチで解決する。

```
連結リストのメモリレイアウト（非連続でよい）:

アドレス 0x200        アドレス 0x350        アドレス 0x120
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ val: 10     │      │ val: 20     │      │ val: 30     │
│ next: 0x350 │─────→│ next: 0x120 │─────→│ next: None  │
└─────────────┘      └─────────────┘      └─────────────┘
メモリ上でバラバラに配置されていても問題ない
```

各要素（ノード）は「データ」と「次のノードへの参照（ポインタ）」を保持する。ノード間はポインタで接続されるため、メモリ上で連続している必要がない。挿入や削除はポインタの付け替えだけで完了するため、要素のシフトは不要である。

### 1.3 連結リストの本質的な強みと弱み

連結リストは万能ではない。配列と相補的な性質を持つデータ構造であり、それぞれ得意な場面が異なる。

**強み:**
- 先頭への挿入・削除が O(1)
- 任意の位置（ポインタが既知の場合）への挿入・削除が O(1)
- サイズの動的変更にリサイズコピーが不要
- 連続メモリ領域が不要

**弱み:**
- ランダムアクセスが O(n)（インデックスによる直接アクセス不可）
- 各ノードにポインタ分の追加メモリが必要
- キャッシュ局所性が低い（CPU キャッシュの恩恵を受けにくい）
- 参照の間接化によるオーバーヘッド

この章では、連結リストの種類（単方向、双方向、循環）ごとに構造と操作を詳細に学び、配列との適切な使い分けができるようになることを目指す。

---

## 2. 単方向連結リスト（Singly Linked List）

### 2.1 基本構造

単方向連結リストは、連結リストの最もシンプルな形態である。各ノードは「データ」と「次のノードへのポインタ」の2つのフィールドを持つ。

```
単方向連結リストの構造:

  head
   │
   ▼
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│ val: "A"  │    │ val: "B"  │    │ val: "C"  │    │ val: "D"  │
│ next: ─────┼──→│ next: ─────┼──→│ next: ─────┼──→│ next: None│
└───────────┘    └───────────┘    └───────────┘    └───────────┘

特徴:
- 各ノードは next ポインタのみ保持（一方向にしか辿れない）
- リストの先頭を指す head ポインタが入口
- 最後のノードの next は None（NULL）
- 逆方向への走査は不可能
```

**ノードクラスの定義:**

```python
class ListNode:
    """単方向連結リストのノード"""
    __slots__ = ('val', 'next')  # メモリ効率のためスロットを使用

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"
```

`__slots__` を指定することで、Python のデフォルトの `__dict__` による属性管理を抑制し、ノード1個あたりのメモリ使用量を削減できる。大量のノードを生成する連結リストでは、この最適化が効果的である。

### 2.2 基本操作

#### 2.2.1 先頭への挿入（prepend）

```
先頭への挿入操作（O(1)):

Before:
  head
   │
   ▼
  [B] → [C] → [D] → None

「A」を先頭に挿入:

Step 1: 新しいノードを作成し、next を現在の head に設定
  new_node        head
     │              │
     ▼              ▼
    [A] ─────────→ [B] → [C] → [D] → None

Step 2: head を新しいノードに更新
  head
   │
   ▼
  [A] → [B] → [C] → [D] → None

→ ポインタの付け替え2回のみ。リスト長に依存しない O(1)
```

#### 2.2.2 末尾への挿入（append）

```
末尾への挿入操作（O(n)):

Before:
  head
   │
   ▼
  [A] → [B] → [C] → None

「D」を末尾に挿入:

Step 1: 末尾ノードまで走査（O(n)）
  head
   │
   ▼
  [A] → [B] → [C] → None
                ↑
              curr（末尾ノードを発見）

Step 2: 末尾ノードの next を新しいノードに設定
  head
   │
   ▼
  [A] → [B] → [C] → [D] → None

→ 末尾までの走査に O(n) が必要
→ tail ポインタを別途保持すれば O(1) に改善可能
```

#### 2.2.3 指定値の削除（delete）

```
中間ノードの削除操作（O(n)):

Before:
  head
   │
   ▼
  [A] → [B] → [C] → [D] → None

「C」を削除:

Step 1: 削除対象の直前のノードまで走査
  head
   │
   ▼
  [A] → [B] → [C] → [D] → None
          ↑     ↑
         prev  target

Step 2: prev.next を target.next に付け替え
  head
   │
   ▼                ┌───────────┐
  [A] → [B] ────────┤→ [D] → None
                    │
          [C] ──────┘  ← 参照が切れてGC対象に

→ 走査に O(n)、ポインタの付け替え自体は O(1)
```

#### 2.2.4 ダミーヘッド（番兵ノード）テクニック

先頭ノードの削除はエッジケースになりやすい。ダミーヘッド（番兵ノード）を導入すると、すべてのノードに「前のノード」が存在するため、場合分けが不要になる。

```
ダミーヘッドを使う場合:

  dummy    head
   │        │
   ▼        ▼
  [0] ──→ [A] → [B] → [C] → None
   ↑
 値は使わない（番兵）

「A」（先頭）を削除する場合も特別扱い不要:

  dummy
   │
   ▼           ┌──────────┐
  [0] ─────────┤→ [B] → [C] → None
               │
     [A] ──────┘

最後に dummy.next を返せばよい
```

### 2.3 完全実装

```python
class ListNode:
    """単方向連結リストのノード"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList:
    """単方向連結リストの完全実装"""

    def __init__(self):
        self.head = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """イテレータプロトコルの実装"""
        curr = self.head
        while curr:
            yield curr.val
            curr = curr.next

    def __repr__(self):
        values = []
        curr = self.head
        count = 0
        while curr and count < 20:  # 無限ループ防止
            values.append(str(curr.val))
            curr = curr.next
            count += 1
        chain = " -> ".join(values)
        if curr:
            chain += " -> ..."
        return f"SinglyLinkedList([{chain}])"

    def is_empty(self):
        """空リスト判定: O(1)"""
        return self.head is None

    def prepend(self, val):
        """先頭に挿入: O(1)"""
        self.head = ListNode(val, self.head)
        self._size += 1

    def append(self, val):
        """末尾に挿入: O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self._size += 1

    def insert_at(self, index, val):
        """指定インデックスに挿入: O(n)

        Args:
            index: 挿入位置（0-based）。0 なら先頭、size なら末尾。
            val: 挿入する値

        Raises:
            IndexError: インデックスが範囲外の場合
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")
        if index == 0:
            self.prepend(val)
            return
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        curr.next = ListNode(val, curr.next)
        self._size += 1

    def delete(self, val):
        """指定値の最初の出現を削除: O(n)

        Args:
            val: 削除する値

        Returns:
            bool: 削除に成功した場合 True、値が見つからなかった場合 False
        """
        # ダミーヘッドテクニックで先頭削除の特殊ケースを回避
        dummy = ListNode(0, self.head)
        curr = dummy
        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
                self.head = dummy.next
                self._size -= 1
                return True
            curr = curr.next
        return False

    def delete_at(self, index):
        """指定インデックスの要素を削除: O(n)

        Args:
            index: 削除位置（0-based）

        Returns:
            削除された値

        Raises:
            IndexError: インデックスが範囲外の場合
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")
        if index == 0:
            val = self.head.val
            self.head = self.head.next
            self._size -= 1
            return val
        curr = self.head
        for _ in range(index - 1):
            curr = curr.next
        val = curr.next.val
        curr.next = curr.next.next
        self._size -= 1
        return val

    def search(self, val):
        """値の検索: O(n)

        Returns:
            int: 値が見つかった場合そのインデックス、見つからない場合 -1
        """
        curr = self.head
        index = 0
        while curr:
            if curr.val == val:
                return index
            curr = curr.next
            index += 1
        return -1

    def get(self, index):
        """インデックスでアクセス: O(n)

        Args:
            index: 取得位置（0-based）

        Returns:
            指定位置の値

        Raises:
            IndexError: インデックスが範囲外の場合
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")
        curr = self.head
        for _ in range(index):
            curr = curr.next
        return curr.val

    def reverse(self):
        """リストを反転（in-place）: O(n)"""
        prev = None
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        self.head = prev

    def to_list(self):
        """Python リストに変換: O(n)"""
        return list(self)

    @classmethod
    def from_list(cls, values):
        """Python リストから連結リストを構築: O(n)

        Args:
            values: イテラブルな値の列

        Returns:
            SinglyLinkedList: 構築された連結リスト
        """
        ll = cls()
        for val in reversed(values):
            ll.prepend(val)
        return ll


# --- 動作確認 ---
if __name__ == "__main__":
    # 構築
    ll = SinglyLinkedList.from_list([1, 2, 3, 4, 5])
    print(ll)                   # SinglyLinkedList([1 -> 2 -> 3 -> 4 -> 5])
    print(f"Length: {len(ll)}")  # Length: 5

    # 挿入
    ll.prepend(0)
    ll.append(6)
    ll.insert_at(3, 99)
    print(ll)  # SinglyLinkedList([0 -> 1 -> 2 -> 99 -> 3 -> 4 -> 5 -> 6])

    # 検索
    print(f"Search 99: index {ll.search(99)}")  # Search 99: index 3
    print(f"Get index 3: {ll.get(3)}")          # Get index 3: 99

    # 削除
    ll.delete(99)
    deleted = ll.delete_at(0)
    print(f"Deleted: {deleted}")  # Deleted: 0
    print(ll)                     # SinglyLinkedList([1 -> 2 -> 3 -> 4 -> 5 -> 6])

    # 反転
    ll.reverse()
    print(ll)  # SinglyLinkedList([6 -> 5 -> 4 -> 3 -> 2 -> 1])

    # イテレーション
    for val in ll:
        print(val, end=" ")  # 6 5 4 3 2 1
    print()
```

### 2.4 計算量まとめ

| 操作 | 時間計算量 | 空間計算量 | 備考 |
|------|-----------|-----------|------|
| `prepend` | O(1) | O(1) | 先頭挿入。最も効率的な操作 |
| `append` | O(n) | O(1) | 末尾走査が必要。tail 保持で O(1) に改善可 |
| `insert_at(i)` | O(i) | O(1) | 挿入位置まで走査。先頭なら O(1) |
| `delete(val)` | O(n) | O(1) | 値の検索に O(n) |
| `delete_at(i)` | O(i) | O(1) | 位置までの走査に O(i) |
| `search` | O(n) | O(1) | 線形探索 |
| `get(i)` | O(i) | O(1) | ランダムアクセス不可 |
| `reverse` | O(n) | O(1) | in-place で反転 |
| `from_list` | O(n) | O(n) | リストからの構築 |

### 2.5 重要なテクニック

#### 2.5.1 リストの反転

連結リストの反転は、面接問題として最も頻出するテーマの一つである。反復版と再帰版の両方を理解しておくことが重要である。

```python
def reverse_iterative(head):
    """連結リストの反転（反復版）: O(n) 時間, O(1) 空間

    3つのポインタ（prev, curr, next_node）を使って
    各ノードの next を前のノードに付け替える。
    """
    prev = None
    curr = head
    while curr:
        next_node = curr.next   # 次のノードを退避
        curr.next = prev        # ポインタを反転
        prev = curr             # prev を進める
        curr = next_node        # curr を進める
    return prev  # 新しい先頭


def reverse_recursive(head):
    """連結リストの反転（再帰版）: O(n) 時間, O(n) 空間

    再帰でリストの末尾に到達し、戻りながらポインタを反転する。
    コールスタックに O(n) の空間を使う点に注意。
    """
    # ベースケース: 空リストまたは1ノード
    if not head or not head.next:
        return head
    # 再帰: head.next 以降を反転
    new_head = reverse_recursive(head.next)
    # head.next のノードが head を指すようにする
    head.next.next = head
    head.next = None
    return new_head
```

```
反復版の動作を追跡:

初期状態: 1 -> 2 -> 3 -> None
         prev=None, curr=1

Step 1: next_node=2, 1.next=None, prev=1, curr=2
        None <- 1    2 -> 3 -> None

Step 2: next_node=3, 2.next=1, prev=2, curr=3
        None <- 1 <- 2    3 -> None

Step 3: next_node=None, 3.next=2, prev=3, curr=None
        None <- 1 <- 2 <- 3

結果: 3 -> 2 -> 1 -> None （prev が新しい head）
```

#### 2.5.2 Floyd のサイクル検出（亀とウサギ）

連結リストにサイクル（循環）が存在するかどうかを検出するアルゴリズム。2つのポインタを異なる速度で進めることで、O(1) の追加空間でサイクルを検出する。

```python
def has_cycle(head):
    """サイクルの存在判定: O(n) 時間, O(1) 空間

    slow（亀）は1ステップずつ、fast（ウサギ）は2ステップずつ進む。
    サイクルがあれば必ず出会い、なければ fast が末尾に到達する。
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


def detect_cycle_start(head):
    """サイクルの開始ノードを特定: O(n) 時間, O(1) 空間

    Phase 1: slow と fast が出会う地点を見つける
    Phase 2: head と出会い地点から同速度で進め、合流点がサイクル開始
    """
    slow = fast = head

    # Phase 1: 出会い地点を見つける
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # サイクルなし

    # Phase 2: サイクル開始ノードを特定
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow
```

```
Floyd のサイクル検出 — なぜ正しく動作するか:

サイクルありの場合:
  head → [1] → [2] → [3] → [4] → [5]
                       ↑              │
                       └──────────────┘

  slow: 1, 2, 3, 4, 5, 3, 4, ...
  fast: 1, 3, 5, 4, 3, 5, 4, ...

  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=5
  Step 3: slow=4, fast=4  ← 出会った！（サイクルあり）

サイクルなしの場合:
  head → [1] → [2] → [3] → [4] → None

  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=None ← fast が None に到達（サイクルなし）

サイクル開始ノード特定の数学的証明:
  F = head からサイクル開始までの距離
  C = サイクルの長さ
  出会い地点はサイクル開始から a ノード先とする

  slow の移動距離: F + a
  fast の移動距離: F + a + kC（k はサイクルを回った回数）

  fast は slow の2倍進むので:
  2(F + a) = F + a + kC
  F + a = kC
  F = kC - a

  つまり head から F ステップ進むと、出会い地点から F ステップ進んだ点
  （= kC - a ステップ = サイクル開始地点）に到達する。
```

#### 2.5.3 Fast/Slow ポインタ（中間ノードの検出）

```python
def find_middle(head):
    """中間ノードを検出: O(n) 時間, O(1) 空間

    slow は1歩、fast は2歩ずつ進む。
    fast がリストの末尾に到達したとき、slow は中間にいる。
    偶数長の場合、2つの中間ノードのうち後者を返す。
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def find_middle_first(head):
    """中間ノードを検出（偶数長の場合は前者を返す）

    偶数長のリスト [1, 2, 3, 4] に対して:
    - find_middle は 3 を返す（後者の中間）
    - find_middle_first は 2 を返す（前者の中間）
    """
    slow = fast = head
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

```
中間ノード検出の動作例:

奇数長: 1 -> 2 -> 3 -> 4 -> 5
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=5  ← fast.next=None で終了
  結果: slow=3 （正確な中間）

偶数長: 1 -> 2 -> 3 -> 4
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3
  Step 2: slow=3, fast=None ← fast が None で終了
  結果: slow=3 （後者の中間）

  find_middle_first の場合:
  Step 0: slow=1, fast=1
  Step 1: slow=2, fast=3  ← fast.next.next=None で終了
  結果: slow=2 （前者の中間）
```

---

## 3. 双方向連結リスト（Doubly Linked List）

### 3.1 基本構造

双方向連結リストでは、各ノードが「前のノードへのポインタ（prev）」と「次のノードへのポインタ（next）」の両方を持つ。これにより、任意のノードから前後両方向に走査できる。

```
双方向連結リストの構造:

  head                                                    tail
   │                                                       │
   ▼                                                       ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ prev:None│    │ prev: ←──┼────│ prev: ←──┼────│ prev: ←──┤
│ val: "A" │    │ val: "B" │    │ val: "C" │    │ val: "D" │
│ next: ───┼──→ │ next: ───┼──→ │ next: ───┼──→ │ next:None│
└──────────┘    └──────────┘    └──────────┘    └──────────┘

  None ← [A] ⇌ [B] ⇌ [C] ⇌ [D] → None

特徴:
- 各ノードは prev と next の2つのポインタを保持
- head と tail の両端からアクセス可能
- 双方向に走査できる
- ポインタが2つあるため、単方向リストよりメモリを多く使う
```

### 3.2 単方向リストとの比較

| 特性 | 単方向リスト | 双方向リスト |
|------|------------|------------|
| ノードあたりのポインタ数 | 1（next のみ） | 2（prev + next） |
| 逆方向走査 | 不可（O(n) で反転が必要） | O(1) で prev を辿る |
| 末尾からの削除 | O(n)（直前ノードの探索が必要） | O(1)（tail.prev で直前ノードにアクセス） |
| 任意ノードの削除 | O(n)（直前ノードの探索が必要） | O(1)（ノード参照があれば prev で直前にアクセス） |
| メモリ使用量 | 少ない | ポインタ1本分多い |
| 実装の複雑さ | シンプル | prev の管理で複雑化 |
| 主な用途 | スタック、シンプルなキュー | LRU キャッシュ、テキストエディタ、ブラウザ履歴 |

双方向リストの最大の利点は「任意のノードを O(1) で削除できる」ことである。単方向リストではノードを削除するために直前のノードを知る必要があり、それには先頭からの走査（O(n)）が不可避である。双方向リストでは `node.prev` で直前のノードに即座にアクセスできる。

### 3.3 完全実装

```python
class DListNode:
    """双方向連結リストのノード"""
    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DListNode({self.val})"


class DoublyLinkedList:
    """双方向連結リストの完全実装

    番兵ノード（sentinel）を使用して実装を簡潔にする。
    head_sentinel と tail_sentinel は常に存在し、実データは持たない。

    構造:
    head_sentinel ⇌ [data1] ⇌ [data2] ⇌ ... ⇌ tail_sentinel
    """

    def __init__(self):
        # 番兵ノードの初期化
        self._head = DListNode(0)  # head sentinel
        self._tail = DListNode(0)  # tail sentinel
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        """前方向イテレータ"""
        curr = self._head.next
        while curr != self._tail:
            yield curr.val
            curr = curr.next

    def __reversed__(self):
        """逆方向イテレータ"""
        curr = self._tail.prev
        while curr != self._head:
            yield curr.val
            curr = curr.prev

    def __repr__(self):
        values = list(self)
        return f"DoublyLinkedList({values})"

    def is_empty(self):
        """空リスト判定: O(1)"""
        return self._size == 0

    def _insert_between(self, val, predecessor, successor):
        """2つのノードの間に新しいノードを挿入: O(1)

        内部ヘルパーメソッド。全ての挿入操作はこれに帰着する。
        """
        new_node = DListNode(val, predecessor, successor)
        predecessor.next = new_node
        successor.prev = new_node
        self._size += 1
        return new_node

    def _remove_node(self, node):
        """指定ノードを削除: O(1)

        内部ヘルパーメソッド。全ての削除操作はこれに帰着する。
        番兵ノードは削除しない。
        """
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self._size -= 1
        node.prev = node.next = None  # 参照をクリア
        return node.val

    def prepend(self, val):
        """先頭に挿入: O(1)"""
        return self._insert_between(val, self._head, self._head.next)

    def append(self, val):
        """末尾に挿入: O(1)

        双方向リストでは tail sentinel の prev に挿入するだけ。
        単方向リストの O(n) と比較して大きな改善。
        """
        return self._insert_between(val, self._tail.prev, self._tail)

    def insert_at(self, index, val):
        """指定インデックスに挿入: O(min(i, n-i))

        先頭と末尾のどちらに近いかに応じて走査方向を選択。
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")

        # 先頭に近いか末尾に近いかで走査方向を決定
        if index <= self._size // 2:
            curr = self._head.next
            for _ in range(index):
                curr = curr.next
            self._insert_between(val, curr.prev, curr)
        else:
            curr = self._tail.prev
            for _ in range(self._size - index - 1):
                curr = curr.prev
            self._insert_between(val, curr, curr.next)

    def pop_front(self):
        """先頭を削除して値を返す: O(1)

        Raises:
            IndexError: リストが空の場合
        """
        if self.is_empty():
            raise IndexError("pop from empty list")
        return self._remove_node(self._head.next)

    def pop_back(self):
        """末尾を削除して値を返す: O(1)

        Raises:
            IndexError: リストが空の場合
        """
        if self.is_empty():
            raise IndexError("pop from empty list")
        return self._remove_node(self._tail.prev)

    def delete(self, val):
        """指定値の最初の出現を削除: O(n)

        Returns:
            bool: 削除に成功した場合 True
        """
        curr = self._head.next
        while curr != self._tail:
            if curr.val == val:
                self._remove_node(curr)
                return True
            curr = curr.next
        return False

    def get(self, index):
        """インデックスでアクセス: O(min(i, n-i))

        双方向の走査を活用し、先頭と末尾のどちらに近いかで方向を選択。
        単方向リストの O(i) に対し、最悪ケースが O(n/2) に改善。
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        if index <= self._size // 2:
            curr = self._head.next
            for _ in range(index):
                curr = curr.next
        else:
            curr = self._tail.prev
            for _ in range(self._size - 1 - index):
                curr = curr.prev
        return curr.val

    def front(self):
        """先頭の値を参照（削除しない）: O(1)"""
        if self.is_empty():
            raise IndexError("front from empty list")
        return self._head.next.val

    def back(self):
        """末尾の値を参照（削除しない）: O(1)"""
        if self.is_empty():
            raise IndexError("back from empty list")
        return self._tail.prev.val

    def reverse(self):
        """リストを反転（in-place）: O(n)

        全ノードの prev と next を入れ替える。
        """
        curr = self._head
        while curr:
            curr.prev, curr.next = curr.next, curr.prev
            curr = curr.prev  # prev と next が入れ替わったので prev 方向に進む
        # 番兵ノードを入れ替え
        self._head, self._tail = self._tail, self._head

    def to_list(self):
        """Python リストに変換: O(n)"""
        return list(self)

    @classmethod
    def from_list(cls, values):
        """Python リストから構築: O(n)"""
        dll = cls()
        for val in values:
            dll.append(val)
        return dll


# --- 動作確認 ---
if __name__ == "__main__":
    dll = DoublyLinkedList.from_list([1, 2, 3, 4, 5])
    print(dll)                     # DoublyLinkedList([1, 2, 3, 4, 5])
    print(f"Length: {len(dll)}")    # Length: 5

    # 両端操作
    dll.prepend(0)
    dll.append(6)
    print(dll)                     # DoublyLinkedList([0, 1, 2, 3, 4, 5, 6])
    print(f"Front: {dll.front()}")  # Front: 0
    print(f"Back: {dll.back()}")    # Back: 6

    # 両端からの削除
    dll.pop_front()
    dll.pop_back()
    print(dll)  # DoublyLinkedList([1, 2, 3, 4, 5])

    # 逆方向走査
    print("Reversed:", list(reversed(dll)))  # Reversed: [5, 4, 3, 2, 1]

    # 反転
    dll.reverse()
    print(dll)  # DoublyLinkedList([5, 4, 3, 2, 1])
```

### 3.4 番兵ノードの設計意図

番兵ノード（sentinel node）を使う設計には明確な利点がある。

```
番兵なしの場合:

  head                    tail
   │                       │
   ▼                       ▼
  [A] ⇌ [B] ⇌ [C] ⇌ [D]

先頭に挿入する場合:
  if self.head is None:       # 空リストの場合
      self.head = self.tail = new_node
  else:                        # 非空リストの場合
      new_node.next = self.head
      self.head.prev = new_node
      self.head = new_node

番兵ありの場合:

  head_sentinel              tail_sentinel
       │                          │
       ▼                          ▼
      [S] ⇌ [A] ⇌ [B] ⇌ [C] ⇌ [S]

先頭に挿入する場合（常に同じコード）:
  self._insert_between(val, self._head, self._head.next)

→ 空リストの場合分けが不要
→ バグの温床となるエッジケースが消える
→ 代償: ノード2個分のメモリオーバーヘッド（通常は無視できる）
```

### 3.5 計算量まとめ

| 操作 | 時間計算量 | 単方向リストとの比較 |
|------|-----------|-------------------|
| `prepend` | O(1) | 同じ |
| `append` | O(1) | 単方向: O(n) → 大幅改善 |
| `insert_at(i)` | O(min(i, n-i)) | 単方向: O(i) → 改善 |
| `pop_front` | O(1) | 同じ |
| `pop_back` | O(1) | 単方向: O(n) → 大幅改善 |
| `delete(node)` | O(1) | 単方向: O(n) → 大幅改善 |
| `get(i)` | O(min(i, n-i)) | 単方向: O(i) → 改善 |
| `reverse` | O(n) | 同じ |

---

## 4. 循環連結リスト（Circular Linked List）

### 4.1 基本構造

循環連結リストでは、最後のノードの `next` が先頭ノードを指す。これにより、リストの末尾から先頭への遷移がシームレスに行える。単方向循環リストと双方向循環リストの両方が存在する。

```
単方向循環連結リスト:

  head
   │
   ▼
  [A] → [B] → [C] → [D] ─┐
   ↑                       │
   └───────────────────────┘

  最後のノード D の next が先頭の A を指す
  → None が存在しない
  → どのノードからでもリスト全体を走査可能

双方向循環連結リスト:

  head
   │
   ▼
  [A] ⇌ [B] ⇌ [C] ⇌ [D]
   ↑                    │
   │    ←────────────── │ (prev)
   └──────────────────→   (next)

  A.prev = D, D.next = A
  → 先頭と末尾の区別が曖昧になる
  → リングバッファ的な用途に適する
```

### 4.2 循環リストの用途

循環リストが特に有効な場面を示す。

**1. ラウンドロビンスケジューリング**
OS のプロセススケジューラでは、各プロセスに均等に CPU 時間を割り当てるラウンドロビン方式がよく使われる。循環リストを使えば、最後のプロセスの次に自然に最初のプロセスに戻る。

**2. 循環バッファ（リングバッファ）**
固定サイズのバッファで、末尾に達したら先頭に戻って上書きする。ストリーミングデータの処理やログの保持に使われる。

**3. マルチプレイヤーゲーム**
複数プレイヤーが順番にアクションを行うターン制ゲームで、最後のプレイヤーの次に最初のプレイヤーに戻る。

**4. ジョセフス問題**
n 人が円形に並び、m 番目の人を順に除外していく古典的な問題。循環リストで自然にモデル化できる。

### 4.3 完全実装

```python
class CircularLinkedList:
    """単方向循環連結リスト

    tail ポインタのみを保持する設計。
    head は tail.next で O(1) アクセス可能。

    空リスト: tail = None
    1ノード:  tail → [A] → (自分自身に戻る)
    複数:     tail → [C] → [A] → [B] → [C]（tail に戻る）

    tail を保持する理由:
    - head は tail.next で取得可能
    - 末尾への挿入が O(1) で可能（tail の直後に挿入し tail を更新）
    - head のみ保持する場合、末尾挿入に O(n) の走査が必要
    """

    def __init__(self):
        self.tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        if not self.tail:
            return
        curr = self.tail.next  # head から開始
        for _ in range(self._size):
            yield curr.val
            curr = curr.next

    def __repr__(self):
        if not self.tail:
            return "CircularLinkedList([])"
        values = list(self)
        return f"CircularLinkedList({values}, circular)"

    def is_empty(self):
        """空リスト判定: O(1)"""
        return self.tail is None

    def prepend(self, val):
        """先頭に挿入: O(1)"""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node  # 自分自身を指す
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # 新ノード → 旧 head
            self.tail.next = new_node       # tail → 新ノード（新 head）
        self._size += 1

    def append(self, val):
        """末尾に挿入: O(1)

        tail の直後に挿入して tail を更新する。
        """
        self.prepend(val)       # 先頭に挿入
        self.tail = self.tail.next  # tail を新ノードに移動 → 実質的に末尾挿入
        self._size += 0  # prepend で既にインクリメント済み

    def pop_front(self):
        """先頭を削除して値を返す: O(1)

        Raises:
            IndexError: リストが空の場合
        """
        if self.is_empty():
            raise IndexError("pop from empty circular list")
        head = self.tail.next
        val = head.val
        if self.tail == head:
            # ノードが1つだけ
            self.tail = None
        else:
            self.tail.next = head.next
        self._size -= 1
        return val

    def rotate(self, k=1):
        """リストを k 回転（先頭が末尾に移動）: O(k)

        ラウンドロビンスケジューリングの次のプロセスへの移動に相当。
        """
        if self.is_empty() or k == 0:
            return
        for _ in range(k % self._size):
            self.tail = self.tail.next

    def search(self, val):
        """値の検索: O(n)

        Returns:
            bool: 値が見つかった場合 True
        """
        if self.is_empty():
            return False
        curr = self.tail.next
        for _ in range(self._size):
            if curr.val == val:
                return True
            curr = curr.next
        return False

    def josephus(self, step):
        """ジョセフス問題を解く

        Args:
            step: 除外する間隔（m 番目を除外）

        Returns:
            list: 除外された順序のリスト
        """
        if self.is_empty():
            return []

        elimination_order = []
        curr = self.tail
        remaining = self._size

        while remaining > 0:
            # step - 1 回進む（step 番目のノードの直前に移動）
            for _ in range(step - 1):
                curr = curr.next
            # curr.next を除外
            eliminated = curr.next
            elimination_order.append(eliminated.val)
            if curr == eliminated:
                # 最後の1人
                curr = None
            else:
                curr.next = eliminated.next
            remaining -= 1

        self.tail = None
        self._size = 0
        return elimination_order


# --- 動作確認 ---
if __name__ == "__main__":
    # 基本操作
    cl = CircularLinkedList()
    for i in range(1, 6):
        cl.append(i)
    print(cl)  # CircularLinkedList([1, 2, 3, 4, 5], circular)

    # 回転
    cl.rotate(2)
    print(cl)  # CircularLinkedList([3, 4, 5, 1, 2], circular)

    # ジョセフス問題: 7人が円形に並び、3番目ごとに除外
    josephus = CircularLinkedList()
    for i in range(1, 8):
        josephus.append(i)
    order = josephus.josephus(3)
    print(f"Josephus elimination order: {order}")
    # [3, 6, 2, 7, 5, 1, 4]
```

### 4.4 循環リスト特有の注意点

循環リストを扱う際に最も危険なのは、**無限ループ**である。通常の連結リストでは `while curr:` や `while curr is not None:` でループを終了できるが、循環リストでは `None` に到達しないため、この条件では永遠にループが続く。

```
無限ループの危険:

  ✗ 間違ったコード:
    curr = head
    while curr:          # ← 循環リストでは永遠に True
        process(curr)
        curr = curr.next

  ○ 正しいコード（カウンタ方式）:
    curr = head
    for _ in range(size):  # ← サイズ分だけ回る
        process(curr)
        curr = curr.next

  ○ 正しいコード（開始地点との比較）:
    curr = head
    while True:
        process(curr)
        curr = curr.next
        if curr == head:   # ← 開始地点に戻ったら終了
            break
```

### 4.5 3種類のリスト比較

| 特性 | 単方向リスト | 双方向リスト | 循環リスト |
|------|------------|------------|-----------|
| ノードのポインタ | next のみ | prev + next | next（+ prev） |
| 末尾 → 先頭 | 不可（O(n)） | 不可（O(n)） | O(1) |
| 先頭 → 末尾 | O(n) | O(1)（tail あり） | O(1)（tail あり） |
| メモリ/ノード | 小 | 中 | 小〜中 |
| 走査方向 | 前方のみ | 双方向 | 前方（循環） |
| 末尾の判定 | next == None | next == sentinel | next == head |
| 無限ループリスク | 低い | 低い | 高い（要注意） |
| 代表的な用途 | スタック、単純なリスト | LRU キャッシュ、Deque | ラウンドロビン、ジョセフス |
| 実装難度 | 易 | 中 | 中 |

---

## 5. 実装パターンと典型アルゴリズム

### 5.1 2つのソート済みリストのマージ

ソート済みの2つの連結リストを1つのソート済みリストにマージする。マージソートの部品として使われるほか、単独でも頻出する面接問題である。

```python
def merge_sorted_lists(l1, l2):
    """2つのソート済み連結リストをマージ: O(n + m) 時間, O(1) 空間

    ダミーヘッドを使って簡潔に実装する。
    新しいノードは作成せず、既存のノードをつなぎ替える。

    Args:
        l1: ソート済みリストの head
        l2: ソート済みリストの head

    Returns:
        マージ後のリストの head
    """
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

    # 残りをそのまま接続
    curr.next = l1 if l1 else l2
    return dummy.next
```

```
マージの動作:

l1: 1 → 3 → 5 → None
l2: 2 → 4 → 6 → None

dummy → ?

Step 1: 1 <= 2 → dummy → 1, l1 = 3
Step 2: 3 > 2  → dummy → 1 → 2, l2 = 4
Step 3: 3 <= 4 → dummy → 1 → 2 → 3, l1 = 5
Step 4: 5 > 4  → dummy → 1 → 2 → 3 → 4, l2 = 6
Step 5: 5 <= 6 → dummy → 1 → 2 → 3 → 4 → 5, l1 = None
残り:           → dummy → 1 → 2 → 3 → 4 → 5 → 6

結果: 1 → 2 → 3 → 4 → 5 → 6
```

### 5.2 連結リストのソート（マージソート）

連結リストに対するソートでは、マージソートが最適である。配列で高効率なクイックソートは、連結リストではランダムアクセスの欠如により性能が劣化する。一方、マージソートはシーケンシャルアクセスのみで動作するため、連結リストと相性がよい。

```python
def sort_linked_list(head):
    """連結リストのマージソート: O(n log n) 時間, O(log n) 空間

    1. リストの中間点で2分割
    2. 各半分を再帰的にソート
    3. ソート済みの2リストをマージ

    空間計算量の O(log n) は再帰スタックの深さ。
    配列のマージソートと異なり、追加の配列は不要。
    """
    # ベースケース: 0ノードまたは1ノード
    if not head or not head.next:
        return head

    # 中間点で分割
    mid = find_middle_first(head)  # 前者の中間を取得
    second_half = mid.next
    mid.next = None  # リストを切断

    # 再帰的にソート
    left = sort_linked_list(head)
    right = sort_linked_list(second_half)

    # マージ
    return merge_sorted_lists(left, right)


# --- 動作確認 ---
if __name__ == "__main__":
    # ソートのテスト
    def build_list(values):
        dummy = ListNode(0)
        curr = dummy
        for v in values:
            curr.next = ListNode(v)
            curr = curr.next
        return dummy.next

    def print_list(head):
        values = []
        while head:
            values.append(str(head.val))
            head = head.next
        print(" -> ".join(values))

    unsorted = build_list([4, 2, 1, 3, 5])
    print("Before sort:", end=" ")
    print_list(unsorted)  # 4 -> 2 -> 1 -> 3 -> 5

    sorted_head = sort_linked_list(unsorted)
    print("After sort: ", end=" ")
    print_list(sorted_head)  # 1 -> 2 -> 3 -> 4 -> 5
```

### 5.3 回文判定

連結リストが回文（前から読んでも後ろから読んでも同じ）かどうかを判定する。

```python
def is_palindrome(head):
    """連結リストの回文判定: O(n) 時間, O(1) 空間

    1. Fast/Slow ポインタで中間点を見つける
    2. 後半を反転する
    3. 前半と後半を比較する
    4. 後半を元に戻す（リストを破壊しない）
    """
    if not head or not head.next:
        return True

    # Step 1: 中間点を見つける
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: 後半を反転
    second_half = reverse_iterative(slow.next)

    # Step 3: 比較
    first_half = head
    second_half_copy = second_half  # 後で元に戻すために保存
    is_palin = True

    while second_half:
        if first_half.val != second_half.val:
            is_palin = False
            break
        first_half = first_half.next
        second_half = second_half.next

    # Step 4: 後半を元に戻す
    slow.next = reverse_iterative(second_half_copy)

    return is_palin
```

```
回文判定の動作例:

リスト: 1 → 2 → 3 → 2 → 1

Step 1: 中間点 = 3（slow が 3 を指す）
  前半: 1 → 2 → 3
  後半: 2 → 1

Step 2: 後半を反転
  後半: 1 → 2

Step 3: 比較
  前半: 1, 2  ←→  後半: 1, 2  ✓ 一致

結果: True（回文）
```

### 5.4 交差するリストの交点検出

2つの単方向リストが途中から合流している場合、合流ノードを見つける。

```python
def find_intersection(headA, headB):
    """2つのリストの交点を検出: O(n + m) 時間, O(1) 空間

    ポインタ A と B を使う。A がリスト A の末尾に達したら
    リスト B の先頭に移動し、B も同様にする。
    両方のポインタは同じ合計距離を進むため、交点で出会う。

    交差がない場合、両方とも None に到達して終了する。
    """
    if not headA or not headB:
        return None

    a, b = headA, headB

    # 両方のポインタが出会うか、両方 None になるまで
    while a != b:
        a = a.next if a else headB
        b = b.next if b else headA

    return a  # 交点ノード、または None（交差なし）
```

```
交点検出の動作:

リスト A: 1 → 2 ┐
                  ├→ 8 → 9 → None
リスト B: 3 → 4 → 5 ┘

長さ: A = 4, B = 5

ポインタ A の経路: 1→2→8→9→(B開始)→3→4→5→8
ポインタ B の経路: 3→4→5→8→9→(A開始)→1→2→8

両方が8（交点）で出会う。
合計移動距離: A = 4 + 4 = 8, B = 5 + 3 = 8（同じ！）

数学的根拠:
  A の独自部分の長さ: a
  B の独自部分の長さ: b
  共通部分の長さ: c

  ポインタ A の合計距離: a + c + b
  ポインタ B の合計距離: b + c + a
  → 常に等しいため、交点で出会う
```

---

## 6. 応用: 実践的なデータ構造

### 6.1 LRU キャッシュ（Least Recently Used Cache）

LRU キャッシュは、双方向連結リストとハッシュマップを組み合わせた実践的なデータ構造であり、面接でも頻出する。キャッシュ容量を超えた場合、最も長い間使われていない（Least Recently Used）エントリを削除する。

```
LRU キャッシュの構造:

  ┌─────────────────────────────────────────────┐
  │  HashMap: key → DListNode                    │
  │  ┌──────┬──────┬──────┬──────┐              │
  │  │ k1→  │ k2→  │ k3→  │ k4→  │              │
  │  └──┼───┴──┼───┴──┼───┴──┼───┘              │
  │     │      │      │      │                   │
  │     ▼      ▼      ▼      ▼                   │
  │  [head] ⇌ [k1] ⇌ [k2] ⇌ [k3] ⇌ [k4] ⇌ [tail]  │
  │  sentinel  ← 最近使われた ─── 古い →  sentinel │
  │                                               │
  │  ・get/put のたびに該当ノードを先頭に移動     │
  │  ・容量超過時は tail 側（最も古い）を削除      │
  └─────────────────────────────────────────────┘

  全操作 O(1):
  - get: HashMap で O(1) 検索 + O(1) でノードを先頭に移動
  - put: HashMap で O(1) 挿入 + O(1) で先頭にノード追加
  - evict: O(1) で tail 側から削除 + O(1) で HashMap から削除
```

```python
class LRUCache:
    """LRU キャッシュ: 双方向連結リスト + ハッシュマップ

    全操作（get, put）が O(1) で動作する。

    Attributes:
        capacity: キャッシュの最大容量
        cache: key から DListNode へのマッピング
        _head: 番兵ノード（最新側）
        _tail: 番兵ノード（最古側）
    """

    class _Node:
        __slots__ = ('key', 'val', 'prev', 'next')

        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity):
        """
        Args:
            capacity: キャッシュの最大容量（1以上）

        Raises:
            ValueError: capacity が1未満の場合
        """
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # 番兵ノード
        self._head = self._Node()
        self._tail = self._Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node):
        """ノードをリストから除去: O(1)"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node):
        """ノードをリスト先頭（head の直後）に追加: O(1)"""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node):
        """ノードをリスト先頭に移動: O(1)"""
        self._remove(node)
        self._add_to_front(node)

    def get(self, key):
        """キーに対応する値を取得: O(1)

        見つかった場合、そのエントリを「最近使用」として先頭に移動する。

        Args:
            key: 検索キー

        Returns:
            値。キーが存在しない場合は -1
        """
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            return node.val
        return -1

    def put(self, key, value):
        """キーと値のペアを挿入/更新: O(1)

        既存のキーの場合は値を更新し先頭に移動。
        新しいキーの場合は先頭に挿入し、容量超過なら最古を削除。

        Args:
            key: キー
            value: 値
        """
        if key in self.cache:
            # 既存のキー: 値を更新して先頭に移動
            node = self.cache[key]
            node.val = value
            self._move_to_front(node)
        else:
            # 新しいキー
            if len(self.cache) >= self.capacity:
                # 容量超過: 最古のエントリ（tail の直前）を削除
                lru_node = self._tail.prev
                self._remove(lru_node)
                del self.cache[lru_node.key]

            # 新しいノードを先頭に追加
            new_node = self._Node(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)

    def __repr__(self):
        items = []
        node = self._head.next
        while node != self._tail:
            items.append(f"{node.key}:{node.val}")
            node = node.next
        return f"LRUCache({' -> '.join(items)}, cap={self.capacity})"


# --- 動作確認 ---
if __name__ == "__main__":
    lru = LRUCache(3)

    lru.put("a", 1)
    lru.put("b", 2)
    lru.put("c", 3)
    print(lru)  # LRUCache(c:3 -> b:2 -> a:1, cap=3)

    # "a" にアクセス → 先頭に移動
    print(lru.get("a"))  # 1
    print(lru)  # LRUCache(a:1 -> c:3 -> b:2, cap=3)

    # 新しいキー "d" を追加 → 最古の "b" が削除される
    lru.put("d", 4)
    print(lru)  # LRUCache(d:4 -> a:1 -> c:3, cap=3)

    # 削除された "b" にアクセス → -1
    print(lru.get("b"))  # -1
```

### 6.2 多項式の表現と演算

連結リストは多項式の表現にも適している。各項（係数と指数）をノードとして保持し、項の追加や削除をポインタ操作で行える。

```python
class TermNode:
    """多項式の項を表すノード"""
    __slots__ = ('coef', 'exp', 'next')

    def __init__(self, coef, exp, next=None):
        self.coef = coef  # 係数
        self.exp = exp    # 指数
        self.next = next

    def __repr__(self):
        if self.exp == 0:
            return f"{self.coef}"
        elif self.exp == 1:
            return f"{self.coef}x"
        else:
            return f"{self.coef}x^{self.exp}"


class Polynomial:
    """多項式クラス（連結リストで実装）

    各項は指数の降順でソートされている。
    例: 3x^3 + 2x^2 + 5x + 1
        → [3,3] → [2,2] → [5,1] → [1,0] → None
    """

    def __init__(self):
        self.head = None

    def add_term(self, coef, exp):
        """項を追加（指数の降順を維持）"""
        if coef == 0:
            return

        new_node = TermNode(coef, exp)

        # 空リストまたは先頭に挿入
        if not self.head or self.head.exp < exp:
            new_node.next = self.head
            self.head = new_node
            return

        # 同じ指数の項が存在する場合は係数を加算
        if self.head.exp == exp:
            self.head.coef += coef
            if self.head.coef == 0:
                self.head = self.head.next
            return

        # 挿入位置を探索
        curr = self.head
        while curr.next and curr.next.exp > exp:
            curr = curr.next

        if curr.next and curr.next.exp == exp:
            curr.next.coef += coef
            if curr.next.coef == 0:
                curr.next = curr.next.next
        else:
            new_node.next = curr.next
            curr.next = new_node

    def __add__(self, other):
        """多項式の加算: O(n + m)"""
        result = Polynomial()
        p1, p2 = self.head, other.head

        while p1 and p2:
            if p1.exp > p2.exp:
                result.add_term(p1.coef, p1.exp)
                p1 = p1.next
            elif p1.exp < p2.exp:
                result.add_term(p2.coef, p2.exp)
                p2 = p2.next
            else:
                total_coef = p1.coef + p2.coef
                if total_coef != 0:
                    result.add_term(total_coef, p1.exp)
                p1 = p1.next
                p2 = p2.next

        while p1:
            result.add_term(p1.coef, p1.exp)
            p1 = p1.next
        while p2:
            result.add_term(p2.coef, p2.exp)
            p2 = p2.next

        return result

    def evaluate(self, x):
        """多項式に x を代入して評価: O(n)"""
        result = 0
        curr = self.head
        while curr:
            result += curr.coef * (x ** curr.exp)
            curr = curr.next
        return result

    def __repr__(self):
        if not self.head:
            return "0"
        terms = []
        curr = self.head
        while curr:
            terms.append(repr(curr))
            curr = curr.next
        return " + ".join(terms)


# --- 動作確認 ---
if __name__ == "__main__":
    # p1 = 3x^3 + 2x^2 + 5
    p1 = Polynomial()
    p1.add_term(3, 3)
    p1.add_term(2, 2)
    p1.add_term(5, 0)
    print(f"p1 = {p1}")  # p1 = 3x^3 + 2x^2 + 5

    # p2 = x^3 + 4x + 2
    p2 = Polynomial()
    p2.add_term(1, 3)
    p2.add_term(4, 1)
    p2.add_term(2, 0)
    print(f"p2 = {p2}")  # p2 = 1x^3 + 4x + 2

    # 加算
    p3 = p1 + p2
    print(f"p1 + p2 = {p3}")  # p1 + p2 = 4x^3 + 2x^2 + 4x + 7

    # 評価
    print(f"p3(2) = {p3.evaluate(2)}")  # p3(2) = 4*8 + 2*4 + 4*2 + 7 = 55
```

### 6.3 スキップリスト（Skip List）の概念

スキップリストは、連結リストに複数レベルの「高速道路」を追加することで、検索を O(log n) に高速化したデータ構造である。平衡二分探索木の代替として使われ、Redis の Sorted Set などの実装で採用されている。

```
スキップリストの構造（概念図）:

Level 3: head ─────────────────────────────────── 9 → None
Level 2: head ────────── 3 ─────────────── 7 ─── 9 → None
Level 1: head ──── 2 ── 3 ──── 5 ──── 7 ──── 9 → None
Level 0: head → 1, 2, 3, 4, 5, 6, 7, 8, 9 → None
         (全要素を含む通常の連結リスト)

検索例: 値 6 を探す
  Level 3: head → 9（大きすぎる。下へ）
  Level 2: head → 3（OK）→ 7（大きすぎる。下へ）
  Level 1: 3 → 5（OK）→ 7（大きすぎる。下へ）
  Level 0: 5 → 6（見つかった！）

→ 全ノードを順に辿る O(n) ではなく、スキップにより O(log n)
→ 各ノードのレベルはランダムに決定（確率的データ構造）
```

スキップリストの詳細実装はこの章の範囲を超えるが、連結リストの発展形として知っておくことは重要である。連結リストの弱点であるランダムアクセスの遅さを、確率的な階層構造によって克服するという発想は示唆に富む。

### 6.4 XOR 連結リスト（メモリ効率の最適化）

XOR 連結リストは、双方向連結リストのメモリ使用量を削減するための工夫である。各ノードが prev と next の2つのポインタの代わりに、`prev XOR next` という1つの値のみを保持する。

```
XOR 連結リストの原理:

通常の双方向リスト（ポインタ2本/ノード）:
  [A] ⇌ [B] ⇌ [C] ⇌ [D]
  B.prev = addr(A), B.next = addr(C)

XOR 連結リスト（ポインタ1本/ノード）:
  [A] - [B] - [C] - [D]
  B.npx = addr(A) XOR addr(C)

走査の仕組み（A から C 方向に進む場合）:
  前のノードのアドレス: addr(A) が既知
  B.npx = addr(A) XOR addr(C)
  addr(C) = B.npx XOR addr(A)  ← XOR の性質: a XOR b XOR a = b

メモリ節約: ポインタ1本分/ノード
代償: ガベージコレクタとの相性が悪い（Python, Java では実用的でない）
    → C/C++ でのみ実用的な手法
```

XOR 連結リストは Python のような言語では実装が困難（アドレスを直接操作できないため）だが、C/C++ では実用的なテクニックである。概念として知っておくことで、メモリ最適化の発想を広げられる。

### 6.5 自己組織化リスト

自己組織化リスト（Self-Organizing List）は、アクセスパターンに基づいてノードの順序を動的に変更することで、頻繁にアクセスされる要素への検索時間を短縮するリストである。

```
自己組織化リストの主要戦略:

1. Move-to-Front（MTF）: アクセスされたノードを先頭に移動
   Before: [D] → [B] → [A] → [C]
   Access A:
   After:  [A] → [D] → [B] → [C]
   → LRU キャッシュと同じ発想

2. Transpose: アクセスされたノードを1つ前に移動
   Before: [D] → [B] → [A] → [C]
   Access A:
   After:  [D] → [A] → [B] → [C]
   → MTF より穏やかな再配置

3. Count: アクセス回数でソート
   Before: [D:3] → [B:1] → [A:5] → [C:2]
   Access A (count 5→6):
   After:  [A:6] → [D:3] → [C:2] → [B:1]
   → 最も正確だが、カウンタの管理が必要
```

LRU キャッシュは Move-to-Front 戦略の応用例であり、実用上もっとも広く使われている戦略である。

---

## 7. 配列との比較分析

### 7.1 計算量の総合比較

```
┌─────────────────────┬──────────────┬──────────────┬────────────────┐
│ 操作                │ 配列         │ 単方向リスト  │ 双方向リスト    │
│                     │ (動的配列)   │              │                │
├─────────────────────┼──────────────┼──────────────┼────────────────┤
│ 先頭に挿入          │ O(n)         │ O(1)  ★     │ O(1)  ★       │
│ 末尾に挿入          │ O(1) 償却 ★ │ O(n)         │ O(1)  ★       │
│ 中間に挿入          │ O(n)         │ O(n)         │ O(n)           │
│ 中間に挿入          │ O(n)         │ O(1)*        │ O(1)*  ★      │
│ (位置が既知)        │              │              │                │
│ 先頭を削除          │ O(n)         │ O(1)  ★     │ O(1)  ★       │
│ 末尾を削除          │ O(1) ★      │ O(n)         │ O(1)  ★       │
│ 任意ノードを削除    │ O(n)         │ O(n)         │ O(1)*  ★      │
│ (参照が既知)        │              │              │                │
│ インデックスアクセス │ O(1)  ★     │ O(n)         │ O(min(i,n-i))  │
│ 値の検索            │ O(n)         │ O(n)         │ O(n)           │
│ 値の検索(ソート済み) │ O(log n) ★  │ O(n)         │ O(n)           │
│ メモリ効率          │ 良い ★      │ やや悪い     │ 悪い           │
│ キャッシュ局所性    │ 良い ★      │ 悪い         │ 悪い           │
└─────────────────────┴──────────────┴──────────────┴────────────────┘

★ = 優位な操作
* = 挿入/削除位置のポインタが既に分かっている場合
```

### 7.2 メモリ使用量の詳細比較

データの型と要素数に応じたメモリ使用量を分析する。

```
64ビット環境での整数値 n 個の格納に必要なメモリ:

配列（Python list）:
  オーバーヘッド: 56 バイト（リストオブジェクト）
  各要素: 8 バイト（ポインタ）+ 28 バイト（int オブジェクト）= 36 バイト
  合計: 56 + 36n バイト
  ※ 実際はオーバーアロケーション（1.125倍程度）がある

単方向連結リスト:
  各ノード: 28 バイト（int）+ 8 バイト（next ポインタ）
           + オブジェクトヘッダ 56 バイト（__slots__ 使用時は少ない）
  合計: 約 72n バイト（__slots__ 未使用時はさらに増加）

双方向連結リスト:
  各ノード: 28 バイト（int）+ 16 バイト（prev + next ポインタ）
           + オブジェクトヘッダ
  合計: 約 80n バイト + 番兵2個分

n = 1,000,000（100万要素）の場合:
  配列:        約 36 MB
  単方向リスト: 約 72 MB（配列の約2倍）
  双方向リスト: 約 80 MB（配列の約2.2倍）
```

### 7.3 キャッシュ局所性の影響

現代のプロセッサでは、CPU キャッシュの活用が性能に大きく影響する。配列はメモリ上で連続しているため、一度のキャッシュラインの読み込みで複数の要素にアクセスできる。一方、連結リストのノードはメモリ上に散在しているため、ノードごとにキャッシュミスが発生する可能性が高い。

```
キャッシュ局所性の比較:

配列（連続メモリ）:
  キャッシュライン（64バイト）に8個の int が収まる
  ┌─────────────────────────────────────────┐
  │ [1] [2] [3] [4] [5] [6] [7] [8]        │ ← 1回のロードで8要素
  └─────────────────────────────────────────┘
  次のキャッシュラインも連続 → プリフェッチ可能

連結リスト（分散メモリ）:
  キャッシュライン1: [Node A: val=1, next=0x...]  ← 1回で1ノードのみ
  キャッシュライン2: [別のデータ]                  ← キャッシュミス
  キャッシュライン3: [Node B: val=2, next=0x...]  ← キャッシュミス
  キャッシュライン4: [別のデータ]                  ← キャッシュミス
  ...

  → 同じ n 要素の走査でも、実測性能は数倍〜10倍以上の差になりうる
  → O(n) vs O(n) でも定数因子が大きく異なる

例外: メモリプール（同じ領域から連続してノードを割り当てる）を
      使用すれば、ある程度キャッシュ局所性を改善できる
```

### 7.4 使い分けの指針

```
配列を使うべき場面:
  ✓ ランダムアクセスが頻繁
  ✓ 末尾への追加・削除が主要操作
  ✓ データサイズが既知または変動が小さい
  ✓ 検索操作が多い（特にソート済みの場合）
  ✓ キャッシュ効率が重要
  ✓ メモリ使用量を最小化したい
  → 多くのケースでは配列が最適解

連結リストを使うべき場面:
  ✓ 先頭への挿入・削除が頻繁
  ✓ 中間の挿入・削除が頻繁かつ位置が既知
  ✓ 最悪ケースの時間保証が必要（動的配列のリサイズを避けたい）
  ✓ 他のデータ構造の内部構造として（LRU キャッシュ、ハッシュテーブルの連鎖法等）
  ✓ 要素の順序変更が頻繁
  → 特定のユースケースに限って有利

実務での経験則:
  → 「まず配列で実装し、プロファイリングで問題が見つかったら連結リストを検討する」
  → 連結リストが最適なケースは全体の5-10%程度
  → ただし、そのケースでは連結リストが圧倒的に有利
```

### 7.5 各言語の標準ライブラリにおける連結リスト

```
各言語における連結リストの提供状況:

┌──────────┬─────────────────┬──────────────────────────────┐
│ 言語     │ クラス名         │ 備考                         │
├──────────┼─────────────────┼──────────────────────────────┤
│ Python   │ collections.deque│ 双方向リスト。list は動的配列 │
│ Java     │ LinkedList      │ 双方向リスト。List/Deque 実装 │
│ C++      │ std::list       │ 双方向リスト                 │
│          │ std::forward_list│ 単方向リスト（C++11〜）      │
│ C#       │ LinkedList<T>   │ 双方向リスト                 │
│ Go       │ container/list  │ 双方向リスト                 │
│ Rust     │ std::collections│ LinkedList（双方向）         │
│          │ ::LinkedList    │ ※ 非推奨に近い扱い           │
│ Haskell  │ [ ] (リスト)    │ 単方向リスト。言語の根幹      │
└──────────┴─────────────────┴──────────────────────────────┘

注目点:
- Python は明示的な LinkedList クラスを提供していない
  → deque が内部的に双方向リストを使用
  → 多くの場面で list（動的配列）が推奨される
- Rust は LinkedList の使用を推奨しておらず、Vec を優先する
  → キャッシュ局所性の観点から
- Haskell ではリスト（単方向連結リスト）が最も基本的なデータ構造
  → 関数型プログラミングでは不変リストが自然な選択
```

---

## 8. アンチパターン

### 8.1 アンチパターン1: 不要な連結リストの使用

**問題:**
配列で十分な場面で連結リストを使ってしまうパターン。「連結リストの方が挿入・削除が速い」という表面的な理解に基づく誤った選択。

```python
# ✗ アンチパターン: 末尾追加と走査だけなら配列で十分
class BadLogBuffer:
    """ログバッファとして連結リストを使用（非推奨）"""
    def __init__(self):
        self.head = None
        self.tail = None

    def add_log(self, message):
        """ログを末尾に追加"""
        node = ListNode(message)
        if not self.tail:
            self.head = self.tail = node
        else:
            self.tail.next = node
            self.tail = node

    def get_all_logs(self):
        """全ログを取得"""
        logs = []
        curr = self.head
        while curr:
            logs.append(curr.val)
            curr = curr.next
        return logs


# ○ 正しいアプローチ: 配列（リスト）を使用
class GoodLogBuffer:
    """ログバッファとして配列を使用（推奨）"""
    def __init__(self):
        self.logs = []

    def add_log(self, message):
        """ログを末尾に追加: O(1) 償却"""
        self.logs.append(message)

    def get_all_logs(self):
        """全ログを取得: O(n)、キャッシュ効率良好"""
        return list(self.logs)

    def get_log(self, index):
        """特定のログを取得: O(1)"""
        return self.logs[index]
```

**なぜ配列の方が適切か:**
- 主な操作が「末尾追加」と「全走査」 → 配列は両方とも効率的
- ランダムアクセス（特定のログの参照）が O(1) で可能
- キャッシュ局所性により走査が高速
- メモリ使用量が少ない
- コードがシンプル

**連結リストを選ぶべき判断基準:**
1. 先頭や中間への挿入・削除が頻繁か？
2. 挿入・削除位置のポインタを別途保持しているか？
3. 最悪ケースの時間保証が必要か？

3つすべてに「いいえ」なら、配列を選ぶべきである。

### 8.2 アンチパターン2: メモリリークとダングリングポインタ

**問題:**
ノードを削除する際にポインタの後始末を怠り、ガベージコレクションが効かない参照が残るパターン。Python ではガベージコレクタが多くのケースをカバーするが、循環参照がある場合は注意が必要。

```python
# ✗ アンチパターン: 循環参照によるメモリリーク
class BadNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.parent_list = None  # リストへの逆参照


class BadLinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = BadNode(val)
        node.parent_list = self  # 循環参照を形成！
        if not self.head:
            self.head = node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = node

    def clear(self):
        """リストをクリア（メモリリークの可能性）"""
        self.head = None
        # ← 各ノードが self（リスト）を参照しているため
        # ← ノード ↔ リスト の循環参照が GC で回収されない可能性


# ○ 正しいアプローチ: 循環参照を避ける、または明示的にクリア
class GoodLinkedList:
    def __init__(self):
        self.head = None

    def clear(self):
        """メモリリークを防ぐクリア"""
        curr = self.head
        while curr:
            next_node = curr.next
            curr.next = None       # 参照をクリア
            curr = next_node
        self.head = None
```

**教訓:**
- ノードからリスト本体への逆参照は循環参照を生む
- 双方向リストの `prev`/`next` も技術的には循環参照だが、Python の世代別 GC で通常は回収される
- 大規模なリストを頻繁に生成・破棄する場合は、明示的な参照のクリアが安全
- C/C++ では手動メモリ管理のため、この問題がさらに深刻になる

### 8.3 アンチパターン3: O(n) 操作の繰り返しによる O(n^2) 化

**問題:**
連結リストの get(i) が O(n) であることを忘れ、ループ内でインデックスアクセスを繰り返すパターン。

```python
# ✗ アンチパターン: O(n^2) になるインデックスベースのアクセス
def print_all_bad(linked_list):
    """全要素を表示（非推奨: O(n^2)）"""
    for i in range(len(linked_list)):
        # get(i) は毎回 head から i 回辿る → O(i)
        # 合計: 0 + 1 + 2 + ... + (n-1) = O(n^2)
        print(linked_list.get(i))


# ○ 正しいアプローチ: イテレータを使用して O(n) で走査
def print_all_good(linked_list):
    """全要素を表示（推奨: O(n)）"""
    for val in linked_list:  # __iter__ で順次走査
        print(val)


# ○ 正しいアプローチ: ノードポインタを直接使用
def print_all_direct(head):
    """全要素を表示（推奨: O(n)）"""
    curr = head
    while curr:
        print(curr.val)
        curr = curr.next
```

**教訓:**
- 連結リストにインデックスでアクセスするたびに O(n) の走査が発生する
- 配列の感覚で `for i in range(n): list.get(i)` とすると O(n^2) になる
- 連結リストの走査はイテレータまたはポインタの直接走査で行う
- この罠は Java の `LinkedList` でも同様であり、`for (int i = 0; i < list.size(); i++)` ではなく `for (T item : list)` を使うべきである

### 8.4 アンチパターン4: スレッドセーフティの欠如

**問題:**
マルチスレッド環境で連結リストを共有する際に、適切な同期機構を設けないパターン。

```python
import threading

# ✗ アンチパターン: スレッドセーフでないリスト操作
class UnsafeSharedList:
    def __init__(self):
        self.head = None

    def prepend(self, val):
        # スレッドAとBが同時にprependすると:
        # A: new_node.next = self.head  (head = [1])
        # B: new_node.next = self.head  (head = [1])  ← 同じ head を読む
        # A: self.head = new_node_A     (head = [A, 1])
        # B: self.head = new_node_B     (head = [B, 1])  ← A のノードが失われる！
        self.head = ListNode(val, self.head)


# ○ 正しいアプローチ: ロックで保護
class ThreadSafeList:
    def __init__(self):
        self.head = None
        self._lock = threading.Lock()

    def prepend(self, val):
        with self._lock:
            self.head = ListNode(val, self.head)

    def pop_front(self):
        with self._lock:
            if not self.head:
                raise IndexError("pop from empty list")
            val = self.head.val
            self.head = self.head.next
            return val
```

---

## 9. 演習問題

### 9.1 基礎レベル

**演習1: 連結リストの基本操作**

以下の操作を持つ単方向連結リストを実装せよ。

- `prepend(val)`: 先頭に値を挿入
- `append(val)`: 末尾に値を挿入
- `delete(val)`: 指定値の最初の出現を削除
- `search(val)`: 値が存在するか判定
- `__len__()`: リストの長さを返す
- `to_list()`: Python リストに変換

```python
# テストケース
ll = SinglyLinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.prepend(0)
assert ll.to_list() == [0, 1, 2, 3]
assert len(ll) == 4
assert ll.search(2) == 2  # index 2
assert ll.search(99) == -1
ll.delete(2)
assert ll.to_list() == [0, 1, 3]
```

**ヒント:** ダミーヘッド（番兵ノード）を使うと、先頭ノードの削除を特別扱いする必要がなくなる。

---

**演習2: 連結リストの反転**

与えられた単方向連結リストを反転する関数を、反復版と再帰版の2通りで実装せよ。

```python
# テストケース
head = build_list([1, 2, 3, 4, 5])
reversed_head = reverse_iterative(head)
assert list_to_array(reversed_head) == [5, 4, 3, 2, 1]

head2 = build_list([1, 2, 3, 4, 5])
reversed_head2 = reverse_recursive(head2)
assert list_to_array(reversed_head2) == [5, 4, 3, 2, 1]

# エッジケース
assert reverse_iterative(None) is None  # 空リスト
assert reverse_iterative(ListNode(42)).val == 42  # 1要素
```

**ヒント（反復版）:** 3つのポインタ `prev`, `curr`, `next_node` を使い、各ステップで `curr.next` を `prev` に付け替える。

**ヒント（再帰版）:** ベースケースは「空リストまたは1ノード」。再帰で末尾まで進み、戻りながらポインタを付け替える。

---

**演習3: 末尾から N 番目のノードを削除**

単方向連結リストの末尾から N 番目のノードを削除する関数を実装せよ。1パスで解くこと（リストを2回走査しない）。

```python
# テストケース
head = build_list([1, 2, 3, 4, 5])
# 末尾から2番目（=4）を削除
new_head = remove_nth_from_end(head, 2)
assert list_to_array(new_head) == [1, 2, 3, 5]

# エッジケース: 先頭を削除
head2 = build_list([1, 2])
new_head2 = remove_nth_from_end(head2, 2)
assert list_to_array(new_head2) == [2]
```

**ヒント:** 2つのポインタ `fast` と `slow` を使う。まず `fast` を N 回進めてから、`fast` と `slow` を同時に進める。`fast` がリスト末尾に到達したとき、`slow` が削除対象の直前にいる。

### 9.2 応用レベル

**演習4: 2つのソート済みリストのマージ**

2つのソート済み単方向リストを、1つのソート済みリストにマージする関数を実装せよ。新しいノードは作成せず、既存のノードをつなぎ替えること。

```python
# テストケース
l1 = build_list([1, 3, 5, 7])
l2 = build_list([2, 4, 6, 8])
merged = merge_sorted(l1, l2)
assert list_to_array(merged) == [1, 2, 3, 4, 5, 6, 7, 8]

# エッジケース
l3 = build_list([1, 2, 3])
merged2 = merge_sorted(l3, None)
assert list_to_array(merged2) == [1, 2, 3]
```

---

**演習5: サイクル検出とサイクル開始ノードの特定**

以下の2つの関数を実装せよ。
1. `has_cycle(head)`: サイクルが存在するか判定
2. `find_cycle_start(head)`: サイクルの開始ノードを返す（サイクルがなければ None）

追加空間は O(1) とすること（ハッシュセットの使用は不可）。

```python
# テストケース
# 1 → 2 → 3 → 4 → 5 → 3（サイクル）
nodes = [ListNode(i) for i in range(1, 6)]
for i in range(4):
    nodes[i].next = nodes[i + 1]
nodes[4].next = nodes[2]  # 5 → 3 でサイクル

assert has_cycle(nodes[0]) is True
assert find_cycle_start(nodes[0]).val == 3

# サイクルなし
normal = build_list([1, 2, 3])
assert has_cycle(normal) is False
assert find_cycle_start(normal) is None
```

**ヒント:** Floyd のサイクル検出アルゴリズム（亀とウサギ）を使う。サイクル開始ノードの特定には、出会い地点と head から同速度で進めるテクニックを使う。

---

**演習6: 連結リストの回文判定**

単方向連結リストが回文かどうかを判定する関数を実装せよ。O(n) 時間、O(1) 追加空間で解くこと。

```python
# テストケース
assert is_palindrome(build_list([1, 2, 3, 2, 1])) is True
assert is_palindrome(build_list([1, 2, 2, 1])) is True
assert is_palindrome(build_list([1, 2, 3])) is False
assert is_palindrome(build_list([1])) is True
assert is_palindrome(None) is True
```

**ヒント:** (1) 中間点を見つけ、(2) 後半を反転し、(3) 前半と後半を比較する。リストを元に戻すために、比較後に後半を再反転する。

### 9.3 発展レベル

**演習7: LRU キャッシュの実装**

双方向連結リストとハッシュマップを使って、以下の操作がすべて O(1) で動作する LRU キャッシュを実装せよ。

- `get(key)`: キーに対応する値を返す。キーが存在しなければ -1 を返す。アクセスしたエントリを「最近使用」にする。
- `put(key, value)`: キーと値のペアを挿入する。キーが既に存在すれば値を更新する。容量を超える場合は最も古いエントリを削除する。

```python
# テストケース
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1       # 1 にアクセス（最近使用に）
cache.put(3, 3)                 # 容量超過 → 2 が削除される
assert cache.get(2) == -1       # 2 は削除済み
cache.put(4, 4)                 # 容量超過 → 1 が削除される
assert cache.get(1) == -1       # 1 は削除済み
assert cache.get(3) == 3
assert cache.get(4) == 4
```

---

**演習8: K 個のソート済みリストのマージ**

K 個のソート済みリストを1つのソート済みリストにマージせよ。全リストの合計要素数を N とするとき、O(N log K) の時間計算量で解くこと。

```python
# テストケース
lists = [
    build_list([1, 4, 7]),
    build_list([2, 5, 8]),
    build_list([3, 6, 9]),
]
merged = merge_k_sorted(lists)
assert list_to_array(merged) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**ヒント:** 最小ヒープ（priority queue）を使う。各リストの先頭要素をヒープに入れ、最小値を取り出して結果に追加し、取り出したリストの次の要素をヒープに追加する。

---

**演習9: 連結リストを使ったテキストエディタバッファ**

双方向連結リストを使って、以下の操作を持つシンプルなテキストエディタバッファを実装せよ。

- `insert(char)`: カーソル位置に文字を挿入し、カーソルを右に移動
- `delete()`: カーソルの左の文字を削除
- `move_left()`: カーソルを左に1つ移動
- `move_right()`: カーソルを右に1つ移動
- `get_text()`: バッファの全テキストを文字列として返す

すべての操作を O(1) で実装すること。

```python
# テストケース
editor = TextEditor()
editor.insert('H')
editor.insert('l')
editor.insert('l')
editor.insert('o')
assert editor.get_text() == "Hllo"
editor.move_left()
editor.move_left()
editor.move_left()
editor.insert('e')
assert editor.get_text() == "Hello"
editor.move_right()
editor.move_right()
editor.move_right()
editor.insert('!')
assert editor.get_text() == "Hello!"
```

**ヒント:** カーソルの位置を「あるノードの直後」として管理する。番兵ノードをカーソルの基準点にすると実装が簡潔になる。

---

## 10. FAQ（よくある質問）

### Q1: Python の `list` は内部的に連結リストですか？

**A:** いいえ。Python の `list` は**動的配列**（dynamic array）で実装されている。内部的には C の配列（ポインタの配列）であり、メモリ上に連続して配置されている。そのため、インデックスアクセスが O(1)、末尾への追加が O(1) 償却で動作する。

Python で連結リスト的な操作が必要な場合は `collections.deque` が代替になる。`deque` は双方向連結リストに基づいた実装（正確にはブロック連結リスト）であり、両端での追加・削除が O(1) で動作する。

```python
from collections import deque

d = deque([1, 2, 3])
d.appendleft(0)  # 先頭に追加: O(1)
d.append(4)      # 末尾に追加: O(1)
d.popleft()      # 先頭から削除: O(1)
d.pop()          # 末尾から削除: O(1)

# ただし、中間のインデックスアクセスは O(n)
# d[i] は deque でも O(n)
```

### Q2: 面接で連結リストの問題を解くコツはありますか？

**A:** 以下のテクニックを覚えておくと、多くの問題に対応できる。

**1. ダミーヘッド（番兵ノード）**
先頭ノードの操作でエッジケースを回避する。マージ、削除、挿入など幅広い問題で有効。

**2. Fast/Slow ポインタ**
- 中間ノードの検出
- サイクル検出
- 回文判定
- 末尾から N 番目のノード

**3. 再帰的思考**
連結リストは再帰的な構造（head + 残りのリスト）であるため、再帰で自然に処理できる。ただし、スタックオーバーフローに注意（O(n) の空間使用）。

**4. 図を描く**
ポインタの操作は頭の中だけで追うと間違えやすい。必ず紙に図を描いてポインタの付け替えを確認する。

**5. よくあるエッジケース**
- 空リスト（head = None）
- 1ノードのリスト
- 2ノードのリスト
- 先頭ノードが対象の場合
- 末尾ノードが対象の場合

### Q3: 連結リストはガベージコレクションにどう影響しますか？

**A:** 連結リストはガベージコレクション（GC）にいくつかの影響を与える。

**1. 参照追跡のコスト**
GC はポインタを辿って到達可能なオブジェクトをマークする（Mark & Sweep）。連結リストのノードはメモリ上に分散しているため、GC がノードを辿る際にキャッシュミスが多発し、GC 停止時間が長くなる傾向がある。

**2. 循環参照の問題**
双方向リストのノードは `prev` と `next` で相互参照している。Python の参照カウント方式だけでは回収できないが、世代別 GC が循環参照を検出して回収する。ただし、`__del__` メソッドを持つオブジェクトが循環参照に含まれると、回収が遅れたり失敗したりする場合がある。

**3. メモリの断片化**
多数の小さなノードオブジェクトの割り当てと解放を繰り返すと、メモリが断片化する。配列は大きな連続領域を使うため断片化しにくい。

**対策:**
- `__slots__` を使ってノードのメモリフットプリントを削減する
- 不要になったリストは明示的にクリア（参照を None に設定）する
- 大規模なデータには配列ベースのデータ構造を検討する

### Q4: 関数型プログラミングにおける連結リストの役割は？

**A:** 関数型プログラミングでは、**不変（immutable）の単方向連結リスト**が最も基本的なデータ構造として使われる。Haskell のリスト `[1, 2, 3]` は内部的には `1 : (2 : (3 : []))` という連結リストである。

不変リストの利点は「構造共有（structural sharing）」にある。既存のリストの先頭に要素を追加しても、元のリストは変更されず、新しいリストは元のリストの tail を共有する。

```
構造共有の例:

xs = [2, 3, 4]
ys = 1 : xs     ← xs の先頭に 1 を追加

メモリ上:
ys: [1] ──┐
          ▼
xs:      [2] → [3] → [4] → []

xs と ys が [2, 3, 4] を共有している
→ コピーのコストがゼロ
→ 並行処理でも安全（データ競合が起きない）
```

### Q5: 配列と連結リストの境界はどこにあるのですか？

**A:** 理論的な計算量だけでなく、実装のコンテキストを含めて判断する必要がある。

**配列が有利な境界条件:**
- 要素数が少ない場合（数百程度まで）は、キャッシュ効率の優位性により配列が圧倒的に速い
- 挿入・削除が全操作の10-20%未満の場合は配列が有利
- メモリが限られた組み込みシステムでは配列が有利

**連結リストが有利な境界条件:**
- 先頭への挿入・削除が操作の大半を占める場合
- ノードの移動（あるリストから別のリストへ）が頻繁な場合（O(1) の splice 操作）
- 最悪ケースの時間保証が必要なリアルタイムシステム
- ハッシュマップと組み合わせて O(1) の削除を実現する場合（LRU キャッシュ等）

現代のハードウェアでは、CPU キャッシュの影響が非常に大きいため、理論上同じ O(n) でも配列の走査が連結リストの走査より5倍〜20倍速いことがある。この定数因子の差を考慮すると、「連結リストが有利になるのは、理論的な計算量が配列よりも明確に良い場合に限られる」と言える。

### Q6: なぜ面接では連結リストの問題が頻出するのですか？

**A:** 連結リストの問題が面接で頻出する理由は複数ある。

1. **ポインタ操作の理解度を測れる**: ポインタの付け替えはバグを生みやすく、正確に操作できるかが基本的なプログラミング力の指標になる。

2. **エッジケースの処理能力を測れる**: 空リスト、1ノード、先頭・末尾の操作など、場合分けが多く、注意深さが要求される。

3. **アルゴリズムの知識を測れる**: Fast/Slow ポインタ、再帰、分割統治など、多くのアルゴリズミックな考え方を連結リストの問題で出題できる。

4. **コードが短い**: ホワイトボードや制限時間のある面接に適した長さのコードで解答できる。

5. **空間計算量の意識を測れる**: 「追加空間 O(1) で解け」という制約を課すことで、空間効率への意識を評価できる。

---

## 11. 実務での連結リスト

### 11.1 実際に使われている場面

連結リストは理論的な学習対象として見られがちだが、実際のソフトウェアシステムでも重要な役割を果たしている。

```
連結リストが実際に使われている場面:

1. OS カーネル:
   - Linux カーネルの list.h: 双方向循環リストのマクロ群
   - プロセスリスト、デバイスドライバリスト
   - メモリのフリーリスト管理

2. キャッシュシステム:
   - LRU キャッシュ（Redis, Memcached, ブラウザキャッシュ）
   - LFU (Least Frequently Used) キャッシュ

3. テキストエディタ:
   - Emacs: テキストをギャップバッファ（配列ベース）+ 連結リストで管理
   - VS Code: ピーステーブル（piece table）で行リストを管理

4. ハッシュテーブル:
   - チェイニング法（衝突解決）で各バケットにリストを使用
   - Java の HashMap は要素数が多いとリスト → 赤黒木に変換

5. メモリアロケータ:
   - フリーリスト: 空きメモリブロックを連結リストで管理
   - バディアロケータ: 各サイズクラスのフリーリスト

6. ブロックチェーン:
   - 各ブロックが前のブロックのハッシュを保持
   - 概念的には単方向連結リスト

7. 関数型言語:
   - Haskell, Clojure, Erlang: 不変の連結リストが言語の基本構造
   - 構造共有による効率的なデータ操作

8. ネットワーキング:
   - Linux のパケットキュー (sk_buff): 双方向リスト
   - TCP の送信/受信バッファ管理
```

### 11.2 Python での実践的な選択

Python で連結リストに相当する機能が必要な場合、まず標準ライブラリの `collections.deque` を検討すべきである。自前で連結リストを実装するのは、以下の場合に限定される。

```
Python での選択指針:

list（動的配列）を使う場合:
  - ランダムアクセスが必要
  - 末尾への追加が主要操作
  - スライス操作が必要
  - 大半のユースケース → まずこれを試す

collections.deque を使う場合:
  - 両端からの追加・削除が必要
  - FIFO キューとして使う
  - 固定長のスライディングウィンドウ（maxlen パラメータ）

自前の連結リストを実装する場合:
  - ハッシュマップとの組み合わせ（LRU キャッシュ等）
  - ノード単位の操作が必要（ノードの移動、スプライシング）
  - 教育目的やアルゴリズム問題の解法
  - 特殊なデータ構造の基盤として
```

---

## 12. まとめ

### 12.1 重要概念の整理

| 概念 | ポイント |
|------|---------|
| 単方向リスト | next ポインタのみ。先頭挿入 O(1)。逆走査不可。最もシンプル |
| 双方向リスト | prev + next。両端操作 O(1)。任意ノードの削除 O(1)。LRU キャッシュの基盤 |
| 循環リスト | 末尾 → 先頭が循環。ラウンドロビン、ジョセフス問題。無限ループに注意 |
| vs 配列 | キャッシュ効率で配列が圧勝。連結リストは挿入・削除とポインタ操作で優位 |
| ダミーヘッド | 番兵ノードでエッジケースを排除。コードの簡潔化と正確性に貢献 |
| Fast/Slow | 中間点検出、サイクル検出、回文判定。2ポインタテクニックの代表 |
| LRU キャッシュ | 双方向リスト + ハッシュマップ。全操作 O(1)。面接頻出 |
| 反転 | 反復版: O(1) 空間。再帰版: O(n) 空間。最頻出の面接問題 |

### 12.2 チェックリスト

この章の学習を終えたら、以下の項目を確認しよう。

- [ ] 単方向・双方向・循環リストの構造と違いを説明できる
- [ ] 各操作（挿入、削除、検索、反転）の計算量を正確に答えられる
- [ ] ダミーヘッドを使って挿入・削除を実装できる
- [ ] Fast/Slow ポインタで中間ノードとサイクルを検出できる
- [ ] LRU キャッシュを双方向リスト + ハッシュマップで実装できる
- [ ] 配列と連結リストの使い分けを適切に判断できる
- [ ] 連結リストのアンチパターンを認識し回避できる

---

## 次に読むべきガイド

→ [[02-stacks-and-queues.md]] — スタックとキュー（連結リストで実装可能な基本データ構造）

---

## 参考文献

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). "Introduction to Algorithms." 4th Edition, MIT Press. Chapter 10: Elementary Data Structures — 連結リストの理論的基盤と操作の解説。
2. Sedgewick, R. & Wayne, K. (2011). "Algorithms." 4th Edition, Addison-Wesley. Chapter 1.3: Bags, Queues, and Stacks — 連結リストを用いたスタック・キューの実装。
3. Skiena, S. S. (2020). "The Algorithm Design Manual." 3rd Edition, Springer. Chapter 3: Data Structures — 連結リストの実践的な使い分けと設計判断。
4. Knuth, D. E. (1997). "The Art of Computer Programming, Volume 1: Fundamental Algorithms." 3rd Edition, Addison-Wesley. Section 2.2: Linear Lists — 連結リストの古典的な解説と数学的分析。
5. Python Documentation. "collections.deque." https://docs.python.org/3/library/collections.html#collections.deque — Python 標準ライブラリにおける連結リストベースのデータ構造。
6. Pugh, W. (1990). "Skip Lists: A Probabilistic Alternative to Balanced Trees." Communications of the ACM, 33(6), 668-676. — スキップリストの原論文。連結リストの発展形として重要。
