# 連結リスト — 単方向・双方向・循環・フロイドのアルゴリズム

> 配列と対をなす線形データ構造「連結リスト」の各種バリエーションと、
> サイクル検出の古典的アルゴリズム、そして実務での応用パターンまでを体系的に学ぶ。

---

## 目次

1. [連結リストとは何か](#1-連結リストとは何か)
2. [リストの種類と構造](#2-リストの種類と構造)
3. [基本実装 — ノードとリストクラス](#3-基本実装--ノードとリストクラス)
4. [コア操作の実装](#4-コア操作の実装)
5. [双方向リストの完全実装](#5-双方向リストの完全実装)
6. [循環リストの実装と応用](#6-循環リストの実装と応用)
7. [フロイドの循環検出アルゴリズム](#7-フロイドの循環検出アルゴリズム)
8. [応用アルゴリズム集](#8-応用アルゴリズム集)
9. [比較表と計算量まとめ](#9-比較表と計算量まとめ)
10. [アンチパターン集](#10-アンチパターン集)
11. [演習問題 — 基礎・応用・発展](#11-演習問題--基礎応用発展)
12. [FAQ — よくある質問](#12-faq--よくある質問)
13. [まとめ](#13-まとめ)
14. [参考文献](#14-参考文献)

---

## 1. 連結リストとは何か

### 1.1 定義と直感的理解

連結リスト (Linked List) は、各要素（ノード）がデータとポインタ（参照）を保持し、
ポインタによって次の要素を指すことで全体の順序を表現する線形データ構造である。

配列がメモリ上の連続領域にデータを格納するのに対し、
連結リストの各ノードはメモリ上の任意の位置に存在できる。
この特性により、挿入・削除がポインタの付け替えだけで完了するという利点を持つ。

```
【配列のメモリレイアウト】

  アドレス:  0x100  0x104  0x108  0x10C  0x110
            ┌──────┬──────┬──────┬──────┬──────┐
  値:       │  10  │  20  │  30  │  40  │  50  │
            └──────┴──────┴──────┴──────┴──────┘
            ← 連続したメモリ領域 →

【連結リストのメモリレイアウト】

  0x200          0x3F0          0x580          0x120
  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │ val: 10 │    │ val: 20 │    │ val: 30 │    │ val: 40 │
  │ next:─────→  │ next:─────→  │ next:─────→  │ next:None│
  └─────────┘    └─────────┘    └─────────┘    └─────────┘
  ← メモリ上でバラバラに配置 →
```

### 1.2 なぜ連結リストを学ぶのか

連結リストは以下の理由から、コンピュータサイエンスの基礎として不可欠である。

1. **ポインタ操作の基本**: ポインタ／参照を使ったデータ管理の最も基本的な形。
   ツリーやグラフなど、より複雑なデータ構造の基盤になる。

2. **動的メモリ管理の理解**: ノードの生成・破棄を通じて、動的メモリ割り当ての
   概念を自然に学べる。

3. **アルゴリズム設計の訓練**: 二つのポインタテクニック（slow/fast）、
   ダミーノードパターンなど、汎用的なアルゴリズム設計手法の入口となる。

4. **技術面接の頻出トピック**: Google、Meta、Amazon など大手テック企業の
   コーディング面接で最も頻繁に出題されるカテゴリの一つである。

### 1.3 連結リストの歴史的背景

連結リストは 1955-1956 年頃、Allen Newell、Cliff Shaw、Herbert A. Simon が
RAND Corporation で IPL (Information Processing Language) を開発する際に考案された。
その後、LISP 言語 (1958, John McCarthy) においてリスト処理が中核的概念として
採用され、連結リストはプログラミング言語とデータ構造の歴史に深く刻まれた。

---

## 2. リストの種類と構造

### 2.1 単方向リスト (Singly Linked List)

各ノードがデータと「次のノードへのポインタ」のみを持つ、最も基本的な形態。

```
単方向リスト (Singly Linked List):

  head
   │
   ▼
  ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
  │ val: "A"  │     │ val: "B"  │     │ val: "C"  │     │ val: "D"  │
  │ next: ─────────→│ next: ─────────→│ next: ─────────→│ next: None│
  └───────────┘     └───────────┘     └───────────┘     └───────────┘

  特徴:
  - 前方走査のみ可能（head → tail）
  - 各ノードはポインタを 1 つだけ保持
  - メモリ使用量が最も少ない
```

**利用場面**: スタックの実装、ハッシュテーブルのチェイン法、
メモリが制約された組み込みシステム。

### 2.2 双方向リスト (Doubly Linked List)

各ノードが前方ポインタ (next) と後方ポインタ (prev) の 2 つを持つ。

```
双方向リスト (Doubly Linked List):

  head                                                          tail
   │                                                             │
   ▼                                                             ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    ┌──────────────┐
  │ prev: None   │     │ prev: ───────────←│ prev: ───────────←│ prev: ──────────←
  │ val: "A"     │     │ val: "B"     │     │ val: "C"      │   │ val: "D"     │
  │ next: ────────────→│ next: ────────────→│ next: ────────────→│ next: None   │
  └──────────────┘     └──────────────┘     └──────────────┘    └──────────────┘

  特徴:
  - 前方・後方の双方向走査が可能
  - 各ノードはポインタを 2 つ保持（メモリ使用量が増加）
  - 任意のノードから前後への移動が O(1)
```

**利用場面**: LRU キャッシュ、テキストエディタのカーソル移動、ブラウザの「戻る/進む」。

### 2.3 循環リスト (Circular Linked List)

末尾ノードのポインタが先頭ノードを指し、リングを形成する。
単方向循環リストと双方向循環リストの 2 種類が存在する。

```
単方向循環リスト:

  head
   │
   ▼
  ┌───────────┐     ┌───────────┐     ┌───────────┐
  │ val: "A"  │     │ val: "B"  │     │ val: "C"  │
  │ next: ─────────→│ next: ─────────→│ next: ──┐  │
  └───────────┘     └───────────┘     └─────────│──┘
   ▲                                             │
   └─────────────────────────────────────────────┘

双方向循環リスト:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
  └→ │ prev: ──┐    │     │ prev: ───────────← │ prev: ───────────←┘
     │ val: "A"│    │     │ val: "B"     │     │ val: "C"     │
     │ next: ────────────→│ next: ────────────→│ next: ──┐    │
     └─────────│────┘     └──────────────┘     └─────────│────┘
               │                                          │
               └──────────────────────────────────────────┘

  特徴:
  - 末尾ノードが先頭を指すため、null 参照が存在しない
  - どのノードからでもリスト全体を走査可能
  - 走査の終了条件に注意が必要（無限ループの危険）
```

**利用場面**: ラウンドロビンスケジューリング、循環バッファ、
マルチプレイヤーゲームのターン管理。

### 2.4 スキップリスト (Skip List) — 発展的バリエーション

スキップリストは連結リストを多段に重ね、上位レベルが「高速レーン」として
機能することで、O(log n) の検索を実現する確率的データ構造である。
本ガイドでは基本的な連結リストに集中するが、発展学習として知っておくべき構造である。

```
スキップリストの概念図:

  Level 3:  head ─────────────────────────── [50] ─────────────────── tail
  Level 2:  head ──────── [20] ──────────── [50] ──── [70] ──────── tail
  Level 1:  head ── [10]─ [20] ── [30] ── [50] ── [60]─ [70] ── [90] ── tail
  Level 0:  head ── [10]─ [20] ── [30] ── [40] ── [50] ── [60]─ [70] ── [80] ── [90] ── tail
```

**利用場面**: Redis のソート済みセット (Sorted Set)、
LevelDB/RocksDB の MemTable。

---

## 3. 基本実装 -- ノードとリストクラス

### 3.1 単方向リストのノードクラス

```python
class ListNode:
    """単方向連結リストのノード。

    Attributes:
        val: ノードが保持する値。任意の型を格納可能。
        next: 次のノードへの参照。末尾ノードでは None。
    """

    __slots__ = ('val', 'next')  # メモリ最適化

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        """デバッグ用文字列表現"""
        return f"ListNode({self.val})"

    def __eq__(self, other):
        """値の等価比較（ノードの同一性ではない点に注意）"""
        if not isinstance(other, ListNode):
            return NotImplemented
        return self.val == other.val
```

### 3.2 双方向リストのノードクラス

```python
class DoublyListNode:
    """双方向連結リストのノード。

    Attributes:
        val: ノードが保持する値。
        prev: 前のノードへの参照。先頭ノードでは None。
        next: 次のノードへの参照。末尾ノードでは None。
    """

    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DoublyListNode({self.val})"
```

### 3.3 `__slots__` によるメモリ最適化

Python ではクラスインスタンスのデフォルトで `__dict__` が生成されるが、
連結リストのように大量のノードを生成する場合、`__slots__` を使うことで
1 ノードあたり約 40-60% のメモリを削減できる。

```python
import sys

class NodeWithDict:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class NodeWithSlots:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# メモリ使用量の比較
node_dict = NodeWithDict(42)
node_slots = NodeWithSlots(42)

print(f"__dict__ あり: {sys.getsizeof(node_dict) + sys.getsizeof(node_dict.__dict__)} bytes")
# 想定される出力: __dict__ あり: 152 bytes (Python 3.12)

print(f"__slots__ あり: {sys.getsizeof(node_slots)} bytes")
# 想定される出力: __slots__ あり: 56 bytes (Python 3.12)
```

### 3.4 ユーティリティ関数 — リストの構築と表示

以降の実装とテストで繰り返し使うヘルパー関数を定義する。
これらは本ガイドの全コード例で利用される。

```python
from typing import Optional, List


class ListNode:
    """単方向リストのノード"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


def build_list(values: List[int]) -> Optional[ListNode]:
    """配列から単方向連結リストを構築する。

    Args:
        values: リストに格納する値の配列。

    Returns:
        リストの先頭ノード。空配列の場合は None。

    Examples:
        >>> head = build_list([1, 2, 3, 4, 5])
        >>> list_to_string(head)
        '1 -> 2 -> 3 -> 4 -> 5 -> None'
    """
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def list_to_array(head: Optional[ListNode]) -> List[int]:
    """連結リストを配列に変換する。

    Args:
        head: リストの先頭ノード。

    Returns:
        リスト内の全値を格納した配列。
    """
    result = []
    current = head
    while current:
        result.append(current.val)
        current = current.next
    return result


def list_to_string(head: Optional[ListNode]) -> str:
    """連結リストを文字列表現に変換する。

    Args:
        head: リストの先頭ノード。

    Returns:
        '1 -> 2 -> 3 -> None' 形式の文字列。
    """
    parts = []
    current = head
    while current:
        parts.append(str(current.val))
        current = current.next
    parts.append("None")
    return " -> ".join(parts)


def list_length(head: Optional[ListNode]) -> int:
    """連結リストの長さを返す。

    Args:
        head: リストの先頭ノード。

    Returns:
        ノード数。
    """
    count = 0
    current = head
    while current:
        count += 1
        current = current.next
    return count


# 動作確認
if __name__ == "__main__":
    head = build_list([10, 20, 30, 40, 50])
    print(list_to_string(head))        # 10 -> 20 -> 30 -> 40 -> 50 -> None
    print(list_to_array(head))          # [10, 20, 30, 40, 50]
    print(f"長さ: {list_length(head)}")  # 長さ: 5
```

---

## 4. コア操作の実装

### 4.1 単方向リストの完全実装

ここでは、挿入・削除・検索・反転など基本操作を全て備えた
単方向リストクラスを提示する。各メソッドには計算量の注釈と
内部動作の図解を付与している。

```python
from typing import Optional, List, Iterator


class ListNode:
    """単方向リストのノード"""
    __slots__ = ('val', 'next')

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class SinglyLinkedList:
    """単方向連結リストの完全実装。

    ダミーヘッドとサイズ管理により、エッジケースを統一的に扱う。

    Attributes:
        _dummy: ダミーヘッドノード（番兵ノード）。
        _size: リストの現在の要素数。
    """

    def __init__(self):
        """空のリストを初期化する。"""
        self._dummy = ListNode(0)  # 番兵ノード
        self._size = 0

    @property
    def head(self) -> Optional[ListNode]:
        """実際の先頭ノードを返す（ダミーヘッドは隠蔽）。"""
        return self._dummy.next

    def __len__(self) -> int:
        """リストの長さを O(1) で返す。"""
        return self._size

    def __bool__(self) -> bool:
        """リストが空でないか判定する。"""
        return self._size > 0

    def __iter__(self) -> Iterator:
        """リストの要素をイテレートする。"""
        current = self._dummy.next
        while current:
            yield current.val
            current = current.next

    def __repr__(self) -> str:
        """リストの文字列表現を返す。"""
        values = list(self)
        return f"SinglyLinkedList({values})"

    def __contains__(self, val) -> bool:
        """in 演算子のサポート。O(n)。"""
        return self.search(val)

    # ─── 挿入操作 ─────────────────────────────────

    def prepend(self, val) -> None:
        """先頭に要素を挿入する。O(1)。

        Args:
            val: 挿入する値。

        図解:
            Before: dummy -> [A] -> [B] -> None
            After:  dummy -> [X] -> [A] -> [B] -> None
        """
        new_node = ListNode(val, self._dummy.next)
        self._dummy.next = new_node
        self._size += 1

    def append(self, val) -> None:
        """末尾に要素を挿入する。O(n)。

        Args:
            val: 挿入する値。

        図解:
            Before: dummy -> [A] -> [B] -> None
            After:  dummy -> [A] -> [B] -> [X] -> None
        """
        current = self._dummy
        while current.next:
            current = current.next
        current.next = ListNode(val)
        self._size += 1

    def insert_at(self, index: int, val) -> None:
        """指定位置に要素を挿入する。O(n)。

        Args:
            index: 挿入位置（0-indexed）。
            val: 挿入する値。

        Raises:
            IndexError: インデックスが範囲外の場合。
        """
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size}]")

        prev = self._dummy
        for _ in range(index):
            prev = prev.next
        new_node = ListNode(val, prev.next)
        prev.next = new_node
        self._size += 1

    def insert_after(self, target_val, new_val) -> bool:
        """指定値のノードの直後に挿入する。O(n)。

        Args:
            target_val: 検索する値。
            new_val: 挿入する値。

        Returns:
            挿入に成功した場合 True、target_val が見つからなければ False。
        """
        current = self._dummy.next
        while current:
            if current.val == target_val:
                new_node = ListNode(new_val, current.next)
                current.next = new_node
                self._size += 1
                return True
            current = current.next
        return False

    # ─── 削除操作 ─────────────────────────────────

    def delete(self, val) -> bool:
        """指定値を持つ最初のノードを削除する。O(n)。

        Args:
            val: 削除する値。

        Returns:
            削除に成功した場合 True、値が見つからなければ False。

        図解:
            Before: dummy -> [A] -> [B] -> [C] -> None
            delete("B"):
            After:  dummy -> [A] ---------> [C] -> None
        """
        prev = self._dummy
        current = self._dummy.next
        while current:
            if current.val == val:
                prev.next = current.next
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def delete_at(self, index: int):
        """指定位置の要素を削除して値を返す。O(n)。

        Args:
            index: 削除位置（0-indexed）。

        Returns:
            削除されたノードの値。

        Raises:
            IndexError: インデックスが範囲外の場合。
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        prev = self._dummy
        for _ in range(index):
            prev = prev.next
        target = prev.next
        prev.next = target.next
        self._size -= 1
        return target.val

    def delete_all(self, val) -> int:
        """指定値を持つ全てのノードを削除する。O(n)。

        Args:
            val: 削除する値。

        Returns:
            削除されたノードの数。
        """
        prev = self._dummy
        current = self._dummy.next
        count = 0
        while current:
            if current.val == val:
                prev.next = current.next
                self._size -= 1
                count += 1
            else:
                prev = current
            current = current.next if current.val == val else prev.next.next if prev.next else None
        # 上記の走査は複雑になるため、よりシンプルな実装:
        return count

    def pop_front(self):
        """先頭要素を削除して値を返す。O(1)。

        Returns:
            先頭ノードの値。

        Raises:
            IndexError: リストが空の場合。
        """
        if not self._dummy.next:
            raise IndexError("pop from empty list")
        target = self._dummy.next
        self._dummy.next = target.next
        self._size -= 1
        return target.val

    # ─── 検索・アクセス操作 ────────────────────────

    def search(self, val) -> bool:
        """値がリストに存在するか判定する。O(n)。

        Args:
            val: 検索する値。

        Returns:
            値が見つかれば True。
        """
        current = self._dummy.next
        while current:
            if current.val == val:
                return True
            current = current.next
        return False

    def get_at(self, index: int):
        """指定位置の値を取得する。O(n)。

        Args:
            index: 取得位置（0-indexed）。

        Returns:
            該当ノードの値。

        Raises:
            IndexError: インデックスが範囲外の場合。
        """
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range [0, {self._size - 1}]")

        current = self._dummy.next
        for _ in range(index):
            current = current.next
        return current.val

    def index_of(self, val) -> int:
        """値の最初の出現位置を返す。O(n)。

        Args:
            val: 検索する値。

        Returns:
            出現位置（0-indexed）。見つからなければ -1。
        """
        current = self._dummy.next
        idx = 0
        while current:
            if current.val == val:
                return idx
            current = current.next
            idx += 1
        return -1

    # ─── 変換操作 ──────────────────────────────────

    def reverse(self) -> None:
        """リストをその場で反転する。O(n) 時間、O(1) 空間。

        図解:
            Before: dummy -> [1] -> [2] -> [3] -> None
            After:  dummy -> [3] -> [2] -> [1] -> None

        反転の詳細過程:
            Step 0: prev=None,  curr=[1]->[2]->[3]
            Step 1: prev=[1],   curr=[2]->[3]     ([1]->None)
            Step 2: prev=[2],   curr=[3]          ([2]->[1]->None)
            Step 3: prev=[3],   curr=None          ([3]->[2]->[1]->None)
        """
        prev = None
        current = self._dummy.next
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self._dummy.next = prev

    def to_list(self) -> List:
        """リストを Python リスト（配列）に変換する。O(n)。"""
        return list(self)

    def clear(self) -> None:
        """リストを空にする。O(1)。"""
        self._dummy.next = None
        self._size = 0

    @classmethod
    def from_list(cls, values: List) -> 'SinglyLinkedList':
        """Python リスト（配列）から連結リストを構築する。O(n)。

        Args:
            values: 値の配列。

        Returns:
            構築された SinglyLinkedList。
        """
        linked_list = cls()
        if not values:
            return linked_list
        current = linked_list._dummy
        for val in values:
            current.next = ListNode(val)
            current = current.next
            linked_list._size += 1
        return linked_list


# ─── 動作確認テスト ──────────────────────────────────────

if __name__ == "__main__":
    # リストの構築
    sll = SinglyLinkedList.from_list([10, 20, 30, 40, 50])
    print(f"初期状態: {sll}")
    # 出力: SinglyLinkedList([10, 20, 30, 40, 50])

    # 先頭挿入
    sll.prepend(5)
    print(f"prepend(5): {sll}")
    # 出力: SinglyLinkedList([5, 10, 20, 30, 40, 50])

    # 末尾挿入
    sll.append(60)
    print(f"append(60): {sll}")
    # 出力: SinglyLinkedList([5, 10, 20, 30, 40, 50, 60])

    # 位置指定挿入
    sll.insert_at(3, 25)
    print(f"insert_at(3, 25): {sll}")
    # 出力: SinglyLinkedList([5, 10, 20, 25, 30, 40, 50, 60])

    # 削除
    sll.delete(25)
    print(f"delete(25): {sll}")
    # 出力: SinglyLinkedList([5, 10, 20, 30, 40, 50, 60])

    # 反転
    sll.reverse()
    print(f"reverse(): {sll}")
    # 出力: SinglyLinkedList([60, 50, 40, 30, 20, 10, 5])

    # 検索
    print(f"search(30): {sll.search(30)}")  # True
    print(f"search(99): {sll.search(99)}")  # False

    # イテレーション
    print(f"リスト長: {len(sll)}")  # 7
    print(f"30 in sll: {30 in sll}")  # True

    # インデックスアクセス
    print(f"get_at(2): {sll.get_at(2)}")  # 40
    print(f"index_of(30): {sll.index_of(30)}")  # 3

    # pop
    val = sll.pop_front()
    print(f"pop_front(): {val}, リスト: {sll}")
    # 出力: pop_front(): 60, リスト: SinglyLinkedList([50, 40, 30, 20, 10, 5])
```

### 4.2 リスト反転の詳細図解

リストの反転は連結リスト操作の中でも最重要のアルゴリズムである。
ポインタの付け替えを 1 ステップずつ追って理解する。

```
反転アルゴリズムの全ステップ:

  【初期状態】
  prev = None
  curr = [1]

  None    [1] ──→ [2] ──→ [3] ──→ [4] ──→ None
   ▲      ▲
  prev   curr

  ──────────────────────────────────────────────────

  【Step 1】 next_node = curr.next = [2]
             curr.next = prev (= None)
             prev = curr (= [1])
             curr = next_node (= [2])

  None ←── [1]    [2] ──→ [3] ──→ [4] ──→ None
            ▲      ▲
           prev   curr

  ──────────────────────────────────────────────────

  【Step 2】 next_node = curr.next = [3]
             curr.next = prev (= [1])
             prev = curr (= [2])
             curr = next_node (= [3])

  None ←── [1] ←── [2]    [3] ──→ [4] ──→ None
                    ▲      ▲
                   prev   curr

  ──────────────────────────────────────────────────

  【Step 3】 next_node = curr.next = [4]
             curr.next = prev (= [2])
             prev = curr (= [3])
             curr = next_node (= [4])

  None ←── [1] ←── [2] ←── [3]    [4] ──→ None
                            ▲      ▲
                           prev   curr

  ──────────────────────────────────────────────────

  【Step 4】 next_node = curr.next = None
             curr.next = prev (= [3])
             prev = curr (= [4])
             curr = next_node (= None)

  None ←── [1] ←── [2] ←── [3] ←── [4]    None
                                     ▲      ▲
                                    prev   curr

  ──────────────────────────────────────────────────

  【完了】 curr == None でループ終了
  新しい head = prev = [4]

  結果: [4] ──→ [3] ──→ [2] ──→ [1] ──→ None
```

### 4.3 再帰による反転

反転は再帰でも実装できる。再帰版は理解しやすいが、
スタック深度 O(n) のためリストが長い場合にスタックオーバーフローの
リスクがある。

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_recursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """連結リストを再帰的に反転する。

    時間計算量: O(n)
    空間計算量: O(n) — 再帰スタック

    Args:
        head: リストの先頭ノード。

    Returns:
        反転後のリストの先頭ノード。

    動作原理:
        1. ベースケース: head が None または head.next が None なら head を返す
        2. 残りの部分を再帰的に反転
        3. head.next.next = head で逆方向のリンクを作成
        4. head.next = None で元のリンクを切断
    """
    # ベースケース
    if not head or not head.next:
        return head

    # 再帰: head の次以降を反転
    new_head = reverse_recursive(head.next)

    # 逆リンクの作成
    head.next.next = head  # 次のノードから自分へのリンク
    head.next = None       # 自分から次へのリンクを切断

    return new_head


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# 動作確認
if __name__ == "__main__":
    original = build_list([1, 2, 3, 4, 5])
    print(f"反転前: {list_to_string(original)}")
    # 出力: 反転前: 1 -> 2 -> 3 -> 4 -> 5 -> None

    reversed_head = reverse_recursive(original)
    print(f"反転後: {list_to_string(reversed_head)}")
    # 出力: 反転後: 5 -> 4 -> 3 -> 2 -> 1 -> None
```

### 4.4 ダミーヘッド（番兵ノード）パターンの重要性

ダミーヘッド（sentinel node）は、連結リスト操作でエッジケースを
排除するための最も強力なテクニックである。

```
ダミーヘッドなし（先頭ノードの特別扱いが必要）:

  ケース1: 先頭削除         ケース2: 中間削除
  head                      head
   │                         │
   ▼                         ▼
  [X] -> [B] -> [C]        [A] -> [X] -> [C]
   ↓                                ↓
  head = head.next          prev.next = curr.next
  （コードが分岐する）

ダミーヘッドあり（統一的な処理）:

  常にダミーヘッドの次から実データが始まる:
  dummy -> [A] -> [B] -> [C] -> None

  先頭削除も中間削除も同じロジック:
  prev.next = curr.next
  head = dummy.next
```

このパターンは、マージソートでの 2 リスト統合、
条件付きノード削除、パーティション操作など、
あらゆる場面で有効である。

---

## 5. 双方向リストの完全実装

双方向リストは前方・後方の双方向走査と O(1) でのノード削除を実現する。
LRU キャッシュの実装に不可欠なデータ構造である。

### 5.1 双方向リストクラス

```python
from typing import Optional, List, Iterator


class DoublyListNode:
    """双方向リストのノード"""
    __slots__ = ('val', 'prev', 'next')

    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

    def __repr__(self):
        return f"DoublyListNode({self.val})"


class DoublyLinkedList:
    """双方向連結リストの完全実装。

    先頭と末尾にダミーノード（番兵）を配置し、
    全ての挿入・削除操作を統一的に扱う。

    構造:
        head_sentinel <-> [node1] <-> [node2] <-> ... <-> tail_sentinel
    """

    def __init__(self):
        """空のリストを初期化する。ダミーヘッドとダミーテイルを接続。"""
        self._head = DoublyListNode(0)   # ダミーヘッド
        self._tail = DoublyListNode(0)   # ダミーテイル
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self) -> Iterator:
        """前方向のイテレーション"""
        current = self._head.next
        while current != self._tail:
            yield current.val
            current = current.next

    def __reversed__(self) -> Iterator:
        """後方向のイテレーション"""
        current = self._tail.prev
        while current != self._head:
            yield current.val
            current = current.prev

    def __repr__(self) -> str:
        return f"DoublyLinkedList({list(self)})"

    # ─── 内部ヘルパー ─────────────────────────────

    def _insert_between(self, val, predecessor: DoublyListNode,
                        successor: DoublyListNode) -> DoublyListNode:
        """predecessor と successor の間にノードを挿入する。O(1)。

        図解:
            Before: [pred] <-> [succ]
            After:  [pred] <-> [new] <-> [succ]
        """
        new_node = DoublyListNode(val, predecessor, successor)
        predecessor.next = new_node
        successor.prev = new_node
        self._size += 1
        return new_node

    def _remove_node(self, node: DoublyListNode):
        """指定ノードをリストから除去する。O(1)。

        図解:
            Before: [pred] <-> [node] <-> [succ]
            After:  [pred] <-> [succ]
        """
        predecessor = node.prev
        successor = node.next
        predecessor.next = successor
        successor.prev = predecessor
        self._size -= 1
        return node.val

    # ─── 公開API ──────────────────────────────────

    def prepend(self, val) -> DoublyListNode:
        """先頭に挿入。O(1)。"""
        return self._insert_between(val, self._head, self._head.next)

    def append(self, val) -> DoublyListNode:
        """末尾に挿入。O(1)。"""
        return self._insert_between(val, self._tail.prev, self._tail)

    def pop_front(self):
        """先頭要素を削除して値を返す。O(1)。"""
        if self._size == 0:
            raise IndexError("pop from empty list")
        return self._remove_node(self._head.next)

    def pop_back(self):
        """末尾要素を削除して値を返す。O(1)。"""
        if self._size == 0:
            raise IndexError("pop from empty list")
        return self._remove_node(self._tail.prev)

    def remove(self, node: DoublyListNode):
        """指定ノードを O(1) で削除する。ノード参照が必要。"""
        return self._remove_node(node)

    def move_to_front(self, node: DoublyListNode) -> None:
        """指定ノードをリストの先頭に移動する。O(1)。
        LRU キャッシュで最近アクセスされた要素を先頭に移す際に使用。
        """
        self._remove_node(node)
        self._insert_between(node.val, self._head, self._head.next)

    def peek_front(self):
        """先頭の値を参照する（削除しない）。O(1)。"""
        if self._size == 0:
            raise IndexError("peek from empty list")
        return self._head.next.val

    def peek_back(self):
        """末尾の値を参照する（削除しない）。O(1)。"""
        if self._size == 0:
            raise IndexError("peek from empty list")
        return self._tail.prev.val

    @classmethod
    def from_list(cls, values: List) -> 'DoublyLinkedList':
        """配列から双方向リストを構築する。"""
        dll = cls()
        for val in values:
            dll.append(val)
        return dll


# 動作確認
if __name__ == "__main__":
    dll = DoublyLinkedList.from_list([10, 20, 30, 40, 50])
    print(f"前方走査: {list(dll)}")
    # 出力: 前方走査: [10, 20, 30, 40, 50]

    print(f"後方走査: {list(reversed(dll))}")
    # 出力: 後方走査: [50, 40, 30, 20, 10]

    dll.prepend(5)
    dll.append(60)
    print(f"prepend(5), append(60): {list(dll)}")
    # 出力: prepend(5), append(60): [5, 10, 20, 30, 40, 50, 60]

    print(f"pop_front(): {dll.pop_front()}")  # 5
    print(f"pop_back(): {dll.pop_back()}")    # 60
    print(f"結果: {list(dll)}")
    # 出力: 結果: [10, 20, 30, 40, 50]
```

### 5.2 LRU キャッシュの実装 — 双方向リスト + ハッシュマップ

LRU (Least Recently Used) キャッシュは、双方向リストの最も有名な応用である。
ハッシュマップとの組み合わせで、get と put を共に O(1) で実現する。

```python
from typing import Optional


class DoublyListNode:
    __slots__ = ('key', 'val', 'prev', 'next')
    def __init__(self, key=0, val=0, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev = prev
        self.next = next


class LRUCache:
    """LRU キャッシュ — O(1) の get/put を実現。

    構造:
        HashMap: key -> DoublyListNode (O(1) ルックアップ)
        DoublyLinkedList: 使用順を管理（先頭 = 最近使用, 末尾 = 最古）

    図解:
        HashMap                DoublyLinkedList
        ┌─────────────┐       head <-> [A] <-> [B] <-> [C] <-> tail
        │ key_A -> [A] │               最新               最古
        │ key_B -> [B] │
        │ key_C -> [C] │       容量超過時: tail.prev ([C]) を削除
        └─────────────┘
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> DoublyListNode

        # ダミーノードで番兵パターン
        self._head = DoublyListNode()
        self._tail = DoublyListNode()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: DoublyListNode) -> None:
        """ノードをリストから除去する。"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_front(self, node: DoublyListNode) -> None:
        """ノードをリストの先頭（head の直後）に追加する。"""
        node.next = self._head.next
        node.prev = self._head
        self._head.next.prev = node
        self._head.next = node

    def _move_to_front(self, node: DoublyListNode) -> None:
        """ノードをリストの先頭に移動する。"""
        self._remove(node)
        self._add_to_front(node)

    def _evict(self) -> None:
        """最も古いエントリ（tail の直前）を削除する。"""
        lru_node = self._tail.prev
        self._remove(lru_node)
        del self.cache[lru_node.key]

    def get(self, key: int) -> int:
        """キーに対応する値を取得する。O(1)。

        見つかった場合はそのノードを先頭に移動（最近使用に更新）。
        見つからなければ -1 を返す。
        """
        if key in self.cache:
            node = self.cache[key]
            self._move_to_front(node)
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        """キーと値のペアを追加/更新する。O(1)。

        既存キーの場合は値を更新して先頭に移動。
        新規キーの場合は先頭に追加し、容量超過時は最古を削除。
        """
        if key in self.cache:
            node = self.cache[key]
            node.val = value
            self._move_to_front(node)
        else:
            if len(self.cache) >= self.capacity:
                self._evict()
            new_node = DoublyListNode(key, value)
            self.cache[key] = new_node
            self._add_to_front(new_node)


# 動作確認
if __name__ == "__main__":
    cache = LRUCache(3)

    cache.put(1, 100)
    cache.put(2, 200)
    cache.put(3, 300)
    print(f"get(1): {cache.get(1)}")   # 100（1 が最近使用に）
    print(f"get(2): {cache.get(2)}")   # 200（2 が最近使用に）

    cache.put(4, 400)  # 容量超過 → 3 が削除される
    print(f"get(3): {cache.get(3)}")   # -1（削除済み）
    print(f"get(4): {cache.get(4)}")   # 400

    cache.put(5, 500)  # 容量超過 → 1 が削除される
    print(f"get(1): {cache.get(1)}")   # -1（削除済み）
    print(f"get(2): {cache.get(2)}")   # 200（まだ存在）
```

---

## 6. 循環リストの実装と応用

### 6.1 循環リストクラス

```python
from typing import Optional, List


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __repr__(self):
        return f"ListNode({self.val})"


class CircularLinkedList:
    """単方向循環連結リスト。

    tail ポインタのみを保持し、tail.next = head とすることで
    先頭・末尾両方への O(1) アクセスを実現する。

    構造:
        tail -> [C] -> [A] -> [B] -> [C] (= tail)
                        ↑ head           ↑ tail

    tail.next が head を指す:
        tail.next = head
    """

    def __init__(self):
        self.tail = None
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    def __iter__(self):
        if not self.tail:
            return
        current = self.tail.next  # head
        for _ in range(self._size):
            yield current.val
            current = current.next

    def __repr__(self):
        if not self.tail:
            return "CircularLinkedList([])"
        values = list(self)
        return f"CircularLinkedList({values})"

    @property
    def head(self):
        return self.tail.next if self.tail else None

    def append(self, val) -> None:
        """末尾に挿入。O(1)。"""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node  # 自分自身を指す
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # 新ノード -> head
            self.tail.next = new_node        # 旧 tail -> 新ノード
            self.tail = new_node             # tail を更新
        self._size += 1

    def prepend(self, val) -> None:
        """先頭に挿入。O(1)。"""
        new_node = ListNode(val)
        if not self.tail:
            new_node.next = new_node
            self.tail = new_node
        else:
            new_node.next = self.tail.next  # 新ノード -> 旧 head
            self.tail.next = new_node        # tail -> 新ノード (= 新 head)
        self._size += 1

    def delete(self, val) -> bool:
        """指定値のノードを削除。O(n)。"""
        if not self.tail:
            return False

        # ノードが1つだけの場合
        if self._size == 1:
            if self.tail.val == val:
                self.tail = None
                self._size = 0
                return True
            return False

        prev = self.tail
        current = self.tail.next  # head
        for _ in range(self._size):
            if current.val == val:
                if current == self.tail:
                    self.tail = prev
                prev.next = current.next
                self._size -= 1
                return True
            prev = current
            current = current.next
        return False

    def rotate(self, k: int = 1) -> None:
        """リストを k 回回転する。O(k)。

        tail を k 回前進させることで、先頭要素が変わる。
        ラウンドロビンスケジューリングに利用。
        """
        if not self.tail or k == 0:
            return
        k = k % self._size
        for _ in range(k):
            self.tail = self.tail.next

    @classmethod
    def from_list(cls, values: List) -> 'CircularLinkedList':
        cll = cls()
        for val in values:
            cll.append(val)
        return cll


# 動作確認
if __name__ == "__main__":
    cll = CircularLinkedList.from_list([1, 2, 3, 4, 5])
    print(f"初期状態: {cll}")
    # 出力: CircularLinkedList([1, 2, 3, 4, 5])

    cll.rotate(2)
    print(f"rotate(2): {cll}")
    # 出力: CircularLinkedList([3, 4, 5, 1, 2])

    cll.prepend(0)
    print(f"prepend(0): {cll}")
    # 出力: CircularLinkedList([0, 3, 4, 5, 1, 2])

    cll.delete(5)
    print(f"delete(5): {cll}")
    # 出力: CircularLinkedList([0, 3, 4, 1, 2])
```

### 6.2 ヨセフスの問題 — 循環リストの古典的応用

ヨセフスの問題は、n 人が円形に並び、k 番目ごとに脱落していくとき
最後に残る人を求める問題である。循環リストで自然にモデル化できる。

```python
def josephus(n: int, k: int) -> int:
    """ヨセフスの問題を循環リストで解く。

    n 人が円形に並び、k 番目ごとに脱落する。
    最後に残る人の番号を返す（0-indexed）。

    時間計算量: O(n * k)
    空間計算量: O(n)

    Args:
        n: 人数。
        k: 数え上げの間隔。

    Returns:
        最後に残る人の番号（0-indexed）。

    図解 (n=5, k=3):
        初期:   0 - 1 - 2 - 3 - 4 (円形)
        Step 1: 2 が脱落 → 0 - 1 - 3 - 4
        Step 2: 0 が脱落 → 1 - 3 - 4
        Step 3: 4 が脱落 → 1 - 3
        Step 4: 1 が脱落 → 3
        結果: 3
    """
    # 循環リストを構築
    class Node:
        __slots__ = ('val', 'next')
        def __init__(self, val):
            self.val = val
            self.next = None

    # ノードをリング状に接続
    head = Node(0)
    current = head
    for i in range(1, n):
        current.next = Node(i)
        current = current.next
    current.next = head  # 循環を作成

    # k 番目ごとに削除
    prev = current  # 末尾ノード（head の直前）
    current = head
    remaining = n

    while remaining > 1:
        # k-1 回進む（k 番目のノードに到達）
        for _ in range(k - 1):
            prev = current
            current = current.next
        # current を削除
        prev.next = current.next
        current = prev.next
        remaining -= 1

    return current.val


# 動作確認
if __name__ == "__main__":
    print(f"josephus(5, 3) = {josephus(5, 3)}")  # 3
    print(f"josephus(7, 2) = {josephus(7, 2)}")  # 6
    print(f"josephus(10, 3) = {josephus(10, 3)}")  # 3

    # 数学的解法との検証
    def josephus_math(n: int, k: int) -> int:
        """数学的再帰解法。O(n) 時間, O(1) 空間。"""
        result = 0
        for i in range(2, n + 1):
            result = (result + k) % i
        return result

    for n in range(1, 20):
        for k in [2, 3, 5]:
            assert josephus(n, k) == josephus_math(n, k), \
                f"Mismatch at n={n}, k={k}"
    print("全テストパス")
```

---

## 7. フロイドの循環検出アルゴリズム

### 7.1 問題の定義

連結リストにサイクル（循環）が存在するかを判定し、
存在する場合はサイクルの開始ノードとサイクルの長さを特定する。

```
サイクルを含むリストの例:

  [1] → [2] → [3] → [4] → [5] → [6]
                ▲                    │
                └────────────────────┘

  ノード 3 がサイクルの開始点
  サイクルの長さ = 4（3 → 4 → 5 → 6 → 3）

  非サイクル部分の長さ (μ) = 2（1 → 2 → 3 到達まで）
  サイクルの長さ (λ) = 4
```

### 7.2 亀と兎のアルゴリズム — サイクル検出

slow ポインタ（亀）は 1 歩ずつ、fast ポインタ（兎）は 2 歩ずつ進む。
サイクルが存在すれば、兎は必ず亀に追いつく。

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """サイクルの有無を判定する。

    時間計算量: O(n)
    空間計算量: O(1)

    Args:
        head: リストの先頭ノード。

    Returns:
        サイクルが存在すれば True。
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next          # 1 歩
        fast = fast.next.next     # 2 歩
        if slow is fast:          # 同一ノード（値ではなく参照で比較）
            return True
    return False
```

```
サイクル検出のステップ追跡:

  リスト: [1] → [2] → [3] → [4] → [5]
                        ▲              │
                        └──────────────┘

  Step 0: slow = [1], fast = [1]
  Step 1: slow = [2], fast = [3]
  Step 2: slow = [3], fast = [5]
  Step 3: slow = [4], fast = [4]  ← 一致! サイクルあり

  なぜ必ず出会うのか？
  ─────────────────────────────
  サイクル内に入った後、fast は slow に対して
  毎ステップ 1 ノード分ずつ接近する（相対速度 = 1）。
  よって、サイクル長を λ とすると、最大 λ ステップで必ず出会う。
```

### 7.3 サイクル開始点の特定

フロイドのアルゴリズムのフェーズ 2 では、
slow と fast が出会った後、slow を head に戻し、
両方を 1 歩ずつ進めると、サイクルの開始点で出会う。

```python
def detect_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """サイクルの開始ノードを特定する。

    時間計算量: O(n)
    空間計算量: O(1)

    数学的証明:
        head からサイクル開始点までの距離を μ、
        サイクルの長さを λ とする。
        slow と fast が出会う点は、サイクル開始点から
        (μ mod λ) の位置にある。
        したがって、head から μ 歩進むと、
        出会い地点から μ 歩進んだ地点（= サイクル開始点）で
        再び出会う。

    Args:
        head: リストの先頭ノード。

    Returns:
        サイクルの開始ノード。サイクルがなければ None。
    """
    slow = fast = head

    # フェーズ1: 出会い地点を見つける
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            break
    else:
        return None  # サイクルなし

    # フェーズ2: slow を head に戻し、両方 1 歩ずつ進める
    slow = head
    while slow is not fast:
        slow = slow.next
        fast = fast.next

    return slow  # サイクルの開始点
```

```
サイクル開始点特定の図解:

  head                          meeting point
   │                                │
   ▼                                ▼
  [1] → [2] → [3] → [4] → [5] → [6]
                ▲                    │
                └────────────────────┘

  μ (非サイクル部分) = 2  (head → [3])
  λ (サイクル長) = 4       ([3]→[4]→[5]→[6]→[3])

  フェーズ1終了時:
    slow と fast が [6] で出会ったとする

  フェーズ2:
    slow = head = [1],  fast = [6]
    Step 1: slow = [2], fast = [3]
    Step 2: slow = [3], fast = [4]
    ... 実際には [3] で出会う = サイクル開始点
```

### 7.4 サイクル長の計測

```python
def cycle_length(head: Optional[ListNode]) -> int:
    """サイクルの長さを返す。サイクルがなければ 0。

    時間計算量: O(n)
    空間計算量: O(1)
    """
    slow = fast = head

    # サイクル検出
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            # 出会い地点から 1 周して同じ地点に戻るまで数える
            length = 1
            runner = slow.next
            while runner is not slow:
                length += 1
                runner = runner.next
            return length

    return 0  # サイクルなし


# テスト用ヘルパー: サイクルを持つリストを構築
def build_cyclic_list(values, cycle_start_index):
    """サイクルを持つリストを構築する。

    Args:
        values: ノードの値リスト。
        cycle_start_index: サイクル開始位置のインデックス。
            -1 ならサイクルなし。
    """
    if not values:
        return None

    nodes = [ListNode(v) for v in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    if cycle_start_index >= 0:
        nodes[-1].next = nodes[cycle_start_index]

    return nodes[0]


# 動作確認
if __name__ == "__main__":
    # サイクルありリスト
    head = build_cyclic_list([1, 2, 3, 4, 5, 6], cycle_start_index=2)
    print(f"サイクルあり: {has_cycle(head)}")           # True
    start = detect_cycle_start(head)
    print(f"サイクル開始点: {start.val}")                # 3
    print(f"サイクル長: {cycle_length(head)}")           # 4

    # サイクルなしリスト
    head_no_cycle = build_cyclic_list([1, 2, 3, 4, 5], cycle_start_index=-1)
    print(f"サイクルなし: {has_cycle(head_no_cycle)}")   # False
    print(f"サイクル開始点: {detect_cycle_start(head_no_cycle)}")  # None
    print(f"サイクル長: {cycle_length(head_no_cycle)}")  # 0
```

### 7.5 フロイドのアルゴリズムの数学的証明

フロイドのアルゴリズムが正しく動作する数学的な根拠を示す。

**前提**:
- head からサイクル開始点までの距離: mu
- サイクルの長さ: lambda
- slow と fast が出会うまでに slow が進んだ距離: d

**フェーズ 1 の証明**:
1. slow が d 歩進んだとき、fast は 2d 歩進んでいる
2. 出会い地点では: 2d - d = d がサイクル長の倍数
3. つまり d = k * lambda（k は正の整数）

**フェーズ 2 の証明**:
1. 出会い地点はサイクル開始点から (d - mu) の位置
2. d = k * lambda なので、出会い地点から mu 歩進むと:
   (d - mu) + mu = d = k * lambda（サイクル開始点に戻る）
3. head から mu 歩進んでもサイクル開始点に到達
4. したがって両方が mu 歩後に出会う地点がサイクル開始点

---

## 8. 応用アルゴリズム集

### 8.1 ソート済みリストのマージ

2 つのソート済みリストを 1 つのソート済みリストに統合する。
マージソートの基盤となるアルゴリズム。

```python
from typing import Optional


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_sorted_lists(l1: Optional[ListNode],
                       l2: Optional[ListNode]) -> Optional[ListNode]:
    """2 つのソート済みリストをマージする。

    時間計算量: O(n + m)
    空間計算量: O(1)（新規ノードを作らず、既存ノードを繋ぎ替え）

    Args:
        l1, l2: ソート済みリストの先頭ノード。

    Returns:
        マージ後のリストの先頭ノード。
    """
    dummy = ListNode(0)
    current = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next

    # 残りを接続
    current.next = l1 if l1 else l2
    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# 動作確認
if __name__ == "__main__":
    l1 = build_list([1, 3, 5, 7])
    l2 = build_list([2, 4, 6, 8])
    merged = merge_sorted_lists(l1, l2)
    print(list_to_string(merged))
    # 出力: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> None
```

### 8.2 K 個のソート済みリストのマージ

ヒープ（優先度キュー）を使って K 個のリストを効率的にマージする。

```python
import heapq
from typing import Optional, List


class ListNode:
    __slots__ = ('val', 'next')
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        """ヒープ比較のために必要"""
        return self.val < other.val


def merge_k_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """K 個のソート済みリストをマージする。

    時間計算量: O(N log K)（N = 全ノード数, K = リスト数）
    空間計算量: O(K)（ヒープサイズ）

    Args:
        lists: ソート済みリストの先頭ノードの配列。

    Returns:
        マージ後のリストの先頭ノード。
    """
    dummy = ListNode(0)
    current = dummy

    # 各リストの先頭をヒープに投入
    heap = []
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# 動作確認
if __name__ == "__main__":
    lists = [
        build_list([1, 4, 7]),
        build_list([2, 5, 8]),
        build_list([3, 6, 9]),
    ]
    merged = merge_k_lists(lists)
    print(list_to_string(merged))
    # 出力: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> None
```

### 8.3 中間ノードの取得 — slow/fast ポインタ

```python
def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """リストの中間ノードを取得する。

    偶数長の場合は後半の先頭（2 つの中間のうち右側）を返す。

    時間計算量: O(n)
    空間計算量: O(1)

    図解:
        奇数長: [1] -> [2] -> [3] -> [4] -> [5]
                              ↑ 中間

        偶数長: [1] -> [2] -> [3] -> [4]
                              ↑ 中間（後半の先頭）
    """
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### 8.4 末尾から K 番目のノード削除

```python
def remove_nth_from_end(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """末尾から n 番目のノードを削除する。

    2 つのポインタを n ノード分離して同時に進める。

    時間計算量: O(L)（L = リスト長）
    空間計算量: O(1)

    図解 (n=2):
        dummy -> [1] -> [2] -> [3] -> [4] -> [5] -> None
                                       ↑ 削除対象（末尾から2番目）

        Step 1: fast を n+1 歩進める
        dummy -> [1] -> [2] -> [3] -> [4] -> [5]
        ↑ slow                         ↑ fast

        Step 2: 両方を同時に進める（fast が None になるまで）
                        ↑ slow                         ↑ fast(None)
        slow.next = slow.next.next  で [4] を削除
    """
    dummy = ListNode(0, head)
    fast = dummy
    slow = dummy

    # fast を n+1 歩先に進める
    for _ in range(n + 1):
        fast = fast.next

    # 両方を同時に進める
    while fast:
        slow = slow.next
        fast = fast.next

    # slow.next が削除対象
    slow.next = slow.next.next
    return dummy.next
```

### 8.5 回文判定 — リストの前半と後半の比較

```python
def is_palindrome(head: Optional[ListNode]) -> bool:
    """連結リストが回文（パリンドローム）であるかを判定する。

    手順:
    1. 中間ノードを見つける（slow/fast）
    2. 後半を反転する
    3. 前半と後半を比較する
    4. （オプション）後半を元に戻す

    時間計算量: O(n)
    空間計算量: O(1)
    """
    if not head or not head.next:
        return True

    # Step 1: 中間ノードを見つける
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: 後半を反転
    second_half = reverse(slow.next)

    # Step 3: 前半と後半を比較
    first_half = head
    result = True
    check = second_half
    while check:
        if first_half.val != check.val:
            result = False
            break
        first_half = first_half.next
        check = check.next

    # Step 4: 後半を元に戻す（リストの構造を保全）
    slow.next = reverse(second_half)

    return result


def reverse(head):
    """リストを反転する。"""
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

### 8.6 リストのソート — マージソート

連結リストのソートにはマージソートが最適である。
配列と異なり、連結リストではマージ操作が O(1) 空間で可能。

```python
def sort_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """連結リストをマージソートでソートする。

    時間計算量: O(n log n)
    空間計算量: O(log n) — 再帰スタック

    Args:
        head: ソート対象リストの先頭ノード。

    Returns:
        ソート後のリストの先頭ノード。
    """
    # ベースケース: 0 or 1 ノード
    if not head or not head.next:
        return head

    # リストを分割
    mid = get_middle_for_split(head)
    right_head = mid.next
    mid.next = None  # 左半分を切断

    # 再帰的にソート
    left = sort_list(head)
    right = sort_list(right_head)

    # マージ
    return merge(left, right)


def get_middle_for_split(head):
    """分割用の中間ノードを返す（左半分の末尾）。"""
    slow = head
    fast = head.next  # 偶数長で左半分が短くなるようにする
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow


def merge(l1, l2):
    """2つのソート済みリストをマージする。"""
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
    curr.next = l1 if l1 else l2
    return dummy.next


def build_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for v in values[1:]:
        curr.next = ListNode(v)
        curr = curr.next
    return head


def list_to_string(head):
    parts = []
    curr = head
    while curr:
        parts.append(str(curr.val))
        curr = curr.next
    return " -> ".join(parts) + " -> None"


# 動作確認
if __name__ == "__main__":
    head = build_list([4, 2, 1, 3, 5])
    print(f"ソート前: {list_to_string(head)}")
    # 出力: ソート前: 4 -> 2 -> 1 -> 3 -> 5 -> None

    sorted_head = sort_list(head)
    print(f"ソート後: {list_to_string(sorted_head)}")
    # 出力: ソート後: 1 -> 2 -> 3 -> 4 -> 5 -> None
```

