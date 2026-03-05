# ブロックチェーンの基礎

> ブロックチェーンは「信頼の問題」をテクノロジーで解決する分散台帳技術である。中央管理者を必要とせず、参加者全員が合意形成によって取引の正当性を担保する。本章ではハッシュチェーンの構造、合意メカニズム（コンセンサスアルゴリズム）、スマートコントラクトの仕組みを基礎から体系的に解説する。

---

## この章で学ぶこと

- [ ] ハッシュチェーンの構造と改ざん耐性の原理を理解する
- [ ] ブロックの内部構造（ヘッダ、マークル木、ナンス）を説明できる
- [ ] 主要なコンセンサスアルゴリズム（PoW, PoS, BFT 系）の違いを比較できる
- [ ] 暗号技術（ハッシュ関数、公開鍵暗号、デジタル署名）の役割を理解する
- [ ] スマートコントラクトの概念と動作原理を把握する
- [ ] DeFi、トークン標準、Layer 2 の基本概念を説明できる
- [ ] ブロックチェーンのトリレンマと現実的な課題を整理できる
- [ ] Python でブロックチェーンの基本構造を実装できる

---

## 1. ブロックチェーンとは何か

### 1.1 中央集権型システムとの対比

現代の情報システムの大半は中央集権型アーキテクチャで構築されている。銀行口座の残高は銀行のデータベースに記録され、SNS のメッセージはプラットフォーム企業のサーバーに保存される。このモデルには明確な利点がある一方で、本質的な脆弱性を内包している。

```
従来の中央集権型システム:

  ┌────────────────────────────────────────────┐
  │              中央管理サーバー                │
  │  ┌────────────────────────────────────┐    │
  │  │        Central Database            │    │
  │  │  ┌─────┐ ┌─────┐ ┌─────┐         │    │
  │  │  │Tx 1 │ │Tx 2 │ │Tx 3 │  ...    │    │
  │  │  └─────┘ └─────┘ └─────┘         │    │
  │  └────────────────────────────────────┘    │
  │          管理者が全権限を保持               │
  └──────────────┬─────────────────────────────┘
         ┌───────┼───────┐
         │       │       │
       ┌─┴─┐  ┌─┴─┐  ┌─┴─┐
       │ A │  │ B │  │ C │   ← ユーザーは管理者を信頼
       └───┘  └───┘  └───┘

  利点:
  - 処理速度が高速（単一DB への読み書き）
  - 権限管理が明確
  - 障害復旧手順が確立されている

  問題点:
  - 単一障害点（Single Point of Failure）
    → 中央 DB の障害でサービス全体が停止
  - 管理者による改ざん・検閲のリスク
    → データの完全性は管理者の誠実さに依存
  - プライバシーの集中管理
    → 大量の個人情報が一箇所に集約される
```

ブロックチェーンはこれらの問題に対する根本的なアプローチを提供する。

```
ブロックチェーン（分散台帳）:

  ┌───┐      ┌───┐      ┌───┐      ┌───┐
  │ A │──────│ B │──────│ C │──────│ D │
  └─┬─┘      └─┬─┘      └─┬─┘      └─┬─┘
    │          │          │          │
  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐  ┌─┴────┐
  │台帳  │  │台帳  │  │台帳  │  │台帳  │
  │コピー│  │コピー│  │コピー│  │コピー│
  └──────┘  └──────┘  └──────┘  └──────┘
  全参加者が同一の台帳のコピーを保持・検証

  核心的特性:
  (1) 分散性:   全参加者がデータの完全なコピーを保持
  (2) 改ざん耐性: 過去のデータの変更が暗号学的に不可能
  (3) 透明性:   全取引が公開され誰でも検証可能
  (4) 非中央集権: 特定の管理者なしにシステムが稼働
  (5) 耐障害性:  一部のノードが停止してもネットワーク全体は稼働継続
```

### 1.2 ブロックチェーンの歴史的背景

ブロックチェーン技術は突然生まれたものではなく、数十年にわたる暗号学と分散システムの研究の集大成である。

| 年代 | 出来事 | 意義 |
|------|--------|------|
| 1976 | Diffie-Hellman 鍵交換の発表 | 公開鍵暗号の基礎理論 |
| 1977 | RSA 暗号の発明 | 実用的な公開鍵暗号の実現 |
| 1979 | Merkle のハッシュ木特許 | データ完全性検証の効率化 |
| 1982 | Lamport のビザンチン将軍問題 | 分散合意の理論的枠組み |
| 1991 | Haber & Stornetta のタイムスタンプチェーン | ハッシュチェーンによる改ざん検知 |
| 1997 | Hashcash（Adam Back） | Proof of Work の原型 |
| 2004 | Reusable Proof of Work（Hal Finney） | PoW トークンの再利用 |
| 2008 | Bitcoin ホワイトペーパー（Satoshi Nakamoto） | 初の実用的ブロックチェーン |
| 2009 | Bitcoin ネットワーク稼働開始 | Genesis Block の生成 |
| 2014 | Ethereum ホワイトペーパー（Vitalik Buterin） | スマートコントラクト基盤 |
| 2015 | Ethereum メインネット稼働 | プログラマブルなブロックチェーン |
| 2020 | Ethereum 2.0 Beacon Chain 稼働 | PoS への移行開始 |
| 2022 | The Merge（Ethereum の PoW→PoS 完全移行） | エネルギー消費 99.95% 削減 |

### 1.3 定義の明確化

「ブロックチェーン」という用語は文脈によって異なる意味で使われることがある。本章では以下の定義に基づいて議論を進める。

**狭義のブロックチェーン**: 暗号学的ハッシュ関数によって連結されたブロックの時系列データ構造。各ブロックは前のブロックのハッシュ値を含み、これにより改ざん検知が可能になる。

**広義のブロックチェーン**: ハッシュチェーン構造に加え、P2P ネットワーク、コンセンサスアルゴリズム、インセンティブ機構を組み合わせた分散型台帳システム全体を指す。

---

## 2. ハッシュチェーンとブロック構造

### 2.1 暗号学的ハッシュ関数の基礎

ブロックチェーンの改ざん耐性の根幹を支えるのが暗号学的ハッシュ関数である。ハッシュ関数は任意長の入力データを固定長のハッシュ値（ダイジェスト）に変換する関数であり、以下の性質を満たす。

**暗号学的ハッシュ関数が満たすべき 5 つの性質:**

1. **決定性（Deterministic）**: 同じ入力に対して常に同じ出力を返す
2. **高速計算**: 任意の入力に対してハッシュ値を効率的に計算できる
3. **原像耐性（Pre-image Resistance）**: ハッシュ値 h から h = H(m) を満たす m を見つけるのが計算上困難
4. **第二原像耐性（Second Pre-image Resistance）**: 入力 m1 が与えられたとき、H(m1) = H(m2) を満たす m2 (m2 ≠ m1) を見つけるのが計算上困難
5. **衝突耐性（Collision Resistance）**: H(m1) = H(m2) を満たす異なる m1, m2 の組を見つけるのが計算上困難

```python
"""
コード例 1: SHA-256 ハッシュ関数の基本的性質を確認する
"""
import hashlib


def sha256(data: str) -> str:
    """文字列を SHA-256 でハッシュ化し、16進数文字列として返す。"""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# --- 決定性の確認 ---
msg = "Hello, Blockchain!"
hash1 = sha256(msg)
hash2 = sha256(msg)
assert hash1 == hash2, "同じ入力には常に同じハッシュ値が返る"
print(f"入力: '{msg}'")
print(f"SHA-256: {hash1}")
# 出力例: SHA-256: 7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284adfa93...

# --- 雪崩効果（Avalanche Effect）の確認 ---
msg_a = "Hello, Blockchain!"
msg_b = "Hello, Blockchain?"  # 末尾の記号を 1 文字変更
hash_a = sha256(msg_a)
hash_b = sha256(msg_b)

# ビット単位での差分を計算
bits_a = bin(int(hash_a, 16))[2:].zfill(256)
bits_b = bin(int(hash_b, 16))[2:].zfill(256)
diff_bits = sum(a != b for a, b in zip(bits_a, bits_b))

print(f"\n--- 雪崩効果 ---")
print(f"入力A: '{msg_a}' -> {hash_a[:16]}...")
print(f"入力B: '{msg_b}' -> {hash_b[:16]}...")
print(f"異なるビット数: {diff_bits}/256 ({diff_bits/256*100:.1f}%)")
# 理想的には約 50% のビットが変化する

# --- 固定長出力の確認 ---
short_input = "A"
long_input = "A" * 10000
print(f"\n--- 固定長出力 ---")
print(f"入力 1 文字:     {sha256(short_input)} (長さ: {len(sha256(short_input))})")
print(f"入力 10000 文字: {sha256(long_input)} (長さ: {len(sha256(long_input))})")
# どちらも 64 文字（256 ビット）の 16 進数文字列
```

### 2.2 ブロックの内部構造

ブロックチェーンの各ブロックは、ヘッダ（Header）とボディ（Body）の 2 つの領域で構成される。ヘッダにはメタデータが格納され、ボディにはトランザクション群が格納される。

```
ブロックの詳細構造（Bitcoin モデル）:

  Block N-1              Block N                Block N+1
  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
  │ Block Header   │    │ Block Header   │    │ Block Header   │
  │ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
  │ │ version    │ │    │ │ version    │ │    │ │ version    │ │
  │ │ prevHash ──┼─┼────│ │ prevHash ──┼─┼────│ │ prevHash   │ │
  │ │ merkleRoot │ │    │ │ merkleRoot │ │    │ │ merkleRoot │ │
  │ │ timestamp  │ │    │ │ timestamp  │ │    │ │ timestamp  │ │
  │ │ difficulty │ │    │ │ difficulty │ │    │ │ difficulty │ │
  │ │ nonce      │ │    │ │ nonce      │ │    │ │ nonce      │ │
  │ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
  ├────────────────┤    ├────────────────┤    ├────────────────┤
  │ Block Body     │    │ Block Body     │    │ Block Body     │
  │ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
  │ │ Tx Count   │ │    │ │ Tx Count   │ │    │ │ Tx Count   │ │
  │ │ Tx 1       │ │    │ │ Tx 1       │ │    │ │ Tx 1       │ │
  │ │ Tx 2       │ │    │ │ Tx 2       │ │    │ │ Tx 2       │ │
  │ │ Tx 3       │ │    │ │ Tx 3       │ │    │ │ Tx 3       │ │
  │ │ ...        │ │    │ │ ...        │ │    │ │ ...        │ │
  │ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
  └────────────────┘    └────────────────┘    └────────────────┘

  ヘッダフィールドの詳細:
  ─────────────────────────────────────────────────────────
  version     : プロトコルバージョン（4 バイト）
  prevHash    : 直前ブロックのヘッダハッシュ（32 バイト）
                → これがチェーン構造の根幹
  merkleRoot  : 全 Tx のマークル木ルートハッシュ（32 バイト）
  timestamp   : ブロック生成時の Unix タイムスタンプ（4 バイト）
  difficulty  : マイニング難易度ターゲット（4 バイト）
  nonce       : PoW で探索する値（4 バイト）
  ─────────────────────────────────────────────────────────
  Bitcoin のヘッダサイズ: 合計 80 バイト（固定長）
```

**prevHash フィールドが改ざん耐性を実現する仕組み:**

```
改ざんの連鎖的影響:

  Block 100 の Tx1 を改ざんしたい場合:

  Step 1: Tx1 のデータを変更
          → merkleRoot が変化

  Step 2: Block 100 のヘッダハッシュが変化
          → Block 101 の prevHash と不一致

  Step 3: Block 101 の prevHash を修正
          → Block 101 のヘッダハッシュが変化
          → Block 102 の prevHash と不一致

  Step 4: Block 102 以降も全て再計算が必要

  Step 5: さらに各ブロックで PoW の再計算が必要
          （= 莫大な計算コスト）

  Step 6: かつ、ネットワークの過半数を超える計算力で
          正規チェーンより速くブロックを生成し続ける必要がある

  結論: 過去のブロックの改ざんは計算上不可能

  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
  │Blk 98│→│Blk 99│→│Blk100│→│Blk101│→│Blk102│
  └──────┘  └──────┘  └──┬───┘  └──────┘  └──────┘
                         │
                     改ざん箇所
                         │
                         ▼
                    ここから先の全ブロックを
                    再計算しなければならない
```

### 2.3 マークル木（Merkle Tree）

マークル木はトランザクションの完全性を効率的に検証するためのハッシュ二分木構造である。Ralph Merkle が 1979 年に特許を取得したデータ構造で、ブロックチェーンにおけるトランザクション検証の基盤となっている。

```
マークル木の構造（4 つのトランザクションの場合）:

                 ┌──────────────────┐
                 │   Merkle Root    │
                 │  H(H_AB + H_CD)  │
                 └────────┬─────────┘
                    ┌─────┴─────┐
              ┌─────┴─────┐  ┌──┴──────────┐
              │   H_AB    │  │    H_CD     │
              │ H(H_A+H_B)│  │ H(H_C+H_D) │
              └─────┬─────┘  └──┬──────────┘
                ┌───┴───┐    ┌──┴───┐
           ┌────┴──┐ ┌──┴───┐ ┌┴────┐ ┌─────┐
           │  H_A  │ │ H_B  │ │ H_C │ │ H_D │
           │H(Tx_A)│ │H(Tx_B)│ │H(Tx_C)│ │H(Tx_D)│
           └───┬───┘ └──┬───┘ └──┬──┘ └──┬──┘
               │        │        │       │
             Tx_A     Tx_B     Tx_C    Tx_D


  検証の効率性（SPV: Simplified Payment Verification）:
  ─────────────────────────────────────────────────────
  Tx_B がブロックに含まれることを検証するには:

  必要な情報: H_A, H_CD, Merkle Root の 3 つだけ

  検証手順:
  1. H_B = H(Tx_B)                   ... 手元で計算
  2. H_AB = H(H_A + H_B)             ... H_A は提供される
  3. Root = H(H_AB + H_CD)           ... H_CD は提供される
  4. 計算した Root とブロックヘッダの merkleRoot を比較

  計算量: O(log n)  ← n はトランザクション数
  例: 1000 Tx → 約 10 回のハッシュ計算で検証完了
  例: 100万 Tx → 約 20 回のハッシュ計算で検証完了
```

```python
"""
コード例 2: マークル木の構築と検証を実装する
"""
import hashlib
from typing import List, Optional, Tuple


def hash_data(data: str) -> str:
    """データを SHA-256 でハッシュ化する。"""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_pair(left: str, right: str) -> str:
    """2 つのハッシュ値を結合してハッシュ化する。"""
    return hashlib.sha256((left + right).encode("utf-8")).hexdigest()


class MerkleTree:
    """マークル木の構築・検証を行うクラス。"""

    def __init__(self, transactions: List[str]):
        if not transactions:
            raise ValueError("トランザクションリストは空にできない")
        self.transactions = transactions
        self.leaves = [hash_data(tx) for tx in transactions]
        self.tree: List[List[str]] = []  # 各レベルのハッシュを保持
        self.root = self._build_tree()

    def _build_tree(self) -> str:
        """マークル木を構築し、ルートハッシュを返す。"""
        current_level = self.leaves[:]
        self.tree.append(current_level[:])

        while len(current_level) > 1:
            next_level = []
            # ノード数が奇数の場合、最後のノードを複製
            if len(current_level) % 2 == 1:
                current_level.append(current_level[-1])
            for i in range(0, len(current_level), 2):
                parent = hash_pair(current_level[i], current_level[i + 1])
                next_level.append(parent)
            self.tree.append(next_level[:])
            current_level = next_level

        return current_level[0]

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        指定インデックスのトランザクションの包含証明（Merkle Proof）を返す。
        戻り値: [(ハッシュ値, 位置)] のリスト。位置は "left" or "right"
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"インデックス {index} は範囲外")

        proof = []
        idx = index

        for level in self.tree[:-1]:  # ルートレベルは除外
            # 奇数個の場合、最後の要素を複製
            level_copy = level[:]
            if len(level_copy) % 2 == 1:
                level_copy.append(level_copy[-1])

            if idx % 2 == 0:
                # 自分が左 → 右の兄弟を証拠に
                sibling_idx = idx + 1
                if sibling_idx < len(level_copy):
                    proof.append((level_copy[sibling_idx], "right"))
            else:
                # 自分が右 → 左の兄弟を証拠に
                proof.append((level_copy[idx - 1], "left"))
            idx //= 2

        return proof

    @staticmethod
    def verify_proof(
        tx_hash: str,
        proof: List[Tuple[str, str]],
        root: str,
    ) -> bool:
        """マークル証明を検証する。"""
        current = tx_hash
        for sibling_hash, position in proof:
            if position == "left":
                current = hash_pair(sibling_hash, current)
            else:
                current = hash_pair(current, sibling_hash)
        return current == root


# --- 使用例 ---
transactions = [
    "Alice -> Bob: 10 BTC",
    "Bob -> Charlie: 5 BTC",
    "Charlie -> Dave: 3 BTC",
    "Dave -> Eve: 1 BTC",
]

tree = MerkleTree(transactions)
print(f"Merkle Root: {tree.root[:16]}...")

# Tx 1（Bob -> Charlie）の包含証明を取得
proof = tree.get_proof(1)
tx_hash = hash_data(transactions[1])
print(f"\nTx: '{transactions[1]}'")
print(f"Tx Hash: {tx_hash[:16]}...")
print(f"Proof steps: {len(proof)}")

# 検証
is_valid = MerkleTree.verify_proof(tx_hash, proof, tree.root)
print(f"検証結果: {'有効' if is_valid else '無効'}")

# 改ざんしたトランザクションの検証
tampered_hash = hash_data("Bob -> Charlie: 50 BTC")
is_valid_tampered = MerkleTree.verify_proof(tampered_hash, proof, tree.root)
print(f"改ざん Tx の検証結果: {'有効' if is_valid_tampered else '無効（改ざん検知）'}")
```

### 2.4 Genesis Block（創世ブロック）

ブロックチェーンの最初のブロックは Genesis Block と呼ばれ、prevHash が存在しない特殊なブロックである。Bitcoin の Genesis Block は 2009 年 1 月 3 日に Satoshi Nakamoto によって生成された。このブロックのコインベーストランザクションには、当時の The Times 紙の見出し "Chancellor on brink of second bailout for banks"（財務大臣、銀行への二度目の救済を検討中）が埋め込まれており、既存の金融システムへの批判的メッセージとして広く知られている。

### 2.5 P2P ネットワークとブロック伝播

ブロックチェーンネットワークは P2P（Peer-to-Peer）構造を採用しており、中央サーバーを介さずにノード間で直接通信する。新しいトランザクションやブロックは「ゴシッププロトコル」によってネットワーク全体に伝播される。

```
P2P ネットワークにおけるブロック伝播:

  Step 1: マイナー M がブロックを発見
  ┌───┐
  │ M │  ← 新ブロックを生成
  └─┬─┘
    │ ブロードキャスト
    ├───────────┬───────────┐
    ▼           ▼           ▼
  ┌───┐     ┌───┐       ┌───┐
  │ A │     │ B │       │ C │  ← 直接接続されたピア
  └─┬─┘     └─┬─┘       └─┬─┘
    │         │            │
    ▼         ▼            ▼
  ┌───┐     ┌───┐       ┌───┐
  │ D │     │ E │       │ F │  ← さらに伝播
  └───┘     └───┘       └───┘

  各ノードの処理手順:
  1. ブロックを受信
  2. ブロックヘッダのハッシュを検証
  3. previous_hash が自身のチェーンの末尾と一致するか確認
  4. 全トランザクションの署名を検証
  5. PoW 条件を満たしているか確認
  6. 検証成功 → 自身のチェーンに追加 + 他のピアに転送
  7. 検証失敗 → ブロックを破棄

  ノードの種類（Bitcoin の場合）:
  ─────────────────────────────────────────
  フルノード:
    全ブロックチェーンデータを保持（約 500 GB+）
    全トランザクションを独立に検証
    完全な自律性を持つ

  ライトノード（SPV ノード）:
    ブロックヘッダのみを保持（約 50 MB）
    マークル証明で特定 Tx の包含を検証
    フルノードに問い合わせが必要

  マイニングノード:
    フルノード + マイニング機能
    新ブロックの生成を試みる
    マイニングプール経由が一般的
```

### 2.6 フォーク（チェーンの分岐）

フォークとはブロックチェーンが分岐する現象であり、一時的フォークとプロトコル変更によるフォークに大別される。

```
一時的フォーク（Temporary Fork / Stale Block）:
──────────────────────────────────────────────

  2 人のマイナーがほぼ同時にブロックを発見した場合:

       ┌──────┐
   ┌──→│Blk N'│  ← マイナー A が発見
   │   │(ver A)│
   │   └──────┘
  ┌──────┐
  │Blk N-1│
  └──┬───┘
   │   ┌──────┐
   └──→│Blk N │  ← マイナー B が発見
       │(ver B)│
       └──────┘

  解決: 「最長チェーン規則（Longest Chain Rule）」
  次のブロックがどちらの上に積まれるかで決まる

       ┌──────┐   ┌──────┐
   ┌──→│Blk N'│──→│Blk N+1│  ← こちらが長い → 正規チェーン
   │   └──────┘   └───────┘
  ┌──────┐
  │Blk N-1│
  └──┬───┘
   │   ┌──────┐
   └──→│Blk N │  ← 孤立ブロック（Orphan Block）として破棄
       └──────┘

  → Bitcoin で「6 承認待ち」が推奨される理由:
    6 ブロック後にフォークが覆る確率は 0.0002% 未満


プロトコルフォーク:
──────────────────────
  ハードフォーク（Hard Fork）:
    後方互換性のないプロトコル変更
    旧ノードは新ブロックを無効と判断 → チェーンが永久分岐
    例: Bitcoin → Bitcoin Cash（2017年、ブロックサイズ論争）
        Ethereum → Ethereum Classic（2016年、The DAO 事件後のロールバック）

  ソフトフォーク（Soft Fork）:
    後方互換性のあるプロトコル変更
    旧ノードも新ブロックを有効と判断
    例: Bitcoin の SegWit（2017年、署名データの分離）
```

---

## 3. コンセンサスメカニズム

### 3.1 分散合意の必要性とビザンチン将軍問題

中央管理者が存在しない分散システムにおいて、全ノードが同一の台帳状態に合意するためには、何らかの合意形成プロトコル（コンセンサスメカニズム）が必要である。この問題の理論的基盤となるのがビザンチン将軍問題である。

```
ビザンチン将軍問題（Byzantine Generals Problem）:
─────────────────────────────────────────────

  設定:
  - 4 人の将軍が敵の城を包囲している
  - 全員が同時に攻撃すれば勝利、バラバラなら敗北
  - 伝令（メッセンジャー）を通じてのみ通信可能
  - ただし、将軍の中に裏切り者がいる可能性がある

           将軍 A（正直）
          /           \
    「攻撃」       「攻撃」
        /               \
  将軍 B（正直）    将軍 C（裏切り者）
        \               /
    「攻撃」       「撤退」  ← 嘘のメッセージ
        \               /
           将軍 D（正直）
           → B「攻撃」C「撤退」... どちらを信じる？

  問題:
  裏切り者が存在する状況で、正直な将軍たちは
  どのようにして正しい合意に到達できるか？

  理論的結果（Lamport, Shostak, Pease 1982）:
  - 裏切り者の数を f とすると、最低 3f + 1 のノードが必要
  - 全体の 1/3 未満が裏切り者であれば合意可能

  ブロックチェーンのコンセンサスメカニズムは
  この問題に対する実用的な解法を提供する
```

### 3.2 Proof of Work（PoW）

PoW は Bitcoin で初めて実用化されたコンセンサスメカニズムであり、計算パズルを最初に解いたノード（マイナー）がブロックを追加する権利を得る仕組みである。

**PoW の動作原理:**

```
PoW パズル:
  SHA-256(block_header) < target

  target は難易度に応じて変動する閾値
  → ハッシュ値の先頭に N 個のゼロが並ぶ nonce を探す

  例: difficulty = 4 (先頭 4 桁がゼロ)
  ┌────────────────────────────────────────────────┐
  │ nonce=0:      a3f2b1c8e9d...  → 条件を満たさない│
  │ nonce=1:      7c4e9d2f0a1...  → 条件を満たさない│
  │ nonce=2:      f1a8b3c7d2e...  → 条件を満たさない│
  │ ...                                             │
  │ nonce=74839:  00009ab3f21...  → 条件を満たさない│
  │ nonce=74840:  00003d8a1f2...  → 条件を満たさない│
  │ ...                                             │
  │ nonce=198247: 0000038f7c1...  → 条件を満たす！   │
  └────────────────────────────────────────────────┘

  非対称性（Asymmetry）:
  - 解を見つける: 困難（総当たり探索、平均数十億回の計算）
  - 解を検証する: 容易（ハッシュ 1 回の計算で済む）
  → この非対称性が PoW の本質

  難易度調整:
  Bitcoin: 2016 ブロックごと（約 2 週間）に調整
  目標: 平均 10 分/ブロック
  - ブロック生成が速すぎる → 難易度上昇
  - ブロック生成が遅すぎる → 難易度低下
```

**51% 攻撃の分析:**

ネットワーク全体の計算力の 51% 以上を掌握した攻撃者は、理論上、正規チェーンより長いチェーンを生成でき、取引の二重支払い（Double Spending）が可能になる。Bitcoin においては天文学的なコストが必要であり現実的に不可能とされるが、ハッシュレートの低い小規模チェーンでは実際に発生事例がある（Ethereum Classic への 51% 攻撃、2019-2020 年）。

```python
"""
コード例 3: Proof of Work マイニングのシミュレーション
"""
import hashlib
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlockHeader:
    """ブロックヘッダを表すデータクラス。"""
    version: int
    prev_hash: str
    merkle_root: str
    timestamp: float
    difficulty: int
    nonce: int = 0

    def to_string(self) -> str:
        """ヘッダ情報を文字列に変換する。"""
        return (
            f"{self.version}{self.prev_hash}{self.merkle_root}"
            f"{self.timestamp}{self.difficulty}{self.nonce}"
        )

    def compute_hash(self) -> str:
        """ヘッダの SHA-256 ハッシュを計算する。"""
        return hashlib.sha256(
            self.to_string().encode("utf-8")
        ).hexdigest()


def mine_block(header: BlockHeader) -> tuple[int, str, float]:
    """
    PoW マイニングを実行する。
    戻り値: (nonce, ハッシュ値, 所要時間)
    """
    target = "0" * header.difficulty
    start_time = time.time()
    attempts = 0

    while True:
        hash_result = header.compute_hash()
        attempts += 1

        if hash_result[:header.difficulty] == target:
            elapsed = time.time() - start_time
            print(f"  ブロック発見！")
            print(f"  Nonce: {header.nonce}")
            print(f"  Hash:  {hash_result}")
            print(f"  試行回数: {attempts:,}")
            print(f"  所要時間: {elapsed:.3f} 秒")
            return header.nonce, hash_result, elapsed

        header.nonce += 1

        # 進捗表示（100万回ごと）
        if attempts % 1_000_000 == 0:
            print(f"  ... {attempts:,} 回試行中 ...")


# --- 難易度ごとのマイニング時間の比較 ---
print("=== PoW マイニングシミュレーション ===\n")

results = []
for difficulty in range(1, 6):
    print(f"--- Difficulty: {difficulty} (先頭 {difficulty} 桁がゼロ) ---")
    header = BlockHeader(
        version=1,
        prev_hash="0" * 64,
        merkle_root="abcdef1234567890" * 4,
        timestamp=time.time(),
        difficulty=difficulty,
    )
    nonce, block_hash, elapsed = mine_block(header)
    results.append((difficulty, nonce, elapsed))
    print()

print("=== 結果まとめ ===")
print(f"{'Difficulty':>12} {'Nonce':>12} {'時間(秒)':>12}")
print("-" * 40)
for d, n, t in results:
    print(f"{d:>12} {n:>12,} {t:>12.3f}")
# difficulty が 1 増えるごとに、平均で約 16 倍の計算が必要になる
```

### 3.3 Proof of Stake（PoS）

PoS は計算リソースの代わりに、保有するトークン（ステーク）に基づいてブロック生成権を割り当てるコンセンサスメカニズムである。PoW の膨大なエネルギー消費を回避しつつ、経済的インセンティブによりネットワークの安全性を確保する。

**比較表: PoW vs PoS**

| 項目 | Proof of Work (PoW) | Proof of Stake (PoS) |
|------|--------------------|--------------------|
| ブロック生成権の根拠 | 計算力（ハッシュレート） | ステーク量（保有トークン） |
| 必要なリソース | 高性能 GPU/ASIC、大量の電力 | ステーク用トークン、通常のサーバー |
| エネルギー消費 | 極めて大（年間 100-150 TWh 級） | 極めて小（PoW の 0.05% 未満） |
| 攻撃コスト | ネットワークの 51% の計算力 | ネットワークの 33% のステーク |
| 不正行為のペナルティ | 電力の浪費（間接的損失） | ステークの没収（Slashing、直接的損失） |
| 参入障壁 | 専用ハードウェアの調達 | トークンの購入 |
| 中央集権化リスク | マイニングプールの寡占 | 大量保有者への権力集中 |
| ファイナリティ | 確率的（6 承認で実質確定） | 確定的（Epoch 単位で確定） |
| 代表的な実装 | Bitcoin, Litecoin, Dogecoin | Ethereum 2.0, Cardano, Polkadot |

**Ethereum の PoS（The Merge, 2022 年 9 月）:**

- バリデータになるには 32 ETH のステークが必要
- Epoch（32 スロット、約 6.4 分）ごとにバリデータ委員会がランダムに選出
- 選出されたバリデータがブロックを提案し、他のバリデータが投票（Attestation）
- 不正行為（二重投票、矛盾する投票）に対してステークが没収される（Slashing）
- The Merge によりエネルギー消費が 99.95% 削減された

**Nothing-at-Stake 問題と対策:**

PoS ではフォーク（チェーンの分岐）が発生した場合、バリデータは計算コストなしに両方のフォークに投票できてしまう。PoW では両方のフォークでマイニングすると計算力が分散するため自然に抑止されるが、PoS ではこの自然な抑止力がない。Ethereum の Casper FFG プロトコルでは、矛盾する投票を行ったバリデータのステークを自動的に没収する Slashing 条件を設けることでこの問題に対処している。

### 3.4 その他のコンセンサスメカニズム

**Delegated Proof of Stake（DPoS）:**

トークン保有者が代表者（デリゲート）に投票し、選出された少数のデリゲートがブロック生成を行う。議会制民主主義に類似した仕組みであり、高速な処理が可能だが、中央集権化のリスクがある。EOS（21 デリゲート）、Tron が代表的な実装である。

**BFT 系（PBFT, Tendermint）:**

ノード間の明示的な投票プロセスにより合意を形成する。全体の 2/3 以上の合意で即座に確定（ファイナリティ）が得られる利点がある一方、参加ノード数が増えると通信量が O(n^2) で増大するため、数十から数百のノード規模に適する。Cosmos（Tendermint BFT）、Hyperledger Fabric が代表例である。

**Proof of Authority（PoA）:**

事前に承認されたバリデータのみがブロック生成を行う。完全な身元確認に基づく信頼モデルであり、パブリックチェーンの分散性は失われるが、処理速度が極めて高速でありプライベート/コンソーシアムチェーンに適する。VeChain、テストネット（Goerli 等）で採用されている。

**Proof of History（PoH）:**

Solana が採用するメカニズムで、PoS と組み合わせて使用される。暗号学的なハッシュチェーンによって時間の経過を証明し、ブロック生成のオーバーヘッドを削減する。SHA-256 の連続計算によって「検証可能な遅延関数（VDF）」に近い機能を実現している。

**比較表: コンセンサスメカニズムの総合比較**

| メカニズム | 速度 (TPS) | 分散性 | エネルギー効率 | ファイナリティ | 代表例 |
|-----------|-----------|--------|--------------|--------------|--------|
| PoW | 3-7 | 高い | 非常に低い | 確率的（約60分） | Bitcoin, Litecoin |
| PoS | 15-100 | 高い | 非常に高い | 確定的（約13分） | Ethereum 2.0, Cardano |
| DPoS | 1000-4000 | 中程度 | 高い | 確定的（数秒） | EOS, Tron |
| BFT 系 | 1000-10000 | 低い（ノード数制限） | 高い | 即時確定 | Cosmos, Hyperledger |
| PoA | 数千以上 | 低い | 高い | 即時確定 | VeChain |
| PoH+PoS | 数千-数万 | 中程度 | 高い | 確定的（数秒） | Solana |

---

## 4. 暗号技術の基盤

### 4.1 公開鍵暗号と楕円曲線暗号

ブロックチェーンにおけるデジタルアイデンティティと取引の認証は、公開鍵暗号に基づいている。特に楕円曲線暗号（ECC: Elliptic Curve Cryptography）が広く使用されている。Bitcoin と Ethereum はいずれも secp256k1 曲線を使用する ECDSA（Elliptic Curve Digital Signature Algorithm）を採用している。

```
鍵の導出プロセス:

  ┌───────────────┐
  │   秘密鍵      │    256 ビットのランダムな整数
  │ (Private Key) │    例: 0x1a2b3c...（32 バイト）
  └───────┬───────┘
          │
          │ 楕円曲線乗算（一方向関数）
          │ 公開鍵 = 秘密鍵 × G（生成点）
          │ ※ 逆算は計算上不可能（離散対数問題）
          ▼
  ┌───────────────┐
  │   公開鍵      │    楕円曲線上の点 (x, y)
  │ (Public Key)  │    非圧縮形式: 65 バイト
  └───────┬───────┘    圧縮形式: 33 バイト
          │
          │ ハッシュ関数の適用
          │ Bitcoin: RIPEMD-160(SHA-256(公開鍵))
          │ Ethereum: Keccak-256(公開鍵) の末尾 20 バイト
          ▼
  ┌───────────────┐
  │  アドレス     │    取引の宛先として使用
  │ (Address)     │    Bitcoin: 1A1zP1... (Base58Check)
  └───────────────┘    Ethereum: 0xAb5801... (16 進数)

  重要な性質:
  秘密鍵 → 公開鍵 → アドレス（一方向のみ）
  アドレス → 公開鍵 → 秘密鍵（逆算不可能）
```

**デジタル署名の仕組み:**

送信者は秘密鍵を用いてトランザクションに署名し、受信者（および全ノード）は送信者の公開鍵を用いて署名を検証する。これにより、(1) トランザクションが秘密鍵の保有者によって作成されたこと（認証）、(2) トランザクションの内容が改ざんされていないこと（完全性）が保証される。

### 4.2 ウォレットの種類と鍵管理

秘密鍵の管理はブロックチェーン利用における最も重要なセキュリティ要素である。秘密鍵を失えば資産は永久に失われ、秘密鍵が漏洩すれば資産は盗まれる。

**ウォレットの分類:**

| 分類 | 種類 | 特徴 | リスク | 例 |
|------|------|------|--------|-----|
| ホットウォレット | ブラウザ拡張 | 利便性が高い、DApp 連携容易 | マルウェア、フィッシング | MetaMask, Phantom |
| ホットウォレット | モバイルアプリ | 持ち運び可能 | デバイス紛失、マルウェア | Trust Wallet, Rainbow |
| コールドウォレット | ハードウェア | オフライン保管、高セキュリティ | 物理的紛失、故障 | Ledger, Trezor |
| コールドウォレット | ペーパーウォレット | 完全オフライン | 物理的損傷、紛失 | 紙への印刷 |

**HD ウォレット（Hierarchical Deterministic Wallet）:**

BIP-32/39/44 で定義された規格であり、単一のシード（通常 12 または 24 個のニーモニックフレーズ）から決定的に無限の鍵ペアを導出できる。バックアップはシードフレーズの保管のみで完了する。

### 4.3 トランザクションの構造

Bitcoin のトランザクションは UTXO（Unspent Transaction Output）モデルを採用しており、Ethereum はアカウントモデルを採用している。

```
UTXO モデル（Bitcoin）:
─────────────────────
  「未使用のお釣り」を管理するモデル

  例: Alice が Bob に 3 BTC を送る
  （Alice は過去に 5 BTC を受け取っている）

  Input:                    Output:
  ┌──────────────────┐     ┌──────────────────┐
  │ UTXO: 5 BTC      │     │ Bob: 3 BTC       │
  │ (Alice 宛の      │ ──→ │ (新しい UTXO)    │
  │  過去の出力)      │     ├──────────────────┤
  │ + Alice の署名   │     │ Alice: 1.999 BTC │
  └──────────────────┘     │ (お釣り UTXO)    │
                           ├──────────────────┤
                           │ 手数料: 0.001 BTC│
                           │ (マイナーへ)      │
                           └──────────────────┘

  Input の合計 = Output の合計 + 手数料
  5 BTC = 3 + 1.999 + 0.001

アカウントモデル（Ethereum）:
─────────────────────────
  口座残高を直接管理するモデル

  ┌─────────────────────────────┐
  │ Alice のアカウント            │
  │ 残高: 10 ETH → 6.999 ETH   │
  │ Nonce: 5 → 6               │
  └─────────────────────────────┘
           │ 3 ETH + Gas
           ▼
  ┌─────────────────────────────┐
  │ Bob のアカウント              │
  │ 残高: 2 ETH → 5 ETH        │
  └─────────────────────────────┘
```

---

## 5. スマートコントラクト

### 5.1 スマートコントラクトとは何か

スマートコントラクトとは、ブロックチェーン上にデプロイされ、事前に定義された条件が満たされたときに自動的に実行されるプログラムのことである。Nick Szabo が 1996 年に概念を提唱し、2015 年の Ethereum メインネット稼働により初めて汎用的な実装が実現された。

「もし条件 X が満たされたら、処理 Y を実行する」というロジックが、第三者の介入なしにブロックチェーン上で自動的に執行される。

**スマートコントラクトの核心的特性:**

1. **不可逆性（Immutability）**: デプロイ後のコード変更は原則不可能（Upgradeable Proxy パターンは例外）
2. **透明性**: コードが公開されており、誰でも検証可能
3. **自動実行**: 条件が満たされれば人間の介入なしに実行される
4. **決定的実行**: 同じ入力に対して全ノードが同じ結果を出力する
5. **ガスコスト**: 実行に計算手数料（Gas）が必要

### 5.2 Ethereum Virtual Machine（EVM）

EVM は Ethereum のスマートコントラクト実行環境であり、チューリング完全なスタックベースの仮想マシンである。全ノードが同一の EVM 上でコードを実行し、結果が一致することを確認する。

```
EVM のアーキテクチャ:

  ┌─────────────────────────────────────────┐
  │          Ethereum Virtual Machine        │
  │  ┌────────────┐  ┌───────────────────┐  │
  │  │   Stack    │  │    Memory         │  │
  │  │ (1024 深度)│  │ (揮発性、呼出毎)  │  │
  │  └────────────┘  └───────────────────┘  │
  │  ┌────────────────────────────────────┐  │
  │  │         Storage                    │  │
  │  │  (永続的、Key-Value ストア)        │  │
  │  │  256-bit Key → 256-bit Value      │  │
  │  └────────────────────────────────────┘  │
  │  ┌────────────────────────────────────┐  │
  │  │         Bytecode                   │  │
  │  │  Solidity → コンパイル → Bytecode  │  │
  │  │  PUSH, POP, ADD, SSTORE, CALL ... │  │
  │  └────────────────────────────────────┘  │
  └─────────────────────────────────────────┘

  アカウントの種類:
  ┌──────────────────────────────┐
  │ EOA（外部所有アカウント）     │
  │ = 人間が秘密鍵で操作         │
  │ - アドレス, 残高, nonce      │
  │ - トランザクションを発行可能 │
  ├──────────────────────────────┤
  │ Contract Account             │
  │ = スマートコントラクト        │
  │ - アドレス, 残高, コード      │
  │ - ストレージ（永続データ）    │
  │ - 他のコントラクトを呼び出し可│
  └──────────────────────────────┘

  ガス（Gas）の仕組み:
  ────────────────────────────────
  各オペコードに Gas コストが定義されている

  操作                     Gas コスト
  ──────────────────────────────────
  ADD（加算）              3
  MUL（乗算）              5
  SLOAD（Storage 読出）    2100
  SSTORE（Storage 書込）   20000（新規）/ 5000（更新）
  CREATE（コントラクト作成）32000
  ──────────────────────────────────

  トランザクション手数料 = Gas Used × Gas Price
  例: Uniswap スワップ
      Gas Used: 約 150,000
      Gas Price: 30 Gwei (= 0.00000003 ETH)
      手数料: 150,000 × 0.00000003 = 0.0045 ETH
```

### 5.3 Solidity によるスマートコントラクト実装例

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DecentralizedCrowdfunding
 * @notice 分散型クラウドファンディングコントラクト
 *
 * 機能:
 * - プロジェクトオーナーが目標金額と期限を設定
 * - 出資者が ETH を出資
 * - 期限内に目標達成 → オーナーが資金を引き出し可能
 * - 期限切れで目標未達 → 出資者に返金
 */
contract DecentralizedCrowdfunding {
    // --- 状態変数 ---
    address public owner;
    uint256 public goal;
    uint256 public deadline;
    uint256 public totalFunded;
    bool public claimed;

    mapping(address => uint256) public contributions;

    // --- イベント ---
    event Funded(address indexed contributor, uint256 amount);
    event Claimed(address indexed owner, uint256 amount);
    event Refunded(address indexed contributor, uint256 amount);

    // --- 修飾子 ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner");
        _;
    }

    modifier beforeDeadline() {
        require(block.timestamp < deadline, "Deadline passed");
        _;
    }

    modifier afterDeadline() {
        require(block.timestamp >= deadline, "Deadline not reached");
        _;
    }

    // --- コンストラクタ ---
    constructor(uint256 _goal, uint256 _durationSeconds) {
        require(_goal > 0, "Goal must be positive");
        require(_durationSeconds > 0, "Duration must be positive");
        owner = msg.sender;
        goal = _goal;
        deadline = block.timestamp + _durationSeconds;
    }

    // --- 出資 ---
    function fund() external payable beforeDeadline {
        require(msg.value > 0, "Must send ETH");
        contributions[msg.sender] += msg.value;
        totalFunded += msg.value;
        emit Funded(msg.sender, msg.value);
    }

    // --- 目標達成時の資金引き出し ---
    function claim() external onlyOwner afterDeadline {
        require(totalFunded >= goal, "Goal not reached");
        require(!claimed, "Already claimed");
        claimed = true;
        uint256 amount = address(this).balance;
        // Checks-Effects-Interactions パターンで再入攻撃を防止
        (bool success, ) = payable(owner).call{value: amount}("");
        require(success, "Transfer failed");
        emit Claimed(owner, amount);
    }

    // --- 目標未達時の返金 ---
    function refund() external afterDeadline {
        require(totalFunded < goal, "Goal was reached");
        uint256 amount = contributions[msg.sender];
        require(amount > 0, "No contribution");
        // Checks-Effects-Interactions パターン
        contributions[msg.sender] = 0;
        (bool success, ) = payable(msg.sender).call{value: amount}("");
        require(success, "Transfer failed");
        emit Refunded(msg.sender, amount);
    }
}
```

### 5.4 Python でスマートコントラクトのロジックを模倣する

スマートコントラクトの概念を理解するため、Solidity のロジックを Python で模倣した実装を示す。

```python
"""
コード例 4: スマートコントラクトの概念を Python で模倣する
分散型エスクロー（第三者預託）の動作を再現
"""
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


class EscrowState(Enum):
    """エスクローの状態遷移。"""
    AWAITING_PAYMENT = auto()
    AWAITING_DELIVERY = auto()
    COMPLETE = auto()
    REFUNDED = auto()


@dataclass
class EscrowContract:
    """
    分散型エスクローのロジックを模倣するクラス。

    実際のスマートコントラクトでは:
    - 状態がブロックチェーン上に永続化される
    - msg.sender による呼び出し元の自動認証
    - ETH の送受信が言語レベルでサポート
    - 全操作がトランザクションとして記録される
    """
    buyer: str
    seller: str
    arbiter: str  # 紛争解決者
    amount: float
    state: EscrowState = EscrowState.AWAITING_PAYMENT
    balance: float = 0.0
    created_at: float = field(default_factory=time.time)
    events: list = field(default_factory=list)

    def _emit(self, event_name: str, data: dict) -> None:
        """イベントを記録する（ブロックチェーンのイベントログに相当）。"""
        entry = {
            "event": event_name,
            "data": data,
            "timestamp": time.time(),
        }
        self.events.append(entry)
        print(f"  [Event] {event_name}: {data}")

    def deposit(self, sender: str, value: float) -> None:
        """
        買い手がエスクローに入金する。
        Solidity の payable 関数に相当。
        """
        if sender != self.buyer:
            raise PermissionError("買い手のみが入金可能")
        if self.state != EscrowState.AWAITING_PAYMENT:
            raise RuntimeError(f"無効な状態: {self.state.name}")
        if value != self.amount:
            raise ValueError(f"正確な金額 {self.amount} を入金すること")

        self.balance += value
        self.state = EscrowState.AWAITING_DELIVERY
        self._emit("Deposited", {"buyer": sender, "amount": value})

    def confirm_delivery(self, sender: str) -> None:
        """
        買い手が商品受取を確認し、売り手に送金する。
        Checks-Effects-Interactions パターンを適用。
        """
        # Checks: 前提条件の検証
        if sender != self.buyer:
            raise PermissionError("買い手のみが確認可能")
        if self.state != EscrowState.AWAITING_DELIVERY:
            raise RuntimeError(f"無効な状態: {self.state.name}")

        # Effects: 状態の更新（送金前に状態を変更）
        payout = self.balance
        self.balance = 0.0
        self.state = EscrowState.COMPLETE

        # Interactions: 外部呼び出し（送金）
        self._emit("DeliveryConfirmed", {
            "buyer": sender,
            "seller": self.seller,
            "amount": payout,
        })
        print(f"  → {self.seller} に {payout} を送金")

    def refund(self, sender: str) -> None:
        """仲裁者が返金を実行する。"""
        if sender != self.arbiter:
            raise PermissionError("仲裁者のみが返金可能")
        if self.state != EscrowState.AWAITING_DELIVERY:
            raise RuntimeError(f"無効な状態: {self.state.name}")

        refund_amount = self.balance
        self.balance = 0.0
        self.state = EscrowState.REFUNDED

        self._emit("Refunded", {
            "arbiter": sender,
            "buyer": self.buyer,
            "amount": refund_amount,
        })
        print(f"  → {self.buyer} に {refund_amount} を返金")


# --- 使用例 ---
print("=== 分散型エスクロー シミュレーション ===\n")

escrow = EscrowContract(
    buyer="Alice",
    seller="Bob",
    arbiter="Charlie",
    amount=1.5,
)
print(f"コントラクト作成: {escrow.buyer} → {escrow.seller}, 金額: {escrow.amount} ETH")
print(f"状態: {escrow.state.name}\n")

# シナリオ 1: 正常な取引
print("--- シナリオ: 正常な取引 ---")
escrow.deposit("Alice", 1.5)
print(f"状態: {escrow.state.name}, 残高: {escrow.balance}")
escrow.confirm_delivery("Alice")
print(f"状態: {escrow.state.name}, 残高: {escrow.balance}\n")

# シナリオ 2: 不正な操作の検出
print("--- シナリオ: 不正な操作の検出 ---")
escrow2 = EscrowContract(buyer="Alice", seller="Bob", arbiter="Charlie", amount=2.0)
escrow2.deposit("Alice", 2.0)
try:
    escrow2.confirm_delivery("Bob")  # 売り手が勝手に確認しようとする
except PermissionError as e:
    print(f"  [拒否] {e}")
try:
    escrow2.deposit("Alice", 2.0)  # 二重入金
except RuntimeError as e:
    print(f"  [拒否] {e}")
```

### 5.5 Layer 2 スケーリングソリューション

Ethereum メインネット（Layer 1）のスループット（約 15 TPS）では大規模な利用に耐えられないため、Layer 2 ソリューションが開発されている。

```
Layer 2 スケーリングの全体像:

  ┌─────────────────────────────────────────────────┐
  │                    Layer 2                       │
  │  ┌──────────────────┐  ┌──────────────────────┐ │
  │  │ Optimistic Rollup│  │    ZK Rollup         │ │
  │  │                  │  │                      │ │
  │  │ Tx をオフチェーン│  │ Tx をオフチェーンで  │ │
  │  │ で実行し、不正が │  │ 実行し、ゼロ知識証明 │ │
  │  │ あれば 7 日以内に│  │ で正しさを数学的に   │ │
  │  │ 異議申立が可能   │  │ 証明してから L1 に   │ │
  │  │                  │  │ 提出                 │ │
  │  │ 例: Arbitrum     │  │ 例: zkSync           │ │
  │  │     Optimism     │  │     StarkNet         │ │
  │  │     Base         │  │     Polygon zkEVM    │ │
  │  └────────┬─────────┘  └──────────┬───────────┘ │
  └───────────┼───────────────────────┼─────────────┘
              │  圧縮データ + 証明     │
              ▼                       ▼
  ┌─────────────────────────────────────────────────┐
  │          Layer 1 (Ethereum Mainnet)              │
  │    セキュリティ基盤 + データ可用性 + 最終確定性    │
  └─────────────────────────────────────────────────┘

  比較: Optimistic Rollup vs ZK Rollup
  ┌──────────────┬────────────────────┬──────────────────┐
  │ 項目          │ Optimistic Rollup  │ ZK Rollup        │
  ├──────────────┼────────────────────┼──────────────────┤
  │ 検証方法      │ 不正証明           │ 有効性証明       │
  │              │ (Fraud Proof)      │ (Validity Proof) │
  │ 引出し時間    │ 約 7 日            │ 数分〜数時間     │
  │ 計算コスト    │ 低い               │ 高い（証明生成） │
  │ EVM 互換性   │ 高い               │ 改善中           │
  │ 成熟度       │ 高い               │ 急速に発展中     │
  └──────────────┴────────────────────┴──────────────────┘
```

---

## 6. DeFi とトークンエコノミクス

### 6.1 DeFi（分散型金融）の概要

DeFi は銀行や証券会社などの金融仲介者を排除し、スマートコントラクトによって金融サービスを提供する仕組みの総称である。2020 年の「DeFi Summer」以降急速に成長し、TVL（Total Value Locked）はピーク時に 1800 億ドル以上に達した。

**主要な DeFi プロトコルカテゴリ:**

| カテゴリ | 機能 | 代表的プロトコル | 仕組み |
|---------|------|-----------------|--------|
| DEX（分散型取引所） | トークン交換 | Uniswap, SushiSwap, Curve | AMM（自動マーケットメイカー） |
| レンディング | 貸し借り | Aave, Compound | 担保付き貸出、変動金利 |
| ステーブルコイン | 価値安定通貨 | MakerDAO (DAI), USDC | 暗号資産/法定通貨担保 |
| デリバティブ | 先物・オプション | dYdX, GMX | オンチェーンデリバティブ |
| 保険 | リスクヘッジ | Nexus Mutual | 分散型保険プール |
| イールドアグリゲータ | 運用最適化 | Yearn Finance | 自動戦略切替 |

**AMM（自動マーケットメイカー）の原理:**

Uniswap の中核は `x * y = k` という定数積公式である。流動性プール内の 2 つのトークンの量の積が常に一定に保たれることで、注文板なしにトークン交換が実現される。

### 6.2 トークン標準

Ethereum 上のトークンは ERC（Ethereum Request for Comments）で標準化されている。

| 標準 | 種類 | 特徴 | 用途例 |
|------|------|------|--------|
| ERC-20 | 代替可能トークン（FT） | 各トークンが同等の価値を持つ | 通貨（USDC）、ガバナンス（UNI） |
| ERC-721 | 非代替性トークン（NFT） | 各トークンが一意の識別子を持つ | デジタルアート、ゲームアイテム |
| ERC-1155 | マルチトークン | FT と NFT を一つのコントラクトで管理 | ゲーム内アイテム全般 |
| ERC-4626 | トークン化ボールト | 利回り付きトークンの標準 | レンディングプール、ステーキング |

---

## 7. ブロックチェーンの限界と課題

### 7.1 スケーラビリティトリレンマ

Vitalik Buterin が提唱したスケーラビリティトリレンマは、分散型システムにおいて「分散性」「セキュリティ」「スケーラビリティ」の 3 つの性質を同時に最大化することが極めて困難であるという命題である。

```
スケーラビリティトリレンマ:

          分散性 (Decentralization)
             /\
            /  \
           /    \
          / 理想 \
         / (不可能)\
        /    領域    \
       /──────────────\
      /                \
  セキュリティ ──────── スケーラビリティ
  (Security)          (Scalability)


  各チェーンのポジション:
  ─────────────────────────────────────────
  Bitcoin:
    分散性 ★★★★★  セキュリティ ★★★★★  スケーラビリティ ★☆☆☆☆
    → 約 7 TPS、完全な分散、最高のセキュリティ

  Ethereum (L1):
    分散性 ★★★★☆  セキュリティ ★★★★★  スケーラビリティ ★★☆☆☆
    → 約 15 TPS、高い分散性、PoS 移行でセキュリティ維持

  Solana:
    分散性 ★★☆☆☆  セキュリティ ★★★☆☆  スケーラビリティ ★★★★★
    → 数千 TPS、バリデータ要件が高く分散性に制約

  Ethereum + L2:
    分散性 ★★★★☆  セキュリティ ★★★★☆  スケーラビリティ ★★★★☆
    → L1 のセキュリティを継承しつつ L2 でスケール

  参考（中央集権型）:
  Visa: 65,000 TPS（ピーク時）
  → 完全な中央集権だがスケーラビリティは圧倒的
```

### 7.2 セキュリティ上の課題

**スマートコントラクトの脆弱性:**

スマートコントラクトはデプロイ後に変更が困難なため、脆弱性が重大な資産損失につながる。

| 脆弱性 | 説明 | 主な被害事例 |
|--------|------|------------|
| 再入攻撃（Reentrancy） | 外部呼出し中に同じ関数が再帰的に呼ばれる | The DAO 事件（2016年、約6000万ドル） |
| 整数オーバーフロー | 算術演算の桁あふれ | Beauty Chain 事件（2018年） |
| フラッシュローン攻撃 | 1 Tx 内で無担保借入→価格操作→返済 | bZx 攻撃（2020年） |
| オラクル操作 | 外部データソースの価格を操作 | Mango Markets（2022年、約1.1億ドル） |
| アクセス制御の不備 | 権限チェックの欠如 | Parity Wallet 凍結（2017年、約1.5億ドル） |

### 7.3 社会的・規制的課題

- **環境問題**: PoW の膨大なエネルギー消費（PoS 移行で緩和傾向）
- **規制の不確実性**: 各国の法規制が異なり、急速に変化している
- **投機とボラティリティ**: 暗号資産の極端な価格変動
- **マネーロンダリング**: 匿名性を悪用した資金洗浄のリスク
- **真の分散性**: マイニングプールや大口ステーカーへの権力集中
- **秘密鍵管理**: ユーザーの自己管理責任と UX の課題

---

## 8. Python によるブロックチェーン実装

### 8.1 最小限のブロックチェーン実装

以下は教育目的のブロックチェーン実装であり、ハッシュチェーン構造、ブロック生成、改ざん検知の基本的な仕組みを示す。

```python
"""
コード例 5: 教育用ブロックチェーンの完全実装
ハッシュチェーン、PoW マイニング、改ざん検知を含む
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Transaction:
    """トランザクション（取引記録）を表すデータクラス。"""
    sender: str
    receiver: str
    amount: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "timestamp": self.timestamp,
        }

    def compute_hash(self) -> str:
        tx_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(tx_string.encode("utf-8")).hexdigest()


@dataclass
class Block:
    """ブロックを表すデータクラス。"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: str = ""

    def compute_hash(self) -> str:
        """ブロックの SHA-256 ハッシュを計算する。"""
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode("utf-8")).hexdigest()


class Blockchain:
    """
    教育用ブロックチェーンの実装。
    PoW マイニング、チェーン検証、改ざん検知をサポートする。
    """

    def __init__(self, difficulty: int = 2):
        """
        Args:
            difficulty: PoW の難易度（ハッシュ先頭のゼロの数）
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self._create_genesis_block()

    def _create_genesis_block(self) -> None:
        """Genesis Block（創世ブロック）を生成する。"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64,
        )
        genesis.hash = self._proof_of_work(genesis)
        self.chain.append(genesis)

    def _proof_of_work(self, block: Block) -> str:
        """PoW マイニングを実行して有効なハッシュを見つける。"""
        target = "0" * self.difficulty
        block.nonce = 0
        computed_hash = block.compute_hash()

        while not computed_hash.startswith(target):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_transaction(self, sender: str, receiver: str, amount: float) -> None:
        """新しいトランザクションをペンディングリストに追加する。"""
        if amount <= 0:
            raise ValueError("送金額は正の数でなければならない")
        tx = Transaction(sender=sender, receiver=receiver, amount=amount)
        self.pending_transactions.append(tx)

    def mine_pending_transactions(self, miner_address: str) -> Block:
        """
        ペンディングトランザクションを含む新しいブロックをマイニングする。
        """
        # マイニング報酬トランザクションを追加
        reward_tx = Transaction(
            sender="NETWORK",
            receiver=miner_address,
            amount=6.25,  # ブロック報酬
        )
        transactions = self.pending_transactions + [reward_tx]

        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=transactions,
            previous_hash=self.chain[-1].hash,
        )

        # PoW マイニング
        new_block.hash = self._proof_of_work(new_block)
        self.chain.append(new_block)

        # ペンディングリストをクリア
        self.pending_transactions = []
        return new_block

    def is_chain_valid(self) -> bool:
        """ブロックチェーン全体の整合性を検証する。"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # ハッシュ値の再計算と比較
            if current.hash != current.compute_hash():
                print(f"  [不正] Block {i}: ハッシュ値が不一致")
                return False

            # 前ブロックとのリンク検証
            if current.previous_hash != previous.hash:
                print(f"  [不正] Block {i}: previous_hash が不一致")
                return False

            # PoW 条件の検証
            if not current.hash.startswith("0" * self.difficulty):
                print(f"  [不正] Block {i}: PoW 条件を満たさない")
                return False

        return True

    def get_balance(self, address: str) -> float:
        """指定アドレスの残高を計算する。"""
        balance = 0.0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    balance -= tx.amount
                if tx.receiver == address:
                    balance += tx.amount
        return balance

    def print_chain(self) -> None:
        """ブロックチェーンの内容を表示する。"""
        for block in self.chain:
            print(f"\n--- Block {block.index} ---")
            print(f"  Timestamp:     {time.ctime(block.timestamp)}")
            print(f"  Previous Hash: {block.previous_hash[:16]}...")
            print(f"  Hash:          {block.hash[:16]}...")
            print(f"  Nonce:         {block.nonce}")
            print(f"  Transactions:  {len(block.transactions)}")
            for tx in block.transactions:
                print(f"    {tx.sender} -> {tx.receiver}: {tx.amount}")


# --- 使用例 ---
print("=== 教育用ブロックチェーン ===\n")

bc = Blockchain(difficulty=2)

# トランザクションの追加
bc.add_transaction("Alice", "Bob", 10.0)
bc.add_transaction("Bob", "Charlie", 5.0)
print("Block 1 をマイニング中...")
block1 = bc.mine_pending_transactions("Miner1")
print(f"Block 1 マイニング完了 (nonce: {block1.nonce})")

bc.add_transaction("Charlie", "Alice", 3.0)
bc.add_transaction("Alice", "Dave", 2.0)
print("Block 2 をマイニング中...")
block2 = bc.mine_pending_transactions("Miner1")
print(f"Block 2 マイニング完了 (nonce: {block2.nonce})")

# チェーン全体の表示
bc.print_chain()

# 残高確認
print(f"\n--- 残高 ---")
for addr in ["Alice", "Bob", "Charlie", "Dave", "Miner1"]:
    print(f"  {addr}: {bc.get_balance(addr):.2f}")

# 整合性検証
print(f"\nチェーンは有効か: {bc.is_chain_valid()}")

# --- 改ざんのデモ ---
print("\n=== 改ざんデモ ===")
print("Block 1 のトランザクションを改ざん...")
bc.chain[1].transactions[0] = Transaction("Alice", "Bob", 1000.0)
print(f"改ざん後のチェーンは有効か: {bc.is_chain_valid()}")
```

---

## 9. アンチパターンと設計上の落とし穴

### 9.1 アンチパターン 1: 再入攻撃に対する無防備な設計

**問題**: 外部コントラクトへの送金を行った後に状態を更新するパターンは、再入攻撃（Reentrancy Attack）を許す致命的な脆弱性となる。2016 年の The DAO 事件では、この脆弱性により約 6000 万ドル相当の ETH が流出した。

```
再入攻撃のメカニズム:

  ┌─────────────────┐        ┌──────────────────┐
  │  攻撃者コントラクト│        │  脆弱なコントラクト│
  │                  │        │                   │
  │  withdraw() 呼出 │───────→│  1. 残高確認: OK   │
  │                  │        │  2. ETH を送金     │
  │  ┌──────────────┐│←───────│     ↓             │
  │  │ receive()    ││        │                   │
  │  │  → 再度      ││───────→│  1. 残高確認: OK!  │
  │  │  withdraw()  ││        │  (まだ更新前)      │
  │  │  を呼び出す  ││        │  2. ETH を送金     │
  │  └──────────────┘│←───────│     ↓             │
  │  ...（繰り返し） │        │  3. 残高=0 に更新  │
  │                  │        │  （ここで初めて更新）│
  └─────────────────┘        └──────────────────┘

  脆弱なコード（擬似コード）:
  ─────────────────────────────
  function withdraw(amount):
      require(balances[msg.sender] >= amount)
      msg.sender.call{value: amount}("")  # ← ここで攻撃者の receive() が呼ばれる
      balances[msg.sender] -= amount       # ← 状態更新が送金の後！

  修正後（Checks-Effects-Interactions パターン）:
  ─────────────────────────────
  function withdraw(amount):
      require(balances[msg.sender] >= amount)   # Checks: 条件確認
      balances[msg.sender] -= amount             # Effects: 状態更新（送金前!）
      msg.sender.call{value: amount}("")         # Interactions: 外部呼出し

  さらに安全な対策:
  - ReentrancyGuard 修飾子の使用（OpenZeppelin 実装）
  - Pull Payment パターン（送金を受取側に委ねる）
```

### 9.2 アンチパターン 2: 「全てをブロックチェーンに載せる」設計

**問題**: 大量のデータや高頻度の更新をブロックチェーン上に直接保存しようとする設計は、ガスコストの爆発と実用性の低下を招く。

```
誤った設計:
─────────────────────────
  ❌ 画像データをブロックチェーンに直接保存
     → 1 MB の画像 ≈ 数千ドルのガスコスト
     → 全ノードが画像データを永久に保持

  ❌ 毎秒のセンサーデータを全てオンチェーンに記録
     → ガスコストが天文学的
     → ネットワーク帯域の浪費

  ❌ 単一組織内部のデータベースをブロックチェーンで置換
     → 分散合意のオーバーヘッドが無駄
     → 従来の RDBMS の方が全面的に優れる

正しい設計:
─────────────────────────
  ✅ オフチェーンストレージ + オンチェーンハッシュ
     → データ本体は IPFS/Arweave に保存
     → ハッシュ値のみをブロックチェーンに記録
     → 改ざん検知とデータ可用性を分離

  ✅ オフチェーン計算 + オンチェーン検証
     → 計算は Layer 2 またはオフチェーンで実行
     → 結果の証明のみを Layer 1 に提出

  ✅ ブロックチェーンが本当に必要な場面を見極める
     → 複数の利害関係者間の信頼問題が存在するか？
     → 中央管理者を信頼できない状況か？
     → 改ざん耐性が不可欠か？
     → 上記のいずれも該当しなければ通常の DB で十分
```

### 9.3 アンチパターン 3: ブロックチェーンの「銀の弾丸」思考

ブロックチェーンを万能の解決策として適用しようとする傾向は、技術選定における典型的な誤りである。以下の判断基準を参考にすべきである。

**ブロックチェーン採用の判断フローチャート:**

```
  Q1: 複数の組織/参加者がデータを共有する必要があるか？
  │
  ├─ No → 通常のデータベースを使用
  │
  └─ Yes
      │
      Q2: 全参加者が信頼できる単一の管理者が存在するか？
      │
      ├─ Yes → 通常のデータベースを使用
      │
      └─ No
          │
          Q3: データの改ざん耐性・透明性が重要か？
          │
          ├─ No → 分散データベース（CockroachDB 等）で十分
          │
          └─ Yes
              │
              Q4: 参加者は不特定多数か？
              │
              ├─ Yes → パブリックチェーン（Ethereum, Solana 等）
              │
              └─ No → コンソーシアムチェーン（Hyperledger 等）
```

### 9.4 セキュリティベストプラクティス

スマートコントラクト開発においてセキュリティを確保するための指針を整理する。

```
スマートコントラクト開発のセキュリティチェックリスト:
─────────────────────────────────────────────────

  設計段階:
  □ Checks-Effects-Interactions パターンを徹底する
  □ ReentrancyGuard（OpenZeppelin）を外部呼出しのある関数に適用
  □ 最小権限の原則: 各関数に必要最小限の権限のみを付与
  □ アクセス制御: onlyOwner, onlyRole 等の修飾子を適切に使用
  □ 整数演算: Solidity 0.8+ の組み込みオーバーフロー検出を活用
  □ Pull Payment パターン: 送金はユーザーが自ら引き出す設計にする

  テスト段階:
  □ 単体テスト: 全関数の正常系・異常系をカバー
  □ ファジングテスト: Foundry の forge-std/Test を使用
  □ 不変条件テスト: コントラクトの不変条件をアサーションで検証
  □ フォークテスト: メインネットの状態をフォークして実環境に近いテスト

  デプロイ前:
  □ 外部監査: 専門のセキュリティ監査会社によるレビュー
  □ バグバウンティ: 脆弱性報告に対する報奨金プログラムを設置
  □ 形式検証: Certora, Halmos 等のツールで数学的に正しさを証明
  □ テストネットでの長期運用テスト

  デプロイ後:
  □ 監視システム: 異常なトランザクションパターンを検知
  □ 緊急停止機能（Circuit Breaker）: 問題発生時にコントラクト一時停止
  □ タイムロック: 重要なパラメータ変更に遅延を設ける
  □ マルチシグ: 管理者権限をマルチシグウォレットで管理
```

```python
"""
コード例 6: Checks-Effects-Interactions パターンのデモンストレーション
再入攻撃に対する脆弱なコードと安全なコードの比較
"""
from typing import Dict


class VulnerableVault:
    """
    脆弱な金庫コントラクトの模倣。
    外部呼出しの後に状態を更新するため、再入攻撃に弱い。
    """

    def __init__(self):
        self.balances: Dict[str, float] = {}

    def deposit(self, user: str, amount: float) -> None:
        self.balances[user] = self.balances.get(user, 0) + amount

    def withdraw(self, user: str, amount: float, callback=None) -> None:
        """脆弱な引き出し: 送金（callback）後に残高を更新する。"""
        balance = self.balances.get(user, 0)
        if balance < amount:
            raise ValueError("残高不足")

        # Interaction が Effects より先 → 再入攻撃に脆弱
        if callback:
            callback(amount)  # ← 攻撃者が再度 withdraw を呼べる

        self.balances[user] = balance - amount  # ← まだ古い balance を参照


class SecureVault:
    """
    安全な金庫コントラクトの模倣。
    Checks-Effects-Interactions パターンを適用。
    """

    def __init__(self):
        self.balances: Dict[str, float] = {}
        self._locked = False  # ReentrancyGuard

    def deposit(self, user: str, amount: float) -> None:
        self.balances[user] = self.balances.get(user, 0) + amount

    def withdraw(self, user: str, amount: float, callback=None) -> None:
        """安全な引き出し: 状態更新後に外部呼出しを行う。"""
        # ReentrancyGuard
        if self._locked:
            raise RuntimeError("再入攻撃を検出: 関数がロック中")
        self._locked = True

        try:
            # Checks: 条件検証
            balance = self.balances.get(user, 0)
            if balance < amount:
                raise ValueError("残高不足")

            # Effects: 状態更新（外部呼出しの前に実行）
            self.balances[user] = balance - amount

            # Interactions: 外部呼出し
            if callback:
                callback(amount)
        finally:
            self._locked = False


# --- 再入攻撃シミュレーション ---
print("=== 再入攻撃シミュレーション ===\n")

# 脆弱なコントラクト
print("--- VulnerableVault（脆弱） ---")
vault = VulnerableVault()
vault.deposit("attacker", 10.0)
stolen = [0.0]
attack_count = [0]


def malicious_callback(amount: float) -> None:
    """攻撃者の receive 関数を模倣。"""
    attack_count[0] += 1
    stolen[0] += amount
    if attack_count[0] < 3:
        try:
            vault.withdraw("attacker", amount, malicious_callback)
        except (ValueError, RecursionError):
            pass


vault.withdraw("attacker", 10.0, malicious_callback)
print(f"  盗まれた金額: {stolen[0]}  再入回数: {attack_count[0]}")

# 安全なコントラクト
print("\n--- SecureVault（安全） ---")
secure = SecureVault()
secure.deposit("attacker", 10.0)


def malicious_callback_secure(amount: float) -> None:
    try:
        secure.withdraw("attacker", amount, malicious_callback_secure)
    except RuntimeError as e:
        print(f"  [防御成功] {e}")


secure.withdraw("attacker", 10.0, malicious_callback_secure)
print(f"  攻撃者の残高: {secure.balances.get('attacker', 0)}")
```

---

## 10. ブロックチェーンの実世界応用

### 10.1 主要なユースケース

ブロックチェーン技術は金融以外にも多様な領域で応用が進んでいる。

| 領域 | ユースケース | 具体例 | ブロックチェーンの利点 |
|------|------------|--------|---------------------|
| サプライチェーン | 製品追跡 | IBM Food Trust | 改ざん耐性のある履歴管理 |
| 医療 | 電子カルテ共有 | MedRec（MIT） | 患者主導のデータ管理 |
| 不動産 | 登記管理 | スウェーデン土地登記局 | 仲介コストの削減 |
| 知的財産 | 著作権管理 | Ascribe | タイムスタンプによる先行権証明 |
| 投票 | 電子投票 | Voatz | 透明性と改ざん耐性の両立 |
| エネルギー | P2P 電力取引 | Power Ledger | 仲介者なしの電力売買 |
| ゲーム | デジタル資産所有 | Axie Infinity | 真のデジタル所有権 |
| 身分証明 | 分散型 ID（DID） | Microsoft ION | 自己主権型アイデンティティ |

### 10.2 CBDC（中央銀行デジタル通貨）

各国の中央銀行が分散台帳技術を応用した法定通貨のデジタル版を検討・開発している。中国のデジタル人民元（e-CNY）は大規模な実証実験が行われ、欧州中央銀行のデジタルユーロも開発が進行中である。CBDC は中央銀行が発行・管理する点で本質的に中央集権的であるが、分散台帳技術の一部の利点（プログラマビリティ、追跡可能性）を活用している。

### 10.3 Web3 と分散型インターネットの展望

Web3 はブロックチェーン技術を基盤として、ユーザーがデータの所有権を取り戻すことを目指すインターネットの新たなパラダイムである。

```
Web の進化:

  Web 1.0（1990年代〜）: 読み取り専用
    静的 HTML ページ、情報の一方向配信
    例: 個人ホームページ、Yahoo! ディレクトリ

  Web 2.0（2000年代〜）: 読み書き
    ユーザー生成コンテンツ、SNS、クラウド
    例: Facebook, YouTube, Twitter
    課題: プラットフォームがデータを独占

  Web 3.0（2020年代〜）: 読み書き + 所有
    ブロックチェーンによるデータ主権
    例: DeFi, NFT, DAO, 分散型 SNS
    課題: UX, スケーラビリティ, 規制

  ┌────────────┬──────────┬──────────┬──────────┐
  │            │ Web 1.0  │ Web 2.0  │ Web 3.0  │
  ├────────────┼──────────┼──────────┼──────────┤
  │ データ管理  │ 分散     │ 中央集権 │ 分散     │
  │ 認証       │ なし     │ OAuth等  │ ウォレット│
  │ 支払い     │ クレカ   │ 電子決済 │ 暗号資産 │
  │ ガバナンス │ なし     │ 企業     │ DAO      │
  └────────────┴──────────┴──────────┴──────────┘
```

---

## 11. 実践演習

### 演習 1:【基礎】ブロックチェーンの手動構築とハッシュチェーン検証

```
以下のステップに従い、3 ブロックのチェーンを手動で構築せよ。
Python の hashlib を使用すること。

Step 1: Genesis Block の作成
  - index: 0
  - previous_hash: "0000000000000000000000000000000000000000000000000000000000000000"
  - data: "Genesis Block"
  - timestamp: 任意の固定値（例: "2024-01-01T00:00:00"）
  - hash = SHA-256(index + previous_hash + data + timestamp)

Step 2: Block 1 の作成
  - index: 1
  - previous_hash: Genesis Block のハッシュ
  - data: "Alice sends 10 BTC to Bob"
  - timestamp: "2024-01-01T00:10:00"

Step 3: Block 2 の作成
  - index: 2
  - previous_hash: Block 1 のハッシュ
  - data: "Bob sends 5 BTC to Charlie"
  - timestamp: "2024-01-01T00:20:00"

Step 4: 検証
  (a) 全ブロックのハッシュを表示し、チェーンの連結を確認せよ
  (b) Block 1 の data を "Alice sends 100 BTC to Bob" に改ざんした場合、
      どのブロックからハッシュが変化するか確認せよ
  (c) 改ざんを検出するバリデーション関数を実装せよ

期待される学習成果:
  - ハッシュチェーンの仕組みを体感的に理解する
  - 1 箇所の改ざんがチェーン全体に波及することを確認する
```

### 演習 2:【応用】PoW 難易度と計算コストの関係分析

```python
# 以下のコードを拡張し、PoW の難易度と計算コストの関係を分析せよ。
#
# 課題:
# (1) difficulty を 1 から 5 まで変化させ、各難易度でのマイニング時間を計測せよ
# (2) 難易度が 1 増加するごとに、平均計算回数が何倍になるか計算せよ
# (3) 結果を表形式で出力し、指数関数的増加を確認せよ
# (4) （発展）5 回試行の平均値を取り、統計的なばらつきも報告せよ

import hashlib
import time
import statistics


def mine_with_stats(data: str, prev_hash: str, difficulty: int) -> dict:
    """
    PoW マイニングを実行し、統計情報を返す。

    戻り値:
        {
            "difficulty": int,
            "nonce": int,
            "hash": str,
            "attempts": int,
            "elapsed_seconds": float,
        }
    """
    target = "0" * difficulty
    nonce = 0
    start = time.time()

    while True:
        text = f"{prev_hash}{data}{nonce}"
        h = hashlib.sha256(text.encode()).hexdigest()
        nonce += 1
        if h[:difficulty] == target:
            return {
                "difficulty": difficulty,
                "nonce": nonce - 1,
                "hash": h,
                "attempts": nonce,
                "elapsed_seconds": time.time() - start,
            }

# ここから自由に拡張すること
```

### 演習 3:【発展】マルチノードコンセンサスシミュレーション

```
分散ネットワーク上でのコンセンサスをシミュレーションするプログラムを設計せよ。

要件:
(1) 5 つの Node クラスのインスタンスを生成する
(2) 各ノードは自身のブロックチェーンのコピーを保持する
(3) トランザクションをブロードキャストする仕組みを実装する
(4) 各ノードが独立にマイニングを試み、最初に成功したノードが
    ブロックを他のノードに伝播する
(5) 他のノードはブロックの妥当性を検証した上で自身のチェーンに追加する
(6) 「最長チェーン規則」を実装する:
    チェーンの長さが異なる場合、最長の有効なチェーンを正規チェーンとして採用する

発展課題:
(7) 1 つのノードを「悪意あるノード」とし、不正なトランザクションを含む
    ブロックを生成させた場合の挙動を観察せよ
(8) フォーク（チェーン分岐）が発生する状況を作り、解決過程を観察せよ

設計ヒント:
  - Node クラス: blockchain, pending_transactions, peers を属性に持つ
  - Network クラス: ノード間の通信を仲介するシミュレーション用クラス
  - 各ノードの validate_block() メソッドで不正ブロックを検出する
```

---

## 12. FAQ

### Q1: ブロックチェーンはどのような問題に適しているのか？適していない場面は？

**適している場面:**
- **複数の利害関係者間のデータ共有**: 互いに信頼できない複数組織が共通のデータを管理する場合（例: 国際サプライチェーンの追跡、貿易金融）
- **改ざん耐性が不可欠な記録**: 不動産登記、学位証明、医療記録など、記録の正当性が重要な場面
- **仲介者の排除**: 国際送金（従来は複数の銀行を経由して数日かかる処理を、ブロックチェーンで直接送金）
- **透明性が求められるプロセス**: 投票システム、寄付金の追跡、公共調達

**適していない場面:**
- **単一組織内のデータ管理**: 組織内部では管理者を信頼できるため、通常のデータベースの方が高速かつ低コスト
- **高速な処理が必要な場面**: 数千 TPS 以上が求められるリアルタイムシステム（ただし Layer 2 の発展により閾値は上昇中）
- **プライバシーが最重要**: パブリックチェーンは全データが公開される（ゼロ知識証明等のプライバシー技術は発展途上）
- **大量データの保存**: ブロックチェーンのストレージコストは通常のクラウドストレージの数千倍以上

### Q2: Bitcoin の「半減期」とは何か？なぜ重要なのか？

半減期（Halving）とは、約 4 年（210,000 ブロック）ごとにマイニング報酬が半分になる仕組みである。

| 時期 | ブロック報酬 | 累積発行率（概算） |
|------|------------|------------------|
| 2009年（開始） | 50 BTC | - |
| 2012年（第1回半減期） | 25 BTC | 約50% |
| 2016年（第2回半減期） | 12.5 BTC | 約75% |
| 2020年（第3回半減期） | 6.25 BTC | 約87.5% |
| 2024年（第4回半減期） | 3.125 BTC | 約93.75% |
| 2140年頃 | 0 BTC | 100%（2100万BTC上限） |

半減期が重要な理由は、Bitcoin の供給スケジュールが完全に予測可能であり、デフレ的な通貨設計を実現している点にある。法定通貨は中央銀行が発行量を裁量的に決定するが、Bitcoin は数学的なルールに基づいて供給量が決定される。

### Q3: プライベートチェーンとパブリックチェーンの違いは？どちらを選ぶべきか？

| 特性 | パブリックチェーン | コンソーシアムチェーン | プライベートチェーン |
|------|------------------|--------------------|--------------------|
| 参加 | 誰でも自由に参加可能 | 承認された組織のみ | 単一組織内 |
| 透明性 | 全取引が完全公開 | 参加者間で共有 | 組織内のみ |
| コンセンサス | PoW, PoS 等 | BFT 系が多い | PoA, Raft 等 |
| 速度 | 遅い（7-15 TPS） | 速い（1000+ TPS） | 非常に速い |
| 分散性 | 高い | 中程度 | 低い |
| 代表例 | Bitcoin, Ethereum | Hyperledger Fabric, R3 Corda | 企業内システム |
| 適する用途 | DeFi, NFT, 公共財 | サプライチェーン, 貿易金融 | 組織内監査証跡 |

エンタープライズ用途ではコンソーシアムチェーンまたはプライベートチェーンが採用されることが多い。ただし「プライベートチェーンは分散データベースと何が違うのか？」という批判は根強く、ブロックチェーンの本質的価値（信頼の分散化）が失われているという指摘もある。

### Q4: スマートコントラクトのバグは修正できるのか？

原則としてデプロイ済みのスマートコントラクトのコードは変更不可能（Immutable）である。ただし以下の戦略が存在する。

1. **Proxy パターン**: ロジック部分を別コントラクトに委譲し、Proxy コントラクトの参照先を変更可能にする設計。OpenZeppelin の UUPS や Transparent Proxy が代表的な実装。
2. **マイグレーション**: 新しいコントラクトをデプロイし、ユーザーに移行を促す。古いコントラクトは停止（pause）機能で無効化する。
3. **ガバナンス**: DAO（分散型自律組織）の投票によってアップグレードの可否を決定する。

いずれの場合も、デプロイ前の徹底的なテストと監査（Audit）が最も重要である。

### Q5: ゼロ知識証明（ZKP）とは何か？ブロックチェーンとどう関係するのか？

ゼロ知識証明とは「ある命題が真であることを、その命題の内容を一切明かすことなく証明する」暗号学的プロトコルである。ブロックチェーンにおいては主に以下の 2 つの用途で注目されている。

1. **スケーラビリティ（ZK Rollup）**: 数千のトランザクションを 1 つのゼロ知識証明に圧縮し、Layer 1 に提出することでスループットを大幅に向上させる。
2. **プライバシー**: トランザクションの金額や送受信者を隠蔽しつつ、取引の正当性を証明する（Zcash の zk-SNARKs など）。

---

## 13. 用語集

| 用語 | 英語表記 | 定義 |
|------|---------|------|
| ハッシュ関数 | Hash Function | 任意長の入力を固定長の出力に変換する関数 |
| マークル木 | Merkle Tree | データの完全性を効率的に検証するハッシュ二分木 |
| ナンス | Nonce | Number Used Once、PoW で探索する一度だけ使用される数値 |
| コンセンサス | Consensus | 分散ノード間で同一の状態に合意するメカニズム |
| ファイナリティ | Finality | トランザクションが確定し覆せなくなる状態 |
| ステーキング | Staking | PoS でバリデータになるためにトークンを預託すること |
| スラッシング | Slashing | 不正行為を行ったバリデータのステークを没収するペナルティ |
| ガス | Gas | EVM でのスマートコントラクト実行に要する計算コストの単位 |
| DApp | Decentralized Application | 分散型アプリケーション |
| TVL | Total Value Locked | DeFi プロトコルに預託されている資産の総額 |
| MEV | Maximal Extractable Value | ブロック内のトランザクション順序操作で得られる利益 |
| DAO | Decentralized Autonomous Organization | スマートコントラクトで運営される分散型自律組織 |

---

## まとめ

| 概念 | 要点 |
|------|------|
| ハッシュチェーン | 各ブロックが前ブロックのハッシュを含むことで改ざん検知チェーンを形成 |
| マークル木 | O(log n) でトランザクションの包含を検証可能なハッシュ二分木 |
| PoW | 計算パズルの非対称性（解くのは困難、検証は容易）を利用したコンセンサス |
| PoS | ステーク量に基づくブロック生成権の割当て。PoW 比で 99.95% 省エネ |
| BFT 系 | 明示的投票による即時ファイナリティ。ノード数に制約あり |
| 公開鍵暗号 | 秘密鍵→公開鍵→アドレスの一方向導出でアイデンティティを管理 |
| スマートコントラクト | ブロックチェーン上で自動実行されるプログラム。EVM で決定的に実行 |
| DeFi | 金融仲介者を排除し、スマートコントラクトで金融サービスを提供 |
| Layer 2 | L1 のセキュリティを継承しつつスループットを向上させるスケーリング技術 |
| トリレンマ | 分散性・セキュリティ・スケーラビリティの同時最大化は極めて困難 |

---

## 次に読むべきガイド

→ [[03-future-of-cs.md]] — コンピュータサイエンスの未来

---

## 参考文献

1. Nakamoto, S. "Bitcoin: A Peer-to-Peer Electronic Cash System." 2008. https://bitcoin.org/bitcoin.pdf
2. Buterin, V. "Ethereum Whitepaper: A Next-Generation Smart Contract and Decentralized Application Platform." 2014. https://ethereum.org/whitepaper
3. Antonopoulos, A. M. *Mastering Bitcoin: Programming the Open Blockchain.* 2nd ed., O'Reilly Media, 2017. ISBN: 978-1491954386
4. Antonopoulos, A. M. & Wood, G. *Mastering Ethereum: Building Smart Contracts and DApps.* O'Reilly Media, 2018. ISBN: 978-1491971949
5. Lamport, L., Shostak, R., & Pease, M. "The Byzantine Generals Problem." *ACM Transactions on Programming Languages and Systems*, Vol. 4, No. 3, 1982, pp. 382-401.
6. Szabo, N. "Smart Contracts: Building Blocks for Digital Markets." 1996. https://www.fon.hum.uva.nl/rob/Courses/InformationInSpeech/CDROM/Literature/LOTwinterschool2006/szabo.best.vwh.net/smart_contracts_2.html
7. Wood, G. "Ethereum: A Secure Decentralised Generalised Transaction Ledger (Yellow Paper)." 2014. https://ethereum.github.io/yellowpaper/paper.pdf
8. Ethereum Foundation. "The Merge." https://ethereum.org/en/roadmap/merge/
