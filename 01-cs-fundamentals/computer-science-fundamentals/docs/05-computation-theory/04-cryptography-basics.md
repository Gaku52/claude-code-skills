# 暗号の基礎

> 現代の暗号はCSの計算複雑性理論に基づいており、「解くのが困難な問題」の存在が安全性を保証する。暗号技術は情報化社会の根幹を支えるインフラであり、通信の秘匿性、データの完全性、そして本人性の検証を可能にする。

## この章で学ぶこと

- [ ] 暗号の歴史的発展と古典暗号から現代暗号への転換を説明できる
- [ ] 共通鍵暗号と公開鍵暗号の違いを数学的根拠とともに説明できる
- [ ] ハッシュ関数の性質と用途を理解する
- [ ] TLS/HTTPS の仕組みを概要レベルで説明できる
- [ ] デジタル署名の仕組みと信頼モデルを理解する
- [ ] 暗号の安全性を計算複雑性の観点から評価できる
- [ ] ポスト量子暗号の必要性と主要候補を説明できる
- [ ] 暗号実装におけるアンチパターンを回避できる

---

## 1. 暗号の歴史と発展

暗号（cryptography）の語源はギリシャ語の kryptos（隠された）と graphein（書く）である。暗号技術は数千年の歴史を持ち、軍事・外交通信から現代のインターネットセキュリティに至るまで、情報を守る手段として進化を続けてきた。

### 1.1 古典暗号

古典暗号とは、コンピュータが存在しない時代に使われていた暗号方式の総称である。多くは「文字の置換」や「文字の転置」に基づいている。

#### シーザー暗号（紀元前1世紀）

ガイウス・ユリウス・カエサルが軍事通信に用いたとされる最も単純な換字式暗号である。アルファベットを固定のシフト数だけずらして暗号化する。

```
シーザー暗号の仕組み:

  シフト数 = 3 の場合:

  平文:   A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  暗号文: D E F G H I J K L M N O P Q R S T U V W X Y Z A B C

  暗号化: HELLO → KHOOR
  復号:   KHOOR → HELLO（逆方向に3ずらす）

  弱点: 鍵空間がわずか25通り → 全探索で即座に解読可能
```

#### ヴィジュネル暗号（16世紀）

シーザー暗号の弱点を克服するため、複数のシフト数を鍵として使用する多表式暗号である。鍵の文字列を繰り返し適用することで、同じ平文文字が異なる暗号文字に変換される。

```
ヴィジュネル暗号:

  鍵: KEY
  平文: HELLOWORLD

  H + K(10) = R
  E + E(4)  = I
  L + Y(24) = J
  L + K(10) = V
  O + E(4)  = S
  W + Y(24) = U
  O + K(10) = Y
  R + E(4)  = V
  L + Y(24) = J
  D + K(10) = N

  暗号文: RIJVSUYV JN

  弱点: 鍵長が判明すると頻度分析で解読可能（カシスキー法）
```

#### エニグマ暗号機（20世紀）

第二次世界大戦でドイツ軍が使用した電気機械式暗号装置である。ローター（回転子）を複数組み合わせ、1文字暗号化するたびにローターが回転することで、事実上の多表式暗号を実現した。鍵空間は約 1.59 × 10^20 に達し、当時としては極めて強固だった。

しかし、アラン・チューリングらの暗号解読チームがポーランドの先行研究を基に Bombe（ボンベ）と呼ばれる解読装置を開発し、構造的弱点を突いてエニグマの解読に成功した。この成功は連合国の勝利に大きく貢献したとされる。

```
エニグマの構造（簡略化）:

  入力 → プラグボード → ローター1 → ローター2 → ローター3
                                                      ↓
                                                  反転器（UKW）
                                                      ↓
       ← プラグボード ← ローター1 ← ローター2 ← ローター3
  出力

  特徴:
  - 1文字ごとにローター1が1ステップ回転（オドメーター式）
  - 同じ文字が自分自身に暗号化されない（構造的弱点）
  - 日替わりの鍵設定（ローター選択、初期位置、プラグ配線）
```

### 1.2 現代暗号への転換

古典暗号から現代暗号への転換点は、1949年のクロード・シャノンによる「Communication Theory of Secrecy Systems」と、1976年のディフィーとヘルマンによる「New Directions in Cryptography」である。

**シャノンの情報理論的安全性（1949年）**

シャノンはワンタイムパッド（鍵が平文と同じ長さのランダムな文字列で、一度しか使わない）が情報理論的に安全であることを証明した。これは鍵の長さが十分であれば、暗号文からいかなる計算能力をもってしても平文の情報を得られないことを意味する。しかし、鍵の長さが平文と同じになるため実用性に乏しい。

**ケルクホフスの原理（1883年提唱、現代でも適用）**

暗号システムの安全性は、アルゴリズムの秘密性ではなく、鍵の秘密性のみに依存すべきである。現代の暗号アルゴリズム（AES、RSA など）は全て公開されており、その安全性は鍵の秘匿に基づいている。

```
暗号の発展年表:

  紀元前     シーザー暗号（換字式）
     |
  16世紀     ヴィジュネル暗号（多表式）
     |
  1883年     ケルクホフスの原理
     |
  1918年     ワンタイムパッド（バーナム暗号）
     |
  1940年代   エニグマ解読（チューリング）
     |
  1949年     シャノンの秘匿通信理論
     |
  1976年     Diffie-Hellman 鍵交換
     |
  1977年     DES（共通鍵暗号の標準）/ RSA（公開鍵暗号）
     |
  1991年     PGP（Pretty Good Privacy）
     |
  2000年     AES（DES の後継として標準化）
     |
  2008年     Bitcoin（暗号技術の応用）
     |
  2018年     TLS 1.3（現代的プロトコル）
     |
  2024年     NIST ポスト量子暗号標準 発表
```

### 1.3 暗号の基本概念と用語

暗号を学ぶうえで不可欠な基本用語を整理する。

| 用語 | 定義 | 具体例 |
|------|------|--------|
| 平文 (plaintext) | 暗号化前の元のデータ | "Hello, World!" |
| 暗号文 (ciphertext) | 暗号化後のデータ | "7f83b1657ff1fc..." |
| 鍵 (key) | 暗号化・復号に使用する秘密のパラメータ | 256ビットのランダムバイト列 |
| 暗号化 (encryption) | 平文を暗号文に変換する操作 | AES-256-GCM で暗号化 |
| 復号 (decryption) | 暗号文を平文に戻す操作 | 正しい鍵で復号 |
| 暗号アルゴリズム | 暗号化・復号の手順 | AES, RSA, ChaCha20 |
| 鍵空間 (key space) | 可能な鍵の総数 | AES-256: 2^256 通り |
| 暗号スイート | 複数の暗号アルゴリズムの組み合わせ | TLS_AES_256_GCM_SHA384 |

```
暗号システムの基本モデル:

  送信者（Alice）                         受信者（Bob）
  ┌──────────┐                          ┌──────────┐
  │  平文 M   │                          │  平文 M   │
  │     ↓     │                          │     ↑     │
  │ 暗号化    │      ┌───────────┐      │  復号     │
  │ E(K, M)   │─────→│  暗号文 C  │─────→│ D(K, C)   │
  │     ↓     │      └───────────┘      │     ↑     │
  │  鍵 K     │                          │  鍵 K     │
  └──────────┘                          └──────────┘

  攻撃者（Eve）は暗号文 C を傍受できるが、
  鍵 K を知らなければ平文 M を復元できない

  正しさの条件: D(K, E(K, M)) = M
```

---

## 2. 共通鍵暗号（対称鍵暗号）

共通鍵暗号（symmetric-key cryptography）は、暗号化と復号に同一の鍵を使用する暗号方式である。公開鍵暗号と比較して計算コストが低く、大量のデータを高速に暗号化できるため、データの暗号化には現在でも主に共通鍵暗号が使われている。

### 2.1 ブロック暗号とストリーム暗号

共通鍵暗号は大きく「ブロック暗号」と「ストリーム暗号」に分類される。

```
分類:

  共通鍵暗号
  ├── ブロック暗号: 固定長ブロック単位で処理
  │   ├── DES（56ビット鍵、非推奨）
  │   ├── 3DES（112ビット相当、非推奨）
  │   └── AES（128/192/256ビット鍵、現行標準）
  │
  └── ストリーム暗号: 1バイトまたは1ビット単位で処理
      ├── RC4（非推奨、TLS で禁止）
      └── ChaCha20（現行推奨、TLS 1.3 採用）
```

**ブロック暗号**は平文を固定サイズのブロック（例: AES では 128 ビット）に分割し、各ブロックを暗号化する。ブロック長より長いデータを扱うには「暗号利用モード」が必要になる。

**ストリーム暗号**は鍵から疑似乱数列（キーストリーム）を生成し、平文と XOR 演算して暗号化する。リアルタイム通信に適している。

### 2.2 AES（Advanced Encryption Standard）

AES は 2001 年に NIST が標準化したブロック暗号であり、ベルギーの暗号学者 Joan Daemen と Vincent Rijmen が設計した Rijndael アルゴリズムが選ばれた。現在、世界で最も広く使われている共通鍵暗号アルゴリズムである。

```
AES の基本パラメータ:

  ┌────────────────┬──────────┬──────────┬──────────┐
  │                │ AES-128  │ AES-192  │ AES-256  │
  ├────────────────┼──────────┼──────────┼──────────┤
  │ 鍵長           │ 128 bit  │ 192 bit  │ 256 bit  │
  │ ブロック長     │ 128 bit  │ 128 bit  │ 128 bit  │
  │ ラウンド数     │ 10       │ 12       │ 14       │
  │ 鍵空間         │ 2^128    │ 2^192    │ 2^256    │
  └────────────────┴──────────┴──────────┴──────────┘

  1ラウンドの処理（4ステップ）:
  1. SubBytes   — S-Box による非線形バイト置換
  2. ShiftRows  — 行単位の巡回シフト
  3. MixColumns — 列単位のガロア体演算（最終ラウンド以外）
  4. AddRoundKey — ラウンド鍵との XOR
```

#### AES の1ラウンドの処理

```
AES ラウンド処理の流れ:

  ┌──────────────────┐
  │   入力ブロック    │  4x4 バイト行列（State）
  │  (128 bits)      │
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │    SubBytes      │  各バイトを S-Box で非線形変換
  │  (非線形置換)    │  → 線形解析攻撃への耐性
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │    ShiftRows     │  行0: シフトなし
  │  (行シフト)      │  行1: 左に1バイト巡回
  │                  │  行2: 左に2バイト巡回
  │                  │  行3: 左に3バイト巡回
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │   MixColumns     │  各列を GF(2^8) 上の
  │  (列混合)        │  行列乗算で変換
  │                  │  → 拡散（diffusion）を実現
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │   AddRoundKey    │  ラウンド鍵との XOR
  │  (鍵加算)        │
  └────────┬─────────┘
           ↓
  ┌──────────────────┐
  │   出力ブロック    │
  └──────────────────┘
```

### 2.3 暗号利用モード

ブロック暗号を実際に使用するには、ブロック長を超えるデータをどのように処理するかを定める「暗号利用モード」が不可欠である。モードの選択は安全性に直結するため、正しい理解が求められる。

#### ECB モード（Electronic Codebook）--- 使用禁止

ECB は各ブロックを独立に暗号化する最も単純なモードだが、同一の平文ブロックが同一の暗号文ブロックになるため、パターンが漏洩する。これは重大な脆弱性であり、実用的な暗号化に ECB を使用してはならない。

```
ECB モードの問題（ペンギン問題）:

  元画像（ビットマップ）       ECB 暗号化後             CBC 暗号化後
  ┌───────────────┐          ┌───────────────┐        ┌───────────────┐
  │   ▓▓▓▓▓▓▓    │          │   ▓▓▓▓▓▓▓    │        │ ░▒▓█░▒▓█░▒▓  │
  │  ▓▓████▓▓▓   │          │  ▓▓████▓▓▓   │        │ █▒░▓▒█░▓█▒░  │
  │ ▓▓██████▓▓   │          │ ▓▓██████▓▓   │        │ ░▓█▒░▓█▒░▓█  │
  │  ▓▓████▓▓    │          │  ▓▓████▓▓    │        │ ▒█░▓█▒░▓█▒░  │
  │   ▓▓▓▓▓▓     │          │   ▓▓▓▓▓▓     │        │ ▓█▒░▓█▒░▓█▒  │
  └───────────────┘          └───────────────┘        └───────────────┘
  ペンギンの形が見える        形がそのまま残る!          完全にランダム化

  → ECB では同じ色のブロックが同じ暗号文になるため、
    画像の輪郭が暗号化後も保存されてしまう
```

#### CBC モード（Cipher Block Chaining）

各ブロックの暗号化前に、前のブロックの暗号文と XOR をとる。最初のブロックには初期化ベクトル（IV）を使用する。同一の平文でも異なる IV により異なる暗号文が生成される。

```
CBC モード:

  IV ─┐
       ↓
  P1 ─⊕─→ E(K) ─→ C1 ─┐
                          ↓
              P2 ─────⊕─→ E(K) ─→ C2 ─┐
                                          ↓
                              P3 ─────⊕─→ E(K) ─→ C3

  暗号化: Ci = E(K, Pi ⊕ C_{i-1}),  C0 = IV
  復号:   Pi = D(K, Ci) ⊕ C_{i-1}

  注意: パディングオラクル攻撃に脆弱な場合がある
```

#### GCM モード（Galois/Counter Mode）--- 現行推奨

GCM は CTR（カウンタ）モードと GHASH（ガロアハッシュ）を組み合わせた認証付き暗号（AEAD: Authenticated Encryption with Associated Data）である。暗号化と同時にデータの改竄検知が可能であり、TLS 1.3 で標準的に使用される。

```
GCM モード（AEAD）:

  鍵 K, IV（96bit推奨）, 平文 P, 関連データ A

  ┌─────────────────────────────────────────┐
  │  CTR モードで暗号化:                     │
  │    Counter = IV || 0...01                │
  │    Ci = Pi ⊕ E(K, Counter + i)          │
  │                                          │
  │  GHASH で認証タグ生成:                   │
  │    T = GHASH(H, A, C) ⊕ E(K, IV||0..0)  │
  │    H = E(K, 0^128)                       │
  └─────────────────────────────────────────┘

  出力: (暗号文 C, 認証タグ T)

  復号時: 認証タグを検証してから復号
  → 改竄された暗号文を復号しない（安全）
```

### 2.4 ChaCha20-Poly1305

ChaCha20 は Daniel J. Bernstein が設計したストリーム暗号であり、Poly1305 と組み合わせて AEAD を実現する。AES-GCM と同等の安全性を持ちながら、AES ハードウェアアクセラレーション（AES-NI）を持たない環境（モバイル端末など）でも高速に動作する。TLS 1.3 で AES-256-GCM と並んで採用されている。

```
ChaCha20 の特徴:

  - 256ビット鍵 + 96ビットノンス + 32ビットカウンタ
  - 20ラウンドの Quarter Round 演算
  - 加算・XOR・ローテーションのみで構成（ARX構造）
    → サイドチャネル攻撃に対してタイミング一定
  - ソフトウェア実装でも高速

  AES-GCM vs ChaCha20-Poly1305:

  ┌──────────────────┬─────────────┬────────────────────┐
  │                  │ AES-256-GCM │ ChaCha20-Poly1305  │
  ├──────────────────┼─────────────┼────────────────────┤
  │ 鍵長             │ 256 bit     │ 256 bit            │
  │ ノンス長         │ 96 bit      │ 96 bit             │
  │ HW加速あり       │ 非常に高速  │ 高速               │
  │ HW加速なし       │ 低速        │ 高速（差が出ない） │
  │ TLS 1.3          │ 標準        │ 標準               │
  │ タイミング安全性 │ 実装依存    │ 構造的に安全       │
  └──────────────────┴─────────────┴────────────────────┘
```

### 2.5 Python で学ぶ共通鍵暗号

以下のコードは Python の `cryptography` ライブラリを用いた AES-256-GCM による暗号化・復号の完全な実装例である。

```python
"""
AES-256-GCM による暗号化・復号のデモ
依存ライブラリ: pip install cryptography
"""
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_aes_gcm(plaintext: bytes, key: bytes, aad: bytes = b"") -> tuple[bytes, bytes]:
    """
    AES-256-GCM で平文を暗号化する。

    Args:
        plaintext: 暗号化する平文データ
        key: 256ビット（32バイト）の鍵
        aad: 関連データ（暗号化しないが認証に含める）

    Returns:
        (nonce, ciphertext): ノンスと暗号文+認証タグの組
    """
    # ノンスは暗号化ごとに一意でなければならない（96ビット推奨）
    nonce = os.urandom(12)  # 96ビット = 12バイト
    aesgcm = AESGCM(key)
    # 暗号文には自動的に認証タグ（16バイト）が付加される
    ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
    return nonce, ciphertext


def decrypt_aes_gcm(nonce: bytes, ciphertext: bytes, key: bytes, aad: bytes = b"") -> bytes:
    """
    AES-256-GCM で暗号文を復号する。

    Args:
        nonce: 暗号化時に使用したノンス
        ciphertext: 暗号文+認証タグ
        key: 256ビット（32バイト）の鍵
        aad: 暗号化時と同じ関連データ

    Returns:
        復号された平文

    Raises:
        cryptography.exceptions.InvalidTag: 認証タグが不正な場合
    """
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, aad)


def main():
    # 256ビット鍵を安全に生成
    key = AESGCM.generate_key(bit_length=256)
    print(f"鍵 (hex): {key.hex()}")
    print(f"鍵長: {len(key) * 8} bits")

    # 平文と関連データ
    plaintext = "暗号学は情報セキュリティの基盤である。".encode("utf-8")
    aad = b"metadata:version=1"

    # 暗号化
    nonce, ciphertext = encrypt_aes_gcm(plaintext, key, aad)
    print(f"\nノンス (hex): {nonce.hex()}")
    print(f"暗号文 (hex): {ciphertext.hex()}")
    print(f"暗号文長: {len(ciphertext)} bytes (平文 {len(plaintext)} + タグ 16)")

    # 復号
    decrypted = decrypt_aes_gcm(nonce, ciphertext, key, aad)
    print(f"\n復号結果: {decrypted.decode('utf-8')}")
    assert decrypted == plaintext, "復号結果が元の平文と一致しない"

    # 改竄検知のデモ: 暗号文を1バイト変更
    tampered = bytearray(ciphertext)
    tampered[0] ^= 0xFF  # 最初のバイトを反転
    try:
        decrypt_aes_gcm(nonce, bytes(tampered), key, aad)
        print("エラー: 改竄が検知されなかった")
    except Exception as e:
        print(f"\n改竄検知成功: {type(e).__name__}")
        print("→ GCM の認証タグにより、暗号文の改竄を検出した")


if __name__ == "__main__":
    main()
```

```
実行結果の例:

  鍵 (hex): a1b2c3d4e5f6...（64文字の16進数）
  鍵長: 256 bits

  ノンス (hex): 1a2b3c4d5e6f7a8b9c0d1e2f
  暗号文 (hex): 8f3a2b1c4d5e...（平文+16バイトのタグ）
  暗号文長: 70 bytes (平文 54 + タグ 16)

  復号結果: 暗号学は情報セキュリティの基盤である。

  改竄検知成功: InvalidTag
  → GCM の認証タグにより、暗号文の改竄を検出した
```

---

## 3. 公開鍵暗号（非対称鍵暗号）

公開鍵暗号（public-key cryptography / asymmetric cryptography）は、1976年にホイットフィールド・ディフィーとマーティン・ヘルマンが発表した革命的な概念である。暗号化と復号に異なる鍵を使用し、共通鍵暗号の最大の課題であった「鍵配送問題」を解決した。

### 3.1 鍵配送問題と公開鍵暗号の誕生

共通鍵暗号では、通信相手と安全に同じ鍵を共有する必要がある。しかし、安全な通信路がないからこそ暗号を使いたいのであり、これは鶏と卵の問題である。

```
鍵配送問題:

  n 人が相互に暗号通信するために必要な鍵の数:

  共通鍵暗号: n(n-1)/2 個の鍵が必要
    2人 →   1鍵
    10人 →  45鍵
    100人 → 4,950鍵
    1000人 → 499,500鍵

  公開鍵暗号: 各人が鍵ペア1組（公開鍵+秘密鍵）を持てばよい
    2人 →   2鍵ペア
    10人 →  10鍵ペア
    100人 → 100鍵ペア
    1000人 → 1,000鍵ペア

  → 鍵管理の計算量が O(n^2) から O(n) に改善
```

### 3.2 RSA 暗号

RSA は 1977 年に Ron Rivest、Adi Shamir、Leonard Adleman の3名が発表した最初の実用的な公開鍵暗号アルゴリズムであり、大きな整数の素因数分解の困難性に安全性の根拠を置く。

```
RSA の鍵生成と暗号化（概要）:

  鍵生成:
  1. 大きな素数 p, q を選ぶ（各1024ビット以上）
  2. n = p × q を計算
  3. φ(n) = (p-1)(q-1) を計算（オイラーのトーシェント関数）
  4. gcd(e, φ(n)) = 1 を満たす e を選ぶ（通常 e = 65537）
  5. e × d ≡ 1 (mod φ(n)) を満たす d を求める

  公開鍵: (n, e)
  秘密鍵: (n, d)

  暗号化: c = m^e mod n
  復号:   m = c^d mod n

  安全性の根拠:
  - n から p, q を求めること（素因数分解）が計算的に困難
  - RSA-2048: n は約617桁の整数
  - 現在の古典コンピュータでは分解不可能
```

#### RSA の数値例（教育用の小さな値）

```
RSA の数値例:

  1. 素数の選択: p = 61, q = 53
  2. n = 61 × 53 = 3233
  3. φ(n) = (61-1)(53-1) = 60 × 52 = 3120
  4. e = 17  (gcd(17, 3120) = 1 を確認)
  5. d = 2753  (17 × 2753 = 46801 = 15 × 3120 + 1)

  公開鍵: (3233, 17)
  秘密鍵: (3233, 2753)

  暗号化 (m = 65 = 'A'):
    c = 65^17 mod 3233 = 2790

  復号:
    m = 2790^2753 mod 3233 = 65

  ※ 実際の RSA では n は2048ビット（617桁）以上を使用
```

### 3.3 楕円曲線暗号（ECC）

楕円曲線暗号（Elliptic Curve Cryptography）は、楕円曲線上の離散対数問題の困難性に基づく暗号方式である。RSA と比較して同等の安全性をはるかに短い鍵長で実現でき、計算効率も良いため、現代のシステムで広く採用されている。

```
楕円曲線の方程式:

  y^2 = x^3 + ax + b  (mod p)

  曲線上の点の加算（P + Q = R）:
  - 2点 P, Q を通る直線と曲線の第3の交点を x軸で反転

  スカラー倍算:
  - nP = P + P + ... + P（n回の加算）

  楕円曲線離散対数問題（ECDLP）:
  - Q = nP が与えられた時、n を求めることが計算的に困難
  - これが安全性の根拠

  鍵長の比較:
  ┌─────────────┬──────────┬──────────┬────────────────┐
  │ セキュリティ │ RSA鍵長  │ ECC鍵長  │ 比率           │
  │ レベル       │          │          │ (RSA/ECC)      │
  ├─────────────┼──────────┼──────────┼────────────────┤
  │  80 bit      │ 1024 bit │ 160 bit  │ 6.4倍          │
  │ 112 bit      │ 2048 bit │ 224 bit  │ 9.1倍          │
  │ 128 bit      │ 3072 bit │ 256 bit  │ 12倍           │
  │ 192 bit      │ 7680 bit │ 384 bit  │ 20倍           │
  │ 256 bit      │15360 bit │ 521 bit  │ 29.5倍         │
  └─────────────┴──────────┴──────────┴────────────────┘

  代表的な曲線:
  - P-256 (secp256r1): NIST 推奨、TLS で広く使用
  - Curve25519: Daniel Bernstein 設計、SSH/Signal で使用
  - secp256k1: Bitcoin で使用
```

### 3.4 Diffie-Hellman 鍵交換

Diffie-Hellman（DH）鍵交換は、安全でない通信路上で2者が共通の秘密鍵を共有するためのプロトコルである。暗号化そのものを行うのではなく、共通鍵暗号で使う鍵の素材を安全に生成する仕組みである。

```
Diffie-Hellman 鍵交換:

  公開パラメータ: 素数 p, 生成元 g

  Alice                                   Bob
  ┌───────────────────┐                 ┌───────────────────┐
  │ 秘密の値 a を選択  │                 │ 秘密の値 b を選択  │
  │ A = g^a mod p      │                 │ B = g^b mod p      │
  │                    │                 │                    │
  │         ─── A を送信 ──────→         │
  │         ←── B を送信 ──────          │
  │                    │                 │                    │
  │ s = B^a mod p      │                 │ s = A^b mod p      │
  │   = (g^b)^a mod p  │                 │   = (g^a)^b mod p  │
  │   = g^(ab) mod p   │                 │   = g^(ab) mod p   │
  └───────────────────┘                 └───────────────────┘

  共通秘密: s = g^(ab) mod p（一致する）

  攻撃者は A = g^a mod p と B = g^b mod p を知っていても、
  a や b を効率的に求められない（離散対数問題）

  現代の実装: ECDH（楕円曲線 Diffie-Hellman）
  → TLS 1.3 の鍵交換は ECDH（X25519 または P-256）を使用
```

### 3.5 Python で学ぶ公開鍵暗号

以下のコードは、楕円曲線 Diffie-Hellman（ECDH）による鍵交換のデモである。

```python
"""
ECDH 鍵交換と HKDF による鍵導出のデモ
依存ライブラリ: pip install cryptography
"""
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def ecdh_key_exchange():
    """
    X25519 を使用した ECDH 鍵交換のデモ。
    Alice と Bob が安全でない通信路上で共通鍵を生成する。
    """
    # Alice の鍵ペア生成
    alice_private = X25519PrivateKey.generate()
    alice_public = alice_private.public_key()

    # Bob の鍵ペア生成
    bob_private = X25519PrivateKey.generate()
    bob_public = bob_private.public_key()

    # 鍵交換: 相手の公開鍵と自分の秘密鍵から共有秘密を導出
    alice_shared = alice_private.exchange(bob_public)
    bob_shared = bob_private.exchange(alice_public)

    # 両者の共有秘密が一致することを確認
    assert alice_shared == bob_shared, "共有秘密が一致しない"
    print(f"共有秘密 (hex): {alice_shared.hex()}")
    print(f"共有秘密の長さ: {len(alice_shared)} bytes (256 bits)")

    # HKDF で共有秘密から暗号鍵を導出
    # 共有秘密をそのまま鍵として使うのではなく、KDF を通すのがベストプラクティス
    derived_key_alice = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(alice_shared)

    derived_key_bob = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"handshake data",
    ).derive(bob_shared)

    assert derived_key_alice == derived_key_bob
    print(f"\n導出鍵 (hex): {derived_key_alice.hex()}")
    print(f"導出鍵の長さ: {len(derived_key_alice)} bytes")
    print("→ この鍵を AES-256-GCM 等の共通鍵暗号で使用する")


if __name__ == "__main__":
    ecdh_key_exchange()
```

```
実行結果の例:

  共有秘密 (hex): 3a1b4c2d5e6f...（64文字の16進数）
  共有秘密の長さ: 32 bytes (256 bits)

  導出鍵 (hex): 7f8e9d0a1b2c...（64文字の16進数）
  導出鍵の長さ: 32 bytes
  → この鍵を AES-256-GCM 等の共通鍵暗号で使用する
```

---

## 4. ハッシュ関数

暗号学的ハッシュ関数（cryptographic hash function）は、任意長の入力データを固定長のハッシュ値（ダイジェスト）に変換する一方向関数である。暗号化とは異なり、ハッシュ値から元のデータを復元することは計算的に不可能であり、データの完全性検証、パスワード保管、デジタル署名の基盤として広く使われる。

### 4.1 ハッシュ関数の必須性質

暗号学的ハッシュ関数が満たすべき3つの主要な性質がある。

```
ハッシュ関数の3つの安全性要件:

  1. 原像耐性（preimage resistance）
     ハッシュ値 h が与えられた時、H(m) = h を満たす m を
     見つけることが計算的に困難

     h → m を見つけられない（一方向性）

  2. 第二原像耐性（second preimage resistance）
     メッセージ m1 が与えられた時、H(m1) = H(m2) を満たす
     m1 ≠ m2 な m2 を見つけることが計算的に困難

     m1 が既知でも、同じハッシュの m2 を見つけられない

  3. 衝突耐性（collision resistance）
     H(m1) = H(m2) かつ m1 ≠ m2 を満たす任意の
     (m1, m2) の組を見つけることが計算的に困難

     ※ 誕生日攻撃: n ビットハッシュの衝突を O(2^(n/2)) で発見可能
       → SHA-256 の衝突耐性は 128 ビット相当

  追加の望ましい性質:
  - 雪崩効果（avalanche effect）: 入力の1ビット変化で出力の約50%が変化
  - 効率性: ハッシュ計算が高速であること
```

### 4.2 主要なハッシュアルゴリズム

```
ハッシュアルゴリズムの比較:

  ┌──────────┬────────────┬──────────┬──────────────────────────┐
  │ アルゴリズム │ 出力長     │ 状態     │ 主な用途                 │
  ├──────────┼────────────┼──────────┼──────────────────────────┤
  │ MD5      │ 128 bit    │ 非推奨   │ レガシーシステム         │
  │ SHA-1    │ 160 bit    │ 非推奨   │ Git（互換性理由で残存）  │
  │ SHA-256  │ 256 bit    │ 推奨     │ TLS, Bitcoin, 汎用       │
  │ SHA-384  │ 384 bit    │ 推奨     │ 高セキュリティ要件       │
  │ SHA-512  │ 512 bit    │ 推奨     │ 高セキュリティ要件       │
  │ SHA3-256 │ 256 bit    │ 推奨     │ SHA-2 の代替             │
  │ BLAKE3   │ 256 bit    │ 推奨     │ 高速用途                 │
  └──────────┴────────────┴──────────┴──────────────────────────┘

  非推奨の理由:
  - MD5: 2004年に衝突攻撃が実証。数秒で衝突ペアを生成可能
  - SHA-1: 2017年に Google が SHAttered 攻撃で衝突を実証
```

#### SHA-256 の構造（Merkle-Damgard 構造）

```
SHA-256 の処理フロー:

  入力メッセージ
       ↓
  パディング（メッセージ長が 512 の倍数になるよう調整）
       ↓
  512ビットブロックに分割: M1, M2, ..., Mn
       ↓
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  IV ─→ [圧縮関数] ─→ [圧縮関数] ─→ ... ─→ ハッシュ値
  │            ↑              ↑                  │
  │           M1             M2                  │
  │                                              │
  │  Merkle-Damgard 構造:                        │
  │  H0 = IV                                     │
  │  Hi = f(H_{i-1}, Mi)  (圧縮関数)             │
  │  ハッシュ値 = Hn                              │
  └──────────────────────────────────────────────┘

  SHA-256 圧縮関数:
  - 8つの32ビットワーキング変数 (a, b, c, d, e, f, g, h)
  - 64ラウンドの演算
  - 各ラウンドで加算、ローテーション、論理演算を使用
```

### 4.3 パスワードハッシュ

パスワードの保存には、汎用ハッシュ関数（SHA-256 など）ではなく、専用のパスワードハッシュ関数を使用しなければならない。汎用ハッシュ関数は「高速であること」が求められるが、パスワードハッシュでは逆に「意図的に低速であること」が重要である。これは、攻撃者のブルートフォース攻撃を遅延させるためである。

```
パスワードハッシュの要件:

  NG: SHA-256(password)
  → GPU で毎秒数十億回のハッシュ計算が可能
  → レインボーテーブル（事前計算済みハッシュ辞書）で即座に解読

  NG: SHA-256(password + salt)
  → ソルトでレインボーテーブルは防げるが、高速すぎる

  OK: 専用パスワードハッシュ関数
  - bcrypt:  Blowfish ベース、コストパラメータで速度調整
  - scrypt:  メモリ消費を要求（GPU/ASIC 耐性）
  - Argon2:  2015年 PHC 優勝、最新推奨
    - Argon2id: サイドチャネル耐性 + GPU耐性（推奨）

  ┌────────────┬──────────────┬──────────────┬──────────────┐
  │            │ bcrypt       │ scrypt       │ Argon2id     │
  ├────────────┼──────────────┼──────────────┼──────────────┤
  │ 設計年     │ 1999         │ 2009         │ 2015         │
  │ CPU耐性    │ ○           │ ○           │ ◎           │
  │ GPU耐性    │ △           │ ○           │ ◎           │
  │ メモリ要求 │ 4KB固定      │ 可変         │ 可変         │
  │ 推奨       │ レガシー対応 │ 十分安全     │ 最推奨       │
  └────────────┴──────────────┴──────────────┴──────────────┘
```

### 4.4 HMAC（Hash-based Message Authentication Code）

HMAC はハッシュ関数と秘密鍵を組み合わせてメッセージ認証コード（MAC）を生成する方式である。データの完全性と真正性を同時に検証でき、API 認証（AWS Signature V4 など）で広く使われている。

```
HMAC の構造:

  HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))

  K' = 鍵（ブロックサイズに調整）
  opad = 0x5c を繰り返したブロック
  ipad = 0x36 を繰り返したブロック

  ┌─────────────────────────────┐
  │ 内部ハッシュ:               │
  │   inner = H((K' ⊕ ipad) || m) │
  │                             │
  │ 外部ハッシュ:               │
  │   HMAC = H((K' ⊕ opad) || inner) │
  └─────────────────────────────┘

  特徴:
  - 単純な H(K || m) や H(m || K) より安全
  - 長さ拡張攻撃（length extension attack）を防止
  - TLS, IPsec, JWT (HS256) 等で広く使用
```

### 4.5 Python で学ぶハッシュ関数

```python
"""
ハッシュ関数と HMAC のデモ
依存ライブラリ: 標準ライブラリのみ（hashlib, hmac）
"""
import hashlib
import hmac
import os


def hash_demo():
    """SHA-256 のハッシュ計算と雪崩効果のデモ。"""
    # 基本的なハッシュ計算
    message = "Hello, Cryptography!"
    hash_value = hashlib.sha256(message.encode()).hexdigest()
    print(f"入力:    '{message}'")
    print(f"SHA-256: {hash_value}")
    print(f"出力長:  {len(hash_value)} 文字 = {len(hash_value) * 4} bits")

    # 雪崩効果: 1文字だけ変更
    message_modified = "Hello, Cryptography?"  # '!' → '?'
    hash_modified = hashlib.sha256(message_modified.encode()).hexdigest()
    print(f"\n入力:    '{message_modified}'")
    print(f"SHA-256: {hash_modified}")

    # ビット単位で異なるビット数を計算
    original_bits = bin(int(hash_value, 16))[2:].zfill(256)
    modified_bits = bin(int(hash_modified, 16))[2:].zfill(256)
    diff_bits = sum(a != b for a, b in zip(original_bits, modified_bits))
    print(f"\n異なるビット数: {diff_bits} / 256 ({diff_bits/256*100:.1f}%)")
    print("→ 1文字の変更で約半数のビットが変化（雪崩効果）")


def hmac_demo():
    """HMAC-SHA256 の計算と検証のデモ。"""
    # HMAC 鍵の生成
    key = os.urandom(32)
    message = b"Transfer $100 to Bob"

    # HMAC 計算
    mac = hmac.new(key, message, hashlib.sha256).hexdigest()
    print(f"\nメッセージ: {message.decode()}")
    print(f"HMAC-SHA256: {mac}")

    # 検証: 正しいメッセージ
    mac_verify = hmac.new(key, message, hashlib.sha256).hexdigest()
    is_valid = hmac.compare_digest(mac, mac_verify)
    print(f"\n検証（正しいメッセージ）: {'OK' if is_valid else 'NG'}")

    # 検証: 改竄されたメッセージ
    tampered_message = b"Transfer $900 to Bob"
    mac_tampered = hmac.new(key, tampered_message, hashlib.sha256).hexdigest()
    is_valid_tampered = hmac.compare_digest(mac, mac_tampered)
    print(f"検証（改竄メッセージ）:   {'OK' if is_valid_tampered else 'NG（改竄検知）'}")

    # compare_digest はタイミング安全な比較を行う
    # 文字列の == 比較はタイミングサイドチャネル攻撃に脆弱
    print("\n注: hmac.compare_digest() はタイミング攻撃に安全な比較を使用")


def multiple_hash_algorithms():
    """複数のハッシュアルゴリズムの比較。"""
    data = b"The quick brown fox jumps over the lazy dog"
    algorithms = ["md5", "sha1", "sha256", "sha384", "sha512", "sha3_256"]

    print(f"\n入力: {data.decode()}\n")
    print(f"{'アルゴリズム':<12} {'出力長':>6}  ハッシュ値（先頭32文字）")
    print("-" * 70)

    for algo_name in algorithms:
        h = hashlib.new(algo_name, data)
        digest = h.hexdigest()
        bits = h.digest_size * 8
        print(f"{algo_name:<12} {bits:>4}bit  {digest[:32]}...")


if __name__ == "__main__":
    hash_demo()
    hmac_demo()
    multiple_hash_algorithms()
```

```
実行結果の例:

  入力:    'Hello, Cryptography!'
  SHA-256: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e7...
  出力長:  64 文字 = 256 bits

  入力:    'Hello, Cryptography?'
  SHA-256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd...

  異なるビット数: 131 / 256 (51.2%)
  → 1文字の変更で約半数のビットが変化（雪崩効果）

  メッセージ: Transfer $100 to Bob
  HMAC-SHA256: 5d41402abc4b2a76b9719d911017c592ae2...

  検証（正しいメッセージ）: OK
  検証（改竄メッセージ）:   NG（改竄検知）

  注: hmac.compare_digest() はタイミング攻撃に安全な比較を使用

  入力: The quick brown fox jumps over the lazy dog

  アルゴリズム   出力長  ハッシュ値（先頭32文字）
  ----------------------------------------------------------------------
  md5           128bit  9e107d9d372bb6826bd81d3542a419d6...
  sha1          160bit  2fd4e1c67a2d28fced849ee1bb76e739...
  sha256        256bit  d7a8fbb307d7809469ca9abcb0082e4f...
  sha384        384bit  ca737f1014a48f4c0b6dd43cb177b0af...
  sha512        512bit  07e547d9586f6a73f73fbac0435ed769...
  sha3_256      256bit  69070dda01975c8c120c3aada1b28239...
```

---

## 5. TLS/HTTPS の仕組み

TLS（Transport Layer Security）は、インターネット上の通信を暗号化し、盗聴・改竄・なりすましを防止するプロトコルである。Web ブラウジング（HTTPS）、メール（SMTPS/IMAPS）、VPN など、現代のインターネット通信のほぼ全てが TLS に依存している。

### 5.1 TLS の目的と提供する保証

TLS は以下の3つのセキュリティ目標を達成する。

```
TLS が提供する3つの保証:

  1. 機密性（Confidentiality）
     → 通信内容を第三者が読めない
     → AES-256-GCM / ChaCha20-Poly1305 で実現

  2. 完全性（Integrity）
     → 通信内容が改竄されていないことを保証
     → HMAC / AEAD の認証タグで実現

  3. 真正性（Authenticity）
     → 通信相手が本物であることを保証
     → X.509 証明書 + デジタル署名で実現

  プロトコルスタック上の位置:

  ┌─────────────────────┐
  │  HTTP / SMTP / ...  │  アプリケーション層
  ├─────────────────────┤
  │       TLS 1.3       │  ← ここで暗号化
  ├─────────────────────┤
  │        TCP          │  トランスポート層
  ├─────────────────────┤
  │        IP           │  ネットワーク層
  └─────────────────────┘
```

### 5.2 TLS 1.3 ハンドシェイク

TLS 1.3 は 2018 年に RFC 8446 として標準化された最新の TLS バージョンであり、TLS 1.2 から大幅に簡素化・高速化された。

```
TLS 1.3 フルハンドシェイク（1-RTT）:

  クライアント                              サーバー
  │                                        │
  │  ClientHello                           │
  │  ├ 対応暗号スイート一覧                │
  │  ├ 対応グループ (X25519, P-256等)      │
  │  ├ key_share (ECDH 公開鍵)             │
  │  └ supported_versions (TLS 1.3)        │
  │─────────────────────────────────────→  │
  │                                        │
  │                         ServerHello    │
  │                  ├ 選択した暗号スイート │
  │                  └ key_share (ECDH公開鍵)│
  │  ←─────────────────────────────────────│
  │                                        │
  │  [ここで共有秘密を導出]                │
  │  shared_secret = ECDH(私鍵, 相手公開鍵)│
  │  → handshake_keys を導出               │
  │                                        │
  │  {EncryptedExtensions}                 │
  │  {Certificate}        (サーバー証明書) │
  │  {CertificateVerify}  (署名)           │
  │  {Finished}           (MAC)            │
  │  ←═══════════════════════════════════  │
  │  (以降 {} 内は暗号化されている)         │
  │                                        │
  │  {Finished}                            │
  │  ═══════════════════════════════════→  │
  │                                        │
  │  ←═══ 暗号化されたアプリケーションデータ ═══→ │
  │                                        │

  TLS 1.3 で削除されたもの（安全性向上）:
  - RSA 鍵交換（前方秘匿性がない）
  - 静的 DH（前方秘匿性がない）
  - CBC モード暗号（パディングオラクル攻撃のリスク）
  - RC4, DES, 3DES, MD5, SHA-1
  - 圧縮（CRIME 攻撃）
  - 再ネゴシエーション

  TLS 1.3 の暗号スイート（5つのみ）:
  - TLS_AES_256_GCM_SHA384
  - TLS_AES_128_GCM_SHA256
  - TLS_CHACHA20_POLY1305_SHA256
  - TLS_AES_128_CCM_SHA256
  - TLS_AES_128_CCM_8_SHA256
```

### 5.3 前方秘匿性（Forward Secrecy）

前方秘匿性（Perfect Forward Secrecy, PFS）とは、長期的な秘密鍵が将来漏洩しても、過去の通信内容が解読されないことを保証する性質である。

```
前方秘匿性の仕組み:

  RSA 鍵交換（TLS 1.2 以前、前方秘匿性なし）:
  ┌────────────────────────────────────────┐
  │ クライアントがプリマスタシークレットを   │
  │ サーバーの RSA 公開鍵で暗号化して送信   │
  │                                        │
  │ 問題: サーバーの秘密鍵が漏洩すると、   │
  │ 過去に記録した全通信を復号可能          │
  └────────────────────────────────────────┘

  ECDHE 鍵交換（TLS 1.3、前方秘匿性あり）:
  ┌────────────────────────────────────────┐
  │ 毎回の接続で新しい一時的な ECDH 鍵ペア │
  │ を生成して鍵交換                        │
  │                                        │
  │ 利点: 長期秘密鍵が漏洩しても、         │
  │ 各セッション固有の一時鍵は復元不可能   │
  │ → 過去の通信は安全                     │
  └────────────────────────────────────────┘

  "E" = Ephemeral（一時的）
  ECDHE の "E" が前方秘匿性を実現
```

### 5.4 X.509 証明書と信頼の連鎖

TLS では X.509 証明書を使用してサーバーの身元を検証する。証明書は認証局（CA: Certificate Authority）によって署名され、信頼の連鎖（Chain of Trust）を形成する。

```
証明書チェーン（信頼の連鎖）:

  ┌───────────────────────────────┐
  │  ルート CA 証明書              │ ← OS/ブラウザに事前搭載
  │  (自己署名)                   │    約100-150の信頼済みルート
  │  例: DigiCert Global Root G2  │
  └───────────┬───────────────────┘
              │ 署名
              ↓
  ┌───────────────────────────────┐
  │  中間 CA 証明書               │ ← ルート CA が署名
  │  例: DigiCert SHA2 Secure     │
  │       Server CA               │
  └───────────┬───────────────────┘
              │ 署名
              ↓
  ┌───────────────────────────────┐
  │  サーバー証明書（リーフ証明書）│ ← 中間 CA が署名
  │  例: www.example.com          │
  │  含まれる情報:                 │
  │  - ドメイン名（SAN）          │
  │  - 公開鍵                     │
  │  - 有効期間                   │
  │  - 発行者（中間CA）の署名     │
  └───────────────────────────────┘

  検証プロセス:
  1. サーバー証明書の署名を中間 CA の公開鍵で検証
  2. 中間 CA 証明書の署名をルート CA の公開鍵で検証
  3. ルート CA がトラストストアに存在することを確認
  4. 証明書が有効期限内であることを確認
  5. 証明書が失効していないことを確認（CRL/OCSP）
  6. ドメイン名がリクエストと一致することを確認
```

### 5.5 0-RTT 再接続

TLS 1.3 では、過去に接続したサーバーへの再接続時に 0-RTT（Zero Round Trip Time）で暗号化データを送信できる機能がある。

```
0-RTT 再接続:

  初回接続後にサーバーから PSK（Pre-Shared Key）を受信

  再接続時:
  クライアント                     サーバー
  │  ClientHello                  │
  │  + early_data (0-RTT データ)  │
  │────────────────────────────→  │  ← 最初のメッセージで
  │                               │     アプリデータ送信可能
  │         ServerHello           │
  │  ←────────────────────────── │
  │                               │
  │  ←══ 暗号化通信 ══→           │

  利点: レイテンシ削減（初回メッセージでデータ送信）
  リスク: リプレイ攻撃の可能性
  → 冪等な操作（GET リクエスト等）にのみ使用すべき
```

---

## 6. デジタル署名

デジタル署名は、メッセージの真正性（送信者の身元確認）と完全性（改竄がないこと）を数学的に保証する技術である。手書きの署名と異なり、メッセージ内容に依存するため偽造が極めて困難であり、否認防止（non-repudiation）を実現する。

### 6.1 デジタル署名の仕組み

```
デジタル署名の処理フロー:

  署名（送信者）:
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  メッセージ M                                │
  │       ↓                                      │
  │  ハッシュ計算: h = H(M)                       │
  │       ↓                                      │
  │  秘密鍵で署名: σ = Sign(秘密鍵, h)           │
  │       ↓                                      │
  │  (M, σ) を送信                               │
  │                                              │
  └──────────────────────────────────────────────┘

  検証（受信者）:
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  (M, σ) を受信                               │
  │       ↓                                      │
  │  ハッシュ計算: h = H(M)                       │
  │       ↓                                      │
  │  公開鍵で検証: Verify(公開鍵, h, σ) → true/false │
  │       ↓                                      │
  │  true: 署名は正当（送信者が本物 & 未改竄）    │
  │  false: 署名は不正（偽造 or 改竄）            │
  │                                              │
  └──────────────────────────────────────────────┘

  暗号化との違い:
  - 暗号化: 公開鍵で暗号化 → 秘密鍵で復号
  - デジタル署名: 秘密鍵で署名 → 公開鍵で検証
  → 鍵の使用方向が逆
```

### 6.2 主要な署名アルゴリズム

```
署名アルゴリズムの比較:

  ┌────────────┬──────────────┬───────────┬──────────────────┐
  │ アルゴリズム │ 安全性根拠   │ 鍵長      │ 用途             │
  ├────────────┼──────────────┼───────────┼──────────────────┤
  │ RSA-PSS    │ 素因数分解   │ 2048+ bit │ TLS, コード署名  │
  │ ECDSA      │ ECDLP        │ 256 bit   │ TLS, Bitcoin     │
  │ Ed25519    │ ECDLP        │ 256 bit   │ SSH, Signal      │
  │ Ed448      │ ECDLP        │ 448 bit   │ 高セキュリティ   │
  └────────────┴──────────────┴───────────┴──────────────────┘

  Ed25519 の特徴:
  - Curve25519 上の Edwards 曲線を使用
  - 決定論的署名（同じメッセージ+鍵で常に同じ署名）
    → 乱数生成の品質に依存しない（ECDSA の脆弱性回避）
  - 高速: 署名 ~50μs, 検証 ~100μs（一般的なCPU）
  - コンパクト: 署名 64バイト, 公開鍵 32バイト
```

### 6.3 Python で学ぶデジタル署名

```python
"""
Ed25519 デジタル署名のデモ
依存ライブラリ: pip install cryptography
"""
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.exceptions import InvalidSignature


def digital_signature_demo():
    """Ed25519 による署名と検証のデモ。"""
    # 鍵ペア生成
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    # 公開鍵のバイト表現（32バイト）
    from cryptography.hazmat.primitives.serialization import (
        Encoding, PublicFormat,
    )
    pub_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    print(f"公開鍵 (hex): {pub_bytes.hex()}")
    print(f"公開鍵の長さ: {len(pub_bytes)} bytes")

    # メッセージに署名
    message = "この文書は改竄されていません。".encode("utf-8")
    signature = private_key.sign(message)
    print(f"\nメッセージ: {message.decode()}")
    print(f"署名 (hex): {signature.hex()}")
    print(f"署名の長さ: {len(signature)} bytes")

    # 署名の検証（正常系）
    try:
        public_key.verify(signature, message)
        print("\n署名検証: OK（正当な署名）")
    except InvalidSignature:
        print("\n署名検証: NG")

    # 署名の検証（改竄検知）
    tampered_message = "この文書は改竄されています。".encode("utf-8")
    try:
        public_key.verify(signature, tampered_message)
        print("署名検証: OK（改竄が検知されなかった — エラー）")
    except InvalidSignature:
        print("署名検証: NG（改竄を検知 — 正常動作）")

    # 別の秘密鍵で署名した場合（なりすまし検知）
    fake_private_key = Ed25519PrivateKey.generate()
    fake_signature = fake_private_key.sign(message)
    try:
        public_key.verify(fake_signature, message)
        print("署名検証: OK（なりすましが検知されなかった — エラー）")
    except InvalidSignature:
        print("署名検証: NG（なりすましを検知 — 正常動作）")


if __name__ == "__main__":
    digital_signature_demo()
```

```
実行結果の例:

  公開鍵 (hex): 7d4a3b2c1e0f...（64文字の16進数）
  公開鍵の長さ: 32 bytes

  メッセージ: この文書は改竄されていません。
  署名 (hex): 8f3a2b1c4d5e6f...（128文字の16進数）
  署名の長さ: 64 bytes

  署名検証: OK（正当な署名）
  署名検証: NG（改竄を検知 — 正常動作）
  署名検証: NG（なりすましを検知 — 正常動作）
```

### 6.4 デジタル署名の応用

デジタル署名は単なるメッセージ認証を超え、現代のソフトウェアエコシステム全体に浸透している。

```
デジタル署名の主要な応用:

  1. コード署名（Code Signing）
     - OS がアプリケーションの出自を検証
     - Apple: コード署名必須、公証（Notarization）
     - Windows: Authenticode 署名
     - Android: APK 署名

  2. パッケージマネージャ
     - apt/yum: GPG 署名でリポジトリの真正性を検証
     - npm/PyPI: Sigstore による署名が普及中
     - Docker: コンテナイメージの署名（Cosign/Notation）

  3. Git コミット署名
     - GPG または SSH 鍵でコミットに署名
     - GitHub: "Verified" バッジの表示
     - git commit -S -m "signed commit"

  4. 電子契約・電子署名
     - PDF 文書のデジタル署名
     - 各国の電子署名法に基づく法的効力

  5. ブロックチェーン
     - トランザクションの署名（ECDSA / Ed25519）
     - Bitcoin: secp256k1 曲線の ECDSA
     - Ethereum: 同上 + EIP-712 typed data signing

  6. JWT（JSON Web Token）
     - RS256: RSA-PSS + SHA-256
     - ES256: ECDSA P-256 + SHA-256
     - EdDSA: Ed25519（RFC 8037）
```

---

