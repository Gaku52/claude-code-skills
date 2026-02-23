# 文字コードとUnicode

> 文字化けの原因は常に「エンコーディングの不一致」であり、UTF-8の仕組みを理解すれば根本的に予防できる。

## この章で学ぶこと

- [ ] ASCII → Unicode → UTF-8 の進化を説明できる
- [ ] UTF-8のバイト構造を手計算で確認できる
- [ ] 文字化けの原因と対策を説明できる
- [ ] UTF-8、UTF-16、UTF-32の違いを説明できる
- [ ] Unicode正規化（NFC/NFD）の概念と実装を理解する
- [ ] 各プログラミング言語での文字列処理の注意点を把握する

## 前提知識

- 2進数と16進数 → 参照: [[00-binary-and-number-systems.md]]

---

## 1. 文字コードの歴史

### 1.1 ASCII（1963年）

```
ASCII: 7ビット = 128文字

  0x00-0x1F: 制御文字（改行LF, タブHT, NULL等）
  0x20:      スペース
  0x30-0x39: 数字 '0'-'9'
  0x41-0x5A: 大文字 'A'-'Z'
  0x61-0x7A: 小文字 'a'-'z'
  0x7F:      DEL

  特徴:
  - 英語圏のみ。日本語、中国語は表現不可能
  - 大文字と小文字の差は 0x20（ビット5の違い）
    'A' = 0x41 = 0100 0001
    'a' = 0x61 = 0110 0001
    →  差 = 0010 0000 = 0x20 = 32
  - 数字 '0'-'9' は 0x30-0x39（下位4ビットが数値そのもの）
```

### 1.2 ASCII完全テーブル

```
ASCII コードテーブル（全128文字）:

  制御文字（0x00-0x1F）:
  Dec  Hex  文字  説明
  ───  ───  ────  ────────────────
    0  0x00  NUL  Null（文字列終端）
    1  0x01  SOH  Start of Heading
    2  0x02  STX  Start of Text
    3  0x03  ETX  End of Text（Ctrl+C）
    4  0x04  EOT  End of Transmission（Ctrl+D）
    7  0x07  BEL  Bell（端末のベル音）
    8  0x08  BS   Backspace
    9  0x09  HT   Horizontal Tab
   10  0x0A  LF   Line Feed（Unix改行）
   11  0x0B  VT   Vertical Tab
   12  0x0C  FF   Form Feed（改ページ）
   13  0x0D  CR   Carriage Return（Mac旧改行）
   27  0x1B  ESC  Escape（ANSIエスケープシーケンスの開始）
   31  0x1F  US   Unit Separator

  改行コードの違い:
  Unix/Linux/macOS: LF (0x0A) = \n
  Windows: CR+LF (0x0D 0x0A) = \r\n
  旧Mac (OS 9以前): CR (0x0D) = \r

  印字可能文字（0x20-0x7E）:
  Dec  Hex  Char │ Dec  Hex  Char │ Dec  Hex  Char
  ───  ───  ──── │ ───  ───  ──── │ ───  ───  ────
   32  0x20  SP  │  64  0x40  @   │  96  0x60  `
   33  0x21  !   │  65  0x41  A   │  97  0x61  a
   34  0x22  "   │  66  0x42  B   │  98  0x62  b
   35  0x23  #   │  67  0x43  C   │  99  0x63  c
   36  0x24  $   │  68  0x44  D   │ 100  0x64  d
   37  0x25  %   │  69  0x45  E   │ 101  0x65  e
   38  0x26  &   │  70  0x46  F   │ 102  0x66  f
   39  0x27  '   │  71  0x47  G   │ 103  0x67  g
   40  0x28  (   │  72  0x48  H   │ 104  0x68  h
   41  0x29  )   │  73  0x49  I   │ 105  0x69  i
   42  0x2A  *   │  74  0x4A  J   │ 106  0x6A  j
   43  0x2B  +   │  75  0x4B  K   │ 107  0x6B  k
   44  0x2C  ,   │  76  0x4C  L   │ 108  0x6C  l
   45  0x2D  -   │  77  0x4D  M   │ 109  0x6D  m
   46  0x2E  .   │  78  0x4E  N   │ 110  0x6E  n
   47  0x2F  /   │  79  0x4F  O   │ 111  0x6F  o
   48  0x30  0   │  80  0x50  P   │ 112  0x70  p
   49  0x31  1   │  81  0x51  Q   │ 113  0x71  q
   50  0x32  2   │  82  0x52  R   │ 114  0x72  r
   51  0x33  3   │  83  0x53  S   │ 115  0x73  s
   52  0x34  4   │  84  0x54  T   │ 116  0x74  t
   53  0x35  5   │  85  0x55  U   │ 117  0x75  u
   54  0x36  6   │  86  0x56  V   │ 118  0x76  v
   55  0x37  7   │  87  0x57  W   │ 119  0x77  w
   56  0x38  8   │  88  0x58  X   │ 120  0x78  x
   57  0x39  9   │  89  0x59  Y   │ 121  0x79  y
   58  0x3A  :   │  90  0x5A  Z   │ 122  0x7A  z
   59  0x3B  ;   │  91  0x5B  [   │ 123  0x7B  {
   60  0x3C  <   │  92  0x5C  \   │ 124  0x7C  |
   61  0x3D  =   │  93  0x5D  ]   │ 125  0x7D  }
   62  0x3E  >   │  94  0x5E  ^   │ 126  0x7E  ~
   63  0x3F  ?   │  95  0x5F  _   │ 127  0x7F  DEL

  実務で重要なASCIIの性質:
  - 'A'-'Z': 0x41-0x5A (ビット5=0)
  - 'a'-'z': 0x61-0x7A (ビット5=1)
  - 大小変換: c ^ 0x20 でトグル
  - '0'-'9': 0x30-0x39 (c - 0x30 で数値に変換)
  - 印字可能文字: 0x20-0x7E の範囲
```

### 1.3 ASCIIの設計思想

```
ASCIIが7ビットである理由:

  1960年代のテレタイプ通信:
  - 5ビット（Baudotコード）: 32文字しか表現不可
  - 6ビット: 64文字（BCDIC等）
  - 7ビット: 128文字 → 英語の全文字 + 制御文字 + 記号
  - 8ビット: 256文字 → 当時は「贅沢」と判断

  残りの1ビット（8ビット目）:
  - パリティチェックに使用（通信エラー検出）
  - 後に ISO 8859 等の拡張文字セットに活用

  大文字/小文字の設計が巧妙な理由:
  - ビット5のON/OFFで切り替え可能
  - ハードウェアで容易に変換可能
  - アルファベット順 ≈ コードの昇順

  制御文字の設計:
  - Ctrl+キー = キーのASCIIコード - 0x40
    Ctrl+A = 0x41 - 0x40 = 0x01 (SOH)
    Ctrl+C = 0x43 - 0x40 = 0x03 (ETX → 割り込み信号)
    Ctrl+D = 0x44 - 0x40 = 0x04 (EOT → EOF)
    Ctrl+G = 0x47 - 0x40 = 0x07 (BEL → ベル音)
    Ctrl+H = 0x48 - 0x40 = 0x08 (BS → バックスペース)
    Ctrl+I = 0x49 - 0x40 = 0x09 (HT → タブ)
    Ctrl+J = 0x4A - 0x40 = 0x0A (LF → 改行)
    Ctrl+M = 0x4D - 0x40 = 0x0D (CR → 復帰)
```

### 1.4 文字コードの混乱期

| コード | 年代 | 対象 | 問題 |
|--------|------|------|------|
| ASCII | 1963 | 英語 | 128文字のみ |
| Latin-1 (ISO-8859-1) | 1987 | 西欧 | 256文字、日本語不可 |
| Shift_JIS | 1982 | 日本語 | Windows標準。可変長。他言語と混在困難 |
| EUC-JP | 1985 | 日本語 | UNIX標準。Shift_JISと互換性なし |
| ISO-2022-JP | 1994 | 日本語 | メール標準。エスケープシーケンスで切替 |
| GB2312/GBK | 1980 | 中国語 | 日本語と互換性なし |
| Big5 | 1984 | 繁体字 | 簡体字と互換性なし |
| KS X 1001 | 1986 | 韓国語 | EUC-KRとして使用 |

→ **各言語・地域ごとに独自の文字コード** → **相互運用性の悪夢**

### 1.5 日本語文字コードの詳細

```
【JIS X 0208（JIS第一・第二水準）】
  - 6,879文字（漢字4,888 + 非漢字1,991）
  - 94×94のマトリクスで管理
  - 区点コード: 区番号(1-94) × 点番号(1-94)

【Shift_JIS（MS漢字コード）】
  構造:
  - 1バイト文字: 0x00-0x7F (ASCII) + 0xA1-0xDF (半角カナ)
  - 2バイト文字: 第1バイト 0x81-0x9F, 0xE0-0xEF
                  第2バイト 0x40-0x7E, 0x80-0xFC

  問題点:
  - 「表」(0x955C) の第2バイト 0x5C = '\' → パス区切りと衝突
  - 「ソ」「能」等で同様の問題（ダメ文字問題）
  - 半角カナがASCIIの上位バイトと競合

  例: "表示" のバイト列
  表 = 0x95 0x5C  (第2バイトが 0x5C = バックスラッシュ)
  示 = 0x8E 0xA6
  → C言語の文字列 "表示" でバックスラッシュがエスケープ文字に

【EUC-JP（Extended Unix Code）】
  構造:
  - 1バイト文字: 0x00-0x7F (ASCII)
  - 2バイト文字: 各バイト 0xA1-0xFE
  - 3バイト文字: 0x8F + 2バイト（JIS第三水準）

  利点:
  - ASCIIとの衝突がない（上位バイトのみ使用）
  - Unix系OSで長年標準

【ISO-2022-JP（JISコード）】
  構造:
  - エスケープシーケンスでASCII/日本語を切替
  - ASCII開始: ESC ( B
  - JIS X 0208開始: ESC $ B
  - 7ビット領域のみ使用（メール送信に適していた）

  例: "ABCあいう" のエンコード
  ESC ( B → ASCII モード → 41 42 43
  ESC $ B → JIS モード → 24 22 24 24 24 26
  ESC ( B → ASCII モードに復帰
```

### 1.6 Unicode の誕生（1991年）

Unicode の理念: **全ての言語の全ての文字に一意のコードポイントを割り当てる**

```
Unicode のコードポイント:

  U+0041  = 'A' (ラテン文字)
  U+3042  = 'あ' (ひらがな)
  U+4E16  = '世' (CJK漢字)
  U+1F600 = '😀' (絵文字)
  U+1F4A9 = '💩' (うんち絵文字)

  範囲: U+0000 〜 U+10FFFF（約111万コードポイント）
  割り当て済み: 約15万文字（2024年時点）
  → 人類が使用する全ての文字を収容可能
```

### 1.7 Unicodeの面（Plane）構造

```
Unicode の17個の面（Plane 0-16）:

  面0: BMP（Basic Multilingual Plane）U+0000 - U+FFFF
    - 最も多く使われる文字の大部分がここに収容
    - ラテン文字、ひらがな、カタカナ、CJK漢字の主要部分
    - 65,536コードポイント

  面1: SMP（Supplementary Multilingual Plane）U+10000 - U+1FFFF
    - 絵文字（U+1F600-U+1F64F 等）
    - 楔形文字、エジプトヒエログリフ
    - 古代文字、音楽記号、数学記号

  面2: SIP（Supplementary Ideographic Plane）U+20000 - U+2FFFF
    - CJK統合漢字拡張B〜（稀少漢字）
    - 約42,711文字

  面3: TIP（Tertiary Ideographic Plane）U+30000 - U+3FFFF
    - CJK統合漢字拡張G〜

  面14: SSP（Supplementary Special-purpose Plane）U+E0000 - U+EFFFF
    - タグ文字、バリエーションセレクタ

  面15-16: PUA（Private Use Areas）
    - 私用領域（企業や個人が自由に定義可能）

  面の可視化:
  ┌───────────────────────────────────────────┐
  │ Plane 0 (BMP): ほぼ全ての現代文字          │
  │ U+0000-U+FFFF                             │
  │ ASCII, ラテン, ギリシャ, キリル,             │
  │ ひらがな, カタカナ, CJK漢字の大部分        │
  ├───────────────────────────────────────────┤
  │ Plane 1 (SMP): 絵文字, 古代文字            │
  │ U+10000-U+1FFFF                           │
  ├───────────────────────────────────────────┤
  │ Plane 2 (SIP): 稀少CJK漢字               │
  │ U+20000-U+2FFFF                           │
  ├───────────────────────────────────────────┤
  │ Plane 3-13: 大部分が未割り当て             │
  ├───────────────────────────────────────────┤
  │ Plane 14 (SSP): 特殊目的                   │
  ├───────────────────────────────────────────┤
  │ Plane 15-16: 私用領域                      │
  └───────────────────────────────────────────┘
```

---

## 2. UTF-8 — 現代の標準

### 2.1 UTF-8のバイト構造

```
UTF-8 エンコーディング規則:

  コードポイント範囲        バイト数  バイト列パターン
  ──────────────────────────────────────────────────
  U+0000  - U+007F         1バイト   0xxxxxxx
  U+0080  - U+07FF         2バイト   110xxxxx 10xxxxxx
  U+0800  - U+FFFF         3バイト   1110xxxx 10xxxxxx 10xxxxxx
  U+10000 - U+10FFFF       4バイト   11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

  先頭バイトの1の数 = バイト数
  継続バイトは必ず 10 で始まる
  → どのバイトからでも文字境界を特定できる（自己同期性）

  バイトの分類（先頭ビットパターン）:
  0xxxxxxx → 1バイト文字の先頭（ASCII互換）
  10xxxxxx → 継続バイト（先頭バイトではない）
  110xxxxx → 2バイト文字の先頭
  1110xxxx → 3バイト文字の先頭
  11110xxx → 4バイト文字の先頭

  自己同期性の意味:
  - 任意のバイトを見て、それが文字の先頭か継続かが即座に分かる
  - ストリームの途中からでも次の文字境界を見つけられる
  - 1バイト破損しても影響は最大1文字（UTF-16では最大2文字影響）
```

### 2.2 具体例（エンコード手順）

```
'A' = U+0041:
  2進数: 100 0001 (7ビット → 1バイトで収まる)
  UTF-8: 0_1000001 = 0x41 (1バイト、ASCIIと完全互換)

'¥' = U+00A5:
  2進数: 10 100101 (8ビット → 2バイト必要)
  テンプレート: 110xxxxx 10xxxxxx
  分割: 00010 100101
  埋め込み: 110_00010 10_100101
  結果: 0xC2 0xA5 (2バイト)

  検算: 00010_100101 = 0x0A5 = 165 = U+00A5 ✓

'あ' = U+3042:
  2進数: 0011 0000 0100 0010 (16ビット → 3バイト必要)
  テンプレート: 1110xxxx 10xxxxxx 10xxxxxx
  分割: 0011 000001 000010
  埋め込み: 1110_0011 10_000001 10_000010
  結果: 0xE3 0x81 0x82 (3バイト)

  エンコード手順:
  U+3042 = 0011 000001 000010
  テンプレート: 1110xxxx 10xxxxxx 10xxxxxx
  埋め込み:     1110_0011 10_000001 10_000010
  結果:         E3 81 82

'漢' = U+6F22:
  2進数: 0110 1111 0010 0010
  分割: 0110 111100 100010
  埋め込み: 1110_0110 10_111100 10_100010
  結果: 0xE6 0xBC 0xA2 (3バイト)

'😀' = U+1F600:
  2進数: 0001 1111 0110 0000 0000 (21ビット → 4バイト必要)
  テンプレート: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  分割: 000 011111 011000 000000
  埋め込み: 11110_000 10_011111 10_011000 10_000000
  結果: 0xF0 0x9F 0x98 0x80 (4バイト)

'𠮷' = U+20BB7（つちよし、JIS第三水準漢字）:
  2進数: 0010 0000 1011 1011 0111
  分割: 000 100000 101110 110111
  埋め込み: 11110_000 10_100000 10_101110 10_110111
  結果: 0xF0 0xA0 0xAE 0xB7 (4バイト)
```

### 2.3 UTF-8の利点

| 特性 | 説明 |
|------|------|
| ASCII互換 | ASCIIテキストはそのままUTF-8として有効 |
| 可変長 | 1-4バイト。英語はコンパクト、全言語をサポート |
| 自己同期 | 任意のバイト位置から文字境界を復元可能 |
| ソート可能 | バイト列の辞書順 ≈ コードポイントの昇順 |
| BOM不要 | エンディアン問題がない（UTF-16と異なり） |
| NUL安全 | U+0000以外に0x00バイトが出現しない（C文字列安全） |
| 広く普及 | Web の 98%以上が UTF-8 |

### 2.4 UTF-8のデコードアルゴリズム

```python
def utf8_decode_manual(byte_sequence):
    """UTF-8バイト列を手動でデコードする（教育用実装）"""
    result = []
    i = 0
    while i < len(byte_sequence):
        b = byte_sequence[i]

        if b < 0x80:
            # 1バイト文字 (0xxxxxxx)
            codepoint = b
            i += 1
        elif b < 0xC0:
            # 継続バイト (10xxxxxx) が先頭に来るのはエラー
            raise ValueError(f"不正な継続バイト: 0x{b:02X} at position {i}")
        elif b < 0xE0:
            # 2バイト文字 (110xxxxx 10xxxxxx)
            if i + 1 >= len(byte_sequence):
                raise ValueError("不完全な2バイト文字")
            codepoint = ((b & 0x1F) << 6) | (byte_sequence[i+1] & 0x3F)
            # オーバーロング検出: U+0080未満は1バイトで表すべき
            if codepoint < 0x80:
                raise ValueError(f"オーバーロング: U+{codepoint:04X}")
            i += 2
        elif b < 0xF0:
            # 3バイト文字 (1110xxxx 10xxxxxx 10xxxxxx)
            if i + 2 >= len(byte_sequence):
                raise ValueError("不完全な3バイト文字")
            codepoint = ((b & 0x0F) << 12) | \
                       ((byte_sequence[i+1] & 0x3F) << 6) | \
                       (byte_sequence[i+2] & 0x3F)
            if codepoint < 0x800:
                raise ValueError(f"オーバーロング: U+{codepoint:04X}")
            # サロゲートペア範囲は不正
            if 0xD800 <= codepoint <= 0xDFFF:
                raise ValueError(f"サロゲート: U+{codepoint:04X}")
            i += 3
        elif b < 0xF8:
            # 4バイト文字 (11110xxx 10xxxxxx 10xxxxxx 10xxxxxx)
            if i + 3 >= len(byte_sequence):
                raise ValueError("不完全な4バイト文字")
            codepoint = ((b & 0x07) << 18) | \
                       ((byte_sequence[i+1] & 0x3F) << 12) | \
                       ((byte_sequence[i+2] & 0x3F) << 6) | \
                       (byte_sequence[i+3] & 0x3F)
            if codepoint < 0x10000:
                raise ValueError(f"オーバーロング: U+{codepoint:04X}")
            if codepoint > 0x10FFFF:
                raise ValueError(f"範囲外: U+{codepoint:04X}")
            i += 4
        else:
            raise ValueError(f"不正な先頭バイト: 0x{b:02X}")

        result.append(chr(codepoint))

    return ''.join(result)


# テスト
test_bytes = bytes([0xE3, 0x81, 0x82])  # 'あ'
print(utf8_decode_manual(test_bytes))  # あ

test_bytes2 = bytes([0xF0, 0x9F, 0x98, 0x80])  # '😀'
print(utf8_decode_manual(test_bytes2))  # 😀


def utf8_encode_manual(text):
    """文字列をUTF-8バイト列に手動エンコード（教育用実装）"""
    result = bytearray()
    for char in text:
        cp = ord(char)
        if cp < 0x80:
            result.append(cp)
        elif cp < 0x800:
            result.append(0xC0 | (cp >> 6))
            result.append(0x80 | (cp & 0x3F))
        elif cp < 0x10000:
            result.append(0xE0 | (cp >> 12))
            result.append(0x80 | ((cp >> 6) & 0x3F))
            result.append(0x80 | (cp & 0x3F))
        else:
            result.append(0xF0 | (cp >> 18))
            result.append(0x80 | ((cp >> 12) & 0x3F))
            result.append(0x80 | ((cp >> 6) & 0x3F))
            result.append(0x80 | (cp & 0x3F))
    return bytes(result)

# テスト
print(utf8_encode_manual("あ").hex())  # e38182
print(utf8_encode_manual("😀").hex())  # f09f9880
```

---

## 3. UTF-16 と UTF-32

### 3.1 UTF-16

```
UTF-16 エンコーディング:

  BMP（U+0000-U+FFFF）: 2バイトでそのまま
  SMP以上（U+10000-U+10FFFF）: サロゲートペア（4バイト）

  サロゲートペアの計算:
  1. コードポイントから 0x10000 を引く（20ビット値になる）
  2. 上位10ビット + 0xD800 → 上位サロゲート（0xD800-0xDBFF）
  3. 下位10ビット + 0xDC00 → 下位サロゲート（0xDC00-0xDFFF）

  例: '😀' U+1F600
  1. 0x1F600 - 0x10000 = 0x0F600
  2. 上位10ビット: 0x0F600 >> 10 = 0x003D → + 0xD800 = 0xD83D
  3. 下位10ビット: 0x0F600 & 0x3FF = 0x0200 → + 0xDC00 = 0xDE00
  4. UTF-16: 0xD83D 0xDE00

  エンディアンの問題:
  UTF-16BE: 0xD8 0x3D 0xDE 0x00（ビッグエンディアン）
  UTF-16LE: 0x3D 0xD8 0x00 0xDE（リトルエンディアン）
  → BOM (U+FEFF) で判定:
    FF FE → リトルエンディアン
    FE FF → ビッグエンディアン

  UTF-16を使用する環境:
  - Windows API (WCHAR, wchar_t)
  - Java (char型, String)
  - JavaScript (String)
  - .NET (System.String)
  - macOS/iOS (NSString / CFString の内部表現)
```

```python
# サロゲートペアの計算

def codepoint_to_utf16(cp):
    """コードポイントからUTF-16エンコーディングを計算"""
    if cp < 0x10000:
        return [cp]
    else:
        cp -= 0x10000
        high = 0xD800 + (cp >> 10)
        low = 0xDC00 + (cp & 0x3FF)
        return [high, low]

def utf16_to_codepoint(units):
    """UTF-16コードユニットからコードポイントを復元"""
    if len(units) == 1:
        return units[0]
    else:
        high, low = units
        return ((high - 0xD800) << 10) + (low - 0xDC00) + 0x10000

# テスト
print([hex(u) for u in codepoint_to_utf16(0x1F600)])  # ['0xd83d', '0xde00']
print(hex(utf16_to_codepoint([0xD83D, 0xDE00])))      # 0x1f600
```

### 3.2 UTF-32

```
UTF-32 エンコーディング:

  全てのコードポイントを固定4バイトで表現
  → 最もシンプルだが、メモリ効率が最悪

  例:
  'A'  = 0x00000041
  'あ' = 0x00003042
  '😀' = 0x0001F600

  利点:
  - 固定長なのでインデックスアクセスが O(1)
  - 実装が極めてシンプル
  - コードポイントの直接比較が容易

  欠点:
  - ASCII文字でも4バイト（UTF-8の4倍）
  - メモリ使用量が多い
  - エンディアンの問題がある（UTF-32BE/UTF-32LE）
  - 実務ではほぼ使われない

  使用される場面:
  - Python 3の内部表現（コードポイントに依る: Latin-1/UCS-2/UCS-4の切替）
  - ICU（International Components for Unicode）の一部
  - テキスト処理ライブラリの内部
```

### 3.3 エンコーディング比較

```
各UTFの比較表:

  文字      UTF-8      UTF-16LE      UTF-32LE
  ─────── ────────── ──────────── ────────────
  'A'       41         41 00         41 00 00 00
  '¥'       C2 A5      A5 00         A5 00 00 00
  'あ'      E3 81 82   42 30         42 30 00 00
  '漢'      E6 BC A2   22 6F         22 6F 00 00
  '😀'      F0 9F 98   3D D8 00 DE   00 F6 01 00
            80

  サイズ比較（"Hello, 世界!" = ASCII 8文字 + 漢字 2文字）:
  UTF-8:  8×1 + 2×3 = 14 bytes
  UTF-16: 8×2 + 2×2 = 20 bytes
  UTF-32: 10×4       = 40 bytes

  サイズ比較（日本語文章 "日本語のテスト" = 7文字）:
  UTF-8:  7×3 = 21 bytes
  UTF-16: 7×2 = 14 bytes
  UTF-32: 7×4 = 28 bytes

  → 日本語テキストでは UTF-16 が最もコンパクト
  → 英語テキストでは UTF-8 が最もコンパクト
  → 混在テキストでは大抵 UTF-8 が有利
  → UTF-32 は常に最大
```

---

## 4. 文字化けの原因と対策

### 4.1 よくある文字化けパターン

```python
# 文字化けの再現と分析

text = "こんにちは"

# UTF-8でエンコードしたバイト列
utf8_bytes = text.encode('utf-8')
# b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf'

# ❌ パターン1: Latin-1としてデコード → 文字化け（最も多い）
wrong = utf8_bytes.decode('latin-1')
# 'ã\x81\x93ã\x82\x93ã\x81«ã\x81¡ã\x81¯'
# → UTF-8のバイトをLatin-1として解釈した典型パターン

# ❌ パターン2: Shift_JISとしてデコード → 別の文字化け
wrong2 = utf8_bytes.decode('shift_jis', errors='replace')
# '縺薙s縺ォ縺。縺ッ' のような意味不明な文字列

# ❌ パターン3: 二重エンコード（UTF-8を更にUTF-8としてエンコード）
double_encoded = text.encode('utf-8').decode('latin-1').encode('utf-8')
# 元に戻すには逆操作が必要
recovered = double_encoded.decode('utf-8').encode('latin-1').decode('utf-8')
print(recovered)  # 'こんにちは'

# ✅ 正しいエンコーディングでデコード
correct = utf8_bytes.decode('utf-8')
# 'こんにちは'
```

### 4.2 文字化けの診断方法

```python
# 文字化けの自動診断

def diagnose_encoding(broken_text, expected_text=None):
    """文字化けテキストのエンコーディングを診断"""
    # よくある文字化けパターンを試す
    patterns = [
        # (encode_as, decode_as, description)
        ('latin-1', 'utf-8', 'UTF-8をLatin-1で開いた'),
        ('cp1252', 'utf-8', 'UTF-8をWindows-1252で開いた'),
        ('shift_jis', 'utf-8', 'UTF-8をShift_JISで開いた'),
        ('utf-8', 'shift_jis', 'Shift_JISをUTF-8で開いた'),
        ('utf-8', 'euc-jp', 'EUC-JPをUTF-8で開いた'),
        ('euc-jp', 'utf-8', 'UTF-8をEUC-JPで開いた'),
    ]

    results = []
    for enc, dec, desc in patterns:
        try:
            recovered = broken_text.encode(enc).decode(dec)
            if expected_text and recovered == expected_text:
                results.append((desc, recovered, "✓ MATCH"))
            elif recovered.isprintable() and not any(c == '\ufffd' for c in recovered):
                results.append((desc, recovered, "? 可能性あり"))
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    return results


# chardetライブラリによる自動検出
# pip install chardet
import chardet

def detect_encoding(byte_data):
    """バイト列のエンコーディングを推定"""
    result = chardet.detect(byte_data)
    return result
    # {'encoding': 'utf-8', 'confidence': 0.99, 'language': ''}

# 使用例
with open('unknown_file.txt', 'rb') as f:
    raw_data = f.read()
    detection = chardet.detect(raw_data)
    print(f"推定: {detection['encoding']} (信頼度: {detection['confidence']:.2%})")
    text = raw_data.decode(detection['encoding'])
```

### 4.3 文字化け対策チェックリスト

```
全レイヤーでUTF-8を統一:

  ┌──────────────────────────────┐
  │ ファイル保存: UTF-8 (BOMなし) │
  ├──────────────────────────────┤
  │ HTTP: Content-Type: text/html;│
  │       charset=utf-8          │
  ├──────────────────────────────┤
  │ HTML: <meta charset="utf-8"> │
  ├──────────────────────────────┤
  │ DB: CHARACTER SET utf8mb4    │
  │    (MySQLのutf8は3バイトまで！│
  │     utf8mb4が真のUTF-8)     │
  ├──────────────────────────────┤
  │ Python: open(f, encoding='utf-8') │
  ├──────────────────────────────┤
  │ JSON: デフォルトUTF-8        │
  └──────────────────────────────┘

  注意: MySQLの"utf8"は3バイトまで（絵文字不可）
  → 必ず"utf8mb4"を使用すること！
```

### 4.4 各環境での文字コード設定

```python
# === Python ===

# ファイル読み書き（常にencodingを明示）
with open('file.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('file.txt', 'w', encoding='utf-8') as f:
    f.write(text)

# Python 3.15+ ではデフォルトがUTF-8に変更予定
# Python 3.7+: UTF-8モード
# PYTHONUTF8=1 環境変数 or python3 -X utf8

# CSVファイル（Excel対応でBOM付きUTF-8が必要な場合）
with open('data.csv', 'w', encoding='utf-8-sig') as f:
    # utf-8-sig = BOM (EF BB BF) + UTF-8
    f.write('名前,年齢\n')

# バイト列と文字列の変換
text = "日本語"
encoded = text.encode('utf-8')     # str → bytes
decoded = encoded.decode('utf-8')  # bytes → str

# エラーハンドリング
text = b'\xff\xfe'.decode('utf-8', errors='replace')   # '??'（置換）
text = b'\xff\xfe'.decode('utf-8', errors='ignore')    # ''（無視）
text = b'\xff\xfe'.decode('utf-8', errors='backslashreplace')  # '\\xff\\xfe'
```

```sql
-- === MySQL ===

-- データベース作成時
CREATE DATABASE mydb
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- テーブル作成時
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100) CHARACTER SET utf8mb4
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 接続時
SET NAMES utf8mb4;
-- または接続パラメータで: charset=utf8mb4

-- 既存テーブルの変更
ALTER TABLE users CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 確認
SHOW VARIABLES LIKE 'character_set%';
SHOW VARIABLES LIKE 'collation%';

-- utf8 vs utf8mb4 の違い:
-- utf8:    最大3バイト（BMP内のみ, 絵文字不可, U+0000-U+FFFF）
-- utf8mb4: 最大4バイト（全Unicode対応, 絵文字OK, U+0000-U+10FFFF）
-- → 常にutf8mb4を使うべき
```

```javascript
// === JavaScript / Node.js ===

// ファイル読み書き
const fs = require('fs');
const text = fs.readFileSync('file.txt', 'utf-8');
fs.writeFileSync('file.txt', text, 'utf-8');

// Buffer操作
const buf = Buffer.from('こんにちは', 'utf-8');
console.log(buf);        // <Buffer e3 81 93 e3 82 93 ...>
console.log(buf.length); // 15 (バイト数)
const str = buf.toString('utf-8');

// TextEncoder / TextDecoder (Web API & Node.js)
const encoder = new TextEncoder();  // デフォルトUTF-8
const decoder = new TextDecoder('utf-8');

const encoded = encoder.encode('Hello, 世界');
const decoded = decoder.decode(encoded);

// Shift_JISのデコード（Node.js: iconv-lite）
// const iconv = require('iconv-lite');
// const text = iconv.decode(buffer, 'Shift_JIS');

// fetch APIでの文字コード指定
// Response.text() はデフォルトでUTF-8
// Content-Type ヘッダーの charset を参照
```

```go
// === Go ===

package main

import (
    "fmt"
    "strings"
    "unicode/utf8"

    "golang.org/x/text/encoding/japanese"
    "golang.org/x/text/transform"
)

func main() {
    // Goの文字列はデフォルトでUTF-8
    s := "こんにちは"
    fmt.Println(len(s))                    // 15 (バイト数)
    fmt.Println(utf8.RuneCountInString(s)) // 5 (文字数)

    // ルーン（Unicodeコードポイント）でのイテレーション
    for i, r := range s {
        fmt.Printf("byte[%d]: U+%04X '%c'\n", i, r, r)
    }

    // UTF-8の妥当性チェック
    fmt.Println(utf8.ValidString(s))  // true

    // Shift_JISからUTF-8への変換
    sjisReader := transform.NewReader(
        strings.NewReader(sjisData),
        japanese.ShiftJIS.NewDecoder(),
    )
    // utf8Data, _ := io.ReadAll(sjisReader)
    _ = sjisReader
}
```

---

## 5. Unicode の落とし穴

### 5.1 結合文字と正規化

```python
# 「が」の2つの表現方法

# 1. 合成済み文字（NFC）: 1文字
ga_nfc = '\u304C'  # 'が' (1コードポイント)
len(ga_nfc)  # 1

# 2. 結合文字（NFD）: 基底文字 + 濁点
ga_nfd = '\u304B\u3099'  # 'か' + '゙' (2コードポイント)
len(ga_nfd)  # 2

# 見た目は同じ 'が' だが、==で比較するとFalse!
ga_nfc == ga_nfd  # False!

# 対策: unicodedata.normalize で正規化
import unicodedata
unicodedata.normalize('NFC', ga_nfd) == ga_nfc  # True
```

### 5.2 4つの正規化形式

```python
import unicodedata

# NFC（Canonical Decomposition, followed by Canonical Composition）
# → 最も一般的。合成済み形式。Webで推奨
nfc = unicodedata.normalize('NFC', text)

# NFD（Canonical Decomposition）
# → macOSのファイルシステム（HFS+）が使用
nfd = unicodedata.normalize('NFD', text)

# NFKC（Compatibility Decomposition, followed by Canonical Composition）
# → 検索/比較に最適。互換文字を統一
nfkc = unicodedata.normalize('NFKC', text)

# NFKD（Compatibility Decomposition）
# → 最も分解された形式
nfkd = unicodedata.normalize('NFKD', text)

# NFC vs NFD の例:
text = "が"  # U+304C
nfc = unicodedata.normalize('NFC', text)
nfd = unicodedata.normalize('NFD', text)
print(len(nfc), [hex(ord(c)) for c in nfc])  # 1 ['0x304c']
print(len(nfd), [hex(ord(c)) for c in nfd])  # 2 ['0x304b', '0x3099']

# NFKC vs NFC の例:
# 全角英数字 → 半角英数字
text2 = "Ａ１"  # U+FF21, U+FF11（全角）
nfc2 = unicodedata.normalize('NFC', text2)    # "Ａ１"（そのまま）
nfkc2 = unicodedata.normalize('NFKC', text2)  # "A1"（半角に変換）
print(nfkc2)  # A1

# 互換分解の例:
# ① → 1  (U+2460 → U+0031)
# ㌔ → キロ (U+3314 → U+30AD U+30ED)
# ﬁ → fi  (U+FB01 → U+0066 U+0069)
# ² → 2   (U+00B2 → U+0032)

# 実務での使い分け:
# - 保存/表示: NFC（最も広くサポート）
# - 検索/比較: NFKC（互換文字を統一）
# - macOSファイル名: NFD（OSが自動変換）
```

### 5.3 サロゲートペア（UTF-16の問題）

```
UTF-16 でのコードポイント表現:

  U+0000 - U+FFFF:   2バイトでそのまま（BMP: Basic Multilingual Plane）
  U+10000 - U+10FFFF: 4バイト（サロゲートペア必要）

  例: '😀' U+1F600
  1. U+1F600 - 0x10000 = 0x0F600
  2. 上位10ビット: 0x003D → + 0xD800 = 0xD83D（上位サロゲート）
  3. 下位10ビット: 0x0200 → + 0xDC00 = 0xDE00（下位サロゲート）
  4. UTF-16: 0xD83D 0xDE00

  → JavaScript の string.length が絵文字で2を返す理由
  '😀'.length === 2  // true! (UTF-16の内部表現)
  [...'😀'].length === 1  // true (イテレータはコードポイント単位)
```

### 5.4 書記素クラスタ（Grapheme Cluster）

```python
# 1つの「見た目の文字」が複数のコードポイントから構成される場合

# 例1: 家族絵文字
family = "👨‍👩‍👧‍👦"
print(len(family))  # 11 (コードポイント数!)
# 構成: 👨 U+1F468 + ZWJ + 👩 U+1F469 + ZWJ + 👧 U+1F467 + ZWJ + 👦 U+1F466
# ZWJ = Zero Width Joiner (U+200D)

# 例2: 国旗絵文字
flag_jp = "🇯🇵"
print(len(flag_jp))  # 2
# 構成: U+1F1EF (Regional Indicator J) + U+1F1F5 (Regional Indicator P)

# 例3: 肌の色修飾子
wave = "👋🏽"
print(len(wave))  # 2
# 構成: U+1F44B (手を振る) + U+1F3FD (肌色修飾子: Medium)

# 例4: 結合文字（アクセント付き文字）
e_acute = "é"  # U+0065 + U+0301 (NFD) or U+00E9 (NFC)

# 正しく「見た目の文字数」を数えるには:
# Python: regex ライブラリ（標準のreではなく第三者ライブラリ）
import regex  # pip install regex
text = "👨‍👩‍👧‍👦こんにちは"
graphemes = regex.findall(r'\X', text)
print(len(graphemes))  # 6 (家族1 + ひらがな5)

# JavaScript:
# Intl.Segmenter API (ES2022+)
# const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
# const segments = [...segmenter.segment(text)];
# segments.length; // 正しい書記素クラスタ数

# Go:
# golang.org/x/text/unicode/norm パッケージ
# rivo/uniseg パッケージ
```

### 5.5 文字列の長さとインデックス

```python
# 各言語での「文字列の長さ」の意味の違い

text = "Hello, 世界! 😀"

# Python 3: コードポイント数
print(len(text))                    # 11
print(len(text.encode('utf-8')))    # 18 (バイト数)

# Pythonでの正しいスライス
# text[7:9] は 'World' の一部ではなく、コードポイント単位

# 各言語の比較:
# ┌─────────────┬──────────────────────┬────────┐
# │ 言語        │ lengthの意味          │ 値     │
# ├─────────────┼──────────────────────┼────────┤
# │ Python 3    │ コードポイント数      │ 11     │
# │ JavaScript  │ UTF-16コードユニット数│ 12*    │
# │ Java        │ UTF-16コードユニット数│ 12*    │
# │ Rust (str)  │ バイト数             │ 18     │
# │ Go          │ バイト数             │ 18     │
# │ C (strlen)  │ バイト数（NULまで）  │ 18     │
# │ Swift       │ 書記素クラスタ数     │ 11**   │
# └─────────────┴──────────────────────┴────────┘
# * 😀が2カウント（サロゲートペア）
# ** Swiftは最も直感的だが、O(n)のコスト
```

```javascript
// JavaScript での文字列長の罠

const text = "Hello, 世界! 😀";

// .length は UTF-16 コードユニット数
console.log(text.length);  // 12 (😀 がサロゲートペアで2カウント)

// コードポイント数
console.log([...text].length);  // 11

// 正しい文字列操作
// ❌ 危険: text[10] は絵文字の上位サロゲートのみ
console.log(text[10]);  // '\uD83D' (壊れた文字)

// ✅ 安全: Array.from() or スプレッド構文
const chars = [...text];
console.log(chars[10]);  // '😀'

// ✅ 安全: codePointAt / String.fromCodePoint
for (const cp of text) {
    console.log(cp.codePointAt(0).toString(16));
}

// 書記素クラスタでの分割 (ES2022+)
const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
const segments = [...segmenter.segment(text)];
console.log(segments.length);  // 11
```

```rust
// Rust での文字列操作

fn main() {
    let text = "Hello, 世界! 😀";

    // .len() はバイト数
    println!("{}", text.len());  // 18

    // .chars().count() はコードポイント数
    println!("{}", text.chars().count());  // 11

    // バイトでのイテレーション
    for b in text.bytes() {
        print!("{:02X} ", b);
    }
    println!();

    // コードポイントでのイテレーション
    for c in text.chars() {
        println!("U+{:04X} '{}'", c as u32, c);
    }

    // 文字列スライスはバイト境界でないとパニック
    // let s = &text[0..7]; // OK (ASCII部分)
    // let s = &text[0..8]; // パニック! UTF-8の途中で切断

    // 安全なスライス
    if text.is_char_boundary(7) {
        let s = &text[0..7];
        println!("{}", s);  // "Hello, "
    }

    // char_indices でバイト位置とcharを取得
    for (i, c) in text.char_indices() {
        println!("byte[{}]: U+{:04X} '{}'", i, c as u32, c);
    }
}
```

---

## 6. 特殊な文字と制御文字

### 6.1 見えない文字

```
注意すべき見えない/紛らわしいUnicode文字:

  【ゼロ幅文字】
  U+200B  Zero Width Space（ゼロ幅スペース）
  U+200C  Zero Width Non-Joiner（ZWNJ）
  U+200D  Zero Width Joiner（ZWJ）
  U+FEFF  BOM / Zero Width No-Break Space

  【方向制御文字】
  U+200E  Left-to-Right Mark（LRM）
  U+200F  Right-to-Left Mark（RLM）
  U+202A  Left-to-Right Embedding
  U+202B  Right-to-Left Embedding
  U+202C  Pop Directional Formatting
  U+2066  Left-to-Right Isolate
  U+2067  Right-to-Left Isolate
  U+2069  Pop Directional Isolate

  【セキュリティ上の懸念】
  - Bidi Override（U+202E）でテキスト方向を逆転
    → ファイル名の偽装: "document.pdf" が実は "document.exe" に見える
  - ゼロ幅文字をパスワードやユーザー名に混入
    → 見た目は同じだが異なる文字列
  - ホモグリフ攻撃: Cyrillicの 'а' (U+0430) vs Latin 'a' (U+0061)
    → "аpple.com" は "apple.com" と見分けがつかない
```

```python
# 見えない文字の検出と除去

import unicodedata

def detect_invisible_chars(text):
    """見えない文字を検出して報告"""
    invisible = []
    for i, char in enumerate(text):
        cat = unicodedata.category(char)
        if cat.startswith('C') and char not in '\n\r\t':
            # C = Control, Cf = Format, Co = Private Use
            invisible.append((i, hex(ord(char)), unicodedata.name(char, '???'), cat))
    return invisible

def remove_invisible_chars(text):
    """見えない文字を除去（改行・タブは保持）"""
    return ''.join(
        c for c in text
        if not unicodedata.category(c).startswith('C')
        or c in '\n\r\t'
    )

# ホモグリフ検出
def detect_homoglyphs(text):
    """ラテン文字に見えるが異なるスクリプトの文字を検出"""
    suspicious = []
    for i, char in enumerate(text):
        if char.isalpha():
            script = unicodedata.name(char, '').split()[0] if unicodedata.name(char, '') else ''
            # ラテン文字に見えるがCyrillicやGreekの文字
            if script in ('CYRILLIC', 'GREEK') and char.lower() in 'abcdefghijklmnopqrstuvwxyz':
                suspicious.append((i, char, hex(ord(char)), script))
    return suspicious


# Unicode カテゴリの概要
categories = """
L  = Letter（文字）
  Lu = Uppercase Letter（大文字）
  Ll = Lowercase Letter（小文字）
  Lt = Titlecase Letter
  Lm = Modifier Letter
  Lo = Other Letter（漢字、ひらがな等）

M  = Mark（結合文字）
  Mn = Nonspacing Mark（濁点等）
  Mc = Spacing Combining Mark
  Me = Enclosing Mark

N  = Number（数字）
  Nd = Decimal Digit Number（0-9等）
  Nl = Letter Number（ローマ数字等）
  No = Other Number（丸数字等）

P  = Punctuation（句読点）
S  = Symbol（記号）
Z  = Separator（空白文字）
  Zs = Space Separator
  Zl = Line Separator
  Zp = Paragraph Separator

C  = Other（制御文字等）
  Cc = Control
  Cf = Format（ZWJ, BOM等）
  Co = Private Use
  Cs = Surrogate
"""
```

### 6.2 異体字セレクタ

```
異体字セレクタ（Variation Selector）:

  同じ漢字でも字形が異なる場合がある:
  「辻」の1点しんにょう vs 2点しんにょう
  「葛」の旧字体 vs 新字体

  IVS（Ideographic Variation Sequence）:
  基底文字 + 異体字セレクタ（U+E0100-U+E01EF）で字形を指定

  例:
  U+8FBB + U+E0100 → 辻（1点しんにょう）
  U+8FBB + U+E0101 → 辻（2点しんにょう）

  注意:
  - フォントがIVSをサポートしている必要がある
  - 多くの環境ではデフォルト字形のみ表示
  - 戸籍・住民票などの公的文書で重要
```

---

## 7. 実務での文字コード処理

### 7.1 Webアプリケーション

```html
<!-- HTML での文字コード指定 -->
<!DOCTYPE html>
<html lang="ja">
<head>
    <!-- 必ず最初の1024バイト以内に -->
    <meta charset="utf-8">
    <!-- Content-Type ヘッダーでも指定 -->
    <!-- Content-Type: text/html; charset=utf-8 -->
    <title>文字コードテスト</title>
</head>
<body>
    <!-- フォームのデフォルトエンコーディング -->
    <form accept-charset="utf-8" method="post">
        <input type="text" name="name" value="">
        <button type="submit">送信</button>
    </form>
</body>
</html>
```

```python
# Flask での文字コード処理

from flask import Flask, request, Response
import json

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def handle_data():
    # リクエストのデコード
    # Flask は Content-Type の charset を自動検出
    text = request.data.decode('utf-8')

    # JSON レスポンス（ensure_ascii=False で日本語をそのまま出力）
    data = {"message": "こんにちは", "status": "ok"}
    response = Response(
        json.dumps(data, ensure_ascii=False),
        content_type='application/json; charset=utf-8'
    )
    return response

# Django の場合:
# settings.py で DEFAULT_CHARSET = 'utf-8'（デフォルト）
# FILE_CHARSET = 'utf-8'
```

### 7.2 データベース操作

```python
# MySQL での絵文字対応

import mysql.connector

# 接続時にutf8mb4を指定
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='password',
    database='mydb',
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci'
)

cursor = conn.cursor()

# 絵文字を含むデータの挿入
cursor.execute(
    "INSERT INTO messages (content) VALUES (%s)",
    ("こんにちは 😀🎉",)
)
conn.commit()

# PostgreSQL の場合:
# デフォルトでUTF-8をフルサポート
# CREATE DATABASE mydb ENCODING 'UTF8' LC_COLLATE 'ja_JP.UTF-8';
```

### 7.3 ファイル入出力

```python
# CSV ファイルの文字コード処理

import csv

# UTF-8 CSV の読み込み
with open('data.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Excel互換のCSV出力（BOM付きUTF-8）
with open('output.csv', 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['名前', '年齢', '住所'])
    writer.writerow(['田中太郎', '30', '東京都'])

# Shift_JIS CSVの読み込み（レガシーシステムから）
with open('legacy.csv', 'r', encoding='shift_jis') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# Shift_JIS → UTF-8 の一括変換
def convert_encoding(input_path, output_path, from_enc='shift_jis', to_enc='utf-8'):
    with open(input_path, 'r', encoding=from_enc) as f_in:
        content = f_in.read()
    with open(output_path, 'w', encoding=to_enc) as f_out:
        f_out.write(content)


# JSON ファイル（常にUTF-8）
import json

data = {"名前": "田中", "趣味": ["読書", "😀"]}

# ensure_ascii=False が重要!
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# 結果: {"名前": "田中", "趣味": ["読書", "😀"]}
# ensure_ascii=True だと: {"\u540d\u524d": "\u7530\u4e2d", ...}
```

### 7.4 コマンドラインでの文字コード

```bash
# ファイルの文字コード判定
file -i document.txt
# document.txt: text/plain; charset=utf-8

# nkf（Network Kanji Filter）で変換
nkf -w input_sjis.txt > output_utf8.txt     # Shift_JIS → UTF-8
nkf -s input_utf8.txt > output_sjis.txt     # UTF-8 → Shift_JIS
nkf --guess input.txt                        # エンコーディング推定

# iconv で変換
iconv -f SHIFT_JIS -t UTF-8 input.txt > output.txt

# Python ワンライナー
python3 -c "
import sys
sys.stdout.buffer.write(
    sys.stdin.buffer.read().decode('shift_jis').encode('utf-8')
)" < input_sjis.txt > output_utf8.txt

# hexdump でバイト列確認
echo -n "あ" | xxd
# 00000000: e381 82                                  ...
# → E3 81 82 = UTF-8の「あ」

# ロケール確認
locale
# LANG=ja_JP.UTF-8

# 環境変数での文字コード設定
export LANG=ja_JP.UTF-8
export LC_ALL=ja_JP.UTF-8
```

---

## 8. 実践演習

### 演習1: UTF-8エンコード（基礎）
以下の文字のUTF-8バイト列を手計算で求めよ:
1. 'Z' (U+005A)
2. '¥' (U+00A5)
3. '漢' (U+6F22)
4. '𠮷' (U+20BB7)

### 演習2: 文字化け解析（応用）
バイト列 `E6 97 A5 E6 9C AC E8 AA 9E` をUTF-8としてデコードし、元の文字列を求めよ。

### 演習3: Unicode正規化の実装（発展）
Pythonで、2つの文字列が「見た目」が同じか（NFC/NFD正規化後に一致するか）を判定する関数を作成せよ。

### 演習4: サロゲートペア計算
以下のコードポイントをUTF-16サロゲートペアに変換せよ:
1. U+1F4A9
2. U+1F1EF (Regional Indicator J)
3. U+20000

### 演習5: 文字コード変換ツール
Pythonで以下の機能を持つCLIツールを実装せよ:
- 入力ファイルのエンコーディング自動判定
- 指定エンコーディングへの変換
- 変換レポートの出力

### 演習解答例

```python
# 演習1 解答

# 'Z' = U+005A
# 2進: 101 1010 (7ビット → 1バイト)
# UTF-8: 0_1011010 = 0x5A
print('Z'.encode('utf-8').hex())  # 5a

# '¥' = U+00A5
# 2進: 10 100101 (8ビット → 2バイト)
# 分割: 00010 100101
# UTF-8: 110_00010 10_100101 = 0xC2 0xA5
print('¥'.encode('utf-8').hex())  # c2a5

# '漢' = U+6F22
# 2進: 0110 1111 0010 0010 (16ビット → 3バイト)
# 分割: 0110 111100 100010
# UTF-8: 1110_0110 10_111100 10_100010 = 0xE6 0xBC 0xA2
print('漢'.encode('utf-8').hex())  # e6bca2

# '𠮷' = U+20BB7
# 2進: 0010 0000 1011 1011 0111 (21ビット → 4バイト)
# 分割: 000 100000 101110 110111
# UTF-8: 11110_000 10_100000 10_101110 10_110111 = 0xF0 0xA0 0xAE 0xB7
print('𠮷'.encode('utf-8').hex())  # f0a0aeb7


# 演習2 解答
bytes_data = bytes([0xE6, 0x97, 0xA5, 0xE6, 0x9C, 0xAC, 0xE8, 0xAA, 0x9E])
result = bytes_data.decode('utf-8')
print(result)  # '日本語'

# 手動デコード:
# E6 97 A5:
#   1110_0110 10_010111 10_100101
#   0110 010111 100101 = 0x65E5 → 不正
#   正しくは: 0110 100111 100101 ... → U+65E5 = '日'


# 演習3 解答
import unicodedata

def visual_equal(s1, s2):
    """2つの文字列が見た目上同じかを判定"""
    # NFC正規化で比較
    nfc1 = unicodedata.normalize('NFC', s1)
    nfc2 = unicodedata.normalize('NFC', s2)
    if nfc1 == nfc2:
        return True

    # NFKC正規化でも比較（互換文字の違いも吸収）
    nfkc1 = unicodedata.normalize('NFKC', s1)
    nfkc2 = unicodedata.normalize('NFKC', s2)
    return nfkc1 == nfkc2

# テスト
print(visual_equal('\u304C', '\u304B\u3099'))  # True (が = か+゛)
print(visual_equal('Ａ', 'A'))                  # True (全角A = 半角A, NFKC)
print(visual_equal('あ', 'ア'))                  # False
```

---

## FAQ

### Q1: BOM（Byte Order Mark）は必要ですか？
**A**: UTF-8では不要（むしろ有害な場合あり）。UTF-16/UTF-32ではエンディアン判定に必要。BOM（U+FEFF）がUTF-8ファイルの先頭にあると、シェルスクリプトやCSV解析でエラーの原因になる。ExcelでCSVを正しく開くためにBOM付きUTF-8（`utf-8-sig`）が必要な場合がある。

### Q2: 絵文字はどう実装されていますか？
**A**: Unicodeで標準化。U+1F600〜に割り当て。肌の色は修飾子（U+1F3FB〜U+1F3FF）で変更。家族絵文字はZWJ（Zero Width Joiner）で複数の絵文字を結合。国旗はRegional Indicator記号のペアで表現（JP → U+1F1EF U+1F1F5）。

### Q3: ASCII以外のファイル名は安全ですか？
**A**: OS依存:
- macOS: NFD正規化を強制（ファイル名が意図しない形に）
- Linux: バイト列として格納（UTF-8推奨だが強制なし）
- Windows: UTF-16で格納（内部はUCS-2の拡張）
安全を期すなら英数字とハイフン/アンダースコアのみ使用。

### Q4: MySQLでutf8ではなくutf8mb4を使うべき理由は？
**A**: MySQLの`utf8`は最大3バイト（BMP内のU+0000-U+FFFFのみ）しかサポートしない独自仕様。絵文字（U+1F600等）やCJK拡張漢字（U+20000以降）は4バイト必要なので`utf8mb4`が必須。`utf8mb4`が本来のUTF-8仕様に準拠。

### Q5: CJK統合漢字とは何ですか？
**A**: 中国語（Chinese）、日本語（Japanese）、韓国語（Korean）で共通する漢字を統合（Han Unification）したUnicodeの領域。例えば「海」は中日韓で字形が微妙に異なるが、同一コードポイント（U+6D77）に割り当てられている。これはUnicodeの最も論争的な設計判断の一つで、適切なフォント選択が重要。

### Q6: 文字列の正しい比較方法は？
**A**: 用途による:
- 完全一致: NFCに正規化してからバイト列比較
- 大小無視: casefold()（Pythonの場合）してから比較（`str.lower()`より正確）
- 検索用: NFKCに正規化して互換文字の違いも吸収
- ロケール依存ソート: ICUのCollationを使用
- セキュリティ: ホモグリフ・見えない文字のチェックも追加

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ASCII | 7ビット128文字。英語のみ。全ての基盤 |
| Unicode | 全言語を統一。コードポイント U+0000〜U+10FFFF |
| UTF-8 | 可変長(1-4バイト)。ASCII互換。Webの98%で使用 |
| UTF-16 | 可変長(2or4バイト)。Windows/Java/JSの内部表現 |
| UTF-32 | 固定長(4バイト)。シンプルだがメモリ非効率 |
| 文字化け | エンコーディング不一致が原因。全レイヤーでUTF-8統一 |
| 正規化 | NFC/NFD/NFKC/NFKDの違いに注意。比較前に正規化必須 |
| 書記素クラスタ | 1「文字」≠1コードポイント。絵文字で顕著 |
| サロゲートペア | BMP外のコードポイントをUTF-16で表す仕組み |
| セキュリティ | ホモグリフ、見えない文字、Bidiオーバーライドに注意 |

---

## 次に読むべきガイド
→ [[02-integer-representation.md]] — 整数表現と2の補数

---

## 参考文献
1. Unicode Consortium. "The Unicode Standard." https://unicode.org/
2. Pike, R. & Thompson, K. "Hello World, or Καλημέρα κόσμε, or こんにちは 世界." UTF-8 Design, 1992.
3. Spolsky, J. "The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets." 2003.
4. W3C. "Character encodings for beginners." https://www.w3.org/International/
5. RFC 3629. "UTF-8, a transformation format of ISO 10646." IETF, 2003.
6. Davis, M. & Suignard, M. "Unicode Security Considerations." Unicode Technical Report #36.
7. Unicode Consortium. "Unicode Normalization Forms." Unicode Standard Annex #15.
8. Unicode Consortium. "Unicode Text Segmentation." Unicode Standard Annex #29.
