# Unicode 正規表現 -- \p{Script}、フラグ、正規化

> グローバルなテキスト処理において、Unicode 対応の正規表現は不可欠である。Unicode プロパティエスケープ(`\p{...}`)、正規化形式(NFC/NFD)、書記体系(Script)によるマッチングを体系的に解説する。

## この章で学ぶこと

1. **Unicode プロパティエスケープの体系** -- `\p{L}` `\p{Script=Han}` 等のカテゴリ・プロパティ分類
2. **Unicode 正規化と正規表現の関係** -- NFC/NFD/NFKC/NFKD が検索結果に与える影響
3. **多言語テキスト処理の実践** -- 日本語・中国語・アラビア語等のマッチング手法

---

## 1. Unicode の基礎

### 1.1 Unicode の構造

```
Unicode コードポイント空間:

U+0000 ─ U+007F    Basic Latin (ASCII)           128文字
U+0080 ─ U+07FF    Latin, Greek, Cyrillic 等     約1,920文字
U+0800 ─ U+FFFF    CJK, ひらがな, カタカナ等      約63,488文字
U+10000 ─ U+10FFFF  絵文字, 古代文字 等           約1,048,576文字

合計: 約149,000文字が割り当て済み (Unicode 16.0)

UTF-8 エンコーディング:
┌─────────────────┬───────────┬──────────────┐
│ コードポイント    │ バイト数   │ 例            │
├─────────────────┼───────────┼──────────────┤
│ U+0000-U+007F   │ 1バイト    │ 'A' = 0x41   │
│ U+0080-U+07FF   │ 2バイト    │ 'é' = C3 A9  │
│ U+0800-U+FFFF   │ 3バイト    │ '漢' = E6 BC A2│
│ U+10000-U+10FFFF│ 4バイト    │ '😀' = F0 9F 98 80│
└─────────────────┴───────────┴──────────────┘
```

### 1.2 Unicode カテゴリ(General Category)

```
┌─────────────────────────────────────────────────┐
│              Unicode General Category             │
├──────┬──────────────────────────────────────────┤
│ L    │ Letter (文字)                              │
│  Lu  │  Uppercase Letter (大文字)                 │
│  Ll  │  Lowercase Letter (小文字)                 │
│  Lt  │  Titlecase Letter (タイトルケース)          │
│  Lm  │  Modifier Letter (修飾文字)                │
│  Lo  │  Other Letter (その他の文字: 漢字、かな等)  │
├──────┼──────────────────────────────────────────┤
│ M    │ Mark (結合文字)                             │
│  Mn  │  Nonspacing Mark (非空白結合文字)           │
│  Mc  │  Spacing Combining Mark                   │
│  Me  │  Enclosing Mark                           │
├──────┼──────────────────────────────────────────┤
│ N    │ Number (数字)                              │
│  Nd  │  Decimal Digit Number (10進数字)           │
│  Nl  │  Letter Number (ローマ数字等)              │
│  No  │  Other Number (分数等)                     │
├──────┼──────────────────────────────────────────┤
│ P    │ Punctuation (句読点)                       │
│ S    │ Symbol (記号)                              │
│ Z    │ Separator (区切り)                          │
│ C    │ Other (制御文字等)                          │
└──────┴──────────────────────────────────────────┘
```

---

## 2. Unicode プロパティエスケープ `\p{...}`

### 2.1 基本構文

```python
# Python: regex モジュール(サードパーティ)が必要
# pip install regex
import regex

text = "Hello 世界 café 123 ١٢٣"

# \p{L} -- 全ての文字(Letter)
print(regex.findall(r'\p{L}+', text))
# => ['Hello', '世界', 'café']

# \p{N} -- 全ての数字(Number)
print(regex.findall(r'\p{N}+', text))
# => ['123', '١٢٣']

# \p{Lu} -- 大文字のみ
print(regex.findall(r'\p{Lu}', text))
# => ['H']

# \P{L} -- 文字以外(否定)
print(regex.findall(r'\P{L}+', text))
# => [' ', ' ', ' ', ' ', '١٢٣']
```

### 2.2 JavaScript での Unicode プロパティ (ES2018+)

```javascript
const text = "Hello 世界 café 123 ١٢٣";

// \p{L} -- 全ての文字
console.log(text.match(/\p{L}+/gu));
// => ['Hello', '世界', 'café']

// \p{Script=Han} -- 漢字のみ
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世']  // '界' も含む場合は [\p{Script=Han}]+

// \p{Emoji} -- 絵文字
const emojiText = "Hello 👋 World 🌍!";
console.log(emojiText.match(/\p{Emoji}/gu));
// => ['👋', '🌍']

// u フラグが必須
// /\p{L}/g  → SyntaxError (u フラグなし)
// /\p{L}/gu → OK
```

### 2.3 Script(書記体系)プロパティ

```python
import regex

text = "日本語テスト English Русский العربية"

# 各書記体系を個別に抽出
print(regex.findall(r'\p{Script=Han}+', text))
# => ['日本語']  (漢字)

print(regex.findall(r'\p{Script=Hiragana}+', text))
# => []  (この例にはひらがななし)

print(regex.findall(r'\p{Script=Katakana}+', text))
# => ['テスト']

print(regex.findall(r'\p{Script=Latin}+', text))
# => ['English']

print(regex.findall(r'\p{Script=Cyrillic}+', text))
# => ['Русский']

print(regex.findall(r'\p{Script=Arabic}+', text))
# => ['العربية']
```

### 2.4 日本語テキストの処理

```python
import regex

text = "東京都は Tokyo とも呼ばれ、人口は約1400万人です。"

# 漢字
kanji = regex.findall(r'\p{Script=Han}+', text)
print(f"漢字: {kanji}")
# => 漢字: ['東京都', '呼', '人口', '約', '万人']

# ひらがな
hiragana = regex.findall(r'\p{Script=Hiragana}+', text)
print(f"ひらがな: {hiragana}")
# => ひらがな: ['は', 'とも', 'ばれ', 'は', 'です']

# カタカナ
katakana = regex.findall(r'\p{Script=Katakana}+', text)
print(f"カタカナ: {katakana}")
# => カタカナ: []

# 日本語文字全般 (漢字 + ひらがな + カタカナ)
japanese = regex.findall(r'[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]+', text)
print(f"日本語: {japanese}")
# => 日本語: ['東京都は', 'とも呼ばれ', '人口は約', '万人です']

# 数字(全角・半角両方)
numbers = regex.findall(r'[\p{Nd}]+', text)
print(f"数字: {numbers}")
# => 数字: ['1400']
```

---

## 3. Unicode 正規化

### 3.1 正規化の4形式

```
NFC  (Canonical Decomposition + Canonical Composition)
NFD  (Canonical Decomposition)
NFKC (Compatibility Decomposition + Canonical Composition)
NFKD (Compatibility Decomposition)

例: "café" の表現方法

NFC:  c a f é        (4文字 -- é は1コードポイント U+00E9)
NFD:  c a f e ◌́      (5文字 -- e + 結合アキュート U+0301)

両方とも同じ見た目だが、バイト列が異なる!

NFKC/NFKD はさらに互換文字を分解:
  "ﬁ" (U+FB01) → "fi" (2文字)
  "①" (U+2460) → "1"
  "Ｈｅｌｌｏ" (全角) → "Hello" (半角)
```

### 3.2 正規化が正規表現に与える影響

```python
import unicodedata
import re

# NFD と NFC で検索結果が変わる例
cafe_nfc = "café"                    # NFC: é = U+00E9
cafe_nfd = "cafe\u0301"             # NFD: e + ◌́ = U+0065 + U+0301

print(f"NFC: {repr(cafe_nfc)}")     # => 'caf\xe9'
print(f"NFD: {repr(cafe_nfd)}")     # => 'cafe\u0301'
print(f"見た目同一: {cafe_nfc} == {cafe_nfd}")  # 見た目は同じ

# 正規表現で "é" を検索
pattern = r'café'
print(bool(re.search(pattern, cafe_nfc)))  # => True
print(bool(re.search(pattern, cafe_nfd)))  # => False!

# 解決策: 検索前に正規化
normalized = unicodedata.normalize('NFC', cafe_nfd)
print(bool(re.search(pattern, normalized)))  # => True
```

### 3.3 実用的な正規化パイプライン

```python
import unicodedata
import re

def normalize_and_search(pattern: str, text: str, form: str = 'NFC') -> list:
    """正規化してから検索する"""
    norm_text = unicodedata.normalize(form, text)
    norm_pattern = unicodedata.normalize(form, pattern)
    return re.findall(norm_pattern, norm_text)

# 全角・半角の混在を処理 (NFKC)
text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３"  # 全角
normalized = unicodedata.normalize('NFKC', text)
print(normalized)         # => "Hello World 123"
print(re.findall(r'\w+', normalized))
# => ['Hello', 'World', '123']
```

---

## 4. Unicode フラグとモード

### 4.1 言語別 Unicode フラグ

```python
import re

text = "café CAFÉ"

# Python 3: デフォルトで Unicode 対応
# \w は Unicode 文字にマッチ
print(re.findall(r'\w+', text))
# => ['café', 'CAFÉ']

# re.ASCII: ASCII のみに制限
print(re.findall(r'\w+', text, re.ASCII))
# => ['caf', 'CAF']   # é がマッチしない

# re.IGNORECASE + Unicode
print(re.findall(r'café', text, re.IGNORECASE))
# => ['café', 'CAFÉ']
```

```javascript
// JavaScript: u フラグ (ES2015+)
const text = "café CAFÉ";

// u フラグなし: サロゲートペアの問題
console.log("😀".match(/^.$/));   // => null (2つのコードユニット)
console.log("😀".match(/^.$/u));  // => ['😀'] (1コードポイント)

// v フラグ (ES2024): u の拡張
// 集合演算: 交差、差分
console.log("aéあ".match(/[\p{L}&&\p{ASCII}]/gv));
// => ['a']  (ASCII かつ文字)
```

### 4.2 大文字小文字変換のUnicode問題

```python
import re

# Unicode の大文字小文字変換は1対1ではない
# ドイツ語の ß → SS (1文字が2文字に)
text = "straße STRASSE"

print(re.findall(r'stra(?:ße|sse)', text, re.IGNORECASE))
# => ['straße', 'STRASSE']

# トルコ語の i/I 問題
# トルコ語: İ (U+0130) ↔ i, I ↔ ı (U+0131)
# 英語:     I ↔ i
# → locale によって IGNORECASE の結果が変わる
```

---

## 5. 絵文字の正規表現

### 5.1 絵文字マッチングの課題

```python
import regex

text = "Hello 👋🏽 World 🇯🇵 Nice 👨‍👩‍👧‍👦"

# 絵文字の構造:
# 👋🏽 = 👋 (U+1F44B) + 🏽 (U+1FFFE, 肌色修飾子) → 2コードポイント
# 🇯🇵 = 🇯 (U+1F1EF) + 🇵 (U+1F1F5)              → 2コードポイント(旗)
# 👨‍👩‍👧‍👦 = 👨 + ZWJ + 👩 + ZWJ + 👧 + ZWJ + 👦   → 7コードポイント

# Python regex モジュール
emojis = regex.findall(r'\p{Emoji_Presentation}', text)
print(emojis)

# より正確な絵文字パターン (書記素クラスタ)
graphemes = regex.findall(r'\X', text)  # \X = 書記素クラスタ
print([g for g in graphemes if regex.match(r'\p{Emoji}', g)])
```

```javascript
// JavaScript (ES2024 v フラグ)
const text = "Hello 👋 World 🌍!";
const emojis = text.match(/\p{Emoji_Presentation}/gu);
console.log(emojis);
// => ['👋', '🌍']
```

---

## 6. ASCII 図解

### 6.1 Unicode プロパティの階層

```
\p{L}  Letter (全文字)
├── \p{Lu}  Uppercase    A B C ... Z  Á É  А Б В
├── \p{Ll}  Lowercase    a b c ... z  á é  а б в
├── \p{Lt}  Titlecase    ǅ ǈ ǋ (まれ)
├── \p{Lm}  Modifier     ʰ ʲ ˈ
└── \p{Lo}  Other        漢 字 あ い う ア イ ウ

\p{N}  Number (全数字)
├── \p{Nd}  Decimal      0-9  ٠-٩  ०-९  ０-９
├── \p{Nl}  Letter Num   Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ
└── \p{No}  Other Num    ½ ¼ ① ②

\p{P}  Punctuation (句読点)
├── \p{Pc}  Connector    _
├── \p{Pd}  Dash         - – —
├── \p{Ps}  Open         ( [ {
├── \p{Pe}  Close        ) ] }
└── ...

\p{S}  Symbol (記号)
├── \p{Sc}  Currency     $ € ¥ £
├── \p{Sm}  Math         + = < > ≤ ≥
└── ...
```

### 6.2 正規化形式の関係図

```
         正準分解
  NFC ◄──────────► NFD
   │                │
   │互換合成         │互換分解
   ▼                ▼
  NFKC ◄─────────► NFKD
         正準分解

例: "ﬁ" (U+FB01 LATIN SMALL LIGATURE FI)

NFC:  ﬁ (そのまま)
NFD:  ﬁ (そのまま -- 正準分解なし)
NFKC: fi (2文字に分解)
NFKD: fi (2文字に分解)

例: "é" (U+00E9 LATIN SMALL LETTER E WITH ACUTE)

NFC:  é        (1文字: U+00E9)
NFD:  e + ◌́    (2文字: U+0065 + U+0301)
NFKC: é        (1文字: U+00E9)
NFKD: e + ◌́    (2文字: U+0065 + U+0301)
```

### 6.3 サロゲートペアの仕組み

```
UTF-16 でのコードポイント表現:

BMP (U+0000 - U+FFFF): そのまま16ビットで表現
  'A' = U+0041 → 0x0041 (1コードユニット)
  '漢' = U+6F22 → 0x6F22 (1コードユニット)

補助面 (U+10000+): サロゲートペア(2つの16ビット値)
  '😀' = U+1F600
  → 0xD83D 0xDE00 (2コードユニット = サロゲートペア)

  計算方法:
  code = 0x1F600 - 0x10000 = 0xF600
  high = (0xF600 >> 10) + 0xD800 = 0xD83D
  low  = (0xF600 & 0x3FF) + 0xDC00 = 0xDE00

JavaScript の . (u フラグなし):
  "😀".length      → 2 (サロゲートペア)
  "😀".match(/./)  → "\uD83D" (上位サロゲートのみ)

JavaScript の . (u フラグあり):
  "😀".match(/./u) → "😀" (正しく1文字として扱う)
```

---

## 7. 比較表

### 7.1 Unicode プロパティのサポート状況

| プロパティ | Python re | Python regex | JavaScript | Java | Perl |
|-----------|----------|-------------|------------|------|------|
| `\p{L}` | 不可 | OK | OK (ES2018+u) | OK | OK |
| `\p{Lu}` | 不可 | OK | OK | OK | OK |
| `\p{Script=Han}` | 不可 | OK | OK | 不可 | OK |
| `\p{Emoji}` | 不可 | OK | OK | 不可 | OK |
| `\p{Block=CJK}` | 不可 | OK | 不可 | OK | OK |
| Unicode対応 `\w` | デフォルト | デフォルト | `/u` 必要 | `UNICODE_CHARACTER_CLASS` | デフォルト |

### 7.2 正規化形式の使い分け

| 形式 | 用途 | 特徴 |
|------|------|------|
| NFC | テキスト保存・交換の標準 | 合成形式。Web標準で推奨 |
| NFD | 分解して処理したい場合 | アクセント記号を分離 |
| NFKC | 検索・照合 | 互換文字を統一(全角→半角等) |
| NFKD | 検索の前処理 | 最大限に分解 |

---

## 8. アンチパターン

### 8.1 アンチパターン: Unicode範囲のハードコード

```python
import re
import regex

# NG: Unicode 範囲を手動で指定
pattern_bad = r'[\u3040-\u309F]+'  # ひらがな範囲をハードコード
# Unicode のバージョンアップで範囲が変わる可能性がある

# OK: Unicode プロパティを使う
pattern_good = r'\p{Script=Hiragana}+'  # regex モジュール

text = "こんにちは"
print(regex.findall(pattern_good, text))
# => ['こんにちは']
```

### 8.2 アンチパターン: 正規化せずに比較

```python
import unicodedata
import re

# NG: 正規化なしで文字列を比較
text_nfc = "caf\u00e9"      # NFC: é (1文字)
text_nfd = "cafe\u0301"     # NFD: e + ́ (2文字)

# 見た目は同じだが...
print(text_nfc == text_nfd)             # => False!
print(re.search(r'café', text_nfd))     # => None!

# OK: 正規化してから比較
text_normalized = unicodedata.normalize('NFC', text_nfd)
print(text_nfc == text_normalized)      # => True
print(re.search(r'café', text_normalized))  # => マッチ
```

---

## 9. FAQ

### Q1: Python の `re` モジュールで `\p{L}` を使うには？

**A**: 標準の `re` モジュールでは使えない。サードパーティの `regex` モジュールを使う:

```bash
pip install regex
```

```python
import regex

text = "Hello 世界"
print(regex.findall(r'\p{L}+', text))
# => ['Hello', '世界']

# re モジュールでの代替手段:
import re
# 方法1: Unicode カテゴリフラグ
print(re.findall(r'[^\W\d_]+', text))  # \W の否定から数字と_を除外
# => ['Hello', '世界']
```

### Q2: 絵文字を正確に検出する最善の方法は？

**A**: 絵文字は複数のコードポイントで構成されるため、単純なパターンでは不十分。書記素クラスタ(`\X`)を使うのが最善:

```python
import regex

text = "Hi 👨‍👩‍👧‍👦 there 🇯🇵"

# \X で書記素クラスタ単位で分割
graphemes = regex.findall(r'\X', text)
emoji_graphemes = [g for g in graphemes if regex.search(r'\p{Emoji}', g) and not regex.match(r'[\d#*]', g)]
print(emoji_graphemes)
```

JavaScript では `Intl.Segmenter` (ES2022)を使う方法もある。

### Q3: 全角・半角を統一して検索するには？

**A**: NFKC 正規化を前処理として適用する:

```python
import unicodedata
import re

text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ　１２３"

# NFKC 正規化: 全角英数字を半角に変換
normalized = unicodedata.normalize('NFKC', text)
print(normalized)  # => "Hello World 123"

# 正規化後に通常の正規表現で検索可能
print(re.findall(r'[A-Za-z]+', normalized))
# => ['Hello', 'World']

print(re.findall(r'\d+', normalized))
# => ['123']
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| `\p{L}` | Unicode の全文字にマッチ |
| `\p{N}` | Unicode の全数字にマッチ |
| `\p{Script=Han}` | 漢字のみにマッチ |
| `\p{Emoji}` | 絵文字にマッチ |
| NFC | 合成形式(Web標準) |
| NFKC | 互換分解+合成(検索向け) |
| `/u` フラグ | JavaScript で Unicode 対応を有効化 |
| `\X` | 書記素クラスタ(regex モジュール) |
| 鉄則 | 検索前に正規化、プロパティはハードコードしない |

## 次に読むべきガイド

- [03-performance.md](./03-performance.md) -- パフォーマンスと ReDoS 対策
- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- 言語別正規表現の違い

## 参考文献

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- Unicode 正規表現の国際標準仕様
2. **Unicode Technical Report #15** "Unicode Normalization Forms" https://unicode.org/reports/tr15/ -- 正規化形式の公式仕様
3. **Mathias Bynens** "JavaScript has a Unicode problem" https://mathiasbynens.be/notes/javascript-unicode -- JavaScript における Unicode の問題点と対策
4. **Python regex module** https://github.com/mrabarnett/mrab-regex -- Python の高機能正規表現モジュール
