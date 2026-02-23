# Unicode 正規表現 -- \p{Script}、フラグ、正規化

> グローバルなテキスト処理において、Unicode 対応の正規表現は不可欠である。Unicode プロパティエスケープ(`\p{...}`)、正規化形式(NFC/NFD)、書記体系(Script)によるマッチングを体系的に解説する。

## この章で学ぶこと

1. **Unicode プロパティエスケープの体系** -- `\p{L}` `\p{Script=Han}` 等のカテゴリ・プロパティ分類
2. **Unicode 正規化と正規表現の関係** -- NFC/NFD/NFKC/NFKD が検索結果に与える影響
3. **多言語テキスト処理の実践** -- 日本語・中国語・アラビア語等のマッチング手法
4. **絵文字の正規表現処理** -- 複合絵文字、書記素クラスタの扱い方
5. **各言語の Unicode サポート差異** -- Python, JavaScript, Java, Go, Rust の違い
6. **実務での正規化パイプライン構築** -- 検索、比較、バリデーションの前処理

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
│  Pc  │  Connector Punctuation (_など)             │
│  Pd  │  Dash Punctuation (- – — など)            │
│  Ps  │  Open Punctuation (( [ { など)             │
│  Pe  │  Close Punctuation () ] } など)            │
│  Pi  │  Initial Quote (« ' " など)                │
│  Pf  │  Final Quote (» ' " など)                  │
│  Po  │  Other Punctuation (. , ; : ! ? など)      │
├──────┼──────────────────────────────────────────┤
│ S    │ Symbol (記号)                              │
│  Sc  │  Currency Symbol ($ € ¥ £ など)            │
│  Sk  │  Modifier Symbol (^ ` ´ ¨ など)           │
│  Sm  │  Math Symbol (+ = < > ± × ÷ など)         │
│  So  │  Other Symbol (© ® ™ ° など)              │
├──────┼──────────────────────────────────────────┤
│ Z    │ Separator (区切り)                          │
│  Zs  │  Space Separator (スペース類)              │
│  Zl  │  Line Separator (行区切り)                 │
│  Zp  │  Paragraph Separator (段落区切り)          │
├──────┼──────────────────────────────────────────┤
│ C    │ Other (制御文字等)                          │
│  Cc  │  Control (制御文字)                        │
│  Cf  │  Format (書式文字: ZWJ, BOM等)             │
│  Cs  │  Surrogate (サロゲート)                    │
│  Co  │  Private Use (私用領域)                    │
│  Cn  │  Unassigned (未割り当て)                   │
└──────┴──────────────────────────────────────────┘
```

### 1.3 Unicode の主要な書記体系(Script)

```
┌──────────────────────────────────────────────────────────┐
│              主要な Unicode Script 一覧                     │
├──────────────────┬───────────────────────────────────────┤
│ Script名          │ 対応文字の例                           │
├──────────────────┼───────────────────────────────────────┤
│ Latin             │ A-Z a-z À-ÿ (ラテン文字)               │
│ Han               │ 漢 字 東 京 (CJK統合漢字)              │
│ Hiragana          │ あ い う え お (ひらがな)               │
│ Katakana          │ ア イ ウ エ オ (カタカナ)               │
│ Hangul            │ 가 나 다 라 (韓国語)                   │
│ Cyrillic          │ А Б В Г (ロシア語等)                   │
│ Arabic            │ ا ب ت ث (アラビア語)                   │
│ Devanagari        │ अ आ इ ई (ヒンディー語等)               │
│ Greek             │ Α Β Γ Δ (ギリシャ語)                  │
│ Thai              │ ก ข ค ง (タイ語)                       │
│ Hebrew            │ א ב ג ד (ヘブライ語)                   │
│ Bengali           │ অ আ ই ঈ (ベンガル語)                  │
│ Tamil             │ அ ஆ இ ஈ (タミル語)                    │
│ Ethiopic          │ ሀ ለ ሐ መ (アムハラ語等)                │
│ Common            │ 0-9 , . ! ? @ (複数書記体系で共有)     │
│ Inherited         │ 結合文字(親の書記体系を継承)            │
└──────────────────┴───────────────────────────────────────┘
```

### 1.4 Unicode のバイナリプロパティ

```python
import regex

# Unicode のバイナリプロパティ(真/偽の値を持つ)
text = "Hello! 123 café Α Β"

# Alphabetic -- アルファベット文字
print(regex.findall(r'\p{Alphabetic}+', text))
# => ['Hello', 'café', 'Α', 'Β']

# White_Space -- 空白文字
print(regex.findall(r'\p{White_Space}+', text))
# => [' ', ' ', ' ', ' ']

# Uppercase / Lowercase
print(regex.findall(r'\p{Uppercase}+', text))
# => ['H', 'Α', 'Β']

print(regex.findall(r'\p{Lowercase}+', text))
# => ['ello', 'caf', 'é']

# ID_Start / ID_Continue -- プログラミング言語の識別子
# ID_Start: 識別子の先頭に使える文字
# ID_Continue: 識別子の2文字目以降に使える文字
identifier_pattern = regex.compile(r'\p{ID_Start}\p{ID_Continue}*')
code_text = "変数名 variable_1 _private 42invalid"
print(identifier_pattern.findall(code_text))
# => ['変数名', 'variable_1', '_private']

# Emoji プロパティ
emoji_text = "Hello 👋 World 🌍 Test 1️⃣ #️⃣"
print(regex.findall(r'\p{Emoji_Presentation}', emoji_text))
# => ['👋', '🌍']

# Extended_Pictographic -- より広い絵文字範囲
print(regex.findall(r'\p{Extended_Pictographic}', emoji_text))
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
# => [' ', ' ', ' ', ' ١٢٣']

# \p{Ll} -- 小文字のみ
print(regex.findall(r'\p{Ll}+', text))
# => ['ello', '世界', 'café']
# 注: 漢字は Lo (Other Letter) だが Ll ではない

# \p{Lo} -- その他の文字（漢字、かな等）
print(regex.findall(r'\p{Lo}+', text))
# => ['世界']
```

### 2.2 JavaScript での Unicode プロパティ (ES2018+)

```javascript
const text = "Hello 世界 café 123 ١٢٣";

// \p{L} -- 全ての文字
console.log(text.match(/\p{L}+/gu));
// => ['Hello', '世界', 'café']

// \p{Script=Han} -- 漢字のみ
console.log(text.match(/\p{Script=Han}+/gu));
// => ['世界']

// \p{Emoji} -- 絵文字
const emojiText = "Hello 👋 World 🌍!";
console.log(emojiText.match(/\p{Emoji}/gu));
// => ['👋', '🌍']

// u フラグが必須
// /\p{L}/g  → SyntaxError (u フラグなし)
// /\p{L}/gu → OK

// v フラグ (ES2024): u の拡張版
// 集合演算が可能
// /[\p{L}&&\p{ASCII}]/gv  -- ASCII 文字かつ Letter
// /[\p{L}--\p{Script=Latin}]/gv  -- ラテン文字以外の Letter
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

### 2.4 Script_Extensions プロパティ

```python
import regex

# Script と Script_Extensions の違い
# Script: 1つの書記体系にのみ属する
# Script_Extensions: 複数の書記体系で使われる文字を含む

# 例: 長音記号 "ー" (U+30FC)
# Script=Katakana だが、Script_Extensions には Hiragana も含む

text = "カタカナ ひらがなー"

# Script=Katakana のみ
print(regex.findall(r'\p{Script=Katakana}+', text))
# => ['カタカナ', 'ー']  -- ー はカタカナのScript

# 漢数字は Script=Han
text2 = "一二三 123"
print(regex.findall(r'\p{Script=Han}+', text2))
# => ['一二三']

# CJK 句読点は Common Script
text3 = "日本語。English"
print(regex.findall(r'\p{Script=Common}', text3))
# => ['。']  -- 句読点は Common
```

### 2.5 日本語テキストの処理

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

### 2.6 日本語固有の文字範囲と正規表現

```python
import regex
import re

# 日本語処理でよく使う Unicode ブロック/範囲
# これらは regex モジュールなしでも re で使える

# ひらがな: U+3040 - U+309F
# カタカナ: U+30A0 - U+30FF
# CJK統合漢字: U+4E00 - U+9FFF
# CJK統合漢字拡張A: U+3400 - U+4DBF
# 全角英数字: U+FF01 - U+FF5E
# 半角カタカナ: U+FF65 - U+FF9F
# CJK記号と句読点: U+3000 - U+303F

text = "東京タワー（とうきょうタワー）は高さ333mの電波塔です。"

# re モジュールで日本語文字をマッチ
# 漢字
print(re.findall(r'[\u4e00-\u9fff]+', text))
# => ['東京', '高', '電波塔']

# ひらがな
print(re.findall(r'[\u3040-\u309f]+', text))
# => ['とうきょう', 'は', 'さ', 'の', 'です']

# カタカナ
print(re.findall(r'[\u30a0-\u30ff]+', text))
# => ['タワー', 'タワー']

# 日本語全体（漢字 + ひらがな + カタカナ）
print(re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text))
# => ['東京タワー', 'とうきょうタワー', 'は高さ', 'の電波塔です']

# 日本語の文章から形態素的な区切りを推定
# （簡易版: 文字種の変わり目で分割）
def split_japanese(text: str) -> list[str]:
    """日本語テキストを文字種の変わり目で分割"""
    # 漢字→ひらがな、カタカナ→漢字 などの境界
    pattern = regex.compile(
        r'[\p{Script=Han}]+'
        r'|[\p{Script=Hiragana}]+'
        r'|[\p{Script=Katakana}ー]+'  # 長音記号を含む
        r'|[\p{Script=Latin}]+'
        r'|[\p{Nd}]+'
        r'|\S'
    )
    return pattern.findall(text)

result = split_japanese("東京タワーは高さ333mの電波塔です")
print(result)
# => ['東京', 'タワー', 'は', '高', 'さ', '333', 'm', 'の', '電波塔', 'です']
```

### 2.7 各言語の数字体系の処理

```python
import regex

# Unicode は様々な数字体系を持つ
text = "Latin: 123, Arabic: ١٢٣, Devanagari: १२३, Thai: ๑๒๓, CJK: ３２１"

# \p{Nd} -- 全ての10進数字
all_digits = regex.findall(r'\p{Nd}+', text)
print(f"全数字: {all_digits}")
# => ['123', '١٢٣', '१२३', '๑๒๓', '３２１']

# 特定の書記体系の数字のみ
# ラテン数字のみ
print(regex.findall(r'[0-9]+', text))
# => ['123']

# アラビア数字
print(regex.findall(r'[\u0660-\u0669]+', text))
# => ['١٢٣']

# 全角数字
print(regex.findall(r'[\uff10-\uff19]+', text))
# => ['３２１']

# Unicode 数字を算用数字に変換
import unicodedata

def normalize_digits(text: str) -> str:
    """全ての Unicode 数字を ASCII 数字に正規化"""
    result = []
    for ch in text:
        if unicodedata.category(ch) == 'Nd':
            # digit_value で数値を取得
            result.append(str(unicodedata.digit(ch)))
        else:
            result.append(ch)
    return ''.join(result)

normalized = normalize_digits(text)
print(f"正規化後: {normalized}")
# => "Latin: 123, Arabic: 123, Devanagari: 123, Thai: 123, CJK: 321"
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

### 3.3 正規化形式の詳細な動作

```python
import unicodedata

# 各正規化形式の詳細な違いを確認

# テストケース1: アクセント付き文字
e_acute = '\u00e9'          # é (NFC形式)
e_combining = 'e\u0301'     # e + 結合アキュート (NFD形式)

print("=== アクセント付き文字 ===")
print(f"NFC:  {repr(unicodedata.normalize('NFC', e_acute))}")   # => '\xe9'
print(f"NFD:  {repr(unicodedata.normalize('NFD', e_acute))}")   # => 'e\u0301'
print(f"NFKC: {repr(unicodedata.normalize('NFKC', e_acute))}")  # => '\xe9'
print(f"NFKD: {repr(unicodedata.normalize('NFKD', e_acute))}")  # => 'e\u0301'

# テストケース2: 合字(リガチャ)
fi_ligature = '\ufb01'  # ﬁ

print("\n=== 合字 ===")
print(f"NFC:  {repr(unicodedata.normalize('NFC', fi_ligature))}")   # => '\ufb01'
print(f"NFD:  {repr(unicodedata.normalize('NFD', fi_ligature))}")   # => '\ufb01'
print(f"NFKC: {repr(unicodedata.normalize('NFKC', fi_ligature))}")  # => 'fi'
print(f"NFKD: {repr(unicodedata.normalize('NFKD', fi_ligature))}")  # => 'fi'

# テストケース3: 全角英数字
fullwidth = '\uff28\uff45\uff4c\uff4c\uff4f'  # Ｈｅｌｌｏ

print("\n=== 全角英数字 ===")
print(f"NFC:  {unicodedata.normalize('NFC', fullwidth)}")   # => Ｈｅｌｌｏ
print(f"NFD:  {unicodedata.normalize('NFD', fullwidth)}")   # => Ｈｅｌｌｏ
print(f"NFKC: {unicodedata.normalize('NFKC', fullwidth)}")  # => Hello
print(f"NFKD: {unicodedata.normalize('NFKD', fullwidth)}")  # => Hello

# テストケース4: 丸付き数字
circled = '\u2460\u2461\u2462'  # ①②③

print("\n=== 丸付き数字 ===")
print(f"NFC:  {unicodedata.normalize('NFC', circled)}")   # => ①②③
print(f"NFKC: {unicodedata.normalize('NFKC', circled)}")  # => 123

# テストケース5: ローマ数字
roman = '\u2160\u2161\u2162'  # ⅠⅡⅢ

print("\n=== ローマ数字 ===")
print(f"NFC:  {unicodedata.normalize('NFC', roman)}")   # => ⅠⅡⅢ
print(f"NFKC: {unicodedata.normalize('NFKC', roman)}")  # => III
```

### 3.4 実用的な正規化パイプライン

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

### 3.5 日本語テキストの正規化パイプライン

```python
import unicodedata
import re

def normalize_japanese_text(text: str) -> str:
    """日本語テキストの包括的な正規化

    以下の処理を行う:
    1. NFKC 正規化（全角英数字→半角、互換文字の統一）
    2. 全角スペース→半角スペース
    3. 連続する空白の統一
    4. 前後の空白除去
    """
    # Step 1: NFKC 正規化
    text = unicodedata.normalize('NFKC', text)

    # Step 2: 全角スペース→半角スペース
    text = text.replace('\u3000', ' ')

    # Step 3: 連続する空白を1つに
    text = re.sub(r'\s+', ' ', text)

    # Step 4: 前後の空白除去
    text = text.strip()

    return text

# テスト
test_cases = [
    "Ｈｅｌｌｏ　Ｗｏｒｌｄ",          # 全角英数字と全角スペース
    "テスト　　テスト",                  # 連続全角スペース
    "①②③の手順",                       # 丸付き数字
    "ﬁnally　ﬁnished",               # 合字
    "  前後に  スペース  ",             # 前後の空白
]

for tc in test_cases:
    result = normalize_japanese_text(tc)
    print(f"  '{tc}' => '{result}'")

# 検索前の正規化パイプライン
def search_normalized(pattern: str, text: str, flags=0) -> list:
    """正規化済みテキストで検索"""
    norm_text = normalize_japanese_text(text)
    norm_pattern = normalize_japanese_text(pattern)
    return re.findall(norm_pattern, norm_text, flags)

# 使用例: 全角・半角混在のテキストから検索
text = "電話番号は０３−１２３４−５６７８です"
results = search_normalized(r'\d{2,4}[-−]\d{4}[-−]\d{4}', text)
print(f"電話番号: {results}")
# => ['03-1234-5678']
```

### 3.6 正規化の注意点とエッジケース

```python
import unicodedata

# 注意点1: NFKC は元に戻せない変換を含む
# 丸付き数字 ① → 1 は不可逆
circled_1 = '\u2460'  # ①
nfkc = unicodedata.normalize('NFKC', circled_1)
print(f"①のNFKC: '{nfkc}' (U+{ord(nfkc):04X})")
# => '1' (U+0031) -- 通常の数字になる

# 注意点2: CJK互換漢字の正規化
# 一部のCJK互換漢字はNFKCで統合される
# 例: U+FA30 "侮" (CJK互換) → U+FA30 のまま（変わらない場合もある）

# 注意点3: カタカナの「ヴ」と結合文字
# "ヴ" (U+30F4) は NFC/NFD どちらでも1文字のまま
vu = '\u30f4'  # ヴ
print(f"NFC: {repr(unicodedata.normalize('NFC', vu))}")   # => '\u30f4'
print(f"NFD: {repr(unicodedata.normalize('NFD', vu))}")   # => '\u30f4'

# 注意点4: 半角カタカナの正規化
# NFKC で半角カタカナは全角カタカナに変換される
halfwidth_katakana = '\uff76\uff80\uff76\uff85'  # ｶﾀｶﾅ
nfkc = unicodedata.normalize('NFKC', halfwidth_katakana)
print(f"半角カタカナのNFKC: '{nfkc}'")
# => 'カタカナ' (全角カタカナに変換)

# 注意点5: 濁点・半濁点の扱い
# 半角カタカナの「ガ」= ｶ + ﾞ (2文字)
# NFKC で全角の「ガ」(1文字) に合成される
ga_halfwidth = '\uff76\uff9e'  # ｶﾞ
ga_nfkc = unicodedata.normalize('NFKC', ga_halfwidth)
print(f"ｶﾞのNFKC: '{ga_nfkc}' (len={len(ga_nfkc)})")
# => 'ガ' (len=1)
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

### 4.2 各言語での Unicode 対応の違い

```python
# Python 3 の Unicode 対応
import re

text = "café naïve résumé"

# Python 3 では \w, \d, \s はデフォルトで Unicode 対応
print(re.findall(r'\w+', text))
# => ['café', 'naïve', 'résumé']

# \b もUnicode 単語境界
print(re.findall(r'\b\w+\b', text))
# => ['café', 'naïve', 'résumé']

# re.ASCII フラグで ASCII モードに制限
print(re.findall(r'\w+', text, re.ASCII))
# => ['caf', 'na', 've', 'r', 'sum']

# インラインフラグでも指定可能
print(re.findall(r'(?a)\w+', text))  # (?a) = re.ASCII
# => ['caf', 'na', 've', 'r', 'sum']
```

```java
// Java の Unicode 対応
import java.util.regex.*;

public class UnicodeJava {
    public static void main(String[] args) {
        String text = "café naïve 東京";

        // デフォルト: \w は [a-zA-Z_0-9] のみ（Unicode非対応）
        Pattern p1 = Pattern.compile("\\w+");
        Matcher m1 = p1.matcher(text);
        while (m1.find()) {
            System.out.println(m1.group());
        }
        // => "caf", "na", "ve", (東京はマッチしない)

        // UNICODE_CHARACTER_CLASS フラグで Unicode 対応
        Pattern p2 = Pattern.compile("\\w+",
            Pattern.UNICODE_CHARACTER_CLASS);
        Matcher m2 = p2.matcher(text);
        while (m2.find()) {
            System.out.println(m2.group());
        }
        // => "café", "naïve", "東京"

        // \p{L} は Java でも使える（フラグ不要）
        Pattern p3 = Pattern.compile("\\p{L}+");
        Matcher m3 = p3.matcher(text);
        while (m3.find()) {
            System.out.println(m3.group());
        }
        // => "café", "naïve", "東京"
    }
}
```

```go
// Go (RE2) の Unicode 対応
package main

import (
    "fmt"
    "regexp"
)

func main() {
    text := "café naïve 東京"

    // Go の \w は ASCII のみ
    re1 := regexp.MustCompile(`\w+`)
    fmt.Println(re1.FindAllString(text, -1))
    // => ["caf", "na", "ve"]

    // Unicode 文字プロパティは使える
    // \p{L} -- Unicode Letter
    re2 := regexp.MustCompile(`\p{L}+`)
    fmt.Println(re2.FindAllString(text, -1))
    // => ["café", "naïve", "東京"]

    // \p{Han} -- 漢字
    re3 := regexp.MustCompile(`\p{Han}+`)
    fmt.Println(re3.FindAllString(text, -1))
    // => ["東京"]

    // \p{Hiragana}, \p{Katakana} も利用可能
    text2 := "ひらがなカタカナ漢字"
    re4 := regexp.MustCompile(`\p{Hiragana}+`)
    fmt.Println(re4.FindAllString(text2, -1))
    // => ["ひらがな"]
}
```

### 4.3 大文字小文字変換のUnicode問題

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

# ケースフォールディング(Case Folding)
# Unicode 標準の大文字小文字統一変換
text = "Straße straße STRASSE"
for word in text.split():
    print(f"  {word} -> casefold: {word.casefold()}")
# => straße -> casefold: strasse
# => straße -> casefold: strasse
# => STRASSE -> casefold: strasse
# casefold() は lower() より積極的に変換する

# 正規表現での Unicode 大文字小文字
# Python の re.IGNORECASE は Unicode 対応
print(re.findall(r'straße', text, re.IGNORECASE))
# => ['Straße', 'straße']
# 注: 'STRASSE' はマッチしない場合がある（実装依存）
```

### 4.4 Unicode 正規表現フラグの包括的比較

```
各言語の Unicode フラグ一覧:

言語          フラグ/オプション              効果
──────        ────────────────              ──────
Python        re.UNICODE (デフォルト)        \w, \d, \s がUnicode対応
              re.ASCII (or (?a))            ASCIIのみに制限
              re.IGNORECASE                 Unicode大文字小文字無視

JavaScript    /u                            Unicode対応(コードポイント単位)
              /v (ES2024)                   /u + 集合演算
              /i                            大文字小文字無視

Java          Pattern.UNICODE_CHARACTER_CLASS  \w, \d がUnicode対応
              Pattern.CASE_INSENSITIVE       大文字小文字無視
              Pattern.UNICODE_CASE           Unicodeケース無視(要CASE_INSENSITIVE)

Go            (デフォルト)                   \p{...} はUnicode対応
                                            \w, \d はASCIIのみ

Rust          (?u) (デフォルト)              Unicode対応
              (?-u)                         ASCII限定
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

### 5.2 絵文字の種類と構造

```
絵文字の種類:

1. 基本絵文字 (Basic Emoji)
   😀 = U+1F600 (1コードポイント)

2. テキスト/絵文字表示切替 (Emoji Presentation)
   ☺️ = ☺ (U+263A) + VS16 (U+FE0F)  -- 絵文字表示
   ☺ = U+263A                         -- テキスト表示

3. 肌色修飾 (Skin Tone Modifier)
   👋🏽 = 👋 (U+1F44B) + 🏽 (U+1F3FD)

4. 国旗 (Regional Indicator)
   🇯🇵 = 🇯 (U+1F1EF) + 🇵 (U+1F1F5)

5. ZWJ シーケンス (Zero Width Joiner)
   👨‍💻 = 👨 (U+1F468) + ZWJ (U+200D) + 💻 (U+1F4BB)
   👨‍👩‍👧‍👦 = 👨 + ZWJ + 👩 + ZWJ + 👧 + ZWJ + 👦

6. キーキャップ (Keycap)
   1️⃣ = 1 (U+0031) + VS16 (U+FE0F) + ⃣ (U+20E3)

7. タグ付き (Tag Sequence) -- 旗のサブ地域
   🏴󠁧󠁢󠁥󠁮󠁧󠁿 = 🏴 + TAG_g + TAG_b + TAG_e + TAG_n + TAG_g + CANCEL_TAG
```

### 5.3 絵文字の包括的な正規表現パターン

```python
import regex

def extract_emojis(text: str) -> list[str]:
    """テキストから全ての絵文字を書記素クラスタ単位で抽出"""
    # \X で書記素クラスタに分割
    graphemes = regex.findall(r'\X', text)

    # 絵文字かどうかを判定
    emoji_pattern = regex.compile(
        r'[\p{Emoji_Presentation}\p{Extended_Pictographic}]'
    )

    emojis = []
    for g in graphemes:
        if emoji_pattern.search(g):
            # 数字やハッシュなどの誤検出を除外
            if not regex.match(r'^[\d#*]$', g):
                emojis.append(g)

    return emojis

# テスト
test_texts = [
    "Hello 😀 World",
    "Flag: 🇯🇵 🇺🇸 🇬🇧",
    "Family: 👨‍👩‍👧‍👦",
    "Skin: 👋🏻 👋🏽 👋🏿",
    "Mix: テスト🎉テスト",
    "Numbers: 1️⃣2️⃣3️⃣",
]

for text in test_texts:
    emojis = extract_emojis(text)
    print(f"  '{text}' => {emojis}")
```

### 5.4 絵文字の除去と置換

```python
import regex
import re

def remove_emojis(text: str) -> str:
    """テキストから全ての絵文字を除去"""
    # 方法1: regex モジュールを使う(推奨)
    return regex.sub(
        r'[\p{Emoji_Presentation}\p{Extended_Pictographic}]'
        r'[\p{Emoji_Modifier}\p{Emoji_Component}\u200d\ufe0f\ufe0e]*',
        '',
        text
    )

def replace_emojis_with_text(text: str) -> str:
    """絵文字をテキスト表現に置換"""
    import unicodedata
    result = []
    for grapheme in regex.findall(r'\X', text):
        name = None
        for ch in grapheme:
            try:
                n = unicodedata.name(ch, None)
                if n and 'EMOJI' not in n.upper():
                    name = n
                    break
            except ValueError:
                continue
        if name and regex.search(r'[\p{Emoji_Presentation}]', grapheme):
            result.append(f'[{name}]')
        else:
            result.append(grapheme)
    return ''.join(result)

# テスト
text = "素晴らしい! 🎉 今日は天気がいい ☀️ 散歩に行こう 🚶"
print(f"元: {text}")
print(f"除去後: {remove_emojis(text)}")
```

### 5.5 JavaScript での絵文字処理

```javascript
// ES2024 の Intl.Segmenter を使った書記素クラスタ分割
function extractEmojis(text) {
    const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
    const segments = [...segmenter.segment(text)];

    return segments
        .filter(seg => /\p{Emoji_Presentation}/u.test(seg.segment))
        .map(seg => seg.segment);
}

console.log(extractEmojis("Hello 😀 World 🌍 Family 👨‍👩‍👧‍👦"));
// => ['😀', '🌍', '👨‍👩‍👧‍👦']

// 文字数のカウント（書記素クラスタ単位）
function graphemeLength(text) {
    const segmenter = new Intl.Segmenter('ja', { granularity: 'grapheme' });
    return [...segmenter.segment(text)].length;
}

console.log("👨‍👩‍👧‍👦".length);           // => 11 (UTF-16コードユニット数)
console.log(graphemeLength("👨‍👩‍👧‍👦"));  // => 1 (見た目の文字数)
```

---

## 6. 書記素クラスタ(Grapheme Cluster)

### 6.1 書記素クラスタとは

```
コードポイントと書記素クラスタの違い:

テキスト: "café"
  コードポイント(NFC): c a f é  → 4個
  書記素クラスタ:      c a f é  → 4個 (同じ)

テキスト: "cafe\u0301" (NFD)
  コードポイント:      c a f e ́  → 5個
  書記素クラスタ:      c a f é  → 4個 (見た目通り)

テキスト: "👨‍👩‍👧‍👦"
  コードポイント:      👨 ZWJ 👩 ZWJ 👧 ZWJ 👦  → 7個
  書記素クラスタ:      👨‍👩‍👧‍👦                       → 1個 (見た目は1文字)

テキスト: "🇯🇵"
  コードポイント:      🇯 🇵  → 2個
  書記素クラスタ:      🇯🇵   → 1個

テキスト: "ก้" (タイ語の「コー」+ マイトー声調記号)
  コードポイント:      ก ้  → 2個
  書記素クラスタ:      ก้   → 1個
```

### 6.2 書記素クラスタを使った正規表現

```python
import regex

# \X -- 書記素クラスタにマッチ(regex モジュール)
text = "café 👨‍👩‍👧‍👦 🇯🇵 naïve"

# コードポイント単位(通常の .)
import re
print(f"コードポイント数: {len(text)}")

# 書記素クラスタ単位(\X)
graphemes = regex.findall(r'\X', text)
print(f"書記素クラスタ数: {len(graphemes)}")
print(f"書記素一覧: {graphemes}")

# 書記素クラスタの先頭N文字を取得
def truncate_graphemes(text: str, max_graphemes: int) -> str:
    """見た目の文字数で切り詰める"""
    graphemes = regex.findall(r'\X', text)
    return ''.join(graphemes[:max_graphemes])

# テスト
long_text = "こんにちは👨‍👩‍👧‍👦世界🌍テスト"
for n in [5, 8, 10]:
    truncated = truncate_graphemes(long_text, n)
    print(f"  先頭{n}書記素: '{truncated}'")
```

### 6.3 文字数カウントの正確な実装

```python
import regex
import unicodedata

def count_characters(text: str) -> dict:
    """テキストの各種文字数をカウント"""
    graphemes = regex.findall(r'\X', text)

    return {
        'bytes_utf8': len(text.encode('utf-8')),
        'bytes_utf16': len(text.encode('utf-16-le')),
        'codepoints': len(text),
        'grapheme_clusters': len(graphemes),
        'nfc_codepoints': len(unicodedata.normalize('NFC', text)),
        'nfd_codepoints': len(unicodedata.normalize('NFD', text)),
    }

# テスト
test_texts = [
    ("ASCII", "Hello"),
    ("日本語", "こんにちは"),
    ("アクセント(NFC)", "caf\u00e9"),
    ("アクセント(NFD)", "cafe\u0301"),
    ("絵文字", "👨‍👩‍👧‍👦"),
    ("国旗", "🇯🇵"),
    ("混合", "Hello世界😀"),
]

for label, text in test_texts:
    counts = count_characters(text)
    print(f"\n{label}: '{text}'")
    for key, value in counts.items():
        print(f"  {key}: {value}")
```

---

## 7. 多言語テキスト処理の実践

### 7.1 多言語テキストの言語判定

```python
import regex

def detect_scripts(text: str) -> dict[str, int]:
    """テキスト内の書記体系を検出してカウント"""
    scripts = {}

    script_patterns = {
        'Latin': r'\p{Script=Latin}',
        'Han': r'\p{Script=Han}',
        'Hiragana': r'\p{Script=Hiragana}',
        'Katakana': r'\p{Script=Katakana}',
        'Cyrillic': r'\p{Script=Cyrillic}',
        'Arabic': r'\p{Script=Arabic}',
        'Devanagari': r'\p{Script=Devanagari}',
        'Hangul': r'\p{Script=Hangul}',
        'Thai': r'\p{Script=Thai}',
        'Greek': r'\p{Script=Greek}',
    }

    for script_name, pattern in script_patterns.items():
        count = len(regex.findall(pattern, text))
        if count > 0:
            scripts[script_name] = count

    return scripts

# テスト
test_texts = [
    "Hello World",
    "こんにちは世界",
    "Hello 世界 Мир العالم",
    "東京タワー Tokyo Tower",
    "한국어 테스트",
]

for text in test_texts:
    scripts = detect_scripts(text)
    print(f"  '{text}' => {scripts}")
```

### 7.2 アラビア語・ヘブライ語(RTL)テキストの処理

```python
import regex

# RTL(右から左)テキストの正規表現処理

# アラビア語テキスト
arabic_text = "مرحبا بالعالم"  # "Hello World" in Arabic
print(regex.findall(r'\p{Script=Arabic}+', arabic_text))
# => ['مرحبا', 'بالعالم']

# ヘブライ語テキスト
hebrew_text = "שלום עולם"  # "Hello World" in Hebrew
print(regex.findall(r'\p{Script=Hebrew}+', hebrew_text))
# => ['שלום', 'עולם']

# RTL と LTR が混在するテキスト
mixed = "Hello مرحبا World عالم"
# ラテン文字とアラビア文字を別々に抽出
latin = regex.findall(r'\p{Script=Latin}+', mixed)
arabic = regex.findall(r'\p{Script=Arabic}+', mixed)
print(f"Latin: {latin}, Arabic: {arabic}")
# => Latin: ['Hello', 'World'], Arabic: ['مرحبا', 'عالم']

# BiDi (双方向)テキストの処理
# Unicode の方向制御文字
# U+200E: LEFT-TO-RIGHT MARK (LRM)
# U+200F: RIGHT-TO-LEFT MARK (RLM)
# U+202A-U+202E: 方向制御文字
# これらは不可視だが、テキスト処理に影響する

# 方向制御文字を除去する正規表現
bidi_cleanup = regex.compile(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]')
clean_text = bidi_cleanup.sub('', mixed)
print(f"BiDi除去後: {clean_text}")
```

### 7.3 中国語(簡体字/繁体字)の処理

```python
import regex

# 簡体字と繁体字の判定
# Unicode では CJK統合漢字として同一コードポイントの場合が多い
# 完全な判定には専用ライブラリが必要

simplified_text = "简体中文测试"  # 簡体字
traditional_text = "繁體中文測試"  # 繁体字
japanese_text = "日本語漢字テスト"

# CJK統合漢字の抽出
for label, text in [("簡体", simplified_text),
                     ("繁体", traditional_text),
                     ("日本語", japanese_text)]:
    cjk = regex.findall(r'\p{Script=Han}+', text)
    print(f"  {label}: {cjk}")

# CJK 統合漢字のコードポイント範囲
# U+4E00-U+9FFF: CJK統合漢字(基本)
# U+3400-U+4DBF: CJK統合漢字拡張A
# U+20000-U+2A6DF: CJK統合漢字拡張B
# U+2A700-U+2B73F: CJK統合漢字拡張C
# U+2B740-U+2B81F: CJK統合漢字拡張D
# U+2B820-U+2CEAF: CJK統合漢字拡張E
# U+2CEB0-U+2EBEF: CJK統合漢字拡張F
# U+30000-U+3134F: CJK統合漢字拡張G
```

### 7.4 韓国語(ハングル)の処理

```python
import regex

korean_text = "한국어 테스트 123 Hello"

# ハングル音節の抽出
hangul = regex.findall(r'\p{Script=Hangul}+', korean_text)
print(f"ハングル: {hangul}")
# => ['한국어', '테스트']

# ハングルの構造:
# ハングル音節 = 初声(子音) + 中声(母音) + 終声(子音、任意)
# U+AC00-U+D7AF: ハングル音節 (11,172文字)
# U+1100-U+11FF: ハングル字母 (Jamo)
# U+3130-U+318F: ハングル互換字母

# ハングル音節文字をJamoに分解
def decompose_hangul(ch: str) -> tuple[str, str, str]:
    """ハングル音節を初声・中声・終声に分解"""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return (ch, '', '')

    # 初声: 19種, 中声: 21種, 終声: 28種(なし含む)
    initial = code // (21 * 28)
    medial = (code % (21 * 28)) // 28
    final = code % 28

    initials = "ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒ"
    medials = "ᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵ"
    finals = "\0ᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ"

    i = initials[initial]
    m = medials[medial]
    f = finals[final] if final > 0 else ''

    return (i, m, f)

# テスト
for ch in "한국":
    i, m, f = decompose_hangul(ch)
    print(f"  {ch} => 初声:{i} 中声:{m} 終声:{f}")
```

### 7.5 インド系文字(デーヴァナーガリー)の処理

```python
import regex

# ヒンディー語のテキスト処理
hindi_text = "नमस्ते दुनिया 123"  # "Hello World" in Hindi

# デーヴァナーガリー文字の抽出
devanagari = regex.findall(r'\p{Script=Devanagari}+', hindi_text)
print(f"デーヴァナーガリー: {devanagari}")
# => ['नमस्ते', 'दुनिया']

# デーヴァナーガリーの数字
hindi_numbers = "१२३४५६७८९०"  # 1234567890 in Devanagari
print(regex.findall(r'\p{Nd}+', hindi_numbers))
# => ['१२३४५६७८९०']

# 結合文字の処理
# デーヴァナーガリーでは結合文字(virama = ्)で子音が連結される
# "स्ते" = स + ् + त + े (4コードポイント, 1書記素クラスタ)
graphemes = regex.findall(r'\X', "नमस्ते")
print(f"書記素クラスタ: {graphemes} (数: {len(graphemes)})")
```

---

## 8. ASCII 図解

### 8.1 Unicode プロパティの階層

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

### 8.2 正規化形式の関係図

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

正規化の選択フロー:

  テキストの保存・交換?
  ├─ Yes → NFC (Web標準、最も一般的)
  └─ No
      検索・照合・比較?
      ├─ Yes → NFKC (互換文字の統一)
      └─ No
          アクセント記号の個別処理?
          ├─ Yes → NFD (分解形式)
          └─ No → NFC (デフォルト推奨)
```

### 8.3 サロゲートペアの仕組み

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

### 8.4 UTF-8 エンコーディングの仕組み

```
UTF-8 のバイト構造:

1バイト文字 (U+0000-U+007F):
  0xxxxxxx
  例: 'A' = 01000001 = 0x41

2バイト文字 (U+0080-U+07FF):
  110xxxxx 10xxxxxx
  例: 'é' (U+00E9) = 11000011 10101001 = 0xC3 0xA9

3バイト文字 (U+0800-U+FFFF):
  1110xxxx 10xxxxxx 10xxxxxx
  例: '漢' (U+6F22) = 11100110 10111100 10100010 = 0xE6 0xBC 0xA2

4バイト文字 (U+10000-U+10FFFF):
  11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
  例: '😀' (U+1F600) = 11110000 10011111 10011000 10000000
                      = 0xF0 0x9F 0x98 0x80

正規表現エンジンはコードポイント単位で動作するため、
UTF-8のバイト構造を直接意識する必要はない。
ただし、バイト列のパターンマッチ(grep -P等)では注意が必要。
```

---

## 9. 比較表

### 9.1 Unicode プロパティのサポート状況

| プロパティ | Python re | Python regex | JavaScript | Java | Go | Rust | Perl |
|-----------|----------|-------------|------------|------|----|------|------|
| `\p{L}` | 不可 | OK | OK (ES2018+u) | OK | OK | OK | OK |
| `\p{Lu}` | 不可 | OK | OK | OK | OK | OK | OK |
| `\p{Script=Han}` | 不可 | OK | OK | 不可 | 不可 | 不可 | OK |
| `\p{Han}` (短縮形) | 不可 | OK | 不可 | 不可 | OK | OK | OK |
| `\p{Emoji}` | 不可 | OK | OK | 不可 | 不可 | 不可 | OK |
| `\p{Block=CJK}` | 不可 | OK | 不可 | OK | 不可 | 不可 | OK |
| `\X` (書記素) | 不可 | OK | 不可 | 不可 | 不可 | 不可 | OK |
| Unicode `\w` | デフォルト | デフォルト | `/u` 必要 | フラグ要 | 不可(`\p{L}`使用) | デフォルト | デフォルト |

### 9.2 正規化形式の使い分け

| 形式 | 用途 | 特徴 | 推奨場面 |
|------|------|------|----------|
| NFC | テキスト保存・交換の標準 | 合成形式。Web標準で推奨 | HTML, JSON, DB保存 |
| NFD | 分解して処理したい場合 | アクセント記号を分離 | テキスト解析、ソート |
| NFKC | 検索・照合 | 互換文字を統一(全角→半角等) | 全文検索、ユーザー入力の正規化 |
| NFKD | 検索の前処理 | 最大限に分解 | インデックス構築 |

### 9.3 \w の言語間比較

| 言語 | `\w` のデフォルト範囲 | Unicode対応方法 |
|------|---------------------|----------------|
| Python 3 | Unicode文字 + 数字 + `_` | デフォルトでUnicode対応 |
| JavaScript (no flag) | `[a-zA-Z0-9_]` | `/u` フラグで拡張なし |
| JavaScript `/u` | `[a-zA-Z0-9_]` | `\p{L}` を使う |
| Java | `[a-zA-Z_0-9]` | `UNICODE_CHARACTER_CLASS` フラグ |
| Go | `[0-9A-Za-z_]` | `\p{L}` を使う |
| Rust | Unicode文字 + 数字 + `_` | `(?-u)` でASCII限定 |
| Perl | Unicode文字 + 数字 + `_` | デフォルトでUnicode対応 |

---

## 10. 実践パターン集

### 10.1 メールアドレスの国際化対応

```python
import regex

# 国際化メールアドレス (RFC 6531)
# ローカルパートとドメインにUnicode文字を許可

# 基本的なメールアドレスパターン(ASCII)
ascii_email = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Unicode対応メールアドレスパターン
unicode_email = r'[\p{L}\p{N}._%+-]+@[\p{L}\p{N}.-]+\.[\p{L}]{2,}'

test_emails = [
    "user@example.com",           # 通常
    "田中@例え.jp",                # 日本語(国際化)
    "пользователь@пример.рф",     # ロシア語(国際化)
    "user@例え.com",               # 混在
]

import re
for email in test_emails:
    ascii_match = bool(re.match(ascii_email, email))
    unicode_match = bool(regex.match(unicode_email, email))
    print(f"  {email}: ASCII={ascii_match}, Unicode={unicode_match}")
```

### 10.2 電話番号の国際対応

```python
import re

# 国際電話番号パターン (E.164 形式)
# +[国番号][番号] で最大15桁
e164_pattern = r'\+[1-9]\d{1,14}'

# 各国の電話番号フォーマット
phone_patterns = {
    'JP': r'(?:0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}|\+81\s?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4})',
    'US': r'(?:\+1[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}',
    'UK': r'(?:\+44[-\s]?)?\d{2,5}[-\s]?\d{3,8}',
}

test_numbers = [
    "+81-90-1234-5678",   # 日本
    "03-1234-5678",        # 日本(市外局番付き)
    "+1 (555) 123-4567",   # アメリカ
    "+44 20 7123 4567",    # イギリス
]

for num in test_numbers:
    for country, pattern in phone_patterns.items():
        if re.search(pattern, num):
            print(f"  {num} => {country}")
            break
```

### 10.3 Unicode 対応のバリデーション

```python
import regex
import unicodedata

def validate_username(username: str) -> tuple[bool, list[str]]:
    """Unicode対応のユーザー名バリデーション

    ルール:
    - 3〜20文字（書記素クラスタ単位）
    - Unicode文字、数字、アンダースコア、ハイフンのみ
    - 先頭は文字のみ
    - 制御文字、不可視文字を含まない
    - 見た目の混乱を避けるため、複数の書記体系の混在を禁止
    """
    errors = []

    # 書記素クラスタ数をカウント
    graphemes = regex.findall(r'\X', username)
    if len(graphemes) < 3:
        errors.append("3文字以上必要")
    if len(graphemes) > 20:
        errors.append("20文字以下にしてください")

    # 許可文字チェック
    if not regex.match(r'^[\p{L}\p{N}_-]+$', username):
        errors.append("文字、数字、_、- のみ使用可能")

    # 先頭文字チェック
    if username and not regex.match(r'^\p{L}', username):
        errors.append("先頭は文字である必要があります")

    # 制御文字チェック
    if regex.search(r'\p{C}', username):
        errors.append("制御文字は使用できません")

    # 書記体系の混在チェック(Confusable攻撃対策)
    scripts = set()
    for ch in username:
        cat = unicodedata.category(ch)
        if cat.startswith('L'):
            try:
                script = unicodedata.script(ch) if hasattr(unicodedata, 'script') else 'Unknown'
            except:
                script = 'Unknown'
            if script not in ('Common', 'Inherited', 'Unknown'):
                scripts.add(script)

    if len(scripts) > 1:
        errors.append(f"複数の書記体系の混在は禁止: {scripts}")

    return (len(errors) == 0, errors)

# テスト
test_usernames = [
    "alice",           # OK: ASCII
    "田中太郎",         # OK: 日本語
    "Алексей",         # OK: キリル文字
    "ab",              # NG: 短すぎる
    "alice_tanaka",    # OK: ASCII + アンダースコア
    "123start",        # NG: 数字で始まる
]

for username in test_usernames:
    valid, errors = validate_username(username)
    status = "OK" if valid else "NG"
    print(f"  '{username}': {status} {errors if errors else ''}")
```

### 10.4 Confusable文字の検出

```python
import regex

# Confusable (混同しやすい文字) の例
# これらはフィッシングやなりすましに悪用される

confusable_pairs = [
    ('a', 'а'),      # Latin 'a' vs Cyrillic 'а' (U+0430)
    ('e', 'е'),      # Latin 'e' vs Cyrillic 'е' (U+0435)
    ('o', 'о'),      # Latin 'o' vs Cyrillic 'о' (U+043E)
    ('p', 'р'),      # Latin 'p' vs Cyrillic 'р' (U+0440)
    ('c', 'с'),      # Latin 'c' vs Cyrillic 'с' (U+0441)
    ('x', 'х'),      # Latin 'x' vs Cyrillic 'х' (U+0445)
    ('0', 'О'),      # Digit '0' vs Cyrillic 'О' (U+041E)
    ('1', 'l'),      # Digit '1' vs Latin 'l'
    ('I', 'l'),      # Latin 'I' vs Latin 'l'
]

# 混合スクリプトの検出
def detect_mixed_scripts(text: str) -> bool:
    """Confusable攻撃の可能性がある混合スクリプトを検出"""
    has_latin = bool(regex.search(r'\p{Script=Latin}', text))
    has_cyrillic = bool(regex.search(r'\p{Script=Cyrillic}', text))
    has_greek = bool(regex.search(r'\p{Script=Greek}', text))

    # ラテン文字とキリル文字、ギリシャ文字の混在は危険
    scripts = sum([has_latin, has_cyrillic, has_greek])
    return scripts > 1

# テスト
suspicious_urls = [
    "example.com",      # 正常
    "ехаmple.com",      # Cyrillic 'е' と 'х' が混在
    "gооgle.com",       # Cyrillic 'о' が混在
    "paypal.com",       # 正常
    "раypal.com",       # Cyrillic 'р' と 'а' が混在
]

for url in suspicious_urls:
    is_mixed = detect_mixed_scripts(url)
    if is_mixed:
        print(f"  [WARNING] '{url}' -- 混合スクリプト検出!")
    else:
        print(f"  [OK] '{url}'")
```

---

## 11. アンチパターン

### 11.1 アンチパターン: Unicode範囲のハードコード

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

### 11.2 アンチパターン: 正規化せずに比較

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

### 11.3 アンチパターン: . で全文字にマッチすると想定

```python
import re

# NG: . は改行にマッチしないだけでなく、
# u フラグなし(JavaScript)ではサロゲートペアの半分にマッチする

# Python では問題ないが...
text = "Hello 😀 World"
print(re.findall(r'.', text))
# Python 3: 正しく1コードポイントずつ

# NG: 絵文字の結合シーケンスが壊れる
text = "👨‍👩‍👧‍👦"
print(re.findall(r'.', text))
# => ['👨', '\u200d', '👩', '\u200d', '👧', '\u200d', '👦']
# ZWJ で結合された絵文字が分解される

# OK: 書記素クラスタ単位で処理
import regex
print(regex.findall(r'\X', text))
# => ['👨\u200d👩\u200d👧\u200d👦']  -- 1つの書記素クラスタ
```

### 11.4 アンチパターン: len() で文字数を判断

```python
# NG: len() はコードポイント数を返す
text1 = "café"           # NFC: 4コードポイント
text2 = "cafe\u0301"     # NFD: 5コードポイント
text3 = "👨‍👩‍👧‍👦"          # 7コードポイント

print(f"len(text1) = {len(text1)}")  # => 4
print(f"len(text2) = {len(text2)}")  # => 5 (見た目は4文字なのに!)
print(f"len(text3) = {len(text3)}")  # => 7 (見た目は1文字なのに!)

# OK: 書記素クラスタ数でカウント
import regex

def visual_length(text: str) -> int:
    """見た目の文字数を返す"""
    return len(regex.findall(r'\X', text))

print(f"visual_length(text1) = {visual_length(text1)}")  # => 4
print(f"visual_length(text2) = {visual_length(text2)}")  # => 4
print(f"visual_length(text3) = {visual_length(text3)}")  # => 1
```

### 11.5 アンチパターン: 特定のエンコーディングを前提とする

```python
import re

# NG: バイト単位で文字を処理
text = "漢字"
# bad: text.encode('utf-8')[0:3] で1文字取得を想定
# UTF-8では漢字は3バイトだが、他のエンコーディングでは異なる

# OK: 文字列レベルで処理
first_char = text[0]  # '漢'
print(first_char)

# NG: 正規表現でバイトパターンを使う
# bad: re.findall(rb'\xe6[\x80-\xbf][\x80-\xbf]', text.encode())
# これはUTF-8 の3バイト文字にマッチするが脆い

# OK: 文字列レベルの正規表現を使う
print(re.findall(r'[\u4e00-\u9fff]', text))
# => ['漢', '字']

# さらに良い: Unicode プロパティを使う
import regex
print(regex.findall(r'\p{Script=Han}', text))
# => ['漢', '字']
```

---

## 12. FAQ

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

### Q4: Unicode のバージョンアップで正規表現の挙動が変わることはあるか？

**A**: ある。Unicode の新バージョンで文字が追加されると、`\p{L}` や `\p{Script=Han}` のマッチ範囲が変わる可能性がある:

```python
# 例: Unicode 15.0 で追加された文字は、
# 古いバージョンの Python/regex では認識されない

# 対策:
# 1. 処理系のバージョンを定期的に更新する
# 2. 特定の Unicode バージョンに依存する場合は明示的にドキュメントする
# 3. ユニットテストで対応文字の範囲を検証する
```

### Q5: 正規表現で Unicode の正規化を自動的に行う方法はあるか？

**A**: 一部のエンジンでは「正規化マッチ」をサポートしているが、一般的ではない:

```python
# ICU (International Components for Unicode) ベースのエンジン:
# CANONICAL_EQUIVALENCE フラグで正規等価なパターンマッチが可能

# Java の場合:
# Pattern.compile("café", Pattern.CANON_EQ)
# これは "café" (NFC) と "cafe\u0301" (NFD) の両方にマッチ

# Python では事前に正規化するのが標準的なアプローチ:
import unicodedata
import re

def canonical_search(pattern: str, text: str):
    """正準等価マッチ"""
    nfc_text = unicodedata.normalize('NFC', text)
    nfc_pattern = unicodedata.normalize('NFC', pattern)
    return re.findall(nfc_pattern, nfc_text)
```

### Q6: `\w` と `\p{L}` の違いは？

**A**: `\w` は「単語文字」で `[\p{L}\p{N}\p{Pc}]`（文字、数字、コネクタ句読点）に相当する。`\p{L}` は「文字」のみ:

```python
import regex

text = "hello_123_世界"

# \w: 文字 + 数字 + _
print(regex.findall(r'\w+', text))
# => ['hello_123_世界']

# \p{L}: 文字のみ
print(regex.findall(r'\p{L}+', text))
# => ['hello', '世界']

# \p{N}: 数字のみ
print(regex.findall(r'\p{N}+', text))
# => ['123']
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| `\p{L}` | Unicode の全文字にマッチ |
| `\p{N}` | Unicode の全数字にマッチ |
| `\p{Script=Han}` | 漢字のみにマッチ |
| `\p{Script=Hiragana}` | ひらがなのみにマッチ |
| `\p{Script=Katakana}` | カタカナのみにマッチ |
| `\p{Emoji}` | 絵文字にマッチ |
| `\X` | 書記素クラスタ(regex モジュール) |
| NFC | 合成形式(Web標準、テキスト保存) |
| NFD | 分解形式(アクセント処理) |
| NFKC | 互換分解+合成(検索向け、全角→半角) |
| NFKD | 互換分解(最大限に分解) |
| `/u` フラグ | JavaScript で Unicode 対応を有効化 |
| `/v` フラグ | JavaScript ES2024 集合演算 |
| 鉄則 | 検索前に正規化、プロパティはハードコードしない |
| 文字数 | len() ではなく書記素クラスタでカウント |

## 次に読むべきガイド

- [03-performance.md](./03-performance.md) -- パフォーマンスと ReDoS 対策
- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- 言語別正規表現の違い

## 参考文献

1. **Unicode Technical Standard #18** "Unicode Regular Expressions" https://unicode.org/reports/tr18/ -- Unicode 正規表現の国際標準仕様
2. **Unicode Technical Report #15** "Unicode Normalization Forms" https://unicode.org/reports/tr15/ -- 正規化形式の公式仕様
3. **Unicode Technical Standard #51** "Unicode Emoji" https://unicode.org/reports/tr51/ -- 絵文字の公式仕様
4. **Mathias Bynens** "JavaScript has a Unicode problem" https://mathiasbynens.be/notes/javascript-unicode -- JavaScript における Unicode の問題点と対策
5. **Python regex module** https://github.com/mrabarnett/mrab-regex -- Python の高機能正規表現モジュール
6. **TC39 RegExp v flag proposal** https://github.com/tc39/proposal-regexp-v-flag -- JavaScript ES2024 の v フラグ仕様
7. **Unicode CLDR** https://cldr.unicode.org/ -- Unicode 共通ロケールデータリポジトリ(ロケール別の文字処理)
8. **Unicode Confusables** https://unicode.org/reports/tr39/ -- Unicode セキュリティメカニズム(Confusable文字の検出)
