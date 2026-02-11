# パフォーマンス -- ReDoS、バックトラック爆発、最適化

> 正規表現のパフォーマンス問題は、セキュリティ脆弱性(ReDoS)からサービス停止まで深刻な影響を及ぼす。バックトラック爆発の原理を正確に理解し、安全で高速なパターンを設計する方法を解説する。

## この章で学ぶこと

1. **ReDoS(Regular Expression Denial of Service)の原理** -- なぜ特定のパターンが指数的に遅くなるのか
2. **バックトラック爆発の検出と回避** -- 危険なパターンの識別方法と安全な代替
3. **正規表現の最適化テクニック** -- コンパイル、アンカー、否定文字クラス、独占的量指定子

---

## 1. ReDoS とは

### 1.1 ReDoS の概要

```
ReDoS (Regular Expression Denial of Service):
悪意のある入力により正規表現エンジンを
指数的な時間計算量に陥らせるサービス拒否攻撃

攻撃の流れ:
┌──────────┐    脆弱なパターン    ┌──────────────┐
│ 攻撃者    │ ─────────────────→ │ Webサーバー   │
│          │   悪意のある入力     │              │
│          │                    │  正規表現      │
│          │                    │  エンジンが    │
│          │                    │  無限ループ状態│
│          │                    │  → CPU 100%   │
│          │                    │  → DoS        │
└──────────┘                    └──────────────┘
```

### 1.2 脆弱なパターンの例

```python
import re
import time

# 脆弱なパターン: (a+)+b
pattern = re.compile(r'(a+)+b')

# 安全な入力: 即座に完了
start = time.time()
pattern.search("aaaaab")
print(f"安全な入力: {time.time() - start:.4f}秒")

# 悪意のある入力: 指数的に遅くなる
# 注意: 以下は実行すると非常に遅くなる
for length in [15, 20, 25]:
    malicious = "a" * length + "c"  # マッチしない入力
    start = time.time()
    pattern.search(malicious)
    elapsed = time.time() - start
    print(f"長さ {length}: {elapsed:.4f}秒")

# 出力例:
# 長さ 15: 0.01秒
# 長さ 20: 0.30秒
# 長さ 25: 9.50秒    ← 指数的増加!
```

### 1.3 バックトラック爆発のメカニズム

```
パターン: (a+)+b
入力:     "aaaaac" (6文字、マッチしない)

エンジンの動作 -- 全ての分割を試行:

試行1: (aaaaa)b     → b ≠ c → 失敗
試行2: (aaaa)(a)b   → b ≠ c → 失敗
試行3: (aaa)(aa)b   → b ≠ c → 失敗
試行4: (aaa)(a)(a)b → b ≠ c → 失敗
試行5: (aa)(aaa)b   → b ≠ c → 失敗
試行6: (aa)(aa)(a)b → b ≠ c → 失敗
試行7: (aa)(a)(aa)b → b ≠ c → 失敗
...

n 文字の 'a' に対して、約 2^n 通りの分割を試行
→ n=25 で約 3,300万通り
→ n=30 で約 10億通り!
```

---

## 2. 危険なパターンの分類

### 2.1 ReDoS脆弱パターンの3類型

```
┌──────────────────────────────────────────────────┐
│              ReDoS 脆弱パターンの類型              │
├──────────────────────────────────────────────────┤
│                                                  │
│ 1. 量指定子のネスト (Nested Quantifiers)          │
│    (a+)+   (a*)*   (a+)*   (a*)+                │
│    → 内側と外側の量指定子が同じ文字にマッチ         │
│                                                  │
│ 2. 選択の重複 (Overlapping Alternation)           │
│    (a|a)+  (a|ab)+  (\w|\d)+                    │
│    → 選択肢が同じ文字にマッチ可能                  │
│                                                  │
│ 3. 量指定子の重複 (Overlapping Quantifiers)       │
│    \d+\d+  a+a+  .*.*                           │
│    → 連続する量指定子が同じ文字を奪い合う           │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 2.2 実際の脆弱パターン例

```python
# 脆弱パターンのカタログ (実行は避けること)

vulnerable_patterns = {
    # 量指定子のネスト
    "(a+)+":           "aaaaaaaaaaaaaaac",
    "(a+)+b":          "aaaaaaaaaaaaaaac",
    "(a*)*b":          "aaaaaaaaaaaaaaac",
    "([a-z]+)+$":      "aaaaaaaaaaaaaaa!",

    # 選択の重複
    "(a|a)+b":         "aaaaaaaaaaaaaaac",
    "(\\w|\\d)+$":     "aaaaaaaaaaaaaaaa!",
    "(.*a){20}":       "aaaaaaaaaaaaaaaaaaaaX",

    # 実世界の脆弱パターン
    # メールアドレス検証
    r"^([a-zA-Z0-9])(([\-.]|[_]+)?([a-zA-Z0-9]+))*(@)":
        "aaaaaaaaaaaaaaaaaaaaa!",

    # URL 検証
    r"^(https?://)([\w-]+(\.[\w-]+)+)(/[\w-./?%&=]*)*$":
        "http://a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p!",
}
```

---

## 3. 安全なパターン設計

### 3.1 修正の基本原則

```python
import re

# 原則1: 量指定子のネストを排除

# NG: (a+)+
# OK: a+
pattern_safe1 = r'a+b'

# 原則2: 選択肢の重複を排除

# NG: (\w|\d)+  (\w は \d を含む)
# OK: \w+
pattern_safe2 = r'\w+$'

# 原則3: 否定文字クラスで明確に制限

# NG: ".*"  (貪欲で " を超えてマッチ)
# OK: "[^"]*"  (引用符内のみ)
pattern_safe3 = r'"[^"]*"'

# 原則4: アトミックグループまたは独占的量指定子を使用
# (Python regex モジュール、Java、PCRE)

# NG: (a+)+b       → バックトラック爆発
# OK: (?>a+)+b     → アトミックグループ (バックトラック禁止)
```

### 3.2 パターン修正例

```python
import re

# 例1: メールアドレス検証
# NG: 脆弱
email_bad = r'^([a-zA-Z0-9])(([\-.]|[_]+)?([a-zA-Z0-9]+))*(@)'

# OK: 安全で実用的
email_good = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# 例2: HTML タグの内容抽出
# NG: 脆弱 (ネストした量指定子)
tag_bad = r'<(\w+)(\s+\w+="[^"]*")*>'

# OK: 安全
tag_good = r'<\w+(?:\s+\w+="[^"]*")*>'

# 例3: CSV フィールド
# NG: 脆弱
csv_bad = r'^("(?:[^"]|"")*"|[^,]*)(,("(?:[^"]|"")*"|[^,]*))*$'

# OK: フィールド単位で処理
csv_good = r'"(?:[^"]|"")*"|[^,]+'

text = '"hello, world","test""quote"",normal'
print(re.findall(csv_good, text))
```

### 3.3 タイムアウト機構

```python
import re
import signal

# Python: signal を使ったタイムアウト(Unix系のみ)
class RegexTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise RegexTimeout("正規表現がタイムアウトしました")

def safe_search(pattern, text, timeout_sec=1):
    """タイムアウト付き正規表現検索"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = re.search(pattern, text)
        signal.alarm(0)  # タイマー解除
        return result
    except RegexTimeout:
        return None

# 使用例
result = safe_search(r'(a+)+b', "a" * 30 + "c", timeout_sec=2)
if result is None:
    print("タイムアウトまたはマッチなし")
```

```javascript
// JavaScript: 安全な正規表現のサードパーティライブラリ
// re2 パッケージ (RE2 エンジンのバインディング)
// npm install re2

// const RE2 = require('re2');
// const pattern = new RE2('(a+)+b');  // RE2 ではこれは自動的に最適化される
```

---

## 4. パフォーマンス最適化

### 4.1 コンパイル(プリコンパイル)

```python
import re
import time

text = "The quick brown fox jumps over the lazy dog" * 1000

# NG: ループ内で毎回コンパイル
start = time.time()
for _ in range(10000):
    re.search(r'\b\w{5}\b', text)
time_uncompiled = time.time() - start

# OK: 事前コンパイル
pattern = re.compile(r'\b\w{5}\b')
start = time.time()
for _ in range(10000):
    pattern.search(text)
time_compiled = time.time() - start

print(f"未コンパイル: {time_uncompiled:.3f}秒")
print(f"コンパイル済: {time_compiled:.3f}秒")
# コンパイル済みのほうが高速(特に繰り返し使用時)

# 注: Python は内部的にキャッシュ(最大512個)を持つが
# 明示的なコンパイルが推奨される
```

### 4.2 アンカーによる高速化

```python
import re

# アンカーがあるとエンジンが不要な位置をスキップできる

# 遅い: 全位置で試行
slow = r'\d{4}-\d{2}-\d{2}'

# 速い: 行頭のみで試行
fast = r'^\d{4}-\d{2}-\d{2}'

# 速い: 単語境界で限定
fast2 = r'\b\d{4}-\d{2}-\d{2}\b'
```

### 4.3 否定文字クラス vs 怠惰量指定子

```python
import re
import time

html = '<div class="test">' * 10000 + 'content</div>'

# 遅い: 怠惰量指定子(バックトラックが発生)
start = time.time()
re.search(r'<div.*?>', html)
lazy_time = time.time() - start

# 速い: 否定文字クラス(バックトラックなし)
start = time.time()
re.search(r'<div[^>]*>', html)
negated_time = time.time() - start

print(f"怠惰量指定子: {lazy_time:.6f}秒")
print(f"否定文字クラス: {negated_time:.6f}秒")
```

### 4.4 最適化テクニック一覧

```python
import re

# 1. 固定文字列の先行チェック
text = "long text without the target pattern..."

# 遅い: 毎回正規表現で検索
if re.search(r'target\s+\w+\s+pattern', text):
    pass

# 速い: 文字列検索で先行チェック
if 'target' in text and 'pattern' in text:
    if re.search(r'target\s+\w+\s+pattern', text):
        pass

# 2. 固定プレフィックスの活用
# エンジンは固定プレフィックスを最適化できる

# 最適化しやすい: 固定文字列で始まる
good = r'ERROR: \w+'

# 最適化しにくい: 可変パターンで始まる
bad = r'.*ERROR: \w+'

# 3. 不要なキャプチャを排除
# キャプチャグループ → 非キャプチャグループ
slow = r'(https?)://([\w.]+)/([\w/]+)'
fast = r'(?:https?)://(?:[\w.]+)/(?:[\w/]+)'

# 4. 選択の順序最適化
# 頻度の高い選択肢を先に
slow_alt = r'(?:rare_pattern|common_pattern)'
fast_alt = r'(?:common_pattern|rare_pattern)'
```

---

## 5. ASCII 図解

### 5.1 バックトラック爆発の可視化

```
パターン: (a+)+b
入力:     "aaac" (4文字)

全試行の木構造:

(aaaa) → b? → c ≠ b → 失敗
│
├── (aaa)(a) → b? → c ≠ b → 失敗
│
├── (aa)(aa) → b? → c ≠ b → 失敗
│   └── (aa)(a)(a) → b? → c ≠ b → 失敗
│
├── (a)(aaa) → b? → c ≠ b → 失敗
│   ├── (a)(aa)(a) → b? → c ≠ b → 失敗
│   └── (a)(a)(aa) → b? → c ≠ b → 失敗
│       └── (a)(a)(a)(a) → b? → c ≠ b → 失敗
│
計: 8通り (入力長 n=4)
n=10 → 512通り
n=20 → 524,288通り
n=30 → 536,870,912通り (5億以上!)
```

### 5.2 安全なパターンと脆弱なパターンの対比

```
脆弱なパターン          安全な代替            理由
──────────────        ──────────────       ──────
(a+)+                 a+                   ネスト不要
(a|a)+                a+                   重複排除
(.*a){n}              最大長チェック         量指定子の入れ子を排除
".*"                  "[^"]*"              範囲を明確に制限
(\w+\s*)+             [\w\s]+              フラット化
(.+)+                 .+                   ネスト不要
(a+|b+)+              [ab]+                フラット化
```

### 5.3 パフォーマンス改善のフローチャート

```
正規表現が遅い?
    │
    ├── パターンを分析
    │   │
    │   ├── 量指定子のネストあり?
    │   │   └── YES → フラット化 or アトミック化
    │   │
    │   ├── 選択肢に重複あり?
    │   │   └── YES → 文字クラスに統合
    │   │
    │   ├── .* を使用?
    │   │   └── YES → [^X]* に置換
    │   │
    │   └── アンカーなし?
    │       └── YES → ^, \b 等を追加
    │
    ├── コンパイルしているか?
    │   └── NO → re.compile() を使用
    │
    ├── 入力が信頼できないか?
    │   ├── YES → RE2/DFA エンジンを検討
    │   └── YES → タイムアウトを設定
    │
    └── それでも遅い?
        └── 正規表現以外の方法を検討
            (文字列操作、パーサー等)
```

---

## 6. 比較表

### 6.1 エンジン別パフォーマンス特性

| エンジン | 最悪計算量 | ReDoS耐性 | 機能 | 言語 |
|---------|-----------|----------|------|------|
| Python re | O(2^n) | なし | 豊富 | Python |
| Python regex | O(2^n) | なし | 最も豊富 | Python |
| JavaScript V8 | O(2^n) | なし | 豊富 | JavaScript |
| Java Pattern | O(2^n) | なし | 豊富 | Java |
| RE2 | O(n) | あり | 限定的 | Go, C++ |
| Rust regex | O(n) | あり | 限定的 | Rust |
| PCRE2 JIT | O(2^n) | なし(JITで高速化) | 最も豊富 | C |
| .NET | O(2^n) | タイムアウトあり | 豊富 | C# |

### 6.2 最適化テクニック効果比較

| テクニック | 効果 | 適用場面 | 実装コスト |
|-----------|------|---------|-----------|
| re.compile() | 小〜中 | 繰り返し使用時 | 低 |
| アンカー追加 | 中 | 位置が既知の場合 | 低 |
| 否定文字クラス | 中〜大 | タグ抽出等 | 低 |
| 独占的量指定子 | 大 | バックトラック防止 | 中 |
| DFAエンジン(RE2) | 最大 | 信頼できない入力 | 高(ライブラリ変更) |
| 先行文字列チェック | 中 | 出現頻度が低い場合 | 低 |
| パターン分割 | 中 | 複雑なパターン | 中 |

---

## 7. 脆弱性の検出ツール

### 7.1 静的解析ツール

```bash
# recheck: 正規表現の脆弱性を検出
# npm install -g recheck
# recheck "(a+)+b"

# redos-checker (Python)
# pip install redos-checker

# safe-regex (JavaScript)
# npm install safe-regex
```

```javascript
// safe-regex の使用例
// const safe = require('safe-regex');
// console.log(safe('(a+)+b'));      // => false (脆弱)
// console.log(safe('[a-z]+'));       // => true  (安全)
// console.log(safe('(a|b|c)+'));     // => true  (安全)
```

```python
# Python での簡易チェッカー
import re

def is_potentially_vulnerable(pattern: str) -> tuple[bool, str]:
    """正規表現パターンの脆弱性を簡易チェック"""
    warnings = []

    # 量指定子のネスト検出
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern):
        warnings.append("量指定子のネストが検出されました")

    # .* の無制限使用検出
    if re.search(r'\.\*(?!\?)', pattern) and '^' not in pattern:
        warnings.append("アンカーなしの .* が検出されました")

    # 選択肢の重複検出(簡易)
    if re.search(r'\((?:[^)]*\|[^)]*)\)[+*]', pattern):
        warnings.append("選択肢を含む繰り返しが検出されました")

    return (len(warnings) > 0, warnings)

# テスト
patterns = [
    r'(a+)+b',
    r'(\w|\d)+',
    r'[a-z]+',
    r'.*error.*',
    r'^[a-z]+$',
]

for p in patterns:
    vulnerable, msgs = is_potentially_vulnerable(p)
    status = "脆弱" if vulnerable else "安全"
    print(f"  {status}: {p}")
    for msg in msgs:
        print(f"    - {msg}")
```

---

## 8. アンチパターン

### 8.1 アンチパターン: ユーザー入力を正規表現に直接使う

```python
import re

# NG: ユーザー入力をそのまま正規表現として使用
def search_bad(user_input: str, text: str):
    return re.search(user_input, text)  # ReDoS攻撃が可能!

# 攻撃例:
# user_input = "(a+)+b"
# text = "a" * 30 + "c"
# → CPU が100%に

# OK: ユーザー入力はエスケープする
def search_safe(user_input: str, text: str):
    escaped = re.escape(user_input)  # メタ文字を全てエスケープ
    return re.search(escaped, text)

# OK: または DFA エンジンを使う
# import re2
# def search_safe_re2(pattern: str, text: str):
#     return re2.search(pattern, text)  # O(n) 保証
```

### 8.2 アンチパターン: 最適化なしの大量データ処理

```python
import re

# NG: 大量のログを非効率に処理
def process_logs_bad(log_lines: list[str]):
    results = []
    for line in log_lines:
        # 毎行で複雑な正規表現を実行
        m = re.search(r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.*)', line)
        if m:
            results.append(m.groups())
    return results

# OK: プリコンパイル + 先行フィルタ
def process_logs_good(log_lines: list[str]):
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+\[(\w+)\]\s+(.*)'
    )
    results = []
    for line in log_lines:
        if '[' in line:  # 先行フィルタ: [ を含む行のみ処理
            m = pattern.search(line)
            if m:
                results.append(m.groups())
    return results
```

---

## 9. FAQ

### Q1: 自分のパターンが ReDoS に脆弱かどうかを判断するには？

**A**: 以下の3ステップでチェックする:

1. **パターンを目視確認**: 量指定子のネスト `(X+)+`、選択の重複 `(a|a)+` がないか
2. **静的解析ツールを使用**: `safe-regex`、`recheck` 等
3. **ストレステスト**: マッチしない長い入力で速度を計測

```python
import re, time

def stress_test(pattern_str, char='a', max_len=30):
    pattern = re.compile(pattern_str)
    for n in range(10, max_len + 1, 5):
        text = char * n + '!'
        start = time.time()
        pattern.search(text)
        elapsed = time.time() - start
        print(f"  n={n}: {elapsed:.4f}秒")
        if elapsed > 1.0:
            print("  → 脆弱性あり!")
            break
```

### Q2: RE2 を使えば全ての問題が解決するか？

**A**: **いいえ**。RE2 は O(n) を保証するが、後方参照やルックアラウンドが使えない。トレードオフを理解した上で選択する:

- 信頼できない入力 → RE2 を強く推奨
- 後方参照が必要 → NFA + タイムアウト + パターン監査
- 内部処理のみ → NFA で通常は問題なし

### Q3: Python の `re` モジュールにタイムアウト機能はあるか？

**A**: 標準の `re` モジュールには **ない**。対策:

1. `signal.alarm()` で外部タイムアウト(Unix系のみ)
2. `regex` モジュールの `timeout` パラメータ(v2021.4.4+)
3. 別プロセスで実行して `multiprocessing` でタイムアウト
4. RE2 バインディング (`google-re2` パッケージ)を使用

```python
# regex モジュールのタイムアウト
import regex
try:
    result = regex.search(r'(a+)+b', 'a' * 30 + 'c', timeout=1)
except TimeoutError:
    print("タイムアウト")
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| ReDoS | 正規表現によるサービス拒否攻撃 |
| 原因 | バックトラック爆発(指数的計算量) |
| 脆弱パターン | `(a+)+`, `(a\|a)+`, `.*` の無制限使用 |
| 防御策1 | 否定文字クラス `[^X]*` で範囲を限定 |
| 防御策2 | 量指定子のネストを排除 |
| 防御策3 | DFA エンジン(RE2等)を使用 |
| 防御策4 | タイムアウト機構を設定 |
| 最適化 | `re.compile()`、アンカー、先行フィルタ |
| 鉄則 | 信頼できない入力には DFA、パターンは静的解析で監査 |

## 次に読むべきガイド

- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- 言語別正規表現(エンジンの違い)
- [../02-practical/03-regex-alternatives.md](../02-practical/03-regex-alternatives.md) -- 正規表現の代替技術

## 参考文献

1. **Russ Cox** "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html, 2007 -- ReDoS の理論的背景と RE2 の設計思想
2. **OWASP** "Regular expression Denial of Service - ReDoS" https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- セキュリティ観点からの ReDoS 解説
3. **James Davis et al.** "The Impact of Regular Expression Denial of Service (ReDoS) in Practice" FSE 2018 -- ReDoS の実世界における影響の学術的調査
4. **Google RE2** https://github.com/google/re2 -- 線形時間保証の正規表現エンジン
