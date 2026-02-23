# パフォーマンス -- ReDoS、バックトラック爆発、最適化

> 正規表現のパフォーマンス問題は、セキュリティ脆弱性(ReDoS)からサービス停止まで深刻な影響を及ぼす。バックトラック爆発の原理を正確に理解し、安全で高速なパターンを設計する方法を解説する。

## この章で学ぶこと

1. **ReDoS(Regular Expression Denial of Service)の原理** -- なぜ特定のパターンが指数的に遅くなるのか
2. **バックトラック爆発の検出と回避** -- 危険なパターンの識別方法と安全な代替
3. **正規表現の最適化テクニック** -- コンパイル、アンカー、否定文字クラス、独占的量指定子
4. **エンジン別の特性理解** -- NFA vs DFA の動作原理とトレードオフ
5. **実務でのセキュリティ対策** -- 入力検証パイプラインの設計
6. **ベンチマーク手法** -- 正規表現パフォーマンスの計測と比較

---

## 1. 正規表現エンジンの基礎知識

### 1.1 NFA と DFA の違い

```
正規表現エンジンの2大アーキテクチャ:

NFA (Non-deterministic Finite Automaton) -- 非決定性有限オートマトン
  動作: パターン主導。パターンの各要素を順に試し、
       失敗したらバックトラックして別の可能性を探る。
  特徴:
  - 後方参照、ルックアラウンド、独占的量指定子をサポート
  - 最悪ケースで指数的な計算量 O(2^n)
  - 使用エンジン: Python re, JavaScript, Java, Perl, PCRE

DFA (Deterministic Finite Automaton) -- 決定性有限オートマトン
  動作: テキスト主導。入力の各文字を1回だけ処理し、
       状態遷移テーブルで次の状態を決定する。
  特徴:
  - バックトラックなし → 常に O(n) の線形時間
  - 後方参照、ルックアラウンドは原理的にサポート不可
  - 使用エンジン: RE2, Rust regex, awk, grep (基本)

ハイブリッド:
  一部のエンジンは DFA と NFA を組み合わせる
  - PCRE2 JIT: NFA ベースだが JIT コンパイルで高速化
  - .NET: NFA ベースだがタイムアウト機構あり
  - RE2: DFA メインだが、一部機能で NFA にフォールバック
```

### 1.2 バックトラッキングの動作原理

```python
import re

# バックトラッキングの詳細な動作を追跡

# パターン: a+b
# 入力: "aaac"

# エンジンの動作:
# 位置0: a+ → "aaa" にマッチ(貪欲)
#         b  → 'c' ≠ 'b' → 失敗
#         バックトラック: a+ → "aa" にマッチ
#         b  → 'a' ≠ 'b' → 失敗
#         バックトラック: a+ → "a" にマッチ
#         b  → 'a' ≠ 'b' → 失敗
# 位置1: a+ → "aa" にマッチ
#         b  → 'c' ≠ 'b' → 失敗
#         バックトラック...
# 位置2: a+ → "a" にマッチ
#         b  → 'c' ≠ 'b' → 失敗
# 位置3: a+ → マッチしない
# 結果: マッチなし

# このパターンは安全 -- バックトラックは線形回数

# 危険なパターン: (a+)+b -- バックトラックが指数的になる
```

### 1.3 NFA エンジンのバックトラック制限

```python
# 各言語のバックトラック制限
#
# Python re:      制限なし（デフォルト）
# Python regex:   timeout パラメータで制限可能
# JavaScript V8:  --regexp-backtracks-limit（デフォルト: 数百万）
# Java:           制限なし（デフォルト）
# .NET:           Regex.MatchTimeout で制限可能
# PCRE2:          pcre2_set_match_limit で制限可能
# Perl:           制限なし（デフォルト）

# JavaScript V8 の例:
# node --regexp-backtracks-limit=10000 script.js

# .NET の例:
# var regex = new Regex(pattern, RegexOptions.None,
#                       TimeSpan.FromSeconds(1));
```

---

## 2. ReDoS とは

### 2.1 ReDoS の概要

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

ReDoS の影響度:
┌─────────────────────────────────────────────┐
│ 影響レベル    │ 状況                          │
├──────────────┼──────────────────────────────┤
│ 軽微         │ 1リクエストの応答遅延          │
│ 中程度       │ ワーカースレッドの枯渇         │
│ 重大         │ サーバー全体のCPU枯渇          │
│ 致命的       │ カスケード障害、全サービス停止  │
└──────────────┴──────────────────────────────┘
```

### 2.2 実際の ReDoS インシデント事例

```
過去の重大な ReDoS インシデント:

1. Stack Overflow (2016)
   原因: HTML サニタイザーの正規表現
   影響: 34分間のダウンタイム
   パターン: 空白の繰り返しパターンに起因

2. Cloudflare (2019)
   原因: WAF ルールの正規表現
   影響: 全世界のサービスに27分間の障害
   パターン: (?:(?:\"|'|\]|\}|\\|\d|(?:nan|infinity|true|false|
             null|undefined|symbol|math)|\`|\-|\+)+[)]*;?((?:\s|
             -|~|!|{}|\|\||\+)*.*(?:.*=.*)))

3. Node.js (2017) -- CVE-2017-15896
   原因: HTTP ヘッダーパーサーの正規表現
   影響: Node.js アプリケーション全般に影響

4. npm (2018) -- event-stream
   原因: パッケージ内の正規表現処理
   影響: npm エコシステム全体に波及
```

### 2.3 脆弱なパターンの例

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

### 2.4 バックトラック爆発のメカニズム

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

なぜ指数的になるのか？
  (a+)+ は「1個以上のaを含むグループを1回以上繰り返す」
  これは "aaa" を以下のように分割できる:
    (aaa)         -- 1グループ
    (aa)(a)       -- 2グループ
    (a)(aa)       -- 2グループ
    (a)(a)(a)     -- 3グループ
  各分割パターンは整数の分割と同等 → 指数的
```

---

## 3. 危険なパターンの分類

### 3.1 ReDoS脆弱パターンの3類型

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

### 3.2 類型別の詳細な分析

```python
# === 類型1: 量指定子のネスト ===

# パターン: (a+)+
# なぜ危険？
# 内側の a+ は「1個以上のa」にマッチ
# 外側の + は「そのグループを1回以上繰り返す」
# → "aaaa" を (aa)(aa), (a)(aaa), (aaa)(a), (a)(a)(aa), ... と
#   指数的に分割可能

# パターン: (\d+)*
# なぜ危険？
# 内側の \d+ は「1個以上の数字」
# 外側の * は「0回以上繰り返す」
# → "12345" を (12)(345), (1)(2345), (123)(45), ... と分割

# === 類型2: 選択の重複 ===

# パターン: (a|ab)+
# なぜ危険？
# "ab" を a + b としてマッチするか、ab としてマッチするか
# の2通りの選択が毎回発生
# → n個の "ab" の連続に対して 2^n 通り

# パターン: (\w|\d)+
# なぜ危険？
# \d は \w に包含される
# 数字に対して \w と \d の両方を試行
# → 数字のみの入力で指数的バックトラック

# === 類型3: 量指定子の重複 ===

# パターン: \d+\d+\d+
# なぜ危険？
# "12345" を各 \d+ にどう分配するか
# (1)(2)(345), (1)(23)(45), (12)(3)(45), ...
# → 組み合わせが爆発
```

### 3.3 実際の脆弱パターン例

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

### 3.4 脆弱パターンの数学的分析

```
バックトラック回数の計算:

パターン (a+)+b に対する入力 "a^n c" のバックトラック回数:

  n文字の a を k個のグループに分割する方法の数は
  C(n-1, k-1) (組み合わせ)

  全グループ数について合計すると:
  Σ_{k=1}^{n} C(n-1, k-1) = 2^(n-1)

  → O(2^n) のバックトラック

具体例:
  n=10:  2^9  =    512 回
  n=20:  2^19 = 524,288 回
  n=25:  2^24 = 16,777,216 回
  n=30:  2^29 = 536,870,912 回 (5億回以上)

  1回のバックトラックに 100ns かかるとすると:
  n=25: 約 1.7秒
  n=30: 約 54秒
  n=35: 約 1,718秒 (約29分)

パターン (a|aa)+ に対する入力 "a^n c":
  フィボナッチ数列的に増加 → O(φ^n) ≈ O(1.618^n)
  指数的だが (a+)+ より遅い増加

パターン (\d+)+ に対する入力 "d^n c":
  (a+)+ と同等 → O(2^n)
```

---

## 4. 安全なパターン設計

### 4.1 修正の基本原則

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

# 原則5: 入力長の制限
# パターン修正だけでなく、入力自体を制限する

def safe_match(pattern, text, max_length=10000):
    """入力長を制限した安全なマッチ"""
    if len(text) > max_length:
        raise ValueError(f"入力が長すぎます: {len(text)} > {max_length}")
    return re.search(pattern, text)
```

### 4.2 パターン修正例

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

# 例4: 空白の処理
# NG: 脆弱 (\s+\s* のような重複)
whitespace_bad = r'(\s+\w+)*\s*$'

# OK: 安全 (明確な区切り)
whitespace_good = r'(?:\s+\w+)*\s*$'
# さらに安全: 否定文字クラスで制限
whitespace_best = r'(?:\s\S+)*\s*$'

# 例5: 複数行テキストの処理
# NG: .* が行を超えてマッチ
multiline_bad = r'START.*END'

# OK: 行内のみマッチ
multiline_good = r'START[^\n]*END'
# または DOTALL フラグを明示的に制御
```

### 4.3 独占的量指定子とアトミックグループ

```python
# 独占的量指定子 (Possessive Quantifier): X++, X*+, X?+
# アトミックグループ (Atomic Group): (?>X)
#
# どちらもバックトラックを禁止する
# → 一度マッチした文字は手放さない

# Python regex モジュールでのアトミックグループ
import regex

# 通常の量指定子: バックトラックする
# パターン: a+b に対して "aaac"
# a+ → "aaa" → b期待だがcが来る → バックトラック
# a+ → "aa" → b期待だがaが来る → バックトラック
# ...

# アトミックグループ: バックトラックしない
# パターン: (?>a+)b に対して "aaac"
# (?>a+) → "aaa" (バックトラック禁止) → b期待だがcが来る → 即座に失敗

pattern_atomic = regex.compile(r'(?>a+)b')
print(pattern_atomic.search("aaab"))   # => マッチ
print(pattern_atomic.search("aaac"))   # => None (即座に失敗)

# 独占的量指定子(Java, PCRE2, regex モジュール)
pattern_possessive = regex.compile(r'a++b')
print(pattern_possessive.search("aaab"))   # => マッチ
print(pattern_possessive.search("aaac"))   # => None

# ReDoS 対策としてのアトミックグループ
# NG: (a+)+b → バックトラック爆発
# OK: (?>(a+))+b → 内側の a+ がアトミック → 爆発しない
safe_pattern = regex.compile(r'(?>(a+))+b')
import time
start = time.time()
safe_pattern.search("a" * 30 + "c")
print(f"アトミック版: {time.time() - start:.4f}秒")
# => 即座に完了
```

```java
// Java での独占的量指定子
import java.util.regex.*;

public class PossessiveExample {
    public static void main(String[] args) {
        // 通常: バックトラックあり
        Pattern greedy = Pattern.compile("(a+)+b");

        // 独占的: バックトラックなし
        Pattern possessive = Pattern.compile("(a++)+b");

        String input = "a".repeat(30) + "c";

        // 独占的量指定子は即座に失敗する
        long start = System.nanoTime();
        possessive.matcher(input).find();
        long elapsed = System.nanoTime() - start;
        System.out.println("独占的: " + (elapsed / 1_000_000) + "ms");
        // => 独占的: 0ms
    }
}
```

### 4.4 タイムアウト機構

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

```python
# multiprocessing を使ったクロスプラットフォームのタイムアウト
import re
from multiprocessing import Process, Queue
import time

def regex_worker(pattern_str, text, result_queue):
    """別プロセスで正規表現を実行"""
    try:
        pattern = re.compile(pattern_str)
        result = pattern.search(text)
        result_queue.put(('success', result is not None))
    except Exception as e:
        result_queue.put(('error', str(e)))

def safe_regex_search(pattern_str, text, timeout_sec=2):
    """プロセスベースのタイムアウト付き正規表現検索"""
    result_queue = Queue()
    p = Process(target=regex_worker, args=(pattern_str, text, result_queue))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join()
        return None, "タイムアウト"

    if not result_queue.empty():
        status, result = result_queue.get()
        return result, status

    return None, "不明なエラー"

# 使用例
result, status = safe_regex_search(r'(a+)+b', "a" * 30 + "c", timeout_sec=2)
print(f"結果: {result}, ステータス: {status}")
```

```javascript
// JavaScript: 安全な正規表現のサードパーティライブラリ
// re2 パッケージ (RE2 エンジンのバインディング)
// npm install re2

// const RE2 = require('re2');
// const pattern = new RE2('(a+)+b');
// RE2 ではこれは自動的に最適化される

// Worker を使ったタイムアウト
function safeRegexTest(pattern, text, timeoutMs = 1000) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(
            URL.createObjectURL(new Blob([`
                self.onmessage = function(e) {
                    const regex = new RegExp(e.data.pattern);
                    const result = regex.test(e.data.text);
                    self.postMessage({ result });
                };
            `]))
        );

        const timer = setTimeout(() => {
            worker.terminate();
            reject(new Error('Regex timeout'));
        }, timeoutMs);

        worker.onmessage = (e) => {
            clearTimeout(timer);
            worker.terminate();
            resolve(e.data.result);
        };

        worker.postMessage({ pattern: pattern.source, text });
    });
}
```

---

## 5. パフォーマンス最適化

### 5.1 コンパイル(プリコンパイル)

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

### 5.2 アンカーによる高速化

```python
import re
import time

# アンカーがあるとエンジンが不要な位置をスキップできる

text = "2024-01-15 Some log entry here" * 10000

# 遅い: 全位置で試行
slow = r'\d{4}-\d{2}-\d{2}'

# 速い: 行頭のみで試行
fast = r'^\d{4}-\d{2}-\d{2}'

# 速い: 単語境界で限定
fast2 = r'\b\d{4}-\d{2}-\d{2}\b'

# ベンチマーク
for label, pattern in [("アンカーなし", slow), ("^付き", fast), ("\\b付き", fast2)]:
    compiled = re.compile(pattern)
    start = time.time()
    for _ in range(1000):
        compiled.search(text)
    elapsed = time.time() - start
    print(f"  {label}: {elapsed:.4f}秒")
```

### 5.3 否定文字クラス vs 怠惰量指定子

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

# なぜ否定文字クラスが速いのか？
#
# 怠惰量指定子 .*?> の動作:
#   1. .  にマッチ (0文字から開始)
#   2. > を試行 → 失敗
#   3. . にもう1文字マッチ
#   4. > を試行 → 失敗
#   5. 繰り返し... (毎回2ステップ)
#
# 否定文字クラス [^>]*> の動作:
#   1. [^>]* で > 以外の文字を一気にマッチ
#   2. > を試行 → 成功
#   (バックトラックが発生しない)
```

### 5.4 最適化テクニック一覧

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

# 5. 具体的な文字クラスの使用
# 遅い: .* (何でもマッチ → バックトラックの元)
slow_dot = r'<.*>'
# 速い: [^>]* (限定的 → バックトラック最小化)
fast_neg = r'<[^>]*>'

# 6. 必要な部分だけマッチ
# 遅い: 全体をマッチしてからグループを取得
slow_full = r'^.*?(\d{4}-\d{2}-\d{2}).*$'
# 速い: 必要な部分だけを探す
fast_part = r'\d{4}-\d{2}-\d{2}'
```

### 5.5 大量データの処理最適化

```python
import re
import time

# 大量のログファイル処理の最適化

def benchmark(name, func, lines, iterations=3):
    """ベンチマーク実行"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(lines)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    print(f"  {name}: {avg:.4f}秒 (平均)")

# テストデータ生成
log_lines = []
for i in range(100000):
    if i % 3 == 0:
        log_lines.append(f"2024-01-15 10:30:{i%60:02d} [ERROR] Something failed: {i}")
    elif i % 3 == 1:
        log_lines.append(f"2024-01-15 10:30:{i%60:02d} [INFO] Processing item {i}")
    else:
        log_lines.append(f"some other line without timestamp {i}")

# 方法1: ナイーブ（毎行で re.search）
def naive_search(lines):
    results = []
    for line in lines:
        m = re.search(r'\[ERROR\]', line)
        if m:
            results.append(line)
    return results

# 方法2: プリコンパイル
error_pattern = re.compile(r'\[ERROR\]')
def compiled_search(lines):
    results = []
    for line in lines:
        if error_pattern.search(line):
            results.append(line)
    return results

# 方法3: 文字列検索で先行フィルタ
def prefiltered_search(lines):
    results = []
    for line in lines:
        if '[ERROR]' in line:
            results.append(line)
    return results

# 方法4: リスト内包表記（Python最適化）
def list_comp_search(lines):
    return [line for line in lines if '[ERROR]' in line]

# ベンチマーク実行
print("ベンチマーク結果 (10万行):")
benchmark("ナイーブ re.search", naive_search, log_lines)
benchmark("プリコンパイル", compiled_search, log_lines)
benchmark("文字列先行フィルタ", prefiltered_search, log_lines)
benchmark("リスト内包表記", list_comp_search, log_lines)
```

### 5.6 正規表現 vs 文字列操作のベンチマーク

```python
import re
import time

# いつ正規表現を使い、いつ文字列操作を使うべきか

text = "user@example.com" * 10000

# タスク: "@" を含むかチェック

# 方法1: re.search
pattern = re.compile(r'@')
start = time.perf_counter()
for _ in range(100000):
    pattern.search(text)
regex_time = time.perf_counter() - start

# 方法2: in 演算子
start = time.perf_counter()
for _ in range(100000):
    '@' in text
string_time = time.perf_counter() - start

print(f"re.search: {regex_time:.4f}秒")
print(f"in 演算子: {string_time:.4f}秒")
print(f"速度比: {regex_time / string_time:.1f}x")

# 一般的なガイドライン:
# - 単純な文字列検索 → in, find(), startswith(), endswith()
# - パターンマッチ → 正規表現
# - 文字列置換(固定) → str.replace()
# - 文字列置換(パターン) → re.sub()
# - 文字列分割(固定) → str.split()
# - 文字列分割(パターン) → re.split()
```

---

## 6. 入力検証のセキュリティ設計

### 6.1 多層防御アーキテクチャ

```
入力検証パイプライン:

┌─────────────────────────────────────────────────┐
│                入力検証パイプライン                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  Layer 1: 長さ制限                               │
│  ├── 最大文字数を制限 (例: 1000文字)              │
│  └── 空文字列チェック                             │
│                                                  │
│  Layer 2: 文字種制限                             │
│  ├── 許可する文字クラスのみ通す                    │
│  └── 制御文字、不可視文字を除去                    │
│                                                  │
│  Layer 3: 簡易パターンチェック                    │
│  ├── 文字列操作で先行チェック                      │
│  └── 明らかに無効な入力を早期排除                  │
│                                                  │
│  Layer 4: 正規表現による詳細検証                  │
│  ├── 安全なパターンを使用                         │
│  ├── タイムアウトを設定                           │
│  └── DFA エンジンを優先                           │
│                                                  │
│  Layer 5: ビジネスロジック検証                    │
│  └── アプリケーション固有のルール                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

```python
import re

class InputValidator:
    """多層防御による入力検証"""

    def __init__(self, max_length=1000, allowed_chars=None,
                 pattern=None, timeout_sec=1):
        self.max_length = max_length
        self.allowed_chars = allowed_chars
        self.pattern = re.compile(pattern) if pattern else None
        self.timeout_sec = timeout_sec

    def validate(self, value: str) -> tuple[bool, str]:
        """入力値を検証"""
        # Layer 1: 長さ制限
        if not value:
            return False, "空の入力"
        if len(value) > self.max_length:
            return False, f"入力が長すぎます ({len(value)} > {self.max_length})"

        # Layer 2: 文字種制限
        if self.allowed_chars:
            invalid = set(value) - set(self.allowed_chars)
            if invalid:
                return False, f"許可されていない文字: {invalid}"

        # Layer 3: 正規表現チェック
        if self.pattern:
            if not self.pattern.match(value):
                return False, "パターンに一致しません"

        return True, "OK"

# 使用例: メールアドレスバリデータ
email_validator = InputValidator(
    max_length=254,
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

test_emails = [
    "user@example.com",
    "a" * 300 + "@test.com",
    "invalid-email",
    "<script>alert('xss')</script>@evil.com",
]

for email in test_emails:
    valid, msg = email_validator.validate(email)
    print(f"  {email[:40]:40s} => {msg}")
```

### 6.2 ユーザー入力の正規表現としての安全な使用

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

# OK: 制限付きの正規表現サブセットのみ許可
def search_limited(user_pattern: str, text: str, max_length=100):
    """安全な正規表現サブセットのみ許可"""
    if len(user_pattern) > max_length:
        raise ValueError("パターンが長すぎます")

    # 危険な構文を検出
    dangerous = [
        r'\(.+[+*]\).+[+*]',  # ネストした量指定子
        r'\(\?\:.*\|.*\)[+*]', # 選択の繰り返し
    ]
    for d in dangerous:
        if re.search(d, user_pattern):
            raise ValueError("危険なパターンが検出されました")

    return re.search(user_pattern, text)
```

### 6.3 WAF(Web Application Firewall)でのパターン設計

```python
import re

# WAF ルールの安全な設計

# NG: 危険な WAF ルール
waf_rules_bad = {
    'sql_injection': r"('.+--)|(--.+')",  # ネストなし但し .+ は危険
    'xss': r"(<script.*>.*</script.*>)",  # .* が危険
}

# OK: 安全な WAF ルール
waf_rules_good = {
    'sql_injection': r"(?:'\s*(?:--|;|/\*)|(?:--|;|/\*)\s*')",
    'xss': r"<script[^>]*>[^<]*</script[^>]*>",
    'path_traversal': r"(?:\.\./|\.\.\\){2,}",
}

def check_waf_rules(input_text: str, rules: dict) -> list[str]:
    """WAF ルールに基づく入力チェック"""
    violations = []
    for rule_name, pattern in rules.items():
        compiled = re.compile(pattern, re.IGNORECASE)
        if compiled.search(input_text):
            violations.append(rule_name)
    return violations

# テスト
test_inputs = [
    "normal input",
    "'; DROP TABLE users; --",
    "<script>alert('xss')</script>",
    "../../../etc/passwd",
]

for inp in test_inputs:
    violations = check_waf_rules(inp, waf_rules_good)
    if violations:
        print(f"  [BLOCKED] '{inp[:40]}' => {violations}")
    else:
        print(f"  [PASS]    '{inp[:40]}'")
```

---

## 7. ベンチマーク手法

### 7.1 正規表現ベンチマークフレームワーク

```python
import re
import time
import statistics

class RegexBenchmark:
    """正規表現のパフォーマンスベンチマーク"""

    def __init__(self, pattern: str, description: str = ""):
        self.pattern = re.compile(pattern)
        self.pattern_str = pattern
        self.description = description

    def run(self, text: str, iterations: int = 1000,
            warmup: int = 100) -> dict:
        """ベンチマークを実行"""
        # ウォームアップ
        for _ in range(warmup):
            self.pattern.search(text)

        # 計測
        times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            self.pattern.search(text)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        return {
            'pattern': self.pattern_str,
            'description': self.description,
            'iterations': iterations,
            'min_ns': min(times),
            'max_ns': max(times),
            'mean_ns': statistics.mean(times),
            'median_ns': statistics.median(times),
            'stdev_ns': statistics.stdev(times) if len(times) > 1 else 0,
            'p95_ns': sorted(times)[int(len(times) * 0.95)],
            'p99_ns': sorted(times)[int(len(times) * 0.99)],
        }

    def print_result(self, result: dict):
        """結果を表示"""
        print(f"パターン: {result['pattern']}")
        print(f"説明: {result['description']}")
        print(f"  平均: {result['mean_ns']/1000:.2f}μs")
        print(f"  中央値: {result['median_ns']/1000:.2f}μs")
        print(f"  P95: {result['p95_ns']/1000:.2f}μs")
        print(f"  P99: {result['p99_ns']/1000:.2f}μs")
        print()

# 使用例
text = "The quick brown fox jumps over the lazy dog" * 100

benchmarks = [
    RegexBenchmark(r'\bfox\b', '単語境界マッチ'),
    RegexBenchmark(r'fox', 'シンプルマッチ'),
    RegexBenchmark(r'(?<=\s)fox(?=\s)', 'ルックアラウンドマッチ'),
    RegexBenchmark(r'.*fox.*', '.* マッチ'),
    RegexBenchmark(r'[^ ]*fox[^ ]*', '否定文字クラスマッチ'),
]

print("=== ベンチマーク結果 ===\n")
for bench in benchmarks:
    result = bench.run(text)
    bench.print_result(result)
```

### 7.2 スケーラビリティテスト

```python
import re
import time

def scalability_test(pattern_str: str, char: str = 'a',
                     lengths: list[int] = None):
    """入力長に対するスケーラビリティをテスト"""
    if lengths is None:
        lengths = [10, 50, 100, 500, 1000, 5000, 10000]

    pattern = re.compile(pattern_str)
    print(f"パターン: {pattern_str}")
    print(f"{'長さ':>10s} {'時間(μs)':>12s} {'比率':>8s}")
    print("-" * 35)

    prev_time = None
    for n in lengths:
        text = char * n
        start = time.perf_counter()
        pattern.search(text)
        elapsed = (time.perf_counter() - start) * 1_000_000  # μs

        ratio = f"{elapsed / prev_time:.1f}x" if prev_time else "-"
        prev_time = elapsed

        print(f"{n:>10d} {elapsed:>12.2f} {ratio:>8s}")

        # 安全のため1秒超えたら中断
        if elapsed > 1_000_000:
            print("  [中断: 1秒超過]")
            break

# テスト
print("=== 安全なパターン ===")
scalability_test(r'[a-z]+$')
print()

print("=== 線形パターン(否定文字クラス) ===")
scalability_test(r'[^b]*b')
print()

# 注意: 以下は長い入力で極めて遅くなる可能性がある
# print("=== 危険なパターン ===")
# scalability_test(r'(a+)+b', lengths=[10, 15, 20, 25])
```

---

## 8. 言語別のセキュリティ対策

### 8.1 Python

```python
import re

# Python での ReDoS 対策

# 対策1: regex モジュールの timeout パラメータ
import regex
try:
    result = regex.search(r'(a+)+b', 'a' * 30 + 'c', timeout=1)
except TimeoutError:
    print("タイムアウト")

# 対策2: google-re2 パッケージ
# pip install google-re2
# import re2
# result = re2.search(r'(a+)+b', 'a' * 30 + 'c')
# RE2 は自動的に安全なパターンに変換

# 対策3: パターンの静的解析
def audit_pattern(pattern_str: str) -> list[str]:
    """パターンの安全性を監査"""
    warnings = []

    # 量指定子のネスト
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern_str):
        warnings.append("量指定子のネスト")

    # .* のアンカーなし使用
    if '.*' in pattern_str and not pattern_str.startswith('^'):
        warnings.append("アンカーなしの .*")

    # 巨大な繰り返し
    large_repeat = re.search(r'\{(\d+)\}', pattern_str)
    if large_repeat and int(large_repeat.group(1)) > 1000:
        warnings.append(f"大きな繰り返し回数: {large_repeat.group(1)}")

    return warnings

# テスト
patterns = [
    r'(a+)+b',
    r'^[a-z]+$',
    r'.*error.*',
    r'a{10000}',
]

for p in patterns:
    warnings = audit_pattern(p)
    status = "WARN" if warnings else "OK"
    print(f"  [{status}] {p}: {warnings or 'clean'}")
```

### 8.2 JavaScript / Node.js

```javascript
// JavaScript / Node.js での ReDoS 対策

// 対策1: re2 パッケージを使用
// const RE2 = require('re2');
// const safe = new RE2('(a+)+b');

// 対策2: safe-regex パッケージで静的解析
// const safe = require('safe-regex');
// if (!safe(userPattern)) {
//     throw new Error('Unsafe regex pattern');
// }

// 対策3: vm モジュールでサンドボックス実行
// const vm = require('vm');
// const script = new vm.Script(`
//     const result = /${pattern}/.test(input);
// `);
// const context = vm.createContext({ input, pattern });
// script.runInContext(context, { timeout: 1000 });

// 対策4: Node.js v20+ の RegExp タイムアウト (experimental)
// --experimental-regexp-engine を使用すると
// RE2 風の安全なエンジンが利用可能になる場合がある

// ベストプラクティス:
// 1. ユーザー入力は必ずエスケープ
function escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 2. パターンの複雑さを制限
function isPatternSafe(pattern) {
    // ネストした量指定子を検出
    if (/\(.+[+*]\).+[+*]/.test(pattern)) return false;
    // パターン長を制限
    if (pattern.length > 200) return false;
    return true;
}

// 3. 入力長を制限
function safeMatch(pattern, text, maxLength) {
    maxLength = maxLength || 10000;
    if (text.length > maxLength) {
        throw new Error('Input too long');
    }
    return new RegExp(pattern).test(text);
}
```

### 8.3 Java

```java
import java.util.regex.*;
import java.util.concurrent.*;

public class SafeRegex {

    // 対策1: タイムアウト付き正規表現マッチ
    public static boolean safeMatch(String pattern, String text,
                                     long timeoutMs)
            throws TimeoutException {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Future<Boolean> future = executor.submit(() -> {
            Pattern p = Pattern.compile(pattern);
            return p.matcher(text).matches();
        });

        try {
            return future.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException e) {
            future.cancel(true);
            throw e;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            executor.shutdownNow();
        }
    }

    // 対策2: 独占的量指定子の使用
    public static void possessiveExample() {
        // 通常: バックトラックあり
        Pattern greedy = Pattern.compile("(a+)+b");

        // 独占的: バックトラック禁止
        Pattern possessive = Pattern.compile("(a++)+b");

        // アトミックグループ
        Pattern atomic = Pattern.compile("(?>a+)+b");
    }

    // 対策3: パターンの事前検証
    public static boolean isPatternSafe(String pattern) {
        // ネストした量指定子を検出
        if (pattern.matches(".*\\(.+[+*]\\).+[+*].*")) {
            return false;
        }
        // パターン長の制限
        if (pattern.length() > 200) {
            return false;
        }
        return true;
    }
}
```

### 8.4 Go (RE2)

```go
package main

import (
    "fmt"
    "regexp"
    "time"
)

func main() {
    // Go の regexp パッケージは RE2 ベース
    // → ReDoS に対して本質的に安全

    pattern := regexp.MustCompile(`(a+)+b`)
    input := string(make([]byte, 30)) + "c"
    for i := range input[:30] {
        input = input[:i] + "a" + input[i+1:]
    }

    start := time.Now()
    pattern.MatchString("a]" + input)
    elapsed := time.Since(start)
    fmt.Printf("RE2: %v\n", elapsed)
    // => 常に高速(O(n))

    // ただし RE2 の制約:
    // - 後方参照 (\1) 不可
    // - ルックアラウンド (?=) (?<=) 不可
    // - 独占的量指定子 (a++) 不可（不要）
    // - アトミックグループ (?>...) 不可（不要）

    // 制約がある機能が必要な場合:
    // github.com/dlclark/regexp2 パッケージを使用
    // ただし ReDoS リスクが発生する
}
```

### 8.5 Rust

```rust
use regex::Regex;
use std::time::Instant;

fn main() {
    // Rust の regex クレートは RE2 同様の DFA ベース
    // → ReDoS に対して本質的に安全

    let re = Regex::new(r"(a+)+b").unwrap();
    // 注: Rust regex は自動的に安全なパターンに変換する

    let input = "a".repeat(100) + "c";
    let start = Instant::now();
    re.is_match(&input);
    let elapsed = start.elapsed();
    println!("Rust regex: {:?}", elapsed);
    // => 常に高速(O(n))

    // ルックアラウンドが必要な場合:
    // fancy-regex クレートを使用
    // use fancy_regex::Regex;
    // ただし ReDoS リスクが発生する

    // regex クレートの制約:
    // - 後方参照不可
    // - ルックアラウンド不可
    // - コンパイル時間の制限あり
    //   (複雑すぎるパターンはコンパイルエラー)
}
```

---

## 9. ASCII 図解

### 9.1 バックトラック爆発の可視化

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

### 9.2 NFA vs DFA の動作比較

```
パターン: a+b
入力: "aaac"

=== NFA エンジン(Python re, JavaScript) ===

状態遷移 + バックトラック:

  位置0:
    a+ → 'a','a','a' (貪欲マッチ)
    b → 'c' ≠ 'b' → バックトラック
    a+ → 'a','a'
    b → 'a' ≠ 'b' → バックトラック
    a+ → 'a'
    b → 'a' ≠ 'b' → 失敗

  位置1: 同様の試行...
  位置2: 同様の試行...
  位置3: a+ マッチ不可 → 失敗

  合計ステップ数: O(n^2) (この場合)

=== DFA エンジン(RE2, Rust regex) ===

状態遷移テーブル:

  現在状態  入力  次の状態
  ────────  ────  ────────
  Start     'a'   State1 (a+)
  State1    'a'   State1 (a+ 繰り返し)
  State1    'b'   Accept (マッチ成功)
  State1    other Fail

  位置0: Start → 'a' → State1
  位置1:         'a' → State1
  位置2:         'a' → State1
  位置3:         'c' → Fail

  合計ステップ数: O(n) (常に線形)
```

### 9.3 安全なパターンと脆弱なパターンの対比

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
\d+\d+                \d{2,}               重複排除
(\w+\.)+              [\w.]+               フラット化
(.*\n)*               [^]*                 フラット化（言語依存）
```

### 9.4 パフォーマンス改善のフローチャート

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
    │   ├── アンカーなし?
    │   │   └── YES → ^, \b 等を追加
    │   │
    │   └── 不要なキャプチャあり?
    │       └── YES → (?:...) に変更
    │
    ├── コンパイルしているか?
    │   └── NO → re.compile() を使用
    │
    ├── 入力が信頼できないか?
    │   ├── YES → RE2/DFA エンジンを検討
    │   ├── YES → タイムアウトを設定
    │   └── YES → 入力長を制限
    │
    ├── 大量データの処理?
    │   ├── YES → 先行フィルタを追加
    │   ├── YES → バッチ処理を検討
    │   └── YES → 並列処理を検討
    │
    └── それでも遅い?
        └── 正規表現以外の方法を検討
            (文字列操作、パーサー、専用ライブラリ等)
```

---

## 10. 比較表

### 10.1 エンジン別パフォーマンス特性

| エンジン | 最悪計算量 | ReDoS耐性 | 機能 | 言語 |
|---------|-----------|----------|------|------|
| Python re | O(2^n) | なし | 豊富 | Python |
| Python regex | O(2^n) | timeout あり | 最も豊富 | Python |
| JavaScript V8 | O(2^n) | backtracks-limit | 豊富 | JavaScript |
| Java Pattern | O(2^n) | なし | 豊富(独占的量指定子) | Java |
| RE2 | O(n) | あり | 限定的 | Go, C++ |
| Rust regex | O(n) | あり | 限定的 | Rust |
| PCRE2 JIT | O(2^n) | match_limit | 最も豊富 | C |
| .NET | O(2^n) | MatchTimeout | 豊富 | C# |
| Oniguruma | O(2^n) | なし | 豊富 | Ruby |

### 10.2 最適化テクニック効果比較

| テクニック | 効果 | 適用場面 | 実装コスト |
|-----------|------|---------|-----------|
| re.compile() | 小～中 | 繰り返し使用時 | 低 |
| アンカー追加 | 中 | 位置が既知の場合 | 低 |
| 否定文字クラス | 中～大 | タグ抽出等 | 低 |
| 独占的量指定子 | 大 | バックトラック防止 | 中 |
| アトミックグループ | 大 | バックトラック防止 | 中 |
| DFAエンジン(RE2) | 最大 | 信頼できない入力 | 高(ライブラリ変更) |
| 先行文字列チェック | 中 | 出現頻度が低い場合 | 低 |
| パターン分割 | 中 | 複雑なパターン | 中 |
| 入力長制限 | 大 | 全ての場面 | 低 |
| タイムアウト設定 | 大 | 外部入力処理 | 中 |

### 10.3 セキュリティ対策の優先度

| 対策 | 優先度 | 効果 | コスト |
|------|--------|------|--------|
| 入力長の制限 | 最高 | 高 | 低 |
| パターンの静的解析 | 高 | 中 | 低 |
| タイムアウトの設定 | 高 | 高 | 中 |
| DFA エンジンの使用 | 中～高 | 最高 | 高 |
| ユーザー入力のエスケープ | 最高 | 高 | 低 |
| パターンの簡素化 | 中 | 中 | 中 |
| コードレビューでの監査 | 中 | 中 | 中 |
| CI/CDでの自動チェック | 高 | 高 | 中 |

---

## 11. 脆弱性の検出ツール

### 11.1 静的解析ツール

```bash
# recheck: 正規表現の脆弱性を検出
# npm install -g recheck
# recheck "(a+)+b"

# redos-checker (Python)
# pip install redos-checker

# safe-regex (JavaScript)
# npm install safe-regex

# semgrep: コード全体の静的解析
# semgrep --config "p/regex-dos" .
```

```javascript
// safe-regex の使用例
// const safe = require('safe-regex');
// console.log(safe('(a+)+b'));      // => false (脆弱)
// console.log(safe('[a-z]+'));       // => true  (安全)
// console.log(safe('(a|b|c)+'));     // => true  (安全)
```

### 11.2 Python での簡易チェッカー

```python
import re

def is_potentially_vulnerable(pattern: str) -> tuple[bool, list[str]]:
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

    # 巨大な繰り返し回数
    large_repeat = re.search(r'\{(\d+)', pattern)
    if large_repeat and int(large_repeat.group(1)) > 1000:
        warnings.append(f"大きな繰り返し回数: {large_repeat.group(1)}")

    # バックリファレンスの使用
    if re.search(r'\\[1-9]', pattern):
        warnings.append("後方参照が使用されています（パフォーマンスに注意）")

    return (len(warnings) > 0, warnings)

# テスト
patterns = [
    r'(a+)+b',
    r'(\w|\d)+',
    r'[a-z]+',
    r'.*error.*',
    r'^[a-z]+$',
    r'(\w+\.)+\w+',
    r'(.)\1{100}',
    r'a{10000}',
]

for p in patterns:
    vulnerable, msgs = is_potentially_vulnerable(p)
    status = "脆弱" if vulnerable else "安全"
    print(f"  {status}: {p}")
    for msg in msgs:
        print(f"    - {msg}")
```

### 11.3 CI/CD パイプラインへの組み込み

```python
#!/usr/bin/env python3
"""
regex_audit.py -- CI/CDパイプラインで正規表現を監査するスクリプト

使用方法:
    python regex_audit.py path/to/source/

戻り値:
    0: 問題なし
    1: 警告あり
    2: 危険なパターンあり
"""

import re
import sys
import os

def find_regex_patterns(filepath: str) -> list[tuple[int, str]]:
    """ソースコードから正規表現パターンを抽出"""
    patterns = []

    # Python: re.compile(), re.search(), re.match() 等
    regex_call = re.compile(
        r're\.(?:compile|search|match|findall|sub|split)\s*\(\s*'
        r'(?:r)?["\'](.+?)["\']'
    )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                for m in regex_call.finditer(line):
                    patterns.append((lineno, m.group(1)))
    except (UnicodeDecodeError, PermissionError):
        pass

    return patterns

def audit_pattern(pattern: str) -> list[str]:
    """パターンの安全性を監査"""
    warnings = []

    # 量指定子のネスト
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern):
        warnings.append("CRITICAL: 量指定子のネスト")

    # .* のアンカーなし使用
    if '.*' in pattern and not pattern.startswith('^'):
        warnings.append("WARNING: アンカーなしの .*")

    # 巨大な繰り返し
    repeat = re.search(r'\{(\d+)', pattern)
    if repeat and int(repeat.group(1)) > 100:
        warnings.append(f"WARNING: 大きな繰り返し回数: {repeat.group(1)}")

    return warnings

def main():
    if len(sys.argv) < 2:
        print("Usage: python regex_audit.py <path>")
        sys.exit(1)

    path = sys.argv[1]
    exit_code = 0

    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                patterns = find_regex_patterns(filepath)
                for lineno, pattern in patterns:
                    warnings = audit_pattern(pattern)
                    for w in warnings:
                        print(f"{filepath}:{lineno}: {w}: {pattern}")
                        if 'CRITICAL' in w:
                            exit_code = max(exit_code, 2)
                        elif 'WARNING' in w:
                            exit_code = max(exit_code, 1)

    if exit_code == 0:
        print("All regex patterns passed audit.")
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
```

---

## 12. 実務でのベストプラクティス

### 12.1 正規表現レビューチェックリスト

```
正規表現コードレビュー チェックリスト:

□ パターンの目的と意図がコメントに記載されているか
□ テストケースが十分にあるか（正常系、異常系、境界値）
□ 量指定子のネストがないか
□ 選択肢の重複がないか
□ .* の使用が適切か（アンカーまたは否定文字クラスの使用を検討）
□ ユーザー入力がパターンに含まれていないか
□ 入力長の制限があるか
□ タイムアウトが設定されているか（外部入力の場合）
□ re.compile() でプリコンパイルされているか
□ 不要なキャプチャグループがないか
□ パターンが過度に複雑でないか（分割を検討）
□ 正規表現以外の手段で代替できないか
```

### 12.2 正規表現のドキュメント化

```python
import re

# 良いドキュメント化の例
EMAIL_PATTERN = re.compile(
    r'''
    ^                       # 文字列の先頭
    [a-zA-Z0-9._%+-]+      # ローカルパート: 英数字と特殊文字
    @                       # @ 記号
    [a-zA-Z0-9.-]+          # ドメイン: 英数字、ドット、ハイフン
    \.                      # ドットセパレータ
    [a-zA-Z]{2,}            # TLD: 2文字以上のアルファベット
    $                       # 文字列の末尾
    ''',
    re.VERBOSE              # 空白とコメントを許可
)

# VERBOSE フラグなしの場合はコメントで補足
# RFC 5322 簡易版。国際化ドメインは未対応。
# ReDoS 安全性: OK（量指定子のネストなし、否定文字クラス使用）
# 最大入力長: 254文字に制限すること
```

---

## 13. アンチパターン

### 13.1 アンチパターン: ユーザー入力を正規表現に直接使う

```python
import re

# NG: ユーザー入力をそのまま正規表現として使用
def search_bad(user_input: str, text: str):
    return re.search(user_input, text)  # ReDoS攻撃が可能!

# OK: ユーザー入力はエスケープする
def search_safe(user_input: str, text: str):
    escaped = re.escape(user_input)  # メタ文字を全てエスケープ
    return re.search(escaped, text)
```

### 13.2 アンチパターン: 最適化なしの大量データ処理

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

### 13.3 アンチパターン: 正規表現の過剰使用

```python
import re

# NG: 単純な操作に正規表現を使う
def check_email_bad(email):
    return bool(re.match(r'.+@.+', email))

# OK: 文字列操作で十分
def check_email_good(email):
    return '@' in email and '.' in email.split('@')[1]

# NG: 固定文字列の置換に正規表現
text = "Hello World"
result_bad = re.sub(r'World', 'Python', text)

# OK: str.replace() を使う
result_good = text.replace('World', 'Python')

# NG: 固定文字列での分割に正規表現
data = "a,b,c,d"
parts_bad = re.split(r',', data)

# OK: str.split() を使う
parts_good = data.split(',')
```

---

## 14. FAQ

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

### Q4: 正規表現を使わずに済む場面はどんなときか？

**A**: 以下の場合は正規表現より効率的な代替手段がある:

```python
# 固定文字列の検索 → in 演算子
if 'error' in log_line:  # re.search(r'error', log_line) より速い

# 先頭/末尾の一致 → startswith() / endswith()
if filename.endswith('.py'):  # re.match(r'.*\.py$', filename) より速い

# 単純な分割 → str.split()
fields = line.split(',')  # re.split(r',', line) より速い

# 単純な置換 → str.replace()
result = text.replace('old', 'new')  # re.sub(r'old', 'new', text) より速い

# 構造化データのパース → 専用パーサー
import json  # JSON には json モジュール
import csv   # CSV には csv モジュール
# HTML → Beautiful Soup / lxml
# XML → ElementTree
# URL → urllib.parse
```

### Q5: 怠惰量指定子と否定文字クラスのどちらを使うべきか？

**A**: 否定文字クラスが使える場合は常にそちらを推奨:

```python
# 怠惰量指定子: .*? (バックトラックが発生する)
# 否定文字クラス: [^X]* (バックトラックが発生しない)

# 例: HTML タグの属性値を抽出
# 遅い: <div class=".*?">
# 速い: <div class="[^"]*">

# 例: 引用符内の文字列を抽出
# 遅い: ".*?"
# 速い: "[^"]*"

# 例: コメントの抽出
# 遅い: /\*.*?\*/  (DOTALL 必要)
# 速い: /\*[^*]*\*+(?:[^/*][^*]*\*+)*/
# ↑ 複雑になる場合は可読性とのトレードオフ
```

### Q6: アトミックグループと独占的量指定子の違いは？

**A**: 同じ機能を異なる構文で提供する。どちらもバックトラックを禁止:

```
アトミックグループ: (?>pattern)
独占的量指定子:    pattern++, pattern*+, pattern?+

(?>a+)  は  a++ と同等
(?>a*)  は  a*+ と同等
(?>a?)  は  a?+ と同等

ただし、アトミックグループはより汎用的:
(?>abc|ab) -- 選択肢全体にアトミックを適用
↑ 独占的量指定子では表現不可

サポート状況:
  独占的量指定子: Java, PCRE, Python regex
  アトミックグループ: Java, PCRE, Python regex, Perl, .NET
  どちらも非対応: Python re, JavaScript, Go, Rust
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| ReDoS | 正規表現によるサービス拒否攻撃 |
| 原因 | バックトラック爆発(指数的計算量) |
| 脆弱パターン | `(a+)+`, `(a\|a)+`, `.*` の無制限使用 |
| NFA | バックトラック型、O(2^n)の可能性 |
| DFA | 線形時間保証 O(n)、機能制限あり |
| 防御策1 | 否定文字クラス `[^X]*` で範囲を限定 |
| 防御策2 | 量指定子のネストを排除 |
| 防御策3 | DFA エンジン(RE2等)を使用 |
| 防御策4 | タイムアウト機構を設定 |
| 防御策5 | 入力長を制限 |
| 最適化 | `re.compile()`、アンカー、先行フィルタ |
| 独占的量指定子 | `a++` -- バックトラック禁止 |
| アトミックグループ | `(?>...)` -- バックトラック禁止 |
| 鉄則 | 信頼できない入力には DFA、パターンは静的解析で監査 |

## 次に読むべきガイド

- [../02-practical/00-language-specific.md](../02-practical/00-language-specific.md) -- 言語別正規表現(エンジンの違い)
- [../02-practical/03-regex-alternatives.md](../02-practical/03-regex-alternatives.md) -- 正規表現の代替技術

## 参考文献

1. **Russ Cox** "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html, 2007 -- ReDoS の理論的背景と RE2 の設計思想
2. **OWASP** "Regular expression Denial of Service - ReDoS" https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS -- セキュリティ観点からの ReDoS 解説
3. **James Davis et al.** "The Impact of Regular Expression Denial of Service (ReDoS) in Practice" FSE 2018 -- ReDoS の実世界における影響の学術的調査
4. **Google RE2** https://github.com/google/re2 -- 線形時間保証の正規表現エンジン
5. **recheck** https://makenowjust-labs.github.io/recheck/ -- 正規表現の ReDoS 脆弱性検出ツール
6. **Cloudflare Outage Post-mortem** https://blog.cloudflare.com/details-of-the-cloudflare-outage-on-july-2-2019/ -- 実際の ReDoS インシデント事例
7. **PCRE2 Documentation** https://www.pcre.org/current/doc/html/ -- PCRE2 のバックトラック制限と最適化
8. **Rust regex crate** https://docs.rs/regex/ -- Rust の O(n) 保証正規表現エンジン
