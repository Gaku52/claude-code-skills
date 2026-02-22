# 正規表現概要

> 正規表現(Regular Expression)の歴史的背景、主要な用途、そしてNFA/DFAエンジンの内部動作原理を体系的に解説する。

## この章で学ぶこと

1. **正規表現の数学的起源と歴史的発展** -- Kleeneの正則集合からPCREまでの系譜
2. **正規表現エンジンの二大方式(NFA/DFA)** -- それぞれの動作原理・性能特性・選択基準
3. **正規表現の適用領域と限界** -- テキスト検索からコンパイラまで、使うべき場面と避けるべき場面
4. **主要エンジンの実装比較** -- 各プログラミング言語のエンジン特性と選定指針
5. **正規表現のデバッグとテスト戦略** -- 効率的なパターン開発手法

---

## 1. 正規表現とは何か

正規表現は **パターンマッチング** のための形式言語である。文字列の集合を有限の記法で表現し、検索・置換・抽出・検証に利用する。

```
パターン: \d{3}-\d{4}
対象文字列: "郵便番号は 100-0001 です"
マッチ結果: "100-0001"
```

### 1.1 基本的な動作モデル

```
入力文字列 ──→ [正規表現エンジン] ──→ マッチ結果
                    ↑
               パターン(正規表現)
```

正規表現エンジンは、与えられたパターンを内部的にオートマトン(有限状態機械)に変換し、入力文字列を1文字ずつ処理してマッチングを行う。

### 1.2 正規表現の構成要素

正規表現パターンは以下の基本要素から構成される:

```
正規表現の構成要素:

1. リテラル文字    -- 'a', 'b', '1' 等、そのまま文字にマッチ
2. メタ文字        -- '.', '*', '+', '?', '|' 等、特殊な意味を持つ
3. 文字クラス      -- [abc], [a-z], \d, \w 等、文字集合を表す
4. 量指定子        -- {n}, {n,m}, *, +, ? 等、繰り返しを指定
5. アンカー        -- ^, $, \b 等、位置を指定
6. グループ化      -- (), (?:), (?=) 等、部分パターンをまとめる
7. エスケープ      -- \., \\, \n 等、メタ文字の無効化や特殊文字
```

### 1.3 正規表現処理の全体フロー

```
                 正規表現パターン
                       │
                       ▼
               ┌──────────────┐
               │   字句解析    │  パターン文字列をトークン列に分解
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │   構文解析    │  トークン列を抽象構文木(AST)に変換
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │ オートマトン  │  AST から NFA/DFA を構築
               │    構築      │
               └──────┬───────┘
                       │
                       ▼
               ┌──────────────┐
               │  マッチング   │  入力文字列に対してオートマトンを実行
               │    実行      │
               └──────┬───────┘
                       │
                       ▼
                 マッチ結果
```

### 1.4 正規表現の表記法の種類

```
主要な表記体系:

1. POSIX BRE (Basic Regular Expression)
   - メタ文字に \ が必要: \(, \), \{, \}, \+, \?
   - 例: grep 'a\(b\|c\)d' file.txt

2. POSIX ERE (Extended Regular Expression)
   - メタ文字をそのまま使用: (, ), {, }, +, ?
   - 例: grep -E 'a(b|c)d' file.txt

3. PCRE (Perl Compatible Regular Expressions)
   - ERE を拡張: 先読み/後読み, 非貪欲量指定子, 名前付きキャプチャ
   - 例: grep -P '(?<=prefix)\w+' file.txt

4. ECMAScript (JavaScript)
   - PCRE のサブセット + 独自拡張(u フラグ, s フラグ等)
   - 例: /pattern/gimsuvy

5. RE2 構文
   - PCRE からバックトラック必須機能を除外
   - 例: 後方参照なし, 先読み/後読みなし
```

各表記体系は互換性がないことが多いため、使用するツールやプログラミング言語の正規表現方言を把握することが重要である。

### 1.5 正規表現パターンの読み方

複雑な正規表現を読む際の手順を示す:

```python
# 例: メールアドレスの簡易パターン
pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

# 分解して読む:
# ^                    -- 文字列の先頭
# [a-zA-Z0-9._%+-]+   -- ローカルパート(1文字以上の英数字と記号)
# @                    -- リテラル '@'
# [a-zA-Z0-9.-]+      -- ドメイン名(1文字以上の英数字とハイフン、ドット)
# \.                   -- リテラル '.'
# [a-zA-Z]{2,}        -- TLD(2文字以上のアルファベット)
# $                    -- 文字列の末尾
```

```python
# 例: 日本の電話番号パターン
pattern = r'^0\d{1,4}-\d{1,4}-\d{4}$'

# 分解:
# ^           -- 先頭
# 0           -- リテラル '0' (市外局番の先頭)
# \d{1,4}     -- 1〜4桁の数字
# -           -- リテラル '-'
# \d{1,4}     -- 1〜4桁の数字(市内局番)
# -           -- リテラル '-'
# \d{4}       -- 4桁の数字(加入者番号)
# $           -- 末尾
```

---

## 2. 歴史的発展

### 2.1 年表

```
1943  McCulloch & Pitts  ─ 神経回路網の数学モデル
  │
1956  Stephen Kleene     ─ 正則集合(Regular Sets)の理論化
  │
1959  Michael Rabin &    ─ 非決定性有限オートマトン(NFA)の形式化
      Dana Scott           チューリング賞受賞(1976年)
  │
1968  Ken Thompson       ─ QED/edエディタに正規表現を実装
  │                        IBM 7094 上でNFAを直接シミュレート
  │
1973  Thompson & Ritchie ─ grep の誕生(Unix V4)
  │                        "Global Regular Expression Print"
  │
1975  Alfred Aho         ─ egrep の開発(DFA方式)
  │                        『コンパイラ—原理・技法・ツール』の著者
  │
1979  AT&T Unix V7       ─ awk の登場(Aho, Weinberger, Kernighan)
  │                        正規表現をプログラミング言語に統合
  │
1986  Henry Spencer      ─ 最初の自由な正規表現ライブラリ
  │                        多くのUNIXツールの基盤に
  │
1986  POSIX 標準化       ─ BRE/ERE を IEEE Std 1003.2 として標準化
  │
1987  Larry Wall         ─ Perl 1.0 に高機能正規表現を搭載
  │                        後方参照、先読み等を導入
  │
1994  Perl 5.0           ─ 正規表現の大幅拡張
  │                        非貪欲量指定子、先読み/後読み
  │
1997  Philip Hazel       ─ PCRE (Perl Compatible Regular Expressions)
  │                        Perlの正規表現を独立ライブラリ化
  │
2002  .NET Framework     ─ バランシンググループを導入
  │                        ネスト構造の限定的なマッチが可能に
  │
2006  Russ Cox           ─ RE2 (線形時間保証エンジン)
  │                        Google で開発、ReDoS を原理的に排除
  │
2012  Rust regex crate   ─ RE2 の思想を継承した Rust 実装
  │                        安全性と性能を両立
  │
2017  ECMAScript 2018    ─ 名前付きキャプチャ、後読みを標準化
  │                        s フラグ(dotAll)の追加
  │
2022  PCRE2 10.40+       ─ JIT コンパイルの改良
  │                        パフォーマンスの大幅向上
  │
2024  ECMAScript 2024    ─ v フラグ(Unicode Sets)の追加
                           文字クラスの集合演算をサポート
```

### 2.2 主要なマイルストーン

| 年代 | 人物/プロジェクト | 貢献 |
|------|-------------------|------|
| 1956 | Stephen Kleene | 「正則表現(regular expression)」の概念を数学的に定式化 |
| 1959 | Rabin & Scott | NFA/DFA の等価性を証明(チューリング賞) |
| 1968 | Ken Thompson | エディタ QED に初の実用的正規表現エンジンを実装 |
| 1973 | grep (Unix) | `g/re/p` -- Global Regular Expression Print |
| 1975 | egrep (Unix) | DFA ベースの高速正規表現マッチング |
| 1979 | awk | テキスト処理言語に正規表現を統合 |
| 1986 | POSIX | BRE/ERE(基本/拡張正規表現)を標準化 |
| 1987 | Perl | 後方参照・先読み等を追加、事実上の標準に |
| 1994 | Perl 5 | 非貪欲量指定子、コードブロック内正規表現 |
| 1997 | PCRE | Perl互換エンジンを独立ライブラリとして提供 |
| 2002 | .NET | バランシンググループでネスト構造に対応 |
| 2006 | RE2 (Google) | DFAベースで ReDoS を原理的に排除 |
| 2017 | ES2018 | 後読み・名前付きキャプチャを JavaScript に追加 |

### 2.3 正規表現理論の数学的基盤

正規表現の理論は計算機科学の基礎であるオートマトン理論と密接に結びついている:

```
Kleene の定理 (1956):

「正則言語」「正規表現で記述可能な言語」「有限オートマトンが受理する言語」
の三者は等価である。

すなわち:
  正規表現 ⟺ NFA ⟺ DFA

変換の方向:
  正規表現 → NFA  : Thompson構成法
  NFA → DFA       : 部分集合構成法
  DFA → 正規表現  : 状態除去法
  DFA → 最小DFA   : Hopcroft のアルゴリズム
```

```python
# Thompson構成法の概念的な実装例
# 正規表現の基本操作に対応するNFA構築

class NFAState:
    """NFA の状態"""
    def __init__(self):
        self.transitions = {}  # 文字 -> [次の状態]
        self.epsilon = []      # ε遷移先
        self.is_accept = False

class NFAFragment:
    """NFA の断片（構築中の部分NFA）"""
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept

def literal(char):
    """リテラル文字 'a' に対する NFA"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.transitions[char] = [accept]
    return NFAFragment(start, accept)

def concatenation(frag1, frag2):
    """連結 ab に対する NFA"""
    frag1.accept.is_accept = False
    frag1.accept.epsilon.append(frag2.start)
    return NFAFragment(frag1.start, frag2.accept)

def alternation(frag1, frag2):
    """選択 a|b に対する NFA"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.epsilon.extend([frag1.start, frag2.start])
    frag1.accept.is_accept = False
    frag2.accept.is_accept = False
    frag1.accept.epsilon.append(accept)
    frag2.accept.epsilon.append(accept)
    return NFAFragment(start, accept)

def kleene_star(frag):
    """Kleene閉包 a* に対する NFA"""
    start = NFAState()
    accept = NFAState()
    accept.is_accept = True
    start.epsilon.extend([frag.start, accept])
    frag.accept.is_accept = False
    frag.accept.epsilon.extend([frag.start, accept])
    return NFAFragment(start, accept)
```

### 2.4 POSIX標準と方言の分岐

```
POSIX 正規表現の二つの標準:

┌─────────────────────────────────────────────────────────────┐
│  BRE (Basic Regular Expression)                             │
│                                                             │
│  特徴:                                                      │
│  - メタ文字としてのグループ化: \( と \)                       │
│  - メタ文字としての選択: 非サポート(一部実装では \|)          │
│  - メタ文字としての量指定: \{ と \}                           │
│  - +, ? はリテラル文字(一部実装では \+, \?)                  │
│                                                             │
│  使用ツール: grep(デフォルト), sed(デフォルト)               │
│                                                             │
│  例: grep 'a\{2,3\}' file.txt                              │
│      → 'aa' または 'aaa' にマッチ                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  ERE (Extended Regular Expression)                          │
│                                                             │
│  特徴:                                                      │
│  - グループ化: ( と )                                        │
│  - 選択: |                                                   │
│  - 量指定: { と }                                            │
│  - +, ? がメタ文字                                           │
│                                                             │
│  使用ツール: grep -E (egrep), sed -E, awk                   │
│                                                             │
│  例: grep -E 'a{2,3}' file.txt                             │
│      → 'aa' または 'aaa' にマッチ                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 正規表現エンジンの種類

### 3.1 NFA vs DFA -- 二大方式

```
┌──────────────────────────────────────────────────────┐
│                正規表現エンジン                        │
├─────────────────────┬────────────────────────────────┤
│   NFA (非決定性)     │    DFA (決定性)                │
│                     │                                │
│ ・バックトラック方式  │ ・状態遷移テーブル方式          │
│ ・パターン駆動       │ ・テキスト駆動                  │
│ ・後方参照をサポート  │ ・後方参照は非サポート           │
│ ・最悪 O(2^n)        │ ・常に O(n)                    │
│                     │                                │
│ 例: Perl, Python,   │ 例: awk, grep(一部),           │
│     Java, .NET,     │     RE2, Rust regex            │
│     JavaScript      │                                │
└─────────────────────┴────────────────────────────────┘
```

### 3.2 NFA の動作例

パターン `a(b|c)d` に対して文字列 `"acd"` をマッチングする場合:

```python
# NFA (Nondeterministic Finite Automaton) の動作シミュレーション
import re

pattern = r'a(b|c)d'
text = "acd"

# 内部的な動作:
# 1. 状態 S0: 'a' を読む → マッチ → S1 へ遷移
# 2. 状態 S1: 'c' を読む → 'b' を試行 → 失敗
#                        → バックトラックして 'c' を試行 → マッチ → S2 へ
# 3. 状態 S2: 'd' を読む → マッチ → 受理状態

result = re.match(pattern, text)
print(result.group())   # => "acd"
print(result.group(1))  # => "c"
```

### 3.3 DFA の動作例

```python
# DFA (Deterministic Finite Automaton) は事前に全状態を展開する
# パターン: a(b|c)d

# 状態遷移テーブル:
# 現在状態 | 入力 'a' | 入力 'b' | 入力 'c' | 入力 'd'
# ---------|----------|----------|----------|--------
# S0       | S1       | -        | -        | -
# S1       | -        | S2       | S2       | -
# S2       | -        | -        | -        | S3(受理)

# DFA ではバックトラックが発生しない
# 各文字につき状態遷移は1回のみ → O(n)

# Rust の regex クレートは DFA ベース
# RE2 も DFA ベース
```

### 3.4 比較表: NFA vs DFA

| 特性 | NFA | DFA |
|------|-----|-----|
| 時間計算量(最悪) | O(2^n) -- 指数的 | O(n) -- 線形 |
| 時間計算量(平均) | O(n) -- 実用的には高速 | O(n) -- 常に線形 |
| 空間計算量 | O(m) パターンサイズ | O(2^m) 最悪(状態爆発) |
| 後方参照 | サポート | 非サポート |
| 先読み/後読み | サポート | 限定的/非サポート |
| 怠惰量指定子 | サポート | N/A (最左最長マッチ) |
| ReDoS脆弱性 | あり | なし |
| 実装の複雑さ | 比較的単純 | 状態テーブル構築が複雑 |
| コンパイル時間 | 短い | 長い(状態展開) |
| 最初のマッチ | 高速(左から右へ) | パターンにより変動 |
| 代表的な実装 | Perl, Python, Java, JS | RE2, Rust regex, awk |

### 3.5 NFA のバックトラック詳細

```
パターン: a.*b
テキスト: "axyzb123"

ステップ 1: 'a' → マッチ
ステップ 2: '.*' → 貪欲に全文字を消費 "xyzb123"
ステップ 3: 'b' → マッチ失敗(文字列末尾)
ステップ 4: バックトラック → '.*' が "xyzb12" まで戻す
ステップ 5: 'b' → '3' とマッチ失敗
ステップ 6: バックトラック → '.*' が "xyzb1" まで戻す
ステップ 7: 'b' → '2' とマッチ失敗
ステップ 8: バックトラック → '.*' が "xyzb" まで戻す
ステップ 9: 'b' → '1' とマッチ失敗
ステップ 10: バックトラック → '.*' が "xyz" まで戻す
ステップ 11: 'b' → 'b' とマッチ成功！

結果: "axyzb"
バックトラック回数: 4回

※ 非貪欲版 a.*?b なら:
ステップ 1: 'a' → マッチ
ステップ 2: '.*?' → 最小で0文字消費
ステップ 3: 'b' → 'x' とマッチ失敗
ステップ 4: '.*?' → 1文字消費 "x"
ステップ 5: 'b' → 'y' とマッチ失敗
ステップ 6: '.*?' → 2文字消費 "xy"
ステップ 7: 'b' → 'z' とマッチ失敗
ステップ 8: '.*?' → 3文字消費 "xyz"
ステップ 9: 'b' → 'b' とマッチ成功！

結果: "axyzb" (同じ結果だが到達経路が異なる)
```

### 3.6 ハイブリッドアプローチ

```
┌─────────────────────────────────────────┐
│         ハイブリッドエンジン              │
│                                         │
│  パターン解析                             │
│      │                                   │
│      ├── 後方参照なし → DFA で実行        │
│      │                                   │
│      └── 後方参照あり → NFA にフォールバック│
│                                         │
│  例: .NET, Rust の fancy-regex           │
└─────────────────────────────────────────┘
```

### 3.7 各言語の正規表現エンジン一覧

| 言語/ツール | エンジン種別 | ライブラリ | 特記事項 |
|------------|-------------|-----------|---------|
| Python | NFA | `re` (C実装) | `regex` モジュールで拡張可能 |
| JavaScript | NFA | V8 Irregexp | JIT最適化あり |
| Java | NFA | `java.util.regex` | 原子グループ非対応(Java 9+で一部対応) |
| C# (.NET) | NFA | `System.Text.RegularExpressions` | バランシンググループ対応 |
| Perl | NFA | 組み込み | 最も機能豊富なNFA実装 |
| Ruby | NFA | Onigmo (鬼雲) | Unicode対応が充実 |
| Go | DFA | `regexp` (RE2ベース) | 後方参照非対応 |
| Rust | DFA | `regex` クレート | 線形時間保証 |
| PHP | NFA | PCRE2 | `preg_*` 関数群 |
| C/C++ | 両方 | PCRE2, RE2, std::regex | 選択可能 |
| awk | DFA | 組み込み | ERE準拠 |
| grep | 両方 | GNU grep | `-G` BRE, `-E` ERE, `-P` PCRE |
| sed | NFA | 組み込み | BRE(デフォルト), ERE(`-E`) |

### 3.8 エンジンの選択フローチャート

```
パターンに後方参照が含まれるか？
    │
    ├── はい → NFA エンジンを使用
    │          │
    │          ├── 信頼できない入力か？
    │          │   │
    │          │   ├── はい → タイムアウト設定必須
    │          │   │          (.NET: MatchTimeout,
    │          │   │           Java: interrupt,
    │          │   │           Python: signal.alarm)
    │          │   │
    │          │   └── いいえ → そのまま NFA を使用
    │          │
    │          └── パフォーマンスが問題か？
    │              │
    │              ├── はい → パターンの書き換えを検討
    │              │          (原子グループ、独占的量指定子)
    │              │
    │              └── いいえ → そのまま使用
    │
    └── いいえ → DFA エンジンが利用可能か？
                │
                ├── はい → DFA を使用 (RE2, Rust regex, Go regexp)
                │          線形時間保証で安全
                │
                └── いいえ → NFA で問題なし
                             (ReDoS パターンを避ける)
```

---

## 4. 正規表現の主要な用途

### 4.1 用途別コード例

```bash
# 1. テキスト検索 (grep)
grep -E 'ERROR|WARN' /var/log/syslog

# 2. テキスト置換 (sed)
sed 's/2025/2026/g' document.txt

# 3. データ抽出 (Python)
python3 -c "
import re
log = '2026-02-11 10:30:45 [ERROR] Connection timeout (192.168.1.1)'
m = re.search(r'(\d{4}-\d{2}-\d{2}) .* \[(\w+)\] (.+)', log)
print(f'日付: {m.group(1)}, レベル: {m.group(2)}, メッセージ: {m.group(3)}')
"

# 4. 入力検証 (JavaScript)
node -e "
const emailPattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
console.log(emailPattern.test('user@example.com'));  // true
console.log(emailPattern.test('invalid@'));           // false
"

# 5. 構文ハイライト -- エディタがキーワードを着色する仕組み
# パターン例: \b(if|else|for|while|return)\b → キーワードとして着色

# 6. ファイル名マッチング (find + grep)
find /var/log -name "*.log" -exec grep -l 'CRITICAL' {} \;

# 7. CSV データの変換 (awk)
awk -F',' '/^2026/ {print $1, $3}' data.csv

# 8. コード内の TODO コメント抽出
grep -rn 'TODO\|FIXME\|HACK\|XXX' --include='*.py' src/
```

### 4.2 ログ解析の実践例

```python
import re
from collections import Counter, defaultdict
from datetime import datetime

# Apache アクセスログの解析
log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+)'       # IPアドレス
    r' - - '
    r'\[(?P<timestamp>[^\]]+)\]'           # タイムスタンプ
    r' "(?P<method>\w+)'                   # HTTPメソッド
    r' (?P<path>[^\s]+)'                   # リクエストパス
    r' HTTP/[\d.]+"'                       # HTTPバージョン
    r' (?P<status>\d{3})'                  # ステータスコード
    r' (?P<size>\d+|-)'                    # レスポンスサイズ
    r'(?: "(?P<referer>[^"]*)")?'          # リファラー(オプション)
    r'(?: "(?P<useragent>[^"]*)")?'        # ユーザーエージェント(オプション)
)

def analyze_access_log(log_file: str):
    """アクセスログを解析して統計情報を出力"""
    status_counter = Counter()
    path_counter = Counter()
    ip_counter = Counter()
    hourly_access = defaultdict(int)
    error_logs = []

    with open(log_file) as f:
        for line in f:
            m = log_pattern.match(line)
            if not m:
                continue

            data = m.groupdict()
            status = int(data['status'])
            path = data['path']
            ip = data['ip']

            status_counter[status] += 1
            path_counter[path] += 1
            ip_counter[ip] += 1

            # 時間帯別集計
            try:
                dt = datetime.strptime(
                    data['timestamp'],
                    '%d/%b/%Y:%H:%M:%S %z'
                )
                hourly_access[dt.hour] += 1
            except ValueError:
                pass

            # エラーログの収集
            if status >= 400:
                error_logs.append({
                    'ip': ip,
                    'path': path,
                    'status': status,
                    'timestamp': data['timestamp']
                })

    return {
        'total_requests': sum(status_counter.values()),
        'status_distribution': dict(status_counter),
        'top_paths': path_counter.most_common(10),
        'top_ips': ip_counter.most_common(10),
        'hourly_distribution': dict(hourly_access),
        'error_count': len(error_logs),
        'recent_errors': error_logs[-10:]
    }
```

### 4.3 データクレンジングの実践例

```python
import re

def clean_text(text: str) -> str:
    """テキストデータのクレンジング"""
    # 連続する空白を1つに統一
    text = re.sub(r'\s+', ' ', text)

    # 全角英数字を半角に変換
    text = re.sub(r'[Ａ-Ｚａ-ｚ０-９]',
                  lambda m: chr(ord(m.group()) - 0xFEE0), text)

    # HTMLタグの除去
    text = re.sub(r'<[^>]+>', '', text)

    # 制御文字の除去（改行・タブは保持）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 先頭・末尾の空白を除去
    text = text.strip()

    return text

def normalize_phone_number(phone: str) -> str:
    """電話番号の正規化"""
    # 数字以外を除去
    digits = re.sub(r'\D', '', phone)

    # 国際電話番号の処理
    if digits.startswith('81') and len(digits) >= 11:
        digits = '0' + digits[2:]

    # フォーマット(固定電話)
    if len(digits) == 10:
        m = re.match(r'(\d{2,4})(\d{2,4})(\d{4})', digits)
        if m:
            return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'

    # フォーマット(携帯電話)
    if len(digits) == 11:
        m = re.match(r'(\d{3})(\d{4})(\d{4})', digits)
        if m:
            return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'

    return phone  # 変換不能な場合はそのまま返す

def extract_urls(text: str) -> list:
    """テキストからURLを抽出"""
    url_pattern = re.compile(
        r'https?://'                    # スキーム
        r'(?:[a-zA-Z0-9]'              # ドメインの先頭文字
        r'(?:[a-zA-Z0-9-]{0,61}'       # ドメイン名本体
        r'[a-zA-Z0-9])?\.)'            # ドメインの末尾
        r'+[a-zA-Z]{2,}'               # TLD
        r'(?::\d{1,5})?'               # ポート(オプション)
        r'(?:/[^\s]*)?'                # パス(オプション)
    )
    return url_pattern.findall(text)

def mask_personal_info(text: str) -> str:
    """個人情報のマスキング"""
    # メールアドレス
    text = re.sub(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        '***@***.***',
        text
    )

    # 電話番号(日本)
    text = re.sub(
        r'0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{4}',
        '***-****-****',
        text
    )

    # クレジットカード番号
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '****-****-****-****',
        text
    )

    # マイナンバー（12桁の数字）
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '****-****-****',
        text
    )

    return text
```

### 4.4 適用領域マップ

| 領域 | 代表的な使い方 | 推奨度 | 備考 |
|------|---------------|--------|------|
| ログ解析 | エラーパターン抽出、集計 | 最適 | 構造化されたログには特に有効 |
| 入力バリデーション | メール、電話番号、郵便番号 | 適切 | ただし過信は禁物、二段階検証を推奨 |
| テキストエディタ | 検索・置換 | 最適 | IDE での一括置換に不可欠 |
| Web スクレイピング | HTML からの情報抽出 | 注意 | HTMLパーサー推奨、正規表現は補助的に |
| コンパイラ/字句解析 | トークン分割 | 適切 | lexer生成器と併用 |
| 自然言語処理 | 形態素解析の前処理 | 限定的 | 専用ライブラリとの組み合わせ |
| データ移行 | フォーマット変換 | 適切 | CSV, TSV の列操作等 |
| セキュリティ | WAF のルール定義 | 注意 | ReDoS リスクの考慮が必要 |
| バイナリ解析 | パターン検出 | 不適切 | バイナリ専用ツールを使用 |
| 設定ファイル | テンプレート展開 | 限定的 | 専用テンプレートエンジンを推奨 |

### 4.5 テキスト処理パイプラインにおける正規表現

```bash
# Unix パイプラインでの正規表現活用例

# 例1: アクセスログからエラーレスポンスのIPアドレスを集計
cat access.log \
  | grep -E '" [45]\d{2} ' \
  | awk '{print $1}' \
  | sort | uniq -c | sort -rn \
  | head -20

# 例2: ソースコードから関数定義を抽出
grep -rn 'def \w\+(' --include='*.py' src/ \
  | sed 's/.*def \(\w\+\)(.*/\1/' \
  | sort | uniq -c | sort -rn

# 例3: JSON ログからエラーメッセージを抽出
cat app.log \
  | grep -oP '"error":\s*"[^"]*"' \
  | sed 's/"error":\s*"\(.*\)"/\1/' \
  | sort | uniq -c | sort -rn

# 例4: Git ログからチケット番号を抽出
git log --oneline \
  | grep -oE '[A-Z]+-[0-9]+' \
  | sort | uniq -c | sort -rn
```

---

## 5. 正規表現の限界

### 5.1 正規表現で表現できないもの

```
チョムスキー階層:
┌─────────────────────────────────────┐
│ タイプ0: 帰納的可算言語              │
│  ┌──────────────────────────────┐   │
│  │ タイプ1: 文脈依存言語         │   │
│  │  ┌───────────────────────┐   │   │
│  │  │ タイプ2: 文脈自由言語  │   │   │
│  │  │  ┌────────────────┐   │   │   │
│  │  │  │ タイプ3: 正則言語│   │   │   │
│  │  │  │ (正規表現)      │   │   │   │
│  │  │  └────────────────┘   │   │   │
│  │  │  例: HTML, JSON,      │   │   │
│  │  │      プログラミング言語│   │   │
│  │  └───────────────────────┘   │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘

正規表現（タイプ3）では、ネストした括弧の
対応関係を数えることができない。
例: ((())), {{{}}}, <div><div></div></div>
```

### 5.2 理論的限界の具体例

```python
# 正規表現では原理的に扱えないパターン

# 1. ネストした括弧の対応
#    a^n b^n (n個のaの後にn個のb) は正則言語ではない
#    例: ab, aabb, aaabbb は受理、aab, abbb は拒否
#    → 正規表現では記述不可能

# 2. 回文の認識
#    例: "abcba", "racecar"
#    → 正規表現では記述不可能

# 3. 素数長の文字列
#    長さが素数の文字列だけにマッチ
#    → 正規表現では記述不可能

# ただし、実用的な「正規表現」(PCRE等)は
# 理論的な正則言語を超える機能を持つ:

import re

# .NET のバランシンググループを使えば
# ネストした括弧のマッチが可能（理論的にはCFGの領域）
# (?<open>\()  (?<close-open>\))

# Perl/PCRE の再帰パターンでも可能
# \((?:[^()]*|(?R))*\)
```

### 5.3 実用上の限界

```python
# 正規表現が適さないケース

# 1. 複雑な文法の解析
# NG: JSON を正規表現でパース
json_text = '{"name": "John", "address": {"city": "Tokyo"}}'
# → JSON パーサーを使うべき

# 2. マルチバイト文字の高度な処理
# NG: 漢字の読み仮名を正規表現で推測
# → 形態素解析器(MeCab, Janome等)を使うべき

# 3. 文脈依存の解析
# NG: Pythonのインデントベースのブロック構造
# → パーサー(AST)を使うべき

# 4. 大規模テキストの構造解析
# NG: 数百MBのXMLファイルを正規表現で処理
# → SAXパーサー等のストリーミング処理を使うべき

# 5. 自然言語の意味解析
# NG: 文の主語と述語を正規表現で抽出
# → NLP ライブラリ(spaCy, GiNZA等)を使うべき
```

---

## 6. アンチパターン

### 6.1 アンチパターン: HTML を正規表現でパースする

```python
# NG: HTML を正規表現で処理しようとする
import re

html = '<div class="outer"><div class="inner">text</div></div>'

# このパターンはネストした div を正しく処理できない
pattern = r'<div[^>]*>(.*?)</div>'
result = re.findall(pattern, html)
print(result)  # => ['<div class="inner">text']  -- 不正確

# OK: HTML パーサーを使う
from html.parser import HTMLParser
# または Beautiful Soup, lxml 等を使用する
```

**理由**: HTML は文脈自由言語であり、正規表現(正則言語)の表現力を超える。ネストした要素の対応を正しく追跡できない。

### 6.2 アンチパターン: 万能バリデーションパターン

```python
# NG: RFC 5322 完全準拠のメールアドレスパターン（実際に存在する）
# 数千文字に及ぶ正規表現 → 保守不能、デバッグ不能

email_pattern_bad = r"""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@..."""
# (省略 -- 実際は数百文字以上)

# OK: 実用的なバリデーション + 確認メール送信
import re

def validate_email(email: str) -> bool:
    """実用的なメール検証 -- 形式チェック + 確認メール"""
    # 基本形式のみチェック
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return False
    # 本当の検証は確認メールを送信して行う
    return True
```

**理由**: 完璧な正規表現を書くより、シンプルなパターン + 別の検証手段を組み合わせるほうが保守性・信頼性ともに高い。

### 6.3 アンチパターン: ReDoS 脆弱パターン

```python
# NG: 壊滅的バックトラック(Catastrophic Backtracking)を引き起こすパターン
import re
import time

# 危険なパターンの例
dangerous_patterns = [
    r'(a+)+$',           # ネストした量指定子
    r'(a|a)+$',          # 重複する選択肢
    r'(a+b?)+$',         # 組み合わせ爆発
    r'([a-zA-Z]+)*$',    # ネストした量指定子（文字クラス版）
]

# 攻撃文字列: 大量の 'a' の後に不一致文字
evil_input = 'a' * 30 + '!'

for pattern in dangerous_patterns:
    start = time.time()
    try:
        re.match(pattern, evil_input)
    except Exception:
        pass
    elapsed = time.time() - start
    print(f'Pattern: {pattern:30s} Time: {elapsed:.3f}s')
    # パターンによっては数秒〜数十秒かかる

# OK: ReDoS に強いパターンの書き方
safe_patterns = [
    r'a+$',              # ネストを避ける
    r'(?:a+)+$',         # 非キャプチャでも危険は同じ → a+$ に書き換え
    r'[a-zA-Z]+$',       # ネストを避ける
]

# OK: タイムアウト付きマッチング (Python 3.11+)
# import signal
# signal.alarm(1)  # 1秒タイムアウト
```

### 6.4 アンチパターン: 可読性の低いパターン

```python
# NG: 一行に詰め込まれた長大なパターン
pattern_bad = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'

# OK: verbose モードで可読性を確保
import re

pattern_good = re.compile(r'''
    ^
    (?:
        (?:
            25[0-5]           # 250-255
            | 2[0-4][0-9]     # 200-249
            | [01]?[0-9][0-9]? # 0-199
        )
        \.                     # ドット区切り
    ){3}                       # 最初の3オクテット
    (?:
        25[0-5]               # 250-255
        | 2[0-4][0-9]         # 200-249
        | [01]?[0-9][0-9]?    # 0-199
    )
    $
''', re.VERBOSE)

# テスト
assert pattern_good.match('192.168.1.1')
assert pattern_good.match('255.255.255.255')
assert not pattern_good.match('256.1.1.1')
assert not pattern_good.match('1.2.3.4.5')
```

### 6.5 アンチパターン: パフォーマンスを考慮しないコンパイル

```python
import re

# NG: ループ内で毎回コンパイル
def search_bad(lines, pattern_str):
    results = []
    for line in lines:
        if re.search(pattern_str, line):  # 毎回コンパイル
            results.append(line)
    return results

# OK: 事前にコンパイル
def search_good(lines, pattern_str):
    pattern = re.compile(pattern_str)  # 1回だけコンパイル
    results = []
    for line in lines:
        if pattern.search(line):       # コンパイル済みオブジェクトを再利用
            results.append(line)
    return results

# OK: さらに良い方法（リスト内包表記）
def search_best(lines, pattern_str):
    pattern = re.compile(pattern_str)
    return [line for line in lines if pattern.search(line)]

# ※ Python の re モジュールは内部的にパターンキャッシュ（最大512個）を
#    持っているため、少数のパターンなら re.search() でも問題ないが、
#    明示的なコンパイルが推奨される
```

---

## 7. 正規表現のデバッグとテスト

### 7.1 段階的なパターン構築

```python
import re

# 複雑なパターンは段階的に構築する

# 目標: Apache ログの解析パターン
# サンプル: 192.168.1.1 - - [10/Feb/2026:13:55:36 +0900] "GET /index.html HTTP/1.1" 200 2326

# ステップ 1: IPアドレス部分
step1 = r'\d+\.\d+\.\d+\.\d+'
assert re.search(step1, '192.168.1.1')

# ステップ 2: タイムスタンプ部分
step2 = r'\[([^\]]+)\]'
assert re.search(step2, '[10/Feb/2026:13:55:36 +0900]')

# ステップ 3: リクエスト行
step3 = r'"(\w+) ([^\s]+) HTTP/[\d.]+"'
assert re.search(step3, '"GET /index.html HTTP/1.1"')

# ステップ 4: ステータスコードとサイズ
step4 = r'(\d{3}) (\d+|-)'
assert re.search(step4, '200 2326')

# ステップ 5: 全体を結合
full_pattern = re.compile(
    rf'({step1})'     # IP
    r' - - '          # ident, auth
    rf'{step2}'       # timestamp
    r' '
    rf'{step3}'       # request
    r' '
    rf'{step4}'       # status, size
)

log_line = '192.168.1.1 - - [10/Feb/2026:13:55:36 +0900] "GET /index.html HTTP/1.1" 200 2326'
m = full_pattern.match(log_line)
assert m is not None
print(f'IP: {m.group(1)}')
print(f'Timestamp: {m.group(2)}')
print(f'Method: {m.group(3)}')
print(f'Path: {m.group(4)}')
print(f'Status: {m.group(5)}')
print(f'Size: {m.group(6)}')
```

### 7.2 テスト駆動での正規表現開発

```python
import re
import pytest

class TestPhoneNumberPattern:
    """電話番号パターンのテスト"""

    pattern = re.compile(r'^0\d{1,4}-\d{1,4}-\d{4}$')

    # 正常系: マッチすべきもの
    @pytest.mark.parametrize("phone", [
        "03-1234-5678",       # 東京
        "06-1234-5678",       # 大阪
        "090-1234-5678",      # 携帯
        "080-1234-5678",      # 携帯
        "0120-123-4567",      # フリーダイヤル
        "0466-12-3456",       # 市外局番4桁
    ])
    def test_valid_phones(self, phone):
        assert self.pattern.match(phone), f"{phone} should match"

    # 異常系: マッチすべきでないもの
    @pytest.mark.parametrize("phone", [
        "1234-5678",          # 先頭が0でない
        "03-1234-567",        # 加入者番号が3桁
        "03-1234-56789",      # 加入者番号が5桁
        "abc-defg-hijk",      # 数字でない
        "",                   # 空文字列
        "03 1234 5678",       # ハイフンでなくスペース
    ])
    def test_invalid_phones(self, phone):
        assert not self.pattern.match(phone), f"{phone} should not match"

class TestEmailPattern:
    """メールアドレスパターンのテスト"""

    pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )

    @pytest.mark.parametrize("email", [
        "user@example.com",
        "user.name@example.co.jp",
        "user+tag@example.org",
        "user123@sub.domain.com",
    ])
    def test_valid_emails(self, email):
        assert self.pattern.match(email)

    @pytest.mark.parametrize("email", [
        "user@",
        "@example.com",
        "user@.com",
        "user@com",
        "",
        "user space@example.com",
    ])
    def test_invalid_emails(self, email):
        assert not self.pattern.match(email)
```

### 7.3 デバッグツールの活用

```python
# Python の re.DEBUG フラグでパターンの内部構造を確認
import re

# パターンの解析結果を表示
re.compile(r'\d{3}-\d{4}', re.DEBUG)
# 出力:
# MAX_REPEAT 3 3
#   IN
#     CATEGORY CATEGORY_DIGIT
# LITERAL 45 ('-')
# MAX_REPEAT 4 4
#   IN
#     CATEGORY CATEGORY_DIGIT

# verbose モードでコメント付きパターン
pattern = re.compile(r"""
    (?P<year>\d{4})     # 年: 4桁の数字
    [-/]                # 区切り: ハイフンまたはスラッシュ
    (?P<month>\d{2})    # 月: 2桁の数字
    [-/]                # 区切り
    (?P<day>\d{2})      # 日: 2桁の数字
""", re.VERBOSE)

result = pattern.match('2026-02-15')
if result:
    print(result.groupdict())
    # => {'year': '2026', 'month': '02', 'day': '15'}
```

---

## 8. 正規表現のベストプラクティス

### 8.1 設計原則

```
正規表現の設計原則:

1. KISS原則 (Keep It Simple, Stupid)
   - 必要最小限のパターンを書く
   - 完璧を目指さず、80%カバーでOKとする
   - 残り20%は別のロジックで処理

2. 段階的精緻化
   - 簡単なパターンから始めて段階的に精度を上げる
   - 各段階でテストを追加する

3. コメント必須
   - 10文字を超えるパターンにはコメントを付ける
   - verbose モードの活用を推奨

4. パフォーマンス考慮
   - 事前コンパイルを基本とする
   - ReDoS パターンを避ける
   - アンカーの活用で検索範囲を限定

5. テスト駆動
   - 正常系・異常系・境界値のテストを書く
   - エッジケースを明示的にカバーする
```

### 8.2 命名規則

```python
# パターンの命名に一貫性を持たせる

# 検証パターン: is_* または validate_*
PATTERN_IS_EMAIL = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
PATTERN_IS_URL = re.compile(r'^https?://[^\s]+$')
PATTERN_IS_IPV4 = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')

# 抽出パターン: extract_* または parse_*
PATTERN_EXTRACT_DATE = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
PATTERN_EXTRACT_IP = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')

# 置換パターン: replace_* または clean_*
PATTERN_CLEAN_WHITESPACE = re.compile(r'\s+')
PATTERN_CLEAN_HTML_TAGS = re.compile(r'<[^>]+>')
```

---

## 9. FAQ

### Q1: 正規表現を学ぶのに最適な順序は？

**A**: 以下の順序を推奨する:

1. リテラルマッチ → メタ文字 → 文字クラス → 量指定子
2. アンカー → グループ → 後方参照
3. 先読み・後読み → Unicode → パフォーマンス

基本構文を習得してから、実際のユースケース(ログ解析、入力検証など)で練習するのが効果的である。

### Q2: NFA と DFA のどちらを選ぶべきか？

**A**: 以下の判断基準で選択する:

- **後方参照や先読みが必要** → NFA(Perl, Python, JavaScript など)
- **信頼されない入力に対するマッチング** → DFA(RE2, Rust regex)で ReDoS を防止
- **パフォーマンスが最優先** → DFA ベースのエンジンを選択
- **機能の豊富さが優先** → NFA ベースのエンジンを選択

多くの場合、言語のデフォルトエンジンを使えばよい。ReDoS が懸念される場合のみ DFA を検討する。

### Q3: 正規表現のデバッグにはどんなツールがあるか？

**A**: 主要なツール:

- **regex101.com** -- パターンのリアルタイムテスト、解説付き(PCRE, Python, JS, Go, Java 対応)
- **Debuggex** -- 正規表現の鉄道図(Railroad Diagram)を表示
- **RegExr** -- インタラクティブな正規表現テスター
- **各言語のverboseモード** -- Python の `re.VERBOSE`、Perl の `/x` 修飾子
- **各言語のデバッグフラグ** -- Python の `re.DEBUG`

### Q4: 正規表現のパフォーマンスが悪いときの対処法は？

**A**: 以下の順序で検討する:

1. **パターンの見直し**: アンカーの追加、不必要なバックトラックの排除
2. **事前コンパイル**: `re.compile()` でパターンオブジェクトを再利用
3. **文字列操作への置き換え**: `str.startswith()`, `str.endswith()`, `in` 演算子
4. **エンジンの変更**: RE2 や Rust regex への移行
5. **アルゴリズムの変更**: 正規表現を使わない解法の検討

### Q5: 正規表現はどの程度の複雑さまで許容すべきか？

**A**: 経験則として:

- **20文字以下**: インラインで使用可能
- **20-50文字**: verbose モードでコメント推奨
- **50-100文字**: 分割して段階的に構築、テスト必須
- **100文字以上**: パーサーやライブラリの使用を検討

可読性が低下したら、正規表現を分割するか、別のアプローチを検討するタイミングである。

### Q6: 正規表現のセキュリティリスクにはどう対処するか？

**A**: 主要なリスクと対策:

1. **ReDoS (Regular Expression Denial of Service)**
   - ネストした量指定子を避ける: `(a+)+` → `a+`
   - 信頼できない入力にはDFAエンジンを使用
   - マッチングにタイムアウトを設定

2. **入力検証のバイパス**
   - `^` と `$` だけでなく `\A` と `\z` を使用（改行文字の考慮）
   - マルチラインモードの意図しない有効化を防ぐ
   - Unicode 正規化を事前に行う

3. **インジェクション攻撃**
   - ユーザー入力をパターンに埋め込まない
   - 埋め込む場合は `re.escape()` で必ずエスケープ

---

## まとめ

| 項目 | 内容 |
|------|------|
| 正規表現の本質 | パターンマッチングのための形式言語 |
| 理論的基盤 | Kleene の正則集合(1956年) |
| 実用化の起点 | Thompson による QED/ed への実装(1968年) |
| 現代の標準 | PCRE (Perl Compatible Regular Expressions) |
| NFA | バックトラック方式、機能豊富、最悪 O(2^n) |
| DFA | 状態遷移方式、高速安定、O(n) 保証 |
| 主な用途 | テキスト検索・置換・抽出・検証 |
| 主な限界 | ネスト構造(HTML等)は原理的に扱えない |
| 主要方言 | POSIX BRE/ERE, PCRE, ECMAScript, RE2 |
| セキュリティ | ReDoS 対策が必須(特にNFAエンジン) |
| ベストプラクティス | 段階的構築、テスト駆動、verbose モード |
| 学習の鍵 | 基本構文→応用構文→実践パターンの順で段階的に |

---

## 次に読むべきガイド

- [01-basic-syntax.md](./01-basic-syntax.md) -- 基本構文: リテラル、メタ文字、エスケープ
- [02-character-classes.md](./02-character-classes.md) -- 文字クラスの詳細
- [03-quantifiers-anchors.md](./03-quantifiers-anchors.md) -- 量指定子とアンカー

---

## 参考文献

1. **Jeffrey E.F. Friedl** "Mastering Regular Expressions, 3rd Edition" O'Reilly Media, 2006 -- 正規表現の決定版バイブル
2. **Russ Cox** "Regular Expression Matching Can Be Simple And Fast" https://swtch.com/~rsc/regexp/regexp1.html, 2007 -- NFA/DFA の理論と実装を明快に解説
3. **Ken Thompson** "Regular Expression Search Algorithm" Communications of the ACM, 11(6):419-422, 1968 -- 正規表現エンジンの原論文
4. **Michael Rabin, Dana Scott** "Finite Automata and Their Decision Problems" IBM Journal of Research and Development, 3(2):114-125, 1959 -- NFA/DFA の等価性を証明した論文
5. **PCRE2 Documentation** https://www.pcre.org/current/doc/html/ -- 現行の PCRE2 リファレンス
6. **RE2 Documentation** https://github.com/google/re2/wiki/Syntax -- RE2 の構文リファレンス
7. **ECMAScript Language Specification** https://tc39.es/ecma262/#sec-regexp-regular-expression-objects -- JavaScript 正規表現の仕様
8. **POSIX Standard** IEEE Std 1003.1-2017, Section 9 "Regular Expressions" -- POSIX 正規表現の公式仕様
