# V8エンジン

> V8はGoogleが開発した高性能JavaScriptエンジンであり、Chrome、Node.js、Denoなど広範なランタイム環境の基盤を形成している。本ガイドではV8の内部アーキテクチャを深掘りし、パーサー、Ignition（インタプリタ）、TurboFan（最適化コンパイラ）、Hidden Class、インラインキャッシュ、ガベージコレクションの各メカニズムを体系的に解説する。

## この章で学ぶこと

- [ ] V8のソースコード処理パイプライン全体像を理解する
- [ ] パーサーの仕組み（Lazy Parsing / Eager Parsing）を把握する
- [ ] Ignition バイトコードインタプリタの動作原理を理解する
- [ ] TurboFan 最適化コンパイラの最適化手法を学ぶ
- [ ] Hidden Class によるオブジェクト表現の内部構造を把握する
- [ ] インラインキャッシュ（IC）の状態遷移を理解する
- [ ] 世代別ガベージコレクションの戦略を学ぶ
- [ ] V8に最適化されたコードの書き方を実践する

---

## 1. V8の全体アーキテクチャ

V8は単なるインタプリタではなく、複数のフェーズを持つ高度なコンパイルパイプラインである。JavaScriptのソースコードは以下のステージを経て実行される。

### 1.1 パイプライン全体図

```
┌─────────────────────────────────────────────────────────────────┐
│                    V8 コンパイルパイプライン                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  JavaScript ソースコード (.js)                                   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                            │
│  │   Scanner        │  字句解析（Tokenizer）                     │
│  │   (Lexer)        │  ソースコードをトークン列に分解              │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   Parser         │  構文解析                                  │
│  │   (Full/Lazy)    │  トークン列からAST（抽象構文木）を構築       │
│  └────────┬────────┘                                            │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   Ignition       │  バイトコードインタプリタ                    │
│  │   (Interpreter)  │  ASTからバイトコードを生成・実行             │
│  │                  │  + フィードバックベクタでプロファイリング      │
│  └────────┬────────┘                                            │
│           │                                                     │
│           │  ホットスポット検出                                   │
│           │  （一定回数以上実行された関数）                        │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   TurboFan       │  最適化コンパイラ                           │
│  │   (Optimizing    │  バイトコード + 型フィードバック →            │
│  │    Compiler)     │  最適化されたマシンコードを生成              │
│  └────────┬────────┘                                            │
│           │                                                     │
│           │  脱最適化（Deoptimization）                          │
│           │  前提条件が崩れた場合、Ignitionに戻る                  │
│           ▼                                                     │
│  ┌─────────────────┐                                            │
│  │   実行            │  最適化マシンコード or バイトコードを実行    │
│  └─────────────────┘                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 歴史的経緯

V8のコンパイルパイプラインは、バージョンを重ねるごとに大きく変遷してきた。

| 時期 | コンパイラ構成 | 特徴 |
|------|---------------|------|
| 2008-2010 | Full-Codegen + Crankshaft なし | 初期の単純なJITコンパイラ |
| 2010-2016 | Full-Codegen + Crankshaft | 2段階JIT。Crankshaftがホットコードを最適化 |
| 2016-2017 | Ignition + TurboFan（段階導入） | 新パイプラインへの移行期 |
| 2017-現在 | Ignition + TurboFan | 現行パイプライン。バイトコード + 最適化JIT |

Full-Codegenは全関数を即座にネイティブコードにコンパイルする方式だったが、起動が遅く、メモリ使用量が多かった。Ignitionの導入により、まずバイトコードとして実行し、本当に必要なコードだけを最適化コンパイルする効率的なアプローチが実現された。

---

## 2. パーサー（Parser）

### 2.1 字句解析（Scanner）

Scannerはソースコードの文字列をトークン（Token）の列に変換する。トークンとはプログラミング言語の最小単位であり、キーワード、識別子、リテラル、演算子などに分類される。

```javascript
// ソースコード
function add(a, b) { return a + b; }

// トークン列に変換
// Token::kFunction  → "function"
// Token::kIdentifier → "add"
// Token::kLeftParen  → "("
// Token::kIdentifier → "a"
// Token::kComma      → ","
// Token::kIdentifier → "b"
// Token::kRightParen → ")"
// Token::kLeftBrace  → "{"
// Token::kReturn     → "return"
// Token::kIdentifier → "a"
// Token::kAdd        → "+"
// Token::kIdentifier → "b"
// Token::kSemicolon  → ";"
// Token::kRightBrace → "}"
```

V8のScannerはUTF-16エンコーディングのソースコードを処理し、1文字先読み（one-character lookahead）を使ってトークンを識別する。数値リテラルや文字列リテラルのようなマルチキャラクタートークンは、専用のスキャンルーチンで処理される。

### 2.2 Lazy Parsing と Eager Parsing

V8のパーサーには2つの動作モードがある。これはV8のパフォーマンス戦略の核心部分である。

```
┌───────────────────────────────────────────────────────────┐
│                  パーサーの2つのモード                       │
├──────────────────────┬────────────────────────────────────┤
│   Eager Parsing      │   Lazy Parsing (PreParser)         │
│   （完全解析）         │   （遅延解析）                      │
├──────────────────────┼────────────────────────────────────┤
│ ・完全なASTを構築     │ ・関数の中身をスキップ               │
│ ・即座に実行される     │ ・変数スコープだけ確認               │
│   コードに使用        │ ・構文エラーだけ検出                 │
│ ・トップレベルコード   │ ・実際に呼び出されるまで              │
│   に適用される        │   解析を延期                        │
│                      │ ・メモリと起動時間を節約              │
├──────────────────────┼────────────────────────────────────┤
│ コスト: 高            │ コスト: 低（完全解析の約半分）        │
│ 対象: 即実行コード     │ 対象: 関数宣言                      │
└──────────────────────┴────────────────────────────────────┘
```

**Lazy Parsing の仕組み:**

```javascript
// トップレベルコード → Eager Parsing（即座に完全解析）
const config = { debug: true };

// 関数宣言 → Lazy Parsing（中身はスキップ）
function processData(data) {
  // この中身は最初の呼び出し時まで解析されない
  const result = data.map(item => item.value * 2);
  return result.filter(v => v > 10);
}

// IIFE（即時実行関数） → Eager Parsing（即座に実行されるため）
(function() {
  console.log("immediately invoked");
})();

// processData が呼び出された時点で初めて完全解析される
processData([{ value: 5 }, { value: 8 }, { value: 15 }]);
```

**Lazy Parsing が有効な理由:**

一般的なWebページでは、読み込まれたJavaScriptコードの30-50%は初期表示で実行されない。Lazy Parsingにより、未使用の関数の解析コストを後回しにできるため、ページの初期読み込み速度が向上する。

**Lazy Parsing の落とし穴:**

```javascript
// アンチパターン: Lazy Parsing が裏目に出るケース
// 関数がすぐに呼ばれるのに、V8がLazy Parsingしてしまう

function heavyComputation() {
  // 大量のコード...
}

// すぐに呼ばれる → Lazy Parse + Re-parse で二重コスト
heavyComputation();

// 対策: V8にEager Parsingのヒントを与える
// 括弧で囲むと、V8はIIFEパターンと認識してEager Parsingする
const heavyComputation2 = (function() {
  // 大量のコード...
});
```

### 2.3 AST（抽象構文木）の構造

パーサーはトークン列からAST（Abstract Syntax Tree）を構築する。ASTはソースコードの構文構造をツリー形式で表現したものである。

```javascript
// ソースコード
function multiply(x, y) {
  return x * y;
}

// 対応するAST（概念的な表現）
//
//  FunctionDeclaration
//  ├── name: "multiply"
//  ├── params: ["x", "y"]
//  └── body: BlockStatement
//      └── ReturnStatement
//          └── BinaryExpression
//              ├── operator: "*"
//              ├── left: Identifier("x")
//              └── right: Identifier("y")
```

V8のASTノードは内部的にC++のクラスで表現され、各ノードはソースコード上の位置情報（SourcePosition）を保持している。これはエラーメッセージやデバッグ情報の生成に使用される。

---

## 3. Ignition バイトコードインタプリタ

### 3.1 Ignitionの役割

Ignitionは2016年にV8に導入されたレジスタベースのバイトコードインタプリタである。ASTからバイトコードを生成し、そのバイトコードを直接実行する。

**Ignition導入の動機:**

1. **メモリ使用量の削減** --- Full-Codegenが生成するネイティブコードに比べ、バイトコードはコンパクト
2. **起動速度の向上** --- バイトコード生成はネイティブコード生成より高速
3. **TurboFanへの情報提供** --- 実行中に収集した型情報をTurboFanに渡す

### 3.2 バイトコードの例

```javascript
// JavaScriptソースコード
function add(a, b) {
  return a + b;
}

// Ignitionが生成するバイトコード（概念的な表現）
// ※ 実際のバイトコードは --print-bytecode フラグで確認できる
//
// Parameter count: 3 (receiver + a + b)
// Register count: 0
// Frame size: 0
//
//   Ldar a1          // レジスタa1（引数a）をアキュムレータにロード
//   Add a2, [0]      // アキュムレータにa2（引数b）を加算
//                     // [0]はフィードバックスロットのインデックス
//   Return            // アキュムレータの値を返す
```

**Node.jsでバイトコードを確認する方法:**

```bash
# --print-bytecode フラグで Ignition のバイトコードを出力
node --print-bytecode --print-bytecode-filter="add" script.js

# 出力例:
# [generated bytecode for function: add (0x...)]
# Bytecode length: 6
# Parameter count 3
# Register count 0
# Frame size 0
#    0 : 25 02             Ldar a1
#    2 : 39 03 00          Add a2, [0]
#    5 : aa                Return
```

### 3.3 レジスタベースとスタックベースの比較

バイトコードインタプリタには大きく2つの方式がある。Ignitionはレジスタベース方式を採用している。

| 特性 | レジスタベース（Ignition） | スタックベース（旧Java VM等） |
|------|--------------------------|------------------------------|
| 命令の形式 | `Add r1, r2, r3` | `Push a; Push b; Add` |
| 命令数 | 少ない（オペランドに直接指定） | 多い（Push/Pop操作が必要） |
| バイトコードサイズ | やや大きい（オペランド指定分） | 小さい |
| 実行速度 | 高速（メモリアクセス削減） | やや遅い（スタック操作のオーバーヘッド） |
| ディスパッチ回数 | 少ない | 多い |
| 採用例 | V8 Ignition, Lua VM | JVM, Python VM, .NET CLR |

### 3.4 フィードバックベクタ（Feedback Vector）

Ignitionは実行中に型情報やプロパティアクセスパターンなどのプロファイル情報を収集する。これは**フィードバックベクタ**と呼ばれるデータ構造に保存される。

```
┌─────────────────────────────────────────────────────────┐
│              フィードバックベクタの構造                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  function calculate(obj) {                              │
│    return obj.x + obj.y;  // 2つのプロパティアクセス      │
│  }                                                      │
│                                                         │
│  フィードバックベクタ:                                    │
│  ┌─────────┬──────────────────────────────────┐         │
│  │ Slot 0  │ LoadIC: obj.x                    │         │
│  │         │ → Map: 0x1234 (Hidden Class)     │         │
│  │         │ → Offset: 12                     │         │
│  │         │ → State: monomorphic             │         │
│  ├─────────┼──────────────────────────────────┤         │
│  │ Slot 1  │ LoadIC: obj.y                    │         │
│  │         │ → Map: 0x1234 (Hidden Class)     │         │
│  │         │ → Offset: 16                     │         │
│  │         │ → State: monomorphic             │         │
│  ├─────────┼──────────────────────────────────┤         │
│  │ Slot 2  │ BinaryOp: +                      │         │
│  │         │ → Hint: SignedSmall              │         │
│  └─────────┴──────────────────────────────────┘         │
│                                                         │
│  この情報がTurboFanに渡され、最適化の判断材料になる        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

フィードバックベクタに蓄積された型情報は、TurboFanが最適化コンパイルを行う際の根拠となる。たとえば「この加算は常にSMI（Small Integer）同士で行われている」という情報があれば、TurboFanは整数加算に特化したマシンコードを生成できる。

---

## 4. TurboFan 最適化コンパイラ

### 4.1 TurboFanの概要

TurboFanはV8の最適化コンパイラであり、IgnitionのバイトコードとフィードバックベクタをもとにSSA（Static Single Assignment）形式の中間表現を構築し、数多くの最適化パスを適用した後、高効率なマシンコードを生成する。

### 4.2 最適化のトリガー条件

TurboFanが関数を最適化コンパイルする条件は以下の通りである。

```javascript
// 最適化のトリガー例
function hotFunction(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    sum += arr[i];
  }
  return sum;
}

// ループの反復回数やバイトコード実行回数が閾値を超えると
// TurboFanによる最適化が開始される
//
// 閾値の目安（V8内部で動的に調整される）:
// - 関数の呼び出し回数
// - ループのバックエッジカウント（ループ1回転ごとにカウント）
// - OSR（On-Stack Replacement）の判定
//
// Node.jsで最適化の状況を確認:
// node --trace-opt --trace-deopt script.js
```

### 4.3 主要な最適化手法

TurboFanが適用する代表的な最適化手法を以下に列挙する。

**インライン展開（Inlining）:**

```javascript
// 最適化前
function square(x) {
  return x * x;
}

function sumOfSquares(a, b) {
  return square(a) + square(b);
}

// TurboFanによるインライン展開後（概念的）
function sumOfSquares_optimized(a, b) {
  return a * a + b * b;  // 関数呼び出しのオーバーヘッドを排除
}
```

インライン展開により、関数呼び出しのオーバーヘッド（スタックフレームの作成、引数の受け渡し、リターンアドレスの管理）が排除される。TurboFanは関数のサイズ、呼び出し頻度、呼び出しの深さなどを考慮してインライン展開の判断を行う。

**定数畳み込み（Constant Folding）:**

```javascript
// 最適化前
const TIMEOUT = 60 * 1000;  // 60秒をミリ秒に変換

// TurboFanによる最適化後
const TIMEOUT = 60000;  // コンパイル時に計算済み
```

**デッドコード除去（Dead Code Elimination）:**

```javascript
// 最適化前
function process(x) {
  const unused = x * 2;  // この結果は使われない
  return x + 1;
}

// TurboFanによる最適化後
function process_optimized(x) {
  return x + 1;  // 不要な計算を除去
}
```

**ループ不変式の外出し（Loop-Invariant Code Motion）:**

```javascript
// 最適化前
function processArray(arr, factor) {
  const results = [];
  for (let i = 0; i < arr.length; i++) {
    const multiplier = factor * 2;  // ループ内で毎回同じ計算
    results.push(arr[i] * multiplier);
  }
  return results;
}

// TurboFanによる最適化後（概念的）
function processArray_optimized(arr, factor) {
  const results = [];
  const multiplier = factor * 2;  // ループの外に移動
  const len = arr.length;         // 長さの取得もループ外へ
  for (let i = 0; i < len; i++) {
    results.push(arr[i] * multiplier);
  }
  return results;
}
```

**型特殊化（Type Specialization）:**

```javascript
// フィードバックベクタの情報:
// → add関数は常にSMI（Small Integer）引数で呼ばれている

function add(a, b) {
  return a + b;
}

// TurboFanが生成するマシンコード（概念的な疑似コード）:
//
// 1. a が SMI か確認（型ガード）
// 2. b が SMI か確認（型ガード）
// 3. SMI同士の整数加算（1命令で完了）
// 4. オーバーフローチェック
// 5. 結果を返す
//
// もし型ガードが失敗したら → 脱最適化
```

### 4.4 脱最適化（Deoptimization）

TurboFanが生成した最適化コードは、型の前提条件に基づいている。この前提が実行時に崩れた場合、V8は**脱最適化**を行い、Ignitionのバイトコード実行に戻す。

```javascript
// 脱最適化が起きるシナリオ

function add(a, b) {
  return a + b;
}

// Phase 1: SMI（整数）で呼び続ける → TurboFanが整数加算に最適化
for (let i = 0; i < 100000; i++) {
  add(i, i + 1);  // 常に整数
}

// Phase 2: 突然文字列を渡す → 脱最適化が発生！
add("hello", " world");
// → TurboFanの最適化コードは整数加算前提なので使えない
// → Ignitionのバイトコードに戻って文字列結合を実行
// → 再度プロファイリングを行い、新たな最適化を検討
```

**脱最適化の種類:**

```
┌─────────────────────────────────────────────────────────┐
│              脱最適化の分類                                │
├─────────────────┬───────────────────────────────────────┤
│ Eager Deopt     │ 型ガードの失敗など、即座に検出される     │
│                 │ 例: 整数を期待した箇所に文字列が来た     │
├─────────────────┼───────────────────────────────────────┤
│ Lazy Deopt      │ コードの実行後、副作用の処理中に検出     │
│                 │ 例: マップの変更が検出された              │
├─────────────────┼───────────────────────────────────────┤
│ Soft Deopt      │ 最適化コードが非効率と判断された場合     │
│                 │ 例: 多態的な呼び出しサイトの検出          │
└─────────────────┴───────────────────────────────────────┘
```

### 4.5 OSR（On-Stack Replacement）

通常の最適化は関数の次の呼び出し時から適用されるが、OSRは**実行中のループの途中**で最適化コードに切り替える技術である。

```javascript
function longRunningLoop() {
  let sum = 0;
  for (let i = 0; i < 10000000; i++) {
    sum += i;
    // ループのバックエッジで最適化判定
    // 閾値を超えた時点で、ループ実行中に最適化コードに切り替え（OSR）
    // → ループ変数 i, sum の状態を最適化コードに引き継ぐ
  }
  return sum;
}

// この関数は1回しか呼ばれないが、ループ内で
// OSRにより最適化される
longRunningLoop();
```

OSRは長時間実行されるループに対して特に有効である。関数が1回しか呼ばれなくても、ループの反復回数が閾値を超えれば最適化が適用される。

---

## 5. Hidden Class（Maps）

### 5.1 Hidden Classの基本概念

JavaScriptのオブジェクトは動的であり、実行時にプロパティの追加・削除が自由にできる。しかし、この柔軟性はプロパティアクセスのパフォーマンスに悪影響を及ぼす。V8はこの問題を**Hidden Class**（V8の内部用語では**Map**）という仕組みで解決している。

Hidden Classはオブジェクトの「形状（Shape）」を記述するメタデータであり、以下の情報を含む:

- プロパティの名前
- プロパティのオフセット（メモリ上の位置）
- プロパティの属性（writable、enumerable、configurable）
- プロトタイプチェーンの参照

### 5.2 Hidden Classの遷移チェーン

オブジェクトにプロパティが追加されるたびに、新しいHidden Classが作成され、遷移チェーンが形成される。

```
┌───────────────────────────────────────────────────────────────┐
│           Hidden Class 遷移チェーンの例                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  const point = {};          // Map M0 (空オブジェクト)         │
│  point.x = 10;             // Map M0 → M1 遷移               │
│  point.y = 20;             // Map M1 → M2 遷移               │
│                                                               │
│                                                               │
│  Map M0          Map M1              Map M2                   │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │ (empty)  │───▶│ x: offset 0  │───▶│ x: offset 0  │         │
│  │          │ x  │              │ y  │ y: offset 1  │         │
│  └─────────┘    └──────────────┘    └──────────────┘         │
│                                                               │
│  遷移情報は Map M0 に保存される:                               │
│  M0.transitions = { "x" → M1 }                               │
│  M1.transitions = { "y" → M2 }                               │
│                                                               │
│  別のオブジェクトが同じ順序でプロパティを追加すると              │
│  既存の遷移チェーンを再利用する:                                │
│                                                               │
│  const point2 = {};         // Map M0（同じ）                 │
│  point2.x = 30;            // Map M0 → M1（再利用）           │
│  point2.y = 40;            // Map M1 → M2（再利用）           │
│                                                               │
│  → point と point2 は同じ Map M2 を共有！                      │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 5.3 Hidden Classの共有と分岐

同じ「形状」のオブジェクトはHidden Classを共有するが、プロパティの追加順序が異なると別のHidden Classが作成される。

```javascript
// ケース1: 同じ順序 → Hidden Classを共有
const a = {};
a.x = 1;
a.y = 2;

const b = {};
b.x = 3;
b.y = 4;
// a と b は同じ Hidden Class

// ケース2: 異なる順序 → 別の Hidden Class
const c = {};
c.y = 2;  // まず y を追加
c.x = 1;  // 次に x を追加
// c は a, b とは異なる Hidden Class

// ケース3: オブジェクトリテラル → 最適化される
const d = { x: 1, y: 2 };
const e = { x: 3, y: 4 };
// d と e は同じ Hidden Class（リテラルの形状が同じ）

// ケース4: delete演算子 → Hidden Classが無効化
const f = { x: 1, y: 2 };
delete f.x;
// f は「遅いモード」（辞書モード）に切り替わる
// → Hidden Class の最適化が失われる
```

### 5.4 In-Object Properties vs. Backing Store

V8はオブジェクトのプロパティを2つの方法で格納する。

```
┌───────────────────────────────────────────────────────────┐
│         プロパティの格納方式                                  │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  【In-Object Properties】                                  │
│  オブジェクト本体に直接格納される                            │
│  ┌──────────────────────────┐                             │
│  │ Object Header            │                             │
│  │ ├── Map pointer          │                             │
│  │ ├── Properties pointer   │                             │
│  │ ├── Elements pointer     │                             │
│  │ ├── In-object prop 0 (x) │  ← 直接アクセス可能          │
│  │ ├── In-object prop 1 (y) │  ← 直接アクセス可能          │
│  │ └── In-object prop 2 (z) │  ← 直接アクセス可能          │
│  └──────────────────────────┘                             │
│  → 最も高速（固定オフセットで直接アクセス）                   │
│  → V8は初期プロパティ数を見積もってスペースを確保             │
│                                                           │
│  【Backing Store（Properties配列）】                        │
│  In-Objectスロットが不足した場合に使用                       │
│  ┌──────────────────────────┐    ┌──────────────┐        │
│  │ Object Header            │    │ Properties    │        │
│  │ ├── Map pointer          │    │ ├── prop 3    │        │
│  │ ├── Properties pointer ──┼───▶│ ├── prop 4    │        │
│  │ ├── Elements pointer     │    │ └── prop 5    │        │
│  │ ├── In-object prop 0     │    └──────────────┘        │
│  │ ├── In-object prop 1     │                             │
│  │ └── In-object prop 2     │                             │
│  └──────────────────────────┘                             │
│  → やや遅い（間接参照が1回必要）                             │
│                                                           │
│  【辞書モード（Slow Properties）】                           │
│  delete演算子使用後や、プロパティ数が非常に多い場合            │
│  → ハッシュテーブルベースの格納                              │
│  → Hidden Class の最適化が無効                              │
│  → 最も遅い                                                │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### 5.5 Elements（配列要素）の格納

オブジェクトの名前付きプロパティとは別に、数値インデックスのプロパティ（配列要素）は**Elements**として別の配列に格納される。

```javascript
// Elements の種類（V8内部のElementsKind）

// PACKED_SMI_ELEMENTS: 全要素が小さな整数
const smiArray = [1, 2, 3, 4, 5];

// PACKED_DOUBLE_ELEMENTS: 浮動小数点数を含む
const doubleArray = [1.1, 2.2, 3.3];

// PACKED_ELEMENTS: オブジェクトや混合型
const mixedArray = [1, "two", { three: 3 }];

// HOLEY_SMI_ELEMENTS: 穴あき配列（整数）
const holeyArray = [1, , 3];  // インデックス1が空

// ElementsKind の遷移（一方向のみ、逆戻りしない！）
//
// PACKED_SMI_ELEMENTS
//     │
//     ├──→ PACKED_DOUBLE_ELEMENTS
//     │        │
//     │        ├──→ PACKED_ELEMENTS
//     │        │
//     │        └──→ HOLEY_DOUBLE_ELEMENTS ──→ HOLEY_ELEMENTS
//     │
//     └──→ HOLEY_SMI_ELEMENTS
//              │
//              ├──→ HOLEY_DOUBLE_ELEMENTS ──→ HOLEY_ELEMENTS
//              │
//              └──→ HOLEY_ELEMENTS

// 一度でも HOLEY になると、PACKED に戻ることはない
const arr = [1, 2, 3];       // PACKED_SMI_ELEMENTS
arr.push(4.5);               // → PACKED_DOUBLE_ELEMENTS（不可逆遷移）
arr.push("hello");           // → PACKED_ELEMENTS（不可逆遷移）
```

---

## 6. インラインキャッシュ（Inline Cache: IC）

### 6.1 インラインキャッシュの基本原理

インラインキャッシュはプロパティアクセスの高速化メカニズムである。JavaScriptのプロパティアクセス `obj.x` は、本来であれば以下のステップが必要となる:

1. オブジェクトのHidden Classを取得
2. Hidden Classのプロパティテーブルで "x" を検索
3. オフセットを取得
4. そのオフセットでメモリからプロパティ値を読み取る

インラインキャッシュはこの検索結果をキャッシュし、同じHidden Classのオブジェクトが来た場合はステップ2-3をスキップする。

```javascript
function getX(obj) {
  return obj.x;  // ← このアクセス箇所にICが生成される
}

// 1回目の呼び出し: IC miss
// → Hidden Classを確認し、"x"のオフセットを検索
// → 結果をICにキャッシュ（Hidden Class → offset のペア）
const p1 = { x: 10, y: 20 };
getX(p1);

// 2回目以降: IC hit
// → Hidden Classがキャッシュと一致 → オフセットを直接使用
// → プロパティテーブルの検索をスキップ
const p2 = { x: 30, y: 40 };  // p1と同じHidden Class
getX(p2);  // 高速アクセス
```

### 6.2 ICの状態遷移（State Machine）

インラインキャッシュは以下の状態を持つ有限状態マシンとして動作する。

```
┌─────────────────────────────────────────────────────────────┐
│              IC 状態遷移図                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   1つ目のMap    ┌──────────────┐          │
│  │ Uninitialized │ ────────────▶ │ Monomorphic  │          │
│  │ (未初期化)     │               │ (単態)        │          │
│  └──────────────┘               └──────┬───────┘          │
│                                        │                   │
│                                 異なるMap  │                   │
│                                        ▼                   │
│                                ┌──────────────┐            │
│                                │ Polymorphic  │            │
│                                │ (多態: 2-4)   │            │
│                                └──────┬───────┘            │
│                                       │                    │
│                                5つ以上のMap │                    │
│                                       ▼                    │
│                                ┌──────────────┐            │
│                                │ Megamorphic  │            │
│                                │ (超多態: 5+)  │            │
│                                └──────────────┘            │
│                                                             │
│  パフォーマンス:                                              │
│  Monomorphic  ≫  Polymorphic  ≫  Megamorphic               │
│  （最速）          （中程度）       （最遅 / 最適化断念）       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 各状態の詳細

**Monomorphic（単態）--- 最速:**

```javascript
// 常に同じHidden Classのオブジェクトが渡される
function getName(person) {
  return person.name;  // IC: monomorphic
}

// 全て同じ形状のオブジェクト
getName({ name: "Alice", age: 30 });
getName({ name: "Bob", age: 25 });
getName({ name: "Charlie", age: 35 });
// → ICは1つのHidden Classだけを記録 → 最速
```

**Polymorphic（多態）--- 中程度:**

```javascript
// 2-4種類のHidden Classが混在
function getArea(shape) {
  return shape.area;  // IC: polymorphic
}

getArea({ area: 100, type: "circle" });     // Hidden Class A
getArea({ area: 200, width: 10, height: 20 }); // Hidden Class B
// → ICは2つのHidden Classを記録 → まだ十分高速
// → 線形検索でマッチするHidden Classを探す
```

**Megamorphic（超多態）--- 最遅:**

```javascript
// 5種類以上のHidden Classが混在
function getValue(obj) {
  return obj.value;  // IC: megamorphic
}

getValue({ value: 1 });
getValue({ value: 2, a: 1 });
getValue({ value: 3, a: 1, b: 2 });
getValue({ value: 4, a: 1, b: 2, c: 3 });
getValue({ value: 5, a: 1, b: 2, c: 3, d: 4 });
getValue({ value: 6, x: 1 });
// → ICがmegamorphic状態 → キャッシュ無効
// → 毎回Hidden Classの検索が必要 → 遅い
// → TurboFanも型特殊化を断念
```

### 6.4 ICの種類

V8にはプロパティアクセス以外にも複数のIC種類がある。

| IC種類 | 対象操作 | 例 |
|--------|----------|-----|
| LoadIC | プロパティ読み取り | `obj.x` |
| StoreIC | プロパティ書き込み | `obj.x = 1` |
| KeyedLoadIC | 動的キーによる読み取り | `obj[key]` |
| KeyedStoreIC | 動的キーによる書き込み | `obj[key] = 1` |
| CallIC | 関数呼び出し | `obj.method()` |
| CompareIC | 比較演算 | `a === b` |
| BinaryOpIC | 二項演算 | `a + b` |

---

## 7. ガベージコレクション（GC）

### 7.1 V8のメモリ構造

V8のヒープメモリは複数の領域に分割されている。

```
┌──────────────────────────────────────────────────────────────┐
│                    V8 ヒープメモリ構造                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────┐              │
│  │          New Space（新世代）                  │              │
│  │  ┌──────────────┬───────────────────┐      │              │
│  │  │  Semi-space   │  Semi-space       │      │              │
│  │  │  (From)       │  (To)             │      │              │
│  │  │  1-16 MB      │  1-16 MB          │      │              │
│  │  └──────────────┴───────────────────┘      │              │
│  │  新しく生成されたオブジェクトが配置される        │              │
│  │  Minor GC（Scavenge）の対象                   │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  ┌────────────────────────────────────────────┐              │
│  │          Old Space（旧世代）                  │              │
│  │  ┌──────────────────────────────────┐      │              │
│  │  │  Old Pointer Space                │      │              │
│  │  │  → 他のオブジェクトへの参照を含む   │      │              │
│  │  ├──────────────────────────────────┤      │              │
│  │  │  Old Data Space                   │      │              │
│  │  │  → 参照を含まないデータ（文字列等） │      │              │
│  │  └──────────────────────────────────┘      │              │
│  │  2回のMinor GCを生き延びたオブジェクト         │              │
│  │  Major GC（Mark-Sweep-Compact）の対象         │              │
│  │  サイズ: --max-old-space-size で設定可能       │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  ┌────────────────────────────────────────────┐              │
│  │          Large Object Space                  │              │
│  │  通常のページに収まらない巨大オブジェクト        │              │
│  │  個別にGC管理される                            │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  ┌────────────────────────────────────────────┐              │
│  │          Code Space                          │              │
│  │  JITコンパイルされたコードが配置される           │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
│  ┌────────────────────────────────────────────┐              │
│  │          Map Space                           │              │
│  │  Hidden Class（Map）オブジェクトが配置される    │              │
│  └────────────────────────────────────────────┘              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 Minor GC（Scavenge）

新世代のGCは**Scavenge**（Cheney's algorithm の変種）を使用する。

```
Scavenge アルゴリズムの流れ:

Step 1: 割り当て
┌────────────────────────────┬────────────────────────────┐
│ From Space (active)         │ To Space (inactive)         │
│ [A][B][C][D][E][ free ]    │ [ empty                  ] │
└────────────────────────────┴────────────────────────────┘

Step 2: GCトリガー（From Space が満杯に近づく）
  → ルートオブジェクトから到達可能なオブジェクトを特定
  → A, C, E が生存、B, D は到達不能（ゴミ）

Step 3: コピー
┌────────────────────────────┬────────────────────────────┐
│ From Space                  │ To Space                    │
│ [A][B][C][D][E][ free ]    │ [A'][C'][E'][ free       ] │
└────────────────────────────┴────────────────────────────┘
  → 生存オブジェクトをTo Spaceにコピー
  → コピー先のアドレスで参照を更新

Step 4: 入れ替え
┌────────────────────────────┬────────────────────────────┐
│ To Space → 新 From Space    │ From Space → 新 To Space    │
│ [A'][C'][E'][ free       ] │ [ empty (解放済み)        ] │
└────────────────────────────┴────────────────────────────┘
  → From と To を入れ替え
  → 旧From Spaceは丸ごと解放（個別のfreeが不要）

昇格（Promotion）:
  → 2回のScavengeを生き延びたオブジェクトはOld Spaceに移動
  → 「長寿命オブジェクト」と判断
```

### 7.3 Major GC（Mark-Sweep-Compact）

旧世代のGCは**Mark-Sweep-Compact**アルゴリズムを使用する。

**Mark フェーズ:**

```
ルートオブジェクト（スタック、グローバル変数等）から開始し、
到達可能な全オブジェクトを再帰的にマークする。

  Root Set
    │
    ├──▶ Object A (marked ✓)
    │      ├──▶ Object D (marked ✓)
    │      └──▶ Object E (marked ✓)
    │
    └──▶ Object B (marked ✓)
           └──▶ Object F (marked ✓)

  Object C (unmarked ✗) → 到達不能 → ゴミ
  Object G (unmarked ✗) → 到達不能 → ゴミ
```

**Sweep フェーズ:**

マークされていないオブジェクトのメモリを解放し、フリーリストに追加する。

**Compact フェーズ:**

メモリの断片化を解消するため、生存オブジェクトを移動して連続したメモリ領域にまとめる。これにより後続の割り当てが高速になる。

### 7.4 インクリメンタルマーキングとコンカレントGC

Major GCは旧世代全体を対象とするため、処理に時間がかかる。これによるメインスレッドの停止時間（Stop-the-world pause）を削減するため、V8は以下の手法を採用している。

```
【従来のStop-the-world GC】
JS実行 ────────┤ GC（長い停止） ├──── JS実行
               └─── 100ms+ ───┘

【インクリメンタルマーキング】
JS実行 ──┤GC├── JS ──┤GC├── JS ──┤GC├── JS実行
         5ms       5ms       5ms
→ GC作業を小さなステップに分割
→ JSの実行と交互に行う

【コンカレントGC】
メインスレッド: JS実行 ──────────────────────── JS実行
バックグラウンド:      ├── GC marking ──┤
→ GC作業の大部分をバックグラウンドスレッドで実行
→ メインスレッドの停止はほぼゼロに近づく

【パラレルGC】
メインスレッド:    ├── GC ──┤
ヘルパースレッド1: ├── GC ──┤
ヘルパースレッド2: ├── GC ──┤
→ 停止は必要だが、複数スレッドで並列処理して時間短縮
```

V8のOrinoco GCプロジェクトにより、これらの手法が組み合わされ、GCの停止時間は多くの場合数ミリ秒以下に抑えられている。

---

## 8. V8の配列最適化

### 8.1 ElementsKind の詳細

V8は配列の要素型に応じて内部表現を切り替える。適切なElementsKindを維持することで、配列操作のパフォーマンスが大幅に向上する。

```javascript
// PACKED_SMI_ELEMENTS: 最も高速
// SMI = Small Integer（31ビット符号付き整数、64ビットプラットフォームでは32ビット）
const smiArray = [1, 2, 3, 4, 5];
// → 要素がunboxed（タグなし）で格納される
// → ポインタ追跡やタグチェックが不要

// PACKED_DOUBLE_ELEMENTS
const doubleArray = [1.1, 2.2, 3.3];
// → IEEE 754 倍精度浮動小数点数として格納
// → ヒープオブジェクトとしてのオーバーヘッドがない

// PACKED_ELEMENTS: 最も汎用的だが最も遅い
const objectArray = [{ x: 1 }, "hello", true];
// → 各要素がヒープオブジェクトへのポインタ
// → GCがポインタを追跡する必要がある
```

### 8.2 配列の最適化ガイドライン

```javascript
// 良い: 型が統一された配列
const numbers = [1, 2, 3, 4, 5];  // PACKED_SMI_ELEMENTS
numbers.push(6);  // OK: SMIのまま

// 悪い: 型の混在で不可逆な遷移が発生
const bad = [1, 2, 3];           // PACKED_SMI_ELEMENTS
bad.push(4.5);                   // → PACKED_DOUBLE_ELEMENTS（不可逆）
bad.push("hello");               // → PACKED_ELEMENTS（不可逆）
// 元のPACKED_SMI_ELEMENTSには二度と戻らない

// 良い: 事前確保
const preallocated = new Array(1000);
// ただし HOLEY_SMI_ELEMENTS になる点に注意

// より良い: fill で初期化
const filled = new Array(1000).fill(0);
// PACKED_SMI_ELEMENTS（穴なし）

// 悪い: 穴あき配列
const holey = [1, , 3];  // HOLEY_SMI_ELEMENTS
// インデックス1のアクセス時にプロトタイプチェーンの検索が必要
// → PACKED に比べて遅い
```

---

## 9. パフォーマンス最適化の実践

### 9.1 型安定性の確保

V8が最も効率的に動作するのは、変数や関数の引数の型が安定している場合である。

```javascript
// アンチパターン1: 同じ変数に異なる型を代入
function unstable() {
  let value = 42;       // SMI
  value = 3.14;         // → Double
  value = "hello";      // → String
  value = { x: 1 };    // → Object
  return value;
}
// → TurboFanが型特殊化できない → 最適化が困難

// 推奨パターン: 変数の型を一貫させる
function stable() {
  const intValue = 42;
  const floatValue = 3.14;
  const strValue = "hello";
  const objValue = { x: 1 };
  return objValue;
}
```

### 9.2 オブジェクト初期化のベストプラクティス

```javascript
// アンチパターン2: 条件付きプロパティ追加
function createUser(name, email, isAdmin) {
  const user = { name, email };
  if (isAdmin) {
    user.role = "admin";        // Hidden Classが分岐
    user.permissions = ["all"];  // さらに分岐
  }
  return user;
}
// → isAdmin=true と isAdmin=false で異なるHidden Class
// → この関数を経由するオブジェクトのICがpolymorphicに

// 推奨パターン: 全プロパティを初期化
function createUserOptimized(name, email, isAdmin) {
  return {
    name,
    email,
    role: isAdmin ? "admin" : null,
    permissions: isAdmin ? ["all"] : null,
  };
}
// → 全オブジェクトが同じHidden Class → ICがmonomorphic

// 推奨パターン: クラスを使用
class User {
  constructor(name, email, isAdmin) {
    this.name = name;
    this.email = email;
    this.role = isAdmin ? "admin" : null;
    this.permissions = isAdmin ? ["all"] : null;
  }
}
// → コンストラクタで全プロパティ初期化 → 安定したHidden Class
```

### 9.3 関数のmonomorphic性を保つ

```javascript
// アンチパターン: 多様な形状のオブジェクトを同じ関数に渡す
function processItem(item) {
  return item.name + ": " + item.value;
}

processItem({ name: "A", value: 1 });
processItem({ name: "B", value: 2, extra: true });       // 別のHidden Class
processItem({ name: "C", value: 3, x: 1, y: 2 });       // また別のHidden Class
processItem({ value: 4, name: "D" });                     // 順序違いで別のHidden Class
// → ICがmegamorphicに → パフォーマンス低下

// 推奨パターン: 統一された形状を使用
class Item {
  constructor(name, value) {
    this.name = name;
    this.value = value;
  }
}

processItem(new Item("A", 1));
processItem(new Item("B", 2));
processItem(new Item("C", 3));
// → 全て同じHidden Class → ICがmonomorphic → 最速
```
