# 計算可能性

> 「全ての問題がコンピュータで解けるわけではない」——この事実はCS最大の発見の一つであり、停止問題の決定不能性がその象徴である。

## この章で学ぶこと

- [ ] チューリングマシンの概念と各種変種を説明できる
- [ ] 停止問題が決定不能であることを理解し、証明を再現できる
- [ ] 決定可能と決定不能の境界を理解する
- [ ] 帰着（Reduction）の概念を使って決定不能性を証明できる
- [ ] チャーチ=チューリングの提唱の意味と限界を理解する
- [ ] 計算可能性理論の実務への影響を説明できる
- [ ] 再帰定理とライスの定理を理解する
- [ ] 計算可能性の歴史的背景と動機を把握する

---

## 1. 歴史的背景と動機

### 1.1 ヒルベルトの計画と決定問題

```
ヒルベルトの計画（1900年〜1930年代）:

  ダヴィッド・ヒルベルトの3つの問い:
  1. 完全性: 数学の全ての真なる命題は証明可能か？
  2. 無矛盾性: 数学は矛盾を含まないか？
  3. 決定可能性: 任意の数学的命題の真偽を機械的に判定できるか？（Entscheidungsproblem）

  ゲーデルの不完全性定理（1931年）:
  - 第一不完全性定理: 十分に強い無矛盾な形式体系には、
    証明も反証もできない命題が存在する
  - 第二不完全性定理: 十分に強い無矛盾な形式体系は、
    自身の無矛盾性を証明できない

  チューリングの回答（1936年）:
  - 決定問題に対する否定的回答
  - 「計算」の概念を厳密に定義するためにチューリングマシンを考案
  - 停止問題の決定不能性を証明

  同時期の貢献:
  - アロンゾ・チャーチ: λ計算（1936年）
  - エミール・ポスト: ポスト生成系（1936年）
  - スティーヴン・クリーネ: 再帰関数（1936年）
  → 全て同等の計算能力を持つことが後に証明された
```

### 1.2 「計算」とは何か

```
「計算可能」の直感的理解:

  日常の「計算」:
  - 足し算、掛け算 → 明らかに計算可能
  - ソート → 明らかに計算可能
  - 素数判定 → 計算可能

  微妙な「計算」:
  - 「この数学の命題は真か？」 → 一般には計算不可能
  - 「このプログラムは停止するか？」 → 計算不可能
  - 「この暗号は安全か？」 → 一般には計算不可能

  計算可能性理論の目的:
  → 「原理的に解ける問題」と「原理的に解けない問題」の境界を見極める

  実務への影響:
  - 解けない問題を解こうとする無駄を避ける
  - 近似解や制限付き解法へのアプローチを導く
  - ソフトウェア検証の限界を理解する
```

---

## 2. チューリングマシン

### 2.1 形式的定義

```
チューリングマシン: 計算の理論的モデル

  構成:
  ┌──────────────────────────────────────────────┐
  │ ... │ B │ 1 │ 0 │ 1 │ 1 │ B │ B │ ...      │ ← 無限テープ
  └──────────────────────────────────────────────┘
                    ↑
                 ヘッド（読み書き）
                 ┌─────┐
                 │ 状態q│ ← 有限制御
                 └─────┘

  形式的定義 M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject):

  Q: 状態の有限集合
  Σ: 入力アルファベット（空白記号Bを含まない）
  Γ: テープアルファベット（Σ ⊂ Γ、B ∈ Γ）
  δ: Q × Γ → Q × Γ × {L, R}  遷移関数
  q₀: 開始状態（q₀ ∈ Q）
  q_accept: 受理状態（q_accept ∈ Q）
  q_reject: 拒否状態（q_reject ∈ Q、q_accept ≠ q_reject）

  動作: (現在の状態, 読んだ記号) → (書く記号, ヘッド移動, 次の状態)

  計算の過程:
  1. 入力文字列がテープに書かれる
  2. ヘッドは最左の記号の上に位置する
  3. 遷移関数に従って動作を繰り返す
  4. 受理状態に到達 → 入力を「受理」
  5. 拒否状態に到達 → 入力を「拒否」
  6. 永遠に停止しない場合もある → 「ループ」
```

### 2.2 具体例: 回文判定チューリングマシン

```
問題: 入力文字列 w ∈ {0, 1}* が回文かどうかを判定する

アルゴリズム:
1. 最初の文字を読んで記憶し、Xで上書き
2. テープ末尾に移動
3. 末尾の文字が記憶した文字と一致するか確認
4. 一致すればXで上書きし、テープ先頭に戻る
5. 全ての文字がXになれば受理

状態:
  q₀: 開始状態
  q₁: 最初の文字が0だった場合、右端へ移動中
  q₂: 最初の文字が1だった場合、右端へ移動中
  q₃: 右端で0を確認後、左端へ移動中
  q₄: 右端で1を確認後、左端へ移動中
  q₅: 左端に戻る途中
  q_accept: 受理
  q_reject: 拒否

遷移表の一部:
  δ(q₀, 0) = (q₁, X, R)    -- 最初が0、記憶してXで上書き
  δ(q₀, 1) = (q₂, X, R)    -- 最初が1、記憶してXで上書き
  δ(q₀, X) = (q₀, X, R)    -- Xをスキップ
  δ(q₀, B) = (q_accept, B, R)  -- 全てXになった、受理

  δ(q₁, 0) = (q₁, 0, R)    -- 右端へ移動中
  δ(q₁, 1) = (q₁, 1, R)
  δ(q₁, X) = (q₁, X, R)
  δ(q₁, B) = (q₃, B, L)    -- 右端に到達、確認へ

  δ(q₃, X) = (q₃, X, L)    -- Xをスキップ
  δ(q₃, 0) = (q₅, X, L)    -- 0が一致、Xで上書き
  δ(q₃, 1) = (q_reject, 1, R)  -- 不一致、拒否

実行例: 入力 "0110"
  ステップ1: [0]110B → X[1]10B  (q₀ → q₁, 0を記憶)
  ステップ2: X[1]10B → X1[1]0B → X11[0]B → X110[B]
  ステップ3: X110[B] → X11[0]B  (q₁ → q₃, 右端到達)
  ステップ4: X11[0]B → X1[1]XB  (0一致、Xで上書き)
  ステップ5: ... (左端に戻り、繰り返す)
  最終的に受理
```

### 2.3 チューリングマシンのシミュレーション（Python）

```python
class TuringMachine:
    """チューリングマシンのシミュレータ"""

    def __init__(self, states, input_alphabet, tape_alphabet,
                 transition_function, start_state,
                 accept_state, reject_state):
        self.states = states
        self.input_alphabet = input_alphabet
        self.tape_alphabet = tape_alphabet
        self.transition = transition_function
        self.start_state = start_state
        self.accept_state = accept_state
        self.reject_state = reject_state

    def run(self, input_string, max_steps=10000):
        """入力文字列に対してTMを実行する"""
        # テープの初期化
        tape = list(input_string) + ['B']
        head = 0
        state = self.start_state
        steps = 0

        history = []

        while steps < max_steps:
            # 現在の設定を記録
            history.append({
                'step': steps,
                'state': state,
                'head': head,
                'tape': ''.join(tape)
            })

            # 受理・拒否の判定
            if state == self.accept_state:
                return 'accept', history
            if state == self.reject_state:
                return 'reject', history

            # テープの拡張（必要に応じて）
            if head < 0:
                tape.insert(0, 'B')
                head = 0
            if head >= len(tape):
                tape.append('B')

            # 遷移の実行
            current_symbol = tape[head]
            key = (state, current_symbol)

            if key not in self.transition:
                return 'reject', history

            new_state, write_symbol, direction = self.transition[key]
            tape[head] = write_symbol
            state = new_state
            head += 1 if direction == 'R' else -1
            steps += 1

        return 'loop', history  # max_steps超過


# 具体例: 2進数を1増やすチューリングマシン
def create_binary_increment_tm():
    """2進数のインクリメントを行うTM"""
    states = {'q0', 'q1', 'q2', 'q_accept'}
    input_alphabet = {'0', '1'}
    tape_alphabet = {'0', '1', 'B'}

    # q0: 右端に移動
    # q1: 右端から左へ、繰り上がり処理
    # q2: 繰り上がり完了、左端に戻る
    transition = {
        # 右端まで移動
        ('q0', '0'): ('q0', '0', 'R'),
        ('q0', '1'): ('q0', '1', 'R'),
        ('q0', 'B'): ('q1', 'B', 'L'),

        # インクリメント処理（右から左へ）
        ('q1', '1'): ('q1', '0', 'L'),  # 繰り上がり
        ('q1', '0'): ('q2', '1', 'L'),  # 繰り上がり停止
        ('q1', 'B'): ('q_accept', '1', 'R'),  # 先頭に繰り上がり

        # 完了、左端に戻る
        ('q2', '0'): ('q2', '0', 'L'),
        ('q2', '1'): ('q2', '1', 'L'),
        ('q2', 'B'): ('q_accept', 'B', 'R'),
    }

    return TuringMachine(
        states, input_alphabet, tape_alphabet,
        transition, 'q0', 'q_accept', 'q_reject'
    )


# 実行例
tm = create_binary_increment_tm()

test_cases = ['0', '1', '10', '11', '101', '111', '1111']
for tc in test_cases:
    result, history = tm.run(tc)
    final_tape = history[-1]['tape'].rstrip('B')
    print(f"  {tc} → {final_tape} ({result})")

# 出力:
#   0 → 1 (accept)
#   1 → 10 (accept)
#   10 → 11 (accept)
#   11 → 100 (accept)
#   101 → 110 (accept)
#   111 → 1000 (accept)
#   1111 → 10000 (accept)
```

### 2.4 チューリングマシンの変種

```
チューリングマシンには多くの変種があるが、全て計算能力は同等:

  1. 多テープチューリングマシン
     ┌───────────────────┐  テープ1（入力用）
     └───────────────────┘
              ↑
     ┌───────────────────┐  テープ2（作業用）
     └───────────────────┘
              ↑
     ┌───────────────────┐  テープ3（出力用）
     └───────────────────┘
              ↑
           ┌─────┐
           │ 状態q│
           └─────┘
     → 1テープで多テープをシミュレート可能（O(t²)ステップ）
     → プログラミングで言えば複数の変数を使うのと同等

  2. 非決定性チューリングマシン (NTM)
     δ: Q × Γ → P(Q × Γ × {L, R})
     → 遷移先が集合（複数の選択肢）
     → 「正しい選択」が存在すれば受理
     → 決定性TMでシミュレート可能（指数的な時間増加）
     → NP問題との関係が深い

  3. 列挙器（Enumerator）
     → 言語の全ての文字列を出力する
     → チューリング認識可能な言語と等価

  4. 万能チューリングマシン (UTM)
     → 任意のTMの記述と入力を受け取り、そのTMをシミュレートする
     → 現代のコンピュータの概念的な原型
     → プログラム内蔵方式の理論的基礎

  5. 制限付きチューリングマシン
     - 線形有界オートマトン (LBA): テープ長が入力長に比例
       → 文脈依存言語を認識
     - プッシュダウンオートマトン (PDA): スタックのみ使用可能
       → 文脈自由言語を認識
     - 有限オートマトン (FA): 内部記憶なし
       → 正規言語を認識
```

### 2.5 万能チューリングマシンの詳細

```python
class UniversalTuringMachine:
    """万能チューリングマシンの概念的実装"""

    def __init__(self):
        pass

    def encode_tm(self, tm):
        """チューリングマシンをエンコードする（ゲーデル数化）"""
        # 状態、アルファベット、遷移関数をシリアライズ
        encoding = {
            'states': list(tm.states),
            'input_alphabet': list(tm.input_alphabet),
            'tape_alphabet': list(tm.tape_alphabet),
            'transitions': {},
            'start_state': tm.start_state,
            'accept_state': tm.accept_state,
            'reject_state': tm.reject_state
        }

        for (state, symbol), (new_state, write, direction) in tm.transition.items():
            key = f"{state},{symbol}"
            encoding['transitions'][key] = {
                'new_state': new_state,
                'write': write,
                'direction': direction
            }

        return encoding

    def simulate(self, encoded_tm, input_string, max_steps=10000):
        """エンコードされたTMを入力文字列上でシミュレートする"""
        # デコード
        transitions = {}
        for key, value in encoded_tm['transitions'].items():
            state, symbol = key.split(',')
            transitions[(state, symbol)] = (
                value['new_state'],
                value['write'],
                value['direction']
            )

        # シミュレーション
        tape = list(input_string) + ['B']
        head = 0
        state = encoded_tm['start_state']

        for step in range(max_steps):
            if state == encoded_tm['accept_state']:
                return 'accept'
            if state == encoded_tm['reject_state']:
                return 'reject'

            if head < 0:
                tape.insert(0, 'B')
                head = 0
            if head >= len(tape):
                tape.append('B')

            key = (state, tape[head])
            if key not in transitions:
                return 'reject'

            new_state, write, direction = transitions[key]
            tape[head] = write
            state = new_state
            head += 1 if direction == 'R' else -1

        return 'timeout'  # ステップ数超過（停止しない可能性）


# 万能TMの意義:
# 1. 一つのマシンで全てのプログラムを実行できる
# 2. 現代のコンピュータ（ノイマン型）の理論的基礎
# 3. 「プログラムもデータとして扱える」という概念の原型
# 4. コンパイラ、インタプリタの理論的正当化
```

### 2.6 チューリング完全性

```
チューリング完全性: 計算システムがチューリングマシンと同等の計算能力を持つこと

  チューリング完全なシステムの例:
  ┌─────────────────────────────────────────────┐
  │ プログラミング言語                             │
  │  - Python, Java, C, C++, Rust, Go            │
  │  - JavaScript, TypeScript, Ruby, PHP         │
  │  - Haskell, OCaml, Lisp, Scheme              │
  │  - アセンブリ言語                              │
  └─────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │ 意外なチューリング完全システム                  │
  │  - CSS（アニメーション + 条件分岐の組合せ）       │
  │  - SQL（再帰CTEを使用）                        │
  │  - Excel（数式のみ）                           │
  │  - PowerPoint（アニメーション）                  │
  │  - マインクラフト（レッドストーン回路）            │
  │  - ライフゲーム（Conway's Game of Life）        │
  │  - LaTeX（マクロシステム）                      │
  │  - sed（ストリームエディタ）                     │
  │  - sendmail.cf（設定ファイル）                   │
  │  - TypeScript の型システム                     │
  └─────────────────────────────────────────────┘

  チューリング完全に必要な最小要件:
  1. 条件分岐（if-then-else）
  2. 無限の記憶領域（任意の量のデータ保存）
  3. 状態の読み書き
  4. 繰り返し（ループまたは再帰）

  チューリング完全でないシステムの例:
  - 正規表現（標準的な定義）
  - 有限オートマトン
  - 全域再帰関数（Total recursive functions）
  - LOOP言語（原始再帰関数のみ）
```

```python
# チューリング完全性の最小例: BFインタプリタ
# BFは8命令のみでチューリング完全

def bf_interpreter(code, input_data=""):
    """Brainf*ckインタプリタ — チューリング完全な最小言語"""
    tape = [0] * 30000
    ptr = 0
    pc = 0
    input_pos = 0
    output = []

    # ブラケットの対応表を構築
    brackets = {}
    stack = []
    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']':
            if stack:
                j = stack.pop()
                brackets[j] = i
                brackets[i] = j

    max_steps = 1000000
    steps = 0

    while pc < len(code) and steps < max_steps:
        cmd = code[pc]
        if cmd == '>':
            ptr = (ptr + 1) % 30000
        elif cmd == '<':
            ptr = (ptr - 1) % 30000
        elif cmd == '+':
            tape[ptr] = (tape[ptr] + 1) % 256
        elif cmd == '-':
            tape[ptr] = (tape[ptr] - 1) % 256
        elif cmd == '.':
            output.append(chr(tape[ptr]))
        elif cmd == ',':
            if input_pos < len(input_data):
                tape[ptr] = ord(input_data[input_pos])
                input_pos += 1
            else:
                tape[ptr] = 0
        elif cmd == '[':
            if tape[ptr] == 0:
                pc = brackets[pc]
        elif cmd == ']':
            if tape[ptr] != 0:
                pc = brackets[pc]

        pc += 1
        steps += 1

    return ''.join(output)


# Hello World in BF
hello_bf = (
    "++++++++[>++++[>++>+++>+++>+<<<<-]"
    ">+>+>->>+[<]<-]>>.>---.+++++++..+++."
    ">>.------------.<<+++++++++++++++.>."
    "+++.------.--------.>>+.>++."
)
print(bf_interpreter(hello_bf))  # "Hello World!\n"

# BFがチューリング完全である理由:
# - 無限（十分に大きい）テープ → 任意の記憶
# - ループ構造 ([...]) → 繰り返し
# - 条件分岐 ([]でゼロチェック) → 条件
# - データの読み書き (+, -, >, <) → 状態操作
```

---

## 3. チャーチ=チューリングの提唱

### 3.1 提唱の内容

```
チャーチ=チューリングの提唱（Church-Turing Thesis）:

  定義:
  「直感的に計算可能な関数は、チューリングマシンで計算可能な関数と
   正確に一致する」

  注意: これは数学的定理ではなく「提唱」（仮説）である
  → 反証は可能だが、これまで反例は見つかっていない

  等価であることが証明されたモデル:
  ┌─────────────────────────────────────────┐
  │                                         │
  │  チューリングマシン ←→ λ計算             │
  │       ↕                  ↕              │
  │  帰納的関数 ←───→ ポスト生成系           │
  │       ↕                  ↕              │
  │  レジスタマシン ←→ マルコフアルゴリズム   │
  │                                         │
  └─────────────────────────────────────────┘

  全て同じ計算能力を持つ → 計算可能性の「ロバスト性」
```

### 3.2 物理的チャーチ=チューリングの提唱

```
物理的チャーチ=チューリングの提唱:

  「物理的に実現可能な全ての計算デバイスは、
   チューリングマシンでシミュレート可能である」

  これに対する議論:

  賛成派の主張:
  - 量子コンピュータもチューリングマシンでシミュレート可能
    （効率は悪いが、計算能力は同等）
  - 今のところ反例となる物理デバイスは見つかっていない

  反対・疑問派の主張:
  - 超計算（Hypercomputation）の可能性
    → ゼノンマシン（無限のステップを有限時間で実行）
    → ブラックホールコンピュータ
    → アナログ計算の無限精度
  - しかし、これらは物理的に実現不可能と考えられている

  実務的な意味:
  - プログラミング言語の選択は「計算能力」に影響しない
  - 全てのプログラミング言語はチューリング完全なら等価
  - 言語の違いは「表現力」「効率」「安全性」のみ
```

### 3.3 強いチャーチ=チューリングの提唱

```
強いチャーチ=チューリングの提唱（Extended Church-Turing Thesis）:

  「効率的に計算可能な関数は、確率的チューリングマシンで
   多項式時間で計算可能な関数と一致する」

  量子コンピュータによる挑戦:
  - ショアのアルゴリズム: 素因数分解を多項式時間で解く
  - 古典的に多項式時間で解けるかは未解明
  → 強い提唱は量子コンピュータにより崩れる可能性がある

  しかし:
  - 量子コンピュータでもNP完全問題は効率的に解けないと予想
  - BQP ⊂ PSPACE は証明済み
  → 量子でも「計算可能性」の境界は変わらない
```

---

## 4. 停止問題

### 4.1 停止問題の形式的定義

```
停止問題（Halting Problem）:

  入力: チューリングマシンMの記述 ⟨M⟩ と入力文字列 w
  出力: Mがwに対して停止するか？

  形式的に:
  HALT = { ⟨M, w⟩ | Mは入力wで停止する }

  定理: HALTは決定不能（undecidable）である

  直感的理解:
  - 「このプログラムは停止するか？」に常に正しく答える
    プログラムは存在しない
  - 個別のプログラムについては判定できることもある
  - しかし「全ての」プログラムに対して判定する万能の方法はない
```

### 4.2 決定不能の証明（詳細版）

```
停止問題の決定不能性の証明（対角線論法による）:

  定理: HALT = { ⟨M, w⟩ | Mは入力wで停止する } は決定不能

  証明:
  仮定: 停止判定プログラム H が存在するとする
    H(⟨M⟩, w) = accept  (Mがwで停止する場合)
    H(⟨M⟩, w) = reject  (Mがwで停止しない場合)

  矛盾の構築:
  プログラム D を以下のように定義する:

    D(⟨M⟩):
      H(⟨M⟩, ⟨M⟩) を実行
      if H が accept を返す:
        while True: pass    # 無限ループ
      else:
        return              # 停止する

  D に自身の記述 ⟨D⟩ を入力する:

  場合1: D(⟨D⟩) が停止すると仮定
    → H(⟨D⟩, ⟨D⟩) = accept
    → Dの定義により、Dは無限ループに入る
    → D(⟨D⟩) は停止しない
    → 矛盾！

  場合2: D(⟨D⟩) が停止しないと仮定
    → H(⟨D⟩, ⟨D⟩) = reject
    → Dの定義により、Dはreturnで停止する
    → D(⟨D⟩) は停止する
    → 矛盾！

  両方の場合で矛盾が生じる
  → 仮定「Hが存在する」が誤り
  → HALTは決定不能 ∎

  この証明のポイント:
  1. 自己参照（D が自分自身を入力に取る）
  2. 対角線論法（カントールの対角線論法と同型）
  3. パラドックス的構造（「嘘つきのパラドックス」に類似）
```

```python
# 停止問題の不可能性を示すPython風の疑似コード

def hypothetical_halts(program_source, input_data):
    """仮に存在するとする停止判定関数（実際には存在しない）"""
    # program_source をinput_data で実行した場合に
    # 停止するかどうかを判定する
    # ... 魔法の実装 ...
    pass

def diagonal(program_source):
    """矛盾を引き起こすプログラム"""
    if hypothetical_halts(program_source, program_source):
        # 停止すると判定されたら、無限ループ
        while True:
            pass
    else:
        # 停止しないと判定されたら、停止
        return

# diagonal のソースコードを取得
diagonal_source = inspect.getsource(diagonal)

# diagonal(diagonal_source) を実行するとどうなるか？
#
# 場合1: hypothetical_halts(diagonal_source, diagonal_source) == True
#   → diagonal は停止すると判定された
#   → しかしdiagonalは無限ループに入る
#   → 停止しない → 矛盾
#
# 場合2: hypothetical_halts(diagonal_source, diagonal_source) == False
#   → diagonal は停止しないと判定された
#   → しかしdiagonalはreturnで停止する
#   → 停止する → 矛盾
#
# どちらの場合も矛盾 → hypothetical_halts は存在し得ない
```

### 4.3 カントールの対角線論法との関係

```
カントールの対角線論法（1891年）と停止問題の証明の構造的類似:

  カントール: 実数の集合は自然数の集合より「大きい」

  自然数と実数の対応表（仮定）:
  n | 実数の小数展開
  ──┼─────────────
  1 | 0.5 2 3 1 7 ...
  2 | 0.3 1 4 1 5 ...
  3 | 0.7 7 7 8 2 ...
  4 | 0.1 4 9 3 6 ...
  5 | 0.8 2 0 5 1 ...
      ↓ ↓ ↓ ↓ ↓
  対角線: 0.5 1 7 3 1 ...
  新しい数: 0.6 2 8 4 2 ... （各桁を+1）
  → この数は表のどの行とも異なる → 矛盾

  停止問題の証明:

  プログラムとその入力の対応表（仮定）:
        | P₁    P₂    P₃    P₄   ...
  ──────┼────────────────────────
  P₁    | 停止  停止  ループ 停止 ...
  P₂    | ループ 停止  停止  ループ ...
  P₃    | 停止  ループ ループ 停止 ...
  P₄    | 停止  停止  停止  ループ ...

  D(Pᵢ) = 対角線の反転:
  D(P₁): ループ（停止の反転）
  D(P₂): ループ（停止の反転）
  D(P₃): 停止（ループの反転）
  D(P₄): 停止（ループの反転）
  → Dはどの行とも異なる → 矛盾

  共通の構造:
  1. 全てを列挙できると仮定
  2. 対角線を取る
  3. 対角線を反転させた新しいものを構築
  4. それが列挙に含まれないことを示す → 矛盾
```

### 4.4 停止問題の実務的帰結

```
停止問題が決定不能であることの実務的な意味:

  1. 完璧なバグ検出器は存在しない
     ┌──────────────────────────────────────────┐
     │ 「全てのバグを検出する静的解析ツール」は     │
     │ 原理的に不可能                              │
     │                                            │
     │ 理由: 「このプログラムにバグがあるか？」は   │
     │ 停止問題に帰着できる                        │
     │                                            │
     │ 実務的対処:                                 │
     │ - 保守的な近似（誤検出あり、見逃しなし）     │
     │ - 楽観的な近似（誤検出なし、見逃しあり）     │
     │ - テストによる検証（有限のケースのみ）       │
     └──────────────────────────────────────────┘

  2. 完璧なウイルス検出器は存在しない
     - マルウェアの全パターンを検出することは不可能
     - ヒューリスティックとシグネチャベースの組み合わせ
     - 行動ベースの検出で補完

  3. 完璧なデッドコード除去は不可能
     - 「この分岐が実行されることはあるか？」は一般に決定不能
     - コンパイラの最適化には限界がある

  4. 完璧な型チェッカーは存在しない
     - 全ての型エラーを検出しつつ、正しいプログラムを
       全て受理する型システムは不可能
     - 実用的な型システムは保守的（型安全だが制限的）

  5. 完璧なリソースリーク検出は不可能
     - メモリリーク、ファイルハンドルリーク等の完全検出は不可能
     - 所有権システム（Rust）は保守的なアプローチの好例
```

```python
# 停止問題の帰結を示す具体例

# 例1: デッドコード検出の限界
def is_dead_code(function_body, line_number):
    """
    指定行が到達不能かを完全に判定することは不可能

    理由: 以下のようなコードを考える
    """
    pass

def example_with_undecidable_reachability():
    """到達可能性が決定不能な例"""
    x = compute_goldbach()  # ゴールドバッハ予想が真なら停止
    if x:
        print("この行は到達可能か？")  # 未解決の数学的問題に依存
        # → デッドコードかどうかを判定することは
        #   ゴールドバッハ予想を証明/反証することに等しい

def compute_goldbach():
    """ゴールドバッハ予想の反例を探す"""
    n = 4
    while True:
        if n % 2 == 0 and not is_sum_of_two_primes(n):
            return n  # 反例が見つかれば停止
        n += 2
    # 反例がなければ永遠に停止しない

def is_sum_of_two_primes(n):
    """nが2つの素数の和で表せるかチェック"""
    for i in range(2, n):
        if is_prime(i) and is_prime(n - i):
            return True
    return False

def is_prime(n):
    """素数判定"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


# 例2: 静的解析の保守的近似
class ConservativeAnalyzer:
    """保守的な静的解析器の例"""

    def check_division_by_zero(self, code_ast):
        """
        ゼロ除算の可能性を保守的にチェック

        完璧な検出は不可能なので:
        - 偽陽性（false positive）は許容する
        - 偽陰性（false negative）は許容しない
        """
        warnings = []

        for node in self.walk(code_ast):
            if self.is_division(node):
                divisor = self.get_divisor(node)
                if not self.can_prove_nonzero(divisor):
                    # ゼロでないことを証明できない → 警告
                    # （実際にはゼロにならないかもしれないが）
                    warnings.append(f"Line {node.line}: 潜在的なゼロ除算")

        return warnings

    def can_prove_nonzero(self, expr):
        """式がゼロでないことを証明できるか（保守的）"""
        # 定数の場合
        if self.is_constant(expr) and self.eval_constant(expr) != 0:
            return True
        # abs(x) > 0 のようなパターン
        if self.matches_positive_pattern(expr):
            return True
        # それ以外は証明できない（保守的にFalse）
        return False
```

---

## 5. 決定可能と決定不能

### 5.1 問題の分類体系

```
言語（問題）の階層:

  ┌─────────────────────────────────────────────────┐
  │               全ての言語                         │
  │  ┌──────────────────────────────────────────┐   │
  │  │         チューリング認識可能               │   │
  │  │  ┌──────────────────────────────────┐    │   │
  │  │  │         決定可能                  │    │   │
  │  │  │  ┌────────────────────────┐      │    │   │
  │  │  │  │    文脈自由             │      │    │   │
  │  │  │  │  ┌──────────────┐     │      │    │   │
  │  │  │  │  │   正規        │     │      │    │   │
  │  │  │  │  └──────────────┘     │      │    │   │
  │  │  │  └────────────────────────┘      │    │   │
  │  │  └──────────────────────────────────┘    │   │
  │  │     半決定可能（RE）                       │   │
  │  └──────────────────────────────────────────┘   │
  │    チューリング認識不能も含む（co-RE の補集合等）  │
  └─────────────────────────────────────────────────┘

  各クラスの特徴:

  正規言語 (REG):
  - 有限オートマトンで認識可能
  - 正規表現で記述可能
  - 例: メールアドレスの形式チェック、識別子のパターン

  文脈自由言語 (CFL):
  - プッシュダウンオートマトンで認識可能
  - 文脈自由文法で記述可能
  - 例: プログラミング言語の構文、括弧の対応

  決定可能言語 (R):
  - 停止するチューリングマシンで認識可能
  - 必ず有限時間で答えが出る
  - 例: 素数判定、グラフの連結性判定

  チューリング認識可能言語 (RE):
  - チューリングマシンで認識可能（ただし停止しない場合あり）
  - 属する場合は有限時間で検出可能
  - 属さない場合は永遠に待つかもしれない
  - 例: 停止問題（停止する場合は検出可能）
```

### 5.2 決定可能な問題の具体例

```python
# 決定可能な問題の例

# 1. 有限オートマトンの受理問題
# A_DFA = { ⟨B, w⟩ | DFA Bが文字列wを受理する }
class DFA:
    """決定性有限オートマトン"""
    def __init__(self, states, alphabet, transitions, start, accepts):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # (state, symbol) -> state
        self.start = start
        self.accepts = accepts

    def accepts_string(self, w):
        """O(|w|) で判定可能 — 常に停止する"""
        state = self.start
        for symbol in w:
            state = self.transitions.get((state, symbol))
            if state is None:
                return False
        return state in self.accepts


# 2. 文脈自由文法の所属判定
# A_CFG = { ⟨G, w⟩ | CFG Gが文字列wを生成する }
def cyk_algorithm(grammar, string):
    """CYKアルゴリズム — O(n³|G|) で判定可能"""
    n = len(string)
    if n == 0:
        return grammar.start in grammar.nullable

    # 初期化
    table = [[set() for _ in range(n)] for _ in range(n)]

    # 長さ1の部分文字列
    for i in range(n):
        for lhs, rhs_list in grammar.rules.items():
            for rhs in rhs_list:
                if len(rhs) == 1 and rhs[0] == string[i]:
                    table[i][i].add(lhs)

    # 長さ2以上の部分文字列
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for lhs, rhs_list in grammar.rules.items():
                    for rhs in rhs_list:
                        if len(rhs) == 2:
                            B, C = rhs
                            if B in table[i][k] and C in table[k+1][j]:
                                table[i][j].add(lhs)

    return grammar.start in table[0][n-1]


# 3. 有限オートマトンの等価性判定
# EQ_DFA = { ⟨A, B⟩ | DFA AとBが同じ言語を認識する }
def dfa_equivalence(dfa_a, dfa_b):
    """
    2つのDFAが等価かどうかを判定
    積オートマトンを構成し、対称差が空かチェック
    常に停止する — 決定可能
    """
    # 対称差 = (L(A) - L(B)) ∪ (L(B) - L(A))
    # 対称差が空 ↔ L(A) = L(B)
    symmetric_diff = construct_symmetric_difference(dfa_a, dfa_b)
    return is_empty(symmetric_diff)

def is_empty(dfa):
    """DFAの言語が空かどうかをBFSで判定"""
    visited = set()
    queue = [dfa.start]
    visited.add(dfa.start)

    while queue:
        state = queue.pop(0)
        if state in dfa.accepts:
            return False  # 受理状態に到達可能 → 空でない
        for symbol in dfa.alphabet:
            next_state = dfa.transitions.get((state, symbol))
            if next_state and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    return True  # 受理状態に到達不能 → 空


# 4. グラフの連結性判定
def is_connected(graph):
    """グラフが連結かどうかを判定 — O(V + E)"""
    if not graph:
        return True

    visited = set()
    start = next(iter(graph))

    def dfs(node):
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return len(visited) == len(graph)
```

### 5.3 決定不能な問題の具体例

```
決定不能な問題（代表的なもの）:

  1. 停止問題 (HALT)
     入力: プログラムPと入力x
     問い: Pはxで停止するか？
     → 決定不能

  2. 全域性問題 (TOTAL)
     入力: プログラムP
     問い: Pは全ての入力で停止するか？
     → 決定不能（停止問題よりさらに難しい）

  3. 等価性問題 (EQ_TM)
     入力: チューリングマシンM₁とM₂
     問い: L(M₁) = L(M₂) か？
     → 決定不能

  4. ポスト対応問題 (PCP)
     入力: ドミノの集合 {(u₁,v₁), (u₂,v₂), ..., (uₙ,vₙ)}
     問い: i₁, i₂, ..., iₖ が存在して
           u_{i₁}u_{i₂}...u_{iₖ} = v_{i₁}v_{i₂}...v_{iₖ} か？
     → 決定不能

  5. タイリング問題
     入力: タイルの集合
     問い: 無限平面をタイルで隙間なく敷き詰められるか？
     → 決定不能

  6. ディオファントス方程式（ヒルベルトの第10問題）
     入力: 整数係数の多変数多項式
     問い: 整数解は存在するか？
     → 決定不能（マチャセビッチの定理、1970年）

  7. モータルマトリクス問題
     入力: 行列の集合
     問い: それらの積で零行列になる組み合わせがあるか？
     → 決定不能
```

```python
# ポスト対応問題（PCP）の具体例

def pcp_brute_force(dominoes, max_length=10):
    """
    PCPを力任せに探索する（一般には停止しない可能性がある）

    ドミノの例:
    [(ab, b), (b, ab), (a, aa)]

    解: 1,2,1,3 → ababba = ababba ✓
    """
    from itertools import product

    for length in range(1, max_length + 1):
        for combo in product(range(len(dominoes)), repeat=length):
            top = ''.join(dominoes[i][0] for i in combo)
            bottom = ''.join(dominoes[i][1] for i in combo)
            if top == bottom:
                return list(combo), top

    return None  # max_length以内では解が見つからない

# 実行例
dominoes = [('ab', 'b'), ('b', 'ab'), ('a', 'aa')]
result = pcp_brute_force(dominoes, max_length=8)
if result:
    indices, matched = result
    print(f"解: {indices}")
    print(f"一致文字列: {matched}")

# PCPが決定不能である意味:
# - 解があるかどうかを判定するアルゴリズムは存在しない
# - 解がある場合は力任せ探索で見つかる（半決定可能）
# - 解がない場合は永遠に探索が終わらない
```

### 5.4 半決定可能性

```
半決定可能（Semi-decidable / Recursively Enumerable）:

  定義: 言語Lが半決定可能 ⟺
    w ∈ L の場合は有限時間で「はい」と答える
    w ∉ L の場合は永遠に答えないかもしれない

  例:
  ┌────────────────────────────────────────────────────┐
  │ 停止問題は半決定可能                                 │
  │                                                    │
  │ 方法: プログラムPを入力xで実際に実行する             │
  │ - 停止した → 「はい、停止します」                    │
  │ - 停止しない → 永遠に待ち続ける（答えられない）       │
  └────────────────────────────────────────────────────┘

  重要な性質:
  - Lが決定可能 ⟺ LとL̄（補集合）がともに半決定可能
  - Lが半決定可能でL̄が半決定可能でない → Lは決定不能
  - 停止問題の補集合（非停止問題）は半決定可能でない

  証明: Lが決定可能 ⟺ LとL̄がともに半決定可能

  (→) Lが決定可能ならば、Lを判定するTM Mが存在。
      Mは常に停止するので、LもL̄も半決定可能。

  (←) LとL̄がともに半決定可能ならば、
      Lの認識器M₁とL̄の認識器M₂が存在。
      入力wに対してM₁とM₂を並列実行すれば、
      必ずどちらかが停止する。
      → M₁が停止すれば受理、M₂が停止すれば拒否
      → 常に停止する → Lは決定可能 ∎
```

```python
# 半決定可能性の実装例

import threading
import time

class SemiDecider:
    """半決定的判定器"""

    def check_halts(self, program, input_data, timeout=None):
        """
        プログラムが停止するかをチェック（半決定的）

        - 停止する場合: True を返す（有限時間で）
        - 停止しない場合: timeout まで待って None を返す
          （timeout なしなら永遠に待つ）
        """
        result = [None]
        error = [None]

        def run_program():
            try:
                exec(program, {'input': input_data})
                result[0] = True
            except Exception as e:
                error[0] = e
                result[0] = True  # 例外で終了 = 停止した

        thread = threading.Thread(target=run_program)
        thread.daemon = True
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return None  # タイムアウト（停止するかどうか不明）
        return result[0]


# 並列実行による決定可能化の例
def parallel_decide(recognizer_l, recognizer_l_complement, input_data):
    """
    LとL̄の両方の認識器がある場合、決定器を構築できる

    前提: LとL̄がともに半決定可能
    結論: Lは決定可能
    """
    result = [None]

    def run_l():
        if recognizer_l(input_data):
            result[0] = True  # w ∈ L

    def run_l_complement():
        if recognizer_l_complement(input_data):
            result[0] = False  # w ∉ L

    t1 = threading.Thread(target=run_l)
    t2 = threading.Thread(target=run_l_complement)
    t1.daemon = True
    t2.daemon = True
    t1.start()
    t2.start()

    # どちらかが停止するまで待つ（必ずどちらかは停止する）
    while result[0] is None:
        time.sleep(0.001)

    return result[0]
```

---

## 6. 帰着（Reduction）

### 6.1 帰着の概念

```
帰着（Reduction）: 問題Aの解法を問題Bの解法に変換すること

  A ≤ B: 「AはBに帰着可能」

  意味:
  - Bが解ければAも解ける
  - Aが解けなければBも解けない（対偶）

  図式:

  入力 w → [変換 f] → f(w) → [Bの判定器] → 答え

  w ∈ A ⟺ f(w) ∈ B

  帰着の種類:

  1. 写像帰着（Mapping Reduction / Many-One Reduction）
     - 計算可能な関数fが存在して w ∈ A ⟺ f(w) ∈ B
     - A ≤ₘ B と書く

  2. チューリング帰着（Turing Reduction）
     - Bを「オラクル」（神託）として使い、Aを決定する
     - A ≤ᵀ B と書く
     - 写像帰着より強力

  帰着の使い方:

  既知: HALTは決定不能

  新問題Xが決定不能であることを示すには:
  1. HALT ≤ₘ X を示す
  2. つまり停止問題をXに変換できることを示す
  3. Xが決定可能なら停止問題も決定可能になる → 矛盾
  → Xは決定不能 ∎
```

### 6.2 帰着の具体例

```python
# 帰着の具体例: 停止問題 → 全域性問題

# 全域性問題: プログラムPが全ての入力で停止するか？
# TOTAL = { ⟨P⟩ | Pは全ての入力で停止する }

# 定理: TOTAL は決定不能

# 証明: HALT ≤ₘ TOTAL̄ を示す（TOTALの補集合に帰着）

def reduce_halt_to_total(program_p, input_w):
    """
    停止問題のインスタンス (P, w) を
    全域性問題のインスタンス Q に変換する
    """
    # 新しいプログラムQを構築:
    # Q(x) = P(w) を実行して停止したら停止、
    #         xは無視する

    q_source = f"""
def Q(x):
    # xは無視して、P(w)を実行する
    {program_p}({input_w})
    return  # P(w)が停止すればQも停止
"""

    # P(w)が停止する → Q(x)は全ての入力xで停止 → Q ∈ TOTAL
    # P(w)が停止しない → Q(x)はどの入力でも停止しない → Q ∉ TOTAL

    # つまり: (P, w) ∈ HALT ⟺ Q ∈ TOTAL

    return q_source


# 帰着の例2: 停止問題 → 等価性問題

# 等価性問題: 2つのTMが同じ言語を認識するか？
# EQ_TM = { ⟨M₁, M₂⟩ | L(M₁) = L(M₂) }

def reduce_halt_to_eq(program_p, input_w):
    """
    停止問題のインスタンス (P, w) を
    等価性問題のインスタンス (M₁, M₂) に変換する
    """
    # M₁: 全ての入力を拒否する（空言語）
    m1 = lambda x: False

    # M₂: P(w)を実行し、停止したら全ての入力を拒否
    #      停止しなければループ
    def m2(x):
        # P(w) を実行
        exec(program_p, {'input': input_w})
        # P(w)が停止した場合のみここに到達
        return False  # 拒否

    # P(w)が停止する → L(M₂) = ∅ = L(M₁) → (M₁,M₂) ∈ EQ_TM
    # P(w)が停止しない → M₂は全入力でループ → L(M₂)は認識不能
    #                   → L(M₁) ≠ L(M₂) → (M₁,M₂) ∉ EQ_TM

    return m1, m2
```

### 6.3 帰着の階層

```
決定不能問題の難しさの階層（算術的階層）:

  Σ₀⁰ = Π₀⁰ = 決定可能問題

  Σ₁⁰: 半決定可能（RE）
    例: HALT（停止問題）
    → 「停止する」場合は検出可能

  Π₁⁰: co-RE（REの補集合）
    例: HALT̄（非停止問題）
    → 「停止しない」場合を検出可能

  Σ₂⁰:
    例: TOTAL（全域性問題）
    → 「全ての入力で停止するか」

  Π₂⁰:
    例: TOTAL̄
    → 「ある入力で停止しないか」

  Σ₃⁰, Π₃⁰, ... と無限に続く

  階層の図示:

  難しさ →
  ──────────────────────────────────────→
  Σ₀⁰    Σ₁⁰    Σ₂⁰    Σ₃⁰    ...
  (決定可能) (RE)  (TOTAL等)

  各レベルで「本質的に難しい」問題が存在
  → 帰着で互いの難しさを比較できる
```

---

## 7. ライスの定理

### 7.1 ライスの定理の内容

```
ライスの定理（Rice's Theorem）:

  定理: チューリングマシンが計算する関数に関する
  非自明な性質は、全て決定不能である

  形式的に:
  Pをチューリング認識可能な言語の性質とする
  （Pは言語の集合の部分集合）

  Pが非自明（P ≠ ∅ かつ P ≠ 全ての言語）ならば、
  { ⟨M⟩ | L(M) ∈ P } は決定不能

  直感的に:
  - 「このプログラムは何を計算するか？」に関する
    非自明な質問には、一般的なアルゴリズムでは答えられない

  具体例（全て決定不能）:

  1. 「TMが空言語を認識するか？」
     → P = {∅}（空言語のみ）: 非自明

  2. 「TMが正規言語を認識するか？」
     → P = 正規言語の集合: 非自明

  3. 「TMが少なくとも一つの文字列を受理するか？」
     → P = {L | L ≠ ∅}: 非自明

  4. 「TMが全ての文字列を受理するか？」
     → P = {Σ*}: 非自明

  5. 「TMが特定の文字列 w を受理するか？」
     → P = {L | w ∈ L}: 非自明

  ライスの定理が適用されない例:

  1. 「TMが100以下の状態を持つか？」
     → これはTMの構造に関する性質（計算する関数ではない）
     → 決定可能（TMの記述を調べれば分かる）

  2. 「TMが入力なしで5ステップ以内に停止するか？」
     → シミュレーションで判定可能
     → しかし「任意のステップ数で停止するか」は決定不能
```

### 7.2 ライスの定理の証明

```
ライスの定理の証明:

  前提:
  - Pを非自明な言語の性質とする
  - P ≠ ∅ かつ P ≠ 全ての言語
  - 空言語 ∅ ∉ P と仮定する（∅ ∈ P の場合はPの補集合で議論）

  証明:

  Pが非自明なので、L₀ ∈ P となる言語L₀が存在する
  L₀を認識するTM M₀ が存在する

  停止問題をライスに帰着する:

  入力 ⟨M, w⟩（Mがwで停止するか？）に対して、
  新しいTM M'を構築:

  M'(x):
    1. Mをwで実行する
    2. もしMが停止したら、M₀をxで実行し結果を返す
    3. もしMが停止しなければ、M'も停止しない

  分析:
  - Mがwで停止する → M'はM₀と同じ動作 → L(M') = L₀ ∈ P
  - Mがwで停止しない → M'は何も受理しない → L(M') = ∅ ∉ P

  つまり: (M, w) ∈ HALT ⟺ L(M') ∈ P

  { ⟨M'⟩ | L(M') ∈ P } が決定可能なら、
  HALT も決定可能になる → 矛盾

  → { ⟨M⟩ | L(M) ∈ P } は決定不能 ∎
```

```python
# ライスの定理の実務的意味

class ProgramAnalyzer:
    """
    ライスの定理により、以下の分析は全て一般には不可能
    """

    def does_program_always_return_positive(self, program):
        """プログラムが常に正の値を返すか？ → 決定不能"""
        raise UndecidableError("ライスの定理により不可能")

    def does_program_use_network(self, program):
        """
        プログラムがネットワークにアクセスするか？

        注意: これは「構文的」にチェックすれば決定可能
        （importやsocket呼び出しの有無を確認）

        しかし「意味的」にチェックは決定不能
        （動的生成されたコードがネットワークアクセスする場合）
        """
        # 構文的チェック（保守的な近似）は可能
        return 'socket' in program or 'requests' in program

    def is_program_equivalent_to(self, program_a, program_b):
        """2つのプログラムが同じ関数を計算するか？ → 決定不能"""
        raise UndecidableError("ライスの定理により不可能")

    def does_program_terminate_for_all_inputs(self, program):
        """プログラムが全入力で停止するか？ → 決定不能"""
        raise UndecidableError("ライスの定理により不可能")


class UndecidableError(Exception):
    """決定不能な問題に対するエラー"""
    pass


# 実務での対処法:
#
# 1. 保守的近似（Sound but Incomplete）
#    - 「安全でない可能性がある」 → 警告を出す
#    - 偽陽性はあるが、見逃しはない
#    例: Rust の借用チェッカー
#
# 2. 楽観的近似（Complete but Unsound）
#    - 「おそらく安全」 → 通す
#    - 見逃しはあるが、偽陽性はない
#    例: 動的型付き言語の実行時チェック
#
# 3. 制限付き正確解
#    - 問題の範囲を制限して、その範囲では正確に判定
#    例: 有界モデル検査（bounded model checking）
#
# 4. 対話的検証
#    - ユーザーにヒントや証明を提供してもらう
#    例: 対話型定理証明器（Coq, Isabelle, Lean）
```

---

## 8. 再帰定理

### 8.1 再帰定理の内容

```
再帰定理（Recursion Theorem / Fixed Point Theorem）:

  定理: 任意の計算可能な変換 t に対して、
  チューリングマシン M が存在し、
  t(⟨M⟩) と M は同じ言語を認識する

  直感的に:
  「任意のプログラム変換に対して、その変換で不動点となる
   プログラムが存在する」

  クリーネの不動点定理とも呼ばれる:
  任意の計算可能関数 f に対して、
  プログラム e が存在して φ_e = φ_{f(e)}
  （eの計算する関数 = f(e)の計算する関数）

  意味:
  - プログラムは「自分自身のソースコード」を知ることができる
  - クワイン（自分自身を出力するプログラム）の存在を保証
  - コンピュータウイルスの自己複製の理論的基礎
  - ゲーデルの不完全性定理の対角化補題の計算論版
```

### 8.2 クワイン（自己出力プログラム）

```python
# クワイン: 自分自身のソースコードを出力するプログラム
# 再帰定理の構成的証明の具体例

# Python のクワイン（最小版）
s='s=%r;print(s%%s)';print(s%s)

# より読みやすい版
def quine():
    """
    再帰定理により、任意のプログラミング言語で
    クワインを構築できることが保証されている
    """
    # 2部構成: データ部（A）とコード部（B）
    # A = Bのテキスト表現
    # B = Aを使って全体を再構成するコード

    A = 'def quine():\n    A = %r\n    B = A %% A\n    print(B)\nquine()'
    B = A % A
    print(B)

# 各言語のクワインの例:

# JavaScript
js_quine = '(function q(){console.log("("+q+")()")})()'

# Ruby
ruby_quine = 's="s=%p;puts s%%s";puts s%s'

# C言語
c_quine = '''
#include <stdio.h>
int main(){char*s="#include <stdio.h>%cint main(){char*s=%c%s%c;printf(s,10,34,s,34);}";printf(s,10,34,s,34);}
'''

# クワインの構造:
# 1. データ部: プログラム全体のテンプレート（一部が空白）
# 2. コード部: 空白部分にデータ部自身を埋め込んで出力
#
# 再帰定理はこの構成が常に可能であることを保証する
```

---

## 9. 計算可能性と実務

### 9.1 ソフトウェア検証への影響

```
計算可能性理論がソフトウェア開発に与える影響:

  ┌─────────────────────────────────────────────────┐
  │ 完璧な自動検証は不可能 → 実用的なアプローチ      │
  └─────────────────────────────────────────────────┘

  1. 型システム（保守的近似の代表例）

     TypeScript の例:
     ```
     function divide(a: number, b: number): number {
       return a / b;  // b=0 のチェックなし
     }
     // TypeScript: 型エラーにはならない
     // → 「全てのランタイムエラーを型で検出」は決定不能

     // より安全なアプローチ:
     function safeDivide(a: number, b: NonZero<number>): number {
       return a / b;
     }
     // → 型レベルでゼロ除算を排除（依存型的アプローチ）
     ```

  2. 静的解析ツール（保守的 / 楽観的）

     ESLint, Pylint, Clippy などの lint ツール:
     - 構文パターンベース → 決定可能な部分のみ
     - 誤検出（false positive）を許容
     - 見逃し（false negative）も存在

     高度な静的解析:
     - Coverity, PVS-Studio: パスセンシティブ解析
     - Abstract Interpretation: 抽象解釈による近似
     - 依然として完璧ではない（ライスの定理）

  3. テスト（有限サンプリング）

     テストの本質的限界:
     - 「テストはバグの存在を示せるが、不在は示せない」
       （ダイクストラ）
     - 無限の入力空間を有限のテストケースでカバー
     - プロパティベーステスト（QuickCheck等）で改善可能

  4. 形式検証（制限付き正確解）

     モデル検査（Model Checking）:
     - 有限状態の場合は完全に検証可能
     - 無限状態 → 抽象化で有限に近似
     - SPIN, TLA+, Alloy など

     定理証明（Theorem Proving）:
     - 人間が証明をガイド
     - Coq, Isabelle, Lean, Agda
     - seL4マイクロカーネル: 機能正当性の完全証明
```

### 9.2 プログラミング言語設計への影響

```python
# 計算可能性理論がプログラミング言語設計に与える影響

# 1. 全域性の保証 vs チューリング完全性
#    全ての入力で停止することを保証する言語は
#    チューリング完全ではない

# Agda（全域関数型言語）の例（概念的）:
# - 全ての関数は停止が保証される
# - 再帰は構造的に減少する引数が必要
# - チューリング完全ではないが、実用上は十分

# 概念的な全域再帰の例
def structural_recursion(n: int) -> int:
    """構造的再帰: 引数が必ず減少する → 停止が保証"""
    if n <= 0:
        return 0
    return 1 + structural_recursion(n - 1)  # n は必ず減少

def non_structural(n: int) -> int:
    """非構造的再帰: 停止が保証されない"""
    if n == 1:
        return 0
    if n % 2 == 0:
        return 1 + non_structural(n // 2)
    else:
        return 1 + non_structural(3 * n + 1)  # コラッツ予想
    # 全ての入力で停止するかは未解明！


# 2. 型推論の決定可能性
#
# Hindley-Milner型推論（ML, Haskell98）: 決定可能
# - 主要型（principal type）が存在
# - 完全な型推論が可能
#
# System F（多相λ計算）の型推論: 決定不能
# - 明示的な型注釈が必要
# - Haskell の一部の拡張機能で該当
#
# TypeScript の型推論: 意図的にチューリング完全
# - 条件型（Conditional Types）+ 再帰型 = チューリング完全
# - 型チェックが停止しない型定義が書ける

# TypeScript の型レベル計算の例（概念的）:
# type Fibonacci<N extends number> = ...
# → 型チェッカーが無限ループする可能性


# 3. マクロシステムと計算可能性
#
# C のプリプロセッサ: チューリング完全ではない
# → マクロ展開は必ず停止する
#
# Rust のマクロ: 制限的なチューリング完全性
# → 再帰深度の制限あり（#![recursion_limit = "..."]）
#
# Template Haskell: チューリング完全
# → コンパイル時に任意の計算が可能
#
# Lisp マクロ: チューリング完全
# → コンパイル時（マクロ展開時）にループ可能
```

### 9.3 コンパイラ最適化の限界

```python
# コンパイラ最適化と計算可能性の限界

class CompilerOptimizer:
    """コンパイラ最適化の限界を示す例"""

    def dead_code_elimination(self, code):
        """
        デッドコード除去

        完璧なデッドコード除去は決定不能（ライスの定理）
        実用的なコンパイラは保守的な近似を使用
        """
        # 到達不能なコード:
        # - 常にFalseの条件分岐 → 検出可能（定数畳み込み）
        # - 関数の戻り値に依存する条件 → 一般には検出不能
        pass

    def constant_propagation(self, code):
        """
        定数伝播

        「この変数は常に定数値か？」は一般に決定不能
        しかし、多くの実用的なケースでは判定可能
        """
        pass

    def loop_optimization(self, code):
        """
        ループ最適化

        「このループは有限回で終了するか？」は決定不能
        → ループ不変式の自動発見にも限界がある

        しかし:
        - for i in range(n): ... → 明らかに有限
        - while condition: ... → 一般には不明
        """
        pass

    def alias_analysis(self, code):
        """
        エイリアス解析

        「この2つのポインタは同じメモリを指すか？」
        → 完璧な解析は決定不能
        → 保守的な近似（Andersen, Steensgaard等）

        実務的影響:
        - Cの restrict キーワード: プログラマがヒントを提供
        - Rust の借用規則: コンパイラが安全に追跡可能
        """
        pass


# 最適化の実例: GCC vs 計算可能性の限界

# 以下のコードのデッドコード除去:
def optimization_example():
    x = complex_computation()  # 副作用があるか不明
    if x > 0:
        print("positive")
    else:
        print("non-positive")

    # コンパイラが知りたいこと:
    # 1. complex_computation() は常に正の値を返すか？ → 決定不能
    # 2. complex_computation() に副作用はあるか？ → 決定不能
    # 3. このif文はどちらの分岐が取られるか？ → 決定不能

    # 結果: コンパイラは両方の分岐を残す（保守的）
```

### 9.4 セキュリティへの影響

```
計算可能性理論のセキュリティへの影響:

  1. 完璧なマルウェア検出は不可能
     ┌──────────────────────────────────────────┐
     │ 定理: 全てのマルウェアを正確に検出する      │
     │ プログラムは存在しない                      │
     │                                            │
     │ 証明: マルウェア検出を停止問題に帰着        │
     │ 任意のプログラムPについて:                  │
     │ 「Pは悪意ある動作をするか？」              │
     │ は非自明な性質 → ライスの定理により決定不能  │
     └──────────────────────────────────────────┘

  2. 完璧な情報フロー解析は不可能
     - 「このプログラムは機密データを漏洩するか？」
     - 非自明な意味的性質 → 決定不能
     - 対策: タイントトラッキング（保守的近似）

  3. 難読化の限界
     - Barak et al. (2001): 完璧な一般的難読化は不可能
     - 「プログラムの実装を完全に隠す」ことは理論的に不可能
     - しかし特定のクラスの難読化（iO等）は可能かもしれない

  4. 実務的な対処
     - シグネチャベース検出: 既知のパターンのみ
     - ヒューリスティック検出: 保守的な近似
     - サンドボックス実行: 実際の動作を観察
     - 形式検証: 限定された性質を正確に検証
```

---

## 10. 計算モデルの等価性

### 10.1 各計算モデルの関係

```
主要な計算モデルとその等価性:

  ┌────────────────────────────────────────────────────┐
  │           チューリング等価な計算モデル               │
  ├────────────────────────────────────────────────────┤
  │                                                    │
  │  チューリングマシン (TM)                            │
  │    - 最も標準的なモデル                             │
  │    - テープ + ヘッド + 有限制御                     │
  │                                                    │
  │  λ計算 (Lambda Calculus)                           │
  │    - チャーチが考案                                 │
  │    - 関数の抽象と適用のみ                           │
  │    - 関数型プログラミングの理論的基礎               │
  │    → 詳細は 05-lambda-calculus.md を参照            │
  │                                                    │
  │  帰納的関数 (Recursive Functions)                   │
  │    - 原始再帰 + μ演算子                            │
  │    - 数学的に最も自然な定式化                       │
  │                                                    │
  │  レジスタマシン (Register Machine)                  │
  │    - 有限個のレジスタ + カウンタ                    │
  │    - 実際のCPUに近いモデル                         │
  │                                                    │
  │  タグシステム (Tag System)                          │
  │    - ポストが考案                                   │
  │    - 文字列の先頭を読んで末尾に追加                │
  │    - 非常に単純だがチューリング完全                 │
  │                                                    │
  │  セルオートマトン (Cellular Automata)               │
  │    - ライフゲーム（ルール110等）                    │
  │    - 格子状のセルが局所規則で変化                   │
  │    - 自然の計算過程のモデル                        │
  │                                                    │
  └────────────────────────────────────────────────────┘

  等価性の証明方法:

  A → B: AをBでシミュレートできることを示す
  B → A: BをAでシミュレートできることを示す
  → A ≡ B（計算能力が等価）

  実際の等価性証明の連鎖:

  TM → λ計算:
    TMの状態遷移をλ式の簡約として表現

  λ計算 → TM:
    β簡約をTMの操作として実装

  TM → レジスタマシン:
    テープの内容をレジスタにエンコード

  レジスタマシン → TM:
    レジスタの値をテープ上に記録
```

### 10.2 計算の限界を超える試み

```
超計算（Hypercomputation）— チューリングマシンを超える試み:

  1. オラクルマシン（Oracle Machine）
     - 停止問題のオラクルを持つTM
     - HALT を O(1) で判定可能
     - しかしオラクルの実現方法が不明
     - 理論的ツールとしては有用（相対的計算可能性）

  2. 加速マシン（Accelerating Machine）
     - ステップnに 1/2ⁿ 秒かかる → 全ステップが1秒で完了
     - 物理的には実現不可能（光速の制限、量子効果等）

  3. 超限チューリングマシン
     - 超限順序数までのステップを実行
     - 数学的には定義可能だが物理的に実現不可能

  4. アナログ計算の無限精度
     - 実数の無限精度を利用
     - 実際には測定の精度に限界がある

  現在の合意:
  → 物理的に実現可能な計算デバイスは
    チューリングマシンの計算能力を超えないと考えられている
  → 量子コンピュータも例外ではない
    （計算「能力」は同等、計算「効率」が異なるだけ）
```

---

## 11. 実践演習

### 演習1: チューリングマシンの設計（基礎）

```
問題: 2進数を1増やすチューリングマシンの遷移表を設計せよ。

入力例: 1011 → 出力: 1100
入力例: 1111 → 出力: 10000

ヒント:
1. まず右端に移動する
2. 右端から左に向かって繰り上がり処理を行う
3. 全ての桁が繰り上がった場合、先頭に1を追加する

解答例:
  状態: {q_start, q_right, q_carry, q_done, q_accept}

  遷移表:
  (q_start, 0) → (q_right, 0, R)  // 右端へ移動
  (q_start, 1) → (q_right, 1, R)
  (q_right, 0) → (q_right, 0, R)
  (q_right, 1) → (q_right, 1, R)
  (q_right, B) → (q_carry, B, L)  // 右端到達、繰り上がり開始
  (q_carry, 0) → (q_done, 1, L)   // 0→1で繰り上がり終了
  (q_carry, 1) → (q_carry, 0, L)  // 1→0で繰り上がり継続
  (q_carry, B) → (q_accept, 1, R) // 先頭に繰り上がり
  (q_done, 0) → (q_done, 0, L)    // 左端へ戻る
  (q_done, 1) → (q_done, 1, L)
  (q_done, B) → (q_accept, B, R)  // 完了
```

### 演習2: 停止問題の帰着（応用）

```
問題: 「全ての入力に対して正しい出力を返すか」を自動判定する
ツールが不可能であることを、停止問題への帰着で証明せよ。

証明:
  CORRECT = { ⟨P, S⟩ | プログラムPが仕様Sに対して
                        全ての入力で正しい }

  HALT ≤ₘ CORRECT を示す:

  任意の (M, w)（Mがwで停止するか？）に対して、
  プログラム P' と仕様 S' を構築:

  P'(x):
    1. Mをwで実行する（xは無視）
    2. Mが停止したら、0を返す

  S': 「全ての入力に対して0を返す」

  分析:
  - Mがwで停止する → P'は全入力で0を返す → (P', S') ∈ CORRECT
  - Mがwで停止しない → P'は全入力でループ → (P', S') ∉ CORRECT

  CORRECTが決定可能なら、HALTも決定可能 → 矛盾
  → CORRECT は決定不能 ∎
```

### 演習3: ライスの定理の適用（発展）

```
問題: 以下の各問題について、ライスの定理が適用できるかどうかを
判定し、適用できる場合は非自明な性質であることを示せ。

1. 「TM Mが空文字列εを受理するか？」
   → ライスの定理適用可能
   → P = {L | ε ∈ L}: 非自明（εを受理する言語もしない言語もある）
   → 決定不能

2. 「TM Mが100個以上の状態を持つか？」
   → ライスの定理適用不可
   → これはTMの「構造的」性質（計算する関数の性質ではない）
   → 決定可能（TMの記述を調べれば判定できる）

3. 「TM Mが認識する言語が正規言語か？」
   → ライスの定理適用可能
   → P = {L | Lは正規言語}: 非自明
   → 決定不能

4. 「TM Mが入力 "hello" を10ステップ以内に受理するか？」
   → ライスの定理適用不可
   → 有限ステップのシミュレーションで判定可能
   → 決定可能

5. 「TM Mが計算する関数が全域か？」（全入力で停止するか）
   → ライスの定理適用可能
   → P = {L | Lは決定可能}: 非自明
   → 決定不能（TOTAL問題と同値）
```

### 演習4: クワインの構築（実装）

```python
"""
演習: Pythonでクワインを構築せよ。

条件:
1. 外部ファイルを読まない
2. 空のプログラムではない
3. 自分自身のソースコードを正確に出力する

ヒント: 2部構成（データ部 + コード部）を使う

解答例:
"""

# 方法1: %r フォーマットを使用
s='s=%r;print(s%%s)';print(s%s)

# 方法2: f-stringを使用（Python 3.12+）
# exec(s:="print(f'exec(s:={chr(34)}{s}{chr(34)})')")

# 方法3: 理解しやすい版
def make_quine():
    """クワインの構造を理解するための段階的構築"""

    # ステップ1: テンプレート（自分自身の骨格）
    template = 'template = {!r}\nprint(template.format(template))'

    # ステップ2: テンプレートに自身を埋め込んで出力
    print(template.format(template))

# 出力結果が入力と同じであることを確認
make_quine()
```

### 演習5: 帰着の実践（発展）

```
問題: 以下の問題が決定不能であることを、適切な帰着を用いて証明せよ。

問題: EMPTY = { ⟨M⟩ | L(M) = ∅ }（TMが空言語を認識するか）

証明方法1: ライスの定理による
  P = {∅}（空言語のみからなる性質）
  空言語は RE に属するが、全ての RE 言語が空ではない → P は非自明
  → ライスの定理により決定不能 ∎

証明方法2: 停止問題からの帰着
  HALT ≤ₘ EMPTȲ を示す（EMPTYの補集合に帰着）

  入力: (M, w)
  構築するTM M':

  M'(x):
    1. Mをwで実行する
    2. Mが停止したら受理

  分析:
  - Mがwで停止する → M'は全ての入力を受理 → L(M') = Σ* ≠ ∅ → M' ∉ EMPTY
  - Mがwで停止しない → M'は何も受理しない → L(M') = ∅ → M' ∈ EMPTY

  つまり: (M, w) ∈ HALT ⟺ ⟨M'⟩ ∉ EMPTY

  EMPTY が決定可能なら HALT も決定可能 → 矛盾
  → EMPTY は決定不能 ∎
```

---

## 12. 計算可能性と現代の話題

### 12.1 機械学習と計算可能性

```
機械学習における計算可能性の問題:

  1. 学習可能性（PAC学習）
     - 「あるクラスの関数は学習可能か？」
     - 一部の問題は計算可能性の壁に直面する
     - Ben-David et al. (2019):
       特定の学習問題は集合論の公理に依存（ZFCから独立）

  2. ニューラルネットワークの検証
     - 「このNNは全ての入力に対して安全か？」
     - 一般的なNNの性質検証は決定不能
     - ReLUネットワークの特定の性質は判定可能

  3. AutoML と Neural Architecture Search
     - 「最適なアーキテクチャを自動で見つける」
     - 無限の探索空間 → ヒューリスティックによる近似
     - No Free Lunch定理との関連

  4. LLM（大規模言語モデル）
     - LLMはチューリング完全か？
     - 有限の文脈長 → 厳密にはチューリング完全ではない
     - しかし実用上は非常に広い計算能力を持つ
     - 外部ツール（コード実行等）と組み合わせればチューリング完全に
```

### 12.2 量子計算と計算可能性

```
量子コンピュータと計算可能性:

  重要な結論:
  ┌───────────────────────────────────────────────────┐
  │ 量子コンピュータはチューリングマシンの計算「能力」を │
  │ 超えない。しかし計算「効率」は超える可能性がある    │
  └───────────────────────────────────────────────────┘

  量子で高速化される問題:
  - 素因数分解: O(n³) [量子] vs O(exp(n^{1/3})) [古典]
  - 非構造化探索: O(√N) [量子] vs O(N) [古典]
  - 量子シミュレーション: 指数的高速化

  量子でも解けない問題:
  - 停止問題 → 依然として決定不能
  - NP完全問題 → おそらく多項式時間では解けない
    （BQP ⊄ NP は未証明だが、予想されている）

  計算可能性への影響:
  - 決定可能/決定不能の境界は変わらない
  - 変わるのは効率（多項式 vs 指数）の部分のみ
  - 計算複雑性理論の一部が変わる可能性
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| チューリングマシン | 計算の理論的モデル。テープ+ヘッド+有限制御で全ての計算を表現 |
| チャーチ=チューリングの提唱 | 計算可能 = チューリングマシンで計算可能。反例は見つかっていない |
| 停止問題 | 決定不能の代表例。完璧なバグ検出器は存在しない |
| 帰着 | 問題AをBに変換。決定不能性の証明に使用 |
| ライスの定理 | プログラムの意味的性質は全て決定不能 |
| 再帰定理 | プログラムは自身を参照可能。クワインの理論的保証 |
| 決定可能性の階層 | 正規 ⊂ CFL ⊂ 決定可能 ⊂ RE ⊂ 全言語 |
| 実務的影響 | 完璧は不可能→保守的/楽観的な近似を使い分ける |

---

## 次に読むべきガイド
→ [[02-complexity-theory.md]] — 計算複雑性理論（決定可能な問題の中の「難しさ」）

---

## 参考文献
1. Sipser, M. "Introduction to the Theory of Computation." Chapters 3-5.
2. Turing, A. M. "On Computable Numbers, with an Application to the Entscheidungsproblem." 1936.
3. Church, A. "An Unsolvable Problem of Elementary Number Theory." 1936.
4. Rice, H. G. "Classes of Recursively Enumerable Sets and Their Decision Problems." 1953.
5. Kleene, S. C. "Introduction to Metamathematics." 1952.
6. Gödel, K. "Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I." 1931.
7. Davis, M. "Computability and Unsolvability." 1958.
8. Rogers, H. "Theory of Recursive Functions and Effective Computability." 1967.
9. Cutland, N. "Computability: An Introduction to Recursive Function Theory." 1980.
10. Arora, S. and Barak, B. "Computational Complexity: A Modern Approach." 2009.
