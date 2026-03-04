# オートマトンと形式言語

> 正規表現、コンパイラ、プロトコル検証——オートマトン理論はCSの理論的基盤であり、実務にも深く浸透している。DFA/NFA の等価性、正規言語のポンピング補題、文脈自由文法の構文解析力、そしてチョムスキー階層による言語クラスの体系的分類を理解することで、計算の本質に迫ることができる。

## この章で学ぶこと

- [ ] 有限オートマトン（DFA/NFA）の定義・構成・等価性を理解する
- [ ] 正規言語の性質と限界（ポンピング補題）を説明できる
- [ ] NFA から DFA への部分集合構成法を実装できる
- [ ] 正規表現とオートマトンの相互変換を行える
- [ ] 文脈自由文法とプッシュダウンオートマトンの関係を理解する
- [ ] チョムスキー階層の4つのレベルを比較・説明できる
- [ ] Python で DFA/NFA シミュレータを実装できる

## 前提知識

- 基本的なプログラミング知識（Python の基礎構文）
- 集合論の基本概念（集合、写像、直積）
- グラフ理論の基礎（有向グラフ、経路）

---

## 1. オートマトンの基礎概念

### 1.1 オートマトンとは何か

オートマトン（automaton、複数形: automata）は、入力記号列を読み取り、内部状態を遷移させながら、最終的にその入力を「受理」するか「拒否」するかを決定する抽象的な計算モデルである。

オートマトン理論が重要である理由は以下の通りである。

1. **計算の限界を明確にする**: どのクラスの機械がどのクラスの問題を解けるかを厳密に定義する
2. **実用的なツールの理論的基盤**: 正規表現エンジン、コンパイラ、プロトコル検証器はすべてオートマトン理論に基づく
3. **設計の指針を与える**: 問題がどの言語クラスに属するかを知ることで、適切なアルゴリズムを選択できる

### 1.2 形式言語の基本用語

オートマトン理論を学ぶ上で不可欠な用語を定義する。

```
用語の定義:

  アルファベット（Σ）: 記号の有限集合
    例: Σ = {0, 1}（2進アルファベット）
    例: Σ = {a, b, c, ..., z}（英小文字アルファベット）

  文字列（string / word）: アルファベットの記号を並べた有限列
    例: Σ = {0, 1} のとき、"0101", "111", "0" は文字列
    空文字列を ε（イプシロン）と書く

  文字列の長さ |w|: 文字列 w に含まれる記号の個数
    |"abc"| = 3,  |ε| = 0

  文字列の連結: w₁ · w₂ = w₁w₂
    "ab" · "cd" = "abcd"
    w · ε = ε · w = w

  Σ*: Σ上のすべての文字列の集合（空文字列を含む）
    Σ = {0, 1} のとき Σ* = {ε, 0, 1, 00, 01, 10, 11, 000, ...}

  Σ⁺: Σ* から空文字列を除いた集合  Σ⁺ = Σ* \ {ε}

  言語（language）: L ⊆ Σ* （Σ*の部分集合）
    例: L = {w ∈ {0,1}* | w は偶数個の0を含む}
    例: L = {aⁿbⁿ | n ≥ 0} = {ε, ab, aabb, aaabbb, ...}
```

### 1.3 オートマトンの分類と計算能力

```
オートマトンの分類（計算能力の昇順）:

  ┌─────────────────────────────────────────────────────────┐
  │  チューリングマシン (TM)                                  │
  │  ┌───────────────────────────────────────────────────┐  │
  │  │  線形有界オートマトン (LBA)                         │  │
  │  │  ┌─────────────────────────────────────────────┐  │  │
  │  │  │  プッシュダウンオートマトン (PDA)              │  │  │
  │  │  │  ┌───────────────────────────────────────┐  │  │  │
  │  │  │  │  有限オートマトン (FA)                  │  │  │  │
  │  │  │  │  DFA / NFA                             │  │  │  │
  │  │  │  │  → 正規言語を認識                      │  │  │  │
  │  │  │  └───────────────────────────────────────┘  │  │  │
  │  │  │  → 文脈自由言語を認識                        │  │  │
  │  │  └─────────────────────────────────────────────┘  │  │
  │  │  → 文脈依存言語を認識                              │  │
  │  └───────────────────────────────────────────────────┘  │
  │  → 帰納的可算言語を認識                                  │
  └─────────────────────────────────────────────────────────┘

  各レベルは真の包含関係:
    正規言語 ⊂ 文脈自由言語 ⊂ 文脈依存言語 ⊂ 帰納的可算言語
```

---

## 2. 有限オートマトン（DFA と NFA）

### 2.1 DFA（決定性有限オートマトン）の定義

DFA は最も基本的なオートマトンであり、各状態において入力記号ごとに遷移先が一意に定まる。

```
DFA の形式的定義:
  M = (Q, Σ, δ, q₀, F)

  Q : 状態の有限集合
  Σ : 入力アルファベット（有限集合）
  δ : 遷移関数  Q × Σ → Q（全域関数）
  q₀: 初期状態  q₀ ∈ Q
  F : 受理状態の集合  F ⊆ Q

  動作:
    1. 初期状態 q₀ から開始
    2. 入力文字列を左から1文字ずつ読む
    3. 現在の状態と読んだ文字に応じて δ で遷移
    4. 入力をすべて読み終えたとき:
       - 現在の状態 ∈ F → 受理（accept）
       - 現在の状態 ∉ F → 拒否（reject）
```

**例: 偶数個の 'a' を含む文字列を受理する DFA**

```
  DFA M₁ = ({q₀, q₁}, {a, b}, δ, q₀, {q₀})

  状態遷移図:

       ┌──b──┐          ┌──b──┐
       │     │          │     │
       ▼     │          ▼     │
    →((q₀))──a──→( q₁ )
       ▲              │
       └──────a───────┘

  ((q₀)): 二重丸は受理状態
  →: 矢印は初期状態

  遷移表:
  ┌────────┬────────┬────────┐
  │ 状態   │ a      │ b      │
  ├────────┼────────┼────────┤
  │ →*q₀   │ q₁     │ q₀     │
  │   q₁   │ q₀     │ q₁     │
  └────────┴────────┴────────┘
  →: 初期状態  *: 受理状態

  トレース例:
  入力 "abba":  q₀ →a→ q₁ →b→ q₁ →b→ q₁ →a→ q₀ → 受理 (q₀ ∈ F)
  入力 "aba":   q₀ →a→ q₁ →b→ q₁ →a→ q₀         → 受理 (q₀ ∈ F)
  入力 "a":     q₀ →a→ q₁                          → 拒否 (q₁ ∉ F)
  入力 "bb":    q₀ →b→ q₀ →b→ q₀                  → 受理 (q₀ ∈ F)
  入力 ε:       q₀                                   → 受理 (q₀ ∈ F)
```

### 2.2 DFA の Python 実装

```python
"""
DFA（決定性有限オートマトン）シミュレータ

任意の DFA を定義し、入力文字列の受理/拒否を判定する。
"""

from typing import Dict, Set, Tuple


class DFA:
    """決定性有限オートマトン（DFA）の実装"""

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transition: Dict[Tuple[str, str], str],
        start_state: str,
        accept_states: Set[str],
    ):
        """
        DFA を初期化する。

        Args:
            states: 状態の集合 Q
            alphabet: 入力アルファベット Σ
            transition: 遷移関数 δ を辞書で表現
                        {(状態, 入力記号): 次の状態}
            start_state: 初期状態 q₀
            accept_states: 受理状態の集合 F
        """
        self.states = states
        self.alphabet = alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_states = accept_states
        self._validate()

    def _validate(self) -> None:
        """DFA の定義が正しいか検証する。"""
        # 初期状態が状態集合に含まれるか
        if self.start_state not in self.states:
            raise ValueError(
                f"初期状態 '{self.start_state}' が状態集合に含まれていない"
            )

        # 受理状態が状態集合の部分集合か
        if not self.accept_states.issubset(self.states):
            invalid = self.accept_states - self.states
            raise ValueError(
                f"受理状態 {invalid} が状態集合に含まれていない"
            )

        # 遷移関数が全域関数か（すべての状態×記号の組に対して定義されているか）
        for state in self.states:
            for symbol in self.alphabet:
                if (state, symbol) not in self.transition:
                    raise ValueError(
                        f"遷移 δ({state}, {symbol}) が未定義"
                    )
                next_state = self.transition[(state, symbol)]
                if next_state not in self.states:
                    raise ValueError(
                        f"遷移先 '{next_state}' が状態集合に含まれていない"
                    )

    def process(self, input_string: str) -> Tuple[bool, list]:
        """
        入力文字列を処理し、受理/拒否を返す。

        Args:
            input_string: 処理する入力文字列

        Returns:
            (受理したか, 状態遷移の履歴)
        """
        current = self.start_state
        history = [current]

        for symbol in input_string:
            if symbol not in self.alphabet:
                raise ValueError(
                    f"入力記号 '{symbol}' がアルファベットに含まれていない"
                )
            current = self.transition[(current, symbol)]
            history.append(current)

        accepted = current in self.accept_states
        return accepted, history

    def accepts(self, input_string: str) -> bool:
        """入力文字列を受理するかどうかを返す。"""
        accepted, _ = self.process(input_string)
        return accepted

    def trace(self, input_string: str) -> str:
        """入力文字列の処理過程を文字列で返す。"""
        accepted, history = self.process(input_string)
        display_input = input_string if input_string else "ε"
        transitions = []
        for i, state in enumerate(history):
            if i < len(input_string):
                transitions.append(f"{state} →{input_string[i]}→ ")
            else:
                transitions.append(state)
        path = "".join(transitions)
        result = "受理" if accepted else "拒否"
        return f"入力 \"{display_input}\": {path} → {result}"


# --- 使用例: 偶数個の 'a' を含む文字列を受理する DFA ---

dfa_even_a = DFA(
    states={"q0", "q1"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): "q1",
        ("q0", "b"): "q0",
        ("q1", "a"): "q0",
        ("q1", "b"): "q1",
    },
    start_state="q0",
    accept_states={"q0"},
)

# テスト
test_cases = ["", "a", "b", "aa", "ab", "abba", "aba", "aabba"]
for tc in test_cases:
    print(dfa_even_a.trace(tc))

# 出力:
# 入力 "ε": q0 → 受理
# 入力 "a": q0 →a→ q1 → 拒否
# 入力 "b": q0 →b→ q0 → 受理
# 入力 "aa": q0 →a→ q1 →a→ q0 → 受理
# 入力 "ab": q0 →a→ q1 →b→ q1 → 拒否
# 入力 "abba": q0 →a→ q1 →b→ q1 →b→ q1 →a→ q0 → 受理
# 入力 "aba": q0 →a→ q1 →b→ q1 →a→ q0 → 受理
# 入力 "aabba": q0 →a→ q1 →a→ q0 →b→ q0 →b→ q0 →a→ q1 → 拒否
```

### 2.3 NFA（非決定性有限オートマトン）の定義

NFA は DFA を拡張したモデルであり、ある状態から同じ入力記号で複数の遷移先が存在してもよく、さらに入力を読まずに遷移する ε 遷移が許される。

```
NFA の形式的定義:
  N = (Q, Σ, δ, q₀, F)

  Q : 状態の有限集合
  Σ : 入力アルファベット（有限集合）
  δ : 遷移関数  Q × (Σ ∪ {ε}) → P(Q)
      ※ P(Q) は Q のべき集合（すべての部分集合の集合）
  q₀: 初期状態  q₀ ∈ Q
  F : 受理状態の集合  F ⊆ Q

  DFA との違い:
  ┌──────────────────┬────────────────────┬───────────────────┐
  │ 特徴             │ DFA                │ NFA               │
  ├──────────────────┼────────────────────┼───────────────────┤
  │ 遷移先の数       │ 各記号に対し正確に1つ│ 0個以上（複数可）  │
  │ ε遷移           │ 不可               │ 可能              │
  │ 遷移関数の型     │ Q × Σ → Q          │ Q × (Σ∪{ε}) → P(Q)│
  │ 受理条件         │ 唯一の計算経路で判定│ いずれかの経路で    │
  │                  │                    │ 受理状態に到達      │
  │ 計算能力         │ 正規言語           │ 正規言語（同じ）    │
  └──────────────────┴────────────────────┴───────────────────┘

  重要な定理: DFA と NFA は計算能力が等価
  → 任意の NFA に対し、同じ言語を受理する DFA が存在する
  → ただし DFA の状態数は最悪 2^n（n は NFA の状態数）
```

**例: "abb" で終わる文字列を受理する NFA**

```
  NFA N₁ = ({q₀, q₁, q₂, q₃}, {a, b}, δ, q₀, {q₃})

  状態遷移図:

       ┌─a,b─┐
       │     │
       ▼     │
    →( q₀ )──a──→( q₁ )──b──→( q₂ )──b──→(( q₃ ))
                                            受理状態

  遷移表:
  ┌────────┬────────────┬────────────┐
  │ 状態   │ a          │ b          │
  ├────────┼────────────┼────────────┤
  │ →q₀    │ {q₀, q₁}  │ {q₀}      │
  │  q₁    │ ∅          │ {q₂}      │
  │  q₂    │ ∅          │ {q₃}      │
  │ *q₃    │ ∅          │ ∅         │
  └────────┴────────────┴────────────┘

  トレース例（入力 "aabb"）:
  非決定的に分岐する計算木:
                    q₀
                   / \
           a→    /   \  ←a
               q₀    q₁
              / \     |
       a→   /   \    |←b
           q₀   q₁   q₂
           |    |     |
    b→     |    |←b   |←b (遷移先なし→消滅)
           q₀   q₂
           |    |
    b→     |    |←b
           q₀   q₃ ← 受理状態に到達！

  少なくとも1つの経路で受理状態に到達 → "aabb" は受理
```

### 2.4 NFA の Python 実装

```python
"""
NFA（非決定性有限オートマトン）シミュレータ

ε遷移をサポートし、部分集合追跡法で受理判定を行う。
"""

from typing import Dict, FrozenSet, Set, Tuple


class NFA:
    """非決定性有限オートマトン（NFA）の実装"""

    def __init__(
        self,
        states: Set[str],
        alphabet: Set[str],
        transition: Dict[Tuple[str, str], Set[str]],
        start_state: str,
        accept_states: Set[str],
    ):
        """
        NFA を初期化する。

        Args:
            states: 状態の集合 Q
            alphabet: 入力アルファベット Σ（εを含まない）
            transition: 遷移関数 δ
                        {(状態, 入力記号またはε): 次の状態の集合}
                        ε遷移はキーに ("状態", "") を使用
            start_state: 初期状態 q₀
            accept_states: 受理状態の集合 F
        """
        self.states = states
        self.alphabet = alphabet
        self.transition = transition
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """
        与えられた状態集合の ε 閉包を計算する。

        ε遷移のみで到達可能なすべての状態を含む集合を返す。
        """
        closure = set(states)
        stack = list(states)

        while stack:
            state = stack.pop()
            # ε遷移先を取得（ε遷移は空文字列 "" で表現）
            epsilon_targets = self.transition.get((state, ""), set())
            for target in epsilon_targets:
                if target not in closure:
                    closure.add(target)
                    stack.append(target)

        return closure

    def process(self, input_string: str) -> Tuple[bool, list]:
        """
        入力文字列を処理し、受理/拒否を返す。

        同時に到達可能な状態の集合を追跡する方法で実装する。
        これは部分集合追跡（on-the-fly subset construction）に相当する。

        Returns:
            (受理したか, 各ステップでの状態集合の履歴)
        """
        # 初期状態のε閉包から開始
        current_states = self.epsilon_closure({self.start_state})
        history = [frozenset(current_states)]

        for symbol in input_string:
            next_states = set()
            for state in current_states:
                targets = self.transition.get((state, symbol), set())
                next_states.update(targets)
            # 遷移先のε閉包を計算
            current_states = self.epsilon_closure(next_states)
            history.append(frozenset(current_states))

        # 現在の状態集合に受理状態が含まれていれば受理
        accepted = bool(current_states & self.accept_states)
        return accepted, history

    def accepts(self, input_string: str) -> bool:
        """入力文字列を受理するかどうかを返す。"""
        accepted, _ = self.process(input_string)
        return accepted

    def trace(self, input_string: str) -> str:
        """入力文字列の処理過程を文字列で返す。"""
        accepted, history = self.process(input_string)
        display_input = input_string if input_string else "ε"
        parts = []
        for i, state_set in enumerate(history):
            sorted_states = sorted(state_set)
            states_str = "{" + ", ".join(sorted_states) + "}"
            if i < len(input_string):
                parts.append(f"{states_str} →{input_string[i]}→ ")
            else:
                parts.append(states_str)
        path = "".join(parts)
        result = "受理" if accepted else "拒否"
        return f"入力 \"{display_input}\": {path} → {result}"


# --- 使用例: "abb" で終わる文字列を受理する NFA ---

nfa_ends_abb = NFA(
    states={"q0", "q1", "q2", "q3"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): {"q0", "q1"},
        ("q0", "b"): {"q0"},
        ("q1", "b"): {"q2"},
        ("q2", "b"): {"q3"},
    },
    start_state="q0",
    accept_states={"q3"},
)

# テスト
test_cases_nfa = ["abb", "aabb", "babb", "ab", "abab", "aabbabb"]
for tc in test_cases_nfa:
    print(nfa_ends_abb.trace(tc))

# 出力:
# 入力 "abb": {q0} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → 受理
# 入力 "aabb": {q0} →a→ {q0, q1} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → 受理
# 入力 "babb": {q0} →b→ {q0} →a→ {q0, q1} →b→ {q0, q2} →b→ {q0, q3} → 受理
# 入力 "ab": {q0} →a→ {q0, q1} →b→ {q0, q2} → 拒否
# 入力 "abab": {q0} →a→ {q0, q1} →b→ {q0, q2} →a→ {q0, q1} →b→ {q0, q2} → 拒否
# 入力 "aabbabb": ... → 受理
```

### 2.5 NFA から DFA への変換（部分集合構成法）

NFA と DFA の等価性を示す構成的証明が部分集合構成法（subset construction）である。NFA の状態集合のべき集合を DFA の状態として使用する。

```
部分集合構成法のアルゴリズム:

  入力: NFA N = (Q_N, Σ, δ_N, q₀_N, F_N)
  出力: DFA D = (Q_D, Σ, δ_D, q₀_D, F_D)

  手順:
    1. q₀_D = ε-closure({q₀_N})
    2. Q_D = {q₀_D}（作業リストに追加）
    3. 作業リストが空になるまで繰り返す:
       a. 作業リストから状態 S を取り出す
       b. 各入力記号 a ∈ Σ について:
          T = ε-closure(∪{δ_N(s, a) | s ∈ S})
          δ_D(S, a) = T
          T が Q_D にまだなければ Q_D に追加し作業リストに入れる
    4. F_D = {S ∈ Q_D | S ∩ F_N ≠ ∅}

  例: "abb" で終わる NFA を DFA に変換

  NFA の状態: {q₀, q₁, q₂, q₃}
  （ε遷移なしなので ε-closure は恒等）

  ステップ1: 初期状態 = {q₀}
  ステップ2:
    {q₀} →a→ {q₀,q₁}  →b→ {q₀}
    {q₀,q₁} →a→ {q₀,q₁}  →b→ {q₀,q₂}
    {q₀,q₂} →a→ {q₀,q₁}  →b→ {q₀,q₃}
    {q₀,q₃} →a→ {q₀,q₁}  →b→ {q₀}

  DFA の状態と対応:
    A = {q₀}       B = {q₀,q₁}
    C = {q₀,q₂}    D = {q₀,q₃}  ← 受理状態

  DFA の遷移表:
  ┌────────┬────────┬────────┐
  │ 状態   │ a      │ b      │
  ├────────┼────────┼────────┤
  │ →A     │ B      │ A      │
  │  B     │ B      │ C      │
  │  C     │ B      │ D      │
  │ *D     │ B      │ A      │
  └────────┴────────┴────────┘
```

### 2.6 部分集合構成法の Python 実装

```python
"""
NFA から DFA への変換（部分集合構成法）

NFA を等価な DFA に変換するアルゴリズムの実装。
"""

from typing import Dict, FrozenSet, Set, Tuple


def nfa_to_dfa(nfa: "NFA") -> "DFA":
    """
    部分集合構成法で NFA を DFA に変換する。

    Args:
        nfa: 変換元の NFA

    Returns:
        等価な DFA
    """
    # DFA の初期状態 = NFA の初期状態のε閉包
    dfa_start = frozenset(nfa.epsilon_closure({nfa.start_state}))

    # DFA の状態集合と遷移関数を構築
    dfa_states: Set[FrozenSet[str]] = set()
    dfa_transition: Dict[Tuple[FrozenSet[str], str], FrozenSet[str]] = {}
    worklist = [dfa_start]
    dfa_states.add(dfa_start)

    while worklist:
        current = worklist.pop()

        for symbol in nfa.alphabet:
            # 現在の DFA 状態（= NFA 状態の集合）の各 NFA 状態から
            # symbol で遷移可能な NFA 状態を集め、ε閉包を取る
            next_nfa_states = set()
            for nfa_state in current:
                targets = nfa.transition.get((nfa_state, symbol), set())
                next_nfa_states.update(targets)

            next_dfa_state = frozenset(
                nfa.epsilon_closure(next_nfa_states)
            )
            dfa_transition[(current, symbol)] = next_dfa_state

            if next_dfa_state not in dfa_states:
                dfa_states.add(next_dfa_state)
                worklist.append(next_dfa_state)

    # DFA の受理状態 = NFA の受理状態を含む DFA 状態
    dfa_accept = {
        s for s in dfa_states
        if s & nfa.accept_states
    }

    # 状態名を読みやすくする
    state_names = {}
    for i, s in enumerate(sorted(dfa_states, key=lambda x: sorted(x))):
        nfa_label = "{" + ",".join(sorted(s)) + "}"
        state_names[s] = f"S{i}_{nfa_label}"

    # DFA オブジェクトを構築
    named_states = {state_names[s] for s in dfa_states}
    named_transition = {
        (state_names[s], sym): state_names[t]
        for (s, sym), t in dfa_transition.items()
    }
    named_start = state_names[dfa_start]
    named_accept = {state_names[s] for s in dfa_accept}

    return DFA(
        states=named_states,
        alphabet=nfa.alphabet,
        transition=named_transition,
        start_state=named_start,
        accept_states=named_accept,
    )


# --- 使用例 ---

# "abb" で終わる文字列を受理する NFA を DFA に変換
nfa = NFA(
    states={"q0", "q1", "q2", "q3"},
    alphabet={"a", "b"},
    transition={
        ("q0", "a"): {"q0", "q1"},
        ("q0", "b"): {"q0"},
        ("q1", "b"): {"q2"},
        ("q2", "b"): {"q3"},
    },
    start_state="q0",
    accept_states={"q3"},
)

dfa = nfa_to_dfa(nfa)

print("=== 変換された DFA ===")
print(f"状態: {dfa.states}")
print(f"初期状態: {dfa.start_state}")
print(f"受理状態: {dfa.accept_states}")
print(f"遷移関数:")
for (state, symbol), target in sorted(dfa.transition.items()):
    print(f"  δ({state}, {symbol}) = {target}")

# 変換後の DFA でテスト
for tc in ["abb", "aabb", "ab", "babb"]:
    print(dfa.trace(tc))
```

### 2.7 DFA の最小化

同じ言語を受理する DFA は複数存在するが、状態数が最小の DFA（最小 DFA）は同型を除いて一意である。最小化アルゴリズムとして Hopcroft のアルゴリズムが知られている。

```
DFA 最小化の基本方針（同値類分割法）:

  1. 到達不能状態の除去:
     初期状態から到達できない状態を削除する

  2. 等価状態の統合:
     2つの状態 p, q が「区別不能」なら統合する

     区別可能の定義:
       状態 p, q が区別可能 ⟺
       ある文字列 w が存在し、δ*(p,w) ∈ F かつ δ*(q,w) ∉ F
       （またはその逆）

  3. テーブル充填法（Table-filling algorithm）:
     a. すべての状態ペア (p,q) を表にする
     b. p ∈ F, q ∉ F（またはその逆）のペアを「区別可能」とマーク
     c. 未マークのペア (p,q) について:
        ある記号 a で (δ(p,a), δ(q,a)) が区別可能なら
        (p,q) も区別可能とマーク
     d. 変更がなくなるまで繰り返す
     e. 未マークのペアは等価 → 統合

  例:
    最小化前（5状態）→ 最小化後（3状態）
    等価な状態を発見し統合することで状態数を削減
```

---

## 3. 正規言語と正規表現

### 3.1 正規言語の定義と特徴付け

正規言語は最も基本的な言語クラスであり、以下の3つの特徴付けが等価である。これは Kleene の定理として知られている。

```
正規言語の等価な特徴付け（Kleene の定理）:

  以下の3つは同じ言語クラスを定義する:

  1. DFA が受理する言語
  2. NFA が受理する言語
  3. 正規表現が表す言語

  すなわち:
    L が DFA で受理可能
    ⟺ L が NFA で受理可能
    ⟺ L が正規表現で表現可能
```

### 3.2 正規表現の形式的定義

```
正規表現の帰納的定義（Σ 上の正規表現）:

  基底:
    1. ∅ は正規表現（空言語 ∅ を表す）
    2. ε は正規表現（{ε} を表す）
    3. 各 a ∈ Σ について、a は正規表現（{a} を表す）

  帰納ステップ（R₁, R₂ が正規表現のとき）:
    4. (R₁ | R₂) は正規表現（和集合: L(R₁) ∪ L(R₂)）
    5. (R₁ · R₂) は正規表現（連結: L(R₁) · L(R₂)）
    6. (R₁*) は正規表現（クリーネ閉包: L(R₁)*）

  演算子の優先順位: * > · > |
    例: ab|c* は (a·b)|(c*) と解釈される

  正規表現の例と対応する言語:
  ┌────────────────┬─────────────────────────────┬──────────────────┐
  │ 正規表現       │ 表す言語                     │ 具体例           │
  ├────────────────┼─────────────────────────────┼──────────────────┤
  │ a*             │ {ε, a, aa, aaa, ...}        │ 0個以上のa       │
  │ a⁺ = aa*       │ {a, aa, aaa, ...}           │ 1個以上のa       │
  │ (a|b)*         │ {a,b}上の全文字列           │ 任意のa,b列      │
  │ a*b*           │ aが0個以上の後にbが0個以上    │ ε, a, b, aab    │
  │ (ab)*          │ abの0回以上の繰り返し        │ ε, ab, abab     │
  │ (a|b)*abb      │ abbで終わる{a,b}上の文字列   │ abb, aabb, babb │
  │ a(a|b)*a       │ aで始まりaで終わる長さ2以上   │ aa, aba, abba   │
  └────────────────┴─────────────────────────────┴──────────────────┘
```

### 3.3 正規表現とオートマトンの相互変換

```
変換の3つの方向:

  正規表現 ──Thompson構成──→ NFA
      ↑                       │
      │                   部分集合構成法
  状態除去法                    │
      │                       ▼
      └────────────────── DFA

  1. Thompson 構成法（正規表現 → NFA）:
     正規表現の構造に沿って帰納的に NFA を構築

     基底:
       ε:  →(s)──ε──→((f))
       a:  →(s)──a──→((f))

     和 R₁|R₂:
                  ┌→ NFA(R₁) ─→┐
       →(s)──ε──┤              ├──ε──→((f))
                  └→ NFA(R₂) ─→┘

     連結 R₁·R₂:
       →(s)── NFA(R₁) ──ε── NFA(R₂) ──→((f))

     閉包 R₁*:
                    ┌──────ε──────┐
                    ↓             │
       →(s)──ε──→ NFA(R₁) ──ε──→((f))
           │                      ↑
           └──────────ε──────────┘

  2. 部分集合構成法（NFA → DFA）:
     → セクション 2.5 で詳述済み

  3. 状態除去法（DFA → 正規表現）:
     DFA の状態を1つずつ除去し、辺のラベルを正規表現に一般化
     最終的に初期状態→受理状態の正規表現を得る
```

### 3.4 正規言語の閉包性

正規言語のクラスは多くの演算に対して閉じている（演算の結果も正規言語になる）。

```
正規言語の閉包性:

  L₁, L₂ が正規言語のとき、以下もすべて正規言語:

  ┌──────────────────────┬─────────────────┬──────────────────────┐
  │ 演算                 │ 定義             │ 証明方法            │
  ├──────────────────────┼─────────────────┼──────────────────────┤
  │ 和集合 L₁ ∪ L₂      │ どちらかに属する │ NFA の並列合成       │
  │ 連結 L₁ · L₂        │ 前後に分割可能   │ NFA の直列合成       │
  │ クリーネ閉包 L₁*     │ 0回以上の連結    │ NFA にε遷移追加     │
  │ 補集合 Σ* \ L₁      │ L₁に属さない     │ DFA の受理/拒否反転  │
  │ 共通部分 L₁ ∩ L₂    │ 両方に属する     │ 積オートマトン       │
  │ 差集合 L₁ \ L₂      │ L₁のみに属する   │ L₁ ∩ (Σ*\L₂)       │
  │ 反転 L₁ᴿ             │ 文字列を逆順に   │ NFA の辺を逆向きに  │
  │ 準同型写像           │ 記号の置換       │ NFA の辺ラベル置換   │
  └──────────────────────┴─────────────────┴──────────────────────┘

  実務での活用:
  - 補集合の閉包性 → 「マッチしない」パターンも正規表現で表現可能
  - 共通部分の閉包性 → 複数条件の AND 結合が可能
  - これらの閉包性により、正規言語の等価性判定が決定可能
```

### 3.5 正規言語のポンピング補題

ポンピング補題は、ある言語が正規言語で**ない**ことを証明するための重要なツールである。

```
正規言語のポンピング補題:

  L が正規言語ならば、ある定数 p ≥ 1 が存在し、
  |w| ≥ p を満たすすべての w ∈ L について、
  w = xyz と分割でき、以下の3条件を満たす:

    (1) |y| > 0      （y は空でない）
    (2) |xy| ≤ p     （xy は先頭 p 文字以内）
    (3) すべての i ≥ 0 について xy^i z ∈ L
        （y を何回繰り返しても L に属する）

  直感的な理解:
    DFA の状態数が p のとき、長さ p 以上の文字列を処理すると
    鳩の巣原理により必ずある状態を2回以上通る。
    その繰り返し部分（ループ）が y に対応する。

    →(q₀)──x──→(qᵢ)──y──→(qᵢ)──z──→((qf))
                  ↑    ↓
                  └────┘  ← このループを0回、1回、2回、...
                             繰り返しても受理される

  ポンピング補題の使い方（背理法）:
    「L が正規言語でない」ことを証明するには:
    1. L が正規言語だと仮定する
    2. ポンピング定数 p が存在する
    3. うまく w ∈ L を選ぶ（|w| ≥ p）
    4. 任意の分割 w = xyz（条件(1)(2)を満たす）に対し、
       ある i で xy^i z ∉ L を示す
    5. 矛盾 → L は正規言語でない
```

**例: L = {aⁿbⁿ | n ≥ 0} が正規言語でないことの証明**

```
  証明:
    L = {ε, ab, aabb, aaabbb, ...} が正規言語だと仮定する。

    1. ポンピング定数 p が存在する
    2. w = aᵖbᵖ を選ぶ（|w| = 2p ≥ p、w ∈ L）
    3. w = xyz と分割する（|y| > 0, |xy| ≤ p）
       |xy| ≤ p なので、xy は w の先頭 p 文字以内
       → x と y は a のみからなる
       x = aˢ, y = aᵗ（t > 0）, z = aᵖ⁻ˢ⁻ᵗbᵖ
    4. i = 2 のとき:
       xy²z = aˢ · a²ᵗ · aᵖ⁻ˢ⁻ᵗbᵖ = aᵖ⁺ᵗbᵖ
       t > 0 なので p+t > p → a の数 > b の数
       → xy²z ∉ L
    5. ポンピング補題に矛盾
       → L は正規言語でない  □

  この結果の意味:
    対応する括弧の検証（「(」と「)」の数が等しいか）は
    有限オートマトンでは不可能 → プッシュダウンオートマトンが必要
```

### 3.6 正規表現の実務での対応

```
正規表現エンジンの実装方式と性能:

  ┌──────────────────┬──────────────┬────────────────┬─────────────┐
  │ 方式             │ 時間計算量   │ 採用言語/エンジン │ 特徴        │
  ├──────────────────┼──────────────┼────────────────┼─────────────┤
  │ DFA ベース       │ O(n)         │ grep (一部)    │ 高速・省メモリ│
  │ NFA シミュレーション│ O(n×m)      │ Go, Rust, RE2  │ 安定した性能  │
  │ バックトラック   │ O(2ⁿ) 最悪   │ Python, JS,    │ 後方参照可能  │
  │                  │              │ Java, Ruby     │ ReDoS リスク │
  └──────────────────┴──────────────┴────────────────┴─────────────┘

  n: 入力文字列の長さ  m: 正規表現パターンの長さ

  ReDoS（Regular Expression Denial of Service）:
    バックトラック方式では、特定のパターンと入力の組み合わせで
    指数的な時間がかかる → サービス拒否攻撃に悪用される

    危険なパターン例: (a+)+$
    入力 "aaaaaaaaaaaaaaaaX" で指数的にバックトラック

    対策:
    - NFA ベースのエンジン（RE2, Go の regexp）を使う
    - タイムアウトを設定する
    - 入力長を制限する
    - パターンを見直す（非貪欲量指定子の活用等）
```

---

## 4. 文脈自由文法とプッシュダウンオートマトン

### 4.1 文脈自由文法（CFG）の定義

文脈自由文法は、プログラミング言語の構文定義に広く使われる形式的な記述方法である。

```
文脈自由文法（CFG）の形式的定義:
  G = (V, Σ, R, S)

  V : 変数（非終端記号）の有限集合
  Σ : 終端記号の有限集合（V ∩ Σ = ∅）
  R : 生成規則の有限集合  A → α（A ∈ V, α ∈ (V ∪ Σ)*）
  S : 開始記号  S ∈ V

  「文脈自由」の意味:
    生成規則 A → α において、左辺が単一の変数 A のみ
    → A がどのような文脈（周囲の記号列）に現れても
      同じ規則で書き換えられる
    （対比: 文脈依存文法では αAβ → αγβ の形を許す）
```

**例: 対応する括弧の言語**

```
  L = {aⁿbⁿ | n ≥ 0}  の文脈自由文法:

  G₁ = ({S}, {a, b}, R, S)
  R:
    S → aSb    （a と b を1組追加）
    S → ε      （基底: 空文字列）

  導出の例（w = "aaabbb"）:
    S ⟹ aSb ⟹ aaSbb ⟹ aaaSbbb ⟹ aaabbb

  導出木（構文木）:
           S
         / | \
        a  S  b
         / | \
        a  S  b
         / | \
        a  S  b
           |
           ε
```

**例: 算術式の文法**

```
  G₂ = ({E, T, F}, {+, *, (, ), id}, R, E)
  R:
    E → E + T | T        （式は項の加算）
    T → T * F | F        （項は因子の乗算）
    F → ( E ) | id       （因子は括弧つき式または識別子）

  導出の例（w = "id + id * id"）:
    E ⟹ E + T
      ⟹ T + T
      ⟹ F + T
      ⟹ id + T
      ⟹ id + T * F
      ⟹ id + F * F
      ⟹ id + id * F
      ⟹ id + id * id

  導出木:
            E
          / | \
         E  +  T
         |   / | \
         T  T  *  F
         |  |     |
         F  F    id
         |  |
        id id

  この導出木は演算子の優先順位を反映:
  * は + より優先度が高い → * のノードが木の下位に位置
```

### 4.2 曖昧性（Ambiguity）

```
曖昧な文法（ambiguous grammar）:

  ある文字列 w に対して、2つ以上の異なる導出木が存在する場合、
  その文法は「曖昧」であるという。

  例: 曖昧な式の文法
    E → E + E | E * E | ( E ) | id

  文字列 "id + id * id" に2つの導出木:

  導出木1（+ が先）:        導出木2（* が先）:
        E                         E
      / | \                     / | \
     E  +  E                   E  *  E
     |   / | \               / | \   |
    id  E  *  E             E  +  E  id
        |     |             |     |
       id    id            id    id

  プログラミング言語では曖昧性は致命的:
  - "2 + 3 * 4" が 20 にも 14 にもなりうる
  - 解決方法:
    a. 文法を書き直して曖昧性を除去（上記 G₂ のように）
    b. パーサーに優先順位と結合性のルールを追加

  重要な定理: 与えられた CFG が曖昧かどうかは一般に決定不能
```

### 4.3 チョムスキー標準形と CYK アルゴリズム

```
チョムスキー標準形（Chomsky Normal Form: CNF）:

  すべての生成規則が以下のいずれかの形:
    A → BC    （右辺は2つの変数）
    A → a     （右辺は1つの終端記号）
    S → ε     （開始記号のみ ε を生成可能）

  任意の CFG は CNF に変換可能（空言語を除く）

  CNF の利点:
    - 導出の各ステップで文字列長が高々1増える
    - 長さ n の文字列の導出はちょうど 2n-1 ステップ
    - CYK アルゴリズムによる効率的な構文解析が可能

CYK アルゴリズム（Cocke-Younger-Kasami）:

  CNF の文法に対する O(n³) の構文解析アルゴリズム

  入力: CNF の文法 G と文字列 w = w₁w₂...wₙ
  出力: w ∈ L(G) かどうか

  手法: 動的計画法
    テーブル T[i][j] = {A ∈ V | A ⟹* wᵢwᵢ₊₁...wⱼ}

    基底: T[i][i] = {A | A → wᵢ ∈ R}
    帰納: T[i][j] = ∪{A | A → BC ∈ R,
                          B ∈ T[i][k], C ∈ T[k+1][j],
                          i ≤ k < j}

    判定: S ∈ T[1][n] ならば w ∈ L(G)

  例: G = ({S, A, B}, {a, b}, R, S)
      R: S → AB, A → a, B → b, S → a

      入力 w = "ab"

      T[1][1] = {A | A → a} = {A, S}
      T[2][2] = {B | B → b} = {B}
      T[1][2] = {X | X → YZ, Y ∈ T[1][1], Z ∈ T[2][2]}
              = {S}  （S → AB, A ∈ T[1][1], B ∈ T[2][2]）

      S ∈ T[1][2] → "ab" ∈ L(G) ✓
```

### 4.4 プッシュダウンオートマトン（PDA）

```
PDA の形式的定義:
  P = (Q, Σ, Γ, δ, q₀, Z₀, F)

  Q  : 状態の有限集合
  Σ  : 入力アルファベット
  Γ  : スタックアルファベット
  δ  : 遷移関数  Q × (Σ ∪ {ε}) × Γ → P(Q × Γ*)
       （現在の状態、入力記号、スタックトップ）→（次の状態、スタック操作）
  q₀ : 初期状態
  Z₀ : スタックの初期記号  Z₀ ∈ Γ
  F  : 受理状態の集合

  PDA = 有限オートマトン + スタック（無限の記憶）

  有限オートマトンとの違い:
  ┌──────────────────────┬─────────────────┬──────────────────┐
  │ 特徴                 │ 有限オートマトン │ PDA              │
  ├──────────────────────┼─────────────────┼──────────────────┤
  │ 記憶装置             │ なし（状態のみ） │ スタック（LIFO）  │
  │ 認識する言語クラス   │ 正規言語         │ 文脈自由言語      │
  │ 典型的な用途         │ 字句解析         │ 構文解析          │
  │ 括弧の対応チェック   │ 不可能           │ 可能              │
  └──────────────────────┴─────────────────┴──────────────────┘
```

**例: L = {aⁿbⁿ | n ≥ 0} を受理する PDA**

```
  PDA P₁:
    状態: {q₀, q₁, q₂}
    入力アルファベット: {a, b}
    スタックアルファベット: {Z₀, A}
    初期状態: q₀
    スタック初期記号: Z₀
    受理状態: {q₂}

  遷移規則:
    δ(q₀, a, Z₀) = {(q₀, AZ₀)}    a を読んで A を push
    δ(q₀, a, A)  = {(q₀, AA)}      a を読んで A を push
    δ(q₀, b, A)  = {(q₁, ε)}       b を読んで A を pop
    δ(q₁, b, A)  = {(q₁, ε)}       b を読んで A を pop
    δ(q₁, ε, Z₀) = {(q₂, Z₀)}     スタックが空（Z₀のみ）→ 受理

  トレース（入力 "aabb"）:
    状態    残り入力    スタック
    q₀      aabb        Z₀
    q₀      abb         AZ₀          ← a を読み A を push
    q₀      bb          AAZ₀         ← a を読み A を push
    q₁      b           AZ₀          ← b を読み A を pop
    q₁      ε           Z₀           ← b を読み A を pop
    q₂      ε           Z₀           ← ε遷移で受理状態へ → 受理！

  トレース（入力 "aab"、拒否される例）:
    状態    残り入力    スタック
    q₀      aab         Z₀
    q₀      ab          AZ₀
    q₀      b           AAZ₀
    q₁      ε           AZ₀          ← スタックに A が残る
    → q₂ へ遷移できない → 拒否
```

### 4.5 CFG と PDA の等価性

```
重要な定理:
  言語 L が文脈自由言語である
  ⟺ L を受理する PDA が存在する
  ⟺ L を生成する CFG が存在する

  変換方向:
    CFG → PDA: 各生成規則をスタック操作に変換
    PDA → CFG: PDA の計算を生成規則でシミュレート

  実務での意味:
  - コンパイラの構文解析器（パーサー）は PDA の一種
  - BNF（バッカスナウア記法）は CFG の別表記
  - yacc/bison, ANTLR 等のパーサージェネレータは
    CFG の定義から PDA を自動生成する
```

### 4.6 文脈自由言語のポンピング補題

```
文脈自由言語のポンピング補題:

  L が文脈自由言語ならば、ある定数 p ≥ 1 が存在し、
  |w| ≥ p を満たすすべての w ∈ L について、
  w = uvxyz と分割でき、以下の3条件を満たす:

    (1) |vy| > 0        （v と y の少なくとも一方は空でない）
    (2) |vxy| ≤ p       （中央部分の長さは p 以下）
    (3) すべての i ≥ 0 について uv^i xy^i z ∈ L

  正規言語版との違い:
    正規言語: 3分割 xyz、y をポンプ
    文脈自由言語: 5分割 uvxyz、v と y を同時にポンプ

  直感的理解:
    導出木が十分大きいと、同じ変数が導出経路上に2回現れる。
    その間の部分を繰り返し展開（ポンプ）できる。

             S
            /|\
           u  A  z
             /|\
            v  A  y
              /|\
             x

    A → ... A ... の部分を0回、1回、2回、... 繰り返せる

  例: L = {aⁿbⁿcⁿ | n ≥ 0} が文脈自由言語でないことの証明

    1. L が文脈自由言語だと仮定し、定数 p をとる
    2. w = aᵖbᵖcᵖ を選ぶ（|w| = 3p ≥ p）
    3. w = uvxyz（|vy| > 0, |vxy| ≤ p）と分割
    4. |vxy| ≤ p なので、vxy は a,b,c のうち高々2種類の
       文字しか含まない
    5. i = 2 のとき uv²xy²z は、
       2種類以下の文字の数だけ増えるが残りは変わらない
       → a,b,c の数が等しくなくなる → uv²xy²z ∉ L
    6. 矛盾 → L は文脈自由言語でない  □
```

---

## 5. チョムスキー階層

### 5.1 チョムスキー階層の全体像

チョムスキー階層は、ノーム・チョムスキーが1956年に提唱した、形式文法と形式言語の階層的分類体系である。

```
チョムスキー階層（Chomsky Hierarchy）:

  ┌─────────┬───────────────┬───────────────────┬──────────────────┐
  │ タイプ  │ 文法           │ 認識する機械       │ 言語クラス       │
  ├─────────┼───────────────┼───────────────────┼──────────────────┤
  │ Type 0  │ 無制限文法     │ チューリングマシン │ 帰納的可算言語   │
  │ Type 1  │ 文脈依存文法   │ 線形有界オートマトン│ 文脈依存言語     │
  │ Type 2  │ 文脈自由文法   │ プッシュダウン     │ 文脈自由言語     │
  │         │               │ オートマトン       │                  │
  │ Type 3  │ 正規文法       │ 有限オートマトン   │ 正規言語         │
  └─────────┴───────────────┴───────────────────┴──────────────────┘

  包含関係（真の包含）:
    正規言語 ⊂ 文脈自由言語 ⊂ 文脈依存言語 ⊂ 帰納的可算言語

  各レベルの境界を示す言語の例:
    正規: a*b*（有限オートマトンで認識可能）
    文脈自由だが正規でない: {aⁿbⁿ | n ≥ 0}
    文脈依存だが文脈自由でない: {aⁿbⁿcⁿ | n ≥ 0}
    帰納的可算だが文脈依存でない: 停止するTMの符号化の集合
    帰納的可算ですらない: 停止問題の補集合
```

### 5.2 各タイプの文法規則の形式

```
文法規則の制約による分類:

  Type 0（無制限文法）:
    規則: α → β（α ∈ (V∪Σ)⁺, β ∈ (V∪Σ)*）
    制約: なし（左辺に終端記号も含められる）

  Type 1（文脈依存文法）:
    規則: αAβ → αγβ（A ∈ V, α,β ∈ (V∪Σ)*, γ ∈ (V∪Σ)⁺）
    制約: |α| ≤ |β|（文字列を短くする規則を禁止）
    意味: A の書き換えが周囲の文脈 α, β に依存する

  Type 2（文脈自由文法）:
    規則: A → α（A ∈ V, α ∈ (V∪Σ)*）
    制約: 左辺は単一の変数のみ
    意味: A はどの文脈でも同じ規則で書き換えられる

  Type 3（正規文法）:
    右線形文法: A → aB | a | ε（A,B ∈ V, a ∈ Σ）
    左線形文法: A → Ba | a | ε
    制約: 右辺の変数は1つ以下、かつ端に配置

  制約の強さ: Type 3 > Type 2 > Type 1 > Type 0
  表現力:     Type 3 < Type 2 < Type 1 < Type 0
```

### 5.3 各レベルの決定可能性

```
各言語クラスの決定問題:

  ┌─────────────────────┬────────┬──────────┬──────────┬────────┐
  │ 問題                │ 正規   │ 文脈自由 │ 文脈依存 │ Type 0 │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ 所属問題            │ ○ O(n) │ ○ O(n³)  │ ○        │ ×     │
  │ w ∈ L?              │        │ CYK      │ PSPACE完全│ 半決定 │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ 空問題              │ ○      │ ○        │ ×        │ ×     │
  │ L = ∅?              │        │          │          │        │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ 等価問題            │ ○      │ ×        │ ×        │ ×     │
  │ L₁ = L₂?            │        │ 決定不能 │ 決定不能 │ 決定不能│
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ 包含問題            │ ○      │ ×        │ ×        │ ×     │
  │ L₁ ⊆ L₂?            │        │          │          │        │
  ├─────────────────────┼────────┼──────────┼──────────┼────────┤
  │ 有限性問題          │ ○      │ ○        │ ×        │ ×     │
  │ |L| < ∞?            │        │          │          │        │
  └─────────────────────┴────────┴──────────┴──────────┴────────┘

  ○: 決定可能  ×: 決定不能

  実務的な意味:
  - 正規言語は最も扱いやすい（ほぼすべての問題が決定可能）
  - 文脈自由言語は所属問題・空問題は解けるが等価性は判定不能
  - Type 0 の所属問題すら決定不能（停止問題に帰着）
```

### 5.4 チョムスキー階層と実務の対応

```
チョムスキー階層の実務対応:

  ┌─────────┬──────────────────────────────────────────────────┐
  │ レベル  │ 実務での対応                                      │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 3  │ ・正規表現によるパターンマッチング                 │
  │ 正規    │ ・字句解析器（レクサー/トークナイザー）             │
  │         │ ・入力バリデーション（メールアドレス、電話番号等）   │
  │         │ ・ネットワークプロトコルの状態管理                  │
  │         │ ・設定ファイルのキーワード認識                      │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 2  │ ・プログラミング言語の構文定義（BNF/EBNF）          │
  │ 文脈自由│ ・構文解析器（パーサー: LL, LR, LALR, PEG）       │
  │         │ ・XML/HTML/JSON の構造解析                          │
  │         │ ・数式パーサー                                      │
  │         │ ・括弧の対応チェック                                │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 1  │ ・型チェック（変数の使用が宣言に依存）              │
  │ 文脈依存│ ・スコープ解析                                      │
  │         │ ・C言語の typedef による文脈依存的なパース           │
  │         │ ・自然言語の一部の構文                              │
  ├─────────┼──────────────────────────────────────────────────┤
  │ Type 0  │ ・汎用プログラミング（チューリング完全な計算）      │
  │ 無制限  │ ・意味解析全般                                      │
  │         │ ・停止性判定 → 決定不能の壁                         │
  └─────────┴──────────────────────────────────────────────────┘
```

### 5.5 コンパイラにおけるチョムスキー階層の適用

```
コンパイラのフロントエンドとチョムスキー階層:

  ソースコード
    │
    ▼ ┌──────────────────────────────────────────────────────┐
  字句解析（レクサー）← Type 3: 正規言語（DFA）              │
    │                  トークンを正規表現で定義               │
    │                  例: ID = [a-zA-Z_][a-zA-Z0-9_]*       │
    │                  例: NUM = [0-9]+(\.[0-9]+)?           │
    ▼                                                        │
  トークン列                                                 │
    │                                                        │
    ▼ ┌──────────────────────────────────────────────────────┤
  構文解析（パーサー）← Type 2: 文脈自由言語（PDA）          │
    │                  BNF/EBNF で構文規則を定義              │
    │                  例: if_stmt → 'if' expr 'then' stmt   │
    │                  LL(k), LR(k), LALR(1) 等の手法        │
    ▼                                                        │
  構文木（AST）                                              │
    │                                                        │
    ▼ ┌──────────────────────────────────────────────────────┤
  意味解析 ← Type 1以上: 文脈依存                             │
    │        型チェック、スコープ解析、                       │
    │        オーバーロード解決等                             │
    ▼                                                        │
  注釈付きAST                                                │
    │                                                        │
    ▼                                                        │
  中間コード生成 → 最適化 → コード生成                       │
  └──────────────────────────────────────────────────────────┘

  例: "int x = 3 + 5;" の処理

  字句解析（Type 3）:
    "int" → KW_INT
    "x"   → ID("x")
    "="   → ASSIGN
    "3"   → INT(3)
    "+"   → PLUS
    "5"   → INT(5)
    ";"   → SEMICOLON

  構文解析（Type 2）:
    declaration
    ├── type: int
    ├── name: x
    └── init_expr
        └── binary_op: +
            ├── left: 3
            └── right: 5

  意味解析（Type 1以上）:
    - x の型は int
    - 3 + 5 は int 型の演算 → 型整合 ✓
    - x はこのスコープで未宣言 → 新規バインディング作成
```

---

## 6. 正規表現エンジンの実装

### 6.1 Thompson 構成法による NFA ベースの正規表現エンジン

以下は、Thompson 構成法を用いて正規表現を NFA に変換し、NFA シミュレーションでマッチングを行うエンジンの実装である。

```python
"""
正規表現エンジン（Thompson 構成法 + NFA シミュレーション）

対応する正規表現の構文:
  - 連結:   ab
  - 選択:   a|b
  - 閉包:   a*
  - 1回以上: a+
  - 0回か1回: a?
  - 括弧:   (a|b)*
  - 任意文字: .

内部で NFA に変換し、入力文字列を O(n*m) でマッチングする。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class NFAState:
    """NFA の状態ノード"""
    id: int
    is_accept: bool = False
    # ε遷移先
    epsilon: List["NFAState"] = field(default_factory=list)
    # 文字遷移: {文字: [遷移先状態]}
    transitions: dict = field(default_factory=dict)


class NFAFragment:
    """NFA の断片（開始状態と末尾状態のペア）"""
    def __init__(self, start: NFAState, accept: NFAState):
        self.start = start
        self.accept = accept


class RegexEngine:
    """Thompson 構成法ベースの正規表現エンジン"""

    def __init__(self):
        self._state_counter = 0

    def _new_state(self, is_accept: bool = False) -> NFAState:
        """新しい NFA 状態を生成する。"""
        state = NFAState(id=self._state_counter, is_accept=is_accept)
        self._state_counter += 1
        return state

    # --- パーサー（正規表現 → 構文木） ---

    def _parse(self, pattern: str) -> "NFAFragment":
        """正規表現をパースして NFA を構築する。"""
        self._pos = 0
        self._pattern = pattern
        nfa = self._parse_expr()
        nfa.accept.is_accept = True
        return nfa

    def _parse_expr(self) -> NFAFragment:
        """式: term ('|' term)*"""
        left = self._parse_term()
        while self._pos < len(self._pattern) and self._peek() == "|":
            self._consume("|")
            right = self._parse_term()
            left = self._build_union(left, right)
        return left

    def _parse_term(self) -> NFAFragment:
        """項: factor*"""
        # 空の項（ε に対応）
        if (self._pos >= len(self._pattern)
                or self._peek() in ("|", ")")):
            s = self._new_state()
            return NFAFragment(s, s)

        left = self._parse_factor()
        while (self._pos < len(self._pattern)
               and self._peek() not in ("|", ")")):
            right = self._parse_factor()
            left = self._build_concat(left, right)
        return left

    def _parse_factor(self) -> NFAFragment:
        """因子: atom ('*' | '+' | '?')?"""
        atom = self._parse_atom()
        if self._pos < len(self._pattern):
            if self._peek() == "*":
                self._consume("*")
                return self._build_star(atom)
            elif self._peek() == "+":
                self._consume("+")
                return self._build_plus(atom)
            elif self._peek() == "?":
                self._consume("?")
                return self._build_optional(atom)
        return atom

    def _parse_atom(self) -> NFAFragment:
        """原子: '(' expr ')' | '.' | 文字"""
        if self._peek() == "(":
            self._consume("(")
            nfa = self._parse_expr()
            self._consume(")")
            return nfa
        elif self._peek() == ".":
            self._consume(".")
            return self._build_dot()
        else:
            ch = self._peek()
            self._pos += 1
            return self._build_char(ch)

    def _peek(self) -> str:
        return self._pattern[self._pos]

    def _consume(self, expected: str) -> None:
        if self._pos < len(self._pattern) and self._peek() == expected:
            self._pos += 1
        else:
            raise ValueError(
                f"Expected '{expected}' at position {self._pos}"
            )

    # --- NFA 構築（Thompson 構成法） ---

    def _build_char(self, ch: str) -> NFAFragment:
        """文字リテラルに対応する NFA 断片を構築する。"""
        start = self._new_state()
        accept = self._new_state()
        start.transitions[ch] = [accept]
        return NFAFragment(start, accept)

    def _build_dot(self) -> NFAFragment:
        """任意文字 '.' に対応する NFA 断片を構築する。"""
        start = self._new_state()
        accept = self._new_state()
        # 特殊キー "ANY" で任意文字マッチを表現
        start.transitions["ANY"] = [accept]
        return NFAFragment(start, accept)

    def _build_concat(
        self, left: NFAFragment, right: NFAFragment
    ) -> NFAFragment:
        """連結: left の受理状態から right の開始状態へε遷移。"""
        left.accept.epsilon.append(right.start)
        left.accept.is_accept = False
        return NFAFragment(left.start, right.accept)

    def _build_union(
        self, left: NFAFragment, right: NFAFragment
    ) -> NFAFragment:
        """選択 (|): 新しい開始状態から左右へε遷移。"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([left.start, right.start])
        left.accept.epsilon.append(accept)
        left.accept.is_accept = False
        right.accept.epsilon.append(accept)
        right.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_star(self, inner: NFAFragment) -> NFAFragment:
        """クリーネ閉包 (*): 0回以上の繰り返し。"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([inner.start, accept])
        inner.accept.epsilon.extend([inner.start, accept])
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_plus(self, inner: NFAFragment) -> NFAFragment:
        """1回以上の繰り返し (+): inner · inner*"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.append(inner.start)
        inner.accept.epsilon.extend([inner.start, accept])
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    def _build_optional(self, inner: NFAFragment) -> NFAFragment:
        """0回か1回 (?): inner | ε"""
        start = self._new_state()
        accept = self._new_state()
        start.epsilon.extend([inner.start, accept])
        inner.accept.epsilon.append(accept)
        inner.accept.is_accept = False
        return NFAFragment(start, accept)

    # --- NFA シミュレーション ---

    def _epsilon_closure(self, states: Set[NFAState]) -> Set[NFAState]:
        """ε閉包を計算する。"""
        closure = set(states)
        stack = list(states)
        while stack:
            state = stack.pop()
            for next_state in state.epsilon:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def _step(
        self, states: Set[NFAState], ch: str
    ) -> Set[NFAState]:
        """1文字読んで遷移し、ε閉包を返す。"""
        next_states = set()
        for state in states:
            # 通常の文字遷移
            if ch in state.transitions:
                next_states.update(state.transitions[ch])
            # 任意文字マッチ
            if "ANY" in state.transitions:
                next_states.update(state.transitions["ANY"])
        return self._epsilon_closure(next_states)

    def match(self, pattern: str, text: str) -> bool:
        """
        正規表現パターンがテキスト全体にマッチするか判定する。

        Args:
            pattern: 正規表現パターン
            text: マッチ対象の文字列

        Returns:
            マッチすれば True
        """
        self._state_counter = 0
        nfa = self._parse(pattern)

        current = self._epsilon_closure({nfa.start})
        for ch in text:
            current = self._step(current, ch)
            if not current:
                return False

        return any(s.is_accept for s in current)

    def search(self, pattern: str, text: str) -> Optional[str]:
        """
        テキスト中で最初にマッチする部分文字列を返す。

        Args:
            pattern: 正規表現パターン
            text: 検索対象の文字列

        Returns:
            最初にマッチした部分文字列、なければ None
        """
        self._state_counter = 0
        nfa = self._parse(pattern)

        for start_pos in range(len(text)):
            current = self._epsilon_closure({nfa.start})
            # 開始位置で即座に受理される場合（ε を受理する NFA）
            if any(s.is_accept for s in current):
                return ""
            for end_pos in range(start_pos, len(text)):
                current = self._step(current, text[end_pos])
                if not current:
                    break
                if any(s.is_accept for s in current):
                    return text[start_pos:end_pos + 1]

        return None


# --- 使用例 ---

engine = RegexEngine()

# 基本的なマッチング
test_patterns = [
    ("abc",      "abc",    True),
    ("a|b",      "a",      True),
    ("a|b",      "b",      True),
    ("a|b",      "c",      False),
    ("a*",       "",       True),
    ("a*",       "aaa",    True),
    ("a+",       "",       False),
    ("a+",       "aaa",    True),
    ("ab?c",     "ac",     True),
    ("ab?c",     "abc",    True),
    ("(a|b)*abb", "aabb",  True),
    ("(a|b)*abb", "ab",    False),
    (".+",       "hello",  True),
]

print("=== 正規表現エンジンのテスト ===")
for pattern, text, expected in test_patterns:
    result = engine.match(pattern, text)
    status = "OK" if result == expected else "NG"
    print(f"  [{status}] match('{pattern}', '{text}') = {result}")

# 検索
print("\n=== 部分文字列検索 ===")
found = engine.search("(a|b)+c", "xxxabcyyy")
print(f"  search('(a|b)+c', 'xxxabcyyy') = '{found}'")

# 出力:
# === 正規表現エンジンのテスト ===
#   [OK] match('abc', 'abc') = True
#   [OK] match('a|b', 'a') = True
#   [OK] match('a|b', 'b') = True
#   [OK] match('a|b', 'c') = False
#   [OK] match('a*', '') = True
#   [OK] match('a*', 'aaa') = True
#   [OK] match('a+', '') = False
#   [OK] match('a+', 'aaa') = True
#   [OK] match('ab?c', 'ac') = True
#   [OK] match('ab?c', 'abc') = True
#   [OK] match('(a|b)*abb', 'aabb') = True
#   [OK] match('(a|b)*abb', 'ab') = False
#   [OK] match('.+', 'hello') = True
#
# === 部分文字列検索 ===
#   search('(a|b)+c', 'xxxabcyyy') = 'abc'
```

### 6.2 実装方式の比較と性能特性

```
正規表現エンジンの実装方式比較:

  ┌────────────────────┬────────────────┬──────────────────────┐
  │ 方式               │ 時間計算量     │ 空間計算量           │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ NFA シミュレーション│ O(n × m)       │ O(m)                 │
  │ （上記の実装）     │ n:入力長       │ m:パターンの状態数   │
  │                    │ m:パターン長   │                      │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ DFA 変換後実行     │ O(n)           │ O(2^m) 最悪          │
  │                    │ マッチング最速 │ 状態爆発のリスク     │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ DFA 遅延構築       │ O(n × m) 平均  │ O(m) 〜 O(2^m)      │
  │ （RE2 方式）       │ キャッシュ活用 │ キャッシュサイズ制限  │
  ├────────────────────┼────────────────┼──────────────────────┤
  │ バックトラック     │ O(2^n) 最悪    │ O(n) スタック        │
  │ （Perl/Python方式）│ 後方参照対応   │ 再帰的               │
  └────────────────────┴────────────────┴──────────────────────┘

  選択の指針:
  - 後方参照が不要で安全性重視 → NFA シミュレーション（Go, Rust）
  - パフォーマンス最優先で小さなパターン → DFA 変換
  - 後方参照が必要 → バックトラック（ただし ReDoS に注意）
  - バランス重視 → DFA 遅延構築（RE2）
```

