# 関数型言語比較（Haskell, Elixir, Elm, F#, OCaml）

> 純粋関数型言語は「副作用のない関数」と「不変データ」を中核に据える。バグの少なさ、テストのしやすさ、並行処理の安全性で独自の強みを持つ。

## この章で学ぶこと

- [ ] 主要関数型言語の特徴と適用領域を把握する
- [ ] 純粋関数型と実用的関数型の違いを理解する

---

## 1. 比較表

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Haskell  │ Elixir   │ Elm      │ F#       │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 純粋性        │ 純粋     │ 実用的   │ 純粋     │ 実用的   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 型付け        │ 静的     │ 動的     │ 静的     │ 静的     │
│              │ 型推論最強│          │ 型推論   │ 型推論   │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 実行環境      │ GHC      │ BEAM/OTP │ JS       │ .NET     │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 並行モデル    │ STM      │ アクター │ なし(SPA)│ async    │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 主な用途      │ 研究     │ Web      │ フロント │ バックエンド│
│              │ 金融     │ リアルタイム│ UI      │ データ   │
│              │ コンパイラ│ IoT      │          │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 学習コスト    │ 高い     │ 中程度   │ 低い     │ 中程度   │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. 各言語のHello World + 特徴

```haskell
-- Haskell: 純粋関数型の頂点
-- 副作用はモナド(IO)で明示
main :: IO ()
main = putStrLn "Hello, World!"

-- 型クラス、モナド、遅延評価
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = quicksort smaller ++ [x] ++ quicksort larger
  where smaller = filter (<= x) xs
        larger  = filter (> x) xs
```

```elixir
# Elixir: Erlang VM上の実用的関数型
# パターンマッチ + パイプ演算子 + アクターモデル
defmodule Greeter do
  def hello(name) do
    "Hello, #{name}!"
    |> String.upcase()
    |> IO.puts()
  end
end

# パイプ演算子（|>）でデータ変換を連鎖
"hello world"
|> String.split()
|> Enum.map(&String.capitalize/1)
|> Enum.join(" ")
# → "Hello World"
```

```elm
-- Elm: フロントエンド特化の純粋関数型
-- ランタイムエラーなし（コンパイラが全てキャッチ）
module Main exposing (main)

import Html exposing (text)

main =
    text "Hello, World!"

-- The Elm Architecture (TEA)
-- Model → Update → View のシンプルなパターン
-- React/Redux に影響を与えた
```

```fsharp
// F#: .NET上の実用的関数型
let greet name =
    printfn "Hello, %s!" name

// パイプ演算子
[1; 2; 3; 4; 5]
|> List.filter (fun x -> x % 2 = 0)
|> List.map (fun x -> x * x)
|> List.sum
// → 20
```

---

## 3. 選択指針

```
学術研究・言語設計の学習    → Haskell
リアルタイムWeb・IoT        → Elixir（Phoenix）
フロントエンドの信頼性      → Elm
.NETエコシステムでFP       → F#
堅牢なバックエンド          → Haskell or Elixir
関数型入門                  → Elm（最もシンプル）
```

---

## まとめ

| 言語 | 一言で表すなら | 最適な場面 |
|------|-------------|----------|
| Haskell | 純粋性の極致 | 研究, 金融, コンパイラ |
| Elixir | 実用的FP+並行 | リアルタイムWeb, IoT |
| Elm | ランタイムエラーゼロ | 信頼性の高いUI |
| F# | .NET上のFP | データ処理, バックエンド |
| OCaml | 実用的+高速 | コンパイラ, 金融 |

---

## 次に読むべきガイド
→ [[../07-language-evolution/00-history-of-languages.md]] — 言語の歴史

---

## 参考文献
1. Lipovaca, M. "Learn You a Haskell for Great Good!" 2011.
2. Thomas, D. "Programming Elixir." 2nd Ed, Pragmatic Bookshelf, 2018.
