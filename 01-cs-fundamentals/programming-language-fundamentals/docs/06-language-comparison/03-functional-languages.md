# 関数型言語比較（Haskell, Elixir, Elm, F#, OCaml）

> 純粋関数型言語は「副作用のない関数」と「不変データ」を中核に据える。バグの少なさ、テストのしやすさ、並行処理の安全性で独自の強みを持つ。本ガイドでは、主要5言語（Haskell, Elixir, Elm, F#, OCaml）を多角的に比較し、設計哲学から実践的な選定基準までを網羅的に解説する。

---

## この章で学ぶこと

- [ ] 主要関数型言語 5 つの特徴と適用領域を正確に把握する
- [ ] 純粋関数型と実用的関数型の設計哲学の違いを理解する
- [ ] 型システム・並行モデル・エコシステムの観点で比較できる
- [ ] プロジェクト要件に応じた言語選定が自律的にできる
- [ ] 各言語の代表的イディオムをコードで書ける
- [ ] アンチパターンを認識し回避策を提示できる

---

## 1. 関数型プログラミングの全体像

### 1.1 関数型プログラミングとは何か

関数型プログラミング（Functional Programming, FP）は、計算をラムダ計算（Lambda Calculus）に基づく数学的関数の適用として捉えるパラダイムである。命令型プログラミングが「状態を変化させる手順」を記述するのに対し、関数型プログラミングは「値から値への変換」を宣言的に記述する。

```
┌─────────────────────────────────────────────────────────────────┐
│               関数型プログラミングの系譜                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ラムダ計算 (1930s, Church)                                       │
│       │                                                         │
│       ├── LISP (1958, McCarthy)                                 │
│       │     ├── Scheme (1975)                                   │
│       │     ├── Common Lisp (1984)                              │
│       │     └── Clojure (2007) ─── JVM上の実用的FP              │
│       │                                                         │
│       ├── ML (1973, Milner)                                     │
│       │     ├── Standard ML (1983)                              │
│       │     ├── OCaml (1996) ─── 実用的+高速                    │
│       │     ├── F# (2005) ─── .NET上のFP                        │
│       │     └── Elm (2012) ─── フロントエンド特化                │
│       │                                                         │
│       └── Haskell (1990)                                        │
│             └── 純粋関数型の研究プラットフォーム                  │
│                                                                 │
│  Erlang/OTP (1986, Ericsson)                                    │
│       └── Elixir (2011) ─── BEAM VM上のモダンFP                 │
│                                                                 │
│  [凡例]                                                         │
│   太線: 直接的な系譜  細線: 影響関係                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 FP の中核概念

関数型プログラミングを支える概念は相互に関連しながら、堅牢なソフトウェア設計を可能にする。

```
┌──────────────────────────────────────────────────────────────────┐
│                  FP の中核概念マップ                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│            ┌─────────────┐                                       │
│            │  純粋関数    │ ← 同じ入力 → 同じ出力                │
│            │ (Pure Func)  │   副作用なし                          │
│            └──────┬───────┘                                       │
│                   │                                               │
│         ┌─────────┼──────────┐                                    │
│         ▼         ▼          ▼                                    │
│  ┌──────────┐ ┌────────┐ ┌──────────────┐                        │
│  │ 不変性   │ │参照透過│ │ 高階関数     │                        │
│  │Immutable │ │Referent│ │Higher-Order  │                        │
│  │  Data    │ │Transpar│ │  Functions   │                        │
│  └────┬─────┘ └───┬────┘ └──────┬───────┘                        │
│       │           │             │                                 │
│       ▼           ▼             ▼                                 │
│  ┌─────────┐ ┌────────────┐ ┌──────────────┐                     │
│  │永続データ│ │ 等式推論   │ │ 関数合成     │                     │
│  │構造     │ │ Equational │ │ Composition  │                     │
│  │Persist. │ │ Reasoning  │ │  f . g       │                     │
│  └─────────┘ └────────────┘ └──────────────┘                     │
│                                                                  │
│  ┌───────────────────────────────────────────┐                   │
│  │ 型システム: 正しさをコンパイル時に保証       │                   │
│  │  代数的データ型 / パターンマッチ / 型推論    │                   │
│  └───────────────────────────────────────────┘                   │
│                                                                  │
│  ┌───────────────────────────────────────────┐                   │
│  │ 副作用管理: モナド / エフェクトシステム      │                   │
│  │  IO モナド (Haskell) / Cmd (Elm) / CE (F#) │                   │
│  └───────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

#### 純粋関数（Pure Function）

同じ引数に対して常に同じ結果を返し、外部の状態を変更しない関数。テストが容易で、並列実行が安全である。

#### 不変性（Immutability）

一度生成したデータは変更しない。変更が必要な場合は新しいデータを生成する。競合状態（Race Condition）の根本的な排除につながる。

#### 参照透過性（Referential Transparency）

式を、その式が評価された値に置き換えても、プログラムの意味が変わらない性質。デバッグと推論を大幅に簡易化する。

#### 高階関数（Higher-Order Functions）

関数を引数として受け取る、あるいは関数を戻り値として返す関数。`map`, `filter`, `fold/reduce` が代表例であり、ループ構造を抽象化する。

---

## 2. 主要5言語の詳細比較

### 2.1 総合比較表

```
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│              │ Haskell   │ Elixir    │ Elm       │ F#        │ OCaml     │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 初版         │ 1990      │ 2011      │ 2012      │ 2005      │ 1996      │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 設計者       │ 委員会    │ J.Valim   │ E.Czaplicki│D.Syme    │ X.Leroy   │
│              │ (学術)    │ (個人)    │ (個人)    │ (MS Research)│(INRIA) │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 純粋性       │ 純粋      │ 実用的    │ 純粋      │ 実用的    │ 実用的    │
│              │           │ (副作用   │           │ (副作用   │ (副作用   │
│              │           │  自由)    │           │  可能)    │  可能)    │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 型付け       │ 静的      │ 動的      │ 静的      │ 静的      │ 静的      │
│              │ HM型推論  │ (型なし)  │ HM型推論  │ HM型推論  │ HM型推論  │
│              │ 型クラス  │           │           │ 型プロバイダ│ モジュール│
│              │ GADTs     │           │           │           │ ファンクタ│
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 評価戦略     │ 遅延評価  │ 正格評価  │ 正格評価  │ 正格評価  │ 正格評価  │
│              │ (lazy)    │ (eager)   │ (eager)   │ (eager)   │ (eager)   │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 実行環境     │ GHC       │ BEAM/OTP  │ JavaScript│ .NET CLR  │ ネイティブ│
│              │ (ネイティブ)│ (VM)    │ (コンパイル)│ (JIT)   │ (バイトコード│
│              │           │           │           │           │  も可能)  │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 並行モデル   │ STM       │ アクター  │ なし(SPA) │ async/    │ Multicore │
│              │ (Software │ (OTP      │ Cmd で    │ Task      │ Domain    │
│              │  Trans.   │  Supervisor│外部通信  │ Parallel  │ (5.0+)    │
│              │  Memory)  │  Tree)    │           │           │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ パッケージ   │ Hackage   │ Hex       │ elm-      │ NuGet     │ opam      │
│ マネージャ   │ /Stackage │           │ packages  │           │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ ビルドツール │ cabal/    │ mix       │ elm make  │ dotnet    │ dune      │
│              │ stack     │           │           │           │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 主な用途     │ 研究      │ Web       │ フロント  │ バックエンド│コンパイラ │
│              │ 金融      │ リアルタイム│ UI       │ データ処理│ 金融      │
│              │ コンパイラ│ IoT       │           │ クラウド  │ システム  │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 学習コスト   │ 高い      │ 中程度    │ 低い      │ 中程度    │ 中〜高    │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 企業採用     │ Meta      │ Discord   │ NoRedInk  │ Microsoft │ Jane Street│
│ 事例         │ (Sigma)   │ Pinterest │ Vendr     │ Jet.com   │ Tezos     │
│              │ Standard  │ Bleacher  │           │ Walmart   │ Bloomberg │
│              │ Chartered │ Report    │           │           │ Docker    │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ コミュニティ │ 中規模    │ 大規模    │ 小規模    │ 中規模    │ 中規模    │
│ 活発度       │ 学術寄り  │ 活発      │ ニッチ    │ MS支援    │ 成長中    │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

### 2.2 パフォーマンス特性比較表

```
┌──────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ 特性         │ Haskell   │ Elixir    │ Elm       │ F#        │ OCaml     │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 起動時間     │ 速い      │ やや遅い  │ N/A(JS)   │ やや遅い  │ 非常に速い│
│              │ (ネイティブ)│(VM起動)  │           │ (CLR起動) │ (ネイティブ)│
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ スループット │ 高い      │ 中程度    │ ブラウザ  │ 高い      │ 非常に高い│
│              │           │ (BEAM    │ 依存      │           │           │
│              │           │  オーバー │           │           │           │
│              │           │  ヘッド)  │           │           │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ メモリ効率   │ 中程度    │ 中程度    │ ブラウザ  │ 高い      │ 非常に高い│
│              │ (遅延評価 │ (プロセス │ 依存      │ (.NET GC) │ (効率的GC)│
│              │  のリスク)│  軽量)    │           │           │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ レイテンシ   │ 中程度    │ 低い      │ 低い      │ 中程度    │ 低い      │
│              │ (GC一時停止│(SLA向き) │ (仮想DOM) │ (GC一時停止│           │
│              │  あり)    │           │           │  あり)    │           │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 並行性能     │ 良い(STM) │ 非常に高い│ N/A       │ 良い      │ 良い      │
│              │           │(数百万    │           │ (Task     │ (Domain   │
│              │           │ プロセス) │           │  並列)    │  5.0+)    │
├──────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ コンパイル   │ 遅い      │ 速い      │ 速い      │ 速い      │ 非常に速い│
│ 速度         │           │           │           │           │           │
└──────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
```

---

## 3. 各言語の深掘り

### 3.1 Haskell ── 純粋関数型の金字塔

#### 設計哲学

Haskell は 1990 年、分散していた関数型言語研究を統一するオープンスタンダードとして誕生した。「避けるべきことを避ける（Avoid success at all costs）」という非公式モットーは、実用的妥協より理論的正しさを優先する姿勢を表している。

#### 型システム

Haskell の型システムは Hindley-Milner 型推論をベースに、型クラス、GADTs（一般化代数的データ型）、型族（Type Families）、依存型に近い機能（DataKinds, TypeFamilies）まで拡張されている。

```haskell
-- Haskell: 型クラスによる多相性（ポリモーフィズム）
-- 型クラスはインタフェースに似た概念だが、事後的に追加可能

class Describable a where
  describe :: a -> String

-- 代数的データ型の定義
data Shape
  = Circle Double          -- 半径
  | Rectangle Double Double -- 幅と高さ
  | Triangle Double Double Double  -- 三辺
  deriving (Show, Eq)

-- 型クラスインスタンスの実装
instance Describable Shape where
  describe (Circle r) =
    "半径 " ++ show r ++ " の円（面積: " ++ show (pi * r * r) ++ "）"
  describe (Rectangle w h) =
    show w ++ " x " ++ show h ++ " の矩形（面積: " ++ show (w * h) ++ "）"
  describe (Triangle a b c) =
    "三辺 " ++ show a ++ ", " ++ show b ++ ", " ++ show c ++ " の三角形"

-- 型クラス制約を利用した汎用関数
describeAll :: Describable a => [a] -> [String]
describeAll = map describe

-- 使用例
-- describeAll [Circle 5.0, Rectangle 3.0 4.0]
-- → ["半径 5.0 の円（面積: 78.53...）", "3.0 x 4.0 の矩形（面積: 12.0）"]
```

#### モナドと副作用管理

Haskell の最大の特徴は、副作用を型レベルで管理するモナドシステムである。

```haskell
-- Haskell: モナドによる副作用管理の例
-- Maybe モナド: 失敗する可能性のある計算の連鎖

import qualified Data.Map as Map

type UserName = String
type Email = String
type UserId = Int

-- ユーザーデータベース（模擬）
userDb :: Map.Map UserId UserName
userDb = Map.fromList [(1, "alice"), (2, "bob"), (3, "charlie")]

emailDb :: Map.Map UserName Email
emailDb = Map.fromList
  [ ("alice", "alice@example.com")
  , ("bob", "bob@example.com")
  ]

-- Maybe モナドによるチェイン
-- 各ステップが失敗する可能性があるが、do記法で直線的に書ける
lookupEmail :: UserId -> Maybe Email
lookupEmail uid = do
  name  <- Map.lookup uid userDb     -- ユーザー名を検索
  email <- Map.lookup name emailDb   -- メールアドレスを検索
  return email

-- lookupEmail 1  → Just "alice@example.com"
-- lookupEmail 2  → Just "bob@example.com"
-- lookupEmail 3  → Nothing  (charlieのメールが未登録)
-- lookupEmail 99 → Nothing  (ユーザーIDが存在しない)

-- IO モナド: 現実世界との対話
-- IO型がついた関数だけが副作用を持てる
main :: IO ()
main = do
  putStrLn "ユーザーIDを入力してください:"
  input <- getLine
  let uid = read input :: UserId
  case lookupEmail uid of
    Just email -> putStrLn ("メールアドレス: " ++ email)
    Nothing    -> putStrLn "メールアドレスが見つかりません"
```

#### 遅延評価

Haskell はデフォルトで遅延評価（Lazy Evaluation）を採用する。値は実際に必要になるまで評価されない。

```haskell
-- Haskell: 遅延評価の威力
-- 無限リストが自然に扱える

-- フィボナッチ数列（無限リスト）
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

-- 必要な分だけ取得
-- take 10 fibs → [0,1,1,2,3,5,8,13,21,34]

-- 素数の無限リスト（エラトステネスの篩）
primes :: [Integer]
primes = sieve [2..]
  where
    sieve (p:xs) = p : sieve [x | x <- xs, x `mod` p /= 0]

-- take 20 primes → [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]

-- 遅延評価による効率的なデータ処理
-- 巨大リストでも最初のN個だけ評価される
firstEvenSquareOver100 :: Maybe Integer
firstEvenSquareOver100 =
  let squares = map (^2) [1..]           -- 無限の平方数
      evenSquares = filter even squares    -- 偶数の平方数だけ
      overHundred = filter (> 100) evenSquares  -- 100超のもの
  in case overHundred of
       (x:_) -> Just x
       []    -> Nothing
-- → Just 144 (12^2)
```

### 3.2 Elixir ── BEAM VM 上の実用的関数型

#### 設計哲学

Elixir は 2011 年に Jose Valim によって設計された。Ruby のような親しみやすい文法と、Erlang/OTP の堅牢な並行処理基盤を融合した言語である。「Let it crash（クラッシュさせよ）」哲学により、障害を前提としたシステム設計を推奨する。

#### パターンマッチとパイプ演算子

```elixir
# Elixir: パターンマッチの多彩な活用

# 関数の引数でパターンマッチ
defmodule UserParser do
  # JSON風マップからユーザー情報を抽出
  def parse(%{"name" => name, "age" => age}) when is_binary(name) and age >= 0 do
    {:ok, %{name: name, age: age}}
  end

  def parse(%{"name" => name}) when is_binary(name) do
    {:ok, %{name: name, age: :unknown}}
  end

  def parse(_invalid) do
    {:error, :invalid_format}
  end

  # パイプ演算子でデータ変換パイプラインを構築
  def process_users(raw_data) do
    raw_data
    |> Enum.map(&parse/1)                          # 各要素をパース
    |> Enum.filter(fn {:ok, _} -> true; _ -> false end)  # 成功のみ残す
    |> Enum.map(fn {:ok, user} -> user end)        # ユーザーデータを抽出
    |> Enum.sort_by(& &1.name)                     # 名前順ソート
    |> Enum.map(&format_user/1)                    # フォーマット
  end

  defp format_user(%{name: name, age: :unknown}) do
    "#{name} (年齢不明)"
  end

  defp format_user(%{name: name, age: age}) do
    "#{name} (#{age}歳)"
  end
end

# 使用例
# data = [
#   %{"name" => "Alice", "age" => 30},
#   %{"name" => "Bob"},
#   %{"invalid" => true},
#   %{"name" => "Charlie", "age" => 25}
# ]
# UserParser.process_users(data)
# → ["Alice (30歳)", "Bob (年齢不明)", "Charlie (25歳)"]
```

#### OTP による耐障害性

```elixir
# Elixir: GenServer + Supervisor による耐障害設計

defmodule Counter do
  use GenServer

  # クライアントAPI
  def start_link(initial_value) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment, do: GenServer.cast(__MODULE__, :increment)
  def decrement, do: GenServer.cast(__MODULE__, :decrement)
  def value,     do: GenServer.call(__MODULE__, :value)

  # サーバーコールバック
  @impl true
  def init(initial_value), do: {:ok, initial_value}

  @impl true
  def handle_cast(:increment, state), do: {:noreply, state + 1}
  def handle_cast(:decrement, state), do: {:noreply, state - 1}

  @impl true
  def handle_call(:value, _from, state), do: {:reply, state, state}
end

defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      {Counter, 0}  # 初期値0でカウンターを起動
    ]

    # one_for_one: 子プロセスがクラッシュしたら、そのプロセスだけ再起動
    Supervisor.init(children, strategy: :one_for_one)
  end
end

# Counter がクラッシュしても Supervisor が自動で再起動する
# これが "Let it crash" 哲学の具体的な実装
```

### 3.3 Elm ── フロントエンド特化の純粋関数型

#### 設計哲学

Elm は 2012 年に Evan Czaplicki の論文から生まれた。「ランタイムエラーをゼロにする」という大胆な目標を掲げ、Web フロントエンド開発に特化した純粋関数型言語である。The Elm Architecture（TEA）は後に React/Redux に影響を与えた。

#### The Elm Architecture（TEA）

```
┌──────────────────────────────────────────────────────────┐
│               The Elm Architecture (TEA)                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐   update    ┌─────────┐   view            │
│   │  Model  │ ──────────► │  Model' │ ──────────┐       │
│   │ (状態)  │             │ (新状態) │           │       │
│   └─────────┘             └─────────┘           │       │
│        ▲                                        ▼       │
│        │                                  ┌─────────┐   │
│        │         Msg (メッセージ)          │  View   │   │
│        └──────────────────────────────── │ (HTML)   │   │
│                                          └─────────┘   │
│                                               │         │
│   ┌─────────────────────────────────────┐    │         │
│   │           Elm Runtime               │    │         │
│   │  ・仮想DOMの差分計算                 │◄───┘         │
│   │  ・バッチ更新                       │               │
│   │  ・副作用の実行（Cmd/Sub）          │               │
│   └─────────────────────────────────────┘               │
│                                                          │
│   [データフロー]                                         │
│   User Event → Msg → update(Model, Msg) → Model'        │
│   → view(Model') → Virtual DOM → Real DOM               │
└──────────────────────────────────────────────────────────┘
```

```elm
-- Elm: カウンターアプリ（TEAの完全な実装例）

module Main exposing (main)

import Browser
import Html exposing (Html, button, div, text, h1, p)
import Html.Attributes exposing (style)
import Html.Events exposing (onClick)


-- MODEL: アプリケーションの状態

type alias Model =
    { count : Int
    , history : List Int
    }

init : Model
init =
    { count = 0
    , history = []
    }


-- UPDATE: 状態遷移ロジック

type Msg
    = Increment
    | Decrement
    | Reset
    | Double

update : Msg -> Model -> Model
update msg model =
    case msg of
        Increment ->
            { model
                | count = model.count + 1
                , history = model.count :: model.history
            }

        Decrement ->
            { model
                | count = model.count - 1
                , history = model.count :: model.history
            }

        Reset ->
            { model
                | count = 0
                , history = model.count :: model.history
            }

        Double ->
            { model
                | count = model.count * 2
                , history = model.count :: model.history
            }


-- VIEW: 状態を HTML に変換する純粋関数

view : Model -> Html Msg
view model =
    div []
        [ h1 [] [ text "Elm カウンター" ]
        , p [] [ text ("現在値: " ++ String.fromInt model.count) ]
        , button [ onClick Decrement ] [ text "-1" ]
        , button [ onClick Increment ] [ text "+1" ]
        , button [ onClick Double ] [ text "x2" ]
        , button [ onClick Reset ] [ text "リセット" ]
        , p [] [ text ("履歴: " ++ historyToString model.history) ]
        ]

historyToString : List Int -> String
historyToString history =
    history
        |> List.reverse
        |> List.map String.fromInt
        |> String.join " -> "


-- MAIN: アプリケーションのエントリポイント

main : Program () Model Msg
main =
    Browser.sandbox
        { init = init
        , update = update
        , view = view
        }
```

### 3.4 F# ── .NET エコシステム上の実用的関数型

#### 設計哲学

F# は 2005 年に Microsoft Research の Don Syme によって開発された。OCaml の影響を強く受けつつ、.NET エコシステムとの完全な相互運用性を実現している。「実用性と関数型の正しさを両立する」言語として、企業での採用が進んでいる。

#### 型プロバイダと計算式

```fsharp
// F#: 代数的データ型とパターンマッチ

// 判別共用体（Discriminated Union）
type Shape =
    | Circle of radius: float
    | Rectangle of width: float * height: float
    | Triangle of base_: float * height: float

// パターンマッチによる面積計算
let area shape =
    match shape with
    | Circle r -> System.Math.PI * r * r
    | Rectangle (w, h) -> w * h
    | Triangle (b, h) -> 0.5 * b * h

// パイプ演算子によるデータ変換
let processShapes shapes =
    shapes
    |> List.map (fun s -> (s, area s))     // 面積を計算
    |> List.sortBy snd                      // 面積順ソート
    |> List.filter (fun (_, a) -> a > 10.0) // 面積10超のみ
    |> List.map (fun (s, a) ->
        sprintf "%A: %.2f" s a)             // 文字列フォーマット

// 計算式（Computation Expressions）
// モナド的な計算を読みやすく書ける F# 独自の機能
type MaybeBuilder() =
    member _.Bind(x, f) =
        match x with
        | Some v -> f v
        | None -> None
    member _.Return(x) = Some x
    member _.ReturnFrom(x) = x
    member _.Zero() = None

let maybe = MaybeBuilder()

// 使用例: Haskell の do記法に相当
let safeDivide x y =
    maybe {
        if y = 0.0 then
            return! None
        else
            return x / y
    }

let calculateResult () =
    maybe {
        let! a = safeDivide 100.0 5.0    // Some 20.0
        let! b = safeDivide a 4.0        // Some 5.0
        let! c = safeDivide b 0.0        // None → 全体が None
        return c
    }
// calculateResult() → None
```

#### 非同期プログラミング

```fsharp
// F#: 非同期ワークフロー

open System.Net.Http

// 非同期計算式（async CE）
let fetchUrlAsync (url: string) =
    async {
        use client = new HttpClient()
        let! response = client.GetStringAsync(url) |> Async.AwaitTask
        return response.Length
    }

// 複数URLの並列フェッチ
let fetchMultiple urls =
    urls
    |> List.map fetchUrlAsync
    |> Async.Parallel        // 並列実行
    |> Async.RunSynchronously

// Active Patterns: F#独自の強力なパターンマッチ拡張
let (|Even|Odd|) n =
    if n % 2 = 0 then Even else Odd

let (|Positive|Negative|Zero|) n =
    if n > 0 then Positive
    elif n < 0 then Negative
    else Zero

let describeNumber n =
    match n with
    | Positive & Even -> sprintf "%d は正の偶数" n
    | Positive & Odd  -> sprintf "%d は正の奇数" n
    | Negative        -> sprintf "%d は負の数" n
    | Zero            -> "ゼロ"
```

### 3.5 OCaml ── 実用性と高性能の融合

#### 設計哲学

OCaml（Objective Caml）は 1996 年にフランスの INRIA で開発された。ML ファミリーの中でも特に実用性を重視し、ネイティブコンパイルによる高いパフォーマンスと、モジュールシステムの強力さで知られる。Jane Street（金融取引会社）が全面的に採用していることでも有名である。

#### モジュールシステムとファンクタ

```ocaml
(* OCaml: モジュールとファンクタ *)

(* モジュール型（シグネチャ）の定義 *)
module type COMPARABLE = sig
  type t
  val compare : t -> t -> int
  val to_string : t -> string
end

(* ファンクタ: モジュールを引数に取り、新しいモジュールを返す *)
module MakeSortedSet (Item : COMPARABLE) = struct
  type element = Item.t
  type t = element list

  let empty = []

  let rec insert x = function
    | [] -> [x]
    | (h :: _) as l when Item.compare x h < 0 -> x :: l
    | h :: t when Item.compare x h > 0 -> h :: insert x t
    | l -> l  (* x = h の場合、重複を許さない *)

  let rec member x = function
    | [] -> false
    | h :: _ when Item.compare x h = 0 -> true
    | h :: t when Item.compare x h > 0 -> member x t
    | _ -> false

  let to_list s = s

  let to_string s =
    let elements = List.map Item.to_string s in
    "{" ^ String.concat ", " elements ^ "}"
end

(* 具体的なモジュールを作る *)
module IntItem : COMPARABLE with type t = int = struct
  type t = int
  let compare = compare
  let to_string = string_of_int
end

module IntSortedSet = MakeSortedSet(IntItem)

(* 使用例 *)
let example =
  IntSortedSet.empty
  |> IntSortedSet.insert 5
  |> IntSortedSet.insert 3
  |> IntSortedSet.insert 7
  |> IntSortedSet.insert 3  (* 重複は無視される *)
  |> IntSortedSet.to_string
(* → "{3, 5, 7}" *)
```

#### OCaml 5.0 のマルチコア対応

```ocaml
(* OCaml 5.0+: エフェクトハンドラとドメイン *)

(* ドメインによる並列計算 *)
let parallel_map f lst =
  let n = List.length lst in
  let results = Array.make n None in
  let domains = List.mapi (fun i x ->
    Domain.spawn (fun () ->
      results.(i) <- Some (f x)
    )
  ) lst in
  List.iter Domain.join domains;
  Array.to_list (Array.map (fun x ->
    match x with Some v -> v | None -> failwith "unreachable"
  ) results)

(* 使用例 *)
(* let heavy_compute x = (* 重い計算 *) x * x *)
(* let results = parallel_map heavy_compute [1; 2; 3; 4; 5; 6; 7; 8] *)

(* パターンマッチの網羅性チェック *)
type color = Red | Green | Blue

let color_to_string = function
  | Red   -> "赤"
  | Green -> "緑"
  | Blue  -> "青"
  (* コンパイラが全パターンの網羅を検証する *)
  (* パターンが不足していれば警告が出る *)
```

---

## 4. クロスカッティング比較

### 4.1 同一問題の実装比較: フィボナッチ数列

各言語のイディオムの違いを、同一の問題（フィボナッチ数列）で比較する。

```haskell
-- Haskell: 遅延評価 + 無限リスト
fibs :: [Integer]
fibs = 0 : 1 : zipWith (+) fibs (tail fibs)

nthFib :: Int -> Integer
nthFib n = fibs !! n
```

```elixir
# Elixir: Stream（遅延列挙）+ パイプ
defmodule Fibonacci do
  def stream do
    Stream.unfold({0, 1}, fn {a, b} -> {a, {b, a + b}} end)
  end

  def nth(n) do
    stream() |> Enum.at(n)
  end

  def first(n) do
    stream() |> Enum.take(n)
  end
end

# Fibonacci.first(10) → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```elm
-- Elm: 再帰 + タプル（無限リストなし）
fibonacci : Int -> List Int
fibonacci n =
    let
        helper i (a, b) acc =
            if i >= n then
                List.reverse acc
            else
                helper (i + 1) (b, a + b) (a :: acc)
    in
    helper 0 (0, 1) []

-- fibonacci 10 → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```fsharp
// F#: Seq（遅延シーケンス）+ パイプ
let fibs =
    Seq.unfold (fun (a, b) -> Some(a, (b, a + b))) (0, 1)

let nthFib n = fibs |> Seq.item n
let firstFibs n = fibs |> Seq.take n |> Seq.toList

// firstFibs 10 → [0; 1; 1; 2; 3; 5; 8; 13; 21; 34]
```

```ocaml
(* OCaml: Seq（遅延シーケンス）+ パイプ *)
let fibs =
  let rec aux a b () =
    Seq.Cons (a, aux b (a + b))
  in
  aux 0 1

let nth_fib n = fibs |> Seq.drop n |> Seq.uncons |> Option.map fst
let first_fibs n = fibs |> Seq.take n |> List.of_seq

(* first_fibs 10 → [0; 1; 1; 2; 3; 5; 8; 13; 21; 34] *)
```

### 4.2 エラーハンドリング比較

| 言語 | 主なエラー型 | パターン | 特徴 |
|------|-------------|---------|------|
| Haskell | `Maybe a`, `Either e a` | モナドバインド(`>>=`, `do`) | 型レベルで失敗を表現。`IO` 内で例外も可能 |
| Elixir | `{:ok, val}` / `{:error, reason}` | `case`/`with`/パターンマッチ | タプルベースの慣習。`try/rescue`も利用可能 |
| Elm | `Maybe a`, `Result err val` | `case`式 | 例外機構なし。全てのエラーが型で表現される |
| F# | `Option<'T>`, `Result<'T,'E>` | 計算式/パターンマッチ | .NETの例外も利用可能だがFP的にはResult推奨 |
| OCaml | `option`, `result` | パターンマッチ/`Result.bind` | 例外も健在。パフォーマンス重視なら例外も選択肢 |

### 4.3 型システムの表現力比較

```
┌────────────────────────────────────────────────────────────────────┐
│                    型システムの表現力スペクトラム                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  弱い型付け ◄──────────────────────────────────────► 強い型付け    │
│                                                                    │
│  Elixir        Elm     OCaml      F#           Haskell             │
│  (動的型付け)  (HM)    (HM+       (HM+         (HM+型クラス+      │
│               　       モジュール  型プロバイダ  GADTs+             │
│                        ファンクタ) Active Pat.) 型族+DataKinds)    │
│                                                                    │
│  [表現可能な型の例]                                                │
│                                                                    │
│  代数的データ型    : Elm, OCaml, F#, Haskell   ← 全静的型言語     │
│  パラメトリック多相: Elm, OCaml, F#, Haskell   ← 全静的型言語     │
│  型クラス/トレイト : Haskell                    ← Haskell独自     │
│  高カインド型      : Haskell                    ← Haskell独自     │
│  型族(Type Family) : Haskell                    ← Haskell独自     │
│  モジュールファンクタ: OCaml                    ← OCaml独自       │
│  型プロバイダ      : F#                         ← F#独自          │
│  行多相(Row Poly.) : OCaml (オブジェクト)       ← OCaml独自       │
│                                                                    │
│  ※ Elixir は動的型付けだが、Dialyzer による事後的型検査が可能     │
│  ※ Elixir の型仕様: @spec, @type によるドキュメント的型注釈       │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. 副作用管理の比較

関数型言語における副作用の管理方法は、各言語の哲学を最も如実に反映する領域である。

### 5.1 副作用管理の戦略

```
┌───────────────────────────────────────────────────────────────────────┐
│                    副作用管理の戦略比較                               │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [Haskell] IO モナド + モナド変換子                                   │
│  ┌─────────────────────────────┐                                      │
│  │ 純粋な世界      │ IO の世界  │                                      │
│  │ ─────────────── │ ────────── │                                      │
│  │ 全ての関数は    │ IOモナド内 │                                      │
│  │ 副作用なし      │ でのみ副作用│                                     │
│  │                 │ が可能     │                                      │
│  │ f :: a -> b     │ g :: IO () │                                      │
│  └─────────────────────────────┘                                      │
│   → 最も厳格。副作用の有無が型に現れる                               │
│                                                                       │
│  [Elm] Cmd/Sub アーキテクチャ                                        │
│  ┌─────────────────────────────┐                                      │
│  │ アプリコード │ Elm Runtime   │                                      │
│  │ ──────────── │ ───────────── │                                      │
│  │ 完全に純粋   │ 副作用を実行  │                                      │
│  │ Cmd を返す   │ 結果をMsgで   │                                      │
│  │ だけ         │ フィードバック│                                      │
│  └─────────────────────────────┘                                      │
│   → ランタイムが副作用を代行。コードは常に純粋                       │
│                                                                       │
│  [Elixir] プロセス分離                                               │
│  ┌─────────────────────────────┐                                      │
│  │ 各プロセスが独立した状態     │                                      │
│  │ メッセージパッシングで通信   │                                      │
│  │ 副作用は自由だが、プロセス   │                                      │
│  │ 境界で隔離される             │                                      │
│  └─────────────────────────────┘                                      │
│   → 言語レベルでは制限なし。アーキテクチャで管理                     │
│                                                                       │
│  [F#] 慣習ベース + 計算式                                            │
│  ┌─────────────────────────────┐                                      │
│  │ 副作用は技術的に自由だが     │                                      │
│  │ Result/Async計算式で         │                                      │
│  │ 明示的に扱う慣習             │                                      │
│  └─────────────────────────────┘                                      │
│   → 自由度が高い。規律はチームに委ねられる                           │
│                                                                       │
│  [OCaml] 慣習ベース + エフェクト(5.0+)                               │
│  ┌─────────────────────────────┐                                      │
│  │ 従来: 副作用は自由           │                                      │
│  │ 5.0+: エフェクトハンドラで   │                                      │
│  │ 構造的に副作用を管理可能     │                                      │
│  └─────────────────────────────┘                                      │
│   → 進化中。エフェクトシステムが新たな選択肢に                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 6. 並行・並列プログラミング比較

### 6.1 並行モデルの詳細

各言語の並行処理モデルは、その実行環境と密接に関連する。

| 比較項目 | Haskell (STM) | Elixir (Actor) | F# (Async) | OCaml (Domain) |
|---------|---------------|----------------|------------|----------------|
| 基本単位 | 軽量スレッド | プロセス | async計算 | ドメイン |
| 共有状態 | STM (トランザクショナルメモリ) | なし (メッセージパッシング) | 可能だが非推奨 | 可能 (注意が必要) |
| スケーラビリティ | 数千スレッド | 数百万プロセス | OS スレッド依存 | コア数に比例 |
| 障害回復 | 例外ハンドリング | Supervisor Tree | try/with | 例外ハンドリング |
| GC の影響 | あり (一時停止) | 最小 (プロセスごとGC) | あり (.NET GC) | 改善中 (5.0+) |
| 適用場面 | 共有メモリの整合性 | 大規模分散システム | I/O バウンド処理 | CPU バウンド並列 |

### 6.2 Elixir の Supervisor Tree

```
┌──────────────────────────────────────────────────────────────┐
│              Elixir Supervisor Tree の例                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    Application                               │
│                        │                                     │
│                   TopSupervisor                               │
│                   (one_for_one)                               │
│                  ┌─────┼─────────┐                           │
│                  │     │         │                           │
│             WebSupervisor  DBPool  CacheServer               │
│             (one_for_all)    │                               │
│              ┌────┼────┐     │                               │
│              │    │    │     │                               │
│          Endpoint Router  Static                             │
│              │                                               │
│              │                                               │
│  [再起動戦略]                                                │
│  one_for_one : 落ちた子だけ再起動                            │
│  one_for_all : 1つ落ちたら全子を再起動                       │
│  rest_for_one: 落ちた子以降の子を再起動                      │
│                                                              │
│  [障害伝播の例]                                              │
│  Router がクラッシュ                                         │
│  → WebSupervisor が検知 (one_for_all)                       │
│  → Endpoint, Router, Static を全て再起動                     │
│  → TopSupervisor には影響なし                                │
│  → DBPool, CacheServer は稼働し続ける                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. エコシステムとツーリング

### 7.1 開発体験比較

| 観点 | Haskell | Elixir | Elm | F# | OCaml |
|------|---------|--------|-----|------|-------|
| IDE サポート | HLS (良好) | ElixirLS (良好) | elm-language-server (良好) | Ionide (優秀) | ocaml-lsp (良好) |
| REPL | GHCi (強力) | IEx (非常に良い) | elm repl (基本的) | F# Interactive (良好) | utop (優秀) |
| テストFW | HSpec, QuickCheck | ExUnit, StreamData | elm-test | Expecto, FsCheck | Alcotest, QCheck |
| フォーマッタ | ormolu, fourmolu | mix format (公式) | elm-format (公式) | fantomas | ocamlformat |
| ドキュメント | Haddock | ExDoc (優秀) | パッケージサイト | XML Doc | odoc |
| デバッグ | GHCi, Debug.Trace | IEx.pry, Observer | Debug.log, elm reactor | VS debugger | ocamldebug |
| プロパティテスト | QuickCheck (元祖) | StreamData | elm-test (組込み) | FsCheck | QCheck |

---

## 8. プロジェクト選定ガイド

### 8.1 意思決定フローチャート

```
┌─────────────────────────────────────────────────────────────────┐
│              関数型言語 選定フローチャート                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Q1: 主な対象ドメインは？                                       │
│  ┌──────┬──────────┬──────────┬──────────┬──────────┐           │
│  │フロント│リアルタイム│.NET環境  │研究/金融 │高性能    │           │
│  │エンド │Web/IoT   │         │コンパイラ│システム  │           │
│  └──┬───┘└──┬───────┘└──┬──────┘└──┬──────┘└──┬──────┘           │
│     │       │           │          │          │                 │
│     ▼       ▼           ▼          ▼          ▼                 │
│   Elm    Elixir        F#       Haskell    OCaml               │
│     │       │           │          │          │                 │
│     │   Q2: 同時接続数は？ │     Q3: 型の    Q4: コンパイル      │
│     │   ┌───┴───┐       │     厳密性は？  速度は重要？         │
│     │ 数千以下 数万以上  │     ┌──┴──┐    ┌──┴──┐              │
│     │   │       │       │    中程度 最高  はい  いいえ          │
│     │   │    Elixir     │     │     │     │     │              │
│     │   │  (Phoenix)    │    F#  Haskell OCaml Haskell         │
│     │   │               │                                       │
│     │   └──→ F# or      │                                       │
│     │       Haskellも可 │                                       │
│     │                    │                                       │
│  [判断補助]                                                     │
│  ・チームが Ruby/Python 経験者中心 → Elixir                    │
│  ・チームが C#/Java 経験者中心     → F#                        │
│  ・チームが学術的背景あり          → Haskell or OCaml          │
│  ・チームが JS/TS 経験者中心       → Elm                       │
│  ・採用のしやすさを重視            → Elixir (コミュニティ大)   │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 ユースケース別推奨言語

| ユースケース | 第1推奨 | 第2推奨 | 理由 |
|-------------|---------|---------|------|
| リアルタイムチャット | Elixir | Haskell | BEAMの数百万プロセスが最適 |
| 金融取引システム | OCaml | Haskell | 低レイテンシ + 型安全性 |
| コンパイラ開発 | Haskell | OCaml | 代数的データ型 + パターンマッチ |
| Web SPA フロントエンド | Elm | F# (Fable) | ランタイムエラーゼロ保証 |
| データパイプライン | F# | Haskell | .NETエコシステム + 型推論 |
| IoT デバイス管理 | Elixir | OCaml | Nerves (Elixir IoT FW) |
| 学術研究・論文実装 | Haskell | OCaml | 理論と実装の一致度 |
| マイクロサービス | Elixir | F# | OTP Supervisor + Phoenix |
| CLIツール | OCaml | Haskell | 起動速度 + ネイティブバイナリ |
| ゲームロジック | F# | Haskell | Unity連携 + 型安全性 |

---

## 9. アンチパターンと回避策

### 9.1 アンチパターン1: 「モナド地獄」（Haskell）

**問題**: モナド変換子（Monad Transformer）のスタックが深くなりすぎ、型シグネチャが読めなくなる。

```haskell
-- アンチパターン: モナド変換子の過剰なネスト
-- 型シグネチャが爆発し、保守不能になる

type AppStack a =
  ExceptT AppError
    (StateT AppState
      (ReaderT AppConfig
        (WriterT [LogEntry]
          IO))) a

-- 問題点:
-- 1. 型シグネチャが巨大で読解困難
-- 2. lift の連鎖が必要（lift . lift . lift ...）
-- 3. スタックの順序変更が大規模リファクタリングを要求
-- 4. パフォーマンスへの悪影響（各層のオーバーヘッド）
```

**回避策**: エフェクトシステムの採用、または ReaderT パターンの活用。

```haskell
-- 回避策1: ReaderT パターン（シンプルで実用的）
-- 1つの ReaderT + IORef で多くのケースをカバー

data AppEnv = AppEnv
  { envConfig :: AppConfig
  , envState  :: IORef AppState
  , envLogger :: LogEntry -> IO ()
  }

type App a = ReaderT AppEnv IO a

-- クリーンな型シグネチャ
getConfig :: App AppConfig
getConfig = asks envConfig

modifyState :: (AppState -> AppState) -> App ()
modifyState f = do
  ref <- asks envState
  liftIO $ modifyIORef ref f

logMessage :: LogEntry -> App ()
logMessage entry = do
  logger <- asks envLogger
  liftIO $ logger entry

-- 回避策2: 型エイリアスで段階的に抽象化
-- 必要な機能だけを制約として表現
class Monad m => HasConfig m where
  getAppConfig :: m AppConfig

class Monad m => HasState m where
  getAppState    :: m AppState
  putAppState    :: AppState -> m ()
```

### 9.2 アンチパターン2: 「プロセスリーク」（Elixir）

**問題**: GenServer や Task を起動したまま適切に終了しない。メモリリークやリソース枯渇を引き起こす。

```elixir
# アンチパターン: 管理されないプロセスの起動

defmodule LeakyModule do
  # 問題: spawn したプロセスが誰にも管理されない
  def process_batch(items) do
    Enum.each(items, fn item ->
      spawn(fn ->
        # 重い処理
        result = heavy_computation(item)
        # 結果を使いたいが、呼び出し元に返す手段がない
        IO.puts("Done: #{result}")
      end)
    end)
    # 問題点:
    # 1. プロセスの完了を待てない
    # 2. エラーが発生しても検知できない
    # 3. プロセスが永遠に待機状態になる可能性
    # 4. メモリが徐々に消費される
  end
end
```

**回避策**: Task.Supervisor を使った構造化された並行処理。

```elixir
# 回避策: Task.Supervisor + タイムアウト管理

defmodule SafeModule do
  # Supervisor 配下で Task を管理する
  def process_batch(items) do
    # Task.Supervisor は Application の Supervisor Tree に登録済みとする
    tasks =
      Enum.map(items, fn item ->
        Task.Supervisor.async(
          MyApp.TaskSupervisor,
          fn -> heavy_computation(item) end,
          shutdown: :brutal_kill  # タイムアウト時の強制終了
        )
      end)

    # 全タスクの完了を待つ（タイムアウト付き）
    results =
      Task.yield_many(tasks, timeout: 30_000)  # 30秒タイムアウト
      |> Enum.map(fn
        {task, {:ok, result}} ->
          {:ok, result}
        {task, {:exit, reason}} ->
          Task.shutdown(task)
          {:error, reason}
        {task, nil} ->
          Task.shutdown(task)   # タイムアウトしたタスクを強制終了
          {:error, :timeout}
      end)

    results
  end
end

# Application の Supervisor Tree に TaskSupervisor を登録
# defmodule MyApp.Application do
#   def start(_type, _args) do
#     children = [
#       {Task.Supervisor, name: MyApp.TaskSupervisor}
#     ]
#     Supervisor.start_link(children, strategy: :one_for_one)
#   end
# end
```

### 9.3 アンチパターン3: 「過剰な型抽象化」（全言語共通）

**問題**: 型レベルプログラミングに没頭し、ビジネスロジックよりも型の設計に時間を使いすぎる。

```haskell
-- アンチパターン: 過剰に抽象化された型

-- 単純な設定値の取得に、過剰な型クラス階層を導入
class (Monad m, MonadReader r m, HasConfig r,
       MonadError AppError m, MonadLogger m,
       MonadMetrics m, MonadCache m) =>
  AppMonad r m | m -> r where
    runApp :: m a -> IO (Either AppError a)

-- 問題点:
-- 1. 型エラーメッセージが数十行に及ぶ
-- 2. 新しいチームメンバーが理解するのに数週間かかる
-- 3. コンパイル時間が大幅に増加
-- 4. 実際のビジネスロジックが型の複雑さに埋もれる
```

**回避策**: YAGNI（You Aren't Gonna Need It）原則の適用。まず具体的な型で書き、必要になってから抽象化する。

---

## 10. 演習問題

### 10.1 初級: パターンマッチとデータ変換

**課題**: 以下の仕様を、好きな関数型言語で実装せよ。

1. `Shape` 型（`Circle`, `Rectangle`, `Triangle`）を定義する
2. 各 Shape の面積を計算する `area` 関数を実装する
3. Shape のリストを受け取り、面積の大きい順にソートして返す関数を実装する
4. 面積が指定値以上の Shape だけをフィルタリングする関数を実装する

**期待される学び**:
- 代数的データ型の定義方法
- パターンマッチの基本
- 高階関数（map, filter, sort）の使い方

**解答例（Haskell）**:

```haskell
data Shape
  = Circle Double
  | Rectangle Double Double
  | Triangle Double Double
  deriving (Show)

area :: Shape -> Double
area (Circle r)      = pi * r * r
area (Rectangle w h) = w * h
area (Triangle b h)  = 0.5 * b * h

sortByAreaDesc :: [Shape] -> [Shape]
sortByAreaDesc = sortBy (flip compare `on` area)

filterByMinArea :: Double -> [Shape] -> [Shape]
filterByMinArea minArea = filter (\s -> area s >= minArea)

-- 使用例
-- let shapes = [Circle 5, Rectangle 3 4, Triangle 6 8, Circle 2]
-- sortByAreaDesc shapes
-- → [Circle 5.0, Triangle 6.0 8.0, Rectangle 3.0 4.0, Circle 2.0]
-- filterByMinArea 20.0 shapes
-- → [Circle 5.0, Triangle 6.0 8.0]
```

### 10.2 中級: エラーハンドリングパイプライン

**課題**: ユーザー登録処理を関数型的に実装せよ。

1. 入力バリデーション（名前: 2〜50文字、メール: `@` を含む、年齢: 0〜150）
2. 各バリデーションは失敗理由を含むエラーを返す
3. 全バリデーションを通過した場合のみ `User` レコードを生成する
4. パイプ演算子やモナディックなエラーチェインを使うこと

**期待される学び**:
- Result/Either 型によるエラーハンドリング
- バリデーションの合成
- Railway Oriented Programming の概念

**解答例（F#）**:

```fsharp
type ValidationError =
    | NameTooShort
    | NameTooLong
    | InvalidEmail
    | InvalidAge of int

type User = {
    Name: string
    Email: string
    Age: int
}

let validateName name =
    if String.length name < 2 then Error NameTooShort
    elif String.length name > 50 then Error NameTooLong
    else Ok name

let validateEmail email =
    if String.exists ((=) '@') email then Ok email
    else Error InvalidEmail

let validateAge age =
    if age >= 0 && age <= 150 then Ok age
    else Error (InvalidAge age)

// Railway Oriented Programming: bind で連鎖
let createUser name email age =
    validateName name
    |> Result.bind (fun validName ->
        validateEmail email
        |> Result.bind (fun validEmail ->
            validateAge age
            |> Result.map (fun validAge ->
                { Name = validName
                  Email = validEmail
                  Age = validAge })))

// createUser "Alice" "alice@example.com" 30 → Ok { Name="Alice"; ... }
// createUser "A" "alice@example.com" 30     → Error NameTooShort
// createUser "Alice" "invalid" 30           → Error InvalidEmail
```

### 10.3 上級: 並行データ処理システム

**課題**: Web クローラーの並行処理部分を設計・実装せよ。

1. URL のリストを受け取り、並行にフェッチする
2. 各フェッチには 10 秒のタイムアウトを設定する
3. 失敗した URL はリトライ（最大 3 回）する
4. 結果を成功/失敗に分類して返す
5. 同時実行数の上限（例: 10）を設けること

**期待される学び**:
- 言語固有の並行処理モデルの実践
- タイムアウトとリトライのパターン
- リソース管理とバックプレッシャー

**ヒント**:
- Haskell: `async` ライブラリ + `STM` でセマフォ実装
- Elixir: `Task.Supervisor` + `Task.async_stream` (max_concurrency オプション)
- F#: `Async.Parallel` + `SemaphoreSlim`
- OCaml: `Lwt` / `Eio` + `Lwt_pool`

---

## 11. 関数型言語の現在と未来

### 11.1 トレンド

1. **マルチコア対応の強化**: OCaml 5.0 のドメインとエフェクトハンドラ、Haskell の改良された並行ランタイム
2. **メインストリーム言語への影響**: Rust のパターンマッチ、Kotlin の data class、Swift の enum + 値型、TypeScript の discriminated unions
3. **エフェクトシステムの台頭**: 代数的エフェクト（Algebraic Effects）が次世代の副作用管理として注目
4. **依存型の実用化**: Idris, Agda で研究されてきた依存型が、Haskell の GHC 拡張を通じて徐々に実用言語に浸透
5. **WebAssembly 対応**: OCaml (wasm_of_ocaml), Haskell (Asterius), F# (Bolero) による Wasm ターゲット

### 11.2 各言語の今後の方向性

| 言語 | 主な進化の方向 | 注目すべき動向 |
|------|--------------|---------------|
| Haskell | 依存型、リニア型 | GHC2021/2024 言語標準、Cabal 改善 |
| Elixir | 型システム導入 | set-theoretic types (v1.17+)、LiveView Native |
| Elm | 安定性重視 | 大きな変更より既存機能の成熟 |
| F# | クロスプラットフォーム | .NET 8+、WASM (Bolero) |
| OCaml | マルチコア、エフェクト | OCaml 5.x 系列、Eio ライブラリ |

---

## 12. FAQ（よくある質問）

### Q1: 関数型言語は本当にプロダクションで使えるのか？

**A**: 大規模な実績が複数ある。Discord は Elixir で 1100 万以上の同時接続を処理している。Jane Street は OCaml で毎日数兆ドル規模の金融取引を処理している。Meta（旧 Facebook）は Haskell で spam フィルタリングシステム（Sigma）を運用し、毎秒 100 万リクエスト以上を処理している。NoRedInk は Elm でフロントエンドを構築し、プロダクションでのランタイムエラーがゼロであると報告している。F# は Microsoft 内部で Azure や Bing の一部に使われている。関数型言語は「学術的な実験」の段階をとうに超えている。

### Q2: 関数型言語の学習順序はどうすべきか？

**A**: 学習者のバックグラウンドによる推奨順序は以下のとおり。

- **JavaScript/TypeScript 経験者**: Elm → Elixir → Haskell
  - Elm は JS のエコシステムに近く、TEA は React/Redux と類似。入門に最適。
- **C#/Java 経験者**: F# → Haskell → OCaml
  - F# は .NET の知識をそのまま活かせる。IDE サポートも充実。
- **Python/Ruby 経験者**: Elixir → Elm → Haskell
  - Elixir の文法は Ruby に近く、パイプ演算子は Python の連鎖メソッドに似る。
- **C/C++ 経験者**: OCaml → Haskell → Elixir
  - OCaml のネイティブコンパイルとパフォーマンス特性は低レベル経験者に馴染みやすい。

共通するのは、Haskell を最後に学ぶという点である。Haskell の概念（モナド、型クラス、遅延評価）は、他の関数型言語の経験があると格段に理解しやすくなる。

### Q3: 関数型言語と命令型言語をどう使い分けるべきか？

**A**: 以下の判断基準が有効である。

関数型言語が向いているケース:
- データ変換パイプラインが中心（ETL、コンパイラ、パーサー）
- 正しさが最優先（金融、医療、航空宇宙）
- 並行・分散処理が必要（リアルタイムシステム、メッセージング）
- ビジネスルールが複雑（状態遷移、バリデーション）

命令型/OOP 言語が向いているケース:
- GUI アプリケーション（状態変化が本質的に多い）
- ゲームのフレーム更新ループ（ミュータブルな状態管理が効率的）
- ハードウェア制御（低レベル操作が必要）
- チームの大多数が命令型の経験のみ（教育コストの考慮）

実際には、多くのモダン言語（Rust, Kotlin, Swift, TypeScript）は関数型の機能を取り込んでおり、パラダイムの境界は曖昧になっている。重要なのは、適材適所で関数型のテクニックを活用する判断力である。

### Q4: モナドとは何か？簡潔に説明してほしい。

**A**: モナドとは「計算のコンテキストを連鎖させるための設計パターン」である。技術的には、`return`（値をコンテキストに包む）と `>>=`（bind: コンテキスト付きの値に関数を適用する）の2つの操作を持つ型クラスである。

身近な例で説明すると:
- `Maybe` モナド: 「失敗するかもしれない」というコンテキスト
- `List` モナド: 「複数の結果があり得る」というコンテキスト
- `IO` モナド: 「外部世界と対話する」というコンテキスト
- `Either` モナド: 「エラー情報付きで失敗するかもしれない」というコンテキスト

モナドを使わない場合、毎回 `if result == null` のようなチェックを書く必要がある。モナドはこのボイラープレートを抽象化し、「正常系のロジック」だけを記述できるようにする。

### Q5: Elixir と Erlang の違いは？なぜ Elixir を選ぶのか？

**A**: Elixir は Erlang VM（BEAM）上で動作し、Erlang の全ライブラリを利用可能だが、以下の点で開発体験が向上している。

1. **構文**: Ruby 風の親しみやすい構文（Erlang は Prolog 風で独特）
2. **マクロ**: メタプログラミングが可能（Erlang にはない）
3. **ツーリング**: mix（ビルドツール）, hex（パッケージ管理）, ExDoc が統合的
4. **Phoenix フレームワーク**: Rails 級の生産性を持つ Web フレームワーク
5. **LiveView**: サーバーサイドレンダリングでリアルタイム UI を構築可能

Erlang の強みである OTP（耐障害性フレームワーク）、BEAM VM（軽量プロセス）、ホットコードスワップは Elixir でもそのまま利用できる。新規プロジェクトでは、開発効率とコミュニティの活発さから Elixir を選択するケースが多い。

---

## 13. まとめ

### 13.1 言語別サマリー

| 言語 | 一言で表すなら | 最適な場面 | 注意点 |
|------|-------------|----------|--------|
| Haskell | 純粋性の極致 | 研究, 金融, コンパイラ | 学習曲線が急、コンパイル遅い |
| Elixir | 実用的FP+並行 | リアルタイムWeb, IoT | CPU密集処理は不得手 |
| Elm | ランタイムエラーゼロ | 信頼性の高いUI | エコシステムが小さい |
| F# | .NET上のFP | データ処理, バックエンド | C#ほどの求人市場がない |
| OCaml | 実用的+高速 | コンパイラ, 金融, システム | ライブラリが少ない分野あり |

### 13.2 関数型言語を学ぶことの価値

関数型言語を学ぶ最大の価値は、「特定の言語を使えるようになること」ではなく、「プログラミングの考え方が変わること」にある。純粋関数、不変性、型による正しさの保証という概念は、どの言語で開発する場合でも応用できる。Rust の所有権システム、React の宣言的 UI、Kubernetes の宣言的設定は、すべて関数型の思想に根ざしている。

---

## 次に読むべきガイド

- [[../07-language-evolution/00-history-of-languages.md]] -- 言語の歴史
- [[../05-paradigm-comparison/01-oop-vs-fp.md]] -- OOP と FP の比較
- [[../../05-type-systems/01-static-vs-dynamic.md]] -- 型システムの基礎

---

## 参考文献

1. Lipovaca, M. "Learn You a Haskell for Great Good!" No Starch Press, 2011. -- Haskell 入門の定番書。モナドやファンクタの直感的な解説で知られる。
2. Thomas, D. "Programming Elixir >= 1.6." Pragmatic Bookshelf, 2018. -- Elixir の包括的な入門書。OTP の実践的な活用法を詳述。
3. Czaplicki, E. "Elm in Action." Manning Publications, 2020. -- Elm の設計哲学と TEA パターンの詳細な解説。
4. Syme, D. et al. "Expert F# 4.0." Apress, 2015. -- F# の設計者自身による包括的なリファレンス。
5. Minsky, Y., Madhavapeddy, A., Hickey, J. "Real World OCaml." O'Reilly Media, 2nd Edition, 2022. -- Jane Street エンジニアによる OCaml の実践ガイド。
6. Bird, R. "Thinking Functionally with Haskell." Cambridge University Press, 2014. -- 関数型思考の本質に迫る教科書。
7. Milewski, B. "Category Theory for Programmers." 2019. -- 圏論の概念をプログラマー向けに解説。モナドの数学的背景を理解するための参考。
8. Peyton Jones, S. "Tackling the Awkward Squad: Monadic Input/Output, Concurrency, Exceptions, and Foreign-language Calls in Haskell." 2001. -- Haskell の IO モナドの設計根拠を示す重要論文。

---

## 用語集

| 用語 | 英語 | 説明 |
|------|------|------|
| 純粋関数 | Pure Function | 同じ入力に対して常に同じ出力を返し、副作用を持たない関数 |
| 参照透過性 | Referential Transparency | 式をその評価結果に置換してもプログラムの意味が変わらない性質 |
| 代数的データ型 | Algebraic Data Type (ADT) | 直和型（Sum Type）と直積型（Product Type）を組み合わせたデータ型 |
| パターンマッチ | Pattern Matching | データの構造に基づいて分岐処理を行う構文機能 |
| 型推論 | Type Inference | 明示的な型注釈なしに、コンパイラが型を自動推論する機能 |
| モナド | Monad | 計算のコンテキストを連鎖させるための抽象パターン |
| 型クラス | Type Class | Haskell におけるアドホック多相性の仕組み。インタフェースに類似 |
| ファンクタ | Functor (ML) | OCaml/SML におけるモジュールを引数に取るパラメータ化されたモジュール |
| 遅延評価 | Lazy Evaluation | 値が実際に必要になるまで計算を遅延させる評価戦略 |
| 正格評価 | Strict/Eager Evaluation | 式が束縛された時点で即座に評価する評価戦略 |
| STM | Software Transactional Memory | トランザクションにより共有メモリへの並行アクセスを安全に行う手法 |
| アクターモデル | Actor Model | 独立したプロセスがメッセージパッシングで通信する並行計算モデル |
| 計算式 | Computation Expression | F# におけるモナディックな計算を記述するための構文糖衣 |
| エフェクトハンドラ | Effect Handler | 代数的エフェクトを処理するための仕組み。OCaml 5.0+ で導入 |
| カリー化 | Currying | 複数引数関数を、1引数関数の連鎖に変換すること |
