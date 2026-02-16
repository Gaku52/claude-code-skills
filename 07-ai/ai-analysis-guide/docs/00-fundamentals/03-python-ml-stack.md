# Python MLスタック — NumPy、pandas、scikit-learn

> Python機械学習エコシステムの中核ライブラリを実践的に習得する

## この章で学ぶこと

1. **NumPy** — 高速な多次元配列演算とブロードキャスティング
2. **pandas** — 表形式データの読み込み・加工・集計の全操作
3. **scikit-learn** — 前処理→学習→評価のパイプライン構築
4. **Matplotlib / Seaborn** — データ可視化の基礎から応用
5. **SciPy** — 科学計算・統計検定・最適化
6. **実務パターン** — プロジェクト構成、テスト、デプロイメント

---

## 1. NumPy — 数値計算の基盤

### 1.1 NumPyのアーキテクチャ

```
Python リスト              NumPy ndarray
┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │         │ 1 │ 2 │ 3 │ 4 │
└─┬─┴─┬─┴─┬─┴─┬─┘         └───┴───┴───┴───┘
  │   │   │   │              連続メモリブロック
  v   v   v   v              (C言語配列と同等)
 obj obj obj obj
 (各要素が別オブジェクト)    → ベクトル化演算で高速
 → ループが必要で低速        → BLAS/LAPACK連携
```

NumPyの内部アーキテクチャを理解することは、高速な数値計算コードを書くうえで不可欠である。NumPyの`ndarray`は以下の3つの主要コンポーネントで構成されている。

1. **データバッファ**: 連続したメモリ領域に同一型の要素が格納される
2. **dtype（データ型）**: 各要素のバイト数と解釈方法を定義する
3. **ストライド**: 各次元方向に1要素進むために必要なバイト数

```python
import numpy as np

# ndarrayの内部構造を確認
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

print(f"shape:    {arr.shape}")        # (2, 3)
print(f"dtype:    {arr.dtype}")        # float64
print(f"strides:  {arr.strides}")      # (24, 8) — 各次元のバイトストライド
print(f"itemsize: {arr.itemsize}")     # 8 バイト（float64）
print(f"nbytes:   {arr.nbytes}")       # 48 バイト（2×3×8）
print(f"flags:\n{arr.flags}")          # C_CONTIGUOUS, F_CONTIGUOUS等
```

### 1.2 配列の作成パターン

```python
import numpy as np

# --- 基本的な配列作成 ---
# ゼロ埋め
zeros = np.zeros((3, 4))                    # 3×4のゼロ行列
ones = np.ones((2, 3, 4))                   # 2×3×4の1埋めテンソル
empty = np.empty((5, 5))                    # 未初期化（高速だが値は不定）

# 等差数列・等比数列
linspace = np.linspace(0, 1, 100)           # 0〜1を100等分
arange = np.arange(0, 10, 0.5)             # 0〜10を0.5刻み
logspace = np.logspace(0, 3, 50)            # 10^0 〜 10^3 の対数等間隔

# 単位行列・対角行列
identity = np.eye(4)                         # 4×4単位行列
diag = np.diag([1, 2, 3, 4])               # 対角行列

# --- 乱数生成（新しいGenerator API推奨） ---
rng = np.random.default_rng(seed=42)

uniform = rng.uniform(0, 1, size=(3, 4))     # 一様分布
normal = rng.normal(loc=0, scale=1, size=1000)  # 正規分布
integers = rng.integers(0, 100, size=50)     # 整数乱数
choice = rng.choice(['A', 'B', 'C'], size=10, p=[0.5, 0.3, 0.2])  # 重み付き選択

# 再現性のある乱数シード管理
seed_seq = np.random.SeedSequence(42)
child_seeds = seed_seq.spawn(4)  # 並列処理用に独立したシード
rngs = [np.random.default_rng(s) for s in child_seeds]
```

### 1.3 ベクトル化演算とブロードキャスティング

```python
import numpy as np
import time

# --- ベクトル化 vs ループの速度比較 ---
n = 1_000_000
a = np.random.randn(n)
b = np.random.randn(n)

# BAD: Pythonループ
start = time.time()
result_loop = [a[i] + b[i] for i in range(n)]
print(f"ループ: {time.time() - start:.4f}秒")

# GOOD: ベクトル化演算
start = time.time()
result_vec = a + b
print(f"ベクトル化: {time.time() - start:.4f}秒")
# → 100倍以上高速

# --- ブロードキャスティング ---
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row = np.array([10, 20, 30])

# 行ベクトルが自動的にブロードキャスト
result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36],
#  [17, 28, 39]]
```

### 1.4 ブロードキャスティングの詳細ルール

ブロードキャスティングはNumPyの最も強力な機能の一つであり、異なる形状の配列間で演算を行う際の暗黙的な拡張規則である。

```
ブロードキャスティングルール:
1. 次元数が少ない配列は、先頭に次元1を追加して揃える
2. 各次元のサイズが一致するか、いずれかが1であれば互換
3. サイズ1の次元は、もう一方のサイズに合わせて拡張される

例:
  A: (3, 4)    B: (4,)
  → B を (1, 4) に変換
  → B を (3, 4) にブロードキャスト

  A: (3, 1, 5)  B: (1, 4, 1)
  → 結果: (3, 4, 5)
```

```python
import numpy as np

# ブロードキャスティングの実用例

# 例1: 各列の平均を引いて標準化
data = np.random.randn(100, 5)  # 100サンプル×5特徴量
col_mean = data.mean(axis=0)     # shape: (5,)
col_std = data.std(axis=0)       # shape: (5,)
normalized = (data - col_mean) / col_std  # (100,5) と (5,) のブロードキャスト

# 例2: 距離行列の計算（ペアワイズ距離）
points = np.random.randn(100, 3)  # 100個の3次元点
# (100,1,3) - (1,100,3) → (100,100,3) → sum → (100,100)
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=-1))

# 例3: 外積の計算
x = np.array([1, 2, 3])
y = np.array([4, 5, 6, 7])
outer = x[:, np.newaxis] * y[np.newaxis, :]  # (3,4)の外積行列

# 例4: one-hotエンコーディング
labels = np.array([0, 2, 1, 0, 3])
n_classes = 4
one_hot = (labels[:, np.newaxis] == np.arange(n_classes)[np.newaxis, :]).astype(int)
# [[1,0,0,0],
#  [0,0,1,0],
#  [0,1,0,0],
#  [1,0,0,0],
#  [0,0,0,1]]
```

### 1.5 高度なインデキシングとスライシング

```python
import numpy as np

arr = np.arange(20).reshape(4, 5)
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

# --- 基本スライス（ビュー：メモリ共有） ---
sub = arr[1:3, 2:4]        # [[7,8],[12,13]]
sub[0, 0] = 999            # arrも変更される！（ビューのため）

# --- ファンシーインデキシング（コピー：メモリ非共有） ---
rows = [0, 2, 3]
cols = [1, 3, 4]
fancy = arr[rows, cols]     # [ 1, 13, 19]（各(row,col)ペアの要素）

# --- ブーリアンインデキシング ---
mask = arr > 10
filtered = arr[mask]        # [11, 12, 13, 14, 15, 16, 17, 18, 19]

# 条件付き代入
arr_copy = arr.copy()
arr_copy[arr_copy > 15] = -1  # 15超の要素を-1に置換

# --- np.where による条件分岐 ---
result = np.where(arr > 10, arr, 0)  # 10超はそのまま、以下は0

# --- 多次元インデキシングの組み合わせ ---
# 特定の行・列を選択
selected = arr[np.ix_([0, 2], [1, 3])]  # [[1,3],[11,13]] の2×2部分行列

# --- ストライドトリック（高度） ---
# スライディングウィンドウを効率的に作成
from numpy.lib.stride_tricks import sliding_window_view
data = np.arange(10)
windows = sliding_window_view(data, window_shape=3)
# [[0,1,2], [1,2,3], [2,3,4], ..., [7,8,9]]
```

### 1.6 線形代数とFFT

```python
import numpy as np

# --- 線形代数 ---
A = np.random.randn(3, 3)
b = np.random.randn(3)

# 連立方程式 Ax = b を解く
x = np.linalg.solve(A, b)
print(f"解: {x}")
print(f"検証 Ax - b ≈ 0: {np.allclose(A @ x, b)}")

# 固有値分解
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"固有値: {eigenvalues}")

# 特異値分解（SVD）
U, s, Vt = np.linalg.svd(A)
print(f"特異値: {s}")
# 再構成: A ≈ U @ diag(s) @ Vt
A_reconstructed = U @ np.diag(s) @ Vt
print(f"再構成誤差: {np.linalg.norm(A - A_reconstructed):.2e}")

# コレスキー分解（正定値対称行列）
C = A @ A.T + np.eye(3)  # 正定値行列を作成
L = np.linalg.cholesky(C)
print(f"C = L @ L.T: {np.allclose(C, L @ L.T)}")

# QR分解
Q, R = np.linalg.qr(A)
print(f"Q直交性: {np.allclose(Q.T @ Q, np.eye(3))}")

# 行列のランク・条件数・ノルム
print(f"ランク: {np.linalg.matrix_rank(A)}")
print(f"条件数: {np.linalg.cond(A):.2f}")
print(f"フロベニウスノルム: {np.linalg.norm(A, 'fro'):.4f}")

# --- FFT（高速フーリエ変換） ---
# 信号解析の例
t = np.linspace(0, 1, 1000)  # 1秒間、1000サンプル
# 50Hzと120Hzの合成信号 + ノイズ
signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
signal += 0.3 * np.random.randn(len(t))

# FFTで周波数成分を抽出
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=t[1] - t[0])
power = np.abs(fft_result) ** 2

# 正の周波数のみ
positive_mask = frequencies > 0
dominant_freq = frequencies[positive_mask][np.argmax(power[positive_mask])]
print(f"支配的周波数: {dominant_freq:.1f} Hz")
```

### 1.7 NumPyのメモリ管理と最適化

```python
import numpy as np

# --- dtype選択によるメモリ最適化 ---
# 画像データ（0-255）にfloat64は無駄
img_bad = np.random.randint(0, 256, (1920, 1080, 3)).astype(np.float64)
img_good = np.random.randint(0, 256, (1920, 1080, 3)).astype(np.uint8)
print(f"float64: {img_bad.nbytes / 1e6:.1f} MB")  # ~49.8 MB
print(f"uint8:   {img_good.nbytes / 1e6:.1f} MB")  # ~6.2 MB

# --- メモリレイアウト（C連続 vs Fortran連続） ---
c_arr = np.array([[1,2,3],[4,5,6]], order='C')    # 行優先（C言語方式）
f_arr = np.array([[1,2,3],[4,5,6]], order='F')    # 列優先（Fortran方式）

# 行方向のアクセスはC連続が高速
# 列方向のアクセスはF連続が高速

# --- コピー vs ビューの判定 ---
original = np.arange(12).reshape(3, 4)
view = original[1:]          # ビュー（メモリ共有）
copy = original[1:].copy()   # コピー（独立メモリ）

print(f"ビュー: {view.base is original}")   # True
print(f"コピー: {copy.base is original}")   # False

# --- np.memmap: ディスク上の大規模配列 ---
# メモリに載りきらない巨大配列を扱う
mmap = np.memmap('/tmp/large_array.dat', dtype='float32',
                 mode='w+', shape=(10000, 10000))
mmap[:100, :100] = np.random.randn(100, 100).astype('float32')
mmap.flush()  # ディスクに書き出し
del mmap      # 解放

# 読み込み（必要な部分だけメモリにロード）
mmap_read = np.memmap('/tmp/large_array.dat', dtype='float32',
                      mode='r', shape=(10000, 10000))
subset = mmap_read[:100, :100]
```

### 1.8 NumPyの汎用関数（ufunc）

```python
import numpy as np

# --- 組み込みufuncの活用 ---
x = np.array([1, 4, 9, 16, 25])

# 数学関数
print(np.sqrt(x))         # 平方根
print(np.log(x))          # 自然対数
print(np.exp(x))          # 指数関数

# 集約関数
data = np.random.randn(1000)
print(f"平均: {np.mean(data):.4f}")
print(f"中央値: {np.median(data):.4f}")
print(f"標準偏差: {np.std(data):.4f}")
print(f"パーセンタイル: {np.percentile(data, [25, 50, 75])}")

# 累積関数
arr = np.array([1, 2, 3, 4, 5])
print(f"累積和: {np.cumsum(arr)}")      # [1, 3, 6, 10, 15]
print(f"累積積: {np.cumprod(arr)}")     # [1, 2, 6, 24, 120]

# --- カスタムufuncの作成 ---
# np.vectorize（簡便だが速度向上なし）
def my_func(x):
    if x > 0:
        return x ** 2
    else:
        return -x

vectorized = np.vectorize(my_func)
result = vectorized(np.array([-2, -1, 0, 1, 2]))

# np.frompyfunc（より高速）
ufunc = np.frompyfunc(my_func, 1, 1)
result = ufunc(np.array([-2, -1, 0, 1, 2]))

# 最も高速: np.whereとベクトル演算の組み合わせ
x = np.array([-2, -1, 0, 1, 2])
result = np.where(x > 0, x ** 2, -x)
```

---

## 2. pandas — データ操作の標準ツール

### 2.1 DataFrameの基本操作とメソッドチェーン

```python
import pandas as pd
import numpy as np

# --- DataFrameの作成と基本操作 ---
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [28, 35, 42, 31, 27],
    "department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
    "salary": [85000, 72000, 95000, 68000, 71000],
    "join_date": pd.to_datetime(["2020-03-15", "2019-07-01", "2018-01-20",
                                  "2021-06-10", "2022-02-28"])
})

# メソッドチェーンでデータ加工
result = (
    df
    .assign(
        tenure_years=lambda x: (pd.Timestamp.now() - x["join_date"]).dt.days / 365,
        salary_rank=lambda x: x["salary"].rank(ascending=False).astype(int)
    )
    .query("age >= 30")
    .sort_values("salary", ascending=False)
    .reset_index(drop=True)
)
print(result)

# --- GroupBy 集計 ---
summary = (
    df
    .groupby("department")
    .agg(
        人数=("name", "count"),
        平均年齢=("age", "mean"),
        平均給与=("salary", "mean"),
        最高給与=("salary", "max"),
    )
    .round(0)
    .sort_values("平均給与", ascending=False)
)
print(summary)
```

### 2.2 データ型と欠損値の管理

```python
import pandas as pd
import numpy as np

# --- データ型の確認と変換 ---
df = pd.DataFrame({
    "id": ["001", "002", "003", "004"],
    "value": ["100", "200", "N/A", "400"],
    "date_str": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"],
    "category": ["A", "B", "A", "C"],
    "flag": [1, 0, 1, 1],
})

# 型変換のベストプラクティス
df_typed = (
    df
    .assign(
        id=lambda x: x["id"].astype("string"),        # 文字列型（pandas StringDtype）
        value=lambda x: pd.to_numeric(x["value"], errors="coerce"),  # 数値に変換（N/A→NaN）
        date=lambda x: pd.to_datetime(x["date_str"]),  # 日付型
        category=lambda x: x["category"].astype("category"),  # カテゴリ型
        flag=lambda x: x["flag"].astype(bool),         # ブール型
    )
    .drop(columns=["date_str"])
)

print(df_typed.dtypes)
print(f"メモリ使用量: {df_typed.memory_usage(deep=True).sum()} bytes")

# --- Nullable型（pandas 1.0+推奨） ---
# 従来: 整数列にNaNがあるとfloat64に昇格
# 新方式: pd.Int64Dtype() で整数のままNaN対応
s = pd.array([1, 2, pd.NA, 4], dtype=pd.Int64Dtype())
print(s)       # [1, 2, <NA>, 4]
print(s.dtype) # Int64

# --- 欠損値の処理パターン ---
df_missing = pd.DataFrame({
    "A": [1, np.nan, 3, np.nan, 5],
    "B": [np.nan, 2, np.nan, 4, 5],
    "C": [1, 2, 3, 4, 5],
})

# 欠損値の確認
print(df_missing.isnull().sum())           # 列ごとの欠損数
print(df_missing.isnull().mean() * 100)    # 列ごとの欠損率(%)

# 補間戦略
df_filled = df_missing.copy()
df_filled["A"] = df_filled["A"].fillna(df_filled["A"].median())      # 中央値で補間
df_filled["B"] = df_filled["B"].interpolate(method="linear")          # 線形補間
df_filled["A_forward"] = df_missing["A"].ffill()                      # 前方補間
df_filled["A_backward"] = df_missing["A"].bfill()                     # 後方補間

# 欠損パターンの可視化用データ
missing_pattern = df_missing.isnull().astype(int)
print(missing_pattern)
```

### 2.3 時系列データの処理

```python
import pandas as pd
import numpy as np

# --- 時系列データの作成 ---
dates = pd.date_range("2023-01-01", periods=365, freq="D")
ts = pd.DataFrame({
    "date": dates,
    "sales": np.random.poisson(100, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 30,
    "temperature": 15 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 3,
})
ts = ts.set_index("date")

# --- リサンプリング ---
# 月次集計
monthly = ts.resample("M").agg({
    "sales": ["sum", "mean", "std"],
    "temperature": "mean",
})
print(monthly.head())

# 週次集計（月曜始まり）
weekly = ts.resample("W-MON").mean()

# --- ローリング統計 ---
ts["sales_ma7"] = ts["sales"].rolling(window=7).mean()           # 7日移動平均
ts["sales_ma30"] = ts["sales"].rolling(window=30).mean()         # 30日移動平均
ts["sales_std7"] = ts["sales"].rolling(window=7).std()           # 7日移動標準偏差
ts["sales_ewm"] = ts["sales"].ewm(span=7).mean()                # 指数加重移動平均

# --- ラグ特徴量・差分 ---
ts["sales_lag1"] = ts["sales"].shift(1)       # 1日前
ts["sales_lag7"] = ts["sales"].shift(7)       # 7日前
ts["sales_diff1"] = ts["sales"].diff(1)       # 1次差分
ts["sales_pct_change"] = ts["sales"].pct_change()  # 変化率

# --- 曜日・月などの特徴量抽出 ---
ts["dayofweek"] = ts.index.dayofweek          # 0=月曜
ts["month"] = ts.index.month
ts["quarter"] = ts.index.quarter
ts["is_weekend"] = ts["dayofweek"].isin([5, 6]).astype(int)
ts["day_of_year"] = ts.index.dayofyear

# --- 期間インデックスとタイムゾーン ---
# タイムゾーン変換
ts_utc = ts.tz_localize("UTC")
ts_jst = ts_utc.tz_convert("Asia/Tokyo")

# ビジネス日カレンダー
biz_days = pd.bdate_range("2024-01-01", "2024-12-31", freq="B")
print(f"2024年の営業日数: {len(biz_days)}")
```

### 2.4 大規模データの効率的な読み込み

```python
import pandas as pd

# --- メモリ最適化読み込み ---
def read_optimized(filepath: str, sample_rows: int = 10000) -> pd.DataFrame:
    """メモリ効率の良いCSV読み込み"""

    # まずサンプルで型を推定
    sample = pd.read_csv(filepath, nrows=sample_rows)

    # 型の最適化マップを作成
    dtype_map = {}
    for col in sample.columns:
        col_type = sample[col].dtype
        if col_type == "int64":
            if sample[col].min() >= 0 and sample[col].max() <= 255:
                dtype_map[col] = "uint8"
            elif sample[col].min() >= -128 and sample[col].max() <= 127:
                dtype_map[col] = "int8"
            elif sample[col].min() >= -32768 and sample[col].max() <= 32767:
                dtype_map[col] = "int16"
            else:
                dtype_map[col] = "int32"
        elif col_type == "float64":
            dtype_map[col] = "float32"
        elif col_type == "object":
            if sample[col].nunique() / len(sample) < 0.5:
                dtype_map[col] = "category"

    # 最適化した型で読み込み
    df = pd.read_csv(filepath, dtype=dtype_map)

    original_mb = sample.memory_usage(deep=True).sum() / 1e6
    optimized_mb = df.head(sample_rows).memory_usage(deep=True).sum() / 1e6
    print(f"メモリ削減: {original_mb:.1f}MB → {optimized_mb:.1f}MB "
          f"({(1-optimized_mb/original_mb)*100:.0f}%削減)")

    return df


# --- チャンク処理（メモリに載りきらないデータ） ---
def process_large_csv(filepath: str, chunksize: int = 100000):
    """巨大CSVをチャンクで処理"""
    results = []

    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
        # 各チャンクに対して処理
        chunk_result = (
            chunk
            .groupby("category")
            .agg({"value": ["sum", "count"]})
        )
        results.append(chunk_result)

        if (i + 1) % 10 == 0:
            print(f"  {(i + 1) * chunksize:,} 行処理完了")

    # チャンク結果を統合
    combined = pd.concat(results)
    final = combined.groupby(level=0).sum()
    return final


# --- Parquet形式の活用（推奨） ---
def csv_to_parquet(csv_path: str, parquet_path: str):
    """CSVからParquetへの変換（圧縮+高速読み込み）"""
    df = pd.read_csv(csv_path)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")

    import os
    csv_size = os.path.getsize(csv_path) / 1e6
    parquet_size = os.path.getsize(parquet_path) / 1e6
    print(f"CSV: {csv_size:.1f}MB → Parquet: {parquet_size:.1f}MB "
          f"({(1-parquet_size/csv_size)*100:.0f}%圧縮)")


# Parquet読み込み（必要な列だけ読む）
# df = pd.read_parquet("data.parquet", columns=["col1", "col2"])
```

### 2.5 マルチインデックスとピボット操作

```python
import pandas as pd
import numpy as np

# --- マルチインデックス ---
arrays = [
    ["東京", "東京", "大阪", "大阪", "名古屋", "名古屋"],
    ["2023Q1", "2023Q2", "2023Q1", "2023Q2", "2023Q1", "2023Q2"],
]
index = pd.MultiIndex.from_arrays(arrays, names=["都市", "四半期"])

df = pd.DataFrame({
    "売上": [100, 120, 80, 90, 60, 70],
    "利益": [30, 35, 20, 25, 15, 18],
}, index=index)

# マルチインデックスのアクセス
print(df.loc["東京"])            # 東京の全データ
print(df.loc[("東京", "2023Q1")]) # 東京のQ1
print(df.xs("2023Q1", level="四半期"))  # 全都市のQ1

# --- ピボットテーブル ---
sales_data = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=100, freq="D"),
    "product": np.random.choice(["A", "B", "C"], 100),
    "region": np.random.choice(["関東", "関西", "九州"], 100),
    "amount": np.random.randint(1000, 10000, 100),
    "quantity": np.random.randint(1, 50, 100),
})

# ピボットテーブル
pivot = pd.pivot_table(
    sales_data,
    values="amount",
    index="product",
    columns="region",
    aggfunc=["sum", "mean", "count"],
    margins=True,           # 合計行・列を追加
    margins_name="合計",
)
print(pivot)

# --- クロス集計 ---
cross = pd.crosstab(
    sales_data["product"],
    sales_data["region"],
    values=sales_data["amount"],
    aggfunc="sum",
    margins=True,
)
print(cross)

# --- stack / unstack ---
stacked = df.stack()       # 列→行（長い形式へ）
unstacked = stacked.unstack(level="都市")  # 行→列（広い形式へ）

# --- melt（ワイド→ロング変換） ---
wide_df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "math": [90, 85],
    "english": [80, 92],
    "science": [88, 78],
})
long_df = wide_df.melt(
    id_vars=["name"],
    value_vars=["math", "english", "science"],
    var_name="subject",
    value_name="score",
)
print(long_df)
```

### 2.6 文字列操作とカテゴリカルデータ

```python
import pandas as pd

# --- 文字列操作（.str アクセサ） ---
df = pd.DataFrame({
    "full_name": ["田中 太郎", "佐藤 花子", "鈴木 一郎", "高橋 美咲"],
    "email": ["tanaka@example.com", "SATO@Example.COM", "suzuki@test.org", "takahashi@example.com"],
    "phone": ["090-1234-5678", "080-2345-6789", "070-3456-7890", "090-4567-8901"],
    "address": ["東京都渋谷区1-2-3", "大阪府北区4-5-6", "東京都新宿区7-8-9", "福岡県博多区10-11-12"],
})

# 文字列メソッド
df["last_name"] = df["full_name"].str.split(" ").str[0]
df["first_name"] = df["full_name"].str.split(" ").str[1]
df["email_lower"] = df["email"].str.lower()
df["email_domain"] = df["email"].str.split("@").str[1].str.lower()
df["phone_clean"] = df["phone"].str.replace("-", "", regex=False)
df["is_tokyo"] = df["address"].str.contains("東京", regex=False)

# 正規表現
df["prefecture"] = df["address"].str.extract(r"^(.+?[都道府県])")

# --- カテゴリカルデータの扱い ---
# 順序付きカテゴリ
satisfaction = pd.Categorical(
    ["満足", "普通", "不満", "満足", "とても満足"],
    categories=["不満", "普通", "満足", "とても満足"],
    ordered=True,
)
s = pd.Series(satisfaction)
print(s > "普通")  # 比較演算が可能

# カテゴリのメモリ効率
large_series = pd.Series(["A", "B", "C"] * 100000)
print(f"object型: {large_series.memory_usage(deep=True) / 1e6:.1f} MB")
print(f"category型: {large_series.astype('category').memory_usage(deep=True) / 1e6:.1f} MB")
```

### 2.7 結合操作の完全ガイド

```python
import pandas as pd

# --- merge（SQLのJOINに相当） ---
orders = pd.DataFrame({
    "order_id": [1, 2, 3, 4, 5],
    "customer_id": [101, 102, 101, 103, 104],
    "amount": [5000, 3000, 7000, 2000, 6000],
})

customers = pd.DataFrame({
    "customer_id": [101, 102, 103, 105],
    "name": ["田中", "佐藤", "鈴木", "高橋"],
    "region": ["東京", "大阪", "東京", "福岡"],
})

# INNER JOIN（両方に存在するもののみ）
inner = orders.merge(customers, on="customer_id", how="inner")

# LEFT JOIN（左テーブル基準）
left = orders.merge(customers, on="customer_id", how="left")

# OUTER JOIN（全て保持）
outer = orders.merge(customers, on="customer_id", how="outer", indicator=True)
print(outer["_merge"].value_counts())

# --- キーが異なる場合 ---
df1 = pd.DataFrame({"id_left": [1, 2], "val": [10, 20]})
df2 = pd.DataFrame({"id_right": [1, 2], "val2": [30, 40]})
merged = df1.merge(df2, left_on="id_left", right_on="id_right")

# --- concat（積み重ね・結合） ---
df_a = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
df_b = pd.DataFrame({"col1": [5, 6], "col2": [7, 8]})

# 縦方向の結合
vertical = pd.concat([df_a, df_b], axis=0, ignore_index=True)

# 横方向の結合
horizontal = pd.concat([df_a, df_b], axis=1)

# --- 条件付き結合（merge_asof：最近傍結合） ---
# 時系列データの近傍マッチング
trades = pd.DataFrame({
    "time": pd.to_datetime(["2024-01-01 09:01:00", "2024-01-01 09:05:30"]),
    "price": [100, 102],
})
quotes = pd.DataFrame({
    "time": pd.to_datetime(["2024-01-01 09:00:00", "2024-01-01 09:03:00",
                            "2024-01-01 09:05:00"]),
    "bid": [99, 101, 101.5],
})
result = pd.merge_asof(trades, quotes, on="time", direction="backward")
print(result)
```

### 2.8 apply と高速な代替手法

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "a": np.random.randn(100000),
    "b": np.random.randn(100000),
    "c": np.random.choice(["X", "Y", "Z"], 100000),
})

# --- apply は最終手段 ---
# BAD: apply（Pythonレベルのループで遅い）
# result = df.apply(lambda row: row["a"] ** 2 + row["b"] ** 2, axis=1)

# GOOD: ベクトル演算（100倍以上高速）
result = df["a"] ** 2 + df["b"] ** 2

# --- 条件分岐のベクトル化 ---
# BAD
# df["label"] = df.apply(lambda row: "high" if row["a"] > 1 else "low", axis=1)

# GOOD: np.where
df["label"] = np.where(df["a"] > 1, "high", "low")

# GOOD: np.select（複数条件）
conditions = [
    df["a"] > 1,
    df["a"] > 0,
    df["a"] > -1,
]
choices = ["very_high", "high", "medium"]
df["grade"] = np.select(conditions, choices, default="low")

# --- groupby + transform ---
# グループ内の標準化
df["a_group_normalized"] = df.groupby("c")["a"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# --- pipe: 関数パイプライン ---
def add_features(df):
    return df.assign(
        ab_product=df["a"] * df["b"],
        ab_ratio=df["a"] / (df["b"] + 1e-8),
    )

def filter_outliers(df, col="a", n_std=3):
    mean, std = df[col].mean(), df[col].std()
    return df[(df[col] > mean - n_std * std) & (df[col] < mean + n_std * std)]

result = (
    df
    .pipe(add_features)
    .pipe(filter_outliers, col="a", n_std=3)
)
```

---

## 3. scikit-learn — MLパイプライン

### 3.1 scikit-learn API設計

```
scikit-learn の一貫したAPI:

  すべての推定器 (Estimator)
  ├── fit(X, y)           # 学習
  ├── predict(X)          # 予測
  ├── score(X, y)         # 評価
  └── get_params()        # パラメータ取得

  変換器 (Transformer) は追加で:
  ├── transform(X)        # 変換
  └── fit_transform(X)    # 学習+変換

  Pipeline で連結:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Scaler   │──>│ PCA      │──>│ Model    │
  │(変換器)  │   │(変換器)  │   │(推定器)  │
  │fit       │   │fit       │   │fit       │
  │transform │   │transform │   │predict   │
  └──────────┘   └──────────┘   └──────────┘
```

### 3.2 データの前処理

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OrdinalEncoder, OneHotEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import pandas as pd

# --- スケーリング手法の比較 ---
X = np.array([[1, 10], [2, 20], [3, 30], [100, 40]])

# StandardScaler: 平均0、標準偏差1
standard = StandardScaler().fit_transform(X)

# MinMaxScaler: 0〜1の範囲にスケーリング
minmax = MinMaxScaler().fit_transform(X)

# RobustScaler: 中央値とIQRによるスケーリング（外れ値に頑健）
robust = RobustScaler().fit_transform(X)

# PowerTransformer: 正規分布に近づける変換
power = PowerTransformer(method="yeo-johnson").fit_transform(X)

# QuantileTransformer: 一様分布または正規分布に変換
quantile = QuantileTransformer(output_distribution="normal").fit_transform(X)

print("StandardScaler:\n", standard)
print("RobustScaler:\n", robust)

# --- 欠損値補完 ---
X_missing = np.array([[1, 2], [np.nan, 3], [7, np.nan], [4, 5]])

# 単純補完
simple = SimpleImputer(strategy="median").fit_transform(X_missing)

# KNN補完（近傍データから推定）
knn_imp = KNNImputer(n_neighbors=2).fit_transform(X_missing)

# --- テキスト特徴量 ---
corpus = [
    "機械学習の基礎を学ぶ",
    "深層学習とニューラルネットワーク",
    "自然言語処理の応用",
    "機械学習による画像認識",
]

# TF-IDF
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(corpus)
print(f"TF-IDF shape: {X_tfidf.shape}")
print(f"特徴量名: {tfidf.get_feature_names_out()[:10]}")

# --- カスタム変換 ---
log_transformer = FunctionTransformer(np.log1p, inverse_func=np.expm1)
X_log = log_transformer.fit_transform(np.array([[1], [10], [100], [1000]]))
```

### 3.3 パイプライン構築

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

# サンプルデータ
df = pd.DataFrame({
    "area": [50, 70, 90, 120, 60, np.nan, 80, 100],
    "rooms": [2, 3, 3, 4, 2, 3, np.nan, 4],
    "location": ["都心", "郊外", "都心", "都心", "郊外", "郊外", "都心", "郊外"],
    "age_years": [5, 10, 3, 1, 20, 15, 8, 12],
    "price": [5000, 4000, 7000, 9000, 3500, 3000, 6000, 4500],
})

X = df.drop(columns=["price"])
y = df["price"]

# 数値列とカテゴリ列で異なる前処理
numeric_features = ["area", "rooms", "age_years"]
categorical_features = ["location"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# 前処理 + モデルの統合パイプライン
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(random_state=42)),
])

# ハイパーパラメータ探索
param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__max_depth": [3, 5, 7],
    "regressor__learning_rate": [0.01, 0.1, 0.3],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error")
grid.fit(X, y)

print(f"最良パラメータ: {grid.best_params_}")
print(f"最良スコア (neg MSE): {grid.best_score_:.2f}")
```

### 3.4 交差検証と評価指標

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold,
    RepeatedStratifiedKFold, LeaveOneOut, TimeSeriesSplit,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    make_scorer,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 分類タスクの交差検証 ---
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_classes=2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 基本的な交差検証
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"精度: {scores.mean():.4f} ± {scores.std():.4f}")

# 複数指標を同時に計算
cv_results = cross_validate(
    model, X, y, cv=5,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=True,
)
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    test_key = f"test_{metric}"
    train_key = f"train_{metric}"
    print(f"{metric:12s}: train={cv_results[train_key].mean():.4f} "
          f"test={cv_results[test_key].mean():.4f}")

# --- 層化K分割（クラス比率を保持） ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
    print(f"Fold {fold+1}: {score:.4f}")

# --- 時系列データの交差検証 ---
tscv = TimeSeriesSplit(n_splits=5)
X_ts, y_ts = make_regression(n_samples=200, n_features=5, random_state=42)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_ts)):
    print(f"Fold {fold+1}: train={len(train_idx)}, val={len(val_idx)}")

# --- 詳細な分類レポート ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n分類レポート:")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 混同行列
cm = confusion_matrix(y_test, y_pred)
print(f"\n混同行列:\n{cm}")

# --- カスタムスコアラー ---
def custom_metric(y_true, y_pred):
    """偽陰性に2倍のペナルティを課すカスタム指標"""
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    return tp / (tp + fp + 2 * fn + 1e-8)

custom_scorer = make_scorer(custom_metric, greater_is_better=True)
scores = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
print(f"\nカスタム指標: {scores.mean():.4f}")
```

### 3.5 特徴量選択

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV,
    SequentialFeatureSelector,
    VarianceThreshold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd

X, y = make_classification(n_samples=500, n_features=30, n_informative=10,
                           n_redundant=5, n_repeated=5, random_state=42)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

# --- 分散ベースのフィルタリング ---
# 分散がほぼ0の特徴量を除去
selector_var = VarianceThreshold(threshold=0.01)
X_filtered = selector_var.fit_transform(X)
print(f"分散フィルタリング: {X.shape[1]} → {X_filtered.shape[1]} 特徴量")

# --- 統計検定ベースの選択 ---
# F値による選択
selector_f = SelectKBest(f_classif, k=10)
X_f = selector_f.fit_transform(X, y)
f_scores = pd.Series(selector_f.scores_, index=feature_names)
print("\nF値 Top10:")
print(f_scores.nlargest(10))

# 相互情報量による選択
selector_mi = SelectKBest(mutual_info_classif, k=10)
X_mi = selector_mi.fit_transform(X, y)
mi_scores = pd.Series(selector_mi.scores_, index=feature_names)
print("\n相互情報量 Top10:")
print(mi_scores.nlargest(10))

# --- RFE（再帰的特徴量除去） ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(model, n_features_to_select=10, step=1)
rfe.fit(X, y)
selected = [f for f, s in zip(feature_names, rfe.support_) if s]
print(f"\nRFE選択特徴量: {selected}")

# --- RFECV（交差検証付きRFE） ---
rfecv = RFECV(model, step=1, cv=5, scoring="accuracy", min_features_to_select=5)
rfecv.fit(X, y)
print(f"\n最適特徴量数: {rfecv.n_features_}")
print(f"最適スコア: {rfecv.cv_results_['mean_test_score'].max():.4f}")

# --- 特徴量重要度（モデルベース） ---
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=feature_names)
print("\n特徴量重要度 Top10:")
print(importances.nlargest(10))

# --- 相関行列による冗長特徴量の除去 ---
def remove_correlated(X, threshold=0.9):
    """高相関の特徴量ペアの片方を除去"""
    corr_matrix = pd.DataFrame(X).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return np.delete(X, to_drop, axis=1), to_drop

X_uncorr, dropped = remove_correlated(X, threshold=0.9)
print(f"\n相関除去: {X.shape[1]} → {X_uncorr.shape[1]} 特徴量（{len(dropped)}列除去）")
```

### 3.6 カスタム変換器の作成

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierClipper(BaseEstimator, TransformerMixin):
    """IQRベースの外れ値クリッピング変換器"""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.array(X)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.factor * IQR
        self.upper_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = np.array(X).copy()
        X = np.clip(X, self.lower_, self.upper_)
        return X

class FeatureInteraction(BaseEstimator, TransformerMixin):
    """特徴量の交互作用を追加する変換器"""

    def __init__(self, interaction_pairs=None):
        self.interaction_pairs = interaction_pairs

    def fit(self, X, y=None):
        if self.interaction_pairs is None:
            n_features = X.shape[1]
            from itertools import combinations
            self.interaction_pairs = list(combinations(range(n_features), 2))
        return self

    def transform(self, X):
        X = np.array(X)
        interactions = []
        for i, j in self.interaction_pairs:
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack([X] + interactions)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """日付列から特徴量を抽出する変換器"""

    def __init__(self, date_column: str, features=None):
        self.date_column = date_column
        self.features = features or [
            "year", "month", "day", "dayofweek",
            "quarter", "is_weekend", "day_of_year"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.date_column])

        result = pd.DataFrame(index=X.index)
        feature_map = {
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "dayofweek": dt.dayofweek,
            "quarter": dt.quarter,
            "is_weekend": dt.dayofweek.isin([5, 6]).astype(int),
            "day_of_year": dt.dayofyear,
            "week_of_year": dt.isocalendar().week.astype(int),
        }

        for feat in self.features:
            if feat in feature_map:
                result[f"{self.date_column}_{feat}"] = feature_map[feat]

        # 元の日付列を除去して返す
        other_cols = X.drop(columns=[self.date_column])
        return pd.concat([other_cols, result], axis=1)


class TargetEncoder(BaseEstimator, TransformerMixin):
    """ターゲットエンコーディング変換器（リーク防止付き）"""

    def __init__(self, columns=None, smoothing: float = 10.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y)
        self.global_mean_ = y.mean()
        self.encoding_map_ = {}

        cols = self.columns or X.select_dtypes(include=["object", "category"]).columns

        for col in cols:
            agg = y.groupby(X[col]).agg(["mean", "count"])
            # スムージング: サンプルが少ないカテゴリはグローバル平均に近づける
            smooth = (agg["count"] * agg["mean"] + self.smoothing * self.global_mean_) / \
                     (agg["count"] + self.smoothing)
            self.encoding_map_[col] = smooth.to_dict()

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, mapping in self.encoding_map_.items():
            X[col] = X[col].map(mapping).fillna(self.global_mean_)
        return X


# パイプラインで使用
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    ("clipper", OutlierClipper(factor=1.5)),
    ("interaction", FeatureInteraction()),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor()),
])
```

### 3.7 ハイパーパラメータ最適化

```python
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from scipy.stats import randint, uniform, loguniform
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# --- GridSearchCV（全組み合わせ探索） ---
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1,
)
grid.fit(X, y)
print(f"Grid最良: {grid.best_params_}, スコア: {grid.best_score_:.4f}")

# --- RandomizedSearchCV（ランダム探索、大規模パラメータ空間向き） ---
param_distributions = {
    "n_estimators": randint(50, 500),
    "max_depth": randint(3, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": uniform(0.1, 0.9),
}
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions, n_iter=100, cv=5, scoring="f1",
    n_jobs=-1, random_state=42, verbose=1,
)
random_search.fit(X, y)
print(f"Random最良: {random_search.best_params_}, スコア: {random_search.best_score_:.4f}")

# --- Optuna（ベイズ最適化、pip install optuna が必要） ---
"""
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1.0, log=True),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print(f"Optuna最良: {study.best_params}")
print(f"スコア: {study.best_value:.4f}")
"""

# --- 結果の分析 ---
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
print("\nTop5パラメータ組み合わせ:")
top5 = results.nsmallest(5, "rank_test_score")[
    ["params", "mean_test_score", "std_test_score", "rank_test_score"]
]
print(top5.to_string())
```

### 3.8 モデルの保存と読み込み

```python
import joblib
import json
from datetime import datetime
from pathlib import Path

def save_model(pipeline, metrics: dict, output_dir: str = "models/"):
    """モデルと付随情報を保存"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{output_dir}/model_{timestamp}.joblib"
    meta_path = f"{output_dir}/model_{timestamp}_meta.json"

    # モデル本体
    joblib.dump(pipeline, model_path)

    # メタデータ
    meta = {
        "timestamp": timestamp,
        "model_type": type(pipeline.named_steps.get("regressor",
                          pipeline.named_steps.get("model"))).__name__,
        "metrics": metrics,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": __import__("platform").python_version(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"モデル保存: {model_path}")
    print(f"メタデータ: {meta_path}")
    return model_path

def load_model(model_path: str):
    """モデルの読み込みと検証"""
    pipeline = joblib.load(model_path)
    meta_path = model_path.replace(".joblib", "_meta.json")

    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"モデル種別: {meta['model_type']}")
        print(f"学習日時: {meta['timestamp']}")
        print(f"評価指標: {meta['metrics']}")

    return pipeline


# --- ONNX形式でのエクスポート（推論高速化） ---
"""
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# sklearn Pipeline → ONNX
initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# ONNX Runtime で推論
import onnxruntime as rt
sess = rt.InferenceSession("model.onnx")
pred = sess.run(None, {"float_input": X_test.astype(np.float32)})[0]
"""
```

### 3.9 アンサンブル学習

```python
from sklearn.ensemble import (
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, AdaBoostClassifier,
    RandomForestClassifier, GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
import numpy as np

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# --- Voting（投票） ---
voting = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    voting="soft",  # 確率ベースの投票
    weights=[2, 2, 1],  # RFとGBに重み
)
scores_voting = cross_val_score(voting, X, y, cv=5, scoring="accuracy")
print(f"Voting: {scores_voting.mean():.4f} ± {scores_voting.std():.4f}")

# --- Stacking（スタッキング） ---
stacking = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=False,  # 元の特徴量をメタ学習器に渡さない
)
scores_stacking = cross_val_score(stacking, X, y, cv=5, scoring="accuracy")
print(f"Stacking: {scores_stacking.mean():.4f} ± {scores_stacking.std():.4f}")

# --- 個別モデルとの比較 ---
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVC": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 4. Matplotlib / Seaborn — データ可視化

### 4.1 Matplotlibの基本構成

```python
import matplotlib.pyplot as plt
import numpy as np

# --- Figure/Axes構成の理解 ---
# matplotlib のオブジェクト階層:
# Figure > Axes > (Line2D, Text, Patch, ...)

# 基本: サブプロットの作成
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 左上: 折れ線グラフ
x = np.linspace(0, 2 * np.pi, 100)
axes[0, 0].plot(x, np.sin(x), label="sin(x)", color="blue")
axes[0, 0].plot(x, np.cos(x), label="cos(x)", color="red", linestyle="--")
axes[0, 0].set_title("三角関数")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 右上: ヒストグラム
data = np.random.randn(1000)
axes[0, 1].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
axes[0, 1].axvline(data.mean(), color="red", linestyle="--", label=f"平均={data.mean():.2f}")
axes[0, 1].set_title("正規分布ヒストグラム")
axes[0, 1].legend()

# 左下: 散布図
x = np.random.randn(200)
y = 2 * x + np.random.randn(200) * 0.5
colors = np.random.rand(200)
axes[1, 0].scatter(x, y, c=colors, cmap="viridis", alpha=0.6, s=30)
axes[1, 0].set_title("散布図")
axes[1, 0].set_xlabel("X")
axes[1, 0].set_ylabel("Y")

# 右下: 棒グラフ
categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]
bars = axes[1, 1].bar(categories, values, color=["#ff6b6b", "#4ecdc4", "#45b7d1",
                                                   "#96ceb4", "#ffeaa7"])
axes[1, 1].set_title("棒グラフ")
for bar, val in zip(bars, values):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(val), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("basic_plots.png", dpi=150, bbox_inches="tight")
plt.close()
```

### 4.2 Seabornによる統計可視化

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# サンプルデータ
np.random.seed(42)
n = 300
df = pd.DataFrame({
    "age": np.random.normal(40, 10, n).astype(int),
    "income": np.random.lognormal(11, 0.5, n).astype(int),
    "education": np.random.choice(["高卒", "大卒", "院卒"], n, p=[0.3, 0.5, 0.2]),
    "satisfaction": np.random.choice(["低", "中", "高"], n, p=[0.2, 0.5, 0.3]),
})

# スタイル設定
sns.set_theme(style="whitegrid", font_scale=1.1)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. 相関行列ヒートマップ
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="RdBu_r", center=0,
            ax=axes[0, 0], fmt=".2f")
axes[0, 0].set_title("相関行列")

# 2. カテゴリ別箱ひげ図
sns.boxplot(data=df, x="education", y="income", ax=axes[0, 1],
            order=["高卒", "大卒", "院卒"], palette="Set2")
axes[0, 1].set_title("学歴別収入分布")

# 3. バイオリンプロット
sns.violinplot(data=df, x="satisfaction", y="age", ax=axes[0, 2],
               order=["低", "中", "高"], palette="muted", inner="quart")
axes[0, 2].set_title("満足度別年齢分布")

# 4. KDEプロット
for edu in ["高卒", "大卒", "院卒"]:
    subset = df[df["education"] == edu]["income"]
    sns.kdeplot(subset, ax=axes[1, 0], label=edu, fill=True, alpha=0.3)
axes[1, 0].set_title("学歴別収入密度")
axes[1, 0].legend()

# 5. カウントプロット
sns.countplot(data=df, x="education", hue="satisfaction", ax=axes[1, 1],
              order=["高卒", "大卒", "院卒"], hue_order=["低", "中", "高"],
              palette="coolwarm")
axes[1, 1].set_title("学歴×満足度")

# 6. 散布図 + 回帰直線
sns.regplot(data=df, x="age", y="income", ax=axes[1, 2],
            scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
axes[1, 2].set_title("年齢と収入の関係")

plt.tight_layout()
plt.savefig("seaborn_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
```

### 4.3 ML結果の可視化

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# --- 学習曲線（過学習診断） ---
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy", n_jobs=-1,
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 学習曲線
axes[0].plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
axes[0].plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
axes[0].fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
axes[0].fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
axes[0].set_xlabel("Training Size")
axes[0].set_ylabel("Score")
axes[0].set_title("学習曲線")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC曲線
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC曲線")
axes[1].legend()

# 特徴量重要度
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
axes[2].barh(range(10), importances[indices], color="steelblue")
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([f"feature_{i}" for i in indices])
axes[2].set_title("特徴量重要度 Top10")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("ml_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
```

---

## 5. SciPy — 科学計算・統計検定

### 5.1 統計検定

```python
from scipy import stats
import numpy as np

np.random.seed(42)

# --- 正規性の検定 ---
data = np.random.normal(100, 15, 200)

# Shapiro-Wilk検定（n < 5000推奨）
stat_sw, p_sw = stats.shapiro(data)
print(f"Shapiro-Wilk: 統計量={stat_sw:.4f}, p値={p_sw:.4f}")

# Kolmogorov-Smirnov検定
stat_ks, p_ks = stats.kstest(data, "norm", args=(data.mean(), data.std()))
print(f"KS検定: 統計量={stat_ks:.4f}, p値={p_ks:.4f}")

# D'Agostino-Pearson検定
stat_dp, p_dp = stats.normaltest(data)
print(f"D'Agostino: 統計量={stat_dp:.4f}, p値={p_dp:.4f}")

# --- 2標本の比較 ---
group_a = np.random.normal(100, 15, 100)
group_b = np.random.normal(105, 15, 120)

# t検定（正規分布・等分散を仮定）
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"\nt検定: t={t_stat:.4f}, p={p_value:.4f}")

# Welchのt検定（等分散を仮定しない）
t_stat_w, p_value_w = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"Welch t検定: t={t_stat_w:.4f}, p={p_value_w:.4f}")

# Mann-Whitney U検定（ノンパラメトリック）
u_stat, p_mw = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
print(f"Mann-Whitney: U={u_stat:.0f}, p={p_mw:.4f}")

# 効果量（Cohen's d）
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

d = cohens_d(group_a, group_b)
print(f"Cohen's d: {d:.4f}")

# --- 多群比較（ANOVA） ---
g1 = np.random.normal(100, 15, 50)
g2 = np.random.normal(105, 15, 50)
g3 = np.random.normal(110, 15, 50)

f_stat, p_anova = stats.f_oneway(g1, g2, g3)
print(f"\nANOVA: F={f_stat:.4f}, p={p_anova:.4f}")

# Kruskal-Wallis（ノンパラメトリック版ANOVA）
h_stat, p_kw = stats.kruskal(g1, g2, g3)
print(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.4f}")

# --- カイ二乗検定（独立性の検定） ---
observed = np.array([[50, 30], [20, 40]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"\nカイ二乗検定: χ²={chi2:.4f}, p={p_chi:.4f}, 自由度={dof}")
print(f"期待度数:\n{expected}")
```

### 5.2 最適化

```python
from scipy.optimize import minimize, curve_fit, minimize_scalar
import numpy as np

# --- 関数の最小化 ---
def rosenbrock(x):
    """ローゼンブロック関数（最適化のベンチマーク）"""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

x0 = np.array([0.0, 0.0])  # 初期点
result = minimize(rosenbrock, x0, method="Nelder-Mead")
print(f"最小点: {result.x}")
print(f"最小値: {result.fun:.6f}")
print(f"収束: {result.success}")

# 制約付き最適化
from scipy.optimize import LinearConstraint, Bounds

# x + y <= 10, x >= 0, y >= 0
result_c = minimize(
    lambda x: (x[0] - 3)**2 + (x[1] - 5)**2,
    x0=[0, 0],
    method="SLSQP",
    constraints={"type": "ineq", "fun": lambda x: 10 - x[0] - x[1]},
    bounds=Bounds(0, np.inf),
)
print(f"\n制約付き最適化: x={result_c.x}, f={result_c.fun:.4f}")

# --- カーブフィッティング ---
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

# ノイズ付きデータ生成
x_data = np.linspace(0, 5, 50)
y_data = 3.0 * np.exp(-1.5 * x_data) + 0.5 + np.random.normal(0, 0.1, 50)

# フィッティング
popt, pcov = curve_fit(exp_decay, x_data, y_data, p0=[3, 1, 0.5])
perr = np.sqrt(np.diag(pcov))  # パラメータの標準誤差

print(f"\nフィッティング結果:")
print(f"a = {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"b = {popt[1]:.4f} ± {perr[1]:.4f}")
print(f"c = {popt[2]:.4f} ± {perr[2]:.4f}")
```

### 5.3 補間とスプライン

```python
from scipy.interpolate import (
    interp1d, CubicSpline, UnivariateSpline,
    RegularGridInterpolator,
)
import numpy as np

# --- 1次元補間 ---
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])

# 線形補間
f_linear = interp1d(x, y, kind="linear")

# 3次スプライン補間
f_cubic = interp1d(x, y, kind="cubic")

# CubicSpline（より高機能）
cs = CubicSpline(x, y)

x_new = np.linspace(0, 5, 100)
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)
y_cs = cs(x_new)
y_cs_deriv = cs(x_new, 1)  # 1次導関数

print(f"x=2.5での値: 線形={f_linear(2.5):.4f}, 3次={f_cubic(2.5):.4f}")
print(f"x=2.5での導関数: {cs(2.5, 1):.4f}")

# --- 2次元補間 ---
x_grid = np.linspace(0, 4, 5)
y_grid = np.linspace(0, 4, 5)
values = np.random.rand(5, 5)

interpolator = RegularGridInterpolator((x_grid, y_grid), values)
point = np.array([[2.1, 3.3]])
result = interpolator(point)
print(f"\n2次元補間 (2.1, 3.3): {result[0]:.4f}")
```

---

## 6. 実務パターン

### 6.1 プロジェクト構成テンプレート

```
ml-project/
├── data/
│   ├── raw/                  # 生データ（変更禁止）
│   ├── processed/            # 加工済みデータ
│   └── external/             # 外部データ
├── notebooks/
│   ├── 01_eda.ipynb          # 探索的データ分析
│   ├── 02_feature_eng.ipynb  # 特徴量エンジニアリング
│   └── 03_modeling.ipynb     # モデリング実験
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # データ読み込み
│   │   └── preprocessor.py   # 前処理
│   ├── features/
│   │   ├── __init__.py
│   │   └── builder.py        # 特徴量生成
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py        # 学習
│   │   └── predictor.py      # 推論
│   └── utils/
│       ├── __init__.py
│       └── metrics.py        # 評価指標
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── models/                   # 学習済みモデル
├── configs/
│   └── config.yaml           # ハイパーパラメータ等
├── Makefile
├── pyproject.toml
└── README.md
```

### 6.2 設定管理

```python
# configs/config.yaml
"""
data:
  raw_path: data/raw/train.csv
  test_path: data/raw/test.csv
  target_column: price

features:
  numeric_columns:
    - area
    - rooms
    - age_years
  categorical_columns:
    - location
    - building_type

model:
  type: gradient_boosting
  params:
    n_estimators: 200
    max_depth: 5
    learning_rate: 0.1
    random_state: 42

training:
  test_size: 0.2
  cv_folds: 5
  scoring: neg_mean_squared_error
"""

# src/config.py
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    raw_path: str
    test_path: str
    target_column: str


@dataclass
class FeatureConfig:
    numeric_columns: list
    categorical_columns: list


@dataclass
class ModelConfig:
    type: str
    params: dict = field(default_factory=dict)


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw["data"]),
            features=FeatureConfig(**raw["features"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
        )


# 使用例
# config = Config.from_yaml("configs/config.yaml")
# print(config.model.params)
```

### 6.3 MLパイプラインのテスト

```python
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class TestPipeline:
    """MLパイプラインのテストスイート"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        X = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100),
        })
        y = (X["feature_1"] + X["feature_2"] > 0).astype(int)
        return X, y

    @pytest.fixture
    def pipeline(self):
        """テスト用パイプライン"""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=10, random_state=42)),
        ])

    def test_pipeline_fit_predict(self, pipeline, sample_data):
        """パイプラインの学習と予測が正常に動作する"""
        X, y = sample_data
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_pipeline_accuracy(self, pipeline, sample_data):
        """最低限の精度を達成する"""
        X, y = sample_data
        pipeline.fit(X, y)
        accuracy = pipeline.score(X, y)
        assert accuracy > 0.7, f"精度が低すぎます: {accuracy:.4f}"

    def test_pipeline_predict_proba(self, pipeline, sample_data):
        """確率予測の出力形式が正しい"""
        X, y = sample_data
        pipeline.fit(X, y)
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_pipeline_unseen_data(self, pipeline, sample_data):
        """未知データに対して予測できる"""
        X, y = sample_data
        pipeline.fit(X, y)
        X_new = pd.DataFrame({
            "feature_1": [0.5, -0.5],
            "feature_2": [1.0, -1.0],
            "feature_3": [0.0, 0.0],
        })
        predictions = pipeline.predict(X_new)
        assert len(predictions) == 2

    def test_scaler_transform(self, sample_data):
        """StandardScalerの変換が正しい"""
        X, _ = sample_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)

    def test_feature_names_preserved(self, pipeline, sample_data):
        """特徴量名が保持される"""
        X, y = sample_data
        pipeline.fit(X, y)
        # sklearn 1.0+ ではget_feature_names_outが利用可能
        scaler = pipeline.named_steps["scaler"]
        assert hasattr(scaler, "feature_names_in_")
        assert list(scaler.feature_names_in_) == list(X.columns)
```

### 6.4 実験トラッキング（MLflow）

```python
"""
MLflowによる実験管理の基本パターン

pip install mlflow
"""

# import mlflow
# import mlflow.sklearn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.datasets import make_classification
# import numpy as np

# --- MLflow実験管理の雛形 ---
"""
# 実験の設定
mlflow.set_experiment("my-classification-experiment")

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# パラメータ候補
configs = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 7},
    {"n_estimators": 300, "max_depth": 10},
]

for config in configs:
    with mlflow.start_run():
        # パラメータの記録
        mlflow.log_params(config)

        # モデルの学習と評価
        model = RandomForestClassifier(**config, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

        # メトリクスの記録
        mlflow.log_metric("accuracy_mean", scores.mean())
        mlflow.log_metric("accuracy_std", scores.std())

        # モデルの保存
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")

        print(f"Config: {config}")
        print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# 結果の確認: mlflow ui
"""

# --- 独自の軽量実験トラッカー ---
import json
from datetime import datetime
from pathlib import Path
import hashlib


class ExperimentTracker:
    """軽量な実験管理ツール"""

    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir) / experiment_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs = []

    def log_run(self, params: dict, metrics: dict, tags: dict = None):
        """実験結果を記録"""
        run = {
            "run_id": hashlib.md5(
                json.dumps(params, sort_keys=True).encode()
            ).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
        }
        self.runs.append(run)

        # 個別ファイルに保存
        run_path = self.base_dir / f"run_{run['run_id']}.json"
        with open(run_path, "w") as f:
            json.dump(run, f, indent=2, ensure_ascii=False)

        return run["run_id"]

    def get_best_run(self, metric: str, mode: str = "max"):
        """最良の実験結果を取得"""
        if not self.runs:
            return None
        key = max if mode == "max" else min
        return key(self.runs, key=lambda r: r["metrics"].get(metric, float("-inf")))

    def summary(self):
        """全実験の概要を表示"""
        import pandas as pd
        if not self.runs:
            print("実験結果がありません")
            return

        rows = []
        for run in self.runs:
            row = {"run_id": run["run_id"], **run["params"], **run["metrics"]}
            rows.append(row)

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df


# 使用例
# tracker = ExperimentTracker("my-experiment")
# tracker.log_run(
#     params={"n_estimators": 100, "max_depth": 5},
#     metrics={"accuracy": 0.92, "f1": 0.90},
#     tags={"model": "RandomForest"},
# )
# tracker.summary()
```

---

## 比較表

### NumPy vs pandas vs Polars

| 項目 | NumPy | pandas | Polars |
|---|---|---|---|
| データ型 | 同型多次元配列 | 異型表形式 | 異型表形式 |
| 速度 | 極速（C/Fortran） | 中速 | 高速（Rust） |
| メモリ効率 | 高い | 中程度 | 高い |
| 遅延評価 | なし | なし | あり（LazyFrame） |
| API | 低レベル | 高レベル | 高レベル |
| 主な用途 | 数値計算、線形代数 | データ加工、EDA | 大規模データ処理 |
| 学習コスト | 中程度 | 低い | 中程度 |
| エコシステム | 広大 | 非常に広大 | 成長中 |
| GPU対応 | CuPy連携 | なし | GPU版開発中 |
| マルチスレッド | 限定的 | GIL制約 | ネイティブ対応 |

### scikit-learn モデル選択チートシート

| データ条件 | 推奨モデル | 訓練速度 | 解釈性 | 精度 |
|---|---|---|---|---|
| 小規模・線形関係 | LinearRegression / LogisticRegression | 極速 | 高い | 中 |
| 中規模・非線形 | RandomForest | 速い | 中程度 | 高い |
| 中規模・高精度 | GradientBoosting | 中程度 | 低い | 高い |
| 大規模・高次元 | SGDClassifier | 極速 | 中程度 | 中 |
| テキスト分類 | MultinomialNB | 極速 | 中程度 | 中 |
| 少量・高精度 | SVM (RBFカーネル) | 遅い | 低い | 高い |
| 外れ値検出 | IsolationForest | 速い | 低い | 中〜高 |
| クラスタリング | KMeans / DBSCAN | 速い | 中程度 | — |
| 次元削減 | PCA / t-SNE / UMAP | 中程度 | 低い | — |

### 前処理手法の選択ガイド

| データの特性 | 推奨スケーリング | 理由 |
|---|---|---|
| 正規分布に近い | StandardScaler | 平均0、標準偏差1に標準化 |
| 範囲が重要（0-1） | MinMaxScaler | 最小最大値でスケーリング |
| 外れ値が多い | RobustScaler | 中央値とIQRで頑健にスケーリング |
| 歪んだ分布 | PowerTransformer | Box-Cox/Yeo-Johnson変換 |
| 一様分布にしたい | QuantileTransformer | 分位数ベースの変換 |
| カテゴリ変数（名義） | OneHotEncoder | ダミー変数化 |
| カテゴリ変数（順序） | OrdinalEncoder | 順序を保持した数値化 |
| 高基数カテゴリ | TargetEncoder | ターゲット値の平均で数値化 |

---

## アンチパターン

### アンチパターン1: pandas のループ処理

```python
# BAD: iterrows で1行ずつ処理（極端に遅い）
for idx, row in df.iterrows():
    df.loc[idx, "new_col"] = row["a"] * row["b"] + row["c"]

# GOOD: ベクトル演算を使用
df["new_col"] = df["a"] * df["b"] + df["c"]

# GOOD: 複雑な条件は np.where か apply
df["category"] = np.where(df["value"] > 100, "high", "low")
```

### アンチパターン2: Pipeline を使わない前処理

```python
# BAD: 前処理とモデルが分離 → テスト時に変換忘れのリスク
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
model = RandomForestClassifier()
model.fit(X_train_s, y_train)
# テスト時に scaler.transform() を忘れる可能性大

# GOOD: Pipeline で一体化
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
pipe.fit(X_train, y_train)        # 前処理+学習が一括
score = pipe.score(X_test, y_test) # 前処理+予測が一括
```

### アンチパターン3: データリーク

```python
# BAD: 全データでfit_transformしてから分割（データリーク）
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_scaled = StandardScaler().fit_transform(X)  # テストデータの情報が漏れる！
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# GOOD: 分割後に学習データのみでfit
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 学習データでfit
X_test_scaled = scaler.transform(X_test)          # テストデータはtransformのみ

# BEST: Pipelineを使えばリークは構造的に防止される
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), RandomForestClassifier())
# cross_val_score内部で正しくfit/transformが分離される
scores = cross_val_score(pipe, X, y, cv=5)
```

### アンチパターン4: 不適切な評価指標

```python
# BAD: 不均衡データでaccuracyのみを評価
# クラス比 95:5 の場合、常に多数派を予測するだけでaccuracy=95%
from sklearn.metrics import accuracy_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")  # 見かけ上は高い

# GOOD: 不均衡データでは複数指標を確認
from sklearn.metrics import classification_report, balanced_accuracy_score
print(classification_report(y_test, y_pred))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")

# GOOD: クラス重み付きの学習
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight="balanced", random_state=42)
# または class_weight={0: 1, 1: 10} で明示的に指定
```

### アンチパターン5: 再現性の欠如

```python
# BAD: シードを固定しない
model = RandomForestClassifier()  # 毎回異なる結果
X_train, X_test = train_test_split(X, y)  # 分割も毎回異なる

# GOOD: すべてのランダム要素にシードを設定
import numpy as np
SEED = 42
np.random.seed(SEED)

model = RandomForestClassifier(random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# BEST: seedをconfigファイルで一元管理
# config.yaml → seed: 42
```

---

## FAQ

### Q1: pandasとPolarsのどちらを使うべき？

**A:** 2024年時点ではpandasが圧倒的にエコシステムが広く、scikit-learnやmatplotlibとの連携もシームレス。Polarsはデータが100万行を超える場合や速度が重要な場面で威力を発揮する。新規プロジェクトでデータサイズが大きいならPolarsを検討し、それ以外はpandasが安全な選択。なお、pandas 2.0以降はArrowバックエンドが利用可能になり、性能差は縮まりつつある。Polarsは遅延評価（LazyFrame）により、クエリプランの最適化が自動で行われるため、大規模データ処理では顕著な速度優位がある。

### Q2: scikit-learnのPipelineはどこまでカスタマイズできる？

**A:** BaseEstimator + TransformerMixinを継承すれば任意の変換器を作成可能。ColumnTransformerで列ごとに異なる処理を適用でき、FeatureUnionで特徴量を結合できる。NestedCVやカスタムスコアラーも組み合わせれば、ほぼ全てのMLワークフローをPipelineで表現できる。さらに、`set_output(transform="pandas")`を使えばDataFrameを出力として維持でき、デバッグが容易になる。

### Q3: JupyterノートブックとPythonスクリプトの使い分けは？

**A:** 探索・可視化・レポーティングにはNotebook、本番パイプラインやテストコードにはスクリプト。Notebookで試行錯誤した後、確定したコードをsrc/以下のモジュールに移すのが理想。NotebookはGit管理しにくいため、nbstripoutでセル出力を除去してからコミットする。また、papermillを使えばNotebookをパラメータ化して自動実行でき、レポート生成パイプラインとして活用できる。

### Q4: NumPyとCuPyの切り替えはどうすればよい？

**A:** CuPyはNumPyと互換性の高いAPIを提供しており、`import cupy as np`とするだけで多くのコードがGPU上で動作する。ただし、データ転送のオーバーヘッドがあるため、小規模データではCPU（NumPy）の方が速い場合もある。`cupy.asnumpy()`と`cupy.asarray()`で明示的に変換する。scikit-learn互換のcuMLライブラリもあり、GPU上でscikit-learnと同じAPIでMLモデルを学習できる。

### Q5: 特徴量エンジニアリングの体系的なアプローチは？

**A:** 以下のステップで進めるのが効果的である。(1) ドメイン知識に基づく特徴量の設計、(2) 基本統計量（平均、分散、歪度、尖度）の算出、(3) 交互作用特徴量の生成、(4) 多項式特徴量の追加、(5) 時系列ならラグ・ローリング統計、(6) カテゴリ変数のエンコーディング、(7) 次元削減（PCA等）。その後、特徴量選択（RFE、重要度ベース）で不要な特徴量を除去する。Featuretoolsのような自動特徴量生成ライブラリも検討に値する。

### Q6: メモリ不足で大規模データを処理できない場合の対策は？

**A:** いくつかの段階的なアプローチがある。(1) dtypeの最適化（float64→float32、int64→int16等）、(2) カテゴリ列のcategory型変換、(3) チャンク処理（`pd.read_csv(chunksize=N)`）、(4) Parquet形式での列指向読み込み、(5) Daskによる分散DataFrame処理、(6) Polarsの遅延評価、(7) Vaexのメモリマップ処理。データベース（SQLite、DuckDB）を中間層として使う手法も有効。DuckDBはSQLでParquetファイルを直接クエリでき、pandasとの連携も良好である。

### Q7: scikit-learnの代わりにXGBoost/LightGBMを使うべき場面は？

**A:** テーブルデータの予測タスクでは、XGBoostやLightGBMはscikit-learnのGradientBoostingよりも高速で高精度な場合が多い。特に(1) データが大規模（10万行以上）、(2) カテゴリ特徴量が多い（LightGBMのネイティブカテゴリ対応）、(3) 欠損値が多い（ネイティブ欠損値対応）、(4) Kaggle等の競技プログラミング、の場合に推奨される。ただし、scikit-learnのPipeline/GridSearchCVとの統合も容易（sklearn API互換ラッパーあり）なので、まずscikit-learnで実装し、性能が不足すればXGBoost/LightGBMに切り替えるのが実務的なアプローチである。

---

## まとめ

| 項目 | 要点 |
|---|---|
| NumPy | ベクトル化演算で高速化。ループを避けブロードキャストを活用 |
| pandas | メソッドチェーンで可読性を高める。大規模データは型最適化 |
| scikit-learn | Pipeline + ColumnTransformer で再現性のあるワークフローを構築 |
| Matplotlib/Seaborn | 探索的分析にはSeaborn、細かいカスタマイズにはMatplotlib |
| SciPy | 統計検定・最適化・補間など科学計算の基盤 |
| モデル保存 | joblibで保存。メタデータ（バージョン、指標）を併せて記録 |
| カスタム変換器 | BaseEstimator + TransformerMixin で独自の前処理をPipeline統合 |
| 特徴量選択 | 統計検定→RFE→重要度の順で段階的に選択 |
| 実験管理 | MLflowまたは軽量トラッカーで再現性を確保 |
| テスト | パイプラインの入出力・精度・エッジケースを必ずテスト |

---

## 次に読むべきガイド

- [../01-classical-ml/00-regression.md](../01-classical-ml/00-regression.md) — 回帰モデルの実装と評価
- [../01-classical-ml/01-classification.md](../01-classical-ml/01-classification.md) — 分類モデルの実装と評価

---

## 参考文献

1. **Jake VanderPlas** "Python Data Science Handbook" O'Reilly Media, 2016 — https://jakevdp.github.io/PythonDataScienceHandbook/
2. **scikit-learn Documentation** "API Reference" — https://scikit-learn.org/stable/modules/classes.html
3. **Wes McKinney** "Python for Data Analysis" 3rd Edition, O'Reilly Media, 2022
4. **NumPy Documentation** "NumPy User Guide" — https://numpy.org/doc/stable/user/
5. **SciPy Documentation** "SciPy Reference Guide" — https://docs.scipy.org/doc/scipy/reference/
6. **Matplotlib Documentation** "Tutorials" — https://matplotlib.org/stable/tutorials/
7. **Seaborn Documentation** "Tutorial" — https://seaborn.pydata.org/tutorial.html
8. **Aurélien Géron** "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" 3rd Edition, O'Reilly Media, 2022
