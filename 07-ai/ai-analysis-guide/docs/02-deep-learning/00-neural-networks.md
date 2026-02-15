# ニューラルネットワーク — パーセプトロン、活性化関数、誤差逆伝播

> ニューラルネットワークの基礎理論をゼロから構築し、深層学習への橋渡しを行う

## この章で学ぶこと

1. **パーセプトロンとMLP** — 単層から多層への発展と表現能力の向上
2. **活性化関数** — ReLU、Sigmoid、Softmax等の特性と選択基準
3. **誤差逆伝播法** — 勾配の自動計算メカニズムとパラメータ最適化
4. **正則化手法** — Dropout、BatchNorm、Weight Decay等の過学習対策
5. **実務的なモデル設計** — アーキテクチャ設計指針とデバッグ手法

---

## 1. ニューラルネットワークの歴史と理論的背景

### 1.1 パーセプトロンから深層学習へ

```
年代別の発展:

1943  McCulloch-Pitts: 形式ニューロンモデル
      ├── 生物学的ニューロンの数学的抽象化
      └── 入力の線形結合 → 閾値関数

1958  Rosenblatt: パーセプトロン
      ├── 単層の学習可能なネットワーク
      ├── パーセプトロン学習則（線形分離可能な問題のみ）
      └── 重み更新: w ← w + η(y - ŷ)x

1969  Minsky & Papert: パーセプトロンの限界
      ├── XOR問題を解けないことを証明
      └── → 第1次AIの冬

1986  Rumelhart, Hinton, Williams: 誤差逆伝播法
      ├── 多層ネットワークの学習を実現
      ├── 連鎖律による勾配計算
      └── → ニューラルネットワーク復活

2006  Hinton: Deep Belief Networks
      ├── 事前学習による深層ネットワークの学習
      └── → 深層学習の幕開け

2012  Krizhevsky: AlexNet
      ├── ImageNet competition優勝
      ├── GPU計算 + ReLU + Dropout
      └── → 深層学習ブーム

2017  Vaswani: Transformer
      ├── Attention Is All You Need
      ├── 自己注意機構
      └── → LLM時代の到来
```

### 1.2 生物学的ニューロンと人工ニューロン

```
生物学的ニューロン:
  樹状突起（入力）→ 細胞体（統合）→ 軸索（出力）→ シナプス（接続）

  ・ 入力信号の重み付き和が閾値を超えると発火
  ・ シナプスの強度が学習に対応（Hebbの法則）
  ・ 約860億のニューロン、100兆のシナプス結合

人工ニューロン:
  x₁w₁ + x₂w₂ + ... + xₙwₙ + b → 活性化関数 → 出力

  ・ 入力 × 重み の線形結合 + バイアス
  ・ 非線形活性化関数による変換
  ・ 学習 = 重みとバイアスの最適化
```

---

## 2. ニューラルネットワークの構造

### 2.1 MLP（多層パーセプトロン）のアーキテクチャ

```
入力層          隠れ層1         隠れ層2         出力層
(d次元)        (h1ユニット)    (h2ユニット)    (c次元)

 x₁ ─────┐   ┌─── h₁¹ ───┐   ┌─── h₁² ───┐   ┌─── y₁
          ├──>│            ├──>│            ├──>│
 x₂ ─────┤   ├─── h₂¹ ───┤   ├─── h₂² ───┤   ├─── y₂
          ├──>│            ├──>│            ├──>│
 x₃ ─────┤   ├─── h₃¹ ───┤   ├─── h₃² ───┤   └─── y₃
          ├──>│            ├──>│            │
 x₄ ─────┘   └─── h₄¹ ───┘   └─── h₃² ───┘

 各矢印 = 重み (w) + バイアス (b)
 各ユニット: z = Σ(wᵢxᵢ) + b  →  a = σ(z)  (活性化関数)

 全結合層の計算:
   Z = XW + b        (線形変換)
   A = σ(Z)          (非線形活性化)
```

### 2.2 万能近似定理（Universal Approximation Theorem）

```
定理:
  十分な数のユニットを持つ1層の隠れ層を持つMLPは、
  任意の連続関数をコンパクト集合上で任意精度で近似できる。

  ⚠️ 注意: 「近似可能」と「学習可能」は別
  ・ 定理は存在を保証するが、SGDで見つけられるかは別問題
  ・ 深いネットワークの方が効率的な表現が可能
  ・ 実務では層を深くして各層のユニット数を抑える

深さの利点:
  ・ 指数的な効率向上（浅いモデルで同じ表現に必要なユニット数は指数的に増大）
  ・ 階層的な特徴抽出（低レベル→高レベルの特徴）
  ・ パラメータ効率が良い
```

### コード例1: ニューラルネットワークのフル実装（NumPyのみ）

```python
import numpy as np

class NeuralNetwork:
    """NumPyだけで実装するフルスクラッチNN"""

    def __init__(self, layer_sizes: list, learning_rate: float = 0.01):
        self.layers = layer_sizes
        self.lr = learning_rate
        self.params = {}
        self.cache = {}

        # Xavier初期化
        for i in range(1, len(layer_sizes)):
            scale = np.sqrt(2.0 / layer_sizes[i-1])
            self.params[f"W{i}"] = np.random.randn(
                layer_sizes[i-1], layer_sizes[i]) * scale
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i]))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_grad(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        """順伝播"""
        self.cache["A0"] = X
        A = X

        # 隠れ層: ReLU
        for i in range(1, len(self.layers) - 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self.relu(Z)
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A

        # 出力層: Softmax
        i = len(self.layers) - 1
        Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
        A = self.softmax(Z)
        self.cache[f"Z{i}"] = Z
        self.cache[f"A{i}"] = A

        return A

    def backward(self, y_onehot):
        """誤差逆伝播"""
        m = y_onehot.shape[0]
        grads = {}
        L = len(self.layers) - 1

        # 出力層の勾配 (Cross-Entropy + Softmax)
        dZ = self.cache[f"A{L}"] - y_onehot
        grads[f"dW{L}"] = (self.cache[f"A{L-1}"].T @ dZ) / m
        grads[f"db{L}"] = np.mean(dZ, axis=0, keepdims=True)

        # 隠れ層の勾配
        for i in range(L - 1, 0, -1):
            dA = dZ @ self.params[f"W{i+1}"].T
            dZ = dA * self.relu_grad(self.cache[f"Z{i}"])
            grads[f"dW{i}"] = (self.cache[f"A{i-1}"].T @ dZ) / m
            grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)

        # パラメータ更新
        for i in range(1, L + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]

        return grads

    def train(self, X, y, epochs=100, verbose=True):
        """学習ループ"""
        # One-hotエンコーディング
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]

        history = []
        for epoch in range(epochs):
            probs = self.forward(X)
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
            self.backward(y_onehot)
            history.append(loss)

            if verbose and (epoch + 1) % 20 == 0:
                acc = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Epoch {epoch+1:4d}  Loss={loss:.4f}  Acc={acc:.4f}")

        return history

# 使用例
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = NeuralNetwork([4, 32, 16, 3], learning_rate=0.1)
history = nn.train(X_train, y_train, epochs=200)

# テスト精度
probs = nn.forward(X_test)
y_pred = np.argmax(probs, axis=1)
print(f"\nテスト精度: {np.mean(y_pred == y_test):.4f}")
```

### コード例1b: ミニバッチ対応の拡張版NN

```python
import numpy as np

class MiniBatchNN:
    """ミニバッチSGD、Dropout、BatchNorm対応のNN"""

    def __init__(self, layer_sizes, learning_rate=0.001,
                 dropout_rate=0.0, use_batchnorm=False):
        self.layers = layer_sizes
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.bn_params = {}
        self.cache = {}
        self.training = True

        # He初期化
        for i in range(1, len(layer_sizes)):
            fan_in = layer_sizes[i-1]
            fan_out = layer_sizes[i]
            self.params[f"W{i}"] = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            self.params[f"b{i}"] = np.zeros((1, fan_out))

            # BatchNormパラメータ
            if use_batchnorm and i < len(layer_sizes) - 1:
                self.bn_params[f"gamma{i}"] = np.ones((1, fan_out))
                self.bn_params[f"beta{i}"] = np.zeros((1, fan_out))
                self.bn_params[f"running_mean{i}"] = np.zeros((1, fan_out))
                self.bn_params[f"running_var{i}"] = np.ones((1, fan_out))

    def batch_norm_forward(self, Z, layer_idx):
        """BatchNormalization順伝播"""
        if self.training:
            mean = np.mean(Z, axis=0, keepdims=True)
            var = np.var(Z, axis=0, keepdims=True)
            Z_norm = (Z - mean) / np.sqrt(var + 1e-8)

            # 移動平均の更新
            momentum = 0.9
            key_mean = f"running_mean{layer_idx}"
            key_var = f"running_var{layer_idx}"
            self.bn_params[key_mean] = momentum * self.bn_params[key_mean] + (1 - momentum) * mean
            self.bn_params[key_var] = momentum * self.bn_params[key_var] + (1 - momentum) * var

            self.cache[f"bn_mean{layer_idx}"] = mean
            self.cache[f"bn_var{layer_idx}"] = var
            self.cache[f"bn_norm{layer_idx}"] = Z_norm
        else:
            mean = self.bn_params[f"running_mean{layer_idx}"]
            var = self.bn_params[f"running_var{layer_idx}"]
            Z_norm = (Z - mean) / np.sqrt(var + 1e-8)

        gamma = self.bn_params[f"gamma{layer_idx}"]
        beta = self.bn_params[f"beta{layer_idx}"]
        return gamma * Z_norm + beta

    def dropout_forward(self, A, layer_idx):
        """Dropout順伝播（逆スケーリング方式）"""
        if self.training and self.dropout_rate > 0:
            mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
            self.cache[f"dropout_mask{layer_idx}"] = mask
            return A * mask / (1 - self.dropout_rate)  # inverted dropout
        return A

    def forward(self, X):
        """順伝播"""
        self.cache["A0"] = X
        A = X

        for i in range(1, len(self.layers) - 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]

            if self.use_batchnorm:
                Z = self.batch_norm_forward(Z, i)

            A = np.maximum(0, Z)  # ReLU
            A = self.dropout_forward(A, i)

            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A

        # 出力層
        L = len(self.layers) - 1
        Z = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = exp_z / exp_z.sum(axis=1, keepdims=True)
        self.cache[f"Z{L}"] = Z
        self.cache[f"A{L}"] = A

        return A

    def compute_loss(self, y_onehot, l2_lambda=0.0):
        """交差エントロピー損失 + L2正則化"""
        L = len(self.layers) - 1
        probs = self.cache[f"A{L}"]
        m = y_onehot.shape[0]

        ce_loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))

        if l2_lambda > 0:
            l2_reg = sum(np.sum(self.params[f"W{i}"] ** 2)
                         for i in range(1, L + 1))
            ce_loss += l2_lambda / (2 * m) * l2_reg

        return ce_loss

    def train(self, X, y, epochs=100, batch_size=32,
              l2_lambda=0.0, verbose=True):
        """ミニバッチ学習"""
        n_classes = len(np.unique(y))
        y_onehot = np.eye(n_classes)[y]
        m = X.shape[0]
        history = {"loss": [], "acc": []}

        for epoch in range(epochs):
            # シャッフル
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                self.training = True
                self.forward(X_batch)
                loss = self.compute_loss(y_batch, l2_lambda)
                self._backward(y_batch, l2_lambda)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                self.training = False
                probs = self.forward(X)
                acc = np.mean(np.argmax(probs, axis=1) == y)
                history["acc"].append(acc)
                print(f"Epoch {epoch+1:4d}  Loss={avg_loss:.4f}  Acc={acc:.4f}")

        return history

    def _backward(self, y_onehot, l2_lambda=0.0):
        """ミニバッチ対応逆伝播"""
        m = y_onehot.shape[0]
        L = len(self.layers) - 1

        dZ = self.cache[f"A{L}"] - y_onehot

        for i in range(L, 0, -1):
            dW = (self.cache[f"A{i-1}"].T @ dZ) / m
            db = np.mean(dZ, axis=0, keepdims=True)

            if l2_lambda > 0:
                dW += (l2_lambda / m) * self.params[f"W{i}"]

            if i > 1:
                dA = dZ @ self.params[f"W{i}"].T

                # Dropout逆伝播
                if self.dropout_rate > 0 and f"dropout_mask{i-1}" in self.cache:
                    dA *= self.cache[f"dropout_mask{i-1}"] / (1 - self.dropout_rate)

                dZ = dA * (self.cache[f"Z{i-1}"] > 0).astype(float)

            self.params[f"W{i}"] -= self.lr * dW
            self.params[f"b{i}"] -= self.lr * db


# 使用例: MNISTの手書き数字分類
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MiniBatchNN(
    layer_sizes=[64, 128, 64, 10],
    learning_rate=0.005,
    dropout_rate=0.2,
    use_batchnorm=True
)
history = model.train(X_train, y_train, epochs=100, batch_size=32)

model.training = False
probs = model.forward(X_test)
y_pred = np.argmax(probs, axis=1)
print(f"テスト精度: {np.mean(y_pred == y_test):.4f}")
```

---

## 3. 活性化関数

### 3.1 活性化関数の比較（図解）

```
ReLU                    Sigmoid                  Tanh
f(x) = max(0, x)       f(x) = 1/(1+e^(-x))    f(x) = (e^x-e^(-x))/(e^x+e^(-x))

  │      /              │    ____               │     ____
  │     /            1.0│___/                 1.0│___/
  │    /               │  /                     │ /
  │   /             0.5│ /                   0.0│/
  │  /                 │/                       │
  │ /               0.0│                    -1.0│
──┤/─────           ───┤─────               ────┤─────
  │                    │                        │

LeakyReLU               GELU                    Swish
f(x) = max(αx, x)      f(x) = x·Φ(x)         f(x) = x·σ(x)

  │      /              │      __/              │      __/
  │     /               │    _/                 │    _/
  │    /                │  _/                   │  _/
  │   /                 │_/                     │_/
  │  /                  │                       │
  │/                    │                       │
─/┤─────            ────┤─────              ────┤─────
 /│                     │                       │
```

### 3.2 活性化関数の数学的詳細

```
■ ReLU (Rectified Linear Unit)
  f(x) = max(0, x)
  f'(x) = 1  (x > 0),  0  (x ≤ 0)
  ・ 計算が高速（比較演算のみ）
  ・ 勾配消失しにくい（正の領域で勾配=1）
  ・ Dead Neuron問題：負の入力が続くと永久に0
  ・ 出力が非対称（0以上のみ）

■ Leaky ReLU
  f(x) = x  (x > 0),  αx  (x ≤ 0)   (通常 α = 0.01)
  f'(x) = 1  (x > 0),  α  (x ≤ 0)
  ・ Dead Neuron問題を緩和
  ・ αをパラメータとして学習可能（PReLU）

■ ELU (Exponential Linear Unit)
  f(x) = x  (x > 0),  α(e^x - 1)  (x ≤ 0)
  ・ 負の出力を許容 → 平均出力が0に近くなる
  ・ 指数関数の計算コスト

■ GELU (Gaussian Error Linear Unit)
  f(x) = x · Φ(x)  （Φ: 標準正規分布の累積分布関数）
  近似: f(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
  ・ BERT、GPT、Vision Transformerで標準採用
  ・ 確率的な解釈が可能

■ Swish / SiLU
  f(x) = x · σ(βx)  （σ: sigmoid, β: 学習可能パラメータ）
  ・ β=1のとき SiLU (Sigmoid Linear Unit)
  ・ GoogleのAutoML探索で発見
  ・ 非単調な関数（x<0で負の出力あり）

■ Mish
  f(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))
  ・ YOLOv4で採用
  ・ Swishと類似の特性だがより滑らか
```

### コード例2: 活性化関数の実装と比較

```python
import numpy as np
import matplotlib.pyplot as plt

class Activations:
    """主要な活性化関数とその導関数"""

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_grad(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_grad(x, alpha=1.0):
        return np.where(x > 0, 1.0, alpha * np.exp(x))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def sigmoid_grad(x):
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_grad(x):
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish(x, beta=1.0):
        return x * Activations.sigmoid(beta * x)

    @staticmethod
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))

# 可視化: 活性化関数とその導関数
x = np.linspace(-4, 4, 1000)
functions = {
    "ReLU": (Activations.relu, Activations.relu_grad),
    "LeakyReLU": (Activations.leaky_relu, Activations.leaky_relu_grad),
    "ELU": (Activations.elu, Activations.elu_grad),
    "Sigmoid": (Activations.sigmoid, Activations.sigmoid_grad),
    "Tanh": (Activations.tanh, Activations.tanh_grad),
    "GELU": (Activations.gelu, None),
    "Swish": (Activations.swish, None),
    "Mish": (Activations.mish, None),
}

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for ax, (name, (func, grad_func)) in zip(axes.flatten(), functions.items()):
    ax.plot(x, func(x), linewidth=2, label=f"{name}")
    if grad_func is not None:
        ax.plot(x, grad_func(x), linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"{name}' (導関数)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_title(name, fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)

plt.tight_layout()
plt.savefig("reports/activation_functions.png", dpi=150)
plt.close()
```

### 3.3 活性化関数の選択指針

```
タスク別推奨:

隠れ層:
  ├── 一般的なMLP / CNN → ReLU（デフォルト）
  ├── Dead Neuron が問題 → LeakyReLU / ELU
  ├── Transformer → GELU
  ├── 深い ResNet → ReLU + Skip Connection
  └── 物体検出（YOLO） → Mish / Swish

出力層:
  ├── 二値分類 → Sigmoid（出力1ノード）
  ├── 多クラス分類 → Softmax（出力Nノード）
  ├── 回帰 → 線形（活性化なし）
  ├── 回帰（正の値のみ） → ReLU / Softplus
  └── 回帰（区間 [a,b]） → Sigmoid × (b-a) + a

実務上の判断基準:
  1. まずReLUで試す（計算効率が最も高い）
  2. 学習が停滞したらLeakyReLU/ELUに切り替え
  3. Transformerベースならデフォルトでbyebyeシグモイド
  4. 最終的にはベンチマークで比較
```

---

## 4. 損失関数の詳細

### 4.1 分類タスクの損失関数

```python
import numpy as np

class LossFunctions:
    """主要な損失関数の実装"""

    @staticmethod
    def binary_cross_entropy(y_true, y_pred, eps=1e-8):
        """二値分類用交差エントロピー"""
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    @staticmethod
    def categorical_cross_entropy(y_true_onehot, y_pred, eps=1e-8):
        """多クラス分類用交差エントロピー"""
        y_pred = np.clip(y_pred, eps, 1.0)
        return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

    @staticmethod
    def focal_loss(y_true_onehot, y_pred, gamma=2.0, alpha=0.25, eps=1e-8):
        """Focal Loss — クラス不均衡対策"""
        y_pred = np.clip(y_pred, eps, 1.0)
        # 正解クラスの確率
        pt = np.sum(y_true_onehot * y_pred, axis=1)
        # 難易度に応じた重み付け
        focal_weight = alpha * (1 - pt) ** gamma
        loss = -focal_weight * np.log(pt)
        return np.mean(loss)

    @staticmethod
    def label_smoothing_ce(y_true, y_pred, num_classes, smoothing=0.1, eps=1e-8):
        """ラベルスムージング交差エントロピー"""
        y_pred = np.clip(y_pred, eps, 1.0)
        # スムーズなラベル: (1-ε)δ(k,y) + ε/K
        smooth_labels = np.full_like(y_pred, smoothing / num_classes)
        smooth_labels[np.arange(len(y_true)), y_true] = 1 - smoothing + smoothing / num_classes
        return -np.mean(np.sum(smooth_labels * np.log(y_pred), axis=1))

    @staticmethod
    def mse_loss(y_true, y_pred):
        """平均二乗誤差（回帰用）"""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae_loss(y_true, y_pred):
        """平均絶対誤差（外れ値に頑健）"""
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """Huber損失（MSEとMAEのハイブリッド）"""
        diff = y_true - y_pred
        return np.mean(np.where(
            np.abs(diff) <= delta,
            0.5 * diff ** 2,
            delta * (np.abs(diff) - 0.5 * delta)
        ))


# 損失関数の挙動比較
y_true = np.array([1.0])
y_preds = np.linspace(0.01, 0.99, 100)

bce_losses = [LossFunctions.binary_cross_entropy(y_true, np.array([p])) for p in y_preds]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_preds, bce_losses, label="Binary CE", linewidth=2)
plt.xlabel("予測確率 p(y=1)")
plt.ylabel("損失")
plt.title("二値交差エントロピー損失（正解=1の場合）")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/bce_loss_curve.png", dpi=150)
plt.close()
```

### 4.2 損失関数の選択ガイド

```
タスク                    損失関数                備考
──────────────────────────────────────────────────────────────
二値分類                  Binary CE               標準
多クラス分類              Categorical CE          標準
クラス不均衡分類          Focal Loss              γ=2.0が一般的
過学習抑制                Label Smoothing CE      ε=0.1が一般的
回帰                      MSE                     外れ値に敏感
回帰（外れ値あり）        MAE / Huber             頑健性が高い
順序回帰                  Ordinal CE              順序関係を保持
物体検出（位置）          Smooth L1 (Huber)       高速R-CNN系
セグメンテーション        Dice Loss + CE          IoU最適化
生成モデル                Adversarial Loss        GAN系
```

---

## 5. 誤差逆伝播法と最適化

### 5.1 逆伝播の計算グラフ

```
順伝播 (Forward):
  x → [Linear] → z → [ReLU] → a → [Linear] → z → [Softmax+CE] → Loss
       W₁,b₁              W₂,b₂

逆伝播 (Backward):
  ∂L/∂x ← ∂L/∂z ← ∂L/∂a ← ∂L/∂z ← ∂L/∂ŷ ← ∂L/∂Loss = 1
            │                  │
            v                  v
         ∂L/∂W₁             ∂L/∂W₂
         ∂L/∂b₁             ∂L/∂b₂

連鎖律 (Chain Rule):
  ∂L/∂W₁ = ∂L/∂z₂ × ∂z₂/∂a₁ × ∂a₁/∂z₁ × ∂z₁/∂W₁
```

### 5.2 逆伝播の各層における勾配計算

```
■ 全結合層の逆伝播
  順伝播: Z = XW + b, A = σ(Z)
  逆伝播:
    ∂L/∂W = Aᵀ_prev · ∂L/∂Z  (重みの勾配)
    ∂L/∂b = mean(∂L/∂Z)       (バイアスの勾配)
    ∂L/∂A_prev = ∂L/∂Z · Wᵀ  (前の層への勾配伝播)

■ ReLU層の逆伝播
  順伝播: A = max(0, Z)
  逆伝播: ∂L/∂Z = ∂L/∂A · 1(Z > 0)

■ Softmax + Cross-Entropy（統合）
  順伝播: ŷ = softmax(Z), L = -Σ yₖ log(ŷₖ)
  逆伝播: ∂L/∂Z = ŷ - y  （非常にシンプル!）

■ BatchNorm層の逆伝播
  順伝播: Z_norm = (Z - μ) / √(σ² + ε), Y = γZ_norm + β
  逆伝播:
    ∂L/∂γ = Σ ∂L/∂Y · Z_norm
    ∂L/∂β = Σ ∂L/∂Y
    ∂L/∂Z_norm = ∂L/∂Y · γ
    ∂L/∂Z = 複雑な式（μ、σ²の勾配も計算が必要）
```

### コード例3: オプティマイザの実装比較

```python
import numpy as np

class Optimizers:
    """主要な最適化アルゴリズムの実装"""

    class SGD:
        def __init__(self, lr=0.01, momentum=0.0):
            self.lr = lr
            self.momentum = momentum
            self.velocity = {}

        def update(self, params, grads):
            for key in params:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
                params[key] += self.velocity[key]

    class NesterovSGD:
        """Nesterovの加速勾配法"""
        def __init__(self, lr=0.01, momentum=0.9):
            self.lr = lr
            self.momentum = momentum
            self.velocity = {}

        def update(self, params, grads):
            for key in params:
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(params[key])
                v_prev = self.velocity[key].copy()
                self.velocity[key] = self.momentum * self.velocity[key] - self.lr * grads[key]
                # Nesterov: 先読みの位置での勾配を使用
                params[key] += -self.momentum * v_prev + (1 + self.momentum) * self.velocity[key]

    class RMSProp:
        """RMSProp — 適応的学習率"""
        def __init__(self, lr=0.001, decay=0.99, eps=1e-8):
            self.lr = lr
            self.decay = decay
            self.eps = eps
            self.cache = {}

        def update(self, params, grads):
            for key in params:
                if key not in self.cache:
                    self.cache[key] = np.zeros_like(params[key])
                self.cache[key] = self.decay * self.cache[key] + (1 - self.decay) * grads[key] ** 2
                params[key] -= self.lr * grads[key] / (np.sqrt(self.cache[key]) + self.eps)

    class Adam:
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.m = {}  # 1次モーメント
            self.v = {}  # 2次モーメント
            self.t = 0

        def update(self, params, grads):
            self.t += 1
            for key in params:
                if key not in self.m:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

                # バイアス補正
                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)

                params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    class AdamW:
        """Adam + Weight Decay（L2正則化の改良版）"""
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                     eps=1e-8, weight_decay=0.01):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.wd = weight_decay
            self.m = {}
            self.v = {}
            self.t = 0

        def update(self, params, grads):
            self.t += 1
            for key in params:
                if key not in self.m:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)

                # Weight Decay は勾配更新とは別に適用
                params[key] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                          + self.wd * params[key])

    class LAMB:
        """LAMB — 大バッチ学習向け（BERT事前学習で使用）"""
        def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                     eps=1e-6, weight_decay=0.01):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.wd = weight_decay
            self.m = {}
            self.v = {}
            self.t = 0

        def update(self, params, grads):
            self.t += 1
            for key in params:
                if key not in self.m:
                    self.m[key] = np.zeros_like(params[key])
                    self.v[key] = np.zeros_like(params[key])

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2

                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)

                # Adam更新量
                update = m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * params[key]

                # Layer-wise適応的学習率
                w_norm = np.linalg.norm(params[key])
                u_norm = np.linalg.norm(update)
                trust_ratio = w_norm / (u_norm + self.eps) if w_norm > 0 and u_norm > 0 else 1.0

                params[key] -= self.lr * trust_ratio * update
```

### コード例4: 学習率スケジューラ

```python
import numpy as np
import matplotlib.pyplot as plt

class LRSchedulers:
    """学習率スケジューリング戦略"""

    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, drop_every=30):
        return initial_lr * (drop_rate ** (epoch // drop_every))

    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        return initial_lr * (decay_rate ** epoch)

    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=1e-6):
        return min_lr + (initial_lr - min_lr) * 0.5 * \
               (1 + np.cos(np.pi * epoch / total_epochs))

    @staticmethod
    def warmup_cosine(initial_lr, epoch, total_epochs,
                      warmup_epochs=10, min_lr=1e-6):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    @staticmethod
    def cosine_annealing_warm_restarts(initial_lr, epoch, T_0=10,
                                        T_mult=2, min_lr=1e-6):
        """Warm Restarts付きCosine Annealing (SGDR)"""
        T_cur = T_0
        epoch_in_cycle = epoch
        while epoch_in_cycle >= T_cur:
            epoch_in_cycle -= T_cur
            T_cur *= T_mult
        return min_lr + (initial_lr - min_lr) * 0.5 * \
               (1 + np.cos(np.pi * epoch_in_cycle / T_cur))

    @staticmethod
    def one_cycle(initial_lr, epoch, total_epochs,
                  max_lr=None, div_factor=25.0, final_div_factor=1e4):
        """1Cycle Policy（Super-Convergence）"""
        if max_lr is None:
            max_lr = initial_lr
        min_lr = max_lr / div_factor
        final_lr = max_lr / final_div_factor
        mid = total_epochs * 0.45

        if epoch < mid:
            # ウォームアップ: min_lr → max_lr
            return min_lr + (max_lr - min_lr) * epoch / mid
        elif epoch < total_epochs * 0.9:
            # クールダウン: max_lr → min_lr
            progress = (epoch - mid) / (total_epochs * 0.9 - mid)
            return max_lr - (max_lr - min_lr) * progress
        else:
            # 最終降下: min_lr → final_lr
            progress = (epoch - total_epochs * 0.9) / (total_epochs * 0.1)
            return min_lr - (min_lr - final_lr) * progress

# 可視化
epochs = range(200)
initial_lr = 0.01

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
schedulers = [
    ("Step Decay", lambda e: LRSchedulers.step_decay(initial_lr, e)),
    ("Exp Decay", lambda e: LRSchedulers.exponential_decay(initial_lr, e)),
    ("Cosine", lambda e: LRSchedulers.cosine_annealing(initial_lr, e, 200)),
    ("Warmup+Cosine", lambda e: LRSchedulers.warmup_cosine(initial_lr, e, 200)),
    ("Warm Restarts", lambda e: LRSchedulers.cosine_annealing_warm_restarts(initial_lr, e)),
    ("1Cycle", lambda e: LRSchedulers.one_cycle(0.001, e, 200, max_lr=0.01)),
]

for ax, (name, func) in zip(axes.flatten(), schedulers):
    lrs = [func(e) for e in epochs]
    ax.plot(epochs, lrs, linewidth=2)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("学習率")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("reports/lr_schedules.png", dpi=150)
plt.close()
```

---

## 6. 正則化手法

### 6.1 正則化手法の体系

```
正則化手法の分類:

■ データレベル
  ├── データ拡張（Augmentation）
  ├── ノイズ注入（入力、重み、勾配）
  └── ラベルスムージング

■ モデルレベル
  ├── L1正則化（Lasso）: |w| → スパース化
  ├── L2正則化（Ridge）: w² → 小さい重み
  ├── ElasticNet: L1 + L2
  └── Weight Decay（AdamW）

■ 構造レベル
  ├── Dropout: ランダムにユニットを無効化
  ├── DropConnect: ランダムに重みを無効化
  ├── Batch Normalization: 内部共変量シフト削減
  ├── Layer Normalization: バッチ非依存の正規化
  └── Stochastic Depth: ランダムに層をスキップ

■ 学習レベル
  ├── Early Stopping: 検証損失の監視
  ├── 学習率スケジューリング
  ├── Gradient Clipping: 勾配爆発防止
  └── Mixup / CutMix: 入力の混合
```

### コード例6: 正則化手法の実装

```python
import numpy as np

class RegularizationDemo:
    """正則化手法のデモ実装"""

    @staticmethod
    def l1_penalty(params, lambda_l1=0.001):
        """L1正則化（Lasso）"""
        penalty = 0.0
        grad_penalty = {}
        for key, w in params.items():
            if key.startswith("W"):  # バイアスには適用しない
                penalty += lambda_l1 * np.sum(np.abs(w))
                grad_penalty[key] = lambda_l1 * np.sign(w)
        return penalty, grad_penalty

    @staticmethod
    def l2_penalty(params, lambda_l2=0.001):
        """L2正則化（Ridge）"""
        penalty = 0.0
        grad_penalty = {}
        for key, w in params.items():
            if key.startswith("W"):
                penalty += 0.5 * lambda_l2 * np.sum(w ** 2)
                grad_penalty[key] = lambda_l2 * w
        return penalty, grad_penalty

    @staticmethod
    def elastic_net(params, lambda_l1=0.001, lambda_l2=0.001, l1_ratio=0.5):
        """ElasticNet正則化"""
        penalty = 0.0
        grad_penalty = {}
        for key, w in params.items():
            if key.startswith("W"):
                l1 = l1_ratio * lambda_l1 * np.sum(np.abs(w))
                l2 = 0.5 * (1 - l1_ratio) * lambda_l2 * np.sum(w ** 2)
                penalty += l1 + l2
                grad_penalty[key] = (
                    l1_ratio * lambda_l1 * np.sign(w) +
                    (1 - l1_ratio) * lambda_l2 * w
                )
        return penalty, grad_penalty

    @staticmethod
    def dropout(A, rate=0.5, training=True):
        """Inverted Dropout"""
        if not training or rate == 0:
            return A, None
        mask = (np.random.rand(*A.shape) > rate).astype(float)
        return A * mask / (1 - rate), mask

    @staticmethod
    def gradient_clipping(grads, max_norm=1.0):
        """勾配クリッピング（ノルムベース）"""
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        clip_coeff = max_norm / (total_norm + 1e-8)
        if clip_coeff < 1.0:
            for key in grads:
                grads[key] *= clip_coeff
        return grads, total_norm

    @staticmethod
    def mixup(X, y_onehot, alpha=0.2):
        """Mixupデータ拡張"""
        lam = np.random.beta(alpha, alpha)
        batch_size = X.shape[0]
        indices = np.random.permutation(batch_size)

        X_mixed = lam * X + (1 - lam) * X[indices]
        y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]

        return X_mixed, y_mixed


# Early Stoppingの実装
class EarlyStopping:
    """Early Stopping with Model Checkpoint"""

    def __init__(self, patience=10, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = np.inf
        self.counter = 0
        self.best_params = None
        self.stopped_epoch = None

    def __call__(self, val_loss, model_params):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                # モデルパラメータのコピーを保存
                self.best_params = {k: v.copy() for k, v in model_params.items()}
            return False  # 学習継続
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 学習停止
            return False

    def get_best_params(self):
        return self.best_params


# 使用例
early_stop = EarlyStopping(patience=15, min_delta=1e-4)

for epoch in range(1000):
    # ... 学習処理 ...
    train_loss = 0.5  # 仮の値
    val_loss = 0.6    # 仮の値

    if early_stop(val_loss, {}):  # model.params を渡す
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 7. 重み初期化の理論

### 7.1 初期化手法の比較

```
■ ゼロ初期化
  W = 0
  問題: 全ユニットが同一の出力 → 対称性が壊れない → 学習不能

■ 小さいランダム値
  W ~ N(0, 0.01²)
  問題: 層が深いと出力が0に収束（勾配消失）

■ 大きいランダム値
  W ~ N(0, 1²)
  問題: 出力が飽和（Sigmoid/Tanh → 勾配消失、ReLU → 勾配爆発）

■ Xavier初期化（Glorot, 2010）
  W ~ N(0, 2/(fan_in + fan_out))  または U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))
  対象: Sigmoid, Tanh
  原理: 各層の分散を保つ

■ He初期化（Kaiming, 2015）
  W ~ N(0, 2/fan_in)
  対象: ReLU, LeakyReLU
  原理: ReLUが半分のユニットを0にするため、分散を2倍に

■ LSUV (Layer-Sequential Unit-Variance)
  手順: (1) 直交行列で初期化 (2) 各層の出力分散が1になるようスケーリング
  利点: 任意の活性化関数に対応

■ fixup初期化
  対象: ResNet（BatchNormなし）
  残差ブロックの重みを0で初期化、スキップ接続をスケーリング
```

### コード例7: 初期化手法の実装と効果比較

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_initialization_effects():
    """異なる初期化手法の出力分布を可視化"""

    np.random.seed(42)
    n_layers = 10
    n_units = 256
    batch_size = 1000

    initializations = {
        "Small Random": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 0.01,
        "Large Random": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * 1.0,
        "Xavier": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out)),
        "He": lambda fan_in, fan_out: np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in),
    }

    fig, axes = plt.subplots(len(initializations), n_layers, figsize=(30, 12))

    for row, (name, init_fn) in enumerate(initializations.items()):
        X = np.random.randn(batch_size, n_units)

        for layer in range(n_layers):
            W = init_fn(n_units, n_units)
            X = np.maximum(0, X @ W)  # ReLU

            ax = axes[row, layer]
            ax.hist(X.flatten(), bins=50, density=True, alpha=0.7)
            ax.set_title(f"Layer {layer+1}" if row == 0 else "")
            if layer == 0:
                ax.set_ylabel(name, fontsize=12)
            ax.set_xlim(-1, 5)

            # 分散と活性化率を表示
            var = np.var(X)
            active_ratio = np.mean(X > 0)
            ax.text(0.5, 0.9, f"var={var:.2e}\nact={active_ratio:.2f}",
                    transform=ax.transAxes, fontsize=7, verticalalignment='top')

    plt.suptitle("初期化手法による各層の出力分布（ReLU活性化）", fontsize=16)
    plt.tight_layout()
    plt.savefig("reports/initialization_comparison.png", dpi=150)
    plt.close()

visualize_initialization_effects()
```

---

## 8. PyTorchによる実装

### コード例8: PyTorchでのMLP実装

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class MLP(nn.Module):
    """PyTorchによるMLP（柔軟な構成）"""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout_rate=0.0, use_batchnorm=True,
                 activation='relu'):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 活性化関数の選択
        act_fn = {
            'relu': nn.ReLU,
            'leaky_relu': lambda: nn.LeakyReLU(0.01),
            'gelu': nn.GELU,
            'silu': nn.SiLU,  # Swish
            'elu': nn.ELU,
            'mish': nn.Mish,
        }

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation in act_fn:
                layers.append(act_fn[activation]())
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, val_loader, epochs=100,
                lr=0.001, weight_decay=1e-4, patience=10):
    """学習ループ（Early Stopping付き）"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # 学習フェーズ
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()

        scheduler.step()

        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break

        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:4d}  "
                  f"Train Loss={train_loss:.4f}  Val Loss={val_loss:.4f}  "
                  f"Train Acc={train_acc:.4f}  Val Acc={val_acc:.4f}  "
                  f"LR={current_lr:.6f}")

    return history


# 使用例
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# TensorDataset作成
train_ds = TensorDataset(
    torch.FloatTensor(X_train), torch.LongTensor(y_train)
)
val_ds = TensorDataset(
    torch.FloatTensor(X_val), torch.LongTensor(y_val)
)
test_ds = TensorDataset(
    torch.FloatTensor(X_test), torch.LongTensor(y_test)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# モデル作成と学習
model = MLP(
    input_dim=64,
    hidden_dims=[256, 128, 64],
    output_dim=10,
    dropout_rate=0.3,
    use_batchnorm=True,
    activation='gelu'
)

print(model)
print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

history = train_model(model, train_loader, val_loader,
                      epochs=200, lr=0.001, patience=20)
```

### コード例8b: 学習曲線の可視化とモデル評価

```python
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_training_history(history):
    """学習曲線の可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('損失の推移')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('精度の推移')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("reports/training_history.png", dpi=150)
    plt.close()


def evaluate_model(model, test_loader, class_names=None):
    """モデルの詳細評価"""
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=== Classification Report ===")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))

    # 混同行列
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('予測')
    plt.ylabel('正解')
    plt.title('混同行列')
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=150)
    plt.close()

    return all_preds, all_labels

# 実行
plot_training_history(history)
class_names = [str(i) for i in range(10)]
preds, labels = evaluate_model(model, test_loader, class_names)
```

### コード例5: scikit-learnでのMLP

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target

pipe = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=500, random_state=42, early_stopping=True,
                  validation_fraction=0.1)
)

param_grid = {
    "mlpclassifier__hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
    "mlpclassifier__activation": ["relu", "tanh"],
    "mlpclassifier__alpha": [0.0001, 0.001, 0.01],  # L2正則化
    "mlpclassifier__learning_rate_init": [0.001, 0.01],
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=1)
grid.fit(X, y)

print(f"最良パラメータ: {grid.best_params_}")
print(f"最良スコア: {grid.best_score_:.4f}")
```

---

## 9. 勾配問題とその対策

### 9.1 勾配消失・勾配爆発

```
■ 勾配消失（Vanishing Gradient）
  原因:
    ・ Sigmoid/Tanh の飽和領域（勾配 → 0）
    ・ 深い層での勾配の乗算（0.25^n → 0）
    ・ 不適切な初期化
  対策:
    ・ ReLU系活性化関数
    ・ He/Xavier初期化
    ・ BatchNormalization
    ・ 残差接続（Skip Connection）
    ・ LSTM/GRU（RNNの場合）

■ 勾配爆発（Exploding Gradient）
  原因:
    ・ 大きな初期重み
    ・ 深い層での勾配の乗算（大きい値^n → ∞）
    ・ RNN（長い系列）
  対策:
    ・ 勾配クリッピング（Gradient Clipping）
    ・ BatchNormalization
    ・ 適切な初期化
    ・ 学習率の調整

■ Dead Neuron（ReLU特有）
  原因:
    ・ 大きな負の入力 → ReLU出力=0 → 勾配=0 → 更新不能
    ・ 学習率が大きすぎる場合に発生しやすい
  対策:
    ・ LeakyReLU / ELU
    ・ 学習率を小さくする
    ・ He初期化で回避確率を下げる
```

### コード例9: 勾配フローの診断

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def diagnose_gradient_flow(model, loss, plot=True):
    """勾配フローの診断ツール"""
    loss.backward()

    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.bar(range(len(ave_grads)), ave_grads, alpha=0.7)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel("平均勾配")
        ax1.set_title("各層の平均勾配")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        ax2.bar(range(len(max_grads)), max_grads, alpha=0.7, color='orange')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel("最大勾配")
        ax2.set_title("各層の最大勾配")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("reports/gradient_flow.png", dpi=150)
        plt.close()

    # 問題の検出
    for name, avg, mx in zip(layers, ave_grads, max_grads):
        if avg < 1e-7:
            print(f"⚠ 勾配消失の可能性: {name} (avg={avg:.2e})")
        if mx > 100:
            print(f"⚠ 勾配爆発の可能性: {name} (max={mx:.2e})")

    return dict(zip(layers, ave_grads))
```

---

## 10. 実務的なデバッグとチューニング

### 10.1 よくある問題と対処法

```
■ 学習が進まない（Loss が減らない）
  チェックリスト:
  1. データ: 前処理が正しいか（正規化、ラベルエンコーディング）
  2. 学習率: 大きすぎ → 発散、小さすぎ → 収束が遅い
  3. 損失関数: タスクに合っているか（分類にMSEを使っていないか）
  4. 出力層: 分類にSoftmax + CE、回帰に線形 + MSE
  5. バグ: .train()/.eval() の切り替え忘れ
  6. データリーク: テストデータの情報が学習に漏れていないか

■ 過学習（Train↑ Val↓）
  対策の優先順位:
  1. データを増やす（最も効果的）
  2. データ拡張（Augmentation）
  3. Dropout を追加（0.2-0.5）
  4. Weight Decay を増やす
  5. モデルを小さくする（層数、ユニット数の削減）
  6. Early Stopping を適用
  7. ラベルスムージング

■ 過少適合（Train↑も Val↑も低い）
  対策:
  1. モデルを大きくする（表現能力不足）
  2. 学習率を調整
  3. エポック数を増やす
  4. 特徴量エンジニアリング
  5. 正則化を弱める（過度な正則化は過少適合を招く）
```

### 10.2 ハイパーパラメータチューニング（Optuna）

```python
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def objective(trial):
    """Optunaの目的関数"""

    # ハイパーパラメータの提案
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(trial.suggest_int(f"n_units_l{i}", 32, 512, log=True))

    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    activation = trial.suggest_categorical("activation",
                                            ["relu", "gelu", "silu", "elu"])

    # データ準備
    digits = load_digits()
    X, y = digits.data, digits.target
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=64, shuffle=False
    )

    # モデル構築
    model = MLP(
        input_dim=64,
        hidden_dims=hidden_dims,
        output_dim=10,
        dropout_rate=dropout_rate,
        use_batchnorm=True,
        activation=activation,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # 学習
    for epoch in range(50):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        # 検証
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                outputs = model(X_b)
                _, predicted = torch.max(outputs, 1)
                val_total += y_b.size(0)
                val_correct += (predicted == y_b).sum().item()

        val_acc = val_correct / val_total

        # Pruning: 途中で性能が悪い試行を打ち切り
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_acc


# 最適化の実行
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=100, timeout=600)

print(f"\n最良の精度: {study.best_value:.4f}")
print(f"最良のパラメータ: {study.best_params}")

# 結果の可視化
fig = optuna.visualization.plot_optimization_history(study)
fig = optuna.visualization.plot_param_importances(study)
```

---

## 比較表

### 活性化関数の特性比較

| 活性化関数 | 出力範囲 | 勾配消失 | 計算コスト | 主な用途 | 備考 |
|---|---|---|---|---|---|
| ReLU | [0, +inf) | 低い | 極低 | 隠れ層（標準） | Dead Neuron問題あり |
| LeakyReLU | (-inf, +inf) | 低い | 極低 | 隠れ層 | Dead Neuron対策 |
| ELU | (-alpha, +inf) | 低い | 低い | 隠れ層 | 負の出力で平均0に近づく |
| GELU | (-0.17, +inf) | 低い | 中程度 | Transformer | BERT, GPT標準 |
| Swish | (-0.28, +inf) | 低い | 中程度 | 深いネットワーク | 自己ゲート機構 |
| Mish | (-0.31, +inf) | 低い | 中程度 | 物体検出 | YOLOv4で採用 |
| Sigmoid | (0, 1) | 高い | 低い | 出力層（二値分類） | 隠れ層には非推奨 |
| Tanh | (-1, 1) | 中程度 | 低い | RNN（ゲート機構） | ゼロ中心出力 |
| Softmax | (0, 1) | - | 中程度 | 出力層（多クラス） | 確率分布を出力 |

### オプティマイザの比較

| オプティマイザ | 学習率調整 | モーメンタム | 適応的学習率 | メモリ | 推奨場面 |
|---|---|---|---|---|---|
| SGD | 手動 | 不可（オプション） | 不可 | 低い | 凸最適化、最終調整 |
| SGD+Momentum | 手動 | あり | 不可 | 低い | CNN学習 |
| Nesterov SGD | 手動 | あり（先読み） | 不可 | 低い | 収束の加速 |
| AdaGrad | 自動 | 不可 | あり | 中程度 | スパースデータ |
| RMSProp | 自動 | 不可 | あり | 中程度 | RNN学習 |
| Adam | 自動 | あり | あり | 高い | 汎用（最も使われる） |
| AdamW | 自動 | あり | あり | 高い | Transformer、大規模モデル |
| LAMB | 自動 | あり | あり（層別） | 高い | 超大バッチ学習 |

### 学習率スケジューラの比較

| スケジューラ | 特徴 | 推奨場面 |
|---|---|---|
| Step Decay | N epoch毎に定率減少 | シンプルなCNN学習 |
| Exponential Decay | 指数的に減少 | 安定した収束が必要 |
| Cosine Annealing | コサインカーブで減少 | 汎用（最も一般的） |
| Warmup + Cosine | 最初に学習率を増加→コサイン減少 | Transformer事前学習 |
| Warm Restarts (SGDR) | コサイン + 周期的リスタート | 複数局所解の探索 |
| 1Cycle Policy | ウォームアップ→クールダウン | Super-Convergence |
| ReduceLROnPlateau | 検証損失停滞時に減少 | 柔軟な適応 |

---

## アンチパターン

### アンチパターン1: 隠れ層にSigmoidを使う

```python
# BAD: 深いネットワークの隠れ層でSigmoid → 勾配消失
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="logistic",  # Sigmoid → 深い層で勾配がほぼ0に
)

# GOOD: 隠れ層にはReLUを使用
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation="relu",  # 勾配消失しにくい
)
```

### アンチパターン2: 重みの初期化を無視する

```python
# BAD: 全て0で初期化 → 対称性が壊れず学習が進まない
W = np.zeros((input_dim, output_dim))

# BAD: 大きすぎるランダム値 → 活性化が飽和
W = np.random.randn(input_dim, output_dim) * 10

# GOOD: He初期化（ReLU用）
W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)

# GOOD: Xavier初期化（Sigmoid/Tanh用）
W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
```

### アンチパターン3: BatchNormとDropoutの順序を間違える

```python
# BAD: Dropout → BatchNorm（分散推定が不安定に）
layer = nn.Sequential(
    nn.Linear(256, 128),
    nn.Dropout(0.3),
    nn.BatchNorm1d(128),  # Dropoutで変わった分散を正規化 → 不安定
    nn.ReLU(),
)

# GOOD: BatchNorm → Activation → Dropout
layer = nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
)

# BETTER: BatchNormを使うならDropoutは不要なことが多い
layer = nn.Sequential(
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
)
```

### アンチパターン4: train/evalモードの切り替え忘れ

```python
# BAD: 推論時にmodel.eval()を忘れる → Dropout/BatchNormが学習時の挙動
model.train()
# ... 学習 ...
# 推論（model.eval()忘れ）
with torch.no_grad():
    output = model(X_test)  # DropoutがONのまま → 性能低下

# GOOD: 推論時は必ずeval()に切り替え
model.eval()
with torch.no_grad():
    output = model(X_test)
# 学習再開時はtrain()に戻す
model.train()
```

### アンチパターン5: 検証データでハイパーパラメータを最適化しすぎる

```python
# BAD: テストデータで繰り返しチューニング → テストデータへの過学習
for lr in [0.001, 0.01, 0.1]:
    model = train(X_train, y_train)
    score = evaluate(X_test, y_test)  # テストで選択 → リーク
    if score > best_score:
        best_lr = lr

# GOOD: Train / Validation / Test の3分割
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Validationでチューニング、Testは最終評価のみ1回
for lr in [0.001, 0.01, 0.1]:
    model = train(X_train, y_train)
    score = evaluate(X_val, y_val)  # Validationで選択
    if score > best_score:
        best_lr = lr

# 最終評価（1回だけ）
final_model = train(X_train, y_train)  # best_lr使用
final_score = evaluate(X_test, y_test)
```

---

## FAQ

### Q1: 隠れ層のユニット数やレイヤー数はどう決める？

**A:** 理論的な最適解はない。経験則: (1) 入力次元と出力次元の間で段階的に減少させる逆ピラミッド型が安定する（例: 256→128→64）、(2) まず小さいモデルで学習曲線を確認し、過少適合なら拡大、過学習ならDropout/正則化を追加、(3) AutoML（Optuna等）でハイパーパラメータ探索。過学習はDropoutやEarly Stoppingで制御する。(4) タスクの複雑さに応じて: 簡単な分類なら1-2層、複雑なパターンなら3-5層が目安。

### Q2: バッチサイズはいくつが良い？

**A:** 一般的に32〜256。小さいバッチは正則化効果があるがノイズが大きい。大きいバッチは安定するがメモリを消費し、汎化性能が低下する傾向がある。GPUのメモリに収まる最大のバッチサイズから始め、学習率をバッチサイズに比例させて調整するのが実践的。最近の研究では、Linear Scaling Rule（バッチサイズを2倍にしたら学習率も2倍）とWarmupの組み合わせが大バッチ学習の標準手法。

### Q3: Early Stoppingはどう設定する？

**A:** 検証損失がN エポック連続で改善しない場合に学習を停止する。Nは5〜20が一般的（patience）。最良の検証スコアを記録したモデルを保存する（Model Checkpoint）。scikit-learnでは `early_stopping=True, n_iter_no_change=10` で設定可能。PyTorchでは自前実装が必要だが、PyTorch Lightningなら `EarlyStopping` コールバックが使える。

### Q4: BatchNormalizationはどこに入れる？

**A:** 一般的には `Linear → BatchNorm → Activation → (Dropout)` の順序が推奨される。BatchNormは各ミニバッチの統計量で正規化するため、バッチサイズが極端に小さい場合（<8）はLayer Normalizationを使うべき。推論時はバッチ統計量ではなく、学習中に計算した移動平均を使う。NLPやTransformerではLayer Normが標準。

### Q5: 勾配消失/爆発をどう検出する？

**A:** (1) 各層の勾配のノルムを監視する（TensorBoardのヒストグラム機能が便利）、(2) 学習初期にLossが減少しない場合は勾配消失を疑う、(3) Lossが急にNaNになった場合は勾配爆発を疑う、(4) `torch.autograd.set_detect_anomaly(True)` でNaN発生箇所を特定。

### Q6: CPUとGPUどちらで学習すべき？

**A:** データサイズとモデルサイズによる。小さいデータセット（〜10万サンプル）で小さいMLP（数千パラメータ）ならCPUで十分。大きなデータセットや深いネットワーク、特にCNN/Transformer系はGPU必須。混合精度学習（FP16）を使えばGPUメモリ使用量を半減でき、学習も高速化される。`torch.cuda.amp` を活用すること。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 構造 | 入力層→隠れ層（ReLU）→出力層（Softmax/Sigmoid） |
| 活性化関数 | 隠れ層: ReLU/GELU、出力層: Softmax（分類）/ 線形（回帰） |
| 損失関数 | 分類: Cross-Entropy、回帰: MSE/Huber、不均衡: Focal Loss |
| 逆伝播 | 連鎖律で勾配を効率的に計算。自動微分が主流 |
| 最適化 | Adam（汎用）、AdamW（大規模）、SGD+Momentum（微調整） |
| 初期化 | He（ReLU用）、Xavier（Sigmoid/Tanh用） |
| 正則化 | Dropout、BatchNorm、Weight Decay、Early Stopping |
| チューニング | Optuna等で自動探索。Train/Val/Testの3分割が基本 |

---

## 次に読むべきガイド

- [01-cnn.md](./01-cnn.md) — 畳み込みニューラルネットワーク（CNN）
- [02-rnn-transformer.md](./02-rnn-transformer.md) — 系列データ向けのRNN/Transformer

---

## 参考文献

1. **Ian Goodfellow, Yoshua Bengio, Aaron Courville** "Deep Learning" MIT Press, 2016 — https://www.deeplearningbook.org/
2. **Diederik P. Kingma, Jimmy Ba** "Adam: A Method for Stochastic Optimization" ICLR 2015
3. **Kaiming He et al.** "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification" ICCV 2015
4. **Sergey Ioffe, Christian Szegedy** "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" ICML 2015
5. **Nitish Srivastava et al.** "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" JMLR 2014
6. **Ilya Loshchilov, Frank Hutter** "Decoupled Weight Decay Regularization" ICLR 2019
7. **Ilya Loshchilov, Frank Hutter** "SGDR: Stochastic Gradient Descent with Warm Restarts" ICLR 2017
8. **Leslie N. Smith** "A disciplined approach to neural network hyper-parameters" arXiv 2018
9. **Xavier Glorot, Yoshua Bengio** "Understanding the difficulty of training deep feedforward neural networks" AISTATS 2010
10. **Dan Hendrycks, Kevin Gimpel** "Gaussian Error Linear Units (GELUs)" arXiv 2016
