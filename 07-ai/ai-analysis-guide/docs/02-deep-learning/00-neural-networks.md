# ニューラルネットワーク — パーセプトロン、活性化関数、誤差逆伝播

> ニューラルネットワークの基礎理論をゼロから構築し、深層学習への橋渡しを行う

## この章で学ぶこと

1. **パーセプトロンとMLP** — 単層から多層への発展と表現能力の向上
2. **活性化関数** — ReLU、Sigmoid、Softmax等の特性と選択基準
3. **誤差逆伝播法** — 勾配の自動計算メカニズムとパラメータ最適化

---

## 1. ニューラルネットワークの構造

### MLP（多層パーセプトロン）のアーキテクチャ

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

---

## 2. 活性化関数

### 活性化関数の比較

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
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def swish(x, beta=1.0):
        return x * Activations.sigmoid(beta * x)

# 可視化
x = np.linspace(-4, 4, 1000)
functions = {
    "ReLU": Activations.relu,
    "LeakyReLU": Activations.leaky_relu,
    "Sigmoid": Activations.sigmoid,
    "Tanh": Activations.tanh,
    "GELU": Activations.gelu,
    "Swish": Activations.swish,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, func) in zip(axes.flatten(), functions.items()):
    ax.plot(x, func(x), linewidth=2)
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_title(name, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)

plt.tight_layout()
plt.savefig("reports/activation_functions.png", dpi=150)
plt.close()
```

---

## 3. 誤差逆伝播法と最適化

### 逆伝播の計算グラフ

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

# 可視化
epochs = range(100)
initial_lr = 0.01

fig, ax = plt.subplots(figsize=(12, 6))
for name, func in [
    ("Step Decay", lambda e: LRSchedulers.step_decay(initial_lr, e)),
    ("Exp Decay", lambda e: LRSchedulers.exponential_decay(initial_lr, e)),
    ("Cosine", lambda e: LRSchedulers.cosine_annealing(initial_lr, e, 100)),
    ("Warmup+Cosine", lambda e: LRSchedulers.warmup_cosine(initial_lr, e, 100)),
]:
    lrs = [func(e) for e in epochs]
    ax.plot(epochs, lrs, label=name, linewidth=2)

ax.set_xlabel("Epoch")
ax.set_ylabel("学習率")
ax.set_title("学習率スケジューリング戦略の比較")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/lr_schedules.png", dpi=150)
plt.close()
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

## 比較表

### 活性化関数の特性比較

| 活性化関数 | 出力範囲 | 勾配消失 | 計算コスト | 主な用途 | 備考 |
|---|---|---|---|---|---|
| ReLU | [0, ∞) | 低い | 極低 | 隠れ層（標準） | Dead Neuron問題あり |
| LeakyReLU | (-∞, ∞) | 低い | 極低 | 隠れ層 | Dead Neuron対策 |
| GELU | (-0.17, ∞) | 低い | 中程度 | Transformer | BERT, GPT標準 |
| Swish | (-0.28, ∞) | 低い | 中程度 | 深いネットワーク | 自己ゲート機構 |
| Sigmoid | (0, 1) | 高い | 低い | 出力層（二値分類） | 隠れ層には非推奨 |
| Tanh | (-1, 1) | 中程度 | 低い | RNN（ゲート機構） | ゼロ中心出力 |
| Softmax | (0, 1) | - | 中程度 | 出力層（多クラス） | 確率分布を出力 |

### オプティマイザの比較

| オプティマイザ | 学習率調整 | モーメンタム | 適応的学習率 | メモリ | 推奨場面 |
|---|---|---|---|---|---|
| SGD | 手動 | △（オプション） | × | 低い | 凸最適化、最終調整 |
| SGD+Momentum | 手動 | ○ | × | 低い | CNN学習 |
| AdaGrad | 自動 | × | ○ | 中程度 | スパースデータ |
| RMSProp | 自動 | × | ○ | 中程度 | RNN学習 |
| Adam | 自動 | ○ | ○ | 高い | 汎用（最も使われる） |
| AdamW | 自動 | ○ | ○ | 高い | Transformer、大規模モデル |

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

---

## FAQ

### Q1: 隠れ層のユニット数やレイヤー数はどう決める？

**A:** 理論的な最適解はない。経験則: (1) 入力次元と出力次元の間で段階的に減少、(2) まず小さいモデルで学習曲線を確認し、過少適合なら拡大、(3) AutoML（Optuna等）でハイパーパラメータ探索。過学習はDropoutやEarly Stoppingで制御する。

### Q2: バッチサイズはいくつが良い？

**A:** 一般的に32〜256。小さいバッチは正則化効果があるがノイズが大きい。大きいバッチは安定するがメモリを消費し、汎化性能が低下する傾向がある。GPUのメモリに収まる最大のバッチサイズから始め、学習率をバッチサイズに比例させて調整するのが実践的。

### Q3: Early Stoppingはどう設定する？

**A:** 検証損失がN エポック連続で改善しない場合に学習を停止する。Nは5〜20が一般的（patience）。最良の検証スコアを記録したモデルを保存する（Model Checkpoint）。scikit-learnでは `early_stopping=True, n_iter_no_change=10` で設定可能。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 構造 | 入力層→隠れ層（ReLU）→出力層（Softmax/Sigmoid） |
| 活性化関数 | 隠れ層: ReLU/GELU、出力層: Softmax（分類）/ 線形（回帰） |
| 逆伝播 | 連鎖律で勾配を効率的に計算。自動微分が主流 |
| 最適化 | Adam（汎用）、AdamW（大規模）、SGD+Momentum（微調整） |
| 初期化 | He（ReLU用）、Xavier（Sigmoid/Tanh用） |

---

## 次に読むべきガイド

- [01-cnn.md](./01-cnn.md) — 畳み込みニューラルネットワーク（CNN）
- [02-rnn-transformer.md](./02-rnn-transformer.md) — 系列データ向けのRNN/Transformer

---

## 参考文献

1. **Ian Goodfellow, Yoshua Bengio, Aaron Courville** "Deep Learning" MIT Press, 2016 — https://www.deeplearningbook.org/
2. **Diederik P. Kingma, Jimmy Ba** "Adam: A Method for Stochastic Optimization" ICLR 2015
3. **Kaiming He et al.** "Delving Deep into Rectifiers" ICCV 2015
