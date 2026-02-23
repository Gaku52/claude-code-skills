# スクリプト言語比較（Python, Ruby, JavaScript, PHP, Perl）

> スクリプト言語は「素早く書いて、素早く動かす」ことに最適化された言語群。それぞれの哲学・強み・エコシステムを比較する。

## この章で学ぶこと

- [ ] 主要スクリプト言語の特徴と適用領域を把握する
- [ ] 各言語の設計哲学の違いを理解する
- [ ] プロジェクトに応じた言語選択ができる
- [ ] 各言語のエコシステム・ツールチェーンを理解する
- [ ] パフォーマンス特性と最適化手法を把握する
- [ ] 各言語のモダンな開発プラクティスを身につける

---

## 1. 比較表

```
┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│              │ Python   │ Ruby     │ JS/TS    │ PHP      │ Perl     │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 設計哲学      │ 明示的   │ 開発者の │ Webの    │ Web特化  │ TMTOWTDI │
│              │ 読みやすさ│ 幸福     │ 言語     │ 実用性   │ 多様な道 │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 型付け        │ 動的+強い│ 動的+強い│ 動的+弱い│ 動的+弱い│ 動的+弱い│
│              │ (型ヒント)│          │ (TS:静的)│ (型宣言) │          │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 主な用途      │ AI/ML    │ Web      │ フルスタック│ Web     │ テキスト │
│              │ データ   │ スクリプト│ ブラウザ  │ CMS     │ 処理     │
│              │ 自動化   │ DevOps   │ サーバー  │ EC      │ 自動化   │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ パッケージ    │ PyPI     │ RubyGems │ npm      │ Packagist│ CPAN     │
│ マネージャ    │ pip/uv   │ bundler  │ npm/pnpm │ composer │ cpanm    │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 主要FW       │ Django   │ Rails    │ Express  │ Laravel  │ Catalyst │
│              │ FastAPI  │ Sinatra  │ Next.js  │ Symfony  │ Mojolicious│
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 実行速度      │ 遅い     │ 遅い     │ 速い(V8) │ 中程度   │ 中程度   │
│ (相対)        │          │          │          │ (OPcache)│          │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 学習コスト    │ 低い     │ 低い     │ 低い     │ 低い     │ 中程度   │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 求人数        │ 非常に多い│ 中程度   │ 最も多い │ 多い     │ 少ない   │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 登場年        │ 1991     │ 1993     │ 1995     │ 1995     │ 1987     │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 作者          │ G.v.Rossum│ Matz    │ B.Eich   │ R.Lerdorf│ L.Wall   │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ 最新安定版    │ 3.12+    │ 3.3+     │ ES2024   │ 8.3+     │ 5.38+    │
│ (2025時点)    │          │          │ Node 22  │          │          │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 2. 設計哲学の違い

### 2.1 Python — 「明示的は暗黙的に勝る」

```python
# "There should be one-- and preferably only one --obvious way to do it."
# 1つの正しいやり方があるべき

# Zen of Python（import this）
# Beautiful is better than ugly.
# Explicit is better than implicit.
# Simple is better than complex.
# Readability counts.
# Errors should never pass silently.

# 特徴: インデントで構造化、読みやすさ最優先
def process_data(data: list[dict]) -> list[str]:
    """アクティブなユーザー名を大文字で返す"""
    return [
        item["name"].upper()
        for item in data
        if item.get("active", False)
    ]

# コンテキストマネージャによるリソース管理
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    records = [row for row in reader if row["status"] == "active"]

# デコレータによる横断的関心事の分離
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# ジェネレータで大規模データの遅延評価
def read_large_file(path: str):
    """メモリ効率の良い大規模ファイル読み込み"""
    with open(path, "r") as f:
        for line in f:
            yield line.strip()

# 型ヒントによるドキュメントとしての型情報（Python 3.12+）
type Point = tuple[float, float]
type Matrix[T] = list[list[T]]

def distance(p1: Point, p2: Point) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
```

**Pythonの設計原則の深掘り**

```python
# 1. EAFP (Easier to Ask Forgiveness than Permission)
#    許可を求めるより許しを乞う方が容易

# LBYL（他言語的アプローチ）- 推奨しない
if hasattr(obj, "name"):
    print(obj.name)

# EAFP（Python的アプローチ）- 推奨
try:
    print(obj.name)
except AttributeError:
    print("name属性がありません")

# 2. ダックタイピング
# 「アヒルのように歩き、アヒルのように鳴くなら、それはアヒルだ」
class Duck:
    def quack(self):
        return "ガーガー"

class Person:
    def quack(self):
        return "私はアヒルのように鳴けます"

def make_quack(thing):
    # Duck型かPerson型かは問わない。quack()メソッドがあればOK
    return thing.quack()

# 3. プロトコルによる構造的部分型（Python 3.8+）
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

class Circle:
    def draw(self) -> None:
        print("○を描画")

def render(shape: Drawable) -> None:
    shape.draw()  # CircleはDrawableを明示的に継承していないがOK

# 4. データクラスによる簡潔なデータモデリング
from dataclasses import dataclass, field
from datetime import datetime

@dataclass(frozen=True)  # frozen=True でイミュータブル
class User:
    name: str
    email: str
    age: int
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)

    def is_adult(self) -> bool:
        return self.age >= 18

# 5. 構造的パターンマッチ（Python 3.10+）
def handle_command(command: dict) -> str:
    match command:
        case {"action": "create", "name": str(name)}:
            return f"作成: {name}"
        case {"action": "delete", "id": int(id_)}:
            return f"削除: ID={id_}"
        case {"action": "list", "filter": {"status": status}}:
            return f"一覧: status={status}"
        case _:
            return "不明なコマンド"
```

### 2.2 Ruby — 「プログラマの幸福」

```ruby
# "Ruby is designed to make programmers happy."
# プログラマの幸福のために設計

# 特徴: 表現力豊か、すべてがオブジェクト
5.times { |i| puts i }
[1, 2, 3].select(&:odd?)  # → [1, 3]
"hello".reverse.upcase     # → "OLLEH"

# Convention over Configuration（Rails哲学）

# 数値もオブジェクト
3.14.round       # → 3
-42.abs          # → 42
1.upto(5) { |n| print n }  # → 12345

# シンボルとハッシュの簡潔さ
user = { name: "Alice", age: 30, active: true }
user[:name]  # → "Alice"

# ブロック・Proc・Lambda の豊かな表現力
# ブロック（最も一般的）
[1, 2, 3].map { |n| n * 2 }  # → [2, 4, 6]

# Proc（手続きオブジェクト）
doubler = Proc.new { |n| n * 2 }
[1, 2, 3].map(&doubler)  # → [2, 4, 6]

# Lambda（引数チェック付き手続き）
validator = ->(n) { n.positive? }
[1, -2, 3, -4].select(&validator)  # → [1, 3]

# メソッドミッシング（メタプログラミング）
class DynamicProxy
  def method_missing(name, *args)
    if name.to_s.start_with?("find_by_")
      attribute = name.to_s.sub("find_by_", "")
      puts "#{attribute}で検索: #{args.first}"
    else
      super
    end
  end

  def respond_to_missing?(name, include_private = false)
    name.to_s.start_with?("find_by_") || super
  end
end

proxy = DynamicProxy.new
proxy.find_by_name("Alice")   # → "nameで検索: Alice"
proxy.find_by_email("a@b.c")  # → "emailで検索: a@b.c"

# オープンクラス（既存クラスの拡張）
class Integer
  def factorial
    return 1 if self <= 1
    self * (self - 1).factorial
  end
end

5.factorial  # → 120

# Enumerableの豊かなメソッド群
users = [
  { name: "Alice", age: 30, dept: "Engineering" },
  { name: "Bob",   age: 25, dept: "Marketing" },
  { name: "Carol", age: 35, dept: "Engineering" },
  { name: "Dave",  age: 28, dept: "Marketing" }
]

# グループ化 → 各グループの平均年齢
users
  .group_by { |u| u[:dept] }
  .transform_values { |members| members.sum { |m| m[:age] }.to_f / members.size }
# → {"Engineering"=>32.5, "Marketing"=>26.5}

# パターンマッチ（Ruby 3.0+）
case { name: "Alice", age: 30, roles: [:admin, :user] }
in { name: String => name, roles: [*, :admin, *] }
  puts "管理者: #{name}"
in { name: String => name, age: (..17) }
  puts "未成年: #{name}"
else
  puts "一般ユーザー"
end
```

### 2.3 JavaScript / TypeScript — 「Webのユビキタス言語」

```typescript
// Web のユビキタス言語
// ブラウザからサーバーまで1言語で

// TypeScript: JavaScript + 型安全性
interface User {
    name: string;
    age: number;
    email?: string;  // オプショナルプロパティ
}

const greet = (user: User): string => `Hello, ${user.name}!`;

// 特徴: イベント駆動、非同期が自然
const data = await fetch("/api").then(r => r.json());

// ユニオン型とリテラル型
type Status = "active" | "inactive" | "pending";
type Result<T> = { ok: true; value: T } | { ok: false; error: string };

function processResult<T>(result: Result<T>): T | never {
    if (result.ok) {
        return result.value;  // T型に絞り込み
    }
    throw new Error(result.error);
}

// ジェネリクスと条件型
type NonNullable<T> = T extends null | undefined ? never : T;
type Flatten<T> = T extends Array<infer U> ? U : T;

// テンプレートリテラル型
type EventName = `on${Capitalize<"click" | "focus" | "blur">}`;
// → "onClick" | "onFocus" | "onBlur"

// Mapped Types で型を動的に構築
type Readonly<T> = { readonly [K in keyof T]: T[K] };
type Optional<T> = { [K in keyof T]?: T[K] };

// 非同期イテレータ
async function* fetchPages(baseUrl: string): AsyncGenerator<Page[]> {
    let page = 1;
    while (true) {
        const response = await fetch(`${baseUrl}?page=${page}`);
        const data = await response.json();
        if (data.items.length === 0) break;
        yield data.items;
        page++;
    }
}

// for await...of で消費
for await (const pages of fetchPages("/api/users")) {
    for (const user of pages) {
        console.log(user.name);
    }
}

// Proxy によるメタプログラミング
const validator = new Proxy<Record<string, unknown>>({}, {
    set(target, prop, value) {
        if (typeof prop === "string" && prop.startsWith("age")) {
            if (typeof value !== "number" || value < 0) {
                throw new TypeError(`${String(prop)} must be a positive number`);
            }
        }
        target[prop as string] = value;
        return true;
    }
});

// Decorator（TypeScript 5.0+, Stage 3）
function log(target: any, context: ClassMethodDecoratorContext) {
    return function(this: any, ...args: any[]) {
        console.log(`Calling ${String(context.name)} with`, args);
        const result = target.call(this, ...args);
        console.log(`Result:`, result);
        return result;
    };
}

class Calculator {
    @log
    add(a: number, b: number): number {
        return a + b;
    }
}
```

### 2.4 PHP — 「Webに特化した実用家」

```php
<?php
// PHP 8.3+: モダンPHP
// Webサーバーサイドの実用的な言語

// 列挙型（PHP 8.1+）
enum Status: string {
    case Active = 'active';
    case Inactive = 'inactive';
    case Pending = 'pending';

    public function label(): string {
        return match($this) {
            Status::Active => '有効',
            Status::Inactive => '無効',
            Status::Pending => '保留中',
        };
    }
}

// Readonly プロパティとコンストラクタプロモーション
class User {
    public function __construct(
        public readonly string $name,
        public readonly int $age,
        public readonly string $email,
        public readonly Status $status = Status::Active,
    ) {}

    public function isAdult(): bool {
        return $this->age >= 18;
    }
}

// 名前付き引数
$user = new User(
    name: 'Alice',
    age: 30,
    email: 'alice@example.com',
);

// ファーストクラスCallable構文（PHP 8.1+）
$users = [
    new User('Alice', 30, 'a@example.com'),
    new User('Bob', 17, 'b@example.com'),
    new User('Carol', 25, 'c@example.com'),
];

$adults = array_filter($users, fn(User $u) => $u->isAdult());
$names = array_map(fn(User $u) => $u->name, $adults);

// Fiber（PHP 8.1+）— 軽量な並行処理
$fiber = new Fiber(function (): void {
    $value = Fiber::suspend('fiber started');
    echo "受け取った値: $value\n";
});

$result = $fiber->start();  // → 'fiber started'
$fiber->resume('hello');     // → "受け取った値: hello"

// match式（PHP 8.0+）— switchの型安全版
function getStatusCode(string $method): int {
    return match($method) {
        'GET'    => 200,
        'POST'   => 201,
        'DELETE'  => 204,
        default  => throw new InvalidArgumentException("不明なメソッド: $method"),
    };
}

// Union型・Intersection型（PHP 8.0+/8.1+）
function processInput(string|int $input): string {
    return match(true) {
        is_string($input) => strtoupper($input),
        is_int($input)    => str_pad((string)$input, 5, '0', STR_PAD_LEFT),
    };
}

// Attribute（PHP 8.0+）— アノテーション
#[Route('/api/users', methods: ['GET'])]
#[Middleware('auth')]
function listUsers(): JsonResponse {
    // ...
}

// ジェネレータでメモリ効率の良い処理
function readCsv(string $path): Generator {
    $handle = fopen($path, 'r');
    $headers = fgetcsv($handle);
    while (($row = fgetcsv($handle)) !== false) {
        yield array_combine($headers, $row);
    }
    fclose($handle);
}

foreach (readCsv('large_file.csv') as $row) {
    // 1行ずつ処理（メモリに全件載せない）
    processRow($row);
}
```

### 2.5 Perl — 「実用性と柔軟性の極致」

```perl
#!/usr/bin/perl
use strict;
use warnings;
use v5.38;

# "There's More Than One Way To Do It" (TMTOWTDI)
# やり方は1つではない

# 正規表現のネイティブサポート（Perlの最大の強み）
my $email_pattern = qr/
    ^
    [a-zA-Z0-9._%+-]+    # ローカル部
    @                      # @記号
    [a-zA-Z0-9.-]+        # ドメイン
    \.                    # ドット
    [a-zA-Z]{2,}          # TLD
    $
/x;  # /x フラグで空白・コメントを許可

my $text = "Contact us at info\@example.com or support\@test.org";
my @emails = ($text =~ /[\w.+-]+@[\w.-]+\.\w{2,}/g);
say "Found: @emails";

# ワンライナーの力（コマンドラインでの文字列処理）
# perl -ne 'print if /ERROR/' access.log
# perl -pe 's/foo/bar/g' input.txt > output.txt
# perl -ane 'print "$F[0]\n"' data.tsv

# 正規表現によるテキスト変換
sub parse_log_line {
    my ($line) = @_;
    if ($line =~ /^(\d{4}-\d{2}-\d{2})\s+(\w+)\s+\[(\w+)\]\s+(.*)$/) {
        return {
            date    => $1,
            level   => $2,
            module  => $3,
            message => $4,
        };
    }
    return undef;
}

# ハッシュ（連想配列）の操作
my %config = (
    host    => 'localhost',
    port    => 8080,
    debug   => 1,
    workers => 4,
);

# ハッシュスライス
my @connection_info = @config{qw(host port)};

# リファレンスとデリファレンス
my $nested = {
    users => [
        { name => 'Alice', age => 30 },
        { name => 'Bob',   age => 25 },
    ],
    metadata => { total => 2, page => 1 },
};

say $nested->{users}[0]{name};  # → "Alice"

# Moose/Moo によるモダンなOOP
package User {
    use Moo;
    use Types::Standard qw(Str Int);

    has name  => (is => 'ro', isa => Str, required => 1);
    has age   => (is => 'ro', isa => Int, required => 1);
    has email => (is => 'rw', isa => Str);

    sub greet {
        my ($self) = @_;
        return "Hello, " . $self->name . "!";
    }
}
```

---

## 3. エコシステム詳細比較

### 3.1 Webフレームワーク

```
Python:
  フルスタック: Django（バッテリー同梱、管理画面自動生成）
  API特化:     FastAPI（非同期、OpenAPI自動生成）
  軽量:        Flask, Starlette
  非同期:      Sanic, aiohttp

Ruby:
  フルスタック: Ruby on Rails（Convention over Configuration）
  軽量:        Sinatra, Hanami
  API:         Grape, Roda
  リアクティブ: Falcon

JavaScript/TypeScript:
  フルスタック: Next.js（SSR/SSG/ISR）, Nuxt.js（Vue版）
  API:         Express, Fastify, Hono
  型安全API:   tRPC, NestJS
  フルスタックTS: Remix, SvelteKit, Astro

PHP:
  フルスタック: Laravel（Eloquent ORM, Blade, Artisan CLI）
  エンタープライズ: Symfony（Bundle系、柔軟性高い）
  API:         Slim, Lumen
  リアクティブ: Swoole + Hyperf

Perl:
  フルスタック: Catalyst, Dancer2
  非同期:      Mojolicious
  CGI:        CGI.pm（レガシーだが歴史的に重要）
```

### 3.2 パッケージ管理の比較

```python
# Python: pip / uv / poetry
# --- requirements.txt ---
fastapi>=0.100.0
pydantic>=2.0.0
sqlalchemy>=2.0.0

# --- pyproject.toml (モダンなアプローチ) ---
# [project]
# name = "myapp"
# version = "0.1.0"
# dependencies = [
#     "fastapi>=0.100.0",
#     "pydantic>=2.0.0",
# ]

# uv（超高速パッケージマネージャ、Rust製）
# uv pip install fastapi
# uv venv
# uv sync
```

```ruby
# Ruby: Bundler + Gemfile
# --- Gemfile ---
source 'https://rubygems.org'

gem 'rails', '~> 7.1'
gem 'pg', '~> 1.5'
gem 'puma', '~> 6.0'

group :development, :test do
  gem 'rspec-rails', '~> 6.0'
  gem 'rubocop', '~> 1.60'
end

# bundle install
# bundle exec rails server
```

```json
// JavaScript: package.json (npm / pnpm / yarn)
{
  "name": "myapp",
  "version": "1.0.0",
  "type": "module",
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.2.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/react": "^18.2.0"
  },
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "lint": "eslint . --ext .ts,.tsx"
  }
}
```

```json
// PHP: composer.json
{
    "require": {
        "php": "^8.2",
        "laravel/framework": "^11.0",
        "laravel/sanctum": "^4.0"
    },
    "require-dev": {
        "phpunit/phpunit": "^11.0",
        "phpstan/phpstan": "^1.10"
    },
    "autoload": {
        "psr-4": {
            "App\\": "app/"
        }
    }
}
```

### 3.3 テストフレームワーク

```python
# Python: pytest（最もポピュラー）
import pytest
from myapp.calculator import Calculator

class TestCalculator:
    @pytest.fixture
    def calc(self):
        return Calculator()

    def test_add(self, calc):
        assert calc.add(2, 3) == 5

    def test_divide_by_zero(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.divide(1, 0)

    @pytest.mark.parametrize("a, b, expected", [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
    ])
    def test_add_parametrized(self, calc, a, b, expected):
        assert calc.add(a, b) == expected
```

```ruby
# Ruby: RSpec（BDD スタイル）
RSpec.describe Calculator do
  subject(:calc) { described_class.new }

  describe '#add' do
    it '2つの数を足す' do
      expect(calc.add(2, 3)).to eq(5)
    end

    context '負の数の場合' do
      it '正しく計算する' do
        expect(calc.add(-1, 1)).to eq(0)
      end
    end
  end

  describe '#divide' do
    it 'ゼロ除算でエラー' do
      expect { calc.divide(1, 0) }.to raise_error(ZeroDivisionError)
    end
  end
end

# Minitest（標準ライブラリ、軽量）
class TestCalculator < Minitest::Test
  def setup
    @calc = Calculator.new
  end

  def test_add
    assert_equal 5, @calc.add(2, 3)
  end
end
```

```typescript
// TypeScript: Vitest / Jest
import { describe, it, expect } from 'vitest';
import { Calculator } from './calculator';

describe('Calculator', () => {
    const calc = new Calculator();

    it('should add two numbers', () => {
        expect(calc.add(2, 3)).toBe(5);
    });

    it('should throw on division by zero', () => {
        expect(() => calc.divide(1, 0)).toThrow('Division by zero');
    });

    it.each([
        [1, 1, 2],
        [0, 0, 0],
        [-1, 1, 0],
    ])('add(%i, %i) = %i', (a, b, expected) => {
        expect(calc.add(a, b)).toBe(expected);
    });
});
```

```php
<?php
// PHP: PHPUnit
use PHPUnit\Framework\TestCase;

class CalculatorTest extends TestCase
{
    private Calculator $calc;

    protected function setUp(): void
    {
        $this->calc = new Calculator();
    }

    public function testAdd(): void
    {
        $this->assertSame(5, $this->calc->add(2, 3));
    }

    public function testDivideByZero(): void
    {
        $this->expectException(\DivisionByZeroError::class);
        $this->calc->divide(1, 0);
    }

    /**
     * @dataProvider additionProvider
     */
    public function testAddParametrized(int $a, int $b, int $expected): void
    {
        $this->assertSame($expected, $this->calc->add($a, $b));
    }

    public static function additionProvider(): array
    {
        return [
            [1, 1, 2],
            [0, 0, 0],
            [-1, 1, 0],
        ];
    }
}
```

---

## 4. 適用領域の詳細比較

### 4.1 領域別最適言語

```
AI / 機械学習           → Python（事実上の唯一の選択肢）
                          理由: NumPy, PyTorch, TensorFlow, scikit-learn
                          Jupyter Notebook での探索的分析

Web バックエンド         → 全て可能。JS/TS, Python, Ruby, PHP
                          大規模: Java/Go を検討すべき
                          スタートアップ: Rails or Next.js

Web フロントエンド       → JavaScript / TypeScript（唯一の選択肢）
                          React, Vue, Svelte, Angular

データ分析              → Python（Pandas, Polars, Matplotlib）
                          R（統計特化、学術分野）

自動化スクリプト         → Python（汎用性）, Bash（OS操作）
                          Perl（テキスト処理に特化）

DevOps / インフラ        → Python（Ansible, SaltStack）
                          Go（Docker, K8s, Terraform）
                          Ruby（Chef, Vagrant - 歴史的）

CMS / EC               → PHP（WordPress, WooCommerce, Magento）
                          世界のWebサイトの75%以上がPHPで動作

スタートアップ MVP      → Ruby on Rails（最速プロトタイピング）
                          Next.js（フルスタックTS）
                          Django（管理画面自動生成）

ゲームスクリプティング    → Lua（Unity, Roblox）
                          Python（Blender, ツール）
                          GDScript（Godot）

教育                    → Python（Scratch → Python パスが定番）
                          JavaScript（即座にブラウザで結果確認）
```

### 4.2 実務シナリオ別の言語選択

```
シナリオ1: SaaS バックエンド API
  推奨: TypeScript (NestJS/Fastify) or Python (FastAPI)
  理由: 型安全性、OpenAPI自動生成、非同期対応
  避けるべき: Perl（エコシステムが古い）

シナリオ2: データパイプライン
  推奨: Python (Pandas + Airflow)
  理由: データ処理ライブラリの充実度
  代替: Node.js（ストリーム処理に強い）

シナリオ3: レガシーシステムのリプレース
  PHP → Laravel: 段階的な移行が容易
  Ruby → Rails: ActiveRecord による DB 抽象化
  Perl → Python: 正規表現処理の移植が比較的容易

シナリオ4: EC サイト構築
  推奨: PHP (Laravel + Stripe/PayPal SDK)
  理由: 実績あるエコシステム、ホスティング容易
  代替: Next.js (Shopify Hydrogen)

シナリオ5: バッチ処理・ETL
  推奨: Python (SQLAlchemy + Celery)
  理由: DB接続ライブラリの充実、タスクキュー
  代替: Node.js（I/O多い場合に有利）
```

---

## 5. パフォーマンス比較

### 5.1 ベンチマーク（概算値）

```
フィボナッチ(40) の実行時間（相対値、V8 JavaScript = 1.0）:

JavaScript (V8/Node.js)  : 1.0x  （JITコンパイルが優秀）
PHP 8.3 (OPcache+JIT)   : 1.5x  （PHP 8のJITで大幅改善）
Ruby 3.3 (YJIT)         : 2.0x  （YJITで大幅改善）
Python 3.12              : 8.0x  （インタープリタ、GIL）
Perl 5.38                : 6.0x  （最適化済みインタープリタ）

※ 実際の Web アプリケーションではI/O待ちが支配的なため、
   CPU ベンチマークほどの差は出ない場合が多い
```

### 5.2 各言語のパフォーマンス改善の歴史

```python
# Python のパフォーマンス改善策
# 1. PyPy（JITコンパイラ実装） — CPython比で5-10倍高速
# 2. Cython（C拡張の自動生成）
# 3. NumPy/Polars（C/Rust実装のデータ処理）
# 4. asyncio（I/O並行処理）
# 5. multiprocessing（GIL回避のマルチプロセス）
# 6. Python 3.13+: Free-threaded Python（GIL除去実験）

# asyncio による並行 I/O
import asyncio
import aiohttp

async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_one(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()

# multiprocessing による CPU 並列処理
from multiprocessing import Pool

def heavy_computation(n: int) -> int:
    return sum(i * i for i in range(n))

with Pool(processes=4) as pool:
    results = pool.map(heavy_computation, [10**6, 10**6, 10**6, 10**6])
```

```ruby
# Ruby のパフォーマンス改善策
# 1. YJIT (Yet Another JIT) — Ruby 3.1+ に統合、3.3で成熟
# 2. Ractor（並列処理、GVL回避）
# 3. Fiber Scheduler（非同期I/O）

# Ractor による並列処理（Ruby 3.0+）
ractors = 4.times.map do |i|
  Ractor.new(i) do |id|
    # 各Ractorは独立したGVLを持つ
    result = (1..1_000_000).sum
    [id, result]
  end
end

results = ractors.map(&:take)
puts results.inspect

# Fiber Scheduler による非同期I/O
require 'async'

Async do
  urls = ['https://example.com', 'https://example.org']
  tasks = urls.map do |url|
    Async do
      # 非ブロッキングHTTPリクエスト
      response = Net::HTTP.get(URI(url))
      response.length
    end
  end
  results = tasks.map(&:wait)
end
```

```javascript
// Node.js のパフォーマンス特性
// V8エンジンのJITコンパイルにより、スクリプト言語で最高レベルの性能

// Worker Threads（CPU集約的処理の並列化）
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';

if (isMainThread) {
    const worker = new Worker(new URL(import.meta.url), {
        workerData: { start: 0, end: 1_000_000 }
    });
    worker.on('message', (result) => {
        console.log(`Result: ${result}`);
    });
} else {
    const { start, end } = workerData;
    let sum = 0;
    for (let i = start; i < end; i++) {
        sum += i * i;
    }
    parentPort.postMessage(sum);
}

// Stream APIによるメモリ効率の良い処理
import { createReadStream } from 'fs';
import { createInterface } from 'readline';

const rl = createInterface({
    input: createReadStream('large_file.txt'),
    crlfDelay: Infinity,
});

let lineCount = 0;
for await (const line of rl) {
    if (line.includes('ERROR')) {
        lineCount++;
    }
}
console.log(`Error lines: ${lineCount}`);
```

---

## 6. 型システムの比較

### 6.1 段階的型付け（Gradual Typing）

```python
# Python: 型ヒント（実行時には無視される）
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items.pop()

    def peek(self) -> T:
        if not self._items:
            raise IndexError("Stack is empty")
        return self._items[-1]

# mypy / Pyright で静的チェック
stack: Stack[int] = Stack()
stack.push(42)
stack.push("hello")  # mypy: error: Argument 1 has incompatible type "str"

# TypedDict（辞書の型指定）
from typing import TypedDict, NotRequired

class UserDict(TypedDict):
    name: str
    age: int
    email: NotRequired[str]

def process_user(user: UserDict) -> str:
    return f"{user['name']} ({user['age']})"
```

```typescript
// TypeScript: 高度な型システム

// Discriminated Unions（判別ユニオン）
type Shape =
    | { kind: "circle"; radius: number }
    | { kind: "rectangle"; width: number; height: number }
    | { kind: "triangle"; base: number; height: number };

function area(shape: Shape): number {
    switch (shape.kind) {
        case "circle":
            return Math.PI * shape.radius ** 2;
        case "rectangle":
            return shape.width * shape.height;
        case "triangle":
            return (shape.base * shape.height) / 2;
    }
    // TypeScript: exhaustiveness check（網羅性チェック）
}

// Template Literal Types
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";
type APIPath = `/api/${string}`;
type APIRoute = `${HTTPMethod} ${APIPath}`;

// Conditional Types + Infer
type ReturnTypeOf<T> = T extends (...args: any[]) => infer R ? R : never;
type ArrayElement<T> = T extends (infer E)[] ? E : never;

// Branded Types（ブランド型でプリミティブを区別）
type UserId = number & { readonly __brand: "UserId" };
type OrderId = number & { readonly __brand: "OrderId" };

function createUserId(id: number): UserId {
    return id as UserId;
}

function getUser(id: UserId): void { /* ... */ }
// getUser(createOrderId(1)); // コンパイルエラー！
```

```php
<?php
// PHP: 型宣言の強化（PHP 8.0+）
declare(strict_types=1);

// Union型 + Intersection型
function processInput(string|int $input): string|int {
    return match(true) {
        is_string($input) => strtoupper($input),
        is_int($input) => $input * 2,
    };
}

// PHPStan / Psalm による高度な静的解析
/**
 * @template T
 * @param array<T> $items
 * @param callable(T): bool $predicate
 * @return array<T>
 */
function filter_items(array $items, callable $predicate): array {
    return array_values(array_filter($items, $predicate));
}

// DNF型（Disjunctive Normal Form, PHP 8.2+）
function process((Countable&Iterator)|null $input): int {
    if ($input === null) {
        return 0;
    }
    return count($input);
}
```

---

## 7. 並行処理モデルの比較

```
┌──────────┬────────────────┬──────────────────────────────┐
│ 言語      │ 並行モデル      │ 特徴                          │
├──────────┼────────────────┼──────────────────────────────┤
│ Python   │ asyncio        │ シングルスレッド非同期I/O        │
│          │ multiprocessing│ マルチプロセス（GIL回避）       │
│          │ threading      │ I/O bound のみ有効（GIL制約）   │
├──────────┼────────────────┼──────────────────────────────┤
│ Ruby     │ Fiber          │ 協調的スレッド（軽量）          │
│          │ Ractor         │ 並列処理（GVL独立）            │
│          │ Thread         │ I/O bound のみ有効（GVL制約）   │
├──────────┼────────────────┼──────────────────────────────┤
│ JS/TS    │ Event Loop     │ シングルスレッド + 非同期I/O     │
│          │ Worker Threads │ CPU集約的処理の並列化           │
│          │ Cluster        │ マルチプロセスサーバー           │
├──────────┼────────────────┼──────────────────────────────┤
│ PHP      │ Fiber          │ 協調的スレッド（PHP 8.1+）      │
│          │ Swoole         │ 非同期イベント駆動              │
│          │ FrankenPHP     │ ワーカーモード                  │
├──────────┼────────────────┼──────────────────────────────┤
│ Perl     │ threads        │ インタープリタクローン           │
│          │ fork           │ プロセスフォーク                │
│          │ AnyEvent       │ イベント駆動                   │
└──────────┴────────────────┴──────────────────────────────┘
```

```python
# Python: asyncio + aiohttp の実践例
import asyncio
import aiohttp
from dataclasses import dataclass

@dataclass
class FetchResult:
    url: str
    status: int
    body_length: int

async def fetch_url(session: aiohttp.ClientSession, url: str) -> FetchResult:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        body = await resp.text()
        return FetchResult(url=url, status=resp.status, body_length=len(body))

async def fetch_all(urls: list[str], concurrency: int = 10) -> list[FetchResult]:
    """セマフォで同時接続数を制限しつつ並行フェッチ"""
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_fetch(session: aiohttp.ClientSession, url: str) -> FetchResult:
        async with semaphore:
            return await fetch_url(session, url)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 実行
urls = [f"https://httpbin.org/get?id={i}" for i in range(100)]
results = asyncio.run(fetch_all(urls, concurrency=20))
```

```typescript
// TypeScript/Node.js: Promise.allSettled + 並行制御
interface FetchResult {
    url: string;
    status: number;
    bodyLength: number;
}

async function fetchWithConcurrency(
    urls: string[],
    concurrency: number = 10
): Promise<FetchResult[]> {
    const results: FetchResult[] = [];

    // chunks に分割して並行実行
    for (let i = 0; i < urls.length; i += concurrency) {
        const chunk = urls.slice(i, i + concurrency);
        const settled = await Promise.allSettled(
            chunk.map(async (url): Promise<FetchResult> => {
                const resp = await fetch(url);
                const body = await resp.text();
                return { url, status: resp.status, bodyLength: body.length };
            })
        );

        for (const result of settled) {
            if (result.status === "fulfilled") {
                results.push(result.value);
            } else {
                console.error(`Failed: ${result.reason}`);
            }
        }
    }

    return results;
}
```

---

## 8. 開発ツールチェーンの比較

### 8.1 リンター・フォーマッター

```
Python:
  リンター: Ruff（超高速、Rust製）, Flake8, Pylint
  フォーマッター: Ruff format, Black
  型チェッカー: mypy, Pyright (Pylance)
  推奨: Ruff + Pyright（2025年のベストプラクティス）

Ruby:
  リンター/フォーマッター: RuboCop（統合型）
  型チェッカー: Sorbet, Steep + RBS
  推奨: RuboCop + Sorbet

JavaScript/TypeScript:
  リンター: ESLint（+ typescript-eslint）, Biome
  フォーマッター: Prettier, Biome
  型チェッカー: tsc（TypeScript Compiler）
  推奨: Biome（リント+フォーマット統合、Rust製で高速）

PHP:
  リンター: PHPStan, Psalm
  フォーマッター: PHP-CS-Fixer, Laravel Pint
  推奨: PHPStan Level 9 + PHP-CS-Fixer

Perl:
  リンター: Perl::Critic
  フォーマッター: Perl::Tidy
  推奨: Perl::Critic + Perl::Tidy
```

### 8.2 REPL・対話的開発

```
Python:
  標準: python3（REPL）
  強化: IPython（タブ補完、マジックコマンド）
  ノートブック: Jupyter Notebook / JupyterLab
  → データサイエンスの探索的開発に最適

Ruby:
  標準: irb（Interactive Ruby）
  強化: pry（デバッグ機能付きREPL）
  → binding.pry でブレークポイント設定可能

JavaScript/Node.js:
  標準: node（REPL）
  ブラウザ: DevTools Console
  → ブラウザ内で即座に実行・確認できる

PHP:
  標準: php -a（対話モード）
  強化: PsySH（Laravel tinker が利用）
  → tinker で Eloquent モデルを対話的に操作

Perl:
  標準: perl -de0（デバッガをREPL代わりに）
  強化: Reply, Devel::REPL
```

### 8.3 デプロイ方法の比較

```
Python:
  PaaS: Heroku, Railway, Render
  サーバーレス: AWS Lambda, Cloud Functions
  コンテナ: Docker + Gunicorn/Uvicorn
  推奨Dockerfile:
    FROM python:3.12-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    COPY . .
    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

Ruby:
  PaaS: Heroku（最も相性が良い）, Render
  コンテナ: Docker + Puma
  推奨: Kamal（Rails公式のデプロイツール、Docker ベース）

JavaScript/Node.js:
  Edge: Cloudflare Workers, Vercel Edge Functions
  サーバーレス: AWS Lambda, Vercel Functions
  PaaS: Vercel（Next.js最適）, Netlify
  コンテナ: Docker + PM2/Node直接

PHP:
  共有ホスティング: ほぼ全てのサーバーで動作（最大の利点）
  PaaS: Laravel Forge, Vapor（サーバーレス）
  コンテナ: Docker + PHP-FPM + Nginx
  → 最も安価にデプロイできる言語の1つ
```

---

## 9. 各言語のモダン化の歩み

### 9.1 Python の進化

```
Python 3.0  (2008): 2→3 の破壊的変更（Unicode標準、print関数化）
Python 3.5  (2015): async/await, 型ヒント(PEP 484)
Python 3.6  (2016): f-string, 変数アノテーション
Python 3.7  (2018): dataclasses, breakpoint()
Python 3.8  (2019): ウォルラス演算子(:=), Protocol
Python 3.9  (2020): 辞書のマージ演算子(|), 型ヒント簡略化
Python 3.10 (2021): 構造的パターンマッチ(match/case)
Python 3.11 (2022): 10-60%高速化, 例外グループ
Python 3.12 (2023): 型パラメータ構文, f-string改善
Python 3.13 (2024): Free-threaded実験的サポート(GIL除去)
```

### 9.2 Ruby の進化

```
Ruby 1.9  (2007): YARV VM, エンコーディング, 新ハッシュ構文
Ruby 2.0  (2013): キーワード引数, Module#prepend
Ruby 2.3  (2015): 安全ナビゲーション演算子(&.)
Ruby 2.6  (2018): JIT(MJIT), 無限Range
Ruby 3.0  (2020): Ractor(並列), Fiber Scheduler, RBS, パターンマッチ
Ruby 3.1  (2021): YJIT, デバッグ改善
Ruby 3.2  (2022): WASI対応, Regexp改善
Ruby 3.3  (2023): YJIT性能向上, Prism Parser
```

### 9.3 JavaScript の進化

```
ES5     (2009): strict mode, Array extras (map, filter, reduce)
ES6/2015(2015): let/const, arrow, class, Promise, module, template literal
ES2016  (2016): Array.includes, ** 演算子
ES2017  (2017): async/await, Object.entries/values
ES2018  (2018): rest/spread properties, for await...of
ES2019  (2019): Array.flat/flatMap, Optional catch
ES2020  (2020): Optional chaining(?.), Nullish coalescing(??)
ES2021  (2021): Logical assignment, String.replaceAll
ES2022  (2022): Top-level await, Array.at, Error cause
ES2023  (2023): Array findLast, Hashbang, Change Array by Copy
ES2024  (2024): Grouping (Object.groupBy), Promise.withResolvers
```

### 9.4 PHP の進化

```
PHP 5.3  (2009): 名前空間, クロージャ, 遅延静的束縛
PHP 5.4  (2012): トレイト, 短い配列構文, ビルトインサーバー
PHP 7.0  (2015): 戻り値型宣言, null合体演算子, 2倍の性能
PHP 7.4  (2019): Arrow関数, 型付きプロパティ, プリロード
PHP 8.0  (2020): JIT, Union型, match式, Named引数, Attribute
PHP 8.1  (2021): Enum, Fiber, Readonly, Intersection型
PHP 8.2  (2022): Readonly class, DNF型, 動的プロパティ非推奨
PHP 8.3  (2023): 型付きクラス定数, json_validate, Override属性
PHP 8.4  (2024): プロパティフック, 非対称可視性
```

---

## 10. 実践的なプロジェクト構成例

### 10.1 Python（FastAPI）プロジェクト

```
myapp/
├── pyproject.toml          # プロジェクト設定・依存関係
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── main.py          # FastAPI アプリケーション
│       ├── config.py        # 設定管理
│       ├── models/          # Pydantic モデル
│       │   ├── __init__.py
│       │   └── user.py
│       ├── routers/         # APIルーター
│       │   ├── __init__.py
│       │   └── users.py
│       ├── services/        # ビジネスロジック
│       │   ├── __init__.py
│       │   └── user_service.py
│       └── repositories/    # データアクセス
│           ├── __init__.py
│           └── user_repo.py
├── tests/
│   ├── conftest.py
│   └── test_users.py
├── Dockerfile
└── docker-compose.yml
```

### 10.2 Ruby on Rails プロジェクト

```
myapp/
├── Gemfile                  # 依存関係
├── config/
│   ├── routes.rb            # ルーティング
│   ├── database.yml         # DB設定
│   └── application.rb       # アプリ設定
├── app/
│   ├── controllers/
│   │   └── users_controller.rb
│   ├── models/
│   │   └── user.rb          # ActiveRecord モデル
│   ├── views/
│   │   └── users/
│   │       ├── index.html.erb
│   │       └── show.html.erb
│   └── services/
│       └── user_service.rb
├── db/
│   └── migrate/             # マイグレーションファイル
├── spec/                    # RSpec テスト
│   ├── models/
│   └── requests/
└── Dockerfile
```

### 10.3 Next.js（TypeScript）プロジェクト

```
myapp/
├── package.json             # 依存関係
├── tsconfig.json            # TypeScript設定
├── next.config.ts           # Next.js設定
├── src/
│   ├── app/                 # App Router
│   │   ├── layout.tsx       # ルートレイアウト
│   │   ├── page.tsx         # ホームページ
│   │   ├── users/
│   │   │   ├── page.tsx     # ユーザー一覧
│   │   │   └── [id]/
│   │   │       └── page.tsx # ユーザー詳細
│   │   └── api/
│   │       └── users/
│   │           └── route.ts # API Route Handler
│   ├── components/
│   │   ├── ui/              # 汎用UIコンポーネント
│   │   └── features/        # 機能別コンポーネント
│   ├── lib/                 # ユーティリティ
│   └── types/               # 型定義
├── tests/
│   └── users.test.tsx
└── Dockerfile
```

### 10.4 Laravel（PHP）プロジェクト

```
myapp/
├── composer.json            # 依存関係
├── artisan                  # CLIツール
├── app/
│   ├── Http/
│   │   ├── Controllers/
│   │   │   └── UserController.php
│   │   └── Middleware/
│   ├── Models/
│   │   └── User.php         # Eloquent モデル
│   └── Services/
│       └── UserService.php
├── config/                  # 設定ファイル
├── database/
│   ├── migrations/          # マイグレーション
│   └── seeders/             # シーダー
├── resources/
│   └── views/               # Blade テンプレート
├── routes/
│   ├── web.php              # Webルート
│   └── api.php              # APIルート
├── tests/
│   ├── Feature/
│   └── Unit/
└── docker-compose.yml
```

---

## 11. 各言語の「キラーアプリ」

```
Python:
  - NumPy / PyTorch / TensorFlow: AI/MLの事実上の標準
  - Jupyter Notebook: 対話的データ分析
  - Django Admin: 自動生成管理画面
  - Ansible: インフラ自動化
  - Scrapy: Webスクレイピング

Ruby:
  - Ruby on Rails: Webアプリケーション開発の革命
  - Homebrew: macOS パッケージマネージャ
  - Vagrant: 仮想環境管理（歴史的）
  - Fastlane: モバイルアプリCI/CD
  - Jekyll: 静的サイトジェネレータ（GitHub Pages標準）

JavaScript/TypeScript:
  - React / Vue / Angular: フロントエンドUI
  - Next.js: フルスタックフレームワーク
  - Electron: デスクトップアプリ（VS Code, Discord, Slack）
  - React Native: モバイルアプリ
  - Three.js: 3D WebGL

PHP:
  - WordPress: 世界のWebの43%
  - Laravel: モダンPHP開発
  - Magento / WooCommerce: EC
  - MediaWiki: Wikipedia のエンジン
  - Drupal: エンタープライズCMS

Perl:
  - CPAN: 先駆的パッケージリポジトリ
  - cPanel: サーバー管理
  - Bugzilla: バグトラッキング
  - Movable Type: 初期のブログシステム
  - BioPerl: バイオインフォマティクス
```

---

## 12. 学習ロードマップ

### 12.1 初学者向け推奨パス

```
完全な初心者:
  Python → JavaScript/TypeScript
  理由: Pythonで基礎概念を学び、JSでWeb開発を学ぶ

Web開発志望:
  JavaScript/TypeScript → (必要に応じて) PHP or Ruby
  理由: フロントエンドは必須。バックエンドは選択

データサイエンス志望:
  Python → SQL → (必要に応じて) R
  理由: Pythonのデータエコシステムは圧倒的

スタートアップ志望:
  TypeScript (Next.js) or Ruby (Rails)
  理由: 最速でMVPを作れる
```

### 12.2 各言語の学習に必要な時間の目安

```
基本文法を覚える:
  Python:     2-4 週間（最も直感的）
  Ruby:       2-4 週間（Python経験者なら1-2週間）
  JavaScript: 2-4 週間（非同期処理の理解にやや時間がかかる）
  PHP:        2-4 週間（Webの文脈で学ぶと早い）
  TypeScript: 1-2 週間（JS経験者）/ 4-6週間（初学者）
  Perl:       4-6 週間（構文が特殊、正規表現の学習も含む）

フレームワークを使いこなす:
  Django:     4-8 週間
  Rails:      4-8 週間（Convention が多い分、覚えることも多い）
  Next.js:    4-8 週間（React の理解が前提）
  Laravel:    4-8 週間
  Express:    2-4 週間（最も軽量）

実務レベル:
  全ての言語で 6-12 ヶ月（実プロジェクト経験が必須）
```

---

## 13. 言語間移行ガイド

### 13.1 Python → TypeScript

```python
# Python
def process_users(users: list[dict]) -> list[str]:
    return [
        user["name"].upper()
        for user in users
        if user.get("active", False)
    ]
```

```typescript
// TypeScript（同じロジック）
interface User {
    name: string;
    active?: boolean;
}

function processUsers(users: User[]): string[] {
    return users
        .filter(user => user.active ?? false)
        .map(user => user.name.toUpperCase());
}
```

### 13.2 Ruby → Python

```ruby
# Ruby
class UserService
  def initialize(repo)
    @repo = repo
  end

  def active_users
    @repo.all
         .select { |u| u.active? }
         .sort_by { |u| u.name }
         .map { |u| u.name.upcase }
  end
end
```

```python
# Python（同じロジック）
class UserService:
    def __init__(self, repo):
        self._repo = repo

    def active_users(self) -> list[str]:
        users = self._repo.all()
        return sorted(
            [u.name.upper() for u in users if u.is_active()],
        )
```

### 13.3 PHP → Python

```php
<?php
// PHP
class UserService {
    public function __construct(
        private readonly UserRepository $repo,
    ) {}

    public function getActiveUsers(): array {
        $users = $this->repo->findAll();
        $active = array_filter($users, fn(User $u) => $u->isActive());
        $names = array_map(fn(User $u) => strtoupper($u->name), $active);
        sort($names);
        return $names;
    }
}
```

```python
# Python（同じロジック）
class UserService:
    def __init__(self, repo: UserRepository) -> None:
        self._repo = repo

    def get_active_users(self) -> list[str]:
        users = self._repo.find_all()
        return sorted(
            u.name.upper() for u in users if u.is_active()
        )
```

---

## まとめ

| 言語 | 一言で表すなら | 最強の領域 | 2025年の状況 |
|------|-------------|----------|-------------|
| Python | 万能の優等生 | AI/ML/データ | 不動の地位。AI時代の最重要言語 |
| Ruby | 開発者の幸福 | Web（Rails） | Rails 8 で復調。スタートアップで健在 |
| JavaScript | Webの支配者 | フルスタックWeb | ブラウザ唯一の言語。エコシステム最大 |
| TypeScript | JS+型安全 | 大規模Web開発 | 事実上のJS標準。採用率80%超 |
| PHP | Web特化の実用家 | CMS/EC | PHP 8系で大幅改善。WordPress健在 |
| Perl | テキスト処理の達人 | レガシー/バイオ | 新規採用は少ないが、既存システムで現役 |

### 選択のフローチャート

```
Q1: 何を作りたい？
├── AI/ML/データ分析 → Python（一択）
├── Web フロントエンド → JavaScript/TypeScript（一択）
├── Web バックエンド
│   ├── スタートアップ → Rails or Next.js
│   ├── 大規模エンタープライズ → Java/Go（スクリプト言語を超える場合）
│   ├── CMS/EC → PHP (Laravel/WordPress)
│   └── APIサーバー → TypeScript (NestJS) or Python (FastAPI)
├── モバイルアプリ → TypeScript (React Native) or Swift/Kotlin
├── 自動化スクリプト → Python or Bash
└── テキスト処理 → Python（汎用）or Perl（正規表現超特化）
```

---

## 次に読むべきガイド
→ [[01-systems-languages.md]] -- システム言語比較

---

## 参考文献
1. "Stack Overflow Developer Survey 2024." stackoverflow.com.
2. Van Rossum, G. "The Python Tutorial." docs.python.org.
3. Matsumoto, Y. "The Ruby Programming Language." O'Reilly, 2008.
4. Flanagan, D. "JavaScript: The Definitive Guide." 7th Ed, O'Reilly, 2020.
5. Lockhart, J. "Modern PHP." O'Reilly, 2015.
6. Tatroe, K. & MacIntyre, P. "Programming PHP." 4th Ed, O'Reilly, 2020.
7. "State of JS 2024." stateofjs.com.
8. "State of Developer Ecosystem 2024." JetBrains.
9. "JetBrains Developer Ecosystem Survey 2024." jetbrains.com.
10. "Python Developer Survey Results 2024." jetbrains.com/lp/devecosystem-2024/python.
