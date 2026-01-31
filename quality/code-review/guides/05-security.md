# セキュリティレビューガイド

## 概要

セキュリティレビューは、コードの脆弱性を発見し、攻撃から保護するための重要なレビューです。OWASP Top 10を中心に、一般的なセキュリティ問題とその対策を確認します。

## 目次

1. [OWASP Top 10](#owasp-top-10)
2. [入力値検証](#入力値検証)
3. [認証と認可](#認証と認可)
4. [データ保護](#データ保護)
5. [セキュアコーディング](#セキュアコーディング)
6. [言語別セキュリティ](#言語別セキュリティ)

---

## OWASP Top 10

### 1. Injection（インジェクション）

#### SQL Injection

```python
# ❌ Bad: SQLインジェクションの脆弱性
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

# 攻撃例: get_user("1 OR 1=1")
# 実行されるSQL: SELECT * FROM users WHERE id = 1 OR 1=1
# 全ユーザーが取得されてしまう

# ✅ Good: プリペアドステートメント
def get_user(user_id):
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))

# ✅ Better: ORMを使用
def get_user(user_id):
    return db.query(User).filter(User.id == user_id).first()
```

#### Command Injection

```typescript
// ❌ Bad: コマンドインジェクション
import { exec } from 'child_process';

function convertImage(filename: string): void {
  exec(`convert ${filename} output.png`);
  // 攻撃例: filename = "input.jpg; rm -rf /"
}

// ✅ Good: 入力値を検証し、安全なAPIを使用
import { execFile } from 'child_process';

function convertImage(filename: string): void {
  // ファイル名を検証
  if (!/^[a-zA-Z0-9_-]+\.(jpg|png)$/.test(filename)) {
    throw new Error('Invalid filename');
  }

  // 引数を配列で渡す（シェル経由しない）
  execFile('convert', [filename, 'output.png'], (error) => {
    if (error) {
      console.error('Conversion failed:', error);
    }
  });
}
```

### 2. Broken Authentication（認証の不備）

```swift
// ❌ Bad: 弱いパスワードポリシー
func validatePassword(_ password: String) -> Bool {
    return password.count >= 6
}

// ✅ Good: 強力なパスワードポリシー
func validatePassword(_ password: String) -> Bool {
    let minLength = 12
    let hasUppercase = password.range(of: "[A-Z]", options: .regularExpression) != nil
    let hasLowercase = password.range(of: "[a-z]", options: .regularExpression) != nil
    let hasNumber = password.range(of: "[0-9]", options: .regularExpression) != nil
    let hasSpecial = password.range(of: "[!@#$%^&*(),.?\":{}|<>]", options: .regularExpression) != nil

    return password.count >= minLength &&
           hasUppercase &&
           hasLowercase &&
           hasNumber &&
           hasSpecial
}

// ❌ Bad: パスワードを平文で保存
func saveUser(email: String, password: String) {
    let user = User(email: email, password: password)
    database.save(user)
}

// ✅ Good: パスワードをハッシュ化
import CryptoKit

func saveUser(email: String, password: String) throws {
    let salt = generateSalt()
    let hashedPassword = hashPassword(password, salt: salt)

    let user = User(
        email: email,
        passwordHash: hashedPassword,
        salt: salt
    )
    database.save(user)
}

func hashPassword(_ password: String, salt: String) -> String {
    let combined = password + salt
    let data = Data(combined.utf8)
    let hashed = SHA256.hash(data: data)
    return hashed.compactMap { String(format: "%02x", $0) }.joined()
}

// ✅ Better: bcryptなど専用ライブラリを使用
import BCrypt

func saveUser(email: String, password: String) throws {
    let hashedPassword = try BCrypt.hash(password)
    let user = User(email: email, passwordHash: hashedPassword)
    database.save(user)
}
```

### 3. Sensitive Data Exposure（機密データの露出）

```go
// ❌ Bad: 機密情報をログに出力
func LoginUser(email, password string) error {
    log.Printf("Login attempt: email=%s, password=%s", email, password)
    // パスワードがログに残る！

    user, err := db.FindUser(email)
    if err != nil {
        return err
    }

    log.Printf("User data: %+v", user)
    // ユーザーのすべてのフィールドがログに残る

    return nil
}

// ✅ Good: 機密情報をマスク
func LoginUser(email, password string) error {
    log.Printf("Login attempt: email=%s", maskEmail(email))

    user, err := db.FindUser(email)
    if err != nil {
        log.Printf("Login failed: %v", err)
        return err
    }

    log.Printf("Login successful: user_id=%s", user.ID)
    // 必要最小限の情報のみログに記録

    return nil
}

func maskEmail(email string) string {
    parts := strings.Split(email, "@")
    if len(parts) != 2 {
        return "***"
    }

    username := parts[0]
    if len(username) <= 2 {
        return "***@" + parts[1]
    }

    return username[0:2] + "***@" + parts[1]
}

// ❌ Bad: 機密データをクライアントに送信
type User struct {
    ID           string
    Email        string
    PasswordHash string  // クライアントに送ってはいけない
    APIKey       string  // クライアントに送ってはいけない
}

func GetUser(id string) (*User, error) {
    return db.FindUser(id)  // すべてのフィールドが返される
}

// ✅ Good: DTOで必要なデータのみ返す
type UserDTO struct {
    ID    string `json:"id"`
    Email string `json:"email"`
    Name  string `json:"name"`
}

func GetUser(id string) (*UserDTO, error) {
    user, err := db.FindUser(id)
    if err != nil {
        return nil, err
    }

    return &UserDTO{
        ID:    user.ID,
        Email: user.Email,
        Name:  user.Name,
    }, nil
}
```

### 4. XML External Entities (XXE)

```typescript
// ❌ Bad: XXE攻撃に脆弱
import { parseString } from 'xml2js';

function parseXML(xmlString: string): void {
  parseString(xmlString, (err, result) => {
    console.log(result);
  });
}

// ✅ Good: 外部エンティティを無効化
import { parseStringPromise } from 'xml2js';

async function parseXML(xmlString: string): Promise<any> {
  const options = {
    // 外部エンティティを無効化
    explicitCharkey: true,
    trim: true,
    normalize: true,
    normalizeTags: true,
    // DTDやエンティティ展開を無効化
    async: false,
    strict: true,
  };

  return await parseStringPromise(xmlString, options);
}
```

### 5. Broken Access Control（アクセス制御の不備）

```python
# ❌ Bad: 認可チェックなし
@app.route('/api/users/<user_id>/profile', methods=['PUT'])
def update_profile(user_id):
    data = request.json
    user = User.query.get(user_id)
    user.name = data['name']
    user.email = data['email']
    db.session.commit()
    return jsonify(user)

# 攻撃例: 他人のプロフィールを更新できてしまう

# ✅ Good: 認可チェック
@app.route('/api/users/<user_id>/profile', methods=['PUT'])
@login_required
def update_profile(user_id):
    # 現在のユーザーが更新権限を持つか確認
    if current_user.id != user_id and not current_user.is_admin:
        abort(403, "You don't have permission to update this profile")

    data = request.json
    user = User.query.get_or_404(user_id)

    user.name = data['name']
    user.email = data['email']
    db.session.commit()

    return jsonify(user)

# ✅ Better: デコレーターで権限チェック
def require_owner_or_admin(f):
    @wraps(f)
    def decorated_function(user_id, *args, **kwargs):
        if current_user.id != user_id and not current_user.is_admin:
            abort(403)
        return f(user_id, *args, **kwargs)
    return decorated_function

@app.route('/api/users/<user_id>/profile', methods=['PUT'])
@login_required
@require_owner_or_admin
def update_profile(user_id):
    # 権限チェック済み
    data = request.json
    user = User.query.get_or_404(user_id)
    user.name = data['name']
    user.email = data['email']
    db.session.commit()
    return jsonify(user)
```

### 6. Security Misconfiguration（セキュリティ設定ミス）

```typescript
// ❌ Bad: 本番環境でデバッグモード
const app = express();

app.use(express.json());

if (process.env.NODE_ENV === 'development') {
  app.use(morgan('dev'));
}

// デバッグモードが常にON
app.set('view cache', false);
app.set('trust proxy', true);

// ✅ Good: 環境に応じた設定
const app = express();

// セキュリティヘッダー
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
}));

// 環境別設定
if (process.env.NODE_ENV === 'production') {
  app.set('view cache', true);
  app.set('trust proxy', 1);
  // エラーの詳細を隠す
  app.use((err, req, res, next) => {
    res.status(500).json({ error: 'Internal server error' });
  });
} else {
  app.use(morgan('dev'));
  // 開発環境では詳細なエラーを表示
  app.use((err, req, res, next) => {
    res.status(500).json({
      error: err.message,
      stack: err.stack,
    });
  });
}
```

### 7. Cross-Site Scripting (XSS)

```javascript
// ❌ Bad: XSS脆弱性
function displayUserComment(comment) {
  document.getElementById('comment').innerHTML = comment;
  // 攻撃例: comment = "<script>alert('XSS')</script>"
}

// ✅ Good: エスケープ処理
function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function displayUserComment(comment) {
  document.getElementById('comment').textContent = comment;
  // または
  document.getElementById('comment').innerHTML = escapeHtml(comment);
}

// ✅ React: 自動エスケープ
function CommentComponent({ comment }) {
  return <div>{comment}</div>;  // 自動的にエスケープされる
}

// ❌ Bad: dangerouslySetInnerHTML
function CommentComponent({ comment }) {
  return <div dangerouslySetInnerHTML={{ __html: comment }} />;
}

// ✅ Good: サニタイズしてから使用
import DOMPurify from 'dompurify';

function CommentComponent({ comment }) {
  const sanitized = DOMPurify.sanitize(comment);
  return <div dangerouslySetInnerHTML={{ __html: sanitized }} />;
}
```

### 8. Insecure Deserialization（安全でないデシリアライゼーション）

```python
# ❌ Bad: pickle使用（任意コード実行の危険）
import pickle

def load_user_data(data):
    return pickle.loads(data)

# ✅ Good: JSON使用
import json

def load_user_data(data):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON data")

# ✅ Better: スキーマ検証
from pydantic import BaseModel, ValidationError

class UserData(BaseModel):
    id: int
    name: str
    email: str

def load_user_data(data: str) -> UserData:
    try:
        return UserData.parse_raw(data)
    except ValidationError as e:
        raise ValueError(f"Invalid user data: {e}")
```

### 9. Using Components with Known Vulnerabilities

```bash
# ❌ Bad: 古い依存関係
{
  "dependencies": {
    "express": "3.0.0",  # 脆弱性あり
    "lodash": "4.17.15"  # 脆弱性あり
  }
}

# ✅ Good: 定期的に更新
npm audit
npm audit fix

# または
yarn audit
yarn audit fix

# 自動更新チェック（CI/CD）
npm install -g npm-check-updates
ncu -u
npm install
```

### 10. Insufficient Logging & Monitoring

```go
// ❌ Bad: ログが不十分
func LoginUser(email, password string) error {
    user, _ := db.FindUser(email)
    // エラーを無視

    if !checkPassword(user, password) {
        return errors.New("login failed")
    }

    return nil
}

// ✅ Good: 適切なログとモニタリング
func LoginUser(email, password string) error {
    user, err := db.FindUser(email)
    if err != nil {
        log.WithFields(log.Fields{
            "email": maskEmail(email),
            "error": err.Error(),
        }).Warn("User not found during login attempt")
        return errors.New("invalid credentials")
    }

    if !checkPassword(user, password) {
        log.WithFields(log.Fields{
            "user_id": user.ID,
            "email":   maskEmail(email),
        }).Warn("Failed login attempt - invalid password")

        // 連続失敗をカウント
        incrementFailedAttempts(user.ID)

        return errors.New("invalid credentials")
    }

    // 成功ログ
    log.WithFields(log.Fields{
        "user_id": user.ID,
        "ip":      getCurrentIP(),
    }).Info("Successful login")

    // 失敗カウントをリセット
    resetFailedAttempts(user.ID)

    return nil
}
```

---

## 入力値検証

### サーバーサイド検証の必須化

```typescript
// ❌ Bad: クライアント側のみで検証
// client.ts
function submitForm(data: FormData): void {
  if (!data.email.includes('@')) {
    alert('Invalid email');
    return;
  }
  // サーバーに送信
  fetch('/api/users', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// server.ts
app.post('/api/users', (req, res) => {
  const user = new User(req.body);
  user.save();  // 検証なし！
});

// ✅ Good: サーバー側でも必ず検証
// server.ts
app.post('/api/users', (req, res) => {
  // バリデーション
  const errors = validateUserData(req.body);
  if (errors.length > 0) {
    return res.status(400).json({ errors });
  }

  const user = new User(req.body);
  user.save();
  res.json(user);
});

function validateUserData(data: any): string[] {
  const errors: string[] = [];

  if (!data.email || !isValidEmail(data.email)) {
    errors.push('Invalid email');
  }

  if (!data.name || data.name.length < 2) {
    errors.push('Name must be at least 2 characters');
  }

  if (!data.password || data.password.length < 12) {
    errors.push('Password must be at least 12 characters');
  }

  return errors;
}
```

---

## 認証と認可

### JWTの安全な使用

```python
# ❌ Bad: セキュアでないJWT
import jwt

def create_token(user_id):
    # 秘密鍵が弱い
    return jwt.encode({'user_id': user_id}, 'secret', algorithm='HS256')

def verify_token(token):
    # 検証なし
    return jwt.decode(token, 'secret', algorithms=['HS256'])

# ✅ Good: セキュアなJWT
import jwt
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv('JWT_SECRET_KEY')  # 環境変数から取得
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 15

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        'user_id': user_id,
        'exp': expire,
        'iat': datetime.utcnow(),
        'type': 'access'
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={
                'verify_exp': True,
                'verify_iat': True,
            }
        )

        if payload.get('type') != 'access':
            raise jwt.InvalidTokenError('Invalid token type')

        return payload

    except jwt.ExpiredSignatureError:
        raise ValueError('Token has expired')
    except jwt.InvalidTokenError as e:
        raise ValueError(f'Invalid token: {e}')
```

---

## データ保護

### 暗号化

```swift
// ❌ Bad: 暗号化せずに保存
func saveAPIKey(_ apiKey: String) {
    UserDefaults.standard.set(apiKey, forKey: "api_key")
}

// ✅ Good: Keychainに保存
import Security

func saveAPIKey(_ apiKey: String) -> Bool {
    let data = apiKey.data(using: .utf8)!

    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrAccount as String: "api_key",
        kSecValueData as String: data,
        kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
    ]

    SecItemDelete(query as CFDictionary)  // 既存を削除
    let status = SecItemAdd(query as CFDictionary, nil)

    return status == errSecSuccess
}

func getAPIKey() -> String? {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrAccount as String: "api_key",
        kSecReturnData as String: true
    ]

    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)

    guard status == errSecSuccess,
          let data = result as? Data,
          let apiKey = String(data: data, encoding: .utf8) else {
        return nil
    }

    return apiKey
}
```

---

## レビューチェックリスト

### セキュリティレビュー完全チェックリスト

#### インジェクション
- [ ] SQLインジェクション対策済み
- [ ] コマンドインジェクション対策済み
- [ ] XSS対策済み

#### 認証・認可
- [ ] 強力なパスワードポリシー
- [ ] パスワードのハッシュ化
- [ ] 適切な認可チェック
- [ ] セッション管理が適切

#### データ保護
- [ ] 機密データが暗号化されている
- [ ] ログに機密情報が含まれない
- [ ] HTTPSを使用
- [ ] 機密データがクライアントに送信されない

#### 入力検証
- [ ] すべての入力が検証されている
- [ ] サーバー側で検証されている
- [ ] ホワイトリスト方式
- [ ] 適切なエラーメッセージ

---

## まとめ

セキュリティは後から追加できません。設計段階から考慮する必要があります。

### 重要ポイント

1. **入力を信用しない**
2. **最小権限の原則**
3. **多層防御**
4. **セキュアなデフォルト**
5. **定期的な監査**

### 次のステップ

- [パフォーマンスレビュー](06-performance.md)
- [保守性レビュー](07-maintainability.md)
