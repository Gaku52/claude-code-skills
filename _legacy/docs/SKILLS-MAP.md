# Claude Code Skills 関連図

## 全体構成（26スキル + 補助スキル）

```mermaid
graph TB
    subgraph Frontend["フロントエンド開発"]
        react[React Development<br/>React開発]
        nextjs[Next.js Development<br/>Next.js開発]
        web[Web Development<br/>Web開発]
        frontend-perf[Frontend Performance<br/>フロントエンド最適化]
        web-a11y[Web Accessibility<br/>Webアクセシビリティ]
    end

    subgraph Backend["バックエンド開発"]
        backend[Backend Development<br/>バックエンド開発]
        nodejs[Node.js Development<br/>Node.js開発]
        python[Python Development<br/>Python開発]
        db[Database Design<br/>データベース設計]
    end

    subgraph Mobile["モバイル開発"]
        ios[iOS Development<br/>iOS開発]
        swiftui[SwiftUI Patterns<br/>SwiftUIパターン]
        ios-security[iOS Security<br/>iOSセキュリティ]
        ios-setup[iOS Project Setup<br/>iOSプロジェクト構成]
        networking[Networking & Data<br/>ネットワーク通信]
    end

    subgraph Tools["開発ツール"]
        cli[CLI Development<br/>CLIツール開発]
        script[Script Development<br/>スクリプト開発]
        mcp[MCP Development<br/>MCP開発]
    end

    subgraph Quality["品質保証"]
        testing[Testing Strategy<br/>テスト戦略]
        qa[Quality Assurance<br/>品質保証]
        code-review[Code Review<br/>コードレビュー]
    end

    subgraph DevOps["DevOps・運用"]
        cicd[CI/CD Automation<br/>CI/CD自動化]
        git[Git Workflow<br/>Git運用]
        deps[Dependency Management<br/>依存関係管理]
    end

    subgraph Knowledge["ナレッジ管理"]
        incident[Incident Logger<br/>インシデント記録]
        lessons[Lessons Learned<br/>教訓管理]
        docs[Documentation<br/>ドキュメント]
    end

    subgraph Other["その他"]
        api-cost[API Cost Skill<br/>APIコスト管理]
    end

    %% フロントエンド内部の関連
    react --> nextjs
    react --> web
    react --> frontend-perf
    web --> web-a11y
    nextjs --> frontend-perf

    %% バックエンド内部の関連
    backend --> nodejs
    backend --> python
    backend --> db
    nodejs --> backend

    %% モバイル内部の関連
    ios --> swiftui
    ios --> ios-security
    ios --> ios-setup
    ios --> networking
    swiftui --> ios-setup

    %% フロントエンド↔バックエンド
    nextjs -.API通信.-> backend
    react -.API通信.-> backend

    %% モバイル↔バックエンド
    ios -.API通信.-> backend
    networking -.-> backend

    %% ツール関連
    cli --> script
    mcp --> cli

    %% 品質保証の関連
    testing --> qa
    code-review --> qa
    testing -.適用.-> react
    testing -.適用.-> backend
    testing -.適用.-> ios

    %% DevOps関連
    git --> cicd
    deps --> cicd
    cicd -.デプロイ.-> nextjs
    cicd -.デプロイ.-> backend
    cicd -.デプロイ.-> ios

    %% ナレッジ管理
    incident --> lessons
    lessons --> docs
    code-review -.フィードバック.-> lessons

    %% その他
    api-cost -.監視.-> backend
    api-cost -.監視.-> mcp

    %% ドキュメント横断
    docs -.適用.-> react
    docs -.適用.-> backend
    docs -.適用.-> ios

    classDef frontend fill:#61dafb,stroke:#333,stroke-width:2px,color:#000
    classDef backend fill:#68a063,stroke:#333,stroke-width:2px,color:#fff
    classDef mobile fill:#fa7343,stroke:#333,stroke-width:2px,color:#fff
    classDef tools fill:#ffd700,stroke:#333,stroke-width:2px,color:#000
    classDef quality fill:#9b59b6,stroke:#333,stroke-width:2px,color:#fff
    classDef devops fill:#3498db,stroke:#333,stroke-width:2px,color:#fff
    classDef knowledge fill:#95a5a6,stroke:#333,stroke-width:2px,color:#000
    classDef other fill:#e74c3c,stroke:#333,stroke-width:2px,color:#fff

    class react,nextjs,web,frontend-perf,web-a11y frontend
    class backend,nodejs,python,db backend
    class ios,swiftui,ios-security,ios-setup,networking mobile
    class cli,script,mcp tools
    class testing,qa,code-review quality
    class cicd,git,deps devops
    class incident,lessons,docs knowledge
    class api-cost other
```

## スキル間の主な関係性

### 1. フロントエンド開発チェーン
- **React** → **Next.js** → **Frontend Performance**
- **Web Development** → **Web Accessibility**

### 2. バックエンド開発チェーン
- **Backend Development** ⇄ **Node.js/Python**
- **Backend** → **Database Design**

### 3. iOS開発チェーン
- **iOS Development** → **SwiftUI Patterns** → **iOS Project Setup**
- **iOS** → **iOS Security** + **Networking & Data**

### 4. 品質保証チェーン
- **Testing Strategy** → **Quality Assurance** ← **Code Review**

### 5. DevOpsチェーン
- **Git Workflow** → **CI/CD Automation** ← **Dependency Management**

### 6. ナレッジ管理チェーン
- **Incident Logger** → **Lessons Learned** → **Documentation**

### 7. 横断的な関係
- **Testing** → すべての開発スキル（React, Backend, iOS）
- **CI/CD** → デプロイ対象（Next.js, Backend, iOS）
- **Documentation** → すべての開発スキル
- **API Cost** → Backend, MCP

## 学習推奨パス

### パス1: フロントエンドエンジニア
1. Web Development
2. React Development
3. Next.js Development
4. Frontend Performance
5. Web Accessibility
6. Testing Strategy

### パス2: バックエンドエンジニア
1. Backend Development
2. Node.js/Python Development
3. Database Design
4. Testing Strategy
5. CI/CD Automation

### パス3: iOSエンジニア
1. iOS Development
2. SwiftUI Patterns
3. iOS Project Setup
4. Networking & Data
5. iOS Security
6. Testing Strategy

### パス4: フルスタックエンジニア
1. Backend Development
2. Database Design
3. React Development
4. Next.js Development
5. Testing Strategy
6. CI/CD Automation
7. Git Workflow

### パス5: DevOpsエンジニア
1. Git Workflow
2. Dependency Management
3. CI/CD Automation
4. Script Development
5. CLI Development

## 文字数統計

| カテゴリ | スキル数 | 合計文字数 |
|---------|---------|-----------|
| フロントエンド | 5 | 約63万字 |
| バックエンド | 4 | 約102万字 |
| モバイル | 5 | 約71万字 |
| 品質保証 | 3 | 約73万字 |
| DevOps | 3 | 約46万字 |
| ナレッジ管理 | 3 | 約29万字 |
| ツール | 3 | 約36万字 |
| その他 | 1 | 約109万字 |

**合計: 約529万字（node_modules除く）**
