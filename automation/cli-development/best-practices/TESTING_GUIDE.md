# CLI Testing Guide

## テスト戦略

### テストピラミッド

```
       /\
      /  \        E2E Tests (5%)
     /____\
    /      \      Integration Tests (15%)
   /________\
  /          \    Unit Tests (80%)
 /____________\
```

## ユニットテスト

### Node.js (Jest)

**セットアップ**:
```bash
npm install -D jest @types/jest ts-jest
```

**jest.config.js**:
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.test.ts',
    '!src/**/*.spec.ts'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
}
```

**コマンドのテスト**:
```typescript
// src/commands/create.test.ts
import { createCommand } from './create'
import { ProjectService } from '../core/ProjectService'

describe('create command', () => {
  let mockProjectService: jest.Mocked<ProjectService>

  beforeEach(() => {
    mockProjectService = {
      create: jest.fn(),
      list: jest.fn(),
      delete: jest.fn()
    } as any
  })

  it('should create a project', async () => {
    const command = createCommand(mockProjectService)
    await command.parseAsync(['node', 'test', 'myapp'])

    expect(mockProjectService.create).toHaveBeenCalledWith('myapp', expect.any(Object))
  })

  it('should validate project name', async () => {
    const command = createCommand(mockProjectService)

    await expect(
      command.parseAsync(['node', 'test', 'Invalid Name'])
    ).rejects.toThrow()
  })
})
```

### Python (pytest)

**セットアップ**:
```bash
pip install pytest pytest-cov
```

**pytest.ini**:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = --strict-markers --strict-config -ra
```

**コマンドのテスト**:
```python
# tests/test_commands.py
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_create_command():
    """Test create command."""
    result = runner.invoke(app, ['create', 'project', 'myapp'])
    assert result.exit_code == 0
    assert 'Creating project: myapp' in result.stdout

def test_create_with_template():
    """Test create with template option."""
    result = runner.invoke(app, ['create', 'project', 'myapp', '--template', 'react'])
    assert result.exit_code == 0
    assert 'react' in result.stdout.lower()

def test_invalid_project_name():
    """Test invalid project name."""
    result = runner.invoke(app, ['create', 'project', 'Invalid Name'])
    assert result.exit_code != 0
    assert 'Invalid' in result.stdout
```

### Go (testing package)

**コマンドのテスト**:
```go
// cmd/create_test.go
package cmd

import (
	"bytes"
	"testing"

	"github.com/spf13/cobra"
)

func TestCreateCommand(t *testing.T) {
	// 出力をキャプチャ
	buf := new(bytes.Buffer)
	rootCmd.SetOut(buf)
	rootCmd.SetErr(buf)
	rootCmd.SetArgs([]string{"create", "myapp"})

	// コマンド実行
	err := rootCmd.Execute()
	if err != nil {
		t.Fatalf("Error executing command: %v", err)
	}

	// 出力を検証
	output := buf.String()
	if !bytes.Contains([]byte(output), []byte("Creating project: myapp")) {
		t.Errorf("Expected output to contain 'Creating project: myapp', got: %s", output)
	}
}

func TestCreateWithTemplate(t *testing.T) {
	buf := new(bytes.Buffer)
	rootCmd.SetOut(buf)
	rootCmd.SetErr(buf)
	rootCmd.SetArgs([]string{"create", "myapp", "--template", "react"})

	err := rootCmd.Execute()
	if err != nil {
		t.Fatalf("Error executing command: %v", err)
	}

	output := buf.String()
	if !bytes.Contains([]byte(output), []byte("react")) {
		t.Errorf("Expected output to contain 'react', got: %s", output)
	}
}
```

## 統合テスト

### Node.js

**実際の CLI を実行**:
```typescript
// tests/integration/cli.test.ts
import { exec } from 'child_process'
import { promisify } from 'util'
import fs from 'fs-extra'
import path from 'path'

const execAsync = promisify(exec)
const CLI_PATH = path.join(__dirname, '../../dist/index.js')
const TEST_DIR = path.join(__dirname, '../../test-projects')

describe('CLI integration tests', () => {
  beforeEach(async () => {
    await fs.ensureDir(TEST_DIR)
    process.chdir(TEST_DIR)
  })

  afterEach(async () => {
    await fs.remove(TEST_DIR)
  })

  it('should create a project', async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} create myapp --skip-install`)

    expect(stdout).toContain('Project created successfully')
    expect(await fs.pathExists('./myapp')).toBe(true)
  })

  it('should handle errors', async () => {
    try {
      await execAsync(`node ${CLI_PATH} create "Invalid Name"`)
      fail('Should have thrown an error')
    } catch (error: any) {
      expect(error.stderr).toContain('Project name must contain')
    }
  })
})
```

### Python

**実際の CLI を実行**:
```python
# tests/integration/test_cli.py
import subprocess
import tempfile
from pathlib import Path

def test_create_project():
    """Test project creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            ['mycli', 'create', 'project', 'myapp'],
            cwd=tmpdir,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert 'Creating project: myapp' in result.stdout
        assert Path(tmpdir, 'myapp').exists()

def test_invalid_input():
    """Test error handling."""
    result = subprocess.run(
        ['mycli', 'create', 'project', 'Invalid Name'],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0
    assert 'Invalid' in result.stdout
```

## E2Eテスト

### インタラクティブプロンプトのテスト

**Node.js (child_process)**:
```typescript
import { spawn } from 'child_process'

describe('Interactive prompts', () => {
  it('should handle interactive input', (done) => {
    const cli = spawn('node', ['dist/index.js', 'create'])

    let output = ''
    cli.stdout.on('data', (data) => {
      output += data.toString()

      // プロンプトに応答
      if (output.includes('Project name:')) {
        cli.stdin.write('myapp\n')
      } else if (output.includes('Template:')) {
        cli.stdin.write('\n') // デフォルト選択
      } else if (output.includes('Create project?')) {
        cli.stdin.write('y\n')
      }
    })

    cli.on('close', (code) => {
      expect(code).toBe(0)
      expect(output).toContain('Project created successfully')
      done()
    })
  })
})
```

**Python (pexpect)**:
```python
import pexpect

def test_interactive_prompts():
    """Test interactive prompts."""
    child = pexpect.spawn('mycli create project')

    # プロンプトに応答
    child.expect('Project name:')
    child.sendline('myapp')

    child.expect('Template:')
    child.sendline('')  # デフォルト選択

    child.expect('Create project?')
    child.sendline('y')

    child.expect('Project created successfully')
    child.close()
    assert child.exitstatus == 0
```

## スナップショットテスト

### ヘルプ出力のスナップショット

**Jest**:
```typescript
describe('Help output', () => {
  it('should match snapshot', () => {
    const program = createProgram()
    const help = program.helpInformation()

    expect(help).toMatchSnapshot()
  })
})
```

**pytest**:
```python
def test_help_output(snapshot):
    """Test help output matches snapshot."""
    result = runner.invoke(app, ['--help'])
    snapshot.assert_match(result.stdout, 'help.txt')
```

## カバレッジレポート

### Node.js

```bash
# カバレッジ生成
npm test -- --coverage

# HTML レポート
npm test -- --coverage --coverageReporters=html

# カバレッジチェック
npm test -- --coverage --coverageThreshold='{"global":{"branches":80,"functions":80,"lines":80}}'
```

### Python

```bash
# カバレッジ生成
pytest --cov=src tests/

# HTML レポート
pytest --cov=src --cov-report=html tests/

# カバレッジチェック
pytest --cov=src --cov-fail-under=80 tests/
```

## CI/CD 統合

### GitHub Actions

**テスト実行**:
```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
```

## テストのベストプラクティス

### 原則

1. **AAA パターン**: Arrange, Act, Assert
2. **独立性**: 各テストは独立して実行可能
3. **再現性**: 同じ結果が得られる
4. **速度**: ユニットテストは高速（< 100ms）
5. **明確性**: テスト名で何をテストしているか明確に

### 例

**Good**:
```typescript
describe('create command', () => {
  it('should create a project with default template', async () => {
    // Arrange
    const service = new ProjectService()

    // Act
    await service.create('myapp', { template: 'default' })

    // Assert
    expect(await fs.pathExists('./myapp')).toBe(true)
  })
})
```

**Bad**:
```typescript
it('test1', async () => {
  // 何をテストしているか不明
  await service.create('myapp')
  expect(true).toBe(true)
})
```

---

*包括的なテストで、堅牢な CLI ツールを構築しましょう。*
