import { exec } from 'child_process'
import { promisify } from 'util'

const execAsync = promisify(exec)

describe('CLI Commands', () => {
  it('should show help', async () => {
    const { stdout } = await execAsync('ts-node src/index.ts --help')
    expect(stdout).toContain('Usage:')
    expect(stdout).toContain('mycli')
  })

  it('should show version', async () => {
    const { stdout } = await execAsync('ts-node src/index.ts --version')
    expect(stdout).toMatch(/\d+\.\d+\.\d+/)
  })

  it('should list projects', async () => {
    const { stdout } = await execAsync('ts-node src/index.ts list')
    expect(stdout).toContain('Projects:')
  })
})
