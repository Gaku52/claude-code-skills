import { Command } from 'commander'
import inquirer from 'inquirer'
import chalk from 'chalk'
import ora from 'ora'

interface CreateOptions {
  template?: string
  typescript?: boolean
  skipInstall?: boolean
}

export function createCommand(): Command {
  return new Command('create')
    .description('Create a new project')
    .argument('[name]', 'Project name')
    .option('-t, --template <template>', 'Template to use', 'default')
    .option('--typescript', 'Use TypeScript', true)
    .option('--skip-install', 'Skip npm install')
    .action(async (name: string | undefined, options: CreateOptions) => {
      try {
        await createProject(name, options)
      } catch (error) {
        console.error(chalk.red('✗ Failed to create project'))
        console.error(error)
        process.exit(1)
      }
    })
}

async function createProject(
  name: string | undefined,
  options: CreateOptions
): Promise<void> {
  // プロジェクト名の取得
  let projectName = name
  if (!projectName) {
    const { name: promptedName } = await inquirer.prompt([
      {
        type: 'input',
        name: 'name',
        message: 'Project name:',
        default: 'my-project',
        validate: (input: string) => {
          if (!/^[a-z0-9-]+$/.test(input)) {
            return 'Project name must contain only lowercase letters, numbers, and hyphens'
          }
          return true
        }
      }
    ])
    projectName = promptedName
  }

  console.log(chalk.cyan(`\n📦 Creating project: ${projectName}\n`))

  // 設定の確認
  console.log(chalk.gray('Configuration:'))
  console.log(`  Template: ${chalk.green(options.template)}`)
  console.log(`  TypeScript: ${chalk.green(options.typescript ? 'Yes' : 'No')}`)

  // プロジェクト作成
  const spinner = ora('Setting up project...').start()

  try {
    // シミュレーション
    await new Promise(resolve => setTimeout(resolve, 1000))
    spinner.text = 'Copying template files...'
    await new Promise(resolve => setTimeout(resolve, 1000))

    if (!options.skipInstall) {
      spinner.text = 'Installing dependencies...'
      await new Promise(resolve => setTimeout(resolve, 2000))
    }

    spinner.succeed(chalk.green('Project created successfully!'))

    // 次のステップ
    console.log(chalk.cyan('\nNext steps:'))
    console.log(`  cd ${projectName}`)
    if (options.skipInstall) {
      console.log('  npm install')
    }
    console.log('  npm run dev')
  } catch (error) {
    spinner.fail(chalk.red('Failed to create project'))
    throw error
  }
}
