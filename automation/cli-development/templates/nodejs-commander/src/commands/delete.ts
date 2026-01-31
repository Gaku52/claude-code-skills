import { Command } from 'commander'
import inquirer from 'inquirer'
import chalk from 'chalk'
import ora from 'ora'

interface DeleteOptions {
  force?: boolean
}

export function deleteCommand(): Command {
  return new Command('delete')
    .description('Delete a project')
    .argument('<name>', 'Project name')
    .option('-f, --force', 'Force delete without confirmation')
    .action(async (name: string, options: DeleteOptions) => {
      try {
        await deleteProject(name, options)
      } catch (error) {
        console.error(chalk.red('✗ Failed to delete project'))
        console.error(error)
        process.exit(1)
      }
    })
}

async function deleteProject(
  name: string,
  options: DeleteOptions
): Promise<void> {
  if (!options.force) {
    const { confirmed } = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'confirmed',
        message: `Delete project "${name}"?`,
        default: false
      }
    ])

    if (!confirmed) {
      console.log(chalk.yellow('Cancelled'))
      return
    }
  }

  const spinner = ora(`Deleting project: ${name}`).start()

  try {
    // シミュレーション
    await new Promise(resolve => setTimeout(resolve, 1000))

    spinner.succeed(chalk.green('Project deleted'))
  } catch (error) {
    spinner.fail(chalk.red('Failed to delete project'))
    throw error
  }
}
