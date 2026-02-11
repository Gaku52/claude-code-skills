import { Command } from 'commander'
import chalk from 'chalk'

interface ListOptions {
  all?: boolean
}

export function listCommand(): Command {
  return new Command('list')
    .description('List all projects')
    .option('-a, --all', 'Show all projects')
    .action(async (options: ListOptions) => {
      await listProjects(options)
    })
}

async function listProjects(options: ListOptions): Promise<void> {
  console.log(chalk.cyan('\n📋 Projects:\n'))

  // サンプルデータ
  const projects = [
    { name: 'myapp', template: 'React', created: '2026-01-03', size: '10 MB' },
    { name: 'api', template: 'Node.js', created: '2026-01-02', size: '5 MB' },
    { name: 'dashboard', template: 'Vue', created: '2026-01-01', size: '8 MB' }
  ]

  projects.forEach(project => {
    console.log(`  ${chalk.bold(project.name)}`)
    console.log(`    Template: ${chalk.green(project.template)}`)
    console.log(`    Created: ${chalk.gray(project.created)}`)
    console.log(`    Size: ${chalk.blue(project.size)}`)
    console.log()
  })

  console.log(chalk.gray(`Total: ${projects.length} project(s)\n`))
}
