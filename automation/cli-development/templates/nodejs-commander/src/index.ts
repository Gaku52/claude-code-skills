#!/usr/bin/env node

import { Command } from 'commander'
import { createCommand } from './commands/create'
import { listCommand } from './commands/list'
import { deleteCommand } from './commands/delete'
import { version } from '../package.json'

const program = new Command()

program
  .name('mycli')
  .description('A powerful CLI tool for project management')
  .version(version)

// グローバルオプション
program
  .option('-v, --verbose', 'Enable verbose output')
  .option('--no-color', 'Disable color output')

// コマンド登録
program.addCommand(createCommand())
program.addCommand(listCommand())
program.addCommand(deleteCommand())

// ヘルプのカスタマイズ
program.addHelpText('after', `

Examples:
  $ mycli create myapp
  $ mycli create myapp --template react
  $ mycli list
  $ mycli delete myapp --force

Documentation: https://github.com/username/mycli
`)

program.parse()
