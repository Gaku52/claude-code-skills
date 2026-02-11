package cmd

import (
	"fmt"
	"time"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var (
	template  string
	skipInstall bool
)

var createCmd = &cobra.Command{
	Use:   "create [name]",
	Short: "Create a new project",
	Long: `Create a new project from a template.

Example:
  mycli create myapp
  mycli create myapp --template react`,
	Args: cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		name := args[0]
		createProject(name)
	},
}

func init() {
	createCmd.Flags().StringVarP(&template, "template", "t", "default", "Template to use")
	createCmd.Flags().BoolVar(&skipInstall, "skip-install", false, "Skip npm install")
}

func createProject(name string) {
	cyan := color.New(color.FgCyan).SprintFunc()
	green := color.New(color.FgGreen).SprintFunc()

	fmt.Printf("\n📦 Creating project: %s\n\n", cyan(name))
	fmt.Printf("  Template: %s\n", green(template))

	// プロジェクト作成シミュレーション
	steps := []string{
		"Creating directory structure",
		"Copying template files",
		"Configuring project",
	}

	if !skipInstall {
		steps = append(steps, "Installing dependencies")
	}

	for _, step := range steps {
		fmt.Printf("  %s...\n", step)
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Printf("\n%s Project created successfully!\n\n", green("✓"))
	fmt.Printf("%s\n", cyan("Next steps:"))
	fmt.Printf("  cd %s\n", name)
	if skipInstall {
		fmt.Println("  npm install")
	}
	fmt.Println("  npm run dev\n")
}
