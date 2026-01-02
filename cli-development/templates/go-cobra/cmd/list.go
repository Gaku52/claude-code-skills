package cmd

import (
	"fmt"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var listAll bool

var listCmd = &cobra.Command{
	Use:   "list",
	Short: "List all projects",
	Long:  `List all projects in the current directory.`,
	Run: func(cmd *cobra.Command, args []string) {
		listProjects()
	},
}

func init() {
	listCmd.Flags().BoolVarP(&listAll, "all", "a", false, "Show all projects")
}

type Project struct {
	Name     string
	Template string
	Created  string
	Size     string
}

func listProjects() {
	cyan := color.New(color.FgCyan).SprintFunc()
	green := color.New(color.FgGreen).SprintFunc()
	gray := color.New(color.FgHiBlack).SprintFunc()
	blue := color.New(color.FgBlue).SprintFunc()

	fmt.Printf("\n%s\n\n", cyan("📋 Projects:"))

	projects := []Project{
		{"myapp", "React", "2026-01-03", "10 MB"},
		{"api", "Node.js", "2026-01-02", "5 MB"},
		{"dashboard", "Vue", "2026-01-01", "8 MB"},
	}

	for _, p := range projects {
		fmt.Printf("  %s\n", color.New(color.Bold).Sprint(p.Name))
		fmt.Printf("    Template: %s\n", green(p.Template))
		fmt.Printf("    Created: %s\n", gray(p.Created))
		fmt.Printf("    Size: %s\n\n", blue(p.Size))
	}

	fmt.Printf("%s\n\n", gray(fmt.Sprintf("Total: %d project(s)", len(projects))))
}
