package cmd

import (
	"fmt"
	"time"

	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var force bool

var deleteCmd = &cobra.Command{
	Use:   "delete [name]",
	Short: "Delete a project",
	Long:  `Delete a project from the filesystem.`,
	Args:  cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		name := args[0]
		deleteProject(name)
	},
}

func init() {
	deleteCmd.Flags().BoolVarP(&force, "force", "f", false, "Force delete without confirmation")
}

func deleteProject(name string) {
	red := color.New(color.FgRed).SprintFunc()
	green := color.New(color.FgGreen).SprintFunc()
	yellow := color.New(color.FgYellow).SprintFunc()

	if !force {
		fmt.Printf("Delete project \"%s\"? [y/N] ", name)
		var response string
		fmt.Scanln(&response)

		if response != "y" && response != "Y" {
			fmt.Printf("%s\n", yellow("Cancelled"))
			return
		}
	}

	fmt.Printf("%s\n", red(fmt.Sprintf("Deleting project: %s", name)))
	time.Sleep(1 * time.Second)
	fmt.Printf("%s\n", green("✓ Project deleted"))
}
