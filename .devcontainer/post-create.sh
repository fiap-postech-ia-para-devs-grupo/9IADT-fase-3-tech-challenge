#!/usr/bin/env bash

set -e

bash .devcontainer/scripts/configure-git.sh

uv sync --frozen

uv run python -m ipykernel install --user --name hospital-assistant-fase3 --display-name "Python (hospital-assistant-fase3)"

# Create .claude/skills symlink pointing to .agents/skills
workspace_dir="$(pwd -P)"
mkdir -p "$workspace_dir/.agents/skills"
mkdir -p "$workspace_dir/.claude"
ln -sf "$workspace_dir/.agents/skills" "$workspace_dir/.claude/skills"
