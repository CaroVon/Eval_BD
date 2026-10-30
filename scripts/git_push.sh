#!/bin/bash
# Usage: git_push.sh <commit-message-file>
set -e
cd ~/myproject/ICML
[ -f gitignore ] && mv -f gitignore .gitignore
bash scripts/collect_results.sh
git add .gitignore README.md DECISIONS.md RESULTS.md src scripts results configs
git status --short | head -60
git commit -q -F "$1"
git push origin main 2>&1 | tail -3
git log --oneline -3
