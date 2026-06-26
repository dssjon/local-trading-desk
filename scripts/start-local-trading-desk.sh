#!/bin/zsh
set -eu

script_dir="${0:A:h}"
project_dir="${TRADINGAGENTS_PROJECT_DIR:-${script_dir:h}}"
cd "$project_dir"

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export TRADINGAGENTS_WEB_HOST="${TRADINGAGENTS_WEB_HOST:-0.0.0.0}"
export TRADINGAGENTS_WEB_PORT="${TRADINGAGENTS_WEB_PORT:-8501}"

exec .venv/bin/tradingagents-web
