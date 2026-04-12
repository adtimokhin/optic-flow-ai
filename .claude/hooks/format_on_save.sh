#!/bin/bash
# Auto-format Python files after Write or Edit tool calls.
# Receives a JSON payload on stdin with tool_name and tool_input.

INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('tool_input', {}).get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null)

if [[ "$FILE" == *.py ]] && [[ -f "$FILE" ]]; then
    black "$FILE" --quiet 2>/dev/null
    isort "$FILE" --quiet 2>/dev/null
fi
