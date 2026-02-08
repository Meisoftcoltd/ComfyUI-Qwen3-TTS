import sys
import os
import ast

def check_syntax(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        ast.parse(source)
        print(f"Syntax OK: {filepath}")
    except SyntaxError as e:
        print(f"Syntax Error in {filepath}: {e}")
        sys.exit(1)

check_syntax("nodes.py")
