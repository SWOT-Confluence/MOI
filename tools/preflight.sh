#!/usr/bin/env bash
# Check what the container image will actually contain, before building it.
#
# The Dockerfile copies ./moi, ./sos_read, ./run_MOI.py and the CalVal CSV from
# the build context.  If the working tree and the commit you are building from
# disagree, the image gets a half-applied change -- which is how commit 423d9d0
# shipped an Output.py calling flp_fit.as_row against a flp_fit.py that did not
# have it yet, and cost a whole basin its output with an AttributeError that no
# local test could have seen.
#
# So this checks the COMMITTED tree, not the working tree.
#
#   bash tools/preflight.sh          # check HEAD
#   bash tools/preflight.sh <ref>    # check some other commit
set -euo pipefail

REF="${1:-HEAD}"
cd "$(git rev-parse --show-toplevel)"

echo "=== working tree vs $REF ==="
DIRTY="$(git status --porcelain -- moi run_MOI.py requirements.txt sos_read || true)"
if [ -n "$DIRTY" ]; then
    echo "UNCOMMITTED changes in files the image copies:"
    echo "$DIRTY" | sed 's/^/    /'
    echo
    echo "The image will NOT contain these. Commit them, or build from the"
    echo "working tree deliberately. Continuing to check $REF as it stands."
    echo
else
    echo "clean"
    echo
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git archive "$REF" | tar -x -C "$TMP"
echo "=== extracted $REF to a temp tree ==="

cd "$TMP"

echo "=== byte-compile ==="
python3 -m compileall -q moi run_MOI.py >/dev/null
echo "ok"

echo "=== unresolved cross-module attributes ==="
# Catches exactly the 423d9d0 failure: module X calls Y.thing() where the
# committed Y has no 'thing'.  Static, so it needs no netCDF4 or scipy.
python3 - <<'PY'
import ast, pathlib, sys

files = {p.stem: p for p in pathlib.Path('moi').glob('*.py')}
trees = {name: ast.parse(p.read_text()) for name, p in files.items()}
defined = {
    name: {n.name for n in ast.walk(t)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
          | {t2.id for n in ast.walk(t) if isinstance(n, ast.Assign)
             for t2 in n.targets if isinstance(t2, ast.Name)}
    for name, t in trees.items()
}

problems = []
for name, tree in trees.items():
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in defined
                and node.value.id != name
                and not node.attr.startswith('__')
                and node.attr not in defined[node.value.id]):
            problems.append('moi/%s.py:%d  %s.%s does not exist in moi/%s.py'
                            % (name, node.lineno, node.value.id, node.attr,
                               node.value.id))

if problems:
    print('\n'.join('  ' + p for p in sorted(set(problems))))
    sys.exit(1)
print('ok')
PY

echo "=== tests that need no netCDF4 ==="
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest -q tests/test_flp_fit.py tests/test_compute_flps.py
else
    echo "pytest not installed here; skipping. Run the full suite where it is."
fi

echo
echo "preflight OK for $REF"
