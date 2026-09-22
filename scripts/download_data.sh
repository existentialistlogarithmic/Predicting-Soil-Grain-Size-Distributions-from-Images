#!/usr/bin/env bash
# Fetch the competition archive into data/raw/.
#
# Prerequisites:
#   1. A Kaggle account that has accepted the competition rules at
#      https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules
#   2. An API token saved to ~/.kaggle/kaggle.json (Kaggle -> Settings -> API ->
#      Create New Token), with permissions set to 600.
set -euo pipefail

COMPETITION="soil-grain-size-from-photos"
DEST="${1:-data/raw}"

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "error: ${HOME}/.kaggle/kaggle.json not found." >&2
  echo "Create an API token at https://www.kaggle.com/settings and save it there." >&2
  exit 1
fi
chmod 600 "${HOME}/.kaggle/kaggle.json"

command -v kaggle >/dev/null 2>&1 || python3 -m pip install --quiet kaggle

mkdir -p "${DEST}"
kaggle competitions download -c "${COMPETITION}" -p "${DEST}"
( cd "${DEST}" && unzip -o -q "${COMPETITION}.zip" && rm -f "${COMPETITION}.zip" )

echo "Downloaded to ${DEST}:"
ls -1 "${DEST}"
