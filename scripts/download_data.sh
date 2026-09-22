#!/usr/bin/env bash
# Fetch the competition archive into data/raw/.
#
# You must first accept the rules at
#   https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules
# Downloads fail with a 403 until you do.
#
# Authenticate either way:
#   A. Environment variables (best for CI and remote containers):
#        export KAGGLE_USERNAME=your-kaggle-username
#        export KAGGLE_KEY=your-api-key
#   B. A token file at ~/.kaggle/kaggle.json
#        (Kaggle -> Settings -> API -> Create New Token)
set -euo pipefail

COMPETITION="soil-grain-size-from-photos"
DEST="${1:-data/raw}"

if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  echo "Authenticating as ${KAGGLE_USERNAME} via environment variables."
elif [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  echo "Authenticating via ${HOME}/.kaggle/kaggle.json."
else
  cat >&2 <<'MSG'
error: no Kaggle credentials found.

Set KAGGLE_USERNAME and KAGGLE_KEY in the environment, or save an API token to
~/.kaggle/kaggle.json. Create a token at https://www.kaggle.com/settings
(API -> Create New Token).
MSG
  exit 1
fi

command -v kaggle >/dev/null 2>&1 || python3 -m pip install --quiet kaggle

mkdir -p "${DEST}"
if ! kaggle competitions download -c "${COMPETITION}" -p "${DEST}"; then
  cat >&2 <<MSG

The download failed. The usual cause is not having accepted the competition
rules yet: open
  https://www.kaggle.com/competitions/${COMPETITION}/rules
click "I Understand and Accept", then run this script again.
MSG
  exit 1
fi

( cd "${DEST}" && unzip -o -q "${COMPETITION}.zip" && rm -f "${COMPETITION}.zip" )

echo "Downloaded to ${DEST}:"
ls -1 "${DEST}"
