#!/bin/bash
#
# Prepares LJSpeech for examples/tacotron2/train_tacotron2.py with this project's own two-step
# preprocessing, the README's "Preprocessing" section for `ljspeech`:
#
#   tensorflow-tts-preprocess --rootdir ./ljspeech --outdir ./dump_ljspeech --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech
#   tensorflow-tts-normalize  --rootdir ./dump_ljspeech --outdir ./dump_ljspeech --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech
#
# Those two commands are setup.py's console scripts for tensorflow_tts.bin.preprocess:preprocess and
# :normalize. The package is not installed here (setup.py pins tensorflow-gpu 2.7.0, not this
# environment's TensorFlow), so each function is called exactly as its console script would call it.
# Run from the project root ($PYTHON may name the interpreter to use).
#
# Input: LJSpeech-1.1.tar.bz2 from https://keithito.com/LJ-Speech-Dataset/ in $LJSPEECH_DIR.
# Outputs, the data under $LJSPEECH_DIR and none of it in this checkout:
#   LJSpeech-1.1/          the archive, extracted (metadata.csv, wavs/)
#   dump_ljspeech/         the two steps' output: train/ and valid/ (ids, raw-feats, norm-feats, ...),
#                          stats*.npy, *_utt_ids.npy, ljspeech_mapper.json
#   ./dump_ljspeech/train, ./dump_ljspeech/valid in this checkout: links to the two splits, at the
#                          path the README's training command reads. ./dump_ljspeech/ is a real
#                          directory, so this project's .gitignore entry `dump_ljspeech/` covers it
#                          (a link in its place would not be ignored).
#   tacotron2-config.yml   examples/tacotron2/conf/tacotron2.v1.yaml with its training-time writes
#                          switched off: no checkpoint and no TensorBoard training log (both
#                          intervals past any run's length). Evaluation and its intermediate
#                          results are left as they are, since evaluation also predicts.
#   ./dump_ljspeech/tacotron2-config.yml in this checkout: a link to it, so the training command can
#                          name it by a relative path (a runner passes `~` through unexpanded, and
#                          the script opens --config as given).
# Idempotent: an extracted archive and a completed dump are kept and not redone. The dump is not
# byte-reproducible (preprocess fits its scalers in the order a process pool returns results), which
# is one more reason it is generated once and kept.

set -euo pipefail

PYTHON="${PYTHON:-python3.10}"
LJSPEECH_DIR="${LJSPEECH_DIR:-$HOME/.cache/python-subjects/assets/ljspeech}"
root="$LJSPEECH_DIR/LJSpeech-1.1"
dump="$LJSPEECH_DIR/dump_ljspeech"

if [[ ! -e $LJSPEECH_DIR/.extracted ]]; then
	tar -xjf "$LJSPEECH_DIR/LJSpeech-1.1.tar.bz2" -C "$LJSPEECH_DIR"
	touch "$LJSPEECH_DIR/.extracted"
fi

if [[ ! -e $dump/.complete ]]; then
	# preprocess imports pyworld (requirements.txt); fail clearly if it is not installed.
	"$PYTHON" -c 'import pyworld' 2>/dev/null || {
		echo "setup.sh: preprocessing needs pyworld, which this interpreter cannot import ($PYTHON)." >&2
		exit 1; }
	# Built beside its final name and renamed, so an interrupted run leaves nothing that reads as done.
	rm -rf "$dump.partial"
	for step in preprocess normalize; do
		[[ $step == preprocess ]] && in="$root" || in="$dump.partial"
		"$PYTHON" -c "import sys; sys.argv[0] = 'tensorflow-tts-$step'
from tensorflow_tts.bin.preprocess import $step
sys.exit($step())" \
			--rootdir "$in" --outdir "$dump.partial" --config preprocess/ljspeech_preprocess.yaml --dataset ljspeech
	done
	touch "$dump.partial/.complete"
	rm -rf "$dump"
	mv "$dump.partial" "$dump"
fi

mkdir -p dump_ljspeech
for split in train valid; do
	if [[ -e dump_ljspeech/$split && ! -L dump_ljspeech/$split ]]; then
		echo "setup.sh: ./dump_ljspeech/$split is not a link to $dump/$split. Move it aside, then re-run." >&2
		exit 1
	fi
	ln -sfn "$dump/$split" "dump_ljspeech/$split"
done

# Regenerated every time from the checked-out config, so it never goes stale against it. A textual
# substitution, not a YAML round trip, so the rest of the file stays byte for byte.
config="$LJSPEECH_DIR/tacotron2-config.yml"
sed -e "s|^save_interval_steps: 2000 |save_interval_steps: 1000000000 |" \
	-e "s|^log_interval_steps: 200 |log_interval_steps: 1000000000 |" \
	examples/tacotron2/conf/tacotron2.v1.yaml > "$config.partial"
if ! grep -q '^save_interval_steps: 1000000000 ' "$config.partial" ||
	! grep -q '^log_interval_steps: 1000000000 ' "$config.partial"; then
	echo "setup.sh: examples/tacotron2/conf/tacotron2.v1.yaml has a training-time write this script does not switch off." >&2
	exit 1
fi
mv "$config.partial" "$config"
if [[ -e dump_ljspeech/tacotron2-config.yml && ! -L dump_ljspeech/tacotron2-config.yml ]]; then
	echo "setup.sh: ./dump_ljspeech/tacotron2-config.yml is not a link to $config. Move it aside, then re-run." >&2
	exit 1
fi
ln -sfn "$config" dump_ljspeech/tacotron2-config.yml
