#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run `video_to_slomo.py` following README instructions.
# Behavior:
#  - Usage: ./sslomo.sh input.mp4 [sf] [fps] [checkpoint]
#  - If checkpoint is not provided, the script will try to download the
#    pretrained checkpoint (Google Drive id from README) into ./checkpoints/
#    using `gdown` if available. If not available, it prints instructions.
#  - Calls: python video_to_slomo.py --video <input> --sf <sf> --output <out> [--fps <fps>] [--checkpoint <ckpt>]
#
usage() {
	cat <<EOF
Usage: $0 input_video [sf] [fps] [checkpoint]

	input_video   Path to source video (required)
	sf            Slow-factor (optional). Defaults to 4
	fps           Output FPS (optional). If omitted, video_to_slomo.py will infer/default.
	checkpoint    Path to checkpoint file (optional). If omitted, script attempts to download the pretrained checkpoint.

Example:
	$0 input.mp4          # creates input_slomo.mp4 with sf=4
	$0 input.mp4 3 90     # creates input_slomo.mp4 with sf=3 and fps=90
	$0 input.mp4 4 120 ckpt/SuperSloMo.ckpt

Note: You can also set environment variable CHECKPOINT to point to a checkpoint file.
EOF
	exit 1
}

if [ "$#" -lt 1 ]; then
	usage
fi

INPUT="$1"
SF="${2:-4}"
FPS="${3:-}"
# precedence: explicit fourth arg -> env CHECKPOINT -> empty
CHECKPOINT="${4:-${CHECKPOINT:-}}"
OUTPUT="${INPUT%.*}_slomo.mkv"

PY_CMD="python"

if ! command -v "$PY_CMD" >/dev/null 2>&1; then
	echo "Error: '$PY_CMD' not found in PATH. Install Python or update PY_CMD in this script." >&2
	exit 2
fi

if [ ! -f "$INPUT" ]; then
	echo "Error: input file '$INPUT' not found." >&2
	exit 3
fi

# If no checkpoint provided, try to download the pretrained model referenced in README
if [ -z "$CHECKPOINT" ]; then
	mkdir -p checkpoints
	DEFAULT_CKPT="./checkpoints/SuperSloMo.ckpt"
	if [ -f "$DEFAULT_CKPT" ]; then
		CHECKPOINT="$DEFAULT_CKPT"
	else
		echo "No checkpoint specified. Attempting to download pretrained checkpoint into ./checkpoints/"
		# Google Drive id from README
		GD_ID="1IvobLDbRiBgZr3ryCRrWL8xDbMZ-KnpF"
		if command -v gdown >/dev/null 2>&1; then
			echo "Downloading pretrained checkpoint with gdown..."
			gdown --id "$GD_ID" -O "$DEFAULT_CKPT" || true
			if [ -f "$DEFAULT_CKPT" ]; then
				CHECKPOINT="$DEFAULT_CKPT"
				echo "Downloaded checkpoint to $DEFAULT_CKPT"
			else
				echo "gdown failed to download the checkpoint. Please download it manually from:"
				echo "  https://drive.google.com/open?id=$GD_ID"
				echo "Or install gdown (pip install gdown) and re-run this script."
			fi
		else
			echo "gdown not found. To auto-download the checkpoint, install gdown (pip install gdown)."
			echo "Pretrained checkpoint URL: https://drive.google.com/open?id=$GD_ID"
			echo "Proceeding without checkpoint. You can set CHECKPOINT env var or pass a path as 4th argument."
		fi
	fi
fi

CMD=("$PY_CMD" "video_to_slomo.py" --video "$INPUT" --sf "$SF" --output "$OUTPUT")
if [ -n "$FPS" ]; then
	CMD+=(--fps "$FPS")
fi
if [ -n "$CHECKPOINT" ]; then
	CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "Converting '$INPUT' -> '$OUTPUT' with sf=$SF${FPS:+, fps=$FPS}${CHECKPOINT:+, checkpoint=$CHECKPOINT}"
echo "Running: ${CMD[*]}"

exec "${CMD[@]}"

