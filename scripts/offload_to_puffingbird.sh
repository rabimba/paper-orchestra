#!/usr/bin/env bash
# ==============================================================================
# PaperOrchestra Remote Offloading Runner for Puffingbird (Dual H100 NVL GPUs)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REMOTE_HOST="${PUFFINGBIRD_HOST:-172.23.79.37}"
REMOTE_USER="${PUFFINGBIRD_USER:-rabimba}"
REMOTE_DIR="${PUFFINGBIRD_PROJECT_DIR:-/data/rabimba/projects/paper-orchestra}"
REMOTE_ENV="${PUFFINGBIRD_ENV:-/data/rabimba/envs/paper-orchestra}"
REMOTE_STORAGE="/data/rabimba"

EXP_ID="EXP-$(date +%Y%m%d_%H%M%S)"
MODE="detached" # "detached" or "wait"
RAW_MATERIALS_DIR="datasets/examples/tsam_cvpr2025/raw_materials"
LATEX_TEMPLATE_DIR="templates/cvpr2025"
OUTPUT_DIR="paper_output_${EXP_ID}"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra args for paper_writing_cli.py]

Options:
  --raw_materials_dir <dir>   Path to raw materials (default: ${RAW_MATERIALS_DIR})
  --latex_template_dir <dir>  Path to LaTeX template (default: ${LATEX_TEMPLATE_DIR})
  --output_dir <dir>          Path for generated paper output (default: ${OUTPUT_DIR})
  --exp_id <id>               Custom experiment session ID (default: ${EXP_ID})
  --wait                      Block, stream log, and sync back generated output when complete
  --status                    Check active jobs and GPU utilization on puffingbird
  --tail                      Tail logs of the most recent remote run
  --sync-back <output_dir>    Manually pull remote output directory to local machine
  -h, --help                  Show this help message

Examples:
  # Launch default sample CVPR 2025 paper offloaded to puffingbird:
  $0

  # Launch with custom materials and wait for output:
  $0 --raw_materials_dir my_project/raw_materials --wait

  # Check active tmux sessions and GPUs on puffingbird:
  $0 --status
EOF
    exit 0
}

# Pre-parse command options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw_materials_dir)
            RAW_MATERIALS_DIR="$2"
            shift 2
            ;;
        --latex_template_dir)
            LATEX_TEMPLATE_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --wait)
            MODE="wait"
            shift
            ;;
        --status)
            echo "=== Remote Job Status on puffingbird ==="
            ssh "${REMOTE_USER}@${REMOTE_HOST}" "
                echo '[Active tmux sessions]:'
                tmux list-sessions 2>/dev/null || echo 'No active tmux sessions.'
                echo ''
                echo '[NVIDIA H100 GPU Status]:'
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
            "
            exit 0
            ;;
        --tail)
            ssh "${REMOTE_USER}@${REMOTE_HOST}" "
                LATEST_LOG=\$(ls -td ${REMOTE_DIR}/runs/*/run.log 2>/dev/null | head -n 1)
                if [ -n \"\${LATEST_LOG}\" ]; then
                    echo \"Tailing \${LATEST_LOG}:\"
                    tail -f -n 50 \"\${LATEST_LOG}\"
                else
                    echo \"No run logs found in ${REMOTE_DIR}/runs/\"
                fi
            "
            exit 0
            ;;
        --sync-back)
            TARGET_SYNC_DIR="${2:-}"
            if [[ -z "${TARGET_SYNC_DIR}" ]]; then
                echo "Error: target output dir required for --sync-back" >&2
                exit 1
            fi
            echo "Syncing back '${TARGET_SYNC_DIR}' from puffingbird..."
            rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${TARGET_SYNC_DIR}" "${REPO_ROOT}/"
            echo "Sync complete."
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "======================================================================"
echo "🎻 PaperOrchestra Remote Offload -> Puffingbird"
echo "Host: ${REMOTE_HOST} | GPUs: 2x NVIDIA H100 NVL (188GB VRAM)"
echo "Experiment ID: ${EXP_ID}"
echo "Storage Root:  ${REMOTE_DIR} (8TB Enterprise SSD)"
echo "======================================================================"

# 1. Connection check
echo "[1/6] Verifying connection to puffingbird..."
ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_USER}@${REMOTE_HOST}" "true" || {
    echo "Error: Unable to connect to ${REMOTE_USER}@${REMOTE_HOST} over SSH." >&2
    exit 1
}

# 2. Sync codebase & raw materials to Puffingbird 8TB drive
echo "[2/6] Syncing code, templates, and materials to puffingbird..."
rsync -avz --delete \
    --exclude '.git' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude 'runs' \
    --exclude 'paper_output*' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    "${REPO_ROOT}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# Copy .env if present locally
if [[ -f "${REPO_ROOT}/.env" ]]; then
    rsync -avz "${REPO_ROOT}/.env" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/.env"
fi

# 3. Update RESEARCH_DIARY.md locally and remotely
echo "[3/6] Logging experiment ${EXP_ID} in RESEARCH_DIARY.md..."
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
GIT_HASH=$(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo "uncommitted")

DIARY_ENTRY="
| \`${EXP_ID}\` | $(date +'%Y-%m-%d %H:%M') | \`${GIT_HASH}\` | Automated Paper Generation (${RAW_MATERIALS_DIR}) | Running | Output: ${OUTPUT_DIR} |
"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    mkdir -p '${REMOTE_DIR}/runs/${EXP_ID}'
    # Prepend to diary if index exists
    if grep -q '## Experiment Index' '${REMOTE_DIR}/RESEARCH_DIARY.md' 2>/dev/null; then
        awk -v entry=\"${DIARY_ENTRY}\" '/## Experiment Index/{print; print \"\"; print entry; next}1' '${REMOTE_DIR}/RESEARCH_DIARY.md' > '${REMOTE_DIR}/RESEARCH_DIARY.tmp' && mv '${REMOTE_DIR}/RESEARCH_DIARY.tmp' '${REMOTE_DIR}/RESEARCH_DIARY.md'
    fi
"

# 4. Construct execution command
REMOTE_CMD="
export HF_HOME='${REMOTE_STORAGE}/.cache/huggingface'
export TORCH_HOME='${REMOTE_STORAGE}/.cache/torch'
export TMPDIR='${REMOTE_STORAGE}/tmp'
export PATH='${REMOTE_STORAGE}/tinytex/.TinyTeX/bin/x86_64-linux:${REMOTE_ENV}/bin:\$PATH'
export PYTHONPATH='${REMOTE_DIR}'

mkdir -p \"\${TMPDIR}\"
cd '${REMOTE_DIR}'

echo '=== PaperOrchestra Job Started: ${EXP_ID} ===' > 'runs/${EXP_ID}/run.log'
date >> 'runs/${EXP_ID}/run.log'

'${REMOTE_ENV}/bin/python' paper_writing_cli.py \\
    --raw_materials_dir '${RAW_MATERIALS_DIR}' \\
    --latex_template_dir '${LATEX_TEMPLATE_DIR}' \\
    --output_dir '${OUTPUT_DIR}' \\
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \\
    >> 'runs/${EXP_ID}/run.log' 2>&1

EXIT_CODE=\$?
echo \"=== Job Finished with exit code \${EXIT_CODE} ===\" >> 'runs/${EXP_ID}/run.log'

# Update diary status
if [ \${EXIT_CODE} -eq 0 ]; then
    sed -i 's/${EXP_ID}.*Running/${EXP_ID} | $(date +'%Y-%m-%d %H:%M') | \`${GIT_HASH}\` | Automated Paper Generation | Completed/' '${REMOTE_DIR}/RESEARCH_DIARY.md' || true
else
    sed -i 's/${EXP_ID}.*Running/${EXP_ID} | $(date +'%Y-%m-%d %H:%M') | \`${GIT_HASH}\` | Automated Paper Generation | Failed (code '\${EXIT_CODE}')/' '${REMOTE_DIR}/RESEARCH_DIARY.md' || true
fi

# Continuous git sync as per skill
git add . && git commit -m 'exp(${EXP_ID}): run status update' && git push origin main || true
"

# 5. Launch detached tmux session on Puffingbird
echo "[4/6] Launching compute job in tmux session 'orchestra_${EXP_ID}' on puffingbird..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    tmux kill-session -t 'orchestra_${EXP_ID}' 2>/dev/null || true
    tmux new-session -d -s 'orchestra_${EXP_ID}' -c '${REMOTE_DIR}' bash -c '$(printf "%q" "${REMOTE_CMD}")'
"

echo "[5/6] Job successfully dispatched to puffingbird!"
echo "   Session Name: orchestra_${EXP_ID}"
echo "   Remote Log:   ${REMOTE_DIR}/runs/${EXP_ID}/run.log"
echo "   Monitor with: $0 --tail"
echo "   Status with:  $0 --status"

# 6. If wait mode requested, stream logs and sync back results
if [[ "${MODE}" == "wait" ]]; then
    echo "[6/6] Waiting for execution to complete (streaming logs)..."
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "
        tail -f '${REMOTE_DIR}/runs/${EXP_ID}/run.log' | while read -r line; do
            echo \"\${line}\"
            if [[ \"\${line}\" == *\"=== Job Finished with exit code\"* ]]; then
                break
            fi
        done
    "
    echo "Syncing back generated output '${OUTPUT_DIR}' from puffingbird..."
    rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/${OUTPUT_DIR}" "${REPO_ROOT}/"
    echo "Paper results synced back to: ${REPO_ROOT}/${OUTPUT_DIR}"
else
    echo "[6/6] Run dispatched in background. To sync results back upon completion, run:"
    echo "   $0 --sync-back ${OUTPUT_DIR}"
fi

echo "Done."
