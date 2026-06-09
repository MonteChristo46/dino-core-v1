#!/usr/bin/env bash
set -euo pipefail

# export_triton.sh — Export trained PatchCore models to Triton Inference Server format.
#
# Usage:
#   ./scripts/export_triton.sh <category> [category2 ...]
#   ./scripts/export_triton.sh mvtec_bottle
#   ./scripts/export_triton.sh mvtec_bottle mvtec_cable
#   ./scripts/export_triton.sh mvtec_bottle --image-size 224
#   ./scripts/export_triton.sh mvtec_bottle --no-clean --output /tmp/repo
#
# Searches results/models/*/models/<category>/ for the latest trained model.
# After export, prompts to start docker compose.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="model_repository"
IMAGE_SIZE=384

usage() {
    echo "Usage: $0 <category> [category2 ...]"
    echo ""
    echo "Export trained PatchCore models to Triton format."
    echo "Searches results/models/*/models/<category>/ for the latest model."
    echo ""
    echo "Options:"
    echo "  --output DIR       Output directory (default: model_repository)"
    echo "  --image-size SIZE  Input image size (default: 384)"
    echo "  --no-clean         Keep existing model_repository (default: always clean)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 mvtec_bottle"
    echo "  $0 mvtec_bottle mvtec_cable --image-size 224"
    echo "  $0 mvtec_bottle --no-clean --output /tmp/repo"
    echo ""
    echo "Available categories:"
    if [[ -d "$PROJECT_DIR/results/models" ]]; then
        for config_dir in "$PROJECT_DIR/results/models"/*/; do
            config="$(basename "$config_dir")"
            if [[ -d "$config_dir/models" ]]; then
                for cat_dir in "$config_dir/models"/*/; do
                    cat="$(basename "$cat_dir")"
                    echo "  $cat  (in $config)"
                done
            fi
        done | sort -u
    else
        echo "  (no results/models/ directory — train models first)"
    fi
    exit 0
}

find_model_dir() {
    local category="$1"
    local best_dir=""
    local best_time=0

    # Search all config dirs for this category
    for config_dir in "$PROJECT_DIR/results/models"/*/; do
        local candidate="$config_dir/models/$category"
        if [[ -d "$candidate" ]]; then
            # Check it has the required files
            if [[ -f "$candidate/patchcore_params.pkl" || -f "$candidate"/*.faiss ]]; then
                local dir_time
                dir_time=$(stat -c %Y "$candidate" 2>/dev/null || echo 0)
                if (( dir_time > best_time )); then
                    best_time=$dir_time
                    best_dir="$candidate"
                fi
            fi
        fi
    done

    echo "$best_dir"
}

NO_CLEAN=false
MODEL_DIRS=()
CATEGORIES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --no-clean)
            NO_CLEAN=true
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            CATEGORIES+=("$1")
            shift
            ;;
    esac
done

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
    echo "Error: No categories specified."
    echo ""
    usage
fi

# Resolve categories to actual model directories
MISSING=()
for cat in "${CATEGORIES[@]}"; do
    resolved="$(find_model_dir "$cat")"
    if [[ -z "$resolved" ]]; then
        MISSING+=("$cat")
    else
        MODEL_DIRS+=("$resolved")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Error: Model not found for: ${MISSING[*]}"
    echo ""
    echo "Run with --help to see available categories."
    exit 1
fi

# Clean existing output by default
if [[ "$NO_CLEAN" == false && -d "$PROJECT_DIR/$OUTPUT_DIR" ]]; then
    echo "Cleaning existing $OUTPUT_DIR..."
    rm -rf "$PROJECT_DIR/$OUTPUT_DIR"
fi

echo "============================================================"
echo "Triton Export"
echo "============================================================"
echo "Categories:    ${CATEGORIES[*]}"
echo "Output:        $PROJECT_DIR/$OUTPUT_DIR"
echo "Image size:    $IMAGE_SIZE"
echo ""
for i in "${!CATEGORIES[@]}"; do
    echo "  ${CATEGORIES[$i]} → ${MODEL_DIRS[$i]}"
done
echo "============================================================"
echo ""

# Run the export script
cd "$PROJECT_DIR"
uv run python scripts/export_for_triton.py \
    --model-dir "${MODEL_DIRS[@]}" \
    --output "$OUTPUT_DIR" \
    --categories "${CATEGORIES[@]}" \
    --image-size "$IMAGE_SIZE"

echo ""
echo "============================================================"
echo "Start Triton server?"
echo "============================================================"
read -p "  Run docker compose up --build? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker compose up --build
fi
