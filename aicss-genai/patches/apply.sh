#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-.}"
BASE_COMMIT="45cac7ca7"

cd "$LLAMA_DIR"

if [ "$(git rev-parse HEAD)" != "$(git rev-parse $BASE_COMMIT)" ]; then
    echo "WARNING: HEAD is not at $BASE_COMMIT. Patches are tested against this commit."
    echo "Current HEAD: $(git log --oneline -1)"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

PATCHES=(
    "0001-sycl-battlemage-optimizations.patch"
    "0002-sycl-remove-dpct-capability-checks.patch"
    "0003-sycl-pad-non-contiguous-stride.patch"
    "0004-sycl-relax-sum-mean-contiguous.patch"
    "0005-sycl-rms-norm-mul-fusion.patch"
    "0006-sycl-add-new-ops.patch"
    "0007-sycl-barrier-local-space-fence.patch"
    "0008-sycl-native-sycl-shuffles.patch"
    "0009-sycl-scratchpad-pool-pvc-cleanup.patch"
    "0010-sycl-small-matmul-onemkl-direct.patch"
    "0011-sycl-fuse-unary-mul.patch"
    "0012-sycl-extend-graph-fusion.patch"
)

for patch in "${PATCHES[@]}"; do
    echo "Applying $patch..."
    if ! git apply --check "$SCRIPT_DIR/$patch" 2>/dev/null; then
        echo "  SKIP (already applied or conflict)"
        continue
    fi
    git apply "$SCRIPT_DIR/$patch"
    echo "  OK"
done

echo ""
echo "All patches applied. Build with:"
echo "  cmake -S . -B build -G Ninja \\"
echo "    -DCMAKE_BUILD_TYPE=Release \\"
echo "    -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \\"
echo "    -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \\"
echo "    -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \\"
echo "    -DGGML_SYCL_DEVICE_ARCH=bmg-g31"
echo "  cmake --build build --target llama-bench -j\$(nproc)"
