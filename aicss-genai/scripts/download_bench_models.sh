#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${LLAMA_MODELS_DIR:-${ROOT_DIR}/models}"
MANIFEST="${LLAMA_BENCH_MANIFEST:-${ROOT_DIR}/scripts/bench_models_manifest.tsv}"
case "${MANIFEST}" in
    /*) ;;
    *) MANIFEST="${ROOT_DIR}/${MANIFEST}" ;;
esac
LOG_FILE="${MODELS_DIR}/bench_model_download.log"

mkdir -p "${MODELS_DIR}"

download_one() {
    local name="$1"
    local expected_size="$2"
    local url="$3"
    local target="${MODELS_DIR}/${name}.gguf"

    if [[ -L "${target}" ]]; then
        printf 'SKIP\t%s\tsymlink\t%s\n' "${name}" "$(date -Is)" | tee -a "${LOG_FILE}"
        return 0
    fi

    if [[ -f "${target}" ]]; then
        local current_size
        current_size="$(stat -c '%s' "${target}")"
        if [[ "${current_size}" == "${expected_size}" ]]; then
            printf 'OK\t%s\t%s\t%s\n' "${name}" "${current_size}" "$(date -Is)" | tee -a "${LOG_FILE}"
            return 0
        fi
    fi

    printf 'START\t%s\t%s\t%s\n' "${name}" "$( [[ -f "${target}" ]] && stat -c '%s' "${target}" || echo 0 )" "$(date -Is)" | tee -a "${LOG_FILE}"
    wget \
        --continue \
        --tries=0 \
        --waitretry=5 \
        --retry-connrefused \
        --timeout=30 \
        --read-timeout=30 \
        --no-verbose \
        --output-document="${target}" \
        "${url}"

    local final_size
    final_size="$(stat -c '%s' "${target}")"
    if [[ "${final_size}" != "${expected_size}" ]]; then
        printf 'ERROR\t%s\t%s\t%s\t%s\n' "${name}" "${final_size}" "${expected_size}" "$(date -Is)" | tee -a "${LOG_FILE}" >&2
        return 1
    fi

    printf 'DONE\t%s\t%s\t%s\n' "${name}" "${final_size}" "$(date -Is)" | tee -a "${LOG_FILE}"
}

: > "${LOG_FILE}"

name=''
tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r name expected_size url || [[ -n "${name-}" ]]; do
    [[ -z "${name}" ]] && continue
    download_one "${name}" "${expected_size}" "${url}"
done

python3 "${ROOT_DIR}/scripts/verify_bench_models.py" --manifest "${MANIFEST}" --models-dir "${MODELS_DIR}" | tee -a "${LOG_FILE}"
