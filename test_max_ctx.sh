#!/usr/bin/env bash
# test_max_ctx.sh — MAX_MODEL_LEN 단계별 안정성 테스트
#
# 사용법:
#   ./test_max_ctx.sh               # 기본값 단계 자동 테스트
#   ./test_max_ctx.sh 32768         # 단일 값 테스트
#   ./test_max_ctx.sh 32768 65536   # 지정 범위만 테스트
#
# 결과는 test_max_ctx_results.log 에 기록됨

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
LOG_FILE="$SCRIPT_DIR/test_max_ctx_results.log"
API_URL="http://localhost:${HOST_PORT:-8000}"
CONTAINER_NAME="vllm-sehyo-qwen3.5-122b-nvfp4"

# 테스트할 MAX_MODEL_LEN 단계
DEFAULT_STEPS=(8192 16384 32768 65536 131072 196608 262144)

HEALTH_TIMEOUT=1200  # 컨테이너 기동 대기 최대 시간(초)
HEALTH_INTERVAL=10   # health 체크 간격(초)

# ───────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

read_env_var() {
    grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d'=' -f2-
}

update_env_var() {
    local key=$1 val=$2
    if grep -qE "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
    else
        echo "${key}=${val}" >> "$ENV_FILE"
    fi
}

wait_healthy() {
    local elapsed=0
    log "  기동 대기 중 (최대 ${HEALTH_TIMEOUT}s)..."
    while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
        local status
        status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "not_found")
        if [[ "$status" == "healthy" ]]; then
            log "  → healthy (${elapsed}s 경과)"
            return 0
        fi
        if [[ "$status" == "not_found" ]] || docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "exited"; then
            log "  → 컨테이너 종료됨 (OOM 또는 오류)"
            return 1
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done
    log "  → 타임아웃"
    return 1
}

test_inference() {
    local ctx_len=$1
    local max_tokens=64
    # 목표: ctx_len의 60%를 입력으로 채움
    # "word " 패턴 반복. Qwen tokenizer 기준 "word" = 1토큰, " " = 1토큰 → "word " ≈ 1~2토큰
    # 안전하게 target = ctx_len * 0.5, 단어 수 = target / 2 (단어당 평균 2토큰 가정)
    local target_tokens=$(( ctx_len * 5 / 10 ))
    local word_count=$(( target_tokens / 2 ))
    local filler
    filler=$(python3 -c "
words = ['the','cat','sat','on','mat','and','dog','ran','far','away']
result = ' '.join(words[i % len(words)] for i in range($word_count))
print(result)
")

    local prompt="Please read this text carefully and then tell me what the 3rd word is: ${filler}"

    log "  inference 테스트 (단어 ${word_count}개, 목표 ~${target_tokens} tokens, max_new=${max_tokens})..."

    local response
    response=$(curl -s --max-time 180 \
        -X POST "${API_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"Sehyo_Qwen3.5-122B-A10B-NVFP4\",
            \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))" <<< "$prompt")}],
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0
        }" 2>&1 || true)

    local actual_tokens
    actual_tokens=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('prompt_tokens','?'))" 2>/dev/null || echo "?")
    log "  실제 prompt_tokens: ${actual_tokens} / ${ctx_len}"

    if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null; then
        log "  → inference 성공"
        return 0
    else
        log "  → inference 실패: ${response:0:300}"
        return 1
    fi
}

backup_env_vars() {
    BACKUP_MAX_MODEL_LEN=$(read_env_var MAX_MODEL_LEN)
    BACKUP_MAX_NUM_BATCHED_TOKENS=$(read_env_var MAX_NUM_BATCHED_TOKENS)
    log "백업: MAX_MODEL_LEN=${BACKUP_MAX_MODEL_LEN}, MAX_NUM_BATCHED_TOKENS=${BACKUP_MAX_NUM_BATCHED_TOKENS}"
}

restore_env_vars() {
    update_env_var MAX_MODEL_LEN "$BACKUP_MAX_MODEL_LEN"
    update_env_var MAX_NUM_BATCHED_TOKENS "$BACKUP_MAX_NUM_BATCHED_TOKENS"
    log "복구: MAX_MODEL_LEN=${BACKUP_MAX_MODEL_LEN}, MAX_NUM_BATCHED_TOKENS=${BACKUP_MAX_NUM_BATCHED_TOKENS}"
}

# ───────────────────────────────────────────────
main() {
    local steps=("${DEFAULT_STEPS[@]}")
    if [[ $# -gt 0 ]]; then
        steps=("$@")
    fi

    echo "" >> "$LOG_FILE"
    log "====== MAX_MODEL_LEN 안정성 테스트 시작 ($(date)) ======"
    log "테스트 단계: ${steps[*]}"

    backup_env_vars

    local last_ok=""
    local trap_set=false

    # 실패 또는 Ctrl+C 시 원래 값으로 복구
    trap 'log "인터럽트 감지, 원래 설정으로 복구..."; restore_env_vars; cd "$SCRIPT_DIR" && docker compose up -d 2>/dev/null; exit 1' INT TERM

    for ctx in "${steps[@]}"; do
        log ""
        log "────── 테스트: MAX_MODEL_LEN=${ctx} ──────"

        update_env_var MAX_MODEL_LEN "$ctx"
        update_env_var MAX_NUM_BATCHED_TOKENS "$ctx"

        log "  컨테이너 재시작..."
        (cd "$SCRIPT_DIR" && docker compose up -d --force-recreate) 2>&1 | tee -a "$LOG_FILE" || true

        if wait_healthy; then
            if test_inference "$ctx"; then
                log "  ✓ MAX_MODEL_LEN=${ctx} — 기동 및 추론 성공"
                last_ok=$ctx
            else
                log "  ✗ MAX_MODEL_LEN=${ctx} — 기동은 됐으나 추론 실패"
                break
            fi
        else
            log "  ✗ MAX_MODEL_LEN=${ctx} — 기동 실패 (OOM 또는 오류)"
            # 컨테이너 로그 마지막 30줄
            docker logs --tail=30 "$CONTAINER_NAME" 2>&1 | tee -a "$LOG_FILE" || true
            break
        fi
    done

    log ""
    log "====== 결과 ======"
    if [[ -n "$last_ok" ]]; then
        log "✓ 안정적으로 동작한 최대값: MAX_MODEL_LEN=${last_ok}"
        log "  → .env 를 ${last_ok} 으로 설정합니다."
        update_env_var MAX_MODEL_LEN "$last_ok"
        update_env_var MAX_NUM_BATCHED_TOKENS "$last_ok"
        (cd "$SCRIPT_DIR" && docker compose up -d --force-recreate) 2>&1 | tee -a "$LOG_FILE"
        wait_healthy || true
    else
        log "✗ 첫 단계부터 실패. 원래 값으로 복구합니다."
        restore_env_vars
        (cd "$SCRIPT_DIR" && docker compose up -d --force-recreate) 2>&1 | tee -a "$LOG_FILE"
        wait_healthy || true
    fi

    log "====== 완료 ($(date)) ======"
    log "전체 로그: $LOG_FILE"
}

main "$@"
