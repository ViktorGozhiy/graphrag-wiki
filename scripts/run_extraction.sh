# ABOUTME: Grinds graph extraction to completion by re-running the resumable batch script.
# ABOUTME: After each stop it resumes from what is done, so a transient failure never loses progress.
set -u

cd "$(dirname "$0")/.." || exit 1
[ -f .env ] && set -a && . ./.env && set +a   # load OPENAI_API_KEY from the repo-root .env
: "${OPENAI_API_KEY:?set OPENAI_API_KEY (in .env or the environment) before running}"

COOLOFF=180       # seconds to wait after a round stops before resuming
MAX_STALLED=5     # give up after this many rounds in a row that add nothing

OUT="data/graph/extractions.jsonl"

count() { [ -f "$OUT" ] && wc -l < "$OUT" || echo 0; }

total=$(uv run python -c "from graphrag_wiki.config import CORPUS_PATH; from graphrag_wiki.corpus import iter_chunks, load_corpus; print(len(list(iter_chunks(load_corpus(CORPUS_PATH)))))")
echo "corpus: $total chunks"

stalled=0
prev=-1
while true; do
    uv run python scripts/extract_graph.py   # streams live (the batch flushes each line)
    done=$(count)
    if [ "$done" -ge "$total" ]; then
        echo "all $total chunks extracted"
        break
    fi
    if [ "$done" -le "$prev" ]; then
        stalled=$((stalled + 1))
        echo "no progress this round ($done/$total) — stalled $stalled/$MAX_STALLED"
        if [ "$stalled" -ge "$MAX_STALLED" ]; then
            echo "no progress for $MAX_STALLED rounds — stopping; resume later"
            break
        fi
    else
        stalled=0
    fi
    prev="$done"
    echo "cooling off ${COOLOFF}s ($((total - done))/$total pending)..."
    sleep "$COOLOFF"
done
