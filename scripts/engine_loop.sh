#!/bin/bash
# Clanker Engine Improvement Loop
# Runs every 15 minutes via cron
# Novel sentence testing, benchmarking, version tracking
#
# Install: crontab -e
# */15 * * * * /var/home/deucebucket/ai-drive/clanker-lang/scripts/engine_loop.sh

cd /var/home/deucebucket/ai-drive/clanker-lang
export PATH="/usr/bin:/usr/local/bin:$PATH"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="benchmarks/logs"
NOVEL_DIR="benchmarks/novel_sentences"
mkdir -p "$LOG_DIR" "$NOVEL_DIR"

LOG="$LOG_DIR/run_${TIMESTAMP}.log"

echo "=== CLANKER ENGINE LOOP ===" > "$LOG"
echo "Timestamp: $TIMESTAMP" >> "$LOG"
echo "" >> "$LOG"

# Run the benchmark suite
python3 scripts/engine_benchmark.py >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== DONE ===" >> "$LOG"
echo "Log: $LOG"
