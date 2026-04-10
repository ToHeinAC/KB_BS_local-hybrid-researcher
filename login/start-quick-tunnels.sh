#!/bin/bash
# Quick Cloudflare Tunnels for Hybrid Researcher
# Generates temporary *.trycloudflare.com URLs
# Safe: only kills tunnels for THIS project's ports (8522/8511)

set -e

echo "Starting Quick Cloudflare Tunnels (Hybrid Researcher)"
echo "======================================================"
echo ""

# Kill only tunnels for our specific ports (preserve brain-nw1 and others)
echo "Cleaning up existing tunnels for ports 8522/8511..."
pkill -f "cloudflared tunnel --url http://localhost:8522" 2>/dev/null || true
pkill -f "cloudflared tunnel --url http://localhost:8511" 2>/dev/null || true
sleep 2

# Backup config files to force true quick tunnels
# (cloudflared uses ~/.cloudflared/config.yml if present, which breaks quick tunnel mode)
CONFIG_BACKUP=""
CRED_BACKUP=""
CRED_FILE=~/.cloudflared/5bc36850-dc39-48be-81ab-fd15f8071bd0.json

if [ -f ~/.cloudflared/config.yml ]; then
    CONFIG_BACKUP=~/.cloudflared/config.yml.quickbackup
    mv ~/.cloudflared/config.yml "$CONFIG_BACKUP"
    echo "Temporarily moved config.yml"
fi

if [ -f "$CRED_FILE" ]; then
    CRED_BACKUP="${CRED_FILE}.quickbackup"
    mv "$CRED_FILE" "$CRED_BACKUP"
    echo "Temporarily moved tunnel credentials"
fi

# Start launcher tunnel (port 8522)
echo "Starting launcher tunnel on port 8522..."
nohup cloudflared tunnel --url http://localhost:8522 > /tmp/hybrid-launcher-tunnel.log 2>&1 &
LAUNCHER_PID=$!
sleep 8

# Start app tunnel (port 8511)
echo "Starting app tunnel on port 8511..."
nohup cloudflared tunnel --url http://localhost:8511 > /tmp/hybrid-app-tunnel.log 2>&1 &
APP_PID=$!
sleep 8

# Restore config files
if [ -n "$CONFIG_BACKUP" ] && [ -f "$CONFIG_BACKUP" ]; then
    mv "$CONFIG_BACKUP" ~/.cloudflared/config.yml
    echo "Restored config.yml"
fi

if [ -n "$CRED_BACKUP" ] && [ -f "$CRED_BACKUP" ]; then
    mv "$CRED_BACKUP" "$CRED_FILE"
    echo "Restored tunnel credentials"
fi

# Extract URLs from logs
echo ""
echo "Extracting tunnel URLs..."
sleep 2

LAUNCHER_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/hybrid-launcher-tunnel.log | head -1 || echo "")
APP_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/hybrid-app-tunnel.log | head -1 || echo "")

if [ -z "$LAUNCHER_URL" ] || [ -z "$APP_URL" ]; then
    echo "Failed to extract URLs. Checking logs..."
    echo ""
    echo "=== Launcher Log ==="
    tail -20 /tmp/hybrid-launcher-tunnel.log
    echo ""
    echo "=== App Log ==="
    tail -20 /tmp/hybrid-app-tunnel.log
    exit 1
fi

# Save URLs for the launcher app to read
echo "$LAUNCHER_URL" > /tmp/hybrid-launcher-url.txt
echo "$APP_URL" > /tmp/hybrid-app-url.txt

echo ""
echo "======================================"
echo "Quick Tunnels Running!"
echo "======================================"
echo ""
echo "Launcher URL:  $LAUNCHER_URL"
echo "Main App URL:  $APP_URL"
echo ""
echo "IMPORTANT: These URLs are temporary!"
echo "They change each time you restart the tunnels."
echo ""
echo "URLs saved to:"
echo "  /tmp/hybrid-launcher-url.txt"
echo "  /tmp/hybrid-app-url.txt"
echo ""
echo "To stop tunnels (this project only):"
echo "  pkill -f 'cloudflared tunnel --url http://localhost:8522'"
echo "  pkill -f 'cloudflared tunnel --url http://localhost:8511'"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/hybrid-launcher-tunnel.log"
echo "  tail -f /tmp/hybrid-app-tunnel.log"
echo ""

# Save execution log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/start-quick-tunnels_log.md"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

cat > "$LOG_FILE" << EOF
# Quick Tunnels Execution Log

**Last run:** $TIMESTAMP

## URLs

- **Launcher:** $LAUNCHER_URL (port 8522)
- **Main App:** $APP_URL (port 8511)

## PIDs

- Launcher tunnel: $LAUNCHER_PID
- App tunnel: $APP_PID

## Stop commands

\`\`\`bash
pkill -f 'cloudflared tunnel --url http://localhost:8522'
pkill -f 'cloudflared tunnel --url http://localhost:8511'
\`\`\`
EOF

echo "Execution log saved to: $LOG_FILE"
