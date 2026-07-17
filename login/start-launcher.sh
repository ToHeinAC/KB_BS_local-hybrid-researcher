#!/bin/bash
# Start the Hybrid Researcher Launcher on port 8522

set -e

echo "Starting Hybrid Researcher Launcher"
echo "===================================="
echo ""

# Check for password
if [ -z "$LAUNCHER_PASSWORD" ]; then
    read -sp "Enter launcher password: " LAUNCHER_PASSWORD
    echo ""
    export LAUNCHER_PASSWORD
fi

# Check if port 8522 is available
if lsof -Pi :8522 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8522 is already in use"
    read -p "Kill existing process? (y/n): " KILL_PROC
    if [ "$KILL_PROC" = "y" ]; then
        lsof -ti:8522 | xargs -r kill -9
        echo "Killed process on port 8522"
        sleep 1
    else
        echo "Cannot start - port 8522 is in use"
        exit 1
    fi
fi

# Change to project root (parent of login/)
cd "$(dirname "$0")/.."

echo ""
echo "Starting launcher on port 8522..."
echo "Access URL: http://localhost:8522"
echo ""
echo "To access remotely, run ./login/start-quick-tunnels.sh first"
echo ""

# Start the launcher.
# --server.baseUrlPath="" is load-bearing: we cd'd into the project root above,
# so the launcher inherits .streamlit/config.toml, whose baseUrlPath = "brain"
# is meant for the *app* (which nginx serves at /brain/). Without this override
# the launcher would answer at :8522/brain/ and 404 at :8522/.
uv run streamlit run login/launcher_app.py --server.port 8522 --server.headless true \
  --server.baseUrlPath=""
