#!/bin/bash
# Start Zotero translation-server and REST API

echo "Starting Zotero translation-server..."
cd /app/translation-server
node src/server.js &
TRANS_PID=$!

# Wait for translation-server to start
sleep 5

echo "Starting Zotero REST API..."
cd /app
python server.py &
API_PID=$!

# Handle shutdown
trap "kill $TRANS_PID $API_PID 2>/dev/null" EXIT

# Wait for either process to exit
wait -n
exit $?
