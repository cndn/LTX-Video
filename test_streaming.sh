#!/bin/bash

# Test the Modal health endpoint first
echo "=== Testing Health Endpoint ==="
curl -s https://zjudn2013--ltxv-webrtc-fixed-webrtc-app.modal.run/health | jq .

echo -e "\n=== Testing Simple Offer ==="
curl -s -X POST https://zjudn2013--ltxv-webrtc-fixed-webrtc-app.modal.run/offer \
  -H "Content-Type: application/json" \
  -d '{
    "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n",
    "type": "offer",
    "prompt": "cat sleeping"
  }' | jq .