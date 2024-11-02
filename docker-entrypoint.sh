#!/bin/sh

# Make sure there's at least 1 reference voice...
if [ ! -e /app/voices/basic_ref_en.wav ]; then
    cp basic_ref_en.wav /app/voices/
fi

if [ ! -e /app/config.yaml ]; then
    cp /app/config.yaml.default /app/config.yaml
fi

exec /usr/local/bin/python server.py \
    --host 0.0.0.0 \
    "$@"
