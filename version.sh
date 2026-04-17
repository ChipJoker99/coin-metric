#!/bin/bash

if [[ ! -f changelog ]]; then
  echo "❌ changelog file not found"
  exit 1
fi

X="$(head -n 1 changelog)"

VERSION=$(echo "$X" | awk '{print $2}')

echo "$VERSION"