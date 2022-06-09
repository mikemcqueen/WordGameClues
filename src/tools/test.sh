#!/bin/bash
_word=20
_file="$_word.json"
_path="data/syns/$_file"
if [[ -f "$_path" ]]; then
    echo "$_path" - file exists
else
    echo "$_path" - no file exists
fi
