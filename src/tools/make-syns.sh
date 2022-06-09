#!/bin/bash

if [ $# -lt 1  ]
then
   echo 'usage: make-syns.sh <words-file>'
   exit -1
fi

_wordsfile=$1
shift

echo "synonyms from $_wordsfile"
while read -r _word; do
    if [ ! -z "$_word" ]; then
        _file="$_word.json"
        _path="data/syns/$_file"
        if [ -f "$_path" ]; then
            echo "$_word" - file exists
        else
            echo "$_word"
            node syn "$_word" > /tmp/"$_file"
            _exitcode=$?
            if [ $_exitcode -ne 0 ]; then
                echo "exitcode: $_exitcode"
            else
                mv /tmp/"$_file" "$_path"
            fi
            sleep 31
        fi
     fi
done < "$_wordsfile"
