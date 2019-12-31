#!/bin/bash
if [ $# -lt 1  ]
then
   echo 'usage: split.sh <filename> [chunks]'
   echo '  chunks defaults to 3'
   exit -1
fi

_filename=$1  #filename
shift
echo "filename: $_filename"

_chunks=3
if [[ $# -gt 0 ]]
then
   _chunks=$1
   shift
fi

_prefix=$filename
if [[ $# -gt 0 ]]
then
   _prefix=$1
   shift
fi

_wcl=$(wc -l $_filename) # line count as ugly string
_lines=$(echo $_wcl | sed -E 's/^[^[[:digit:]]*([[:digit:]]+).*$/\1/') # line count

_lpc=$(($_lines / $_chunks)) # lines per chunk

echo "lines: $_lines, chunks: $_chunks, lpc: $_lpc"

split -l $_lpc $_filename $_prefix.
