#!/bin/bash
if [ $# -lt 1  ]
then
   echo 'usage: split.sh <filename> [lines-per-chunk] [output-prefix]'
   echo '  LPC defaults to 750'
   exit 2
fi

_filename=$1  #filename
shift
echo "filename: $_filename"

_lpc=750
if [[ $# -gt 0 ]]
then
   _lpc=$1
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

#_lpc=$(($_lines / $_chunks)) # lines per chunk

_chunks=$(($_lines / $_lpc)) # chunks
_last=$(($_lines - $_chunks * $_lpc))

echo "lines: $_lines, chunks: $_chunks, lpc: $_lpc, last chunk: $_last"

split -l $_lpc $_filename $_prefix.
