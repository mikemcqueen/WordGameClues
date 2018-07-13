#!/bin/bash
if [ $# -lt 4  ]
then
   echo 'usage: pairs.sh <clue-num> <min-word-length>  <words-file> <letters-file> [used letters]'
   exit -1
fi

_cn=$1  #clue num
shift
_mwl=$1  #min word length
shift

_wordsfile=$1
shift
_lettersfile=$1
shift

_rwl=5 #required word length

echo "pairs from $_wordsfile"
while read -r _word
do
    if [ ! -z $_word ]
    then
        echo $_word
        ./ancc --two -m $_mwl --dict sentence/$_cn/reduced "`cat $_lettersfile`" -u "$_word $@" | grep -E "[^ ']{$_rwl}" > $_cn/pairs.$_word.$_mwl.$_rwl
    fi
done < "$_wordsfile"
