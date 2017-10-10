#!/bin/bash
if [ $# -lt 2  ]
then
   echo 'usage: update_all.sh <clue-type> <clue-count> [--production] [note.name]'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1   #clue count
shift

_name=""
while [[ $# -gt 0 ]]
do
      if [[ $1 == "--production" ]]
      then
	  echo "---PRODUCTION---"
	  _production=$1
	  _save=--save
      elif [[ -z $_name ]]
      then
	  _name=$1
      else
	  echo "multiple note names supplied: 1) $_note 2) $1"
	  exit -1
      fi
      shift
done

echo "clues: $_ct, count: $_cc, note: $_note"

_base=$_ct.c2-$_cc.x2

_out=tmp/update_all.err
echo $(date) >> $_out

#_force=--force

if [[ ! -z $_name ]]
then
    #
    #  update one note
    #
    _note=$_base.$_name
    node note -$_ct --update=$_note $1 $2 $_production $_save 2>> $_out
    if [[ $? -ne 0 ]]
    then
	echo "update failed for $_note"
	exit -1
    fi
else
    #
    # update all notes
    #
    node note -$_ct --match $_ct.c2-$_cc.x2 $1 $2 $_production $_save --update 2>> $_out
    if [[ $? -ne 0 ]]
    then
	echo "update all failed"
	exit -1
    fi
fi

echo "Generating new clues.."
node ../clues -$_ct -c2,$_cc -x2 > tmp/$_ct.c2-$_cc.x2 2>> $_out

if [[ ! -z $_name ]]
then
    ./filtermerge.sh $_ct $_cc $_name $_production 
else
    #for each name in  ../../data/words/$_ct.txt
    _wordsfile="../../data/words/$_ct.txt"
    echo "update all from $_wordsfile"
    while read -r _word
    do
	if [ ! -z $_word ]
	then
	    ./filtermerge.sh $_ct $_cc $_word $_production 
	fi
    done < "$_wordsfile"
fi
    
#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors
