#!/bin/bash
if [ $# -lt 2  ]
then
   echo 'usage: update_all.sh <clue-type> <clue-count>'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1   #clue count
shift

if [ $1 == "--production" ]
then
    echo "---PRODUCTION---"
    _production=$1
    shift
    _save=--save
fi

echo "clues: $_ct, count: $_cc"

name=sugar
#_note=p3s.c2-$_cc.x2.$name

_out=tmp/update_all.err
echo $(date) >> $_out

#_force=--force

if [ $_note ]
then
    #
    # test update one note
    #
    node note -$_ct --update=$_note $1 $2 $_force #--save
    if [ $? -ne 0 ]
    then
	echo "update failed for $_note"
	exit -1
    fi
    # need to chop off tailing word of note name here, that's our name (e.g. sugar)
else
    #
    # update all notes
    #
    node note -$_ct --match $_ct.c2-$_cc.x2 $1 $2 $_production $_save --update 2>> $_out
    if [ $? -ne 0 ]
    then
	echo "update all failed"
	exit -1
    fi

    echo "Generating new clues.."
    node ../clues -$_ct -c2,$_cc -x2 > tmp/$_ct.c2-$_cc.x2 2>> $_out

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
    exit -1
fi


#TODO : (auto remove .filtered, no need for --note)
#dump errors in a file 2>tmp/p3s.update.errors
