#!/bin/bash
if [ $# -lt 3  ]
then
   echo 'usage: filtermerge.sh <clue-type> <clue-count> <word> [--production] [--generate]'
   exit -1
fi

_ct=$1  #clue type
shift
_cc=$1  #clue count
shift
_name=$1  #word
echo "Name: $_name"
shift

while [[ $# -gt 0 ]]
do
      if [[ $1 == "--production" ]]
      then
	  echo "---PRODUCTION---"
	  _production=$1
	  _save=--save
      elif [[ $1 == "--generate" ]]
      then
	   _generate=true
      elif [[ $1 == "--article" ]]
      then
	   _article=$1
      else
	  echo "unknown option, $1"
	  exit -1
      fi
      shift
done

_out=tmp/filtermerge.err
echo $(date) >> $_out

_base=$_ct.c2-$_cc.x2
if [[ $_generate ]]
then
    echo "Generating new clues.."
    node ../clues -$_ct -c2,$_cc -x2 > tmp/$_base 2>> $_out
fi

_note=$_ct.c2-$_cc.x2.$_name
_filtered=$_note
if [[ ! -z $_article ]]
then
    _filtered=$_filtered.article
fi
_filtered=$_filtered.filtered

echo "Grepping..."
grep $_name tmp/$_base > tmp/$_note

echo "Filtering..."
node filter -$_ct tmp/$_note $_article > tmp/$_filtered 2>> $_out
if [ $? -ne 0 ]
then
    echo "filter failed on $_note"
    exit -1
fi

echo "Merging..."
node note-merge -$_ct tmp/$_filtered --note $_note --force-create $_production 2>> $_out
if [ $? -ne 0 ]
then
    echo "merge failed on $_note.filtered"
    exit -1
fi
