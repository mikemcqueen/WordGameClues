//
// PECO.JS
//
// PErmutations/COmbinations
//

'use strict';

module.exports = Peco;

var LOGGING = false;

// args:
//  listArray
//
// -or-
//
//  sum       
//  count     
//  max       
//  required  

function Peco(args) {
    this.listArray  = args.listArray;
    this.sum        = args.sum;
    this.count      = args.count;
    this.max        = args.max;
    this.required   = args.required; // list of required numbers

    if (this.listArray) {
	if (!this.max) {
	    throw new Error('Peco: must specify max');
	}
    }
    else {
	if (!this.sum) {
	    throw new Error('Peco: must specify sum > 0, ' + args.sum +
			    ', or addends: ' + args.listArray);
	}
	if (!this.count && !this.max) {
	    throw new Error('Peco: must specify either count -or- max');
	}
	if (this.count && this.max) {
	    throw new Error('Peco: cannot specify both count -and- max');
	}
	if ((this.count > this.sum) || (this.max > this.sum)) {
	    throw new Error('Peco: count/max > sum');
	}
    }
}

//

Peco.prototype.log = function(text) {
    console.log(text);
}

//

Peco.prototype.getPermutations = function() {
    return this.getAllAddends(false);
}

//

Peco.prototype.getCombinations = function() {
    var count;
    
    if (this.listArray) {
	return this.getListCombinations(this.listArray);
    }
    else {
	return this.getAllAddends(true);
    }
}

//

Peco.prototype.getAllAddends = function(combFlag) {
    var count;
    var list;
    
    if (this.count) {
	list = this.getAddendsForCount(this.count, combFlag);
    }
    else {
	for (count = 2; count <= this.max; ++count) {
	    list = this.getAddendsForCount(count, combFlag, list);
	}
    }

    // sanity check
    if (!list.length) {
	throw new Error('Peco: empty list!');
    }
    return list;
}

//

Peco.prototype.getAddendsForCount = function(count, combFlag, pecoList) {
    var last = this.sum - (this.count - 1);
    
    if (count > this.sum) {
	throw new Error('Peco: count > sum');
    }

    if (LOGGING) {
	this.log('Peco: for ' + count + ', last=' + last + 
		 ', combFlag: ' + combFlag);
    }

    return this.buildResult({
	count:    count,
	combFlag: combFlag,
	pecoList: pecoList
    });
}

//

Peco.prototype.getListCombinations = function(listArray) {
    var pecoList;

    return this.buildResult({
	listArray: listArray,
	combFlag:       true
    });
}

//
// args:
//  count
//  listArray
//  combFlag
//  pecoList (internal)
//

Peco.prototype.buildResult = function(args) {
    var list;

    if (!args.pecoList) {
	args.pecoList = [];
    }

    if (args.listArray) {
	list = this.listFirst(args.listArray, args.combFlag);
    }
    else if (args.count) {
	list = this.first(args.count, args.combFlag);
    }
    else {
	throw new Error('missing arg, count: ' + args.count +
			', listArray: ' + args.listArray);
    }

    if (!list) {
	if (args.count) {
	    throw new Error('no permutations/combinations');
	}
	console.log('no permutations/combinations');
	return null;
    }

    do {
	args.pecoList.push(list);
	if (args.listArray) {
	    list = this.listNext(args.listArray, args.combFlag);
	}
	else {
	    list = this.next(args.combFlag);
	}
    } while (list);

    return args.pecoList;
}

//

Peco.prototype.listFirst = function(listArray, combFlag) {
    var last;
    var start;
    var srcCount;
    var index;
    var list;
    
    if (!listArray) {
	throw new Error('invalid countListArray, ' + listArray);
    }
    listArray.forEach(list => {
	if (LOGGING) {
	    console.log('list: ' + list);
	}
    });

    srcCount = listArray.length;
    start = 0;
    this.indexList = [];
    for (index = 0; index < srcCount; ++index) {
	this.indexList.push({
	    first:  start, 
	    index:  start,
	    last:   listArray[index].length - 1
	});
    }

    if (LOGGING) {
	this.log ('srcCount: ' + srcCount + ' indexList.length: ' + this.indexList.length);
    }

    if (this.getIndexSum() <= this.max) {
	list = this.getPecoList();
    }
    else {
	list = this.next(combFlag);
	if (!list) {
	    return null;
	}
    }
    return list;
}

//
//

Peco.prototype.listNext = function(combFlag)
{
    var lastIndex = this.indexList.length - 1;
    var index;
    var start;
    var sum;
    var inner;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    for (;;) {
	index = lastIndex;
	while (++this.indexList[index].index > this.indexList[index].last) {
	    /*
	    if (combFlag) { // combinations
		start = ++this.indexList[index].first;
		for (inner = index + 1; inner < this.indexList.length; ++inner) {
		    this.indexList[inner].index = this.indexList[inner].first = start;
		}
	    }
	    */
	    this.indexList[index].index = this.indexList[index].first;
	    --index;
	    if (index < 0) {
		return null;
	    }
	}
	if (this.getIndexSum() > this.max) {
	    continue;
	}
	/*
	if (this.required) {
	    if (!this.required.every(size => {
		return list.indexOf(Number(size)) > -1;
	    })) {
		continue;
	    }
	}
	*/
	break;
    }
    return this.getPecoList();
}

//
//

Peco.prototype.first = function(srcCount, combFlag) {
    var start;
    var last;
    var list;
    var index;

    if ((srcCount < 1) || (srcCount > this.sum)) {
	throw new Error('invalid srcCount, ' + srcCount);
    }

    last = this.sum - (srcCount - 1);
    start = 1;
    this.indexList = [];
    for (index = 1; index <= srcCount; ++index) {
	this.indexList.push({
	    first:  start, 
	    index:  start,
	    last:   last
	});
    }
  
    if (LOGGING) {
	this.log ('srcCount: ' + srcCount + ' indexList.length: ' + this.indexList.length);
    }

    if (this.getIndexSum() == this.sum) {
	list = this.getPecoList();
    }
    else {
	list = this.next(combFlag);
	if (!list) {
	    throw new Error('Peco: no permutations/combinations');
	    return null;
	}
    }
    return list;
}
    
//
//

Peco.prototype.next = function(combFlag)
{
    var lastIndex = this.indexList.length - 1;
    var index;
    var start;
    var sum;
    var inner;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    for (;;) {
	index = lastIndex;
	while (++this.indexList[index].index > this.indexList[index].last) {
	    if (combFlag) { // combinations
		start = ++this.indexList[index].first;
		for (inner = index + 1; inner < this.indexList.length; ++inner) {
		    this.indexList[inner].index = this.indexList[inner].first = start;
		}
	    }
	    this.indexList[index].index = this.indexList[index].first;
	    --index;
	    if (index < 0) {
		return null;
	    }
	}
	if (this.sum) {
	    if (this.getIndexSum() != this.sum) {
		continue;
	    }
	}
	else {
	    // IS this used? or was this a listNext integration hack?
	    if (this.getIndexSum() > this.max) {
		continue;
	    }
	}
	if (this.required) {
	    if (!this.required.every(size => {
		return list.indexOf(Number(size)) > -1;
	    })) {
		continue;
	    }
	}
	break;
    }
    return this.getPecoList();
}

// NOTE: this is basically getPecoList().getSum()

Peco.prototype.getIndexSum = function() {
    var sum;
    sum = 0;
    this.indexList.forEach((indexObj, index) => {
	if (this.listArray) {
	    sum += this.listArray[index][indexObj.index];
	}
	else {
	    sum += indexObj.index;
	}
    });
    return sum;
}

//

Peco.prototype.getPecoList = function() {
    var list = [];

    this.indexList.forEach((indexObj, index) => {
	if (this.listArray) {
	    list.push(this.listArray[index][indexObj.index]);
	}
	else {
	    list.push(indexObj.index);
	}
    });
    return list;
}

//

function display(prefix, pecoList) {
    var s = '';
    pecoList.forEach(peco => {
	if (s.length > 0) {
	    s += ',';
	}
	s += peco.index;
    });
    console.log(prefix + s);
}

