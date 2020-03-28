//
// peco.js
//
// PErmutations/COmbinations
//

'use strict';

//
// TODO:
// make this a work with hasNext(), next().
// then make it stream.
//
// Use log.debug instead of logging.
//

const _           = require('lodash');
const Debug       = require('debug')('peco');

//

const FORCE_QUIET = false;
const LOGGING = false;

function makeNew(args) {
    return new Peco(args);
}

// args:
//  listArray
//  max       
//
// -or-
//
//  sum       
//  count     create lists of this length
//    -or-
//  max       create lists of max length
//
//
//  require:  list of required numbers, e.g. [2, 4]
//  exclude:  list of excluded numbers, e.g. [3, 5]

function Peco(args) {
    if (LOGGING) {
        this.log('Peco: sum: '+ args.sum +','  + typeof args.sum +
                 ', max: ' + args.max+','+ typeof args.max + 
                 ', count: ' + args.count +','+ typeof args.count +
                 ', require: ' + args.require +
                 ', exclude: ' + args.exclude +
		 `, listArray: ${args.listArray ? args.listArray.length : args.listArray}`);
    }
    
    this.listArray  = args.listArray;
    this.sum        = args.sum;
    this.count      = args.count;
    this.max        = args.max;
    this.require    = args.require; // list of required numbers
    this.exclude    = args.exclude; // list of exluded numbers

    if (this.listArray) {
        if (!this.max) {
            throw new Error('Peco: must specify max with listArray');
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
        if (this.count > this.sum) {
            throw new Error(`Peco: count(${this.count}) > sum(${this.sum})`);
        }
        if (this.require && this.exclude &&
            _.intersection(this.require, this.exclude).length > 0)
        {
            throw new Error('Peco: require and exclude contain same number(s), ' +
                            _.intersection(this.require, this.exclude));
        }
    }
}

//

function setLogging(flag) {
    if (!flag || (flag && !FORCE_QUIET)) {
        LOGGING = flag;
    }
}

//

Peco.prototype.log = function(text) {
    if (LOGGING) {
        console.log(text);
    }
}

//

Peco.prototype.getPermutations = function() {
    if (this.listArray) {
        throw new Error('permutations on lists not supported');
    }
    else {
        return this.getAllAddends(false);
    }
}

//

Peco.prototype.getCombinations = function() {
    if (this.listArray) {
        return this.getListCombinations(this.listArray);
    } else {
        return this.getAllAddends(true);
    }
}

//

Peco.prototype.getAllAddends = function(combFlag) {
    let list;
    if (this.count) {
        list = this.getAddendsForCount(this.count, combFlag);
    } else {
        for (let count = 2; count <= this.max; ++count) {
            list = this.getAddendsForCount(count, combFlag, list);
        }
    }
    return list;
}

//

Peco.prototype.getAddendsForCount = function(count, combFlag, pecoList, quiet) {
    if (count > this.sum) {
        throw new Error(`Peco: count(${count}) > sum(${this.sum})`);
    }
    if (LOGGING) {
        const last = this.sum - (count - 1);
        this.log('Peco: for ' + count + ', last=' + last + 
                 ', combFlag: ' + combFlag);
    }
    return this.buildResult({
        count,
        combFlag,
        pecoList,
        quiet
    });
}

//

Peco.prototype.getListCombinations = function(listArray) {
    return this.buildResult({
        listArray,
        combFlag:  true
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
    let list;

    if (!args.pecoList) {
        args.pecoList = [];
    }

    if (args.listArray) {
        list = this.listFirst(args.listArray, args.combFlag);
    } else if (args.count) {
        list = this.first(args.count, args.combFlag);
    } else {
        throw new Error('missing arg, count: ' + args.count +
                        ', listArray: ' + args.listArray);
    }
    if (!list) {
        if (LOGGING) {
            this.log('Peco: no ' + args.count + ' in ' +  list);
        }
        return [];
    }
    do {
        if (LOGGING) {
            this.log('Peco: adding: ' + list);
        }
        args.pecoList.push(list);
        if (args.listArray) {
            list = this.listNext(args.listArray, args.combFlag);
        } else {
            list = this.next(args.combFlag);
        }
    } while (list);

    return args.pecoList;
}

//

Peco.prototype.listFirst = function (listArray, combFlag) {
    let last;
    let start;
    let srcCount;
    let index;
    let list;
    
    if (!listArray) {
        throw new Error('invalid countListArray, ' + listArray);
    }
    listArray.forEach(list => {
        if (LOGGING) {
            this.log('list: ' + list);
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
	this.log(`pecoList, getIndexSum = ${this.getIndexSum()}`);
        list = this.getPecoList();
    } else {
	this.log(`next, getIndexSum = ${this.getIndexSum()}`);
        list = this.next(combFlag);
    }
    return list;
}

//
//

Peco.prototype.listNext = function (combFlag) {
    let lastIndex = this.indexList.length - 1;
    let index;
    let start;
    let sum;
    let inner;

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
        if (this.require) {
            if (!this.require.every(size => {
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

Peco.prototype.first = function (srcCount, combFlag) {
    if ((srcCount < 1) || (srcCount > this.sum)) {
        throw new Error('invalid srcCount, ' + srcCount);
    }

    // initIndexList;
    let last = this.sum - (srcCount - 1);
    let start = 1;
    this.indexList = [];
    for (let index = 1; index <= srcCount; index += 1) {
        this.indexList.push({
            first:  start, 
            index:  start,
            last:   last
        });
    }
  
    if (LOGGING) {
        this.log('first: ' + this.indexListToJSON());
        this.log ('srcCount: ' + srcCount + 
                  ', indexList.length: ' + this.indexList.length +
                  ', this.getIndexSum(): ' + this.getIndexSum() +
                  ', this.sum: ' + this.sum);
    }
    if (!this.isValidIndex()) {
        return this.next(combFlag);     
    }
    return this.getPecoList();
}
    
//

Peco.prototype.next = function (combFlag) {
    let lastIndex = this.indexList.length - 1;
    let index;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    do {
        index = lastIndex;
        while (++this.indexList[index].index > this.indexList[index].last) {
            if (combFlag) { // combinations
                let start = ++this.indexList[index].first;
                for (let inner = index + 1; inner < this.indexList.length; inner += 1) {
                    this.indexList[inner].index = this.indexList[inner].first = start;
                }
            }
            this.indexList[index].index = this.indexList[index].first;
            --index;
            if (index < 0) {
                return null;
            }
        }
        Debug('-----\nnext: ' + this.indexListToJSON());
        Debug('indexList.length: ' + this.indexList.length +
              ', this.getIndexSum(): ' + this.getIndexSum() +
              ', this.sum: ' + this.sum);
    } while (!this.isValidIndex());

    return this.getPecoList();
}

//

Peco.prototype.isValidIndex = function () {
    if (this.getIndexSum() != this.sum) {
        if (LOGGING) {
            this.log('Mismatch sums, ' + this.getIndexSum() +
                     ' != ' + this.sum);
        }
        return false;
    }
    if (this.require &&
        !this.indexContainsAll(this.require)) {
        if (LOGGING) {
            this.log('Missing required number: ' + this.require);
        }
        return false;
    }
    if (this.exclude &&
        this.indexContainsAny(this.exclude)) {
        if (LOGGING) {
            this.log('Contains excluded number: ' + this.exclude);
        }
        return false;
    }
    return true;
}

// NOTE: this is basically getPecoList().getSum()

Peco.prototype.getIndexSum = function() {
    let sum;
    sum = 0;
    this.indexList.forEach((indexObj, index) => {
        if (this.listArray) {
            sum += this.listArray[index][indexObj.index];
        } else {
            sum += indexObj.index;
        }
    });
    return sum;
}

//
//

Peco.prototype.indexListToJSON = function() {
    let s;
    s = '';
    this.indexList.forEach((indexObj, index) => {
        if (s.length > 0) {
            s += ',';
        }
        if (this.listArray) {
            s += this.listArray[index][indexObj.index];
        } else {
            s += indexObj.index;
        }
    });
    return '[' + s + ']';
}

//

Peco.prototype.indexContainsAll = function(list) {
    // NOTE: requires list to have unique numbers
    return _.size(_.intersection(this.getPecoList(), list)) === _.size(list);
}

//

Peco.prototype.indexContainsAny = function(list) {
    return _.size(_.intersection(this.getPecoList(), list)) > 0;
}

//

Peco.prototype.getPecoList = function() {
    let list = [];
    this.indexList.forEach((indexObj, index) => {
        if (this.listArray) {
            list.push(this.listArray[index][indexObj.index]);
        } else {
            list.push(indexObj.index);
        }
    });
    return list;
}

//

function display(prefix, pecoList) {
    let s = '';
    pecoList.forEach(peco => {
        if (s.length > 0) {
            s += ',';
        }
        s += peco.index;
    });
    console.log(prefix + s);
}


module.exports = {
    makeNew              : makeNew,
    logging              : LOGGING,
    setLogging           : setLogging
};
