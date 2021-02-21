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

const Flag = {
    Combinations: 1,
    NoDuplicates: 2
};


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
            throw new Error('Peco: must specify sum > 0 (' + args.sum +
                            '), or addends: ' + args.listArray);
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
};

//

Peco.prototype.getPermutations = function() {
    if (this.listArray) {
        throw new Error('permutations on lists not supported');
    }
    else {
        return this.getAllAddends(false);
    }
};

//

Peco.prototype.getCombinations = function() {
    if (this.listArray) {
        return this.getListCombinations(this.listArray);
    } else {
        return this.getAllAddends(true);
    }
};

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
};

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
	flags: combFlag ? Flag.Combinations : 0,
        pecoList,
        quiet
    });
};

//

Peco.prototype.firstCombination = function () {
    this.first = true;
    return this.nextCombination();
};

//

Peco.prototype.nextCombination = function () {
    return this.buildNextResult({
        listArray: this.listArray,
        flags: Flag.Combinations | Flag.NoDuplicates
    });
};

//
// args:
//  listArray
//  flags
//

Peco.prototype.buildNextResult = function (args) {
    if (this.first) {
	if (args.listArray) {
	    this.first = false;
            return this.listFirst(args.listArray, args.flags);
	} else if (args.count) {
	    this.first = false;
            return this.first(args.count, args.flags); // TODO flags
	}
        throw new Error('missing arg, count: ' + args.count +
                        ', listArray: ' + args.listArray);
    }
    if (args.listArray) {
        return this.listNext(args.flags);
    } else {
        return this.next(args.flags); // TODO flags
    }
};

//

Peco.prototype.getListCombinations = function (listArray) {
    return this.buildResult({
        listArray,
	flags: Flag.Combinations
    });
};

//
// args:
//  count
//  listArray
//  combFlag
//  pecoList (internal)
//

Peco.prototype.buildResult = function (args) {
    if (!args.pecoList) {
        args.pecoList = [];
    }

    let list;
    if (args.listArray) {
        list = this.listFirst(args.listArray, args.flags);
    } else if (args.count) {
        list = this.first(args.count, args.flags);
    } else {
        throw new Error('missing arg, count: ' + args.count +
                        ', listArray: ' + args.listArray);
    }
    if (!list) {
	console.log(`no list`);
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
            list = this.listNext(args.flags);
        } else {
            list = this.next(args.flags);
        }
    } while (list);

    return args.pecoList;
};

//

Peco.prototype.listFirst = function (listArray, flags) {
    let last;
    let list;
    
    if (!listArray) {
        throw new Error('invalid countListArray, ' + listArray);
    }

    let srcCount = listArray.length;
    let start = 0;
    this.indexList = [];
    this.hash = new Set();
    for (let index = 0; index < srcCount; ++index) {
        this.indexList.push({
            first:  start,
            index:  start,
            last:   (flags & Flag.NoDuplicates) ? listArray[index].length - (srcCount - index) : listArray[index].length - 1
        });
	this.hash.add(start);
	if (flags & Flag.NoDuplicates) ++start;
    }
    //console.log(`lastList: ${this.indexList.map(entry => entry.last)}`);

    if (LOGGING) {
        this.log ('srcCount: ' + srcCount + ' indexList.length: ' + this.indexList.length);
    }

    const sum = this.getIndexSum();
    if (sum <= this.max) {
	this.log(`pecoList, getIndexSum ${sum}`);
        list = this.getPecoList();
    } else {
	this.log(`next, getIndexSum ${sum}, max ${this.max}`);
        list = this.listNext(flags);
    }
    return list;
};

//
//

Peco.prototype.listNext = function (flags) {
    const lastIndex = this.indexList.length - 1;
    //console.log(`lastIndex ${lastIndex}`);

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    for (;;) {
        let index = lastIndex;
        while ((++this.indexList[index].index) > this.indexList[index].last) {
	    /*
            if (flags & Flag.Combinations) { // combinations
	    //} else {
		    let start = ++this.indexList[index].first;
                    for (let inner = index + 1; inner <= lastIndex; ++inner) {
			this.indexList[inner].index = this.indexList[inner].first = start;
		    }
                }
            }
	    */
            this.indexList[index].index = this.indexList[index].first;
            --index;
            if (index < 0) {
                return null;
            }
        }
	if ((flags & Flag.Combinations) && (flags & Flag.NoDuplicates)) {
            for (let inner = index + 1; inner <= lastIndex; ++inner) {
		//if (this.indexList[inner].first < previousFirst) throw new Error (`inner ${inner} < prevFirsT ${previousFirst}`);
		let newFirst = this.indexList[inner - 1].index + 1;
		if (newFirst > this.indexList[inner].last) {
		    throw new Error (`newFirst ${newFirst} > last ${this.indexList[inner].last}`);
		}
		this.indexList[inner].first = newFirst;
		//console.log(`index[${inner}].first = ${newFirst}`);
		this.indexList[inner].index = this.indexList[inner].first;
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
};

//

Peco.prototype.first = function (srcCount, flags) {
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
        return this.next(flags);     
    }
    return this.getPecoList();
};
    
//

Peco.prototype.next = function (flags) {
    let lastIndex = this.indexList.length - 1;
    let index;

    // if last index is maxed reset to zero, increment next-to-last index, etc.
    do {
        index = lastIndex;
        while (++this.indexList[index].index > this.indexList[index].last) {
            if (flags & Flag.Combinations) { // combinations
                let start = ++this.indexList[index].first;
                for (let inner = index + 1; inner < this.indexList.length; ++inner) {
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
};

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
};

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
};

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
};

//

Peco.prototype.indexContainsAll = function(list) {
    // NOTE: requires list to have unique numbers
    return _.size(_.intersection(this.getPecoList(), list)) === _.size(list);
};

//

Peco.prototype.indexContainsAny = function(list) {
    return _.size(_.intersection(this.getPecoList(), list)) > 0;
};

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
};

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
};


module.exports = {
    makeNew              : makeNew,
    logging              : LOGGING,
    setLogging           : setLogging
};
