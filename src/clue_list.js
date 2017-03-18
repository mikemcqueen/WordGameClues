'use strict';

//

const _       = require('lodash');
const Expect  = require('chai').expect;
const Fs      = require('fs');

//
//

function display() {
    var arr = [];
    
    this.forEach(function(clue) {
	arr.push(clue.name);
    });
    console.log(arr.toString());
    
    return this;
}

//
//

function persist(filename) {
    if (Fs.exists(filename)) {
	throw new Error('file already exists: ' + filename);
    }
    Fs.writeFileSync(filename, this.toJSON());
}

//

function toJSON () {
    var result = '[\n';
    var first = true;

    this.forEach(clue => {
	if (!first) {
	    result += ',\n';
	}
	else { 
	    first = false;
	}
	result += "  " + clueToJSON(clue);
    });
    if (!first) {
	result += '\n';
    }
    result += ']';

    return result;
}

//

function save (filename) {
    Fs.writeFileSync(filename, this.toJSON(), { encoding: 'utf8' });
}

//

function clueToJSON (clue) {
    let s = '{';

    if (clue.name) {
	s += ` "name": "${clue.name}", ${format2(clue.name, 15)}`;
    }
    s += `"src": "${clue.src}"`;
    if (clue.actual) {
	s += `, ${format2(clue.src, 2)}"actual": "${clue.actual}"`;
    }
    if (clue.note) {
	s+= ', "note" : "' + clue.note + '"';
    }
    if (clue.x) {
	s+= ', "x" : ' + clue.x;
    }
    if (clue.ignore) {
	s+= ', "ignore" : ' + clue.ignore;
    }
    else if (clue.skip) {
	s+= ', "skip" : ' + clue.skip;
    }
    s += ' }';

    return s;
}

//

function format2 (text, span)
{
    var result = "";
    for (var len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//
//

function init () {
    return this;
}

//
//

function makeKey () {
    return this.map(clue => clue.name).sort().toString();
}

//

function sortSources () {
    for (let clue of this) {
	clue.src = clue.src.split(',').sort().toString();
    }
    return this;
}

//

function getSameSrcList (startIndex, options = {}) {
    let list = makeNew();
    let mismatch;
    if (!_.isUndefined(startIndex) && startIndex >=0 && startIndex < this.length) {
	let src = this[startIndex].src;
	list.push(..._.filter(this, (value, index) => {
	    if (index < startIndex) return false;
	    if (!_.isUndefined(mismatch) && !options.allowMismatch) return false;
	    if (this[index].src === src) return true;
	    if (_.isUndefined(mismatch)) mismatch = index;
	    return false;
	}));
    }
    return [list, mismatch];
}

//

function sortedBySrc () {
    let srcHash = {};
    let sorted = makeNew();
    this.sortSources();
    for (let index = 0; index < this.length; index += 1) {
	let src = this[index].src
	if (_.has(srcHash, src)) continue;
	srcHash[src] = true;
	sorted.push(..._.filter(this, (value, innerIdx) => {
	    return innerIdx >= index && this[innerIdx].src === src;
	}));
    }
    return sorted;
}

//

function clueSetActual (clue, actualClue) {
    Expect(actualClue.src).to.exist;
    clue.actual = actualClue.src;
}

//

function clueMergeFrom (toClue, fromClue, options) {
    let warnings = 0;

    if (!_.has(toClue, 'actual')) {
	clueSetActual(toClue, fromClue);
    } else if (toClue.actual !== fromClue.src) {
	console.log(`WARNING! mismatched actual for ${toClue.name}, to:${toClue.actual} from:${fromClue.src}`);
	warnings += 1;
    }
    if (toClue.x !== fromClue.x) {
	if (_.isUndefined(toClue.x)) {
	    toClue.x = fromClue.x;
	} else {
	    console.log(`WARNING! mismatched X for ${toClue.name}, to:${toClue.x} from:${fromClue.x}`);
	    warnings += 1;
	}
    }
    if (toClue.note !== fromClue.note) {
	if (_.isUndefined(toClue.note)) {
	    toClue.note = fromClue.note;
	} else {
	    console.log(`WARNING! mismatched note for ${toClue.name}, to:${toClue.note} from:${fromClue.note}`);
	    warnings += 1;
	}
    }
    if (toClue.skip !== fromClue.skip) {
	if (_.isUndefined(toClue.skip)) {
	    toClue.skip = fromClue.skip;
	} else {
	    console.log(`WARNING! mismatched skip for ${toClue.name}, to:${toClue.skip} from:${fromClue.skip}`);
	    warnings += 1;
	}
    }
    if (toClue.ignore !== fromClue.ignore) {
	if (_.isUndefined(toClue.ignore)) {
	    toClue.ignore = fromClue.ignore;
	} else {
	    console.log(`WARNING! mismatched ignore for ${toClue.name}, to:${toClue.ignore} from:${fromClue.ignore}`);
	    warnings += 1;
	}
    }
    return warnings;
}

//

function sameSrcMergeFrom (fromList, options = {}) {
    Expect(fromList.length, 'fromList.length < this.length').is.at.least(this.length);
    let warnings = 0;
    for (let [index, clue] of this.entries()) {
	Expect(clue.name, `name mismatch, ${clue.name} vs ${fromList[index].name}`)
	    .to.equal(fromList[index].name);
	warnings += clueMergeFrom(clue, fromList[index], options);
	this[index] = clue;
    }
    // append remaing non-matching-name clues, but don' allow duplicate names
    for (let clue of fromList.slice(this.length, fromList.length)) {
	// check if clue.name already exists in this list
	Expect(_.find(this, ['name', clue.name])).to.be.undefined; // TODO: define/use Clue.NAME
	Expect(options.src).to.be.a('string');
	clueSetActual(clue, clue);
	clue.src = options.src;
	this.push(clue);
    }
    return [this, warnings];
}

//

function mergeFrom (fromList, options = {}) {
    let merged = makeNew();
    let toIndex = 0;
    let fromIndex = 0;
    let warnings = 0;
    let srcNum = 0;
    while (!_.isUndefined(toIndex) || !_.isUndefined(fromIndex)) {
	let [sameSrcFromList, nextFromIndex] = fromList.getSameSrcList(fromIndex, options);
	let [sameSrcToList, nextToIndex] = this.getSameSrcList(toIndex, options);
	// use the src from TO list if available
	if (!_.isEmpty(sameSrcToList)) {
	    srcNum = _.toNumber(sameSrcToList[0].src);
	} else {
	    srcNum += 1;
	}
	Expect(srcNum).to.be.above(0);
	//console.log(`to: ${sameSrcToList.toJSON()}`);
	let mergeOptions = { src: srcNum.toString() };
	let [sameSrcMerged, mergeWarnings] = sameSrcToList.sameSrcMergeFrom(sameSrcFromList, mergeOptions);
	merged.push(...sameSrcMerged);
	warnings += mergeWarnings;
	
	toIndex = nextToIndex;
	fromIndex = nextFromIndex;
    }
    Expect(toIndex, 'looks like toList is bigger than fromlist, manually fix it').to.be.undefined;
    Expect(fromIndex, 'pigs fly').to.be.undefined;
    return [merged, warnings];
}

//
//

function assignMethods (list) {
    list.display          = display;
    list.getSameSrcList   = getSameSrcList;
    list.init             = init;
    list.makeKey          = makeKey;
    list.mergeFrom        = mergeFrom;
    list.toJSON           = toJSON;
    list.save             = save;
    list.sameSrcMergeFrom = sameSrcMergeFrom;
    list.sortedBySrc      = sortedBySrc;
    list.sortSources      = sortSources;

    return list;
}

// objectFrom(args)
//
// args: see makeFrom()

function objectFrom (args) {
    let clueList = [];

    if (args.filename) {
	clueList = JSON.parse(Fs.readFileSync(args.filename, 'utf8'));
    }
    else if (args.array) {
	if (!Array.isArray(args.array)) {
	    throw new Error('bad array');
	}
	clueList = args.array;
    }
    else {
	console.log('args:');
	_.keys(args).forEach(key => {
	    console.log('  ' + key + ' : ' + args[key]);
	});
	throw new Error('missing argument');
    }
    
    return clueList;
}

//
//

function makeNew () {
    return assignMethods(Object([]));
}

// makeFrom()
//
// args:
//   filename: filename string
//   optional: optional flag suppresses file error
//   array:    js array

function makeFrom (args) {
    let object = objectFrom(args);
    return assignMethods(Object(object)).init();
}

//

module.exports = {
    makeNew:  makeNew,
    makeFrom: makeFrom,
    makeKey:  makeKey
};

