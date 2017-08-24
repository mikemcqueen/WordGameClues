//
// name-count.js
//

'use strict';

//

var _           = require('lodash');

//

function makeNew(name, count, index) {
    return new NameCount(name, count, index);
}

function makeCopy(nc) {
    return new NameCount(nc.name, nc.count, nc.index);
}

function NameCount(name, count, index) {
    var splitList;
    
    if (!name) return;

    this.name  = name;
    this.count = _.toNumber(count);
    this.index = _.toNumber(index);

    if (!this.count) {
	splitList = name.split(':');
	if (splitList.length > 1) {
	    this.name = splitList[0];
	    this.count = _.toNumber(splitList[1]);
	    splitList = splitList[1].split('.');
	    if (splitList.length > 1) {
		this.index = _.toNumber(splitList[1]);
	    }
	}
    }
}

//

function makeListFromCsv(csv) {
    return makeListFromNameList(csv.split(','));
}

//

function makeListFromNameList(nameList) {
    var ncList;
    ncList = [];
    nameList.forEach(name => {
	ncList.push(new NameCount(name));
    });
    return ncList;
}

//

function makeCountList(ncList) {
    var countList;
    countList = [];
    ncList.forEach(nc => countList.push(_.toNumber(nc.count)));
    return countList;
}

//

function makeNameList(ncList) {
    var nameList;
    nameList = [];
    ncList.forEach(nc => nameList.push(nc.name));
    return nameList;
}

//

function makeCanonicalName(name, count, index) {
    var s = name;
    if (count) {
	s += ':' + count;
	if (index) {
	    s += '.' + index;
	}
    }
    return s;
}

//

function makeNameMap(ncList) {
    let nameMap = {};
    ncList.forEach(nc => {
	if (!nameMap[nc.name]) {
	    nameMap[nc.name] = [ nc.src ]; 
	}
	else {
	    nameMap[nc.name].push(nc.src);
	}
    });
    return nameMap;
}

//

function makeCountMap(ncList) {
    let countMap = {};
    ncList.forEach(nc => {
	if (!countMap[nc.src]) {
	    countMap[nc.src] = [ nc.name ]; 
	}
	else {
	    countMap[nc.src].push(nc.name);
	}
    });
}

//

function listToJSON(ncList) {
    var s;

    if (!ncList.length) { return '[]'; }

    s = '[';
    ncList.forEach((nc, index) => {
	if (index > 0) {
	    s += ',\n';
	}
	s += nc.toJSON();
    });
    s += ']';
    return s;
}

//

function listContains(ncListContains, nc) {
    return _.find(ncListContains, ['name', nc.name, 'count', nc.count]);
}

//
    
function listContainsAll(ncListContains, ncList) {
    return ncList.every(nc => {
	return _.find(ncListContains, ['name', nc.name, 'count', nc.count]);
    });

    /*
    console.log('listContainsAll, contains: ' + ncListContains + 
		', list: ' + ncList +
		', result: ' + ret);
    return ret;
    */
}

//////////////////////////////////////////////////////////////////////////////

NameCount.prototype.toString = function() {
    return makeCanonicalName(this.name, this.count, this.index);
}

//
//

NameCount.prototype.setIndex = function(index) {
    this.index = index;
}

//

NameCount.prototype.equals = function(nc) {
    return (nc.count === this.count) && (nc.name == this.name);
}

//

/*
NameCount.prototype.log = function() {
    console.log('NameCount: ' + this);
}
*/

//

NameCount.prototype.toJSON = function() {
    return '{ "name": ' + this.name + ', "count": ' + this.count + ' }';
}

//

NameCount.prototype.logList = function(list) {
    var str;

    list.forEach(nc => {
	if (str.length > 0) {
	    str += ',';
	}
	str += nc;
    });
    console.log(str);
}

//

module.exports = {
    makeNew              : makeNew,
    makeCopy             : makeCopy,
    makeNameList         : makeNameList,
    makeCountList        : makeCountList,
    makeListFromNameList : makeListFromNameList,
    makeListFromCsv      : makeListFromCsv,
    makeCanonicalName    : makeCanonicalName,
    makeNameMap          : makeNameMap,
    makeCountMap         : makeCountMap,
    listToJSON           : listToJSON,
    listContains         : listContains,
    listContainsAll      : listContainsAll
};
