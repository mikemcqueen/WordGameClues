'use strict';

var nameCountExports = {
    makeNew              : makeNew,
    makeNameList         : makeNameList,
    makeCountList        : makeCountList,
    makeListFromNameList : makeListFromNameList,
    makeListFromCsv      : makeListFromCsv,
    makeCanonicalName    : makeCanonicalName,
    makeNameMap          : makeNameMap,
    makeCountMap         : makeCountMap
};

module.exports = nameCountExports;

//module.exports = NameCount;

// TODO: some way to export a constructor, and a "static" function
// like makeCanonicalName?

function makeNew(name, count, index) {
    return new NameCount(name, count, index);
}

function NameCount(name, count, index) {
    var splitList;
    
    if (!name) return;

    this.name  = name;
    this.count = count;
    this.index = index;

    if (!this.count) {
	splitList = name.split(':');
	if (splitList.length > 1) {
	    this.name = splitList[0];
	    this.count = Number(splitList[1]);
	    splitList = splitList[1].split('.');
	    if (splitList.length > 1) {
		this.index = Number(splitList[1]);
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
    ncList.forEach(nc => countList.push(Number(nc.count)));
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
    var nameMap;
    nameMap = {};
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
    var countMap;
    countMap = {};
    ncList.forEach(nc => {
	if (!countMap[nc.src]) {
	    countMap[nc.src] = [ nc.name ]; 
	}
	else {
	    countMap[nc.src].push(nc.name);
	}
    });
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

NameCount.prototype.log = function() {
    console.log('NameCount: ' + this);
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


