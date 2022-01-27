//
// name-count.js
//

'use strict';

//

const _           = require('lodash');

//

function makeNew(name, count, index) {
    return new NameCount(name, count, index);
}

function makeCopy(nc) {
    return new NameCount(nc.name, nc.count, nc.index);
}

function NameCount(name, count, index) {
    let splitList;
    
    if (!name) return;

    this.name  = name;
    this.count = _.toNumber(count);
    this.index = index ? _.toNumber(index) : undefined;

/*
    if (_.isNaN(this.count)) {
	console.log(`BAD name:${name} count:${count}`);
    }
*/
    
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

function count (nc) {
    return _.toNumber(nc.count);
}

//

function makeListFromCsv (csv) {
    return makeListFromNameList(csv.split(','));
}

//

function makeListFromNameList (nameList) {
    let ncList = [];
    nameList.forEach(name => {
        ncList.push(new NameCount(name));
    });
    return ncList;
}

//

function makeCountList(ncList) {
    return ncList.map(nc => count(nc));
}

//

function makeNameList(ncList) {
    return ncList.map(nc => nc.name);
}

//

function makeCanonicalName(name, count, index = undefined) {
    let s = name;
    if (count) {
        s += ':' + count;
        if (index) {
            s += '.' + index;
        }
    }
    return s;
}

function listToString (ncList) {
    if (!ncList) return _.toString(ncList);
    return ncList.map(nc => nc.toString()).toString();
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
    let s;

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
//    this.cachedCanonicalName = this.cashedCanonicalName || 
//    return this.cachedCanonicalName;
};

//
//

NameCount.prototype.setIndex = function(index) {
    this.index = index;
};

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
    return {"name": `"${this.name}"`, "count": this.count};
};

/*
NameCount.prototype.toJSON = function() {
    return this.toJSON(`{"name": "${this.name}", "count": ${this.count}}`;
};
*/

//

NameCount.prototype.logList = function(list) {
    let str;

    list.forEach(nc => {
        if (str.length > 0) {
            str += ',';
        }
        str += nc;
    });
    console.log(str);
};

//

module.exports = {
    count,
    makeNew,
    makeCopy,
    makeNameList,
    makeCountList,
    makeListFromNameList,
    makeListFromCsv,
    makeCanonicalName,
    makeNameMap,
    makeCountMap,
    listToString,
    listToJSON,
    listContains,
    listContainsAll
};
