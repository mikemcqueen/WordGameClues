//
// name-count.js
//

'use strict';

//

import _ from 'lodash';

//
//

interface NameCount {
    name: string;
    count: number;
    index?: number;
}

export type Type = NameCount;
export type List = NameCount[];

//
//

export function makeNew (name: string, count: number, index?: number): NameCount {
    return new NameCount_t(name, count, index);
}

function makeCopy(nc: NameCount): NameCount {
    return new NameCount_t(nc.name, nc.count, nc.index);
}

function NameCount_t (name: string, count?: number, index?: number) {
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

export function count (nc: NameCount): number {
    return nc.count;
}

//

export function makeListFromCsv (csv) {
    return makeListFromNameList(csv.split(','));
}

//

function makeListFromNameList (nameList: string[]): NameCount[] {
    let ncList: NameCount[] = [];
    nameList.forEach(name => {
        ncList.push(new NameCount_t(name));
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

export function makeCanonicalName (name: string, count: number, index?: number): string {
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

export function listContains(ncListContains, nc) {
    return _.find(ncListContains, ['name', nc.name, 'count', nc.count]);
}

//
    
export function listContainsAll(ncListContains, ncList) {
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

NameCount_t.prototype.toString = function(): string {
    return makeCanonicalName(this.name, this.count, this.index);
//    this.cachedCanonicalName = this.cashedCanonicalName || 
//    return this.cachedCanonicalName;
};

//
//

NameCount_t.prototype.setIndex = function(index) {
    this.index = index;
};

//

NameCount_t.prototype.equals = function(nc) {
    return (nc.count === this.count) && (nc.name == this.name);
}

//

/*
NameCount_t.prototype.log = function() {
    console.log('NameCount: ' + this);
}
*/

//

NameCount_t.prototype.toJSON = function() {
    return {"name": `"${this.name}"`, "count": this.count};
};

/*
NameCount_t.prototype.toJSON = function() {
    return this.toJSON(`{"name": "${this.name}", "count": ${this.count}}`;
};
*/

//

NameCount_t.prototype.logList = function(list) {
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

/*
module.exports = {
    makeCopy,
    makeNameList,
    makeCountList,
    makeListFromNameList,
    makeListFromCsv,
    makeNameMap,
    makeCountMap,
    listToString,
    listToJSON,
    listContains,
    listContainsAll
};
*/
