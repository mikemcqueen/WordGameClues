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

export function makeNew (name: string, count?: number, index?: number): NameCount {
    return new NameCount_t(name, count, index);
}

function makeCopy(nc: NameCount): NameCount {
    return new NameCount_t(nc.name, nc.count, nc.index);
}

function NameCount_t (name: string, count?: number, index?: number) {
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
        let splitList = name.split(':');
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

export function makeListFromCsv (csv: string): NameCount[] {
    return listFromStrList(csv.split(','));
}

//

export function nameListFromCsv (csv: string): string[] {
    return nameListFromStrList(csv.split(','));
}

//

export function listFromStrList (strList: string[]): NameCount[] {
    return strList.map(ncStr => new NameCount_t(ncStr));
/*
    let ncList: NameCount[] = nameList.map(nameOrNcStr => {
        ncList.push(new NameCount_t(name));
    });
    return ncList;
*/
}

export function nameListFromStrList (strList: string[]): string[] {
    return listFromStrList(strList).map(nc => nc.name);
}

//

export function listToNameList (ncList: NameCount[]): string[] {
    return ncList.map(nc => nc.name);
}

//

export function listToCountList (ncList: NameCount[]): number[] {
    return ncList.map(nc => nc.count);
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

export function listToString (ncList: List): string {
    if (!ncList) return _.toString(ncList);
    return ncList.map(nc => nc.toString()).toString();
}

//

function listToJSON (ncList: List) {
    let s: string;

    if (!ncList.length) { return '[]'; }

    s = '[';
    ncList.forEach((nc, index) => {
        if (index > 0) {
            s += ',\n';
        }
        s += toJSON(nc);
    });
    s += ']';
    return s;
}

//

export function listContains (ncListContains: List, nc: NameCount) {
    return _.find(ncListContains, ['name', nc.name, 'count', nc.count]);
}

//
    
export function listContainsAll (ncListContains: List, ncList: List) {
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

NameCount_t.prototype.setIndex = function(index: number) {
    this.index = index;
};

//

NameCount_t.prototype.equals = function(nc: NameCount) {
    return (nc.count === this.count) && (nc.name == this.name);
}

//

let toJSON = function (nc: NameCount): NameCount {
    return { name: `"${nc.name}"`, count: nc.count };
};

//

NameCount_t.prototype.logList = function (list: List) {
    let str: string = '';

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
