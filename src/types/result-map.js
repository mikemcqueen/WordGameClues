//
// result-map.js
//

'use strict';

//

const _               = require('lodash');
const Debug           = require('debug')('result-map');
const Expect          = require('should/as-function');
const NameCount       = require('./name-count');

//

const PRIMARY_KEY   = '__primary';
const SOURCES_KEY   = '__sources';
const FINAL_KEY     = '__final';

//

function makeNew () {
    return new ResultMap(); //assignMethods(Object({}));
}

//
//

function ResultMap () {
    let obj = assignMethods(Object({}));

    obj.internal_map = {};
    
    return obj;
}


//
//

function assignMethods (obj) {
    obj.addPrimaryLists             = addPrimaryLists;
    obj.addResult                   = addResult;
    obj.addResultAtNcList           = addResultAtNcList;
    obj.addSourcesToPathList        = addSourcesToPathList;
    obj.getPrimaryPathList          = getPrimaryPathList;
    obj.allInAnyNameList            = allInAnyNameList;
    obj.recursiveGetPrimaryPathList = recursiveGetPrimaryPathList;
    obj.addAllPrimary               = addAllPrimary;
    obj.addPrimary                  = addPrimary;
    obj.resolvePrimary              = resolvePrimary;
    obj.merge                       = merge;
    obj.mergeNcList                 = mergeNcList;
    obj.recursiveMergeMaps          = recursiveMergeMaps;
    obj.ensureUniquePrimaryLists    = ensureUniquePrimaryLists;
    obj.dump                        = dump;
    obj.map                         = map;

    return obj;
}

//

function map () {
    return this.internal_map;
}

//
//

function addPrimaryLists (ncList, nameSrcList) {
    nameSrcList = _.clone(nameSrcList);
    ncList.forEach(nc => {
        let index;
    
        index = _.findIndex(nameSrcList, { name: nc.name });
        if (index === -1) {
            throw new Error('no ' + nc.name + ' in ' + nameSrcList);
        }
        this.map()[nc] = [ nameSrcList[index] ];
        _.pullAt(nameSrcList, [ index ]);
    });
    if (!_.isEmpty(nameSrcList)) {
        throw new Error('nameSrcList has items remaining, ' + nameSrcList);
    }
    return this;
}

//
//

//rootKey
//origNcList:
//primaryNcList:
//nameSrcList:    nameSrcList
//ncNameListPairs:buildResult.ncNameListPairs
//
function addResult (args) {
    let primaryNcList;
    let nameSrcList;
    let result;
    
    primaryNcList = _.clone(args.primaryNcList);
    nameSrcList = _.clone(args.nameSrcList);

    Debug('++addResult' +
          ', this.map.size: ' + _.size(this.map()) +
          ', rootKey: ' + args.rootKey +
          ', origNcList: ' + args.origNcList +
          indentNewline() + '  primaryNcList: ' + primaryNcList +
          indentNewline() + '  nameSrcList: ' + nameSrcList);
    this.dump();

    Expect(_.size(primaryNcList)).is.equal(_.size(nameSrcList));
    this.addAllPrimary(args.origNcList, primaryNcList, nameSrcList);
    Expect(_.size(primaryNcList)).is.equal(_.size(nameSrcList));
    /*
    if (args.rootKey) {
        this.addResultAtRoot({
            origNcList:     args.origNcList,
            primaryNcList:  primaryNcList,
            nameSrcList:    nameSrcList
        });
    }
    else */
    if (!_.isEmpty(primaryNcList)) {
        this.addResultAtNcList({
            ncList:         args.origNcList,
            primaryNcList:  primaryNcList,
            nameSrcList:    nameSrcList,
            ncNameListPairs:args.ncNameListPairs
        });
    }

    Debug('--addResult');
    this.dump();

    return this;
}

//
//ncList:
//primaryNcList:
//nameSrcList:    nameSrcList
//ncNameListPairs:buildResult.ncNameListPairs
//
function addResultAtNcList (args) {
    let pathList;

    Debug('++addResultAtNcList' +
             ', ncList: ' + args.ncList +
             indentNewline() + '  ncNameListPairs: ' + args.ncNameListPairs +
             indentNewline() + '  primaryNcList: ' + args.primaryNcList +
             indentNewline() + '  nameSrcList: ' + args.nameSrcList);

    pathList = this.getPrimaryPathList(args.primaryNcList, args.ncNameListPairs);

    if (_.isEmpty(pathList)) {
        throw new Error('empty pathList');
    }

    pathList.forEach(path => {
        Debug('path: ' + path.path +
                 ', primary: ' + path.primaryNcCsv + 
                 ', processLast: ' + path.processLast);
    });

    this.addSourcesToPathList(pathList, args.nameSrcList, args.ncNameListPairs);

    pathList.forEach(path => {
        Debug('path: ' + path.path +
                 ', primary: ' + path.primaryNcCsv +
                 ', nameSrcList: ' + path.nameSrcList);
    });

    // sources are in the pathLst, now add the items at those paths
    pathList.forEach(path => {
        let at = path.path ? 
            _.at(this.map(), path.path) :
            [ this.map() ];
        let list;
        if (_.size(at) != 1) {
            throw new Error('too much at');
        }
        at = at[0];
        Debug('at: ' + at + '(' + _.size(at) + '), typeof ' + (typeof at));
        Debug('at.keys: ' + _.keys(at));
        list = at[path.primaryNcCsv];
        list.push(_.toString(path.nameSrcList));
    });

    return this;
}

//
//

function addSourcesToPathList (pathList, nameSrcList, ncNameListPairs) {
    let passTwo = false;

    nameSrcList = _.clone(nameSrcList);

    for (;;) {
        pathList.forEach(path => {
            let index;
            if ((path.processLast && !passTwo) ||
                (!path.processLast && passTwo))
            {
                return; // forEach.continue
            }
            path.nameSrcList = [];
            if (!NameCount.makeListFromCsv(path.primaryNcCsv).every(nc => {
                index = _.findIndex(nameSrcList, { name: nc.name });
                if (index === -1) {
                    Debug('reverting, primary clue not in nameSrcList, ' + nc.name +
                          ', list: ' + nameSrcList);
                    return false; // every.exit
                }
                Debug('addSources: adding ' + nameSrcList[index] +
                         ' to ' + nc);
                path.nameSrcList.push(nameSrcList[index]);
                _.pullAt(nameSrcList, [ index ]);
                return true;  // every.continue
            })) {
                path.nameSrcList.forEach(nameSrc => {
                    nameSrcList.push(nameSrc);
                });
                path.nameSrcList = null;
            }
        });
        if (passTwo) {
            break;
        }
        passTwo = true;
    }
    if (!_.isEmpty(nameSrcList)) {
        throw new Error('items remain in nameSrcList, ' + nameSrcList);
    }
    return this;
}

//
//

function getPrimaryPathList (primaryNcList, ncNameListPairs) {
    let pathList;
    pathList = [];
    _.keys(this.map()).forEach(key => {
        let nc;
        let valid = false;
        if (_.isArray(this.map()[key])) {
            // this does a better job at filtering valid lists
            if (ncNameListPairs) {
                if (this.allInAnyNameList(
                    _.map(_.split(key, ','), NameCount.makeNew), ncNameListPairs))
                {
                    valid = true;
                }
            }
            else if (!primaryNcList ||
                     NameCount.listContains(primaryNcList, NameCount.makeNew(key))) {
                valid = true;
            }
            if (valid) {
                pathList.push({
                    path:         null,
                    primaryNcCsv: key
                });
            }
        }
        else {
            pathList = _.concat(pathList, this.recursiveGetPrimaryPathList(
                primaryNcList, key, this.map()[key]));
        }
    });
    return pathList;
}

//
//

function allInAnyNameList (ncList, ncNameListPairs) {
    Debug('++allInAnyNameList ncList: ' + ncList);
    return ncNameListPairs.some(ncNameListPair => {
        let nameList = ncNameListPair[1];
        if (_.size(nameList) != _.size(ncList)) {
            return false;
        }
        return ncList.every(nc => {
            return _.includes(nameList, nc.name);
        });
    });
}

//
//

function recursiveGetPrimaryPathList (primaryNcList, path, map) {
    let arrayFound = false;
    let pathList = [];
    let processLast = false;

    // first pass, check for multiple array keys that match primaryNcList
    if (primaryNcList) {
        _.forOwn(map, (value, key) => {
            if (_.isArray(value)) { // array means primary 
                if (NameCount.listContainsAll(primaryNcList, NameCount.makeListFromCsv(key))) {
                    if (arrayFound) {
                        processLast = true;
                    }
                    arrayFound = true;
                }
            }
        });
    }

    _.forOwn(map, (value, key) => {
        if (_.isArray(value)) { // array means primary 
            if (!primaryNcList ||
                NameCount.listContainsAll(primaryNcList, NameCount.makeListFromCsv(key)))
            {
                /*
                if (primaryNcList && arrayFound) {
                    throw new Error('two potential primary list matches in ' + path);
                }
                */
                pathList.push({
                    path:         path,
                    primaryNcCsv: key,
                    processLast:  processLast
                });
                arrayFound = true;
            }
        }
        else {
            pathList = _.concat(pathList, this.recursiveGetPrimaryPathList(
                primaryNcList, path + '.' + key, map[key]));
        }
    });
    return pathList;
}


/*
//
//rootKey
//origNcList:
//primaryNcList:
//nameSrcList:
//
function addResultAtRoot (args) {
    let map;
    let listKey;
    let list;

    // ensure root key exists
    if (!this.map()[args.rootKey]) {
        throw new Error('missing rootKey, ' + args.rootKey);
    }
    map = this.map()[args.rootKey];
    listKey = _.toString(args.primaryNcList);

    if (!map[listKey]) {
        throw new Error('rootKey: ' + args.rootKey +
                        ', missing list: ' + listKey);
    }

    list = map[listKey];
    if (!_.isArray(list)) {
         throw new Error('rootKey: ' + args.rootKey +
                        ' list: ' + listKey +
                        ' is not an array, type: ' + (typeof list));
    }
    list.push(_.toString(args.nameSrcList));
}
*/

//
//

function addAllPrimary (origNcList, mutatingPrimaryNcList, mutatingNameSrcList) {
    Debug('++addAllPrimary' +
             ', origNcList: ' + origNcList +
             indentNewline() + '  mutatingPrimaryNcList: ' + mutatingPrimaryNcList +
             indentNewline() + '  mutatingNameSrcList: ' + mutatingNameSrcList);

    // find primary NCs in the orignal NC list
    origNcList.forEach((nc, ncIndex) => {
        let primaryIndex;
        let nameSrc;

        if (nc.count > 1) {
            return; // forEach.continue
        }
        // is original primary NC in supplied mutatingPrimaryNcList?
        if (!_.includes(mutatingPrimaryNcList, nc)) {
            return; // forEach.continue
        }

        nameSrc = _.find(mutatingNameSrcList, ['name', nc.name]);
        if (!nameSrc) {
            throw new Error('no ' + nc.name + ' in ' + mutatingNameSrcList);
        }

        // add the primary NC to map
        this.addPrimary( _.toString(nc), _.toString(nameSrc));
        primaryIndex = _.indexOf(mutatingPrimaryNcList, nc);
        if (primaryIndex === -1) {
            throw new Error('alternate universe, nc: ' + nc +
                            ', mutatingPrimaryNcList: ' + mutatingPrimaryNcList);
        }

        // remove the primary NC and it's corresponding name:src from
        // the supplied lists
        _.pullAt(mutatingPrimaryNcList, [primaryIndex]);
        _.pullAt(mutatingNameSrcList, _.indexOf(mutatingNameSrcList, nameSrc));
    });

    Debug('--addAllPrimary' +
             indentNewline() + '  primaryNcList: ' + mutatingPrimaryNcList +
             indentNewline() + '  nameSrcList: ' + mutatingNameSrcList);
    this.dump();

    return this;
}

//
//

function addPrimary (ncPrimaryStr, nameSrcStr) {
    let list;

    if (!this.map()[ncPrimaryStr]) {
        if (!this.resolvePrimary(ncPrimaryStr)) {
            throw new Error('failure to resolve pending primary, ' + ncPrimaryStr +
                            ', for: ' + nameSrcStr);
        }
    }
    list = this.map()[ncPrimaryStr] = [];
    if (!_.isArray(list)) {
        throw new Error('pending primary list, ' + list +
                        ' is not an array, type: ' + (typeof list));
    }
    Debug('addPrimary: adding ' + nameSrcStr +
          ' to ' + ncPrimaryStr);
    list.push(nameSrcStr);
    return this;
}

//
//

function resolvePrimary (ncPrimaryStr) {
    let primaryNcStrList;
    let index;

    primaryNcStrList = this.map()[PRIMARY_KEY];
    if (!primaryNcStrList || _.isEmpty(primaryNcStrList)) {
        Debug('missing or empty unresolved primary list, ' + primaryNcStrList +
                 ', for nc: ' + ncPrimaryStr);
        Debug('keys: ' + _.keys(this.map()));
        return false;
    }

    index = _.indexOf(primaryNcStrList, ncPrimaryStr);
    if (index === -1) {
        Debug('nc not in unresolved list, ' + ncPrimaryStr +
                 ', list: ' + primaryNcStrList);
        return false;
    }

    Debug('found unresolved pending primary nc: ' + ncPrimaryStr +
             ', at index: ' + index);

    _.pullAt(primaryNcStrList, [ index ]);
    if (_.isEmpty(primaryNcStrList)) {
        delete this.map()[PRIMARY_KEY];
    }
    return true;
}

//
//

function merge (fromMap, ncList) {
    let rootKey;
    let subKey;
    let key;
    let map;
    let list;

    Debug('++merge' + ', ncList: ' + ncList);
    Debug('before resultMap:');
    this.dump();
    Debug('before fromMap:');
    fromMap.dump();

    if (ncList) {
        this.mergeNcList(fromMap, ncList);
    }
    else {
        this.recursiveMergeMaps(this.map(), fromMap.map());
    }

    Debug('--merge');
    Debug('  after resultMap:');
    this.dump();

    return this;
}

//
//

function mergeNcList (fromMap, ncList) {
    Debug('++mergeNcList' +
             ', this.keys: ' + _.keys(this.map()) +
             ', fromMap.keys: ' + _.keys(fromMap.map()) +
             ', ncList: ' + ncList);

    ncList.forEach(nc => {
        let map;
        let srcNameList;
        let keys;

        map = this.map()[nc];
        if (!map) {
            throw new Error('resultMap missing nc, ' + nc);
        }
        srcNameList = map[SOURCES_KEY];
        if (!srcNameList || _.isEmpty(srcNameList)) {
            throw new Error('missing or empty srcNameList, ' + srcNameList);
        }
        keys = _.keys(fromMap.map());
        if (_.isEmpty(keys)) {
            throw new Error('empty keys');
        }
        keys.forEach(ncStr => {
            let keyNc = NameCount.makeNew(ncStr);
            let index;
            index = _.indexOf(srcNameList, keyNc.name);
            if (index > -1) {
                // copy sub-map from fromMap to resultMap
                map[ncStr] = fromMap.map()[ncStr];
                // delete sub-map in fromMap;
                delete fromMap.map()[ncStr];
                // delete key from srcNameList
                _.pullAt(srcNameList, [index]);
            }
        });

        if (!_.isEmpty(srcNameList)) {
            throw new Error('srcNameList has remaining items' +
                            ', size: ' + _.size(srcNameList) + 
                            ', length: ' + srcNameList.length +
                            ', list: ' + srcNameList);
        }
        // delete sources key from resultMap
        delete map[SOURCES_KEY];
    });
    if (!_.isEmpty(fromMap.map())) {
        throw new Error('fromMap has remaining keys, ' + _.keys(fromMap.map()));
    }
    return this;
}

//
//

function recursiveMergeMaps (toSeq, fromSeq) {
    // empty resultSeq, add everything from pendingSeq
    if (_.isEmpty(toSeq)) {
        _.forEach(fromSeq, (value, key) => {
            toSeq[key] = value;
        });
        return;
    }

    if (_.isArray(fromSeq) != _.isArray(toSeq)) {
        throw new Error('array/object type mismatch');
    }

    if (_.isArray(toSeq)) {
        // ???
    }
    else {
        _.forOwn(fromSeq, (value, key) => {
            if (_.has(toSeq, key)) {
                this.recursiveMergeMaps(toSeq[key], fromSeq[key]);
            }
            else {
                toSeq[key] = value;
            }
        });
    }
}

//
//

function ensureUniquePrimaryLists () {
    let pathList;
    pathList = this.getPrimaryPathList();
    pathList.forEach(path => {
        let at = path.path ? _.at(this.map(), path.path) : [ this.map() ];
        let list;
        if (_.size(at) !== 1) {
            throw new Error('too much at, ');
        }
        at = at[0];

        Debug('at: ' + at + '(' + _.size(at) + '), typeof ' + (typeof at));
        Debug('at.keys: ' + _.keys(at));

        list = at[path.primaryNcCsv];
        at[path.primaryNcCsv] = _.uniq(list);
    });
    return this;
}

//
//

function dump () {
    dumpMap(this.map());
    return this;
}

// args:
//  header:
//  result:
//  primary:  t/f
//  compound: t/f
//
function dumpMap (seq, level) {
//    console.log(_(seq).toJSON());
//    return;
    if (!level) level = 0;
    if (typeof seq === 'object') {
        Debug('' + indent() + spaces(2 * level) + (_.isArray(seq) ? '[' : '{'));
        ++level;

        if (_.isArray(seq)) {
            seq.forEach(elem => {
                if (typeof elem === 'object') {
                    dumpMap(elem, level + 1);
                }
                else {
                    Debug('' + indent() + spaces(2*level) + elem);
                }
            });
        }
        else {
            _.forOwn(seq, function(value, key) {
                if (typeof value === 'object') {
                    Debug('' + indent() + spaces(2 * level) + key + ':');
                    dumpMap(value, level + 1);
                }
                else {
                    Debug('' + indent() + spaces(2 * level) + key + ': ' + value);
                }
            });
        }
        --level;
        Debug('' + indent() + spaces(2 * level) + (_.isArray(seq) ? ']' : '}'));
    }
}

function indent () {
    return ' '; //spaces(this.logLevel);
}

function indentNewline () {
    return '\n' + indent();
}

function spaces (length) {
    let count;
    let result = '';
    for (count = 0; count < length; result += ' ', count++);
    return result;
}

module.exports = {
    makeNew                : makeNew,
    dumpMap                : dumpMap
};
