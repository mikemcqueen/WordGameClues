//
// clue-list.ts
//

'use strict';

//

const _         = require('lodash');
const Ajv       = require("ajv")
const Debug     = require('debug')('clue-list');
const Expect    = require('should/as-function');
const Fs        = require('fs-extra');
const Stringify = require('stringify-object');

import * as Clue from './clue';

type ClueList = Clue.Type[];
export type Type = ClueList;

type PrimaryClueList = Clue.PrimaryType[];
export type PrimaryType = PrimaryClueList;

export const Schema = {
    "$id": "https://wordgameclues.com/schemas/clue-list",
    "type": "array",
    "items": { "$ref": "https://wordgameclues.com/schemas/clue" }
};

export const PrimarySchema = {
    "$id": "https://wordgameclues.com/schemas/primary-clue-list",
    "type": "array",
    "items": { "$ref": "https://wordgameclues.com/schemas/primary-clue" }
};


const ajv = new Ajv({schemas: [Clue.Schema, Clue.PrimarySchema, Schema, PrimarySchema]});
const validateList = ajv.compile(Schema);
const validatePrimaryList = ajv.compile(PrimarySchema);
//const serialize = ajv.compileSerializer(Clue.ListSchema);
//const parse = ajv.compileParser(Clue.ListSchema);

//
//

function display () {
    let arr: string[] = [];
    
    this.forEach(function(clue) {
        arr.push(clue.name);
    });
    console.log(arr.toString());
    
    return this;
}

//
//

function persist (filename: string) {
    if (Fs.exists(filename)) {
        throw new Error('file already exists: ' + filename);
    }
    Fs.writeFileSync(filename, this.toJSON());
}

//

function toJSON (): string {
    let result = '[\n';
    let first = true;

    this.forEach(clue => {
        if (!first) {
            result += ',\n';
        }
        else { 
            first = false;
        }
        result += `  ${Clue.toJSON(clue)}`;
    });
    if (!first) {
        result += '\n';
    }
    result += ']';

    return result;
}

//

function save (filename: string) {
    Fs.writeFileSync(filename, this.toJSON(), { encoding: 'utf8' });
}

//

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

function getSameSrcList (startIndex, options: any = {}) {
    Expect(startIndex).is.a.Number();
    let list = makeNew();
    let mismatch = -1;
    if (startIndex >=0 && startIndex < this.length) {
        let src = this[startIndex].src;
        list.push(..._.filter(this, (clue, index) => {
            if (index < startIndex) return false;
            if (mismatch >= 0 && !options.allowMismatch) return false;
            if (clue.src === src) return true;
            if (mismatch < 0) mismatch = index;
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
    Expect(actualClue.src).is.a.String();
    clue.actual = actualClue.src;
    return clue;
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

function sameSrcMergeFrom (fromList, options: any = {}) {
    Debug(`fromList: ${Stringify(fromList)}, to this: ${Stringify(this)}`);
    Expect(fromList.length).is.aboveOrEqual(this.length);
    let warnings = 0;
    for (let [index, clue] of this.entries()) {
        Expect(clue.name)//, `name mismatch, from ${fromList[index].name} to ${clue.name}`)
            .is.equal(fromList[index].name);
        warnings += clueMergeFrom(clue, fromList[index], options);
        this[index] = clue;
    }
    // append remaing non-matching-name clues, but don' allow duplicate names
    for (let clue of fromList.slice(this.length, fromList.length)) {
        // check if clue.name already exists in this list
        Expect(_.find(this, ['name', clue.name])).is.undefined(); // TODO: define/use Clue.NAME
        Expect(options.src).is.a.String();
        clue = clueSetActual(_.clone(clue), clue);
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
    while (toIndex >= 0 || fromIndex >= 0) {
        let [sameSrcFromList, nextFromIndex] = fromList.getSameSrcList(fromIndex, options);
        console.log(`from: ${sameSrcFromList.map(o => o.name)}`);
        let [sameSrcToList, nextToIndex] = this.getSameSrcList(toIndex, options);
        // use the src from TO list if available
        if (!_.isEmpty(sameSrcToList)) {
            srcNum = _.toNumber(sameSrcToList[0].src);
        } else {
            srcNum += 1;
        }
        Expect(srcNum).is.above(0);
        //console.log(`to: ${sameSrcToList.toJSON()}`);
        let mergeOptions = { src: srcNum.toString() };
        let [sameSrcMerged, mergeWarnings] = sameSrcToList.sameSrcMergeFrom(sameSrcFromList, mergeOptions);
        merged.push(...sameSrcMerged);
        warnings += mergeWarnings;
        
        toIndex = nextToIndex;
        fromIndex = nextFromIndex;
    }
    Expect(toIndex).is.equal(-1); // looks like toList is bigger than fromlist, manually fix it
    Expect(fromIndex).is.equal(-1); // pigs fly
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

function objectFrom (args: any) {
    let clueList: ClueList = [];

    if (args.filename) {
        try {
            const json = Fs.readFileSync(args.filename, 'utf8');
            clueList = JSON.parse(json);
            //console.error(`${args.filename}, primary=${args.primary}`);
            const validate =  args.primary ? validatePrimaryList : validateList;
            const valid = validate(clueList);
            if (!valid) {
                console.error(validate.errors);
                throw new Error(`invalid json`);
            }
            /*
            clueList = parse(json);
            if (clueList === undefined) {
                console.log(parse.message); // error message from the last parse call
                console.log(` position(${parse.position})`); // error position in string
            }
            */
        }
        catch(e) {
	    throw new Error(`${args.filename}, ${e}`);
        }
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
    return Object(clueList); // TODO:  why Object() ?
}

//
//

export function makeNew () {
    return assignMethods(Object([]));
}

// makeFrom()
//
// args:
//   filename: filename string
//   optional: optional flag suppresses file error
//   array:    js array

export function makeFrom (args: any) {
    let object = objectFrom(args);
    return assignMethods(Object(object)).init();
}

//

module.exports = {
    makeNew:  makeNew,
    makeFrom: makeFrom,
    makeKey:  makeKey
};

