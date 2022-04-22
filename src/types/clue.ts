//
// clue.ts
//

'use strict';

import _ from 'lodash';

type CountedPropertyName = "synonym" | "homonym";

enum CountedPropertyEnum {
    Synonym = "synonym",
    Homonym = "homonym"
}

interface TotalPrimary {
    total: number;
    primary: number;
}

// try instead:  interface { [key: CountedPropertyName]: TotalPrimary }
export type PropertyCountMap = Record<CountedPropertyName, TotalPrimary>;

interface ClueCommon {
    note?: string;
    ignore?: boolean;
    skip?: boolean;
    synonym?: boolean;
    homonym?: boolean;

    propertyCounts?: PropertyCountMap;
}

interface Clue extends ClueCommon {
    name: string;
    src: string;
}

// for primary sources only
interface PrimaryClue extends ClueCommon {
    name?: string;
    src?: string;

    num?: string | number;
    source?: string;
    target?: string;
    implied?: string;
    require?: string;
    _?: string;

    restrictToSameClueNumber: boolean;
}

export type Type = Clue;
export type PrimaryType = PrimaryClue;

//
//

export const Schema = {
    "$id": "https://wordgameclues.com/schemas/clue",
    "type": "object",
    "properties": {
        "name":    { type: "string" },
        "src":     { type: "string" },

        "note":    { type: "string" },
        "ignore":  { type: "boolean" },
        "skip":  { type: "boolean" },
        "synonym": { type: "boolean" },
        "homonym": { type: "boolean" },
    },
    "required": ["name", "src"],
    "additionalProperties": false
};

export const PrimarySchema = {
    "$id": "https://wordgameclues.com/schemas/primary-clue",
    "type": "object",
    "properties": {
        // name/src not required in primary clue
        "name":    { type: "string" },
        "src":     { type: "string" },

        "note":    { type: "string" },
        "ignore":  { type: "boolean" },
        "skip":    { type: "boolean" },
        "synonym": { type: "boolean" },
        "homonym": { type: "boolean" },
        
        // for primary sources only

        "num":     { type: "string" } ,
        "source":  { type: "string" },
        "target":  { type: "string" },

        "implied": { type: "string" },
        "require": { type: "string" },
        "_": { type: "string" },
        
        "restrictToSameClueNumber": { type: "boolean" },
    },
    // TODO: require name if ignore != true
    // TODO: enforce src=same if synonym=true
    "additionalProperties": false
};

function initTotalPrimary(clue: PrimaryClue, propertyName: CountedPropertyName): TotalPrimary {
    //console.error(`init: ${clue.name}`);
    return {
        total: clue[propertyName] ? 1 : 0,
        primary: clue["sources"] ? 0 : 1
    }
}

function addTotalPrimary(toTotalPrimary: TotalPrimary, fromTotalPrimary: TotalPrimary): void {
    toTotalPrimary.total += fromTotalPrimary.total;
    toTotalPrimary.primary += fromTotalPrimary.primary;
}

export function addAllPropertyCounts(toClue: PrimaryClue, fromClue: PrimaryClue): void {
    //console.error(`add: to ${toClue.name} from ${fromClue.name}`);
    Object.values(CountedPropertyEnum).forEach((propertyName: string) => {
        //console.error(`add: to[name](${toClue[propertyName]}) from[name](${fromClue[propertyName]})`);
        addTotalPrimary(toClue.propertyCounts![propertyName], fromClue.propertyCounts![propertyName]);
    });
}

export function initPropertyCounts (clue: PrimaryClue): void {
    clue.propertyCounts = {
        synonym: {
            total: 0, primary: 0
        },
        homonym: {
            total: 0, primary: 0
        }
    };
    for (let propertyName of Object.values(CountedPropertyEnum)) {
        //console.error(`propertyName: ${propertyName}`);
        clue.propertyCounts![propertyName] = initTotalPrimary(clue, propertyName);
    }
}

//
//

function format2 (text: string, span: number) {
    let result = "";
    for (let len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//

export function toJSON (clue: Clue): string {
    let s = '{';
    if (clue.name) {
        s += ` "name": "${clue.name}", ${format2(clue.name, 15)}`;
    }
    s += `"src": "${clue.src}"`;
    // TODO: loop
    if (clue.note) {
        s+= `, "note": "${clue.note}"`;
    }
    if (clue.ignore) {
        s+= `, "ignore": ${clue.ignore}`;
    }
    if (clue.skip) {
        s+= `, "skip": ${clue.skip}`;
    }
    if (clue.synonym) {
        s+= `, "synonym": ${clue.synonym}`;
    }
    if (clue.homonym) {
        s+= `, "homonym": ${clue.homonym}`;
    }
    s += ' }';

    return s;
}

