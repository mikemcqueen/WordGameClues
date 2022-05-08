//
// clue.ts
//

'use strict';

import _ from 'lodash'; // TODO: need a smaller dummy import
//let Stringify = require('stringify-object');

export namespace PropertyName {
    export enum Enum {
        Synonym = "synonym",
        Homonym = "homonym"
    };

    export const Synonym = Enum.Synonym;
    export const Homonym = Enum.Homonym;

    export type Any = Enum.Synonym |  Enum.Homonym;
}

export namespace PropertyCounts {
    export interface Type {
        total: number;
        primary: number;
    };

    export type Map = {
        [key in PropertyName.Enum]: Type;
    };
}

interface Common {
    name: string;
    src: string;

    note?: string;
    ignore?: boolean;
    skip?: boolean;
    synonym?: boolean;
    homonym?: boolean;
}

// for primary sources only
interface PrimaryClue extends Common {
/*
    name: string;
    src: string;
*/
    
    num: number;
    source?: string;
    target?: string;
    implied?: string;
    require?: string;
    _?: string;

    restrictToSameClueNumber?: boolean;

    // runtime only, not in schema
    propertyCounts?: PropertyCounts.Map;
}

interface CompoundClue extends Common {
/*    
    name: string;
    src: string;
*/
}

export type Primary = PrimaryClue;
export type Compound = CompoundClue;
export type Any = PrimaryClue | CompoundClue;

//
//

export const Schema = {
    "$id": "https://wordgameclues.com/schemas/compound-clue",
    "type": "object",
    "properties": {
        "name":    { type: "string" },
        "src":     { type: "string" },

        "note":    { type: "string" },
        "ignore":  { type: "boolean" },
        "skip":    { type: "boolean" },
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
        "_":       { type: "string" },
        
        "restrictToSameClueNumber": { type: "boolean" },
    },
    // TODO: require name if ignore != true
    // TODO: enforce src=same if synonym=true
    "additionalProperties": false
};

export namespace PropertyCounts {
    export function empty() {
        return {
            total: 0,
            primary: 0
        }
    }

    export function initAll (clue: PrimaryClue): void {
        let propertyCounts = {};
        Object.values(PropertyName.Enum).forEach((propertyName: PropertyName.Any) => {
            //console.error(`propertyName: ${propertyName}`);
            propertyCounts[propertyName] = getCounts(clue, propertyName);
        });
        clue.propertyCounts = propertyCounts as Map;
    }

    export function getCounts (clue: Any, propertyName: PropertyName.Any): Type {
        const hasProperty = !!clue[propertyName];
        return {
            total: hasProperty ? 1 : 0,
            primary: hasProperty && isPrimary(clue) && !clue.source ? 1 : 0
        }
    }

    export function addAll (toClue: PrimaryClue, fromClue: PrimaryClue): void {
        //console.error(`add: to ${toClue.name} from ${fromClue.name}`);
        Object.values(PropertyName.Enum)
            .forEach((propertyName: PropertyName.Any) => {
                //console.error(`add: to[name](${toClue[propertyName]}) from[name](${fromClue[propertyName]})`);
                add(toClue.propertyCounts![propertyName], fromClue.propertyCounts![propertyName]);
            });
    }

    export function add (to: Type, from: Type | undefined): void {
        if (from) {
            to.total += from.total;
            to.primary += from.primary;
        }
    }
}

//

export function isPrimary (clue: Any): clue is PrimaryClue {
    return 'num' in clue; // (clue as PrimaryClue).num !== undefined;
}

export function isCompound (clue: Any): clue is CompoundClue {
    return !isPrimary(clue);
}

//
//

function format2 (text: string, span: number) {
    let result = "";
    for (let len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//
// uh. this is named wrong. toString() ?

export function toJSON (clue: /*PrimaryClue*/ Common, options: any = {}): string {
    let s = '{';
    if (clue.name) {
        s += ` "name": "${clue.name}", ${format2(clue.name, 15)}`;
    }
    s += `"src": "${clue.src}"`;
    // TODO: loop
    if (clue.note) {
        s += `, "note": "${clue.note}"`;
    }
    if (clue.ignore) {
        s += `, "ignore": ${clue.ignore}`;
    }
    if (clue.skip) {
        s += `, "skip": ${clue.skip}`;
    }
    if (clue.synonym) {
        s += `, "synonym": ${clue.synonym}`;
    }
    if (clue.homonym) {
        s += `, "homonym": ${clue.homonym}`;
    }
    if (isPrimary(clue)) {
        if (options.synonym) {
            s += `, "syn total": ${clue.propertyCounts!.synonym!.total}, "syn primary": ${clue.propertyCounts!.synonym!.primary}`;
        }
    }
    s += ' }';

    return s;
}
