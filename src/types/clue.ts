//
// clue.ts
//

'use strict';

import _ from 'lodash'; // TODO: need a smaller dummy import

export namespace CountedProperty {
    export enum Enum {
        Synonym = "synonym",
        Homonym = "homonym"
    };

//    export type Name = "synonym" | "homonym"; // TODO: Enum.Synonym |  Enum.Homonym;

    export interface Counts {
        total: number;
        primary: number;
    };

    export type Map = {
        [key in Enum]: Counts;
    };

    //export type Map = Record<CountedProperty.Name, PropertyCount.Type>;
}

interface Common {
    note?: string;
    ignore?: boolean;
    skip?: boolean;
    synonym?: boolean;
    homonym?: boolean;

    propertyCounts?: CountedProperty.Map;
}

interface Clue extends Common {
    name: string;
    src: string;
}

// for primary sources only
interface PrimaryClue extends Common {
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

export namespace CountedProperty {
    export function initAll (clue: PrimaryClue): void {
        let propertyCounts = {};
        for (let propertyName of Object.values(CountedProperty.Enum)) {
            //console.error(`propertyName: ${propertyName}`);
            propertyCounts[propertyName] = get(clue, propertyName);
        }
        clue.propertyCounts = propertyCounts as Map;
    }

    function get (clue: PrimaryClue, property: Enum): Counts {
        const hasProperty = Boolean(clue[property]);
        return {
            total: hasProperty ? 1 : 0,
            primary: hasProperty && !clue["sources"] ? 1 : 0
        }
    }

    export function addAll (toClue: PrimaryClue, fromClue: PrimaryClue): void {
        //console.error(`add: to ${toClue.name} from ${fromClue.name}`);
        Object.values(Enum).forEach((propertyName: string) => {
            //console.error(`add: to[name](${toClue[propertyName]}) from[name](${fromClue[propertyName]})`);
            add(toClue.propertyCounts![propertyName], fromClue.propertyCounts![propertyName]);
        });
    }

    export function add (to: Counts, from: Counts | undefined): void {
        if (from) {
            to.total += from.total;
            to.primary += from.primary;
        }
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

export function toJSON (clue: Clue, options: any = {}): string {
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
    if (options.synonym) {
        s += `, "syn total": ${clue.propertyCounts!.synonym!.total}, "syn primary": ${clue.propertyCounts!.synonym!.primary}`;
    }
    s += ' }';

    return s;
}
