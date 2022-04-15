//
// clue.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"


interface ClueCommon {
    note?: string;
    ignore?: boolean;
    skip?: boolean;
    synonym?: boolean;
    homonym?: boolean;
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

//
//

function format2 (text: string, span: number) {
    let result = "";
    for (let len = text.toString().length; len < span; ++len) { result += " "; }
    return result;
}

//

export function toJSON (clue: Clue) {
    let s = '{';

    if (clue.name) {
        s += ` "name": "${clue.name}", ${format2(clue.name, 15)}`;
    }
    s += `"src": "${clue.src}"`;
    if (clue.note) {
        s+= `, "note": "${clue.note}"`;
    }
    if (clue.ignore) {
        s+= `, "ignore": ${clue.ignore}`;
    }
    else if (clue.skip) {
        s+= `, "skip": ${clue.skip}`;
    }
    s += ' }';

    return s;
}
