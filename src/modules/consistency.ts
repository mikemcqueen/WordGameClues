//
// consistency.ts
//

'use strict';

import _ from 'lodash'; // import statement to signal that we are a "module"

const Native      = require('../../../build/experiment.node');

const Assert      = require('assert');
const Debug       = require('debug')('consistency');
const Duration    = require('duration');
const PrettyMs    = require('pretty-ms');
//const stringify   = require('javascript-stringify').stringify;
//const Stringify2  = require('stringify-object');

//import * as Clue from '../types/clue';
import * as ClueList from '../types/clue-list';
import * as ClueManager from './clue-manager';
//import * as NameCount from '../types/name-count';
import * as PreCompute from './cm-precompute';
//import * as Sentence from '../types/sentence';
//import * as Source from './source';

let get_unique_combos = (first: number, last: number): Set<string> => {
    let combos = new Set<string>();
    for (let i = first; i <= last; ++i) {
        const clues = ClueManager.getClueList(i) as ClueList.Compound;
        if (!clues) continue;
        for (let clue of clues) {
            if (!combos.has(clue.src)) {
                combos.add(clue.src);
            }
        }
    }
    return combos;
}

export const check = (options: any): void => {
    const max_sources = 30;
    let combos = get_unique_combos(2, max_sources);
    let inconsistent_combos = new Set<string>();
    for (let combo of combos) {
        const nameList = combo.split(',').sort();
        const pc_args = {
            xor: nameList,
            merge_only: true,
            max: 2,
            max_sources,
            quiet: true,
            ignoreErrors: options.ignoreErrors
        };
        const pc_result = PreCompute.preCompute(2, max_sources, pc_args);
        if (pc_result) {
            if (!Native.checkClueConsistency(nameList)) {
                inconsistent_combos.add(combo);
            }
        } else {
            console.error(`Consistency::Precompute failed at ${combo}.`);
        }
        //console.error(combo);
    }
    if (inconsistent_combos.size) {
        console.error(`inconsistent combos:`);
        for (let combo of inconsistent_combos) {
            console.error(combo);
        }
    } else {
        console.error(`No inconsistent combos`);
    }
};
