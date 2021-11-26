/*
 * words.ts
 */

'use strict';

import * as _ from "lodash";
import _Debug from "debug";
const Debug = _Debug("words");
import * as Es from "event-stream";
import * as Fs from "fs";

const skipped = new Set();
let total_skipped = 0;
let total_duplicates = 0;

//
//
//

function base (word) {
    // strip l' prefix
    const l_apo = 'l\'';
    if (_.startsWith(word, l_apo)) {
        return word.slice(l_apo.length, word.length);
    }

    // strip ' suffix from words with allowed preceding character
    // NOTE: doesn't seem to be working. 
    const apo_suffix_chars = 'ns';
    if (_.endsWith(word, '\'') &&
        _.includes(apo_suffix_chars, word.charAt(word.length - 1))) {

        return word.slice(0, word.length - 1);
    }

    // strip suffixes
    const exts = ['\'s', 'n\'t', '\'ve'];
    for (const ext of exts) {
        if (_.endsWith(word, ext)) {
            return word.slice(0, word.length - ext.length);
        }
    }
    return word;
}

function keep (word) {
    //const allowed = [];
    // keep allowed words
    //if (_.includes(allowed, word)) return true;

    // keep o'prefix
    if (_.startsWith(word, 'o\'') && word.length > 2) return true;

    // don't keep remaining contractions
    if (_.includes(word, '\'')) return false;

    // keep everything else
    return true;
}

function valid (word) {
    if (keep(word)) return true;
    Debug(`skip: ${word}`);
    skipped.add(word);
    total_skipped += 1;
    return false;
}

function clean (word) {
    Debug(`clean in : ${word}`);
    const repl = _.flatMap(word.replace(/[0-9,.:;\"\?\!\(\)_]+/g, ' ').split(' '), x => x)
              .filter(x => !_.isEmpty(x))
              .map(x => base(x))
              .filter(x => valid(x));
    Debug(`clean out: ${repl}`);
    return repl;
}

function get_words (line) {
    let words = _.flatMap(line.split(' '), word => word.split('-'));
    Debug(`words: ${words}, ${words.length}`);
    return _.flatMap(words, word  => clean(_.toLower(word)));
}

interface Dict {
    dict: Set<string>;
}

interface DictResult {
    dict: Set<string>;
    result: Array<string>;
};

type LoadArg = /*Dict |*/ DictResult;
type LoadFunc = (line: string, arg: LoadArg) => void;

// TS: restrict arg type
let load = (filename: string, func: LoadFunc, arg: LoadArg): Promise<LoadArg> => {
    return new Promise((resolve, reject) => {
        const s = Fs.createReadStream(filename)
            .pipe(Es.split())
            .pipe(Es.mapSync(line => {
                
                // pause the readstream
                s.pause();
                
		func(line, arg);

                // resume the readstream, possibly from a callback
                s.resume();
            })).on('error', err => {
                console.error('Error while reading file.', err);
                reject();
            }).on('end', () => {
                Debug('Read entire file.');
                resolve(arg);
            });
    });
};

let load_dict_line_func = (line: string, arg: Dict): void => {  
    const words = get_words(line);
    words.forEach(word => {
	arg.dict.add(word);
    });
};
                
export let load_dict = (filename: string) : Promise<Set<string>> => {
    return load(filename, load_dict_line_func, { dict: new Set<string>(), result: [] /*NOTE WOULD LIKE TO REMOVE*/ })
	.then(obj => obj.dict);
};

let load_anresult_line_func = (line: string, arg: DictResult): void => {
    const words = get_words(line);
    for (const word of words) {
	if (!arg.dict.has(word)) return;
    }
    arg.result.push(line);
};
                
export let load_anresult = (filename: string, dict: Set<string>): Promise<Array<string>> => {
    if (!dict) throw new Error(`bad dict`);
    return load(filename, load_anresult_line_func, { dict, result: [] })
	.then(obj => obj.result);
};

module.exports = {
    load_dict,
    load_anresult
};
