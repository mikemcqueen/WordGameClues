// pairs.ts

import * as Json from './json';
import * as Remaining from "./remaining";
import * as Solutions from "./solutions";
const Assert = require('assert');
const Stringify = require("javascript-stringify").stringify;
const StringifyObj = require("stringify-object");

export const Options = [
    [ 'w', 'words', 'words from words.json' ],
    [ 's', 'single-solutions', 'single-word solution words' ],
    [ 'm', 'multi-solutions',  'multi-word solution words' ],
    [ 'a', 'all-solutions',    'all solutions words' ],
    [ 'o', 'other=FILE+',      'words from FILE' ]
];

export const show_help = (): void => {
    console.log('Usage: node cm pairs [-w] [-s] [-m] [-a] [-o <words-file>]...');
    console.log('\nGenerate pairs from words in words.json, solutions.json, and/or another words file.');
};

// flag to indicate that words from this source come from solution words and
// do not consist of letters that must exist in "remaining" letters. 
const SolutionFlag = 0x10;
const WordSourceId = Object.freeze({ 
    Words: 1,
    Other: 2,
    SingleSolutions: SolutionFlag + 1,
    MultiSolutions: SolutionFlag + 2,
    AllSolutions: SolutionFlag + 3
});

const concat = (first: string, second: string): string => {
    let result = first;
    if (result.length > 0) {
        result += ' ';
    }
    return result + second;
};

const get_word_source_ids = (options: any): number[] => {
    const option_id_map = Object.freeze({
        'words': WordSourceId.Words,
        'single-solutions': WordSourceId.SingleSolutions,
        'multi-solutions': WordSourceId.MultiSolutions,
        'all-solutions': WordSourceId.AllSolutions
    });
    const ids: number[] = [];
    for (const option in option_id_map) {
        if (options[option]) {
            ids.push(option_id_map[option]);
        }
    }
    if (options.other) {
        for (let other of options.other) {
            ids.push(WordSourceId.Other);
        }
    }
    return ids;
};

type SolutionWordType = {
    single?: boolean;
    multi?: boolean;
};

const get_word_type = (word: string): SolutionWordType => {
    return word.indexOf(' ') === -1 ? { single: true } : { multi: true };
};

const is_word_type_allowed = (word_type: SolutionWordType,
                              allowed_word_type: SolutionWordType): boolean => {
    return (word_type.single! && allowed_word_type.single!) ||
        (word_type.multi! && allowed_word_type.multi!);
};

const get_solution_words = (word_type: SolutionWordType): string[] => {
    const result: string[] = [];
    const solutions = Solutions.get_filtered();
    const words: string[] = Array.from(solutions.keys());
    for (const word of words) {
        if (is_word_type_allowed(get_word_type(word), word_type)) {
            result.push(word);
        }
    }
    return result;
};

const is_solution_source_id = (src_id: number): boolean => {
    return (src_id & SolutionFlag) !== 0;
};

type WordList = {
    src_id: number;
    words: string[];
};

const get_word_list = (src_id: number, filename?: string): WordList => {
    let words: string[] = [];
    switch (src_id) {
        case WordSourceId.Words:
            words = Json.load('words.json');
            break;
        case WordSourceId.Other:
            Assert(filename);
            words = Json.load(filename!);
            break;
        case WordSourceId.SingleSolutions:
            words = get_solution_words({ single: true });
            break;
        case WordSourceId.MultiSolutions:
            words = get_solution_words({ multi: true });
            break;
        case WordSourceId.AllSolutions:
            words = get_solution_words({ single: true, multi: true });
            break;
        default:
            throw new Error(`Invalid word source id: ${src_id}`);
    }
    return { src_id, words };
};

const get_word_lists = (src_ids: number[], other?: string[]): [WordList, WordList|null] => {
    let word_lists: WordList[] = [];
    let other_idx = 0;
    for (const src_id of src_ids) {
        word_lists.push(get_word_list(src_id, other?.[other_idx]));
        if (src_id === WordSourceId.Other) ++other_idx;
    }
    return [word_lists[0], word_lists.length > 1 ? word_lists[1] : null];
};

const show_pairs = (list1: string[], list2: string[], letter_counts: Remaining.LetterCounts): number => {
    let count = 0;

    let used_words = new Set<string>();
    for (const word1 of list1) {
        used_words.add(word1);
        let remaining1 = Remaining.remove_letters(letter_counts, word1);
        // Some words in list1 may be "solution" (e.g. to folder names) words. in which case,
        // remove_letters() may (and hopefully will) fail, because the letters of the word
        // didn't actually come from remaining letters. In that case, just skip testing the
        // first word and only test the second word for remaining letters.
        // TODO:
        // Note that this is kind of a dumb way to determine that a word is a solution word,
        // because what if a solution word actually does contain only letters that match
        // remaining letters? Then we're kinda fucked and producing bad pairs. I need to be
        // able to signal somehow that a word (or list of words) is a solution word.
        let remaining2_required = false;
        if (!remaining1) {
            remaining1 = letter_counts;
            remaining2_required = true;
        }
        for (const word2 of list2) {
            if (used_words.has(word2)) continue;
            let remaining2 = Remaining.remove_letters(remaining1, word2);
            if (remaining2) {
                console.log(`${word1} ${word2}`);
                count += 1;
            } else if (remaining2_required) {
                Assert(remaining2, `${word1} ${word2}`);
            }
        }
    }
    return count;
};

const show_all_pairs = (args: string[]): number => {
    const remaining = Remaining.letter_counts();
    const words: string[] = Json.load('words.json');
    // Assert(validate_letters(words, remaining));
    let words1 = words;
    let words2 = words;
    if (args.length) {
        words1 = Json.load(args[0]);
    }
    // TODO: do something with "solution" words
    return show_pairs(words1, words2, remaining);
};

const is_all_solution_source_ids = (src_ids: number[]): boolean => {
    for (const id of src_ids) {
        if (!is_solution_source_id(id)) return false;
    }
    return true;
};

const get_word_type_from_source_id = (src_id: number): SolutionWordType => {
    switch (src_id) {
        case WordSourceId.SingleSolutions: return { single: true };
        case WordSourceId.MultiSolutions:  return { multi: true };
        case WordSourceId.AllSolutions:    return { single: true, multi: true };
        default: throw new Error(`Invalid word source id, ${src_id}`);
    }
};

const has_any = (target: Set<string>|undefined, source: Set<string>|undefined): boolean => {
    if (!target || !source) return false;
    for (const key of source) {
        if (target.has(key)) return true;
    }
    return false;
};

const show_solution_pairs = (allowed_first_words: SolutionWordType,
                             allowed_second_words: SolutionWordType): number => {
    let count = 0;
    const solutions = Solutions.get_filtered();
    const words: string[] = Array.from(solutions.keys());
    //console.error(`words: ${Stringify(words)}\nsolutions: ${Stringify(solutions)}`);
    for (let i = 0; i < words.length - 1; ++i) {
        const first_word = words[i];
        if (!is_word_type_allowed(get_word_type(first_word), allowed_first_words)) continue;
        const first_value = solutions.get(first_word)!;
        const first_deps = first_value.depends as Set<string>;
        for (let j = i + 1; j < words.length; ++j) {
            const second_word = words[j];
            if (!is_word_type_allowed(get_word_type(second_word), allowed_second_words)) continue;
            const deps = solutions.get(second_word)!.depends as Set<string>;
            // skip pairs with dependency conflicts
            if (has_any(first_deps, deps) || deps?.has(first_word)) continue;
            // skip "known good" pairs
            if (first_value.hasOwnProperty(second_word)) continue;
            console.log(`${first_word} ${second_word}`);
            count += 1;
        }
    }
    return count;
};

export const run = (args: string[], options: any): number => {
    const src_ids = get_word_source_ids(options);
    if (!src_ids.length) {
        console.error('At least one word source must be specified.');
        return -1;
    }
    if (src_ids.length > 2) {
        console.error(`At most two word sources may be specified. (${src_ids.length})`);
        return -1;
    }
    console.error(`${Stringify(src_ids)} (${src_ids.length})`);
    let count = 0;
    if (is_all_solution_source_ids(src_ids)) {
        const first_word_type = get_word_type_from_source_id(src_ids[0]);
        count = show_solution_pairs(first_word_type, src_ids.length === 1 ?
            first_word_type : get_word_type_from_source_id(src_ids[1]));
    } else {
        const [words1, words2] = get_word_lists(src_ids, options.other);
        count = show_pairs(words1, words2, Remaining.letter_counts());
    }
    console.error(`pairs: ${count}`);
    return 0;
};
