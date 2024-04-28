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
    [ 'f', 'file=FILE+',       'words from FILE' ]
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
    File: 2,
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
    if (options.file) {
        for (let file of options.file) {
            ids.push(WordSourceId.File);
        }
    }
    return ids;
};

type WordCountType = {
    single?: boolean;
    multi?: boolean;
};

const get_word_count = (word: string): WordCountType => {
    return word.indexOf(' ') === -1 ? { single: true } : { multi: true };
};

const get_word_count_from_source_id = (src_id: number): WordCountType => {
    switch (src_id) {
        case WordSourceId.SingleSolutions: return { single: true };
        case WordSourceId.MultiSolutions:  return { multi: true };
        case WordSourceId.AllSolutions:    return { single: true, multi: true };
        default: throw new Error(`Invalid word source id, ${src_id}`);
    }
};

const is_word_count_allowed = (word_count: WordCountType,
                               allowed_word_counts: WordCountType): boolean => {
    return (word_count.single! && allowed_word_counts.single!) ||
        (word_count.multi! && allowed_word_counts.multi!);
};

type WordType = {
    word: string;
    depends?: Set<string>;
}

const list_from_strings = (words: string[]): WordType[] => {
    const result: WordType[] = [];
    for (const word of words) {
        result.push({ word });
    }
    return result;
};

const list_to_string_set = (words: WordType[]): Set<string> => {
    const result = new Set<string>();
    for (const wt of words) {
        result.add(wt.word);
    }
    return result;
}

type WordList = {
    src_id: number;
    words: WordType[]; // string[];
};

const get_solution_words = (word_count: WordCountType): WordType[] => {
    const result: WordType[] = [];
    const solutions = Solutions.get_filtered();
    const words: string[] = Array.from(solutions.keys());
    for (const word of words) {
        if (is_word_count_allowed(get_word_count(word), word_count)) {
            result.push({ word, depends: solutions.get(word)!.depends as Set<string> });
        }
    }
    return result;
};

const is_solution_source_id = (src_id: number): boolean => {
    return (src_id & SolutionFlag) !== 0;
};

const get_word_list = (src_id: number, filename?: string): WordList => {
    let words: WordType[] = [];
    switch (src_id) {
        case WordSourceId.Words:
            words = list_from_strings(Json.load('words.json'));
            break;
        case WordSourceId.File:
            Assert(filename);
            words = list_from_strings(Json.load(filename!));
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

const get_word_lists = (src_ids: number[], filenames?: string[]): [WordList, WordList] => {
    let word_lists: WordList[] = [];
    let filename_idx = 0;
    for (const src_id of src_ids) {
        word_lists.push(get_word_list(src_id, filenames?.[filename_idx]));
        if (src_id === WordSourceId.File) ++filename_idx;
    }
    return [word_lists[0], word_lists.length > 1 ? word_lists[1] : word_lists[0]];
};

const is_disjoint = (set1: Set<string>|undefined, set2: Set<string>|undefined): boolean => {
    if (set1 && set2) {
        for (const value of set2) {
            if (set1.has(value)) return false;
        }
    }
    return true;
};

const has_dependency_conflict = (word1: WordType, word2: WordType): boolean => {
    if (word2.depends?.has(word1.word)) return true;
    if (word1.depends?.has(word2.word)) return true;
    return !is_disjoint(word1.depends, word2.depends);
};

const filter_pair = (word1: WordType, word2: WordType, shown_pairs: Set<string>): boolean => {
    // same word
    if (word1.word === word2.word) return false;
    // pair already shown
    if (shown_pairs.has(`${word1.word} ${word2.word}`)) return false;
    // reversed pair already shown
    if (shown_pairs.has(`${word2.word} ${word1.word}`)) return false;
    // pair with dependency conflict
    if (has_dependency_conflict(word1, word2)) return false;
    return true;
};

const show_pairs = (words1: WordList, words2: WordList,
                    letter_counts: Remaining.LetterCounts): number =>
{
    let shown_pairs = new Set<string>();
    for (const word1 of words1.words) {
        let remaining = letter_counts;
        // if it's a solution word, don't remove letters from remaining
        if (!is_solution_source_id(words1.src_id)) {
            remaining = Remaining.remove_letters(remaining, word1.word)!;
            if (!remaining) {
                continue;
            }
        }
        for (const word2 of words2.words) {
            if (!filter_pair(word1, word2, shown_pairs)) continue;

            // skip "known good" pairs
            // TODO
            // TODO
            // TODO
            //if (first_value.hasOwnProperty(second_word)) continue;

            // if it's a solution word, don't remove letters from remaining.
            if (!is_solution_source_id(words2.src_id) &&
                !Remaining.remove_letters(remaining, word2.word))
            {
                continue;
            }
            const pair = `${word1.word} ${word2.word}`;
            console.log(pair);
            shown_pairs.add(pair);
        }
    }
    return shown_pairs.size;
};

const is_all_solution_source_ids = (src_ids: number[]): boolean => {
    for (const id of src_ids) {
        if (!is_solution_source_id(id)) return false;
    }
    return true;
};

const show_solution_pairs = (allowed_first_words: WordCountType,
                             allowed_second_words: WordCountType): number => {
    let count = 0;
    const solutions = Solutions.get_filtered();
    const words: string[] = Array.from(solutions.keys());
    //console.error(`words: ${Stringify(words)}\nsolutions: ${Stringify(solutions)}`);
    for (let i = 0; i < words.length - 1; ++i) {
        const first_word = words[i];
        if (!is_word_count_allowed(get_word_count(first_word), allowed_first_words)) continue;
        const first_value = solutions.get(first_word)!;
        const first_deps = first_value.depends as Set<string>;
        for (let j = i + 1; j < words.length; ++j) {
            const second_word = words[j];
            if (!is_word_count_allowed(get_word_count(second_word), allowed_second_words)) continue;
            const deps = solutions.get(second_word)!.depends as Set<string>;
            // skip pairs with dependency conflicts
            if (!is_disjoint(first_deps, deps) || deps?.has(first_word)) continue;
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
    if (options.verbose) {
        console.error(`${Stringify(src_ids)} (${src_ids.length})`);
    }
    let count = 0;
/*
    if (is_all_solution_source_ids(src_ids)) {
        const first_word_type = get_word_count_from_source_id(src_ids[0]);
        count = show_solution_pairs(first_word_type, src_ids.length === 1 ?
            first_word_type : get_word_count_from_source_id(src_ids[1]));
    } else
*/
    {
        const [words1, words2] = get_word_lists(src_ids, options.file);
        //console.error(`words1: ${StringifyObj(words1)}\nwords2: ${StringifyObj(words2)}`);
        count = show_pairs(words1, words2, Remaining.letter_counts());
    }
    console.error(`pairs: ${count}`);
    return 0;
};
