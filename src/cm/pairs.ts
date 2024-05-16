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
    console.log('Usage: node cm pairs [-w] [-s] [-m] [-a] [-f FILE]...');
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

const is_allowed_word_count = (word_count: WordCountType,
                               allowed_word_counts: WordCountType): boolean => {
    return (word_count.single! && allowed_word_counts.single!) ||
        (word_count.multi! && allowed_word_counts.multi!);
};

type WordType = {
    word: string;
    // for solution words:
    props?: any;            // all properties
    depends?: Set<string>;  // depends property
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
    for (const word of words) {
        result.add(word.word);
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
        if (is_allowed_word_count(get_word_count(word), word_count)) {
            const props = solutions.get(word)!;
            result.push({ word, props, depends: props.depends as Set<string> });
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
    if (word1.depends?.has(word2.word)) return true;
    if (word2.depends?.has(word1.word)) return true;
    if (!is_disjoint(word1.depends, word2.depends)) return true;
    return false;
};

const is_known_good_pair = (word1: WordType, word2: WordType): boolean => {
    if (word1.props?.hasOwnProperty(word2.word)) return true;
    if (word2.props?.hasOwnProperty(word1.word)) return true;
    return false;
};

const make_pair = (word1: WordType, word2: WordType): string => {
    return `${word1.word},${word2.word}`;
}

const allow_pair = (word1: WordType, word2: WordType, shown_pairs: Set<string>): boolean => {
    // same word
    if (word1.word === word2.word) return false;
    // pair already shown
    if (shown_pairs.has(make_pair(word1, word2))) return false;
    // reverse pair already shown
    if (shown_pairs.has(make_pair(word2, word1))) return false;
    // pair with known solution, e.g. "stinky", "french cheese"
    if (is_known_good_pair(word1, word2)) return false;
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
        // remove letters from remaining for non-solution words
        if (!is_solution_source_id(words1.src_id)) {
            remaining = Remaining.remove_letters(remaining, word1.word)!;
            if (!remaining) {
                continue;
            }
        }
        for (const word2 of words2.words) {
            // remove_letters from remaining for non-solution words must succeed
            if (!is_solution_source_id(words2.src_id) &&
                !Remaining.remove_letters(remaining, word2.word))
            {
                continue;
            }
            if (!allow_pair(word1, word2, shown_pairs)) continue;
            const pair = make_pair(word1, word2);
            console.log(pair);
            shown_pairs.add(pair);
        }
    }
    return shown_pairs.size;
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
    const [words1, words2] = get_word_lists(src_ids, options.file);
    const count = show_pairs(words1, words2, Remaining.letter_counts());
    if (options.verbose) {
        console.error(`pairs: ${count}`);
    }
    return 0;
};
