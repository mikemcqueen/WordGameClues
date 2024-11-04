// pairs.ts

import * as Json from './json';
import * as Remaining from "./remaining";
import * as Solutions from "./solutions";
import { stringify as Stringify } from 'javascript-stringify';
const Assert = require('assert');
const Events = require('events');
const Fs = require('fs');
const Readline = require('readline');
const StringifyObj = require("stringify-object");

export const Options = [
    [ 'w', 'words',              'use words from words.json' ],
    [ 'l', 'with-letter=LETTER', 'use words from words.json containing LETTER' ],
    [ 's', 'single-solutions',   'use single-word solution words' ],
    [ 'm', 'multi-solutions',    'use multi-word solution words' ],
    [ 'a', 'all-solutions',      'use all solutions words' ],
    [ 'f', 'file=JSON+',         'use words from JSON file' ],
    [ 'c', 'letter-counts',      'calculate pair-counts for words containing each remaining letter (using words.json)' ],
    [ 'o', 'old',                'use old method (build set of all pairs)' ],
    [ 'p', 'flip',               'flip order of the two word lists' ],
    [ '',  'analyze=FILE',       'analyze pairs in text FILE. current only and default option is 2nd word frequency' ]
];

export const show_help = (): void => {
    // TODO: updatee
    console.log('Usage (outdated): node cm pairs [-w] [-s] [-m] [-a] [-f FILE]...');
    console.log('\nGenerate pairs from words in words.json, solutions.json, and/or another words file.');
};

type WordCountType = {
    single?: boolean;
    multi?: boolean;
};

type WordType = {
    str: string;
    // for solution words:
    props?: any;            // all properties
    depends?: Set<string>;  // depends property
};

type WordList = {
    src_id: number;
    words: WordType[];
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
        'with-letter': WordSourceId.Words,
        'single-solutions': WordSourceId.SingleSolutions,
        'multi-solutions': WordSourceId.MultiSolutions,
        'all-solutions': WordSourceId.AllSolutions
    });
    const ids: number[] = [];
    for (const option in option_id_map) {
        if (options[option] !== undefined) {
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

const list_from_strings = (words: string[]): WordType[] => {
    const result: WordType[] = [];
    for (const word of words) {
        result.push({ str: word });
    }
    return result;
};

const list_to_string_set = (words: WordType[]): Set<string> => {
    const result = new Set<string>();
    for (const word of words) {
        result.add(word.str);
    }
    return result;
}
/*
const copy_word_list = (words: WordList): WordList => {
    return { src_id: words.src_id, word: words.slice() };
};
*/

const get_solution_words = (word_count: WordCountType): WordType[] => {
    const result: WordType[] = [];
    const solutions = Solutions.get_filtered();
    const words: string[] = Array.from(solutions.keys());
    for (const word of words) {
        if (is_allowed_word_count(get_word_count(word), word_count)) {
            const props = solutions.get(word)!;
            result.push({ str: word, props, depends: props.depends as Set<string> });
        }
    }
    return result;
};

const is_solution_source_id = (src_id: number): boolean => {
    return (src_id & SolutionFlag) !== 0;
};

const validate_word_list = (words: WordType[]): boolean => {
    let unique_words = new Set<string>();
    for (let word of words) {
        if (unique_words.has(word.str)) {
            console.error(`duplicate word '${word.str}' in wordlist`);
            return false;
        }
        unique_words.add(word.str);
    }
    return true;
}

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
    if (!validate_word_list(words)) {
        process.exit(-1);
    }
    return { src_id, words };
};

const get_word_lists = (src_ids: number[], options: any): WordList[] => {
    let word_lists: WordList[] = [];
    let filename_idx = 0;
    for (const src_id of src_ids) {
        const list = get_word_list(src_id, options.file?.[filename_idx]);
        word_lists.push(list);
        if (src_id === WordSourceId.File) ++filename_idx;
    }
    let flip_order = !!options.flip;
    if (options['with-letter']) {
        Assert(word_lists.length === 1);
        const list1 = word_lists[0];
        word_lists.push({
            src_id: list1.src_id,
            words:  list1.words.filter(word => word.str.includes(options['with-letter']))
        });
        // flip order, unless --flip was specified
        flip_order = !flip_order;
    }
    if (word_lists.length === 1) {
        // use same list twice
        word_lists.push(word_lists[0]);
    } else if (word_lists.length === 2) {
        if ((word_lists[0].src_id === WordSourceId.Words)
            && (word_lists[1].src_id !== WordSourceId.Words)) {
            // if we ended up with a Words source as list1, and a non-words source as
            // list2, flip the order, unless --flip was specified. This enables the
            // version of show_pairs that doesn't run out of memory.
            flip_order = !flip_order;
        }
    } else {
        Assert(false);
    }
    if (flip_order) {
        Assert(word_lists.length === 2, 'Option --flip requires two word sources');
        word_lists = [word_lists[1], word_lists[0]];
    }
    return word_lists;
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
    if (word1.depends?.has(word2.str)) return true;
    if (word2.depends?.has(word1.str)) return true;
    if (!is_disjoint(word1.depends, word2.depends)) return true;
    return false;
};

const is_known_good_pair = (word1: WordType, word2: WordType): boolean => {
    if (word1.props?.hasOwnProperty(word2.str)) return true;
    if (word2.props?.hasOwnProperty(word1.str)) return true;
    return false;
};

const make_pair = (word1: WordType, word2: WordType): string => {
    return [word1.str, word2.str].sort().join(',');
}

type Stats = {
    same_word: number;
    used_word: number;
    known_good: number;
    dep_conflict: number;
    first_invalid: number;
    second_invalid: number;
};

const new_stats = (): Stats => {
    return {
        same_word: 0,
        used_word: 0,
        known_good: 0,
        dep_conflict: 0,
        first_invalid: 0,
        second_invalid: 0
    };
};

const sum_stats = (stats: Stats): number => {
    return stats.same_word + stats.used_word + stats.known_good
        + stats.dep_conflict + stats.first_invalid + stats.second_invalid;
};

const show_stats = (valid: number, stats: Stats): void => {
    console.error(`same word:    ${stats.same_word}`);
    console.error(`used word:    ${stats.used_word}`);
    console.error(`known good:   ${stats.known_good}`);
    console.error(`dep conflict: ${stats.dep_conflict}`);
    console.error(`1st invalid:  ${stats.first_invalid}`);
    console.error(`2nd invalid:  ${stats.second_invalid}`);
    const sum = sum_stats(stats);
    console.error(`-----`);
    console.error(`all stats: ${sum}`);
    console.error(`total:     ${sum + valid}`);
};

let pair_stats = new_stats();

const allow_pair = (word1: WordType, word2: WordType,
                    shown_pairs?: Set<string>,
                    used_words?: Set<string>): boolean =>
{
    // same word
    if (word1.str === word2.str) {
        pair_stats.same_word++;
        return false;
    }
    // pair already shown
    if (shown_pairs) {
        if (shown_pairs.has(make_pair(word1, word2))) return false;
    }
    // already used (??)
    if (used_words?.has(word2.str)) {
        pair_stats.used_word++;
        return false;
    }
    // pair with known solution
    if (is_known_good_pair(word1, word2)) {
        pair_stats.known_good++;
        return false;
    }
    // pair with dependency conflict
    if (has_dependency_conflict(word1, word2)) {
        pair_stats.dep_conflict++;
        return false;
    }
    return true;
};

const show_pairs_old = (words1: WordList, words2: WordList,
                        letter_counts: Remaining.LetterCounts): number =>
{
    let shown_pairs = new Set<string>();
    for (const word1 of words1.words) {
        let remaining = letter_counts;
        // remove letters from remaining for non-solution words
        if (!is_solution_source_id(words1.src_id)) {
            remaining = Remaining.remove_letters(remaining, word1.str)!;
            if (!remaining) {
                pair_stats.first_invalid++;
                continue;
            }
        }
        for (const word2 of words2.words) {
            // remove_letters from remaining for non-solution words must succeed
            if (!is_solution_source_id(words2.src_id) &&
                !Remaining.remove_letters(remaining, word2.str))
            {
                pair_stats.second_invalid++;
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

const show_pairs = (words1: WordList, words2: WordList,
                    letter_counts: Remaining.LetterCounts,
                    count_only?: boolean): number =>
{
    let num_pairs = 0;
    let used_words = new Set<string>();
    for (const word1 of words1.words) {
        let remaining = letter_counts;
        // remove letters from remaining for non-solution words
        if (!is_solution_source_id(words1.src_id)) {
            remaining = Remaining.remove_letters(remaining, word1.str)!;
            if (!remaining) {
                pair_stats.first_invalid++;
                continue;
            }
        }
        used_words.add(word1.str);
        for (const word2 of words2.words) {
            // remove_letters from remaining for non-solution words must succeed
            if (!is_solution_source_id(words2.src_id) &&
                !Remaining.remove_letters(remaining, word2.str))
            {
                pair_stats.second_invalid++;
                continue;
            }
            if (!allow_pair(word1, word2, undefined, used_words)) continue;
            if (!count_only) {
                console.log(make_pair(word1, word2));
            }
            num_pairs++;
        }
    }
    return num_pairs;
};

type LetterCountData = {
    count: number;
    num_words1: number;
    num_words2: number;
}

const show_letter_counts = (letter_counts: Remaining.LetterCounts): void => {
    let map = new Map<string, LetterCountData>;
    const letters = Remaining.to_letters(letter_counts, false);
    for (const letter of letters) {
        const words2 = get_word_list(WordSourceId.Words);
        const words1 = {
            src_id: words2.src_id,
            words:  words2.words.filter(word => word.str.includes(letter))
        };
        const count = show_pairs(words1, words2, letter_counts.slice(), true);
        map.set(letter, {
            count,
            num_words1: words1.words.length,
            num_words2: words2.words.length
        });
        process.stderr.write('.');
    }
    process.stderr.write('\r');
    let keys = Array.from(map.keys()).sort((a, b) => map.get(a)!.count - map.get(b)!.count);
    for (const key of keys) {
        const lcd = map.get(key)!;
        console.log(`${key}: ${lcd.count} -- words1(${lcd.num_words1})` +
            `, words2(${lcd.num_words2})`);
    }
};

const validate_options = (options: any): boolean => {
    return true;
};

const validate_source_ids = (src_ids: number[], options: any): boolean => {
    if (!src_ids.length) {
        console.error('At least one word source must be specified.');
        return false;
    }
    if (src_ids.length > 2) {
        console.error(`At most two word sources may be specified. (${src_ids.length})`);
        return false;
    }
    if (options['with-letter'] && (src_ids.length > 1)) {
        console.error(`Option --with-letter may not be combined with any other word-source option`);
        return false;
    }
    return true;
};

const process_line = (line: string, map: Map<string, number>): void => {
    const words = line.split(',');
    Assert(words.length === 2);
    const word = words[1];
    if (!map.has(word)) {
        map.set(word, 1);
    } else {
        map.set(word, map.get(word)! + 1);
    }
};

const process_file = async (filename: string): Promise<Map<string, number>> => {
    let map = new Map<string, number>();
    try {
        const rl = Readline.createInterface({
            input: Fs.createReadStream(filename),
            crlfDelay: Infinity
        });
        rl.on('line', line => process_line(line, map));
        await Events.once(rl, 'close');
    } catch (err) {
        console.error(err);
    }
    return map;
};

const dump = (freq_map: Map<string, number>): void => {
    const sorted_entries = Array.from(freq_map.entries()).sort((a, b) => a[1] - b[1]);
    let total = 0;
    for (const [key, value] of sorted_entries) {
        console.log(`${key} (${value})`);
        total += value;
    }
    console.log(`${sorted_entries.length} unique words, ${total} total occurences`);
};

const show_2nd_word_frequency = (filename: string): Promise<number> => {
    return process_file(filename).then(freq_map => {
        dump(freq_map);
        return 0;
    });
};

export const run = (args: string[], options: any): Promise<number> => {
    if (options['letter-counts']) {
        show_letter_counts(Remaining.letter_counts());
        return Promise.resolve(0);
    }
    if (options['analyze']) {
        return show_2nd_word_frequency(options.analyze);
    }
    if (!validate_options(options)) return Promise.resolve(-1); //return -1;
    const src_ids = get_word_source_ids(options);
    if (!validate_source_ids(src_ids, options)) return Promise.resolve(-1); // return -1;
    if (options.verbose) {
        console.error(`src_ids: ${Stringify(src_ids)} (${src_ids.length})`);
    }
    const [words1, words2] = get_word_lists(src_ids, options);
    if (options.verbose) {
        console.error(`word list1(${words1.words.length})` +
            `, list2(${words2.words.length})`);
    }
    // When to use show_pairs_old:
    // * when option --old is specified
    // * or when two word source options are specified, and the 2nd one is not --words
    // when to use new:
    // * when only one word source option is specified
    // * or when two word source options are specified, and the 2nd one is --words
    //
    // Note that by default the --words list is always put in 2nd slot. Option --flip
    // can override that and cause the version of show_pairs to change.
    let count;
    if (options.old || ((src_ids.length > 1) && (words2.src_id !== WordSourceId.Words))) {
        console.error(`WARNING: using show_pairs_old()`);
        count = show_pairs_old(words1, words2, Remaining.letter_counts());
    } else {
        count = show_pairs(words1, words2, Remaining.letter_counts());
    }
    if (options.verbose || options.count) {
        console.error(`pairs:     ${count}`);
    }
    if (options.verbose) {
        show_stats(count, pair_stats);
    }
    return Promise.resolve(0);
};
