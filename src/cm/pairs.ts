// pairs.ts

import * as Json from './json';
import * as Remaining from "./remaining";
import * as Solutions from "./solutions";
const Assert = require('assert');

const Stringify = require("stringify-object");

const concat = (first: string, second: string): string => {
    let result = first;
    if (result.length > 0) {
        result += ' ';
    }
    return result + second;
};

/*
const old_show_pairs = (primary_words: string[], secondary_words: string[] | undefined,
    remaining: Remaining.LetterCounts, max_depth: number = 2, depth: number = 1,
    preceding_words: string = ''): void =>
{
    for (let i = 0; i < primary_words.length; ++i) {
        let word = primary_words[i];
        const remaining_minus_word = Remaining.has_letters(remaining, word);
        if (!remaining_minus_word) continue;
        const gram = concat(preceding_words, word);
        if (depth === max_depth) {
            console.log(gram);
        }
        if ((i < primary_words.length - 1) && (depth < max_depth)) {
            show_pairs(secondary_words || primary_words.slice(i + 1), undefined,
                remaining_minus_word, max_depth, depth + 1, gram);
        }
    }
};
*/

const show_pairs = (list1: string[], list2: string[], letter_counts: Remaining.LetterCounts): number => {
    let count = 0;
    for (const word1 of list1) {
        let remaining1 = Remaining.remove_letters(letter_counts, word1);
        // because some words in list1 may be "solution" (to folder names) words.
        // *may* is not as strict as i'd like.
        let remaining2_required = false;
        if (!remaining1) {
            remaining1 = letter_counts;
            remaining2_required = true;
        }
        for (const word2 of list2) {
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
    } else {
        words2 = words2.slice(1);
    }
    // TODO: append solution words to words1
    return show_pairs(words1, words2, remaining);
};

const has_any = (target: Set<string>|undefined, source: Set<string>|undefined): boolean => {
    if (!target || !source) return false;
    for (const key of source) {
        if (target.has(key)) return true;
    }
    return false;
};

const show_solution_pairs = (): number => {
    let count = 0;
    const solutions = Solutions.get_filtered();
    const keys: string[] = Array.from(solutions.keys());
    for (let i = 0; i < keys.length - 1; ++i) {
        const first_key = keys[i];
        const first_value = solutions.get(first_key)!;
        const first_deps = solutions.get(first_key)!.depends as Set<string>;
        for (let j = i + 1; j < keys.length; ++j) {
            const second_key = keys[j];
            const deps = solutions.get(second_key)!.depends as Set<string>;
            // skip pairs with dependency conflicts
            if (has_any(first_deps, deps) || deps?.has(first_key)) continue;
            // skip "known good" pairs
            if (first_value.hasOwnProperty(second_key)) continue;
            console.log(`${first_key} ${second_key}`);
            count += 1;
        }
    }
    return count;
};

export const run = (args: string[]): number => {
    console.error(`pairs.run args: ${JSON.stringify(args)}`);
    let count = 0;
    if (args.length && (args[0] === 'solutions')) {
        count = show_solution_pairs();
    } else {
        count = show_all_pairs(args);
    }
    console.error(`pairs: ${count}`);
    return 0;
};
