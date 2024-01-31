// pairs.ts

import * as Json from './json';
import * as Remaining from "./remaining";
import * as Solutions from "./solutions";

const Stringify = require("stringify-object");

const concat = (first: string, second: string): string => {
    let result = first;
    if (result.length > 0) {
        result += ' ';
    }
    result += second;
    return result;
};

const show_pairs = (primary_words: string[], secondary_words: string[] | undefined,
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

const show_all_pairs = (args: string[]): void => {
    const letters: string = Json.load("remain.json");
    let remaining = Remaining.make_letter_counts(letters);
    const words: string[] = Json.load("words.json");
    // assert(validate_letters(words, remaining));
    let primary_words = undefined;
    if (args.length) {
        primary_words = Json.load(args[0]);
    }
    show_pairs(primary_words || words, primary_words ? words : undefined, remaining);
};

const has_any = (target: Set<string>|undefined, source: Set<string>|undefined): boolean => {
    if (!target || !source) return false;
    for (const key of source) {
        if (target.has(key)) return true;
    }
    return false;
};

const get_solution_pairs = (): string[] => {
    let pairs: string[] = [];
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
            pairs.push(`${first_key} ${second_key}`);
        }
    }
    return pairs;
};

const show_solution_pairs = (): void => {
    console.error("solution pairs");
    const pairs = get_solution_pairs();
    console.error(`${Stringify(pairs)}`);
};

export const run = (args: string[]): number => {
    console.error(`pairs.run args: ${JSON.stringify(args)}`);
    if (args.length && (args[0] === 'solutions')) {
        show_solution_pairs();
    } else {
        show_all_pairs(args);
    }
    return 0;
};
