import * as _ from 'lodash';
import * as Json from '../cm/json';
import * as Remaining from "../cm/remaining";

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
}

export const run = (args: string[]): number => {
    console.error(`pairs.run args: ${Stringify(args)}`);
    const letters: string = Json.load("remain.json");
    let remaining = Remaining.make_letter_counts(letters);
    const words: string[] = Json.load("words.json");
    // assert(validate_letters(words, remaining));
    let primary_words = undefined;
    if (args.length) {
        primary_words = Json.load(args[0]);
    }
    show_pairs(primary_words || words, primary_words ? words : undefined, remaining);
    return 0;
};
