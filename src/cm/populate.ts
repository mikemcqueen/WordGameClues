// populate.ts

import * as _ from 'lodash';
import * as Json from '../cm/json';
import * as Remaining from "../cm/remaining";

const Assert = require('assert');

const enum_directories = (primary_words: string[], secondary_words: string[],
    remaining: Remaining.LetterCounts, func: any, base_dir: string = ''): boolean =>
{
    let any_appended = false;
    for (let i = 0; i < primary_words.length; ++i) {
        let word = primary_words[i];
        const new_remaining = Remaining.remove_letters(remaining, word);
        if (!new_remaining) continue;
        any_appended = true;
        const dir = base_dir + word;
        const count = Remaining.total_counts(new_remaining);
        func(dir, count);
        if (!count) continue;
        let any_appended_recursively = false;
        if (i < primary_words.length - 1) {
            any_appended_recursively = enum_directories(primary_words.slice(i + 1),
                secondary_words, new_remaining, func, dir + '/');
        }
        if (!any_appended_recursively && secondary_words.length) {
            enum_directories(secondary_words, [], new_remaining, func, dir + '/');
        }
    }
    return any_appended;
}

export const run = (args: string[]): number => {
    console.log(`populate.run args: ${JSON.stringify(args)}`);
    const letters: string = Json.load("remain.json");
    let remaining = Remaining.make_letter_counts(letters);
    let primary_words: string[] = Json.load("words.json");
    let secondary_words: string[] = [];
    if (args.length) {
        secondary_words = primary_words;
        primary_words = Json.load(args[0]);
    }
    // Assert(validate_letters(words, remaining));
    enum_directories(primary_words, secondary_words, remaining,
        (dir: string, num_remaining: number) => {
            if (!num_remaining) console.log(dir);
        });
    return 0;
};
