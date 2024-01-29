import * as _ from 'lodash';
import * as Json from '../cm/json';
import * as Remaining from "../cm/remaining";

const Assert = require('assert');
const Fs = require('fs-extra');
//const Peco = require('../modules/peco');

//return this.parse(process.argv.slice(2));

const make_directories = (words: string[], remaining: Remaining.LetterCounts,
     base_dir: string = ''): void =>
{
    for (let i = 0; i < words.length; ++i) {
        let word = words[i];;
        const remaining_minus_word = Remaining.has_letters(remaining, word);
        if (!remaining_minus_word) continue;
        const dir = base_dir + word;
        // Fs.mkdir (dir);
        console.log(dir);
        if (i < words.length - 1) {
            make_directories(words.slice(i + 1), remaining_minus_word, dir + '/');
        }
    }
}

export const run = (args: string[]): number => {
    console.log('populate.run');
    const letters: string = Json.load("remain.json");
    let remaining = Remaining.make_letter_counts(letters);
    const words: string[] = Json.load("words.json");
    // assert(validate_letters(words, remaining));
    make_directories(words, remaining);
    return 0;
};
