// lines.ts
//
// I don't remember what I was doing here.

const Assert = require('assert');
const Events = require('events');
const Fs = require('fs');
const Readline = require('readline');
import * as Remaining from './remaining';

type LetterCount = {
    letter: string;
    count: number;
};

const init_letter_counts = (letters: string): LetterCount[] => {
    return Array.from({length: letters.length}, (_, idx) =>
        ({ letter: letters[idx], count: 0 }));
};

const process_line = (line: string, letter_counts: LetterCount[]): void => {
    for (let lc of letter_counts) {
        if (line.includes(lc.letter)) {
            lc.count += 1;
        }
    }
};

const process_file = async (filename: string, letters: string): Promise<LetterCount[]> => {
    try {
        const rl = Readline.createInterface({
            input: Fs.createReadStream(filename),
            crlfDelay: Infinity
        });
        let letter_counts = init_letter_counts(letters);
        rl.on('line', line => process_line(line, letter_counts));
        await Events.once(rl, 'close');
        return letter_counts;
    } catch (err) {
        console.error(err);
        return [];
    }
};

const dump = (letter_counts: LetterCount[]): void => {
    letter_counts.sort((a, b) => a.count - b.count);
    for (let lc of letter_counts) {
        console.log(`${lc.letter}: ${lc.count}`);
    }
};

export const run = (args: string[], options: any): Promise<number> => {
    if (!args.length) {
        console.error('missing filename');
        return Promise.resolve(-1);
    }
    const letters = Remaining.letters(process.cwd(), false);
    console.error(`letters: ${letters} (${letters.length})`);
    return process_file(args[0], letters).then(letter_counts => {
        dump(letter_counts);
        return 0;
    });
};
