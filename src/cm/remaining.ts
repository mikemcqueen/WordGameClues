import * as _ from 'lodash';
const Assert = require('assert');

export type LetterCounts = Int32Array;
/*
    starting_letters: string;
    counts: Int32Array;
}
*/

const lower_a = 'a'.charCodeAt(0);
const lower_z = 'z'.charCodeAt(0);

const get_ascii_code = (letter: string): number => {
    Assert(letter.length === 1);
    return letter.charCodeAt(0);
}

const is_lower_alpha_ascii = (code: number): boolean => {
    return (code >= lower_a) && (code <= lower_z);
}

const is_lower_alpha = (letter: string): boolean => {
    return is_lower_alpha_ascii(get_ascii_code(letter));
}

const get_letter_index = (letter: string): number => {
    const code = get_ascii_code(letter);
    Assert(is_lower_alpha_ascii(code));
    return code - lower_a;
}

export const make_letter_counts = (starting_letters: string): LetterCounts => {
    let remaining: LetterCounts = new Int32Array(26);
    for (const letter of starting_letters) {
        if (!is_lower_alpha(letter)) continue;
        remaining[get_letter_index(letter)] += 1;
    }
    return remaining;
};

export const has_letters = (remaining: LetterCounts, letters: string): LetterCounts|undefined => {
    let new_remaining = remaining.slice(0);
    for (const letter of letters) {
        if (!is_lower_alpha(letter)) continue;
        const idx = get_letter_index(letter);
        if (!new_remaining[idx]) return undefined;
        new_remaining[idx] -= 1;
    }
    return new_remaining;
}
