// remaining.ts

import * as Folder from './folder';
import * as Json from './json';
import * as Solutions from './solutions';
const Assert = require('assert');

export type LetterCounts = Int16Array;

const FILE = 'remain.json';
const LOWER_A = 'a'.charCodeAt(0);
const lower_z = 'z'.charCodeAt(0);

export const topmost_dir = (starting_dir?: string): string => {
    return Solutions.find_dir(starting_dir);
};

export const load = (dir: string): string => {
    const s = Json.load(Folder.make_path(dir, FILE));
    Assert(typeof s === 'string' || s instanceof String, "bad remain.json");
    return s;
};

export const load_topmost = (): string => {
    return load(topmost_dir());
};

const get_ascii_code = (letter: string): number => {
    Assert(letter.length === 1);
    return letter.charCodeAt(0);
};

const is_lower_alpha_ascii = (code: number): boolean => {
    return (code >= LOWER_A) && (code <= lower_z);
};

const is_lower_alpha = (letter: string): boolean => {
    return is_lower_alpha_ascii(get_ascii_code(letter));
};

const get_letter_index = (letter: string): number => {
    const code = get_ascii_code(letter);
    Assert(is_lower_alpha_ascii(code));
    return code - LOWER_A;
};

export const make_letter_counts = (letters: string): LetterCounts => {
    let remaining: LetterCounts = new Int16Array(26);
    for (const letter of letters) {
        if (!is_lower_alpha(letter)) continue;
        remaining[get_letter_index(letter)] += 1;
    }
    return remaining;
};

export const remove_letters = (counts: LetterCounts, letters: string): LetterCounts|undefined => {
    let new_counts = counts.slice();
    for (const letter of letters) {
        if (!is_lower_alpha(letter)) continue;
        const idx = get_letter_index(letter);
        if (!new_counts[idx]) return undefined;
        new_counts[idx] -= 1;
    }
    return new_counts;
};

const total_counts = (counts: LetterCounts): number => {
    return counts.reduce((total, count) => total + count, 0);
};

// convert lettercounts to charcode array
const flatten = (counts: LetterCounts): Uint16Array => {
    let result = new Uint16Array(total_counts(counts));
    let result_idx = 0;
    for (let i = 0; i < counts.length; ++i) {
        let count = counts[i];
        while (count--) {
            result[result_idx++] = LOWER_A + i;
        }
    }
    return result;
};

const to_letters = (counts: LetterCounts): string => {
    return String.fromCharCode(...flatten(counts));
};

const show_letter_counts = (counts: LetterCounts): void => {
    for (let i = 0; i < counts.length; ++i) {
        const count = counts[i];
        if (count) {
            console.log(`${String.fromCharCode(LOWER_A + i)}: ${count}`);
        }
    }
};

/*
const make_letter_map = (letters: string): Map<string, number> => {
    let map = new Map<string, number>();
    for (let i = 0; i < letters.length; ++i) {
        const letter = letters.charAt(i);
        if (letter === '.') continue; // hack
        const count: number = map.get(letter) || 0;
        map.set(letter, count + 1);
    }
    return map;
}

const show_letter_map = (map: Map<string, number>): void => {
    for (const [k, v] of map.entries()) {
        console.error(`${k}: ${v}`);
    }
}

const is_empty_letter_map = (map: Map<string, number>): boolean => {
    for (const key of map.keys()) {
        if (map.get(key)) {
            console.error(`${key}: ${map.get(key)}`);
            return false;
        }
    }
    return true;
};

const remove_letters = (source: string, letters: string): string => {
    let result: string = '';
    let letter_map = make_letter_map(letters);
    //show_letter_map(letter_map);
    for (let i = 0; i < source.length; ++i) {
        const letter = source.charAt(i);
        const count: number = letter_map.get(letter) || 0;
        if (count > 0) {
            letter_map.set(letter, count - 1);
        } else {
            result = result.concat(letter);
        }
    }
    Assert(is_empty_letter_map(letter_map));
    return result;
}
*/

export const letter_counts = (dir?: string): LetterCounts => {
    const topmost = topmost_dir(dir);
    let remain = make_letter_counts(load(topmost));
    //console.error(`topmost: ${remain}`);
    if (dir !== topmost) {
        const child_dirs = Folder.get_child_dirs(topmost);
        let cur_dir = topmost.slice();
        for (const child_dir of child_dirs) {
            remain = remove_letters(remain, child_dir)!;
        }
    }
    return remain;
}

export const letters = (dir?: string): string => {
    return to_letters(letter_counts(dir));
};

export const run = (args: string[]): number => {
    console.log(letters());
    return 0;
};
