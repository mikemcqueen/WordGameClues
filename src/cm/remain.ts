// remain.ts

import * as Folder from './folder';
import * as Json from './json';
import * as Solutions from './solutions';
const Assert = require('assert');

const FILE = 'remain.json';

export const topmost_dir = (starting_dir?: string): string => {
    return Solutions.find_dir(starting_dir);
}

export const load = (dir: string): string => {
    const s = Json.load(Folder.make_path(dir, FILE));
    Assert(typeof s === 'string' || s instanceof String, "bad remain.json");
    return s;
};

export const load_topmost = (): string => {
    return load(topmost_dir());
}

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

export const calculate = (dir?: string): string => {
    const topmost = topmost_dir(dir);
    let remain = load(topmost);
    //console.error(`topmost: ${remain}`);
    if (dir !== topmost) {
        const child_dirs = Folder.get_child_dirs(topmost);
        let cur_dir = topmost.slice();
        for (const child_dir of child_dirs) {
            remain = remove_letters(remain, child_dir);
        }
    }
    return remain;
}

export const run = (args: string[]): number => {
    console.log(calculate());
    return 0;
};
