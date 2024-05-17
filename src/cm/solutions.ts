import * as Folder from './folder';
import * as Json from './json';
const Assert = require('assert');

export const Options = [
    [ 'a', 'all',          'do not filter by parent directory names' ],
    [ '', 'words[=COUNT]', 'show solutions "words", optionally containing exactly COUNT words' ]
];

export const show_help = (): void => {
    console.log('Usage: node cm solutions [-a] [--words [COUNT]]');
    console.log('\nShow solutions word-map, or solution words only (filtered by parent directory names by default).');
};

const FILENAME = 'solutions.json';

export type MapEntry = {
  [key: string]: string[] | Set<string>;
};
export type MapType = Map<string, MapEntry>;

type FileMapEntry = Record<string, string[]>;
type FileMapType = Record<string, FileMapEntry>;

const is_map = (o) => o instanceof Map;
const is_set = (o) => o instanceof Set;

const my_replacer = (key, value) => {
    if (is_map(value)) {
        return { ['M']: Array.from(value) };
    } else if (is_set(value)) {
        return { ['S']: Array.from(value) };
    }
    return value;
};

export const show = (solutions: MapType): void => {
    for (let key of solutions.keys()) {
        console.error(`${key}: ${JSON.stringify(solutions.get(key)!, my_replacer)}`);
    }
};

const do_transform = (file_map: FileMapType, map: MapType, valid_keys: Set<string>): boolean => {
    //console.error(`transform, keys(${valid_keys.size}): ${Array.from(valid_keys).toString()}`);
    let changed = false;
    const outer_keys = Object.keys(file_map).filter(key => valid_keys.has(key));
    for (const outer_key of outer_keys) {
        if (!map.has(outer_key)) {
            map.set(outer_key, {});
            changed = true;
        }
        const inner_map = file_map[outer_key]!;
        const inner_keys = Object.keys(inner_map).filter(key => valid_keys.has(key));
        for (const inner_key of inner_keys) {
            const values = inner_map[inner_key]!;
            if (inner_key !== 'solutions') {
                // TODO:
                //if (!add_key(map, outer_key, inner_key, values)) continue;
                let obj = map.get(outer_key)!;
                // skip key if already processed
                if (obj.hasOwnProperty(inner_key)) continue;
                // e.g. add "actress" (inner) to "wonder woman" (outer)
                obj[inner_key] = values;
                if (!map.has(inner_key)) map.set(inner_key, {});
                obj = map.get(inner_key)!;
                Assert(!obj.hasOwnProperty(outer_key));
                // e.g. add  "wonder woman" (inner) to "actress" (outer)
                obj[outer_key] = values;
                changed = true;
            }
            for (const value of values) {
                if (inner_key === 'solutions') {
                    Assert(!map.has(value));
                    map.set(value, { depends: new Set([outer_key]) });
                    changed = true;
                } else {
                    Assert(!map.has(value));
                    map.set(value, { depends: new Set([outer_key, inner_key]) });
                }
            }
        }
    }
    return changed;
};

const transform = (file_map: FileMapType, valid_keys: Set<string> = new Set()): MapType => {
    let map: MapType = new Map;
    // copy and "append"
    valid_keys = new Set(valid_keys);
    valid_keys.add('solutions');
    while (do_transform(file_map, map, valid_keys)) {
        if (0) {
            console.error(`transform result:`);
            show(map);
            console.error(`----------------`);
        }
        if (!valid_keys.size) break;
        valid_keys = new Set(map.keys());
    }
    return map;
};

export const find_dir = (starting_dir: string = process.cwd()): string => {
    return Folder.find_parent_with(starting_dir, FILENAME);
};

const load = (starting_dir?: string): FileMapType => {
    return Json.load(Folder.make_path(find_dir(starting_dir), FILENAME));
};

export const get_all = (): MapType => {
    return transform(load());
};

const filter = (file_map: FileMapType, valid_keys: Set<string>): FileMapType => {
    let result: FileMapType = {};
    const keys = Object.keys(file_map);
    for (const key of keys) {
        if (valid_keys.has(key)) {
            result[key] = file_map[key];
        }
    }
    return result;
};

const fix_names = (names: string[]): string[] => {
    return names.map(name => name.replace('.', ' '));
};

export const get_filtered = (): MapType => {
    const name_list = Folder.get_parent_names_until('solutions.json');
    const names = new Set(fix_names(name_list));
    const map = transform(filter(load(), names), names);
    // to support the case where parent diretory names haven't been added to
    // the solutions.json map yet, do a 2nd pass adding all the names that
    // haven't yet been added to the result.
    // this is a partial solution that only works on "filtered" results; a
    // broader solution is outlined in todo.cm
    for (const name of names) {
        if (!map.has(name)) {
            map.set(name, {});
        }
    }
    return map;
};

export const show_words = (solutions: MapType, required_word_count: number): void => {
    for (let key of solutions.keys()) {
        if (!required_word_count || (key.split(' ').length === required_word_count)) {
            console.log(key);
        }
    }
};

const get_num_primary_options = (options: any): number => {
    return Number(!!options.words) + (options.all|0);
};

// for maximum flexibility, if another optional-value option is added,
// should pass process.argv here.
const get_required_word_count = (option: string, args: string[]) : number => {
    let count = 0;
    if (!option.length) {
        if (args.length) count = Number(args[0]);
    } else {
        count = Number(option);
    }
    Assert(!isNaN(count));
    return count;
};

export const run = (args: string[], options: any): number => {
    const num_options = get_num_primary_options(options);
    // TODO: remove restriction. allow filtering for both show and show_words.
    if (num_options > 1) {
        console.error('Only one of --words or --all may be specified.');
        return -1;
    }
    if (options.all) {
        show(get_all());
    } else if (options.words !== undefined) {
        // NOTE: args[1] is not very forward-compatible
        show_words(get_filtered(), get_required_word_count(options.words, args.slice(1)));
    } else {
        show(get_filtered());
    }
    return 0;
};
