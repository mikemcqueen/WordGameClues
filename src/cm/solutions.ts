import * as Folder from './folder';
import * as Json from './json';
const Assert = require('assert');
const Fs = require('fs-extra');

export type MapEntry = {
  [key: string]: string[] | Set<string>;
};
export type MapType = Map<string, MapEntry>;

type FileMapEntry = Record<string, string[]>;
type FileMapType = Record<string, FileMapEntry>;

/*
export const get_all = (starting_dir: string = process.cwd()): Map<string, string[]> => {
    const root_dir = Folder.find_root(starting_dir);
    const child_dirs = starting_dir.slice(root_dir.length).split('/').filter(dir => dir.length);
    //if (show) console.error(`root_dir: ${root_dir} child_dirs: ${child_dirs.toString()}`);
    let result = new Map<string, string[]>();
    let dir = Folder.make_path(root_dir.slice(), child_dirs[0]);
    // skip first level below root (no solutions)
    for (let idx = 1; idx < child_dirs.length; ++idx) {
        const child = child_dirs[idx];
        dir = Folder.make_path(dir, child);
        const solutions = Json.load(Folder.make_path(dir, 'solutions.json'));
        Assert(!result.has(child));
        result.set(child, solutions);
    }
    return result;
}

export const old_show_all = (): void => {
    const solutions = get_all();
    for (let key of solutions.keys()) {
        console.error(`${key}: ${solutions.get(key)!.toString()}`);
    }
};
*/

const transform = (file_map: FileMapType): MapType => {
    let map: MapType = new Map();
    const outer_keys = Object.keys(file_map);
    for (const outer_key of outer_keys) {
        Assert(!map.has(outer_key)); // unexpected dupe, but not surprised if it happens
        map.set(outer_key, {});
        const inner_map = file_map[outer_key]!;;
        const inner_keys = Object.keys(inner_map);
        for (const inner_key of inner_keys) {
            const values = inner_map[inner_key]!;
            if (inner_key !== 'solutions') {
                // e.g. add "actress" (inner) to "wonder woman" (outer)
                let obj = map.get(outer_key)!;
                Assert(!obj.hasOwnProperty(inner_key));
                obj[inner_key] = values;

                // e.g. add  "wonder woman" (inner) to "actress" (outer)
                if (!map.has(inner_key)) map.set(inner_key, {});
                obj = map.get(inner_key)!;
                if (!obj.hasOwnProperty(outer_key)) obj[outer_key] = [];
                (obj[outer_key] as string[]).push(...values);
            }
            for (const value of values) {
                if (inner_key === 'solutions') {
                    Assert(!map.has(value)); // unexpected dupe, but not surprised if it happens
                    map.set(value, { depends: new Set([outer_key]) });
                } else { // if (!map.has(value)) {
                    Assert(!map.has(value)); // unexpected dupe, possibly tricky if it happens
                    map.set(value, { depends: new Set([outer_key, inner_key]) });
                }
            }
        }
    }
    return map;
};

const load = (starting_dir: string = process.cwd()): FileMapType => {
    const file = 'solutions.json';
    const dir = Folder.find_parent_with(starting_dir, file);
    return Json.load(Folder.make_path(dir, file)); // as FileMapType);
};

export const get_all = (): MapType => {
    return transform(load());
}

const filter = (source: FileMapType, keys: Set<string>): FileMapType => {
    let result: FileMapType = {};
    for (const key of Object.keys(source)) {
        if (keys.has(key)) {
            result[key] = source[key];
        }
    }
    return result;
};

const fixup = (names: Set<string>): Set<string> => {
    let result = new Set<string>();
    for (const name of names.keys()) {
        result.add(name.replace('.', ' '));
    }
    return result;
}

export const get_filtered = (): MapType => {
    const names = Folder.get_parent_names_until('solutions.json');
    return transform(filter(load(), fixup(names)));
}

const is_map = (o) => o instanceof Map;
const is_set = (o) => o instanceof Set;

const my_replacer = (key, value) => {
    if (is_map(value)) {
        return { ['M']: Array.from(value) };
    } else if (is_set(value)) {
        return { ['S']: Array.from(value) };
    }
    return value;
}

export const show = (solutions: MapType): void => {
    for (let key of solutions.keys()) {
        console.error(`${key}: ${JSON.stringify(solutions.get(key)!, my_replacer)}`);
    }
};

export const run = (args: string[]): number => {
    if (args.length && (args[0] === 'all')) {
        show(get_all());
    } else {
        show(get_filtered());
    }
    return 0;
}
