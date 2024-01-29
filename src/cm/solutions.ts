import * as Folder from './folder';
import * as Json from './json';
const Assert = require('assert');
const Fs = require('fs-extra');
//const Stringify = require('stringify-object');

export const get_all = (starting_dir: string = process.cwd(), show = false): string[] => {
    const root_dir = Folder.find_root(starting_dir);
    const child_dirs = starting_dir.slice(root_dir.length).split('/').filter(dir => dir.length);
    if (show) console.error(`root_dir: ${root_dir} child_dirs: ${child_dirs.toString()}`);
    let all_solutions: string[] = [];
    let dir = Folder.make_path(root_dir.slice(), child_dirs[0]);
    // skip first level below root (no solutions)
    for (let idx = 1; idx < child_dirs.length; ++idx) {
        dir = Folder.make_path(dir, child_dirs[idx]);
        const solutions = Json.load(Folder.make_path(dir, 'solutions.json'));
        if (show) console.error(`${child_dirs[idx]}: ${solutions.toString()}`);
        all_solutions.push(...solutions);
    }
    if (show) console.error(`all_solutions ${all_solutions.toString()}`);
    return all_solutions;
}

export const show_all = (): void => {
    get_all(process.cwd(), true);
};
