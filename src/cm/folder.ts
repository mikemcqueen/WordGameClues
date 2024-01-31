import * as _ from 'lodash';
const Assert = require('assert');
const Fs = require('fs-extra');

const ROOT_FILE = 'cm-root';

export const make_path = (dir: string, file: string): string => {
    if (!dir.length) return file;
    const delim = dir[dir.length - 1] === '/' ? '' : '/';
    return `${dir}${delim}${file}`;
};

export const file_exists = (dir: string, file: string): boolean => {
    return Fs.pathExistsSync(make_path(dir, file));
};


export const find_parent_with = (dir: string, filename: string): string => {
    // because logic is slightly simpler if true
    Assert((dir.length > 0) && (dir[0] === '/'), 'absolute path required');
    while (!file_exists(dir, filename)) {
        Assert(dir.length > 1, `${filename} not found`);
        const last_index = dir.lastIndexOf('/');
        dir = dir.slice(0, last_index);
    }
    return dir;
};

export const get_parent_names_until = (filename: string): string[] => {
    const current_dir = process.cwd();
    const root = find_parent_with(current_dir, filename);
    const parents = current_dir.slice(root.length);
    return parents.split('/').filter(name => name.length);
};

export const find_root = (dir: string): string => {
//    return find_file_up(dir, ROOT_FILE);
    // because logic is slightly simpler if true
    Assert((dir.length > 0) && (dir[0] === '/'), 'absolute path required');
    while (!file_exists(dir, ROOT_FILE)) {
        //console.error(dir);
        Assert(dir.length > 1, 'root not found');
        const last_index = dir.lastIndexOf('/');
        dir = dir.slice(0, last_index);
    }
    return dir;
};
