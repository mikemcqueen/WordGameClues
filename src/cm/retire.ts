// retire.ts

import * as Folder from './folder';
import Fs from 'fs-extra';
import * as Json from './json';
const Assert = require('assert');

export const Options = [
    [ '',  'latest',      'display latest retired pairs file path' ],
    [ '',  'next',        'display next retired pairs file path and increment latest_idx' ],
    [ '',  'revert=PATH', 'revert the latest retired pairs file PATH and decrement latest_idx' ],
    [ 'd', 'dry-run',     'dry-run: latest_idx not updated' ]
];

export const show_help = (): void => {
    console.log('Usage: node cm retire --latest|--next|--revert=PATH\n' +
        '\nDisplay the name of the latest, or next, retired pairs file path, or revert\n' +
        'the latest retired pairs path.');
};

const RETIRED_DIR = 'retired';
const LATEST_IDX_FILENAME = 'latest.idx';
const PAIRS_FILENAME_PREFIX = 'pairs';

const get_retired_dir = (): string => {
    const root_dir = Folder.find_root(process.cwd());
    const retired_dir = Folder.make_path(root_dir, RETIRED_DIR);
    Fs.ensureDirSync(retired_dir);
    return retired_dir;
};

const make_latest_idx_path = (): string => {
    return Folder.make_path(get_retired_dir(), LATEST_IDX_FILENAME);
};

const make_pairs_path = (idx: number): string => {
    return Folder.make_path(get_retired_dir(), `${PAIRS_FILENAME_PREFIX}.${idx}`);
};

const set_latest_idx = (idx: number, options: any): void => {
    const latest_idx_path = Folder.make_path(get_retired_dir(), LATEST_IDX_FILENAME);
    if (options?.['dry-run']) {
        console.error(`dry-run: writeFile(path: ${latest_idx_path}, idx: ${idx})`);
    } else {
        Fs.writeFileSync(latest_idx_path, `${idx}`);
    }
};

const ensure_latest_idx_file = (options: any): void => {
    if (!Fs.existsSync(make_latest_idx_path())) {
        set_latest_idx(0, options);
    }
};

const get_latest_idx = (options: any): number => {
    ensure_latest_idx_file(options);
    const latest_idx_path = make_latest_idx_path();
    Assert(Fs.existsSync(latest_idx_path), `latest index file doesn't exist: ${latest_idx_path}`);
    return Number(Fs.readFileSync(latest_idx_path));
};

const get_latest = (options: any): string => {
    const latest_idx = get_latest_idx(options);
    let latest_pairs_path = make_pairs_path(latest_idx);
    // edge case: the "latest" pairs file doesn't exist. this could happen if this
    // is the first retired file (latest_idx = 0), or we're retrying after a prior
    // retire failure. 
    if (!Fs.existsSync(latest_pairs_path)) {
        if (latest_idx === 0) {
            // there is in fact no latest pairs file.
            return '';
        }
        Assert(latest_idx > 0, `latest_idx: ${latest_idx}`);
        // maybe we're recovering from a prior retire failure. in which case,
        // the pairs file for the previous index should exist. if not, things
        // have gone unrecoverably wrong.
        latest_pairs_path = make_pairs_path(latest_idx - 1);
        if (!Fs.existsSync(latest_pairs_path)) {
            throw new Error(`latest and previous pairs files don't exist, ` +
                `manual intervention required`);
        }
        set_latest_idx(latest_idx - 1, options);
    }
    return latest_pairs_path;
};

const get_next = (options: any): string => {
    const latest_idx = get_latest_idx(options);
    let next_pairs_path = make_pairs_path(latest_idx);
    // edge case: the "latest" pairs file doesn't exist. this could happen if this
    // is the first retired file (next_idx = 0), or we're retrying after a prior
    // retire failure. next == latest in this case.
    if (Fs.existsSync(next_pairs_path)) {
        // common case: the "latest" pairs file exists; increment to next index.
        next_pairs_path = make_pairs_path(latest_idx + 1);
        // the "next" pairs file should *never* exist. if it does, something is
        // wrong (probably a prior retire failure). instruct the user to manually
        // intervene (probably by deleting the pairs file).
        if (Fs.existsSync(next_pairs_path)) {
            throw new Error(`next pairs file already exists, ` +
                ` manual intervention required: ${next_pairs_path}`);
        }
        set_latest_idx(latest_idx + 1, options);
    }
    return next_pairs_path;
};

const revert_latest = (path: string, options: any): boolean => {
    const latest_idx = get_latest_idx(options);
    let latest_pairs_path = make_pairs_path(latest_idx);
    if (path !== latest_pairs_path) {
        console.error('invalid path supplied');
        return false;
    }
    if (latest_idx > 0) {
        set_latest_idx(latest_idx - 1, options);
    }
    return true;
};

const get_num_primary_options = (options: any): number => {
    return (options.latest|0) + (options.next|0) + Number(!!options.revert);
};

export const run = (args: string[], options: any): number => {
    const num_options = get_num_primary_options(options);
    if (!num_options) {
        console.error('One of --latest, --next, or --revert must be specified.');
        return -1;
    } else if (num_options > 1) {
        console.error('Only one of --latest, --next, or --revert may be specified.');
        return -1;
    }
    if (options.latest) {
        console.log(get_latest(options));
    } else if (options.next) {
        console.log(get_next(options));
    } else if (options.revert) {
        if (!revert_latest(options.revert, options)) {
            return -1;
        }
    }
    return 0;
};
