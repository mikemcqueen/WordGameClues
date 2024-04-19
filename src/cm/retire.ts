// retire.ts

import * as Folder from './folder';
import Fs from 'fs-extra';
import * as Json from './json';
const Assert = require('assert');

const RETIRED_DIR = 'retired';
const LATEST_IDX_FILENAME = 'latest.idx';
const PAIRS_FILENAME_PREFIX = 'pairs';

const get_retired_dir = (): string => {
    const root_dir = Folder.find_root(process.cwd());
    const retired_dir = Folder.make_path(root_dir, RETIRED_DIR);
    Fs.ensureDirSync(retired_dir);
    return retired_dir;
};

const make_pairs_path = (idx: number): string => {
    return Folder.make_path(get_retired_dir(), `${PAIRS_FILENAME_PREFIX}.${idx}`);
};

const set_latest_idx = (idx: number, latest_idx_path?: string): void => {
    if (!latest_idx_path) {
        latest_idx_path = Folder.make_path(get_retired_dir(), LATEST_IDX_FILENAME);
    }
    Fs.writeFileSync(latest_idx_path, `${idx}`);
};

const ensure_latest_idx_file = (): void => {
    const latest_idx_path = Folder.make_path(get_retired_dir(), LATEST_IDX_FILENAME);
    if (!Fs.existsSync(latest_idx_path)) {
        set_latest_idx(0, latest_idx_path);
    }
};

const get_latest_idx = (): number => {
    ensure_latest_idx_file();
    const latest_idx_path = Folder.make_path(get_retired_dir(), LATEST_IDX_FILENAME);
    Assert(Fs.existsSync(latest_idx_path), `latest index file doesn't exist: ${latest_idx_path}`);
    return Number(Fs.readFileSync(latest_idx_path));
};

const get_latest = (): string => {
    const latest_idx = get_latest_idx();
    let latest_pairs_path = make_pairs_path(latest_idx);
    // edge case: the "latest" pairs file doesn't exit. this could happen if this
    // is the first retired file (latest_idx = 0), or we're retrying after a prior
    // retire failure. 
    if (!Fs.existsSync(latest_pairs_path)) {
        if (latest_idx === 0) {
            // there is in fact no latest pairs file.
            return '';
        } else {
            Assert(latest_idx > 0, `latest_idx: ${latest_idx}`);
            // maybe we're recovering from a prior retire failure. in which case,
            // the pairs file for the previous index should exist. if not, things
            // have gone unrecoverably wrong.
            latest_pairs_path = make_pairs_path(latest_idx - 1);
            if (Fs.existsSync(latest_pairs_path)) {
                set_latest_idx(latest_idx - 1);
            } else {
                Assert(false, `latest and previous pairs files don't exist, manual ` +
                    `intervention required`);
            }
        }
    }
    return latest_pairs_path;
};

const get_next = (): string => {
    const latest_idx = get_latest_idx();
    let next_pairs_path = make_pairs_path(latest_idx);
    // edge case: the "latest" pairs file doesn't exit. this could happen if this
    // is the first retired file (next_idx = 0), or we're retrying after a prior
    // retire failure. next == latest in this case.
    if (Fs.existsSync(next_pairs_path)) {
        // common case: the "latest" pairs file exists; increment to next index.
        next_pairs_path = make_pairs_path(latest_idx + 1);
        // the "next" pairs file should *never* exist. if it does, something is
        // wrong (probably a prior retire failure). instruct the user to manually
        // intervene (probably by deleting the pairs file).
        if (Fs.existsSync(next_pairs_path)) {
            console.error(`next pairs file already exists, manual intervention required: ${next_pairs_path}`);
            Assert(false);
        }
        set_latest_idx(latest_idx + 1);
    }
    return next_pairs_path;
};

const revert_latest = (path: string): boolean => {
    const latest_idx = get_latest_idx();
    let latest_pairs_path = make_pairs_path(latest_idx);
    if (path !== latest_pairs_path) {
        console.error('invalid path supplied');
        return false;
    }
    if (latest_idx > 0) {
        set_latest_idx(latest_idx - 1);
    }
    return true;
};

export const run = (args: string[], options: any): number => {
    //console.error(`retire.run args: ${JSON.stringify(args)}`);
    if (args.length) {
        if (args[0] === 'latest') {
            const latest = get_latest();
            console.log(latest);
        } else if (args[0] === 'next') {
            const next = get_next();
            console.log(next);
        } else if (args[0] === 'revert') {
            if (!args[1]) {
                console.error(`missing path`);
                return -1;
            }
            if (!revert_latest(args[1])) {
                return -1;
            }
        } else {
            console.error(`unrecognized argument: ${args[0]}`);
            return -1;
        }
    } else {
        console.error(`missing argument`);
        return -1;
    }
    return 0;
};
