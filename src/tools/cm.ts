// cm.ts

import * as _ from 'lodash';
import * as Pairs from '../cm/pairs';
import * as Populate from '../cm/populate';
import * as Solutions from '../cm/solutions';

const Opt   = require('node-getopt')
    .create([
        ['h', 'help',                'this screen' ]
    ])
    .bindHelp('Usage: cm populate');

const commands = [ "populate", "pairs", "show-solutions" ];


const is_command = (cmd: string, cmds: string[] = commands): boolean => {
    return cmds.includes(cmd);
}

const execute_command = (cmd: string, args: string[]): number => {
    switch(cmd) {
        case "populate": return Populate.run(args);
        case "pairs": return Pairs.run(args);
        case "show-solutions": Solutions.show_all(); break;
    }
    return -1;
};

const main = async (): Promise<number> => {
    const args = process.argv.slice(2);
    if (!args.length || !is_command(args[0])) {
        Opt.showHelp();
        return -1;
    }
    return execute_command(args[0], args.slice(1));
};

main().catch(err => {
    console.error(err, err.stack);
});
