// cm.ts

import * as Lines from '../cm/lines';
import * as Pairs from '../cm/pairs';
import * as Populate from '../cm/populate';
import * as Remaining from '../cm/remaining';
import * as Retire from '../cm/retire';
import * as Solutions from '../cm/solutions';

const Opt = require('node-getopt')
    .create([
        ['h', 'help',                'this screen' ]
    ])
    .bindHelp('Usage: cm populate|pairs|solutions|remain|lines');

const commands = [ "populate", "pairs", "solutions", "remain", "retire", "lines" ];

const is_command = (cmd: string, cmds: string[] = commands): boolean => {
    return cmds.includes(cmd);
}

const execute_command = async (cmd: string, args: string[]): Promise<number> => {
    switch(cmd) {
        case "populate": return Populate.run(args);
        case "pairs": return Pairs.run(args);
        case "solutions": return Solutions.run(args);
        case "remain": return Remaining.run(args);
        case "retire": return Retire.run(args);
        case "lines": return Lines.run(args);
    }
    return -1;
};

// "main"
((): Promise<void> => {
    const args = process.argv.slice(2);
    if (!args.length || !is_command(args[0])) {
        Opt.showHelp();
        process.exit(-1);
    }
    execute_command(args[0], args.slice(1)).then(code => {
        process.exit(code);
    });
    return Promise.resolve();
})().catch(err => {
    console.error(err, err.stack);
    process.exit(-1);
});
