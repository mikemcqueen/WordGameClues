// cm.ts

//const Getopt = require('node-getopt');
import * as Getopt from 'node-getopt';
import * as Lines from '../cm/lines';
import * as Pairs from '../cm/pairs';
import * as Populate from '../cm/populate';
import * as Remaining from '../cm/remaining';
import * as Retire from '../cm/retire';
import * as Solutions from '../cm/solutions';
import { stringify as Stringify } from 'javascript-stringify';

interface OptionsInterface {
    parse: (args: string[]) => any;
    setHelp: (text: string) => Object;
    showHelp: () => void;
}

type OptionsModule = {
    // TODO: remove these optionals eventually?
    Options?: string[][];
    show_help?: () => void;
    run: (args: string[], options: any) => number|Promise<number>;
};

// command name | module | command description
type CommandDefinitionTuple = [string, OptionsModule|null, string];

const Commands: CommandDefinitionTuple[] = [
    [ 'lines', Lines, 'lines blah blah' ],
    [ 'pairs', Pairs, 'generate word pairs' ],
    [ 'populate', Populate, 'blah blah populate' ],
    [ 'remain', Remaining, 'display remaining letters in a clue folder-hierarchy' ],
    [ 'retire', Retire, 'retire word pairs from a file' ] ,
    [ 'solutions', Solutions, 'blah blah solutions' ],
    [ 'help', null, 'help COMMAND for command-specific help' ]
];

const Options = [
    [ 'v', 'verbose', 'more output' ],
    [ 'h', 'help',    'this screen' ]
];

const show_commands = (commands: CommandDefinitionTuple[]): void => {
    console.log('\nCommands:');
    for (let tuple of commands) {
        console.log(`  ${tuple[0]}${' '.repeat(15 - tuple[0].length)}${tuple[2]}`);
    }
};

const get_module = (cmd: string, commands: CommandDefinitionTuple[] = Commands):
    OptionsModule|null =>
{
    for (let tuple of commands) {
        if (tuple[0] === cmd) return tuple[1];
    }
    return null;
};

const is_command = (cmd: string): boolean => {
    return get_module(cmd) != null;
};

const is_help_requested = (args: string[]): boolean => {
    return !args.length || !is_command(args[0]) || (args[0] === 'help');
};

const build_options = (module?: OptionsModule): OptionsInterface => {
    const options_list = module?.Options ? module.Options.concat(Options) : Options;
    const options: OptionsInterface = Getopt.create(options_list);
    options.setHelp('\nOptions:\n[[OPTIONS]]');
    return options;
};

const show_default_help = (): void => {
    console.log('Usage: node cm COMMAND [OPTION...]');
    show_commands(Commands);
    build_options().showHelp();
};

const show_module_help = (module: OptionsModule): void => {
    // TODO: make this property non-optional, remove this conditional
    if (module.show_help) {
        module.show_help();
        build_options(module).showHelp();
    }
    else
        show_default_help();
};

const show_help = (args: string[]): void => {
    if (!args.length) {
        show_default_help();
        return;
    }
    let cmd = args[0];
    if (cmd === 'help') {
        if (args.length === 1) {
            show_default_help();
            return;
        }
        cmd = args[1];
    }
    if (!is_command(cmd)) {
        console.error(`${cmd} is not a valid COMMAND.\n`);
        show_default_help();
        return;
    }
    show_module_help(get_module(cmd)!);
};

const execute_command = async (cmd: string, args: string[],
    commands: CommandDefinitionTuple[] = Commands): Promise<number> =>
{
    const module = get_module(cmd, commands);
    if (!module) throw new Error('should never happen!');
    const opt = build_options(module).parse(args);
    const options = opt.options;
    // TODO: this is a code smell that we wait to parse '--help' option here,
    // rather than doing it in is_help_requested. should call get_module()
    // in main(), and pass around the module, probably.
    if (options.help) {
        show_help(args);
        return 0;
    }
    if (opt.argv.length) {
        console.error(`Unrecognized parameter(s): ${Stringify(opt.argv)}`)
        return -1;
    }
    if (options.verbose) {
        console.error(`cmd: ${cmd}, args: ${Stringify(opt.argv)}, options: ${Stringify(options)}`);
    }
    return module.run(opt.argv, options);
};

// "main"
((): Promise<number> => {
    return new Promise((resolve, reject) => {
        const args = process.argv.slice(2); // ignore: node cm
        if (is_help_requested(args)) {
            show_help(args);
            resolve(0);
        }
        execute_command(args[0], args.slice(1))
            .then(code => { resolve(code); })
            .catch(e => { reject(e); });
    });
})().then(code => {
    process.exit(code);
}).catch(err => {
    console.error(err); //, err.stack);
    process.exit(-1);
});
