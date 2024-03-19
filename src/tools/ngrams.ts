const Assert = require('assert');
const Events = require('events');
const Fs = require('fs');
const Readline = require('readline');

type Stats = {
    min_occur: number;
    max_occur: number;
    sum_occur: number;
    total_lines: number;
    discarded_lines: number;
};

let stats: Stats = {
    min_occur: 999,
    max_occur: 0,
    sum_occur: 0,
    total_lines: 0,
    discarded_lines: 0
};

const dump_stats = (sats: Stats): void => {
    const num_lines = stats.total_lines - stats.discarded_lines;
    const percent_lines = Math.round(num_lines / stats.total_lines * 100);
    const avg_occur = Math.round(stats.sum_occur / num_lines);
    console.error(`min: ${stats.min_occur}, max: ${stats.max_occur}, avg: ${avg_occur}` +
        `, lines: ${num_lines} of ${stats.total_lines} (${percent_lines}%)`);
};

const update_occur = (stats: Stats, sum: number): void => {
    stats.sum_occur += sum;
    if (sum > stats.max_occur) stats.max_occur = sum;
    if (sum < stats.min_occur) stats.min_occur = sum;
};

const process_dates = (line: string): number => {
    const dates = line.split('\t');
    let sum = 0;
    for (const csv of dates) {
        const values = csv.split(',');
        Assert(values.length === 3);
        sum += Number(values[1]);
    }
    return sum;
};

const underscored = (line: string): boolean => {
    return line.startsWith('_') && line.endsWith('_');
};

const is_alpha = (ch: string): boolean => {
    return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z");
};

const num_alpha = (line: string): number => {
    let num = 0;
    for (const ch of line) {
        if (is_alpha(ch)) num += 1;
    }
    return num;;
};

const is_alpha_code = (code: number): boolean => {
    return (code >= 65 && code <= 90) || (code >= 97 && code <= 122);
};

const num_alpha_code = (line: string): number => {
    let num = 0;
    const len = line.length;
    for (let i = 0; i < len; ++i) {
        if (is_alpha_code(line.charCodeAt(i))) num += 1;
    }
    return num;;
};

//const UnderscoreCharCode = '_'.charCodeAt(0);

const filter = (gram: string): string|undefined => {
    const first_underscore = gram.indexOf('_');
    if (first_underscore > -1) {
        if (first_underscore === 0) {
            return undefined;
        }
        gram = gram.slice(0, first_underscore);
    }
    // # alpha characters = 0 or 1
    if (num_alpha_code(gram) < 2) return undefined;
    return gram;
};

const filter_all = (grams: string[]): string|undefined => {
    let filtered: string[] = [];
    for (const gram of grams) {
        const result = filter(gram);
        if (!result) return undefined;
        filtered.push(result);
    }
    return filtered.join(' ');
};

const process_line = (line: string): void  => {
    const first_tab = line.indexOf('\t');
    let sum = -1;
    if (first_tab > -1) {
        sum = process_dates(line.slice(first_tab + 1));
        line = line.slice(0, first_tab);
    }
    const grams = line.split(' ');
    stats.total_lines += 1;
    Assert(grams.length === 2);
    const result = filter_all(grams);
    if (!result) {
        stats.discarded_lines += 1;
    } else {
        let sum_str = '';
        if (sum > -1) {
            sum_str = ` ${sum}`;
            update_occur(stats, sum);
        }
        console.log(`${result}${sum_str}`);
    }
};

const process_file = async (filename: string): Promise<void> => {
    try {
        const rl = Readline.createInterface({
            input: Fs.createReadStream(filename),
            crlfDelay: Infinity
        });
        rl.on('line', process_line);
        await Events.once(rl, 'close');
    } catch (err) {
        console.error(err);
    }
};

(function main() {
    if (process.argv.length < 3) {
        console.error('missing filename');
        process.exit(-1);
    }
    const filename = process.argv[2];
    process_file(filename).then(_ => dump_stats(stats));
})();
