'use strict';

const Debug = require('debug')('anfilter');
const Fs    = require('fs-extra');
import * as Words from "../modules/words.js";
const Opt   = require('node-getopt')
      .create([
          ['h', 'help',                'this screen' ]
      ])
      .bindHelp(
          'Usage: node anfilter <an output file>'
      );

let main = async () => {
    const opt = Opt.parseSystem();
    if (opt.argv.length < 1) {
        Opt.showHelp();
        return 1;
    }

    const filename = opt.argv[0];
    const dict_filename = "words";
    Debug(`filename: ${filename}`);
    Debug(`dict_filename: ${dict_filename}`);
    
    const dict = await Words.load_dict(dict_filename);
    console.error(`dict(${dict.size})`);
    const result = await Words.load_anresult(filename, dict);
    console.error(`result(${result.length})`);
    
    //result.forEach(line => { console.log(line); } );
    return 0;
};

main().catch(err => {
    console.log(err, err.stack);
});
