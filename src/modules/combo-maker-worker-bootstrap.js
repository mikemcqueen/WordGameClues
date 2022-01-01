/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ComboWorker = require('./combo-maker-worker');
const Stringify   = require('stringify-object');

let bootstrap = (args) => {
    console.log(`bootstrapping sum(${args.sum}) max(${args.max}) args: ${Stringify(args)}`);
    ComboWorker.doWork(args);
    return 0;
};

let entrypoint = (args) => {
    return bootstrap(args);
};

process.once('message', code => {
    let d = JSON.parse(code).data;
    //console.log(d);
    eval(d);
});

module.exports = {
    entrypoint
};
