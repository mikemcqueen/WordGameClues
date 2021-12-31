/* eslint no-eval: off, no-restricted-globals: off */

'use strict';

const ComboMakerWorker = require('./combo-maker-worker');

let bootstrap = (data) => {
    console.log(`bootstrapping ${data}`);
    ComboMakerWorker.doWork(data);
    return 0;
};

let entrypoint = (data) => {
    return bootstrap(data);
};

process.once('message', code => {
    let d = JSON.parse(code).data;
    //console.log(d);
    // TODO: assign env.global to actual global state somehow (within stringify'd code probably)
    eval(d);
});

module.exports = {
    entrypoint
};
