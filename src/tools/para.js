'use strict';

let Parallel = require('paralleljs');
let ComboMakerWorkerBootstrap = require('../modules/combo-maker-worker-bootstrap');

function test(data) {
    console.log(`test: ${data}`);
    return 0;
}

async function main() {
    let data = [0, 1, 2, 3];
    let p = new Parallel(data, {
	//evalPath: 'https://raw.github.com/parallel-js/parallel.js/master/lib/eval.js'
	evalPath: '${__dirname}/../../modules/combo-maker-worker-bootstrap.js'
    });
    
    console.log('begin');
    p.map(ComboMakerWorkerBootstrap.entrypoint).then(_ => { console.log('end'); });
    //p.map(test).then(_ => { console.log('end'); });
}

main().catch(e => { console.error(e); });
