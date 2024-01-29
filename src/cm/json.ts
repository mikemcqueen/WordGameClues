import * as _ from 'lodash';
const Fs = require('fs-extra');

export const load = (filename: string): any => {
    let obj: any;
    try {
        const json = Fs.readFileSync(filename, 'utf8');
        obj  = JSON.parse(json);
    } catch(e) {
        throw new Error(`${filename}, ${e}`);
    }
    return obj;
}
