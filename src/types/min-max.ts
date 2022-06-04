//
// min-max.ts
//

'use strict';

import _ from 'lodash'; // TODO: need a smaller dummy import
const Assert      = require('assert');

export interface Type {
    min: number;
    max: number;
};

function initValue(value: number|string, fallbackValue?: number): number {
    let numberValue: number = _.toNumber(value); 
    if (_.isNaN(numberValue)) {
        Assert(!_.isUndefined(fallbackValue) && !_.isNaN(fallbackValue),
               `numberValue is ${numberValue}, fallbackValue is ${fallbackValue}`);
        numberValue = fallbackValue as number;
    }
    return numberValue;
}

export function init(min: number|string, max: number|string,
                     fallbackMin?: number, fallbackMax?: number): Type
{
    return {
        min: initValue(min, fallbackMin),
        max: initValue(max, fallbackMax)
    };
}
