//
// min-max.ts
//

'use strict';

import _ from 'lodash';
const Assert = require('assert');

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

export function init(min: number|string = 1000000000, max: number|string = 0,
    fallbackMin?: number, fallbackMax?: number): Type
{
    return {
        min: initValue(min, fallbackMin),
        max: initValue(max, fallbackMax)
    };
}

export const update = (minMax: Type, value: number): void => {
    if (value < minMax.min) {
        minMax.min = value;
    }
    if (value > minMax.max) {
        minMax.max = value;
    }
}

export const add = (to: Type, from: Type): void => {
    to.min += from.min;
    to.max += from.max
}

