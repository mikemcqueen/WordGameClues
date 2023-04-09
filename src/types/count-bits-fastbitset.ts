const FastBitSet = require("fastbitset");

export type Type = any;

export let makeNew = () : Type => {
    return new FastBitSet();
}

export let makeFrom = (values: any) : Type => {
    return new FastBitSet(values);
}

export let set = (cb: any, num: number) : void => {
    cb.add(num);
}

export let setMany = (cb: Type, nums: number[]) : void => {
    for (let num of nums) {
	cb.add(num);
    }
}

export let or = (cb1: Type, cb2: Type): Type => {
    return cb1.new_union(cb2);
}

export let and = (cb1: Type, cb2: Type): Type => {
    return cb1.new_intersection(cb2);
}

export let orInPlace = (cb1: Type, cb2: Type): Type => {
    return cb1.union(cb2);
}

export let intersects = (cb1: Type, cb2: Type) : boolean => {
    return cb1.intersects(cb2);
}
