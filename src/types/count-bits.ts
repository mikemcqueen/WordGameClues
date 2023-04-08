const RoaringBitmap32 = require("roaring/RoaringBitmap32");

export type Type = any;

export let makeNew = (size = 4) : Type => {
    return new RoaringBitmap32();
}

export let set = (cb: any, num: number) : void => {
    cb.add(num);
}

export let setMany = (cb: Type, nums: number[]) : void => {
    cb.addMany(nums);
}

export let or = (cb1: Type, cb2: Type): Type => {
    return makeNew().orInPlace(cb1).orInPlace(cb2);
}

export let intersects = (cb1: Type, cb2: Type) : boolean => {
    return cb1.intersects(cb2);
}

export let optimize = (cb: Type) : Type => {
    for (let bit of cb.bits) {
	let result = bit.runOptimize();
//	if (!result) console.error('runOptimize() failed');
    }
    return cb;
}
