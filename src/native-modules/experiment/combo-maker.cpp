#if 0
//
//
let mergeSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean | undefined): AnySourceData => {
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        source1 = source1 as LazySourceData;
        source2 = source2 as LazySourceData;
        const result: LazySourceData = {
            primaryNameSrcList,
            ncList,
            synonymCounts: Clue.PropertyCounts.merge(
                getSynonymCountsForValidateResult(source1.validateResultList[0]),
                getSynonymCountsForValidateResult(source2.validateResultList[0])),
            validateResultList: [
                (source1 as LazySourceData).validateResultList[0],
                (source2 as LazySourceData).validateResultList[0]
            ]
        };
        return result;
    }
    source1 = source1 as SourceData;
    source2 = source2 as SourceData;
    const mergedSource: SourceData = {
        primaryNameSrcList,
        ncList,
        synonymCounts: Clue.PropertyCounts.merge(source1.synonymCounts, source2.synonymCounts),
        sourceNcCsvList: [...source1.sourceNcCsvList, ...source2.sourceNcCsvList]
    };
    // TODO: still used?
    mergedSource.ncCsv = NameCount.listToSortedString(mergedSource.ncList);
    return mergedSource;
};
#endif

#if 0
//
//
let mergeCompatibleSources = (source1: AnySourceData, source2: AnySourceData, args: MergeArgs): AnySourceData[] => {
    // TODO: this logic could be part of mergeSources
    // also, uh, isn't there a primarySrcArray I can be using here?
    return allCountUnique(source1.primaryNameSrcList, source2.primaryNameSrcList)
        ? [mergeSources(source1, source2, args.lazy)]
        : [];
};
#endif

#if 0
//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], args: MergeArgs): AnySourceData[] => {
    let mergedSourcesList: AnySourceData[] = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, args))
        }
    }
    return mergedSourcesList;
};
#endif

#if 0
//
//
let getSynonymCounts = (sourceList: AnySourceData[]): Clue.PropertyCounts.Type => {
    return sourceList.reduce(
        (counts, source) => Clue.PropertyCounts.add(counts, source.synonymCounts),
        Clue.PropertyCounts.empty());
};
                      
//
//
let sourceListHasPropertyCountInBounds = (sourceList: AnySourceData[], minMax: MinMax.Type): boolean => {
    const synonymCounts = getSynonymCounts(sourceList);
    const inBounds = propertyCountsIsInBounds(synonymCounts, minMax);
    if (!inBounds) {
        if (0) {
            console.error(`oob: [${NameCount.listToNameList(sourceListToNcList(sourceList))}]` +
                `, syn-total(${synonymCounts.total})`);
        }
    }
    return inBounds;
}
#endif

#if 0
//
//
let mergeAllCompatibleSources = (ncList: NameCount.List, args: MergeArgs): AnySourceData[] => {
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let sourceList = getSourceList(ncList[0], args);
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList = getSourceList(ncList[ncIndex], args);
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList, args);
        if (!sourceListHasPropertyCountInBounds(sourceList, args.synonymMinMax)) sourceList = [];
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(sourceList)) break;
    }
    return sourceList;
};
#endif

#if 0
//
//
let buildSourceListsForUseNcData = (useNcDataLists: NCDataList[], args: MergeArgs): SourceList[] => {
    let sourceLists: SourceList[] = [];
    // TODO: This is to prevent duplicate sourceLists. I suppose I could use a Set or Map, above?
    let hashList: StringBoolMap[] = [];
    for (let ncDataList of useNcDataLists) {
        for (let [sourceListIndex, useNcData] of ncDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            // give priority to any min/max args specific to an NcData, for example, through --xormm,
            // but fallback to the values we were called with
            const mergeArgs = useNcData.synonymMinMax ? { synonymMinMax: useNcData.synonymMinMax } : args;
            const sourceList = mergeAllCompatibleSources(useNcData.ncList, mergeArgs) as SourceList;
            for (let source of sourceList) {
                let key = NameCount.listToString(_.sortBy(source.primaryNameSrcList, NameCount.count));
                if (!hashList[sourceListIndex][key]) {
                    sourceLists[sourceListIndex].push(source as SourceData);
                    hashList[sourceListIndex][key] = true;
                }
            }
        }
    }
    return sourceLists;
};
#endif
