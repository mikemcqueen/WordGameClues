import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.List;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.TreeSet;
import java.util.TreeMap;
import java.util.Comparator;

class Freqsort
{
    private static enum MatchType
    {
	UNDEFINED, ALL, PROPER, NON_PROPER;
    }

    private static Map<String, Freq> freqSort;
    private static Map<Character, Integer> letterCounts = new TreeMap<Character, Integer>();
    private static Map<Character, Integer> ignoreLetterCounts = new HashMap<Character, Integer>();
    private static Map<Character, BigDecimal> letterFrequencyMap = new HashMap<Character, BigDecimal>();
    private static MathContext mc = new MathContext(5, RoundingMode.HALF_UP);
    private static BigDecimal TWO = new BigDecimal(2, mc);
    private static BigDecimal ONE_HALF = new BigDecimal(.5, mc);
    private static boolean multiply = false;
    private static boolean add = false;
    private static boolean interleave = false;
    private static boolean wordsOnly = false;
    private static MatchType matchType = MatchType.ALL;

    private static BigDecimal _value = BigDecimal.ONE;
    private static Comparator<StringFreq> sortFunc;

    static
    {
        letterFrequencyMap.put('a', 	new BigDecimal(8.167, mc));
        letterFrequencyMap.put('b', 	new BigDecimal(1.492, mc));
        letterFrequencyMap.put('c', 	new BigDecimal(2.782, mc));
        letterFrequencyMap.put('d', 	new BigDecimal(4.253, mc));
        letterFrequencyMap.put('e', 	new BigDecimal(12.702, mc));
        letterFrequencyMap.put('f', 	new BigDecimal(2.228, mc));
        letterFrequencyMap.put('g', 	new BigDecimal(2.015, mc));
        letterFrequencyMap.put('h', 	new BigDecimal(6.094, mc));
        letterFrequencyMap.put('i', 	new BigDecimal(6.966, mc));
        letterFrequencyMap.put('j', 	new BigDecimal(0.153, mc));
        letterFrequencyMap.put('k', 	new BigDecimal(0.772, mc));
        letterFrequencyMap.put('l', 	new BigDecimal(4.025, mc));
        letterFrequencyMap.put('m', 	new BigDecimal(2.406, mc));
        letterFrequencyMap.put('n', 	new BigDecimal(6.749, mc));
        letterFrequencyMap.put('o', 	new BigDecimal(7.507, mc));
        letterFrequencyMap.put('p', 	new BigDecimal(1.929, mc));
        letterFrequencyMap.put('q', 	new BigDecimal(0.095, mc));
        letterFrequencyMap.put('r', 	new BigDecimal(5.987, mc));
        letterFrequencyMap.put('s', 	new BigDecimal(6.327, mc));
        letterFrequencyMap.put('t', 	new BigDecimal(9.056, mc));
        letterFrequencyMap.put('u', 	new BigDecimal(2.758, mc));
        letterFrequencyMap.put('v', 	new BigDecimal(0.978, mc));
        letterFrequencyMap.put('w', 	new BigDecimal(2.360, mc));
        letterFrequencyMap.put('x', 	new BigDecimal(0.150, mc));
        letterFrequencyMap.put('y', 	new BigDecimal(1.974, mc));
        letterFrequencyMap.put('z', 	new BigDecimal(0.074, mc));
    }

    public static void main(String[] args) throws IOException
    {
        if (args.length < 3)
        {
            System.out.println("Usage: java Freqsort [-i] [-n|-p|-a] <value> <sentence-file> <dict-file> [ignore-string1 ... ignore-stringN]");
            return;
	}

	int index = 0;
	for (; index < args.length; ++index) {
	    if ((args[index].charAt(0) == '-') && 
		(args[index].length() > 1))
	    {
		switch (args[index].charAt(1)) {
		case 'i': interleave = true;		    break;
		case 'n': matchType = MatchType.NON_PROPER; break;
		case 'p': matchType = MatchType.PROPER;     break;
		case 'a': matchType = MatchType.ALL;        break;
		case 'w': wordsOnly = true; break;
		default : throw new IllegalArgumentException("-" + String.valueOf(args[index].charAt(1)));
		}
	    }
	    else break;
	}

	String valueArg = args[index++];
	char x = Character.toLowerCase(valueArg.charAt(0));
	if (x == 'x') {
	    multiply = true;
	} else if (x == '+') {
	    add = true;
	    _value = new BigDecimal(valueArg.substring(1));
        } else if (x != '0') {
	    _value = new BigDecimal(valueArg);
	}

	freqSort = new TreeMap<String, Freq>();
	if (!interleave) {
	    sortFunc = new StringFreq.ByFreq();
	} else {
	    sortFunc = new StringFreq.ByLengthAndFreq();
        }

        String sentence = loadSentence(args[index++]).toLowerCase();

	boolean anyIgnore = false;
	boolean lastUppercase = false;
        String ignoring = "";
	for (int argIndex = index + 1; argIndex < args.length; ++argIndex) {
	    if (Character.isUpperCase(args[argIndex].charAt(0)))
		lastUppercase = true;

	    ignoreLetters(args[argIndex], ignoreLetterCounts);
            ignoring = ignoring.concat(args[argIndex]);
	    anyIgnore = true;
	}
        Map<Character, Integer> sentenceLetterCounts = new HashMap<>();
        ignoreLetters(sentence, sentenceLetterCounts);
        if (!validateIgnoreLetters(sentenceLetterCounts, ignoreLetterCounts)) {
            System.out.println("ignore letters: '" + ignoring + "' do not exist in '" + sentence + "'");
            return;
        }
        System.out.println("ignoring: " + ignoring);
	
	if (matchType == MatchType.UNDEFINED) {
	    if (anyIgnore) {
		if (lastUppercase)
		    matchType = MatchType.NON_PROPER;
		else
		    matchType = MatchType.PROPER;
	    } else {
		matchType = MatchType.ALL; // Default if no ignore words and no flags specified
	    }
	}

        int letterCount = buildLetterCountMap(sentence);

        loadDict(args[index], letterCount);

        if (freqSort.isEmpty()) {
            System.out.println("no results");
        } else {
            dump(freqSort, wordsOnly);
        }
    }

    private static String loadSentence(String filename) throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        return reader.readLine().toLowerCase();
    }

    private static int buildLetterCountMap(String sentence)
    {
	int total = 0;

        for (char ch : sentence.toCharArray()) {
            if (!Character.isLowerCase(ch)) {
                continue;
            }
	    if (isIgnoreLetter(ch)) {
		continue;
	    }
            int count = 0;
            if (letterCounts.containsKey(ch)) {
                count = letterCounts.get(ch);
            }
            letterCounts.put(ch, ++count);
            ++total;
        }
	return total;
    }
 
    private static boolean isIgnoreLetter(char ch) 
    {
	if (!ignoreLetterCounts.containsKey(ch)) {
	    return false;
	}
	int count = ignoreLetterCounts.get(ch);
	if (count > 1) {
	    ignoreLetterCounts.put(ch, --count);
	}
	else {
	    ignoreLetterCounts.remove(ch);
	}
	return true;
    }

    private static void loadDict(String filename, int letterCount) throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = reader.readLine()) != null) {
	    boolean proper = Character.isUpperCase(line.charAt(0));

	    if (((matchType == MatchType.PROPER) && !proper) ||
		((matchType == MatchType.NON_PROPER) && proper))
	    {
		continue;
	    }

	    line = line.toLowerCase();
	    
	    BigDecimal freq = getFreq2(line, letterCount);
	    if (freq != null) {
		freqSort.put(line, new Freq(freq));
	    }
        }
    }

    private static BigDecimal getFreq2(String word, int totalLetterCount)
    {
	Map<Character, Integer> localLetterCounts = new HashMap<Character, Integer>(letterCounts);
	BigDecimal freq = multiply ? BigDecimal.ONE : BigDecimal.ZERO;
	BigDecimal count = BigDecimal.ZERO;
	BigDecimal mean = BigDecimal.ONE;

	for (char ch : word.toCharArray())
        {
            if (!Character.isLowerCase(ch)) {
                continue;
            }
	    if (!localLetterCounts.containsKey(ch)) {
		// impossible to make this word
		return null;
	    }
	    int letterCount = localLetterCounts.get(ch);
	    if (letterCount > 1) {		
		localLetterCounts.put(ch, letterCount - 1);
	    } else {
		localLetterCounts.remove(ch);
	    }
	
	    Factor factor = getFactor(ch, letterCount, totalLetterCount);
	    if (multiply) {
		freq = freq.multiply(factor.factor.add(BigDecimal.ONE), mc);
	    } else if (add) {
		BigDecimal value = factor.factor.multiply(mean);
	        freq = freq.add(value);
		freq = freq.add(_value);
	    } else {
	        freq = freq.add(factor.factor);
	        freq = freq.multiply(_value);
	    }
	    count = count.add(BigDecimal.ONE);
	    if (add) {
		mean = freq.divide(count, mc);
	    }
	    
	    --totalLetterCount;
	}
	// if (!multiply)
	{
	    if (0 != count.compareTo(BigDecimal.ZERO))
		freq = freq.divide(count, mc);
	}
	return freq;
    }

    private static Factor getFactor(char ch, int letterCount, int totalLetterCount)
    {
	return new Factor(ch, letterCount, totalLetterCount);
    }

    private static class Factor
    {
	BigDecimal actual;
	BigDecimal expected;
	BigDecimal factor;
	
	public Factor(char ch, int letterCount, int totalLetterCount)
	{
	    actual = new BigDecimal(letterCount, mc).divide(new BigDecimal(totalLetterCount), mc).scaleByPowerOfTen(2);
	    expected = letterFrequencyMap.get(ch);
	    factor = actual.divide(expected, mc);
	}
    }

    private static void ignoreLetters(String word, Map<Character, Integer> map)
    {
        for (char ch : word.toCharArray()) {
            if (!Character.isLowerCase(ch)) {
                continue;
            }
            int count = 0;
            if (map.containsKey(ch)) {
                count = map.get(ch);
            }
            map.put(ch, ++count);
	}
    }

    private static void dump(Map<String, Freq> frequencyMap, boolean wordsOnly)
    {
        //BigDecimal bdTotal = new BigDecimal(total);
	List<StringFreq> freqList = new ArrayList<StringFreq>();
        for (Map.Entry<String, Freq> entry : frequencyMap.entrySet()) {
	    freqList.add(new StringFreq(entry.getKey(), entry.getValue().freq));
        }
	java.util.Collections.sort(freqList, sortFunc);
	for (StringFreq sf : freqList) {
            String result = sf.string;
            if (!wordsOnly) {
                result = result.concat(", " + sf.freq);
            }
            System.out.println(result);//sf.string + ", " + sf.freq);
	}
    }

    private static boolean validateIgnoreLetters(Map<Character, Integer> sentenceLetters,
                                                 Map<Character, Integer> ignoreLetters)
    {
        for (Character k : ignoreLetters.keySet()) {
            if (!sentenceLetters.containsKey(k) || (ignoreLetters.get(k) > sentenceLetters.get(k))) return false;
        }
        return true;
    }

    static class Freq
    {
	BigDecimal freq;

	Freq(BigDecimal freq)
	{
	    this.freq = freq;
	}
    }

    static class StringFreq
    {
	String string;
	BigDecimal freq;

	StringFreq(String string, BigDecimal freq)
	{
	    this.string = string;
	    this.freq = freq;
	}

	static class ByFreq implements Comparator<StringFreq>
	{
	    @Override
	    public int compare(StringFreq e1, StringFreq e2)
	    {
		return e1.freq.compareTo(e2.freq);
	    }  

	}

	static class ByLengthAndFreq implements Comparator<StringFreq>
	{
	    @Override
	    public int compare(StringFreq e1, StringFreq e2)
	    {
		if (e1.string.length() > e2.string.length())
		{
		    return 1;
		}
		if (e1.string.length() < e2.string.length())
		{
		    return -1;
		}

		return e1.freq.compareTo(e2.freq);
	    }  
	}
    }
}
