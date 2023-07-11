import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.math.MathContext;
import java.math.RoundingMode;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.util.Comparator;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashSet;
import java.util.LinkedHashMap;
import java.util.HashSet;
import java.util.TreeMap;



class Lettercount
{
    private static Map<Character, Integer> letterCounts = new TreeMap<Character, Integer>();
    private static Map<Character, Integer> ignoreLetterCounts = new HashMap<Character, Integer>();
    private static Map<Character, BigDecimal> letterFrequencyMap = new HashMap<Character, BigDecimal>();
    private static MathContext mc = new MathContext(5, RoundingMode.HALF_UP);
    private static BigDecimal TWO = new BigDecimal(2, mc);
    private static BigDecimal ONE = new BigDecimal(1, mc);
    private static BigDecimal ONE_HALF = new BigDecimal(.5, mc);
    private static BigDecimal CV_HIGH = new BigDecimal(1.8, mc);
    private static BigDecimal CV_AVERAGE = new BigDecimal(1.59, mc);
    private static BigDecimal CV_LOW = new BigDecimal(1.4, mc);
    public static BigDecimal total;
    public static DecimalFormat df = new DecimalFormat();

    static
    {
        df.setMaximumFractionDigits(2);
        df.setMinimumFractionDigits(2);
        df.setGroupingUsed(false);
    }

    static
    {
        letterFrequencyMap.put('a',     new BigDecimal(8.167, mc));
        letterFrequencyMap.put('b',     new BigDecimal(1.492, mc));
        letterFrequencyMap.put('c',     new BigDecimal(2.782, mc));
        letterFrequencyMap.put('d',     new BigDecimal(4.253, mc));
        letterFrequencyMap.put('e',     new BigDecimal(12.702, mc));
        letterFrequencyMap.put('f',     new BigDecimal(2.228, mc));
        letterFrequencyMap.put('g',     new BigDecimal(2.015, mc));
        letterFrequencyMap.put('h',     new BigDecimal(6.094, mc));
        letterFrequencyMap.put('i',     new BigDecimal(6.966, mc));
        letterFrequencyMap.put('j',     new BigDecimal(0.153, mc));
        letterFrequencyMap.put('k',     new BigDecimal(0.772, mc));
        letterFrequencyMap.put('l',     new BigDecimal(4.025, mc));
        letterFrequencyMap.put('m',     new BigDecimal(2.406, mc));
        letterFrequencyMap.put('n',     new BigDecimal(6.749, mc));
        letterFrequencyMap.put('o',     new BigDecimal(7.507, mc));
        letterFrequencyMap.put('p',     new BigDecimal(1.929, mc));
        letterFrequencyMap.put('q',     new BigDecimal(0.095, mc));
        letterFrequencyMap.put('r',     new BigDecimal(5.987, mc));
        letterFrequencyMap.put('s',     new BigDecimal(6.327, mc));
        letterFrequencyMap.put('t',     new BigDecimal(9.056, mc));
        letterFrequencyMap.put('u',     new BigDecimal(2.758, mc));
        letterFrequencyMap.put('v',     new BigDecimal(0.978, mc));
        letterFrequencyMap.put('w',     new BigDecimal(2.360, mc));
        letterFrequencyMap.put('x',     new BigDecimal(0.150, mc));
        letterFrequencyMap.put('y',     new BigDecimal(1.974, mc));
        letterFrequencyMap.put('z',     new BigDecimal(0.074, mc));
    }
 
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: java Lettercount [-r] <filename> [string1 ... stringN]");
            return;
        }
        
        boolean quiet = false;
        boolean remainOnly = false;
        int argIndex = 0;

        if (args[argIndex].equals("-r")) {
            ++argIndex;
            quiet = true;
            remainOnly = true;
        }

        String sentence = load(args[argIndex++]);
        int totalChars = sentence.length();
        if (!quiet) {
            System.out.println("total: " + totalChars);
        }
        
        String ignoring = "";
        for (; argIndex < args.length; ++argIndex) {
            ignoreLetters(args[argIndex], ignoreLetterCounts);
            ignoring = ignoring.concat(args[argIndex]);
        }
        Map<Character, Integer> sentenceLetterCounts = new HashMap<>();
        ignoreLetters(sentence, sentenceLetterCounts);
        if (!validateIgnoreLetters(sentenceLetterCounts, ignoreLetterCounts)) {
            System.out.println("ignore letters: '" + ignoring + "' do not exist in '" + sentence + "'");
            return;
        }
        if (!quiet) {
            System.out.println("ignoring: " + ignoring);
        }
        
        String vowels = "aeiou";
        int numVowels = 0;
        String remain = "";
        for (char ch : sentence.toCharArray()) {
            if (!Character.isLowerCase(ch)) {
                continue;
            }
            if (isIgnoreLetter(ch)) {
                continue;
            }
            if (vowels.indexOf(ch) != -1) {
                ++numVowels;
            }
            int count = 0;
            if (letterCounts.containsKey(ch)) {
                count = letterCounts.get(ch);
            }
            letterCounts.put(ch, ++count);
            remain += ch;
        }
        total = new BigDecimal(remain.length());
        if (remainOnly) {
            System.out.println(remain);
            return;
        }
        if (!quiet) {
            System.out.println("remain: " + remain.length() + ", " + remain);

            Factor factor = new Factor(remain.length(), numVowels);
            boolean negative = factor.factor.compareTo(factor.expected) == -1;
            String actual = negative ? "(" + df.format(factor.actual) + ")" : df.format(factor.actual);
            if (factor.actual.compareTo(CV_HIGH) == 1) {
                actual += " +++";
            }
            else if (factor.actual.compareTo(CV_LOW) == -1) {
                //System.out.println(df.format(factor.factor) + " is less than " + df.format(CV_LOW));
                actual += " ---";
            }
            System.out.println("C/V ratio: " + df.format(factor.expected) + " " + actual);
        }
        dump(letterCounts);
    }
    
    private static String load(String filename) throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        return reader.readLine().toLowerCase();
    }

    private static void ignoreLetters(String word, Map<Character, Integer> map)
    {
        for (char ch : word.toCharArray())
        {
            if (!Character.isLowerCase(ch))
            {
                continue;
            }
            int count = 0;
            if (map.containsKey(ch))
            {
                count = map.get(ch);
            }
            map.put(ch, ++count);
        }
    }

    private static boolean isIgnoreLetter(char ch) 
    {
        if (!ignoreLetterCounts.containsKey(ch))
        {
            return false;
        }
        int count = ignoreLetterCounts.get(ch);
        if (count > 1)
        {
            ignoreLetterCounts.put(ch, --count);
        }
        else
        {
            ignoreLetterCounts.remove(ch);
        }
        return true;
    }

    private static void dump(Map<Character, Integer> map)
    {
        List<Map.Entry<Character, Integer>> entryList = new ArrayList<>(map.entrySet());
        java.util.Collections.sort(entryList, new ByFactor());
        for (Map.Entry<Character, Integer> entry : entryList) {
            Factor factor = getFactor(entry);
            
            String factorString = df.format(factor.factor);

            if (factor.factor.compareTo(TWO) == 1) {
                factorString += " +++";
            }
            else if (factor.factor.compareTo(ONE_HALF) == -1) {
                factorString += " ---";
                //factorString = "-" + factor.factor.toString();
            }
            boolean negative = factor.factor.compareTo(ONE) == -1;
            String actual = negative ? "(" + df.format(factor.actual) + ")" : df.format(factor.actual);
                
            System.out.println("'" + entry.getKey() + "' : " + entry.getValue() + "  " +
                               df.format(factor.expected) + "  " + actual + "  " + factorString);
        }
    }

    private static boolean validateIgnoreLetters(Map<Character, Integer> sentenceLetters,
                                                 Map<Character, Integer> ignoreLetters)
    {
        for (Character k : ignoreLetters.keySet()) {
            if (!sentenceLetters.containsKey(k) || (ignoreLetters.get(k) > sentenceLetters.get(k))) {
                return false;
            }
        }
        return true;
    }

    private static Factor getFactor(Map.Entry<Character, Integer> entry)
    {
        return new Factor(entry);
    }

    private static class Factor
    {
        BigDecimal expected;
        BigDecimal actual;
        BigDecimal factor;
        
        public Factor(Map.Entry<Character, Integer> entry)
        {
            actual = new BigDecimal(entry.getValue(), mc).divide(total, mc).scaleByPowerOfTen(2);
            expected = letterFrequencyMap.get(entry.getKey());
            factor = actual.divide(expected, mc);
        }

        public Factor(int length, int numVowels) {
            actual =  new BigDecimal(length - numVowels, mc)
                .divide(new BigDecimal(numVowels, mc), mc);
            expected = CV_AVERAGE;
            factor = actual.divide(expected, mc);
        }

    }

    /*
    {
        List<Map.Entry<String, Integer>> entryList = new ArrayList<Map.Entry<String, Integer>>(map.entrySet());
        java.util.Collections.sort(entryList, new ByValue());
        for (Map.Entry<String, Integer> entry : entryList)
        {
            System.out.println(entry.getKey() + ", " + entry.getValue());
        }
    }
    */

    public static class ByFactor implements Comparator<Map.Entry<Character, Integer>>
    {
        public ByFactor() {}

        public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2)
        {
            return getFactor(o1).factor.compareTo(getFactor(o2).factor);
        }

        public boolean equals(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2)
        {
            return getFactor(o1).factor.compareTo(getFactor(o2).factor) == 0;
        }

    }
}
