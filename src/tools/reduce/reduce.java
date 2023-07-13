import java.io.BufferedReader;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;


class Reduce
{
    private static int DEFAULT_LINE_LEVEL = 3;
    private static Set<String> words = new TreeSet<String>();

    public static void main(String[] args) throws IOException
    {
        if (args.length < 2)
        {
            System.out.println("usage: reduce <words-file> <words-to-remove-file> [words-to-remove-file...]");
            return;
        }

        /*
        int level = Integer.valueOf(args[0]);
        if (level < 1 || level > 5)
        {
            System.out.println("first argument must be an integer in the range [1..5]");
            return;
        }
        */
        int level = DEFAULT_LINE_LEVEL;

        load(args[0], words);

        for (int argIndex = 1; argIndex < args.length; ++argIndex)
        {
            reduce(words, args[argIndex], level);
        }
        
        dump(words);
    }

    public static void load(String filename, Set<String> words) throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line;
        while ((line = reader.readLine()) != null)
        {
            words.add(line);
        }
    }

    public static void reduce(Set<String> words, String filename, int level) throws IOException
    {
        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line;
        while ((line = reader.readLine()) != null)
        {
            int lineLevel = DEFAULT_LINE_LEVEL;
            if (line.indexOf(",") != -1)
            {
                String[] values = line.split(",");
                line = values[0];
                lineLevel = Integer.valueOf(values[1].trim());
            }
            if (lineLevel < level)
            {
                words.remove(line);
            }
        }
    }

    public static void dump(Set<String> words)
    {
        for (String word : words)
        {
            System.out.println(word);
        }
    }

}
