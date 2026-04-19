// pairs.cpp - Find word pairs that fit within a sentence's letter pool.
// Usage: pairs <wordlist> <sentence_file> [min_word_length]
// Compile: g++ -std=c++23 -O2 -o pairs pairs.cpp

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <print>
#include <string>
#include <vector>

static constexpr int ALPHA = 26;

using Counts = uint8_t[ALPHA];

static void letter_counts(const char* s, Counts out) {
    memset(out, 0, ALPHA);
    for (; *s; ++s) {
        char c = *s;
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c >= 'a' && c <= 'z') out[c - 'a']++;
    }
}

static bool subtract(const Counts pool, const Counts wc, Counts out) {
    for (int i = 0; i < ALPHA; ++i) {
        if (wc[i] > pool[i]) return false;
        out[i] = pool[i] - wc[i];
    }
    return true;
}

static bool fits(const Counts pool, const Counts wc) {
    for (int i = 0; i < ALPHA; ++i) {
        if (wc[i] > pool[i]) return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::println(stderr, "Usage: {} <wordlist> <sentence_file> [min_word_length]", argv[0]);
        return 1;
    }

    setvbuf(stdout, nullptr, _IOFBF, 1 << 22);

    const int min_len = (argc > 3) ? atoi(argv[3]) : 1;

    // Read sentence
    FILE* sf = fopen(argv[2], "r");
    if (!sf) { perror(argv[2]); return 1; }
    char sentence[4096] = {};
    if (!fgets(sentence, sizeof(sentence), sf)) { fclose(sf); return 1; }
    fclose(sf);
    size_t slen = strlen(sentence);
    while (slen > 0 && (sentence[slen-1] == '\n' || sentence[slen-1] == '\r'))
        sentence[--slen] = '\0';

    Counts sc;
    letter_counts(sentence, sc);

    // Read and filter wordlist
    FILE* wf = fopen(argv[1], "r");
    if (!wf) { perror(argv[1]); return 1; }

    struct Word {
        std::string text;
        Counts wc;
    };

    std::vector<Word> words;
    char line[4096];
    while (fgets(line, sizeof(line), wf)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if ((int)len < min_len) continue;
        Counts wc;
        letter_counts(line, wc);
        Counts remaining;
        if (subtract(sc, wc, remaining)) {
            words.push_back({std::string(line), {}});
            memcpy(words.back().wc, wc, ALPHA);
        }
    }
    fclose(wf);

    const size_t n = words.size();
    long long total = 0;
    Counts remaining;

    for (size_t i = 0; i < n; ++i) {
        const Word& w1 = words[i];
        if (!subtract(sc, w1.wc, remaining)) continue;
        for (size_t j = i + 1; j < n; ++j) {
            const Word& w2 = words[j];
            if (fits(remaining, w2.wc)) {
                std::println("{},{}", w1.text, w2.text);
                ++total;
            }
        }
    }

    std::println("{}", total);
    return 0;
}
