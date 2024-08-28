#pragma once

constexpr char* cuda_strcpy(char* dest, const char* src) {
  int i = 0;
  do {
    dest[i] = src[i];
  } while (src[i++] != 0);
  return dest;
}

constexpr char* cuda_strcat(char* dest, const char* src) {
  int i = 0;
  while (dest[i] != 0)
    i++;
  cuda_strcpy(dest + i, src);
  return dest;
}

// Yet, another good itoa implementation
// returns: the length of the number string
// https://stackoverflow.com/questions/3440726/what-is-the-proper-way-of-implementing-a-good-itoa-function
constexpr int cuda_itoa(int value, char *sp, int radix = 10) {
  char tmp[32];
  char* tp = tmp;
  unsigned v;

  int sign = (radix == 10 && value < 0);
  if (sign)
    v = -value;
  else
    v = (unsigned)value;

  while (v || tp == tmp) {
    auto i = v % radix;
    v /= radix;
    if (i < 10)
      *tp++ = (char)i + '0';
    else
      *tp++ = (char)i + 'a' - 10;
  }

  auto len = int(tp - tmp);

  if (sign) {
    *sp++ = '-';
    len++;
  }

  while (tp > tmp)
    *sp++ = *--tp;

  *sp = '\0';
  return len;
}

