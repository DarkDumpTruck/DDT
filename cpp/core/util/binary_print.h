#pragma once
#include <cstdio>
#include <string>

using std::string;

template <class T>
void bprint(T x, int begin = 8) {
  while (begin) {
    begin--;
    putchar(x & (1 << begin) ? '1' : '0');
  }
}

template <class T>
std::string bsprint(T x, int begin = 8) {
  std::string result;
  result.reserve(begin);
  while (begin) {
    begin--;
    result += (x & (((T)1) << begin) ? '1' : '0');
  }
  return result;
}