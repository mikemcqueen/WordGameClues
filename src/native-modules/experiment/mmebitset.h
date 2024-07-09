/*******************************************************************************
    Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/
#ifndef _MMEBITSET_H
#define _MMEBITSET_H 1

#ifndef assert
#include <assert.h>
#endif // assert
// TODO: remove
#ifdef __CUDA_ARCH__
#include <cuda_runtime.h>
#endif
#include <cstdint>

//typedef unsigned int uint32_t;

namespace mme {

template <typename SizeT>
inline void hash_combine(SizeT& seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

//
// Similar to C++ std::bitset, but simplified and with a few specialized
// methods added: extract(), insert().
//
template <int size>
class bitset {
public:
  using word_type = uint32_t;
  //constexpr int word_count = (size+31)/32;

  constexpr bitset() { reset(); }

    /*
    explicit bitset(uint32_t b) {
        reset();
        bits[0] = b;
    }
    */

  // word count
  constexpr inline static auto wc() {
    return (size + 31) / 32;
  }

  constexpr bool any() const {
    for (int i{}; i < wc(); ++i) {
      if (bits[i])
        return true;
    }
    return false;
  }

  constexpr auto count() const {
    int n{};
    for (int i{}; i < wc(); ++i) {
#ifndef __CUDA_ARCH__ // v- host code
      auto w = bits[i];
      while (w) {
        w &= (w - 1);
        n++;
      }
#else // v- device code  ^- host code
      n += __popc(bits[i]);
#endif
    }
    return n;
  }

  bitset<size>& operator|=(const bitset<size>& other) {
    for (int i{}; i < wc(); ++i)
      bits[i] |= other.bits[i];
    return *this;
  }

  // test "XOR compatibility" without constructing new object
  constexpr bool intersects(const bitset<size>& other) const {
    for (int i{}; i < wc(); ++i) {
      if (bits[i] & other.bits[i]) {
        return true;
      }
    }
    return false;
  }

  // test "AND compatibility" without constructing new object
  constexpr bool is_subset_of(const bitset<size>& other) const {
    for (int i{}; i < wc(); ++i) {
      if (bits[i] && ((bits[i] & other.bits[i]) != bits[i])) {
        return false;
      }
    }
    return true;
  }

#ifdef __CUDA_ARCH__
  // shared memory implementation, one bit word every blockDim.x words
  constexpr bool shared_intersects(word_type* shrd_mem) const {
    for (unsigned int i{}, o{threadIdx.x}; i < wc(); i++, o += blockDim.x) {
      if (bits[i] & shrd_mem[o]) return true;
    }
    return false;
  }
#endif

  constexpr word_type word(int i) const {
    return bits[i];
  }

  void reset() {
    for (int i=0; i<(size+31)/32; i++)
      bits[i] = 0;
  }
  
  /*
  void set() {
    for (int i=0; i<wc(); i++)
      bits[i] = ~0;
  }
  */
  
  void set(unsigned int bit, bool value = true) {
    unsigned int mask = 1 << (bit & 0x1F);
    if (value) bits[bit/32] |= mask;
    else bits[bit/32] &= ~mask;
  }
  
  constexpr bool test(int word, unsigned int relative_bit) const {
    unsigned int mask = 1 << relative_bit;
    return (bits[word] & mask) ? true : false;
  }

  constexpr bool test(unsigned int bit) const {
    unsigned int mask = 1 << (bit & 0x1F);
    return (bits[bit/32] & mask) ? true : false;
  }

  #if 0
  constexpr bool operator [] (unsigned int bit) const {
    return test(bit);
  }
  #endif
  
  bitset<size> operator | (const bitset<size> &a) const {
    bitset<size> rv;
    for (int i=0; i<wc(); i++)
      rv.bits[i] = bits[i] | a.bits[i];
    return rv;
  }
  
  constexpr bitset<size> operator & (const bitset<size> &a) const {
    bitset<size> rv;
    for (int i=0; i<wc(); i++)
      rv.bits[i] = bits[i] & a.bits[i];
    return rv;
  }
  
  bitset<size> operator - (const bitset<size> &a) const {
    bitset<size> rv;
    for (int i=0; i<wc(); i++)
      rv.bits[i] = bits[i] & ~a.bits[i];
    return rv;
  }
  
  bool operator == (const bitset<size> &x) const {
    for (int i=0; i<wc(); i++)
      if (bits[i] != x.bits[i])
        return false;
    return true;
  }
  
  bool operator != (const bitset<size> &x) const {
    return !(*this == x);
  }

    /*
    template <int h, int l>
    bitset<h-l+1> extract() const {
        bitset<h-l+1> rv;

        for (int i=0; i<=h-l; i++) {
            rv.set(i, (*this)[l+i]);
        }

        return rv;
    }

    template <int h, int l>
    void insert(bitset<h-l+1> v) {
        for (int i=0; i<=h-l; i++) {
            this->set(l+i, v[i]);
        }
    }

    template <int h, int l>
    void insert(uint32_t v) {
        bitset<h-l+1> bv(v);
        insert<h,l>(bv);
    }

    uint32_t toUint() const {
        assert(size <= 32);
        return bits[0];
    }
    */

  auto hash() const {
    size_t seed{};
    for (int i=0; i<wc(); i++)
      hash_combine(seed, (size_t)bits[i]);
    return seed;
  }

protected:
  word_type bits[wc()];
};


} // namespace mme

#endif // _MMEBITSET_H
