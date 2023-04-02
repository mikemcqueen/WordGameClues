#ifndef include_peco_h
#define include_peco_h

#include <algorithm>
#include <numeric>
#include <vector>
//#include <span>

template<int MaxLength>
class Peco {
public:
  // todo: std::span
  Peco(std::vector<uint32_t>& lengths) :
    size_((int)lengths.size())
  {
    for (auto i = 0u; i < lengths.size(); ++i) {
      lengths_[i] = lengths[i];
    }
    reset_state();
  }

  using take_type = std::uint32_t*;

  //const?
  take_type first_combination() {
    reset_indices();
    return take();
  }

  //const?
  take_type next_combination() {
    if (indices_[0] < lengths_[0]) {
      for (int i = size_ - 1; i >= 0; --i) {
	indices_[i]++;
	if (i > 0 && indices_[i] == lengths_[i]) {
	  indices_[i] = 0;
	  continue;
	}
	break;
      }
    }
    return take();
  }

private:
  //const?
  take_type take() {
    if (indices_[0] < lengths_[0]) {
      return indices_.data();
    }
    return nullptr;
  }

  void reset_state() {
    //indices_.resize(size_);
    reset_indices();
    //    state_.resize(lengths_.size());
    //result_.resize(lengths_.size());
    /*
    for (auto i = 0u; i < lengths_.size(); ++i) {
      reset(i);
    }
    */
  }

  void reset_indices() {
    //std::fill(indices_.begin(), indices_.begin() + size_, 0);
    indices_.fill(0);
  }

  /*
  void reset(int i) {
    vector<int>& v = state_[i];
    v.resize(lengths_[i]);
    std::iota(v.begin(), v.end(), 0);
  }
  */

  std::array<std::uint32_t, MaxLength> lengths_;
  std::array<std::uint32_t, MaxLength> indices_;
  int size_;
  //std::vector<std::vector<int>> state_;
  //std::vector<int> result_;
};

#endif //  include_peco_h
