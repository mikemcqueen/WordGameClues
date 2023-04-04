#ifndef include_peco_h
#define include_peco_h

#include <algorithm>
#include <numeric>
#include <vector>
//#include <span>

class Peco {
public:
  // todo: std::span
  Peco(std::vector<int>& lengths) : lengths_(std::move(lengths)) {
    max_ = 1;
    for (const auto len : lengths_) {
      max_ *= len;
    }
    indices_.resize(lengths_.size());
  }

  //const?
  using take_type = std::vector<int>*;

  take_type first_combination() {
    index_ = 0;
    reset_indices();
    return take();
  }

  take_type next_combination() {
    if (index_ < max_) ++index_;
    return take();
  }

private:
  take_type take() {
    if (index_ == max_) return nullptr;
    int remain = index_;
    for (int i = (int)lengths_.size() - 1; i >= 0; --i) {
      auto val = remain % lengths_[i];
      indices_[i] = val;
      if (val > 0) break;
      remain /= lengths_[i];
    }
    return &indices_;
  }

  void reset_state() {
    //    state_.resize(lengths_.size());
    //result_.resize(lengths_.size());
    /*
    for (auto i = 0u; i < lengths_.size(); ++i) {
      reset(i);
    }
    */
  }

  void reset_indices() {
    std::fill(indices_.begin(), indices_.begin() + lengths_.size(), 0);
    //indices_.fill(0);
  }

  /*
  void reset(int i) {
    vector<int>& v = state_[i];
    v.resize(lengths_[i]);
    std::iota(v.begin(), v.end(), 0);
  }
  */

  std::vector<int> lengths_;
  std::vector<int> indices_;
  int max_;
  int index_;
  //std::vector<std::vector<int>> state_;
  //std::vector<int> result_;
};

#endif //  include_peco_h
