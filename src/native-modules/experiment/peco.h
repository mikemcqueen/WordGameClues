#ifndef include_peco_h
#define include_peco_h

#include <algorithm>
#include <numeric>
#include <vector>

class Peco {
public:
  Peco(std::vector<int> lengths) : lengths_(std::move(lengths)) {}

  //const?
  vector<int>* first_combination() {
    reset_state();
    return take();
  }

  //const?
  vector<int>* next_combination() {
    if (!state_[0].empty()) {
      for (int i = (int)state_.size() - 1; i >= 0; --i) {
	state_[i].pop_back();
	if (i > 0 && state_[i].empty()) {
	  reset(i);
	  continue;
	}
	break;
      }
    }
    if (state_[0].empty()) {
      for (auto i = 1u; i < state_.size(); ++i) {
	state_[i].clear();
      }
    }
    return take();
  }

private:
  //const?
  vector<int>* take() {
    result_.clear();
    if (!state_[0].empty()) {
      for (const auto& v : state_) {
	result_.push_back(*v.rbegin());
      }
    }
    return &result_;
  }

  void reset_state() {
    state_.resize(lengths_.size());
    for (auto i = 0u; i < lengths_.size(); ++i) {
      reset(i);
    }
  }

  void reset(int i) {
    vector<int>& v = state_[i];
    v.resize(lengths_[i]);
    std::iota(v.rbegin(), v.rend(), 0);
  }

  std::vector<int> lengths_;
  std::vector<std::vector<int>> state_;
  std::vector<int> result_;
};

#endif //  include_peco_h
