#ifndef include_peco_h
#define include_peco_h

#include <algorithm>
#include <numeric>
#include <vector>
#include <forward_list>
#include <cassert>

class Peco {
public:
  using IndexList = std::forward_list<int>;
  using IndexListVector = std::vector<IndexList>;
  //const?
  using take_type = std::vector<int>*;

  Peco() = delete;
  Peco(IndexListVector&& indexLists) : index_lists_(std::move(indexLists)) {
    assert(index_lists_.size() >= 2);
    result_.resize(index_lists_.size());
  }

  take_type first_combination() {
    reset_iterators(index_lists_.size());
    done_ = false;
    return take();
  }

  take_type next_combination() {
    int i{};
    for (i = iterators_.size() - 1; i >= 0; --i) {
      auto& il = index_lists_[i];
      assert(std::next(iterators_[i]) != il.end());
      ++iterators_[i];
      if (std::next(iterators_[i]) != il.end()) break;
      iterators_[i] = il.before_begin();
    }
    if (i < 0) done_ = true;
    return take();
  }

  static IndexListVector initial_indices(const std::vector<int>& lengths) {
    IndexListVector indexLists;
    indexLists.resize(lengths.size());
    for (size_t i{}; i < indexLists.size(); ++i) {
      initialize_list(indexLists[i], lengths[i]);
    }
    return indexLists;
  }

private:
  take_type take() {
    if (done_) return nullptr;
    for (size_t i{}; i < iterators_.size(); ++i) {
      result_[i] = *std::next(iterators_[i]);
    }
    return &result_;
  }

  void reset_iterators(int size) {
    iterators_.resize(size);
    for (int i{}; i < size; ++i) {
      assert(!index_lists_[i].empty());
      iterators_[i] = index_lists_[i].before_begin();
    }
  }

  static void initialize_list(IndexList& indexList, int size) {
    indexList.clear();
    for (int i{ size - 1 }; i >= 0; --i) {
      indexList.emplace_front(i);
    }
  }

  IndexListVector index_lists_;
  std::vector<IndexList::const_iterator> iterators_;
  std::vector<int> result_;
  bool done_;
};

#endif //  include_peco_h
