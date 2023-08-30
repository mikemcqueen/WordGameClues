#ifndef include_peco_h
#define include_peco_h

#include <algorithm>
#include <numeric>
#include <vector>
#include <forward_list>
#include <cassert>

class Peco {
public:
  using index_t = uint32_t;
  using IndexVector = std::vector<index_t>;
  using IndexList = std::forward_list<index_t>;
  using IndexListVector = std::vector<IndexList>;
  // TODO: const?
  using take_type = IndexVector*;

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

  // TODO:
  // these static methods actually made sense to be in this class at one point,
  // but I think they're kinda standalone at this point and don't really belong
  // here
  static IndexListVector initial_indices(const std::vector<size_t>& lengths) {
    IndexListVector indexLists;
    indexLists.resize(lengths.size());
    for (size_t i{}; i < indexLists.size(); ++i) {
      initialize_list(indexLists[i], lengths[i]);
    }
    return indexLists;
  }

  static std::vector<IndexVector> to_vectors(const IndexListVector& idx_lists) {
    std::vector<IndexVector> idx_vectors(idx_lists.size());
    for (size_t i{}; i < idx_lists.size(); ++i) {
      const auto& idx_list = idx_lists.at(i);
      auto& idx_vector = idx_vectors.at(i);
      idx_vector.resize(std::distance(idx_list.begin(), idx_list.end()));
      std::copy(idx_list.begin(), idx_list.end(), idx_vector.begin());
    }
    return idx_vectors;
  }

private:
  take_type take() {
    if (done_) return nullptr;
    for (size_t i{}; i < iterators_.size(); ++i) {
      result_[i] = *std::next(iterators_[i]);
    }
    return &result_;
  }

  void reset_iterators(size_t size) {
    iterators_.resize(size);
    for (size_t i{}; i < size; ++i) {
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
  std::vector<index_t> result_;
  bool done_;
};

#endif //  include_peco_h
