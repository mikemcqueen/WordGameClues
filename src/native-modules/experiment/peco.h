#ifndef INCLUDE_PECO_H
#define INCLUDE_PECO_H

#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <forward_list>
#include <numeric>
#include <ranges>
#include <vector>

class Peco {
public:
  using index_t = uint32_t;
  using IndexVector = std::vector<index_t>;
  using IndexList = std::forward_list<index_t>;
  using IndexListVector = std::vector<IndexList>;
  // TODO: std::optional
  using take_type = IndexVector*;

private:
  
  template <typename T>
  // TODO: requires integral_type
  static void initialize_list(IndexList& idx_list, T size) {
    idx_list.clear();
    for (int i{(int)size - 1}; i >= 0; --i) {
      idx_list.push_front(i);
    }
  }

  static void make_addends_helper(int sum, int count, int start,
    std::vector<int>& current, std::vector<std::vector<int>>& result) {
    if (!count) {
      if (!sum) {
        result.push_back(current);
      }
      return;
    }
    for (auto i = start; i <= sum; ++i) {
      current.push_back(i);
      make_addends_helper(sum - i, count - 1, i, current, result);
      current.pop_back();  // backtrack
    }
  }

public:
  // TODO: it may have made sense for some of these static methods to be in this
  // class at one time, but it has become rather polluted with them, and it may
  // be time to re-think the organization of this class/namespace
  // maybe something like namespace peco { class Generator }
  template <typename T>
  // TODO: requires integral_type
  static auto initial_indices(const std::vector<T>& lengths) {
    IndexListVector idx_lists;
    idx_lists.resize(lengths.size());
    for (size_t i{}; i < idx_lists.size(); ++i) {
      initialize_list(idx_lists.at(i), lengths.at(i));
    }
    return idx_lists;
  }

  static auto to_vectors(const IndexListVector& idx_lists) {
    std::vector<IndexVector> idx_vectors(idx_lists.size());
    for (size_t i{}; i < idx_lists.size(); ++i) {
      const auto& idx_list = idx_lists.at(i);
      auto& idx_vector = idx_vectors.at(i);
      idx_vector.resize(std::distance(idx_list.begin(), idx_list.end()));
      std::copy(idx_list.begin(), idx_list.end(), idx_vector.begin());
    }
    return idx_vectors;
  }

  // converts a std::vector (cm::IndexList) to a std::forward_list
  // (Peco::IndexList). Yes poor name choices.
  static auto make_index_list(const IndexVector& idx_vec) {
    IndexList idx_list;
    for (auto i : idx_vec | std::views::reverse) {
      idx_list.push_front(i);
    }
    return idx_list;
  }

  static auto make_index_list(size_t size) {
    IndexList idx_list;
    initialize_list(idx_list, size);
    return idx_list;
  }

  // TODO: add 'max' param, loop over it
  static auto make_addends(int sum, int count) {
    std::vector<std::vector<int>> result;
    std::vector<int> current;
    make_addends_helper(sum, count, 1, current, result);
    return result;
  }

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
    for (i = (int)iterators_.size() - 1; i >= 0; --i) {
      const auto& il = index_lists_[i];
      assert(std::next(iterators_[i]) != il.end());
      ++iterators_[i];
      if (std::next(iterators_[i]) != il.end()) break;
      iterators_[i] = il.before_begin();
    }
    if (i < 0) done_ = true;
    return take();
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

  IndexListVector index_lists_;
  std::vector<IndexList::const_iterator> iterators_;
  std::vector<index_t> result_;
  bool done_;
};

#endif //  INCLUDE_PECO_H
