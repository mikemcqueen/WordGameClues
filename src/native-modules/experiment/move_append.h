#if 0
template <typename T>
typename std::vector<T>::iterator append(std::vector<T>& dst, const std::vector<T>& src)
{
    typename std::vector<T>::iterator result;
    if (dst.empty()) {
        dst = src;
        result = std::begin(dst);
    } else {
        result = dst.insert(std::end(dst), std::cbegin(src), std::cend(src));
    }
    return result;
}

template <typename T>
typename std::vector<T>::const_iterator append(std::vector<T>& dst, std::vector<T>&& src)
{
    typename std::vector<T>::const_iterator result;
    if (dst.empty()) {
        dst = std::move(src);
        result = std::cbegin(dst);
    } else {
        result = dst.insert(std::end(dst),
                             std::make_move_iterator(std::begin(src)),
                             std::make_move_iterator(std::end(src)));
    }
    src.clear();
    src.shrink_to_fit();
    return result;
}
#endif

