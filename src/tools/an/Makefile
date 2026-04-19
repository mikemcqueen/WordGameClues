ann: ancc.o
	g++ -g3 -ggdb -o ann ancc.o

ancc.o: ancc.cc
	g++ -std=c++20 --debug -D_GLIBCXX_DEBUG -Wall -Wextra -Wno-char-subscripts -c ancc.cc

pairs: pairs.cpp
	g++ -std=c++23 -O2 -Wall -Wextra -o pairs pairs.cpp

gan_qsortcc.o: gan_qsortcc.cc
	g++ -std=c++20 --debug -D_GLIBCXX_DEBUG -Wall -Wextra -Wno-register -c gan_qsortcc.cc
