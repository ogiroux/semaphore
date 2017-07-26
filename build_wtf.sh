xcrun clang++ -D__test_wtf -O3 -W -ISource/WTF -ISource/WTF/benchmarks -LWebKitBuild/Release -lWTF -framework Foundation -licucore -std=c++14 -fvisibility=hidden -I../../GitHub/semaphore/include -O3 -std=c++14 -pthread ../../GitHub/semaphore/lib/semaphore.cpp ../../GitHub/semaphore/test.cpp -lpthread -o test

