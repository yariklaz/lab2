// Microsoft Visual C++ Compiler (MSVC) C++20

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <random>
#include <thread>
#include <chrono>

using namespace std;
using namespace chrono;

template<typename Func>
double measure_time(Func fn) {
    auto time_start = steady_clock::now();
    fn();
    auto time_end = steady_clock::now();
    duration<double> delta = time_end - time_start;
    return delta.count();
}

inline int fast_operation(int value) {
    return value * 2;
}

inline int slow_operation(int value) {
    int res = value;
    for (int i = 0; i < 100; i++) {
        res = (res * 3 + 7) % 1000003;
    }
    return res;
}

template<typename InputIt, typename OutputIt, typename Func>
void parallel_transform(InputIt beginIt, InputIt endIt, OutputIt resultIt, Func op, int threadCount) {
    int total = endIt - beginIt;
    if (threadCount <= 1 || total == 0) {
        transform(beginIt, endIt, resultIt, op);
        return;
    }

    threadCount = min(threadCount, total);
    int partSize = total / threadCount;
    vector<thread> threads;

    for (int i = 0; i < threadCount; i++) {
        int partStart = i * partSize;
        int partEnd = (i == threadCount - 1) ? total : (i + 1) * partSize;
        threads.emplace_back([=]() {
            transform(beginIt + partStart, beginIt + partEnd, resultIt + partStart, op);
            });
    }

    for (auto& th : threads) th.join();
}

void run_test(const vector<int>& inputData, bool useSlow) {
    auto func = useSlow ? slow_operation : fast_operation;
    vector<int> output(inputData.size());
    cout << "\nTest (" << (useSlow ? "slow" : "fast") << " operation)\n";

    double t_basic = measure_time([&]() {
        transform(inputData.begin(), inputData.end(), output.begin(), func);
        });
    cout << "std::transform: " << t_basic << " sec\n";

    double t_seq = measure_time([&]() {
        transform(execution::seq, inputData.begin(), inputData.end(), output.begin(), func);
        });
    double t_par = measure_time([&]() {
        transform(execution::par, inputData.begin(), inputData.end(), output.begin(), func);
        });
    double t_par_unseq = measure_time([&]() {
        transform(execution::par_unseq, inputData.begin(), inputData.end(), output.begin(), func);
        });

    cout << "seq: " << t_seq << " sec\n";
    cout << "par: " << t_par << " sec\n";
    cout << "par_unseq: " << t_par_unseq << " sec\n";

    cout << "\nMy parallel transform (threads test):\n";
    unsigned maxThreads = thread::hardware_concurrency();
    double bestTime = 1e9;
    int bestThreads = 1;

    for (int k = 1; k <= (int)maxThreads * 4; k++) {
        double current = measure_time([&]() {
            parallel_transform(inputData.begin(), inputData.end(), output.begin(), func, k);
            });
        cout << "Threads = " << k << " -> " << current << " sec\n";
        if (current < bestTime) {
            bestTime = current;
            bestThreads = k;
        }
    }

    cout << "Best = " << bestThreads
        << ", hardware threads = " << maxThreads
        << ", ratio = " << (double)bestThreads / maxThreads << endl;
}

int main() {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(1, 1000);
    vector<int> sizes = { 1000, 100000, 1000000 };

    for (int size : sizes) {
        vector<int> arr(size);
        generate(arr.begin(), arr.end(), [&]() { return dist(gen); });
        cout << "\n===== N = " << size << " =====\n";
        run_test(arr, false);
        run_test(arr, true);
    }
    return 0;
}