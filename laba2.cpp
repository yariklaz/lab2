#include <algorithm>
#include <execution>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <thread>
#include <cmath>
#include <functional>
#include <sstream>

using clk = std::chrono::high_resolution_clock;
using ns = std::chrono::duration<double, std::milli>;

struct Timer {
    clk::time_point start;
    Timer() : start(clk::now()) {}
    void reset() { start = clk::now(); }
    double elapsed_ms() const { return ns(clk::now() - start).count(); }
};

template <typename T, typename UnaryOp>
void parallel_transform_custom(const std::vector<T>& in, std::vector<T>& out, UnaryOp op, size_t K) {
    const size_t n = in.size();
    if (n == 0) return;
    if (K <= 1) {
        std::transform(in.begin(), in.end(), out.begin(), op);
        return;
    }
    K = std::min(K, n);
    std::vector<std::thread> threads;
    threads.reserve(K);
    size_t base = n / K;
    size_t rem = n % K;
    size_t offset = 0;
    for (size_t i = 0; i < K; ++i) {
        size_t len = base + (i < rem ? 1 : 0);
        size_t start = offset;
        size_t end = start + len; 
        threads.emplace_back([start, end, &in, &out, op]() {
            std::transform(in.begin() + start, in.begin() + end, out.begin() + start, op);
            });
        offset = end;
    }
    for (auto& t : threads) t.join();
}

std::vector<double> make_random_vector(size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::vector<double> v; v.reserve(n);
    for (size_t i = 0; i < n; ++i) v.push_back(dist(rng));
    return v;
}

inline double fast_op(double x) {
    return x + 1.0;
}

inline double slow_op(double x) {
    double y = x;
    for (int i = 0; i < 50; ++i) {
        y = std::sqrt(y + 1.0) * std::cbrt(y + 1.0) - std::log(y + 1.0000001);
    }
    return y;
}

template <typename T, typename UnaryOp>
double run_transform_seq(const std::vector<T>& in, std::vector<T>& out, UnaryOp op) {
    Timer t;
    std::transform(in.begin(), in.end(), out.begin(), op);
    return t.elapsed_ms();
}

template <typename Policy, typename T, typename UnaryOp>
double run_transform_policy(Policy&& policy, const std::vector<T>& in, std::vector<T>& out, UnaryOp op) {
    Timer t;
    std::transform(std::forward<Policy>(policy), in.begin(), in.end(), out.begin(), op);
    return t.elapsed_ms();
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const unsigned hw_threads = std::thread::hardware_concurrency();
    std::cout << "Hardware threads (std::thread::hardware_concurrency): " << hw_threads << "\n";

    std::vector<size_t> sizes = { 1000, 10'000, 100'000, 1'000'000, 5'000'000 };

    std::vector<uint64_t> seeds = { 12345ULL, 67890ULL };

    std::vector<size_t> K_values;
    for (size_t k = 1; k <= hw_threads * 4 + 8; k *= 2) K_values.push_back(k);
    K_values.push_back(hw_threads);
    std::sort(K_values.begin(), K_values.end());
    K_values.erase(std::unique(K_values.begin(), K_values.end()), K_values.end());

    struct OpCase { std::string name; std::function<double(double)> op; };
    std::vector<OpCase> cases = {
        {"fast", fast_op},
        {"slow", slow_op}
    };

    std::cout << "\n===== transform benchmark (will print times in ms) =====\n";
    std::cout << "Note: compile with -O3 for release timings; to compare optimization levels, build both -O0 and -O3.\n\n";

    for (auto& c : cases) {
        std::cout << "OPERATION: " << c.name << "\n";
        for (size_t n : sizes) {
            std::vector<double> in = make_random_vector(n, seeds[0] + n);
            std::vector<double> out(n);

            std::cout << "\nSequence length: " << n << "\n";

            double t_seq = run_transform_seq(in, out, c.op);
            std::cout << std::setw(30) << std::left << "std::transform (sequential)" << std::setw(12) << std::right << t_seq << " ms\n";

            double t_policy_seq = run_transform_policy(std::execution::seq, in, out, c.op);
            std::cout << std::setw(30) << std::left << "std::transform (policy: seq)" << std::setw(12) << std::right << t_policy_seq << " ms\n";

            double t_policy_par = run_transform_policy(std::execution::par, in, out, c.op);
            std::cout << std::setw(30) << std::left << "std::transform (policy: par)" << std::setw(12) << std::right << t_policy_par << " ms\n";

            double t_policy_par_unseq = run_transform_policy(std::execution::par_unseq, in, out, c.op);
            std::cout << std::setw(30) << std::left << "std::transform (policy: par_unseq)" << std::setw(12) << std::right << t_policy_par_unseq << " ms\n";

            std::cout << "\nCustom parallel transform (split into K parts). Results (K | time ms):\n";
            std::cout << std::setw(8) << "K" << std::setw(16) << "time(ms)" << "\n";
            double best_time = std::numeric_limits<double>::infinity();
            size_t best_K = 1;
            for (size_t K : K_values) {
                std::vector<double> out_custom(n);
                Timer t;
                parallel_transform_custom(in, out_custom, c.op, K);
                double elapsed = t.elapsed_ms();
                std::cout << std::setw(8) << K << std::setw(16) << elapsed << "\n";
                if (elapsed < best_time) { best_time = elapsed; best_K = K; }
            }
            std::cout << "Best K = " << best_K << "  (time = " << best_time << " ms)\n";
            if (hw_threads > 0) {
                std::cout << "Best K / hardware_threads = " << std::fixed << std::setprecision(2)
                    << (double)best_K / (double)hw_threads << "\n";
            }

            std::cout << "\n---------------------------------------------------------------\n";
        }
        std::cout << "\n===============================================================\n\n";
    }

    std::cout << "All experiments finished. Redirect output to a file if you want to attach results to the report.\n";
    std::cout << "Example: ./transform_benchmark > result.txt\n";
    return 0;
}