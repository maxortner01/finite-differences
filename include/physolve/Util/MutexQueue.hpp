#pragma once

#include <queue>
#include <mutex>

namespace ps::Util
{
    template<typename T>
    struct MutexQueue
    {
        void push(T value)
        {
            std::lock_guard g(mutex);
            queue.push(value);
        }

        T pop()
        {
            std::lock_guard g(mutex);
            T v = std::move(queue.front());
            queue.pop();
            return v;
        }
        
    private:
        std::queue<T> queue;
        std::mutex mutex;
    };
}