#pragma once
#include <thread>

#ifdef NUM_THREADS
const int ProcessorCnt = NUM_THREADS;
#else
const int ProcessorCnt = std::max(std::thread::hardware_concurrency(), 1U);
#endif

