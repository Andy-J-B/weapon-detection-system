#include <iostream>
#include <boost/asio.hpp>

namespace asio = boost::asio;

int main() {
    // 1. Create the I/O execution context
    asio::io_context io;

    // 2. Create a timer object associated with the context
    asio::steady_timer timer(io, asio::chrono::seconds(5));

    std::cout << "Waiting 5 seconds..." << std::endl;

    // 3. Wait synchronously for the timer to expire
    timer.wait();

    std::cout << "Timer expired!" << std::endl;

    return 0;
}