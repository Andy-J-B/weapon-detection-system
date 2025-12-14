/********************************************************************
 *  Boost.Asio HTTP POST server for ESP32‑CAM snapshots
 *
 *  - Listens on TCP port 8080 
 *  - Parses HTTP headers, extracts Content‑Length
 *  - Reads the JPEG payload into a std::vector<char>
 *  - Sends a minimal HTTP/1.1 200 OK response
 *
 *  Hook OpenCV processing after the body has been collected
 *  (see comment in extract_body()).
 *
 *  Build (Linux/macOS):
 *      g++ -std=c++14 -O2 -Wall -Wextra -lboost_system -lpthread \
 *          -o esp32_server esp32_server.cpp
 *
 *  Requires Boost (>=1.66 for std::make_shared with asio) and a C++11+
 *  compiler.
 ********************************************************************/

#include <boost/asio.hpp>
#include <iostream>
#include <memory> // Memory management, smart pointers
#include <string>
#include <unordered_map> // O(1) lookup time dict
#include <vector>
#include <algorithm> // Used for algorithms to help with helper functions? like find_if_not
#include <cctype> // character logic
#include <sstream> // allow you to treat strings as streams, enabling you to perform formatted input and output operations on them
#include "opencv2/core.hpp"

namespace asio = boost::asio;
using tcp       = asio::ip::tcp;

/* ------------------------------------------------------------------
 *  Helper – trim whitespace from both ends of a string.
 * ------------------------------------------------------------------ */
static inline std::string trim(const std::string& s)
{
    auto start = std::find_if_not(s.begin(), s.end(),
                                  [](unsigned char c){ return std::isspace(c); });
    auto end   = std::find_if_not(s.rbegin(), s.rend(),
                                  [](unsigned char c){ return std::isspace(c); }).base();
    return (start < end) ? std::string(start, end) : std::string{};
}

/* ------------------------------------------------------------------
 *  Session – one client connection.
 *  Handles: header → body → response.
 * ------------------------------------------------------------------ */
class session : public std::enable_shared_from_this<session>
{
public:
    explicit session(tcp::socket socket)
        : socket_(std::move(socket))
    {}

    void start() { read_header(); }

private:
    /* --------------------------------------------------------------
     *  1. Read the HTTP header until the blank line (CRLFCRLF)
     * -------------------------------------------------------------- */
    void read_header()
    {
        auto self = shared_from_this();
        asio::async_read_until(socket_, buffer_, "\r\n\r\n",
            [this, self](boost::system::error_code ec, std::size_t bytes_transferred)
            {
                if (!ec)
                    parse_header(bytes_transferred);
                else
                    close_socket();
            });
    }

    /* --------------------------------------------------------------
     *  2. Parse request line + headers, locate Content‑Length.
     * -------------------------------------------------------------- */
    void parse_header(std::size_t /*bytes_transferred*/)
    {
        std::istream request_stream(&buffer_);
        std::string request_line;
        std::getline(request_stream, request_line);
        if (request_line.back() == '\r') request_line.pop_back();

        std::istringstream line_stream(request_line);
        line_stream >> method_ >> target_ >> http_version_;

        // Read header lines
        std::string header_line;
        while (std::getline(request_stream, header_line) && header_line != "\r")
        {
            if (header_line.back() == '\r') header_line.pop_back();
            auto colon = header_line.find(':');
            if (colon != std::string::npos)
            {
                std::string name  = trim(header_line.substr(0, colon));
                std::string value = trim(header_line.substr(colon + 1));
                // HTTP header names are case‑insensitive; store lower‑case key
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c){ return std::tolower(c); });
                headers_[name] = value;
            }
        }

        // If this is a POST we expect a body
        if (method_ == "POST")
        {
            auto it = headers_.find("content-length");
            if (it != headers_.end())
            {
                try
                {
                    content_length_ = static_cast<std::size_t>(std::stoul(it->second));
                }
                catch (const std::exception&)
                {
                    send_response("400 Bad Request\r\n\r\nInvalid Content‑Length");
                    return;
                }

                // Simple sanity check – you can adjust the limit
                if (content_length_ > max_body_size_)
                {
                    send_response("413 Payload Too Large\r\n\r\n");
                    return;
                }

                read_body();
                return;
            }
            else
            {
                // POST without a length – treat as bad request
                send_response("411 Length Required\r\n\r\n");
                return;
            }
        }

        // For non‑POST (e.g., health‑check GET) just reply
        send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nReady");
    }

    /* --------------------------------------------------------------
     *  3. Read the POST body (the JPEG image)
     * -------------------------------------------------------------- */
    void read_body()
    {
        // The streambuf may already contain part or all of the body
        std::size_t bytes_already = buffer_.size();
        if (bytes_already >= content_length_)
        {
            // Whole body already buffered – we can extract it directly
            extract_body();
            return;
        }

        // Need to read the remaining bytes
        auto self = shared_from_this();
        asio::async_read(socket_, buffer_,
                         asio::transfer_exactly(content_length_ - bytes_already),
                         [this, self](boost::system::error_code ec, std::size_t)
                         {
                             if (!ec)
                                 extract_body();
                             else
                                 close_socket();
                         });
    }

    /* --------------------------------------------------------------
     *  4. Pull the JPEG bytes out of the streambuf into a vector.
     *  This is the place to hand the data to OpenCV.
     * -------------------------------------------------------------- */
    void extract_body()
    {
        // Move the payload into a contiguous vector
        body_.resize(content_length_);
        std::istream body_stream(&buffer_);
        body_stream.read(body_.data(), static_cast<std::streamsize>(content_length_));

        // -----------------------------------------------------------------
        //  ***** INSERT YOUR OpenCV / Weapon‑Detection code here *****
        // -----------------------------------------------------------------
        // Example placeholder:
        // cv::Mat img = cv::imdecode(cv::Mat(body_), cv::IMREAD_COLOR);
        // bool threat = run_yolo(img);
        cv::Mat img = cv::imdecode(cv::Mat(body_), cv::IMREAD_COLOR);
        bool threat = run_yolo(img);
        // -----------------------------------------------------------------

        // For now just acknowledge receipt
        if (threat) {
        // Respond with a status that indicates a threat was found (e.g., 200 OK with a specific body)
        // Or, if this were an API designed to reject threats, a 403 Forbidden might be considered.
        // Sticking to 200 OK for now, but making the body clear.
        send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nTHREAT DETECTED!");
    } else {
        // Acknowledge receipt and successful processing
        send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nImage processed. No threat detected.");
    }
    }

    /* --------------------------------------------------------------
     *  5. Send a minimal HTTP response and close the socket.
     * -------------------------------------------------------------- */
    void send_response(const std::string& status_and_headers)
    {
        // Always terminate with an empty line (CRLF) to finish the header block
        std::string response = "HTTP/1.1 " + status_and_headers;
        if (response.back() != '\n')
            response += "\r\n";
        response += "Connection: close\r\n\r\n";

        auto self = shared_from_this();
        asio::async_write(socket_, asio::buffer(response),
                          [this, self](boost::system::error_code /*ec*/, std::size_t /*len*/)
                          {
                              close_socket();
                          });
    }

    /* --------------------------------------------------------------
     *  Close the socket – the session object will be destroyed
     *  automatically when the last shared_ptr goes out of scope.
     * -------------------------------------------------------------- */
    void close_socket()
    {
        boost::system::error_code ignored_ec;
        socket_.shutdown(tcp::socket::shutdown_both, ignored_ec);
        socket_.close(ignored_ec);
    }

    /* ----------------------------------------------------------------
     *  Member data
     * ---------------------------------------------------------------- */
    tcp::socket                       socket_;
    asio::streambuf                   buffer_;          // header + body (partial)
    std::string                       method_;
    std::string                       target_;
    std::string                       http_version_;
    std::unordered_map<std::string,std::string> headers_;
    std::size_t                       content_length_ = 0;
    const std::size_t                 max_body_size_ = 512 * 1024; // 512 KB limit
    std::vector<char>                 body_;            // final JPEG payload
};

/* ------------------------------------------------------------------
 *  Server – accepts new connections and spawns a session for each.
 * ------------------------------------------------------------------ */
class server
{
public:
    server(asio::io_context& io, unsigned short port)
        : acceptor_(io, tcp::endpoint(tcp::v4(), port))
        
    {
        start_accept();
    }

private:
    void start_accept()
    {
        acceptor_.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket)
            {
                if (!ec)
                {
                    // Hand the socket to its own session object.
                    std::make_shared<session>(std::move(socket))->start();
                }
                // Regardless of success or failure we keep listening.
                start_accept();
            });
    }

    tcp::acceptor acceptor_;
};

/* ------------------------------------------------------------------
 *  main() – entry point.
 * ------------------------------------------------------------------ */
int main()
{
    try
    {
        constexpr unsigned short listen_port = 8080;   // change if you like
        asio::io_context io;

        server s(io, listen_port);
        std::cout << "ESP32‑CAM HTTP server listening on port "
                  << listen_port << std::endl;

        io.run();   // blocks until all work is finished (i.e. never)
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
