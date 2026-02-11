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
 ********************************************************************/

#include <boost/asio.hpp>
#include <iostream>
#include <memory> // Memory management, smart pointers
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm> // Used for algorithms to help with helper functions? like find_if_not
#include <cctype> // character logic
#include <sstream> //  treat strings as streams, enabling performing formatted input and output operations on them
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>  
#include <opencv2/highgui.hpp>  

namespace asio = boost::asio;
using tcp       = asio::ip::tcp;

/* Helper – trim whitespace from both ends of a string. */
static inline std::string trim(const std::string& s)
{
    auto start = std::find_if_not(s.begin(), s.end(),
                                  [](unsigned char c){ return std::isspace(c); });
    auto end   = std::find_if_not(s.rbegin(), s.rend(),
                                  [](unsigned char c){ return std::isspace(c); }).base();
    return (start < end) ? std::string(start, end) : std::string{};
}

/* AI - Runs  */
bool run_yolo(cv::Mat& image) {

    std::cout << "Starting yolo inference\n";
    static const std::string modelPath = "/Users/Andy_1/dev/code/programs/GitHub/weapon-detection-system/server/best.onnx";
    static const float CONF_THRESH = 0.35f;             // detection confidence (objectness × class‑score)
    static const float NMS_THRESH  = 0.45f;       
    static const cv::Size INPUT_SIZE(640, 640); 

    // 0 : knife, 1 : handgun
    static const std::vector<int> weaponClassIds = {0, 1};

    static cv::dnn::Net net;
    static bool netInitialized = false;


    if (!netInitialized)
    {
        try
        {
            net = cv::dnn::readNetFromONNX(modelPath);
            std::cout << "✅ Loaded YOLO model from '" << modelPath << "' ("
                  << net.getLayerNames().size() << " layers)\n";
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            netInitialized = true;
        }
        catch (const cv::Exception& e)
        {
            std::cerr << "⚠️  Failed to load YOLO model from '" << modelPath
                      << "': " << e.what() << std::endl;
            return false;   // without a model we cannot detect anything
        }
    }

    if (image.empty())
    {
        std::cerr << "⚠️  run_yolo received an empty image." << std::endl;
        return false;
    }

    // Pre-process
    cv::Mat blob = cv::dnn::blobFromImage(
        image,                // source image (BGR)
        1.0 / 255.0,         // scale factor – YOLO expects values in [0,1]
        INPUT_SIZE,         // resize to the size used during training
        cv::Scalar(),       // mean subtraction (none)
        true,               // swap RB (OpenCV loads BGR, YOLO was trained on RGB)
        false               // crop – we want a straight resize with letter‑box padding handled by the network
    );

    // Forward Pass
    net.setInput(blob);


    cv::Mat preds;
    try
    {
        preds = net.forward();   // single output
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "⚠️  Inference error: " << e.what() << std::endl;
        return false;
    }

    // The output may be a 3‑D Mat; flatten it to a 2‑D view for easier indexing.
    // Expected shape: (1, N, C)  →  we drop the leading 1.
    const int numDetections = preds.size[1];
    const int numChannels   = preds.size[2];   // typically 5 + num_classes

    // Safety check – if the tensor shape is not what we expect, abort.
    if (numDetections == 0 || numChannels <= 5)
    {
        std::cerr << "⚠️  Unexpected network output shape." << std::endl;
        return false;
    }

    // Create a view that treats the data as a matrix of [N x C] floats.
    cv::Mat detections(numDetections, numChannels, CV_32F, preds.ptr<float>());

    for (int i = 0; i < numDetections; ++i)
    {
        const float objScore = detections.at<float>(i, 4);
        if (objScore < CONF_THRESH)               // filter out low‑objectness boxes early
            continue;

        // Class scores start at column 5.
        cv::Mat scores = detections.row(i).colRange(5, numChannels);
        cv::Point maxClassIdx;
        double   maxClassScore = 0.0;
        cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &maxClassIdx);

        const float confidence = static_cast<float>(objScore * maxClassScore);
        if (confidence < CONF_THRESH)
            continue;   // discard low‑confidence detections

        int classId = maxClassIdx.x;   // zero‑based index into the model’s class list
        
        // Decide whether there's a weapon detected
        bool isWeapon = false;
        if (!weaponClassIds.empty())
        {
            // If you supplied an explicit whitelist, check membership.
            isWeapon = std::find(weaponClassIds.begin(),
                                 weaponClassIds.end(),
                                 classId) != weaponClassIds.end();
        }
        else
        {
            // No whitelist supplied – treat *any* detection as a threat.
            isWeapon = true;
        }
        std::cout << "Deciding whether this is a weapon\n";

        if (isWeapon)
        {
            std::cout << "Detected a weapon!"<< std::endl;
            return true;
            
        }
    }
    // No Detection
    std::cout << "Didn't detect a weapon."<< std::endl;
    return false;
}

/* 
 *  Session – one client connection.
 *  Handles: header → body → response.
 */
class session : public std::enable_shared_from_this<session>
{
public:
    explicit session(tcp::socket socket)
        : socket_(std::move(socket))
    {}

    void start() { read_header(); }

private:
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

    void extract_body()
    {
        // Move the payload into a contiguous vector
        body_.resize(content_length_);
        std::istream body_stream(&buffer_);
        body_stream.read(body_.data(), static_cast<std::streamsize>(content_length_));

        cv::Mat img = cv::imdecode(cv::Mat(body_), cv::IMREAD_COLOR);
        bool threat = false;
        try {
            threat = run_yolo(img);
        } catch (const cv::Exception& e) {
            // Handle potential OpenCV runtime errors during detection
            std::cerr << "OpenCV Error during run_yolo: " << e.what() << std::endl;
            send_response("500 Internal Server Error\r\nContent-Type: text/plain\r\n\r\nImage processing error");
            return;
        } catch (const std::exception& e) {
            // Handle other potential errors
            std::cerr << "Detection Error: " << e.what() << std::endl;
            send_response("500 Internal Server Error\r\nContent-Type: text/plain\r\n\r\nImage processing error");
            return;
        }
        

        // For now just acknowledge receipt
        if (threat) {
        send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nTHREAT DETECTED!");
    } else {
        // Acknowledge receipt and successful processing
        send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nImage processed. No threat detected.");
    }
    }


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
    void close_socket()
    {
        boost::system::error_code ignored_ec;
        socket_.shutdown(tcp::socket::shutdown_both, ignored_ec);
        socket_.close(ignored_ec);
    }

    tcp::socket                       socket_;
    asio::streambuf                   buffer_;          // header + body (partial)
    std::string                       method_;
    std::string                       target_;
    std::string                       http_version_;
    std::unordered_map<std::string,std::string> headers_;
    std::size_t                       content_length_ = 0;
    const std::size_t                 max_body_size_ = 2 * 1024 * 1024;
    std::vector<char>                 body_;            // final JPEG payload
};

/* 
 *  Server – accepts new connections and spawns a session for each.
 */
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
