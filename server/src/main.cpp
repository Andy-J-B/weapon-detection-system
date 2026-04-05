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

#include "secrets.h"
#include <algorithm> // Used for algorithms to help with helper functions? like find_if_not
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/bind/bind.hpp>
#include <cctype> // character logic
#include <cstdlib>
#include <iostream>
#include <memory> // Memory management, smart pointers
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream> //  treat strings as streams, enabling performing formatted input and output operations on them
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace asio = boost::asio;
using tcp = asio::ip::tcp;

static inline std::string trim(const std::string &s) {
  auto start = std::find_if_not(
      s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); });
  auto end = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) {
               return std::isspace(c);
             }).base();
  return (start < end) ? std::string(start, end) : std::string{};
}

void sendPhoneAlert(const std::string &message) {
  std::string token = PUSHOVER_TOKEN;
  std::string user = PUSHOVER_USER;

  std::string cmd = "curl -s \
        --form-string \"token=" +
                    token + "\" \
        --form-string \"user=" +
                    user + "\" \
        --form-string \"message=" +
                    message + "\" \
        https://api.pushover.net/1/messages.json > /dev/null";

  // Run it in the background
  std::system((cmd + " &").c_str());
}

bool detectWeapon(const cv::Mat &image, float confThresh = 0.35f,
                  float nmsThresh = 0.45f) {
  // Load the model
  static const std::string modelPath = MODEL_PATH;

  static cv::dnn::Net net;
  static bool initialized = false;

  if (!initialized) {
    try {
      net = cv::dnn::readNetFromONNX(modelPath);
      net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      std::cout << "✅ Loaded ONNX model from '" << modelPath << "'\n";
      initialized = true;
    } catch (const cv::Exception &e) {
      std::cerr << "❌ Failed to load ONNX model: " << e.what() << "\n";
      return false;
    }
  }

  if (image.empty()) {
    std::cerr << "⚠️  Received an empty image.\n";
    return false;
  }

  // preprocess the image
  const cv::Size INPUT_SIZE(640, 640);
  cv::Mat resized;
  float r = std::min(INPUT_SIZE.width / static_cast<float>(image.cols),
                     INPUT_SIZE.height / static_cast<float>(image.rows));
  int new_unpad_w = static_cast<int>(std::round(image.cols * r));
  int new_unpad_h = static_cast<int>(std::round(image.rows * r));
  cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

  int dw = INPUT_SIZE.width - new_unpad_w;
  int dh = INPUT_SIZE.height - new_unpad_h;
  dw /= 2;
  dh /= 2;
  cv::Mat padded;
  cv::copyMakeBorder(resized, padded, dh, dh, dw, dw, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));

  cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, INPUT_SIZE,
                                        cv::Scalar(), false, false);

  // forward pass into nn
  net.setInput(blob);
  cv::Mat out;
  try {
    out = net.forward();
  } catch (const cv::Exception &e) {
    std::cerr << "❌ Inference failed: " << e.what() << "\n";
    return false;
  }

  // get detections
  const int N = out.size[1];
  const int C = out.size[2]; // = 5 + nc  (nc == 1)
  CV_Assert(C >= 6);         // sanity check

  cv::Mat dets(N, C, CV_32F, out.ptr<float>());

  std::vector<int> keepIdx;
  std::vector<float> keepConf;
  std::vector<cv::Rect> keepBox;
  std::vector<int> keepCls;

  for (int i = 0; i < N; ++i) {
    const float *row = dets.ptr<float>(i);
    float objScore = row[4];
    if (objScore < confThresh)
      continue;

    int bestClass = -1;
    float bestClsScore = -1.f;
    for (int c = 5; c < C; ++c) {
      float clsScore = row[c];
      if (clsScore > bestClsScore) {
        bestClsScore = clsScore;
        bestClass = c - 5;
      }
    }
    if (bestClass < 0)
      continue;

    float confidence = objScore * bestClsScore;
    if (confidence < confThresh)
      continue;

    float cx = row[0];
    float cy = row[1];
    float w = row[2];
    float h = row[3];

    int x = static_cast<int>((cx - w / 2.0f) * INPUT_SIZE.width);
    int y = static_cast<int>((cy - h / 2.0f) * INPUT_SIZE.height);
    int bw = static_cast<int>(w * INPUT_SIZE.width);
    int bh = static_cast<int>(h * INPUT_SIZE.height);

    x -= dw;
    y -= dh;

    // Scale back to original image size (undo the letter‑box scaling)
    x = static_cast<int>(std::round(x / r));
    y = static_cast<int>(std::round(y / r));
    bw = static_cast<int>(std::round(bw / r));
    bh = static_cast<int>(std::round(bh / r));

    // Clip to image bounds
    x = std::max(0, std::min(x, image.cols - 1));
    y = std::max(0, std::min(y, image.rows - 1));
    bw = std::max(0, std::min(bw, image.cols - x));
    bh = std::max(0, std::min(bh, image.rows - y));

    keepIdx.push_back(i);
    keepConf.push_back(confidence);
    keepBox.emplace_back(x, y, bw, bh);
    keepCls.push_back(bestClass);
  }

  // -----------------------------------------------------------
  // 6️⃣ NMS – remove duplicate boxes (if any)
  // -----------------------------------------------------------
  std::vector<int> nmsIndices;
  cv::dnn::NMSBoxes(keepBox, keepConf, confThresh, nmsThresh, nmsIndices);

  for (int idx : nmsIndices) {
    int clsId = keepCls[idx];
    const char *clsName = (clsId == 0) ? "Weapon" : "Human";

    const cv::Rect &box = keepBox[idx];
    float conf = keepConf[idx];

    std::cout << "[NMS KEEP] class=" << clsName << " (id=" << clsId << ")"
              << ", confidence=" << conf << ", bbox=(" << box.x << "," << box.y
              << "," << box.width << "," << box.height << ")\n";
  }

  // -----------------------------------------------------------
  // 7️⃣ Final answer
  // -----------------------------------------------------------
  if (nmsIndices.empty()) {
    std::cout << "🟢 No weapon detected (background).\n";
    return false;
  }

  bool weaponFound = false;
  bool humanFound = false;
  float bestWeaponConf = 0.f;
  float bestHumanConf = 0.f;

  for (int idx : nmsIndices) {
    int clsId = keepCls[idx];
    float conf = keepConf[idx];

    if (clsId == 0) { // weapon
      weaponFound = true;
      bestWeaponConf = std::max(bestWeaponConf, conf);
    } else if (clsId == 1) { // human
      humanFound = true;
      bestHumanConf = std::max(bestHumanConf, conf);
    }
  }

  // ---- LOG THE SUMMARY ----
  if (weaponFound) {
    std::cout << "🔴 Weapon detected! best confidence = " << bestWeaponConf
              << "\n";
  }
  if (humanFound) {
    std::cout << "🟡 Human detected!  best confidence = " << bestHumanConf
              << "\n";
  }
  if (!weaponFound && !humanFound) {
    std::cout << "🟢 No weapon / human detected.\n";
  }

  // Return true if a weapon was present (your original contract)
  return weaponFound;
}

/*
 *  Session – one client connection.
 *  Handles: header → body → response.
 */
class session : public std::enable_shared_from_this<session> {
public:
  explicit session(tcp::socket socket, boost::asio::ssl::context &ctx)
      : stream_(std::move(socket), ctx) {}
  void start() {
    auto self = shared_from_this();
    stream_.async_handshake(
        asio::ssl::stream_base::server,
        [this, self](const boost::system::error_code &error) {
          handle_handshake(error);
        });
  }

  void handle_handshake(const boost::system::error_code &error) {
    if (!error) {
      read_header();
    } else {
      perror("Error while handling handle_handshake");
    }
  }

private:
  void read_header() {
    auto self = shared_from_this();
    asio::async_read_until(stream_, buffer_, "\r\n\r\n",
                           [this, self](boost::system::error_code ec,
                                        std::size_t bytes_transferred) {
                             if (!ec)
                               parse_header(bytes_transferred);
                             else
                               close_socket();
                           });
  }

  void parse_header(std::size_t /*bytes_transferred*/) {
    std::istream request_stream(&buffer_);
    std::string request_line;
    std::getline(request_stream, request_line);
    if (!request_line.empty() && request_line.back() == '\r')
      request_line.pop_back();

    std::istringstream line_stream(request_line);
    line_stream >> method_ >> target_ >> http_version_;

    // Read header lines
    std::string header_line;
    while (std::getline(request_stream, header_line) && header_line != "\r") {
      if (!header_line.empty() && header_line.back() == '\r')
        header_line.pop_back();
      auto colon = header_line.find(':');
      if (colon != std::string::npos) {
        std::string name = trim(header_line.substr(0, colon));
        std::string value = trim(header_line.substr(colon + 1));
        // HTTP header names are case‑insensitive; store lower‑case key
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        headers_[name] = value;
      }
    }

    // If this is a POST we expect a body
    if (method_ == "POST") {
      auto it = headers_.find("content-length");
      if (it != headers_.end()) {
        try {
          content_length_ = static_cast<std::size_t>(std::stoul(it->second));
        } catch (const std::exception &) {
          send_response("400 Bad Request\r\n\r\nInvalid Content‑Length");
          return;
        }

        // Simple sanity check – you can adjust the limit
        if (content_length_ > max_body_size_) {
          send_response("413 Payload Too Large\r\n\r\n");
          return;
        }

        read_body();
        return;
      } else {
        // POST without a length – treat as bad request
        send_response("411 Length Required\r\n\r\n");
        return;
      }
    }

    // For non‑POST (e.g., health‑check GET) just reply
    send_response("200 OK\r\nContent-Type: text/plain\r\n\r\nReady");
  }

  void read_body() {
    // The streambuf may already contain part or all of the body
    std::size_t bytes_already = buffer_.size();
    if (bytes_already >= content_length_) {
      // Whole body already buffered – we can extract it directly
      extract_body();
      return;
    }

    // Need to read the remaining bytes
    auto self = shared_from_this();
    asio::async_read(stream_, buffer_,
                     asio::transfer_exactly(content_length_ - bytes_already),
                     [this, self](boost::system::error_code ec, std::size_t) {
                       if (!ec)
                         extract_body();
                       else
                         close_socket();
                     });
  }

  void extract_body() {
    // Move the payload into a contiguous vector
    body_.resize(content_length_);
    std::istream body_stream(&buffer_);
    body_stream.read(body_.data(),
                     static_cast<std::streamsize>(content_length_));

    cv::Mat img = cv::imdecode(cv::Mat(body_), cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "❌ Failed to decode JPEG body." << std::endl;
      send_response("400 Bad Request\r\n\r\nInvalid Image Data");
      return;
    }

    bool weapon = false;

    try {
      weapon = detectWeapon(img);
    } catch (const cv::Exception &e) {
      // Handle potential OpenCV runtime errors during detection
      std::cerr << "OpenCV Error during run_yolo: " << e.what() << std::endl;
      send_response("500 Internal Server Error\r\nContent-Type: "
                    "text/plain\r\n\r\nImage processing error");
      return;
    } catch (const std::exception &e) {
      // Handle other potential errors
      std::cerr << "Detection Error: " << e.what() << std::endl;
      send_response("500 Internal Server Error\r\nContent-Type: "
                    "text/plain\r\n\r\nImage processing error");
      return;
    }

    if (weapon) {
      sendPhoneAlert("🚨 THREAT DETECTED: A weapon has been identified!");
    }

    // Build a JSON payload
    std::ostringstream payload;
    payload << "{ \"threat\" : " << (weapon ? "true" : "false") << " }";

    std::string body = payload.str();
    std::string status = "200 OK";
    std::string headers = "Content-Type: application/json\r\n"
                          "Content-Length: " +
                          std::to_string(body.size()) + "\r\n";

    send_response(status + "\r\n" + headers + "\r\n" + body);
  }

  void send_response(const std::string &status_and_headers) {
    // Always terminate with an empty line (CRLF) to finish the header block
    response_buffer_ = "HTTP/1.1 " + status_and_headers;
    if (response_buffer_.back() != '\n')
      response_buffer_ += "\r\n";
    response_buffer_ += "Connection: close\r\n\r\n";

    auto self = shared_from_this();
    asio::async_write(stream_, asio::buffer(response_buffer_),
                      [this, self](boost::system::error_code /*ec*/,
                                   std::size_t /*len*/) { close_socket(); });
  }
  void close_socket() {
    boost::system::error_code ec;
    stream_.shutdown(ec);

    if (ec) {
      if (ec == boost::asio::error::eof ||
          ec == boost::asio::ssl::error::stream_truncated) {
      } else {
        std::cerr << "SSL shutdown error: " << ec.message() << std::endl;
      }
    }

    boost::system::error_code ec_ignored;
    stream_.lowest_layer().close(ec_ignored);

    if (ec_ignored) {
      if (ec_ignored == boost::asio::error::eof ||
          ec_ignored == boost::asio::ssl::error::stream_truncated) {
      } else {
        std::cerr << "SSL shutdown error: " << ec_ignored.message()
                  << std::endl;
      }
    }
  }

  asio::ssl::stream<tcp::socket> stream_;
  asio::streambuf buffer_; // header + body (partial)
  std::string method_;
  std::string target_;
  std::string http_version_;
  std::unordered_map<std::string, std::string> headers_;
  std::size_t content_length_ = 0;
  const std::size_t max_body_size_ = 2 * 1024 * 1024;
  std::vector<char> body_; // final JPEG payload
  std::string response_buffer_;
};

/*
 *  Server – accepts new connections and spawns a session for each.
 */
class server {
public:
  server(asio::io_context &io, unsigned short port)
      : acceptor_(io, tcp::endpoint(tcp::v4(), port)),
        ctx(asio::ssl::context::sslv23)

  {

    ctx.set_options(asio::ssl::context::default_workarounds |
                    asio::ssl::context::no_sslv2 |
                    asio::ssl::context::no_sslv3);

    try {
      ctx.use_certificate_chain_file("../certs/server.crt");
      ctx.use_private_key_file("../certs/server.key", asio::ssl::context::pem);
    } catch (const std::exception &e) {
      throw std::runtime_error("Failed to load TLS certs.");
    }
    start_accept();
  }

private:
  void start_accept() {
    acceptor_.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
          if (!ec) {
            // Hand the socket to its own session object.
            std::make_shared<session>(std::move(socket), ctx)->start();
          }
          // Regardless of success or failure we keep listening.
          start_accept();
        });
  }

  tcp::acceptor acceptor_;
  asio::ssl::context ctx;
};

int main() {
  try {
    constexpr unsigned short listen_port = 8080; // change if you like
    asio::io_context io;

    server s(io, listen_port);
    std::cout << "ESP32‑CAM HTTP server listening on port " << listen_port
              << std::endl;

    io.run(); // blocks until all work is finished (i.e. never)
  } catch (const std::exception &ex) {
    std::cerr << "Fatal error: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
