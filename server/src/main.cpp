#include <ctime>
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/function.hpp> // REQUIRED for the completion callback

using boost::asio::ip::tcp;
using namespace std;

// Helper function to generate the current daytime string
std::string make_daytime_string()
{
  using namespace std;
  time_t now = time(0);
  return ctime(&now);
}

// Forward declaration of tcp_connection
class tcp_connection;

// Define a shared pointer type for tcp_connection
typedef boost::shared_ptr<tcp_connection> pointer;

// The tcp_connection class handles a single client connection
class tcp_connection
  : public boost::enable_shared_from_this<tcp_connection>
{
public:
  // Factory method to create an instance
  static pointer create(boost::asio::io_context& io_context)
  {
    return pointer(new tcp_connection(io_context));
  }

  // Returns a reference to the socket
  tcp::socket& socket()
  {
    return socket_;
  }

  // Initiates the asynchronous write operation to send data to the client.
  // The 'on_complete' callback is executed when the write is finished.
  void start(boost::function<void()> on_complete)
  {
    on_complete_ = on_complete;
    
    // In a real application, you would read the request headers here.
    // We keep the daytime example simple for demonstration.
    message_ = make_daytime_string();

    // Initiate the asynchronous write operation
    boost::asio::async_write(socket_, boost::asio::buffer(message_),
        boost::bind(&tcp_connection::handle_write, shared_from_this(),
          boost::asio::placeholders::error,
          boost::asio::placeholders::bytes_transferred));
  }

private:
  // Private constructor to enforce use of the static create() method
  tcp_connection(boost::asio::io_context& io_context)
    : socket_(io_context)
  {
  }

  // Handler for the asynchronous write operation
  void handle_write(const boost::system::error_code& error,
      size_t /*bytes_transferred*/)
  {
    // 1. Connection is finished. If no error, execute the callback
    if (!error && on_complete_)
    {
      // This calls tcp_server::start_accept() to begin listening again.
      on_complete_();
    }
    
    // The connection object will be destroyed when shared_from_this() is released.
  }

  tcp::socket socket_;
  std::string message_;

  // Buffer for incoming request data (must be large enough for image + headers)
  // 524288 bytes (512 KB) for SXGA (1280x1024) JPEG at Quality 10.
  boost::array<char, 524288> buffer_; 

  // Function object to hold the cleanup callback provided by the server
  boost::function<void()> on_complete_;
};

// The tcp_server class handles accepting new client connections
class tcp_server
{
public:
  // Constructor initializes the acceptor to listen on TCP port 80 (HTTP)
  tcp_server(boost::asio::io_context& io_context)
    : io_context_(io_context),
      acceptor_(io_context, tcp::endpoint(tcp::v4(), 80))
  {
    // Start accepting the FIRST connection
    start_accept();
  }

private:
  // Creates a socket and initiates an asynchronous accept operation
  void start_accept()
  {
    // Print status to indicate server is available
    std::cout << "Server now listening for a single connection..." << std::endl;

    // Create a new connection object for the incoming client
    pointer new_connection =
      tcp_connection::create(io_context_);

    // Initiate the asynchronous accept operation.
    // The server STOPS here until a client connects.
    acceptor_.async_accept(new_connection->socket(),
        boost::bind(&tcp_server::handle_accept, this, new_connection,
          boost::asio::placeholders::error));
  }

  // Handler for the asynchronous accept operation
  void handle_accept(pointer new_connection,
      const boost::system::error_code& error)
  {
    if (!error)
    {
      std::cout << "Client connected. Handling request and suspending listening." << std::endl;

      // CRITICAL CHANGE: We pass a callback to the connection.
      // This callback calls tcp_server::start_accept() AFTER the connection
      // is fully handled and closed.
      new_connection->start(
        boost::bind(&tcp_server::start_accept, this)
      );
      
      // NOTICE: We DO NOT call start_accept() here. 
      // This prevents the server from listening for a second client.
    }
    else
    {
      // If the accept fails (e.g., connection reset), we should still
      // restart the listening process for the next attempt.
      std::cerr << "Accept failed: " << error.message() << std::endl;
      start_accept();
    }
  }

  boost::asio::io_context& io_context_;
  tcp::acceptor acceptor_;
};

// The main function sets up and runs the server
int main()
{
  try
  {
    boost::asio::io_context io_context;
    tcp_server server(io_context);

    std::cout << "HTTP Server started on port 80 (TCP v4)." << std::endl;
    std::cout << "Waiting for first connection..." << std::endl;

    // Run the io_context object.
    io_context.run();
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  return 0;
}