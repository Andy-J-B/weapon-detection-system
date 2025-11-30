#include <ctime>
#include <iostream>
#include <string>
#include <boost/asio.hpp>
#include <boost/bind/bind.hpp>
#include <boost/enable_shared_from_this.hpp>

using boost::asio::ip::tcp;
using namespace std;

// Helper function to generate the current daytime string
std::string make_daytime_string()
{
  using namespace std; // For time_t, time and ctime;
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

  // Initiates the asynchronous write operation to send data to the client
  void start()
  {
    // The data to be sent is stored in the class member message_
    // so it remains valid until the asynchronous operation is complete.
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
  void handle_write(const boost::system::error_code& /*error*/,
      size_t /*bytes_transferred*/)
  {
    // No action is needed here in this simple example, 
    // as the connection is finished after sending the data.
  }

  tcp::socket socket_;
  std::string message_;
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
    start_accept();
  }

private:
  // Creates a socket and initiates an asynchronous accept operation
  void start_accept()
  {
    // Create a new connection object for the incoming client
    pointer new_connection =
      tcp_connection::create(io_context_);

    // Initiate the asynchronous accept operation
    acceptor_.async_accept(new_connection->socket(),
        boost::bind(&tcp_server::handle_accept, this, new_connection,
          boost::asio::placeholders::error));
  }

  // Handler for the asynchronous accept operation
  void handle_accept(pointer new_connection,
      const boost::system::error_code& error)
  {
    // If no error occurred during accept
    if (!error)
    {
      // Start the connection (i.e., send the daytime string)
      new_connection->start();
    }

    // Initiate the next accept operation to wait for another client
    start_accept();
  }

  boost::asio::io_context& io_context_;
  tcp::acceptor acceptor_;
};

// The main function sets up and runs the server
int main()
{
  try
  {

    // The io_context object provides I/O services
    boost::asio::io_context io_context;

    // Create the server object, which starts listening for connections
    tcp_server server(io_context);

    // Run the io_context object. This blocks and performs the asynchronous operations
    // on the server's behalf until there is no more work to do (which is never,
    // in this server example, unless it's stopped).
    io_context.run();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}