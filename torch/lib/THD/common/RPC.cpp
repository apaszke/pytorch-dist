#include <cstdarg>
#include <cstring>
#include <stdexcept>
#include "_RPC.h"
#include "../master/THDTensor.h"

namespace thd {

RPCMessage::RPCMessage(std::string& str) : _msg(str) {};

const char *RPCMessage::data() {
  return _msg.data() + _offset;
}

const char *RPCMessage::read(size_t num_bytes) {
  if (_offset + num_bytes >= _msg.length())
    throw std::out_of_range("invalid access: out of bounds");
  const char *ret_val = _msg.data() + _offset;
  _offset += num_bytes;
  return ret_val;
}

namespace {
template <typename real, typename return_type = double>
return_type _readValue(RPCMessage& msg) {
  constexpr size_t type_size = sizeof(real);
  real ret_val;
  memcpy(&ret_val, msg.read(type_size), type_size);
  return (return_type)ret_val;
}

template<typename real>
void _appendData(std::string& str, real data) {
  constexpr size_t type_size = sizeof(real);
  char *data_ptr = (char*)&data;
  str.append(data_ptr, type_size);
}

// The following notation comes from:
// docs.python.org/3/library/struct.html#module-struct
template<typename T>
struct rpc_traits {};

template<>
struct rpc_traits<double> {
  static constexpr char scalar_char = 'd';
};

template<>
struct rpc_traits<char> {
  static constexpr char scalar_char = 'c';
};

template<>
struct rpc_traits<float> {
  static constexpr char scalar_char = 'f';
};

template<>
struct rpc_traits<int> {
  static constexpr char scalar_char = 'i';
};

template<>
struct rpc_traits<long> {
  static constexpr char scalar_char = 'l';
};

template<>
struct rpc_traits<short> {
  static constexpr char scalar_char = 'h';
};

template <typename T, typename ...Args>
void packIntoString(std::string& str, const T& arg, const Args&... args) {
  if (std::is_same<T, const THDTensor&>::value) {
    _appendData<char>(str, 'T');
    _appendData<unsigned long long>(str, arg.tensor_id);
  } else {
    _appendData<char>(str, rpc_traits<T>::scalar_char);
    _appendData<unsigned long long>(str, arg.tensor_id);
  }
  packIntoString(str, args...);
}

void packIntoString(RPCMessage& message) {}
}

template <typename ...Args>
RPCMessage packMessage(function_id_type fid, uint16_t num_args,
    const Args&... args) {
  std::string msg;
  _appendData<function_id_type>(msg, fid);
  _appendData<uint16_t>(msg, num_args);
  packIntoString(msg, args...);
  return RPCMessage(msg);
}

uint16_t unpackFunctionId(RPCMessage& raw_message) {
  return _readValue<uint16_t, uint16_t>(raw_message);
}

Tensor *unpackTensor(RPCMessage& raw_message) {
  char type = *raw_message.read(sizeof(char));

  if (type == 'T')
    return NULL; //_readValue<uint65_t>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}
double unpackScalar(RPCMessage& raw_message) {
  char type = *(raw_message.read(sizeof(char)));

  if (type == 'd')
    return _readValue<double>(raw_message);
  else if (type == 'f')
    return _readValue<float>(raw_message);
  else if (type == 'c')
    return _readValue<char>(raw_message);
  else if (type == 'i')
    return _readValue<int>(raw_message);
  else if (type == 'l')
    return _readValue<long>(raw_message);
  else if (type == 'h')
    return _readValue<short>(raw_message);
  throw std::invalid_argument("Wrong real type in the raw message");
}

} // thd namespace
