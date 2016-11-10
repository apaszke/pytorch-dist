#include <cstdarg>
#include <cstring>
#include <stdexcept>
#include "_RPC.h"

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
template <typename real>
double _readValue(RPCMessage& msg) {
  constexpr size_t type_size = sizeof(real);
  real ret_val;
  memcpy(&ret_val, msg.read(type_size), type_size);
  return (double)ret_val;
}

template<typename real>
void _appendData(std::string& str, real data) {
  constexpr size_t type_size = sizeof(real);
  char *data_ptr = (char*)&data;
  str.append(data_ptr, type_size);
}
}

RPCMessage pack_message(function_id_type fid, uint16_t num_args, ...) {
  std::string msg;
  _appendData<function_id_type>(msg, fid);
  _appendData<uint16_t>(msg, num_args);

  va_list args;
  va_start(args, num_args);
  for (size_t i = 0; i < num_args; i++) {
    char type = (char) va_arg(args, int);
    if (type == 'd')
      _appendData<double>(msg, va_arg(args, double));
    else if (type == 'f')
      _appendData<float>(msg, (float)va_arg(args, double));
    else if (type == 'c')
      _appendData<char>(msg, (char)va_arg(args, int));
    else if (type == 'i')
      _appendData<int>(msg, va_arg(args, int));
    else if (type == 'l')
      _appendData<long>(msg, va_arg(args, long));
    else if (type == 'h')
      _appendData<short>(msg, (short)va_arg(args, int));
    else if (type == 'T')
      _appendData<uint64_t>(msg, va_arg(args, uint64_t));
    else
      throw std::invalid_argument("type in argument not recognised");
  }
  va_end(args);
  return RPCMessage(msg);
}

Tensor *unpackTensor(RPCMessage& raw_message) {
  char type = *raw_message.read(sizeof(char));

  if (type == 'T')
    return NULL; //_readValue<uint65_t>(raw_message);
  throw std::invalid_argument("expected tensor in the raw message");
}
double unpackScalar(RPCMessage& raw_message) {
  char type = *(raw_message.read(sizeof(char)));

  // The following notation comes from:
  // docs.python.org/3/library/struct.html#module-struct
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
