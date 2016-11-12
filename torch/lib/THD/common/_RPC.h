#pragma once

#include <cstdint>
#include <string>
#include "_Tensor.h"

using function_id_type = uint16_t;

namespace thd {

class RPCMessage {
public:
  RPCMessage();
  RPCMessage(std::string &str);
  const char *data();
  const char *read(size_t num_bytes);
private:
  std::string _msg;
  size_t _offset;
};

// The type defines the length of a variable
template <typename ...Args>
RPCMessage packMessage(function_id_type fid, uint16_t num_args, const Args&... args);
Tensor *unpackTensor(RPCMessage& raw_message);
double unpackScalar(RPCMessage& raw_message);

}
