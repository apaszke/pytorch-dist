#include <iostream>
#include <cassert>
#include <typeinfo>

#include "../_THD.h"

using namespace std;
using namespace thd;

int main() {
  RPCMessage msg = packMessage(1, 3, 1.0f, 100l, -12);
  return 0;
}
