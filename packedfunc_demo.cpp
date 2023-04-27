#include <tvm/runtime/packed_func.h>
/*
 * g++ -std=c++17 -O2 -fPIC -I/workdir/tvm/include -I/workdir/tvm/3rdparty/dmlc-core/include -I/workdir/tvm/3rdparty/dlpack/include -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> -o lib/packedfunc_demo   packedfunc_demo.cpp -ltvm_runtime -L/workdir/tvm/build -ldl -pthread
 * */

void MyAdd(tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
  // automatically convert arguments to desired type.
  int a = args[0];
  int b = args[1];
  // automatically assign value return to rv
  *rv = a + b;
}

void CallPacked() {
  tvm::runtime::PackedFunc myadd = tvm::runtime::PackedFunc(MyAdd);
  // get back 3
  int c = myadd(1, 2);
  printf("%d\n",c);
}

int main() {
  CallPacked();
  return 0;
}
