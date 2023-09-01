#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <stdio.h>
class PythonCppTest : public torch::CustomClassHolder {
private:
public:
  explicit PythonCppTest() {}

  torch::Tensor test_copy(torch::Tensor &keys, int64_t n) {
    const uint64_t *key_ptr = static_cast<const uint64_t *>(keys.data_ptr());
    torch::Tensor result = torch::empty({n});
    float *value_ptr = static_cast<float *>(result.data_ptr());
    for(int i = 0; i < n; i++){
      value_ptr[i] = float(key_ptr[i]);
      printf("%ld ", key_ptr[i]);
    }
    printf("\n");
    return result;
  }
};

TORCH_LIBRARY(python_cpp_test, m) {
  m.class_<PythonCppTest>("PythonCppTest")
      .def(torch::init())
      .def("test_copy", &PythonCppTest::test_copy)
      ;
}