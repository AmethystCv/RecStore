#include "grpc_ps_client.h"
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

class PythonParameterClient : public torch::CustomClassHolder, public ParameterClient {
private:
  int emb_dim_;
public:
  explicit PythonParameterClient(const std::string &host, int64_t port, int64_t shard, int64_t emb_dim)
      : ParameterClient(host, port, shard), emb_dim_(emb_dim) {}

  torch::Tensor GetParameter(torch::Tensor &keys, bool perf = true) {
    const uint64_t *key_ptr = static_cast<const uint64_t *>(keys.data_ptr());
    torch::Tensor result = torch::empty({keys.size(0), emb_dim_});
    float *value_ptr = static_cast<float *>(result.data_ptr());
    ConstArray<uint64_t> keys_array(key_ptr, keys.size(0));
    ParameterClient::GetParameter(keys_array, value_ptr, perf);
    return result;
  }

  bool PutParameter(
    torch::Tensor &keys,
    torch::Tensor &values){

    auto keys_accessor = keys.accessor<int64_t, 1>();
    auto values_accessor = values.accessor<float, 2>();

    put_param_status_.clear();
    put_param_requests_.clear();
    put_param_responses_.clear();
    put_param_resonse_readers_.clear();

    int request_num =
      (keys.size(0) + MAX_PARAMETER_BATCH - 1) / MAX_PARAMETER_BATCH;
    put_param_status_.resize(request_num);
    put_param_requests_.resize(request_num);
    put_param_responses_.resize(request_num);

    for (int start = 0, index = 0; start < keys.size(0);
        start += MAX_PARAMETER_BATCH, ++index) {
      int key_size = std::min((int)(keys.size(0) - start), MAX_PARAMETER_BATCH);

      auto &status = put_param_status_[index];
      auto &request = put_param_requests_[index];
      auto &response = put_param_responses_[index];
      ParameterCompressor compressor;
      std::vector<std::string> blocks;
      for (int i = start; i < start + key_size; i++) {
        ParameterPack parameter_pack;
        parameter_pack.key = keys_accessor[i];
        parameter_pack.dim = values_accessor[i].size(0);
        parameter_pack.emb_data = static_cast<float*>(values_accessor[i].data());
        compressor.AddItem(parameter_pack, &blocks);
      }
      compressor.ToBlock(&blocks);
      CHECK_EQ(blocks.size(), 1);
      request.mutable_parameter_value()->swap(blocks[0]);
      grpc::ClientContext context;
      std::unique_ptr<grpc::ClientAsyncResponseReader<PutParameterResponse>> rpc = stubs_[0]->AsyncPutParameter(&context, request, &cq);
      rpc->Finish(&response, &status, reinterpret_cast<void*>(index));
    }

    int cnt = 0;
    while(cnt != request_num){
      void *got_tag;
      bool ok = false;
      cq.Next(&got_tag, &ok);
      if(!ok){
        LOG(ERROR) << "error";
      }
      cnt++;
    }
    return true;
  }

  bool LoadFakeData(int64_t data_size){
    return ParameterClient::LoadFakeData(data_size);
  }
  
  bool ClearPS(){
    return ParameterClient::ClearPS();
  }
};

TORCH_LIBRARY(grpc_ps_client_python, m) {
  m.class_<PythonParameterClient>("PythonParameterClient")
      .def(torch::init<const std::string &, int64_t, int64_t, int64_t>())
      .def("GetParameter", &PythonParameterClient::GetParameter)
      .def("PutParameter", &PythonParameterClient::PutParameter)
      .def("LoadFakeData", &PythonParameterClient::LoadFakeData)
      .def("ClearPS", &PythonParameterClient::ClearPS)
      ;
}