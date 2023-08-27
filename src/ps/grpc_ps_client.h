#pragma once

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "base/array.h"
#include "flatc.h"
#include "parameters.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"

#include "folly/Portability.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/init/Init.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using xmhps::CommandRequest;
using xmhps::CommandResponse;
using xmhps::GetParameterRequest;
using xmhps::GetParameterResponse;
using xmhps::PSCommand;
using xmhps::PutParameterRequest;
using xmhps::PutParameterResponse;

using base::ConstArray;

static const int MAX_PARAMETER_BATCH = 2000;

class ParameterClient {
public:
  explicit ParameterClient(const std::string &host, int port, int shard);
  ~ParameterClient() {}

  // this interface assume all keys with the same embedding dimension
  bool GetParameter(ConstArray<uint64_t> &keys, float *values, int64_t model_name,
                    bool perf = true);

  inline int shard() const { return shard_; }

  bool ClearPS();

  bool LoadFakeData(int64_t data);

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path);

  bool PutParameter(const std::vector<uint64_t> &keys,
                    const std::vector<std::vector<float>> &values);

protected:
  bool Initialize() { return true; }
  std::string host_;
  int port_;
  int shard_;
  int nr_clients_;
  std::vector<float> cache_;
  std::vector<int32_t> offset_;
  std::vector<int> get_param_key_sizes_;
  std::vector<Status> get_param_status_;
  std::vector<GetParameterRequest> get_param_requests_;
  std::vector<GetParameterResponse> get_param_responses_;
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<GetParameterResponse>>>
      get_param_resonse_readers_;
  std::vector<Status> put_param_status_;
  std::vector<PutParameterRequest> put_param_requests_;
  std::vector<PutParameterResponse> put_param_responses_;
  std::vector<
      std::unique_ptr<grpc::ClientAsyncResponseReader<PutParameterResponse>>>
      put_param_resonse_readers_;
  std::shared_ptr<Channel> channel_;
  std::vector<std::unique_ptr<xmhps::ParameterService::Stub>> stubs_;
  grpc::CompletionQueue cq;
  grpc::CompletionQueue put_cq;
};
