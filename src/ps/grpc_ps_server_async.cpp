#include <folly/executors/CPUThreadPoolExecutor.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <gflags/gflags.h>

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "base/array.h"
#include "base/base.h"
#include "base/timer.h"
#include "cache_ps_impl.h"
#include "flatc.h"
#include "parameters.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"
#include "provider.h"
#include "atomic_queue/atomic_queue.h"

#include "base_ps_server.h"

using grpc::Server;
using grpc::ServerAsyncResponseWriter;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using xmhps::CommandRequest;
using xmhps::CommandResponse;
using xmhps::GetParameterRequest;
using xmhps::GetParameterResponse;
using xmhps::PSCommand;
using xmhps::PutParameterRequest;
using xmhps::PutParameterResponse;

DEFINE_int32(thread_num, 16, "Thread num");
DEFINE_string(port, "15000", "Server Port");
DEFINE_int64(dict_capability, 12543770LL, "Dict capability");
DEFINE_int32(value_size, 128, "Value size");
DEFINE_int64(memory_pool_size, 1*1024*1024*1024LL, "Memory pool size");
DEFINE_int32(max_batch_size, 10000, "Max batch size");

const int GET_PATA_REQ_NUM = 10240;
const int PUT_PATA_REQ_NUM = 6400;
const int COMMAND_REQ_NUM = 100;
const int RESERVE_NUM = 512;
const int TOTAL_NUM = GET_PATA_REQ_NUM + PUT_PATA_REQ_NUM + COMMAND_REQ_NUM + RESERVE_NUM;

class DispatchParam {
public:
  enum Status {
    RECEIVED,
    PROCESSED,
  };
  enum RequestType{
    GET,
    PUT,
    COMMAND,
  };
  Status status;
  RequestType request_type;
  ::grpc::ServerContext ctx;
  xmh::Timer timer;
  DispatchParam(RequestType request_type, const char *name) : status(PROCESSED), request_type(request_type), timer(name) {}
};

class PutRequestParam : public DispatchParam{
public:
  xmhps::PutParameterRequest request;
  ServerAsyncResponseWriter<PutParameterResponse> responder;
  PutParameterResponse reply;
  PutRequestParam() : DispatchParam(PUT, "PUT timer"), responder(&ctx) {}
};

class GetRequestParam : public DispatchParam {
public:
  xmhps::GetParameterRequest request;
  ServerAsyncResponseWriter<GetParameterResponse> responder;
  GetParameterResponse reply;
  GetRequestParam() : DispatchParam(GET, "GET timer"), responder(&ctx) {}
};

class CommandRequestParam : public DispatchParam {
public:
  xmhps::CommandRequest request;
  ServerAsyncResponseWriter<CommandResponse> responder;
  CommandResponse reply;
  CommandRequestParam() : DispatchParam(COMMAND, "COMMAND timer"), responder(&ctx) {}
};

class ParameterServiceImpl final : public xmhps::ParameterService::AsyncService {
  
private:
  CachePS *cache_ps_;
  int thread_num_;
  std::vector<std::thread> thread_pool;
  std::atomic<uint64_t> get_key_cnt;
public:
  atomic_queue::AtomicQueue<DispatchParam *, TOTAL_NUM> task_queue;

  void Monitor(){
    while(true){
      std::this_thread::sleep_for(std::chrono::seconds(1));
      double total_bytes = get_key_cnt.exchange(0);
      total_bytes *= 128;
      total_bytes /= 1024 * 1024;
      std::cout << "\rThroughput " << total_bytes << " MB/s" << std::flush;
    }
  }


  ParameterServiceImpl(CachePS *cache_ps, int thread_num) : cache_ps_(cache_ps), thread_num_(thread_num) {
    for(int i = 0; i < thread_num_; i++){
      thread_pool.push_back(std::thread(&ParameterServiceImpl::Process, this, i));
    }
    thread_pool.push_back(std::thread(&ParameterServiceImpl::Monitor, this));
    get_key_cnt = 0;
  }


  void DoGetParameter(GetRequestParam *under_process, int tid){
    GetParameterRequest *request = &under_process->request;
    GetParameterResponse *reply = &under_process->reply;
    base::ConstArray<uint64_t> keys_array(request->keys());
    bool isPerf = request->has_perf() && request->perf();
    if (isPerf) {
      xmh::PerfCounter::Record("PS Get Keys", keys_array.Size());
    }
    xmh::Timer timer_ps_get_req("PS GetParameter Req");
    ParameterCompressor compressor;
    std::vector<std::string> blocks;
    // FB_LOG_EVERY_MS(INFO, 1000)
    //     << "[PS] Getting " << keys_array.Size() << " keys";

    std::vector<ParameterPack> packs;
    cache_ps_->GetParameterRun2Completion(keys_array, packs, tid);
    for (auto each : packs) {
      compressor.AddItem(each, &blocks);
    }
    compressor.ToBlock(&blocks);
    CHECK_EQ(blocks.size(), 1);
    reply->mutable_parameter_value()->swap(blocks[0]);
    if (isPerf) {
        timer_ps_get_req.end();
    } else {
      timer_ps_get_req.destroy();
    }
    under_process->timer.end();
    under_process->responder.Finish(under_process->reply, Status::OK, under_process);
    get_key_cnt.fetch_add(keys_array.Size());
  }

  void DoPutParameter(PutRequestParam *under_process, int tid){
    PutParameterRequest *request = &under_process->request;
    const ParameterCompressReader *reader =
        reinterpret_cast<const ParameterCompressReader *>(
            request->parameter_value().data());
    cache_ps_->PutParameter(reader, tid);
    under_process->timer.end();
    under_process->responder.Finish(under_process->reply, Status::OK, under_process);
  }

  void DoCommand(CommandRequestParam *under_process, int tid){
    CommandRequest *request = &under_process->request;
    xmh::Timer timer_ps_get_req("PS Command Req");
    if (request->command() == PSCommand::CLEAR_PS) {
      LOG(WARNING) << "[PS Command] Clear All";
      cache_ps_->Clear();
    } else if (request->command() == PSCommand::RELOAD_PS) {
      LOG(WARNING) << "[PS Command] Reload PS";
      CHECK_NE(request->arg1().size(), 0);
      CHECK_NE(request->arg2().size(), 0);
      CHECK_EQ(request->arg1().size(), 1);
      LOG(WARNING) << "model_config_path = " << request->arg1()[0];
      for (int i = 0; i < request->arg2().size(); i++) {
        LOG(WARNING) << fmt::format("emb_file {}: {}", i, request->arg2()[i]);
      }
      std::vector<std::string> arg1;
      for (auto &each : request->arg1()) {
        arg1.push_back(each);
      }
      std::vector<std::string> arg2;
      for (auto &each : request->arg2()) {
        arg2.push_back(each);
      }
      cache_ps_->Initialize(arg1, arg2);
    } else if(request->command() == PSCommand::LOAD_FAKE_DATA) {
      uint64_t fake_data_size = *((uint64_t *)request->arg1(0).data());
      cache_ps_->LoadFakeData(fake_data_size);
    } else {
      LOG(FATAL) << "invalid command";
    }
    timer_ps_get_req.end();
    under_process->timer.end();
    under_process->responder.Finish(under_process->reply, Status::OK, under_process);
  }

  void worker_func(int work_id){
    DispatchParam *under_process;
    while(true){
      under_process = this->task_queue.pop();
      if(under_process->request_type == DispatchParam::RequestType::GET){
        this->DoGetParameter(static_cast<GetRequestParam *>(under_process), work_id);
        continue;
      }
      if(under_process->request_type == DispatchParam::RequestType::PUT){
        this->DoPutParameter(static_cast<PutRequestParam *>(under_process), work_id);
        continue;
      }
      if(under_process->request_type == DispatchParam::RequestType::COMMAND){
        this->DoCommand(static_cast<CommandRequestParam *>(under_process), work_id);
        continue;
      }
    }
  }

  void Process(int work_id){
    base::bind_core(work_id + 32);
    worker_func(work_id);
  }

};

namespace recstore {
class GRPCParameterServer : public BaseParameterServer {
 public:
  GRPCParameterServer() = default;

  void Run() {
    std::string server_address("0.0.0.0:");
    server_address += FLAGS_port;
    auto cache_ps = std::make_unique<CachePS>(FLAGS_dict_capability, FLAGS_value_size, FLAGS_memory_pool_size, FLAGS_thread_num, FLAGS_max_batch_size);  // 1GB dict
    ParameterServiceImpl service(cache_ps.get(), FLAGS_thread_num);
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case it corresponds to an *synchronous* service.
    builder.RegisterService(&service);
    std::unique_ptr<grpc::ServerCompletionQueue> cq = builder.AddCompletionQueue();
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    for(int i = 0; i < GET_PATA_REQ_NUM; i++){
      GetRequestParam* get_params = new GetRequestParam();
      service.RequestGetParameter(&get_params->ctx, &get_params->request, &get_params->responder, cq.get(), cq.get(), get_params);
    }
    for(int i = 0; i < PUT_PATA_REQ_NUM; i++){
      PutRequestParam* put_params = new PutRequestParam();
      service.RequestPutParameter(&put_params->ctx, &put_params->request, &put_params->responder, cq.get(), cq.get(), put_params);
    }
    for(int i = 0; i < COMMAND_REQ_NUM; i++){
      CommandRequestParam* command_params = new CommandRequestParam();
      service.RequestCommand(&command_params->ctx, &command_params->request, &command_params->responder, cq.get(), cq.get(), command_params);
    }
    while(true){
      void *tag;
      bool ok;
      GPR_ASSERT(cq->Next(&tag, &ok));
      GPR_ASSERT(ok);
      DispatchParam *param = static_cast<DispatchParam *>(tag);
      if(param->status == DispatchParam::Status::PROCESSED){
        param->status = DispatchParam::Status::RECEIVED;
        param->timer.start();
        service.task_queue.push(param);
      } else {
        param->status = DispatchParam::PROCESSED;
        if(param->request_type == DispatchParam::RequestType::GET){
          GetRequestParam *param = static_cast<GetRequestParam *>(tag);
          delete param;
          param = new GetRequestParam();
          service.RequestGetParameter(&param->ctx, &param->request, &param->responder, cq.get(), cq.get(), param);
          continue;
        }
        if(param->request_type == DispatchParam::RequestType::PUT){
          PutRequestParam *param = static_cast<PutRequestParam *>(tag);
          delete param;
          param = new PutRequestParam();
          service.RequestPutParameter(&param->ctx, &param->request, &param->responder, cq.get(), cq.get(), param);
          continue;
        }
        if(param->request_type == DispatchParam::RequestType::COMMAND){
          CommandRequestParam *param = static_cast<CommandRequestParam *>(tag);
          delete param;
          param = new CommandRequestParam();
          service.RequestCommand(&param->ctx, &param->request, &param->responder, cq.get(), cq.get(), param);
          continue;
        }
      }
    }
  }
};
}  // namespace recstore

int main(int argc, char **argv) {
  std::ignore = folly::Init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);
  nlohmann::json ex = nlohmann::json::parse(R"(
  {
    "pi": 3.141,
    "happy": true
  }
  )");

  recstore::GRPCParameterServer ps;
  ps.Init(ex);
  ps.Run();
  gflags::ShutDownCommandLineFlags();
  return 0;
}