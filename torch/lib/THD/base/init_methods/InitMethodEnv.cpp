#include "InitMethod.hpp"
#include "InitMethodUtils.hpp"

namespace thd {
namespace init {

namespace {

constexpr char RANK_ENV[] = "RANK";
constexpr char WORLD_SIZE_ENV[] = "WORLD_SIZE";
constexpr char MASTER_PORT_ENV[] = "MASTER_PORT";
constexpr char MASTER_ADDR_ENV[] = "MASTER_ADDR";

const char* must_getenv(const char* env) {
  const char* value = std::getenv(env);
  if (value == nullptr) {
    throw std::logic_error(std::string("") + "failed to read the " + env +
        " environmental variable; maybe you forgot to set it?");
  }
  return value;
}

std::tuple<port_type, rank_type> load_master_env() {
  auto port = convertToPort(std::stoul(must_getenv(MASTER_PORT_ENV)));

  rank_type world_size = std::stoul(must_getenv(WORLD_SIZE_ENV));
  if (world_size == 0)
    throw std::domain_error(std::string(WORLD_SIZE_ENV) + " env variable cannot be 0");

  return std::make_tuple(port, world_size);
}


std::tuple<std::string, port_type> load_worker_env() {
  std::string str_port = must_getenv(MASTER_PORT_ENV);
  auto port = convertToPort(std::stoul(str_port));
  return std::make_tuple(must_getenv(MASTER_ADDR_ENV), port);
}

rank_type getRank(int rank) {
  const char *env_rank_str = std::getenv(RANK_ENV);
  int env_rank = rank;
  if (env_rank_str != nullptr)
    env_rank = std::stol(env_rank_str);
  if (rank != -1 && env_rank != rank)
    throw std::runtime_error("rank specified both as an environmental variable "
      "and to the initializer");

  return convertToRank(env_rank);
}

rank_type getWorldSize(int world_size) {
  const char *env_world_size_str = std::getenv(WORLD_SIZE_ENV);
  int env_world_size = world_size;
  if (env_world_size_str != nullptr)
    env_world_size = std::stol(env_world_size_str);
  if (world_size != -1 && env_world_size != world_size)
    throw std::runtime_error("world_size specified both as an environmental variable "
      "and to the initializer");

  return convertToRank(env_world_size);
}

} // anonymous namespace

InitMethod::Config initEnv(int world_size, std::string group_name, int rank) {
  InitMethod::Config config;

  config.rank = getRank(rank);
  config.world_size = getWorldSize(world_size);

  if (group_name != "") {
    throw std::runtime_error("group_name is not supported in env:// init method");
  }

  if (config.rank == 0) {
    std::tie(config.master.listen_port, std::ignore) = load_master_env();
    std::tie(config.master.listen_socket, std::ignore) = listen(config.master.listen_port);
    config.public_address = discoverWorkers(config.master.listen_socket,
                                            config.world_size);
  } else {
    std::tie(config.worker.master_addr, config.worker.master_port) = load_worker_env();
    std::tie(std::ignore, config.public_address) =
      discoverMaster({config.worker.master_addr}, config.worker.master_port);
  }
  return config;
}

} // namespace init
} // namespace thd
