#include <string.h>
#include "models/lenet/lenet.cpp"

int main(int argc, char *argv[]) {
  bool debug = false;
  int batch_size = 1;
  std::vector<int> profile_indices;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
      debug = true;
      break;
    } else if (strncmp(argv[i], "--batchsize=", 12) == 0) {
      batch_size = std::atoi(argv[i] + 12);
      if (batch_size <= 0) {
        std::cerr << "Invalid batch size. It must be a positive integer." << std::endl;
        return EXIT_FAILURE;
      }
    } else if (strncmp(argv[i], "--profile=", 10) == 0) {
      char* profile_values = argv[i] + 10;
      char* token = strtok(profile_values, ",");
      while (token != nullptr) {
        int index = std::atoi(token);
        if (index >= 0) {
          profile_indices.push_back(index);
        }
        token = strtok(nullptr, ",");
      }
    }
  }

  if (debug) {
    std::cout << "Welcome to CCNN!" << std::endl;
    run_model_debug(batch_size);
    std::cout << "Done." << std::endl;
  } else {
    run_model(batch_size, profile_indices);
  }

  return EXIT_SUCCESS;
}
