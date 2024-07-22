#include "models/test.cpp"

int main(int argc, char *argv[]) {
  bool debug = false;
  int batch_size = 1;

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
    }
  }

  if (debug) {
    std::cout << "Welcome to CCNN!" << std::endl;
    run_model_debug(batch_size);
    std::cout << "Done." << std::endl;
  } else {
    run_model(batch_size);
  }

  return EXIT_SUCCESS;
}
