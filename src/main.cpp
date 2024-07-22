#include "models/test.cpp"

int main(int argc, char *argv[]) {
  bool debug = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--debug") == 0 || strcmp(argv[i], "-d") == 0) {
      debug = true;
      break;
    }
  }

  if (debug) {
    std::cout << "Welcome to CCNN!" << std::endl;
    run_model_debug();
    std::cout << "Done." << std::endl;
  } else {
    run_model();
  }

  return EXIT_SUCCESS;
}
