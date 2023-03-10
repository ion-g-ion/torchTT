
#include "full.h"
#include "amen_solve.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_full", &full, "TT to full");
  m.def("amen_solve", &amen_solve, "TT to full");
}







