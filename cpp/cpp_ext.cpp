
#include "full.h"
#include "amen_solve.h"
#include "compression.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_full", &full, "TT to full");
  m.def("amen_solve", &amen_solve, "TT to full");
  m.def("round_this", &round_this, "Implace rounding");
}







