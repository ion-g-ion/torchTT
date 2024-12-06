#include "full.h"
#include "amen_solve.h"
#include "compression.h"
#include "dmrg_mv.h"

/// Functions from cpp to import in python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tt_full", &full, "TT to full");
  m.def("amen_solve", &amen_solve, "AMEn solve");
  m.def("round_this", &round_this, "Implace rounding");
  m.def("dmrg_mv", &dmrg_mv, "DMRG matrix vector product");
}







