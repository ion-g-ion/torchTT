#pragma once

#include <torch/extension.h>
#include <vector>
#include <array>
#include <iostream>
#include <chrono>
#include "BLAS.h"

// torch::NoGradGuard no_grad;

#define NO_PREC 0
#define C_PREC 1
#define R_PREC 2



