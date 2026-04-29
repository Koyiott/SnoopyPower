#pragma once
#include <assert.h>
#define Xil_AssertVoid(expr)         do { (void)0; } while (0)
#define Xil_AssertNonvoid(expr)      do { (void)0; } while (0)
/* If you prefer real checks, use:  assert(expr);  */

