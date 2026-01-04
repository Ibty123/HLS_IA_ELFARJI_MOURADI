/**
  ******************************************************************************
  * @file    pool_fixed.c
  * @brief   Max-pooling layers for LeNet (FIXED POINT version)
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#include "lenet_cnn_fixed_point.h"


/* ============================================================================
 *  POOL1  (24×24×20  →  12×12×20)
 * ============================================================================
 *
 *  MaxPool2x2, stride 2, aucun changement d’échelle (toujours en fixed-point)
 * ============================================================================
 */
void Pool1_24x24x20_2x2x20_2_0_fixed(
        short input [CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],    // IN
        short output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH])    // OUT
{
    unsigned short z, y, x;

    for (z = 0; z < POOL1_NBOUTPUT; z++) {
        for (y = 0; y < POOL1_HEIGHT; y++) {
            for (x = 0; x < POOL1_WIDTH; x++) {

                unsigned short in_y = (unsigned short)(y * POOL1_STRIDE);
                unsigned short in_x = (unsigned short)(x * POOL1_STRIDE);

                /* Max-pooling 2×2 */
                short max_val = input[z][in_y][in_x];

                short v1 = input[z][in_y][in_x + 1];
                if (v1 > max_val) max_val = v1;

                short v2 = input[z][in_y + 1][in_x];
                if (v2 > max_val) max_val = v2;

                short v3 = input[z][in_y + 1][in_x + 1];
                if (v3 > max_val) max_val = v3;

                output[z][y][x] = max_val;
            }
        }
    }
}


/* ============================================================================
 *  POOL2  (8×8×40  →  4×4×40)
 * ============================================================================
 *
 *  Identique à POOL1, tailles différentes.
 * ============================================================================
 */
void Pool2_8x8x40_2x2x40_2_0_fixed(
        short input [CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],    // IN
        short output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH])    // OUT
{
    unsigned short z, y, x;

    for (z = 0; z < POOL2_NBOUTPUT; z++) {
        for (y = 0; y < POOL2_HEIGHT; y++) {
            for (x = 0; x < POOL2_WIDTH; x++) {

                unsigned short in_y = (unsigned short)(y * POOL2_STRIDE);
                unsigned short in_x = (unsigned short)(x * POOL2_STRIDE);

                /* Max-pooling 2×2 */
                short max_val = input[z][in_y][in_x];

                short v1 = input[z][in_y][in_x + 1];
                if (v1 > max_val) max_val = v1;

                short v2 = input[z][in_y + 1][in_x];
                if (v2 > max_val) max_val = v2;

                short v3 = input[z][in_y + 1][in_x + 1];
                if (v3 > max_val) max_val = v3;

                output[z][y][x] = max_val;
            }
        }
    }
}
