/**
  ******************************************************************************
  * @file    conv_fixed.c
  * @brief   Convolution layers for LeNet (FIXED POINT version)
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#include "lenet_cnn_fixed_point.h"


/* ReLU FIXED POINT */
static inline short relu_fixed(short x)
{
    return (x > 0) ? x : 0;
}


/* ============================================================================
 *  CONV1  (28×28×1  →  24×24×20)
 * ============================================================================
 *
 *  FLOAT: sum += input * kernel + bias
 *  FIXED: acc += in_fp * k_fp
 *         acc += b_fp << FIXED_POINT
 *         out_fp = acc >> FIXED_POINT
 */
void Conv1_28x28x1_5x5x20_1_0_fixed(
        short input[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],                       // IN
        short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],       // IN
        short bias[CONV1_NBOUTPUT],                                          // IN
        short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH])             // OUT
{
    unsigned short k, z, y, x, ky, kx;

    for (k = 0; k < CONV1_NBOUTPUT; k++) {
        for (y = 0; y < CONV1_HEIGHT; y++) {
            for (x = 0; x < CONV1_WIDTH; x++) {

                /* Accumulate in 32-bit to avoid overflow */
                int acc = ((int)bias[k]) << FIXED_POINT;

                for (z = 0; z < IMG_DEPTH; z++) {
                    for (ky = 0; ky < CONV1_DIM; ky++) {
                        for (kx = 0; kx < CONV1_DIM; kx++) {

                            unsigned short in_y = (unsigned short)(y + ky);
                            unsigned short in_x = (unsigned short)(x + kx);

                            acc += ( (int)input[z][in_y][in_x] *
                                     (int)kernel[k][z][ky][kx] );
                        }
                    }
                }

                /* Return to FIXED range */
                acc >>= FIXED_POINT;

                output[k][y][x] = relu_fixed((short)acc);
            }
        }
    }
}



/* ============================================================================
 *  CONV2  (12×12×20  →  8×8×40)
 * ============================================================================
 *
 *  Same exact pattern, only dimensions change.
 */
void Conv2_12x12x20_5x5x40_1_0_fixed(
        short input[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],              // IN
        short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],  // IN
        short bias[CONV2_NBOUTPUT],                                          // IN
        short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH])             // OUT
{
    unsigned short k, z, y, x, ky, kx;

    for (k = 0; k < CONV2_NBOUTPUT; k++) {
        for (y = 0; y < CONV2_HEIGHT; y++) {
            for (x = 0; x < CONV2_WIDTH; x++) {

                int acc = ((int)bias[k]) << FIXED_POINT;

                for (z = 0; z < POOL1_NBOUTPUT; z++) {
                    for (ky = 0; ky < CONV2_DIM; ky++) {
                        for (kx = 0; kx < CONV2_DIM; kx++) {

                            unsigned short in_y = (unsigned short)(y + ky);
                            unsigned short in_x = (unsigned short)(x + kx);

                            acc += ( (int)input[z][in_y][in_x] *
                                     (int)kernel[k][z][ky][kx] );
                        }
                    }
                }

                acc >>= FIXED_POINT;

                output[k][y][x] = relu_fixed((short)acc);
            }
        }
    }
}
