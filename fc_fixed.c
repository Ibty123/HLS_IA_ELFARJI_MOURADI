/**
  ******************************************************************************
  * @file    fc_fixed.c
  * @brief   Fully connected layers (FC1, FC2) + Softmax for FIXED POINT LeNet
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#include "lenet_cnn_fixed_point.h"
#include <math.h>   // uniquement pour exp() dans softmax (autorisé CPU)


// ------------------------------
//  Helpers
// ------------------------------

static inline short relu_fixed(short x)
{
    return (x > 0) ? x : 0;
}


// ------------------------------
//  Fully Connected Layer FC1
// ------------------------------
//
// input   : [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]   (short)
// kernel  : [FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH] (short)
// bias    : [FC1_NBOUTPUT] (short)
// output  : [FC1_NBOUTPUT] (short)
//
// FLOAT equivalent:
//   y = ReLU( b + Σ input * weight )
//
// FIXED equivalent:
//   y_fp = ReLU( b_fp + Σ (in_fp * w_fp >> FIXED_POINT) )
//
void Fc1_40_400_fixed(
        short input [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short bias  [FC1_NBOUTPUT],
        short output[FC1_NBOUTPUT])
{
    unsigned short k, z, y, x;

    for (k = 0; k < FC1_NBOUTPUT; k++) {

        // accumulator must be larger than short → int recommended
        int acc = ((int)bias[k]) << FIXED_POINT;

        for (z = 0; z < POOL2_NBOUTPUT; z++) {
            for (y = 0; y < POOL2_HEIGHT; y++) {
                for (x = 0; x < POOL2_WIDTH; x++) {

                    acc += ( (int)input[z][y][x] * (int)kernel[k][z][y][x] );
                }
            }
        }

        // shift back to fixed-point range
        acc = acc >> FIXED_POINT;

        output[k] = relu_fixed((short)acc);
    }
}


// ------------------------------
//  Fully Connected Layer FC2
// ------------------------------
//
// input   : [400] short
// kernel  : [10][400] short
// bias    : [10] short
// output  : [10] short (logits in fixed point)
//
// Same formula as FC1.
//
void Fc2_400_10_fixed(
        short input [FC1_NBOUTPUT],
        short kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
        short bias  [FC2_NBOUTPUT],
        short output[FC2_NBOUTPUT])
{
    unsigned short k, i;

    for (k = 0; k < FC2_NBOUTPUT; k++) {

        int acc = ((int)bias[k]) << FIXED_POINT;

        for (i = 0; i < FC1_NBOUTPUT; i++) {
            acc += ( (int)input[i] * (int)kernel[k][i] );
        }

        acc = acc >> FIXED_POINT;

        output[k] = (short)acc;   // pas de ReLU ici (logits)
    }
}



// ------------------------------
//  Softmax FIXED POINT
//  (version du prof, améliorée + stabilisée)
// ------------------------------
//
// vector_in  : [10] short (logits)
// vector_out : [10] float (probabilities)
//
// Convertit logits FP16 → float,
// applique softmax stable : exp(x - max)
//
void Softmax_fixed(short vector_in[FC2_NBOUTPUT], float vector_out[FC2_NBOUTPUT])
{
    unsigned short k;
    float f[FC2_NBOUTPUT];

    // Convertir en float réel
    for (k = 0; k < FC2_NBOUTPUT; k++) {
        f[k] = (float)vector_in[k] / (float)(1 << FIXED_POINT);
    }

    // Softmax stable
    float maxval = f[0];
    for (k = 1; k < FC2_NBOUTPUT; k++) {
        if (f[k] > maxval) maxval = f[k];
    }

    float sum = 0.0f;
    for (k = 0; k < FC2_NBOUTPUT; k++) {
        f[k] = expf(f[k] - maxval);
        sum += f[k];
    }

    if (sum < 1e-12f) {
        float inv = 1.0f / FC2_NBOUTPUT;
        for (k = 0; k < FC2_NBOUTPUT; k++)
            vector_out[k] = inv;
        return;
    }

    for (k = 0; k < FC2_NBOUTPUT; k++) {
        vector_out[k] = f[k] / sum;
    }
}
