/**
  ******************************************************************************
  * @file    lenet_cnn_fixed_point.h
  * @brief   Header for LeNet FIXED POINT implementation
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#ifndef LENET_CNN_FIXED_POINT_H
#define LENET_CNN_FIXED_POINT_H


/**************************************
 *  FIXED POINT FORMAT
 **************************************/
#define FIXED_POINT 8   // Q8 (excellent compromis précision / performance)

/*
   Une valeur réelle r devient :
       fixed = (short)( r * (1 << FIXED_POINT) )

   Tous les poids, biais, entrées et activations sont en `short`.
   Accumulateurs dans le .c : int  (sécurité overflow).
*/


/**************************************
 *  DIMENSIONS (identiques version FLOAT)
 **************************************/

#define IMG_WIDTH   28
#define IMG_HEIGHT  28
#define IMG_DEPTH   1

/* ---------- CONV1 ---------- */
#define CONV1_DIM       5
#define CONV1_NBOUTPUT  20
#define CONV1_STRIDE    1
#define CONV1_PAD       0
#define CONV1_WIDTH     ( ((IMG_WIDTH  - CONV1_DIM + 2*CONV1_PAD)  / CONV1_STRIDE) + 1 )
#define CONV1_HEIGHT    ( ((IMG_HEIGHT - CONV1_DIM + 2*CONV1_PAD) / CONV1_STRIDE) + 1 )

/* ---------- POOL1 ---------- */
#define POOL1_DIM       2
#define POOL1_NBOUTPUT  CONV1_NBOUTPUT
#define POOL1_STRIDE    2
#define POOL1_PAD       0
#define POOL1_WIDTH     ( ((CONV1_WIDTH  - POOL1_DIM + 2*POOL1_PAD) / POOL1_STRIDE) + 1 )
#define POOL1_HEIGHT    ( ((CONV1_HEIGHT - POOL1_DIM + 2*POOL1_PAD) / POOL1_STRIDE) + 1 )

/* ---------- CONV2 ---------- */
#define CONV2_DIM       5
#define CONV2_NBOUTPUT  40
#define CONV2_STRIDE    1
#define CONV2_PAD       0
#define CONV2_WIDTH     ( ((POOL1_WIDTH  - CONV2_DIM + 2*CONV2_PAD) / CONV2_STRIDE) + 1 )
#define CONV2_HEIGHT    ( ((POOL1_HEIGHT - CONV2_DIM + 2*CONV2_PAD) / CONV2_STRIDE) + 1 )

/* ---------- POOL2 ---------- */
#define POOL2_DIM       2
#define POOL2_NBOUTPUT  CONV2_NBOUTPUT
#define POOL2_STRIDE    2
#define POOL2_PAD       0
#define POOL2_WIDTH     ( ((CONV2_WIDTH  - POOL2_DIM + 2*POOL2_PAD) / POOL2_STRIDE) + 1 )
#define POOL2_HEIGHT    ( ((CONV2_HEIGHT - POOL2_DIM + 2*POOL2_PAD) / POOL2_STRIDE) + 1 )

/* ---------- FC layers ---------- */
#define FC1_NBOUTPUT    400
#define FC2_NBOUTPUT    10


/**************************************
 *  PROTOTYPES DES FONCTIONS FIXED POINT
 **************************************/

/* ---------- Convolution layers ---------- */
void Conv1_28x28x1_5x5x20_1_0_fixed(
        short input [IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
        short kernel[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
        short bias  [CONV1_NBOUTPUT],
        short output[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH]);

void Conv2_12x12x20_5x5x40_1_0_fixed(
        short input [POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH],
        short kernel[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
        short bias  [CONV2_NBOUTPUT],
        short output[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH]);


/* ---------- Pooling layers ---------- */
void Pool1_24x24x20_2x2x20_2_0_fixed(
        short input [CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH],
        short output[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH]);

void Pool2_8x8x40_2x2x40_2_0_fixed(
        short input [CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH],
        short output[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH]);


/* ---------- Fully Connected layers ---------- */
void Fc1_40_400_fixed(
        short input [POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short kernel[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short bias  [FC1_NBOUTPUT],
        short output[FC1_NBOUTPUT]);

void Fc2_400_10_fixed(
        short input [FC1_NBOUTPUT],
        short kernel[FC2_NBOUTPUT][FC1_NBOUTPUT],
        short bias  [FC2_NBOUTPUT],
        short output[FC2_NBOUTPUT]);


/* ---------- Softmax fixed point ---------- */
void Softmax_fixed(short vector_in[FC2_NBOUTPUT],
                   float vector_out[FC2_NBOUTPUT]);



/**************************************
 *  BUFFERS GLOBAUX (optionnel)
 *  Vous pouvez les déclarer ici OU dans le .c selon votre organisation
 **************************************/

/*
short INPUT_FP[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
short CONV1_KERNEL_FP[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
short CONV1_BIAS_FP[CONV1_NBOUTPUT];

short CONV2_KERNEL_FP[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
short CONV2_BIAS_FP[CONV2_NBOUTPUT];

short FC1_KERNEL_FP[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
short FC1_BIAS_FP[FC1_NBOUTPUT];

short FC2_KERNEL_FP[FC2_NBOUTPUT][FC1_NBOUTPUT];
short FC2_BIAS_FP[FC2_NBOUTPUT];

short FC2_OUTPUT_FP[FC2_NBOUTPUT];
float SOFTMAX_OUTPUT[FC2_NBOUTPUT];
*/

#endif
