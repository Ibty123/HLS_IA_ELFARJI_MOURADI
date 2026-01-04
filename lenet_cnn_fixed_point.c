/**
  ******************************************************************************
  * @file    lenet_cnn_fixed_point.c
  * @brief   Full LeNet forward pass in FIXED POINT
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "lenet_cnn_fixed_point.h"
#include "Weights.h"

const int labels_legend[10] = {0,1,2,3,4,5,6,7,8,9};

/**************************************
 *  BUFFERS GLOBAUX FIXED POINT
 **************************************/

short INPUT_FP[IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH];
/*
short CONV1_KERNEL_FP[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM];
short CONV1_BIAS_FP[CONV1_NBOUTPUT];

short CONV2_KERNEL_FP[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM];
short CONV2_BIAS_FP[CONV2_NBOUTPUT];

short FC1_KERNEL_FP[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
short FC1_BIAS_FP[FC1_NBOUTPUT];

short FC2_KERNEL_FP[FC2_NBOUTPUT][FC1_NBOUTPUT];
short FC2_BIAS_FP[FC2_NBOUTPUT];
*/
short FC2_OUTPUT_FP[FC2_NBOUTPUT];

float SOFTMAX_OUTPUT[FC2_NBOUTPUT];


/**************************************
 *  PROTOTYPES DES FONCTIONS UTILS FIXED
 *  (implémentées dans utils_fixed.c)
 **************************************/
#ifndef __SYNTHESIS__
void ReadConv1Weights_float(char *filename, char *dataset, float W[CONV1_DIM][CONV1_DIM][IMG_DEPTH][CONV1_NBOUTPUT]);
void ReadConv1Bias_float   (char *filename, char *dataset, float *b);

void ReadConv2Weights_float(char *filename, char *dataset, float W[CONV2_DIM][CONV2_DIM][CONV1_NBOUTPUT][CONV2_NBOUTPUT]);
void ReadConv2Bias_float   (char *filename, char *dataset, float *b);

void ReadFc1Weights_float  (char *filename, char *dataset, float W[POOL2_HEIGHT*POOL2_WIDTH*POOL2_NBOUTPUT][FC1_NBOUTPUT]);
void ReadFc1Bias_float     (char *filename, char *dataset, float *b);

void ReadFc2Weights_float  (char *filename, char *dataset, float W[FC1_NBOUTPUT][FC2_NBOUTPUT]);
void ReadFc2Bias_float     (char *filename, char *dataset, float *b);

void ConvertWeightsToFixed();
#endif
void NormalizeImg_fixed(unsigned char *input, short *output, short width, short height);
void ReadPgmFile(char *filename, unsigned char *pix);


/**************************************
 *  TOP LEVEL FIXED POINT
 **************************************/
//#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
//#pragma HLS INTERFACE s_axilite port=input bundle=CTRL
//#pragma HLS INTERFACE s_axilite port=out bundle=CTRL

void lenet_cnn_fixed(
        short  input   [IMG_DEPTH][IMG_HEIGHT][IMG_WIDTH],
        short  conv1_k [CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
        short  conv1_b [CONV1_NBOUTPUT],
        short  conv2_k [CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
        short  conv2_b [CONV2_NBOUTPUT],
        short  fc1_k   [FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short  fc1_b   [FC1_NBOUTPUT],
        short  fc2_k   [FC2_NBOUTPUT][FC1_NBOUTPUT],
        short  fc2_b   [FC2_NBOUTPUT],
        short  out     [FC2_NBOUTPUT])
{
    short conv1_out[CONV1_NBOUTPUT][CONV1_HEIGHT][CONV1_WIDTH];
    short pool1_out[POOL1_NBOUTPUT][POOL1_HEIGHT][POOL1_WIDTH];
    short conv2_out[CONV2_NBOUTPUT][CONV2_HEIGHT][CONV2_WIDTH];
    short pool2_out[POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH];
    short fc1_out[FC1_NBOUTPUT];

    Conv1_28x28x1_5x5x20_1_0_fixed(input, conv1_k, conv1_b, conv1_out);
    Pool1_24x24x20_2x2x20_2_0_fixed(conv1_out, pool1_out);
    Conv2_12x12x20_5x5x40_1_0_fixed(pool1_out, conv2_k, conv2_b, conv2_out);
    Pool2_8x8x40_2x2x40_2_0_fixed(conv2_out, pool2_out);
    Fc1_40_400_fixed(pool2_out, fc1_k, fc1_b, fc1_out);
    Fc2_400_10_fixed(fc1_out, fc2_k, fc2_b, out);
}


/**************************************
 *  PROGRAMME PRINCIPAL
 **************************************/
#ifndef __SYNTHESIS__
int main() 
{
    /*char *hdf5_file = "lenet_weights.hdf5";

    // noms des datasets dans le .hdf5
    char *conv1_W = "conv2d_1/conv2d_1/kernel:0";
    char *conv1_b = "conv2d_1/conv2d_1/bias:0";

    char *conv2_W = "conv2d_2/conv2d_2/kernel:0";
    char *conv2_b = "conv2d_2/conv2d_2/bias:0";

    char *fc1_W   = "dense_1/dense_1/kernel:0";
    char *fc1_b   = "dense_1/dense_1/bias:0";

    char *fc2_W   = "dense_2/dense_2/kernel:0";
    char *fc2_b   = "dense_2/dense_2/bias:0";
*/
    char *label_file_name = "mnist/t10k-labels-idx1-ubyte";

    FILE *label_file = fopen(label_file_name, "r");
    if (!label_file) {
        printf("ERROR: Could not open labels file\n");
        return -1;
    }

    // discard header (8 bytes)
    unsigned char tmp;
    for (int i = 0; i < 8; i++) fscanf(label_file, "%c", &tmp);


    /********************************************
     * 1. LECTURE DES POIDS (FLOAT)
     ********************************************/
    /*float buffer_conv1_W[CONV1_DIM][CONV1_DIM][IMG_DEPTH][CONV1_NBOUTPUT];
    float buffer_conv2_W[CONV2_DIM][CONV2_DIM][CONV1_NBOUTPUT][CONV2_NBOUTPUT];
    float buffer_fc1_W[POOL2_HEIGHT*POOL2_WIDTH*POOL2_NBOUTPUT][FC1_NBOUTPUT];
    float buffer_fc2_W[FC1_NBOUTPUT][FC2_NBOUTPUT];

    float buffer_conv1_B[CONV1_NBOUTPUT];
    float buffer_conv2_B[CONV2_NBOUTPUT];
    float buffer_fc1_B[FC1_NBOUTPUT];
    float buffer_fc2_B[FC2_NBOUTPUT];

    ReadConv1Weights_float(hdf5_file, conv1_W, buffer_conv1_W);
    ReadConv1Bias_float   (hdf5_file, conv1_b, buffer_conv1_B);

    ReadConv2Weights_float(hdf5_file, conv2_W, buffer_conv2_W);
    ReadConv2Bias_float   (hdf5_file, conv2_b, buffer_conv2_B);

    ReadFc1Weights_float  (hdf5_file, fc1_W, buffer_fc1_W);
    ReadFc1Bias_float     (hdf5_file, fc1_b, buffer_fc1_B);

    ReadFc2Weights_float  (hdf5_file, fc2_W, buffer_fc2_W);
    ReadFc2Bias_float     (hdf5_file, fc2_b, buffer_fc2_B);
*/

    /********************************************
     * 2. CONVERSION DES POIDS EN FIXED POINT
     ********************************************/
    /*ConvertWeightsToFixed(buffer_conv1_W, buffer_conv1_B,
                          buffer_conv2_W, buffer_conv2_B,
                          buffer_fc1_W,  buffer_fc1_B,
                          buffer_fc2_W,  buffer_fc2_B,
                          CONV1_KERNEL_FP, CONV1_BIAS_FP,
                          CONV2_KERNEL_FP, CONV2_BIAS_FP,
                          FC1_KERNEL_FP,  FC1_BIAS_FP,
                          FC2_KERNEL_FP,  FC2_BIAS_FP);
*/

    /********************************************
     * 3. BOUCLE DE TEST SUR MNIST
     ********************************************/
    unsigned int error = 0;
    unsigned int n = 0;

   while(1) {

        unsigned char label_raw;
        if (fscanf(label_file, "%c", &label_raw) == EOF) break;

        int label = label_raw;

        // Construire nom du fichier
        char img_file[128];
        sprintf(img_file, "mnist/t10k-images-idx3-ubyte[%05d].pgm", n);

        // Lire image
        unsigned char img_px[IMG_WIDTH * IMG_HEIGHT];
        ReadPgmFile(img_file, img_px);

        // Normalisation -> fixed-point
        NormalizeImg_fixed(img_px, (short*)INPUT_FP, IMG_WIDTH, IMG_HEIGHT);
    

        // Exécution du CNN
        lenet_cnn_fixed(INPUT_FP,
                        CONV1_KERNEL, CONV1_BIAS,
                        CONV2_KERNEL, CONV2_BIAS,
                        FC1_KERNEL,  FC1_BIAS,
                        FC2_KERNEL,  FC2_BIAS,
                        FC2_OUTPUT_FP);


        // Softmax
        Softmax_fixed(FC2_OUTPUT_FP, SOFTMAX_OUTPUT);

        // Trouver prediction
        float max = SOFTMAX_OUTPUT[0];
        int pred = 0;

        for (int k = 1; k < FC2_NBOUTPUT; k++) {
            if (SOFTMAX_OUTPUT[k] > max) {
                max = SOFTMAX_OUTPUT[k];
                pred = k;
            }
        }
        if (n == 0) {
                printf("\nSoftmax output:\n");
                for (int k = 0; k < FC2_NBOUTPUT; k++) {
                    printf("%.2f%% ", SOFTMAX_OUTPUT[k] * 100.0f);
                }
                printf("\n");

                printf("Predicted: %d    Actual: %d\n", pred, label);
            }

            if (pred != label)
                error++;

            n++;
    }

    fclose(label_file);

    printf("\nTEST FINISHED\n");
    printf("Errors: %d / %d\n", error, n);
    printf("Success rate: %.2f%%\n", 100.0f * (1.0f - (float)error/n));

    return 0;
}
#endif
