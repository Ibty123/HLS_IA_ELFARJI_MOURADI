/**
  ******************************************************************************
  * @file    utils_fixed.c
  * @brief   Utilities for FIXED POINT LeNet (weights conversion, image
  *          normalization, HDF5 reading of float weights)
  * @note    Designed for Vivado HLS synthesis
  ******************************************************************************
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lenet_cnn_fixed_point.h"


/****************************************************
 * 1) LECTURE DES POIDS FLOAT DEPUIS HDF5
 ****************************************************/


/* ------------------ CONV1 ------------------ 
void ReadConv1Weights_float(char *filename, char *datasetname,
                            float W[CONV1_DIM][CONV1_DIM][IMG_DEPTH][CONV1_NBOUTPUT])
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, W);

    H5Dclose(dataset);
    H5Fclose(file);
}

void ReadConv1Bias_float(char *filename, char *datasetname, float *b)
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, b);

    H5Dclose(dataset);
    H5Fclose(file);
}


 ------------------ CONV2 ------------------
void ReadConv2Weights_float(char *filename, char *datasetname,
                            float W[CONV2_DIM][CONV2_DIM][CONV1_NBOUTPUT][CONV2_NBOUTPUT])
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, W);

    H5Dclose(dataset);
    H5Fclose(file);
}
*/
/*
void ReadConv2Bias_float(char *filename, char *datasetname, float *b)
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, b);

    H5Dclose(dataset);
    H5Fclose(file);
}

*/
/* ------------------ FC1 ------------------ 
 Buffer format : [POOL2_H * POOL2_W * POOL2_NBOUTPUT][FC1_NBOUTPUT]
void ReadFc1Weights_float(char *filename, char *datasetname,
                          float W[POOL2_HEIGHT*POOL2_WIDTH*POOL2_NBOUTPUT][FC1_NBOUTPUT])
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, W);

    H5Dclose(dataset);
    H5Fclose(file);
}

void ReadFc1Bias_float(char *filename, char *datasetname, float *b)
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, b);

    H5Dclose(dataset);
    H5Fclose(file);
}
*/

/* ------------------ FC2 ------------------ 
void ReadFc2Weights_float(char *filename, char *datasetname,
                          float W[FC1_NBOUTPUT][FC2_NBOUTPUT])
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, W);

    H5Dclose(dataset);
    H5Fclose(file);
}
*/
/*
void ReadFc2Bias_float(char *filename, char *datasetname, float *b)
{
    hid_t file, dataset;
    herr_t status;

    file    = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file, datasetname, H5P_DEFAULT);

    status = H5Dread(dataset, H5T_NATIVE_FLOAT,
                     H5S_ALL, H5S_ALL, H5P_DEFAULT, b);

    H5Dclose(dataset);
    H5Fclose(file);
}

*/

/****************************************************
 * 2) CONVERSION FLOAT → FIXED POINT
 ****************************************************/

void ConvertWeightsToFixed(
        float conv1_W[CONV1_DIM][CONV1_DIM][IMG_DEPTH][CONV1_NBOUTPUT],
        float conv1_B[CONV1_NBOUTPUT],
        float conv2_W[CONV2_DIM][CONV2_DIM][CONV1_NBOUTPUT][CONV2_NBOUTPUT],
        float conv2_B[CONV2_NBOUTPUT],
        float fc1_W[POOL2_HEIGHT*POOL2_WIDTH*POOL2_NBOUTPUT][FC1_NBOUTPUT],
        float fc1_B[FC1_NBOUTPUT],
        float fc2_W[FC1_NBOUTPUT][FC2_NBOUTPUT],
        float fc2_B[FC2_NBOUTPUT],

        /* OUTPUT FIXED */
        short conv1_W_fp[CONV1_NBOUTPUT][IMG_DEPTH][CONV1_DIM][CONV1_DIM],
        short conv1_B_fp[CONV1_NBOUTPUT],
        short conv2_W_fp[CONV2_NBOUTPUT][POOL1_NBOUTPUT][CONV2_DIM][CONV2_DIM],
        short conv2_B_fp[CONV2_NBOUTPUT],
        short fc1_W_fp[FC1_NBOUTPUT][POOL2_NBOUTPUT][POOL2_HEIGHT][POOL2_WIDTH],
        short fc1_B_fp[FC1_NBOUTPUT],
        short fc2_W_fp[FC2_NBOUTPUT][FC1_NBOUTPUT],
        short fc2_B_fp[FC2_NBOUTPUT])
{
    unsigned short k, z, y, x;

    /* ---------- CONV1 ---------- */
    for (k = 0; k < CONV1_NBOUTPUT; k++) {
        conv1_B_fp[k] = (short)(conv1_B[k] * (1 << FIXED_POINT));

        for (z = 0; z < IMG_DEPTH; z++)
            for (y = 0; y < CONV1_DIM; y++)
                for (x = 0; x < CONV1_DIM; x++)
                    conv1_W_fp[k][z][y][x] =
                        (short)(conv1_W[y][x][z][k] * (1 << FIXED_POINT));
    }

    /* ---------- CONV2 ---------- */
    for (k = 0; k < CONV2_NBOUTPUT; k++) {
        conv2_B_fp[k] = (short)(conv2_B[k] * (1 << FIXED_POINT));

        for (z = 0; z < CONV1_NBOUTPUT; z++)
            for (y = 0; y < CONV2_DIM; y++)
                for (x = 0; x < CONV2_DIM; x++)
                    conv2_W_fp[k][z][y][x] =
                        (short)(conv2_W[y][x][z][k] * (1 << FIXED_POINT));
    }

    /* ---------- FC1 ---------- */
    for (k = 0; k < FC1_NBOUTPUT; k++) {

        fc1_B_fp[k] = (short)(fc1_B[k] * (1 << FIXED_POINT));

        for (z = 0; z < POOL2_NBOUTPUT; z++)
            for (y = 0; y < POOL2_HEIGHT; y++)
                for (x = 0; x < POOL2_WIDTH; x++) {

                    int idx = (y * POOL2_WIDTH * POOL2_NBOUTPUT)
                            + (x * POOL2_NBOUTPUT)
                            + z;

                    fc1_W_fp[k][z][y][x] =
                        (short)(fc1_W[idx][k] * (1 << FIXED_POINT));
                }
    }

    /* ---------- FC2 ---------- */
    for (k = 0; k < FC2_NBOUTPUT; k++) {

        fc2_B_fp[k] = (short)(fc2_B[k] * (1 << FIXED_POINT));

        for (z = 0; z < FC1_NBOUTPUT; z++)
            fc2_W_fp[k][z] = (short)(fc2_W[z][k] * (1 << FIXED_POINT));
    }
}



/****************************************************
 * 3) NORMALISATION DES IMAGES MNIST → FIXED POINT
 ****************************************************/

void NormalizeImg_fixed(unsigned char *input, short *output,
                        short width, short height)
{
    int size = width * height;
    int i;

    /* (pixel / 255) * 2^Q */
    for (i = 0; i < size; i++)
        output[i] = (short)((input[i] * (1 << FIXED_POINT)) / 255);
}



/****************************************************
 * 4) LECTURE DES IMAGES PGM MNIST
 ****************************************************/

void ReadPgmFile(char *filename, unsigned char *pix)
{
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR: Cannot open %s\n", filename);
        exit(1);
    }

    char header[10];
    int width, height, max;
    fscanf(f, "%s", header);
    fscanf(f, "%d", &width);
    fscanf(f, "%d", &height);
    fscanf(f, "%d", &max);

    for (int i = 0; i < width * height; i++)
        fscanf(f, "%c", &pix[i]);

    fclose(f);
}
