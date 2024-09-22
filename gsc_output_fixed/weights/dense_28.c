/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 128
#define FC_UNITS 64


const int16_t dense_28_bias[FC_UNITS] = {14, 67, 29, 27, -42, -42, 122, -9, -42, 30, 53, 139, 35, 67, 57, -64, 46, -13, 62, -31, -49, -16, 71, -12, 119, -11, 81, 74, 27, -31, -4, 65, 4, 42, 58, 66, 16, -12, -44, -12, -51, 80, -15, -22, 102, 8, -42, -9, -31, 5, -84, -23, 93, 70, -49, 52, 54, -49, 2, 79, 23, -3, 33, 51}
;

const int16_t dense_28_kernel[FC_UNITS][INPUT_SAMPLES] = {{10, 39, 3, -48, -19, 8, 3, -5, 39, -34, -14, 67, 33, -78, -42, -5, -26, 16, -6, 19, 30, 12, -10, -44, -23, 37, 45, -10, 6, -4, -3, 83, -41, -13, -31, 29, -11, -36, -16, -40, 45, 18, 10, 11, -35, 53, -2, 32, 18, -19, 17, 27, -11, -8, 15, 3, 105, 28, -67, 23, -13, -18, -23, -15, -18, 13, 5, -73, 6, 9, 6, 2, -51, -12, -6, -112, 28, -11, 64, 9, -60, 22, -19, -30, 28, -8, 9, 5, -5, -49, 15, 31, 27, 4, 1, 44, 14, -29, -78, -97, 6, -16, -36, 12, -4, 27, 1, 25, -43, -34, 8, 107, 26, 29, 14, 24, 101, 5, -195, -19, -2, -20, 123, -29, -168, 5, 11, -13}
, {25, -25, -59, 6, 36, 41, -4, -99, 12, -111, 18, -42, 50, 41, -10, 2, -32, -5, -99, -69, 26, 1, -25, 0, 4, -2, -31, 9, 32, -4, 21, -25, -66, 22, -6, 1, -45, -6, 12, 30, 67, 19, -26, 26, -11, 49, 26, 15, -20, -56, -2, 31, 4, -14, 11, 34, -14, 15, 11, 28, 16, -18, -23, -6, 16, -21, 51, 138, 16, 12, 64, -4, -3, -12, 7, -43, 15, -58, 56, -32, -14, 29, -26, 12, 68, 5, -8, 11, -9, -29, 22, -1, 68, -28, 6, 42, -6, 20, -49, -125, 27, 10, 55, -37, 8, 14, -44, 4, -51, -144, -66, 28, 0, -20, 39, -9, 23, -31, -25, 1, 28, 10, 65, 40, -163, -13, -15, -19}
, {-6, -13, 14, 14, -43, 4, 22, -1, 26, 74, -12, -47, -12, -30, 42, 12, 25, 29, 15, -30, -8, 0, -116, 20, -80, 25, -31, 4, 22, -96, 8, 18, -32, -18, 15, 10, -74, -41, 3, -25, 23, 13, 40, -3, 17, -54, -3, 8, 4, -45, 7, 94, -10, -11, -15, -21, 43, 25, 21, 38, -15, -23, -20, -15, -48, -2, -21, -2, 10, -13, -12, 11, 44, 27, 8, -60, 19, -38, -21, 5, -34, -8, -35, -66, 12, -4, 21, 22, 11, -52, -2, -102, 36, -101, -1, -38, -1, 83, -30, 28, 34, -16, -43, 22, 15, 10, 0, 27, 3, -48, -51, -34, 14, -44, -7, 10, -108, 40, -93, -28, 11, -22, 49, -18, 26, 25, 22, -24}
, {-18, 32, 3, 8, 11, 13, 14, -31, 121, -78, -6, 7, -3, -77, -58, -1, 46, 16, 22, -53, -14, -10, -103, 45, -30, 24, 31, 7, -50, 7, -4, 44, 83, 8, 40, -11, -41, -83, 25, 9, 9, 15, -87, 14, 41, -5, 24, -42, 1, -19, 5, 68, 1, 9, 24, 2, 9, -12, -17, 19, -12, 5, -38, -7, 25, 20, 65, -103, 20, 21, 75, 2, 34, -6, -30, -59, 2, -28, 24, -9, -58, 14, 37, -30, 3, -3, 23, -7, 25, 0, 1, -16, 24, 43, -19, -41, 6, -15, -54, -26, -23, -20, 28, -16, 32, -1, -42, 10, -19, -50, -8, 40, 16, -111, 7, -7, -16, -22, -53, 1, -4, 48, 33, 0, 93, 0, 25, -58}
, {2, -11, -104, -41, -40, -11, -20, -55, -59, 9, -24, -36, -22, -101, -17, 11, 36, -8, 46, -56, 0, 9, -8, -65, 25, 7, -16, -14, -37, -51, -28, -14, -30, -51, -39, -5, -44, -66, 2, -25, -167, -9, 74, 5, -63, 32, -30, 3, -4, -45, 5, 13, 3, 7, -2, 10, 5, -9, -7, -48, 10, 13, -65, 14, 26, -28, -100, 53, -6, -6, -62, 24, -61, 2, -1, -69, 5, -9, -17, 60, -38, 11, -22, -40, 9, -4, -18, -22, 14, -47, -6, -155, -46, -113, -17, -103, -2, 57, -42, -24, -9, -1, -22, -28, -29, -6, -37, -7, -12, -30, -69, -62, 13, -76, -13, 4, 17, 6, 58, 5, 4, -97, 4, -31, -8, 2, -8, -3}
, {-25, -6, -44, -28, -46, 0, -5, -1, -46, -28, 16, -49, -14, -20, -34, 8, 31, 8, 29, -77, 1, -9, -20, -75, 1, 10, -94, 14, 3, 9, 9, -58, -17, -21, -4, -4, -16, -58, -25, -10, -28, -22, -8, 7, -27, 15, 13, -11, -13, -3, -31, 1, -20, -5, 6, -54, -56, -25, -83, -20, 12, -28, -22, -5, -27, 1, 4, -14, -22, 0, 1, 4, -34, 7, -3, 14, -14, -30, -9, -49, 9, -10, -43, -10, 3, -1, -8, 14, 5, -18, -25, -12, -28, -64, -13, -39, -23, 42, -29, -69, -10, 1, 24, -20, -11, 10, -8, 5, -43, -32, -33, -10, -20, -6, -1, -14, -44, -61, -8, -1, -13, -33, -6, -27, 1, -9, -12, -6}
, {60, -16, -22, 18, 16, -34, -36, -9, -145, -37, 32, 20, 58, 27, 88, -16, 23, 22, -3, -175, -14, -20, -105, 82, -108, 14, 13, -15, 11, 80, -20, -47, -81, -15, -154, 14, -19, -30, -1, -3, -3, -34, -30, -20, -48, -40, -19, -27, 4, -143, 16, -127, 1, 11, -9, 19, -23, -21, 34, 8, 0, 47, -9, -7, -55, -7, -9, 4, -13, -7, -76, -14, 2, -8, 79, 41, 4, -1, 51, -85, 19, -19, 45, -33, -8, 28, -6, -6, 20, 38, 5, -21, -30, 29, 12, 11, 14, 51, -86, -144, -18, -13, 33, -28, 52, -5, -60, 12, 12, -32, 6, -81, -1, -61, -57, 12, 52, -20, -71, -36, -15, 40, 55, -12, -48, 0, -17, -139}
, {-12, -13, -30, 2, -40, 5, 13, -32, -18, 15, 0, 4, 8, -10, -22, -4, -18, -23, -30, -8, 13, -18, -23, -29, -17, -1, 3, 9, -14, -12, 21, -17, -8, -10, -24, -14, -47, -32, -12, -5, -21, -8, -20, 2, -28, -15, -27, 9, 17, -9, 5, 2, 7, -27, 3, -27, -15, -27, -46, -26, -22, 13, -32, -5, -26, -5, -13, 0, 8, -27, 7, 31, 0, 12, 13, -25, -11, -16, 9, -33, -37, -4, -21, -17, -24, -20, 12, 5, -12, -30, 11, -4, -1, -23, -14, -13, -3, -9, -16, 7, -4, -13, 10, -24, -5, 26, -20, 11, -2, -20, -9, 8, 11, -43, 0, -9, -27, -25, -26, 1, -21, -26, -24, -9, 22, 1, 13, -40}
, {-3, -14, -17, -19, -20, -12, 14, -78, 33, 41, 4, 12, -3, -25, -49, 2, -27, -13, -94, -43, -2, 6, -69, -97, -63, -23, -2, -27, -10, -37, -24, -69, -76, 51, -47, 10, -6, -32, 3, -30, -66, -30, -21, 14, -49, 19, -19, 12, -29, 7, 14, -40, -60, -25, -18, -62, 20, 7, -48, -1, -6, 5, -119, -19, -45, -1, -26, 73, -19, 13, 36, 23, -41, -20, -40, -11, 4, -24, -29, -40, -54, -10, 2, -54, -51, 5, -16, -18, 10, -20, 5, -67, 4, -27, -13, -48, 7, -37, -83, -39, -1, -6, -4, 14, -27, -8, 38, -20, -22, -60, -20, -33, -4, -59, 8, -19, -43, -114, -22, 4, 8, 18, 48, -43, -4, -23, -7, -21}
, {8, -13, -38, 86, 17, 28, 44, -14, -5, -89, 9, -63, -9, -150, 70, 10, -113, 18, 7, -12, -6, 8, -24, 74, 3, -2, 20, -1, 14, -50, 5, 18, -19, 26, -31, 29, -24, 20, -7, 0, -62, 12, -55, 5, 62, 32, 9, 7, -6, 11, 6, -219, -151, 18, 14, -78, 6, 4, -73, 69, 18, -18, -27, -10, -50, 21, -63, -35, 15, 13, 10, 79, -24, 9, 25, 8, -25, -16, -81, 28, 43, 21, -54, -12, -43, -7, 14, 15, 1, -93, 15, -54, 26, -48, -24, -65, -6, 36, 9, -81, -66, -18, -39, 27, -4, 5, -5, 19, 18, 30, -68, -74, -3, 28, 25, 25, -50, 17, -114, -1, -8, -70, -7, 17, 0, 23, 4, 34}
, {52, 56, 6, -36, 21, -35, -38, -27, -82, 31, 17, -50, -4, -105, 15, 0, 27, -13, 0, 29, -17, 14, -84, 45, 9, 17, 50, 1, -75, 3, 22, 5, -19, -74, -54, 3, 3, -5, -19, 16, -10, -18, 53, 13, -6, -27, -2, 91, 5, 17, -7, 11, 60, -19, -9, 31, 73, -1, 4, 6, 2, 32, -3, 1, -113, -8, -136, -121, -22, -14, -103, -12, -86, -4, 2, 18, -24, -31, 36, -61, 1, -13, 25, -6, -106, 10, -7, 1, -12, 10, -13, -52, -11, 39, 14, -110, 4, 14, -33, -9, -8, 13, -54, 15, -13, 1, 90, -20, 19, -19, -12, -24, 13, 8, -15, -10, 38, -4, -72, 13, 18, 1, -54, 21, -84, -13, -14, -44}
, {3, -5, 26, -178, -34, -1, 3, -70, 3, -85, 6, -85, -30, -41, 53, -11, -164, -27, -63, 2, 23, 13, -5, -70, 21, 68, -4, 16, 16, -7, -5, -44, -21, 37, -88, 2, 7, -12, 18, 57, 24, 21, -34, -26, 65, 10, -32, 41, -27, 19, 27, 56, -20, 14, 26, -12, -95, -9, -35, 43, 16, 2, 32, 1, -11, -29, -36, -39, 13, -9, -86, -27, -64, -24, 46, -1, -8, 24, -63, 31, -73, 19, -15, -57, -20, -11, 26, -6, 25, -47, 8, -35, -66, -55, -33, -31, 0, 16, 7, -89, -9, 20, 41, 27, -29, -24, 41, -1, -18, 13, -109, -29, 5, 17, 15, 24, 18, -92, 55, -26, -28, -56, 15, 38, -87, 11, 11, 23}
, {-16, 52, -88, -68, -29, 20, -15, 24, -39, 34, 17, 53, -51, -87, -36, -1, -6, -21, -74, -20, -12, 22, -34, -69, -32, 22, -7, 29, 51, -74, 10, -4, -59, -23, -83, 2, 8, -37, -2, 5, -54, -13, -17, 10, -81, 39, -8, -37, -16, 29, 8, 54, -116, -2, -2, 93, 35, -17, -86, -27, 3, 3, 19, -12, -39, -12, -19, -10, -5, 9, -57, -24, 20, 17, -39, -6, 27, -78, -85, -92, -61, -5, 23, -39, -81, -7, 19, 14, -6, -131, -10, 57, -9, -79, 15, -55, -26, 21, -11, 33, 30, -3, 69, -29, 17, 40, 38, 10, 8, -6, -15, 61, -12, 67, 2, -4, -29, -37, -19, 1, 24, -6, 1, -9, -20, -6, -9, 12}
, {-31, 44, -45, -19, 19, 19, 12, 0, -13, 30, -8, -11, -9, -55, -63, -20, -29, -55, -9, -41, 3, 6, 1, -107, 3, 77, -18, 15, -32, -41, -9, 42, -56, 36, -3, -27, 6, -21, 18, 32, 42, -38, -17, 8, 31, 2, -13, 19, -16, 35, -30, 23, -38, 4, -3, -47, 10, 15, 25, 25, -11, 34, 16, -15, 44, -16, -26, 67, 0, -15, -102, 131, 2, 5, -40, -37, -25, -20, 51, -9, 31, -9, -135, -25, -90, -24, 11, -20, 9, 7, -8, -12, 13, 19, 48, 83, 0, 58, 39, 38, 64, -27, 53, -17, 32, -8, -40, -7, -67, -67, -49, -42, 10, -66, 21, -20, 32, 52, 83, 7, -19, -41, 23, 65, 81, -9, -10, -78}
, {72, 43, 2, 2, 11, 0, -3, -18, 16, -11, -26, -2, -9, 0, -16, 16, -74, 11, -28, 14, 6, 3, -66, 3, 64, 14, 2, 30, 45, -30, 14, 12, -79, -13, 25, -13, 7, -84, -18, 6, -18, -14, 44, -11, 24, 66, 16, -28, -35, -52, 6, 17, -24, 19, 0, -45, -1, 2, 19, 59, 17, 39, 4, -9, 69, 11, -49, -3, 25, 21, -85, -51, -26, -11, -70, -50, 17, 5, -9, 12, -40, 7, 20, -81, -144, 6, 19, 24, -5, -21, 22, 60, 24, -8, 21, -103, -6, 6, 21, 12, -12, 17, -13, -5, -3, -16, -62, -6, -109, -22, -40, -35, 27, -34, 9, 16, -127, -25, 0, -14, -2, 47, 4, -34, -16, 1, -8, 8}
, {-24, -15, -23, -10, -26, 18, -13, -66, -6, -49, 15, -25, -17, -12, -19, -21, -19, -20, -7, -34, -9, 5, -38, -5, -58, 13, -26, -4, -41, -40, -22, -86, -13, -17, -27, 15, 2, -18, 13, -25, -29, 5, -92, 10, -19, -20, 3, 10, -14, -26, -11, -25, 5, -1, -23, -25, -60, -11, -35, -20, -10, 11, -36, 9, -32, 3, -48, -11, -27, -2, -27, -19, -40, -6, -25, -48, 3, 6, 7, -41, 2, -10, -12, -49, -5, -3, 6, -24, 3, -62, 6, -54, -3, -19, -21, -18, -10, -79, -23, 8, 0, 4, -25, -2, 12, 18, -20, -25, -8, -16, 0, 12, 7, -2, -13, -4, -33, -16, 0, -12, 5, 6, -26, -37, -24, -6, 4, -27}
, {83, -26, -65, 72, 53, 31, 38, 18, -30, 45, 22, -9, 4, -45, 26, 26, 26, 9, 19, 17, -11, 12, -63, 69, 3, 4, -60, 12, -80, 0, 16, 22, -39, 2, 76, 12, -17, -17, -8, 22, -104, 25, -115, 19, -22, 43, 17, 11, -32, 38, 7, 82, 35, 26, -3, -76, 36, -10, -51, 61, 4, 5, -1, 23, 43, 7, -153, -1, -8, 19, 17, 3, 14, 4, -61, -22, 24, 14, -72, 47, 16, 5, -31, 29, -86, -5, 1, -14, -14, -42, -6, 25, 1, 26, -17, -96, 4, 108, 25, 29, 69, -14, -10, 42, -8, -9, -7, 1, 39, -38, -30, -44, 1, 1, -12, -8, 19, -15, 5, 1, 23, 8, 18, 31, 102, -1, 28, 10}
, {-3, 9, 3, -22, 6, -29, -20, 77, 6, 35, 7, -34, -2, -37, -65, -1, 9, 7, 12, -24, 3, 2, -52, -24, -62, 14, -10, 12, 10, -18, 2, -26, -22, -40, 4, 12, 55, -31, 12, 5, 12, 3, 9, -30, -85, 4, -9, -23, -17, -15, -2, -64, -65, 0, -29, -71, -9, 6, -52, 0, 14, 3, -100, 14, 17, -15, 7, -14, -4, -8, 21, 28, 18, -10, -46, 39, -1, -64, 48, -50, -15, 13, -23, 25, -52, 1, 7, 0, 2, -41, -18, -62, -20, -35, -5, -10, -23, 63, -79, -7, -6, -3, -19, 13, 13, 17, -23, 10, -42, -34, -110, -6, 8, -7, 2, 9, 32, -127, -62, 3, 14, -76, -8, 6, 101, 5, 9, -12}
, {-2, 7, -30, 89, -24, -55, 35, 10, -14, -43, 8, 0, 47, -85, 39, -9, -61, 0, -2, -1, -11, 13, -78, 42, -37, 15, 13, -9, -5, -55, 12, 4, 7, -37, -70, 7, -49, 15, -11, 0, 44, -13, -9, 8, -11, -9, 4, 30, -14, 4, -15, -60, -41, -7, -6, 8, -13, 14, 41, 33, 18, 23, 31, 28, -50, 11, -28, 102, 21, 18, -1, -8, -3, -14, -11, 19, 1, 11, -23, -35, -6, 0, 53, -49, 39, 8, -14, -4, -9, 19, 12, -114, -32, -56, -12, 19, 19, -16, 7, -81, -42, -4, -8, -1, 13, -17, 43, 18, 47, 5, 42, -69, 21, 9, 22, -6, -69, 25, -114, 6, -15, -12, 13, 23, 36, 9, 28, 7}
, {-17, -1, -17, -17, -54, -11, -7, -25, -42, 14, -24, -18, -16, -5, -12, -1, -17, -21, -69, -34, -30, -16, -24, 12, -45, -17, -33, 12, -27, -47, -19, 1, -2, -31, -4, 9, -54, -26, 6, 0, -24, -7, -17, 3, -10, 9, 6, -16, -5, 10, -10, 60, -69, -2, 8, -44, -33, 1, -39, -6, -7, -25, -23, 8, -31, -10, -31, -8, -2, 8, 14, -11, -17, 3, -23, -92, 3, -68, -63, -15, -21, -28, 4, -27, -22, -23, -24, 6, -19, -22, -17, -26, 12, -67, -21, -29, -12, -31, -72, -12, 11, -9, 13, 12, -24, 9, -44, -19, -27, -3, -94, 12, 6, -9, -1, -15, -1, -39, -18, -24, 8, -4, 7, -27, -12, -24, -19, -17}
, {12, -10, 5, -51, -36, -45, -24, -35, 28, -126, 1, 21, -29, 33, 30, -2, 50, -3, -65, 1, 4, -24, -29, -57, 14, -5, -48, 9, -62, -9, -28, -33, 8, -63, -54, 5, -30, -78, -25, 10, 54, -27, -45, -26, -82, 0, -17, 6, 4, -54, -16, -98, 33, -30, 27, -23, -10, 13, -59, 57, -5, -32, -42, 11, -59, -6, -33, -58, -28, 12, 14, 9, -48, -18, -62, -95, -21, -100, 14, -27, -14, -15, -77, -45, -63, 11, -27, -25, -12, 60, -2, 8, -21, -153, -18, -55, 4, 59, -89, 50, -30, -30, -8, -19, 7, 12, 100, -14, -45, -83, -33, 36, 4, 4, -15, -30, 39, -95, -76, 13, 0, -72, 84, 58, -62, -1, 9, -25}
, {6, 22, -21, -30, -22, -3, -28, -12, -4, -23, -3, -4, 19, 0, 10, 26, -7, -10, -2, -2, -15, -2, -9, -4, -32, -11, -1, -9, 12, -20, -6, -23, 0, -4, -2, -23, -6, -1, 2, 29, 12, 25, -14, -28, -26, 16, -3, 13, -28, -17, 6, -18, -20, -15, 8, -20, 10, -12, -20, -10, 7, 2, -8, 8, -43, 2, -6, -12, -7, 11, -21, 16, -19, -8, -4, -20, -29, -5, -23, -30, -35, 2, -7, -4, -16, -24, 14, 11, 4, -3, -14, -5, -4, -3, -7, -15, 16, -17, -15, -18, 7, 7, 12, -5, 10, -13, -20, 14, -7, 7, -51, -20, 9, -31, -5, -4, 12, -29, 6, -18, 4, -24, -26, -32, 0, -24, 17, -28}
, {-14, 76, -2, 8, -36, -11, 7, -30, -48, -40, 27, -80, -33, -112, -15, -7, -104, -17, 5, -1, 4, -1, 23, 12, -21, 5, 31, -3, -28, -67, 8, -27, -92, 29, 44, -7, -6, -36, 17, 2, -4, -7, -5, 13, 1, 31, -14, -18, -1, -22, -5, 24, -58, 13, -8, 46, 56, -27, 36, 41, 25, -25, 4, 23, 16, -7, -88, -74, 5, -1, 37, -129, -36, -5, -49, -61, -4, -14, -18, -41, -26, -15, -25, -6, -6, -15, 6, 0, -7, -37, -11, -22, 12, -32, -23, 83, 13, -40, 25, -127, 18, 2, -20, -25, -17, -38, -47, 12, -21, -18, 13, -76, -17, 54, -16, 3, 7, 26, -39, -17, 8, 44, 35, 4, 8, 6, 16, 8}
, {6, 8, -73, 3, -44, 2, -2, -57, -36, 50, 5, -23, 11, 109, -42, 23, -8, 9, -52, -28, 22, -13, -22, -4, -27, 24, -6, -6, -42, -55, 22, -20, -25, -21, -30, 25, -24, -68, -7, 27, -6, 23, 77, 0, -35, -48, 8, -4, 14, 24, -6, -114, -52, -2, -3, -9, 19, -4, 0, -12, -31, 9, -25, -7, -3, 3, 4, 0, -5, -5, -19, 9, -38, 18, 14, 10, 4, 13, -63, 44, -32, 30, -9, -9, -15, 8, -22, 4, 7, -2, -8, 23, 28, -19, -19, -33, 14, -44, -20, -71, 57, 8, 17, -5, 18, -18, -15, -20, -19, -9, 0, -13, -19, -54, -18, -12, 12, -80, -118, -20, 10, -13, -6, -25, 18, 18, 14, -83}
, {50, 49, 25, 56, -7, -14, 28, -102, 39, 12, 19, -14, 18, -57, 18, 7, -81, 62, 21, 61, -1, -10, 79, 69, -5, -28, 60, -11, -124, -12, -10, 26, -57, 10, 19, -22, 19, -19, -28, 43, -8, 21, -142, 20, 3, 21, 44, -23, -38, -2, 2, 70, -66, 9, 21, 63, 41, -13, 12, -17, 6, -22, -54, 16, -47, -11, -76, 35, 13, 23, 80, 20, -70, -18, -61, 25, -12, 7, -35, 24, -1, 12, 33, 27, 71, -10, 24, 26, 16, -31, 3, -48, -66, 10, -4, -1, 15, 8, -5, -72, -20, 8, 10, 42, -2, -14, 41, 24, -7, 63, -11, -46, -1, 24, 20, 18, 19, 23, 5, -11, 20, 54, 17, -21, -74, -11, -13, 29}
, {11, -23, -35, -3, -108, -32, -16, 46, -31, 9, 13, -48, -11, -8, -17, 2, -52, -7, -16, 13, 10, 3, 9, 11, -52, 5, -48, -7, -13, -52, 11, -30, -32, -43, 0, -27, -81, 3, -12, 19, -28, 10, -23, -19, -7, -2, -4, -8, 7, 27, -17, -15, -87, -4, -26, -39, 40, -1, -62, -24, 14, -18, -86, 4, -90, -13, -5, -94, -2, -7, -29, 12, -23, 6, -25, 17, 10, -14, -45, -6, 22, -10, -25, 2, -96, -14, -25, 11, -2, 28, 10, -48, -5, -25, 10, 4, 2, -35, -25, 65, -3, -26, 12, -19, 5, -20, -37, 0, -61, -23, 24, -49, 12, -138, 2, -15, -15, -150, -22, 12, -24, -98, -4, 6, 47, 0, -5, -83}
, {43, 19, -75, 4, 15, 79, 8, 73, -33, 108, 27, 55, 29, 113, -13, -13, 15, -23, 67, -71, 7, 17, -72, -13, 10, -77, -61, -20, 80, -6, -1, -45, 39, -41, -31, -9, 6, -10, 16, 25, -40, 8, -8, 4, -12, -54, 36, 48, 52, -31, -5, -64, 11, -14, 14, -44, -67, -11, 28, 70, 9, -12, 43, 19, 2, 24, -80, -51, 13, -28, 48, 8, 14, 28, -36, -24, -38, -53, 17, 10, 14, 7, 46, -44, 15, 20, 6, 6, 9, 54, -5, 2, 12, 3, 20, -15, 2, 115, 6, 118, 42, -2, -18, -72, 27, -12, -27, -1, -5, -77, -1, 48, 12, -46, -18, -4, -38, 14, -51, -12, -10, -1, -33, 66, 129, 20, 13, -80}
, {47, 65, -51, -46, 32, 34, 19, -4, -108, 89, -16, 24, -8, 41, -29, -21, 4, -17, -4, -35, -4, -9, -98, 1, -2, 68, -43, -27, -92, 110, -2, -10, -16, 55, -105, 5, 8, -25, 3, 57, -13, 3, -47, -23, 30, 33, -40, 11, 6, -93, -19, -54, 61, -30, -6, -13, -66, -15, 6, 50, -27, -29, -3, 8, -7, 6, -56, -56, -25, -13, 41, -24, -16, -6, -24, -37, -24, -14, -6, -42, 22, -20, -127, 12, -44, -20, -11, 8, -14, 39, -10, 65, 5, 26, -8, -26, -26, 40, -15, -45, 78, 4, -22, 71, -10, -6, 52, -28, 9, 18, 6, 48, -18, -59, 17, -11, -13, -25, 33, 15, -24, -60, 21, 22, 58, -25, -19, -120}
, {10, 13, -69, -17, -92, -7, 21, 9, -70, -67, -30, -30, -16, 10, 4, 8, -42, 14, 58, -52, -5, -11, -87, 20, -81, 14, 24, 2, 5, -54, -13, -114, -79, -18, 2, 8, -76, -76, -25, -11, -26, 5, -56, -11, -45, -29, -23, 22, 5, -146, 7, -47, -82, -23, 3, 7, -80, -5, -94, -16, 13, -18, -35, -15, -109, -12, -80, 59, 10, -17, 134, -8, -13, -21, -91, 33, 9, -2, -38, 45, 57, -25, -40, 5, -98, -11, 13, 10, 2, -3, -15, 12, 18, -116, -8, -92, -2, -79, -2, 7, 5, 22, -54, 22, -11, 5, -32, -24, -103, -24, -96, -139, 6, -185, -7, 14, -49, -65, -21, -32, 6, -53, 44, 16, -42, -28, 1, -17}
, {-8, -16, -7, 13, -41, -33, -1, -9, -23, -17, 0, 67, -3, -12, 66, -9, -24, -31, 21, -37, -1, -2, -19, -16, -30, 25, -95, -22, -92, -29, 9, -16, -55, -54, -18, 8, -34, -27, -17, -33, -1, 10, -75, -13, -15, 7, 1, -23, -22, 5, 12, -20, -1, 11, -10, -44, -52, -9, -93, -25, -13, 3, -35, -11, -15, -10, -63, 20, 9, -4, -97, -16, 3, 0, -24, 62, -21, -42, -10, -65, -75, 8, -17, -32, -50, 14, 8, 9, -8, -117, 5, -94, -5, -28, -28, -61, 11, -22, -48, 7, 5, -2, -2, 6, -18, -4, -6, -19, -21, -79, -18, -80, -18, -47, -5, 13, 12, 14, -29, 11, 20, 7, 6, -18, -52, -22, 2, -9}
, {90, -31, -48, -94, -9, -41, 11, -57, 0, -62, 1, 40, -16, 54, -9, 20, -4, -17, 43, 40, -7, -8, -57, -4, 58, -47, 28, 4, -10, 56, -5, -4, 48, -35, -97, 24, -46, -18, -5, -3, -46, 3, -40, 28, -16, 3, 4, 0, 27, -23, -4, 44, 14, -11, 16, -14, 28, 14, -26, 67, -11, -2, -41, 20, -76, -11, 27, -71, 22, 7, -8, 5, -59, 8, -5, -51, 0, -60, 11, -24, 27, 2, 36, -66, 22, -25, 22, 2, -14, 31, -10, 50, 4, -5, -14, -101, 15, 16, -25, -18, 47, 2, 20, 45, 35, -1, 41, 18, 8, -24, 19, 56, -1, 32, -5, 2, 46, -11, -71, 10, -12, 33, 130, 15, -66, 25, 31, 26}
, {-5, -30, -69, 20, 42, 42, -11, 17, -48, 72, -14, 76, 12, -30, -57, -8, 50, -10, 40, -3, -3, 0, 15, 27, -104, 21, 45, -1, 1, 50, 22, -26, 14, -82, -124, -19, 50, -34, 15, -31, -38, -27, -71, -7, -16, 31, -3, 3, -7, 26, 12, -48, 2, -30, -7, -88, 22, -29, -25, -12, 10, -38, -31, -7, 11, 8, -46, -125, -30, -14, -37, -2, -11, -7, 22, -1, -44, -13, 40, 27, 30, -7, -95, 19, -97, 16, 17, -7, -24, 51, -23, 16, -61, 35, -17, -19, 1, 147, 38, -4, 53, -5, 42, 28, -42, 10, 5, -24, 28, -28, 30, 4, -28, -51, -3, 3, -16, 22, 29, -4, -4, -7, 232, 6, 4, -5, -11, -74}
, {-25, -17, -83, 18, -50, -18, 7, -49, -55, -43, 31, -63, 18, -27, -46, -9, 39, 6, -10, -31, 3, -13, -59, -19, -63, -1, 9, -1, 6, -79, -22, 67, -11, -4, -3, -11, -34, -40, -5, -9, -28, 15, 23, 4, -13, -42, 12, 5, -2, -18, -9, -14, -40, -13, -12, 1, 20, 13, -24, -42, -5, -22, -48, 28, -33, -27, -38, 15, 11, 21, -24, 6, -70, 3, -14, -28, -14, -76, 47, -73, -48, -3, -32, -82, 15, 5, 8, -14, 12, -41, 23, -24, 0, -73, 19, -42, 29, -40, -24, -71, 6, -8, 22, 24, 3, 9, -36, -13, -40, -54, -58, 24, -5, -51, -8, 2, -24, -65, -85, 27, -6, -46, -1, -23, 15, 5, 12, -102}
, {-68, 24, 33, -89, -90, -48, -4, 55, 3, -47, -6, -47, -45, -45, 4, -11, -69, -31, 33, 4, -5, 13, 5, -16, 28, 25, -40, -5, 48, -94, -20, 35, -3, -25, 110, 5, -93, -50, -9, 6, 34, -9, -61, 14, 21, -48, -1, 29, 11, -124, 7, -59, 44, 14, -13, 33, 9, -28, 56, 13, -14, 3, 25, -2, 13, 28, 20, 11, 13, 12, -25, -107, 18, 0, -37, -53, 0, -1, 70, -13, -87, -11, 18, -34, 68, 0, -3, -29, -26, 23, -7, -34, 9, -58, -15, -34, 12, -32, 15, -27, -25, -6, -16, 10, 1, 5, 32, -4, -72, -66, -65, 87, 21, 22, -6, -21, -12, -39, -76, -3, -3, 24, -59, 26, -104, 6, 14, 11}
, {20, -27, -27, 83, -9, -10, 17, -25, 6, -18, -32, -4, 57, -27, -21, -29, 25, -1, -9, -29, 0, 11, -72, -64, -56, 30, 2, 31, 42, -12, 0, 50, -13, 47, 37, 0, 40, -33, 13, -71, 63, 7, -51, 0, -34, 43, -39, -18, -21, -67, -5, -150, -20, -12, 5, -70, -47, 14, 2, -12, 9, 7, 22, -11, -9, -17, 6, 3, -20, -24, -107, -50, 66, -3, 10, 11, -19, 21, 57, -51, -2, -21, -19, 30, 16, -7, 3, -7, 9, 31, -15, -57, -58, 41, -5, -118, 24, -24, 35, -36, -39, -21, -16, -6, -11, 7, 7, 6, -11, -7, 8, -84, -6, -29, -4, -10, -105, 32, -70, -27, 13, -11, 33, 37, -67, -27, -24, -81}
, {28, -46, -38, -42, -107, -78, -11, 0, -51, -65, -2, 23, -65, 21, 49, 5, 35, -25, 57, 26, -28, -2, -9, -84, 69, 8, 38, -9, 47, -2, 49, 25, -55, -63, -207, -3, 29, -48, 14, 41, -53, -7, -22, 0, -39, -36, -1, -4, 29, -79, 4, 113, -31, 6, -1, -94, -98, 10, -11, -3, -1, -4, -90, -7, -6, -15, -55, 2, -10, -28, 45, -50, -34, 21, -76, -37, -2, -85, -4, -47, 6, -22, 21, -99, 36, -2, -5, -17, -30, -15, -21, 37, -57, -89, 32, -30, 6, 22, -21, -70, -4, -6, -60, 35, -2, 18, -7, -16, -37, -61, -87, -86, -18, -112, 6, -6, -66, -87, -68, -5, 2, -69, -90, 83, 91, -16, -14, -86}
, {-11, 28, -23, -5, 33, -5, 16, -61, 61, 52, -5, 37, 7, -53, -19, -14, 3, -8, -154, -35, 27, 27, -16, 12, -24, 5, -93, 29, -32, -40, 12, 18, -54, -37, -25, -2, -66, -111, -8, 24, -44, 8, -28, -7, -14, -45, 17, 17, 5, -45, 7, -81, -18, 5, 23, 17, -19, -14, -97, -18, 2, 3, -48, -13, -152, 8, -81, -29, 17, 14, -86, -23, -42, -10, -60, -15, -10, -125, -13, -60, -53, 24, 11, -32, -52, -9, 16, 22, -11, -2, 0, 43, 10, -28, 11, -24, 7, -19, -10, 35, 17, -21, 18, -1, -25, 19, 42, 18, 34, -39, -33, 40, 6, -52, 29, -7, 7, -62, -46, 13, -11, -52, -44, -70, -13, 16, -10, -53}
, {-7, -25, -39, -16, -61, 12, -25, -51, -42, -31, -28, -13, 5, -27, -5, 2, -20, -8, -20, -25, 8, -15, -4, -24, -3, -20, -30, 7, -21, 23, -12, -53, -35, 0, -5, 5, -22, -10, -5, 0, -17, -16, -66, 13, 5, -4, 13, -15, -15, -33, -4, -30, 1, 11, 11, -59, -8, 1, -16, -4, -17, -12, -67, 1, -37, -27, -55, 8, -18, 4, -36, 17, -13, -23, -44, -49, -20, -65, 20, -43, -73, 11, -46, -28, 21, -27, -16, -3, -3, -12, -18, -19, 1, -28, -30, -70, 12, -39, -78, -60, -20, -31, -2, 9, 4, -1, -11, -19, -32, -22, -16, -46, 13, -7, -25, 1, 0, -64, 45, 4, -16, -46, 0, -27, -55, -25, -9, -4}
, {4, -12, 3, -23, -42, -15, 21, -18, -20, -12, 11, -1, 8, -8, -24, 13, -38, 13, -24, -38, -3, 8, -32, -9, -27, 1, -12, 28, -17, -21, 1, -46, -56, -8, 19, -24, -36, -45, 3, -3, -74, 16, -10, 15, -28, 2, 25, -13, 11, -22, -11, -5, -27, -1, 6, -30, -11, -1, -35, -46, 28, -51, -27, 3, -28, -8, -17, -50, 6, -8, -20, -17, -39, 14, -32, -31, 25, -51, -5, -24, -29, -10, 1, -23, 16, 21, 26, 6, 25, -21, -13, -9, 13, -32, 2, -17, 7, -4, -7, 8, -14, 8, 29, 11, -13, 7, -12, 23, -48, -21, -4, -22, -15, 7, -13, -21, -11, -19, 15, 9, -21, 2, 9, -11, 26, 15, 3, -5}
, {2, 39, -147, -5, -22, 5, 14, -74, -82, -31, -27, -43, 14, -65, -61, -15, -60, 12, -14, -51, 3, -10, -52, 7, -52, -20, -52, -12, -13, -72, -15, -35, -11, 4, 3, -9, -78, -66, -23, -1, -1, -14, -47, 5, -48, -35, 13, -5, -33, -38, -16, -77, 72, 13, 5, 38, -53, 3, -29, -15, -12, 9, 8, 3, -60, 8, -23, -28, 12, 5, 3, -31, -27, -18, -64, 10, 5, -81, 63, -49, -48, 14, -48, -6, -73, -25, 10, 11, 4, -48, 11, -135, -9, 5, -25, -11, 10, 61, 11, -48, 8, -2, -32, -11, -12, -13, -103, 2, -36, -5, 40, 15, -15, -46, 3, 13, -8, 46, -15, -14, 9, -3, -10, 30, -26, 14, 13, -177}
, {-26, -22, -3, -12, -36, 3, -4, -54, -27, -34, 1, -39, 6, -41, -13, -19, -6, 10, 10, 13, -18, -7, -22, -17, -52, -9, -17, -6, 2, -13, 20, -86, -56, -68, -68, -6, -36, -62, -19, -21, 11, 1, -14, -6, -41, -40, -15, -5, -5, 4, 5, -8, 31, 7, -3, -31, 68, 1, 14, 62, 6, -24, -44, 10, -8, -3, -41, -36, -23, -24, -12, 16, -26, -14, -50, -84, 6, -99, -43, -103, 3, -19, -19, -64, -100, -19, -3, -17, -10, -38, -28, -66, 5, -126, 7, -19, -10, -26, -102, -22, -21, 7, -20, 8, -9, -19, 18, -24, -21, 95, -76, -75, -15, -27, -30, 5, 3, -40, -1, 12, 4, -13, -26, -88, 10, -7, -6, 5}
, {-30, 20, -36, -6, -34, 10, -3, -49, 61, -54, -8, -42, -12, -12, 34, 13, -19, 4, -12, -24, 17, 4, -59, -52, -34, -1, -1, 3, -115, 51, -25, -18, -86, 19, 44, 5, 25, -26, 4, -13, -19, 3, -50, -2, -44, 6, 10, -30, 13, 23, 4, 35, -147, 1, -14, -51, -25, 20, 40, 46, 5, 13, -37, 5, 12, -9, 25, 27, -9, 22, 105, -24, 71, -31, -76, -24, 4, -51, -31, 53, 8, -8, 24, 64, -27, -1, -15, 0, -7, 4, 3, -3, 49, 41, -13, -104, 15, 1, -40, -168, 87, 6, 31, 21, 38, -14, -18, 6, -20, -10, 20, -10, 29, -4, 14, 16, -42, 35, 18, -11, -16, -42, 37, 9, -107, 12, 13, 14}
, {7, -14, -16, -6, -45, -12, -10, -23, 9, 6, -23, -9, -9, -25, -24, -25, -23, 13, -11, -11, -11, -24, -47, -4, 8, -9, -18, 8, 11, -2, 27, -30, 1, -2, -9, 17, -11, -18, 1, -17, -10, 12, -25, -6, -5, 13, -11, 6, -6, -4, -18, -37, -15, -13, -3, -32, -6, 14, -33, -28, -28, -17, -14, 2, -28, -14, -39, 18, 7, -12, -1, -4, -16, 14, -2, -27, -4, -36, 5, -44, -23, 1, -32, -26, -14, -3, -26, -21, 14, -19, 13, -8, -5, 5, -11, -15, -12, 4, -51, -20, -28, -34, 3, -7, -23, 7, -15, 11, -33, -11, -21, -20, -14, 0, -8, 6, 11, -42, -14, 6, -26, -4, 12, -27, 25, 4, -29, -40}
, {-24, 13, -42, -8, -56, -6, -26, -11, -32, -11, 12, 11, -29, -22, -54, -11, -49, -30, -65, -60, 10, -12, 21, -17, -57, -8, -95, 12, 10, 22, 2, -2, -24, -40, -17, -12, -4, -39, -30, -11, 7, -7, -22, 7, -39, -40, -6, -12, 15, -11, -14, -9, -47, 14, -26, -60, -73, 10, 4, -42, -8, -29, -63, -23, -51, -6, -1, -21, 13, -3, -34, -7, 3, -14, -29, 3, -29, -122, -36, -98, -40, -28, -54, -31, -38, 9, -29, -25, 12, -23, -27, -47, -28, -95, -19, -33, 1, 6, -92, -102, 12, -30, -2, 4, -10, -4, -29, -1, -38, -16, -14, 1, -2, -25, -11, -17, -38, 8, -4, 4, 13, -18, -13, -26, 7, 8, -4, 6}
, {4, 5, 22, 11, -58, 26, 12, -16, 19, 76, -9, -78, 18, -114, 36, 24, -134, -4, -25, 4, 1, -11, 10, 38, -45, 14, -3, 22, -30, 15, -13, 12, -104, 4, -107, 27, 16, -31, 5, 4, 65, 14, -13, -1, -28, 95, -4, 21, -25, 37, 28, -52, -163, 1, 15, -125, -12, -10, -28, 29, 4, -31, -40, -10, -33, 12, -55, -40, 3, 15, 67, -73, 31, -6, -13, -15, -9, 15, -43, 43, -30, -1, -24, 51, -11, 5, 23, 27, 26, -42, 14, -5, 56, 32, 15, 10, 10, -50, 1, -103, -63, -8, -25, -12, -18, -19, -6, 0, -5, 13, -35, -69, -2, -3, -7, -1, 0, 19, -144, 14, -12, -81, 1, -28, -129, 25, 6, 19}
, {-15, 57, 64, 65, -15, -15, 32, 51, -19, -138, -5, -13, 32, -162, -66, 25, -34, -19, -58, 1, -14, 24, -23, 36, 10, 8, 41, 11, 7, 6, 9, -8, -5, 42, 67, 20, 9, -32, 20, 17, 68, -10, -23, -10, 61, 64, 19, 15, -1, 27, -5, -4, -38, -13, 15, 4, -42, 27, 3, 79, 29, 25, -15, 26, 3, -20, 7, 80, -7, 22, 56, -49, -58, -9, 41, -67, 9, 19, -76, -53, -68, 16, 8, -37, -65, 20, -15, 20, -15, -28, 9, 3, -6, -18, -1, -1, 11, 28, -36, 4, -48, -32, -3, -19, 8, 5, -71, -6, -57, -15, -2, -31, -2, -18, 7, -8, -95, -21, -73, -17, 14, -68, 22, 11, -45, -15, 23, 44}
, {4, 11, -30, -18, -27, -1, 4, -10, -13, -8, -11, -24, 7, 7, -29, -25, -36, -20, -20, -24, -16, -1, -22, -3, -19, -3, -11, 3, 4, -32, 1, -62, 17, -8, -4, 11, 10, -27, -19, -15, -7, -24, -8, -27, -3, 2, 10, 1, -8, -22, 18, 12, -34, 4, -16, -44, -21, -6, -26, -40, -4, 16, -55, -4, -62, 22, -46, 5, -19, 20, -13, 6, 12, -3, 1, -18, 2, -22, -15, -18, -18, -3, -41, -68, -40, 11, 19, -28, 0, 1, 9, -19, -28, -23, -1, -4, -19, -5, -32, -42, -14, -15, -22, 0, -4, 17, -44, 1, -17, -17, -10, 3, -13, -27, 20, 21, -27, -29, -12, 2, -4, -24, 14, -24, -19, -20, -15, 4}
, {12, -22, -3, 7, -10, 17, -2, -24, -18, -7, -10, -11, -1, 10, -25, 20, -12, 17, -42, -27, -19, 17, -48, -17, -6, 21, -8, -23, 2, -36, -13, -23, -15, -11, -10, -8, -15, -29, -7, -10, 15, -10, 24, 13, -35, -19, -10, 3, -18, 2, 20, -3, -13, -7, -12, -27, -6, 14, -32, 7, 5, 13, -5, -5, -50, -14, -14, -17, -21, -1, -3, -55, -18, 14, -25, 17, -11, -7, 12, -18, -18, 0, -32, -19, -34, 8, -5, -18, -20, -33, -14, -35, -20, -8, -22, -18, 7, 7, -9, -12, -6, -10, 8, 20, -3, -1, -23, 21, -15, -59, -47, 10, 12, -30, -17, -5, 4, -10, 11, 12, -19, -1, 18, -21, -6, -19, -27, -17}
, {38, -16, 24, 3, -2, 18, 27, -132, -15, -31, 22, -19, -34, -94, -23, -4, 20, 15, -34, -56, 25, 8, -63, -71, -6, -37, -49, 19, 48, 38, -10, -68, -38, 38, 26, 14, 29, -22, -14, -5, -22, -7, -55, 7, -20, -89, -14, 8, -2, -43, -11, 50, 13, 10, -13, 16, -11, 21, 15, 7, 17, 13, -56, 2, -84, -6, 8, -47, 10, 22, 5, -19, -86, 22, -15, -113, 17, 1, -70, -28, -37, -6, 7, -74, -89, 18, 3, 7, 14, 12, 11, -107, -15, -88, -4, 27, 17, 8, -50, -49, -7, -4, 5, 34, -9, -19, -52, 24, -69, -18, -89, -70, -2, -31, 3, 0, -69, -100, -24, 20, 8, -11, 30, -9, -38, 21, -9, 1}
, {0, 30, -28, -51, -12, -64, -9, -111, -77, -43, -7, 87, 2, -43, 7, -20, 23, -13, 19, -57, -27, -28, 5, 17, -51, -17, -37, 9, -32, 53, 36, 25, 80, 17, -116, -11, -57, -125, 17, -55, -18, 8, 18, -26, -24, 12, 4, 18, 18, -43, 10, -44, 13, 0, 5, -83, -31, -15, 41, -87, -19, 14, -35, 1, -22, 1, -1, 7, -8, -27, -26, -34, 35, -26, -36, -7, 7, -99, -40, -70, -91, -6, 0, 13, 8, 12, 9, 7, -23, 2, -20, -18, 79, 47, 3, 42, 3, -55, 25, 46, -25, 11, 27, 14, 15, 6, 68, 9, -25, 6, -131, 97, 3, -53, -11, -27, -98, -132, 20, -10, -9, -81, -93, -11, -77, -4, -9, -56}
, {-5, -12, -70, -17, -48, -7, 18, 18, -35, -11, 18, -64, -27, -3, -33, -26, -18, 5, -16, 9, -9, -1, -44, 17, -80, -3, -73, 8, 27, -71, -2, -21, -15, -30, 9, -26, -16, -59, 16, -8, -83, -11, -5, -11, -49, -18, 0, -8, 21, -50, -1, -14, -9, -3, 8, -40, -18, -15, -100, -48, -10, 16, -14, -3, -35, 8, -86, 36, -30, -20, 28, 10, -20, -17, 19, -24, 4, -21, -16, -53, -46, 18, -27, -88, -7, -16, -12, -8, -16, -89, -12, -4, -5, -45, -3, -28, -26, 15, -28, -34, -16, -7, -24, -20, 12, -6, -42, -6, -33, -49, -61, -11, 24, -41, 26, 6, 21, -49, 9, -17, 7, -137, -26, -85, 1, 4, -9, -17}
, {-6, -11, 12, -46, 64, -25, -18, -43, -36, 10, 9, -103, 8, -47, -65, -6, 49, -43, -64, 11, 12, -25, -79, -59, -9, -10, -51, 0, 2, 3, -6, 26, -63, -38, -25, 0, -73, -93, -27, -7, -14, -13, -21, -18, -51, -59, 7, -14, 9, -33, -30, 41, 1, -30, 18, -84, -33, -30, -27, -37, 4, -2, -85, -3, 29, -8, -47, -40, -13, -14, 32, -87, 40, -1, -38, -103, 4, -53, -20, -51, -52, 2, -40, 19, -46, -22, 0, -21, -27, 38, -9, -32, -30, -39, -5, 17, 14, 24, -64, 29, -30, -15, 7, 19, -20, -20, -68, 7, -88, -85, 23, 74, -29, -30, -20, -21, -42, -45, -106, 3, -20, -118, 60, -55, 35, -28, -14, -67}
, {-13, -6, 7, -30, -13, 21, 23, -116, -35, -5, -7, 14, -41, 0, -13, 18, 42, -13, -4, -19, 19, 6, 11, 51, 53, 41, 40, 6, 3, 24, -28, -58, 8, -24, -75, 14, -17, 19, -19, -36, 5, 16, 9, 17, 13, -130, -22, -53, -2, -78, 17, 84, -15, -15, 13, -31, 82, -2, 69, 21, -6, -24, -64, 8, -15, 3, -22, 7, 7, -13, 41, -93, -8, 9, 13, -13, -13, -60, 29, 52, 36, 2, 23, -48, 6, 11, 4, 16, 21, -4, 7, 34, 49, -50, -39, -7, 8, 29, -44, -67, 119, 6, -26, 10, 30, -5, -3, -19, -147, -61, -6, 22, -6, 19, 29, -6, 62, -45, -143, -4, 12, 51, 80, 2, -14, -1, 9, 51}
, {69, -16, -28, 14, 1, -164, -18, -22, -26, 91, 8, 61, -60, -17, 34, 12, -64, -1, 20, 26, -28, -28, 64, 28, 38, -38, -67, 18, 114, 12, -32, -61, 2, 31, -58, 1, 5, -31, 19, -5, 0, 13, 22, -24, -1, -23, 10, -6, 14, -1, 16, 44, 31, -8, 15, -57, 35, 6, -19, -14, 10, 14, -63, -10, -14, 10, -114, -74, -4, -7, -50, 18, -18, 3, -117, -35, -2, -21, -77, 52, -4, -20, 52, -50, -5, -21, -1, -23, 24, -68, 4, -84, -50, 24, -17, -3, -27, 16, -7, 57, -55, -24, -15, 38, 7, -12, 17, -1, -46, 18, -50, 59, -14, 19, 49, -23, 52, -49, -69, 4, -8, -38, 58, -8, 81, -25, -11, 13}
, {-16, -3, -30, 25, -118, -34, 13, -25, -35, -52, 6, -38, -19, -22, 14, -23, -40, -8, -81, -41, 9, 11, -49, -33, 70, -22, 18, -4, -46, 62, 3, -8, 47, -39, -81, 10, -47, -29, -20, 13, -74, 4, -56, -19, -16, -27, 3, 3, -8, -53, -20, -97, -43, -14, 11, -68, 131, -23, -24, -11, -3, -12, -77, -10, -91, 19, -11, -15, -2, -25, -19, 8, -95, -8, -3, -103, 3, -38, -56, -112, -78, 8, -16, 13, -138, -9, 16, -28, 8, -3, -32, -77, 4, -7, 8, -34, -3, -15, -63, 3, 14, 6, 5, 6, 16, 16, -57, -8, -159, -65, -10, 14, -5, 8, -1, -12, -35, -29, -22, 6, -2, -54, -7, 8, -105, -3, -30, -68}
, {-7, 32, 68, -37, -69, 14, -8, 19, 82, 40, 26, -18, -5, 58, 22, 17, -15, -19, 10, -46, 9, 29, -45, -133, 40, -4, -38, -6, 127, -98, -29, 71, -16, -45, -103, 21, 23, 7, -16, -1, -17, 12, 78, -3, 21, -113, -8, -17, -1, -96, 22, -52, 26, 1, -34, -84, -51, -11, 40, 18, -15, 78, -7, 18, 5, -10, 26, -4, 8, -16, 4, -51, -48, -17, 10, 28, -7, -20, -50, 16, 40, -11, -3, -142, 37, 7, 1, -14, -6, -68, -3, -65, -23, -1, 4, -60, 23, -28, 9, -5, -59, 8, -20, 9, 10, -28, 45, -2, -55, -27, -83, -16, 3, -64, -14, 2, -54, 18, -118, -7, -21, -33, 17, 34, 72, 0, 16, 0}
, {-10, 21, -32, -79, -63, -23, -8, -36, -69, -101, 17, -90, 15, -94, 70, -18, -35, -12, -50, 11, -26, 2, 27, 6, 12, 25, -34, -27, -66, 43, 35, 33, -61, -29, 97, -29, 89, 20, -7, 7, -41, -4, -30, -22, 11, -39, -11, 14, -4, 5, 10, 84, -60, 10, -10, 10, -56, -22, -25, -50, -4, 23, -1, -23, -3, 9, -8, 42, -5, 26, -55, 1, 5, 24, 64, 20, -21, 30, -32, 60, 31, 8, 16, -35, 7, -19, 3, -15, -23, -42, -30, 19, -11, 11, -21, -56, -29, -48, 3, -7, 66, -4, -26, -18, -1, -10, -57, 3, 26, 8, -29, -92, 10, -3, -38, 10, 68, -22, 58, -8, -14, -31, 27, 12, -10, -21, -6, 68}
, {-21, -20, 7, 10, -84, -11, 12, -16, -67, 86, -5, -157, 7, -93, 35, -26, -61, -6, -19, -2, -21, -19, 10, 5, -24, -19, -11, -2, -1, -55, -2, -45, 39, -69, -11, -9, -14, -112, -30, 40, -42, -23, -28, -22, -72, 0, 3, -24, 5, 2, 7, -19, -61, 2, -3, -29, 41, -10, 6, 4, -17, -24, 13, -24, -90, 13, -72, -25, 7, -31, -21, -20, 7, -13, -37, -89, 4, -93, -47, -55, -40, -1, -57, -84, -76, -16, 15, -13, -19, 5, -25, -94, -19, -11, 9, 39, 7, -22, -25, -96, -8, -19, 3, 10, -24, -24, -49, -2, -57, -12, -61, -130, -17, -49, -24, -22, -35, 1, 12, 5, -18, -57, 12, -46, -54, -24, -11, -42}
, {-21, 17, -94, -23, -52, 8, 0, -14, -33, 111, 6, -78, -16, -29, -25, -17, -78, -14, -115, 8, -1, -4, -58, -46, -19, -3, 29, 7, -48, 67, 14, -7, -68, 9, -32, 12, 31, -45, -3, -1, -17, -27, -38, 7, 5, -21, -4, -14, -3, -58, 1, -75, -105, -8, 9, -54, 61, 14, -57, -10, 4, -2, -53, -8, -65, 12, -34, -23, -18, 14, -66, 21, -12, 22, 15, -94, -11, -19, -61, 10, -128, 14, -2, -11, -56, -2, 8, -14, -9, 35, -27, 44, -19, -86, -26, 41, -15, 11, -84, -13, 12, -21, 5, -12, -1, -4, -98, -18, -23, -25, -84, -20, 3, -42, -3, -7, -47, -15, -61, -23, -13, -10, -28, -44, -48, -10, 6, -123}
, {54, -51, -30, -15, -25, 23, -31, -24, -26, 23, -18, -51, 3, -34, 15, 4, 32, 20, -14, -5, 25, 14, 21, 12, 2, 4, -70, -12, -67, 56, -4, -19, 96, 3, -19, -17, 34, 17, -14, -23, 34, 2, 73, 1, -22, 23, -1, 7, -56, -59, -21, -42, 22, 14, 1, -51, 9, 19, -55, 49, -8, 22, -26, -7, -8, 17, 39, -139, -10, -16, -7, -189, 57, -1, -69, -62, -30, -39, -53, 52, 12, -11, 19, 49, -160, 16, 10, 10, -3, 6, -4, 81, -36, 35, -5, -27, 12, 34, 14, -5, 102, 13, 95, 35, 4, 15, 12, 12, -82, -21, -30, 37, -8, -85, 13, 11, 84, -45, -65, 30, 10, -24, -110, -17, -4, 22, 8, -27}
, {0, -37, -181, 75, 39, -43, 50, 69, -43, -2, 18, -5, 37, -55, 34, -2, -50, 37, 33, -44, 26, 22, -49, 51, -74, -38, 61, 14, 24, -5, 5, 21, -90, -101, -7, -27, -1, -6, -4, 22, -114, 10, 0, -8, -45, 47, 5, -40, -11, 26, 24, 89, -56, 24, 3, -46, -24, -3, 19, 28, 19, -7, 12, 29, -65, 4, -55, -32, 10, 13, -89, -10, 5, 0, -14, 10, 27, 12, 4, -29, 27, -3, -29, 23, 0, -17, 16, 24, 5, 50, 20, -78, 8, 44, 15, -66, 25, 14, 22, 29, -4, 5, -5, 13, 14, 3, 16, 5, 0, 67, 41, -86, 13, -60, -21, -6, -145, 33, 9, 9, -6, 25, 165, 50, 148, -12, 21, 21}
, {2, 4, -43, -7, -58, -13, 9, -25, -1, 0, -20, -54, 14, -36, 12, -1, -57, -4, -112, -38, -11, -17, -4, -23, 11, 6, -4, -16, -4, -10, -13, 20, 13, -26, -7, -14, 13, 18, -22, 22, 20, 8, -61, -6, -18, -20, -8, -24, -7, -24, -28, -27, -14, -23, 14, -45, 138, -13, 3, 16, -20, -12, -48, 4, -24, 1, 0, -28, 6, 12, -39, -34, -22, 3, -3, 46, -26, -62, -41, -15, -4, -24, 76, -10, -17, -5, -8, 3, 7, -8, -13, -77, -7, -13, -18, -30, -9, -44, -54, -16, -2, -11, -48, 14, -13, 13, -19, -27, -44, -89, -74, -84, -6, -14, -26, -15, -46, -78, -52, -7, 0, 42, -19, 14, -14, 4, 8, 0}
, {16, -4, -53, 31, 7, -24, 27, 13, 6, -2, -13, 54, -25, -67, 55, 1, -84, 7, 41, -42, 16, -6, -10, 41, -36, -2, -48, 4, 54, -23, -7, 54, 21, -41, -50, -21, -8, 19, 30, 6, -94, 6, 38, 21, -84, 62, -12, -15, -4, -6, -12, 78, 56, 22, -2, -18, -60, 19, -8, 54, -12, 15, -38, 20, -18, -4, -27, 22, -10, 14, 42, -52, -23, 5, -54, 23, 1, 11, -52, -4, 21, 3, 76, -2, 7, -1, 18, 1, 25, -75, 2, -61, 51, 21, 13, 44, -4, 30, -16, 12, 27, 4, 18, 30, 51, 9, 14, 1, 2, 48, -42, 3, 27, -97, -2, 26, -35, -9, -51, 17, 19, -99, 61, -1, 145, 17, 10, 28}
, {40, 14, 63, -88, -73, -40, 5, -100, 25, 48, 6, -142, 22, -75, 21, 8, -46, -5, 6, -17, 0, 14, 57, -67, 18, -5, 23, -13, 93, -79, -8, 10, -8, 29, 27, -7, 107, -46, 13, -11, 61, 5, 46, 26, 31, 5, -4, 5, 9, -7, -3, 20, 8, -7, 13, -97, -41, 10, -1, -48, 6, -4, -27, -17, -25, 22, -11, 41, -22, 11, -89, -74, 56, -1, 3, -36, -25, -34, -113, 25, -13, -25, -21, -23, -9, 26, -6, -21, -11, -93, -20, -65, 32, 21, 3, -38, 8, -44, -4, -19, -84, 4, -19, -3, 7, 4, 85, -4, -76, -9, -27, -15, -17, -31, -1, 1, -2, -32, -102, 19, 5, -36, 3, 56, 9, 11, 7, 28}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS