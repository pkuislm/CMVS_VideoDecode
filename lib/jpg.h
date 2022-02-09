#pragma once

#ifndef JPG_H
#define JPG_H

#include <cmath>
#include <string>
#include <vector>

typedef unsigned char byte;
typedef unsigned int uint;

// Start of Frame markers, non-differential, Huffman coding
const byte SOF0 = 0xC0; // Baseline DCT

// Define Huffman Table(s)
const byte DHT = 0xC4;

// Other Markers
const byte SOI = 0xD8; // Start of Image
const byte EOI = 0xD9; // End of Image
const byte SOS = 0xDA; // Start of Scan
const byte DQT = 0xDB; // Define Quantization Table(s)

// APPN Markers
const byte APP0 = 0xE0;

struct QuantizationTable {
    uint table[64] = { 0 };
    bool set = false;
};

struct HuffmanTable {
    byte offsets[17] = { 0 };
    byte symbols[176] = { 0 };
    uint codes[176] = { 0 };
    bool set = false;
};

struct Block {
    union {
        int y[64] = { 0 };
        int r[64];
    };
    union {
        int cb[64] = { 0 };
        int g [64];
    };
    union {
        int cr[64] = { 0 };
        int b [64];
    };
    int* operator[](uint i) {
        switch (i) {
            case 0:
                return y;
            case 1:
                return cb;
            case 2:
                return cr;
            default:
                return nullptr;
        }
    }
};

struct BMPImage {
    uint height = 0;
    uint width = 0;

    Block* blocks = nullptr;

    uint blockHeight = 0;
    uint blockWidth = 0;
};

const byte zigZagMap[] = {
        0,   1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63
};

// IDCT scaling factors
const float m0 = 2.0 * std::cos(1.0 / 16.0 * 2.0 * M_PI);
const float m1 = 2.0 * std::cos(2.0 / 16.0 * 2.0 * M_PI);
const float m3 = 2.0 * std::cos(2.0 / 16.0 * 2.0 * M_PI);
const float m5 = 2.0 * std::cos(3.0 / 16.0 * 2.0 * M_PI);
const float m2 = m0 - m5;
const float m4 = m0 + m5;

const float s0 = std::cos(0.0 / 16.0 * M_PI) / std::sqrt(8);
const float s1 = std::cos(1.0 / 16.0 * M_PI) / 2.0;
const float s2 = std::cos(2.0 / 16.0 * M_PI) / 2.0;
const float s3 = std::cos(3.0 / 16.0 * M_PI) / 2.0;
const float s4 = std::cos(4.0 / 16.0 * M_PI) / 2.0;
const float s5 = std::cos(5.0 / 16.0 * M_PI) / 2.0;
const float s6 = std::cos(6.0 / 16.0 * M_PI) / 2.0;
const float s7 = std::cos(7.0 / 16.0 * M_PI) / 2.0;

const QuantizationTable qTableY100 = {
        {
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1
        },
        true
};

const QuantizationTable qTableCbCr100 = {
        {
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1
        },
        true
};

const QuantizationTable* const qTables100[] = { &qTableY100, &qTableCbCr100, &qTableCbCr100 };

void RGBToYCbCr(const BMPImage& image);
void forwardDCT(const BMPImage& image);
void quantize(const BMPImage& image);
void writeJPG(const BMPImage& image, std::vector<byte>& buff);

#endif
