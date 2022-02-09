//
// Created by Pkuism on 2022/2/9.
//
#include "jpg.h"
#include <vector>
#include <iostream>

HuffmanTable hDCTableY = {
        { 0, 0, 1, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12 },
        { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b },
        {},
        false
};

HuffmanTable hDCTableCbCr = {
        { 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12 },
        { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b },
        {},
        false
};

HuffmanTable hACTableY = {
        { 0, 0, 2, 3, 6, 9, 11, 15, 18, 23, 28, 32, 36, 36, 36, 37, 162 },
        {
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
                0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
                0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
                0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
                0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
                0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
                0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
                0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
                0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
                0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
                0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
                0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
                0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
                0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
                0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                0xf9, 0xfa
        },
        {},
        false
};

HuffmanTable hACTableCbCr = {
        { 0, 0, 2, 3, 5, 9, 13, 16, 20, 27, 32, 36, 40, 40, 41, 43, 162 },
        {
                0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
                0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
                0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
                0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
                0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
                0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
                0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
                0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
                0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
                0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
                0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
                0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
                0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
                0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                0xf9, 0xfa
        },
        {},
        false
};

HuffmanTable* const dcTables[] = { &hDCTableY, &hDCTableCbCr, &hDCTableCbCr };
HuffmanTable* const acTables[] = { &hACTableY, &hACTableCbCr, &hACTableCbCr };

// convert all pixels in a block from RGB color space to YCbCr
void RGBToYCbCrBlock(Block& block) {
    for (uint cy = 0; cy < 8; ++cy) {
        for (uint x = 0; x < 8; ++x) {
            const uint pixel = cy * 8 + x;
            int y  =  0.2990 * block.r[pixel] + 0.5870 * block.g[pixel] + 0.1140 * block.b[pixel] - 128;
            int cb = -0.1687 * block.r[pixel] - 0.3313 * block.g[pixel] + 0.5000 * block.b[pixel];
            int cr =  0.5000 * block.r[pixel] - 0.4187 * block.g[pixel] - 0.0813 * block.b[pixel];
            if (y  < -128) y  = -128;
            if (y  >  127) y  =  127;
            if (cb < -128) cb = -128;
            if (cb >  127) cb =  127;
            if (cr < -128) cr = -128;
            if (cr >  127) cr =  127;
            block.y[pixel]  = y;
            block.cb[pixel] = cb;
            block.cr[pixel] = cr;
        }
    }
}

// convert all pixels from RGB color space to YCbCr
void RGBToYCbCr(const BMPImage& image) {
    for (uint y = 0; y < image.blockHeight; ++y) {
        for (uint x = 0; x < image.blockWidth; ++x) {
            RGBToYCbCrBlock(image.blocks[y * image.blockWidth + x]);
        }
    }
}

// perform 1-D FDCT on all columns and rows of a block component
//   resulting in 2-D FDCT
void forwardDCTBlockComponent(int* const component) {
    for (uint i = 0; i < 8; ++i) {
        const float a0 = component[0 * 8 + i];
        const float a1 = component[1 * 8 + i];
        const float a2 = component[2 * 8 + i];
        const float a3 = component[3 * 8 + i];
        const float a4 = component[4 * 8 + i];
        const float a5 = component[5 * 8 + i];
        const float a6 = component[6 * 8 + i];
        const float a7 = component[7 * 8 + i];

        const float b0 = a0 + a7;
        const float b1 = a1 + a6;
        const float b2 = a2 + a5;
        const float b3 = a3 + a4;
        const float b4 = a3 - a4;
        const float b5 = a2 - a5;
        const float b6 = a1 - a6;
        const float b7 = a0 - a7;

        const float c0 = b0 + b3;
        const float c1 = b1 + b2;
        const float c2 = b1 - b2;
        const float c3 = b0 - b3;
        const float c4 = b4;
        const float c5 = b5 - b4;
        const float c6 = b6 - c5;
        const float c7 = b7 - b6;

        const float d0 = c0 + c1;
        const float d1 = c0 - c1;
        const float d2 = c2;
        const float d3 = c3 - c2;
        const float d4 = c4;
        const float d5 = c5;
        const float d6 = c6;
        const float d7 = c5 + c7;
        const float d8 = c4 - c6;

        const float e0 = d0;
        const float e1 = d1;
        const float e2 = d2 * m1;
        const float e3 = d3;
        const float e4 = d4 * m2;
        const float e5 = d5 * m3;
        const float e6 = d6 * m4;
        const float e7 = d7;
        const float e8 = d8 * m5;

        const float f0 = e0;
        const float f1 = e1;
        const float f2 = e2 + e3;
        const float f3 = e3 - e2;
        const float f4 = e4 + e8;
        const float f5 = e5 + e7;
        const float f6 = e6 + e8;
        const float f7 = e7 - e5;

        const float g0 = f0;
        const float g1 = f1;
        const float g2 = f2;
        const float g3 = f3;
        const float g4 = f4 + f7;
        const float g5 = f5 + f6;
        const float g6 = f5 - f6;
        const float g7 = f7 - f4;

        component[0 * 8 + i] = g0 * s0;
        component[4 * 8 + i] = g1 * s4;
        component[2 * 8 + i] = g2 * s2;
        component[6 * 8 + i] = g3 * s6;
        component[5 * 8 + i] = g4 * s5;
        component[1 * 8 + i] = g5 * s1;
        component[7 * 8 + i] = g6 * s7;
        component[3 * 8 + i] = g7 * s3;
    }
    for (uint i = 0; i < 8; ++i) {
        const float a0 = component[i * 8 + 0];
        const float a1 = component[i * 8 + 1];
        const float a2 = component[i * 8 + 2];
        const float a3 = component[i * 8 + 3];
        const float a4 = component[i * 8 + 4];
        const float a5 = component[i * 8 + 5];
        const float a6 = component[i * 8 + 6];
        const float a7 = component[i * 8 + 7];

        const float b0 = a0 + a7;
        const float b1 = a1 + a6;
        const float b2 = a2 + a5;
        const float b3 = a3 + a4;
        const float b4 = a3 - a4;
        const float b5 = a2 - a5;
        const float b6 = a1 - a6;
        const float b7 = a0 - a7;

        const float c0 = b0 + b3;
        const float c1 = b1 + b2;
        const float c2 = b1 - b2;
        const float c3 = b0 - b3;
        const float c4 = b4;
        const float c5 = b5 - b4;
        const float c6 = b6 - c5;
        const float c7 = b7 - b6;

        const float d0 = c0 + c1;
        const float d1 = c0 - c1;
        const float d2 = c2;
        const float d3 = c3 - c2;
        const float d4 = c4;
        const float d5 = c5;
        const float d6 = c6;
        const float d7 = c5 + c7;
        const float d8 = c4 - c6;

        const float e0 = d0;
        const float e1 = d1;
        const float e2 = d2 * m1;
        const float e3 = d3;
        const float e4 = d4 * m2;
        const float e5 = d5 * m3;
        const float e6 = d6 * m4;
        const float e7 = d7;
        const float e8 = d8 * m5;

        const float f0 = e0;
        const float f1 = e1;
        const float f2 = e2 + e3;
        const float f3 = e3 - e2;
        const float f4 = e4 + e8;
        const float f5 = e5 + e7;
        const float f6 = e6 + e8;
        const float f7 = e7 - e5;

        const float g0 = f0;
        const float g1 = f1;
        const float g2 = f2;
        const float g3 = f3;
        const float g4 = f4 + f7;
        const float g5 = f5 + f6;
        const float g6 = f5 - f6;
        const float g7 = f7 - f4;

        component[i * 8 + 0] = g0 * s0;
        component[i * 8 + 4] = g1 * s4;
        component[i * 8 + 2] = g2 * s2;
        component[i * 8 + 6] = g3 * s6;
        component[i * 8 + 5] = g4 * s5;
        component[i * 8 + 1] = g5 * s1;
        component[i * 8 + 7] = g6 * s7;
        component[i * 8 + 3] = g7 * s3;
    }
}

// perform FDCT on all MCUs
void forwardDCT(const BMPImage& image) {
    for (uint y = 0; y < image.blockHeight; ++y) {
        for (uint x = 0; x < image.blockWidth; ++x) {
            for (uint i = 0; i < 3; ++i) {
                forwardDCTBlockComponent(image.blocks[y * image.blockWidth + x][i]);
            }
        }
    }
}

// quantize a block component based on a quantization table
void quantizeBlockComponent(const QuantizationTable& qTable, int* const component) {
    for (uint i = 0; i < 64; ++i) {
        component[i] /= (signed)qTable.table[i];
    }
}

// quantize all MCUs
void quantize(const BMPImage& image) {
    for (uint y = 0; y < image.blockHeight; ++y) {
        for (uint x = 0; x < image.blockWidth; ++x) {
            for (uint i = 0; i < 3; ++i) {
                quantizeBlockComponent(*qTables100[i], image.blocks[y * image.blockWidth + x][i]);
            }
        }
    }
}

class BitWriter {
private:
    byte nextBit = 0;
    std::vector<byte>& data;

public:
    BitWriter(std::vector<byte>& d) :
            data(d)
    {}

    void writeBit(uint bit) {
        if (nextBit == 0) {
            data.push_back(0);
        }
        data.back() |= (bit & 1) << (7 - nextBit);
        nextBit = (nextBit + 1) % 8;
        if (nextBit == 0 && data.back() == 0xFF) {
            data.push_back(0);
        }
    }

    void writeBits(uint bits, uint length) {
        for (uint i = 1; i <= length; ++i) {
            writeBit(bits >> (length - i));
        }
    }
};

// generate all Huffman codes based on symbols from a Huffman table
void generateCodes(HuffmanTable& hTable) {
    uint code = 0;
    for (uint i = 0; i < 16; ++i) {
        for (uint j = hTable.offsets[i]; j < hTable.offsets[i + 1]; ++j) {
            hTable.codes[j] = code;
            code += 1;
        }
        code <<= 1;
    }
}

uint bitLength(int v) {
    uint length = 0;
    while (v > 0) {
        v >>= 1;
        length += 1;
    }
    return length;
}

bool getCode(const HuffmanTable& hTable, byte symbol, uint& code, uint& codeLength) {
    for (uint i = 0; i < 16; ++i) {
        for (uint j = hTable.offsets[i]; j < hTable.offsets[i + 1]; ++j) {
            if (symbol == hTable.symbols[j]) {
                code = hTable.codes[j];
                codeLength = i + 1;
                return true;
            }
        }
    }
    return false;
}

bool encodeBlockComponent(
        BitWriter& bitWriter,
        int* const component,
        int& previousDC,
        const HuffmanTable& dcTable,
        const HuffmanTable& acTable
) {
    // encode DC value
    int coeff = component[0] - previousDC;
    previousDC = component[0];

    uint coeffLength = bitLength(std::abs(coeff));
    if (coeffLength > 11) {
        std::cout << "Error - DC coefficient length greater than 11\n";
        return false;
    }
    if (coeff < 0) {
        coeff += (1 << coeffLength) - 1;
    }

    uint code = 0;
    uint codeLength = 0;
    if (!getCode(dcTable, coeffLength, code, codeLength)) {
        std::cout << "Error - Invalid DC value\n";
        return false;
    }
    bitWriter.writeBits(code, codeLength);
    bitWriter.writeBits(coeff, coeffLength);

    // encode AC values
    for (uint i = 1; i < 64; ++i) {
        // find zero run length
        byte numZeroes = 0;
        while (i < 64 && component[zigZagMap[i]] == 0) {
            numZeroes += 1;
            i += 1;
        }

        if (i == 64) {
            if (!getCode(acTable, 0x00, code, codeLength)) {
                std::cout << "Error - Invalid AC value\n";
                return false;
            }
            bitWriter.writeBits(code, codeLength);
            return true;
        }

        while (numZeroes >= 16) {
            if (!getCode(acTable, 0xF0, code, codeLength)) {
                std::cout << "Error - Invalid AC value\n";
                return false;
            }
            bitWriter.writeBits(code, codeLength);
            numZeroes -= 16;
        }

        // find coeff length
        coeff = component[zigZagMap[i]];
        coeffLength = bitLength(std::abs(coeff));
        if (coeffLength > 10) {
            std::cout << "Error - AC coefficient length greater than 10\n";
            return false;
        }
        if (coeff < 0) {
            coeff += (1 << coeffLength) - 1;
        }

        // find symbol in table
        byte symbol = numZeroes << 4 | coeffLength;
        if (!getCode(acTable, symbol, code, codeLength)) {
            std::cout << "Error - Invalid AC value\n";
            return false;
        }
        bitWriter.writeBits(code, codeLength);
        bitWriter.writeBits(coeff, coeffLength);
    }

    return true;
}

// encode all the Huffman data from all MCUs
std::vector<byte> encodeHuffmanData(const BMPImage& image) {
    std::vector<byte> huffmanData;
    BitWriter bitWriter(huffmanData);

    int previousDCs[3] = { 0 };

    for (uint i = 0; i < 3; ++i) {
        if (!dcTables[i]->set) {
            generateCodes(*dcTables[i]);
            dcTables[i]->set = true;
        }
        if (!acTables[i]->set) {
            generateCodes(*acTables[i]);
            acTables[i]->set = true;
        }
    }

    for (uint y = 0; y < image.blockHeight; ++y) {
        for (uint x = 0; x < image.blockWidth; ++x) {
            for (uint i = 0; i < 3; ++i) {
                if (!encodeBlockComponent(
                        bitWriter,
                        image.blocks[y * image.blockWidth + x][i],
                        previousDCs[i],
                        *dcTables[i],
                        *acTables[i])) {
                    return std::vector<byte>();
                }
            }
        }
    }

    return huffmanData;
}

// helper function to write a 2-byte short integer in big-endian
void putShort(std::vector<byte>& buff, const uint v) {
    buff.push_back((v >> 8) & 0xFF);
    buff.push_back((v >> 0) & 0xFF);
}

void writeQuantizationTable(std::vector<byte>& buff, byte tableID, const QuantizationTable& qTable) {
    buff.push_back(0xFF);
    buff.push_back(DQT);
    putShort(buff, 67);
    buff.push_back(tableID);
    for (unsigned char i : zigZagMap) {
        buff.push_back(qTable.table[i]);
    }
}

void writeStartOfFrame(std::vector<byte>& buff, const BMPImage& image) {
    buff.push_back(0xFF);
    buff.push_back(SOF0);
    putShort(buff, 17);
    buff.push_back(8);
    putShort(buff, image.height);
    putShort(buff, image.width);
    buff.push_back(3);
    for (uint i = 1; i <= 3; ++i) {
        buff.push_back(i);
        buff.push_back(0x11);
        buff.push_back(i == 1 ? 0 : 1);
    }
}

void writeHuffmanTable(std::vector<byte>& buff, byte acdc, byte tableID, const HuffmanTable& hTable) {
    buff.push_back(0xFF);
    buff.push_back(DHT);
    putShort(buff, 19 + hTable.offsets[16]);
    buff.push_back(acdc << 4 | tableID);
    for (uint i = 0; i < 16; ++i) {
        buff.push_back(hTable.offsets[i + 1] - hTable.offsets[i]);
    }
    for (uint i = 0; i < 16; ++i) {
        for (uint j = hTable.offsets[i]; j < hTable.offsets[i + 1]; ++j) {
            buff.push_back(hTable.symbols[j]);
        }
    }
}

void writeStartOfScan(std::vector<byte>& buff) {
    buff.push_back(0xFF);
    buff.push_back(SOS);
    putShort(buff, 12);
    buff.push_back(3);
    for (uint i = 1; i <= 3; ++i) {
        buff.push_back(i);
        buff.push_back(i == 1 ? 0x00 : 0x11);
    }
    buff.push_back(0);
    buff.push_back(63);
    buff.push_back(0);
}

void writeAPP0(std::vector<byte>& buff) {
    buff.push_back(0xFF);
    buff.push_back(APP0);
    putShort(buff, 16);
    buff.push_back('J');
    buff.push_back('F');
    buff.push_back('I');
    buff.push_back('F');
    buff.push_back(0);
    buff.push_back(1);
    buff.push_back(2);
    buff.push_back(0);
    putShort(buff, 100);
    putShort(buff, 100);
    buff.push_back(0);
    buff.push_back(0);
}

void writeJPG(const BMPImage& image, std::vector<byte>& buff) {
    std::vector<byte> huffmanData = encodeHuffmanData(image);
    if (huffmanData.size() == 0) {
        return;
    }

    // SOI
    buff.push_back(0xFF);
    buff.push_back(SOI);

    // APP0
    writeAPP0(buff);

    // DQT
    writeQuantizationTable(buff, 0, qTableY100);
    writeQuantizationTable(buff, 1, qTableCbCr100);

    // SOF
    writeStartOfFrame(buff, image);

    // DHT
    writeHuffmanTable(buff, 0, 0, hDCTableY);
    writeHuffmanTable(buff, 0, 1, hDCTableCbCr);
    writeHuffmanTable(buff, 1, 0, hACTableY);
    writeHuffmanTable(buff, 1, 1, hACTableCbCr);

    // SOS
    writeStartOfScan(buff);

    // ECS
    buff.insert(buff.end(),huffmanData.begin(),huffmanData.end());

    // EOI
    buff.push_back(0xFF);
    buff.push_back(EOI);
}
