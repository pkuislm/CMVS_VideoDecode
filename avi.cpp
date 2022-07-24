//
// Created by Pkuism on 2022/2/9.
//
#include <fstream>
#include <vector>
#include "avi.h"

typedef unsigned char byte;
// AVI atoms

#define AVIIF_KEYFRAME 0x00000010
// Tried to output directly through outFile.put ms ,file_ptr Will instead store header in file and then
// read and output it

void fwrite_DWORD(std::ofstream& outFile, DWORD word)
{
    unsigned char *p;

    p = (unsigned char *)&word;
    int i;
    for (i = 0; i < 4; i++) {
        outFile.put(p[i]);
    }
}

void fwrite_WORD(std::ofstream& outFile, WORD word)
{
    unsigned char *p;

    p = (unsigned char *)&word;
    int i;
    for (i = 0; i < 2; i++) {
        outFile.put(p[i]);
    }
}

void writeFrames(std::ofstream& outFile, std::vector<byte>& buff)
{

    CHUNK data;
    data.dwFourCC = '00db';
    outFile.put('0');
    outFile.put('0');
    outFile.put('d');
    outFile.put('b');

    data.dwSize = buff.size();
    fwrite_DWORD(outFile, data.dwSize);

    for (auto i : buff)
        outFile.put(i);
    if (buff.size() % 2){
        outFile.put('\0');
        buff.push_back('\0');
    }


}

void writeEnd(std::ofstream& outFile, std::vector<int>& frameSizes){

    outFile.put('i');
    outFile.put('d');
    outFile.put('x');
    outFile.put('1');
    unsigned long index_length = 4 * 4 * frameSizes.size();
    fwrite_DWORD(outFile, index_length);

    unsigned long AVI_KEYFRAME = 16;

    unsigned long offset_count = 4;

    for (auto i : frameSizes){
        outFile.put('0');
        outFile.put('0');
        outFile.put('d');
        outFile.put('b');
        fwrite_DWORD(outFile, AVI_KEYFRAME);
        fwrite_DWORD(outFile, offset_count);
        fwrite_DWORD(outFile, i);
        offset_count += i + 8;
    }

}

void writeAVI(std::ofstream& outFile, int jpgs_width, int jpgs_height, unsigned long fps, int nbr_of_jpgs, unsigned int len)
{

    RIFF RIFF_LIST;

    RIFF_LIST.dwRIFF = 'RIFF';
    outFile.put('R');
    outFile.put('I');
    outFile.put('F');
    outFile.put('F');

    RIFF_LIST.dwSize = 150 + 12 + len + 8 * nbr_of_jpgs + 8 + 4 * 4 * nbr_of_jpgs;
    fwrite_DWORD(outFile, RIFF_LIST.dwSize);

    RIFF_LIST.dwFourCC = 'AVI ';
    outFile.put('A');
    outFile.put('V');
    outFile.put('I');
    outFile.put(' ');
    // 	RIFF_LIST.data = WAIT WITH THIS

    LIST hdrl;
    hdrl.dwList = 'LIST';
    outFile.put('L');
    outFile.put('I');
    outFile.put('S');
    outFile.put('T');
    hdrl.dwSize = 208;
    fwrite_DWORD(outFile, hdrl.dwSize);
    hdrl.dwFourCC = 'hdrl';
    outFile.put('h');
    outFile.put('d');
    outFile.put('r');
    outFile.put('l');

    MainAVIHeader avih;

    avih.dwFourCC = 'avih';
    outFile.put('a');
    outFile.put('v');
    outFile.put('i');
    outFile.put('h');
    avih.dwSize = 56;
    fwrite_DWORD(outFile, avih.dwSize);

    avih.dwMicroSecPerFrame = 1000000 / fps;
    fwrite_DWORD(outFile, avih.dwMicroSecPerFrame);

    avih.dwMaxBytesPerSec = 7000;
    fwrite_DWORD(outFile, avih.dwMaxBytesPerSec);

    avih.dwPaddingGranularity = 0;
    fwrite_DWORD(outFile, avih.dwPaddingGranularity);

    // dwFlags set to 16, do not know why!
    avih.dwFlags = 16;
    fwrite_DWORD(outFile, avih.dwFlags);

    avih.dwTotalFrames = nbr_of_jpgs;
    fwrite_DWORD(outFile, avih.dwTotalFrames);

    avih.dwInitialFrames = 0;
    fwrite_DWORD(outFile, avih.dwInitialFrames);

    avih.dwStreams = 1;
    fwrite_DWORD(outFile, avih.dwStreams);

    avih.dwSuggestedBufferSize = 0;
    fwrite_DWORD(outFile, avih.dwSuggestedBufferSize);

    avih.dwWidth = jpgs_width;
    fwrite_DWORD(outFile, avih.dwWidth);

    avih.dwHeight = jpgs_height;
    fwrite_DWORD(outFile, avih.dwHeight);

    avih.dwReserved[0] = 0;
    fwrite_DWORD(outFile, avih.dwReserved[0]);
    avih.dwReserved[1] = 0;
    fwrite_DWORD(outFile, avih.dwReserved[1]);
    avih.dwReserved[2] = 0;
    fwrite_DWORD(outFile, avih.dwReserved[2]);
    avih.dwReserved[3] = 0;
    fwrite_DWORD(outFile, avih.dwReserved[3]);

    LIST strl;
    strl.dwList = 'LIST';
    outFile.put('L');
    outFile.put('I');
    outFile.put('S');
    outFile.put('T');
    strl.dwSize = 132;
    fwrite_DWORD(outFile, strl.dwSize);

    strl.dwFourCC = 'strl';
    outFile.put('s');
    outFile.put('t');
    outFile.put('r');
    outFile.put('l');

    AVIStreamHeader strh;
    strh.dwFourCC = 'strh';
    outFile.put('s');
    outFile.put('t');
    outFile.put('r');
    outFile.put('h');

    strh.dwSize = 48;
    fwrite_DWORD(outFile, strh.dwSize);
    strh.fccType = 'vids';
    outFile.put('v');
    outFile.put('i');
    outFile.put('d');
    outFile.put('s');
    strh.fccHandler = 'MJPG';
    outFile.put('M');
    outFile.put('J');
    outFile.put('P');
    outFile.put('G');
    strh.dwFlags = 0;
    fwrite_DWORD(outFile, strh.dwFlags);
    strh.wPriority = 0; // +2 = 14
    fwrite_WORD(outFile, strh.wPriority);
    strh.wLanguage = 0; // +2 = 16
    fwrite_WORD(outFile, strh.wLanguage);
    strh.dwInitialFrames = 0; // +4 = 20
    fwrite_DWORD(outFile, strh.dwInitialFrames);
    strh.dwScale = 1; // +4 = 24
    fwrite_DWORD(outFile, strh.dwScale);
    // insert FPS
    strh.dwRate = fps; // +4 = 28
    fwrite_DWORD(outFile, strh.dwRate);
    strh.dwStart = 0; // +4 = 32
    fwrite_DWORD(outFile, strh.dwStart);
    // insert nbr of jpegs
    strh.dwLength = nbr_of_jpgs; // +4 = 36
    fwrite_DWORD(outFile, strh.dwLength);

    strh.dwSuggestedBufferSize = 0; // +4 = 40
    fwrite_DWORD(outFile, strh.dwSuggestedBufferSize);
    strh.dwQuality = 0; // +4 = 44
    fwrite_DWORD(outFile, strh.dwQuality);
    // Specifies the size of a single sample of data.
    // This is set to zero if the samples can vary in size.
    // If this number is nonzero, then multiple samples of data
    // can be grouped into a single chunk within the file.
    // If it is zero, each sample of data (such as a video frame) must be in a separate chunk.
    // For video streams, this number is typically zero, although
    // it can be nonzero if all video frames are the same size.
    //
    strh.dwSampleSize = 0; // +4 = 48
    fwrite_DWORD(outFile, strh.dwSampleSize);

    EXBMINFOHEADER strf;

    strf.dwFourCC = 'strf';
    outFile.put('s');
    outFile.put('t');
    outFile.put('r');
    outFile.put('f');
    strf.dwSize = 40;
    fwrite_DWORD(outFile, strf.dwSize);

    strf.biSize = 40;
    fwrite_DWORD(outFile, strf.biSize);

    strf.biWidth = jpgs_width;
    fwrite_DWORD(outFile, strf.biWidth);
    strf.biHeight = jpgs_height;
    fwrite_DWORD(outFile, strf.biHeight);
    strf.biPlanes = 1;
    fwrite_WORD(outFile, strf.biPlanes);
    strf.biBitCount = 24;
    fwrite_WORD(outFile, strf.biBitCount);
    strf.biCompression = 'MJPG';
    outFile.put('M');
    outFile.put('J');
    outFile.put('P');
    outFile.put('G');

    strf.biSizeImage = ((strf.biWidth * strf.biBitCount / 8 + 3) & 0xFFFFFFFC) * strf.biHeight;
    fwrite_DWORD(outFile, strf.biSizeImage);
    strf.biXPelsPerMeter = 0;
    fwrite_DWORD(outFile, strf.biXPelsPerMeter);
    strf.biYPelsPerMeter = 0;
    fwrite_DWORD(outFile, strf.biYPelsPerMeter);
    strf.biClrUsed = 0;
    fwrite_DWORD(outFile, strf.biClrUsed);
    strf.biClrImportant = 0;
    fwrite_DWORD(outFile, strf.biClrImportant);

    outFile.put('L');
    outFile.put('I');
    outFile.put('S');
    outFile.put('T');

    DWORD ddww = 16;
    fwrite_DWORD(outFile, ddww);
    outFile.put('o');
    outFile.put('d');
    outFile.put('m');
    outFile.put('l');

    outFile.put('d');
    outFile.put('m');
    outFile.put('l');
    outFile.put('h');

    DWORD szs = 4;
    fwrite_DWORD(outFile, szs);

    // nbr of jpgs
    DWORD totalframes = nbr_of_jpgs;
    fwrite_DWORD(outFile, totalframes);

    LIST movi;
    movi.dwList = 'LIST';
    outFile.put('L');
    outFile.put('I');
    outFile.put('S');
    outFile.put('T');

    movi.dwSize = len + 4 + 8 * nbr_of_jpgs;
    fwrite_DWORD(outFile, movi.dwSize);
    movi.dwFourCC = 'movi';
    outFile.put('m');
    outFile.put('o');
    outFile.put('v');
    outFile.put('i');
}