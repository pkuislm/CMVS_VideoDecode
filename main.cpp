#include <cstring>
#include <iostream>
#include <fstream>
#include <mmintrin.h>
#include <vector>
#include "jpg.h"
#include "avi.h"

typedef union  __declspec(aligned(8)) __m64_i{
    unsigned __int64    m64_u64;
    __m64               m64_tm64;
    __int32             m64_i32[2];
    __int64             m64_i64;
    unsigned __int32    m64_u32[2];
} __m64_i;

struct pixelA{
    byte r;
    byte g;
    byte b;
    byte a;
};

struct jbpdInfo {
    uint width = 0;
    uint height = 0;
    //int unk1;
    //int unk2;
    //int unk3;
    //int unk4;
    //int unk5;
    //int unk6;
    QuantizationTable quantizationTables[3];
    //int unk8;
    //int unk9;
    //int unk10;
    //byte* frameAddr;
    //uint frameSize;
    //int ret;
    //int unk11;
    //uint unk12;
    //uint width2;
    //uint height2;
    //uint colorBits;
    //int unk13;
    //short unk14;
    //int unk15;
};

struct cmv_head {
    byte magic[3]; //CMV
    byte cmv_version;
    int cmv_frame_start_offset;
    int cmv_size;
    int unk;
    int cmv_frame_count;
    int unk2;
    int unk3;
    int frame_width;
    int frame_height;
    int unk4[2];
};

struct cmv_frame_index {
    int frame_index;
    int cmv_frame_size;
    int cmv_original_frame_size; // height * width * 3 (RGB)
    int frame_type; //2 == video, 0 == audio ?
    int offset; //in file : cmv_frame_start_offset + offset
};

struct cmvVideo{
    cmv_head head;
    cmv_frame_index* index;
};

#pragma region Consts
const byte zigZagMap_ex1[64] = {
        1 , 8 , 16, 9 , 2 , 3 , 10,
        17, 24, 32, 25, 18, 11, 4 , 5 ,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6 , 7 , 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,0
};
const __m64_i mmxOpConstant[40]={
         0x0001000100010001,  0x0000040000000400,
         0x0020002000200020,  0x001F001F001F001F,
         0x32EC32EC32EC32EC,  0x6A0A6A0A6A0A6A0A,
         0xAB0EAB0EAB0EAB0E,  0xB505B505B505B505,
         0xC000400040004000,  0xAC6122A322A3539F,
        0x40004000C0004000, 0xDD5DAC61539FDD5D,
        0xA73B4B42324958C5, 0xCDB7EE5811A84B42,
        0x4B4211A811A83249, 0xA73BCDB74B42A73B,
        0xA73B58C558C558C5, 0x8C04300B300B73FC,
        0x58C558C5A73B58C5, 0xCFF58C0473FCCFF5,
        0x84DF686245BF7B21, 0xBA41E782187E6862,
        0x6862187E187E45BF, 0x84DFBA41686284DF,
        0xAC61539F539F539F, 0x92BF2D412D416D41,
        0x539F539FAC61539F, 0xD2BF92BF6D41D2BF,
        0x8C04625441B373FC, 0xBE4DE8EE17126254,
        0x62541712171241B3, 0x8C04BE4D62548C04,
        0xB4BE4B424B424B42, 0x9DAC28BA28BA6254,
        0x4B424B42B4BE4B42, 0xD7469DAC6254D746,
        0x979E587E3B216862, 0xC4DFEB3D14C3587E,
        0x587E14C314C33B21, 0x979EC4DF587E979E
};
__m64_i mmxOpConstant2[5] = {
        0x5810581058105810, 0x5B605B605B605B60,
        0x7164716471647164, 0x59BC59BC59BC59BC,
        0x0080008000800080
};
int mmxOpCIndex[] = {
        8,16,24,32,
        8,32,24,16
};
static int tbl[1024];
static int tab3[128]{ 0 };
#pragma endregion

#ifdef useless

void initJBPDInfo(jbpdInfo* result){
    result->unk2 = 24;
    result->unk4 = 0;
    result->unk5 = 1;
    result->unk1 = 1;
    result->unk3 = 1;
    result->unk11 = 0;
    result->height = 0;
    result->unk6 = 80;
    result->frameSize = 0;
    result->width = 0;
    result->unk8 = 1;
    result->frameAddr = nullptr;
    result->ret = 0;
    result->colorBits = 24;
    result->unk9 = 1;
    result->unk10 = 0;
    result->unk15 = 0;
}

void GetInfoFromFrameFlag(jbpdInfo* info){
    if(!info->frameAddr)
        return;
    int* frame = reinterpret_cast<int*>(info->frameAddr);
    info->width = *reinterpret_cast<short*>(reinterpret_cast<short*>(frame) + 8);
    info->height = *reinterpret_cast<short*>(reinterpret_cast<short*>(frame) + 9);
    info->unk3 = (frame[2] >> 30) & 1;
    info->unk4 = frame[2] >> 31;
    switch((frame[2] >> 28) & 3){
        case 3:
            info->unk5 = -1;
            break;
        case 2:
            info->unk5 = 2;
            break;
        case 1:
            info->unk5 = 1;
            break;
        default:
            info->unk5 = 0;
            break;
    }
    switch(info->unk5){
        case 0:
            info->width2 = (info->width + 7) & 0xFFFFFFF8;
            info->height2 = (info->height + 7) & 0xFFFFFFF8;
            break;
        case 1:
            info->width2 = (info->width + 15) & 0xFFFFFFF0;
            info->height2 = (info->height + 15) & 0xFFFFFFF0;
            break;
        case 2:
            info->width2 = (info->width + 31) & 0xFFFFFFE0;
            info->height2 = (info->height + 15) & 0xFFFFFFF0;
            break;
        default:
            return;
    }
    info->unk1 = *reinterpret_cast<short *>(reinterpret_cast<byte *>(frame) + 8);
    info->unk2 = frame[5];
    info->unk6 = frame[3];
    info->unk8 = (info->width & 7) == 0 && (info->height & 7) == 0;
    info->unk9 = (frame[2] >> 23) & 1;
    info->unk10 = (frame[2] >> 22) & 1;
    info->unk12 = (info->width2 * info->colorBits) >> 3;
}

#endif

#pragma region mmxFunc

void mmxPostProcess(__m64_i *input)
{
    __m64 v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26;

    for(int i = 0; i < 16; i+=2){
        v1 = _m_punpcklwd(_mm_cvtsi32_si64(input[i].m64_i32[0]), _mm_cvtsi32_si64(input[i + 1].m64_u32[0]));
        v2 = _m_punpcklwd(_mm_cvtsi32_si64(input[i].m64_u32[1]), _mm_cvtsi32_si64(input[i + 1].m64_u32[1]));
        v3 = _m_punpckldq(v1, v1);
        v4 = _m_punpckhdq(v1, v1);
        v5 = _m_punpckldq(v2, v2);
        v6 = _m_punpckhdq(v2, v2);
        v7 = _m_paddd(_m_paddd(_m_pmaddwd(mmxOpConstant[mmxOpCIndex[i / 2] + 2].m64_tm64, v3), _m_pmaddwd(mmxOpConstant[mmxOpCIndex[i / 2] + 3].m64_tm64, v5)), mmxOpConstant[1].m64_tm64);
        v8 = _m_paddd(_m_paddd(_m_pmaddwd(v3, mmxOpConstant[mmxOpCIndex[i / 2]].m64_tm64), _m_pmaddwd(v5, mmxOpConstant[mmxOpCIndex[i / 2] + 1].m64_tm64)), mmxOpConstant[1].m64_tm64);
        v9 = _m_paddd(_m_pmaddwd(mmxOpConstant[mmxOpCIndex[i / 2] + 6].m64_tm64, v4), _m_pmaddwd(mmxOpConstant[mmxOpCIndex[i / 2] + 7].m64_tm64, v6));
        v10 = _m_paddd(_m_pmaddwd(v4, mmxOpConstant[mmxOpCIndex[i / 2] + 4].m64_tm64), _m_pmaddwd(v6, mmxOpConstant[mmxOpCIndex[i / 2] + 5].m64_tm64));
        input[i].m64_tm64 = _m_packssdw(_m_psradi(_m_paddd(v10, v8), 0xBu), _m_psradi(_m_paddd(v9, v7), 0xBu));
        v11 = _m_packssdw(_m_psradi(_m_psubd(v7, v9), 0xBu), _m_psradi(_m_psubd(v8, v10), 0xBu));
        input[i + 1].m64_tm64 = _m_por(_m_psrldi(v11, 0x10u), _m_pslldi(v11, 0x10u));
    }

    for(int i = 0; i < 2; ++i){
        v1 = input[i + 10].m64_tm64;
        v2 = input[i + 6].m64_tm64;
        v3 = input[i + 14].m64_tm64;
        v4 = _m_paddsw(_m_paddsw(_m_pmulhw(v1, mmxOpConstant[6].m64_tm64), v1), v2);
        v5 = _m_psubsw(v1, _m_paddsw(_m_pmulhw(mmxOpConstant[6].m64_tm64, v2), v2));
        v6 = _m_paddsw(_m_pmulhw(v3, mmxOpConstant[4].m64_tm64), input[i + 2].m64_tm64);
        v7 = _m_paddsw(_m_pmulhw(input[i + 12].m64_tm64, mmxOpConstant[5].m64_tm64), input[i + 4].m64_tm64);
        v8 = _m_psubsw(_m_pmulhw(mmxOpConstant[4].m64_tm64, input[i + 2].m64_tm64), v3);
        v9 = _m_psubsw(_m_pmulhw(mmxOpConstant[5].m64_tm64, input[i + 4].m64_tm64), input[i + 12].m64_tm64);
        v10 = _m_paddsw(_m_psubsw(v8, v5), mmxOpConstant[0].m64_tm64);
        input[i + 14].m64_tm64= _m_paddsw(_m_paddsw(v4, v6), mmxOpConstant[0].m64_tm64);
        v11 = _m_psubsw(v6, v4);
        v12 = _m_paddsw(v11, v10);
        input[i + 6].m64_tm64 = _m_paddsw(v8, v5);
        v13 = _m_psubsw(v11, v10);
        v14 = _m_por(_m_paddsw(v12, _m_pmulhw(mmxOpConstant[7].m64_tm64, v12)), mmxOpConstant[0].m64_tm64);
        v15 = _m_por(_m_paddsw(_m_pmulhw(mmxOpConstant[7].m64_tm64, v13), v13), mmxOpConstant[0].m64_tm64);
        v16 = _m_paddsw(input[i + 8].m64_tm64, input[i + 0].m64_tm64);
        v17 = _m_psubsw(input[i + 0].m64_tm64, input[i + 8].m64_tm64);
        v18 = _m_paddsw(_m_paddsw(v16, v7), mmxOpConstant[2].m64_tm64);
        v19 = _m_paddsw(_m_paddsw(v17, v9), mmxOpConstant[2].m64_tm64);
        v20 = _m_psubsw(v17, v9);
        v21 = _m_paddsw(_m_psubsw(v16, v7), mmxOpConstant[3].m64_tm64);
        v22 = v19;
        input[i].m64_tm64 = _m_psrawi(_m_paddsw(input[i + 14].m64_tm64, v18), 6u);
        v23 = _m_paddsw(v20, mmxOpConstant[3].m64_tm64);
        input[i + 2].m64_tm64 = _m_psrawi(_m_paddsw(v19, v14), 6u);
        v24 = _m_paddsw(input[i + 6].m64_tm64, v21);
        v25 = _m_psubsw(v21, input[i + 6].m64_tm64);
        input[i + 4].m64_tm64 = _m_psrawi(_m_paddsw(v23, v15), 6u);
        v26 = _m_psubsw(v18, input[i + 14].m64_tm64);
        input[i + 6].m64_tm64 = _m_psrawi(v24, 6u);
        input[i + 8].m64_tm64 = _m_psrawi(v25, 6u);
        input[i + 10].m64_tm64 = _m_psrawi(_m_psubsw(v23, v15), 6u);
        input[i + 12].m64_tm64 = _m_psrawi(_m_psubsw(v22, v14), 6u);
        input[i + 14].m64_tm64 = _m_psrawi(v26, 6u);
    }
}

void mmxProcess(byte *dst, int dstSize, __m64_i *input)
{
    __m64_i *v3, *v4, *v5;
    __m64 v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
            v30, v31, v32, v33, v34, v35, v36, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53;


    v4 = input;
    v5 = input + 64;
    for(int j = 0; j < 2; ++j){
        v3 = reinterpret_cast<__m64_i *>(dst) + (4 * j);
        for(int i = 0; i < 4; ++i){
            v7 = v5[16].m64_tm64;
            v8 = _m_pmullw(_m_paddw(_m_pmulhw(mmxOpConstant2[0].m64_tm64, v5[0].m64_tm64), _m_psllwi(_m_pmulhw(mmxOpConstant2[1].m64_tm64, v7), 1u)),_mm_set_pi64x(-1));
            v9 = _m_pmulhw(mmxOpConstant2[2].m64_tm64, _m_psllwi(v5->m64_tm64, 2u));
            v10 = _m_pmulhw(mmxOpConstant2[3].m64_tm64, _m_psllwi(v7, 2u));
            v11 = _m_punpckldq(_m_punpcklwd(v9, v8), v10);
            v12 = _m_psrlqi(v10, 0x10u);
            v13 = _m_psrlqi(v9, 0x10u);
            v14 = _m_psrlqi(v8, 0x10u);
            v15 = _m_punpckldq(_m_punpcklwd(v13, v14), v12);
            v16 = _m_psrlqi(v12, 0x10u);
            v17 = _m_psrlqi(v13, 0x10u);
            v18 = _m_psrlqi(v14, 0x10u);
            v19 = _m_punpckldq(_m_punpcklwd(v17, v18), v16);
            v20 = _m_punpckldq(_m_punpcklwd(_m_psrlqi(v17, 0x10u), _m_psrlqi(v18, 0x10u)), _m_psrlqi(v16, 0x10u));
            v21 = _m_paddw(v4[0].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v22 = _m_punpcklwd(v21, v21);
            v23 = _m_psrlqi(v21, 0x10u);
            v24 = _m_punpcklwd(v23, v23);
            v25 = _m_psrlqi(v23, 0x10u);
            v3[0].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v22, v22), v11), _m_paddsw(_m_punpcklwd(v24, v24), v11));
            v26 = _m_punpcklwd(v25, v25);
            v27 = _m_psrlqi(v25, 0x10u);
            v28 = _m_punpcklwd(v27, v27);
            v29 = _m_paddw(v4[1].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v3[1].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v26, v26), v15), _m_paddsw(_m_punpcklwd(v28, v28), v15));
            v30 = _m_punpcklwd(v29, v29);
            v31 = _m_psrlqi(v29, 0x10u);
            v32 = _m_punpcklwd(v31, v31);
            v33 = _m_psrlqi(v31, 0x10u);
            v3[2].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v30, v30), v19), _m_paddsw(_m_punpcklwd(v32, v32), v19));
            v34 = _m_punpcklwd(v33, v33);
            v35 = _m_psrlqi(v33, 0x10u);
            v36 = _m_punpcklwd(v35, v35);
            v3[3].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v34, v34), v20), _m_paddsw(_m_punpcklwd(v36, v36), v20));
            v3 = reinterpret_cast<__m64_i *>(reinterpret_cast<byte *>(v3) + dstSize);
            v38 = _m_paddw(v4[2].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v39 = _m_punpcklwd(v38, v38);
            v40 = _m_psrlqi(v38, 0x10u);
            v41 = _m_punpcklwd(v40, v40);
            v42 = _m_psrlqi(v40, 0x10u);
            v3[0].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v39, v39), v11), _m_paddsw(_m_punpcklwd(v41, v41), v11));
            v43 = _m_punpcklwd(v42, v42);
            v44 = _m_psrlqi(v42, 0x10u);
            v45 = _m_punpcklwd(v44, v44);
            v46 = _m_paddw(v4[3].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v3[1].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v43, v43), v15), _m_paddsw(_m_punpcklwd(v45, v45), v15));
            v47 = _m_punpcklwd(v46, v46);
            v48 = _m_psrlqi(v46, 0x10u);
            v49 = _m_punpcklwd(v48, v48);
            v50 = _m_psrlqi(v48, 0x10u);
            v3[2].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v47, v47), v19), _m_paddsw(_m_punpcklwd(v49, v49), v19));
            v51 = _m_punpcklwd(v50, v50);
            v52 = _m_psrlqi(v50, 0x10u);
            v53 = _m_punpcklwd(v52, v52);
            v3[3].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v51, v51), v20), _m_paddsw(_m_punpcklwd(v53, v53), v20));
            v3 = reinterpret_cast<__m64_i *>(reinterpret_cast<byte *>(v3) + dstSize);
            v5 += 2;
            v4 += 4;
        }
        v5 -= 7;
    }

    dst += dstSize * sizeof(__m64_i);
    v4 = input + 32;
    v5 = input + 72;
    for(int j = 0; j < 2; ++j){
        v3 = reinterpret_cast<__m64_i *>(dst) + (4 * j);
        for(int i = 0; i < 4; ++i){
            v7 = v5[16].m64_tm64;
            v8 = _m_pmullw(_m_paddw(_m_pmulhw(mmxOpConstant2[0].m64_tm64, v5[0].m64_tm64), _m_psllwi(_m_pmulhw(mmxOpConstant2[1].m64_tm64, v7), 1u)),_mm_set_pi64x(-1));
            v9 = _m_pmulhw(mmxOpConstant2[2].m64_tm64, _m_psllwi(v5[0].m64_tm64, 2u));
            v10 = _m_pmulhw(mmxOpConstant2[3].m64_tm64, _m_psllwi(v7, 2u));
            v11 = _m_punpckldq(_m_punpcklwd(v9, v8), v10);
            v12 = _m_psrlqi(v10, 0x10u);
            v13 = _m_psrlqi(v9, 0x10u);
            v14 = _m_psrlqi(v8, 0x10u);
            v15 = _m_punpckldq(_m_punpcklwd(v13, v14), v12);
            v16 = _m_psrlqi(v12, 0x10u);
            v17 = _m_psrlqi(v13, 0x10u);
            v18 = _m_psrlqi(v14, 0x10u);
            v19 = _m_punpckldq(_m_punpcklwd(v17, v18), v16);
            v20 = _m_punpckldq(_m_punpcklwd(_m_psrlqi(v17, 0x10u), _m_psrlqi(v18, 0x10u)), _m_psrlqi(v16, 0x10u));
            v21 = _m_paddw(v4[0].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v22 = _m_punpcklwd(v21, v21);
            v23 = _m_psrlqi(v21, 0x10u);
            v24 = _m_punpcklwd(v23, v23);
            v25 = _m_psrlqi(v23, 0x10u);
            v3[0].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v22, v22), v11), _m_paddsw(_m_punpcklwd(v24, v24), v11));
            v26 = _m_punpcklwd(v25, v25);
            v27 = _m_psrlqi(v25, 0x10u);
            v28 = _m_punpcklwd(v27, v27);
            v29 = _m_paddw(v4[1].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v3[1].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v26, v26), v15), _m_paddsw(_m_punpcklwd(v28, v28), v15));
            v30 = _m_punpcklwd(v29, v29);
            v31 = _m_psrlqi(v29, 0x10u);
            v32 = _m_punpcklwd(v31, v31);
            v33 = _m_psrlqi(v31, 0x10u);
            v3[2].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v30, v30), v19), _m_paddsw(_m_punpcklwd(v32, v32), v19));
            v34 = _m_punpcklwd(v33, v33);
            v35 = _m_psrlqi(v33, 0x10u);
            v36 = _m_punpcklwd(v35, v35);
            v3[3].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v34, v34), v20), _m_paddsw(_m_punpcklwd(v36, v36), v20));
            v3 = reinterpret_cast<__m64_i *>(reinterpret_cast<byte *>(v3) + dstSize);
            v38 = _m_paddw(v4[2].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v39 = _m_punpcklwd(v38, v38);
            v40 = _m_psrlqi(v38, 0x10u);
            v41 = _m_punpcklwd(v40, v40);
            v42 = _m_psrlqi(v40, 0x10u);
            v3[0].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v39, v39), v11), _m_paddsw(_m_punpcklwd(v41, v41), v11));
            v43 = _m_punpcklwd(v42, v42);
            v44 = _m_psrlqi(v42, 0x10u);
            v45 = _m_punpcklwd(v44, v44);
            v46 = _m_paddw(v4[3].m64_tm64, mmxOpConstant2[4].m64_tm64);
            v3[1].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v43, v43), v15), _m_paddsw(_m_punpcklwd(v45, v45), v15));
            v47 = _m_punpcklwd(v46, v46);
            v48 = _m_psrlqi(v46, 0x10u);
            v49 = _m_punpcklwd(v48, v48);
            v50 = _m_psrlqi(v48, 0x10u);
            v3[2].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v47, v47), v19), _m_paddsw(_m_punpcklwd(v49, v49), v19));
            v51 = _m_punpcklwd(v50, v50);
            v52 = _m_psrlqi(v50, 0x10u);
            v53 = _m_punpcklwd(v52, v52);
            v3[3].m64_tm64 = _m_packuswb(_m_paddsw(_m_punpcklwd(v51, v51), v20), _m_paddsw(_m_punpcklwd(v53, v53), v20));
            v3 = reinterpret_cast<__m64_i *>(reinterpret_cast<byte *>(v3) + dstSize);
            v5 += 2;
            v4 += 4;
        }
        v5 -= 7;
    }
}

#pragma endregion

class bitHelper{
private:
    int bitShift;
    int* readPos;
    uint current;
public:
    bitHelper(){
        bitShift = 0;
        current = 0;
        readPos = nullptr;
    }
    int get_bit(){
        int bit = current & 1;
        current = current >> 1;
        bitShift++;
        if(bitShift == 32){
            bitShift = 0;
            readPos++;
            current = *readPos;
        }
        return bit;
    }
    void resetRead(int* pos){
        //std::cout << "Read position has been reset to " << pos << std::endl;
        readPos = pos;
        current = *pos;
        bitShift = 0;
    }
};

int findSmall(int size, int* arr){
    int tmp[102]{ 0 };
    memset(tbl, 0, 4096);
    for(int i = -size;; --i){
        int index_small = 0;
        int index_small2 = -100;
        int val_small = 0x7D2B74FF;
        for(int k = 0; k < size; ++k){
            if(arr[k] < val_small){
                index_small = k;
                val_small = arr[k];
            }
        }
        val_small = 0x7D2B74FF;
        for(int l = 0; l < size; ++l){
            if(l != index_small && arr[l] < val_small){
                index_small2 = l;
                val_small = arr[l];
            }
        }
        if(index_small < 0 || index_small2 < 0)
            break;
        tmp[index_small] = size;
        tmp[index_small2] = i;
        tbl[size] = index_small;
        tbl[size + 512] = index_small2;

        arr[size] = arr[index_small] + arr[index_small2];
        size ++;
        arr[index_small] = 0x7D2B7500;
        arr[index_small2] = 0x7D2B7500;
    }
    return size;
}

int getTblVal(bitHelper &bh, int a1, int a2){
    int result = a1 - 1;
    while ( result >= a2 ){
        result = tbl[0x200 * bh.get_bit() + result];// 512 + x或者0 + x，对应奇偶数表
    }
    return result;
}

void emptyM64Arrs(__m64_i *arrs, uint size){
    for(int i = 0; i < size / sizeof(__m64_i); ++i)
        arrs[i].m64_i64 = 0ll;
}

int getTabPos(bitHelper &bh)
{
    int* pos = tab3;
    while ( bh.get_bit() ){
        ++pos;
    };
    return *pos;
}

void decompressCMVFrame(pixelA** units, int* unitsSize, byte* const frame, const int  frameSize, const int flag){
    int* intptrFrame = reinterpret_cast<int *>(frame);
    bool exSec = false;
    jbpdInfo info{};
    int tab[32]{ 0 };
    int tab2[32]{ 0 };
    short dcValTmp[32768]{ 0 };
    __m64_i Pix[96];
    bitHelper bh;

    //initJBPDInfo(&info);
    //info.frameAddr = frame;
    info.width = *reinterpret_cast<short*>(reinterpret_cast<short*>(frame) + 8);
    info.height = *reinterpret_cast<short*>(reinterpret_cast<short*>(frame) + 9);
    //info.frameSize = frameSize;
    //info.colorBits = 32;
    //GetInfoFromFrameFlag(&info);

    byte* frameArr = frame + (*reinterpret_cast<int *>(frame + 4) + 4);
    memcpy(tab, frameArr, 64);
    memcpy(tab2, frameArr + 64, 64);
    frameArr += 128;
    for(uint i = 0; i < 16; ++i)
        tab3[i] = frameArr[i] + 1;
    frameArr +=16;
    if(intptrFrame[2] & 0x8000000){
        for (uint i = 0; i < 64; ++i){
            info.quantizationTables[0].table[i] = frameArr[zigZagMap[i]];
            info.quantizationTables[1].table[i] = frameArr[zigZagMap[i] + 64];
        }
        frameArr += 128;
    }

    auto* afDqt = reinterpret_cast<short *>(frameArr) + 2;

    if(intptrFrame[2] & 0x1000)
        exSec = true;

    uint v70 = *frameArr;
    uint v71 = *(frameArr + 1);
    frameArr += (*reinterpret_cast<short *>(frameArr + 2) * 2) + 6;
    //byte* pexSec = frameArr;
    if(exSec)
        frameArr += ((info.width / 16 * info.height / 16 + 7) >> 3) + 2;
    if(!flag)
        exSec = false;

    bh.resetRead(reinterpret_cast<int *>(frameArr));

    int sRet = findSmall(16, tab);

    uint v19 = 3 * (info.width / 16) * (info.height / 16);
    uint v21 = v19 * 2;
    int num;
    for(uint i = 0; i < v21; ++i){
        num = 0;
        uint v24 = getTblVal(bh, sRet, 16);
        for(uint j = v24; j > 0; --j){           //summary:获得这v53个bit，转换为int
            num = (num << 1) | bh.get_bit();
        }
        if(num < (1 << ( v24 - 1)))
            num -= (1 << v24) - 1;
        dcValTmp[i] = static_cast<short>(num);
        if(i != 0)
            dcValTmp[i] = static_cast<short>(num + dcValTmp[i - 1]);
    }

    bh.resetRead(reinterpret_cast<int *>(frameArr + intptrFrame[7]));//第二个区段

    sRet = findSmall(16, tab2);
    uint v30 = v70 * v71;
    short* pdcValTmp = dcValTmp;
    if(exSec && v30){
        //Executes when have extra block and flag!=0
        //只有当flag不为0的时候才会执行到这里
    }else if(v30){
        for(uint i = 0; i < v30; ++i){                       //While v75 -checked
            short outerCycle = *afDqt & 0x00FF;
            short innerCycle = *afDqt >> 8;
            byte* unit = reinterpret_cast<byte*>(*units);//对应一个0x40000的内存块，unit指向起始位置
            int unitSize = *unitsSize;
            int blockSize = 16 * unitSize;

            afDqt++;
            unitsSize++;
            units++;

            for(uint j = 0; j < outerCycle; ++j){           //While v67 -checked
                int unitTmp = 0;//在这个内存块中的偏移

                for(uint k = 0; k < innerCycle; ++k){       //While !v20
                    emptyM64Arrs(Pix, 0x300);
                    auto* pPix = reinterpret_cast<short *>(Pix);//指向__m64[0]，共计768字节，index最大允许384

                    for(uint l = 0; l < 6; ++l){
                        pPix[0] = info.quantizationTables[l >> 2].table[0] * *pdcValTmp;
                        pdcValTmp++;
                        for(int m = 0; m < 63;){
                            int v53 = getTblVal(bh, sRet, 16);
                            num = 0;
                            if(v53 == 0xF)
                                break;
                            if(v53){
                                for(int n = v53; n > 0 ; --n){
                                    num = (num << 1) | bh.get_bit();
                                }
                                if(num < (1 << ( v53 - 1)))
                                    num -= (1 << v53) - 1;
                                pPix[zigZagMap_ex1[m]] = info.quantizationTables[l >> 2].table[m + 1] * num;
                                m++;
                            }else{
                                m += getTabPos(bh);
                            }
                        }
                        pPix += 64;//即已经填充16个 __m64
                    }//循环一共填充 16 * 6 = 96 个 __m64
                    //Huge Shit Here
                    mmxPostProcess(Pix);
                    mmxPostProcess(Pix + 16);
                    mmxPostProcess(Pix + 32);
                    mmxPostProcess(Pix + 48);
                    mmxPostProcess(Pix + 64);
                    mmxPostProcess(Pix + 80);
                    mmxProcess(unit + unitTmp, unitSize, Pix);
                    unitTmp += 64;
                }
                unit += blockSize;
            }
        }
    }
    _m_empty();
}

bool extractCMVFrames(std::ifstream &cmvFile, cmvVideo cmv, const std::string outFilename){
    cmv.index = reinterpret_cast<cmv_frame_index *>(malloc((cmv.head.cmv_frame_count+1) * sizeof(cmv_frame_index)));
    cmvFile.read(reinterpret_cast<char *>(cmv.index), (cmv.head.cmv_frame_count+1) * sizeof(cmv_frame_index));

    byte* frame = nullptr;
    int* unitSizes = nullptr;
    pixelA** unitBuff = nullptr;
    byte* bimage = nullptr;

    uint uWidth = (cmv.head.frame_width + 255) / 256;
    uint uHeight = (cmv.head.frame_height + 255) / 256;
    uint uCount = uWidth * uHeight;

    unitSizes = new int[uCount]{ 0 };
    unitBuff = new pixelA*[uCount]{ 0 };
    //分配需要的内存
    bimage = new byte[uCount * 0x30000];
    for(int i = 0; i < uCount; i++){
        unitSizes[i] = 0x400;                       //16(长) * 16(宽) * 4(通道)
        unitBuff[i] = new pixelA[0x10000]{ 0 };     //4(通道) * 256(长) * 256(宽)
    }
    BMPImage image;
    image.height = cmv.head.frame_height;
    image.width = cmv.head.frame_width;
    image.blockHeight = (cmv.head.frame_height + 7) / 8;
    image.blockWidth = (cmv.head.frame_width + 7) / 8;
    image.blocks = new Block[image.blockHeight * image.blockWidth];
    std::vector<byte> jpgBuff;
    std::vector<int> frameSizes;
    jpgBuff.reserve(0x64000);
    frameSizes.reserve(0x64000);

    unsigned long aviSize = 0;
    std::ofstream ot(outFilename, std::ios::binary);
    writeAVI(ot, image.width, image.height, 24, cmv.head.cmv_frame_count, 1);

    long long fstart = cmv.head.cmv_frame_start_offset;
    for(int i = 0; i <= cmv.head.cmv_frame_count; ++i){
        frame = new byte[cmv.index[i].cmv_frame_size];
        cmvFile.seekg(fstart + cmv.index[i].offset, std::ios::beg);
        cmvFile.read(reinterpret_cast<char *>(frame), cmv.index[i].cmv_frame_size);
        if(cmv.index[i].frame_type!=2){
            if(cmv.index[i].cmv_frame_size == cmv.index[i].cmv_original_frame_size){
                std::cout << "Found sound frame at " << i << " ." << std::endl;
                const std::size_t pos = outFilename.find_last_of('.');
                const std::string outsFilename = (pos == std::string::npos) ? (outFilename + ".ogg") : (outFilename.substr(0, pos) + ".ogg");
                std::ofstream so(outsFilename, std::ios::binary);
                so.write(reinterpret_cast<const char *>(frame), cmv.index[i].cmv_frame_size);
                so.close();
                continue;
            }else{
                std::cout << "Unknown frame type: " << cmv.index[i].frame_type << std::endl;
                continue;
            }
            std::cout << "Unknown frame type: " << cmv.index[i].frame_type << std::endl;
            continue;
        }

        //解压
        decompressCMVFrame(unitBuff,unitSizes,frame,cmv.index[i].cmv_frame_size,0);
        delete[] frame;

        uint imgindex = 0;
        for(uint y = 0; y < cmv.head.frame_height; ++y){
            const uint mcuRow = y / 256;
            const uint pixelRow = y % 256;
            for(uint x = 0; x < cmv.head.frame_width; x++){
                const uint mcuColumn = x / 256;
                const uint pixelColumn = x % 256;
                const uint mcuIndex = mcuRow * uWidth + mcuColumn;
                const uint pixelIndex = pixelRow * 256 + pixelColumn;
                bimage[imgindex + 0] = unitBuff[mcuIndex][pixelIndex].b;
                bimage[imgindex + 1] = unitBuff[mcuIndex][pixelIndex].g;
                bimage[imgindex + 2] = unitBuff[mcuIndex][pixelIndex].r;
                imgindex += 3;
            }
        }
        imgindex = 0;
        for(uint y = 0; y < cmv.head.frame_height; ++y){
            const uint blockRow = y / 8;
            const uint pixelRow = y % 8;
            for(uint x = 0; x < cmv.head.frame_width; x++){
                const uint blockColumn = x / 8;
                const uint pixelColumn = x % 8;
                const uint blockIndex = blockRow * image.blockWidth + blockColumn;
                const uint pixelIndex = pixelRow * 8 + pixelColumn;
                image.blocks[blockIndex].r[pixelIndex] = bimage[imgindex + 0];
                image.blocks[blockIndex].g[pixelIndex] = bimage[imgindex + 1];
                image.blocks[blockIndex].b[pixelIndex] = bimage[imgindex + 2];
                imgindex += 3;
            }
        }
        //转换为jpg
        RGBToYCbCr(image);
        forwardDCT(image);
        quantize(image);
        writeJPG(image, jpgBuff);
        //写入AVI
        writeFrames(ot, jpgBuff);
        std::cout<<"Converted " << i << '/' << cmv.head.cmv_frame_count << " frame.." << std::endl;
        aviSize += jpgBuff.size();
        frameSizes.push_back(jpgBuff.size());
        jpgBuff.clear();
    }
    ot.seekp(0,std::ios::beg);
    writeAVI(ot,image.width,image.height,24,frameSizes.size(),aviSize);
    ot.seekp(0,std::ios::end);
    writeEnd(ot, frameSizes);
    ot.close();

    delete[] image.blocks;
    delete[] bimage;
    delete[] unitSizes;
    for(int i = 0; i < uCount; ++i)
        delete[] unitBuff[i];
    delete[] unitBuff;
    return true;
}


int main(int argc, char** argv){
    if(argc<2){
        std::cout << "Pkuism : 2022/2/9\nCMV Video decoder\n\nUsage: CMVDecode.exe [video.cmv]" << std::endl;
        return 0;
    }

    const std::string filename(argv[1]);
    std::ifstream in = std::ifstream(filename,std::ios::in | std::ios::binary);
    if(in.is_open()){
        cmvVideo video;
        in.read(reinterpret_cast<char *>(&video.head), sizeof(cmv_head));
        if(video.head.magic[0] == 'C' && video.head.magic[1] == 'M' && video.head.magic[2] == 'V'){
            std::cout << "Resolution: " << video.head.frame_width << 'x' << video.head.frame_height << '\n';
            std::cout << "Total frames: " << video.head.cmv_frame_count << std::endl;
            const std::size_t pos = filename.find_last_of('.');
            const std::string outFilename = (pos == std::string::npos) ? (filename + ".avi") : (filename.substr(0, pos) + ".avi");
            if(!extractCMVFrames(in, video, outFilename))
                std::cout << "Error occurs when try to extract frame data." << std::endl;
        }
    }
    return 0;
}