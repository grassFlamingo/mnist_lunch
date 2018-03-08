// #include<stdio.h>
// #include<stdlib.h>

/*
 * command
 * gcc -fPIC -shared filename.c -o outputfilename.so
 * 
 * in python
 * from ctypes import cdll
 * filemodel = cdll.LoadLibrary("filename.so")
 * filemodel.callsomefunction(some parameter)
 */
typedef unsigned char byte;


typedef union{
    byte byteData[4];
    int intData;
}BytesAndInt;

int pybytes_to_int32(byte* bytes, int offset){
    // byte ihave4bytes[4] = (byte[4])bytes;
    // offset += 3;
    byte* p = &bytes[offset + 3];
    BytesAndInt bai;
    for(int i = 0; i < 4; i++, p--){
        bai.byteData[i] = *p;
    }
    return bai.intData;
}

// int main(){
//     byte bytes[4] = {0, 0, 8, 3};
//     printf("%X %X %X %X\n", bytes[0], bytes[1], bytes[2], bytes[3]);
//     printf("%X\n", pybytes_to_int32(bytes));
// }