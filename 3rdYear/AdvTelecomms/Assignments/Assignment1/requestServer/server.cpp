#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "../include/networking.h"

#define DEFAULT_PORT 80
#define BUF_SIZE 1024

using namespace std;

char *readUrl2(char *szUrl, long &bytesReturedOut, char **headerOut) {
    const int bufSize = 512;
    char readBuffer[bufSize], sendBuffer[bufSize], tmpBuffer[bufSize];
    char *tmpResult = NULL, *result;
    int conn;
    string server, filepath, filename;
    long totalBytesRead, thisReadSize, headerLen;

    mParseUrl(szUrl, server, filepath, filename);
    conn = connectToServer((char *) server.c_str(), 80);

    sprintf(tmpBuffer, "GET %s HTTP/1.0", filepath.c_str());
    strcpy(sendBuffer, tmpBuffer);
    strcat(sendBuffer, "\r\n");
    sprintf(tmpBuffer, "Host: %s", server.c_str());
    strcat(sendBuffer, tmpBuffer);
    strcat(sendBuffer, "\r\n");
    strcat(sendBuffer, "\r\n");
    send(conn, sendBuffer, strlen(sendBuffer), 0);

    printf("Buffer being sent:\n%s", sendBuffer);

    totalBytesRead = 0;
    while(1) {
        memset(readBuffer, 0, bufSize);
        thisReadSize = recv(conn, readBuffer, bufSize, 0);

        if (thisReadSize <= 0) break;

        tmpResult = (char *)realloc(tmpResult, thisReadSize + totalBytesRead);

        memcpy(tmpResult + totalBytesRead, readBuffer, thisReadSize);
        totalBytesRead += thisReadSize;
    }

    headerLen = getHeaderLength(tmpResult);
    long contenLen = totalBytesRead - headerLen;
    result = new char[contenLen + 1];
    memcpy(result, tmpResult + headerLen, contenLen);
    result[contenLen] = 0x0;
    char *myTmp;

    myTmp = new char[headerLen + 1];
    strncpy(myTmp, tmpResult, headerLen);
    myTmp[headerLen] = '\0';
    delete(tmpResult);
    *headerOut = myTmp;

    bytesReturedOut = contenLen;
    close(conn);
    return(result);
}

int main(int argc, char *argv[]) {
    const int bufLen = 1024;
    //char szUrl[] = "http://www.codeproject.com/Questions/427350/calling-a-website-from-cplusplus";
    char szUrl[] = "http://www.google.com";
    long fileSize;
    char *memBuffer, *headerBuffer;
    FILE *fp;

    memBuffer = headerBuffer = NULL;

    memBuffer = readUrl2(szUrl, fileSize, &headerBuffer);
    printf("returned from readUrl\n");
    printf("data returned: %s\n", memBuffer);
    if (fileSize != 0) {
        printf("Got some data\n");
        fp = fopen("downloaded.file", "wb");
        fwrite(memBuffer, 1, fileSize, fp);
        fclose(fp);
        delete(memBuffer);
        delete(headerBuffer);
    }
    return 0;
}