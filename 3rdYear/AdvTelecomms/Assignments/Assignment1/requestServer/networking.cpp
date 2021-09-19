#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;

int connectToServer(char *szServerName, int port) {
    struct hostent *hp;
    unsigned int addr;
    struct sockaddr_in server;
    int conn;

    if ((conn = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0) {
        return -1;
    }

    if (inet_addr(szServerName) == INADDR_NONE) {
        hp = gethostbyname(szServerName);
    } else {
        addr = inet_addr(szServerName);
        hp = gethostbyaddr((char *) &addr, sizeof(addr), AF_INET);
    }

    if (hp == NULL) {
        close(conn);
        return -1;
    }

    server.sin_addr.s_addr = *((unsigned long *)hp->h_addr);
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    if (connect(conn, (struct sockaddr *) &server, sizeof(server))) {
        close(conn);
        return -1;
    }
    return conn;
}

void mParseUrl(char *mUrl, string &serverName, string &filepath, string &filename) {
    string::size_type n;
    string url = mUrl;

    if (url.substr(0, 7) == "http://") url.erase(0, 7);

    if (url.substr(0, 8) == "https://") url.erase(0, 8);

    n = url.find('/');
    if (n != string::npos) {
        serverName = url.substr(0, n);
        filepath = url.substr(n);
        n = filepath.rfind('/');
        filename = filepath.substr(n + 1);
    } else {
        serverName = url;
        filepath = "/";
        filename = "";
    }
}

int getHeaderLength(char *content) {
    const char *srchStr1 = "\r\n\r\n", *srchStr2 = "\n\r\n\r";
    char *findPos;
    int ofset = -1;

    findPos = strstr(content, srchStr1);
    if (findPos != NULL) {
        ofset = findPos - content;
        ofset += strlen(srchStr1);
    } else {
        findPos = strstr(content, srchStr2);
        if (findPos != NULL) {
            ofset = findPos - content;
            ofset += strlen(srchStr2);
        }
    }
    return ofset;
}